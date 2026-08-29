"""Sliding-window dataset for the hourly CNN.

Each sample:
  past:   (4, 24)  channels [flow_m3s, precip_mm, temp_c, sm] hours t-24..t-1
  future: (1, 12)  channel  [precip_mm]                       hours t..t+11
  target: (12,)    flow_m3s                                    hours t..t+11

The 4th past channel is ERA5-Land surface soil moisture — antecedent-wetness
proxy for groundwater storage, gives the CNN a meaningful signal of how
saturated the basin is before a storm hits.

Normalization: log1p on flow + precip (right-skewed), z-score everything.
Normalization stats are computed on the training split only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .fetch import DATA_DIR


PAST_STEPS = 24
FUTURE_STEPS = 12


@dataclass
class Scaler:
    """Per-feature normalization stats. Stored alongside model checkpoints."""
    flow_log_mean: float; flow_log_std: float
    precip_log_mean: float; precip_log_std: float
    temp_mean: float; temp_std: float
    sm_mean: float = 0.35; sm_std: float = 0.10   # ERA5 surface SM defaults

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def encode_flow(self, x):
        return (np.log1p(np.clip(x, 0, None)) - self.flow_log_mean) / self.flow_log_std
    def decode_flow(self, z):
        return np.expm1(z * self.flow_log_std + self.flow_log_mean)
    def encode_precip(self, x):
        return (np.log1p(np.clip(x, 0, None)) - self.precip_log_mean) / self.precip_log_std
    def encode_temp(self, x):
        return (x - self.temp_mean) / self.temp_std
    def encode_sm(self, x):
        return (x - self.sm_mean) / self.sm_std


def encode_window(df: pd.DataFrame, past_idx: pd.DatetimeIndex,
                  fut_idx: pd.DatetimeIndex, scaler: Scaler
                  ) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Assemble and encode one inference window from an hourly frame.

    Single source of truth for the fill policy shared by live inference and
    the backtest replay in predict.py, so the two paths can't drift:
      flow    no filling here — fetch already bridged small interior gaps,
              so a NaN left over is real missing data and refuses the window
      precip  missing hours count as 0 mm
      temp    interpolate up to 4 h
      sm      ffill/bfill inside the window, then scaler.sm_mean

    Returns (past_enc (4, 24), fut_enc (1, 12), raw) where raw carries the
    unencoded flow/precip arrays for display, or None if flow or temp still
    has NaNs after filling.
    """
    flow_past = df['flow_m3s'].reindex(past_idx).values
    precip_past = df['precip_mm'].reindex(past_idx).fillna(0).values
    precip_fut = df['precip_mm'].reindex(fut_idx).fillna(0).values
    temp_past = df['temp_c'].reindex(past_idx).interpolate(limit=4).values
    if 'sm_surface' in df.columns:
        sm_past = (df['sm_surface'].reindex(past_idx)
                   .ffill().bfill().fillna(scaler.sm_mean).values)
    else:
        sm_past = np.full(len(past_idx), scaler.sm_mean)
    if np.isnan(flow_past).any() or np.isnan(temp_past).any():
        return None
    past_enc = np.stack([
        scaler.encode_flow(flow_past),
        scaler.encode_precip(precip_past),
        scaler.encode_temp(temp_past),
        scaler.encode_sm(sm_past),
    ], axis=0).astype(np.float32)
    fut_enc = scaler.encode_precip(precip_fut)[None, :].astype(np.float32)
    return past_enc, fut_enc, {'flow': flow_past,
                               'precip_past': precip_past,
                               'precip_fut': precip_fut}


def build_windows(df: pd.DataFrame, scaler: Scaler | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Slide a 24+12 window across the hourly DataFrame, dropping any window
    with missing flow values. Returns (past, future, target, target_t0).

    Past stream has 4 channels: flow, precip, temperature, surface soil
    moisture. Future stream is just forecast precip.
    """
    flow = df['flow_m3s'].values
    precip = df['precip_mm'].fillna(0).values
    temp = df['temp_c'].fillna(np.nanmean(df['temp_c'])).values
    # ERA5 SM has gaps near present; ffill in the caller covers the lag.
    sm = (df['sm_surface'].ffill().bfill()
          .fillna(scaler.sm_mean if scaler is not None else 0.35).values)
    idx = df.index

    n = len(df)
    win = PAST_STEPS + FUTURE_STEPS
    if n < win:
        return (np.zeros((0, 4, PAST_STEPS)), np.zeros((0, 1, FUTURE_STEPS)),
                np.zeros((0, FUTURE_STEPS)), [])

    past_list, fut_list, tgt_list, t0_list = [], [], [], []
    for i in range(n - win + 1):
        f = flow[i:i + win]
        if np.isnan(f).any():
            continue
        p_past = precip[i:i + PAST_STEPS]
        p_fut = precip[i + PAST_STEPS:i + PAST_STEPS + FUTURE_STEPS]
        t_past = temp[i:i + PAST_STEPS]
        sm_past = sm[i:i + PAST_STEPS]
        flow_past = flow[i:i + PAST_STEPS]
        flow_fut = flow[i + PAST_STEPS:i + PAST_STEPS + FUTURE_STEPS]

        if scaler is not None:
            flow_past = scaler.encode_flow(flow_past)
            flow_fut_scaled = scaler.encode_flow(flow_fut)
            p_past = scaler.encode_precip(p_past)
            p_fut = scaler.encode_precip(p_fut)
            t_past = scaler.encode_temp(t_past)
            sm_past = scaler.encode_sm(sm_past)
        else:
            flow_fut_scaled = flow_fut

        past_list.append(np.stack([flow_past, p_past, t_past, sm_past], axis=0))
        fut_list.append(p_fut[None, :])
        tgt_list.append(flow_fut_scaled)
        t0_list.append(idx[i + PAST_STEPS])

    if not past_list:
        return (np.zeros((0, 4, PAST_STEPS)), np.zeros((0, 1, FUTURE_STEPS)),
                np.zeros((0, FUTURE_STEPS)), [])
    return (np.stack(past_list).astype(np.float32),
            np.stack(fut_list).astype(np.float32),
            np.stack(tgt_list).astype(np.float32),
            t0_list)


def fit_scaler(df: pd.DataFrame) -> Scaler:
    """Compute log/z-score stats on the train portion of one gauge's record."""
    flow = df['flow_m3s'].dropna().values
    precip = df['precip_mm'].dropna().values
    temp = df['temp_c'].dropna().values
    sm = df['sm_surface'].dropna().values
    flow_log = np.log1p(np.clip(flow, 0, None))
    precip_log = np.log1p(np.clip(precip, 0, None))
    return Scaler(
        flow_log_mean=float(flow_log.mean()), flow_log_std=float(flow_log.std() + 1e-9),
        precip_log_mean=float(precip_log.mean()), precip_log_std=float(precip_log.std() + 1e-9),
        temp_mean=float(temp.mean()), temp_std=float(temp.std() + 1e-9),
        sm_mean=float(sm.mean()) if len(sm) else 0.35,
        sm_std=float(sm.std() + 1e-9) if len(sm) else 0.10,
    )


class WindowDataset(Dataset):
    def __init__(self, past, future, target):
        self.past = torch.from_numpy(past)
        self.future = torch.from_numpy(future)
        self.target = torch.from_numpy(target)

    def __len__(self): return len(self.past)
    def __getitem__(self, i):
        return self.past[i], self.future[i], self.target[i]


def load_site_windows(gauge_id: str, train_frac: float = 0.70,
                      val_frac: float = 0.15
                      ) -> tuple[WindowDataset, WindowDataset, WindowDataset,
                                 Scaler, list]:
    """Load a gauge's hourly parquet, split temporally, build windows."""
    df = pd.read_parquet(DATA_DIR / gauge_id / 'hourly.parquet').sort_index()
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    scaler = fit_scaler(train_df.dropna(subset=['flow_m3s']))

    train = WindowDataset(*build_windows(train_df, scaler)[:3])
    val = WindowDataset(*build_windows(val_df, scaler)[:3])
    test_p, test_f, test_t, test_t0 = build_windows(test_df, scaler)
    test = WindowDataset(test_p, test_f, test_t)
    return train, val, test, scaler, test_t0
