"""Train one CNN per gauge. CPU-only, single-process, no MPS (PyTorch MPS has
flaky 1D conv kernels in 2.5)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import load_site_windows, PAST_STEPS, FUTURE_STEPS
from .model import FloodCNN, nse_loss
from .sites import BY_ID

REPO = Path(__file__).resolve().parents[1]
CKPT_DIR = Path(__file__).resolve().parent / 'checkpoints'


def nse_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Standard NSE for evaluation (in physical units, after denormalization)."""
    pred = pred.flatten(); target = target.flatten()
    mask = np.isfinite(pred) & np.isfinite(target)
    p, t = pred[mask], target[mask]
    if len(t) == 0:
        return float('nan')
    return 1.0 - ((p - t) ** 2).sum() / (((t - t.mean()) ** 2).sum() + 1e-9)


def train_gauge(gauge_id: str, epochs: int = 20, batch_size: int = 64,
                lr: float = 1e-3, hidden: int = 32) -> dict:
    site = BY_ID[gauge_id]
    print(f'\n=== Training {site["short"]} ({gauge_id}) ===')

    train_ds, val_ds, test_ds, scaler, test_t0 = load_site_windows(gauge_id)
    print(f'  windows: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    model = FloodCNN(hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                            factor=0.5, patience=3)

    best_val = float('inf')
    best_state = None
    history = []
    t_start = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for past, fut, tgt in train_dl:
            pred = model(past, fut)
            loss = nse_loss(pred, tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(past)
        train_loss /= len(train_ds)

        # validate
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for past, fut, tgt in val_dl:
                pred = model(past, fut)
                val_loss += nse_loss(pred, tgt).item() * len(past)
            val_loss /= len(val_ds)

        scheduler.step(val_loss)
        history.append({'epoch': ep, 'train_loss': train_loss, 'val_loss': val_loss})
        marker = '  *' if val_loss < best_val else ''
        print(f'  ep {ep:2d}  train {train_loss:.4f}  val {val_loss:.4f}{marker}')
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test eval — denormalize, compute NSE per-hour-ahead and overall
    model.eval()
    all_pred, all_tgt = [], []
    with torch.no_grad():
        for past, fut, tgt in test_dl:
            all_pred.append(model(past, fut).numpy())
            all_tgt.append(tgt.numpy())
    all_pred = np.concatenate(all_pred)
    all_tgt = np.concatenate(all_tgt)
    pred_m3s = scaler.decode_flow(all_pred)
    tgt_m3s = scaler.decode_flow(all_tgt)
    nse_overall = nse_score(pred_m3s, tgt_m3s)
    nse_per_hour = [nse_score(pred_m3s[:, h], tgt_m3s[:, h])
                    for h in range(FUTURE_STEPS)]

    elapsed = time.time() - t_start
    print(f'  test NSE (12h ensemble): {nse_overall:.3f}  '
          f'(per hour: 1h={nse_per_hour[0]:.3f}, 6h={nse_per_hour[5]:.3f}, '
          f'12h={nse_per_hour[11]:.3f})')
    print(f'  time: {elapsed:.1f}s')

    # Save
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = {
        'state_dict': model.state_dict(),
        'config': {'hidden': hidden, 'past_steps': PAST_STEPS, 'future_steps': FUTURE_STEPS},
        'scaler': scaler.to_dict(),
        'site': site,
        'metrics': {
            'nse_overall': float(nse_overall),
            'nse_per_hour': [float(x) for x in nse_per_hour],
            'train_windows': len(train_ds),
            'val_windows': len(val_ds),
            'test_windows': len(test_ds),
        },
        'history': history,
    }
    torch.save(ckpt, CKPT_DIR / f'{gauge_id}.pt')
    print(f'  saved {CKPT_DIR / (gauge_id + ".pt")}')
    return ckpt['metrics']


if __name__ == '__main__':
    import sys
    gauge = sys.argv[1] if len(sys.argv) > 1 else '01646500'
    train_gauge(gauge)
