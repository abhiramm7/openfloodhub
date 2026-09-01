"""Run one controller on one storm. Shared by the tuning and results scripts."""
import numpy as np

from pondmpc import GammaLike, gamma_like
from pondmpc.planner import MPPI
import pondmpc.basin as B


def run_mppi(rainfall, model_builder, flow_threshold=1.0, substeps=1, **kw):
    sc = GammaLike(rainfall=rainfall, flow_threshold=flow_threshold)
    model = model_builder()
    pl = MPPI(model, sc.env.order, flow_threshold=flow_threshold, **kw)
    pl.set_forecast(sc.rainfall)

    orig = B.Basin.step
    B.Basin.step = lambda self, i, v, dt, _o=orig: _o(self, i, v, dt, 4)

    def model_step(self, i, v, dt, _o=orig, _s=substeps):
        return _o(self, i, v, dt, _s)

    done, k = False, 0
    while not done:
        d = sc.state()
        B.Basin.step = model_step
        a = pl(d)
        B.Basin.step = lambda self, i, v, dt, _o=orig: _o(self, i, v, dt, 4)
        done = sc.step(a)
        B.Basin.step = model_step
        pl.after_step(sc.rainfall[min(k, len(sc.rainfall) - 1)], a)
        B.Basin.step = orig
        k += 1
    return sc
