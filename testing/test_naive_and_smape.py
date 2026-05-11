"""Tests for `backend.app.inference` helpers (naive forecast, sMAPE)."""

from __future__ import annotations

import numpy as np

from app.inference import seasonal_naive_forecast, smape


def test_seasonal_naive_uses_weekly_lag_when_available():
    n = 500
    load = np.linspace(10.0, 50.0, n)
    t = 300
    H = 24
    out = seasonal_naive_forecast(load, t, H)
    assert out.shape == (H,)
    for k in range(H):
        i = t + k
        j = i - 168
        assert j >= 0
        assert out[k] == load[j]


def test_smape_zero_when_perfect():
    y = np.array([10.0, 20.0])
    yhat = np.array([10.0, 20.0])
    assert smape(y, yhat) < 0.01
