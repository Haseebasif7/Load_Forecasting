"""Tests for `src.windows.split_indices` (chronological test-window origins)."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.windows import split_indices


def test_split_indices_shapes_and_monotonicity():
    n = 5000
    val_start = 1500
    test_start = 3500
    W, H = config.WINDOW, config.HORIZON
    out = split_indices(val_start, test_start, n, W, H)

    train_t, val_t, test_t = out["train"], out["val"], out["test"]
    assert len(train_t) > 0 and len(val_t) > 0 and len(test_t) > 0
    assert int(train_t[0]) == W
    assert int(test_t[0]) >= test_start
    assert int(test_t[-1]) <= n - H
    assert np.all(np.diff(test_t) == 1)


def test_split_indices_rejects_empty_test():
    """Late test_start leaves no valid prediction origins before n - H."""
    n, W, H = 600, config.WINDOW, config.HORIZON
    val_start = 400
    test_start = 580  # max(W, test_start) > n - H + 1  => empty test_t
    with pytest.raises(ValueError, match="Empty split"):
        split_indices(val_start, test_start, n, W, H)
