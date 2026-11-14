"""Flame presence and extinction time from a per-second luminosity series."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BurnInterval:
    ignition_s: float | None
    extinction_s: float | None
    duration_s: float
    censored: bool  # True when the flame is still present at the last sample


def flame_mask(flame_px: np.ndarray, min_px: int = 500, smooth: int = 3) -> np.ndarray:
    """Boolean flame-present mask with a majority filter to drop single-sample flicker."""
    raw = np.asarray(flame_px) > min_px
    if smooth <= 1 or raw.size < smooth:
        return raw
    kernel = np.ones(smooth)
    votes = np.convolve(raw.astype(float), kernel, mode="same")
    return votes > smooth / 2


def burn_interval(t_s: np.ndarray, flame_px: np.ndarray, min_px: int = 500,
                  smooth: int = 3, min_burn_s: float = 3.0) -> BurnInterval:
    """Longest contiguous flame-present run; extinction is its last sample."""
    t = np.asarray(t_s, dtype=float)
    mask = flame_mask(flame_px, min_px, smooth)
    best, start = (0, 0, 0), None
    for i, on in enumerate(np.append(mask, False)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start > best[0]:
                best = (i - start, start, i - 1)
            start = None
    n, a, b = best
    if n == 0 or t[b] - t[a] < min_burn_s:
        return BurnInterval(None, None, 0.0, False)
    censored = b == len(t) - 1
    return BurnInterval(float(t[a]), None if censored else float(t[b]), float(t[b] - t[a]), censored)
