"""Deterministic unit conversion. No guessing: unknown units raise."""
from __future__ import annotations

import re

# canonical unit -> {alias: (factor, offset)}   value_canonical = value * factor + offset
_TABLE: dict[str, dict[str, tuple[float, float]]] = {
    "mole fraction": {"mole_frac": (1, 0), "fraction": (1, 0), "%": (0.01, 0), "percent": (0.01, 0),
                      "vol%": (0.01, 0), "ppm": (1e-6, 0)},
    "kPa": {"kpa": (1, 0), "atm": (101.325, 0), "bar": (100, 0), "psi": (6.894757, 0), "psia": (6.894757, 0),
            "mmhg": (0.1333224, 0), "torr": (0.1333224, 0)},
    "cm/s": {"cm/s": (1, 0), "mm/s": (0.1, 0), "m/s": (100, 0)},
    "mm/s": {"mm/s": (1, 0), "cm/s": (10, 0)},
    "mm": {"mm": (1, 0), "cm": (10, 0), "m": (1000, 0), "micron": (0.001, 0), "um": (0.001, 0), "in": (25.4, 0)},
    "cm": {"cm": (1, 0), "mm": (0.1, 0), "m": (100, 0)},
    "s": {"s": (1, 0), "sec": (1, 0), "ms": (0.001, 0), "min": (60, 0), "minutes": (60, 0)},
    "K": {"k": (1, 0), "c": (1, 273.15)},
    "mg/s": {"mg/s": (1, 0), "mg/min": (1 / 60, 0)},
    "sccm": {"sccm": (1, 0), "slpm": (1000, 0)},
    "ppm": {"ppm": (1, 0)},
    "W": {"w": (1, 0)},
    "mm2/s": {"mm2/s": (1, 0), "mm^2/s": (1, 0)},
    "count": {"count": (1, 0)},
    "1": {"1": (1, 0), "ratio": (1, 0)},
}

_ALIASES = {"mm hg": "mmhg", "°c": "c", "degc": "c", "sec.": "sec", "seconds": "s", "second": "s",
            "% by vol": "%", "microns": "micron", "µm": "um", "inch": "in", '"': "in", "mole fraction": "mole_frac"}


def norm_unit(u: str | None) -> str | None:
    if u is None:
        return None
    u = re.sub(r"\s+", " ", u.strip().lower())
    return _ALIASES.get(u, u) or None


def known(unit: str, canonical: str) -> bool:
    return norm_unit(unit) in _TABLE.get(canonical, {})


def convert(value: float, unit: str | None, canonical: str | None) -> float:
    """Convert value from `unit` to the `canonical` unit."""
    if canonical is None or unit is None:
        return value
    u = norm_unit(unit)
    try:
        factor, offset = _TABLE[canonical][u]
    except KeyError as exc:
        raise ValueError(f"no conversion from {unit!r} to {canonical!r}") from exc
    return value * factor + offset
