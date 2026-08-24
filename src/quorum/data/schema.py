"""The shared attribute taxonomy.

Every attribute a population can be synthesized on, and every level it can take, is
declared here once. Both ground-truth sources are forced onto these levels, so a
marginal from one and a topline from the other are talking about the same people.

Levels are ordered tuples, not sets: the order fixes column order in cell tables,
design matrices and reports, which is what makes runs byte-reproducible.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

LEVELS: Mapping[str, tuple[str, ...]] = {
    "age_band": ("18-24", "25-34", "35-44", "45-54", "55-64", "65+"),
    "sex": ("male", "female"),
    "education": ("less_than_hs", "high_school", "some_college", "bachelors", "graduate"),
    "race": ("white", "black", "other"),
    "marital": ("married", "widowed", "divorced", "separated", "never_married"),
}

ATTRIBUTES: tuple[str, ...] = tuple(LEVELS)

#: Lower bound of each age band. The final band is open ended.
AGE_BAND_EDGES: tuple[int, ...] = (18, 25, 35, 45, 55, 65)

#: Adults only. Both sources are restricted to this universe before anything else.
MINIMUM_AGE = 18


def age_band(age: float | int | np.ndarray) -> str | np.ndarray:
    """Map an age in years onto its band. Vectorized when given an array."""
    if isinstance(age, np.ndarray):
        idx = np.searchsorted(np.asarray(AGE_BAND_EDGES), age, side="right") - 1
        idx = np.clip(idx, 0, len(AGE_BAND_EDGES) - 1)
        bands = np.array(LEVELS["age_band"])
        out = bands[idx]
        out[age < MINIMUM_AGE] = ""
        return out
    if age < MINIMUM_AGE:
        raise ValueError(f"age {age} is below the adult universe ({MINIMUM_AGE}+)")
    idx = int(np.searchsorted(np.asarray(AGE_BAND_EDGES), age, side="right") - 1)
    return LEVELS["age_band"][min(idx, len(AGE_BAND_EDGES) - 1)]


def validate_levels(attribute: str, values: Sequence[str]) -> None:
    """Raise if any value is outside the declared levels for ``attribute``.

    Called at every boundary where external data enters. A silently unmapped level
    would show up much later as a mysteriously empty poststratification cell.
    """
    if attribute not in LEVELS:
        raise KeyError(f"unknown attribute {attribute!r}; known: {list(LEVELS)}")
    allowed = set(LEVELS[attribute])
    seen = {v for v in values if v is not None and v == v}  # drop NaN
    unexpected = sorted(seen - allowed)
    if unexpected:
        raise ValueError(
            f"{attribute} contains levels outside the taxonomy: {unexpected}; "
            f"allowed: {list(LEVELS[attribute])}"
        )


def cell_count(attributes: Sequence[str] | None = None) -> int:
    """Number of poststratification cells spanned by ``attributes``."""
    attrs = tuple(attributes) if attributes is not None else ATTRIBUTES
    total = 1
    for a in attrs:
        total *= len(LEVELS[a])
    return total
