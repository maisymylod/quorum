from __future__ import annotations

import numpy as np
import pytest

from quorum.data.schema import ATTRIBUTES, LEVELS, age_band, cell_count, validate_levels


@pytest.mark.parametrize(
    "age, expected",
    [(18, "18-24"), (24, "18-24"), (25, "25-34"), (44, "35-44"), (64, "55-64"), (65, "65+"), (105, "65+")],
)
def test_age_band_boundaries(age, expected):
    assert age_band(age) == expected


def test_age_band_rejects_children():
    with pytest.raises(ValueError, match="adult universe"):
        age_band(17)


def test_age_band_is_vectorized_and_blanks_children():
    out = age_band(np.array([17.0, 18.0, 70.0]))
    assert list(out) == ["", "18-24", "65+"]


def test_every_attribute_has_ordered_levels():
    assert ATTRIBUTES == tuple(LEVELS)
    for attribute, levels in LEVELS.items():
        assert len(set(levels)) == len(levels), attribute
        assert all(isinstance(v, str) for v in levels)


def test_validate_levels_accepts_known_values():
    validate_levels("sex", ["male", "female", None])


def test_validate_levels_rejects_unknown_values():
    with pytest.raises(ValueError, match="outside the taxonomy"):
        validate_levels("sex", ["male", "unspecified"])


def test_validate_levels_rejects_unknown_attribute():
    with pytest.raises(KeyError, match="unknown attribute"):
        validate_levels("height", ["tall"])


def test_cell_count_multiplies_levels():
    assert cell_count(["sex", "race"]) == 6
    assert cell_count() == 6 * 2 * 5 * 3 * 5
