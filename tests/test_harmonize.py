from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quorum.data.harmonize import (
    effective_sample_size,
    harmonize_acs,
    harmonize_gss,
    share_standard_errors,
    weighted_shares,
)


@pytest.fixture
def acs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AGEP": [17, 19, 30, 44, 67, 52],
            "SEX": [1, 1, 2, 2, 1, 2],
            "SCHL": [12, 16, 20, 21, 24, 3],
            "RAC1P": [1, 1, 2, 9, 1, 6],
            "MAR": [5, 5, 1, 3, 2, 4],
            "PWGTP": [10, 10, 20, 5, 15, 8],
        }
    )


def test_acs_drops_minors_and_maps_every_attribute(acs_frame):
    out = harmonize_acs(acs_frame)
    assert len(out) == 5  # the 17 year old is outside the adult universe
    assert list(out["age_band"]) == ["18-24", "25-34", "35-44", "65+", "45-54"]
    assert list(out["education"]) == ["high_school", "some_college", "bachelors", "graduate", "less_than_hs"]
    assert list(out["race"]) == ["white", "black", "other", "white", "other"]
    assert list(out["marital"]) == ["never_married", "married", "divorced", "widowed", "separated"]
    assert out["weight"].sum() == 58


def test_acs_drops_rows_with_unmappable_codes(acs_frame):
    acs_frame.loc[1, "MAR"] = 9  # not a documented marital code
    out = harmonize_acs(acs_frame)
    assert len(out) == 4


def test_acs_drops_non_positive_weights(acs_frame):
    acs_frame.loc[2, "PWGTP"] = 0
    assert len(harmonize_acs(acs_frame)) == 4


def test_acs_requires_its_columns(acs_frame):
    with pytest.raises(KeyError, match="missing columns"):
        harmonize_acs(acs_frame.drop(columns=["SCHL"]))


@pytest.fixture
def gss_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2024] * 5,
            "age": [22, 39, 71, 98, 55],
            "sex": [1, 2, 2, 1, 1],
            "degree": [0, 2, 4, 1, 3],
            "race": [1, 2, 3, 1, 1],
            "marital": [5, 1, 2, 3, 4],
            "wtssps": [1.0, 2.0, 0.5, 1.0, 1.5],
            "natfare": [1, 2, 3, 1, 8],
        }
    )


def test_gss_maps_demographics_and_keeps_items(gss_frame):
    out = harmonize_gss(gss_frame)
    bands = list(out["age_band"])
    assert bands[:3] == ["18-24", "35-44", "65+"]
    assert pd.isna(bands[3])
    assert bands[4] == "55-64"
    assert list(out["education"]) == ["less_than_hs", "some_college", "graduate", "high_school", "bachelors"]
    assert list(out["race"]) == ["white", "black", "other", "white", "white"]
    assert "natfare" in out.columns


def test_gss_out_of_range_age_becomes_missing_not_wrong(gss_frame):
    out = harmonize_gss(gss_frame)
    # 98 is a reserved code, not a 98 year old
    assert pd.isna(out.loc[3, "age_band"])


def test_gss_requires_a_weight_column(gss_frame):
    with pytest.raises(KeyError, match="weight column"):
        harmonize_gss(gss_frame, weight_column="wtssall")


def test_gss_requires_demographics(gss_frame):
    with pytest.raises(KeyError, match="missing columns"):
        harmonize_gss(gss_frame.drop(columns=["degree"]))


def test_weighted_shares_excludes_non_substantive_answers():
    values = pd.Series(["yes", "no", "no answer", "yes"])
    weights = pd.Series([1.0, 1.0, 99.0, 3.0])
    shares = weighted_shares(values, weights, ("yes", "no"))
    np.testing.assert_allclose(shares, [0.8, 0.2])


def test_weighted_shares_returns_nan_when_nothing_qualifies():
    shares = weighted_shares(pd.Series(["dk"]), pd.Series([1.0]), ("yes", "no"))
    assert np.isnan(shares).all()


def test_effective_sample_size_penalizes_unequal_weights():
    assert effective_sample_size(np.ones(100)) == pytest.approx(100.0)
    uneven = np.array([10.0] + [1.0] * 99)
    assert effective_sample_size(uneven) < 100.0
    assert effective_sample_size(np.array([])) == 0.0


def test_standard_errors_use_the_effective_sample_size():
    shares = np.array([0.5, 0.5])
    tight = share_standard_errors(shares, 1000.0)
    loose = share_standard_errors(shares, 250.0)
    assert (loose > tight).all()
    assert tight[0] == pytest.approx(np.sqrt(0.25 / 1000))
    assert np.isnan(share_standard_errors(shares, 0.0)).all()
