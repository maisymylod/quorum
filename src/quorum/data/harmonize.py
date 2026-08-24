"""Force each public source onto the shared taxonomy.

These are crosswalks, and crosswalks are where quiet bias enters a simulation. Two
rules apply throughout: every mapping is explicit (no ``else`` that silently absorbs
unmapped codes into a residual category unless the source itself defines one), and
every output is validated against :func:`quorum.data.schema.validate_levels` before
it leaves this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quorum.data.schema import ATTRIBUTES, LEVELS, MINIMUM_AGE, age_band, validate_levels

# -- American Community Survey public use microdata -------------------------------
# Codes are from the ACS PUMS data dictionary. SCHL is educational attainment,
# RAC1P is recoded detailed race (first race reported), MAR is marital status.

ACS_COLUMNS = ("AGEP", "SEX", "SCHL", "RAC1P", "MAR", "PWGTP")

_ACS_SEX = {1: "male", 2: "female"}
_ACS_RACE = {1: "white", 2: "black"}  # every other RAC1P code folds into "other"
_ACS_MARITAL = {
    1: "married",
    2: "widowed",
    3: "divorced",
    4: "separated",
    5: "never_married",
}


def _acs_education(schl: pd.Series) -> pd.Series:
    """Map SCHL to the five-level taxonomy.

    Codes 1-15 are anything short of a high school credential, 16-17 are diploma or
    GED, 18-20 are some college through an associate degree, 21 is a bachelor's, and
    22-24 are master's, professional and doctorate.
    """
    schl = pd.to_numeric(schl, errors="coerce")
    out = pd.Series(pd.NA, index=schl.index, dtype="object")
    out[schl.between(1, 15)] = "less_than_hs"
    out[schl.between(16, 17)] = "high_school"
    out[schl.between(18, 20)] = "some_college"
    out[schl == 21] = "bachelors"
    out[schl.between(22, 24)] = "graduate"
    return out


def harmonize_acs(frame: pd.DataFrame) -> pd.DataFrame:
    """Return adult ACS person records on the shared taxonomy.

    Output columns are the taxonomy attributes plus ``weight`` (PWGTP, the person
    weight). Records missing any attribute are dropped and counted by the caller
    rather than imputed: an imputation here would propagate into the marginals that
    everything downstream is raked to.
    """
    missing = [c for c in ACS_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"ACS frame is missing columns {missing}")

    age = pd.to_numeric(frame["AGEP"], errors="coerce")
    adults = age >= MINIMUM_AGE
    frame = frame.loc[adults]
    age = age.loc[adults]

    out = pd.DataFrame(index=frame.index)
    out["age_band"] = age_band(age.to_numpy(dtype=float))
    out["sex"] = pd.to_numeric(frame["SEX"], errors="coerce").map(_ACS_SEX)
    out["education"] = _acs_education(frame["SCHL"])
    race = pd.to_numeric(frame["RAC1P"], errors="coerce")
    out["race"] = np.where(race.isin(_ACS_RACE), race.map(_ACS_RACE), "other")
    out["race"] = out["race"].where(race.notna())
    out["marital"] = pd.to_numeric(frame["MAR"], errors="coerce").map(_ACS_MARITAL)
    out["weight"] = pd.to_numeric(frame["PWGTP"], errors="coerce")

    out = out.dropna(subset=list(ATTRIBUTES) + ["weight"])
    out = out[out["weight"] > 0]
    for attribute in ATTRIBUTES:
        validate_levels(attribute, out[attribute].unique())
    return out.reset_index(drop=True)


# -- General Social Survey ---------------------------------------------------------
# Read with convert_categoricals=False so the numeric codes below apply directly.
# GSS reserved codes for missingness are large positive integers, which is why every
# mapping is a dict lookup rather than a range.

GSS_DEMOGRAPHIC_COLUMNS = ("year", "age", "sex", "degree", "race", "marital")

_GSS_SEX = {1: "male", 2: "female"}
_GSS_DEGREE = {
    0: "less_than_hs",
    1: "high_school",
    2: "some_college",  # associate or junior college
    3: "bachelors",
    4: "graduate",
}
_GSS_RACE = {1: "white", 2: "black", 3: "other"}
_GSS_MARITAL = {
    1: "married",
    2: "widowed",
    3: "divorced",
    4: "separated",
    5: "never_married",
}

#: GSS tops out its age variable here; 89 means "89 or older".
GSS_MAX_AGE = 89


def harmonize_gss(frame: pd.DataFrame, weight_column: str = "wtssps") -> pd.DataFrame:
    """Return adult GSS respondents on the shared taxonomy.

    Keeps every non-demographic column untouched so that attitude items travel with
    their respondent, and adds ``weight`` from ``weight_column``.
    """
    missing = [c for c in GSS_DEMOGRAPHIC_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"GSS frame is missing columns {missing}")
    if weight_column not in frame.columns:
        raise KeyError(f"GSS frame is missing weight column {weight_column!r}")

    age = pd.to_numeric(frame["age"], errors="coerce")
    age = age.where(age.between(MINIMUM_AGE, GSS_MAX_AGE))

    out = frame.copy()
    out["age_band"] = pd.Series(
        np.where(age.notna(), age_band(age.fillna(0).to_numpy(dtype=float)), None),
        index=frame.index,
    )
    out["sex"] = pd.to_numeric(frame["sex"], errors="coerce").map(_GSS_SEX)
    out["education"] = pd.to_numeric(frame["degree"], errors="coerce").map(_GSS_DEGREE)
    out["race"] = pd.to_numeric(frame["race"], errors="coerce").map(_GSS_RACE)
    out["marital"] = pd.to_numeric(frame["marital"], errors="coerce").map(_GSS_MARITAL)
    out["weight"] = pd.to_numeric(frame[weight_column], errors="coerce")

    out = out.dropna(subset=["weight"])
    out = out[out["weight"] > 0]
    for attribute in ATTRIBUTES:
        validate_levels(attribute, out[attribute].dropna().unique())
    return out.reset_index(drop=True)


def weighted_shares(
    values: pd.Series, weights: pd.Series, levels: tuple[str, ...]
) -> np.ndarray:
    """Weighted share of ``values`` in each level, in ``levels`` order.

    Rows whose value is not in ``levels`` are excluded, which is how survey
    non-response codes are dropped rather than counted as a substantive answer.
    """
    mask = values.isin(levels) & weights.notna() & (weights > 0)
    if not mask.any():
        return np.full(len(levels), np.nan)
    grouped = weights[mask].groupby(values[mask]).sum()
    shares = np.array([float(grouped.get(level, 0.0)) for level in levels])
    return shares / shares.sum()


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish's effective sample size.

    Unequal weights cost precision. Reporting a survey topline's standard error from
    the raw count would overstate how tightly the ground truth itself is pinned down,
    and the accuracy harness compares against that truth, so it has to be right.
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights) & (weights > 0)]
    if weights.size == 0:
        return 0.0
    return float(weights.sum() ** 2 / np.square(weights).sum())


def share_standard_errors(shares: np.ndarray, effective_n: float) -> np.ndarray:
    """Binomial standard error of each share at the effective sample size."""
    if effective_n <= 0:
        return np.full_like(np.asarray(shares, dtype=float), np.nan)
    shares = np.asarray(shares, dtype=float)
    return np.sqrt(np.clip(shares * (1.0 - shares), 0.0, None) / effective_n)
