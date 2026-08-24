"""Shared fixtures.

The tiny population here is deliberately hand-written rather than synthesized: tests of
the core objects should fail when the core objects break, not when synthesis does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quorum.core.population import Population
from quorum.core.spec import SimulationSpec

# Levels come from the real taxonomy so that fixtures exercise the same encoding
# path as production data.
AGE_BANDS = ["18-24", "35-44", "65+"]
EDUCATION = ["high_school", "bachelors"]


@pytest.fixture
def small_population() -> Population:
    rng = np.random.default_rng(7)
    # Weights correlate with age band on purpose, so a weighted marginal is
    # distinguishable from a raw count share.
    band_weight = {"18-24": 1.0, "35-44": 2.0, "65+": 4.0}
    rows = []
    for i in range(120):
        age_band = AGE_BANDS[i % 3]
        rows.append(
            {
                "age_band": age_band,
                "education": EDUCATION[i % 2],
                "trait_openness": float(rng.random()),
                "weight": band_weight[age_band],
            }
        )
    return Population(pd.DataFrame(rows), name="small")


@pytest.fixture
def spec_dict() -> dict:
    return {
        "name": "test-sim",
        "seed": 11,
        "population": {
            "size": 500,
            "attributes": ["age_band", "education"],
            "traits": ["openness"],
        },
        "scenario": {
            "question_id": "q1",
            "prompt": "Do you agree?",
            "options": ["Agree", "Disagree"],
        },
        "predictor": {"stratify_by": ["age_band"], "archetypes": 12},
        "estimator": {"dimensions": ["age_band"]},
        "world": {"network": {"dimensions": ["age_band"]}},
    }


@pytest.fixture
def spec(spec_dict: dict) -> SimulationSpec:
    return SimulationSpec.from_dict(spec_dict)
