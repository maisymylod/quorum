"""The ablation grid: every configuration the backtest compares.

Each entry is a set of overrides on one base spec, so the grid is a description of
what to vary rather than a pile of near-duplicate configuration files. Because every
stage is chosen by name through the contracts, adding an arm to the grid is a few
lines here and no change anywhere else.

The grid exists to answer questions the architecture otherwise only asserts. Does
poststratification help? Does partial pooling help? Does the joint structure in real
microdata matter, or would independent draws with the same margins do? Does peer
influence earn its place? Does the hybrid actually beat asking the model about every
agent, once cost is counted? Every one of those is an entry here rather than a claim
in a README.
"""

from __future__ import annotations

import copy
from typing import Any, Sequence

from quorum.core.spec import SimulationSpec
from quorum.data.targets import Question

DEFAULT_ATTRIBUTES = ["age_band", "sex", "education", "race", "marital"]
DEFAULT_STRATIFY = ["age_band", "education"]
DEFAULT_TRAITS = ["openness", "engagement"]

#: The population the whole grid runs on unless an entry says otherwise. Every arm of
#: a wording experiment runs on the same one, which is what makes the arms comparable.
BASE_SPEC: dict[str, Any] = {
    "seed": 20260824,
    "population": {
        "source": "marginals",
        "targets": "data/vendor/acs_marginals.json",
        "microdata": "data/vendor/acs_microdata.csv.gz",
        "size": 20_000,
        "attributes": DEFAULT_ATTRIBUTES,
        "traits": DEFAULT_TRAITS,
    },
    "predictor": {
        "kind": "hybrid",
        "archetypes": 300,
        "stratify_by": DEFAULT_STRATIFY,
        "propagator": "multinomial_logit",
        "trait_noise": 0.05,
        "provider": {"name": "stub", "model": "stub", "cache_dir": "data/cache"},
    },
    "estimator": {
        "poststratify": True,
        "dimensions": DEFAULT_STRATIFY,
        "mrp": True,
        "draws": 1000,
        "level": 0.90,
    },
    "budget": {"max_usd": 25.0, "max_calls": 60_000},
}

#: Overrides per configuration, deep-merged onto the base.
ENGINES: dict[str, dict[str, Any]] = {
    # Baselines. Neither measures anything, so neither claims an interval.
    "uniform": {
        "predictor": {"kind": "uniform"},
        "estimator": {"poststratify": False, "draws": 0},
    },
    "prior": {
        "predictor": {"kind": "prior"},
        "estimator": {"poststratify": False, "draws": 0},
    },
    # The production path.
    "hybrid": {},
    # What each piece of the production path is worth.
    "hybrid-no-poststrat": {"estimator": {"poststratify": False, "draws": 0}},
    "hybrid-no-pooling": {"estimator": {"mrp": False}},
    "hybrid-cell-mean": {"predictor": {"propagator": "cell_mean"}},
    "hybrid-no-structure": {"predictor": {"propagator": "mean"}},
    "hybrid-independent-population": {"population": {"microdata": None, "source": "fixture"}},
    "hybrid-with-influence": {
        "world": {
            "network": {"enabled": True, "mean_degree": 8, "homophily": 0.7,
                        "dimensions": DEFAULT_STRATIFY},
            "dynamics": {"enabled": True, "rounds": 3, "confidence": 0.25,
                         "susceptibility": 0.2},
        }
    },
    # How much the archetype budget buys.
    "hybrid-100": {"predictor": {"archetypes": 100}},
    "hybrid-600": {"predictor": {"archetypes": 600}},
    # The thing the hybrid is a shortcut for. Deliberately on a small population,
    # because asking a model about every agent is exactly what does not scale.
    "llm-every-agent": {
        "population": {"size": 600},
        "predictor": {"kind": "llm"},
    },
}

#: The subset run by default. The full grid is available but slower and dearer.
DEFAULT_GRID = (
    "uniform",
    "prior",
    "hybrid-no-structure",
    "hybrid-cell-mean",
    "hybrid",
    "hybrid-no-poststrat",
    "hybrid-no-pooling",
    "hybrid-independent-population",
    "hybrid-with-influence",
    "hybrid-100",
    "hybrid-600",
    "llm-every-agent",
)


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` onto a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def build_spec(
    engine: str,
    group_id: str,
    arms: Sequence[Question],
    provider: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> SimulationSpec:
    """Build the spec that runs ``arms`` under configuration ``engine``.

    Every arm becomes an arm of one scenario, so they share a population and a seed.
    Their prompts are the verbatim question wording, which is the only thing that
    differs between them and therefore the only thing that can explain a predicted gap.
    """
    if engine not in ENGINES:
        raise KeyError(f"unknown engine {engine!r}; known: {sorted(ENGINES)}")
    if not arms:
        raise ValueError("a spec needs at least one arm")

    options = list(arms[0].options)
    mismatched = [a.id for a in arms if list(a.options) != options]
    if mismatched:
        raise ValueError(f"arms {mismatched} do not share the options of {arms[0].id}")

    payload = merge(BASE_SPEC, ENGINES[engine])
    if provider:
        payload = merge(payload, {"predictor": {"provider": provider}})
    if overrides:
        payload = merge(payload, overrides)
    payload["name"] = f"{engine}:{group_id}"
    payload["scenario"] = {
        "question_id": group_id,
        "options": options,
        "arms": [
            {"id": arm.id, "label": arm.arm_label or arm.id, "prompt": arm.text}
            for arm in arms
        ],
    }
    return SimulationSpec.from_dict(payload)
