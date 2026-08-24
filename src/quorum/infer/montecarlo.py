"""Uncertainty by resampling the agents that were actually measured.

:mod:`quorum.infer.mrp` derives its interval from a model: assume the cells are drawn
from a Dirichlet and the posterior follows. This module derives one from resampling
instead: draw a new archetype sample with replacement, refit the propagator on it,
predict the population again, and look at how far the answer moves.

Two routes to the same quantity, resting on different assumptions. Reporting both, and
whether they agree, is worth more than reporting either alone, because when a
parametric interval is wrong it is usually wrong quietly.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from quorum.core.population import Population

#: A callable that refits on a resampled archetype set and predicts for the frame.
Refit = Callable[[Population, np.ndarray], np.ndarray]


def bootstrap_draws(
    frame: Population,
    sample: Population,
    responses: np.ndarray,
    refit: Refit,
    draws: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Return ``(draws, n_options)`` toplines from resampled archetype sets.

    The resampling is over archetypes, not over agents. Resampling the synthetic
    population would measure how big the population is, which the author chose, rather
    than how much was measured, which is the thing in doubt.
    """
    responses = np.asarray(responses, dtype=float)
    if responses.shape[0] != len(sample):
        raise ValueError(f"expected {len(sample)} response rows, got {responses.shape[0]}")
    if draws < 1:
        raise ValueError("draws must be at least 1")

    rng = np.random.default_rng(seed)
    n = len(sample)
    out = np.empty((draws, responses.shape[1]))
    frame_weights = frame.weights

    for draw in range(draws):
        picked = rng.integers(0, n, size=n)
        resampled = sample.subset(_mask_from_indices(picked, n))
        # subset() drops duplicates, so weight each kept archetype by how many times
        # it was drawn. That is what makes this a bootstrap rather than a subsample.
        multiplicity = np.bincount(picked, minlength=n)
        kept = multiplicity > 0
        resampled = resampled.with_weights(sample.weights[kept] * multiplicity[kept])
        predicted = refit(resampled, responses[kept])
        out[draw] = (predicted * frame_weights[:, None]).sum(axis=0) / frame_weights.sum()
    return out


def interval_agreement(a: np.ndarray, b: np.ndarray, level: float = 0.90) -> float:
    """How closely two sets of draws agree on their intervals, in share units.

    The largest absolute difference between the two intervals' endpoints. Small means
    the parametric and resampled routes tell the same story; large means at least one
    of them rests on an assumption that is not holding.
    """
    lo = (1.0 - level) / 2.0
    left = np.quantile(a, [lo, 1.0 - lo], axis=0)
    right = np.quantile(b, [lo, 1.0 - lo], axis=0)
    return float(np.max(np.abs(left - right)))


def _mask_from_indices(indices: np.ndarray, n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[indices] = True
    return mask
