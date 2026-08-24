"""Raking: iterative proportional fitting of survey weights onto target marginals.

Sampling from microdata gets the joint structure of a population roughly right and its
margins only approximately right. Raking fixes the margins exactly, one attribute at a
time, in a loop: scale every agent's weight by the ratio of the target share to the
achieved share for the level it occupies, repeat over attributes, iterate until nothing
moves. The joint structure the sample brought with it survives, because every agent in
a given cell is scaled by the same factor.

The procedure has a known failure mode. If the sample contains no agent in some level
that the targets require, no reweighting can produce one, and the loop will happily run
forever chasing a share it cannot reach. That case is detected and reported rather than
silently converged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass(slots=True)
class RakingResult:
    """Outcome of a raking run, including the evidence that it worked."""

    weights: np.ndarray
    iterations: int
    converged: bool
    max_deviation: float
    history: list[float] = field(default_factory=list)
    empty_levels: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def summary(self) -> str:
        state = "converged" if self.converged else "did not converge"
        return (
            f"raking {state} after {self.iterations} iterations, "
            f"max marginal deviation {self.max_deviation:.2e}"
        )


def rake(
    codes: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    initial_weights: np.ndarray | None = None,
    max_iterations: int = 200,
    tolerance: float = 1e-9,
) -> RakingResult:
    """Scale weights until every attribute's weighted shares hit its targets.

    Parameters
    ----------
    codes:
        Per attribute, an integer array giving each agent's level index. Integer codes
        rather than labels because the inner loop runs once per attribute per
        iteration and :func:`numpy.bincount` on codes is the whole cost of it.
    targets:
        Per attribute, the target share of each level, in level-index order. Must sum
        to 1 within each attribute.
    initial_weights:
        Starting weights. Defaults to uniform. Passing the sample's own design weights
        here is what makes this a reweighting of a real sample rather than a fresh
        allocation.

    Returns
    -------
    RakingResult
        Carries the weights and the convergence evidence. Callers are expected to
        check :attr:`RakingResult.converged`, and the fidelity gate does.
    """
    attributes = list(targets)
    if not attributes:
        raise ValueError("raking needs at least one target attribute")

    for attribute in attributes:
        if attribute not in codes:
            raise KeyError(f"no agent codes supplied for attribute {attribute!r}")
    n = len(codes[attributes[0]])
    for attribute in attributes:
        if len(codes[attribute]) != n:
            raise ValueError(f"codes for {attribute!r} have a different length")
        total = float(np.sum(targets[attribute]))
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"targets for {attribute!r} sum to {total:.8f}, not 1")

    weights = (
        np.ones(n, dtype=float) if initial_weights is None
        else np.asarray(initial_weights, dtype=float).copy()
    )
    if weights.shape != (n,):
        raise ValueError(f"initial_weights must have shape ({n},)")
    if np.any(weights < 0):
        raise ValueError("initial_weights must be non-negative")

    # A level the sample cannot represent is unreachable by any reweighting. Recording
    # it up front turns a silent non-convergence into an explained one.
    empty: dict[str, tuple[str, ...]] = {}
    for attribute in attributes:
        target = np.asarray(targets[attribute], dtype=float)
        present = np.bincount(codes[attribute], minlength=len(target)) > 0
        missing = np.flatnonzero((target > 0) & ~present)
        if missing.size:
            empty[attribute] = tuple(str(i) for i in missing)

    history: list[float] = []
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        for attribute in attributes:
            target = np.asarray(targets[attribute], dtype=float)
            code = codes[attribute]
            achieved = np.bincount(code, weights=weights, minlength=len(target))
            total = achieved.sum()
            if total <= 0:
                raise ValueError("all weights collapsed to zero during raking")
            achieved = achieved / total
            # Levels with no representation stay at their current weight rather than
            # multiplying by an infinite factor.
            factor = np.ones_like(target)
            live = achieved > 0
            factor[live] = target[live] / achieved[live]
            weights = weights * factor[code]

        deviation = _max_deviation(codes, targets, weights)
        history.append(deviation)
        if deviation <= tolerance:
            converged = True
            break

    return RakingResult(
        weights=weights,
        iterations=iterations,
        converged=converged,
        max_deviation=history[-1] if history else float("inf"),
        history=history,
        empty_levels=empty,
    )


def _max_deviation(
    codes: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray], weights: np.ndarray
) -> float:
    worst = 0.0
    for attribute, target in targets.items():
        achieved = np.bincount(codes[attribute], weights=weights, minlength=len(target))
        achieved = achieved / achieved.sum()
        worst = max(worst, float(np.max(np.abs(achieved - np.asarray(target, dtype=float)))))
    return worst


def encode(values: Sequence[str], levels: Sequence[str]) -> np.ndarray:
    """Map labels onto level indices, rejecting anything outside ``levels``."""
    lookup = {level: index for index, level in enumerate(levels)}
    try:
        return np.array([lookup[v] for v in values], dtype=int)
    except KeyError as exc:  # pragma: no cover - guarded upstream by validate_levels
        raise ValueError(f"value {exc.args[0]!r} is not one of {list(levels)}") from exc
