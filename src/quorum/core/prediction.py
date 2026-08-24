"""Prediction objects: what the simulation loop produces and what gets scored.

Everything downstream of response prediction (estimation, scoring, publication)
consumes these two types, which is what lets the predictors be swapped freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class ResponseDistribution:
    """A probability distribution over a question's response options.

    Not a single choice. A simulated respondent that returns only its modal answer
    throws away the information that makes an aggregate topline calibrated, so an
    agent's response is a distribution and aggregation is a weighted average of them.
    """

    options: tuple[str, ...]
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        p = np.asarray(self.probabilities, dtype=float)
        object.__setattr__(self, "probabilities", p)
        if p.shape != (len(self.options),):
            raise ValueError(
                f"expected {len(self.options)} probabilities for {len(self.options)} "
                f"options, got shape {p.shape}"
            )
        if not np.all(np.isfinite(p)):
            raise ValueError("probabilities must be finite")
        if np.any(p < -_TOLERANCE):
            raise ValueError("probabilities must be non-negative")
        if abs(p.sum() - 1.0) > 1e-3:
            raise ValueError(f"probabilities must sum to 1, got {p.sum():.6f}")

    @classmethod
    def uniform(cls, options: Sequence[str]) -> "ResponseDistribution":
        options = tuple(options)
        return cls(options, np.full(len(options), 1.0 / len(options)))

    @classmethod
    def from_mapping(
        cls, mapping: dict[str, float], options: Sequence[str]
    ) -> "ResponseDistribution":
        """Build from a partial ``{option: probability}`` mapping, renormalizing.

        LLMs return sloppy distributions: missing options, probabilities summing to
        0.97 or 1.04. Normalizing here, once, keeps that leniency out of every caller.
        """
        options = tuple(options)
        raw = np.array([max(0.0, float(mapping.get(o, 0.0))) for o in options])
        total = raw.sum()
        if total <= 0:
            raise ValueError(f"mapping {mapping!r} has no mass on options {options!r}")
        return cls(options, raw / total)

    def __getitem__(self, option: str) -> float:
        return float(self.probabilities[self.options.index(option)])

    @property
    def mode(self) -> str:
        return self.options[int(np.argmax(self.probabilities))]

    def as_dict(self) -> dict[str, float]:
        return {o: float(p) for o, p in zip(self.options, self.probabilities)}


@dataclass(frozen=True, slots=True)
class Prediction:
    """A population-level answer, with uncertainty and provenance.

    ``draws`` carries the posterior or Monte Carlo sample the interval is computed
    from; keeping the draws rather than only their summary is what lets calibration,
    coverage and treatment-effect contrasts be computed after the fact without
    re-running the simulation.
    """

    question_id: str
    options: tuple[str, ...]
    distribution: np.ndarray
    draws: np.ndarray | None = None
    segments: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        d = np.asarray(self.distribution, dtype=float)
        object.__setattr__(self, "distribution", d)
        if d.shape != (len(self.options),):
            raise ValueError(f"distribution shape {d.shape} does not match options")
        if abs(d.sum() - 1.0) > 1e-3:
            raise ValueError(f"distribution must sum to 1, got {d.sum():.6f}")
        if self.draws is not None:
            draws = np.asarray(self.draws, dtype=float)
            object.__setattr__(self, "draws", draws)
            if draws.ndim != 2 or draws.shape[1] != len(self.options):
                raise ValueError(
                    f"draws must be (n_draws, {len(self.options)}), got {draws.shape}"
                )

    def as_distribution(self) -> ResponseDistribution:
        return ResponseDistribution(self.options, self.distribution)

    def interval(self, level: float = 0.90) -> np.ndarray:
        """Equal-tailed credible interval per option, shape ``(n_options, 2)``.

        Falls back to a zero-width interval when the prediction carries no draws,
        so callers do not have to branch. ``has_uncertainty`` distinguishes the cases.
        """
        if self.draws is None:
            return np.stack([self.distribution, self.distribution], axis=1)
        lo = (1.0 - level) / 2.0
        return np.quantile(self.draws, [lo, 1.0 - lo], axis=0).T

    @property
    def has_uncertainty(self) -> bool:
        return self.draws is not None

    def share(self, option: str) -> float:
        return float(self.distribution[self.options.index(option)])

    def as_dict(self) -> dict[str, float]:
        return {o: float(p) for o, p in zip(self.options, self.distribution)}
