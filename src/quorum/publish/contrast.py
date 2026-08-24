"""Comparing two arms of a simulation.

The number a decision actually turns on is usually not a topline but a difference:
does this wording move people, and by how much, and is the movement bigger than what
the simulation can resolve? Reporting the two toplines and leaving the reader to
subtract them hides the last part, because the difference has its own uncertainty and
it is not the sum of the two.

Both arms are estimated from the same population, so their draws are dependent. Taking
the difference draw by draw keeps that dependence instead of assuming it away, which is
what makes the resulting interval narrower, and correct.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quorum.core.prediction import Prediction


@dataclass(frozen=True, slots=True)
class Contrast:
    """The difference between two arms, per response option."""

    reference: str
    other: str
    options: tuple[str, ...]
    difference: np.ndarray
    draws: np.ndarray | None = None

    @classmethod
    def between(
        cls, reference_arm: str, reference: Prediction, other_arm: str, other: Prediction
    ) -> "Contrast":
        if reference.options != other.options:
            raise ValueError("arms must share their response options to be contrasted")
        draws = None
        if reference.draws is not None and other.draws is not None:
            n = min(len(reference.draws), len(other.draws))
            draws = other.draws[:n] - reference.draws[:n]
        return cls(
            reference=reference_arm,
            other=other_arm,
            options=reference.options,
            difference=other.distribution - reference.distribution,
            draws=draws,
        )

    def interval(self, level: float = 0.90) -> np.ndarray:
        if self.draws is None:
            return np.stack([self.difference, self.difference], axis=1)
        lo = (1.0 - level) / 2.0
        return np.quantile(self.draws, [lo, 1.0 - lo], axis=0).T

    def resolves(self, option: str, level: float = 0.90) -> bool:
        """Whether the interval for ``option`` excludes zero.

        Not "is the effect real", which a simulation cannot establish. It is "did the
        simulation resolve a direction at all", which is the most a reader should take
        from it and more than a bare point estimate offers.

        A contrast with no draws has no interval, and a zero-width interval around a
        non-zero point excludes zero trivially. Answering "yes" there would turn an
        absence of uncertainty information into a claim of confidence, which is the
        exact overstatement this method exists to prevent.
        """
        if self.draws is None:
            return False
        index = self.options.index(option)
        low, high = self.interval(level)[index]
        return bool(low > 0 or high < 0)

    def shift(self, option: str) -> float:
        return float(self.difference[self.options.index(option)])
