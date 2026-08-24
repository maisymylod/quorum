"""Did synthesis actually produce the population it was asked for?

Raking reports whether its own loop converged, which is not the same question. These
metrics compare the finished population against the targets it was supposed to match,
and are enforced as a gate rather than printed: a population that misses its margins
is a wrong answer waiting to happen, and it should fail loudly at synthesis time
rather than quietly at scoring time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quorum.core.population import Population
from quorum.data.schema import LEVELS
from quorum.data.targets import MarginalTargets


@dataclass(frozen=True, slots=True)
class AttributeFidelity:
    """How closely one attribute's achieved shares match its targets."""

    attribute: str
    total_absolute_error: float
    max_deviation: float
    srmse: float
    achieved: np.ndarray
    target: np.ndarray

    def worst_level(self, levels: tuple[str, ...]) -> tuple[str, float]:
        index = int(np.argmax(np.abs(self.achieved - self.target)))
        return levels[index], float(self.achieved[index] - self.target[index])


@dataclass(frozen=True, slots=True)
class FidelityReport:
    """Fidelity across every attribute, plus the headline numbers the gate uses."""

    attributes: dict[str, AttributeFidelity]
    population_size: int

    @property
    def total_absolute_error(self) -> float:
        """Summed absolute share error across all attributes.

        In share units, so a value of 0.01 means one percentage point of misallocated
        population summed over every level of every attribute.
        """
        return float(sum(a.total_absolute_error for a in self.attributes.values()))

    @property
    def max_deviation(self) -> float:
        """Largest single-level share error anywhere. The strictest of the three."""
        return float(max(a.max_deviation for a in self.attributes.values()))

    @property
    def srmse(self) -> float:
        """Worst standardized root mean square error across attributes.

        Standardizing by the mean target share is what makes an attribute with many
        small levels comparable to one with two large levels.
        """
        return float(max(a.srmse for a in self.attributes.values()))

    def as_dict(self) -> dict[str, float]:
        return {
            "total_absolute_error": self.total_absolute_error,
            "max_deviation": self.max_deviation,
            "srmse": self.srmse,
            "population_size": float(self.population_size),
        }

    def table(self) -> str:
        lines = [f"{'attribute':<14} {'TAE':>10} {'max dev':>10} {'SRMSE':>10}"]
        for name, fidelity in self.attributes.items():
            lines.append(
                f"{name:<14} {fidelity.total_absolute_error:>10.2e} "
                f"{fidelity.max_deviation:>10.2e} {fidelity.srmse:>10.2e}"
            )
        return "\n".join(lines)


def marginal_fidelity(population: Population, targets: MarginalTargets) -> FidelityReport:
    """Compare a population's weighted marginals against the targets it was raked to."""
    attributes = [a for a in population.attributes if a in targets.marginals]
    if not attributes:
        raise ValueError("population and targets share no attributes")

    results: dict[str, AttributeFidelity] = {}
    for attribute in attributes:
        levels = LEVELS[attribute]
        achieved_series = population.marginals(attribute)
        achieved = np.array([float(achieved_series.get(level, 0.0)) for level in levels])
        target = targets.vector(attribute)
        error = achieved - target
        mean_target = float(np.mean(target))
        results[attribute] = AttributeFidelity(
            attribute=attribute,
            total_absolute_error=float(np.sum(np.abs(error))),
            max_deviation=float(np.max(np.abs(error))),
            srmse=float(np.sqrt(np.mean(error**2)) / mean_target) if mean_target > 0 else float("inf"),
            achieved=achieved,
            target=target,
        )
    return FidelityReport(attributes=results, population_size=len(population))


def joint_divergence(
    population: Population, reference: Population, dimensions: list[str]
) -> float:
    """Total variation distance between two populations' joint cell distributions.

    Every synthesizer here matches the one-way margins by construction, so comparing
    them on margins says nothing. This compares the thing they actually differ on:
    how the attributes travel together.
    """
    left = population.cells(dimensions).set_index(dimensions)["share"]
    right = reference.cells(dimensions).set_index(dimensions)["share"]
    combined = left.align(right, fill_value=0.0)
    return float(0.5 * np.abs(combined[0].to_numpy() - combined[1].to_numpy()).sum())
