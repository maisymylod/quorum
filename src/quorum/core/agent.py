"""The agent object.

An ``Agent`` is a *view* onto one row of a :class:`~quorum.core.population.Population`,
not the unit of storage. Populations are held columnar so that a 100k-agent run is a
handful of numpy arrays rather than 100k Python objects; ``Agent`` exists for the places
that genuinely need a single respondent at a time, above all prompt rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Agent:
    """One simulated respondent.

    Attributes
    ----------
    id:
        Stable index into the owning population. Deterministic across runs with the
        same seed and spec.
    attributes:
        Observed characteristics that the population was synthesized on, e.g.
        ``{"age_band": "35-44", "education": "bachelors"}``. These are the dimensions
        that marginals and poststratification are defined over.
    traits:
        Latent numeric dispositions in ``[0, 1]``, drawn during synthesis. They give
        agents within the same demographic cell room to differ, which is what keeps a
        simulated topline from collapsing onto a single cell-level point estimate.
    weight:
        Survey weight. The population represents a real target population only when
        aggregated with these weights.
    """

    id: int
    attributes: Mapping[str, Any]
    traits: Mapping[str, float] = field(default_factory=dict)
    weight: float = 1.0

    def cell(self, dimensions: tuple[str, ...]) -> tuple[Any, ...]:
        """Return this agent's poststratification cell over ``dimensions``."""
        return tuple(self.attributes[d] for d in dimensions)

    def describe(self, dimensions: tuple[str, ...] | None = None) -> str:
        """Render the agent as a short natural-language persona line.

        Used for prompt construction. Deterministic: dimension order is the order
        given, or the population's column order, never a dict iteration accident.
        """
        dims = dimensions if dimensions is not None else tuple(self.attributes)
        parts = [f"{d.replace('_', ' ')}: {self.attributes[d]}" for d in dims]
        return "; ".join(parts)
