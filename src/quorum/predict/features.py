"""Turning agents into a design matrix.

One place decides how an agent becomes numbers, because every model that consumes
agents has to agree on it. Column order comes from the declared level order in the
taxonomy rather than from whatever levels happen to appear in a given sample, so a
model fitted on 300 archetypes can be applied to 100,000 agents without the columns
silently shifting underneath it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quorum.core.population import TRAIT_PREFIX, Population
from quorum.data.schema import LEVELS


@dataclass(frozen=True, slots=True)
class DesignSpace:
    """A fixed encoding of agents into features.

    Parameters
    ----------
    attributes:
        Categorical attributes to one-hot encode. The first level of each is dropped,
        so the intercept is identified and the matrix is not collinear.
    traits:
        Latent trait columns to pass through as continuous features.
    """

    attributes: tuple[str, ...]
    traits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = [a for a in self.attributes if a not in LEVELS]
        if unknown:
            raise KeyError(f"unknown attributes {unknown}")

    @property
    def columns(self) -> tuple[str, ...]:
        names: list[str] = []
        for attribute in self.attributes:
            names.extend(f"{attribute}={level}" for level in LEVELS[attribute][1:])
        names.extend(f"trait:{t}" for t in self.traits)
        return tuple(names)

    def encode(self, population: Population) -> np.ndarray:
        """Return the ``(n_agents, n_features)`` design matrix for ``population``."""
        missing = [a for a in self.attributes if a not in population.attributes]
        if missing:
            raise KeyError(f"population is missing attributes {missing}")
        missing_traits = [t for t in self.traits if t not in population.traits]
        if missing_traits:
            raise KeyError(f"population is missing traits {missing_traits}")

        frame = population.frame
        blocks: list[np.ndarray] = []
        for attribute in self.attributes:
            values = frame[attribute].to_numpy()
            levels = LEVELS[attribute][1:]
            blocks.append(
                np.stack([(values == level).astype(float) for level in levels], axis=1)
                if levels
                else np.zeros((len(frame), 0))
            )
        for trait in self.traits:
            blocks.append(frame[TRAIT_PREFIX + trait].to_numpy(dtype=float)[:, None])
        if not blocks:
            return np.zeros((len(frame), 0))
        return np.concatenate(blocks, axis=1)
