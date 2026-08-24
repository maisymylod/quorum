"""Population synthesizers: target marginals in, a weighted synthetic population out.

Two implementations, and the difference between them is the point. Both hit the target
marginals exactly, because both finish by raking. Only one carries the joint structure
of the real population, and the ablation grid uses the pair to show what that structure
is worth: whether knowing that a level of education travels with an age band, rather
than treating the two as independent, changes the answer the simulation gives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quorum.core.population import TRAIT_PREFIX, Population
from quorum.data.schema import LEVELS
from quorum.data.targets import MarginalTargets
from quorum.synthesis.ipf import RakingResult, encode, rake


class _BaseSynthesizer:
    """Shared machinery: trait assignment and the final rake onto targets."""

    def __init__(
        self,
        targets: MarginalTargets,
        attributes: tuple[str, ...],
        traits: tuple[str, ...] = (),
        max_iterations: int = 200,
        tolerance: float = 1e-9,
    ) -> None:
        missing = [a for a in attributes if a not in targets.marginals]
        if missing:
            raise KeyError(f"no target marginals for {missing}")
        self.targets = targets
        self.attributes = tuple(attributes)
        self.traits = tuple(traits)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.last_raking: RakingResult | None = None

    def _finish(self, frame: pd.DataFrame, seed: int, name: str) -> Population:
        rng = np.random.default_rng(seed + 1)
        for trait in self.traits:
            frame[TRAIT_PREFIX + trait] = rng.random(len(frame))

        codes = {a: encode(frame[a].tolist(), LEVELS[a]) for a in self.attributes}
        target_vectors = {a: self.targets.vector(a) for a in self.attributes}
        result = rake(
            codes,
            target_vectors,
            initial_weights=frame["weight"].to_numpy(dtype=float),
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )
        self.last_raking = result
        frame["weight"] = result.weights
        # Weights are rescaled so they sum to the represented population rather than
        # to the agent count. Downstream everything uses shares, but a report that
        # says "12.4 million adults" needs the scale to be real.
        frame["weight"] *= self.targets.population_total / frame["weight"].sum()
        columns = list(self.attributes) + [TRAIT_PREFIX + t for t in self.traits] + ["weight"]
        return Population(frame[columns], name=name)


class MicrodataSynthesizer(_BaseSynthesizer):
    """Resample real microdata, then rake it onto the targets.

    Sampling with replacement from a weighted microdata seed reproduces the joint
    distribution of the source. Raking then corrects the margins, which drift both
    from sampling noise and from the seed being a subsample of a single survey year.
    """

    def __init__(
        self,
        seed_frame: pd.DataFrame,
        targets: MarginalTargets,
        attributes: tuple[str, ...],
        traits: tuple[str, ...] = (),
        **kwargs,
    ) -> None:
        super().__init__(targets, attributes, traits, **kwargs)
        missing = [c for c in self.attributes if c not in seed_frame.columns]
        if missing:
            raise KeyError(f"microdata seed is missing columns {missing}")
        if "weight" not in seed_frame.columns:
            raise KeyError("microdata seed is missing a weight column")
        self.seed_frame = seed_frame.reset_index(drop=True)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        targets: MarginalTargets,
        attributes: tuple[str, ...],
        traits: tuple[str, ...] = (),
        **kwargs,
    ) -> "MicrodataSynthesizer":
        return cls(pd.read_csv(path), targets, attributes, traits, **kwargs)

    def synthesize(self, size: int, seed: int) -> Population:
        if size < 1:
            raise ValueError("size must be at least 1")
        rng = np.random.default_rng(seed)
        weights = self.seed_frame["weight"].to_numpy(dtype=float)
        probabilities = weights / weights.sum()
        picked = rng.choice(len(self.seed_frame), size=size, replace=True, p=probabilities)
        frame = self.seed_frame.iloc[picked].reset_index(drop=True).copy()
        frame["weight"] = 1.0
        return self._finish(frame, seed, name="microdata")


class IndependenceSynthesizer(_BaseSynthesizer):
    """Draw each attribute independently from its marginal, then rake.

    A deliberate straw man. It matches every one-way margin exactly and gets every
    interaction between them wrong, which makes it the control that shows whether the
    joint structure in real microdata is doing any work.
    """

    def synthesize(self, size: int, seed: int) -> Population:
        if size < 1:
            raise ValueError("size must be at least 1")
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame(index=range(size))
        for attribute in self.attributes:
            levels = LEVELS[attribute]
            frame[attribute] = rng.choice(
                list(levels), size=size, p=self.targets.vector(attribute)
            )
        frame["weight"] = 1.0
        return self._finish(frame, seed, name="independence")
