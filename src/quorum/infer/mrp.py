"""Multilevel regression and poststratification, and the simpler estimator it replaces.

Poststratification is the step that turns "what the agents we asked said" into "what
the population would say". The sample that was actually measured is not the population:
the archetype sample deliberately over-represents small cells so that every cell is
covered at all. Reweighting each cell's estimate by that cell's real share of the
population undoes that by construction.

The multilevel half is :mod:`quorum.infer.pooling`. Together they are MRP: model the
cells with partial pooling, then weight them by how common they really are.

Uncertainty comes from the sample, which is where it actually lives. A synthetic
population can be made as large as anyone likes, so the spread of a topline over
100,000 agents says nothing; the number of agents a model was actually asked about
says everything. The posterior draws reflect the second.
"""

from __future__ import annotations

import numpy as np

from quorum.core.population import Population
from quorum.core.prediction import Prediction
from quorum.infer.pooling import PooledCells, pool_cells, posterior_draws
from quorum.world.context import Scenario

#: Concentration used when partial pooling is switched off for the ablation.
_NO_POOLING = 1e-6


class DirectEstimator:
    """Aggregate agent responses with population weights, and claim no uncertainty.

    The honest minimum. Used as the control in the ablation, and whenever a predictor
    exposes no sample for the uncertainty to be derived from.
    """

    name = "direct"

    def estimate(
        self, population: Population, responses: np.ndarray, scenario: Scenario, seed: int
    ) -> Prediction:
        distribution = population.weighted_distribution(responses)
        return Prediction(
            question_id=scenario.question_id,
            options=scenario.options,
            distribution=distribution,
            metadata={"estimator": self.name, "arm": scenario.arm},
        )


class MRPEstimator:
    """Partial-pool cell estimates from the sample, then poststratify onto the frame."""

    name = "mrp"

    def __init__(
        self,
        dimensions: tuple[str, ...],
        draws: int = 2000,
        level: float = 0.90,
        concentration: float | None = None,
        pool: bool = True,
    ) -> None:
        if not dimensions:
            raise ValueError("poststratification needs at least one dimension")
        if not 0.0 < level < 1.0:
            raise ValueError("level must be between 0 and 1")
        self.dimensions = tuple(dimensions)
        self.draws = draws
        self.level = level
        self.concentration = concentration
        self.pool = pool
        self.last_pooled: PooledCells | None = None

    def estimate(
        self,
        frame: Population,
        sample: Population,
        sample_responses: np.ndarray,
        scenario: Scenario,
        seed: int,
    ) -> Prediction:
        """Estimate the population answer from what the sample said.

        Parameters
        ----------
        frame:
            The full synthetic population. Supplies the cell weights, and nothing else.
        sample:
            The agents that were actually measured.
        sample_responses:
            Their response distributions, one row each.
        """
        sample_responses = np.asarray(sample_responses, dtype=float)
        if sample_responses.shape[0] != len(sample):
            raise ValueError(
                f"expected {len(sample)} response rows, got {sample_responses.shape[0]}"
            )

        dimensions = list(self.dimensions)
        cells = frame.cells(dimensions)
        cell_weights = cells["share"].to_numpy(dtype=float)
        n_cells = len(cells)

        # Evidence per cell, in units of agents measured. Sample weights are
        # normalized so a cell's evidence is a count of respondents rather than a
        # count of the millions of people they stand for, which would make every
        # posterior absurdly tight.
        lookup = {tuple(row): index for index, row in enumerate(cells[dimensions].to_numpy())}
        sample_cells = np.array(
            [lookup.get(tuple(row), -1) for row in sample.frame[dimensions].to_numpy()]
        )
        weights = sample.weights
        weights = weights / weights.mean() if weights.mean() > 0 else weights

        counts = np.zeros((n_cells, sample_responses.shape[1]))
        known = sample_cells >= 0
        np.add.at(counts, sample_cells[known], sample_responses[known] * weights[known, None])

        # Pooling off means each cell speaks only for itself. A concentration this
        # small leaves an occupied cell at its own average and an empty one at the
        # global average, which is the no-pooling estimator with the degenerate case
        # handled rather than left as a division by zero.
        concentration = self.concentration if self.pool else _NO_POOLING
        pooled = pool_cells(counts, concentration=concentration)
        self.last_pooled = pooled

        point = cell_weights @ pooled.posterior_mean
        draws = None
        if self.draws > 0:
            samples = posterior_draws(pooled, self.draws, seed)
            draws = np.einsum("c,dco->do", cell_weights, samples)

        segments = self._segments(frame, pooled, dimensions)
        return Prediction(
            question_id=scenario.question_id,
            options=scenario.options,
            distribution=point / point.sum(),
            draws=draws,
            segments=segments,
            metadata={
                "estimator": self.name,
                "arm": scenario.arm,
                "cells": n_cells,
                "occupied_cells": int((counts.sum(axis=1) > 0).sum()),
                "concentration": round(float(pooled.concentration), 4),
                "mean_shrinkage": round(float(pooled.shrinkage.mean()), 4),
                "concentration_at_bound": bool(pooled.at_bound),
                "level": self.level,
            },
        )

    def _segments(
        self, frame: Population, pooled: PooledCells, dimensions: list[str]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Break the answer down by each poststratification dimension.

        Built by reweighting the same pooled cells rather than by re-estimating, so a
        segment breakdown always adds back up to the topline it sits under.
        """
        cells = frame.cells(dimensions)
        weights = cells["share"].to_numpy(dtype=float)
        out: dict[str, dict[str, np.ndarray]] = {}
        for dimension in dimensions:
            levels = cells[dimension].to_numpy()
            by_level: dict[str, np.ndarray] = {}
            for level in dict.fromkeys(levels):
                # cells() only returns occupied cells, so the mass is always positive.
                mask = levels == level
                mass = weights[mask].sum()
                by_level[str(level)] = (weights[mask] @ pooled.posterior_mean[mask]) / mass
            out[dimension] = by_level
        return out
