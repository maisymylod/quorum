"""Baseline predictors: the bar a real simulation has to clear.

An accuracy number means nothing on its own. Predicting that a three-option spending
question splits close to the average of every other three-option spending question is
already right to within a few points, and any engine that cannot beat that is not
earning its cost. These are here so the headline is always a comparison.

The prior baseline is fitted on a calibration split and scored on a held-out one.
Fitting it on the questions it is then scored against would hand it the answers.
"""

from __future__ import annotations

import numpy as np

from quorum.core.population import Population
from quorum.data.targets import QuestionBank
from quorum.world.context import Scenario


class UniformPredictor:
    """Every option equally likely. The floor."""

    name = "uniform"

    def predict(self, population: Population, scenario: Scenario, seed: int) -> np.ndarray:
        k = len(scenario.options)
        return np.full((len(population), k), 1.0 / k)


class PriorPredictor:
    """Predict the average answer to questions of this shape.

    A strong baseline and an uncomfortable one: on a question set where most items
    lean the same way, it is hard to beat without actually modelling the question.
    """

    name = "prior"

    def __init__(self, priors: dict[int, np.ndarray]) -> None:
        self.priors = {int(k): np.asarray(v, dtype=float) for k, v in priors.items()}

    @classmethod
    def fit(cls, bank: QuestionBank) -> "PriorPredictor":
        """Average the toplines of every question, grouped by option count.

        Grouping by option count rather than pooling everything is what keeps a
        two-option question from being predicted with a three-option average.
        """
        grouped: dict[int, list[np.ndarray]] = {}
        for question in bank:
            grouped.setdefault(len(question.options), []).append(question.topline)
        if not grouped:
            raise ValueError("cannot fit a prior from an empty question bank")
        return cls({k: np.mean(np.stack(v), axis=0) for k, v in grouped.items()})

    def predict(self, population: Population, scenario: Scenario, seed: int) -> np.ndarray:
        k = len(scenario.options)
        prior = self.priors.get(k)
        if prior is None:
            prior = np.full(k, 1.0 / k)
        return np.tile(prior / prior.sum(), (len(population), 1))
