"""Propagators: carry a handful of measured responses to a whole population.

This is the machinery that makes simulating a large population affordable. Something
expensive produces response distributions for a stratified sample of agents; a
propagator learns the relationship between agent features and those responses, and
scores everybody else for free. The expensive step then costs the sample size rather
than the population size, and the population size stops being a budget question.

Three implementations, in increasing order of how much structure they assume. They
exist as a set because which one is right is an empirical question the ablation grid
settles, and because the cheapest of them is a fair control for the other two.
"""

from __future__ import annotations

import numpy as np

from quorum.core.population import Population
from quorum.predict.features import DesignSpace

_EPSILON = 1e-9


def _validate(sample: Population, responses: np.ndarray) -> np.ndarray:
    responses = np.asarray(responses, dtype=float)
    if responses.ndim != 2 or responses.shape[0] != len(sample):
        raise ValueError(
            f"expected a ({len(sample)}, n_options) response matrix, got {responses.shape}"
        )
    if responses.shape[1] < 2:
        raise ValueError("responses need at least two options")
    return responses


class MeanPropagator:
    """Give every agent the sample's average response.

    The control. It uses the measured responses and nothing about who gave them, so
    any accuracy a richer propagator has over this one is accuracy that came from
    modelling the population rather than from the sample average.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None

    def fit(self, sample: Population, responses: np.ndarray) -> "MeanPropagator":
        responses = _validate(sample, responses)
        self.mean_ = sample.weighted_distribution(responses)
        return self

    def predict(self, population: Population) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("propagator must be fitted before predicting")
        return np.tile(self.mean_, (len(population), 1))


class CellMeanPropagator:
    """Average the sample's responses within each cell of ``dimensions``.

    Assumes nothing beyond the cells themselves, which makes it robust and blunt: a
    cell the sample never reached falls back to the overall mean, and a cell reached
    once inherits that single agent's answer with no shrinkage.
    """

    def __init__(self, dimensions: tuple[str, ...]) -> None:
        if not dimensions:
            raise ValueError("CellMeanPropagator needs at least one dimension")
        self.dimensions = tuple(dimensions)
        self.cell_means_: dict[tuple, np.ndarray] = {}
        self.fallback_: np.ndarray | None = None

    def fit(self, sample: Population, responses: np.ndarray) -> "CellMeanPropagator":
        responses = _validate(sample, responses)
        frame = sample.frame
        weights = sample.weights
        keys = list(map(tuple, frame[list(self.dimensions)].to_numpy()))
        totals: dict[tuple, np.ndarray] = {}
        counts: dict[tuple, float] = {}
        for index, key in enumerate(keys):
            weight = weights[index]
            totals[key] = totals.get(key, 0.0) + responses[index] * weight
            counts[key] = counts.get(key, 0.0) + weight
        self.cell_means_ = {k: totals[k] / counts[k] for k in totals}
        self.fallback_ = sample.weighted_distribution(responses)
        return self

    def predict(self, population: Population) -> np.ndarray:
        if self.fallback_ is None:
            raise RuntimeError("propagator must be fitted before predicting")
        keys = map(tuple, population.frame[list(self.dimensions)].to_numpy())
        return np.stack([self.cell_means_.get(key, self.fallback_) for key in keys])


class MultinomialLogitPropagator:
    """Fit a softmax regression from agent features onto response distributions.

    The responses being fitted are distributions, not choices, and throwing that away
    by taking each agent's modal answer would discard most of the signal a small
    sample carries. Instead each sampled agent is expanded into one weighted row per
    option, so a respondent who was 60/40 between two answers trains the model as
    exactly that rather than as a hard vote.
    """

    def __init__(self, design: DesignSpace, regularization: float = 1.0, max_iter: int = 500) -> None:
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        self.design = design
        self.regularization = regularization
        self.max_iter = max_iter
        self.model_ = None
        self.fallback_: np.ndarray | None = None
        self.n_options_ = 0

    def fit(self, sample: Population, responses: np.ndarray) -> "MultinomialLogitPropagator":
        from sklearn.linear_model import LogisticRegression

        responses = _validate(sample, responses)
        self.n_options_ = responses.shape[1]
        self.fallback_ = sample.weighted_distribution(responses)

        features = self.design.encode(sample)
        n, k = responses.shape
        rows = np.repeat(features, k, axis=0)
        labels = np.tile(np.arange(k), n)
        weights = (responses * sample.weights[:, None]).reshape(-1)

        live = weights > _EPSILON
        # An option no sampled agent gave any weight to cannot be fitted, and a model
        # with a single surviving class is not a model. Fall back rather than crash.
        if features.shape[1] == 0 or len(np.unique(labels[live])) < 2:
            self.model_ = None
            return self

        # Recent scikit-learn is multinomial by default and has dropped the
        # multi_class argument, so it is deliberately not passed.
        model = LogisticRegression(C=1.0 / self.regularization, max_iter=self.max_iter)
        model.fit(rows[live], labels[live], sample_weight=weights[live])
        self.model_ = model
        return self

    def predict(self, population: Population) -> np.ndarray:
        if self.fallback_ is None:
            raise RuntimeError("propagator must be fitted before predicting")
        if self.model_ is None:
            return np.tile(self.fallback_, (len(population), 1))
        probabilities = self.model_.predict_proba(self.design.encode(population))
        # The fitted classes may be a subset of the options if the sample never
        # touched one. Scatter back into full option space so the matrix shape is a
        # property of the question, not of the sample.
        out = np.zeros((len(population), self.n_options_))
        out[:, self.model_.classes_] = probabilities
        return out / out.sum(axis=1, keepdims=True)


def apply_trait_noise(
    responses: np.ndarray, traits: np.ndarray, scale: float, seed: int
) -> np.ndarray:
    """Disperse responses within a cell using each agent's latent traits.

    A propagator can only separate agents its features can see, so every agent in a
    cell leaves it with an identical answer. Real people in the same cell disagree,
    and a simulation that reports them as unanimous will produce intervals that are
    too narrow. The perturbation is applied in log space and renormalized, so it
    cannot push a probability outside the simplex.
    """
    responses = np.asarray(responses, dtype=float)
    if scale <= 0:
        return responses.copy()
    rng = np.random.default_rng(seed)
    traits = np.asarray(traits, dtype=float)
    if traits.size == 0:
        direction = rng.normal(size=responses.shape)
    else:
        # Centre the traits so the perturbation has no systematic direction: it should
        # widen the spread of answers within a cell without moving the cell's mean.
        centred = traits - traits.mean(axis=0, keepdims=True)
        loadings = rng.normal(size=(centred.shape[1], responses.shape[1]))
        direction = centred @ loadings
    perturbed = np.log(np.clip(responses, _EPSILON, None)) + scale * direction
    perturbed = np.exp(perturbed - perturbed.max(axis=1, keepdims=True))
    return perturbed / perturbed.sum(axis=1, keepdims=True)
