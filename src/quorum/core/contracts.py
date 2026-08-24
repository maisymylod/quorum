"""The interfaces the simulation product is assembled from.

Each stage of the loop is a Protocol, so any implementation that satisfies the shape
can be dropped in: a classical predictor swapped for an LLM one, an in-memory provider
swapped for a live API, a stub scorer swapped for the real harness. The runner and the
CLI are written against these names and nothing else, which is what makes the ablation
grid in :mod:`quorum.eval.ablation` a matter of configuration rather than code.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from quorum.core.population import Population
from quorum.core.prediction import Prediction


@runtime_checkable
class PopulationSynthesizer(Protocol):
    """Audience generation: targets in, a weighted synthetic population out."""

    def synthesize(self, size: int, seed: int) -> Population: ...


@runtime_checkable
class WorldModel(Protocol):
    """World modeling: what happens between a population and the answers it gives.

    Peer influence over a social graph, exposure to a stimulus, anything that makes
    one agent's answer depend on another's. Takes the population's own responses and
    returns revised ones, and never mutates either, so a run can branch across arms.
    """

    def influence(
        self, population: Population, responses: np.ndarray, seed: int
    ) -> np.ndarray: ...


@runtime_checkable
class ResponsePredictor(Protocol):
    """Response prediction: a per-agent distribution over the question's options.

    Returns an ``(n_agents, n_options)`` matrix. Aggregation is the population's job,
    not the predictor's, so that poststratification can happen after the fact.
    """

    def predict(self, population: Population, scenario: Any, seed: int) -> np.ndarray: ...


@runtime_checkable
class Estimator(Protocol):
    """Aggregation and uncertainty: agent-level responses to a population answer."""

    def estimate(
        self, population: Population, responses: np.ndarray, scenario: Any, seed: int
    ) -> Prediction: ...


@runtime_checkable
class Scorer(Protocol):
    """Accuracy: a prediction and a known answer in, named metrics out."""

    def score(self, prediction: Prediction, truth: Sequence[float]) -> dict[str, float]: ...


@runtime_checkable
class Publisher(Protocol):
    """Publication: raw output to a decision-ready artifact on disk."""

    def publish(self, prediction: Prediction, destination: str) -> str: ...


@runtime_checkable
class LLMProvider(Protocol):
    """A batched text-completion backend.

    Deliberately narrow. The simulation loop needs exactly one thing from a model
    provider, and holding the surface to one method is what makes the offline stub a
    genuine substitute rather than a partial one.
    """

    name: str
    model: str

    def complete(self, prompts: Sequence[str], system: str, max_tokens: int) -> list[Any]: ...
