"""Peer influence over the social graph.

Bounded confidence, not plain averaging. Each round an agent moves a little way toward
the average opinion of the neighbours it finds close enough to listen to, and ignores
the rest. The bound is what keeps the model from collapsing every population to
consensus, which is both what plain averaging does and what real populations do not.

Whether this helps prediction is not assumed. It is off by default and sits in the
ablation grid next to everything else, so the question is answered rather than argued.
"""

from __future__ import annotations

import numpy as np

from quorum.core.population import Population
from quorum.world.network import HomophilyNetwork, SocialGraph


class NoInfluence:
    """The null world model: agents answer independently."""

    def influence(
        self, population: Population, responses: np.ndarray, seed: int
    ) -> np.ndarray:
        return np.asarray(responses, dtype=float)


class BoundedConfidenceInfluence:
    """Move agents toward like-minded neighbours, ignoring distant ones."""

    def __init__(
        self,
        network: HomophilyNetwork,
        rounds: int = 3,
        confidence: float = 0.25,
        susceptibility: float = 0.2,
    ) -> None:
        if rounds < 0:
            raise ValueError("rounds must be non-negative")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.0 <= susceptibility <= 1.0:
            raise ValueError("susceptibility must be between 0 and 1")
        self.network = network
        self.rounds = rounds
        self.confidence = confidence
        self.susceptibility = susceptibility
        self.last_graph: SocialGraph | None = None

    def influence(
        self, population: Population, responses: np.ndarray, seed: int
    ) -> np.ndarray:
        responses = np.asarray(responses, dtype=float)
        if responses.ndim != 2 or responses.shape[0] != len(population):
            raise ValueError(
                f"expected a ({len(population)}, n_options) response matrix, "
                f"got {responses.shape}"
            )
        if self.rounds == 0 or self.susceptibility == 0.0:
            return responses.copy()

        graph = self.network.build(population, seed)
        self.last_graph = graph
        sources = np.repeat(np.arange(graph.n_agents), graph.degrees)
        current = responses.copy()

        for _ in range(self.rounds):
            source_opinions = current[sources]
            neighbour_opinions = current[graph.neighbours]
            # Total variation distance, the natural metric between two distributions
            # over the same options, and the one the confidence bound is stated in.
            distance = 0.5 * np.abs(source_opinions - neighbour_opinions).sum(axis=1)
            listens = distance <= self.confidence

            totals = np.zeros_like(current)
            counts = np.zeros(graph.n_agents)
            np.add.at(totals, sources[listens], neighbour_opinions[listens])
            np.add.at(counts, sources[listens], 1.0)

            heard = counts > 0
            average = np.zeros_like(current)
            average[heard] = totals[heard] / counts[heard, None]
            current[heard] = (
                (1.0 - self.susceptibility) * current[heard]
                + self.susceptibility * average[heard]
            )

        # Rounding drift over several rounds is tiny but real, and every consumer
        # downstream assumes rows are distributions.
        return current / current.sum(axis=1, keepdims=True)
