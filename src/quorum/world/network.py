"""A social graph over a synthetic population.

People are not independent draws and their opinions are not either. The graph here is
deliberately simple, and deliberately homophilous: an agent's neighbours are drawn
mostly from agents like itself, in the attribute space the population was synthesized
on. That single property is what makes peer influence over the graph do something other
than push everybody toward the same answer.

The construction is O(n * degree) and holds only an edge list, so a 100k-agent graph is
a couple of integer arrays rather than a dict of adjacency sets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quorum.core.population import Population


@dataclass(frozen=True, slots=True)
class SocialGraph:
    """An undirected graph in flat neighbour-list form.

    ``neighbours`` holds every edge endpoint back to back; ``offsets`` marks where each
    agent's block starts, so agent ``i``'s neighbours are
    ``neighbours[offsets[i]:offsets[i + 1]]``. This is the compressed layout the
    influence loop wants, and it keeps the whole graph in two arrays.
    """

    neighbours: np.ndarray
    offsets: np.ndarray
    n_agents: int

    def __post_init__(self) -> None:
        if self.offsets.shape != (self.n_agents + 1,):
            raise ValueError("offsets must have one more entry than there are agents")

    def neighbours_of(self, agent: int) -> np.ndarray:
        return self.neighbours[self.offsets[agent] : self.offsets[agent + 1]]

    @property
    def degrees(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def n_edges(self) -> int:
        """Undirected edge count. Every edge appears twice in the neighbour list."""
        return int(len(self.neighbours) // 2)

    def homophily(self, labels: np.ndarray) -> float:
        """Share of edges joining two agents with the same label.

        The measured counterpart of the ``homophily`` parameter the graph was built
        with, and the thing the tests assert on: a stated parameter that the built
        graph does not exhibit is a bug that would otherwise go unnoticed.
        """
        if len(self.neighbours) == 0:
            return float("nan")
        sources = np.repeat(np.arange(self.n_agents), self.degrees)
        return float(np.mean(labels[sources] == labels[self.neighbours]))


class HomophilyNetwork:
    """Builds a graph where similar agents are more likely to be connected.

    Each agent proposes edges. With probability ``homophily`` the other endpoint is
    drawn from agents sharing its cell over ``dimensions``; otherwise from the
    population at large. Proposals are then symmetrized and deduplicated, so the
    realized degree distribution is a consequence of the process rather than imposed,
    and ``mean_degree`` is approached rather than enforced exactly.
    """

    def __init__(
        self, mean_degree: int = 8, homophily: float = 0.7, dimensions: tuple[str, ...] = ()
    ) -> None:
        if mean_degree < 1:
            raise ValueError("mean_degree must be at least 1")
        if not 0.0 <= homophily <= 1.0:
            raise ValueError("homophily must be between 0 and 1")
        self.mean_degree = mean_degree
        self.homophily = homophily
        self.dimensions = tuple(dimensions)

    def build(self, population: Population, seed: int) -> SocialGraph:
        n = len(population)
        if n < 2:
            raise ValueError("a social graph needs at least two agents")
        rng = np.random.default_rng(seed)

        dims = list(self.dimensions) if self.dimensions else list(population.attributes[:1])
        cell_ids = population.cell_index(dims)
        order = np.argsort(cell_ids, kind="stable")
        sorted_cells = cell_ids[order]
        # Where each cell's block begins in the sorted order, so a same-cell draw is a
        # uniform pick inside one contiguous slice.
        boundaries = np.flatnonzero(np.diff(sorted_cells)) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [n]])

        # Each proposal becomes an undirected edge, which lands in both endpoints'
        # neighbour lists. Proposing mean_degree edges per agent would therefore give
        # a realized degree of roughly twice that, so halve it here and let
        # mean_degree mean what it says.
        proposals = max(1, round(self.mean_degree / 2))
        sources = np.repeat(np.arange(n), proposals)
        same_cell = rng.random(n * proposals) < self.homophily

        targets = rng.integers(0, n, size=n * proposals)
        if same_cell.any():
            cells_of_sources = cell_ids[sources[same_cell]]
            lo = starts[cells_of_sources]
            hi = ends[cells_of_sources]
            picks = lo + (rng.random(int(same_cell.sum())) * (hi - lo)).astype(int)
            targets[same_cell] = order[np.clip(picks, lo, hi - 1)]

        keep = sources != targets
        sources, targets = sources[keep], targets[keep]

        # Symmetrize, then drop duplicate undirected edges.
        low = np.minimum(sources, targets)
        high = np.maximum(sources, targets)
        edges = np.unique(np.stack([low, high], axis=1), axis=0)
        both = np.concatenate([edges, edges[:, ::-1]], axis=0)

        order = np.lexsort((both[:, 1], both[:, 0]))
        both = both[order]
        degrees = np.bincount(both[:, 0], minlength=n)
        offsets = np.concatenate([[0], np.cumsum(degrees)])
        return SocialGraph(neighbours=both[:, 1].copy(), offsets=offsets, n_agents=n)
