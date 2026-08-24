"""The population object: a weighted, columnar store of simulated agents.

Design note. The obvious modelling of a population is ``list[Agent]``. It does not
survive contact with scale: at 100k agents every marginal, every stratification and
every reweight becomes a Python loop. ``Population`` instead wraps a single
:class:`pandas.DataFrame` with an explicit weight column, so the operations the
simulation loop actually performs (marginals, cells, stratified sampling, raking,
poststratification) are vectorized, and materializing an :class:`~quorum.core.agent.Agent`
is an explicit, rare act.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd

from quorum.core.agent import Agent

WEIGHT_COLUMN = "weight"
TRAIT_PREFIX = "trait_"


class Population:
    """A weighted set of synthetic agents.

    Parameters
    ----------
    frame:
        One row per agent. Must contain a ``weight`` column. Columns prefixed
        ``trait_`` are treated as latent traits; every other non-weight column is an
        observed attribute that marginals may be defined over.
    name:
        Human label carried into run records and reports.
    """

    def __init__(self, frame: pd.DataFrame, name: str = "population") -> None:
        if WEIGHT_COLUMN not in frame.columns:
            raise ValueError(f"population frame must have a {WEIGHT_COLUMN!r} column")
        weights = frame[WEIGHT_COLUMN].to_numpy(dtype=float)
        if not np.all(np.isfinite(weights)):
            raise ValueError("population weights must all be finite")
        if np.any(weights < 0):
            raise ValueError("population weights must be non-negative")
        if weights.sum() <= 0:
            raise ValueError("population weights must sum to a positive number")
        self._frame = frame.reset_index(drop=True)
        self.name = name

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_records(
        cls,
        records: Sequence[dict[str, Any]],
        name: str = "population",
        weight: float | Sequence[float] = 1.0,
    ) -> "Population":
        """Build a population from plain dicts, mostly for tests and small fixtures."""
        frame = pd.DataFrame.from_records(list(records))
        if WEIGHT_COLUMN not in frame.columns:
            frame[WEIGHT_COLUMN] = weight
        return cls(frame, name=name)

    # -- shape -----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._frame)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Population(name={self.name!r}, n={len(self)}, "
            f"attributes={list(self.attributes)}, weight_sum={self.weight_sum:,.0f})"
        )

    @property
    def frame(self) -> pd.DataFrame:
        """The backing frame. Returned as a copy so callers cannot mutate state."""
        return self._frame.copy()

    @property
    def weights(self) -> np.ndarray:
        return self._frame[WEIGHT_COLUMN].to_numpy(dtype=float)

    @property
    def weight_sum(self) -> float:
        return float(self.weights.sum())

    @property
    def attributes(self) -> tuple[str, ...]:
        """Observed attribute columns, in stable order."""
        return tuple(
            c
            for c in self._frame.columns
            if c != WEIGHT_COLUMN and not c.startswith(TRAIT_PREFIX)
        )

    @property
    def traits(self) -> tuple[str, ...]:
        return tuple(
            c[len(TRAIT_PREFIX) :]
            for c in self._frame.columns
            if c.startswith(TRAIT_PREFIX)
        )

    # -- aggregation -----------------------------------------------------------

    def marginals(self, dimension: str) -> pd.Series:
        """Weighted share of the population in each level of ``dimension``.

        Sums to 1. This is the quantity synthesis is raked against and predictions
        are poststratified onto, so it is defined once, here.
        """
        if dimension not in self._frame.columns:
            raise KeyError(f"unknown dimension {dimension!r}")
        grouped = self._frame.groupby(dimension, observed=True)[WEIGHT_COLUMN].sum()
        return (grouped / grouped.sum()).sort_index()

    def all_marginals(self, dimensions: Sequence[str] | None = None) -> dict[str, pd.Series]:
        dims = tuple(dimensions) if dimensions is not None else self.attributes
        return {d: self.marginals(d) for d in dims}

    def cells(self, dimensions: Sequence[str]) -> pd.DataFrame:
        """Weighted poststratification cells over ``dimensions``.

        Returns one row per occupied cell with columns ``dimensions + ["weight", "share"]``.
        """
        dims = list(dimensions)
        if not dims:
            raise ValueError("cells() needs at least one dimension")
        grouped = (
            self._frame.groupby(dims, observed=True)[WEIGHT_COLUMN]
            .sum()
            .reset_index()
            .sort_values(dims)
            .reset_index(drop=True)
        )
        grouped["share"] = grouped[WEIGHT_COLUMN] / grouped[WEIGHT_COLUMN].sum()
        return grouped

    def cell_index(self, dimensions: Sequence[str]) -> np.ndarray:
        """Integer cell id per agent, aligned to ``cells(dimensions)`` row order."""
        dims = list(dimensions)
        cells = self.cells(dims)
        key = pd.MultiIndex.from_frame(cells[dims])
        lookup = pd.Series(np.arange(len(cells)), index=key)
        agent_key = pd.MultiIndex.from_frame(self._frame[dims])
        return lookup.reindex(agent_key).to_numpy(dtype=int)

    def weighted_mean(self, values: np.ndarray) -> float:
        """Weighted mean of a per-agent quantity."""
        values = np.asarray(values, dtype=float)
        if values.shape[0] != len(self):
            raise ValueError(f"expected {len(self)} values, got {values.shape[0]}")
        w = self.weights
        return float(np.dot(values, w) / w.sum())

    def weighted_distribution(self, values: np.ndarray) -> np.ndarray:
        """Weighted column means of a per-agent ``(n_agents, n_options)`` matrix.

        This is the aggregation step of the simulation loop: agent-level response
        distributions in, one population-level distribution out.
        """
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[0] != len(self):
            raise ValueError(f"expected a ({len(self)}, n_options) matrix, got {values.shape}")
        w = self.weights
        return (values * w[:, None]).sum(axis=0) / w.sum()

    # -- transformation --------------------------------------------------------

    def with_weights(self, weights: np.ndarray) -> "Population":
        """Return a copy carrying new weights. The original is untouched."""
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (len(self),):
            raise ValueError(f"expected {len(self)} weights, got {weights.shape}")
        frame = self._frame.copy()
        frame[WEIGHT_COLUMN] = weights
        return Population(frame, name=self.name)

    def with_column(self, name: str, values: np.ndarray) -> "Population":
        frame = self._frame.copy()
        frame[name] = values
        return Population(frame, name=self.name)

    def subset(self, mask: np.ndarray) -> "Population":
        frame = self._frame.loc[np.asarray(mask, dtype=bool)].reset_index(drop=True)
        return Population(frame, name=self.name)

    # -- sampling --------------------------------------------------------------

    def sample(self, n: int, seed: int, weighted: bool = True) -> "Population":
        """Draw ``n`` agents without replacement, deterministically."""
        if n >= len(self):
            return Population(self._frame.copy(), name=self.name)
        rng = np.random.default_rng(seed)
        if weighted:
            p = self.weights / self.weight_sum
            idx = rng.choice(len(self), size=n, replace=False, p=p)
        else:
            idx = rng.choice(len(self), size=n, replace=False)
        idx = np.sort(idx)
        return Population(self._frame.iloc[idx].reset_index(drop=True), name=self.name)

    def stratified_sample(
        self, n: int, dimensions: Sequence[str], seed: int
    ) -> tuple["Population", np.ndarray]:
        """Draw ``n`` agents spread proportionally across cells of ``dimensions``.

        Returns the sample and the index of each sampled agent in this population.
        This is the sampling frame the hybrid predictor spends its LLM budget on:
        every occupied cell gets at least one representative, and cells get extra
        agents in proportion to their weight.
        """
        dims = list(dimensions)
        cell_ids = self.cell_index(dims)
        cells = self.cells(dims)
        n_cells = len(cells)
        if n < n_cells:
            raise ValueError(
                f"cannot cover {n_cells} cells of {dims} with only {n} draws; "
                "raise the archetype budget or coarsen the stratification"
            )

        shares = cells["share"].to_numpy(dtype=float)
        alloc = _largest_remainder(shares, n, minimum=1)

        rng = np.random.default_rng(seed)
        chosen: list[int] = []
        for cid in range(n_cells):
            members = np.flatnonzero(cell_ids == cid)
            take = min(int(alloc[cid]), len(members))
            picked = rng.choice(members, size=take, replace=False)
            chosen.extend(int(i) for i in picked)
        index = np.sort(np.array(chosen, dtype=int))
        return Population(self._frame.iloc[index].reset_index(drop=True), name=self.name), index

    # -- single-agent access ---------------------------------------------------

    def agent(self, i: int) -> Agent:
        """Materialize agent ``i``. Deliberately explicit; see the module docstring."""
        row = self._frame.iloc[i]
        return Agent(
            id=int(i),
            attributes={c: row[c] for c in self.attributes},
            traits={t: float(row[TRAIT_PREFIX + t]) for t in self.traits},
            weight=float(row[WEIGHT_COLUMN]),
        )

    def iter_agents(self) -> Iterator[Agent]:
        for i in range(len(self)):
            yield self.agent(i)

    # -- reproducibility -------------------------------------------------------

    def fingerprint(self) -> str:
        """Stable content hash. Two runs producing the same population agree here."""
        h = hashlib.sha256()
        h.update(self.name.encode())
        for col in sorted(self._frame.columns):
            h.update(col.encode())
            values = self._frame[col].to_numpy()
            if values.dtype.kind in "fc":
                h.update(np.round(values.astype(float), 9).tobytes())
            else:
                h.update("\x1f".join(map(str, values)).encode())
        return h.hexdigest()[:16]


def _largest_remainder(shares: np.ndarray, total: int, minimum: int = 0) -> np.ndarray:
    """Apportion ``total`` across ``shares`` with the largest-remainder method.

    Guarantees each entry gets at least ``minimum`` and the allocation sums exactly to
    ``total``, which naive rounding does not.
    """
    shares = np.asarray(shares, dtype=float)
    k = len(shares)
    if minimum * k > total:
        raise ValueError(f"cannot give {k} entries a minimum of {minimum} out of {total}")
    remaining = total - minimum * k
    exact = shares / shares.sum() * remaining
    floor = np.floor(exact).astype(int)
    short = remaining - int(floor.sum())
    if short > 0:
        order = np.argsort(-(exact - floor), kind="stable")
        floor[order[:short]] += 1
    return floor + minimum
