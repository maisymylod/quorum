"""Partial pooling of cell-level estimates.

A stratified sample of a few hundred archetypes spread over several hundred
poststratification cells leaves most cells with one or two observations. Taking each
cell's own average at face value produces estimates that swing wildly on no evidence,
and a topline built from them inherits the swing. Ignoring cells entirely and using one
global average throws away the structure the whole exercise is about.

Partial pooling is the answer in between: every cell's estimate is pulled toward the
global one by an amount that depends on how much evidence the cell actually has. A cell
with thirty observations barely moves; a cell with one is mostly the global average.

The model is Dirichlet-multinomial. Cell probabilities are drawn from a Dirichlet
centred on the global distribution with concentration ``alpha``, and ``alpha`` is not
guessed: it is estimated from how much the cells genuinely differ, after subtracting
the part of that difference explained by each cell having seen little evidence. No
sampler and no external dependency, because the posterior is conjugate and can be
drawn from directly.

The posterior width still rests on treating soft response distributions as if they
were counts, which they are not. :mod:`quorum.infer.montecarlo` derives the same
interval by resampling instead, and
:func:`quorum.infer.montecarlo.interval_agreement` measures how far the two routes
disagree, so that assumption is checked rather than trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
_EPSILON = 1e-12


#: Search range for the pooling strength. The upper end is effectively "these cells
#: are indistinguishable"; a fit that lands there is reported rather than hidden.
CONCENTRATION_BOUNDS = (1e-2, 1e4)


def fit_concentration(
    counts: np.ndarray, mu: np.ndarray, bounds: tuple[float, float] = CONCENTRATION_BOUNDS
) -> float:
    """Estimate the pooling strength from how much the cells actually differ.

    Method of moments, not maximum likelihood, and the reason is the shape of the
    evidence. Cell counts here are *soft*: an agent who answered 60/40 contributes 0.6
    and 0.4 rather than a vote. The Dirichlet-multinomial likelihood is written for
    integer draws, and evaluated on fractional ones its ``gammaln`` terms diverge as
    the prior shrinks, so it penalizes weak pooling for a purely numerical reason and
    saturates at maximal pooling however different the cells plainly are. That is not
    a tuning problem; it is the wrong likelihood for this data.

    Matching moments avoids the question. Under a Dirichlet with concentration
    ``alpha``, option ``k`` varies across cells with variance
    ``mu_k (1 - mu_k) / (alpha + 1)``. Measure how much the cells actually vary and
    solve for ``alpha``.

    No multinomial noise term is subtracted, and that is deliberate. These cell means
    are averages of predicted distributions rather than tallies of drawn answers, so
    they carry far less sampling noise than a multinomial would; subtracting a
    multinomial correction over-corrects, wipes out the measured variation, and pins
    the fit at maximal pooling with every cell collapsed onto the global average.
    Omitting it leaves the estimate conservative in the safe direction: slightly too
    little pooling, so posteriors come out slightly wide. Whether they are too wide is
    exactly what :func:`quorum.infer.montecarlo.interval_agreement` is for.
    """
    counts = np.asarray(counts, dtype=float)
    mu = np.asarray(mu, dtype=float)
    totals = counts.sum(axis=1)
    occupied = totals > 0
    if occupied.sum() < 2:
        return float(bounds[1])

    means = counts[occupied] / totals[occupied, None]
    evidence = totals[occupied]
    weights = evidence / evidence.sum()

    # Total variance available to explain, and how much of it the cells show.
    # Aggregating over options before dividing matters: solving option by option and
    # averaging lets a single option whose cells happen to agree drag the estimate to
    # infinity, however plainly the others disagree.
    spread = float(np.sum(mu * (1.0 - mu)))
    observed = float(np.sum((weights[:, None] * (means - mu) ** 2).sum(axis=0)))
    if spread <= 1e-12 or observed <= 1e-12:
        return float(bounds[1])
    return float(np.clip(spread / observed - 1.0, bounds[0], bounds[1]))


@dataclass(frozen=True, slots=True)
class PooledCells:
    """Posterior over each cell's response distribution."""

    posterior_mean: np.ndarray
    concentration: float
    prior_mean: np.ndarray
    evidence: np.ndarray
    at_bound: bool = False
    """True when the fitted pooling strength ran into the end of its search range.

    At the top it means the evidence cannot tell the cells apart, so every cell has
    collapsed onto the global average and a segment breakdown will be flat. That is a
    real finding about the sample size, but it looks identical to a bug unless it is
    said out loud."""

    @property
    def shrinkage(self) -> np.ndarray:
        """Per cell, the share of its estimate that came from the prior.

        1.0 means the cell had no evidence of its own and is entirely the global
        average; 0.0 would mean the global average had no influence. Reported rather
        than hidden, because how much a cell-level number was invented is exactly what
        a reader of a segment breakdown needs to know.
        """
        totals = self.evidence.sum(axis=1)
        return self.concentration / (self.concentration + totals)


def pool_cells(
    counts: np.ndarray, concentration: float | None = None
) -> PooledCells:
    """Shrink each cell's observed distribution toward the global one.

    Parameters
    ----------
    counts:
        ``(n_cells, n_options)`` of evidence. Fractional counts are expected and
        correct: an agent that answered 60/40 contributes 0.6 and 0.4, not a vote.
    concentration:
        Pooling strength. Fitted from the data when omitted.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2:
        raise ValueError(f"counts must be (n_cells, n_options), got {counts.shape}")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")

    grand_total = counts.sum()
    if grand_total <= 0:
        raise ValueError("cannot pool cells with no evidence at all")
    prior_mean = counts.sum(axis=0) / grand_total

    if concentration is None:
        alpha = fit_concentration(counts, prior_mean)
        at_bound = not (CONCENTRATION_BOUNDS[0] * 1.01 < alpha < CONCENTRATION_BOUNDS[1] * 0.99)
    else:
        alpha = float(concentration)
        at_bound = False
    if alpha <= 0:
        raise ValueError("concentration must be positive")

    posterior = alpha * prior_mean + counts
    posterior_mean = posterior / posterior.sum(axis=1, keepdims=True)
    return PooledCells(
        posterior_mean=posterior_mean,
        concentration=alpha,
        prior_mean=prior_mean,
        evidence=counts,
        at_bound=at_bound,
    )


def posterior_draws(pooled: PooledCells, draws: int, seed: int) -> np.ndarray:
    """Sample ``(draws, n_cells, n_options)`` from each cell's posterior.

    The posterior is Dirichlet, so this is exact rather than a chain that has to be
    checked for convergence.
    """
    if draws < 1:
        raise ValueError("draws must be at least 1")
    rng = np.random.default_rng(seed)
    posterior = pooled.concentration * pooled.prior_mean + pooled.evidence
    n_cells, n_options = posterior.shape
    out = np.empty((draws, n_cells, n_options))
    for cell in range(n_cells):
        out[:, cell, :] = rng.dirichlet(np.maximum(posterior[cell], _EPSILON), size=draws)
    return out
