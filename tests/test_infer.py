from __future__ import annotations

import numpy as np
import pytest

from quorum.core.population import Population
from quorum.infer.montecarlo import bootstrap_draws, interval_agreement
from quorum.infer.mrp import DirectEstimator, MRPEstimator
from quorum.infer.pooling import (
    CONCENTRATION_BOUNDS,
    fit_concentration,
    pool_cells,
    posterior_draws,
)
from quorum.predict.propagate import CellMeanPropagator
from quorum.world.context import Scenario

OPTIONS = ("too little", "about right", "too much")


@pytest.fixture
def scenario() -> Scenario:
    return Scenario("q", "Are we spending too little?", OPTIONS)


# -- pooling -----------------------------------------------------------------------


def test_concentration_tracks_how_different_the_cells_are():
    """More heterogeneous cells must yield weaker pooling, not just a different number."""
    rng = np.random.default_rng(0)
    mu = np.array([0.5, 0.3, 0.2])
    fits = []
    for true_alpha in (2.0, 10.0, 50.0):
        cells = rng.dirichlet(true_alpha * mu, size=200)
        counts = np.stack([rng.multinomial(40, p) for p in cells]).astype(float)
        fits.append(fit_concentration(counts, counts.sum(axis=0) / counts.sum()))
    assert fits[0] < fits[1] < fits[2]
    assert 1.0 < fits[0] < 5.0


def test_identical_cells_pool_all_the_way_and_say_so():
    pooled = pool_cells(np.tile([5.0, 3.0, 2.0], (20, 1)))
    assert pooled.at_bound
    assert pooled.concentration == pytest.approx(CONCENTRATION_BOUNDS[1])
    assert pooled.shrinkage.min() > 0.99


def test_pooling_shrinks_cells_toward_the_global_mean_without_erasing_them():
    counts = np.array([[6.0, 2.0, 2.0]] * 10 + [[2.0, 4.0, 4.0]] * 10)
    pooled = pool_cells(counts)
    assert not pooled.at_bound
    high, low = pooled.posterior_mean[0, 0], pooled.posterior_mean[-1, 0]
    # Pulled toward each other, but still distinguishable.
    assert 0.2 < low < 0.4 < high < 0.6
    assert high - low < 0.4  # the raw gap was 0.4


def test_a_cell_with_less_evidence_shrinks_more():
    counts = np.array([[30.0, 10.0, 10.0], [0.6, 0.2, 0.2], [2.0, 4.0, 4.0]])
    shrinkage = pool_cells(counts).shrinkage
    assert shrinkage[1] > shrinkage[2] > shrinkage[0]


def test_an_empty_cell_becomes_the_global_average():
    counts = np.array([[6.0, 2.0, 2.0], [2.0, 4.0, 4.0], [0.0, 0.0, 0.0]])
    pooled = pool_cells(counts)
    np.testing.assert_allclose(pooled.posterior_mean[2], pooled.prior_mean, atol=1e-9)
    assert pooled.shrinkage[2] == pytest.approx(1.0)


def test_an_explicit_concentration_overrides_the_fit():
    counts = np.array([[6.0, 2.0, 2.0]] * 5 + [[2.0, 4.0, 4.0]] * 5)
    assert pool_cells(counts, concentration=1.0).concentration == 1.0
    assert not pool_cells(counts, concentration=1.0).at_bound


def test_fit_needs_at_least_two_occupied_cells():
    counts = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    assert fit_concentration(counts, np.full(3, 1 / 3)) == CONCENTRATION_BOUNDS[1]


@pytest.mark.parametrize(
    "counts, message",
    [
        (np.zeros((3, 3)), "no evidence"),
        (np.array([[-1.0, 1.0]]), "non-negative"),
        (np.zeros(3), "must be"),
    ],
)
def test_pooling_validates_its_input(counts, message):
    with pytest.raises(ValueError, match=message):
        pool_cells(counts)


def test_pooling_rejects_a_non_positive_concentration():
    with pytest.raises(ValueError, match="concentration must be positive"):
        pool_cells(np.ones((2, 3)), concentration=-1.0)


def test_posterior_draws_are_tighter_where_there_is_more_evidence():
    pooled = pool_cells(np.array([[300.0, 100.0, 100.0], [3.0, 1.0, 1.0]]), concentration=1.0)
    draws = posterior_draws(pooled, 800, seed=1)
    assert draws.shape == (800, 2, 3)
    np.testing.assert_allclose(draws.sum(axis=2), 1.0)
    assert draws[:, 0, 0].std() < draws[:, 1, 0].std()


def test_posterior_draws_are_deterministic_and_need_a_positive_count():
    pooled = pool_cells(np.array([[6.0, 2.0, 2.0], [2.0, 4.0, 4.0]]))
    np.testing.assert_allclose(posterior_draws(pooled, 50, 3), posterior_draws(pooled, 50, 3))
    with pytest.raises(ValueError, match="draws must be"):
        posterior_draws(pooled, 0, 1)


# -- estimators --------------------------------------------------------------------


def test_direct_estimator_is_a_weighted_average_with_no_claim_of_uncertainty(scenario):
    population = Population.from_records([{"g": "a"}, {"g": "b"}], weight=[3.0, 1.0])
    responses = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    prediction = DirectEstimator().estimate(population, responses, scenario, seed=1)
    np.testing.assert_allclose(prediction.distribution, [0.75, 0.25, 0.0])
    assert not prediction.has_uncertainty


def test_mrp_reweights_the_sample_onto_the_population(scenario, small_population):
    """The sample over-represents small cells by design; poststratification undoes it."""
    frame = small_population
    sample, index = frame.stratified_sample(24, ["age_band", "education"], seed=1)
    graduate = (sample.frame["education"] == "bachelors").to_numpy()
    responses = np.where(
        graduate[:, None], np.array([0.8, 0.1, 0.1]), np.array([0.1, 0.45, 0.45])
    )

    prediction = MRPEstimator(("age_band", "education"), draws=500).estimate(
        frame, sample, responses, scenario, seed=2
    )
    frame_share = float((frame.frame["education"] == "bachelors").mean())
    assert prediction.share("too little") == pytest.approx(
        0.8 * frame_share + 0.1 * (1 - frame_share), abs=0.12
    )
    assert prediction.has_uncertainty
    assert prediction.metadata["cells"] == 6


def test_mrp_breaks_the_answer_down_by_dimension(scenario, small_population):
    sample, index = small_population.stratified_sample(24, ["age_band", "education"], seed=1)
    graduate = (sample.frame["education"] == "bachelors").to_numpy()
    responses = np.where(graduate[:, None], np.array([0.9, 0.05, 0.05]), np.array([0.1, 0.45, 0.45]))
    prediction = MRPEstimator(("age_band", "education"), draws=200).estimate(
        small_population, sample, responses, scenario, seed=2
    )
    education = prediction.segments["education"]
    assert education["bachelors"][0] > education["high_school"][0]


def test_mrp_intervals_widen_when_less_was_measured(scenario, small_population):
    rng = np.random.default_rng(0)
    estimator = MRPEstimator(("age_band", "education"), draws=1500, concentration=5.0)

    widths = []
    for size in (12, 60):
        sample, index = small_population.stratified_sample(size, ["age_band", "education"], seed=1)
        responses = rng.dirichlet([3.0, 3.0, 3.0], size=len(sample))
        prediction = estimator.estimate(small_population, sample, responses, scenario, seed=2)
        interval = prediction.interval(0.90)
        widths.append(float(np.mean(interval[:, 1] - interval[:, 0])))
    assert widths[0] > widths[1]


def test_mrp_with_pooling_off_keeps_cells_further_apart(scenario, small_population):
    sample, index = small_population.stratified_sample(24, ["age_band", "education"], seed=1)
    graduate = (sample.frame["education"] == "bachelors").to_numpy()
    responses = np.where(graduate[:, None], np.array([0.9, 0.05, 0.05]), np.array([0.1, 0.45, 0.45]))

    def gap(pool: bool) -> float:
        prediction = MRPEstimator(("age_band", "education"), draws=0, pool=pool).estimate(
            small_population, sample, responses, scenario, seed=2
        )
        segments = prediction.segments["education"]
        return float(segments["bachelors"][0] - segments["high_school"][0])

    assert gap(pool=False) > gap(pool=True)


def test_mrp_can_skip_drawing(scenario, small_population):
    sample, index = small_population.stratified_sample(12, ["age_band"], seed=1)
    prediction = MRPEstimator(("age_band",), draws=0).estimate(
        small_population, sample, np.full((len(sample), 3), 1 / 3), scenario, seed=1
    )
    assert not prediction.has_uncertainty


def test_mrp_validates_its_inputs(scenario, small_population):
    with pytest.raises(ValueError, match="at least one dimension"):
        MRPEstimator(())
    with pytest.raises(ValueError, match="level must be"):
        MRPEstimator(("age_band",), level=1.5)
    sample, index = small_population.stratified_sample(12, ["age_band"], seed=1)
    with pytest.raises(ValueError, match="expected 12 response rows"):
        MRPEstimator(("age_band",)).estimate(
            small_population, sample, np.zeros((3, 3)), scenario, seed=1
        )


# -- bootstrap ---------------------------------------------------------------------


def test_bootstrap_produces_a_spread_around_the_point_estimate(small_population):
    frame = small_population
    sample, index = frame.stratified_sample(24, ["age_band", "education"], seed=1)
    rng = np.random.default_rng(0)
    responses = rng.dirichlet([2.0, 2.0, 2.0], size=len(sample))

    def refit(resampled, resampled_responses):
        propagator = CellMeanPropagator(("age_band", "education"))
        propagator.fit(resampled, resampled_responses)
        return propagator.predict(frame)

    draws = bootstrap_draws(frame, sample, responses, refit, draws=60, seed=2)
    assert draws.shape == (60, 3)
    np.testing.assert_allclose(draws.sum(axis=1), 1.0)
    assert draws.std(axis=0).min() > 0


def test_bootstrap_narrows_as_the_sample_grows(small_population):
    frame = small_population
    rng = np.random.default_rng(1)

    def spread(size: int) -> float:
        sample, index = frame.stratified_sample(size, ["age_band", "education"], seed=1)
        responses = rng.dirichlet([2.0, 2.0, 2.0], size=len(sample))

        def refit(resampled, resampled_responses):
            propagator = CellMeanPropagator(("age_band", "education"))
            propagator.fit(resampled, resampled_responses)
            return propagator.predict(frame)

        return float(bootstrap_draws(frame, sample, responses, refit, draws=80, seed=3).std(axis=0).mean())

    assert spread(12) > spread(96)


def test_bootstrap_validates_its_input(small_population):
    sample, index = small_population.stratified_sample(12, ["age_band"], seed=1)
    with pytest.raises(ValueError, match="expected 12 response rows"):
        bootstrap_draws(small_population, sample, np.zeros((3, 3)), lambda s, r: None)
    with pytest.raises(ValueError, match="draws must be"):
        bootstrap_draws(small_population, sample, np.zeros((12, 3)), lambda s, r: None, draws=0)


def test_interval_agreement_is_zero_for_identical_draws_and_grows_with_disagreement():
    rng = np.random.default_rng(0)
    a = rng.dirichlet([10, 10, 10], size=500)
    b = rng.dirichlet([10, 10, 10], size=500)
    wide = rng.dirichlet([1, 1, 1], size=500)
    assert interval_agreement(a, a) == pytest.approx(0.0)
    assert interval_agreement(a, wide) > interval_agreement(a, b)
