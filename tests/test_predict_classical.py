from __future__ import annotations

import numpy as np
import pytest

from quorum.core.population import Population
from quorum.data.targets import Question, QuestionBank
from quorum.predict import (
    CellMeanPropagator,
    DesignSpace,
    MeanPropagator,
    MultinomialLogitPropagator,
    PriorPredictor,
    UniformPredictor,
    apply_trait_noise,
)
from quorum.world.context import Scenario

OPTIONS = ("too little", "about right", "too much")


@pytest.fixture
def graduates_disagree(small_population) -> tuple[Population, np.ndarray]:
    """A population whose answer depends on an attribute, with a known direction."""
    frame = small_population.frame
    lift = np.where(frame["education"].to_numpy() == "bachelors", 0.3, -0.2)
    raw = np.clip(np.stack([0.34 + lift, 0.33 - lift / 2, 0.33 - lift / 2], axis=1), 0.01, None)
    return small_population, raw / raw.sum(axis=1, keepdims=True)


# -- design space ------------------------------------------------------------------


def test_design_space_columns_follow_the_taxonomy(small_population):
    design = DesignSpace(("education",), ("openness",))
    # Columns cover every level after the first, whether or not this population
    # happens to contain them.
    assert design.columns == (
        "education=high_school",
        "education=some_college",
        "education=bachelors",
        "education=graduate",
        "trait:openness",
    )
    matrix = design.encode(small_population)
    assert matrix.shape == (len(small_population), 5)
    assert set(np.unique(matrix[:, 2])) == {0.0, 1.0}
    assert matrix[:, 1].max() == 0.0


def test_design_space_drops_the_first_level_to_stay_identifiable():
    design = DesignSpace(("age_band",))
    assert "age_band=18-24" not in design.columns
    assert len(design.columns) == 5


def test_design_space_rejects_unknown_and_absent_fields(small_population):
    with pytest.raises(KeyError, match="unknown attributes"):
        DesignSpace(("height",))
    with pytest.raises(KeyError, match="missing attributes"):
        DesignSpace(("race",)).encode(small_population)
    with pytest.raises(KeyError, match="missing traits"):
        DesignSpace(("education",), ("charisma",)).encode(small_population)


def test_design_space_with_nothing_to_encode(small_population):
    assert DesignSpace((), ()).encode(small_population).shape == (len(small_population), 0)


# -- propagators -------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["mean", "cell", "logit"])
def test_propagators_return_valid_distributions(graduates_disagree, kind):
    population, responses = graduates_disagree
    sample, index = population.stratified_sample(24, ["age_band", "education"], seed=1)
    propagator = _build(kind)
    propagator.fit(sample, responses[index])
    out = propagator.predict(population)
    assert out.shape == (len(population), 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


@pytest.mark.parametrize("kind", ["cell", "logit"])
def test_structured_propagators_recover_a_real_signal(graduates_disagree, kind):
    """The mean propagator cannot see who answered; the other two must."""
    population, responses = graduates_disagree
    sample, index = population.stratified_sample(24, ["age_band", "education"], seed=1)
    out = _build(kind).fit(sample, responses[index]).predict(population)

    degree = population.frame["education"].to_numpy() == "bachelors"
    assert out[degree, 0].mean() > out[~degree, 0].mean() + 0.2

    flat = MeanPropagator().fit(sample, responses[index]).predict(population)
    assert flat[degree, 0].mean() == pytest.approx(flat[~degree, 0].mean())


def test_cell_propagator_falls_back_for_cells_the_sample_never_reached(graduates_disagree):
    population, responses = graduates_disagree
    young = population.frame["age_band"].to_numpy() == "18-24"
    sample = population.subset(young)
    propagator = CellMeanPropagator(("age_band", "education")).fit(sample, responses[young])
    out = propagator.predict(population)
    old = population.frame["age_band"].to_numpy() == "65+"
    np.testing.assert_allclose(out[old], np.tile(propagator.fallback_, (old.sum(), 1)))


def test_logit_propagator_falls_back_when_the_sample_is_unanimous(graduates_disagree):
    population, _ = graduates_disagree
    sample, index = population.stratified_sample(24, ["age_band", "education"], seed=1)
    unanimous = np.tile([1.0, 0.0, 0.0], (len(sample), 1))
    propagator = MultinomialLogitPropagator(DesignSpace(("education",))).fit(sample, unanimous)
    assert propagator.model_ is None
    np.testing.assert_allclose(propagator.predict(population), unanimous[:1].repeat(len(population), 0))


def test_logit_propagator_falls_back_without_features(graduates_disagree):
    population, responses = graduates_disagree
    sample, index = population.stratified_sample(24, ["age_band", "education"], seed=1)
    propagator = MultinomialLogitPropagator(DesignSpace((), ())).fit(sample, responses[index])
    assert propagator.model_ is None


def test_logit_propagator_scatters_back_into_full_option_space(graduates_disagree):
    """A sample that never used an option must still yield a full-width matrix."""
    population, responses = graduates_disagree
    sample, index = population.stratified_sample(24, ["age_band", "education"], seed=1)
    trimmed = responses[index].copy()
    trimmed[:, 2] = 0.0
    trimmed /= trimmed.sum(axis=1, keepdims=True)
    out = MultinomialLogitPropagator(DesignSpace(("education",))).fit(sample, trimmed).predict(population)
    assert out.shape[1] == 3
    assert out[:, 2].max() == pytest.approx(0.0)


@pytest.mark.parametrize("kind", ["mean", "cell", "logit"])
def test_propagators_refuse_to_predict_before_fitting(graduates_disagree, kind):
    population, _ = graduates_disagree
    with pytest.raises(RuntimeError, match="must be fitted"):
        _build(kind).predict(population)


@pytest.mark.parametrize("kind", ["mean", "cell", "logit"])
def test_propagators_validate_the_response_matrix(graduates_disagree, kind):
    population, _ = graduates_disagree
    with pytest.raises(ValueError, match="response matrix"):
        _build(kind).fit(population, np.zeros((3, 3)))


def test_propagators_need_at_least_two_options(graduates_disagree):
    population, _ = graduates_disagree
    with pytest.raises(ValueError, match="at least two options"):
        MeanPropagator().fit(population, np.ones((len(population), 1)))


def test_cell_propagator_needs_a_dimension():
    with pytest.raises(ValueError, match="at least one dimension"):
        CellMeanPropagator(())


def test_logit_propagator_needs_positive_regularization():
    with pytest.raises(ValueError, match="regularization"):
        MultinomialLogitPropagator(DesignSpace(("education",)), regularization=0.0)


def _build(kind: str):
    return {
        "mean": lambda: MeanPropagator(),
        "cell": lambda: CellMeanPropagator(("age_band", "education")),
        "logit": lambda: MultinomialLogitPropagator(DesignSpace(("age_band", "education"))),
    }[kind]()


# -- trait noise -------------------------------------------------------------------


def test_trait_noise_widens_spread_without_moving_the_topline(graduates_disagree):
    population, responses = graduates_disagree
    sample, index = population.stratified_sample(24, ["age_band", "education"], seed=1)
    flat = CellMeanPropagator(("age_band", "education")).fit(sample, responses[index]).predict(population)
    traits = population.frame[["trait_openness"]].to_numpy()

    noisy = apply_trait_noise(flat, traits, scale=0.5, seed=2)
    assert noisy.std(axis=0).mean() > flat.std(axis=0).mean()
    shift = population.weighted_distribution(noisy) - population.weighted_distribution(flat)
    assert np.max(np.abs(shift)) < 0.03
    np.testing.assert_allclose(noisy.sum(axis=1), 1.0)


def test_trait_noise_is_a_no_op_at_zero_scale(graduates_disagree):
    population, responses = graduates_disagree
    np.testing.assert_allclose(apply_trait_noise(responses, np.zeros((len(population), 1)), 0.0, 1), responses)


def test_trait_noise_works_without_traits(graduates_disagree):
    population, responses = graduates_disagree
    out = apply_trait_noise(responses, np.zeros((len(population), 0)), 0.3, seed=1)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


def test_trait_noise_is_deterministic(graduates_disagree):
    population, responses = graduates_disagree
    traits = population.frame[["trait_openness"]].to_numpy()
    a = apply_trait_noise(responses, traits, 0.4, seed=5)
    b = apply_trait_noise(responses, traits, 0.4, seed=5)
    c = apply_trait_noise(responses, traits, 0.4, seed=6)
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)


# -- baselines ---------------------------------------------------------------------


def test_uniform_baseline(small_population):
    scenario = Scenario("q", "Do you agree?", OPTIONS)
    out = UniformPredictor().predict(small_population, scenario, seed=1)
    np.testing.assert_allclose(out, 1.0 / 3.0)


def test_prior_baseline_averages_by_option_count():
    bank = _bank(
        [
            ("a", ("yes", "no"), [0.6, 0.4]),
            ("b", ("yes", "no"), [0.8, 0.2]),
            ("c", OPTIONS, [0.5, 0.3, 0.2]),
        ]
    )
    prior = PriorPredictor.fit(bank)
    np.testing.assert_allclose(prior.priors[2], [0.7, 0.3])
    np.testing.assert_allclose(prior.priors[3], [0.5, 0.3, 0.2])


def test_prior_baseline_predicts_the_matching_prior(small_population):
    prior = PriorPredictor({3: np.array([0.5, 0.3, 0.2])})
    out = prior.predict(small_population, Scenario("q", "Agree?", OPTIONS), seed=1)
    np.testing.assert_allclose(out[0], [0.5, 0.3, 0.2])


def test_prior_baseline_falls_back_to_uniform_for_an_unseen_shape(small_population):
    prior = PriorPredictor({2: np.array([0.6, 0.4])})
    out = prior.predict(small_population, Scenario("q", "Agree?", OPTIONS), seed=1)
    np.testing.assert_allclose(out[0], 1.0 / 3.0)


def test_prior_baseline_needs_questions():
    with pytest.raises(ValueError, match="empty question bank"):
        PriorPredictor.fit(_bank([]))


def _bank(items) -> QuestionBank:
    questions = {
        qid: Question(
            id=qid,
            text=f"{qid}?",
            options=options,
            topline=np.array(topline, dtype=float),
            standard_error=np.zeros(len(options)),
            n=100,
            effective_n=100.0,
        )
        for qid, options, topline in items
    }
    return QuestionBank(source={}, year=2024, weight_variable="w", questions=questions)
