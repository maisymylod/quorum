from __future__ import annotations

import numpy as np
import pytest

from quorum.core.spec import SimulationSpec
from quorum.data.targets import MarginalTargets, QuestionBank
from quorum.eval.configurations import BASE_SPEC, build_spec
from quorum.exec.runner import Simulation

ROOT = "."


@pytest.fixture(scope="module")
def bank() -> QuestionBank:
    return QuestionBank.load("data/vendor/gss_questions.json")


@pytest.fixture(scope="module")
def targets() -> MarginalTargets:
    return MarginalTargets.load(BASE_SPEC["population"]["targets"])


@pytest.fixture(scope="module")
def welfare_arms(bank):
    experiment = next(e for e in bank.experiments if e.id == "welfare")
    return [bank[a] for a in experiment.arms]


def _spec(engine: str, arms, **overrides) -> SimulationSpec:
    small = {"population": {"size": 800}, "predictor": {"archetypes": 60}, "estimator": {"draws": 200}}
    for key, value in overrides.items():
        small.setdefault(key, {}).update(value)
    return build_spec(engine, "welfare", arms, overrides=small)


def test_a_run_produces_a_prediction_for_every_arm(welfare_arms, targets, bank):
    result = Simulation(_spec("hybrid", welfare_arms), targets=targets, root=ROOT).run()
    assert set(result.predictions) == {"natfare", "natfarey"}
    for prediction in result.predictions.values():
        np.testing.assert_allclose(prediction.distribution.sum(), 1.0)
        assert prediction.has_uncertainty


def test_every_arm_runs_against_the_same_population(welfare_arms, targets):
    """Arms that differed in population could not be compared on the gap between them."""
    result = Simulation(_spec("hybrid", welfare_arms), targets=targets, root=ROOT).run()
    assert result.record.population_fingerprint == result.population.fingerprint()
    assert len(result.population) == 800


def test_the_run_is_reproducible(welfare_arms, targets):
    a = Simulation(_spec("hybrid", welfare_arms), targets=targets, root=ROOT).run()
    b = Simulation(_spec("hybrid", welfare_arms), targets=targets, root=ROOT).run()
    assert a.record.reproducibility_key() == b.record.reproducibility_key()
    for arm in a.predictions:
        np.testing.assert_allclose(a.predictions[arm].distribution, b.predictions[arm].distribution)


def test_a_different_seed_gives_a_different_population(welfare_arms, targets):
    a = Simulation(_spec("hybrid", welfare_arms), targets=targets, root=ROOT).run()
    other = _spec("hybrid", welfare_arms).model_copy(update={"seed": 99})
    b = Simulation(other, targets=targets, root=ROOT).run()
    assert a.record.population_fingerprint != b.record.population_fingerprint


def test_synthesis_fidelity_is_enforced_at_run_time(welfare_arms, targets):
    result = Simulation(_spec("hybrid", welfare_arms), targets=targets, root=ROOT).run()
    assert result.fidelity.max_deviation < 1e-6


def test_a_stub_run_says_so_in_its_record(welfare_arms, targets):
    result = Simulation(_spec("hybrid", welfare_arms), targets=targets, root=ROOT).run()
    assert any("stub" in note for note in result.record.notes)


def test_baselines_claim_no_interval(welfare_arms, targets, bank):
    result = Simulation(
        _spec("uniform", welfare_arms), targets=targets, root=ROOT
    ).run()
    for prediction in result.predictions.values():
        assert not prediction.has_uncertainty


def test_the_prior_baseline_needs_a_calibration_bank(welfare_arms, targets):
    with pytest.raises(ValueError, match="calibration question bank"):
        Simulation(_spec("prior", welfare_arms), targets=targets, root=ROOT).run()


def test_the_prior_baseline_runs_when_given_one(welfare_arms, targets, bank):
    calibration, _ = bank.split([a.id for a in welfare_arms])
    result = Simulation(
        _spec("prior", welfare_arms), targets=targets, prior_bank=calibration, root=ROOT
    ).run()
    assert set(result.predictions) == {"natfare", "natfarey"}


def test_peer_influence_can_be_switched_on_from_the_spec(welfare_arms, targets):
    result = Simulation(
        _spec("hybrid-with-influence", welfare_arms), targets=targets, root=ROOT
    ).run()
    assert set(result.predictions) == {"natfare", "natfarey"}


def test_an_independent_population_still_hits_its_margins(welfare_arms, targets):
    result = Simulation(
        _spec("hybrid-independent-population", welfare_arms), targets=targets, root=ROOT
    ).run()
    assert result.fidelity.max_deviation < 1e-6


def test_the_gap_helper_reads_across_arms(welfare_arms, targets):
    result = Simulation(_spec("hybrid", welfare_arms), targets=targets, root=ROOT).run()
    gap = result.gap("too little", "natfare", "natfarey")
    expected = result.predictions["natfarey"].share("too little") - result.predictions[
        "natfare"
    ].share("too little")
    assert gap == pytest.approx(expected)
    assert result.topline().question_id == "welfare"


def test_the_run_record_carries_cost_and_provenance(welfare_arms, targets):
    result = Simulation(_spec("hybrid", welfare_arms), targets=targets, root=ROOT).run()
    record = result.record
    assert record.spec_fingerprint
    assert record.population_size == 800
    assert record.llm_calls > 0
    assert record.wall_seconds > 0
    assert set(record.results) == {"natfare", "natfarey"}


def test_the_anthropic_provider_is_selected_from_the_spec(welfare_arms, targets, bank):
    from quorum.predict.provider import AnthropicProvider
    from quorum.world.context import Scenario

    spec = _spec("hybrid", welfare_arms).model_copy(
        update={
            "predictor": _spec("hybrid", welfare_arms).predictor.model_copy(
                update={
                    "provider": _spec("hybrid", welfare_arms).predictor.provider.model_copy(
                        update={"name": "anthropic", "model": "claude-opus-5"}
                    )
                }
            )
        }
    )
    simulation = Simulation(spec, targets=targets, root=ROOT)
    scenario = Scenario.from_spec(spec, "natfare")
    provider = simulation.build_provider(scenario)
    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-opus-5"
    assert provider.options == scenario.options


def test_asking_the_model_about_every_agent_is_a_selectable_configuration(welfare_arms, targets):
    from quorum.predict.hybrid import DirectLLMPredictor
    from quorum.world.context import Scenario

    spec = build_spec("llm-every-agent", "welfare", welfare_arms)
    simulation = Simulation(spec, targets=targets, root=ROOT)
    predictor = simulation.build_predictor(Scenario.from_spec(spec, "natfare"))
    assert isinstance(predictor, DirectLLMPredictor)

    result = simulation.run()
    # One call per agent per arm, which is the cost the hybrid exists to avoid.
    assert result.record.llm_calls == 2 * spec.population.size


def test_a_population_too_small_to_cover_every_level_is_refused_with_a_reason(
    welfare_arms, targets
):
    """Raking cannot invent an agent, and the error should say that rather than
    leave a reader staring at a table of small numbers."""
    spec = build_spec(
        "hybrid", "welfare", welfare_arms, overrides={"population": {"size": 30}}
    )
    with pytest.raises(RuntimeError, match="too small to cover every level"):
        Simulation(spec, targets=targets, root=ROOT).run()
