from __future__ import annotations

import json

import numpy as np
import pytest

from quorum.exec.cost import Budget, BudgetExceeded
from quorum.predict.cache import NullCache, ResponseCache
from quorum.predict.hybrid import DirectLLMPredictor, HybridPredictor, build_propagator
from quorum.predict.llm import LLMResponder
from quorum.predict.provider import Completion, StubProvider
from quorum.world.context import Scenario

OPTIONS = ("too little", "about right", "too much")


@pytest.fixture
def scenario() -> Scenario:
    return Scenario("natfare", "Are we spending too little on welfare?", OPTIONS)


@pytest.fixture
def responder() -> LLMResponder:
    return LLMResponder(StubProvider(options=OPTIONS), budget=Budget(max_usd=10.0))


class ScriptedProvider:
    """Returns exactly the texts it was given, in order."""

    name = "scripted"

    def __init__(self, texts: list[str], model: str = "scripted") -> None:
        self.texts = texts
        self.model = model
        self.seen: list[str] = []

    def complete(self, prompts, system, max_tokens=512):
        out = []
        for prompt in prompts:
            self.seen.append(prompt)
            text = self.texts[(len(self.seen) - 1) % len(self.texts)]
            out.append(Completion(text=text, input_tokens=10, output_tokens=5, model=self.model))
        return out


# -- prompts -----------------------------------------------------------------------


def test_the_system_prompt_carries_the_question_and_options(responder, scenario):
    system = responder.system_prompt(scenario)
    assert scenario.text in system
    assert "too little | about right | too much" in system
    assert "OPTIONS:" in system


def test_context_appears_in_the_system_prompt_when_given(responder):
    scenario = Scenario("q", "Agree?", ("yes", "no"), context="It is 2024.")
    assert "CONTEXT: It is 2024." in responder.system_prompt(scenario)


def test_the_agent_prompt_is_the_only_part_that_varies(responder, small_population):
    first = responder.agent_prompt(small_population, 0)
    second = responder.agent_prompt(small_population, 1)
    assert "age band" in first
    assert first != second


# -- parsing ---------------------------------------------------------------------


def test_parse_reads_a_clean_answer():
    parsed = LLMResponder.parse(json.dumps({"too little": 0.5, "about right": 0.3, "too much": 0.2}), OPTIONS)
    np.testing.assert_allclose(parsed.probabilities, [0.5, 0.3, 0.2])


def test_parse_survives_surrounding_prose():
    text = 'Here is my answer:\n{"too little": 1, "about right": 1, "too much": 2}\nHope that helps.'
    parsed = LLMResponder.parse(text, OPTIONS)
    np.testing.assert_allclose(parsed.probabilities, [0.25, 0.25, 0.5])


def test_parse_renormalizes_a_sloppy_distribution():
    parsed = LLMResponder.parse('{"too little": 0.6, "about right": 0.3, "too much": 0.2}', OPTIONS)
    assert parsed.probabilities.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "no json here",
        "{not valid json}",
        "[1, 2, 3]",
        '{"commentary": "I decline"}',
        '{"too little": 0, "about right": 0, "too much": 0}',
    ],
)
def test_parse_returns_none_rather_than_inventing_an_answer(text):
    assert LLMResponder.parse(text, OPTIONS) is None


# -- responding --------------------------------------------------------------------


def test_responding_produces_one_distribution_per_agent(responder, small_population, scenario):
    rows = responder.respond(small_population, scenario)
    assert rows.shape == (len(small_population), 3)
    np.testing.assert_allclose(rows.sum(axis=1), 1.0, atol=1e-3)
    assert responder.stats.live_calls == len(small_population)
    assert responder.stats.parse_failure_rate == 0.0


def test_a_second_run_is_served_from_cache(tmp_path, small_population, scenario):
    cache = ResponseCache.open(tmp_path)
    first = LLMResponder(StubProvider(options=OPTIONS), cache=cache, budget=Budget(max_usd=10.0))
    rows = first.respond(small_population, scenario)

    second = LLMResponder(StubProvider(options=OPTIONS), cache=ResponseCache.open(tmp_path), budget=Budget(max_usd=10.0))
    replayed = second.respond(small_population, scenario)

    np.testing.assert_allclose(rows, replayed)
    assert second.stats.live_calls == 0
    assert second.stats.cache_hits == len(small_population)


def test_unparseable_answers_become_nan_not_a_guess(small_population, scenario):
    responder = LLMResponder(ScriptedProvider(["not json"]), cache=NullCache(), budget=Budget(max_usd=10.0))
    rows = responder.respond(small_population, scenario)
    assert np.isnan(rows).all()
    assert responder.stats.parse_failure_rate == 1.0


def test_a_run_stops_when_it_would_breach_its_budget(small_population, scenario):
    responder = LLMResponder(
        ScriptedProvider(['{"too little": 1, "about right": 1, "too much": 1}'], model="claude-opus-5"),
        budget=Budget(max_usd=1e-9, max_calls=10_000),
    )
    with pytest.raises(BudgetExceeded):
        responder.respond(small_population, scenario)


def test_a_run_stops_when_it_would_breach_its_call_limit(small_population, scenario):
    responder = LLMResponder(StubProvider(options=OPTIONS), budget=Budget(max_usd=1e6, max_calls=5), batch_size=4)
    with pytest.raises(BudgetExceeded, match="limit is 5"):
        responder.respond(small_population, scenario)


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size"):
        LLMResponder(StubProvider(), batch_size=0)


def test_responder_stats_serialize(responder, small_population, scenario):
    responder.respond(small_population, scenario)
    assert responder.stats.as_dict()["agents"] == float(len(small_population))


# -- hybrid ------------------------------------------------------------------------


def test_the_hybrid_calls_the_model_once_per_archetype_not_once_per_agent(
    small_population, scenario, responder
):
    predictor = HybridPredictor(responder, ("age_band", "education"), archetypes=12)
    out = predictor.predict(small_population, scenario, seed=1)

    assert out.shape == (len(small_population), 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    assert responder.stats.agents == 12
    assert responder.stats.agents < len(small_population)
    assert predictor.diagnostics.cells == 6
    assert predictor.diagnostics.coverage == 1.0


def test_the_hybrid_drops_archetypes_it_could_not_read(small_population, scenario):
    good = '{"too little": 0.5, "about right": 0.3, "too much": 0.2}'
    responder = LLMResponder(
        ScriptedProvider([good, "garbage"]), cache=NullCache(), budget=Budget(max_usd=10.0)
    )
    predictor = HybridPredictor(responder, ("age_band", "education"), archetypes=12)
    predictor.predict(small_population, scenario, seed=1)
    assert predictor.diagnostics.usable_archetypes == 6
    assert predictor.diagnostics.coverage == pytest.approx(0.5)


def test_the_hybrid_refuses_to_guess_when_nothing_parsed(small_population, scenario):
    responder = LLMResponder(ScriptedProvider(["garbage"]), cache=NullCache(), budget=Budget(max_usd=10.0))
    predictor = HybridPredictor(responder, ("age_band",), archetypes=6)
    with pytest.raises(RuntimeError, match="no archetype produced a usable answer"):
        predictor.predict(small_population, scenario, seed=1)


def test_the_hybrid_can_disperse_within_cells(small_population, scenario, responder):
    plain = HybridPredictor(responder, ("age_band", "education"), archetypes=12, propagator="cell_mean")
    noisy = HybridPredictor(
        responder,
        ("age_band", "education"),
        archetypes=12,
        propagator="cell_mean",
        traits=("openness",),
        trait_noise=0.4,
    )
    flat_out = plain.predict(small_population, scenario, seed=1)
    noisy_out = noisy.predict(small_population, scenario, seed=1)
    assert noisy_out.std(axis=0).mean() > flat_out.std(axis=0).mean()


def test_the_hybrid_diagnostics_serialize(small_population, scenario, responder):
    predictor = HybridPredictor(responder, ("age_band",), archetypes=6)
    predictor.predict(small_population, scenario, seed=1)
    payload = predictor.diagnostics.as_dict()
    assert payload["propagator"] == "multinomial_logit"
    assert payload["responder_agents"] == 6.0


@pytest.mark.parametrize(
    "kwargs, message",
    [({"archetypes": 0}, "archetypes"), ({"stratify_by": ()}, "stratification dimension")],
)
def test_the_hybrid_validates_its_configuration(responder, kwargs, message):
    settings = {"stratify_by": ("age_band",), "archetypes": 4, **kwargs}
    with pytest.raises(ValueError, match=message):
        HybridPredictor(responder, **settings)


def test_build_propagator_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown propagator"):
        build_propagator("magic", ("age_band",))


# -- direct predictor --------------------------------------------------------------


def test_the_direct_predictor_asks_about_every_agent(small_population, scenario, responder):
    out = DirectLLMPredictor(responder).predict(small_population, scenario, seed=1)
    assert out.shape == (len(small_population), 3)
    assert responder.stats.agents == len(small_population)


def test_the_direct_predictor_fills_unreadable_agents_from_the_rest(small_population, scenario):
    good = '{"too little": 0.5, "about right": 0.3, "too much": 0.2}'
    responder = LLMResponder(ScriptedProvider([good, "garbage"]), cache=NullCache(), budget=Budget(max_usd=10.0))
    out = DirectLLMPredictor(responder).predict(small_population, scenario, seed=1)
    # Dropping them instead would silently reweight the population.
    assert out.shape[0] == len(small_population)
    assert not np.isnan(out).any()


def test_the_direct_predictor_refuses_to_guess_when_nothing_parsed(small_population, scenario):
    responder = LLMResponder(ScriptedProvider(["garbage"]), cache=NullCache(), budget=Budget(max_usd=10.0))
    with pytest.raises(RuntimeError, match="no agent produced a usable answer"):
        DirectLLMPredictor(responder).predict(small_population, scenario, seed=1)
