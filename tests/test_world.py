from __future__ import annotations

import numpy as np
import pytest

from quorum.core.spec import SimulationSpec
from quorum.world import BoundedConfidenceInfluence, HomophilyNetwork, NoInfluence, Scenario
from quorum.world.network import SocialGraph


# -- scenario ----------------------------------------------------------------------


def test_scenario_resolves_the_default_arm(spec):
    scenario = Scenario.from_spec(spec)
    assert scenario.arm == "default"
    assert scenario.text == "Do you agree?"
    assert scenario.options == ("Agree", "Disagree")


def test_scenario_resolves_each_named_arm(spec_dict):
    spec_dict["scenario"]["arms"] = [
        {"id": "welfare", "label": "welfare", "prompt": "spending on welfare"},
        {"id": "poor", "label": "the poor", "prompt": "assistance to the poor"},
    ]
    spec_dict["scenario"]["prompt"] = ""
    spec = SimulationSpec.from_dict(spec_dict)
    scenarios = Scenario.arms_from_spec(spec)
    assert [s.arm for s in scenarios] == ["welfare", "poor"]
    assert scenarios[1].arm_label == "the poor"


def test_scenario_rejects_an_unknown_arm(spec):
    with pytest.raises(KeyError, match="unknown arm"):
        Scenario.from_spec(spec, arm="nope")


def test_scenario_fingerprint_separates_arms_by_wording():
    a = Scenario("q", "spending on welfare", ("yes", "no"))
    b = Scenario("q", "assistance to the poor", ("yes", "no"))
    assert a.fingerprint() != b.fingerprint()
    assert a.fingerprint() == Scenario("q", "spending on welfare", ("yes", "no")).fingerprint()


@pytest.mark.parametrize(
    "text, options, message",
    [("q?", ("only",), "at least two"), ("   ", ("a", "b"), "question text")],
)
def test_scenario_validation(text, options, message):
    with pytest.raises(ValueError, match=message):
        Scenario("q", text, options)


# -- network -----------------------------------------------------------------------


def test_graph_reaches_the_requested_mean_degree(small_population):
    graph = HomophilyNetwork(mean_degree=6, homophily=0.5, dimensions=("age_band",)).build(
        small_population, seed=1
    )
    assert 4.0 < graph.degrees.mean() < 8.0
    assert graph.n_edges == len(graph.neighbours) // 2


def test_homophily_parameter_shows_up_in_the_built_graph(small_population):
    labels = small_population.frame["age_band"].to_numpy()
    low = HomophilyNetwork(8, 0.0, ("age_band",)).build(small_population, seed=2)
    high = HomophilyNetwork(8, 0.95, ("age_band",)).build(small_population, seed=2)
    assert high.homophily(labels) > low.homophily(labels) + 0.3


def test_graph_is_symmetric_and_has_no_self_loops(small_population):
    graph = HomophilyNetwork(6, 0.6, ("age_band",)).build(small_population, seed=3)
    for agent in range(len(small_population)):
        for neighbour in graph.neighbours_of(agent):
            assert neighbour != agent
            assert agent in graph.neighbours_of(neighbour)


def test_graph_construction_is_deterministic(small_population):
    build = lambda seed: HomophilyNetwork(6, 0.7, ("age_band",)).build(small_population, seed)
    np.testing.assert_array_equal(build(4).neighbours, build(4).neighbours)
    assert not np.array_equal(build(4).neighbours, build(5).neighbours)


@pytest.mark.parametrize(
    "kwargs, message",
    [({"mean_degree": 0}, "mean_degree"), ({"homophily": 1.5}, "homophily")],
)
def test_network_validates_its_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        HomophilyNetwork(**kwargs)


def test_network_needs_two_agents(small_population):
    single = small_population.subset(np.arange(len(small_population)) == 0)
    with pytest.raises(ValueError, match="at least two agents"):
        HomophilyNetwork().build(single, seed=1)


def test_graph_rejects_inconsistent_offsets():
    with pytest.raises(ValueError, match="one more entry"):
        SocialGraph(np.array([1, 0]), np.array([0, 1]), n_agents=5)


def test_homophily_of_an_empty_graph_is_undefined():
    graph = SocialGraph(np.array([], dtype=int), np.zeros(3, dtype=int), n_agents=2)
    assert np.isnan(graph.homophily(np.array(["a", "b"])))


# -- influence ---------------------------------------------------------------------


@pytest.fixture
def responses(small_population) -> np.ndarray:
    rng = np.random.default_rng(0)
    raw = rng.dirichlet([2.0, 2.0, 2.0], size=len(small_population))
    return raw


def test_no_influence_is_the_identity(small_population, responses):
    np.testing.assert_allclose(
        NoInfluence().influence(small_population, responses, seed=1), responses
    )


def test_influence_narrows_disagreement_without_moving_the_aggregate(small_population, responses):
    network = HomophilyNetwork(8, 0.5, ("age_band",))
    model = BoundedConfidenceInfluence(network, rounds=4, confidence=1.0, susceptibility=0.4)
    after = model.influence(small_population, responses, seed=2)
    assert after.std(axis=0).mean() < responses.std(axis=0).mean()
    before_topline = small_population.weighted_distribution(responses)
    after_topline = small_population.weighted_distribution(after)
    assert np.max(np.abs(after_topline - before_topline)) < 0.05
    np.testing.assert_allclose(after.sum(axis=1), 1.0)


def test_a_tight_confidence_bound_stops_influence_spreading(small_population, responses):
    network = HomophilyNetwork(8, 0.5, ("age_band",))
    tight = BoundedConfidenceInfluence(network, rounds=4, confidence=0.01, susceptibility=0.4)
    loose = BoundedConfidenceInfluence(network, rounds=4, confidence=1.0, susceptibility=0.4)
    tight_spread = tight.influence(small_population, responses, seed=3).std(axis=0).mean()
    loose_spread = loose.influence(small_population, responses, seed=3).std(axis=0).mean()
    assert tight_spread > loose_spread


@pytest.mark.parametrize("kwargs", [{"rounds": 0}, {"susceptibility": 0.0}])
def test_influence_is_a_no_op_when_switched_off(small_population, responses, kwargs):
    model = BoundedConfidenceInfluence(HomophilyNetwork(), **kwargs)
    np.testing.assert_allclose(model.influence(small_population, responses, seed=1), responses)


def test_influence_records_the_graph_it_used(small_population, responses):
    model = BoundedConfidenceInfluence(HomophilyNetwork(6, 0.5, ("age_band",)))
    model.influence(small_population, responses, seed=1)
    assert model.last_graph is not None
    assert model.last_graph.n_agents == len(small_population)


def test_influence_validates_the_response_matrix(small_population):
    model = BoundedConfidenceInfluence(HomophilyNetwork())
    with pytest.raises(ValueError, match="response matrix"):
        model.influence(small_population, np.zeros((3, 2)), seed=1)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"rounds": -1}, "rounds"),
        ({"confidence": 2.0}, "confidence"),
        ({"susceptibility": -0.1}, "susceptibility"),
    ],
)
def test_influence_validates_its_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BoundedConfidenceInfluence(HomophilyNetwork(), **kwargs)
