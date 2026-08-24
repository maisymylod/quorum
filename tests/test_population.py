from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quorum.core.agent import Agent
from quorum.core.population import Population, _largest_remainder


def test_requires_weight_column():
    with pytest.raises(ValueError, match="weight"):
        Population(pd.DataFrame({"age_band": ["18-34"]}))


@pytest.mark.parametrize(
    "weights, message",
    [
        ([1.0, np.nan], "finite"),
        ([1.0, -1.0], "non-negative"),
        ([0.0, 0.0], "positive"),
    ],
)
def test_rejects_bad_weights(weights, message):
    frame = pd.DataFrame({"age_band": ["a", "b"], "weight": weights})
    with pytest.raises(ValueError, match=message):
        Population(frame)


def test_attributes_and_traits_are_separated(small_population):
    assert small_population.attributes == ("age_band", "education")
    assert small_population.traits == ("openness",)


def test_marginals_sum_to_one_and_are_weighted(small_population):
    marginals = small_population.marginals("age_band")
    assert pytest.approx(marginals.sum()) == 1.0
    # weights vary by row, so a weighted marginal must differ from the raw count share
    counts = small_population.frame["age_band"].value_counts(normalize=True).sort_index()
    assert not np.allclose(marginals.to_numpy(), counts.to_numpy())


def test_marginals_rejects_unknown_dimension(small_population):
    with pytest.raises(KeyError):
        small_population.marginals("income_band")


def test_cells_cover_the_population(small_population):
    cells = small_population.cells(["age_band", "education"])
    assert len(cells) == 6
    assert pytest.approx(cells["share"].sum()) == 1.0
    assert pytest.approx(cells["weight"].sum()) == small_population.weight_sum


def test_cell_index_aligns_with_cells(small_population):
    dims = ["age_band", "education"]
    cells = small_population.cells(dims)
    idx = small_population.cell_index(dims)
    assert idx.min() == 0 and idx.max() == len(cells) - 1
    # every agent's cell id must point back at its own attribute values
    frame = small_population.frame
    for i in (0, 5, 77, 119):
        row = cells.iloc[idx[i]]
        assert row["age_band"] == frame.iloc[i]["age_band"]
        assert row["education"] == frame.iloc[i]["education"]


def test_weighted_distribution_matches_hand_computation():
    frame = pd.DataFrame({"g": ["a", "b"], "weight": [3.0, 1.0]})
    population = Population(frame)
    responses = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(population.weighted_distribution(responses), [0.75, 0.25])


def test_weighted_distribution_validates_shape(small_population):
    with pytest.raises(ValueError, match="matrix"):
        small_population.weighted_distribution(np.zeros((3, 2)))


def test_weighted_mean_validates_length(small_population):
    with pytest.raises(ValueError, match="expected 120 values"):
        small_population.weighted_mean(np.zeros(5))


def test_with_weights_does_not_mutate_the_original(small_population):
    before = small_population.weight_sum
    reweighted = small_population.with_weights(np.ones(len(small_population)))
    assert reweighted.weight_sum == len(small_population)
    assert small_population.weight_sum == before


def test_with_weights_validates_shape(small_population):
    with pytest.raises(ValueError, match="expected 120 weights"):
        small_population.with_weights(np.ones(3))


def test_subset_and_with_column(small_population):
    tagged = small_population.with_column("flag", np.arange(len(small_population)))
    young = tagged.subset(tagged.frame["age_band"].to_numpy() == "18-34")
    assert set(young.frame["age_band"]) == {"18-34"}
    assert "flag" in young.frame.columns


def test_sample_is_deterministic_and_sized(small_population):
    a = small_population.sample(30, seed=3)
    b = small_population.sample(30, seed=3)
    c = small_population.sample(30, seed=4)
    assert len(a) == 30
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_sample_larger_than_population_returns_everything(small_population):
    assert len(small_population.sample(10_000, seed=1)) == len(small_population)


def test_unweighted_sample_ignores_weights(small_population):
    sample = small_population.sample(30, seed=3, weighted=False)
    assert len(sample) == 30


def test_stratified_sample_covers_every_cell(small_population):
    dims = ["age_band", "education"]
    sample, index = small_population.stratified_sample(20, dims, seed=5)
    assert len(sample) == 20
    assert len(index) == 20
    covered = set(map(tuple, sample.frame[dims].to_numpy()))
    assert covered == set(map(tuple, small_population.frame[dims].to_numpy()))


def test_stratified_sample_refuses_an_impossible_budget(small_population):
    with pytest.raises(ValueError, match="cannot cover"):
        small_population.stratified_sample(3, ["age_band", "education"], seed=1)


def test_agent_view_round_trips(small_population):
    agent = small_population.agent(4)
    assert isinstance(agent, Agent)
    assert agent.id == 4
    assert set(agent.attributes) == {"age_band", "education"}
    assert 0.0 <= agent.traits["openness"] <= 1.0
    assert agent.cell(("age_band",)) == (agent.attributes["age_band"],)
    assert "age band" in agent.describe()


def test_iter_agents_yields_every_agent(small_population):
    assert sum(1 for _ in small_population.iter_agents()) == len(small_population)


def test_fingerprint_is_sensitive_to_values(small_population):
    other = small_population.with_column("extra", np.ones(len(small_population)))
    assert small_population.fingerprint() != other.fingerprint()


def test_from_records_defaults_the_weight():
    population = Population.from_records([{"g": "a"}, {"g": "b"}])
    assert population.weight_sum == 2.0


def test_largest_remainder_sums_exactly_and_respects_the_minimum():
    alloc = _largest_remainder(np.array([0.7, 0.2, 0.1]), total=10, minimum=1)
    assert alloc.sum() == 10
    assert alloc.min() >= 1


def test_largest_remainder_rejects_an_infeasible_minimum():
    with pytest.raises(ValueError, match="cannot give"):
        _largest_remainder(np.array([0.5, 0.5]), total=1, minimum=1)
