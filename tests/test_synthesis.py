from __future__ import annotations

import pandas as pd
import pytest

from quorum.data.targets import DEFAULT_MARGINALS, MarginalTargets
from quorum.synthesis import IndependenceSynthesizer, MicrodataSynthesizer, marginal_fidelity
from quorum.synthesis.validate import joint_divergence

ATTRIBUTES = ("age_band", "sex", "education", "race", "marital")
SEED_PATH = "data/vendor/acs_microdata.csv.gz"

#: The synthesis gate. Raking is exact arithmetic, so anything above this is a bug,
#: not sampling noise.
MAX_MARGINAL_DEVIATION = 1e-6


@pytest.fixture(scope="module")
def targets() -> MarginalTargets:
    return MarginalTargets.load(DEFAULT_MARGINALS)


@pytest.fixture(scope="module")
def microdata(targets) -> MicrodataSynthesizer:
    return MicrodataSynthesizer.from_csv(SEED_PATH, targets, ATTRIBUTES, ("openness",))


def test_microdata_synthesis_hits_every_target_margin(microdata, targets):
    population = microdata.synthesize(20_000, seed=20260824)
    report = marginal_fidelity(population, targets)
    assert report.max_deviation < MAX_MARGINAL_DEVIATION, report.table()
    assert microdata.last_raking.converged


def test_independence_synthesis_also_hits_every_target_margin(targets):
    synthesizer = IndependenceSynthesizer(targets, ATTRIBUTES, ("openness",))
    report = marginal_fidelity(synthesizer.synthesize(20_000, seed=5), targets)
    assert report.max_deviation < MAX_MARGINAL_DEVIATION, report.table()


def test_the_two_synthesizers_differ_only_in_joint_structure(targets, microdata):
    # Both match every one-way margin, so any divergence between them lives entirely
    # in how attributes travel together. Age and marital status is the clearest case:
    # widowhood is concentrated among the old, and independence cannot know that.
    independence = IndependenceSynthesizer(targets, ATTRIBUTES)
    a = microdata.synthesize(40_000, seed=7)
    b = independence.synthesize(40_000, seed=7)
    assert joint_divergence(b, a, ["age_band", "marital"]) > 0.15
    assert joint_divergence(a, a, ["age_band", "marital"]) == pytest.approx(0.0)


def test_synthesis_is_deterministic_and_seed_sensitive(microdata):
    a = microdata.synthesize(3_000, seed=1)
    b = microdata.synthesize(3_000, seed=1)
    c = microdata.synthesize(3_000, seed=2)
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_weights_scale_to_the_represented_population(microdata, targets):
    population = microdata.synthesize(5_000, seed=3)
    assert population.weight_sum == pytest.approx(targets.population_total, rel=1e-9)


def test_traits_are_assigned_in_the_unit_interval(microdata):
    population = microdata.synthesize(2_000, seed=4)
    assert population.traits == ("openness",)
    values = population.frame["trait_openness"].to_numpy()
    assert values.min() >= 0.0 and values.max() <= 1.0


def test_population_carries_only_the_declared_columns(microdata):
    population = microdata.synthesize(500, seed=9)
    assert population.attributes == ATTRIBUTES


@pytest.mark.parametrize("synthesizer_kind", ["microdata", "independence"])
def test_size_must_be_positive(targets, microdata, synthesizer_kind):
    synthesizer = (
        microdata if synthesizer_kind == "microdata" else IndependenceSynthesizer(targets, ATTRIBUTES)
    )
    with pytest.raises(ValueError, match="at least 1"):
        synthesizer.synthesize(0, seed=1)


def test_synthesizer_rejects_attributes_with_no_targets(targets):
    with pytest.raises(KeyError, match="no target marginals"):
        IndependenceSynthesizer(targets.subset(["sex"]), ("sex", "race"))


def test_microdata_synthesizer_requires_its_columns(targets):
    frame = pd.DataFrame({"sex": ["male"], "weight": [1.0]})
    with pytest.raises(KeyError, match="missing columns"):
        MicrodataSynthesizer(frame, targets, ATTRIBUTES)


def test_microdata_synthesizer_requires_a_weight_column(targets):
    frame = pd.DataFrame({a: ["x"] for a in ATTRIBUTES})
    with pytest.raises(KeyError, match="missing a weight column"):
        MicrodataSynthesizer(frame, targets, ATTRIBUTES)


def test_fidelity_report_surfaces_the_worst_level(microdata, targets):
    report = marginal_fidelity(microdata.synthesize(2_000, seed=6), targets)
    level, error = report.attributes["age_band"].worst_level(("18-24", "25-34", "35-44", "45-54", "55-64", "65+"))
    assert level in {"18-24", "25-34", "35-44", "45-54", "55-64", "65+"}
    assert abs(error) < MAX_MARGINAL_DEVIATION
    assert set(report.as_dict()) == {"total_absolute_error", "max_deviation", "srmse", "population_size"}
    assert "attribute" in report.table()


def test_fidelity_requires_shared_attributes(microdata, targets):
    population = microdata.synthesize(200, seed=8)
    with pytest.raises(ValueError, match="share no attributes"):
        marginal_fidelity(population, targets.subset([]))
