from __future__ import annotations

import numpy as np
import pytest

from quorum.core.prediction import Prediction, ResponseDistribution


def test_uniform_distribution():
    d = ResponseDistribution.uniform(["a", "b", "c", "d"])
    np.testing.assert_allclose(d.probabilities, 0.25)
    assert d.mode == "a"


def test_from_mapping_renormalizes_a_sloppy_model_answer():
    d = ResponseDistribution.from_mapping({"Agree": 0.6, "Disagree": 0.3}, ["Agree", "Disagree", "Neither"])
    assert pytest.approx(d.probabilities.sum()) == 1.0
    assert d["Neither"] == 0.0
    assert d.mode == "Agree"


def test_from_mapping_ignores_options_the_model_invented():
    d = ResponseDistribution.from_mapping({"Agree": 1.0, "Maybe": 5.0}, ["Agree", "Disagree"])
    np.testing.assert_allclose(d.probabilities, [1.0, 0.0])


def test_from_mapping_rejects_an_empty_answer():
    with pytest.raises(ValueError, match="no mass"):
        ResponseDistribution.from_mapping({"Maybe": 1.0}, ["Agree", "Disagree"])


def test_from_mapping_clamps_negative_mass():
    d = ResponseDistribution.from_mapping({"Agree": 1.0, "Disagree": -0.5}, ["Agree", "Disagree"])
    np.testing.assert_allclose(d.probabilities, [1.0, 0.0])


@pytest.mark.parametrize(
    "probs, message",
    [
        ([0.5, 0.5, 0.0], "expected 2 probabilities"),
        ([0.5, np.nan], "finite"),
        ([1.5, -0.5], "non-negative"),
        ([0.2, 0.2], "sum to 1"),
    ],
)
def test_distribution_validation(probs, message):
    with pytest.raises(ValueError, match=message):
        ResponseDistribution(("a", "b"), np.array(probs, dtype=float))


def test_distribution_lookup_and_dict():
    d = ResponseDistribution(("a", "b"), np.array([0.3, 0.7]))
    assert d["b"] == pytest.approx(0.7)
    assert d.as_dict() == pytest.approx({"a": 0.3, "b": 0.7})


def test_prediction_without_draws_has_a_degenerate_interval():
    p = Prediction("q", ("a", "b"), np.array([0.4, 0.6]))
    assert not p.has_uncertainty
    np.testing.assert_allclose(p.interval(), [[0.4, 0.4], [0.6, 0.6]])
    assert p.share("b") == pytest.approx(0.6)


def test_prediction_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    draws = rng.dirichlet([20, 30], size=4000)
    p = Prediction("q", ("a", "b"), draws.mean(axis=0), draws=draws)
    interval = p.interval(0.90)
    assert p.has_uncertainty
    for option in range(2):
        lo, hi = interval[option]
        assert lo < p.distribution[option] < hi
        assert 0.0 <= lo < hi <= 1.0


def test_prediction_validates_shapes():
    with pytest.raises(ValueError, match="does not match options"):
        Prediction("q", ("a", "b"), np.array([1.0]))
    with pytest.raises(ValueError, match="must sum to 1"):
        Prediction("q", ("a", "b"), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="draws must be"):
        Prediction("q", ("a", "b"), np.array([0.5, 0.5]), draws=np.zeros((5, 3)))


def test_prediction_converts_to_a_distribution():
    p = Prediction("q", ("a", "b"), np.array([0.25, 0.75]))
    assert p.as_distribution().mode == "b"
    assert p.as_dict() == pytest.approx({"a": 0.25, "b": 0.75})
