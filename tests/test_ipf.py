from __future__ import annotations

import numpy as np
import pytest

from quorum.synthesis.ipf import encode, rake


def test_raking_hits_the_targets_it_is_given():
    codes = {"sex": np.array([0, 0, 0, 1])}
    result = rake(codes, {"sex": np.array([0.5, 0.5])})
    assert result.converged
    weights = result.weights
    assert weights[:3].sum() == pytest.approx(weights[3])


def test_raking_reconciles_two_attributes_at_once():
    rng = np.random.default_rng(0)
    n = 4000
    codes = {
        "age_band": rng.integers(0, 3, n),
        "education": rng.integers(0, 2, n),
    }
    targets = {"age_band": np.array([0.5, 0.3, 0.2]), "education": np.array([0.7, 0.3])}
    result = rake(codes, targets)
    assert result.converged
    for attribute, target in targets.items():
        achieved = np.bincount(codes[attribute], weights=result.weights)
        np.testing.assert_allclose(achieved / achieved.sum(), target, atol=1e-8)


def test_raking_preserves_the_joint_structure_of_the_sample():
    # Every agent in the same cell must be scaled by the same factor, which is what
    # lets raking fix margins without inventing associations.
    codes = {"sex": np.array([0, 0, 1, 1]), "race": np.array([0, 0, 1, 1])}
    result = rake(codes, {"sex": np.array([0.25, 0.75]), "race": np.array([0.25, 0.75])})
    assert result.weights[0] == pytest.approx(result.weights[1])
    assert result.weights[2] == pytest.approx(result.weights[3])


def test_raking_starts_from_supplied_weights():
    codes = {"sex": np.array([0, 1])}
    flat = rake(codes, {"sex": np.array([0.5, 0.5])}, initial_weights=np.array([1.0, 1.0]))
    tilted = rake(codes, {"sex": np.array([0.5, 0.5])}, initial_weights=np.array([9.0, 1.0]))
    # Both reach the same shares, but the second keeps the scale it was handed.
    assert flat.weights.sum() != pytest.approx(tilted.weights.sum())


def test_raking_reports_levels_the_sample_cannot_represent():
    codes = {"sex": np.array([0, 0, 0])}
    result = rake(codes, {"sex": np.array([0.5, 0.5])}, max_iterations=5)
    assert not result.converged
    assert result.empty_levels["sex"] == ("1",)


def test_raking_reports_non_convergence_within_the_iteration_budget():
    rng = np.random.default_rng(1)
    codes = {"a": rng.integers(0, 4, 500), "b": rng.integers(0, 4, 500)}
    targets = {"a": np.full(4, 0.25), "b": np.full(4, 0.25)}
    result = rake(codes, targets, max_iterations=1, tolerance=1e-15)
    assert not result.converged
    assert result.iterations == 1
    assert "did not converge" in result.summary()


def test_history_decreases_towards_convergence():
    rng = np.random.default_rng(2)
    codes = {"a": rng.integers(0, 3, 800), "b": rng.integers(0, 2, 800)}
    result = rake(codes, {"a": np.array([0.6, 0.3, 0.1]), "b": np.array([0.4, 0.6])})
    assert result.history[0] > result.history[-1]
    assert "converged" in result.summary()


@pytest.mark.parametrize(
    "targets, message",
    [
        ({}, "at least one target"),
        ({"sex": np.array([0.4, 0.4])}, "sum to"),
    ],
)
def test_raking_validates_targets(targets, message):
    with pytest.raises(ValueError, match=message):
        rake({"sex": np.array([0, 1])}, targets)


def test_raking_requires_codes_for_every_target():
    with pytest.raises(KeyError, match="no agent codes"):
        rake({"sex": np.array([0, 1])}, {"race": np.array([0.5, 0.5])})


def test_raking_rejects_ragged_codes():
    codes = {"sex": np.array([0, 1]), "race": np.array([0])}
    targets = {"sex": np.array([0.5, 0.5]), "race": np.array([1.0])}
    with pytest.raises(ValueError, match="different length"):
        rake(codes, targets)


@pytest.mark.parametrize(
    "weights, message",
    [(np.ones(3), "must have shape"), (np.array([-1.0, 1.0]), "non-negative")],
)
def test_raking_validates_initial_weights(weights, message):
    with pytest.raises(ValueError, match=message):
        rake({"sex": np.array([0, 1])}, {"sex": np.array([0.5, 0.5])}, initial_weights=weights)


def test_raking_rejects_all_zero_weights():
    with pytest.raises(ValueError, match="collapsed to zero"):
        rake(
            {"sex": np.array([0, 1])},
            {"sex": np.array([0.5, 0.5])},
            initial_weights=np.zeros(2),
        )


def test_encode_maps_labels_to_level_order():
    np.testing.assert_array_equal(encode(["b", "a", "b"], ["a", "b"]), [1, 0, 1])


def test_encode_rejects_an_unknown_label():
    with pytest.raises(ValueError, match="not one of"):
        encode(["c"], ["a", "b"])
