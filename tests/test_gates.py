from __future__ import annotations

import pytest

from quorum.eval import gates


def _summaries(**overrides) -> dict[str, dict[str, float]]:
    base = {
        "prior": {"mae": 0.12},
        "hybrid": {
            "questions": 41.0,
            "mae": 0.05,
            "skill_mae": 0.55,
            "ece": 0.03,
            "interval_coverage": 0.88,
            "gap_sign_accuracy": 0.9,
            "gap_mae": 0.05,
        },
    }
    for engine, values in overrides.items():
        base.setdefault(engine, {}).update(values)
    return base


def test_prediction_gates_are_skipped_for_the_stub():
    results = gates.check(_summaries(), provider="stub")
    skipped = [r.gate.name for r in results if r.skipped]
    assert "reads the wording" in skipped
    assert "calibrated" in skipped
    # Plumbing gates still run, and a skipped gate is not a failure.
    assert not gates.failures(results)
    assert any(r.gate.name == "every question scored" and not r.skipped for r in results)


def test_prediction_gates_run_for_a_real_provider():
    results = gates.check(_summaries(), provider="anthropic")
    assert not any(r.skipped for r in results)
    assert not gates.failures(results)


def test_a_flat_engine_fails_the_wording_gate():
    """An engine that predicts both arms alike must not pass, however low its MAE."""
    results = gates.check(
        _summaries(hybrid={"gap_sign_accuracy": 0.4, "mae": 0.01}), provider="anthropic"
    )
    failed = {r.gate.name for r in gates.failures(results)}
    assert "reads the wording" in failed


def test_an_engine_that_cannot_beat_the_baseline_fails():
    results = gates.check(_summaries(hybrid={"skill_mae": -0.2}), provider="anthropic")
    assert "beats the prior baseline" in {r.gate.name for r in gates.failures(results)}


def test_an_overconfident_interval_fails():
    results = gates.check(_summaries(hybrid={"interval_coverage": 0.4}), provider="anthropic")
    assert "intervals mean what they say" in {r.gate.name for r in gates.failures(results)}


def test_a_backtest_that_skipped_questions_fails():
    results = gates.check(_summaries(hybrid={"questions": 4.0}), provider="stub")
    assert "every question scored" in {r.gate.name for r in gates.failures(results)}


def test_a_missing_metric_is_a_failure_not_a_pass():
    """A gate whose metric never arrived must not be silently satisfied."""
    results = gates.check({"prior": {"mae": 0.12}}, provider="stub")
    assert "every question scored" in {r.gate.name for r in gates.failures(results)}


def test_every_gate_explains_itself():
    for gate in gates.GATES:
        assert gate.rationale
        assert gate.direction in {"at_most", "at_least"}
        assert gate.applies in {"any", "model"}


def test_results_describe_themselves():
    results = gates.check(_summaries(), provider="anthropic")
    text = "\n".join(r.describe() for r in results)
    assert "pass" in text
    failed = gates.check(_summaries(hybrid={"ece": 0.5}), provider="anthropic")
    assert "FAIL" in "\n".join(r.describe() for r in failed)
    skipped = gates.check(_summaries(), provider="stub")
    assert "skip" in "\n".join(r.describe() for r in skipped)


@pytest.mark.parametrize(
    "direction, bound, value, expected",
    [("at_most", 0.1, 0.05, True), ("at_most", 0.1, 0.2, False),
     ("at_least", 0.7, 0.9, True), ("at_least", 0.7, 0.5, False)],
)
def test_a_gate_compares_in_the_direction_it_declares(direction, bound, value, expected):
    gate = gates.Gate("g", "m", bound, direction, "hybrid", "because")
    assert gate.holds(value) is expected
