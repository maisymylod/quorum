from __future__ import annotations

import numpy as np
import pytest

from quorum.data.targets import MarginalTargets, Question, QuestionBank
from quorum.eval import metrics, report
from quorum.eval.configurations import BASE_SPEC, ENGINES, build_spec, merge
from quorum.eval.harness import (
    GAP_NOISE_FLOOR,
    Backtest,
    BacktestResult,
    ExperimentScore,
    QuestionScore,
    question_groups,
)

OPTIONS = ("too little", "about right", "too much")


# -- metrics -----------------------------------------------------------------------


def test_a_perfect_prediction_scores_zero_on_every_distance():
    truth = np.array([0.5, 0.3, 0.2])
    scores = metrics.score_all(truth, truth)
    for key in ("mae", "max_error", "total_variation", "earth_movers", "brier"):
        assert scores[key] == pytest.approx(0.0)


def test_the_metrics_disagree_in_the_way_they_should():
    """An adjacent mistake must cost less than an opposite one, ordinally."""
    truth = np.array([0.6, 0.3, 0.1])
    adjacent = np.array([0.3, 0.6, 0.1])
    opposite = np.array([0.1, 0.3, 0.6])
    assert metrics.earth_movers_distance(adjacent, truth) < metrics.earth_movers_distance(opposite, truth)
    # Total variation cannot tell them apart, which is why both are reported.
    assert metrics.total_variation(adjacent, truth) == pytest.approx(0.3)


def test_log_score_punishes_confident_wrongness_harder_than_absolute_error():
    truth = np.array([0.6, 0.4])
    timid = np.array([0.5, 0.5])
    confident = np.array([0.999, 0.001])
    assert metrics.mean_absolute_error(confident, truth) > metrics.mean_absolute_error(timid, truth)
    assert metrics.log_score(confident, truth) > 3 * metrics.log_score(timid, truth)


def test_within_truth_interval_uses_the_surveys_own_noise():
    truth = np.array([0.50, 0.50])
    close = np.array([0.51, 0.49])
    assert metrics.within_truth_interval(close, truth, np.array([0.02, 0.02])) == 1.0
    assert metrics.within_truth_interval(close, truth, np.array([0.001, 0.001])) == 0.0


def test_interval_coverage_counts_options_inside_the_band():
    interval = np.array([[0.4, 0.6], [0.1, 0.2]])
    assert metrics.interval_coverage(interval, np.array([0.5, 0.5])) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="interval must be"):
        metrics.interval_coverage(np.zeros((2, 3)), np.array([0.5, 0.5]))


def test_skill_score_can_report_doing_worse_than_the_baseline():
    assert metrics.skill_score(0.05, 0.10) == pytest.approx(0.5)
    assert metrics.skill_score(0.20, 0.10) == pytest.approx(-1.0)
    assert metrics.skill_score(0.0, 0.0) == 0.0
    assert metrics.skill_score(0.1, 0.0) == float("-inf")


def test_calibration_error_is_zero_for_a_calibrated_engine_and_grows_when_biased():
    rng = np.random.default_rng(0)
    truths = rng.random(4000)
    assert metrics.expected_calibration_error(truths, truths) == pytest.approx(0.0)
    assert metrics.expected_calibration_error(np.clip(truths + 0.2, 0, 1), truths) > 0.15


def test_reliability_curve_bins_and_counts():
    predictions = np.array([0.05, 0.15, 0.95])
    truths = np.array([0.10, 0.10, 0.90])
    mean_prediction, mean_truth, counts = metrics.reliability_curve(predictions, truths, bins=10)
    assert counts.sum() == 3
    assert mean_prediction[0] == pytest.approx(0.05)
    assert np.isnan(mean_truth[5])


def test_metrics_validate_their_inputs():
    with pytest.raises(ValueError, match="shape mismatch"):
        metrics.mean_absolute_error(np.array([0.5, 0.5]), np.array([1.0]))
    with pytest.raises(ValueError, match="one distribution"):
        metrics.mean_absolute_error(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="same shape"):
        metrics.expected_calibration_error(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="empty set"):
        metrics.expected_calibration_error(np.array([]), np.array([]))


def test_earth_movers_is_undefined_for_a_single_option():
    assert metrics.earth_movers_distance(np.array([1.0]), np.array([1.0])) == 0.0


# -- configuration grid ------------------------------------------------------------


def test_merge_is_deep_and_does_not_mutate():
    base = {"a": {"b": 1, "c": 2}}
    merged = merge(base, {"a": {"b": 9}})
    assert merged == {"a": {"b": 9, "c": 2}}
    assert base["a"]["b"] == 1


def test_every_configuration_in_the_grid_builds(bank_fixture):
    arms = [bank_fixture["a"], bank_fixture["b"]]
    for engine in ENGINES:
        spec = build_spec(engine, "split", arms)
        assert spec.name.endswith(":split")
        assert [a.id for a in spec.scenario.arms] == ["a", "b"]


def test_arms_of_one_group_share_a_spec_and_therefore_a_population(bank_fixture):
    spec = build_spec("hybrid", "split", [bank_fixture["a"], bank_fixture["b"]])
    prompts = spec.scenario.arm_prompts()
    assert prompts["a"] != prompts["b"]
    assert spec.seed == BASE_SPEC["seed"]


def test_build_spec_rejects_arms_that_disagree_on_options(bank_fixture):
    other = Question(
        id="c", text="C?", options=("yes", "no"),
        topline=np.array([0.5, 0.5]), standard_error=np.zeros(2), n=10, effective_n=10.0,
    )
    with pytest.raises(ValueError, match="do not share the options"):
        build_spec("hybrid", "split", [bank_fixture["a"], other])


def test_build_spec_validates_its_arguments(bank_fixture):
    with pytest.raises(KeyError, match="unknown engine"):
        build_spec("magic", "split", [bank_fixture["a"]])
    with pytest.raises(ValueError, match="at least one arm"):
        build_spec("hybrid", "split", [])


# -- harness -----------------------------------------------------------------------


def test_question_groups_keep_experiment_arms_together(bank_fixture):
    groups = question_groups(bank_fixture)
    assert len(groups) == 1
    group_id, arms, experiment = groups[0]
    assert group_id == "split"
    assert [a.id for a in arms] == ["a", "b"]
    assert experiment is not None


def test_question_groups_can_be_filtered(bank_fixture):
    assert question_groups(bank_fixture, only=["nothing"]) == []


def test_a_wording_gap_below_the_noise_floor_is_not_scored_on_its_sign():
    tiny = ExperimentScore("e", "yes", ("a", "b"), true_gap=0.005, predicted_gap=-0.4)
    real = ExperimentScore("e", "yes", ("a", "b"), true_gap=0.4, predicted_gap=0.3)
    assert not tiny.is_scoreable
    assert real.is_scoreable and real.sign_matches
    assert real.error == pytest.approx(0.1)
    assert GAP_NOISE_FLOOR > 0


def test_a_result_with_no_questions_has_no_summary():
    with pytest.raises(ValueError, match="no scored questions"):
        BacktestResult(engine="x").summary()


def test_a_flat_predictor_gets_zero_gap_correlation_rather_than_nan():
    result = BacktestResult(
        engine="flat",
        questions=[_score("q1"), _score("q2")],
        experiments=[
            ExperimentScore(f"e{i}", "too little", ("a", "b"), true_gap=0.1 * i, predicted_gap=0.0)
            for i in range(4)
        ],
    )
    assert result.experiment_summary()["gap_correlation"] == 0.0


def test_skill_is_reported_against_a_baseline():
    good = BacktestResult(engine="good", questions=[_score("q", error=0.02)])
    poor = BacktestResult(engine="poor", questions=[_score("q", error=0.10)])
    skill = good.with_skill_against(poor)
    assert skill["skill_mae"] > 0.7


# -- report ------------------------------------------------------------------------


def test_the_report_leads_with_a_warning_when_the_stub_produced_the_numbers():
    results = {"hybrid": BacktestResult(engine="hybrid", questions=[_score("q")])}
    text = report.render(results, 2024, "stub", "stub", 20_000, 300)
    assert "not from a model" in text
    assert text.index("not from a model") < text.index("## Leaderboard")


def test_the_report_omits_the_warning_for_a_real_provider():
    results = {"hybrid": BacktestResult(engine="hybrid", questions=[_score("q")])}
    assert "not from a model" not in report.render(results, 2024, "anthropic", "claude-opus-5", 1, 1)


def test_the_leaderboard_orders_by_error_and_marks_skill():
    results = {
        "prior": BacktestResult(engine="prior", questions=[_score("q", error=0.10)]),
        "hybrid": BacktestResult(engine="hybrid", questions=[_score("q", error=0.02)]),
    }
    table = report.leaderboard(results)
    assert table.index("`hybrid`") < table.index("`prior`")
    assert "+" in table


def test_the_experiment_table_flags_a_wrong_sign():
    result = BacktestResult(
        engine="e",
        questions=[_score("q")],
        experiments=[ExperimentScore("welfare", "too little", ("a", "b"), 0.37, -0.10)],
    )
    assert "**no**" in report.experiment_table(result)
    assert "No wording experiments" in report.experiment_table(BacktestResult(engine="e"))


def test_the_report_refuses_to_render_nothing():
    with pytest.raises(ValueError, match="no results"):
        report.render({}, 2024, "stub", "stub", 1, 1)
    with pytest.raises(ValueError, match="no results"):
        report.leaderboard({})


# -- end to end against the real question bank -------------------------------------


def test_a_backtest_runs_against_the_real_ground_truth():
    bank = QuestionBank.load("data/vendor/gss_questions.json")
    targets = MarginalTargets.load(BASE_SPEC["population"]["targets"])
    small = {"population": {"size": 400}, "predictor": {"archetypes": 40}, "estimator": {"draws": 100}}
    factory = lambda engine, group, arms: build_spec(engine, group, arms, overrides=small)
    backtest = Backtest(bank, factory, root=".", targets=targets)

    result = backtest.run("hybrid", only=["welfare", "cappun"])
    assert len(result.questions) == 3  # two arms plus one standalone question
    assert len(result.experiments) == 1
    summary = result.summary()
    assert 0.0 <= summary["mae"] <= 1.0
    assert "gap_sign_accuracy" in summary
    assert result.llm_calls > 0


def test_the_prior_baseline_never_sees_the_questions_it_is_scored_on():
    """Leave-one-group-out, so the baseline cannot be handed its own answers."""
    bank = QuestionBank.load("data/vendor/gss_questions.json")
    targets = MarginalTargets.load(BASE_SPEC["population"]["targets"])
    small = {"population": {"size": 300}}
    factory = lambda engine, group, arms: build_spec(engine, group, arms, overrides=small)
    result = Backtest(bank, factory, root=".", targets=targets).run("prior", only=["welfare"])

    welfare = next(q for q in result.questions if q.question_id == "natfarey")
    # If it had been fitted on the whole bank it would have seen this 70/20/9 split.
    assert abs(welfare.prediction[0] - welfare.truth[0]) > 0.1


# -- helpers -----------------------------------------------------------------------


def _score(question_id: str, error: float = 0.05) -> QuestionScore:
    truth = np.array([0.5, 0.3, 0.2])
    prediction = np.clip(truth + np.array([error, -error, 0.0]), 1e-6, None)
    prediction = prediction / prediction.sum()
    return QuestionScore(
        question_id=question_id,
        text=f"{question_id}?",
        options=OPTIONS,
        prediction=prediction,
        truth=truth,
        scores=metrics.score_all(prediction, truth),
    )


@pytest.fixture
def bank_fixture() -> QuestionBank:
    def question(qid: str, topline: list[float]) -> Question:
        return Question(
            id=qid, text=f"Spending on {qid}?", options=OPTIONS,
            topline=np.array(topline), standard_error=np.full(3, 0.01),
            n=900, effective_n=700.0, experiment="split", arm_label=qid,
        )

    from quorum.data.targets import Experiment

    return QuestionBank(
        source={}, year=2024, weight_variable="w",
        questions={"a": question("a", [0.3, 0.4, 0.3]), "b": question("b", [0.7, 0.2, 0.1])},
        experiments=(Experiment("split", "wording split", ("a", "b"), "too little"),),
    )


@pytest.mark.parametrize(
    "value, expected", [(float("nan"), "-"), (0.12345, "0.1235"), (12345.0, "12,345")]
)
def test_the_report_formats_awkward_numbers(value, expected):
    assert report._format(value) == expected


def test_run_notes_reach_the_report():
    result = BacktestResult(engine="hybrid", questions=[_score("q")], notes=["stub in use"])
    text = report.render({"hybrid": result}, 2024, "anthropic", "claude-opus-5", 1, 1)
    assert "Notes from the run" in text
    assert "stub in use" in text
