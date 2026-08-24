from __future__ import annotations

import numpy as np
import pytest

from quorum.core.prediction import Prediction
from quorum.data.targets import MarginalTargets, QuestionBank
from quorum.eval.configurations import BASE_SPEC, build_spec
from quorum.exec.runner import Simulation
from quorum.publish import Contrast, publish, render_html, render_markdown
from quorum.publish.charts import interval_plot, legend, stacked_bar
from quorum.publish.report import contrasts

OPTIONS = ("too little", "about right", "too much")


@pytest.fixture(scope="module")
def result():
    bank = QuestionBank.load("data/vendor/gss_questions.json")
    targets = MarginalTargets.load(BASE_SPEC["population"]["targets"])
    experiment = next(e for e in bank.experiments if e.id == "welfare")
    spec = build_spec(
        "hybrid",
        "welfare",
        [bank[a] for a in experiment.arms],
        overrides={
            "population": {"size": 900},
            "predictor": {"archetypes": 60},
            "estimator": {"draws": 300},
        },
    )
    return Simulation(spec, targets=targets, root=".").run()


# -- contrast ----------------------------------------------------------------------


def _prediction(shares, draws=None) -> Prediction:
    return Prediction("q", OPTIONS, np.array(shares, dtype=float), draws=draws)


def test_a_contrast_is_the_difference_between_two_arms():
    contrast = Contrast.between("a", _prediction([0.3, 0.4, 0.3]), "b", _prediction([0.7, 0.2, 0.1]))
    np.testing.assert_allclose(contrast.difference, [0.4, -0.2, -0.2])
    assert contrast.shift("too little") == pytest.approx(0.4)


def test_a_contrast_keeps_the_dependence_between_arms():
    """Arms share a population, so differencing draw by draw is tighter and right."""
    rng = np.random.default_rng(0)
    shared = rng.dirichlet([30, 30, 30], size=800)
    a = _prediction(shared.mean(axis=0), draws=shared)
    b = _prediction(shared.mean(axis=0), draws=shared)
    contrast = Contrast.between("a", a, "b", b)
    # Identical draws: the difference is exactly zero, not merely centred on it.
    assert np.abs(contrast.draws).max() == pytest.approx(0.0)
    assert not contrast.resolves("too little")


def test_a_contrast_resolves_a_direction_when_the_interval_excludes_zero():
    rng = np.random.default_rng(1)
    a = _prediction([0.3, 0.4, 0.3], draws=rng.dirichlet([30, 40, 30], size=800))
    b = _prediction([0.7, 0.2, 0.1], draws=rng.dirichlet([70, 20, 10], size=800))
    contrast = Contrast.between("a", a, "b", b)
    assert contrast.resolves("too little")
    interval = contrast.interval(0.90)
    assert interval[0][0] > 0


def test_a_contrast_without_draws_claims_no_interval():
    contrast = Contrast.between("a", _prediction([0.5, 0.3, 0.2]), "b", _prediction([0.4, 0.4, 0.2]))
    np.testing.assert_allclose(contrast.interval()[:, 0], contrast.interval()[:, 1])
    assert not contrast.resolves("too little")


def test_a_contrast_needs_matching_options():
    other = Prediction("q", ("yes", "no"), np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="share their response options"):
        Contrast.between("a", _prediction([0.5, 0.3, 0.2]), "b", other)


# -- charts ------------------------------------------------------------------------


def test_a_stacked_bar_is_self_contained_svg():
    svg = stacked_bar(np.array([0.5, 0.3, 0.2]), OPTIONS)
    assert svg.startswith("<svg") and svg.count("<rect") == 3
    assert "http" not in svg  # nothing to fetch
    assert "50%" in svg


def test_a_narrow_segment_gets_no_label_rather_than_an_unreadable_one():
    assert stacked_bar(np.array([0.99, 0.01]), ("a", "b")).count("<text") == 1


def test_the_interval_plot_draws_a_zero_line_only_when_zero_is_in_range():
    spanning = interval_plot(("a",), np.array([0.0]), np.array([[-0.1, 0.1]]))
    positive = interval_plot(("a",), np.array([0.5]), np.array([[0.4, 0.6]]))
    assert "stroke-dasharray" in spanning
    assert "stroke-dasharray" not in positive


def test_the_interval_plot_survives_a_zero_width_interval():
    svg = interval_plot(("a",), np.array([0.5]), np.array([[0.5, 0.5]]))
    assert svg.startswith("<svg")


def test_the_legend_escapes_its_labels():
    assert "&lt;script&gt;" in legend(("<script>",))


# -- report ------------------------------------------------------------------------


def test_the_html_report_is_one_self_contained_file(result):
    html = render_html(result)
    assert html.startswith("<!doctype html>")
    assert "<style>" in html
    for marker in ("http://", "https://", "<script", "<img"):
        assert marker not in html


def test_the_report_shows_every_arm_and_the_contrast_between_them(result):
    html = render_html(result)
    for arm in result.predictions:
        assert arm in html
    assert "What changing the wording did" in html
    assert "Assistance to the poor" in html


def test_the_report_states_where_its_numbers_came_from(result):
    html = render_html(result)
    assert "Provenance" in html
    assert result.record.spec_fingerprint in html
    assert result.record.population_fingerprint in html
    assert "Constructed" in html and "Real" in html


def test_a_stub_report_says_so_before_any_number(result):
    html = render_html(result)
    assert "offline stub" in html
    assert html.index("offline stub") < html.index("What the population said")


def test_a_real_provider_report_carries_no_stub_warning(result):
    result.record.provider = "anthropic"
    result.record.model = "claude-opus-5"
    try:
        html = render_html(result)
        assert "offline stub" not in html
        assert "claude-opus-5" in html
    finally:
        result.record.provider = "stub"
        result.record.model = "stub"


def test_the_markdown_report_carries_the_same_content(result):
    text = render_markdown(result)
    assert text.startswith("# welfare")
    assert "Provenance" in text
    for arm in result.predictions:
        assert arm in text
    assert "offline stub" in text


def test_publishing_writes_every_requested_format(tmp_path, result):
    written = publish(result, tmp_path)
    assert set(written) == {"html", "markdown", "json"}
    for path in written.values():
        assert path.exists() and path.stat().st_size > 0


def test_publishing_honours_a_narrower_format_list(tmp_path, result):
    trimmed = result.spec.model_copy(
        update={"output": result.spec.output.model_copy(update={"formats": ["markdown"]})}
    )
    result.spec, original = trimmed, result.spec
    try:
        assert set(publish(result, tmp_path)) == {"markdown"}
    finally:
        result.spec = original


def test_a_single_arm_run_has_nothing_to_contrast(result):
    single = type(result)(
        spec=result.spec,
        record=result.record,
        population=result.population,
        predictions={next(iter(result.predictions)): next(iter(result.predictions.values()))},
        fidelity=result.fidelity,
    )
    assert contrasts(single) == []
    assert "What changing the wording did" not in render_html(single)
