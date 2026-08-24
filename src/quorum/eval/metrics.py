"""Scoring a predicted distribution against a known one.

Several metrics rather than one, because they disagree in useful ways. Mean absolute
error is what a reader of a topline notices. Total variation is the honest summary of
how far apart two distributions are. Earth mover's distance is the one that knows the
options are ordered, so predicting "about right" when the answer was "too little" is
a smaller mistake than predicting "too much". Brier and log score reward calibration
rather than the mode.

Two of these compare against something other than the truth's point estimate, and both
matter. ``within_truth_interval`` asks whether the prediction is inside the ground
truth's *own* sampling interval, which is the fairest bar available: a survey topline
from 1,600 respondents is itself an estimate, and a prediction cannot reasonably be
asked to beat its noise. ``skill_score`` asks whether the prediction beat a baseline,
because an error of three points means nothing until you know what guessing scores.
"""

from __future__ import annotations

import numpy as np

_EPSILON = 1e-12


def _check(prediction: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if prediction.shape != truth.shape:
        raise ValueError(f"shape mismatch: prediction {prediction.shape}, truth {truth.shape}")
    if prediction.ndim != 1:
        raise ValueError("metrics compare one distribution against one truth")
    return prediction, truth


def mean_absolute_error(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Average absolute error per option, in share units."""
    prediction, truth = _check(prediction, truth)
    return float(np.mean(np.abs(prediction - truth)))


def max_absolute_error(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction, truth = _check(prediction, truth)
    return float(np.max(np.abs(prediction - truth)))


def total_variation(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Half the summed absolute difference. Zero identical, one disjoint."""
    prediction, truth = _check(prediction, truth)
    return float(0.5 * np.sum(np.abs(prediction - truth)))


def earth_movers_distance(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Ordinal distance, normalized so one is the worst possible answer.

    Only meaningful when the options are ordered, which for these questions they are:
    too little, about right, too much is a scale, and being adjacent to the truth is
    genuinely better than being at the far end of it.
    """
    prediction, truth = _check(prediction, truth)
    if len(prediction) < 2:
        return 0.0
    return float(np.sum(np.abs(np.cumsum(prediction - truth))) / (len(prediction) - 1))


def brier_score(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Summed squared error. Proper: minimized by reporting the true distribution."""
    prediction, truth = _check(prediction, truth)
    return float(np.sum((prediction - truth) ** 2))


def log_score(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Cross entropy of the truth under the prediction. Lower is better.

    Punishes confident wrongness far harder than absolute error does, which is the
    point of including it: a simulator that says 99 percent when the answer is 60 is
    doing something worse than being three points off.
    """
    prediction, truth = _check(prediction, truth)
    return float(-np.sum(truth * np.log(np.clip(prediction, _EPSILON, None))))


def within_truth_interval(
    prediction: np.ndarray, truth: np.ndarray, standard_error: np.ndarray, z: float = 1.96
) -> float:
    """Share of options where the prediction lands inside the truth's own interval.

    The ground truth is a survey estimate, not a constant. Asking a prediction to sit
    inside the interval the survey itself would report is the fair version of "was it
    right", and it is stricter than it sounds: at 1,600 respondents that interval is
    about plus or minus two points.
    """
    prediction, truth = _check(prediction, truth)
    standard_error = np.asarray(standard_error, dtype=float)
    inside = np.abs(prediction - truth) <= z * standard_error
    return float(np.mean(inside))


def interval_coverage(interval: np.ndarray, truth: np.ndarray) -> float:
    """Share of options whose true value lies inside the predicted interval.

    Compared against the interval's nominal level, this is the calibration check that
    matters: a 90 percent interval that contains the truth 55 percent of the time is
    not a 90 percent interval.
    """
    interval = np.asarray(interval, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if interval.shape != (len(truth), 2):
        raise ValueError(f"interval must be (n_options, 2), got {interval.shape}")
    inside = (truth >= interval[:, 0]) & (truth <= interval[:, 1])
    return float(np.mean(inside))


def skill_score(error: float, baseline_error: float) -> float:
    """Fraction of a baseline's error removed. Zero is no better, one is perfect.

    Negative means the engine did worse than the baseline, which is a result worth
    being able to report rather than an outcome to be avoided.
    """
    if baseline_error <= 0:
        return 0.0 if error <= 0 else float("-inf")
    return float(1.0 - error / baseline_error)


def expected_calibration_error(
    predictions: np.ndarray, truths: np.ndarray, bins: int = 10
) -> float:
    """Average gap between a predicted probability and how often it was right.

    Pool every (question, option) pair across a backtest, bin by predicted
    probability, and compare each bin's average prediction to the truth's average
    share in it. A well calibrated engine that says 0.30 is right about 30 percent of
    the time across everything it said 0.30 about.
    """
    predictions = np.asarray(predictions, dtype=float).ravel()
    truths = np.asarray(truths, dtype=float).ravel()
    if predictions.shape != truths.shape:
        raise ValueError("predictions and truths must have the same shape")
    if predictions.size == 0:
        raise ValueError("cannot compute calibration on an empty set")
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(predictions, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        total += mask.mean() * abs(predictions[mask].mean() - truths[mask].mean())
    return float(total)


def reliability_curve(
    predictions: np.ndarray, truths: np.ndarray, bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per bin: mean prediction, mean truth, and how many pairs landed there."""
    predictions = np.asarray(predictions, dtype=float).ravel()
    truths = np.asarray(truths, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(predictions, edges[1:-1]), 0, bins - 1)
    mean_prediction = np.full(bins, np.nan)
    mean_truth = np.full(bins, np.nan)
    counts = np.zeros(bins)
    for b in range(bins):
        mask = index == b
        counts[b] = mask.sum()
        if mask.any():
            mean_prediction[b] = predictions[mask].mean()
            mean_truth[b] = truths[mask].mean()
    return mean_prediction, mean_truth, counts


def score_all(
    prediction: np.ndarray,
    truth: np.ndarray,
    standard_error: np.ndarray | None = None,
    interval: np.ndarray | None = None,
) -> dict[str, float]:
    """Every point metric at once, for one question."""
    scores = {
        "mae": mean_absolute_error(prediction, truth),
        "max_error": max_absolute_error(prediction, truth),
        "total_variation": total_variation(prediction, truth),
        "earth_movers": earth_movers_distance(prediction, truth),
        "brier": brier_score(prediction, truth),
        "log_score": log_score(prediction, truth),
    }
    if standard_error is not None:
        scores["within_truth_interval"] = within_truth_interval(prediction, truth, standard_error)
    if interval is not None:
        scores["interval_coverage"] = interval_coverage(interval, truth)
    return scores
