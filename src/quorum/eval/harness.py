"""The backtest: run the engine over a bank of real questions and score every answer.

This is the part of the repository that decides whether anything else in it works. It
takes a configuration, runs it against questions whose published answers the engine
never sees, and reports how far off it was, next to what a baseline scored on the same
questions. An error of three points is not a result until you know what guessing gets.

Wording experiments are scored separately and more strictly. For those, the ground
truth includes a difference that is *caused*, not merely observed: the same population
answered two forms of one question, assigned at random, so the gap between the arms is
attributable to the wording alone. Predicting both toplines well while predicting no
gap would look fine on average error and would mean the engine is not reading the
question at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from quorum.core.spec import SimulationSpec
from quorum.data.targets import Experiment, Question, QuestionBank
from quorum.eval import metrics
from quorum.exec.runner import Simulation

#: A wording gap smaller than this is inside the noise of two survey halves, so its
#: sign is not something a simulator can fairly be scored on.
GAP_NOISE_FLOOR = 0.03


@dataclass(slots=True)
class QuestionScore:
    """One question, one arm, scored."""

    question_id: str
    text: str
    options: tuple[str, ...]
    prediction: np.ndarray
    truth: np.ndarray
    scores: dict[str, float]
    experiment: str | None = None
    interval: np.ndarray | None = None

    @property
    def mae(self) -> float:
        return self.scores["mae"]


@dataclass(slots=True)
class ExperimentScore:
    """One randomized wording split, scored on the gap it produced."""

    experiment_id: str
    contrast_option: str
    arms: tuple[str, ...]
    true_gap: float
    predicted_gap: float

    @property
    def error(self) -> float:
        return abs(self.predicted_gap - self.true_gap)

    @property
    def is_scoreable(self) -> bool:
        """Whether the true gap is big enough for its sign to mean anything."""
        return abs(self.true_gap) >= GAP_NOISE_FLOOR

    @property
    def sign_matches(self) -> bool:
        return np.sign(self.predicted_gap) == np.sign(self.true_gap)


@dataclass(slots=True)
class BacktestResult:
    """Everything one configuration scored, and what it cost."""

    engine: str
    questions: list[QuestionScore] = field(default_factory=list)
    experiments: list[ExperimentScore] = field(default_factory=list)
    cost_usd: float = 0.0
    llm_calls: int = 0
    wall_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        if not self.questions:
            raise ValueError("a backtest with no scored questions has no summary")
        pooled_prediction = np.concatenate([q.prediction for q in self.questions])
        pooled_truth = np.concatenate([q.truth for q in self.questions])

        out = {
            "questions": float(len(self.questions)),
            "mae": float(np.mean([q.scores["mae"] for q in self.questions])),
            "max_error": float(np.mean([q.scores["max_error"] for q in self.questions])),
            "total_variation": float(np.mean([q.scores["total_variation"] for q in self.questions])),
            "earth_movers": float(np.mean([q.scores["earth_movers"] for q in self.questions])),
            "brier": float(np.mean([q.scores["brier"] for q in self.questions])),
            "log_score": float(np.mean([q.scores["log_score"] for q in self.questions])),
            "ece": metrics.expected_calibration_error(pooled_prediction, pooled_truth),
            "cost_usd": round(self.cost_usd, 6),
            "llm_calls": float(self.llm_calls),
        }
        scored = [q for q in self.questions if "within_truth_interval" in q.scores]
        if scored:
            out["within_truth_interval"] = float(
                np.mean([q.scores["within_truth_interval"] for q in scored])
            )
        covered = [q for q in self.questions if "interval_coverage" in q.scores]
        if covered:
            out["interval_coverage"] = float(
                np.mean([q.scores["interval_coverage"] for q in covered])
            )
        out.update(self.experiment_summary())
        return out

    def experiment_summary(self) -> dict[str, float]:
        if not self.experiments:
            return {}
        scoreable = [e for e in self.experiments if e.is_scoreable]
        out = {
            "experiments": float(len(self.experiments)),
            "gap_mae": float(np.mean([e.error for e in self.experiments])),
        }
        if scoreable:
            out["gap_sign_accuracy"] = float(
                np.mean([e.sign_matches for e in scoreable])
            )
            out["scoreable_experiments"] = float(len(scoreable))
        if len(self.experiments) > 2:
            true = np.array([e.true_gap for e in self.experiments])
            predicted = np.array([e.predicted_gap for e in self.experiments])
            # A flat predictor has zero variance and no correlation to report, which
            # is itself the finding: it did not react to the wording at all.
            if predicted.std() > 1e-9 and true.std() > 1e-9:
                out["gap_correlation"] = float(np.corrcoef(true, predicted)[0, 1])
            else:
                out["gap_correlation"] = 0.0
        return out

    def with_skill_against(self, baseline: "BacktestResult") -> dict[str, float]:
        """Summary plus how much of the baseline's error this configuration removed."""
        mine, theirs = self.summary(), baseline.summary()
        out = dict(mine)
        for metric in ("mae", "total_variation", "brier", "log_score", "gap_mae"):
            if metric in mine and metric in theirs:
                out[f"skill_{metric}"] = metrics.skill_score(mine[metric], theirs[metric])
        return out


#: Builds the spec for one question group. Given the arms to run, returns a spec.
SpecFactory = Callable[[str, str, Sequence[Question]], SimulationSpec]


def question_groups(
    bank: QuestionBank, only: Iterable[str] | None = None
) -> list[tuple[str, list[Question], Experiment | None]]:
    """Split a bank into runnable groups.

    A wording experiment is one group with several arms, because its arms must be run
    against the same population for the gap between them to mean anything. Every other
    question is a group of one.
    """
    wanted = set(only) if only is not None else None
    groups: list[tuple[str, list[Question], Experiment | None]] = []
    claimed: set[str] = set()

    for experiment in bank.experiments:
        arms = [bank[a] for a in experiment.arms if a in bank.questions]
        if len(arms) < 2:
            continue
        claimed.update(a.id for a in arms)
        if wanted is None or experiment.id in wanted:
            groups.append((experiment.id, arms, experiment))

    for question in bank:
        if question.id in claimed:
            continue
        if wanted is None or question.id in wanted:
            groups.append((question.id, [question], None))
    return groups


class Backtest:
    """Runs a configuration over a question bank and scores every answer."""

    def __init__(
        self,
        bank: QuestionBank,
        spec_factory: SpecFactory,
        root: str = ".",
        prior_bank: QuestionBank | None = None,
        targets=None,
    ) -> None:
        """``prior_bank`` defaults to leave-one-group-out, which is what you want."""
        self.bank = bank
        self.spec_factory = spec_factory
        self.root = root
        self.prior_bank = prior_bank
        self.targets = targets

    def run(self, engine: str, only: Iterable[str] | None = None) -> BacktestResult:
        result = BacktestResult(engine=engine)
        for group_id, arms, experiment in question_groups(self.bank, only):
            spec = self.spec_factory(engine, group_id, arms)
            # The prior baseline is fitted on every question except the ones it is
            # about to be scored on. Fitting it on the whole bank would hand it the
            # answers and quietly make the bar it sets unbeatable.
            prior_bank = self.prior_bank or self.bank.split([a.id for a in arms])[0]
            simulation = Simulation(
                spec, targets=self.targets, prior_bank=prior_bank, root=self.root
            )
            run = simulation.run()

            result.cost_usd += run.record.cost_usd
            result.llm_calls += run.record.llm_calls
            result.wall_seconds += run.record.wall_seconds
            for note in run.record.notes:
                if note not in result.notes:
                    result.notes.append(note)

            for question in arms:
                prediction = run.predictions[question.id]
                result.questions.append(
                    QuestionScore(
                        question_id=question.id,
                        text=question.text,
                        options=question.options,
                        prediction=prediction.distribution,
                        truth=question.topline,
                        interval=prediction.interval(spec.estimator.level)
                        if prediction.has_uncertainty
                        else None,
                        experiment=experiment.id if experiment else None,
                        scores=metrics.score_all(
                            prediction.distribution,
                            question.topline,
                            standard_error=question.standard_error,
                            interval=prediction.interval(spec.estimator.level)
                            if prediction.has_uncertainty
                            else None,
                        ),
                    )
                )

            if experiment is not None:
                option = experiment.contrast_option
                reference, other = experiment.arms[0], experiment.arms[1]
                result.experiments.append(
                    ExperimentScore(
                        experiment_id=experiment.id,
                        contrast_option=option,
                        arms=(reference, other),
                        true_gap=self.bank[other].share(option) - self.bank[reference].share(option),
                        predicted_gap=run.predictions[other].share(option)
                        - run.predictions[reference].share(option),
                    )
                )
        return result
