"""Typed loaders for the vendored ground-truth artifacts.

Two files, two jobs. ``acs_marginals.json`` is what a synthesized population is raked
to. ``gss_questions.json`` is what a prediction is scored against. Neither is ever
read as a raw dict outside this module, so a shape change in the vendored data is a
load-time failure rather than a silent accuracy regression.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from quorum.data.schema import LEVELS

DEFAULT_MARGINALS = "data/vendor/acs_marginals.json"
DEFAULT_QUESTIONS = "data/vendor/gss_questions.json"


@dataclass(frozen=True, slots=True)
class MarginalTargets:
    """Population shares each attribute must be raked to."""

    source: dict[str, Any]
    universe: str
    population_total: float
    records: int
    marginals: dict[str, dict[str, float]]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MARGINALS) -> "MarginalTargets":
        payload = json.loads(Path(path).read_text())
        targets = cls(
            source=payload["source"],
            universe=payload["universe"],
            population_total=float(payload["population_total"]),
            records=int(payload["records"]),
            marginals={k: dict(v) for k, v in payload["marginals"].items()},
        )
        targets.validate()
        return targets

    def validate(self) -> None:
        for attribute, shares in self.marginals.items():
            if attribute not in LEVELS:
                raise ValueError(f"marginals contain unknown attribute {attribute!r}")
            expected = set(LEVELS[attribute])
            if set(shares) != expected:
                raise ValueError(
                    f"marginals for {attribute} cover {sorted(shares)}, "
                    f"expected {sorted(expected)}"
                )
            total = sum(shares.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"marginals for {attribute} sum to {total:.8f}, not 1")

    @property
    def attributes(self) -> tuple[str, ...]:
        return tuple(a for a in LEVELS if a in self.marginals)

    def vector(self, attribute: str) -> np.ndarray:
        """Target shares for ``attribute`` in canonical level order."""
        shares = self.marginals[attribute]
        return np.array([shares[level] for level in LEVELS[attribute]], dtype=float)

    def subset(self, attributes: Sequence[str]) -> "MarginalTargets":
        missing = [a for a in attributes if a not in self.marginals]
        if missing:
            raise KeyError(f"no target marginals for {missing}")
        return MarginalTargets(
            source=self.source,
            universe=self.universe,
            population_total=self.population_total,
            records=self.records,
            marginals={a: dict(self.marginals[a]) for a in attributes},
        )


@dataclass(frozen=True, slots=True)
class Question:
    """One survey item with its published answer.

    ``topline`` is the weighted share of each option among substantive answers.
    ``standard_error`` is computed at the Kish effective sample size, so the harness
    can ask the fair question: is the prediction inside the ground truth's own
    sampling interval?
    """

    id: str
    text: str
    options: tuple[str, ...]
    topline: np.ndarray
    standard_error: np.ndarray
    n: int
    effective_n: float
    experiment: str | None = None
    arm_label: str | None = None
    segments: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    def segment(self, dimension: str, level: str) -> np.ndarray | None:
        values = self.segments.get(dimension, {}).get(level)
        return None if values is None else np.asarray(values, dtype=float)

    def share(self, option: str) -> float:
        return float(self.topline[self.options.index(option)])


@dataclass(frozen=True, slots=True)
class Experiment:
    """A randomized wording split: the same population, two or more question forms.

    The point of keeping these as first class objects is that a wording split has an
    answer key an ordinary topline does not. The population is identical by
    construction, so any difference between arms is caused by the wording, and a
    simulator that cannot reproduce it is not modelling the question at all.
    """

    id: str
    label: str
    arms: tuple[str, ...]
    contrast_option: str

    def gap(self, bank: "QuestionBank", reference: str | None = None) -> dict[str, float]:
        """Observed difference in ``contrast_option`` share against the first arm."""
        base_id = reference or self.arms[0]
        base = bank[base_id].share(self.contrast_option)
        return {arm: bank[arm].share(self.contrast_option) - base for arm in self.arms}


@dataclass(frozen=True, slots=True)
class QuestionBank:
    """The scored question set for one survey year."""

    source: dict[str, Any]
    year: int
    weight_variable: str
    questions: dict[str, Question]
    experiments: tuple[Experiment, ...] = ()

    @classmethod
    def load(cls, path: str | Path = DEFAULT_QUESTIONS) -> "QuestionBank":
        payload = json.loads(Path(path).read_text())
        questions = {}
        for item in payload["questions"]:
            questions[item["id"]] = Question(
                id=item["id"],
                text=item["text"],
                options=tuple(item["options"]),
                topline=np.array(item["topline"], dtype=float),
                standard_error=np.array(item["standard_error"], dtype=float),
                n=int(item["n"]),
                effective_n=float(item["effective_n"]),
                experiment=item.get("experiment"),
                arm_label=item.get("arm_label"),
                segments=item.get("segments", {}),
            )
        experiments = tuple(
            Experiment(
                id=e["id"],
                label=e["label"],
                arms=tuple(e["arms"]),
                contrast_option=e["contrast_option"],
            )
            for e in payload.get("experiments", [])
        )
        bank = cls(
            source=payload["source"],
            year=int(payload["year"]),
            weight_variable=payload["weight_variable"],
            questions=questions,
            experiments=experiments,
        )
        bank.validate()
        return bank

    def validate(self) -> None:
        for question in self.questions.values():
            if len(question.options) != len(question.topline):
                raise ValueError(f"{question.id}: options and topline disagree in length")
            total = float(question.topline.sum())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"{question.id}: topline sums to {total:.8f}, not 1")
        for experiment in self.experiments:
            unknown = [a for a in experiment.arms if a not in self.questions]
            if unknown:
                raise ValueError(f"experiment {experiment.id} references unknown arms {unknown}")
            options = {self.questions[a].options for a in experiment.arms}
            if len(options) != 1:
                raise ValueError(f"experiment {experiment.id} has arms with different options")
            if experiment.contrast_option not in next(iter(options)):
                raise ValueError(
                    f"experiment {experiment.id} contrasts on "
                    f"{experiment.contrast_option!r}, which is not a response option"
                )

    def __getitem__(self, question_id: str) -> Question:
        return self.questions[question_id]

    def __iter__(self) -> Iterator[Question]:
        return iter(self.questions.values())

    def __len__(self) -> int:
        return len(self.questions)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self.questions)

    def split(self, holdout: Sequence[str]) -> tuple["QuestionBank", "QuestionBank"]:
        """Partition into a calibration set and a held-out set.

        Recalibration fitted on the questions it is then scored on would report
        accuracy the engine does not have, so the split is explicit and the harness
        refuses to score a recalibrated prediction on the calibration half.
        """
        holdout_set = set(holdout)
        unknown = holdout_set - set(self.questions)
        if unknown:
            raise KeyError(f"holdout references unknown questions {sorted(unknown)}")
        keep = {k: v for k, v in self.questions.items() if k not in holdout_set}
        held = {k: v for k, v in self.questions.items() if k in holdout_set}
        return (
            QuestionBank(self.source, self.year, self.weight_variable, keep, self.experiments),
            QuestionBank(self.source, self.year, self.weight_variable, held, self.experiments),
        )


def load_ground_truth(
    marginals: str | Path = DEFAULT_MARGINALS, questions: str | Path = DEFAULT_QUESTIONS
) -> tuple[MarginalTargets, QuestionBank]:
    """Load both sides of the ground truth together."""
    return MarginalTargets.load(marginals), QuestionBank.load(questions)
