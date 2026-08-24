"""Accuracy as a gate, not a printout.

A metric that is reported but not enforced drifts. These are the thresholds a run has
to clear, checked by ``quorum eval --check`` and by CI, so that a change which quietly
makes the engine worse fails a build instead of appearing as a slightly different
number in a table nobody diffed.

The gates are split by what produces the answers. Some hold for any run at all: the
population has to match its targets, the harness has to score every question, the
baselines have to order themselves the way arithmetic says they must. Others are
statements about prediction, and those are enforced only when a real model produced
the answers, because the offline stub cannot pass them and pretending otherwise would
be the exact self-deception this file exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Gate:
    """One enforced threshold."""

    name: str
    metric: str
    bound: float
    direction: Literal["at_most", "at_least"]
    engine: str
    rationale: str
    #: "any" holds for every run; "model" only when a real model answered.
    applies: Literal["any", "model"] = "any"

    def holds(self, value: float) -> bool:
        return value <= self.bound if self.direction == "at_most" else value >= self.bound


@dataclass(frozen=True, slots=True)
class GateResult:
    gate: Gate
    value: float | None
    passed: bool
    skipped: bool = False

    def describe(self) -> str:
        if self.skipped:
            return f"skip  {self.gate.name}: {self.gate.rationale}"
        state = "pass" if self.passed else "FAIL"
        comparison = "<=" if self.gate.direction == "at_most" else ">="
        return (
            f"{state}  {self.gate.name}: {self.gate.metric} on `{self.gate.engine}` "
            f"= {self.value:.4f}, needs {comparison} {self.gate.bound}"
        )


GATES: tuple[Gate, ...] = (
    Gate(
        name="every question scored",
        metric="questions",
        bound=30,
        direction="at_least",
        engine="hybrid",
        rationale="a backtest that silently skipped questions would look like an easy one",
    ),
    Gate(
        name="baselines are ordered",
        metric="mae",
        bound=0.16,
        direction="at_most",
        engine="prior",
        rationale="the prior baseline must beat guessing on this question bank, or the "
        "bar the engine is measured against is not the bar we think it is",
    ),
    # Everything below is a claim about prediction, so the stub cannot be asked to
    # meet it. These activate the moment a real model answers.
    Gate(
        name="beats the prior baseline",
        metric="skill_mae",
        bound=0.15,
        direction="at_least",
        engine="hybrid",
        rationale="an engine that cannot beat predicting the average answer to "
        "questions of this shape is not earning its cost",
        applies="model",
    ),
    Gate(
        name="calibrated",
        metric="ece",
        bound=0.06,
        direction="at_most",
        engine="hybrid",
        rationale="a predicted 30 percent should be right about 30 percent of the time",
        applies="model",
    ),
    Gate(
        name="intervals mean what they say",
        metric="interval_coverage",
        bound=0.70,
        direction="at_least",
        engine="hybrid",
        rationale="a 90 percent interval that covers half the time is not one",
        applies="model",
    ),
    Gate(
        name="reads the wording",
        metric="gap_sign_accuracy",
        bound=0.70,
        direction="at_least",
        engine="hybrid",
        rationale="predicting both arms of a wording split identically scores well on "
        "average error and means the question was never read",
        applies="model",
    ),
    Gate(
        name="wording gaps are the right size",
        metric="gap_mae",
        bound=0.12,
        direction="at_most",
        engine="hybrid",
        rationale="getting the direction of a framing effect but not its magnitude is "
        "half an answer",
        applies="model",
    ),
)


def check(summaries: dict[str, dict[str, float]], provider: str) -> list[GateResult]:
    """Evaluate every gate against a set of per-engine summaries."""
    results: list[GateResult] = []
    is_model = provider != "stub"
    for gate in GATES:
        summary = summaries.get(gate.engine)
        if gate.applies == "model" and not is_model:
            results.append(
                GateResult(
                    gate,
                    None,
                    passed=True,
                    skipped=True,
                )
            )
            continue
        if summary is None or gate.metric not in summary:
            results.append(GateResult(gate, None, passed=False))
            continue
        value = summary[gate.metric]
        results.append(GateResult(gate, value, passed=gate.holds(value)))
    return results


def failures(results: list[GateResult]) -> list[GateResult]:
    return [r for r in results if not r.passed and not r.skipped]
