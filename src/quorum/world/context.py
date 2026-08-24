"""The question, as one arm of it is actually asked.

A scenario is a spec's scenario block resolved down to a single arm. Keeping the arm
resolved rather than carried around as a branch is what makes a wording experiment two
ordinary runs over the same population instead of a special case threaded through the
predictor.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from quorum.core.spec import SimulationSpec


@dataclass(frozen=True, slots=True)
class Scenario:
    """One question, one wording, one set of response options."""

    question_id: str
    text: str
    options: tuple[str, ...]
    arm: str = "default"
    arm_label: str = ""
    context: str = ""

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError("a scenario needs at least two response options")
        if not self.text.strip():
            raise ValueError("a scenario needs question text")

    @classmethod
    def from_spec(cls, spec: SimulationSpec, arm: str | None = None) -> "Scenario":
        prompts = spec.scenario.arm_prompts()
        if arm is None:
            arm = next(iter(prompts))
        if arm not in prompts:
            raise KeyError(f"unknown arm {arm!r}; spec defines {sorted(prompts)}")
        labels = {a.id: a.label for a in spec.scenario.arms}
        return cls(
            question_id=spec.scenario.question_id,
            text=prompts[arm],
            options=tuple(spec.scenario.options),
            arm=arm,
            arm_label=labels.get(arm, ""),
            context=spec.scenario.context,
        )

    @classmethod
    def arms_from_spec(cls, spec: SimulationSpec) -> tuple["Scenario", ...]:
        return tuple(cls.from_spec(spec, arm) for arm in spec.scenario.arm_prompts())

    def fingerprint(self) -> str:
        """Content hash of everything a respondent would see.

        The wording is inside the hash on purpose: two arms of a split-ballot
        experiment differ only in text, and a cache that ignored the text would serve
        one arm's answers to the other and quietly erase the effect being measured.
        """
        payload = "\x1f".join([self.question_id, self.text, self.context, *self.options])
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
