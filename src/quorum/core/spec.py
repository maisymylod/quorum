"""The simulation spec: one declarative file that fully determines a run.

This is the contract a new simulation is stood up from. Everything the engine needs
lives here (the audience, the world, the question, the predictor stack, the budget),
so standing up a new simulation is authoring a YAML file rather than writing code, and
two people running the same spec with the same seed get the same numbers.

The spec is also the reproducibility key: :meth:`SimulationSpec.fingerprint` hashes the
canonical form and that hash is stamped into every run record and cached artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PopulationSpec(_Base):
    """Audience generation settings."""

    source: Literal["marginals", "fixture"] = "marginals"
    targets: str = Field(
        default="data/vendor/acs_marginals.json",
        description="Path to the target marginals the population is raked to.",
    )
    microdata: str | None = Field(
        default=None,
        description="Optional microdata table to resample joint structure from. "
        "Without it, synthesis falls back to an independence prior plus raking.",
    )
    size: int = Field(default=20_000, ge=1)
    attributes: list[str] = Field(
        default_factory=lambda: ["age_band", "sex", "education", "race", "income_band"]
    )
    traits: list[str] = Field(default_factory=lambda: ["openness", "engagement"])

    @model_validator(mode="after")
    def _non_empty_attributes(self) -> "PopulationSpec":
        if not self.attributes:
            raise ValueError("population.attributes must list at least one dimension")
        if len(set(self.attributes)) != len(self.attributes):
            raise ValueError("population.attributes contains duplicates")
        return self


class NetworkSpec(_Base):
    """Social graph the world model runs peer influence over."""

    enabled: bool = False
    mean_degree: int = Field(default=8, ge=1)
    homophily: float = Field(default=0.7, ge=0.0, le=1.0)
    dimensions: list[str] = Field(default_factory=lambda: ["age_band", "education"])


class DynamicsSpec(_Base):
    """Bounded-confidence opinion updating over the network."""

    enabled: bool = False
    rounds: int = Field(default=3, ge=0)
    confidence: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Neighbours further than this in "
        "opinion space are ignored, which is what makes the model bounded-confidence "
        "rather than plain averaging toward consensus."
    )
    susceptibility: float = Field(default=0.2, ge=0.0, le=1.0)


class WorldSpec(_Base):
    network: NetworkSpec = Field(default_factory=NetworkSpec)
    dynamics: DynamicsSpec = Field(default_factory=DynamicsSpec)


class ArmSpec(_Base):
    """One experimental arm: the same question asked a different way."""

    id: str
    label: str
    prompt: str


class ScenarioSpec(_Base):
    """The question put to the population."""

    question_id: str
    prompt: str = ""
    options: list[str]
    arms: list[ArmSpec] = Field(default_factory=list)
    context: str = ""

    @model_validator(mode="after")
    def _check(self) -> "ScenarioSpec":
        if len(self.options) < 2:
            raise ValueError("scenario.options needs at least two response options")
        if len(set(self.options)) != len(self.options):
            raise ValueError("scenario.options contains duplicates")
        if not self.prompt and not self.arms:
            raise ValueError("scenario needs either a prompt or at least one arm")
        if len({a.id for a in self.arms}) != len(self.arms):
            raise ValueError("scenario.arms contains duplicate ids")
        return self

    def arm_prompts(self) -> dict[str, str]:
        """Prompt text per arm id, with a single-arm default for unframed questions."""
        if self.arms:
            return {a.id: a.prompt for a in self.arms}
        return {"default": self.prompt}


class ProviderSpec(_Base):
    """Which model backend the LLM half of the predictor talks to."""

    name: Literal["stub", "anthropic"] = "stub"
    model: str = "claude-opus-5"
    max_concurrency: int = Field(default=8, ge=1)
    max_tokens: int = Field(default=512, ge=16)
    effort: Literal["low", "medium", "high", "xhigh", "max"] = Field(
        default="low",
        description="Reasoning depth. Answering one survey question as one persona "
        "does not need much, and this is multiplied by the archetype count.",
    )
    cache_dir: str = "data/cache"
    batch_size: int = Field(default=32, ge=1)


class PredictorSpec(_Base):
    """How agents' responses are produced.

    ``hybrid`` is the production path: spend the LLM budget on a stratified set of
    archetypes, fit a classical propagator to those responses, and score the whole
    population with it. ``llm`` (every agent called) and ``classical`` (no model at
    all) exist so the ablation grid can show what the hybrid is actually buying.
    """

    kind: Literal["hybrid", "llm", "classical", "prior", "uniform"] = "hybrid"
    archetypes: int = Field(default=300, ge=1)
    stratify_by: list[str] = Field(default_factory=lambda: ["age_band", "education"])
    propagator: Literal["multinomial_logit", "cell_mean", "nearest"] = "multinomial_logit"
    provider: ProviderSpec = Field(default_factory=ProviderSpec)
    trait_noise: float = Field(
        default=0.05, ge=0.0, le=1.0,
        description="Within-cell dispersion applied to propagated responses.",
    )


class EstimatorSpec(_Base):
    """Aggregation and uncertainty."""

    poststratify: bool = True
    dimensions: list[str] = Field(default_factory=lambda: ["age_band", "education"])
    mrp: bool = True
    draws: int = Field(default=2000, ge=0)
    level: float = Field(default=0.90, gt=0.0, lt=1.0)


class BudgetSpec(_Base):
    """Hard stop on spend. Enforced before a call is made, not after."""

    max_usd: float = Field(default=5.0, ge=0.0)
    max_calls: int = Field(default=5000, ge=0)


class OutputSpec(_Base):
    dir: str = "artifacts"
    formats: list[Literal["json", "markdown", "html"]] = Field(
        default_factory=lambda: ["json", "markdown", "html"]
    )


class SimulationSpec(_Base):
    """A complete, self-contained description of one simulation."""

    name: str
    seed: int = 20260824
    population: PopulationSpec = Field(default_factory=PopulationSpec)
    world: WorldSpec = Field(default_factory=WorldSpec)
    scenario: ScenarioSpec
    predictor: PredictorSpec = Field(default_factory=PredictorSpec)
    estimator: EstimatorSpec = Field(default_factory=EstimatorSpec)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    output: OutputSpec = Field(default_factory=OutputSpec)

    @model_validator(mode="after")
    def _dimensions_exist(self) -> "SimulationSpec":
        known = set(self.population.attributes)
        for field_name, dims in (
            ("predictor.stratify_by", self.predictor.stratify_by),
            ("estimator.dimensions", self.estimator.dimensions),
            ("world.network.dimensions", self.world.network.dimensions),
        ):
            missing = [d for d in dims if d not in known]
            if missing:
                raise ValueError(
                    f"{field_name} refers to {missing} which are not in "
                    f"population.attributes {sorted(known)}"
                )
        return self

    # -- io --------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimulationSpec":
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"{path} does not contain a YAML mapping")
        return cls.model_validate(raw)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationSpec":
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    def canonical(self) -> str:
        """Canonical JSON. Key order fixed, so the hash is stable across machines."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:16]
