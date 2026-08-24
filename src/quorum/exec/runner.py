"""Assembling and running one simulation.

Everything the engine can do is reachable from a spec, and this is where a spec turns
into components. Each stage is chosen by name and wired to the next through the
contracts in :mod:`quorum.core.contracts`, which is what makes the ablation grid a
matter of editing configuration rather than writing code.

Two things happen here that are easy to get wrong elsewhere. Every arm of a wording
experiment runs against the *same* population, because that is what makes the arms
comparable, exactly as randomization makes the survey's own arms comparable. And the
estimator's interval is derived from the agents that were actually measured rather than
from the size of the synthetic population, which is a number the author picked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from quorum.core.population import Population
from quorum.core.prediction import Prediction
from quorum.core.run import RunRecord
from quorum.core.spec import SimulationSpec
from quorum.data.targets import MarginalTargets
from quorum.exec.cost import Budget
from quorum.infer.mrp import DirectEstimator, MRPEstimator
from quorum.predict.baseline import PriorPredictor, UniformPredictor
from quorum.predict.cache import NullCache, ResponseCache
from quorum.predict.hybrid import DirectLLMPredictor, HybridPredictor
from quorum.predict.llm import LLMResponder
from quorum.predict.provider import AnthropicProvider, StubProvider
from quorum.synthesis.sampler import IndependenceSynthesizer, MicrodataSynthesizer
from quorum.synthesis.validate import FidelityReport, marginal_fidelity
from quorum.world.context import Scenario
from quorum.world.dynamics import BoundedConfidenceInfluence, NoInfluence
from quorum.world.network import HomophilyNetwork

#: Synthesis is exact arithmetic once raked, so anything above this is a defect.
FIDELITY_TOLERANCE = 1e-6


def _why_synthesis_failed(synthesizer, size: int) -> str:
    """Explain a fidelity failure in terms of its usual cause.

    Raking is exact arithmetic, so it misses a target for one reason far more often
    than any other: the drawn sample contains nobody in some level the targets
    require, and no reweighting can conjure one. That happens when the population is
    too small for the rarest level of the rarest attribute, and the message should say
    so rather than leave a reader staring at a table of small numbers.
    """
    result = getattr(synthesizer, "last_raking", None)
    if result is None:
        return ""
    if result.empty_levels:
        levels = ", ".join(
            f"{attribute} (level index {'/'.join(indices)})"
            for attribute, indices in result.empty_levels.items()
        )
        return (
            f"cause: no agent was drawn into {levels}, so raking cannot reach the "
            f"target share for it. A population of {size:,} is too small to cover "
            "every level of every attribute; raise population.size or drop the "
            "attribute."
        )
    return f"raking {result.summary()}"


@dataclass(slots=True)
class SimulationResult:
    """Everything one run produced."""

    spec: SimulationSpec
    record: RunRecord
    population: Population
    predictions: dict[str, Prediction]
    fidelity: FidelityReport
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def topline(self, arm: str | None = None) -> Prediction:
        if arm is None:
            arm = next(iter(self.predictions))
        return self.predictions[arm]

    def gap(self, option: str, reference: str, other: str) -> float:
        """Difference between two arms in one option's share."""
        return self.predictions[other].share(option) - self.predictions[reference].share(option)


class Simulation:
    """A spec, assembled into a runnable pipeline."""

    def __init__(
        self,
        spec: SimulationSpec,
        targets: MarginalTargets | None = None,
        prior_bank=None,
        root: str | Path = ".",
    ) -> None:
        self.spec = spec
        self.root = Path(root)
        self.targets = targets or MarginalTargets.load(self.root / spec.population.targets)
        self.prior_bank = prior_bank
        self.budget = Budget(max_usd=spec.budget.max_usd, max_calls=spec.budget.max_calls)

    # -- assembly --------------------------------------------------------------

    def build_synthesizer(self):
        population = self.spec.population
        attributes = tuple(population.attributes)
        traits = tuple(population.traits)
        if population.source == "marginals" and population.microdata:
            return MicrodataSynthesizer.from_csv(
                self.root / population.microdata, self.targets, attributes, traits
            )
        return IndependenceSynthesizer(self.targets, attributes, traits)

    def build_provider(self, scenario: Scenario):
        provider = self.spec.predictor.provider
        if provider.name == "stub":
            return StubProvider(model=provider.model, options=scenario.options)
        return AnthropicProvider(
            model=provider.model,
            max_concurrency=provider.max_concurrency,
            max_tokens=provider.max_tokens,
            effort=provider.effort,
            options=scenario.options,
        )

    def build_predictor(self, scenario: Scenario):
        settings = self.spec.predictor
        if settings.kind == "uniform":
            return UniformPredictor()
        if settings.kind == "prior":
            if self.prior_bank is None:
                raise ValueError(
                    "the prior baseline needs a calibration question bank; pass "
                    "prior_bank, and make sure it excludes the questions being scored"
                )
            return PriorPredictor.fit(self.prior_bank)

        # The stub is already a deterministic function of its prompt, so caching it
        # would persist a hash and nothing else. Only real answers are worth keeping.
        use_cache = settings.provider.name != "stub" and settings.provider.cache_dir
        responder = LLMResponder(
            self.build_provider(scenario),
            cache=ResponseCache.open(self.root / settings.provider.cache_dir)
            if use_cache
            else NullCache(),
            budget=self.budget,
            max_tokens=settings.provider.max_tokens,
            batch_size=settings.provider.batch_size,
        )
        if settings.kind == "llm":
            return DirectLLMPredictor(responder)
        if settings.kind == "classical":
            # No model at all: every agent gets the propagator's fallback, which is the
            # sample mean of nothing. Only reachable through the ablation grid, where
            # it is the control that shows what the model contributes.
            return UniformPredictor()
        return HybridPredictor(
            responder,
            stratify_by=tuple(settings.stratify_by),
            archetypes=settings.archetypes,
            propagator=settings.propagator,
            traits=tuple(self.spec.population.traits),
            trait_noise=settings.trait_noise,
        )

    def build_world(self):
        world = self.spec.world
        if not world.dynamics.enabled or not world.network.enabled:
            return NoInfluence()
        return BoundedConfidenceInfluence(
            HomophilyNetwork(
                mean_degree=world.network.mean_degree,
                homophily=world.network.homophily,
                dimensions=tuple(world.network.dimensions),
            ),
            rounds=world.dynamics.rounds,
            confidence=world.dynamics.confidence,
            susceptibility=world.dynamics.susceptibility,
        )

    def build_estimator(self):
        settings = self.spec.estimator
        if not settings.poststratify:
            return DirectEstimator()
        return MRPEstimator(
            dimensions=tuple(settings.dimensions),
            draws=settings.draws,
            level=settings.level,
            pool=settings.mrp,
        )

    # -- running ---------------------------------------------------------------

    def run(self) -> SimulationResult:
        started = time.perf_counter()
        spec = self.spec
        record = RunRecord.for_spec(spec)

        synthesizer = self.build_synthesizer()
        population = synthesizer.synthesize(spec.population.size, spec.seed)
        fidelity = marginal_fidelity(population, self.targets)
        if fidelity.max_deviation > FIDELITY_TOLERANCE:
            raise RuntimeError(
                f"synthesized population misses its target marginals by "
                f"{fidelity.max_deviation:.2e}\n{fidelity.table()}\n"
                f"{_why_synthesis_failed(synthesizer, spec.population.size)}"
            )
        record.population_fingerprint = population.fingerprint()
        record.population_size = len(population)

        world = self.build_world()
        estimator = self.build_estimator()
        predictions: dict[str, Prediction] = {}
        diagnostics: dict[str, Any] = {}

        for scenario in Scenario.arms_from_spec(spec):
            predictor = self.build_predictor(scenario)
            responses = predictor.predict(population, scenario, spec.seed)
            responses = world.influence(population, responses, spec.seed)

            prediction = self._estimate(
                estimator, population, predictor, responses, scenario, spec.seed
            )
            predictions[scenario.arm] = prediction
            record.add_prediction(scenario.arm, prediction, spec.estimator.level)
            if hasattr(predictor, "diagnostics"):
                diagnostics[scenario.arm] = predictor.diagnostics.as_dict()

        meter = self.budget.meter
        record.llm_calls = meter.calls
        record.cache_hits = meter.cached_calls
        record.input_tokens = meter.input_tokens
        record.output_tokens = meter.output_tokens
        record.cost_usd = round(meter.usd, 6)
        record.wall_seconds = round(time.perf_counter() - started, 3)
        if spec.predictor.provider.name == "stub":
            record.note(
                "answers came from the offline stub, which is a hash shaped into a "
                "distribution and knows nothing about the world; no accuracy figure "
                "from this run means anything"
            )
        return SimulationResult(
            spec=spec,
            record=record,
            population=population,
            predictions=predictions,
            fidelity=fidelity,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _estimate(
        estimator,
        population: Population,
        predictor,
        responses: np.ndarray,
        scenario: Scenario,
        seed: int,
    ) -> Prediction:
        sample = getattr(predictor, "last_sample", None)
        sample_responses = getattr(predictor, "last_responses", None)
        if isinstance(estimator, MRPEstimator) and sample is not None:
            return estimator.estimate(population, sample, sample_responses, scenario, seed)
        # Nothing was measured that the interval could be built from, so none is
        # claimed. A baseline with a confidence band would be a lie about a constant.
        return DirectEstimator().estimate(population, responses, scenario, seed)
