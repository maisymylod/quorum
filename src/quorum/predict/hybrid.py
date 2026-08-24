"""The hybrid predictor: a model on a sample, classical machinery on the rest.

The constraint this exists to break is simple arithmetic. Asking a model about every
agent makes a simulation cost O(population), so a hundred thousand agents means a
hundred thousand calls and the population size becomes a budget decision rather than a
modelling one.

The hybrid spends the model budget on a stratified sample of archetypes chosen so that
every cell of the population is represented, fits a propagator to what the model said
about them, and scores everybody else with it. The model cost becomes O(archetypes),
independent of population size, and the population size goes back to being a modelling
decision. Whether the accuracy survives that trade is not assumed here: the same
interface runs in the ablation grid against the all-model and no-model extremes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quorum.core.population import TRAIT_PREFIX, Population
from quorum.predict.features import DesignSpace
from quorum.predict.propagate import (
    CellMeanPropagator,
    MeanPropagator,
    MultinomialLogitPropagator,
    apply_trait_noise,
)
from quorum.world.context import Scenario


@dataclass(slots=True)
class HybridDiagnostics:
    """What the hybrid actually did, so the shortcut can be audited."""

    archetypes: int = 0
    usable_archetypes: int = 0
    cells: int = 0
    propagator: str = ""
    responder: dict[str, float] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return self.usable_archetypes / self.archetypes if self.archetypes else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "archetypes": self.archetypes,
            "usable_archetypes": self.usable_archetypes,
            "archetype_coverage": round(self.coverage, 4),
            "cells": self.cells,
            "propagator": self.propagator,
            **{f"responder_{k}": v for k, v in self.responder.items()},
        }


def build_propagator(
    kind: str, stratify_by: tuple[str, ...], traits: tuple[str, ...] = ()
) -> Any:
    if kind == "cell_mean":
        return CellMeanPropagator(stratify_by)
    if kind == "multinomial_logit":
        return MultinomialLogitPropagator(DesignSpace(stratify_by, traits))
    if kind == "mean":
        return MeanPropagator()
    raise ValueError(f"unknown propagator {kind!r}")


class HybridPredictor:
    """Measure a stratified sample with a model, propagate to the population."""

    name = "hybrid"

    def __init__(
        self,
        responder,
        stratify_by: tuple[str, ...],
        archetypes: int = 300,
        propagator: str = "multinomial_logit",
        traits: tuple[str, ...] = (),
        trait_noise: float = 0.0,
    ) -> None:
        if archetypes < 1:
            raise ValueError("archetypes must be at least 1")
        if not stratify_by:
            raise ValueError("the hybrid predictor needs at least one stratification dimension")
        self.responder = responder
        self.stratify_by = tuple(stratify_by)
        self.archetypes = archetypes
        self.propagator_kind = propagator
        self.traits = tuple(traits)
        self.trait_noise = trait_noise
        self.diagnostics = HybridDiagnostics()

    def predict(self, population: Population, scenario: Scenario, seed: int) -> np.ndarray:
        sample, _ = population.stratified_sample(
            min(self.archetypes, len(population)), list(self.stratify_by), seed
        )
        responses = self.responder.respond(sample, scenario, seed)

        # Agents whose answer could not be parsed are dropped rather than imputed.
        # A fabricated response would be fitted as if it were evidence.
        usable = ~np.isnan(responses).any(axis=1)
        if not usable.any():
            raise RuntimeError(
                "no archetype produced a usable answer; the provider returned nothing "
                "that could be read as a distribution"
            )
        sample = sample.subset(usable)
        responses = responses[usable]

        propagator = build_propagator(self.propagator_kind, self.stratify_by, self.traits)
        propagator.fit(sample, responses)
        predicted = propagator.predict(population)

        if self.trait_noise > 0 and self.traits:
            traits = population.frame[[TRAIT_PREFIX + t for t in self.traits]].to_numpy()
            predicted = apply_trait_noise(predicted, traits, self.trait_noise, seed + 2)

        self.diagnostics = HybridDiagnostics(
            archetypes=int(len(usable)),
            usable_archetypes=int(usable.sum()),
            cells=len(population.cells(list(self.stratify_by))),
            propagator=self.propagator_kind,
            responder=getattr(self.responder, "stats", None).as_dict()
            if hasattr(self.responder, "stats")
            else {},
        )
        return predicted


class DirectLLMPredictor:
    """Ask the model about every single agent.

    The thing the hybrid is a shortcut for, kept as a first-class predictor so the
    shortcut can be measured against it rather than argued about. Only affordable on
    small populations, which is the entire point.
    """

    name = "llm"

    def __init__(self, responder) -> None:
        self.responder = responder

    def predict(self, population: Population, scenario: Scenario, seed: int) -> np.ndarray:
        responses = self.responder.respond(population, scenario, seed)
        usable = ~np.isnan(responses).any(axis=1)
        if not usable.any():
            raise RuntimeError("no agent produced a usable answer")
        if not usable.all():
            # Unlike the hybrid, this predictor has no sample to fall back on, so an
            # unparsed agent is filled with the population's own mean rather than
            # dropped, which would silently reweight the population.
            fallback = responses[usable].mean(axis=0)
            responses[~usable] = fallback
        return responses
