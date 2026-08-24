"""Response prediction: from an agent to a distribution over a question's options."""

from quorum.predict.baseline import PriorPredictor, UniformPredictor
from quorum.predict.cache import NullCache, ResponseCache, cache_key
from quorum.predict.features import DesignSpace
from quorum.predict.hybrid import DirectLLMPredictor, HybridPredictor, build_propagator
from quorum.predict.llm import LLMResponder
from quorum.predict.propagate import (
    CellMeanPropagator,
    MeanPropagator,
    MultinomialLogitPropagator,
    apply_trait_noise,
)
from quorum.predict.provider import AnthropicProvider, Completion, StubProvider

__all__ = [
    "AnthropicProvider",
    "CellMeanPropagator",
    "Completion",
    "DesignSpace",
    "DirectLLMPredictor",
    "HybridPredictor",
    "LLMResponder",
    "MeanPropagator",
    "MultinomialLogitPropagator",
    "NullCache",
    "PriorPredictor",
    "ResponseCache",
    "StubProvider",
    "UniformPredictor",
    "apply_trait_noise",
    "build_propagator",
    "cache_key",
]
