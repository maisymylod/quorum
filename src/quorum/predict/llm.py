"""Asking a model how a particular person would answer.

One design decision runs through this module: the model is asked for a distribution
over the response options, not for a choice. A simulated respondent that returns only
its modal answer throws away exactly the information that makes an aggregate topline
calibrated, and a population of confident agents produces a confident population,
which is not what a real one looks like.

The prompt is split so that everything shared between agents comes first. See
:mod:`quorum.predict.provider` for why that is worth doing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from quorum.core.population import Population
from quorum.core.prediction import ResponseDistribution
from quorum.exec.cost import Budget, BudgetExceeded
from quorum.predict.cache import NullCache, ResponseCache
from quorum.predict.provider import Completion, embed_options
from quorum.world.context import Scenario

SYSTEM_PREAMBLE = """\
You are simulating how one specific person would answer a survey question.

You will be given a short profile of that person. Answer as that person would answer, \
not as you would, and not as an average person would. Their circumstances should show \
in the answer where the question touches on them, and should not where it does not.

Reply with a probability distribution over the response options: for each option, how \
likely is it that this person gives that answer? Someone who is certain should have \
nearly all their weight on one option. Someone genuinely torn should be spread out. \
The probabilities should sum to 1.\
"""


@dataclass(slots=True)
class ResponderStats:
    """What one pass over a population cost and how much of it worked."""

    agents: int = 0
    live_calls: int = 0
    cache_hits: int = 0
    parse_failures: list[str] = field(default_factory=list)

    @property
    def parse_failure_rate(self) -> float:
        return len(self.parse_failures) / self.agents if self.agents else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "agents": float(self.agents),
            "live_calls": float(self.live_calls),
            "cache_hits": float(self.cache_hits),
            "parse_failures": float(len(self.parse_failures)),
            "parse_failure_rate": round(self.parse_failure_rate, 4),
        }


class LLMResponder:
    """Turns a population into a matrix of response distributions, one row per agent."""

    def __init__(
        self,
        provider,
        cache: ResponseCache | NullCache | None = None,
        budget: Budget | None = None,
        max_tokens: int = 512,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.provider = provider
        self.cache = cache if cache is not None else NullCache()
        self.budget = budget if budget is not None else Budget()
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.stats = ResponderStats()

    # -- prompts ---------------------------------------------------------------

    def system_prompt(self, scenario: Scenario) -> str:
        """The half of the prompt every agent shares, and therefore the cached half."""
        parts = [SYSTEM_PREAMBLE, "", f"QUESTION: {scenario.text}"]
        if scenario.context:
            parts.append(f"CONTEXT: {scenario.context}")
        parts.append("RESPONSE OPTIONS: " + " | ".join(scenario.options))
        return embed_options("\n".join(parts), scenario.options)

    def agent_prompt(self, population: Population, index: int) -> str:
        """The half that varies: one person."""
        agent = population.agent(index)
        return f"Profile of the person answering: {agent.describe(population.attributes)}."

    # -- responding ------------------------------------------------------------

    def respond(
        self, population: Population, scenario: Scenario, seed: int = 0
    ) -> np.ndarray:
        """Return an ``(n_agents, n_options)`` matrix of response distributions.

        Rows whose answer could not be parsed come back as NaN rather than as a
        silently invented uniform. A caller that averages over them will get NaN and
        notice; a caller that means to drop them can, and the hybrid predictor does.
        """
        system = self.system_prompt(scenario)
        prompts = [self.agent_prompt(population, i) for i in range(len(population))]
        self.stats = ResponderStats(agents=len(prompts))

        completions: list[Completion | None] = [None] * len(prompts)
        pending: list[int] = []
        for index, prompt in enumerate(prompts):
            hit = self.cache.get(self.provider.model, system, prompt, self.max_tokens)
            if hit is not None:
                completions[index] = hit
                self.stats.cache_hits += 1
                self.budget.meter.record(self.provider.model, cached=True)
            else:
                pending.append(index)

        for start in range(0, len(pending), self.batch_size):
            chunk = pending[start : start + self.batch_size]
            self.budget.check(len(chunk), self._estimate_usd(system, prompts, chunk))
            answers = self.provider.complete(
                [prompts[i] for i in chunk], system, self.max_tokens
            )
            for index, completion in zip(chunk, answers):
                completions[index] = completion
                self.stats.live_calls += 1
                self.budget.meter.record(
                    completion.model or self.provider.model,
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                    cache_read_tokens=completion.cache_read_tokens,
                    cache_write_tokens=completion.cache_write_tokens,
                )
                self.cache.put(
                    self.provider.model, system, prompts[index], self.max_tokens, completion
                )

        rows = np.full((len(prompts), len(scenario.options)), np.nan)
        for index, completion in enumerate(completions):
            parsed = self.parse(completion.text if completion else "", scenario.options)
            if parsed is None:
                self.stats.parse_failures.append(prompts[index])
            else:
                rows[index] = parsed.probabilities
        return rows

    def _estimate_usd(self, system: str, prompts: list[str], chunk: list[int]) -> float:
        """Rough forward cost of a chunk, for the budget check that precedes it.

        Deliberately crude and deliberately an overestimate: it charges every prompt
        the full system prefix even though all but the first will hit the prompt
        cache. A budget guard that underestimates is not a guard.
        """
        from quorum.exec.cost import price_of

        input_price, output_price = price_of(self.provider.model)
        characters = sum(len(system) + len(prompts[i]) for i in chunk)
        input_tokens = characters / 4
        output_tokens = self.max_tokens * len(chunk) * 0.25
        return (input_tokens * input_price + output_tokens * output_price) / 1e6

    @staticmethod
    def parse(text: str, options: tuple[str, ...]) -> ResponseDistribution | None:
        """Read a model answer into a distribution, or return None if it cannot be.

        The response format is schema-constrained, so this should not fail. It is
        written to fail softly anyway, because a provider swap or a schema-free
        fallback would otherwise turn a bad answer into a wrong number.
        """
        text = text.strip()
        if not text:
            return None
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        numeric = {k: float(v) for k, v in payload.items() if isinstance(v, (int, float))}
        if not numeric:
            return None
        try:
            return ResponseDistribution.from_mapping(numeric, options)
        except ValueError:
            return None


__all__ = ["BudgetExceeded", "LLMResponder", "ResponderStats", "SYSTEM_PREAMBLE"]
