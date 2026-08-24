"""Token and dollar accounting, and a budget that stops a run before it overspends.

A simulation that calls a model once per agent has a cost that scales with the
population, which is the constraint the whole hybrid architecture exists to break.
That argument is only worth making if the cost is measured rather than asserted, so
every call is metered and every run reports what it spent.

The budget is checked before a call is made, not after. A guard that notices the
overspend afterwards is an incident report, not a budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Published list prices in US dollars per million tokens, as (input, output).
#: Cache reads bill at a fraction of the input rate and cache writes at a premium,
#: which is why the two are metered separately below.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "stub": (0.0, 0.0),
}

#: Cache reads bill at roughly a tenth of the input rate; writing to the cache costs
#: about a quarter more than an ordinary input token.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25

#: The Batch API trades latency for half price. Simulated respondents do not need to
#: answer in real time, which makes this close to free money at scale.
BATCH_DISCOUNT = 0.5


class BudgetExceeded(RuntimeError):
    """Raised before a call that would take a run past its budget."""


def price_of(model: str) -> tuple[float, float]:
    """Input and output price per million tokens, or zero for an unpriced model."""
    return MODEL_PRICES.get(model, (0.0, 0.0))


@dataclass(slots=True)
class CostMeter:
    """Running total of what a simulation has spent."""

    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    usd: float = 0.0
    by_model: dict[str, float] = field(default_factory=dict)

    def record(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        batch: bool = False,
        cached: bool = False,
    ) -> float:
        """Meter one completion and return what it cost.

        A completion served from the local response cache costs nothing and is counted
        separately, so the headline spend is what a fresh run would actually pay and
        the cache hit rate stays visible next to it.
        """
        if cached:
            self.cached_calls += 1
            return 0.0

        input_price, output_price = price_of(model)
        discount = BATCH_DISCOUNT if batch else 1.0
        cost = discount * (
            input_tokens * input_price
            + cache_read_tokens * input_price * CACHE_READ_MULTIPLIER
            + cache_write_tokens * input_price * CACHE_WRITE_MULTIPLIER
            + output_tokens * output_price
        ) / 1e6

        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_read_tokens += cache_read_tokens
        self.cache_write_tokens += cache_write_tokens
        self.usd += cost
        self.by_model[model] = self.by_model.get(model, 0.0) + cost
        return cost

    @property
    def total_calls(self) -> int:
        return self.calls + self.cached_calls

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_calls / self.total_calls if self.total_calls else 0.0

    @property
    def prompt_cache_hit_rate(self) -> float:
        """Share of billed input tokens served from the model's prompt cache.

        Every simulated respondent is asked the same question with a different
        persona. Putting the shared half of the prompt first makes it a cacheable
        prefix, and this is the number that says whether that worked.
        """
        billed = self.input_tokens + self.cache_read_tokens
        return self.cache_read_tokens / billed if billed else 0.0

    def per_thousand(self, agents: int) -> float:
        return self.usd / agents * 1000 if agents else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "calls": float(self.calls),
            "cached_calls": float(self.cached_calls),
            "input_tokens": float(self.input_tokens),
            "output_tokens": float(self.output_tokens),
            "cache_read_tokens": float(self.cache_read_tokens),
            "cache_write_tokens": float(self.cache_write_tokens),
            "usd": round(self.usd, 6),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "prompt_cache_hit_rate": round(self.prompt_cache_hit_rate, 4),
        }


@dataclass(slots=True)
class Budget:
    """A hard ceiling on a run, enforced before each call."""

    max_usd: float = 5.0
    max_calls: int = 5000
    meter: CostMeter = field(default_factory=CostMeter)

    def check(self, upcoming_calls: int, estimated_usd: float = 0.0) -> None:
        """Raise if the next batch of calls would breach the budget."""
        if self.meter.calls + upcoming_calls > self.max_calls:
            raise BudgetExceeded(
                f"{self.meter.calls} calls made, {upcoming_calls} more requested, "
                f"limit is {self.max_calls}"
            )
        if self.meter.usd + estimated_usd > self.max_usd:
            raise BudgetExceeded(
                f"${self.meter.usd:.4f} spent, ${estimated_usd:.4f} more estimated, "
                f"limit is ${self.max_usd:.2f}"
            )

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.max_usd - self.meter.usd)

    @property
    def remaining_calls(self) -> int:
        return max(0, self.max_calls - self.meter.calls)
