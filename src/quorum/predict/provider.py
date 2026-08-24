"""Model backends behind one narrow interface.

The simulation loop needs exactly one thing from a model provider: given a shared
system prompt and a list of per-agent prompts, return a completion for each. Holding
the surface to that makes the offline stub a genuine substitute rather than a partial
one, and it is what lets the whole pipeline, evaluation included, run in CI with no
network.

Two prompt-shaping decisions live here because they are the difference between a
simulation that is affordable at scale and one that is not.

**The shared half of the prompt goes first.** Every simulated respondent is asked the
same question; only the persona differs. Putting the task instructions, the question
and the response options in the system prompt and the persona in the user message
makes the shared half a cacheable prefix, so it is billed once at full rate and then
at a tenth of it. :attr:`CostMeter.prompt_cache_hit_rate` reports whether that worked.

**The answer is a schema, not a request.** The response format is constrained to a
JSON object with one number per response option, so a malformed answer is a provider
error rather than a parsing problem downstream.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

#: Current models reject sampling parameters, so a run is not reproducible by pinning
#: a temperature. Reproducibility comes from the committed response cache instead.
SAMPLING_IS_UNAVAILABLE = True


@dataclass(frozen=True, slots=True)
class Completion:
    """One model answer, with everything the cost meter needs."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model: str = ""
    source: str = "live"

    @property
    def cached(self) -> bool:
        return self.source == "cache"

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source: str = "cache") -> "Completion":
        return cls(
            text=payload["text"],
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            cache_read_tokens=int(payload.get("cache_read_tokens", 0)),
            cache_write_tokens=int(payload.get("cache_write_tokens", 0)),
            model=payload.get("model", ""),
            source=source,
        )


def response_schema(options: Sequence[str]) -> dict[str, Any]:
    """A JSON schema demanding one number per response option.

    Requiring every option, and forbidding any other key, is what turns "the model
    usually returns parseable JSON" into a guarantee. The numbers are still
    renormalized downstream, because a schema can require a number without requiring
    that a set of them sums to one.
    """
    return {
        "type": "object",
        "properties": {option: {"type": "number"} for option in options},
        "required": list(options),
        "additionalProperties": False,
    }


class StubProvider:
    """A deterministic offline stand-in for a model.

    It exists so the pipeline, the tests and the evaluation harness can run with no
    network and no key. It is not a model and its answers carry no knowledge of the
    world: they are a hash of the prompt shaped into a distribution. Any accuracy
    figure produced with this provider is a figure about plumbing, not about
    prediction, and the run record says so.
    """

    name = "stub"

    def __init__(self, model: str = "stub", options: Sequence[str] = ()) -> None:
        self.model = model
        self.options = tuple(options)

    def complete(
        self, prompts: Sequence[str], system: str, max_tokens: int = 512
    ) -> list[Completion]:
        return [self._one(prompt, system) for prompt in prompts]

    def _one(self, prompt: str, system: str) -> Completion:
        options = self.options or _options_from_schema(system)
        digest = hashlib.sha256((system + "\x1f" + prompt).encode()).digest()
        weights = [1.0 + digest[i % len(digest)] / 32.0 for i in range(len(options))]
        total = sum(weights)
        answer = {option: round(w / total, 4) for option, w in zip(options, weights)}
        text = json.dumps(answer)
        return Completion(
            text=text,
            input_tokens=len(system) // 4 + len(prompt) // 4,
            output_tokens=len(text) // 4,
            model=self.model,
            source="stub",
        )


class AnthropicProvider:
    """Calls Claude, concurrently, with the shared prompt prefix cached."""

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5",
        max_concurrency: int = 8,
        max_tokens: int = 512,
        effort: str = "low",
        options: Sequence[str] = (),
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self.model = model
        self.max_concurrency = max_concurrency
        self.max_tokens = max_tokens
        self.effort = effort
        self.options = tuple(options)
        self._api_key = api_key
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "the anthropic package is required for live calls; "
                    "install quorum[llm] or run with the stub provider"
                ) from exc
            key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._client = AsyncAnthropic(api_key=key) if key else AsyncAnthropic()
        return self._client

    def complete(
        self, prompts: Sequence[str], system: str, max_tokens: int | None = None
    ) -> list[Completion]:
        if not prompts:
            return []
        return asyncio.run(self._complete_async(list(prompts), system, max_tokens or self.max_tokens))

    async def _complete_async(
        self, prompts: list[str], system: str, max_tokens: int
    ) -> list[Completion]:
        client = self._ensure_client()
        semaphore = asyncio.Semaphore(self.max_concurrency)
        options = self.options or _options_from_schema(system)

        async def one(prompt: str) -> Completion:
            async with semaphore:
                response = await client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    # The stable half of the prompt, marked cacheable. The persona in
                    # the user message is the only part that varies per agent.
                    system=[
                        {
                            "type": "text",
                            "text": system,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": prompt}],
                    output_config={
                        "effort": self.effort,
                        "format": {"type": "json_schema", "schema": response_schema(options)},
                    },
                )
                return _from_response(response, self.model)

        return list(await asyncio.gather(*(one(p) for p in prompts)))


def _from_response(response: Any, model: str) -> Completion:
    text = next((b.text for b in response.content if b.type == "text"), "")
    usage = response.usage
    return Completion(
        text=text,
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        model=getattr(response, "model", model),
        source="live",
    )


_OPTIONS_MARKER = "OPTIONS:"


def embed_options(system: str, options: Sequence[str]) -> str:
    """Append a machine-readable option list to a system prompt.

    The provider has to know the option set to build the response schema, and the
    prompt already contains it in prose. Restating it in one parseable line keeps the
    two from drifting apart.
    """
    return f"{system}\n\n{_OPTIONS_MARKER} {json.dumps(list(options))}"


def _options_from_schema(system: str) -> tuple[str, ...]:
    for line in reversed(system.splitlines()):
        if line.startswith(_OPTIONS_MARKER):
            return tuple(json.loads(line[len(_OPTIONS_MARKER) :].strip()))
    raise ValueError(
        "the system prompt carries no option list; build it with embed_options()"
    )
