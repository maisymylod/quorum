from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from quorum.predict.cache import NullCache, ResponseCache, cache_key
from quorum.predict.provider import (
    AnthropicProvider,
    Completion,
    StubProvider,
    _options_from_schema,
    embed_options,
    response_schema,
)

OPTIONS = ("too little", "about right", "too much")


# -- schema and option plumbing ----------------------------------------------------


def test_response_schema_demands_every_option_and_nothing_else():
    schema = response_schema(OPTIONS)
    assert set(schema["properties"]) == set(OPTIONS)
    assert schema["required"] == list(OPTIONS)
    assert schema["additionalProperties"] is False


def test_options_round_trip_through_the_system_prompt():
    system = embed_options("You are simulating a person.", OPTIONS)
    assert _options_from_schema(system) == OPTIONS


def test_a_system_prompt_without_options_is_an_error():
    with pytest.raises(ValueError, match="no option list"):
        _options_from_schema("just some text")


# -- stub provider -----------------------------------------------------------------


def test_stub_is_deterministic_and_prompt_sensitive():
    provider = StubProvider(options=OPTIONS)
    system = embed_options("system", OPTIONS)
    first = provider.complete(["persona A"], system)[0]
    again = provider.complete(["persona A"], system)[0]
    other = provider.complete(["persona B"], system)[0]
    assert first.text == again.text
    assert first.text != other.text


def test_stub_returns_a_valid_distribution_over_the_options():
    completion = StubProvider(options=OPTIONS).complete(["p"], embed_options("s", OPTIONS))[0]
    payload = json.loads(completion.text)
    assert set(payload) == set(OPTIONS)
    assert sum(payload.values()) == pytest.approx(1.0, abs=1e-3)
    assert completion.source == "stub"


def test_stub_reads_the_options_out_of_the_prompt_when_not_configured():
    completion = StubProvider().complete(["p"], embed_options("s", OPTIONS))[0]
    assert set(json.loads(completion.text)) == set(OPTIONS)


# -- anthropic provider ------------------------------------------------------------


class _FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        options = list(kwargs["output_config"]["format"]["schema"]["properties"])
        answer = {option: 1.0 / len(options) for option in options}
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(answer))],
            usage=SimpleNamespace(
                input_tokens=120,
                output_tokens=30,
                cache_read_input_tokens=900,
                cache_creation_input_tokens=0,
            ),
            model=kwargs["model"],
        )


class _FakeClient:
    def __init__(self) -> None:
        self.messages = _FakeMessages()


def test_provider_sends_the_shared_prefix_as_a_cacheable_system_block():
    client = _FakeClient()
    provider = AnthropicProvider(model="claude-opus-5", options=OPTIONS, client=client)
    provider.complete(["persona A", "persona B"], embed_options("system", OPTIONS))

    assert len(client.messages.calls) == 2
    call = client.messages.calls[0]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["messages"][0]["content"] == "persona A"
    assert call["output_config"]["format"]["type"] == "json_schema"
    # Sampling parameters are rejected by current models and must not be sent.
    assert "temperature" not in call
    assert "top_p" not in call


def test_provider_reports_token_usage_including_prompt_cache_reads():
    provider = AnthropicProvider(options=OPTIONS, client=_FakeClient())
    completion = provider.complete(["persona"], embed_options("system", OPTIONS))[0]
    assert completion.input_tokens == 120
    assert completion.cache_read_tokens == 900
    assert completion.source == "live"


def test_provider_short_circuits_on_an_empty_batch():
    provider = AnthropicProvider(options=OPTIONS, client=_FakeClient())
    assert provider.complete([], "system") == []


def test_provider_validates_concurrency():
    with pytest.raises(ValueError, match="max_concurrency"):
        AnthropicProvider(max_concurrency=0)


# -- response cache ----------------------------------------------------------------


def test_cache_key_depends_on_every_input():
    base = cache_key("m", "sys", "prompt", 512)
    assert base != cache_key("m2", "sys", "prompt", 512)
    assert base != cache_key("m", "sys2", "prompt", 512)
    assert base != cache_key("m", "sys", "prompt2", 512)
    assert base != cache_key("m", "sys", "prompt", 256)
    assert base == cache_key("m", "sys", "prompt", 512)


def test_cache_round_trips_through_disk(tmp_path):
    cache = ResponseCache.open(tmp_path)
    completion = Completion(text="{}", input_tokens=5, output_tokens=2, model="m")
    cache.put("m", "sys", "prompt", 512, completion)

    reopened = ResponseCache.open(tmp_path)
    found = reopened.get("m", "sys", "prompt", 512)
    assert found is not None
    assert found.text == "{}"
    assert found.input_tokens == 5
    assert found.source == "cache"
    assert len(reopened) == 1


def test_cache_reports_misses_and_hit_rate(tmp_path):
    cache = ResponseCache.open(tmp_path)
    assert cache.get("m", "sys", "prompt", 512) is None
    cache.put("m", "sys", "prompt", 512, Completion(text="{}", model="m"))
    assert cache.get("m", "sys", "prompt", 512) is not None
    assert cache.hits == 1 and cache.misses == 1
    assert cache.hit_rate == pytest.approx(0.5)


def test_cache_file_name_is_safe_for_any_model_string(tmp_path):
    cache = ResponseCache.open(tmp_path)
    cache.put("weird/model:name", "sys", "prompt", 512, Completion(text="{}"))
    written = list(tmp_path.iterdir())
    assert len(written) == 1
    assert "/" not in written[0].name


def test_cache_ignores_blank_lines_in_its_file(tmp_path):
    cache = ResponseCache.open(tmp_path)
    cache.put("m", "sys", "prompt", 512, Completion(text="{}"))
    path = next(tmp_path.iterdir())
    path.write_text(path.read_text() + "\n\n")
    assert ResponseCache.open(tmp_path).load("m") == 1


def test_loading_a_missing_cache_file_is_not_an_error(tmp_path):
    assert ResponseCache.open(tmp_path).load("never-used") == 0


def test_null_cache_remembers_nothing():
    cache = NullCache()
    assert cache.put("m", "s", "p", 1, Completion(text="{}")) == ""
    assert cache.get("m", "s", "p", 1) is None
    assert len(cache) == 0


def test_completion_knows_where_it_came_from():
    assert Completion(text="{}", source="cache").cached
    assert not Completion(text="{}", source="live").cached


def test_provider_builds_a_real_client_when_none_is_injected():
    provider = AnthropicProvider(api_key="test-key-not-used")
    client = provider._ensure_client()
    assert client is provider._ensure_client()  # constructed once, then reused
    assert hasattr(client, "messages")
