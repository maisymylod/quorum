from __future__ import annotations

import pytest

from quorum.exec.cost import BATCH_DISCOUNT, Budget, BudgetExceeded, CostMeter, price_of


def test_known_and_unknown_models_are_priced():
    assert price_of("claude-opus-5") == (5.00, 25.00)
    assert price_of("claude-haiku-4-5") == (1.00, 5.00)
    assert price_of("not-a-model") == (0.0, 0.0)


def test_recording_a_call_charges_input_and_output():
    meter = CostMeter()
    cost = meter.record("claude-opus-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(30.0)
    assert meter.usd == pytest.approx(30.0)
    assert meter.calls == 1
    assert meter.by_model["claude-opus-5"] == pytest.approx(30.0)


def test_batching_halves_the_price():
    live = CostMeter().record("claude-opus-5", input_tokens=1_000_000)
    batched = CostMeter().record("claude-opus-5", input_tokens=1_000_000, batch=True)
    assert batched == pytest.approx(live * BATCH_DISCOUNT)


def test_prompt_cache_reads_are_much_cheaper_than_fresh_input():
    fresh = CostMeter().record("claude-opus-5", input_tokens=1_000_000)
    cached = CostMeter().record("claude-opus-5", cache_read_tokens=1_000_000)
    written = CostMeter().record("claude-opus-5", cache_write_tokens=1_000_000)
    assert cached == pytest.approx(fresh * 0.1)
    assert written > fresh


def test_a_locally_cached_call_costs_nothing_and_is_counted_separately():
    meter = CostMeter()
    meter.record("claude-opus-5", input_tokens=1000, output_tokens=100)
    assert meter.record("claude-opus-5", cached=True) == 0.0
    assert meter.calls == 1
    assert meter.cached_calls == 1
    assert meter.total_calls == 2
    assert meter.cache_hit_rate == pytest.approx(0.5)


def test_prompt_cache_hit_rate_measures_billed_input():
    meter = CostMeter()
    meter.record("claude-opus-5", input_tokens=200, cache_read_tokens=800)
    assert meter.prompt_cache_hit_rate == pytest.approx(0.8)


def test_rates_are_zero_before_anything_happens():
    meter = CostMeter()
    assert meter.cache_hit_rate == 0.0
    assert meter.prompt_cache_hit_rate == 0.0
    assert meter.per_thousand(0) == 0.0


def test_cost_per_thousand_agents():
    meter = CostMeter()
    meter.usd = 2.0
    assert meter.per_thousand(20_000) == pytest.approx(0.1)


def test_meter_serializes_for_the_run_record():
    meter = CostMeter()
    meter.record("claude-opus-5", input_tokens=10, output_tokens=5)
    payload = meter.as_dict()
    assert payload["calls"] == 1.0
    assert "prompt_cache_hit_rate" in payload


def test_budget_stops_a_run_before_it_overspends():
    budget = Budget(max_usd=1.0, max_calls=100)
    budget.meter.record("claude-opus-5", input_tokens=10_000, output_tokens=2_000)
    budget.check(1, estimated_usd=0.01)
    with pytest.raises(BudgetExceeded, match="limit is \\$1.00"):
        budget.check(1, estimated_usd=5.0)


def test_budget_stops_a_run_on_call_count():
    budget = Budget(max_usd=100.0, max_calls=2)
    budget.meter.calls = 2
    with pytest.raises(BudgetExceeded, match="limit is 2"):
        budget.check(1)


def test_budget_reports_what_is_left():
    budget = Budget(max_usd=1.0, max_calls=10)
    budget.meter.usd = 0.4
    budget.meter.calls = 3
    assert budget.remaining_usd == pytest.approx(0.6)
    assert budget.remaining_calls == 7
    budget.meter.usd = 2.0
    assert budget.remaining_usd == 0.0
