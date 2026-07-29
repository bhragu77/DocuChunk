"""
Model pricing + cost computation (Phase 9B).

Prices are DATA, kept here (a dedicated pricing config module) and overridable
from the environment — never hard-coded inside the tracer. Cost is computed from
the provider-returned token usage; USD is the stored unit, INR is a display
conversion using a configurable rate.

Env overrides (all optional):
  TRACE_MODEL_PRICES  JSON: {"model-name": {"input": <usd_per_1M>, "output": <usd_per_1M>}}
                      merged over the defaults below (per-model override / addition).
  TRACE_USD_TO_INR    float, default 83.0 — display-only conversion rate.

A model with no known price yields cost None (recorded as "unknown", never a
fabricated number).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger("app.observability.pricing")

# USD price per 1,000,000 tokens, {model: {"input": x, "output": y}}. These are
# starting points — override per deployment via TRACE_MODEL_PRICES. Keep the map
# small and current; unknown models simply report no cost.
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    # Google Gemini (public list prices, USD / 1M tokens).
    "gemini-3.1-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    # Rolling "-latest" aliases. These resolve to a concrete model server-side, but
    # they arrive here as their own name — and an unpriced name silently produced
    # cost=None, which is how a routine model switch disabled cost attribution
    # across the whole system without a single error.
    "gemini-flash-lite-latest": {"input": 0.10, "output": 0.40},
    "gemini-flash-latest": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    # OpenAI-compatible examples.
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # The offline stub is free — lets cost plumbing be demonstrated end-to-end
    # without a paid key (0 USD is a real, correct cost for the stub).
    "stub-1": {"input": 0.0, "output": 0.0},
    "stub-verify": {"input": 0.0, "output": 0.0},
}

DEFAULT_USD_TO_INR = 83.0

TRACE_MODEL_PRICES_ENV = "TRACE_MODEL_PRICES"
TRACE_USD_TO_INR_ENV = "TRACE_USD_TO_INR"


def get_prices() -> dict[str, dict[str, float]]:
    """Return the effective price table: defaults with env overrides merged on top."""
    prices = {k: dict(v) for k, v in DEFAULT_PRICES.items()}
    raw = os.getenv(TRACE_MODEL_PRICES_ENV, "").strip()
    if raw:
        try:
            override = json.loads(raw)
            for model, p in override.items():
                entry = prices.get(model, {})
                entry.update({k: float(v) for k, v in p.items()})
                prices[model] = entry
        except Exception:
            logger.debug("tracing: could not parse %s", TRACE_MODEL_PRICES_ENV, exc_info=True)
    return prices


def usd_to_inr_rate() -> float:
    try:
        return float(os.getenv(TRACE_USD_TO_INR_ENV, "") or DEFAULT_USD_TO_INR)
    except ValueError:
        return DEFAULT_USD_TO_INR


_WARNED_UNPRICED: set[str] = set()


def resolve_price(model: str) -> Optional[dict[str, float]]:
    """Price lookup with prefix fallback, warning ONCE per unpriced model.

    Providers ship new and aliased model names constantly. An exact-match-only table
    returns None for anything unrecognised, and because None is a legitimate value
    ("not priced") it propagates all the way to the report as a blank cell — which is
    indistinguishable from "this call was free". That is precisely how cost
    attribution was lost silently when GEN_MODEL moved to a `-latest` alias.

    The prefix fallback lets `gemini-3.1-flash-lite-preview-09` inherit
    `gemini-3.1-flash-lite` pricing (approximate, and better than nothing), and the
    one-time warning makes the gap visible instead of blank.
    """
    prices = get_prices()
    exact = prices.get(model)
    if exact is not None:
        return exact
    # Longest matching prefix wins, so a more specific entry is preferred.
    candidates = [k for k in prices if model.startswith(k)]
    if candidates:
        best = max(candidates, key=len)
        return prices[best]
    if model not in _WARNED_UNPRICED:
        _WARNED_UNPRICED.add(model)
        logger.warning(
            "no price entry for model %r — cost will be reported as unknown. "
            "Add it to DEFAULT_PRICES or TRACE_MODEL_PRICES.", model,
        )
    return None


def cost_usd(model: Optional[str], input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    """USD cost for a call, or None if the model is unpriced or tokens are missing.

    cost = input_tokens/1e6 * price_in + output_tokens/1e6 * price_out
    """
    if not model or input_tokens is None or output_tokens is None:
        return None
    price = resolve_price(model)
    if price is None:
        return None
    total = (input_tokens / 1_000_000.0) * price.get("input", 0.0) + (
        output_tokens / 1_000_000.0
    ) * price.get("output", 0.0)
    # Round to a sane precision (micro-dollars) — usage-based costs are tiny.
    return round(total, 8)


def to_inr(usd: Optional[float]) -> Optional[float]:
    if usd is None:
        return None
    return round(usd * usd_to_inr_rate(), 6)
