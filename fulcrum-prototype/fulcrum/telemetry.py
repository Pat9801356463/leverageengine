"""
Telemetry — runtime observability for the Fulcrum pipeline.

Round-2 requirement: "Runtime telemetry covering latency, model calls, token
usage and estimated cost."

Every pipeline stage is wrapped by `Telemetry.stage(...)`, which records wall
time and any model calls made inside it. LLM cost is accounted separately from
deterministic compute so the LLM-vs-non-LLM split is measurable, not asserted.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any


# Published per-million-token prices (USD). Kept in one place so the cost model
# is auditable rather than scattered through the codebase.
MODEL_PRICES = {
    "claude-haiku-4-5":  {"in": 1.00, "out": 5.00},
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "deterministic":     {"in": 0.00, "out": 0.00},
}

USD_TO_INR = 87.0


@dataclass
class ModelCall:
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cached: bool = False

    @property
    def cost_usd(self) -> float:
        if self.cached:
            return 0.0
        p = MODEL_PRICES.get(self.model, {"in": 0.0, "out": 0.0})
        return (self.input_tokens / 1e6) * p["in"] + (self.output_tokens / 1e6) * p["out"]


@dataclass
class StageRecord:
    name: str
    latency_ms: float
    kind: str                       # "deterministic" | "statistical" | "ml" | "causal" | "llm" | "optimisation"
    notes: str = ""
    rows_processed: int | None = None


@dataclass
class Telemetry:
    stages: list[StageRecord] = field(default_factory=list)
    calls: list[ModelCall] = field(default_factory=list)
    _cache: dict = field(default_factory=dict, repr=False)

    @contextmanager
    def stage(self, name: str, kind: str = "deterministic", notes: str = ""):
        t0 = time.perf_counter()
        holder: dict[str, Any] = {}
        try:
            yield holder
        finally:
            dt = (time.perf_counter() - t0) * 1000
            self.stages.append(
                StageRecord(
                    name=name,
                    latency_ms=round(dt, 2),
                    kind=kind,
                    notes=notes,
                    rows_processed=holder.get("rows"),
                )
            )

    def record_model_call(
        self, stage: str, model: str, input_tokens: int, output_tokens: int,
        latency_ms: float, cache_key: str | None = None,
    ) -> ModelCall:
        cached = cache_key is not None and cache_key in self._cache
        if cache_key is not None:
            self._cache[cache_key] = True
        call = ModelCall(stage, model, input_tokens, output_tokens, latency_ms, cached)
        self.calls.append(call)
        return call

    # ---- reporting ----------------------------------------------------

    def total_latency_ms(self) -> float:
        return round(sum(s.latency_ms for s in self.stages), 2)

    def latency_by_kind(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in self.stages:
            out[s.kind] = round(out.get(s.kind, 0.0) + s.latency_ms, 2)
        return out

    def total_tokens(self) -> dict[str, int]:
        return {
            "input": sum(c.input_tokens for c in self.calls),
            "output": sum(c.output_tokens for c in self.calls),
        }

    def total_cost(self) -> dict[str, float]:
        usd = sum(c.cost_usd for c in self.calls)
        return {"usd": round(usd, 6), "inr": round(usd * USD_TO_INR, 4)}

    def llm_share(self) -> dict[str, float]:
        """The LLM-vs-non-LLM breakdown the Round-2 brief asks for, measured."""
        by_kind = self.latency_by_kind()
        total = sum(by_kind.values()) or 1.0
        llm_ms = by_kind.get("llm", 0.0)
        return {
            "llm_latency_pct": round(100 * llm_ms / total, 2),
            "non_llm_latency_pct": round(100 * (total - llm_ms) / total, 2),
            "llm_stage_count": sum(1 for s in self.stages if s.kind == "llm"),
            "non_llm_stage_count": sum(1 for s in self.stages if s.kind != "llm"),
        }

    def report(self) -> dict:
        return {
            "total_latency_ms": self.total_latency_ms(),
            "latency_by_kind_ms": self.latency_by_kind(),
            "model_calls": len(self.calls),
            "tokens": self.total_tokens(),
            "estimated_cost": self.total_cost(),
            "llm_vs_non_llm": self.llm_share(),
            "stages": [asdict(s) for s in self.stages],
            "calls": [asdict(c) | {"cost_usd": round(c.cost_usd, 6)} for c in self.calls],
        }

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.report(), f, indent=2)
