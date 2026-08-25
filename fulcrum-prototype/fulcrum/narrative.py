"""
S7 - Narrative: turn a finished evidence object into prose, safely.

The single most important rule in this module:

    THE LANGUAGE MODEL DOES NO ARITHMETIC AND HAS NO DATABASE ACCESS.

Every number was computed by deterministic, checkable code in S2-S6 before
the renderer is called. The renderer's only job is language. A validator then
extracts every numeral from the generated text and asserts it appears in the
evidence object; text containing an unsourced number is REJECTED, not warned
about. That is roughly forty lines of code and it closes the hallucinated-
statistic failure mode almost entirely.

Two renderers implement the same interface:
  * DeterministicRenderer - templates. Default, so the prototype is
    reproducible and costs nothing to demo.
  * LLMRenderer - calls a model. Produces better prose; goes through the
    SAME validator. Swapping renderers cannot weaken the safety property,
    which is the point of putting the guarantee in the validator rather
    than in the prompt.

Row-level security is enforced in the NARRATIVE, not only in the query. A
regional manager's text must not reference another region's figures even if
those figures reached the evidence object.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

# matches 12, 12.5, 1,234, -3.2, 45% ...
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")

# Numbers that are never a factual claim about the business: list positions,
# dates, small counts in phrases like "3 of 5 tests". Whitelisted so the
# validator stays strict about everything else.
SAFE_TOKENS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
               "12", "24", "2026", "100"}


@dataclass
class EvidenceObject:
    """Everything the renderer is allowed to know. Nothing else is in scope."""
    finding_id: str
    analysis_week: str
    kpi: str
    slice_label: str
    slice_spec: dict

    observed_change_pct: float
    change_vs_forecast_pct: float
    forecast: float
    actual: float
    interval: list[float]

    localisation: dict
    pvm: dict
    funnel: dict

    accepted_causes: list[dict]
    rejected_causes: list[dict]

    confidence: dict
    recommendation: dict

    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def numeric_vocabulary(self) -> set[str]:
        """Every number the renderer is permitted to utter."""
        vocab: set[str] = set()

        def walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, (list, tuple, set)):
                for v in o:
                    walk(v)
            elif isinstance(o, bool):
                return
            elif isinstance(o, (int, float)):
                vocab.update(_number_forms(float(o)))
            elif isinstance(o, str):
                for m in NUM_RE.findall(o):
                    vocab.add(m.replace(",", "").rstrip("."))

        walk(self.to_dict())
        vocab |= SAFE_TOKENS
        return vocab


def _number_forms(x: float) -> set[str]:
    """All the renderings a number might legitimately take in prose."""
    out: set[str] = set()
    for v in (x, abs(x)):
        for nd in (0, 1, 2):
            out.add(f"{v:.{nd}f}".rstrip(".") if nd == 0 else f"{v:.{nd}f}")
        out.add(str(int(round(v))))
        for scale, _ in ((1e5, "L"), (1e7, "Cr"), (1e3, "k"), (1e6, "M")):
            s = v / scale
            for nd in (0, 1, 2):
                out.add(f"{s:.{nd}f}")
        for nd in (0, 1, 2):        # percentage-point / ratio renderings
            out.add(f"{v*100:.{nd}f}")
    return {o.rstrip(".") for o in out}


class ValidationError(Exception):
    pass


def validate_numeric_grounding(text: str, evidence: EvidenceObject) -> dict:
    """Reject any numeral in `text` that is not derivable from `evidence`."""
    vocab = evidence.numeric_vocabulary()
    found = [m.replace(",", "").rstrip(".") for m in NUM_RE.findall(text)]
    ungrounded = [n for n in found if n not in vocab]
    return {
        "numbers_in_text": len(found),
        "ungrounded": ungrounded,
        "passed": len(ungrounded) == 0,
    }


# =====================================================================
# Renderers
# =====================================================================

class Renderer(Protocol):
    name: str
    def render(self, evidence: EvidenceObject, persona: dict) -> tuple[str, dict]: ...


def _rls_filter(evidence: EvidenceObject, persona: dict) -> EvidenceObject:
    """Row-level security applied to the EVIDENCE, before rendering.

    Filtering only the query is insufficient: aggregate context, placebo
    weights and comparison regions all carry other units' figures into the
    evidence object, from where a renderer would happily quote them.
    """
    scope = persona.get("row_scope", {})
    allowed = scope.get("region")
    if allowed in (None, "ALL"):
        return evidence

    d = json.loads(json.dumps(evidence.to_dict(), default=str))

    def scrub(o):
        if isinstance(o, dict):
            return {k: ("[redacted: outside your entitlement]"
                        if _is_foreign_region_key(k, allowed) else scrub(v))
                    for k, v in o.items()}
        if isinstance(o, list):
            return [scrub(v) for v in o]
        return o

    d = scrub(d)
    d["provenance"] = dict(d.get("provenance") or {})
    d["provenance"]["rls_applied"] = f"region in {allowed}"
    return EvidenceObject(**d)


REGION_NAMES = {"North", "South", "East", "West", "Central", "Northeast"}


def _is_foreign_region_key(key: str, allowed: list[str]) -> bool:
    return key in REGION_NAMES and key not in allowed


class DeterministicRenderer:
    name = "deterministic"

    def render(self, evidence: EvidenceObject, persona: dict) -> tuple[str, dict]:
        t0 = time.perf_counter()
        ev = _rls_filter(evidence, persona)
        depth = persona.get("insight_depth", "operational")
        text = (self._strategic(ev) if depth == "strategic" else self._operational(ev))
        meta = {"renderer": self.name, "latency_ms": (time.perf_counter() - t0) * 1000,
                "input_tokens": 0, "output_tokens": 0, "model": "deterministic"}
        return text, meta

    # -- operational: one lever, inside their authority, and what to watch
    def _operational(self, e: EvidenceObject) -> str:
        rec = e.recommendation
        acc = e.accepted_causes[0] if e.accepted_causes else None
        f = e.funnel
        lines = [
            f"{e.slice_label} - week ending {e.analysis_week}.",
            "",
            f"{e.kpi} came in {e.change_vs_forecast_pct:.1f}% below forecast "
            f"(week-on-week move was {e.observed_change_pct:.1f}%, so most of the "
            f"headline drop was expected seasonality).",
        ]
        if f.get("available"):
            lines.append(
                f"Orders held roughly steady ({f['gross_orders_change_pct']:.1f}%); "
                f"cancellations rose from {f['cancellation_rate_baseline']*100:.1f}% "
                f"to {f['cancellation_rate_current']*100:.1f}%. "
                f"This is a fulfilment problem, not a demand problem.")
        if acc:
            lines += ["", f"Likely cause: {acc['label']} (from {acc['start']})."]
        if rec.get("persona_actions"):
            lines += ["", "Your action:"]
            for a in rec["persona_actions"]:
                lines.append(f"  - {a['action']} (owner: {a['owner']})")
                mp = a["monitoring_plan"]
                lines.append(f"    Watch {mp['metric']} for {mp['window_weeks']} weeks; "
                             f"target {mp['success_threshold']}.")
        if rec.get("withheld_count"):
            lines.append(f"\n{rec['withheld_count']} further action(s) require "
                         f"authority above your level and have been routed to their owners.")
        return "\n".join(lines)

    # -- strategic: the bundle, the runner-up, the budget, the residual
    def _strategic(self, e: EvidenceObject) -> str:
        rec, conf = e.recommendation, e.confidence
        lines = [
            f"{e.kpi} - {e.slice_label}, week ending {e.analysis_week}.",
            "",
            f"Movement: {e.observed_change_pct:.1f}% week-on-week, but "
            f"{e.change_vs_forecast_pct:.1f}% against forecast - the difference is "
            f"seasonality that was already expected.",
        ]
        loc = e.localisation
        if loc.get("best_slice_share_of_gap_pct"):
            lines.append(f"Localisation: {loc['best_slice_share_of_gap_pct']:.1f}% of the "
                         f"shortfall sits in a single slice.")
        if e.pvm.get("available"):
            lines.append(f"Decomposition: volume effect dominates "
                         f"(units {e.pvm['units_change_pct']:.1f}%, "
                         f"ASP {e.pvm['asp_change_pct']:.1f}%) - not a pricing problem.")
        if e.accepted_causes:
            a = e.accepted_causes[0]
            lines += ["", f"Cause accepted: {a['label']}, effective {a['start']}."]
            if a.get("mechanism_effect_pp") is not None:
                lines.append(f"  Effect: +{a['mechanism_effect_pp']:.2f}pp on "
                             f"{a['mechanism_kpi']}, worth "
                             f"Rs {abs(a['revenue_impact_per_week'])/1e5:.1f}L per week.")
            if a.get("tests_passed"):
                lines.append(f"  Survived {a['tests_passed']} falsification tests.")
        if e.rejected_causes:
            lines += ["", "Rejected, with reasons:"]
            for r in e.rejected_causes:
                lines.append(f"  - {r['label']}: {r['reason']}")
        if rec.get("chosen"):
            c = rec["chosen"]
            lines += ["", f"Recommended bundle: {c['labels']}",
                      f"  Cost Rs {c['cost']/1e5:.1f}L against an envelope of "
                      f"Rs {rec['budget']/1e5:.1f}L; "
                      f"expected recovery Rs {c['value_per_week']/1e5:.1f}L per week."]
            if rec.get("runner_up"):
                r = rec["runner_up"]
                lines.append(f"  Runner-up: {r['labels']} at Rs {r['cost']/1e5:.1f}L "
                             f"recovering Rs {r['value_per_week']/1e5:.1f}L per week - "
                             f"rejected on value per rupee.")
        lines += ["", f"Confidence: {conf['confidence']:.2f} ({conf['mode']}). "
                      f"{conf['rationale']}"]
        if conf.get("unexplained_inr", 0) > 0:
            lines.append(f"Unexplained: Rs {conf['unexplained_inr']/1e5:.1f}L per week "
                         f"remains unattributed and is NOT covered by the accepted cause.")
            for t in conf.get("fired_triggers", []):
                if t.get("discriminating_test"):
                    lines.append(f"  Next test: {t['discriminating_test']}")
                    break
        return "\n".join(lines)


class LLMRenderer:
    """Production renderer. Same validator, so the guarantee is unchanged.

    Kept behind the same interface deliberately: if the safety property lived
    in the prompt, swapping renderers could silently weaken it.
    """
    name = "llm"

    def __init__(self, client=None, model: str = "claude-haiku-4-5"):
        self.client, self.model = client, model

    def render(self, evidence: EvidenceObject, persona: dict) -> tuple[str, dict]:
        if self.client is None:
            raise RuntimeError("LLMRenderer needs a client; the prototype defaults to "
                               "DeterministicRenderer so the demo is reproducible.")
        ev = _rls_filter(evidence, persona)
        payload = json.dumps(ev.to_dict(), default=str)
        prompt = (
            "You are rendering a finished analysis into prose for the persona below.\n"
            "HARD RULES:\n"
            "1. Use ONLY numbers present in the evidence JSON. Never compute a new one.\n"
            "2. Never introduce a fact that is not in the evidence JSON.\n"
            "3. If a value is marked redacted, do not mention it or work around it.\n"
            f"PERSONA: {json.dumps(persona)}\n"
            f"EVIDENCE: {payload}\n"
        )
        t0 = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model, max_tokens=800,
            messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return text, {
            "renderer": self.name, "model": self.model,
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }


def render_for_persona(evidence: EvidenceObject, persona: dict,
                       renderer: Renderer | None = None) -> dict:
    """Render, then VALIDATE. Ungrounded numerals fail the artefact."""
    renderer = renderer or DeterministicRenderer()
    text, meta = renderer.render(evidence, persona)
    check = validate_numeric_grounding(text, evidence)
    if not check["passed"]:
        text = ("[BLOCKED] The generated narrative contained numbers that are not "
                "present in the verified evidence object: "
                f"{check['ungrounded'][:8]}. Nothing was sent.")
    return {
        "persona": persona["id"],
        "channel": persona.get("channel"),
        "text": text,
        "validation": check,
        "render_meta": meta,
    }
