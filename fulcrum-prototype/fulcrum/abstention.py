"""
S5 - Confidence and abstention.

Round-2 requirement: "Communicates uncertainty and abstains when evidence is
insufficient or contradictory."

Abstention here is a POLICY WITH ENUMERATED TRIGGERS, not a fallback for when
something breaks. Each trigger has a defined condition, a defined response,
and - where the engine refuses - a DISCRIMINATING TEST: the specific data
that would resolve the question. An engine that says "I don't know, and here
is what would tell us" is more useful than one that always has an answer.

The triggers are deliberately enumerated in one place so the abstention
language is consistent everywhere, rather than reimplemented three different
ways across the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np


class Mode(str, Enum):
    CONFIDENT = "confident"        # one explanation dominates, residual small
    COMPETING = "competing"        # several survive, none dominant
    ABSTAIN = "abstain"            # evidence insufficient or contradictory


@dataclass
class AbstentionTrigger:
    code: str
    fired: bool
    detail: str
    discriminating_test: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConfidenceVerdict:
    mode: Mode
    confidence: float
    explained_fraction: float
    unexplained_inr: float
    triggers: list[AbstentionTrigger]
    accepted_hypotheses: list[str]
    competing_hypotheses: list[str]
    rationale: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["mode"] = self.mode.value
        d["triggers"] = [t.as_dict() for t in self.triggers]
        return d

    @property
    def fired(self) -> list[AbstentionTrigger]:
        return [t for t in self.triggers if t.fired]


# thresholds, in one place so they can be tuned and audited
RESIDUAL_ABSTAIN = 0.40        # >40% of the movement unexplained -> abstain
RESIDUAL_FLAG = 0.15           # >15% -> report explicitly but still answer
MIN_HISTORY_WEEKS = 60
CI_OVERLAP_TOL = 0.60          # competing hypotheses if intervals overlap this much


def evaluate(
    total_gap_inr: float,
    accepted: list,                 # list[Hypothesis]
    rejected: list,
    freshness: dict,                # dict[str, SourceFreshness]
    conflicts: list[dict],
    n_history_weeks: int,
    excluded_kpis: dict[str, str],
    oos_calibration: bool = True,
) -> ConfidenceVerdict:
    triggers: list[AbstentionTrigger] = []

    # ---- T1: unexplained residual --------------------------------------
    explained = sum(abs(h.effect.get("revenue_impact_per_week") or 0.0) for h in accepted)
    total = abs(total_gap_inr) if total_gap_inr else 0.0
    frac = min(explained / total, 1.0) if total > 0 else 0.0
    residual = max(total - explained, 0.0)

    triggers.append(AbstentionTrigger(
        code="UNEXPLAINED_RESIDUAL",
        fired=bool(1 - frac > RESIDUAL_ABSTAIN),
        detail=(f"{100*(1-frac):.1f}% of the movement "
                f"(Rs {residual/1e5:.1f}L/week) is not attributed to any "
                f"hypothesis that survived causal testing"),
        discriminating_test=(
            "Pull marketplace search-ranking and impression logs for the affected "
            "weeks. A ranking demotion would produce a demand-side shortfall with "
            "no event-log entry and no support-ticket signature, which is exactly "
            "the pattern in the residual."),
    ))

    # ---- T2: competing hypotheses --------------------------------------
    competing = []
    if len(accepted) > 1:
        ivs = []
        for h in accepted:
            ci = h.effect.get("revenue_impact_ci")
            if ci:
                ivs.append((h.hid, min(ci), max(ci)))
        for i in range(len(ivs)):
            for j in range(i + 1, len(ivs)):
                _, a0, a1 = ivs[i]
                _, b0, b1 = ivs[j]
                inter = max(0.0, min(a1, b1) - max(a0, b0))
                union = max(a1, b1) - min(a0, b0)
                if union > 0 and inter / union > CI_OVERLAP_TOL:
                    competing = [ivs[i][0], ivs[j][0]]
    triggers.append(AbstentionTrigger(
        code="COMPETING_EXPLANATIONS",
        fired=bool(competing),
        detail=(f"hypotheses {competing} have overlapping effect intervals; "
                "the data cannot rank them" if competing else
                "no two accepted hypotheses have indistinguishable effects"),
        discriminating_test=(
            "Run a staggered rollout: apply the remedy for one hypothesis in a "
            "subset of pincodes and hold the rest. The split identifies which "
            "mechanism is load-bearing." if competing else ""),
    ))

    # ---- T3: stale sources ---------------------------------------------
    stale = [f.source for f in freshness.values() if f.is_stale]
    triggers.append(AbstentionTrigger(
        code="STALE_SOURCE",
        fired=bool(stale),
        detail=(f"sources beyond their contracted staleness tolerance: {stale}"
                if stale else "all contributing sources within tolerance"),
        discriminating_test="Re-run once the named source has refreshed." if stale else "",
    ))

    # ---- T4: cross-source disagreement ---------------------------------
    triggers.append(AbstentionTrigger(
        code="SOURCE_CONFLICT",
        fired=bool(conflicts),
        detail=(f"{len(conflicts)} reconciliation check(s) failed: "
                f"{[c['check'] for c in conflicts]}" if conflicts else
                "cross-source reconciliation checks passed"),
        discriminating_test=("Reconcile the disagreeing systems before acting; the "
                             "engine will not choose between them." if conflicts else ""),
    ))

    # ---- T5: insufficient history --------------------------------------
    triggers.append(AbstentionTrigger(
        code="SPARSE_HISTORY",
        fired=bool(n_history_weeks < MIN_HISTORY_WEEKS),
        detail=(f"only {n_history_weeks} weeks of history; a seasonal baseline and "
                f"a valid pre-period control both require ~{MIN_HISTORY_WEEKS}"
                if n_history_weeks < MIN_HISTORY_WEEKS else
                f"{n_history_weeks} weeks of history available"),
        discriminating_test=("Report the deviation but withhold any causal claim until "
                             "a full seasonal cycle is observed, or borrow strength from "
                             "comparable established slices with the uncertainty widened."
                             if n_history_weeks < MIN_HISTORY_WEEKS else ""),
    ))

    # ---- T6: grain policy ----------------------------------------------
    triggers.append(AbstentionTrigger(
        code="GRAIN_POLICY",
        fired=bool(excluded_kpis),
        detail=(f"KPIs excluded from weekly attribution by contract: "
                f"{list(excluded_kpis)}" if excluded_kpis else "no grain exclusions"),
        discriminating_test=("These metrics cannot support a weekly claim at their "
                             "native grain; interpolating them would present "
                             "interpolation as evidence." if excluded_kpis else ""),
    ))

    # ---- T7: calibration integrity -------------------------------------
    triggers.append(AbstentionTrigger(
        code="CALIBRATION_INTEGRITY",
        fired=not oos_calibration,
        detail=("conformal intervals were calibrated on in-sample residuals; the "
                "coverage guarantee is not valid" if not oos_calibration else
                "intervals calibrated on out-of-sample residuals"),
        discriminating_test="Extend history until a clean train/calibration split fits."
        if not oos_calibration else "",
    ))

    # ---- resolve the mode ----------------------------------------------
    blocking = {"STALE_SOURCE", "SOURCE_CONFLICT", "CALIBRATION_INTEGRITY"}
    fired = {t.code for t in triggers if t.fired}

    if fired & blocking:
        mode = Mode.ABSTAIN
        rationale = ("Abstaining: the evidence base itself is compromised "
                     f"({sorted(fired & blocking)}). No causal claim is made.")
    elif "UNEXPLAINED_RESIDUAL" in fired and not accepted:
        mode = Mode.ABSTAIN
        rationale = ("Abstaining: a material movement is present but no candidate "
                     "cause survived causal testing.")
    elif "COMPETING_EXPLANATIONS" in fired:
        mode = Mode.COMPETING
        rationale = ("Multiple explanations survive with indistinguishable effects. "
                     "Presenting all rather than selecting one.")
    elif accepted:
        mode = Mode.CONFIDENT
        rationale = (f"One explanation survived all falsification tests and accounts "
                     f"for {100*frac:.0f}% of the movement.")
    else:
        mode = Mode.ABSTAIN
        rationale = "Abstaining: no hypothesis survived causal testing."

    # confidence: starts from explained fraction, penalised per fired trigger
    conf = frac
    for t in triggers:
        if t.fired and t.code != "GRAIN_POLICY":
            conf *= 0.82
    if mode is Mode.ABSTAIN:
        conf = min(conf, 0.35)

    return ConfidenceVerdict(
        mode=mode,
        confidence=float(np.clip(conf, 0.0, 0.99)),
        explained_fraction=float(frac),
        unexplained_inr=float(residual),
        triggers=triggers,
        accepted_hypotheses=[h.hid for h in accepted],
        competing_hypotheses=competing,
        rationale=rationale,
    )
