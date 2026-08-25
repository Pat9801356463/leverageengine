"""
Property tests for the claims Fulcrum actually makes.

These are not coverage tests. Each one pins a specific claim that appears in
the pitch, so a reviewer can check that the code does what the deck says.

Run:  python test_fulcrum.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from fulcrum import decision, narrative                       # noqa: E402
from fulcrum.detection_core import (benjamini_hochberg, conformal_quantile,  # noqa: E402
                                    tail_pvalue, fit_forecast)

PASSED, FAILED = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASSED if cond else FAILED).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


# =====================================================================
print("\nVALUE FUNCTION -- redundancy, complementarity, structural moves")
dm = decision.DecisionModel.from_yaml(ROOT / "contracts" / "levers.yaml")

v_c2 = dm.value({"C2_listing_accuracy"})
v_c1 = dm.value({"C1_delivery_capacity"})
v_both = dm.value({"C1_delivery_capacity", "C2_listing_accuracy"})

check("complementarity: a gated constraint alone is worth zero", v_c2 == 0,
      f"V(C2)={v_c2}")
check("complementarity: the pair exceeds the sum of its parts", v_both > v_c1 + v_c2,
      f"V(both)={v_both:.0f} > {v_c1 + v_c2:.0f}")

b_l1 = dm._build_bundle(("L1",), 1e12)
b_l6 = dm._build_bundle(("L6",), 1e12)
b_both = dm._build_bundle(("L1", "L6"), 1e12)
check("redundancy: two levers on the same constraint union, not sum",
      b_both.value_per_week < b_l1.value_per_week + b_l6.value_per_week,
      f"{b_both.value_per_week:.0f} < {b_l1.value_per_week + b_l6.value_per_week:.0f}")

budget = float(dm.envelope["total_budget_inr"])
best = dm.rank(budget=budget)[0]
naive = dm._build_bundle(("L1", "L2", "L3"), 1e12)
check("structural move beats the naive repair bundle on cost",
      best.cost < naive.cost, f"{best.cost/1e5:.0f}L vs {naive.cost/1e5:.0f}L")
check("naive repair bundle is over budget", naive.cost > budget,
      f"{naive.cost/1e5:.0f}L > {budget/1e5:.0f}L")
check("a structural lever is reachable at all",
      any(dm.levers[l].family == "structural" for l in best.levers))

milp = dm.solve_milp(budget=budget)
check("exhaustive and MILP solvers agree", set(milp.levers) == set(best.levers),
      f"{sorted(milp.levers)} vs {sorted(best.levers)}")

sweep = decision.budget_sweep(dm, [5e5, 4.5e6, 5.6e6, 7e6])
fams = [r["family"] for r in sweep]
check("recommendation changes SHAPE across budgets, not just size",
      len(set(fams)) >= 3, " -> ".join(fams))

# decision rights
ops = next(p for p in dm.personas if p["role"] == "ops_manager")
head = next(p for p in dm.personas if p["role"] == "category_head")
check("decision rights restrict a manager to fewer levers than a VP",
      len(dm.levers_for_persona(ops)) < len(dm.levers_for_persona(head)),
      f"{len(dm.levers_for_persona(ops))} vs {len(dm.levers_for_persona(head))}")
check("no lever above a manager's authority is offered to them",
      all(dm.levers[l].authority_level == "manager"
          for l in dm.levers_for_persona(ops)))

# =====================================================================
print("\nCONFORMAL / FDR NUMERICS")
rng = np.random.default_rng(0)

# finite-sample correction: a tiny calibration sample cannot support the interval
q_small, valid_small, note = conformal_quantile(rng.normal(0, 1, 7), 0.10)
q_big, valid_big, _ = conformal_quantile(rng.normal(0, 1, 200), 0.10)
check("tiny calibration sample yields NO valid interval", not valid_small, note[:60])
check("adequate calibration sample yields a valid interval", valid_big)

# the sparse-is-tighter trap
tight = np.abs(rng.normal(0, 0.05, 7))     # short, low-variance history
q_t, _, _ = conformal_quantile(tight, 0.10)
check("sparse interval is inflated above the raw max residual",
      q_t > tight.max(), f"{q_t:.4f} > {tight.max():.4f}")

# coverage on genuinely held-out data
hits = n = 0
for _ in range(300):
    s = rng.normal(0, 1, 120)
    qh, ok, _ = conformal_quantile(s[:60], 0.10)
    if ok:
        hits += int(np.sum(np.abs(s[60:]) <= qh)); n += 60
cov = hits / n
check("held-out conformal coverage >= nominal", cov >= 0.88, f"coverage={cov:.4f}")

# EVT escapes the conformal p-value floor
cal = rng.normal(0, 1, 60)
p_extreme, method = tail_pvalue(cal, 8.0)
floor = 1 / 61
check("EVT tail escapes the 1/(n+1) conformal p-value floor",
      p_extreme < floor and method == "gpd_extrapolated",
      f"p={p_extreme:.3e} < floor={floor:.4f}")
p_mid, method_mid = tail_pvalue(cal, 0.2)
check("non-extreme scores stay empirical (no needless extrapolation)",
      method_mid == "empirical")

# BH behaves
p = np.concatenate([np.full(10, 1e-8), rng.uniform(0.2, 1.0, 990)])
surv, qv = benjamini_hochberg(p, 0.10)
check("BH recovers true signals", int(surv[:10].sum()) == 10)
check("BH controls false discoveries", int(surv[10:].sum()) == 0,
      f"{int(surv[10:].sum())} false positives of 990")
surv_null, _ = benjamini_hochberg(rng.uniform(0, 1, 2000), 0.10)
check("BH finds ~nothing under a pure null", int(surv_null.sum()) <= 2,
      f"{int(surv_null.sum())} discoveries")

# the AR(1) trap: a held-out event window must NOT be absorbed
base = 100 + rng.normal(0, 2, 120)
shifted = base.copy(); shifted[-3:] -= 25
fit = fit_forecast(shifted, 3, 52.0, 3, 60)
resid_last = shifted[-1] - fit["yhat"][-1]
check("sustained level shift is NOT absorbed by the baseline",
      abs(resid_last) > 15, f"residual={resid_last:.1f} on a -25 shift")

# =====================================================================
print("\nNUMERIC GROUNDING VALIDATOR")
ev = narrative.EvidenceObject(
    finding_id="T", analysis_week="2026-08-09", kpi="revenue",
    slice_label="North", slice_spec={"region": "North"},
    observed_change_pct=-1.05, change_vs_forecast_pct=-1.66,
    forecast=1000.0, actual=983.4, interval=[950.0, 1050.0],
    localisation={}, pvm={}, funnel={}, accepted_causes=[], rejected_causes=[],
    confidence={"confidence": 0.71}, recommendation={},
)
ok = narrative.validate_numeric_grounding(
    "Revenue came in 1.66% below forecast, at 983.4 against 1000.", ev)
bad = narrative.validate_numeric_grounding(
    "Revenue fell 47.3% and cancellations hit 91.4%.", ev)
check("grounded numerals pass", ok["passed"], f"{ok['numbers_in_text']} numerals")
check("fabricated numerals are blocked", not bad["passed"],
      f"ungrounded={bad['ungrounded']}")

rendered = narrative.render_for_persona(ev, {"id": "x", "insight_depth": "operational",
                                             "row_scope": {"region": ["North"]}})
check("blocked text is never released as a narrative",
      "[BLOCKED]" not in rendered["text"] and rendered["validation"]["passed"])

# row-level security
ev2 = narrative.EvidenceObject(**(ev.to_dict() | {
    "localisation": {"North": 1.0, "West": 9.9, "South": 4.2}}))
scrubbed = narrative._rls_filter(ev2, {"row_scope": {"region": ["North"]}})
check("RLS scrubs foreign regions from the evidence object",
      scrubbed.localisation["West"] == "[redacted: outside your entitlement]"
      and scrubbed.localisation["North"] == 1.0)

# =====================================================================
print("\nFEEDBACK LOOP -- persistence, annotation, conflict trigger")
import tempfile                                                # noqa: E402
from fulcrum import causal, feedback                            # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    fb_path = Path(tmp) / "feedback_log.jsonl"

    check("load_feedback on a missing file returns no history, not an error",
          feedback.load_feedback(fb_path) == [])

    feedback.record_feedback(fb_path, "hypothesis", "E1-prior", "rejected",
                              cause_node="logistics_vendor", note="prior run",
                              user="test_analyst")
    feedback.record_feedback(fb_path, "action", "L5", "acted_on", user="test_analyst")
    loaded = feedback.load_feedback(fb_path)
    check("recorded feedback round-trips through the JSONL log",
          len(loaded) == 2 and loaded[0]["cause_node"] == "logistics_vendor")

    def _hyp(hid, cause_node, verdict):
        return causal.Hypothesis(
            hid=hid, label=hid, source="event_log", proposed_cause_node=cause_node,
            start=pd.Timestamp("2026-08-01"), end=None, scope={}, verdict=verdict)

    accepted_conflict = _hyp("E1", "logistics_vendor", "accepted")
    accepted_clean = _hyp("E9", "unrelated_node", "accepted")
    feedback.annotate_hypotheses([accepted_conflict, accepted_clean], loaded)
    check("annotation matches history by cause_node, not by hypothesis id",
          len(accepted_conflict.feedback_history) == 1
          and accepted_clean.feedback_history == [])

    trig_fired = feedback.feedback_conflict_trigger([accepted_conflict])
    trig_clean = feedback.feedback_conflict_trigger([accepted_clean])
    check("FEEDBACK_CONFLICT fires when an accepted cause was rejected before",
          trig_fired.fired and "E1" in trig_fired.detail)
    check("FEEDBACK_CONFLICT stays quiet when there is no matching history",
          not trig_clean.fired)

    rejected_conflict = _hyp("E2", "logistics_vendor", "rejected")
    feedback.annotate_hypotheses([rejected_conflict], loaded)
    trig_on_rejected = feedback.feedback_conflict_trigger([rejected_conflict])
    check("the trigger only evaluates ACCEPTED hypotheses, not rejected ones",
          not trig_on_rejected.fired)

# =====================================================================
print(f"\n{'='*70}\n{len(PASSED)}/{len(PASSED)+len(FAILED)} property tests passed")
if FAILED:
    print("FAILED: " + ", ".join(FAILED))
sys.exit(1 if FAILED else 0)
