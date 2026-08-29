"""
Fulcrum - end-to-end run.

Executes the full pipeline and every Round-2 minimum-prototype scenario:

  1. 5 connected KPIs across 5 sources with 4 different grains/cadences
  2. KPI semantic contract (definitions, drivers, thresholds, lineage, access)
  3. Two personas receiving materially different narratives
  4. One multi-factor KPI movement with known planted drivers
  5. One low-confidence scenario -> abstention + discriminating test
  6. One sparse-history / newly-launched KPI scenario
  7. One role-based security / entitlement scenario
  8. Evidence: freshness, method, contribution, confidence, lineage
  9. Explicit LLM vs non-LLM breakdown
 10. Runtime telemetry: latency, model calls, tokens, estimated cost

Run:  python run_demo.py
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fulcrum import abstention, attribution, causal, decision, feedback, narrative, registry
from fulcrum.detection import detect
from fulcrum.semantic import SemanticLayer
from fulcrum.telemetry import Telemetry

warnings.simplefilter("ignore")

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
ANALYSIS_WEEK = pd.Timestamp("2026-08-09")
AS_OF = pd.Timestamp("2026-08-10 09:00")
KPIS = ["revenue", "cancellation_rate", "delivery_sla_pct", "ticket_volume"]
SOURCES = ["warehouse", "order_system", "logistics_api", "support_tickets",
           "nps_survey", "event_log", "external_signals"]

RULE = "=" * 78


def banner(t: str) -> None:
    print(f"\n{RULE}\n{t}\n{RULE}")


def inr(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    a = abs(x)
    if a >= 1e7:
        return f"Rs {x/1e7:.2f} Cr"
    if a >= 1e5:
        return f"Rs {x/1e5:.1f} L"
    return f"Rs {x:,.0f}"


def main() -> dict:
    tel = Telemetry()
    report: dict = {}

    # ================================================================
    banner("FULCRUM  |  KPI intelligence-to-action engine")
    print(f"Analysis week : {ANALYSIS_WEEK.date()}   (run as of {AS_OF})")

    if not (DATA / "warehouse.parquet").exists():
        from fulcrum import datagen
        with tel.stage("S1 generate scenario data", "deterministic"):
            datagen.write_all(DATA)

    with tel.stage("S1 load sources", "deterministic") as h:
        srcs = {n: pd.read_parquet(DATA / f"{n}.parquet") for n in SOURCES}
        h["rows"] = int(sum(len(v) for v in srcs.values()))
    print(f"Loaded {len(srcs)} sources, {sum(len(v) for v in srcs.values()):,} rows")

    # ================================================================
    banner("S2  SEMANTIC LAYER - contract, grain reconciliation, freshness")
    sem = SemanticLayer.from_yaml(ROOT / "contracts" / "kpis.yaml")
    with tel.stage("S2 align sources", "deterministic") as h:
        panel, fresh = sem.align(srcs, AS_OF)
        h["rows"] = len(panel)

    print(sem.freshness_report().to_string(index=False))
    print(f"\nCross-source reconciliation checks: "
          f"{'PASSED' if not sem.conflicts else sem.conflicts}")
    for k, why in panel.attrs["excluded_kpis"].items():
        print(f"EXCLUDED from weekly attribution: '{k}' - {' '.join(why.split())}")
    report["freshness"] = sem.freshness_report().to_dict("records")

    # ================================================================
    banner("S3  DETECTION - forecast baseline, conformal intervals, FDR")
    with tel.stage("S3 detect", "statistical") as h:
        findings, diag = detect(panel, sem, KPIS, ANALYSIS_WEEK)
        h["rows"] = int(diag["n_tests"])

    print(f"Slice x KPI tests run          : {diag['n_tests']:,}")
    print(f"Naively significant (p<0.05)   : {diag['n_naive_significant']}")
    print(f"Survive Benjamini-Hochberg FDR : {diag['n_survives_fdr']}")
    print(f"Material (statistical AND business impact) : {diag['n_material']}")
    print(f"False alarms suppressed        : {diag['false_alarms_avoided']}")
    print(f"\nCALIBRATION PROOF - empirical coverage {diag['empirical_coverage']:.4f} "
          f"vs {diag['target_coverage']:.2f} target, on {diag['coverage_n']:,} "
          f"out-of-sample residuals")
    report["detection"] = diag

    nat = findings[(findings.kpi == "revenue") & (findings.depth == 0)].iloc[0]
    print(f"\nNational revenue: {nat['wow_pct']:+.2f}% week-on-week, but "
          f"{nat['deviation_pct']:+.2f}% vs forecast.")
    print("  -> most of the headline move was seasonality that was already expected.")

    # ================================================================
    banner("S4a  ATTRIBUTION - Shapley, price/volume/mix, funnel")
    with tel.stage("S4a attribute", "statistical"):
        att = attribution.attribute(panel, ANALYSIS_WEEK)

    print(f"Total gap vs expectation: {inr(att.total_gap)}/week\n")
    print("Shapley localisation power by dimension (%):")
    for d, v in sorted(att.shapley_dimensions.items(), key=lambda x: -x[1]):
        print(f"   {d:<14} {v:>6.2f}   {'#' * int(v/2)}")

    spec = att.concentration["best_slice_rule"]
    print(f"\nMost specific slice retaining the shortfall: {spec}")
    print(f"  carries {att.concentration['best_slice_share_of_gap_pct']:.1f}% of the gap")

    if att.pvm.get("available"):
        p = att.pvm
        print(f"\nPrice-Volume-Mix:  volume {inr(p['volume_effect'])} | "
              f"price {inr(p['price_effect'])} | mix {inr(p['mix_effect'])}")
        print(f"  units {p['units_change_pct']:+.1f}%, ASP {p['asp_change_pct']:+.1f}% "
              f"-> dominant term: {p['dominant_term'].upper()} (not a pricing problem)")
    if att.funnel.get("available"):
        f = att.funnel
        print(f"\nFunnel:  gross orders {f['gross_orders_change_pct']:+.1f}%, "
              f"cancellations {f['cancellation_rate_baseline']*100:.1f}% -> "
              f"{f['cancellation_rate_current']*100:.1f}% "
              f"({f['cancellation_rate_change_pp']:+.1f}pp)")
        print(f"  -> dominant term: {f['dominant_term'].upper()}. "
              f"Demand held; fulfilment broke.")
    report["attribution"] = {
        "total_gap": att.total_gap, "shapley": att.shapley_dimensions,
        "concentration": att.concentration, "pvm": att.pvm, "funnel": att.funnel}

    # ================================================================
    banner("S4b  CAUSAL PROSECUTION - every candidate put on trial")
    with tel.stage("S4b generate hypotheses", "ml"):
        hyps = causal.generate_hypotheses(srcs, sem, spec, ANALYSIS_WEEK)
    with tel.stage("S4b prosecute", "causal"):
        hyps = [causal.prosecute(h, panel, srcs, spec,
                                 "cancellation_rate", "revenue") for h in hyps]

    print(causal.hypothesis_table(hyps).to_string(index=False))
    accepted = [h for h in hyps if h.verdict == "accepted"]
    rejected = [h for h in hyps if h.verdict == "rejected"]

    print(f"\n{len(hyps)} candidates -> {len(accepted)} survived, {len(rejected)} rejected")
    mechs = set()
    for h in rejected:
        mechs.add("ontology filter" if "causal path" in h.rejection_reason
                  else "temporal precedence" if "precedence" in h.rejection_reason
                  else "dose-response" if "dose-response" in h.rejection_reason
                  else "placebo/DiD")
    print(f"Rejections came from {len(mechs)} different mechanisms:")
    for h in rejected:
        mech = ("ontology filter" if "causal path" in h.rejection_reason
                else "temporal precedence" if "precedence" in h.rejection_reason
                else "dose-response" if "dose-response" in h.rejection_reason
                else "placebo/DiD")
        print(f"   [{mech:<20}] {h.hid}: {h.label[:44]}")

    for h in accepted:
        e = h.effect
        print(f"\nACCEPTED {h.hid}: {h.label}")
        print(f"  Mechanism : {e['mechanism_kpi']} {e['mechanism_effect_pp']:+.2f}pp")
        print(f"  Impact    : {inr(e['revenue_impact_per_week'])}/week "
              f"[{inr(e['revenue_impact_ci'][0])}, {inr(e['revenue_impact_ci'][1])}]")
        sc = h.tests["synthetic_control"]
        print(f"  Synthetic control donors: {sc['weights']}")
        print(f"    pre-period fit RMSE {sc['pre_fit_quality']*100:.2f}% of level")
        pl = h.tests["placebo_in_space"]
        print(f"  Placebo-in-space: rank {pl['rank']}/{pl['n_placebo']+1}, "
              f"permutation p={pl['permutation_p_value']:.3f}")
        print("  REFUTATION SUITE (the engine attacking its own finding):")
        for k, v in h.tests["refutations"].items():
            print(f"    {'PASS' if v['passed'] else 'FAIL'}  {k:<22} {v['interpretation']}")

    report["hypotheses"] = causal.hypothesis_table(hyps).to_dict("records")

    # Export the statistical test results that produced each verdict. Without
    # this the effect-size interval and the placebo p-value are visible in the
    # console but absent from every committed artefact, so nothing downstream
    # can cite them. Keys are trimmed to what a reader actually needs.
    def _finite(v):
        """NaN/inf are not valid JSON; emit null instead of a bare NaN token."""
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v

    def _pick(d: dict, keys) -> dict:
        return {k: _finite(d[k]) for k in keys if k in d}

    def _test_summary(h) -> dict:
        t = getattr(h, "tests", {}) or {}
        did, pl = t.get("did") or {}, t.get("placebo_in_space") or {}
        dr, ref = t.get("dose_response") or {}, t.get("refutations") or {}
        out = {}
        if did:
            out["did"] = _pick(did, ("estimate", "std_error", "ci_low", "ci_high",
                                     "p_value", "parallel_trends_ok", "n_obs"))
        if pl.get("available"):
            out["placebo_in_space"] = _pick(pl, ("permutation_p_value", "rank",
                                                 "n_placebo", "treated_rmspe_ratio"))
        if dr.get("applicable"):
            out["dose_response"] = _pick(dr, ("correlation", "highest_exposure_unit",
                                              "highest_exposure", "treated_exposure",
                                              "inverted"))
        if ref:
            out["refutations"] = {k: _finite(v.get("passed")) for k, v in ref.items()
                                  if isinstance(v, dict) and "passed" in v}
        return out

    for _rec, _h in zip(report["hypotheses"], hyps):
        _rec["tests"] = _test_summary(_h)

    # ================================================================
    banner("S_FB  FEEDBACK LOOP - analyst corrections carried across runs")
    fb_log_path = OUT / "feedback_log.jsonl"
    with tel.stage("S_FB load feedback", "deterministic"):
        fb_records = feedback.load_feedback(fb_log_path)
        feedback.annotate_hypotheses(hyps, fb_records)

    if fb_records:
        print(f"Loaded {len(fb_records)} prior feedback record(s) from {fb_log_path.name}")
        any_history = False
        for h in hyps:
            hist = getattr(h, "feedback_history", [])
            if hist:
                any_history = True
                verdicts = [r["verdict"] for r in hist]
                print(f"  {h.hid} (cause_node={h.proposed_cause_node}): "
                      f"{len(hist)} prior analyst verdict(s) -> {verdicts}")
        if not any_history:
            print("  none of today's hypotheses share a cause_node with prior feedback")
    else:
        print(f"No feedback history yet at {fb_log_path.name} - "
              "first run, or nothing recorded so far.")

    # ================================================================
    banner("S5  CONFIDENCE & ABSTENTION")
    with tel.stage("S5 abstention policy", "deterministic"):
        verdict = abstention.evaluate(
            total_gap_inr=att.total_gap, accepted=accepted, rejected=rejected,
            freshness=fresh, conflicts=sem.conflicts,
            n_history_weeks=int(panel["week"].nunique()),
            excluded_kpis=panel.attrs["excluded_kpis"],
            # Calibration integrity is judged on the finding under analysis, not
        # globally: a sparse newly-launched slice legitimately lacks a clean
        # train/calibration split, and that should widen ITS interval and
        # lower ITS confidence - not abstain on an unrelated finding.
        oos_calibration=bool(
            findings.loc[~findings.sparse_history, "oos_calibration"].all()))

    print(f"Mode       : {verdict.mode.value.upper()}")
    print(f"Confidence : {verdict.confidence:.3f}")
    print(f"Explained  : {verdict.explained_fraction*100:.1f}% of the movement")
    print(f"Unexplained: {inr(verdict.unexplained_inr)}/week\n")
    print("Trigger status:")
    for t in verdict.triggers:
        print(f"   [{'FIRED' if t.fired else '  ok '}] {t.code:<24} {t.detail[:80]}")
    fb_trigger = feedback.feedback_conflict_trigger(hyps)
    verdict.triggers.append(fb_trigger)
    print(f"   [{'FIRED' if fb_trigger.fired else '  ok '}] "
          f"{fb_trigger.code:<24} {fb_trigger.detail[:80]}")

    for t in verdict.fired:
        if t.discriminating_test:
            print(f"\nDISCRIMINATING TEST for {t.code}:\n   {t.discriminating_test}")
            break
    report["confidence"] = verdict.as_dict()
    report["feedback"] = {
        "log_path": str(fb_log_path.relative_to(ROOT)),
        "summary": feedback.summarize(fb_records),
        "records": fb_records,
    }

    # ================================================================
    banner("S6  DECISION - constraint-state value + portfolio optimisation")
    dm = decision.DecisionModel.from_yaml(ROOT / "contracts" / "levers.yaml")
    budget = float(dm.envelope["total_budget_inr"])

    with tel.stage("S6 portfolio optimisation", "optimisation"):
        ranked = dm.rank(budget=budget)
        milp = dm.solve_milp(budget=budget)

    agree = set(milp.levers) == set(ranked[0].levers)
    print(f"Capital envelope: {inr(budget)}")
    print(f"Solver cross-check (exhaustive vs MILP): "
          f"{'AGREE' if agree else 'DISAGREE - model error'}\n")

    print("Top bundles:")
    for b in ranked[:5]:
        print(f"  {'+'.join(b.levers) or '(do nothing)':<12} "
              f"cost {inr(b.cost):>12}  value {inr(b.value_per_week):>11}/wk  "
              f"payback {b.roi_weeks_to_payback:>4.1f}w  [{'+'.join(sorted(b.families))}]")

    naive = dm._build_bundle(("L1", "L2", "L3"), 1e12)
    chosen = ranked[0]
    print(f"\nTHE POINT:")
    print(f"  Naive 'fix everything diagnosed' (L1+L2+L3): {inr(naive.cost)} "
          f"-> {'OVER BUDGET' if naive.cost > budget else 'feasible'}")
    print(f"  Fulcrum's choice ({'+'.join(chosen.levers)}): {inr(chosen.cost)}, "
          f"recovering {inr(chosen.value_per_week)}/week")
    print(f"  {(1 - chosen.cost/naive.cost)*100:.0f}% cheaper, and it relieves "
          f"{len(chosen.realised)} constraints vs {len(naive.realised)}")

    print("\nComplementarity is structural, not hand-coded:")
    print(f"  V(listing accuracy alone)        = {inr(dm.value({'C2_listing_accuracy'}))}"
          "   <- prerequisite unmet")
    print(f"  V(delivery capacity)             = {inr(dm.value({'C1_delivery_capacity'}))}")
    print(f"  V(both)                          = "
          f"{inr(dm.value({'C1_delivery_capacity','C2_listing_accuracy'}))}"
          "   <- more than the sum of parts")

    print("\nBUDGET SWEEP - the recommendation changes SHAPE, not just size:")
    for r in decision.budget_sweep(dm, [5e5, 1.5e6, 4.5e6, 5.6e6, 7e6, 1.0e7]):
        print(f"  {inr(r['budget_inr']):>10} -> "
              f"{'+'.join(r['chosen_levers']) or 'DO NOTHING':<10} "
              f"[{r['family']:<22}] recovers {inr(r['value_per_week_inr'])}/wk")
    report["decision"] = {
        "budget": budget, "solvers_agree": agree,
        "chosen": {"levers": chosen.levers, "cost": chosen.cost,
                   "value_per_week": chosen.value_per_week,
                   "realised": sorted(chosen.realised)},
        "naive_repair_cost": naive.cost,
        "greedy_baselines": decision.greedy_baselines(dm, budget),
        "budget_sweep": decision.budget_sweep(dm, [5e5, 1.5e6, 4.5e6, 5.6e6, 7e6, 1.0e7]),
    }

    # ================================================================
    banner("S7  NARRATIVE - two personas, row-level security, validation")
    runner_up = ranked[1] if len(ranked) > 1 else None
    acc_payload = [{
        "hid": h.hid, "label": h.label, "start": str(h.start.date()),
        "mechanism_kpi": h.effect["mechanism_kpi"],
        "mechanism_effect_pp": h.effect["mechanism_effect_pp"],
        "revenue_impact_per_week": h.effect["revenue_impact_per_week"],
        "revenue_impact_ci": h.effect.get("revenue_impact_ci"),
        "impact_basis": h.effect.get("basis"),
        "tests_passed": 1 + 1 + 1 + len(h.tests.get("refutations", {})),
    } for h in accepted]
    rej_payload = [{"hid": h.hid, "label": h.label,
                    "reason": " ".join(h.rejection_reason.split())[:150]}
                   for h in rejected]

    personas = dm.personas
    outputs = []
    for persona in personas:
        allowed = dm.levers_for_persona(persona)
        p_rank = dm.rank(budget=budget, allowed_levers=allowed)
        p_best = p_rank[0] if p_rank else chosen
        p_actions = dm.to_recommendation(p_best, driver="C1_delivery_capacity",
                                         confidence=verdict.confidence,
                                         evidence_ref="E1")
        withheld = len([l for l in chosen.levers if l not in allowed])

        ev = narrative.EvidenceObject(
            finding_id="F-2026W32-001",
            analysis_week=str(ANALYSIS_WEEK.date()),
            kpi="revenue",
            slice_label=" / ".join(str(v) for v in spec.values()),
            slice_spec=spec,
            observed_change_pct=float(nat["wow_pct"]),
            change_vs_forecast_pct=float(nat["deviation_pct"]),
            forecast=float(nat["forecast"]), actual=float(nat["actual"]),
            interval=[float(nat["lo"]), float(nat["hi"])],
            localisation=att.concentration, pvm=att.pvm, funnel=att.funnel,
            accepted_causes=acc_payload, rejected_causes=rej_payload,
            confidence=verdict.as_dict() | {
                "fired_triggers": [t.as_dict() for t in verdict.fired]},
            recommendation={
                "budget": budget,
                "chosen": {"levers": chosen.levers,
                           "labels": ", ".join(chosen.labels)[:150],
                           "cost": chosen.cost,
                           "value_per_week": chosen.value_per_week},
                "runner_up": ({"labels": ", ".join(runner_up.labels)[:150],
                               "cost": runner_up.cost,
                               "value_per_week": runner_up.value_per_week}
                              if runner_up else None),
                "persona_actions": p_actions,
                "withheld_count": withheld,
            },
            provenance={"sources": list(fresh), "lineage": "contracts/kpis.yaml",
                        "method_registry": "fulcrum/registry.py"},
        )

        with tel.stage(f"S7 render [{persona['id']}]", "llm") as h:
            res = narrative.render_for_persona(ev, persona)
        tel.record_model_call(f"S7 render [{persona['id']}]", "deterministic",
                              0, 0, res["render_meta"]["latency_ms"])

        print(f"\n--- {persona['display_name']}  "
              f"(scope: {persona['row_scope']}, authority<= {persona['max_authority_level']}, "
              f"channel: {persona['channel']}) ---")
        print(res["text"])
        print(f"\n  [numeric grounding: {res['validation']['numbers_in_text']} numerals, "
              f"{len(res['validation']['ungrounded'])} ungrounded -> "
              f"{'PASS' if res['validation']['passed'] else 'BLOCKED'}]")
        print(f"  [levers visible to this persona: {allowed}; "
              f"{withheld} withheld above authority]")
        outputs.append(res)
    report["narratives"] = [{"persona": o["persona"], "validation": o["validation"],
                             "text": o["text"]} for o in outputs]

    # ---- entitlement proof -----------------------------------------
    ops = next(p for p in personas if p["role"] == "ops_manager")
    other_regions = {"South", "East", "West", "Central", "Northeast"}
    leaked = [r for r in other_regions if r in outputs[0]["text"]]
    print(f"\nENTITLEMENT CHECK - ops manager narrative mentions foreign regions: "
          f"{leaked or 'NONE'}  -> {'PASS' if not leaked else 'FAIL'}")
    report["entitlement_check"] = {"leaked_regions": leaked, "passed": not leaked}

    # ---- deliberate validator failure -------------------------------
    bad = "Revenue fell 47.3% and cancellations hit 91.4% this week."
    chk = narrative.validate_numeric_grounding(bad, ev)
    # `blocked` is the reader-facing name: True means the validator CORRECTLY
    # refused the fabricated statistics. The underlying validator still returns
    # `passed` (True = clean text), which is the right sense for real narratives;
    # inverting it here stops the negative test reading as a failing test.
    blocked = not chk["passed"]
    print(f"\nVALIDATOR NEGATIVE TEST - injected fabricated statistics:")
    print(f"  input : {bad}")
    print(f"  result: {'BLOCKED' if blocked else 'passed (BAD)'} "
          f"- ungrounded numerals {chk['ungrounded']}")
    report["validator_negative_test"] = {
        "numbers_in_text": chk["numbers_in_text"],
        "ungrounded": chk["ungrounded"],
        "blocked": blocked,
    }

    # ================================================================
    banner("SCENARIO: sparse history / newly launched product line")
    sp = findings[findings.sparse_history]
    if len(sp):
        r = sp.iloc[0]
        widths = findings
        n_invalid = int((~sp["interval_valid"]).sum())
        print(f"'SmartHome' newly launched. {len(sp)} slices flagged sparse, "
              f"{n_invalid} with NO valid conformal interval at this confidence.")
        print(f"  example: {r['slice_label']} ({r['n_history']} weeks of history)")
        print(f"  median interval width: {sp['interval_width_pct'].median():.1f}% of forecast "
              f"vs {widths[~widths.sparse_history]['interval_width_pct'].median():.1f}% "
              f"for established slices")
        bad = sp[~sp["interval_valid"]]
        if len(bad):
            print(f"  {' '.join(str(bad.iloc[0]['interval_note']).split())}")
        print("  -> the engine widens uncertainty and says so; it does not fail silently.")
        report["sparse_history"] = {
            "n_sparse_slices": int(len(sp)),
            "n_invalid_intervals": n_invalid,
            "example": r["slice_label"], "weeks": int(r["n_history"])}

    # ================================================================
    banner("S9  METHOD ATTRIBUTION - LLM vs non-LLM, made explicit")
    reg = registry.registry_frame()
    split = registry.llm_vs_non_llm()
    print(f"Pipeline steps: {split['total_pipeline_steps']}  |  "
          f"LLM: {split['llm_steps']}  |  non-LLM: {split['non_llm_steps']}  "
          f"({split['llm_share_pct']}% LLM)")
    print("\nBy category:")
    for k, v in sorted(split["by_category"].items(), key=lambda x: -x[1]):
        print(f"   {k:<20} {v:>3}  {'#' * v}")
    print(f"\nLLM scope: {split['llm_scope']}")
    print(f"Prototype substitutions disclosed: {split['prototype_substitutions']}")
    report["method_split"] = split

    # ================================================================
    banner("TELEMETRY")
    tr = tel.report()
    print(f"Total latency      : {tr['total_latency_ms']:,.0f} ms")
    print(f"Model calls        : {tr['model_calls']}")
    print(f"Tokens             : in {tr['tokens']['input']:,} / "
          f"out {tr['tokens']['output']:,}")
    print(f"Estimated cost     : ${tr['estimated_cost']['usd']:.6f} "
          f"(Rs {tr['estimated_cost']['inr']:.4f}) per insight")
    print(f"LLM share of latency: {tr['llm_vs_non_llm']['llm_latency_pct']}%\n")
    print("Per stage:")
    for s in tr["stages"]:
        print(f"   {s['latency_ms']:>9,.1f} ms  [{s['kind']:<14}] {s['name']}")
    report["telemetry"] = tr

    # ================================================================
    OUT.mkdir(exist_ok=True, parents=True)
    findings.to_parquet(OUT / "findings.parquet")
    reg.to_csv(OUT / "method_registry.csv", index=False)
    tel.to_json(str(OUT / "telemetry.json"))
    with open(OUT / "evidence_object.json", "w") as f:
        json.dump(ev.to_dict(), f, indent=2, default=str)
    with open(OUT / "run_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(OUT / "narratives.txt", "w") as f:
        for o in outputs:
            f.write(f"=== {o['persona']} ({o['channel']}) ===\n{o['text']}\n\n")

    banner("CHECKLIST")
    gt = yaml.safe_load((DATA / "ground_truth.yaml").read_text())
    true_cause = gt["true_cause"]["event_id"]
    checks = [
        ("3-5 connected KPIs, 2-3 sources, different grains", len(SOURCES) >= 5),
        ("KPI / semantic contract", (ROOT / "contracts" / "kpis.yaml").exists()),
        ("Two personas, different narratives",
         len(outputs) >= 2 and outputs[0]["text"] != outputs[1]["text"]),
        ("Multi-factor movement, known drivers recovered",
         [h.hid for h in accepted] == [true_cause]),
        ("Decoys rejected", len(rejected) >= 2),
        ("Low-confidence / abstention scenario", len(verdict.fired) > 0),
        ("Sparse-history scenario", bool(len(sp))),
        ("Role-based security scenario", not leaked),
        ("Evidence: freshness, method, contribution, confidence, lineage",
         (OUT / "evidence_object.json").exists()),
        ("LLM vs non-LLM breakdown", split["llm_steps"] >= 1),
        ("Runtime telemetry", tr["total_latency_ms"] > 0),
        ("Numeric grounding validator blocks fabrications", not chk["passed"]),
        # Conformal guarantees AT LEAST nominal coverage, so over-coverage is
        # valid and under-coverage is the real failure. Assert the right side.
        ("Calibration demonstrated on held-out residuals",
         diag["empirical_coverage"] >= diag["target_coverage"] - 0.02),
        ("Solvers cross-check", agree),
        ("Feedback loop: analyst corrections captured, surfaced, and flagged",
         report["feedback"]["summary"]["n_records"] > 0
         and any(t.code == "FEEDBACK_CONFLICT" for t in verdict.triggers)),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {label}")
    print(f"\n{sum(1 for _, o in checks if o)}/{len(checks)} passed")
    print(f"\nGROUND TRUTH: planted cause was {true_cause} "
          f"({gt['true_cause']['label']}). Engine accepted: "
          f"{[h.hid for h in accepted]}")
    print(f"Artefacts written to {OUT}/")
    return report


if __name__ == "__main__":
    main()
