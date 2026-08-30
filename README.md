# Fulcrum — KPI Intelligence-to-Action Engine

Accenture Innovation Challenge 2026 · Round 2 prototype
Problem statement 3 — **BusinessIntelligence.ai**

**Live demo — [Fulcrum case file](https://pat9801356463.github.io/leverageengine/dashboard.html)**  ·  one self-contained static page; renders without JavaScript.

---

## The thesis in one line

> Most root-cause tools tell you what to fix. **Fixing everything that is broken is not the same as doing the best thing you can afford.**

A dashboard shows *what* changed. A good analyst explains *why*. Neither answers the question a business actually faces: **given a budget, what is the single best move?**

Fulcrum closes that last mile — and refuses to answer when the evidence does not support one.

---

## Quick start

```bash
cd fulcrum-prototype
pip install -r requirements.txt
python run_demo.py
```

The engine lives in `fulcrum-prototype/`; the paths under **Layout** below are relative to it.

Runtime ≈ 37 s, dominated by 2,016 conformal forecasts. Artefacts land in `outputs/`.

---

## What the run demonstrates

| Round-2 requirement | Where |
|---|---|
| 3–5 connected KPIs, 2–3 sources, different grains/cadences | 5 KPIs, 6 sources, 4 native grains (`contracts/kpis.yaml`) |
| KPI / semantic contract | `contracts/kpis.yaml` — definitions, drivers, thresholds, lineage, access |
| Two personas, different narratives | Ops Manager vs Category Head (`contracts/levers.yaml`) |
| Multi-factor movement with known drivers | Planted causal chain + 2 decoys (`fulcrum/datagen.py`) |
| Low-confidence / abstention scenario | 7 enumerated triggers (`fulcrum/abstention.py`) |
| Sparse-history KPI | Newly-launched product line, 14 weeks |
| Role-based security | RLS enforced in the **narrative**, not just the query |
| Evidence: freshness, method, contribution, confidence, lineage | `outputs/evidence_object.json` |
| LLM vs non-LLM breakdown | `outputs/method_registry.csv` — 1 of 27 steps is generative |
| Runtime telemetry | `outputs/telemetry.json` |

---

## Architecture

```
S1  Scenario data ......... planted ground truth + 2 decoys + 1 unexplainable shock
S2  Semantic layer ........ contract, grain reconciliation, freshness, conflict checks
S3  Detection ............. forecast baseline → conformal+ACI intervals → EVT p-value
                            → Benjamini-Hochberg FDR → dual materiality
S4a Attribution ........... exact Shapley over dimensions, price-volume-mix, funnel
S4b Causal prosecution .... precedence → DiD → synthetic control → placebo → dose-response
                            → refutation suite
S5  Abstention ............ 7 triggers, each with a discriminating test
S6  Decision .............. constraint-state value function → MILP portfolio, budget-bound
S7  Narrative ............. persona rendering + RLS + numeric grounding validator
S8  Telemetry ............. latency, model calls, tokens, cost per insight
S9  Method registry ....... every step mapped to its technique, with justification
```

---

## Results from the reference run

**Detection**
- 2,016 slice × KPI tests → 338 naively significant → 272 survive FDR → **115 material**
- Empirical coverage **0.898** against a 0.90 target, on **52,080 held-out residuals**

**Causal prosecution** — 3 candidates, 1 survives, and the two decoys die by *different* mechanisms:

| Candidate | Verdict | Killed by |
|---|---|---|
| E1 Logistics vendor switch | **accepted** | survived all 6 tests |
| E2 Price increase +3% | rejected | temporal precedence — cause post-dates its effect |
| E3 Competitor sale | rejected | dose-response inverted — exposure highest where revenue held up |

Accepted effect: **+10.07pp** on cancellation rate, worth **₹42.6 L/week** [₹36.5 L, ₹48.8 L].

**Decision** — the whole thesis, in one comparison:

- Naive "repair everything diagnosed" (L1+L2+L3): **₹95 L** → over budget, infeasible
- Fulcrum's choice: **₹67 L**, recovering **₹35.5 L/week**, relieving 4 constraints

The budget slider changes the recommendation's *shape*, not just its size:

| Budget | Chosen | Family |
|---|---|---|
| ₹5 L | do nothing | — |
| ₹15 L | L4 | repair |
| ₹45 L | L1+L7 | repair + goal-modifying |
| **₹56 L** | **L5** | **structural** |
| ₹70 L | L4+L5 | repair + structural |

---

## Three design decisions worth defending

**1. Value belongs to constraint states, not to actions.**
An action is a priced way of relieving a *set* of constraints; `V(S)` gates each constraint on its prerequisites. Redundancy (union, not sum) and complementarity (gated to zero) then fall out of the definition rather than being special-cased. `V({listing_accuracy})` = ₹0 while deliveries are still failing; `V({delivery, listing})` exceeds the sum of its parts. This is also why a *structural* move — "exit the channel" — is reachable at all: it is not the repair of any diagnosed cause, so a diagnosis-driven recommender can never propose it.

**2. The LLM does no arithmetic and has no database access.**
It receives a finished evidence object and writes prose. A validator then extracts every numeral from the output and asserts it appears in the evidence; ungrounded text is **blocked**, not warned about. The guarantee lives in the validator rather than the prompt, so swapping the deterministic renderer for a frontier model cannot weaken it. Negative test included in the run.

**3. Abstention is a policy, not a fallback.**
Seven enumerated triggers, each with a *discriminating test* naming the data that would resolve it. The engine says "₹X L/week is unexplained, and here is the log that would settle it" rather than stretching an accepted cause to cover the whole gap.

---

## Bugs found and fixed during the build

Kept deliberately, because each is a trap a reviewer may probe:

- **AR(1) baselines absorb sustained level shifts.** An autoregressive term learns the broken state within one step and reports no anomaly. The event window is now held out of the fit.
- **Conformal p-values are floored at 1/(n+1).** At ~0.016 with a 60-point calibration window, *no* finding could ever clear a BH threshold across 2,016 tests — multiplicity control would be silently vacuous. The interval still carries the coverage guarantee; a peaks-over-threshold GPD tail supplies a p-value that can go small.
- **Sparse slices produced *narrower* intervals.** Short, low-variance histories yield small residuals, so the least trustworthy slices looked the most certain. Fixed with the finite-sample conformal quantile `ceil((n+1)(1-α))/n`; where that exceeds `n`, no valid interval exists and the engine says so.
- **Coverage measured on its own calibration set is circular.** Now a genuine holdout: quantile from the first half, coverage measured on the untouched second half.
- **Global binary segmentation is the wrong changepoint tool** for onset detection — it returns the most *balanced* split. A shift planted at 2026-07-27 was located at 2024-09-15. Replaced with out-of-control onset detection.
- **Mapping every discovered topic to one causal node** let an unrelated "app / crash / login" cluster inherit the real cause's signal. Topics now map to distinct nodes and face the ontology filter.
- **Corroboration was being counted as a rival hypothesis.** A delivery-complaint spike sits *on* E1's causal path, so it is evidence for E1, not a competing cause. Merging it removed a spurious second acceptance.

---

## Honest limitations

- **Synthetic data.** Public retail datasets carry no labelled causal ground truth, so a causal engine cannot be *validated* on them — only demonstrated. The generator plants a known chain plus two decoys specifically so every rejection is checkable. The trade-off is that the data is kinder than production.
- **The constraint value function is partly declared, not learned.** `blocked_value` per constraint is elicited in `contracts/levers.yaml`. In production it comes from the causal layer's CATE estimates where data allows and from domain experts where it does not. Pretending it is fully learned would be false.
- **The renderer is templated by default.** An `LLMRenderer` sits behind the same interface and the same validator, but the demo defaults to deterministic output so runs are reproducible and free.
- **TF-IDF + KMeans stands in for BERTopic.** Short templated ticket text separates poorly; production would use embeddings → UMAP → HDBSCAN. Disclosed in the method registry rather than hidden.
- **No sequential/RL policy.** The MILP solves the single-period budget problem. The contextual-bandit-with-knapsacks extension is roadmap, not deliverable.
- **Detection is ~34 s for 2,016 series.** Fine for a nightly batch; the interactive path would need caching and incremental refits.

---

## Layout

```
contracts/kpis.yaml        KPI semantic contract + causal DAG
contracts/levers.yaml      constraints, levers, objective, envelope, personas
fulcrum/datagen.py         S1  scenario data with planted ground truth
fulcrum/semantic.py        S2  contracts, grain reconciliation, freshness
fulcrum/detection_core.py  S3  conformal / ACI / EVT / BH numerics
fulcrum/detection.py       S3  detection pipeline
fulcrum/attribution.py     S4a Shapley, price-volume-mix, funnel
fulcrum/causal.py          S4b hypotheses, DiD, synthetic control, refuters
fulcrum/abstention.py      S5  confidence and abstention policy
fulcrum/decision.py        S6  constraint-state value, MILP portfolio
fulcrum/narrative.py       S7  personas, RLS, numeric grounding validator
fulcrum/telemetry.py       S8  latency, tokens, cost
fulcrum/registry.py        S9  method attribution
run_demo.py                end-to-end orchestrator
test_fulcrum.py            property tests for the load-bearing claims
```
