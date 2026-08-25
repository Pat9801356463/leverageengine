"""
S9 - Method attribution registry.

Round-2 requirement, quoted: "The LLM should not be treated as the source of
quantitative truth. Teams should explicitly demonstrate when they use
deterministic logic, SQL, business rules, statistics, traditional ML, causal
inference, retrieval or LLMs - and why."

This registry is that answer as a machine-readable artefact rather than a
bullet on a slide. It is also honest about the two places where the prototype
substitutes a lighter method for the production choice.
"""

from __future__ import annotations

import pandas as pd

# category | method | why this and not something else | prod substitution
REGISTRY = [
    dict(phase="0 Frame", step="KPI semantic contract",
         category="business_rules", method="Declarative YAML contract",
         why="A metric definition is a governance artefact, not something to infer. "
             "Learning what 'revenue' means would be both wrong and unauditable.",
         substitution=""),
    dict(phase="0 Frame", step="Causal DAG",
         category="business_rules", method="Curated, human-approved edge list",
         why="Structure-learning algorithms (PC, NOTEARS) may PROPOSE edges, but a "
             "causal graph asserted from observational business data is "
             "indefensible. The graph is curated; discovery would only suggest.",
         substitution="Production adds PC-algorithm edge proposals for human review."),

    dict(phase="1 Sense", step="Multi-source grain reconciliation",
         category="deterministic", method="Contract-driven aggregation (sum vs "
                                          "weighted mean by declared weight column)",
         why="Ratio KPIs must be weighted, never plain-averaged. This is arithmetic, "
             "not inference; making it statistical would add noise to an identity.",
         substitution=""),
    dict(phase="1 Sense", step="Freshness and lineage stamping",
         category="deterministic", method="Ingest-time metadata propagation",
         why="Enables a claim to be REFUSED on staleness grounds. No model needed.",
         substitution=""),

    dict(phase="2 Detect", step="Baseline expectation",
         category="statistics", method="Trend + annual Fourier harmonics, ridge-fitted, "
                                       "event window held out",
         why="Comparing to last week confuses seasonality with news. No AR term: an "
             "AR baseline absorbs a sustained level shift within one step and reports "
             "no anomaly - the commonest way a detector misses a real regime change.",
         substitution="Production: pooled Temporal Fusion Transformer across slices, "
                      "which also yields the donor-similarity matrix for S4."),
    dict(phase="2 Detect", step="Prediction intervals",
         category="statistics", method="Split conformal + Adaptive Conformal Inference",
         why="Distribution-free coverage without assuming Gaussian, homoscedastic "
             "residuals. ACI chosen over EnbPI/SPCI on published benchmark evidence "
             "that ACI attains nominal coverage where those under-cover; it is also "
             "far cheaper (no bootstrap ensemble).",
         substitution=""),
    dict(phase="2 Detect", step="Ranking p-value",
         category="statistics", method="Peaks-over-threshold GPD tail extrapolation",
         why="Conformal p-values are floored at 1/(n+1) (~0.016 here), so no finding "
             "could ever clear a BH threshold across ~1,700 tests - multiplicity "
             "control would be vacuous. The conformal INTERVAL remains the decision "
             "boundary; EVT supplies a p-value that can go small.",
         substitution=""),
    dict(phase="2 Detect", step="Multiplicity control",
         category="statistics", method="Benjamini-Hochberg FDR",
         why="~1,700 simultaneous tests at alpha=0.05 manufacture ~85 false alarms by "
             "construction. FDR is the right error rate for a ranked worklist; "
             "family-wise control would be far too conservative here.",
         substitution=""),
    dict(phase="2 Detect", step="Materiality",
         category="business_rules", method="Statistical significance AND contract "
                                           "business-impact floor",
         why="A statistically bulletproof Rs 40,000 deviation is not material. Ranking "
             "by p-value alone is the classic failure of automated anomaly detection.",
         substitution=""),

    dict(phase="3 Localise", step="Dimension attribution",
         category="statistics", method="Exact Shapley over 2^4 coalitions",
         why="Region, product, channel and segment overlap; the same rupee of "
             "shortfall sits inside several at once. Shapley is the unique attribution "
             "with efficiency + symmetry + null-player, so credit is assigned once. "
             "Exact enumeration - no sampling error to defend.",
         substitution=""),
    dict(phase="3 Localise", step="Price-volume-mix bridge",
         category="deterministic", method="Accounting identity",
         why="revenue = units x ASP is exact by definition. Modelling it would add "
             "noise to something that is true.",
         substitution=""),
    dict(phase="3 Localise", step="Funnel split",
         category="deterministic", method="net = gross x (1 - cancel rate)",
         why="Separates 'fewer people ordered' from 'people ordered then walked away'. "
             "Different remedies; no aggregate KPI distinguishes them.",
         substitution=""),

    dict(phase="4 Hypothesise", step="Topic discovery on ticket text",
         category="traditional_ml", method="TF-IDF + KMeans + top-term naming",
         why="Unsupervised recovery of complaint themes, then a spike test inside the "
             "affected slice and window. Topics are mapped to DISTINCT causal-DAG "
             "nodes - mapping them all to one node lets an unrelated cluster inherit "
             "the real cause's signal.",
         substitution="Production: BERTopic (embeddings -> UMAP -> HDBSCAN -> c-TF-IDF), "
                      "which separates short templated text far better."),
    dict(phase="4 Hypothesise", step="Ontology filter",
         category="business_rules", method="Reachability query on the approved DAG",
         why="A spike in app-crash tickets is real but has no declared path to revenue. "
             "Filtering here stops the causal tests from ever running on it.",
         substitution=""),

    dict(phase="5 Prosecute", step="Temporal precedence",
         category="causal_inference", method="Out-of-control onset detection vs event date",
         why="If the effect began before the cause, the candidate is dead regardless of "
             "correlation strength. Cheapest test, so it runs first. Note: global "
             "binary segmentation is the WRONG tool - it returns the most balanced "
             "split, not the most recent onset.",
         substitution=""),
    dict(phase="5 Prosecute", step="Difference-in-differences",
         category="causal_inference", method="Two-way fixed effects + parallel-trends check",
         why="Differences out both unit-level levels and shocks common to all units. "
             "Parallel trends is CHECKED on the pre-period, not assumed.",
         substitution="Production: DoWhy / EconML for heterogeneous effects."),
    dict(phase="5 Prosecute", step="Synthetic control",
         category="causal_inference", method="Abadie SCM, non-negative weights summing to 1",
         why="Builds a counterfactual that tracks the treated unit pre-event. Convex "
             "weights prevent extrapolation outside the donor hull.",
         substitution="Production: CausalImpact (Bayesian structural time series) for "
                      "posterior intervals."),
    dict(phase="5 Prosecute", step="Placebo inference",
         category="causal_inference", method="Permutation over donor units",
         why="Exact permutation p-value AND the placebo-in-space refuter in one pass. "
             "If untreated units show comparable gaps, there is nothing to explain.",
         substitution=""),
    dict(phase="5 Prosecute", step="Refutation suite",
         category="causal_inference", method="Placebo-in-time, random common cause, "
                                             "leave-donors-out, dose-response",
         why="The engine attacks its own finding before shipping it. This is the layer "
             "that separates a causal claim from a correlation.",
         substitution="Production: DoWhy's built-in refuters, same tests."),

    dict(phase="6-7 Decide", step="Constraint-state value function",
         category="deterministic", method="V(S) with prerequisite gating",
         why="Actions have no intrinsic value; constraint STATES do. Redundancy "
             "(union not sum) and complementarity (gated to zero) fall out of the "
             "definition instead of being special-cased.",
         substitution=""),
    dict(phase="6-7 Decide", step="Portfolio selection",
         category="optimisation", method="Exhaustive enumeration cross-checked against "
                                         "MILP (PuLP/CBC)",
         why="Exhaustive is exact and fully transparent at this size; the MILP is the "
             "scalable path. Running both and asserting agreement is a cheap "
             "correctness check on the model itself.",
         substitution="Production: contextual bandit with knapsack constraints for the "
                      "sequential budget problem."),
    dict(phase="6-7 Decide", step="Decision rights",
         category="business_rules", method="Authority-level filter on the lever catalogue",
         why="A persona is only offered levers it can actually authorise; the rest are "
             "routed to their owners.",
         substitution=""),

    dict(phase="8 Narrate", step="Prose generation",
         category="llm", method="Templated renderer (LLM behind the same interface)",
         why="The ONLY generative step. It receives a finished evidence object and does "
             "no arithmetic and no retrieval.",
         substitution="Production: frontier LLM, low temperature, schema-constrained. "
                      "Prototype defaults to templates for reproducibility."),
    dict(phase="8 Narrate", step="Numeric grounding validator",
         category="deterministic", method="Numeral extraction + set membership check",
         why="Every number in the output must exist in the evidence object or the "
             "artefact is blocked. ~40 lines; closes the hallucinated-statistic "
             "failure mode. Lives in the validator, not the prompt, so swapping "
             "renderers cannot weaken it.",
         substitution="Production adds an NLI entailment check (AlignScore/SummaC) for "
                      "non-numeric claims."),
    dict(phase="8 Narrate", step="Row-level security",
         category="business_rules", method="Evidence-level scrub before rendering",
         why="Filtering the query alone is insufficient - comparison regions and donor "
             "weights carry other units' figures into the evidence object.",
         substitution=""),

    dict(phase="9 Learn", step="Abstention policy",
         category="business_rules", method="Enumerated triggers with discriminating tests",
         why="Abstention is a policy with defined conditions, not a fallback. Each "
             "refusal names the data that would resolve it.",
         substitution=""),
    dict(phase="9 Learn", step="Calibration tracking",
         category="statistics", method="Empirical coverage on out-of-sample residuals",
         why="Calibration is demonstrated, not asserted. In-sample residuals would make "
             "the coverage guarantee circular.",
         substitution="Production adds reliability diagrams, Brier score and ECE over "
                      "the outcome ledger."),
]


def registry_frame() -> pd.DataFrame:
    return pd.DataFrame(REGISTRY)


def llm_vs_non_llm() -> dict:
    df = registry_frame()
    n = len(df)
    llm = int((df["category"] == "llm").sum())
    return {
        "total_pipeline_steps": n,
        "llm_steps": llm,
        "non_llm_steps": n - llm,
        "llm_share_pct": round(100 * llm / n, 1),
        "by_category": df["category"].value_counts().to_dict(),
        "llm_scope": ("Prose rendering only. The LLM performs no arithmetic, has no "
                      "database access, and every numeral it emits is validated against "
                      "a pre-computed evidence object before the artefact is released."),
        "prototype_substitutions": int((df["substitution"] != "").sum()),
    }
