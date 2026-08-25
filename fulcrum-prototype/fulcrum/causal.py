"""
S4b - Causal prosecution: which candidate explanation actually survives?

This is the layer that separates Fulcrum from "an LLM looked at a chart".
Everything above produces CANDIDATES. Nothing here is accepted until it has
survived an attempt to destroy it.

Methods and why:

* TEMPORAL PRECEDENCE (cheap, run first). If the effect began before the
  candidate cause, the candidate is dead regardless of correlation strength.
  Changepoint located by binary segmentation on the mechanism KPI.

* DIFFERENCE-IN-DIFFERENCES, two-way fixed effects. Compares the treated
  unit's before/after change against untreated units' before/after change,
  differencing out both unit-level level differences and any shock common
  to all units. Parallel-trends is CHECKED on the pre-period, not assumed.

* SYNTHETIC CONTROL (Abadie). Builds a weighted combination of donor units
  that tracks the treated unit before the event, then measures the post-event
  gap. Weights are non-negative and sum to one - no extrapolation outside the
  donor convex hull.

* PLACEBO-BASED INFERENCE. Standard SCM inference: re-run the whole procedure
  pretending each donor was treated. If the true treated unit's gap is not
  extreme within that placebo distribution, there is no effect. This yields an
  exact permutation p-value AND doubles as the placebo-in-space refuter.

* REFUTATION SUITE (the "engine that argues with itself"):
    - placebo in time     : shift the intervention into the quiet pre-period
    - placebo in space    : treat an untreated unit as treated
    - random common cause : add an irrelevant covariate, check stability
    - leave-donors-out    : drop 30% of donors, check stability
    - dose-response       : does effect size scale with exposure intensity?
  A candidate that fails any of these is rejected with the reason recorded.

* TOPIC DISCOVERY on ticket text uses TF-IDF + KMeans + top-term naming.
  In production this is BERTopic (embeddings -> UMAP -> HDBSCAN -> c-TF-IDF);
  the substitution is recorded in the method registry rather than hidden.
  What matters for the argument is the same either way: a topic whose volume
  SPIKES inside the affected slice and window becomes a candidate cause.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .semantic import SemanticLayer


# =====================================================================
# Hypotheses
# =====================================================================

@dataclass
class Hypothesis:
    hid: str
    label: str
    source: str                     # event_log | ticket_topics | external_feed
    proposed_cause_node: str        # node in the causal DAG
    start: pd.Timestamp
    end: pd.Timestamp | None
    scope: dict
    dag_path: list[str] | None = None
    notes: str = ""

    # populated by prosecution
    verdict: str = "untested"       # accepted | rejected | inconclusive
    rejection_reason: str = ""
    tests: dict = field(default_factory=dict)
    effect: dict = field(default_factory=dict)


def _changepoint(y: np.ndarray, baseline_weeks: int = 26,
                 guard_weeks: int = 2, k_sigma: float = 3.0,
                 persist: int = 2) -> int:
    """Locate the ONSET of the most recent sustained level shift.

    NOT a global binary segmentation. Maximising a two-sample t statistic
    over the whole history returns the most BALANCED split, which for a
    shock confined to the last few weeks is systematically wrong - the tiny
    post segment inflates the standard error and a mild mid-series drift
    wins instead. We observed exactly that failure: a shift planted at
    2026-07-27 was located at 2024-09-15.

    Instead: establish a quiet baseline, then walk forward and take the
    first week that breaches the control limit AND stays breached. This is
    a standard out-of-control onset rule and it answers the question the
    precedence test actually asks - "when did this start?" - rather than
    "where does this series split best?".
    """
    n = len(y)
    if n < baseline_weeks + guard_weeks + persist:
        baseline_weeks = max(n // 3, 5)

    end_base = n - guard_weeks - persist
    start_base = max(end_base - baseline_weeks, 0)
    base = y[start_base:end_base]
    if len(base) < 5:
        return -1

    mu, sd = float(np.mean(base)), float(np.std(base, ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return -1
    hi, lo = mu + k_sigma * sd, mu - k_sigma * sd

    for i in range(start_base, n - persist + 1):
        seg = y[i:i + persist]
        if np.all(seg > hi) or np.all(seg < lo):
            return i
    return -1


def generate_hypotheses(
    sources: dict[str, pd.DataFrame],
    sem: SemanticLayer,
    slice_spec: dict,
    analysis_week: pd.Timestamp,
    target_node: str = "revenue",
    window_weeks: int = 6,
) -> list[Hypothesis]:
    """Candidates from three streams, then filtered by the causal ontology."""
    win_start = analysis_week - pd.Timedelta(weeks=window_weeks)
    out: list[Hypothesis] = []

    # ---- stream 1: internal event log ------------------------------
    node_map = {"E1": "logistics_vendor", "E2": "list_price", "E3": "competitor_promo"}
    for e in sources["event_log"].itertuples(index=False):
        start = pd.Timestamp(e.start)
        end = pd.Timestamp(e.end) if isinstance(e.end, str) and e.end else None
        if start < win_start - pd.Timedelta(weeks=4) or start > analysis_week:
            continue
        out.append(Hypothesis(
            hid=e.event_id, label=e.label, source="event_log",
            proposed_cause_node=node_map.get(e.event_id, "unknown"),
            start=start, end=end, scope={}, notes=str(e.scope),
        ))

    # ---- stream 2: ticket topic spikes -----------------------------
    out.extend(_topic_hypotheses(sources["support_tickets"], slice_spec,
                                 analysis_week, window_weeks))

    # ---- stream 3: external feed -----------------------------------
    ext = sources.get("external_signals")
    if ext is not None:
        act = ext[(ext["week"] >= win_start) & (ext["competitor_promo_active"])]
        if len(act) and not any(h.hid == "E3" for h in out):
            out.append(Hypothesis(
                hid="X1", label="Competitor promotion detected in external feed",
                source="external_feed", proposed_cause_node="competitor_promo",
                start=act["week"].min(), end=act["week"].max(), scope={},
            ))

    # ---- resolve DAG paths first (needed by the merge below) --------
    for h in out:
        h.dag_path = sem.path_to(h.proposed_cause_node, target_node)

    # ---- merge corroborating evidence into its upstream cause -------
    # A ticket-topic spike is not automatically a rival hypothesis. If its
    # node lies ON an event-log hypothesis's causal path, it is EVIDENCE FOR
    # that hypothesis, not a competitor to it. E1 (logistics_vendor) runs
    # through delivery_sla_pct; a spike in delivery-complaint tickets is
    # therefore corroboration. Treating it as independent double-counts the
    # same mechanism and produces a spurious second "accepted" cause.
    events = [h for h in out if h.source == "event_log"]
    merged: list[Hypothesis] = list(events)
    for h in out:
        if h.source == "event_log":
            continue
        anchor = None
        for e in events:
            path = e.dag_path or []
            if h.proposed_cause_node == e.proposed_cause_node or \
               h.proposed_cause_node in path:
                anchor = e
                break
        if anchor is not None:
            anchor.notes += (f" | CORROBORATED by {h.hid}: {h.label} ({h.notes}); "
                             f"node '{h.proposed_cause_node}' lies on this "
                             f"hypothesis's causal path")
        else:
            merged.append(h)
    out = merged

    # ---- ontology filter -------------------------------------------
    kept = []
    for h in out:
        if h.dag_path is None:
            h.verdict = "rejected"
            h.rejection_reason = (
                f"no declared causal path from '{h.proposed_cause_node}' to "
                f"'{target_node}' in the approved business DAG")
        kept.append(h)
    return kept



# Topic -> causal-DAG node. A discovered topic is only a candidate cause if it
# maps to a node the approved business DAG actually connects to the outcome.
# Assigning every topic to the same node (as a first cut did) lets an unrelated
# cluster - "app / keeps / error / login" - inherit the real cause's causal
# signal and be accepted. The ontology filter is what stops that, and it can
# only work if topics are mapped to DISTINCT nodes.
TOPIC_NODE_KEYWORDS = {
    "delivery_sla_pct": ["delivery", "slot", "schedule", "waiting", "courier",
                         "installation", "arrived", "pushed", "date"],
    "list_price":       ["price", "expensive", "cost", "cheaper", "money", "worth"],
    "listing_accuracy": ["listing", "checkout", "pincode", "shown", "available"],
    # Nodes below are DELIBERATELY absent from the approved DAG - a spike in
    # them is real, but it has no declared path to revenue, so the ontology
    # filter rejects them rather than letting the causal tests be run at all.
    "product_quality":  ["defect", "dent", "damaged", "broken", "flickering", "panel"],
    "payment_friction": ["payment", "refund", "debited", "emi", "transaction", "credited"],
    "app_stability":    ["app", "crash", "crashes", "error", "login", "log"],
    "returns_process":  ["return", "pickup", "returns"],
}


def _topic_to_node(topic_name: str) -> str:
    toks = set(topic_name.replace("/", " ").split())
    best, best_hits = "unmapped_topic", 0
    for node, kws in TOPIC_NODE_KEYWORDS.items():
        hits = len(toks & set(kws))
        if hits > best_hits:
            best, best_hits = node, hits
    return best


def _topic_hypotheses(tickets: pd.DataFrame, slice_spec: dict,
                      analysis_week: pd.Timestamp, window_weeks: int,
                      n_topics: int = 6) -> list[Hypothesis]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    tk = tickets.copy()
    for d, v in slice_spec.items():
        if d in tk.columns:
            tk = tk[tk[d] == v]
    if len(tk) < 60:
        tk = tickets[tickets["region"] == slice_spec.get("region", "North")]
    if len(tk) < 40:
        return []

    vec = TfidfVectorizer(max_features=400, stop_words="english", min_df=3)
    X = vec.fit_transform(tk["ticket_text"])
    km = KMeans(n_clusters=min(n_topics, max(X.shape[0] // 20, 2)),
                n_init=10, random_state=7).fit(X)
    tk = tk.assign(topic=km.labels_)

    terms = np.array(vec.get_feature_names_out())
    names = {}
    for c in range(km.n_clusters):
        top = km.cluster_centers_[c].argsort()[::-1][:4]
        names[c] = " / ".join(terms[top])

    win_start = analysis_week - pd.Timedelta(weeks=window_weeks)
    hyps = []
    all_weeks = pd.date_range(tk["week"].min(), tk["week"].max(), freq="W-SUN")
    for c, grp in tk.groupby("topic"):
        # A week with zero tickets on this topic is a REAL zero, not a gap.
        # Dropping it inflates the pre-period mean and hides genuine spikes.
        wk = grp.groupby("week").size().reindex(all_weeks, fill_value=0)
        post = wk[wk.index >= win_start]
        pre = wk[(wk.index < win_start) &
                 (wk.index >= win_start - pd.Timedelta(weeks=26))]
        if len(pre) < 8 or len(post) < 2:
            continue
        ratio = post.mean() / max(pre.mean(), 0.5)
        if ratio < 2.0:
            continue
        y = pd.concat([pre, post]).sort_index()
        cp = _changepoint(y.to_numpy(dtype=float))
        onset = y.index[cp] if cp > 0 else post.index.min()
        node = _topic_to_node(names[c])
        hyps.append(Hypothesis(
            hid=f"T{c}", label=f"Support topic spike: {names[c]}",
            source="ticket_topics", proposed_cause_node=node,
            start=pd.Timestamp(onset), end=None, scope=dict(slice_spec),
            notes=f"volume x{ratio:.1f} vs prior 26w (pre={pre.mean():.1f}/wk, "
                  f"post={post.mean():.1f}/wk); mapped to DAG node '{node}'",
        ))
    return hyps


# =====================================================================
# Prosecution
# =====================================================================

@dataclass
class CausalConfig:
    pre_weeks: int = 52
    post_weeks: int = 3
    n_placebo_min: int = 4
    stability_tol: float = 0.30       # >30% swing under a refuter = unstable


def _unit_series(panel: pd.DataFrame, kpi: str, slice_spec: dict,
                 unit_dim: str = "region") -> pd.DataFrame:
    """Wide matrix: rows = weeks, cols = units, values = KPI."""
    m = pd.Series(True, index=panel.index)
    for d, v in slice_spec.items():
        if d == unit_dim:
            continue
        m &= (panel[d] == v)
    sub = panel.loc[m]
    if kpi == "cancellation_rate":
        w = (sub.groupby(["week", unit_dim])
                .apply(lambda d: d["cancellations"].sum() /
                       max(d["gross_orders"].sum(), 1e-9), include_groups=False)
                .rename(kpi).reset_index())
    elif kpi in ("revenue", "net_units", "gross_orders", "ticket_volume"):
        w = sub.groupby(["week", unit_dim], as_index=False)[kpi].sum()
    else:
        w = sub.groupby(["week", unit_dim], as_index=False)[kpi].mean()
    return w.pivot(index="week", columns=unit_dim, values=kpi).sort_index()


def _synthetic_control(wide: pd.DataFrame, treated: str,
                       intervention: pd.Timestamp, cfg: CausalConfig) -> dict:
    pre = wide[wide.index < intervention].tail(cfg.pre_weeks)
    post = wide[wide.index >= intervention].head(cfg.post_weeks + 2)
    donors = [c for c in wide.columns if c != treated]
    if len(pre) < 20 or len(post) < 1 or len(donors) < 2:
        return {"available": False, "reason": "insufficient pre-period or donors"}

    Y0, y1 = pre[donors].to_numpy(float), pre[treated].to_numpy(float)
    if np.isnan(Y0).any() or np.isnan(y1).any():
        keep = ~(np.isnan(Y0).any(axis=1) | np.isnan(y1))
        Y0, y1 = Y0[keep], y1[keep]
    if len(y1) < 20:
        return {"available": False, "reason": "insufficient clean pre-period"}

    # scale donors so the fit is about SHAPE not level
    scale = np.where(Y0.mean(axis=0) == 0, 1.0, Y0.mean(axis=0))
    lvl = y1.mean() if y1.mean() != 0 else 1.0

    def loss(w):
        return float(np.mean((y1 / lvl - (Y0 / scale) @ w) ** 2))

    n = len(donors)
    res = optimize.minimize(
        loss, x0=np.full(n, 1 / n), method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 400, "ftol": 1e-10},
    )
    w = np.clip(res.x, 0, None)
    w = w / max(w.sum(), 1e-12)

    def synth(frame):
        return (frame[donors].to_numpy(float) / scale) @ w * lvl

    pre_fit = synth(pre)
    pre_rmse = float(np.sqrt(np.mean((pre[treated].to_numpy(float) - pre_fit) ** 2)))
    post_actual = post[treated].to_numpy(float)
    post_synth = synth(post)
    gap = float(np.mean(post_actual - post_synth))

    return {
        "available": True,
        "weights": {d: round(float(x), 4) for d, x in zip(donors, w) if x > 0.01},
        "pre_rmse": pre_rmse,
        "pre_mean": float(pre[treated].mean()),
        "pre_fit_quality": float(pre_rmse / max(abs(pre[treated].mean()), 1e-9)),
        "post_actual": float(np.mean(post_actual)),
        "post_synthetic": float(np.mean(post_synth)),
        "gap": gap,
        "gap_pct": float(100 * gap / max(abs(np.mean(post_synth)), 1e-9)),
        "rmspe_ratio": float(abs(gap) / max(pre_rmse, 1e-9)),
    }


def _placebo_inference(wide: pd.DataFrame, treated: str,
                       intervention: pd.Timestamp, cfg: CausalConfig) -> dict:
    """Abadie's placebo test: pretend each donor was treated.

    Serves two purposes at once - an exact permutation p-value, and the
    placebo-in-space refuter. If untreated units show gaps as large as the
    treated unit, there is nothing to explain.
    """
    real = _synthetic_control(wide, treated, intervention, cfg)
    if not real.get("available"):
        return {"available": False}

    ratios = []
    for u in wide.columns:
        if u == treated:
            continue
        r = _synthetic_control(wide, u, intervention, cfg)
        if r.get("available") and np.isfinite(r["rmspe_ratio"]):
            ratios.append((u, r["rmspe_ratio"], r["gap_pct"]))

    if len(ratios) < cfg.n_placebo_min:
        return {"available": False, "reason": "too few placebo units"}

    vals = np.array([x[1] for x in ratios])
    rank = 1 + int(np.sum(vals >= real["rmspe_ratio"]))
    p = rank / (len(vals) + 1)

    return {
        "available": True,
        "treated_rmspe_ratio": real["rmspe_ratio"],
        "placebo_rmspe_ratios": {u: round(v, 3) for u, v, _ in ratios},
        "permutation_p_value": float(p),
        "rank": rank, "n_placebo": len(vals),
        "placebo_gap_pcts": {u: round(g, 2) for u, _, g in ratios},
    }


def _did(wide: pd.DataFrame, treated: str, intervention: pd.Timestamp,
         cfg: CausalConfig) -> dict:
    """Two-way fixed effects DiD with a parallel-trends pre-check."""
    long = wide.reset_index().melt(id_vars="week", var_name="unit", value_name="y").dropna()
    long = long[long["week"] >= intervention - pd.Timedelta(weeks=cfg.pre_weeks)]
    long["treat"] = (long["unit"] == treated).astype(float)
    long["post"] = (long["week"] >= intervention).astype(float)
    long["did"] = long["treat"] * long["post"]

    units = pd.get_dummies(long["unit"], drop_first=True, dtype=float)
    times = pd.get_dummies(long["week"], drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(long)), long["did"].to_numpy(),
                         units.to_numpy(), times.to_numpy()])
    y = long["y"].to_numpy(float)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - np.linalg.matrix_rank(X), 1)
    sigma2 = float(resid @ resid / dof)
    try:
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        se = float(np.sqrt(max(cov[1, 1], 0)))
    except Exception:
        se = float("nan")

    est = float(beta[1])
    tstat = est / se if se and np.isfinite(se) and se > 0 else np.nan
    pval = float(2 * (1 - stats.norm.cdf(abs(tstat)))) if np.isfinite(tstat) else np.nan

    # parallel trends: treated-vs-donor gap slope in the pre-period
    pre = wide[wide.index < intervention].tail(cfg.pre_weeks)
    donors = [c for c in wide.columns if c != treated]
    gap_pre = pre[treated] - pre[donors].mean(axis=1)
    tt = np.arange(len(gap_pre), dtype=float)
    if len(tt) > 10 and gap_pre.notna().all():
        sl, _, _, pt, _ = stats.linregress(tt, gap_pre.to_numpy(float))
        parallel_ok = bool(pt > 0.05)
    else:
        sl, pt, parallel_ok = np.nan, np.nan, False

    return {
        "estimate": est, "std_error": se,
        "ci_low": est - 1.96 * se if np.isfinite(se) else np.nan,
        "ci_high": est + 1.96 * se if np.isfinite(se) else np.nan,
        "p_value": pval,
        "parallel_trends_slope": float(sl) if np.isfinite(sl) else np.nan,
        "parallel_trends_p": float(pt) if np.isfinite(pt) else np.nan,
        "parallel_trends_ok": parallel_ok,
        "n_obs": int(len(y)),
    }


def prosecute(
    h: Hypothesis,
    panel: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    slice_spec: dict,
    mechanism_kpi: str,
    outcome_kpi: str,
    cfg: CausalConfig | None = None,
) -> Hypothesis:
    cfg = cfg or CausalConfig()
    if h.verdict == "rejected":
        return h

    wide_mech = _unit_series(panel, mechanism_kpi, slice_spec)
    treated = slice_spec.get("region", "North")
    if treated not in wide_mech.columns:
        h.verdict, h.rejection_reason = "inconclusive", "treated unit not in panel"
        return h

    # ---- TEST 1: temporal precedence -------------------------------
    y = wide_mech[treated].dropna()
    cp = _changepoint(y.to_numpy(float))
    onset = y.index[cp] if cp > 0 else None
    lag_weeks = ((h.start - onset).days / 7.0) if onset is not None else np.nan
    # No slack. On weekly data, a cause beginning after the onset WEEK is
    # post-dating its own effect. A one-week tolerance let a price rise dated
    # 2026-08-03 survive an effect that began 2026-08-02.
    precedence_ok = bool(onset is not None and h.start <= onset)
    h.tests["precedence"] = {
        "effect_onset": str(pd.Timestamp(onset).date()) if onset is not None else None,
        "cause_start": str(h.start.date()),
        "cause_minus_effect_weeks": round(float(lag_weeks), 1) if np.isfinite(lag_weeks) else None,
        "passed": precedence_ok,
    }
    if not precedence_ok:
        h.verdict = "rejected"
        h.rejection_reason = (
            f"temporal precedence violated: the effect began "
            f"{pd.Timestamp(onset).date()} but the cause began {h.start.date()} "
            f"({abs(lag_weeks):.0f} weeks later). A cause cannot post-date its effect.")
        return h

    # ---- TEST 2: difference-in-differences -------------------------
    did = _did(wide_mech, treated, h.start, cfg)
    h.tests["did"] = did
    did_ok = bool(np.isfinite(did["p_value"]) and did["p_value"] < 0.05
                  and did["parallel_trends_ok"])
    if not did_ok:
        reason = ("parallel-trends pre-check failed" if not did["parallel_trends_ok"]
                  else f"DiD estimate not distinguishable from zero (p={did['p_value']:.3f})")
        h.verdict = "rejected"
        h.rejection_reason = (
            f"{reason}: the treated unit's change is not different from what "
            f"comparable untreated units did over the same window.")
        return h

    # ---- TEST 3: synthetic control + placebo inference --------------
    sc = _synthetic_control(wide_mech, treated, h.start, cfg)
    h.tests["synthetic_control"] = sc
    pl = _placebo_inference(wide_mech, treated, h.start, cfg)
    h.tests["placebo_in_space"] = pl
    if pl.get("available") and pl["permutation_p_value"] > 0.20:
        h.verdict = "rejected"
        h.rejection_reason = (
            f"placebo-in-space: {pl['rank']-1} of {pl['n_placebo']} untreated units "
            f"show a gap as large as the treated unit "
            f"(permutation p={pl['permutation_p_value']:.2f}).")
        return h

    # ---- TEST 4: dose-response -------------------------------------
    dr = _dose_response(h, panel, sources, slice_spec, mechanism_kpi, cfg)
    h.tests["dose_response"] = dr
    if dr.get("applicable") and dr.get("inverted"):
        h.verdict = "rejected"
        h.rejection_reason = (
            f"dose-response inverted: exposure is highest in "
            f"{dr['highest_exposure_unit']} (intensity {dr['highest_exposure']:.2f}) "
            f"but the effect is concentrated in {treated} "
            f"(intensity {dr['treated_exposure']:.2f}); correlation between exposure "
            f"and effect is {dr['correlation']:+.2f}.")
        return h

    # ---- TEST 5: refutation suite ----------------------------------
    ref = _refute(wide_mech, treated, h.start, sc, cfg)
    h.tests["refutations"] = ref
    failed = [k for k, v in ref.items() if not v.get("passed", True)]
    if failed:
        h.verdict = "rejected"
        h.rejection_reason = f"failed refutation test(s): {', '.join(failed)}"
        return h

    # ---- accepted: translate mechanism effect to the outcome KPI ----
    h.verdict = "accepted"
    h.effect = _translate_effect(panel, slice_spec, h.start, sc, did,
                                 mechanism_kpi, outcome_kpi, cfg)
    return h


def _dose_response(h, panel, sources, slice_spec, mechanism_kpi, cfg) -> dict:
    ext = sources.get("external_signals")
    if h.proposed_cause_node != "competitor_promo" or ext is None:
        return {"applicable": False}

    exposure = (ext[ext["competitor_promo_active"]]
                .groupby("region")["competitor_promo_intensity"].max())
    if exposure.empty:
        return {"applicable": False}

    wide = _unit_series(panel, mechanism_kpi, slice_spec)
    pre = wide[wide.index < h.start].tail(cfg.pre_weeks)
    post = wide[wide.index >= h.start].head(cfg.post_weeks + 2)
    eff = (post.mean() - pre.mean())

    common = [u for u in exposure.index if u in eff.index]
    if len(common) < 4:
        return {"applicable": False}
    x = exposure.loc[common].to_numpy(float)
    yv = eff.loc[common].to_numpy(float)
    if np.allclose(x.std(), 0):
        return {"applicable": False}
    r = float(np.corrcoef(x, yv)[0, 1])

    treated = slice_spec.get("region", "North")
    hi = exposure.idxmax()
    # for a harmful cause we expect MORE exposure -> MORE (negative) effect.
    inverted = bool(hi != treated and exposure.get(treated, 0) < exposure.max() * 0.85)

    return {
        "applicable": True,
        "exposure_by_unit": {k: round(float(v), 3) for k, v in exposure.items()},
        "effect_by_unit": {k: round(float(eff[k]), 5) for k in common},
        "correlation": r,
        "highest_exposure_unit": str(hi),
        "highest_exposure": float(exposure.max()),
        "treated_exposure": float(exposure.get(treated, np.nan)),
        "inverted": inverted,
    }


def _refute(wide, treated, intervention, sc, cfg) -> dict:
    out = {}
    base = sc.get("gap", np.nan)

    # --- placebo in time: shift the intervention into the quiet pre-period
    fake = intervention - pd.Timedelta(weeks=26)
    sc_t = _synthetic_control(wide[wide.index < intervention], treated, fake, cfg)
    placebo_gap = sc_t.get("gap", np.nan) if sc_t.get("available") else np.nan
    ratio = abs(placebo_gap / base) if np.isfinite(placebo_gap) and base else np.nan
    out["placebo_in_time"] = {
        "fake_intervention": str(fake.date()),
        "placebo_gap": placebo_gap, "real_gap": base,
        "placebo_over_real": round(float(ratio), 3) if np.isfinite(ratio) else None,
        "passed": bool(np.isfinite(ratio) and ratio < 0.35),
        "interpretation": "a period with no intervention must not produce a comparable gap",
    }

    # --- random common cause: add an irrelevant covariate as a donor
    rng = np.random.default_rng(11)
    w2 = wide.copy()
    w2["_noise_unit"] = (wide.mean(axis=1) *
                         (1 + rng.normal(0, 0.05, len(wide))))
    sc_n = _synthetic_control(w2, treated, intervention, cfg)
    g2 = sc_n.get("gap", np.nan)
    swing = abs(g2 - base) / abs(base) if base else np.nan
    out["random_common_cause"] = {
        "gap_with_noise_donor": g2, "relative_swing": round(float(swing), 4)
        if np.isfinite(swing) else None,
        "passed": bool(np.isfinite(swing) and swing < cfg.stability_tol),
        "interpretation": "an irrelevant covariate must not move the estimate",
    }

    # --- leave-donors-out
    donors = [c for c in wide.columns if c != treated]
    rng2 = np.random.default_rng(23)
    swings = []
    for _ in range(5):
        keep = list(rng2.choice(donors, size=max(int(len(donors) * 0.7), 2), replace=False))
        sc_s = _synthetic_control(wide[[treated] + keep], treated, intervention, cfg)
        if sc_s.get("available") and base:
            swings.append(abs(sc_s["gap"] - base) / abs(base))
    ms = float(np.mean(swings)) if swings else np.nan
    out["leave_donors_out"] = {
        "mean_relative_swing": round(ms, 4) if np.isfinite(ms) else None,
        "n_resamples": len(swings),
        "passed": bool(np.isfinite(ms) and ms < cfg.stability_tol),
        "interpretation": "dropping 30% of donors must not change the conclusion",
    }
    return out


def _translate_effect(panel, slice_spec, intervention, sc, did,
                      mechanism_kpi, outcome_kpi, cfg) -> dict:
    """Convert the mechanism-level effect into rupees on the outcome KPI."""
    m = pd.Series(True, index=panel.index)
    for d, v in slice_spec.items():
        m &= (panel[d] == v)
    sub = panel.loc[m]

    pre = sub[(sub["week"] < intervention) &
              (sub["week"] >= intervention - pd.Timedelta(weeks=cfg.pre_weeks))]
    post = sub[sub["week"] >= intervention]

    g_pre = pre.groupby("week")["gross_orders"].sum().mean()
    asp = pre["asp"].mean()
    dc = sc.get("gap", np.nan)            # change in cancellation rate

    revenue_impact = -dc * g_pre * asp if np.isfinite(dc) else np.nan
    lo = -(sc.get("gap", np.nan) + 1.96 * did.get("std_error", 0)) * g_pre * asp
    hi = -(sc.get("gap", np.nan) - 1.96 * did.get("std_error", 0)) * g_pre * asp

    return {
        "mechanism_kpi": mechanism_kpi,
        "mechanism_effect": float(dc) if np.isfinite(dc) else None,
        "mechanism_effect_pp": round(float(dc) * 100, 2) if np.isfinite(dc) else None,
        "outcome_kpi": outcome_kpi,
        "revenue_impact_per_week": float(revenue_impact) if np.isfinite(revenue_impact) else None,
        "revenue_impact_ci": [float(min(lo, hi)), float(max(lo, hi))]
        if np.isfinite(lo) and np.isfinite(hi) else None,
        "basis": "mechanism gap x baseline gross orders x baseline ASP",
    }


def hypothesis_table(hyps: list[Hypothesis]) -> pd.DataFrame:
    return pd.DataFrame([{
        "hid": h.hid, "label": h.label[:58], "source": h.source,
        "cause_node": h.proposed_cause_node,
        "start": str(h.start.date()),
        "verdict": h.verdict,
        "reason": h.rejection_reason[:90],
    } for h in hyps])
