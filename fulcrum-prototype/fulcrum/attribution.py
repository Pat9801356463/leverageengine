"""
S4a - Attribution: WHERE did the movement happen, and through which term?

Method choices and why:

* SHAPLEY over dimensions - region, product, channel and segment overlap.
  Naively reporting "the biggest sub-group" double-counts, because the same
  rupee of shortfall sits inside North, inside LargeAppliances and inside
  Marketplace simultaneously. Shapley values are the unique attribution
  satisfying efficiency (contributions sum to the total), symmetry and the
  null-player property, so overlapping dimensions get credit exactly once.
  With 4 dimensions we enumerate all 16 coalitions EXACTLY - no sampling,
  no approximation error to defend.

* PRICE-VOLUME-MIX - deterministic accounting identity, NOT a model.
  revenue = units x ASP is exact. Making this statistical would add noise
  to something that is true by definition. Flagged as `deterministic` in
  the method registry for exactly this reason.

* FUNNEL split - net_units = gross_orders x (1 - cancellation_rate).
  This is the step that distinguishes "customers stopped buying" from
  "customers bought and then walked away". Those have completely different
  remedies, and no aggregate KPI can tell them apart.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

DIMS = ["region", "product_line", "channel", "segment"]


@dataclass
class AttributionResult:
    total_gap: float
    leaf_contributions: pd.DataFrame
    shapley_dimensions: dict[str, float]
    concentration: dict
    pvm: dict
    funnel: dict


def _forecast_leaves(panel: pd.DataFrame, analysis_week: pd.Timestamp,
                     kpi: str, lookback: int = 8) -> pd.DataFrame:
    """Cheap per-leaf expectation for gap attribution.

    We reuse a seasonal-naive-plus-drift expectation here rather than
    re-running the full detection model on every leaf: attribution needs a
    consistent decomposition of the aggregate gap, and the aggregate gap is
    already anchored by the detection layer. Using the heavy model per leaf
    would be ~150x more compute for a second-order change in the split.
    """
    hist = panel[panel["week"] < analysis_week]
    cur = panel[panel["week"] == analysis_week]

    recent = (hist[hist["week"] >= analysis_week - pd.Timedelta(weeks=lookback)]
              .groupby(DIMS, as_index=False)[kpi].mean()
              .rename(columns={kpi: "recent_mean"}))
    yoy = (hist[(hist["week"] >= analysis_week - pd.Timedelta(weeks=53)) &
                (hist["week"] <= analysis_week - pd.Timedelta(weeks=51))]
           .groupby(DIMS, as_index=False)[kpi].mean()
           .rename(columns={kpi: "yoy_mean"}))
    yoy_recent = (hist[(hist["week"] >= analysis_week - pd.Timedelta(weeks=53 + lookback)) &
                       (hist["week"] <= analysis_week - pd.Timedelta(weeks=53))]
                  .groupby(DIMS, as_index=False)[kpi].mean()
                  .rename(columns={kpi: "yoy_recent_mean"}))

    out = (cur[DIMS + [kpi]].rename(columns={kpi: "actual"})
           .merge(recent, on=DIMS, how="left")
           .merge(yoy, on=DIMS, how="left")
           .merge(yoy_recent, on=DIMS, how="left"))

    # seasonal factor = how this week compares to the trailing window, a year ago
    with np.errstate(divide="ignore", invalid="ignore"):
        seas = out["yoy_mean"] / out["yoy_recent_mean"]
    seas = pd.Series(seas).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.5, 2.0)
    out["expected"] = out["recent_mean"] * seas
    out["gap"] = out["actual"] - out["expected"]
    return out.dropna(subset=["expected"])


def _explained_fraction(leaves: pd.DataFrame, subset: tuple[str, ...]) -> float:
    """Value function for the Shapley game.

    v(S) = fraction of total squared gap captured by the group means when the
    data is partitioned by dimension subset S. v({}) = 0 by construction.
    This measures how much LOCALISATION POWER each dimension provides.
    """
    g = leaves["gap"].to_numpy()
    denom = float(np.sum(g ** 2))
    if denom <= 0:
        return 0.0
    if not subset:
        return 0.0
    grp = leaves.groupby(list(subset))["gap"].transform("mean").to_numpy()
    return float(np.sum(grp ** 2) / denom)


def shapley_dimensions(leaves: pd.DataFrame, dims: list[str] = None) -> dict[str, float]:
    """Exact Shapley values over 2^|dims| coalitions."""
    dims = dims or DIMS
    n = len(dims)
    v: dict[tuple, float] = {}
    for r in range(n + 1):
        for sub in itertools.combinations(dims, r):
            v[tuple(sorted(sub))] = _explained_fraction(leaves, sub)

    from math import factorial
    phi = {d: 0.0 for d in dims}
    for d in dims:
        others = [x for x in dims if x != d]
        for r in range(len(others) + 1):
            for sub in itertools.combinations(others, r):
                s = tuple(sorted(sub))
                s_with = tuple(sorted(s + (d,)))
                w = factorial(len(s)) * factorial(n - len(s) - 1) / factorial(n)
                phi[d] += w * (v[s_with] - v[s])
    total = sum(phi.values()) or 1.0
    return {d: round(100 * phi[d] / total, 2) for d in dims}


def price_volume_mix(panel: pd.DataFrame, slice_spec: dict,
                     analysis_week: pd.Timestamp, base_weeks: int = 8) -> dict:
    """Exact three-way decomposition of a revenue change.

    Volume: what if units changed but every price and the mix stayed put?
    Price:  what if prices changed but units did not?
    Mix:    what if the SHARE of expensive vs cheap products shifted?

    These are different problems with different fixes, which is why the split
    matters more than the aggregate.
    """
    m = pd.Series(True, index=panel.index)
    for d, val in slice_spec.items():
        m &= (panel[d] == val)
    sub = panel.loc[m]

    cur = sub[sub["week"] == analysis_week]
    base = sub[(sub["week"] < analysis_week) &
               (sub["week"] >= analysis_week - pd.Timedelta(weeks=base_weeks))]
    if cur.empty or base.empty:
        return {"available": False}

    key = "product_line"
    # Baseline must be the AVERAGE WEEKLY TOTAL, not the average of leaf rows.
    # Mixing per-row means with cross-row sums silently rescales the baseline
    # by the number of leaves and produces nonsense growth rates.
    bw = base.groupby(["week", key], as_index=False).agg(
        u=("net_units", "sum"), r=("revenue", "sum"))
    b = bw.groupby(key, as_index=False).agg(u=("u", "mean"), r=("r", "mean"))
    b["p"] = b["r"] / b["u"].replace(0, np.nan)
    c = cur.groupby(key, as_index=False).agg(u=("net_units", "sum"), r=("revenue", "sum"))
    c["p"] = c["r"] / c["u"].replace(0, np.nan)

    j = b.merge(c, on=key, how="outer", suffixes=("0", "1")).fillna(0.0)
    U0, U1 = j["u0"].sum(), j["u1"].sum()
    R0, R1 = j["r0"].sum(), j["r1"].sum()
    P0 = R0 / U0 if U0 else 0.0

    growth = (U1 / U0) if U0 else 1.0
    volume = (U1 - U0) * P0
    mix = float(np.sum((j["u1"] - j["u0"] * growth) * (j["p0"] - P0)))
    price = float(np.sum(j["u1"] * (j["p1"] - j["p0"])))
    total = R1 - R0
    residual = total - (volume + mix + price)

    return {
        "available": True,
        "baseline_revenue": float(R0), "current_revenue": float(R1),
        "total_change": float(total),
        "volume_effect": float(volume),
        "price_effect": float(price),
        "mix_effect": float(mix),
        "residual": float(residual),
        "dominant_term": max(
            [("volume", abs(volume)), ("price", abs(price)), ("mix", abs(mix))],
            key=lambda x: x[1])[0],
        "units_change_pct": float(100 * (U1 / U0 - 1)) if U0 else np.nan,
        "asp_change_pct": float(100 * ((R1 / U1) / (R0 / U0) - 1)) if U0 and U1 else np.nan,
    }


def funnel_split(panel: pd.DataFrame, slice_spec: dict,
                 analysis_week: pd.Timestamp, base_weeks: int = 8) -> dict:
    """net_units = gross_orders x (1 - cancellation_rate).

    Separates 'fewer people ordered' from 'people ordered and cancelled'.
    """
    m = pd.Series(True, index=panel.index)
    for d, val in slice_spec.items():
        m &= (panel[d] == val)
    sub = panel.loc[m]

    cur = sub[sub["week"] == analysis_week]
    base = sub[(sub["week"] < analysis_week) &
               (sub["week"] >= analysis_week - pd.Timedelta(weeks=base_weeks))]
    if cur.empty or base.empty:
        return {"available": False}

    g1 = float(cur["gross_orders"].sum())
    g0 = float(base.groupby("week")["gross_orders"].sum().mean())
    c1 = float(cur["cancellations"].sum() / max(g1, 1e-9))
    c0 = float(base["cancellations"].sum() / max(base["gross_orders"].sum(), 1e-9))

    n0, n1 = g0 * (1 - c0), g1 * (1 - c1)
    orders_effect = (g1 - g0) * (1 - c0)
    cancel_effect = g0 * (-(c1 - c0))
    interaction = (n1 - n0) - orders_effect - cancel_effect

    return {
        "available": True,
        "gross_orders_baseline": g0, "gross_orders_current": g1,
        "gross_orders_change_pct": 100 * (g1 / g0 - 1) if g0 else np.nan,
        "cancellation_rate_baseline": c0, "cancellation_rate_current": c1,
        "cancellation_rate_change_pp": 100 * (c1 - c0),
        "net_units_change": n1 - n0,
        "orders_effect": orders_effect,
        "cancellation_effect": cancel_effect,
        "interaction": interaction,
        "dominant_term": "cancellation" if abs(cancel_effect) > abs(orders_effect) else "orders",
    }


def attribute(panel: pd.DataFrame, analysis_week: pd.Timestamp,
              kpi: str = "revenue", top_n: int = 10) -> AttributionResult:
    leaves = _forecast_leaves(panel, analysis_week, kpi)
    total_gap = float(leaves["gap"].sum())

    leaves = leaves.copy()
    leaves["share_of_gap_pct"] = 100 * leaves["gap"] / total_gap if total_gap else np.nan
    top = (leaves.reindex(leaves["gap"].abs().sort_values(ascending=False).index)
                 .head(top_n)[DIMS + ["actual", "expected", "gap", "share_of_gap_pct"]])

    phi = shapley_dimensions(leaves)

    neg = leaves[leaves["gap"] < 0].sort_values("gap")
    cum = neg["gap"].cumsum() / neg["gap"].sum() if len(neg) else pd.Series(dtype=float)
    n_for_80 = int((cum <= 0.80).sum() + 1) if len(cum) else 0

    # Best slice rule = the MOST SPECIFIC rule that still carries a large
    # share of the shortfall. Maximising share alone always returns the
    # shallowest rule ("region = North"), which is true but not actionable;
    # a fix is scoped to a channel and a product line, not to a region.
    RETENTION = 0.45
    candidates = []
    for r in range(1, len(DIMS) + 1):
        for sub in itertools.combinations(DIMS, r):
            grp = leaves.groupby(list(sub))["gap"].sum()
            worst = grp.idxmin()
            share = float(grp.min() / total_gap) if total_gap else 0.0
            key = worst if isinstance(worst, tuple) else (worst,)
            candidates.append((r, share, dict(zip(sub, key))))

    deep = [c for c in candidates if c[1] >= RETENTION]
    if deep:
        best_depth = max(c[0] for c in deep)
        _, best_share, best_rule = max(
            [c for c in deep if c[0] == best_depth], key=lambda x: x[1])
    else:
        _, best_share, best_rule = max(candidates, key=lambda x: x[1])

    return AttributionResult(
        total_gap=total_gap,
        leaf_contributions=top,
        shapley_dimensions=phi,
        concentration={
            "n_negative_leaves": int((leaves["gap"] < 0).sum()),
            "n_leaves_for_80pct_of_shortfall": n_for_80,
            "best_slice_rule": best_rule,
            "best_slice_share_of_gap_pct": round(100 * best_share, 2),
        },
        pvm=price_volume_mix(panel, best_rule or {}, analysis_week),
        funnel=funnel_split(panel, best_rule or {}, analysis_week),
    )
