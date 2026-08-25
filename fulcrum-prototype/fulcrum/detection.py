"""
S3 - Detection: is this movement news, or is it noise?

Method choices and WHY (this is the "which technique and why" the Round-2
brief asks teams to make explicit):

1. BASELINE = expectation, not last period.
   Comparing a KPI to last week or last year confuses seasonality with news.
   We fit a per-slice baseline (linear trend + annual Fourier harmonics +
   AR(1)) and test the residual. "Down 8% when you forecast down 9%" is
   not a finding; "down 8% when you forecast down 2%" is.

2. INTERVALS = split conformal + Adaptive Conformal Inference (ACI).
   Chosen over parametric intervals because business KPI residuals are
   heteroscedastic and non-Gaussian. Chosen over EnbPI/SPCI on evidence:
   a 2026 benchmark of conformal algorithms for time-series forecasting
   found ACI and Global-CP reached nominal coverage while EnbPI and SPCI
   under-covered on ARIMA-style forecasters. ACI is also far cheaper -
   no bootstrap ensemble.
   ACI update (Gibbs & Candes 2021):  alpha_{t+1} = alpha_t + gamma*(alpha - err_t)

3. P-VALUES = conformal p-values, not Gaussian tail probabilities.
   The p-value is the rank of the test residual among calibration
   residuals. Distribution-free, and consistent with the interval.

4. MULTIPLICITY = Benjamini-Hochberg FDR.
   ~2,000 slice x KPI tests. At alpha=0.05 that manufactures ~100 false
   alarms by construction. BH controls the expected *proportion* of false
   discoveries, which is the right error rate when the output is a ranked
   worklist rather than a single decision.

5. MATERIALITY = statistical AND business.
   The contract carries a min_business_impact per KPI. A statistically
   bulletproof Rs 40,000 deviation is not material; a marginally
   significant Rs 30L one is. Ranking by p-value alone is the classic
   failure mode of automated anomaly detection.

We also REPORT EMPIRICAL COVERAGE on held-out calibration data, so
calibration is demonstrated rather than asserted.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .semantic import SemanticLayer
from .detection_core import (fit_forecast, aci_alpha, tail_pvalue,
                             benjamini_hochberg, conformal_quantile)

DIMS = ["region", "product_line", "channel", "segment"]

# minimum history before we will attempt a seasonal baseline at all
MIN_HISTORY_FULL = 60
MIN_HISTORY_ANY = 10


@dataclass
class DetectionConfig:
    fourier_k: int = 3
    period: float = 52.0
    calibration_weeks: int = 52
    alpha: float = 0.10            # target miscoverage -> 90% intervals
    aci_gamma: float = 0.02        # ACI learning rate
    fdr_q: float = 0.10
    eval_window: int = 3           # recent weeks held out of the baseline fit


def _slice_masks(df: pd.DataFrame) -> list[tuple[dict, pd.Series]]:
    """Every combination of dimension levels, including partial (aggregate)
    slices. This is the hierarchy the FDR correction is applied across."""
    out = []
    levels = {d: [None] + sorted(df[d].dropna().unique().tolist()) for d in DIMS}
    for combo in itertools.product(*[levels[d] for d in DIMS]):
        spec = {d: v for d, v in zip(DIMS, combo) if v is not None}
        mask = pd.Series(True, index=df.index)
        for d, v in spec.items():
            mask &= (df[d] == v)
        out.append((spec, mask))
    return out


def detect(
    panel: pd.DataFrame,
    sem: SemanticLayer,
    kpis: list[str],
    analysis_week: pd.Timestamp,
    cfg: DetectionConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Returns (findings, diagnostics)."""
    cfg = cfg or DetectionConfig()
    rows = []
    coverage_hits, coverage_n = 0, 0

    slices = _slice_masks(panel)

    for kpi in kpis:
        contract = sem.contract(kpi)
        ok, _ = contract.can_support_weekly()
        if not ok:
            continue

        agg = contract.aggregation
        wcol = contract.weight_by

        for spec, mask in slices:
            sub = panel.loc[mask]
            if sub.empty:
                continue

            if agg == "sum":
                s = sub.groupby("week")[kpi].sum()
            elif agg == "weighted_mean" and wcol and wcol in sub.columns:
                num = sub.groupby("week").apply(
                    lambda d: np.average(d[kpi], weights=np.maximum(d[wcol], 1e-9)),
                    include_groups=False)
                s = num
            else:
                s = sub.groupby("week")[kpi].mean()

            s = s.sort_index().dropna()
            if len(s) < MIN_HISTORY_ANY or s.index.max() != analysis_week:
                continue

            y = s.to_numpy(dtype=float)
            if np.allclose(y.std(), 0):
                continue

            fit = fit_forecast(y, cfg.fourier_k, cfg.period,
                               cfg.eval_window, MIN_HISTORY_FULL)
            yhat, cal, seasonal = fit["yhat"], fit["cal_resid"], fit["seasonal"]
            if len(cal) < 6:
                continue

            a_adapt = aci_alpha(cal, cfg.alpha, cfg.aci_gamma)
            q, interval_valid, interval_note = conformal_quantile(cal, a_adapt)

            # HONEST COVERAGE TEST. Deriving q from the calibration residuals
            # and then checking how many of those same residuals it covers is
            # circular - it returns ceil((n+1)(1-a))/n by construction. Split
            # instead: set the quantile on the first half, measure coverage on
            # the untouched second half. One genuine out-of-sample check per
            # series, ~1,700 series.
            if len(cal) >= 16:
                half = len(cal) // 2
                q_ho, _, _ = conformal_quantile(cal[:half], cfg.alpha)
                if np.isfinite(q_ho):
                    coverage_hits += int(np.sum(np.abs(cal[half:]) <= q_ho))
                    coverage_n += len(cal) - half

            actual, forecast = float(y[-1]), float(yhat[-1])
            lo, hi = forecast - q, forecast + q
            dev = actual - forecast

            # conformal interval = decision boundary; EVT tail = ranking p-value
            score = abs(dev)
            p, p_method = tail_pvalue(cal, score)

            # business impact in the KPI's own units
            imp = contract.min_business_impact()
            if imp is None:
                business_impact, threshold, unit = abs(dev), 0.0, contract.unit
            else:
                unit, threshold = imp
                if unit == "pp":
                    business_impact = abs(dev) * 100
                else:
                    business_impact = abs(dev)

            rows.append({
                "kpi": kpi,
                "slice_spec": spec,
                "slice_label": " / ".join(spec.get(d, "ALL") for d in DIMS),
                "depth": len(spec),
                "n_history": len(y),
                "seasonal_model": seasonal,
                "actual": actual,
                "forecast": forecast,
                "lo": lo, "hi": hi,
                "deviation": dev,
                "deviation_pct": 100 * dev / forecast if forecast else np.nan,
                "wow_pct": 100 * (y[-1] / y[-2] - 1) if len(y) > 1 and y[-2] else np.nan,
                "outside_interval": bool(actual < lo or actual > hi),
                "p_value": float(p),
                "p_method": p_method,
                "business_impact": float(business_impact),
                "impact_unit": unit,
                "impact_threshold": float(threshold),
                "meets_business_threshold": bool(business_impact >= threshold),
                "aci_alpha": round(a_adapt, 4),
                "sparse_history": len(y) < MIN_HISTORY_FULL,
                "oos_calibration": fit["oos_calibration"],
                "interval_valid": interval_valid,
                "interval_note": interval_note,
                "interval_width_pct": float(200 * q / abs(forecast)) if forecast else np.nan,
                "n_calibration": fit["n_cal"],
            })

    findings = pd.DataFrame(rows)
    if findings.empty:
        return findings, {"coverage": None, "n_tests": 0}

    # ---- Benjamini-Hochberg across ALL tests -------------------------
    survives, q_vals = benjamini_hochberg(findings["p_value"].to_numpy(), cfg.fdr_q)
    findings["survives_fdr"] = survives
    findings["q_value"] = q_vals

    findings["material"] = findings["survives_fdr"] & findings["meets_business_threshold"]

    diagnostics = {
        "n_tests": int(len(findings)),
        "n_naive_significant": int((findings["p_value"] <= 0.05).sum()),
        "n_survives_fdr": int(findings["survives_fdr"].sum()),
        "n_material": int(findings["material"].sum()),
        "false_alarms_avoided": int((findings["p_value"] <= 0.05).sum()
                                    - findings["survives_fdr"].sum()),
        "empirical_coverage": round(coverage_hits / max(coverage_n, 1), 4),
        "target_coverage": 1 - cfg.alpha,
        "coverage_n": int(coverage_n),
    }
    return findings, diagnostics
