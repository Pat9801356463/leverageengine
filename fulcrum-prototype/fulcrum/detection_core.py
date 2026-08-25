"""Core numerics for S3 detection, separated so they can be unit-tested."""

from __future__ import annotations

import numpy as np
from scipy import stats


def design(t: np.ndarray, n_ref: int, fourier_k: int, period: float,
           seasonal: bool) -> np.ndarray:
    cols = [np.ones_like(t), t / max(n_ref, 1)]
    if seasonal:
        for k in range(1, fourier_k + 1):
            cols.append(np.sin(2 * np.pi * k * t / period))
            cols.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(cols)


def _ridge(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    lam = 1e-6 * max(np.trace(X.T @ X), 1.0) / X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def fit_forecast(
    y: np.ndarray, fourier_k: int, period: float,
    eval_window: int, min_seasonal: int, cal_frac: float = 0.40,
) -> dict:
    """Two-stage fit.

    Stage A (calibration): fit on the first (1-cal_frac) of the pre-event
      history, predict the remaining cal_frac. Those residuals are genuinely
      OUT-OF-SAMPLE and are what the conformal layer calibrates against.
      Using in-sample residuals here would make the coverage guarantee
      circular - the interval would be validated on data the model had
      already seen.

    Stage B (forecast): refit on ALL pre-event history and forecast across
      the held-out event window. The event window is excluded from the fit
      so a sustained level shift is not absorbed into the baseline.
    """
    n = len(y)
    seasonal = n >= min_seasonal
    t = np.arange(n, dtype=float)
    X = design(t, n, fourier_k, period, seasonal)
    p = X.shape[1]

    n_eval = int(min(eval_window, max(n // 10, 1)))
    n_pre = n - n_eval
    if n_pre < p + 6:
        n_eval = 1
        n_pre = n - 1

    # ---- Stage A: out-of-sample calibration residuals ----------------
    n_cal = int(max(round(n_pre * cal_frac), 8))
    n_cal = min(n_cal, n_pre - (p + 4)) if n_pre - (p + 4) > 0 else 0

    if n_cal >= 8:
        split = n_pre - n_cal
        beta_a = _ridge(X[:split], y[:split])
        cal_resid = y[split:n_pre] - X[split:n_pre] @ beta_a
    else:
        # too little history for a clean split - fall back and FLAG it
        beta_a = _ridge(X[:n_pre], y[:n_pre])
        cal_resid = y[:n_pre] - X[:n_pre] @ beta_a
        n_cal = len(cal_resid)

    # ---- Stage B: forecast the event window --------------------------
    beta_b = _ridge(X[:n_pre], y[:n_pre])
    yhat = X @ beta_b

    return {
        "yhat": yhat,
        "cal_resid": cal_resid,
        "seasonal": seasonal,
        "n_eval": n_eval,
        "n_cal": int(n_cal),
        "oos_calibration": n_cal >= 8,
    }


def aci_alpha(cal_resid: np.ndarray, alpha: float, gamma: float,
              max_steps: int = 60) -> float:
    """Adaptive Conformal Inference (Gibbs & Candes 2021).

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

    Chosen over EnbPI/SPCI on published evidence: a 2026 benchmark of
    conformal methods for time-series forecasting found ACI reached nominal
    coverage while EnbPI and SPCI under-covered. ACI is also far cheaper -
    no bootstrap ensemble to train.
    """
    s = np.abs(cal_resid)
    if len(s) < 4:
        return alpha
    s = s[-max_steps:]
    a = alpha
    srt = np.sort(s[: max(len(s) // 3, 3)])
    for i in range(len(srt), len(s)):
        past = s[:i]
        q = np.quantile(past, float(np.clip(1 - a, 0.01, 0.999)))
        err = 1.0 if s[i] > q else 0.0
        a = float(np.clip(a + gamma * (alpha - err), 0.005, 0.5))
    return a


def tail_pvalue(cal_resid: np.ndarray, score: float,
                u_q: float = 0.80, min_exc: int = 10) -> tuple[float, str]:
    """P-value for |residual| against the calibration sample.

    The conformal p-value is bounded below by 1/(n+1). With a ~60-point
    calibration window that floor is ~0.016, so NO finding could ever clear
    a Benjamini-Hochberg threshold across ~1,700 simultaneous tests - the
    entire multiplicity correction would be vacuous.

    The conformal INTERVAL remains the decision boundary (it carries the
    distribution-free coverage guarantee). For RANKING and FDR we need a
    p-value that can go small, so we fit a Generalized Pareto to the
    calibration exceedances above the 80th percentile and extrapolate the
    tail (peaks-over-threshold EVT). Inside the empirical range we use the
    empirical estimate; only beyond it do we extrapolate, and we always
    label which was used.
    """
    a = np.abs(np.asarray(cal_resid, dtype=float))
    a = a[np.isfinite(a)]
    n = len(a)
    if n == 0:
        return 1.0, "degenerate"

    emp = float((1.0 + np.sum(a >= score)) / (n + 1.0))
    u = float(np.quantile(a, u_q))
    exc = a[a > u] - u

    # Only extrapolate where it actually matters: when the empirical
    # estimate is pinned near its 1/(n+1) floor. Elsewhere the empirical
    # value is both adequate and cheaper (GPD MLE dominates runtime).
    if score <= u or len(exc) < min_exc or emp > 3.0 / (n + 1.0):
        return emp, "empirical"

    try:
        c, _, scale = stats.genpareto.fit(exc, floc=0.0)
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(c):
            return emp, "empirical"
        tail = float(stats.genpareto.sf(score - u, c, loc=0.0, scale=scale))
        p = (len(exc) / n) * tail
        if not np.isfinite(p) or p <= 0:
            p = float(np.finfo(float).tiny)
        # NOTE: deliberately NOT capped at the empirical floor - escaping
        # that floor is the whole purpose of the extrapolation.
        # Floor at 1e-12: extrapolating a 60-point sample further into the
        # tail than that is not credible, and reporting q=1e-300 would be
        # false precision.
        return float(min(max(p, 1e-12), 1.0)), "gpd_extrapolated"
    except Exception:
        return emp, "empirical"


def benjamini_hochberg(p: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (survives, q_values). Step-up procedure, monotone q-values."""
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1)
    crit = q * ranks / m
    passed = p_sorted <= crit
    k = int(np.max(np.where(passed)[0])) + 1 if passed.any() else 0

    survives_sorted = np.zeros(m, dtype=bool)
    if k > 0:
        survives_sorted[:k] = True

    q_raw = p_sorted * m / ranks
    q_sorted = np.minimum.accumulate(q_raw[::-1])[::-1].clip(0, 1)

    survives = np.empty(m, dtype=bool)
    q_vals = np.empty(m, dtype=float)
    survives[order] = survives_sorted
    q_vals[order] = q_sorted
    return survives, q_vals


def conformal_quantile(cal_resid: np.ndarray, alpha: float) -> tuple[float, bool, str]:
    """Finite-sample-corrected conformal quantile.

    The split-conformal guarantee requires the ceil((n+1)(1-alpha))/n empirical
    quantile, NOT the plain (1-alpha) quantile. The distinction is irrelevant
    for n=60 and decisive for n=7.

    When ceil((n+1)(1-alpha)) > n, no finite (1-alpha) interval exists from
    this calibration sample at all. That is the honest answer for a
    newly-launched slice, and it is the OPPOSITE of what a naive
    implementation does: a short, low-variance history produces small
    residuals and therefore a spuriously TIGHT interval, so the least
    trustworthy slices look the most certain. We inflate instead and mark the
    interval invalid.
    """
    a = np.abs(np.asarray(cal_resid, dtype=float))
    a = a[np.isfinite(a)]
    n = len(a)
    if n == 0:
        return float("inf"), False, "no calibration residuals"

    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        # No valid interval exists. Inflate the observed spread by the
        # t-vs-normal ratio at this sample size and flag it.
        from scipy import stats as _st
        dof = max(n - 1, 1)
        infl = float(_st.t.ppf(1 - alpha / 2, dof) / _st.norm.ppf(1 - alpha / 2))
        infl *= float(np.sqrt((n + 1) / n))
        return float(a.max() * infl), False, (
            f"only {n} calibration points; ceil((n+1)(1-alpha))={k} > n, so no valid "
            f"{100*(1-alpha):.0f}% conformal interval exists. Interval inflated x"
            f"{infl:.2f} and marked unreliable.")

    return float(np.sort(a)[k - 1]), True, ""
