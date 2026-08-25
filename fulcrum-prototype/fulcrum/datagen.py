"""
S1 - Scenario & data foundation.

Why synthetic rather than a public Kaggle retail set: public retail datasets
carry no labelled causal ground truth. Without knowing which driver actually
moved a KPI, a causal engine cannot be *validated* - only demonstrated. This
generator plants a known causal chain plus two deliberately-tempting decoys, so
every downstream claim ("we rejected the competitor sale") is checkable.

Planted causal chain (this is the DAG the engine must recover, not be told):

    vendor_switch -> delivery_sla_pct DOWN -> cancellation_rate UP
                                           -> ticket_volume UP
                          cancellation_rate UP -> net_units DOWN -> revenue DOWN

Planted scenarios
-----------------
E1  TRUE CAUSE   logistics vendor switch, 2026-07-27, North region only.
E2  DECOY        price rise +3%, 2026-08-03, LargeAppliances national.
                 Deliberately starts AFTER the cancellation spike -> must fail
                 the temporal-precedence test.
E3  DECOY        competitor sale, 2026-08-03..08-10, national, with HIGHER
                 intensity in West than North -> must fail difference-in-
                 differences and invert dose-response.
AMB unexplained  East x SmallAppliances demand shock with no event and no
                 ticket signal -> must trigger abstention.
SPA sparse       'SmartHome' product line launched 8 weeks before the analysis
                 date -> must widen intervals, not silently fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path

SEED = 20260809
ANALYSIS_WEEK = pd.Timestamp("2026-08-09")   # week ending; the week under study
N_WEEKS = 156

REGIONS = ["North", "South", "East", "West", "Central", "Northeast"]
PRODUCTS = ["LargeAppliances", "SmallAppliances", "Kitchen", "Personal"]
CHANNELS = ["Marketplace", "OwnSite", "Retail"]
SEGMENTS = ["Consumer", "Enterprise"]

SPARSE_PRODUCT = "SmartHome"          # launched late; sparse-history scenario
SPARSE_WEEKS = 14

TICKET_TOPICS = {
    "delivery_delay": [
        "delivery date pushed again no slot available",
        "still waiting for my delivery nobody called to schedule",
        "they gave me a delivery date three weeks out unacceptable",
        "no delivery slot shown at checkout for my pincode",
        "installation team never arrived on the scheduled day",
        "courier keeps rescheduling my appliance delivery",
    ],
    "price_complaint": [
        "price went up since last week why so expensive",
        "found the same model cheaper elsewhere price too high",
        "cost has increased not worth the money now",
    ],
    "product_defect": [
        "unit arrived with a dent on the door panel",
        "machine stopped working after two days defective",
        "display panel flickering right out of the box",
    ],
    "payment_issue": [
        "payment failed but amount was debited from account",
        "emi option not showing at checkout payment problem",
        "refund not credited back after failed transaction",
    ],
    "return_request": [
        "want to return this item how do i start a return",
        "return pickup not scheduled after raising request",
    ],
    "app_bug": [
        "app crashes when i open the order tracking page",
        "cannot log in to the app keeps showing error",
    ],
}


@dataclass
class PlantedEvent:
    event_id: str
    label: str
    start: str
    end: str | None
    scope: dict
    mechanism: str
    is_true_cause: bool


PLANTED_EVENTS = [
    PlantedEvent(
        "E1", "Logistics vendor switched (3PL contract change)",
        "2026-07-27", None, {"region": ["North"]},
        "vendor -> delivery_sla_pct down -> cancellation_rate up -> revenue down",
        True,
    ),
    PlantedEvent(
        "E2", "List price increase +3%",
        "2026-08-03", None, {"product_line": ["LargeAppliances"]},
        "price -> asp up, mild volume elasticity. Starts AFTER the effect began.",
        False,
    ),
    PlantedEvent(
        "E3", "Competitor promotional sale",
        "2026-07-20", "2026-08-10", {"region": "ALL"},
        "competitor -> gross_orders down, national, strongest in West not North.",
        False,
    ),
]


def _weeks() -> pd.DatetimeIndex:
    return pd.date_range(end=ANALYSIS_WEEK, periods=N_WEEKS, freq="W-SUN")


# Indian retail annual shape: monsoon/pre-festive softness in Aug,
# Onam/Navratri/Diwali surge Oct-Nov, post-festive slump in Feb.
MONTH_PROFILE = {
    1: 0.96, 2: 0.92, 3: 1.00, 4: 1.03, 5: 1.04, 6: 0.99,
    7: 0.95, 8: 0.89, 9: 1.03, 10: 1.19, 11: 1.14, 12: 1.05,
}


def _seasonal(weeks: pd.DatetimeIndex) -> np.ndarray:
    """Annual seasonality from a smooth interpolation of the month profile.

    Anchored to calendar position (not series index) so the August trough
    lands on the analysis week regardless of history length.
    """
    # place each month's multiplier at its mid-point (day 15) and interpolate
    frac = (weeks.dayofyear.to_numpy() - 1) / 365.25
    anchors_x, anchors_y = [], []
    for m, v in MONTH_PROFILE.items():
        anchors_x.append((pd.Timestamp(2025, m, 15).dayofyear - 1) / 365.25)
        anchors_y.append(v)
    # wrap for periodic interpolation
    ax = np.array([anchors_x[-1] - 1.0] + anchors_x + [anchors_x[0] + 1.0])
    ay = np.array([anchors_y[-1]] + anchors_y + [anchors_y[0]])
    return np.interp(frac, ax, ay) - 1.0


def _slice_frame() -> pd.DataFrame:
    rows = []
    for r in REGIONS:
        for p in PRODUCTS + [SPARSE_PRODUCT]:
            for c in CHANNELS:
                for s in SEGMENTS:
                    rows.append({"region": r, "product_line": p, "channel": c, "segment": s})
    return pd.DataFrame(rows)


def generate(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    weeks = _weeks()
    t = np.arange(N_WEEKS, dtype=float)
    slices = _slice_frame()

    e1_start = pd.Timestamp("2026-07-27")
    e2_start = pd.Timestamp("2026-08-03")
    e3_start, e3_end = pd.Timestamp("2026-07-20"), pd.Timestamp("2026-08-10")
    amb_start = pd.Timestamp("2026-07-27")
    sparse_start = weeks[-SPARSE_WEEKS]

    # region-level baseline scale and competitor exposure (dose)
    region_scale = dict(zip(REGIONS, [1.00, 0.86, 0.63, 0.92, 0.71, 0.44]))
    # NOTE: competitor spend is HIGHER in West than North -> inverts dose-response
    competitor_dose = {"North": 0.6, "South": 0.7, "East": 0.5,
                       "West": 1.0, "Central": 0.6, "Northeast": 0.4}
    product_scale = {"LargeAppliances": 1.0, "SmallAppliances": 0.42,
                     "Kitchen": 0.55, "Personal": 0.30, SPARSE_PRODUCT: 0.18}
    channel_scale = {"Marketplace": 1.0, "OwnSite": 0.70, "Retail": 0.55}
    segment_scale = {"Consumer": 1.0, "Enterprise": 0.35}
    base_asp = {"LargeAppliances": 42000, "SmallAppliances": 6800,
                "Kitchen": 14500, "Personal": 3900, SPARSE_PRODUCT: 9500}

    # how exposed a slice is to a delivery-SLA failure
    def sla_sensitivity(product, channel):
        p = {"LargeAppliances": 1.0, "Kitchen": 0.55, SPARSE_PRODUCT: 0.45,
             "SmallAppliances": 0.22, "Personal": 0.12}[product]
        c = {"Marketplace": 1.0, "OwnSite": 0.45, "Retail": 0.15}[channel]
        return p * c

    records = []
    for _, sl in slices.iterrows():
        r, p, c, sg = sl.region, sl.product_line, sl.channel, sl.segment
        scale = (region_scale[r] * product_scale[p] * channel_scale[c]
                 * segment_scale[sg] * 900)

        level = scale * rng.uniform(0.9, 1.1)
        trend = 0.00055 * rng.uniform(0.5, 1.5)
        seas = _seasonal(weeks)
        noise = rng.normal(0, 0.035, N_WEEKS)

        gross = level * (1 + trend * t) * (1 + seas) * (1 + noise)
        gross = np.clip(gross, 1.0, None)

        asp = np.full(N_WEEKS, base_asp[p], dtype=float) * (1 + rng.normal(0, 0.006, N_WEEKS))

        # --- baseline delivery SLA and cancellation --------------------
        sla = np.clip(0.945 + rng.normal(0, 0.012, N_WEEKS), 0.5, 0.999)
        sens = sla_sensitivity(p, c)

        post_e1 = (weeks >= e1_start)
        post_e2 = (weeks >= e2_start)
        in_e3 = (weeks >= e3_start) & (weeks <= e3_end)
        post_amb = (weeks >= amb_start)

        # E1 - TRUE CAUSE: vendor switch collapses SLA in North.
        # Calibrated so the most-exposed slice (Large/Marketplace, sens=1.0)
        # moves from ~7.5% to ~19% cancellations, matching the case narrative.
        if r == "North":
            sla = np.where(post_e1, sla - 0.115 * (0.35 + 0.65 * sens), sla)
            sla = np.clip(sla, 0.30, 0.999)

        # cancellation rate is CAUSED by SLA (this is the real mechanism)
        base_cancel = 0.055 + 0.02 * (c == "Marketplace") + rng.normal(0, 0.004, N_WEEKS)
        cancel = base_cancel + 1.05 * sens * np.clip(0.945 - sla, 0, None)
        cancel = np.clip(cancel, 0.005, 0.65)

        # E2 - DECOY: price rise (starts AFTER cancellations already moved)
        if p == "LargeAppliances":
            asp = np.where(post_e2, asp * 1.03, asp)
            gross = np.where(post_e2, gross * 0.988, gross)   # mild elasticity only

        # E3 - DECOY: competitor sale, national, strongest in West
        gross = np.where(in_e3, gross * (1 - 0.030 * competitor_dose[r]), gross)

        # AMB - unexplained demand shock, no event, no ticket signal
        if r == "East" and p == "SmallAppliances":
            gross = np.where(post_amb, gross * 0.90, gross)

        net_units = gross * (1 - cancel)
        revenue = net_units * asp

        # SPA - sparse history: product does not exist before launch
        mask = np.ones(N_WEEKS, dtype=bool)
        if p == SPARSE_PRODUCT:
            mask = weeks >= sparse_start

        # tickets driven by cancellations + SLA failures
        ticket_lambda = (30 * (gross / gross.mean())
                         + 900 * np.clip(cancel - base_cancel, 0, None)
                         + 260 * np.clip(0.945 - sla, 0, None))

        for i in range(N_WEEKS):
            if not mask[i]:
                continue
            records.append({
                "week": weeks[i], "region": r, "product_line": p,
                "channel": c, "segment": sg,
                "gross_orders": gross[i],
                "cancellations": gross[i] * cancel[i],
                "net_units": net_units[i],
                "asp": asp[i],
                "revenue": revenue[i],
                "delivery_sla_pct": sla[i],
                "cancellation_rate": cancel[i],
                "_ticket_lambda": ticket_lambda[i],
            })

    fact = pd.DataFrame.from_records(records)

    # ---- SOURCE 1: warehouse (daily grain, rolled to weekly here) ------
    warehouse = fact[["week", "region", "product_line", "channel", "segment",
                      "gross_orders", "net_units", "asp", "revenue"]].copy()
    warehouse["_source"] = "warehouse"

    # ---- SOURCE 2: order system (transaction grain, near real-time) ----
    orders = fact[["week", "region", "product_line", "channel", "segment",
                   "gross_orders", "cancellations", "cancellation_rate"]].copy()
    orders["_source"] = "order_system"

    # ---- SOURCE 3: logistics API (hourly grain) ------------------------
    logistics = fact[["week", "region", "product_line", "channel", "segment",
                      "delivery_sla_pct"]].copy()
    logistics["_source"] = "logistics_api"

    # ---- SOURCE 4: support platform (raw ticket text) -----------------
    tickets = _generate_tickets(fact, weeks, rng)

    # ---- SOURCE 5: NPS survey (MONTHLY grain - deliberately stale) -----
    nps = _generate_nps(fact, rng)

    # ---- SOURCE 6: external competitive-intel feed ---------------------
    # Per-region competitor promo intensity. This is what makes the
    # dose-response test possible: E3 is a real national event, but its
    # INTENSITY is highest in West, where revenue held up fine.
    ext_rows = []
    for wk in weeks:
        for r in REGIONS:
            active = (wk >= e3_start) and (wk <= e3_end)
            ext_rows.append({
                "week": wk, "region": r,
                "competitor_promo_intensity": competitor_dose[r] if active else 0.0,
                "competitor_promo_active": bool(active),
            })
    external = pd.DataFrame(ext_rows)

    events = pd.DataFrame([asdict(e) for e in PLANTED_EVENTS])
    events["scope"] = events["scope"].apply(lambda d: yaml.safe_dump(d, default_flow_style=True).strip())

    return {
        "warehouse": warehouse,
        "order_system": orders,
        "logistics_api": logistics,
        "support_tickets": tickets,
        "nps_survey": nps,
        "event_log": events,
        "external_signals": external,
    }


def _generate_tickets(fact: pd.DataFrame, weeks, rng) -> pd.DataFrame:
    """Raw ticket text. Topics are NOT labelled in the output the engine sees -
    they are recovered by unsupervised clustering in causal.py."""
    rows = []
    e1_start = pd.Timestamp("2026-07-27")
    e2_start = pd.Timestamp("2026-08-03")

    # subsample slices to keep ticket volume realistic
    agg = (fact.groupby(["week", "region", "product_line", "channel"], as_index=False)
                .agg(lam=("_ticket_lambda", "sum"),
                     cancel=("cancellation_rate", "mean"),
                     sla=("delivery_sla_pct", "mean")))

    for row in agg.itertuples(index=False):
        n = int(rng.poisson(max(row.lam * 0.05, 0.2)))
        if n <= 0:
            continue
        # topic mix depends on the true underlying state
        sla_gap = max(0.945 - row.sla, 0.0)
        w = {
            "delivery_delay": 0.10 + 7.0 * sla_gap,
            "price_complaint": 0.14 + (0.16 if (row.week >= e2_start and
                                                row.product_line == "LargeAppliances") else 0.0),
            "product_defect": 0.22,
            "payment_issue": 0.18,
            "return_request": 0.20,
            "app_bug": 0.16,
        }
        keys = list(w)
        probs = np.array([w[k] for k in keys], dtype=float)
        probs = probs / probs.sum()
        picks = rng.choice(len(keys), size=n, p=probs)
        for pk in picks:
            topic = keys[pk]
            text = rng.choice(TICKET_TOPICS[topic])
            rows.append({
                "week": row.week, "region": row.region,
                "product_line": row.product_line, "channel": row.channel,
                "ticket_text": text,
                "_latent_topic": topic,       # ground truth only; not used by engine
            })
    return pd.DataFrame(rows)


def _generate_nps(fact: pd.DataFrame, rng) -> pd.DataFrame:
    """Monthly grain. Deliberately cannot resolve a weekly movement."""
    f = fact.copy()
    f["month"] = f["week"].dt.to_period("M").dt.to_timestamp()
    g = (f.groupby(["month", "region"], as_index=False)
           .agg(sla=("delivery_sla_pct", "mean"), cancel=("cancellation_rate", "mean")))
    g["nps"] = (52 - 140 * (0.945 - g["sla"]).clip(lower=0)
                - 90 * (g["cancel"] - 0.06).clip(lower=0)
                + rng.normal(0, 2.4, len(g)))
    # survey lags: the latest month is not yet published
    latest = g["month"].max()
    g = g[g["month"] < latest]
    return g[["month", "region", "nps"]]


def ground_truth() -> dict:
    return {
        "analysis_week": str(ANALYSIS_WEEK.date()),
        "true_cause": {
            "event_id": "E1",
            "label": "Logistics vendor switched",
            "start": "2026-07-27",
            "scope": {"region": ["North"]},
            "mechanism": ("vendor -> delivery_sla_pct DOWN -> cancellation_rate UP "
                          "-> net_units DOWN -> revenue DOWN; also -> ticket_volume UP"),
            "most_affected_slice": "North / LargeAppliances / Marketplace",
        },
        "decoys": [
            {"event_id": "E2", "label": "Price increase +3%",
             "why_it_should_fail": "starts 2026-08-03, AFTER the cancellation spike "
                                   "began 2026-07-27 -> fails temporal precedence; "
                                   "also applied nationally with no effect in controls"},
            {"event_id": "E3", "label": "Competitor sale",
             "why_it_should_fail": "passes precedence (starts 2026-07-20, before onset) so it must be killed by dose-response: national scope but effect concentrated in North; "
                                   "competitor dose is HIGHER in West (1.0) than North (0.6) "
                                   "-> dose-response inverted"},
        ],
        "unexplained_by_design": {
            "slice": "East / SmallAppliances",
            "start": "2026-07-27",
            "note": "demand shock with NO event-log entry and NO ticket signal. "
                    "The engine MUST abstain rather than attribute it to E1.",
        },
        "sparse_history": {
            "product_line": SPARSE_PRODUCT,
            "weeks_of_history": SPARSE_WEEKS,
            "note": "must produce wider intervals and a low-confidence flag, not a crash",
        },
        "stale_source": {
            "kpi": "nps",
            "note": "monthly grain, latest month unpublished -> cannot support a weekly claim",
        },
    }


def write_all(outdir: str | Path) -> dict[str, pd.DataFrame]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = generate()
    for name, df in data.items():
        df.to_parquet(outdir / f"{name}.parquet", index=False)
    with open(outdir / "ground_truth.yaml", "w") as f:
        yaml.safe_dump(ground_truth(), f, sort_keys=False)
    return data


if __name__ == "__main__":
    d = write_all("data")
    for k, v in d.items():
        print(f"{k:18s} rows={len(v):>8,}  cols={list(v.columns)[:6]}")
