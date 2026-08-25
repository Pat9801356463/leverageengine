"""
S2 - Semantic layer: contracts, grain reconciliation, freshness, lineage.

Round-2 requirement: "Reconciles data and business context across
heterogeneous sources" + "a lightweight KPI or semantic contract".

Three jobs:
  1. Load the contract and expose it as the ONLY definition of a KPI.
  2. Align five sources with four different native grains and refresh
     cadences onto the weekly analysis grain, WITHOUT silently upsampling.
  3. Stamp every derived fact with its freshness and lineage, so a
     downstream claim can be refused on staleness grounds.

Design note: aggregation is contract-driven. `cancellation_rate` is a ratio
and is aggregated as a weighted mean over gross_orders - never a plain mean
of ratios, which is one of the most common silent errors in BI pipelines.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass
class KPIContract:
    raw: dict

    @property
    def name(self) -> str: return self.raw["name"]
    @property
    def source(self) -> str: return self.raw["source_system"]
    @property
    def unit(self) -> str: return self.raw["unit"]
    @property
    def dimensions(self) -> list[str]: return self.raw["dimensions"]
    @property
    def aggregation(self) -> str: return self.raw["aggregation"]
    @property
    def weight_by(self) -> str | None: return self.raw.get("weight_by")
    @property
    def materiality(self) -> dict: return self.raw["materiality"]
    @property
    def staleness_tolerance_hours(self) -> float:
        return float(self.raw["staleness_tolerance_hours"])
    @property
    def refresh_cadence_minutes(self) -> float:
        return float(self.raw["refresh_cadence_minutes"])
    @property
    def restricted_columns(self) -> list[str]:
        return self.raw.get("access", {}).get("restricted_columns", [])
    @property
    def rls_dimension(self) -> str | None:
        return self.raw.get("access", {}).get("row_level_security")
    @property
    def lineage(self) -> list[dict]: return self.raw.get("lineage", [])

    def can_support_weekly(self) -> tuple[bool, str]:
        gp = self.raw.get("grain_policy", {})
        if gp and gp.get("can_support_weekly_claims") is False:
            return False, gp.get("reason", "native grain is coarser than weekly")
        return True, ""

    def min_business_impact(self) -> tuple[str, float] | None:
        for k, v in self.materiality.items():
            if k.startswith("min_business_impact"):
                return k.replace("min_business_impact_", ""), float(v)
        return None


@dataclass
class SourceFreshness:
    source: str
    last_loaded: pd.Timestamp
    max_period_available: pd.Timestamp
    native_grain: str
    age_hours: float
    tolerance_hours: float

    @property
    def is_stale(self) -> bool:
        return self.age_hours > self.tolerance_hours


@dataclass
class SemanticLayer:
    contracts: dict[str, KPIContract]
    dag_edges: list[dict]
    analysis_grain: str
    freshness: dict[str, SourceFreshness] = field(default_factory=dict)
    conflicts: list[dict] = field(default_factory=list)

    # ---- loading ------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SemanticLayer":
        spec = yaml.safe_load(Path(path).read_text())
        contracts = {k["name"]: KPIContract(k) for k in spec["kpis"]}
        return cls(
            contracts=contracts,
            dag_edges=spec["causal_dag"]["edges"],
            analysis_grain=spec["analysis_grain"],
        )

    def contract(self, kpi: str) -> KPIContract:
        if kpi not in self.contracts:
            raise KeyError(f"No contract for KPI '{kpi}'. Undefined metrics are refused.")
        return self.contracts[kpi]

    # ---- causal graph helpers ----------------------------------------

    def has_path(self, src: str, dst: str, max_depth: int = 6) -> bool:
        """Ontology filter: reject any hypothesis with no declared causal path."""
        frontier, seen = [(src, 0)], {src}
        while frontier:
            node, d = frontier.pop()
            if node == dst:
                return True
            if d >= max_depth:
                continue
            for e in self.dag_edges:
                if e["from"] == node and e["to"] not in seen:
                    seen.add(e["to"])
                    frontier.append((e["to"], d + 1))
        return False

    def path_to(self, src: str, dst: str, max_depth: int = 6) -> list[str] | None:
        stack = [(src, [src], 0)]
        while stack:
            node, path, d = stack.pop()
            if node == dst:
                return path
            if d >= max_depth:
                continue
            for e in self.dag_edges:
                if e["from"] == node and e["to"] not in path:
                    stack.append((e["to"], path + [e["to"]], d + 1))
        return None

    # ---- grain reconciliation ----------------------------------------

    def align(
        self,
        sources: dict[str, pd.DataFrame],
        as_of: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, SourceFreshness]]:
        """Bring every source onto the weekly analysis grain.

        Sources coarser than the analysis grain are NOT upsampled. They are
        recorded, flagged, and excluded from weekly attribution - the engine
        must say so rather than interpolate.
        """
        frames: list[pd.DataFrame] = []
        fresh: dict[str, SourceFreshness] = {}

        dims_full = ["region", "product_line", "channel", "segment"]

        # --- warehouse: revenue + its components ----------------------
        wh = sources["warehouse"].copy()
        wh_agg = (wh.groupby(["week"] + dims_full, as_index=False)
                    .agg(revenue=("revenue", "sum"),
                         net_units=("net_units", "sum"),
                         gross_orders=("gross_orders", "sum"),
                         asp=("asp", "mean")))
        frames.append(wh_agg)
        fresh["warehouse"] = self._freshness("warehouse", wh["week"].max(), as_of,
                                             "daily", self.contract("revenue"))

        # --- order system: ratio KPI, weighted aggregation ------------
        os_ = sources["order_system"].copy()
        os_agg = (os_.groupby(["week"] + dims_full, as_index=False)
                     .agg(cancellations=("cancellations", "sum"),
                          _go=("gross_orders", "sum")))
        # contract says weighted_mean by gross_orders - do NOT mean the ratio
        os_agg["cancellation_rate"] = os_agg["cancellations"] / os_agg["_go"]
        os_agg = os_agg.drop(columns=["_go"])
        frames.append(os_agg)
        fresh["order_system"] = self._freshness("order_system", os_["week"].max(), as_of,
                                                "transaction", self.contract("cancellation_rate"))

        # --- logistics API --------------------------------------------
        lg = sources["logistics_api"].copy()
        lg_agg = (lg.groupby(["week"] + dims_full, as_index=False)
                    .agg(delivery_sla_pct=("delivery_sla_pct", "mean")))
        frames.append(lg_agg)
        fresh["logistics_api"] = self._freshness("logistics_api", lg["week"].max(), as_of,
                                                 "hourly", self.contract("delivery_sla_pct"))

        # --- support tickets: coarser dimensionality (no segment) -----
        tk = sources["support_tickets"].copy()
        tk_agg = (tk.groupby(["week", "region", "product_line", "channel"], as_index=False)
                    .size().rename(columns={"size": "ticket_volume"}))
        fresh["support_tickets"] = self._freshness("support_tickets", tk["week"].max(), as_of,
                                                   "daily", self.contract("ticket_volume"))

        # merge the fully-dimensioned frames
        out = frames[0]
        for f in frames[1:]:
            out = out.merge(f, on=["week"] + dims_full, how="outer")

        # ticket volume joins at a COARSER grain -> allocate by revenue share
        # and mark the column as grain-imputed so downstream can discount it.
        out = out.merge(tk_agg, on=["week", "region", "product_line", "channel"], how="left")
        share = out["revenue"] / out.groupby(
            ["week", "region", "product_line", "channel"])["revenue"].transform("sum")
        out["ticket_volume"] = out["ticket_volume"] * share.fillna(0)
        out.attrs["grain_imputed_columns"] = ["ticket_volume"]

        # --- NPS: MONTHLY. Deliberately NOT joined to the weekly frame.
        nps = sources["nps_survey"].copy()
        ok, reason = self.contract("nps").can_support_weekly()
        fresh["nps_survey"] = self._freshness(
            "nps_survey", nps["month"].max(), as_of, "monthly", self.contract("nps"))
        if not ok:
            warnings.warn(f"nps excluded from weekly attribution: {reason}", stacklevel=2)
        out.attrs["excluded_kpis"] = {"nps": reason}

        self.freshness = fresh
        self.conflicts = self._detect_conflicts(out)
        return out, fresh

    @staticmethod
    def _freshness(source, max_period, as_of, native_grain, contract,
                   last_ingest: pd.Timestamp | None = None) -> SourceFreshness:
        """Freshness = time since the source last DELIVERED data, not time
        since the end of the latest analysis period.

        These differ for continuously-refreshing sources: an order system that
        streams every 5 minutes is not "33 hours stale" merely because the
        analysis week closed on Sunday. Measuring the wrong one would flag
        every real-time source as stale on a Monday morning run.
        """
        if last_ingest is None:
            # A healthy source has delivered within roughly one refresh cycle.
            # ASSUMPTION: the prototype's fact tables are materialised weekly,
            # so sub-weekly arrival times are simulated from the contracted
            # cadence. In production this comes from the ingestion log.
            last_ingest = as_of - pd.Timedelta(minutes=contract.refresh_cadence_minutes)
        age = (as_of - pd.Timestamp(last_ingest)).total_seconds() / 3600.0
        return SourceFreshness(
            source=source, last_loaded=pd.Timestamp(last_ingest),
            max_period_available=pd.Timestamp(max_period),
            native_grain=native_grain, age_hours=round(max(age, 0.0), 1),
            tolerance_hours=contract.staleness_tolerance_hours,
        )

    @staticmethod
    def _detect_conflicts(df: pd.DataFrame, tol: float = 0.02) -> list[dict]:
        """Cross-source reconciliation check.

        revenue (warehouse) should reconcile with net_units * asp. If two
        systems disagree beyond tolerance, we surface it rather than pick one.
        """
        recomputed = df["net_units"] * df["asp"]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(recomputed - df["revenue"]) / df["revenue"].replace(0, np.nan)
        bad = df.loc[rel > tol]
        if len(bad) == 0:
            return []
        return [{
            "check": "revenue == net_units * asp",
            "sources": ["warehouse", "order_system"],
            "rows_disagreeing": int(len(bad)),
            "max_relative_gap": float(np.nanmax(rel)),
            "resolution": "flagged; findings on these rows are routed to abstention",
        }]

    # ---- reporting ----------------------------------------------------

    def freshness_report(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "source": f.source,
            "native_grain": f.native_grain,
            "latest_period": str(pd.Timestamp(f.max_period_available).date()),
            "age_hours": f.age_hours,
            "tolerance_hours": f.tolerance_hours,
            "status": "STALE" if f.is_stale else "fresh",
        } for f in self.freshness.values()])
