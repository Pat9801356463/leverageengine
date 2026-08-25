"""
S6 - Decision layer: which BUNDLE of moves, within budget?

The central claim of Fulcrum lives here.

A conventional root-cause tool produces one recommended fix per diagnosed
cause and scores each in isolation. That is how you arrive at three
individually-justified repairs costing Rs 95L when a single structural move
costing Rs 55L solves the same problem - the phone whose battery and GPU
upgrades together cost more than a new phone.

The fix is a change of primitive:

    An action has NO intrinsic value.
    An action is a priced way of relieving a SET of constraints.
    Value belongs to the resulting CONSTRAINT STATE.

        V(S) = sum over c in S of  blocked_value[c]  if requires[c] subset of S
                                   0                 otherwise

Three behaviours fall out of that definition instead of being special-cased:

  * REDUNDANCY      two levers relieving the same constraint contribute a
                    union, not a sum.
  * COMPLEMENTARITY relieving C2 without its prerequisite C1 yields zero -
                    fixing listings while deliveries still fail buys nothing.
  * STRUCTURAL      "exit the channel" is simply a lever whose relieved-set
                    covers three constraints at once. It is not the repair of
                    anything, which is exactly why a diagnosis-driven
                    recommender can never propose it.

Two solvers are run and cross-checked:
  * exhaustive enumeration over 2^n bundles - exact, fully transparent, and
    the right choice while n is small (transparency beats elegance in review).
  * a MILP (PuLP/CBC) - the scalable path. If the two disagree, the model is
    wrong, so we assert they agree.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import pulp
import yaml

AUTHORITY_ORDER = {"manager": 1, "director": 2, "vp": 3, "cxo": 4}


@dataclass
class Constraint:
    id: str
    label: str
    blocked_value: float
    requires: list[str]
    owner: str
    evidence_kpi: str


@dataclass
class Lever:
    id: str
    label: str
    family: str
    cost: float
    relieves: list[str]
    owner: str
    authority_level: str
    reversibility: str
    lead_time_weeks: int
    monitoring_metric: str
    monitoring_window_weeks: int
    success_threshold: str


@dataclass
class Bundle:
    levers: list[str]
    labels: list[str]
    cost: float
    relieved: set[str]
    realised: set[str]
    value_per_week: float
    families: set[str]
    feasible: bool
    infeasible_reason: str = ""

    @property
    def roi_weeks_to_payback(self) -> float:
        return self.cost / self.value_per_week if self.value_per_week > 0 else float("inf")

    @property
    def value_per_rupee(self) -> float:
        return self.value_per_week / self.cost if self.cost > 0 else float("inf")


@dataclass
class DecisionModel:
    constraints: dict[str, Constraint]
    levers: dict[str, Lever]
    objective: dict
    envelope: dict
    personas: list[dict] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DecisionModel":
        spec = yaml.safe_load(Path(path).read_text())
        cons = {c["id"]: Constraint(
            id=c["id"], label=c["label"],
            blocked_value=float(c["blocked_value_inr_per_week"]),
            requires=list(c.get("requires", [])),
            owner=c["owner"], evidence_kpi=c["evidence_kpi"]) for c in spec["constraints"]}
        lev = {l["id"]: Lever(
            id=l["id"], label=" ".join(str(l["label"]).split()), family=l["family"],
            cost=float(l["cost_inr"]), relieves=list(l.get("relieves", [])),
            owner=l["owner"], authority_level=l["authority_level"],
            reversibility=l["reversibility"], lead_time_weeks=int(l["lead_time_weeks"]),
            monitoring_metric=l["monitoring_metric"],
            monitoring_window_weeks=int(l["monitoring_window_weeks"]),
            success_threshold=str(l["success_threshold"])) for l in spec["levers"]}
        return cls(constraints=cons, levers=lev, objective=spec["objective"],
                   envelope=spec["capital_envelope"], personas=spec.get("personas", []))

    # ---- the value function -------------------------------------------

    def realised_constraints(self, relieved: set[str]) -> set[str]:
        """Apply prerequisite gating until the set stops changing."""
        out = set()
        changed = True
        while changed:
            changed = False
            for cid in relieved:
                if cid in out:
                    continue
                c = self.constraints[cid]
                if all(r in relieved for r in c.requires):
                    out.add(cid)
                    changed = True
        return out

    def value(self, relieved: set[str]) -> float:
        return sum(self.constraints[c].blocked_value
                   for c in self.realised_constraints(relieved))

    # ---- candidate bundles --------------------------------------------

    def _build_bundle(self, combo: tuple[str, ...], budget: float) -> Bundle:
        cost = sum(self.levers[l].cost for l in combo)
        relieved = set().union(*[set(self.levers[l].relieves) for l in combo]) if combo else set()
        realised = self.realised_constraints(relieved)
        return Bundle(
            levers=list(combo),
            labels=[self.levers[l].label for l in combo],
            cost=cost, relieved=relieved, realised=realised,
            value_per_week=self.value(relieved),
            families={self.levers[l].family for l in combo},
            feasible=cost <= budget,
            infeasible_reason=("" if cost <= budget else
                               f"cost Rs {cost/1e5:.1f}L exceeds envelope Rs {budget/1e5:.1f}L"),
        )

    def enumerate_bundles(self, budget: float | None = None,
                          allowed_levers: list[str] | None = None) -> list[Bundle]:
        budget = budget if budget is not None else float(self.envelope["total_budget_inr"])
        ids = [l for l in self.levers if l != "L0"]
        if allowed_levers is not None:
            ids = [l for l in ids if l in allowed_levers]

        out = [self._build_bundle((), budget)]           # the null action
        for r in range(1, len(ids) + 1):
            for combo in itertools.combinations(ids, r):
                out.append(self._build_bundle(combo, budget))

        # drop dominated bundles: same realised set, higher cost
        best: dict[frozenset, Bundle] = {}
        for b in out:
            k = frozenset(b.realised)
            if k not in best or b.cost < best[k].cost:
                best[k] = b
        keep = set(id(b) for b in best.values())
        return [b for b in out if id(b) in keep or not b.feasible]

    # ---- MILP (the scalable path) --------------------------------------

    def solve_milp(self, budget: float | None = None,
                   allowed_levers: list[str] | None = None) -> Bundle:
        budget = budget if budget is not None else float(self.envelope["total_budget_inr"])
        ids = [l for l in self.levers if l != "L0"]
        if allowed_levers is not None:
            ids = [l for l in ids if l in allowed_levers]

        prob = pulp.LpProblem("fulcrum_portfolio", pulp.LpMaximize)
        x = {l: pulp.LpVariable(f"x_{l}", cat="Binary") for l in ids}
        z = {c: pulp.LpVariable(f"z_{c}", cat="Binary") for c in self.constraints}

        prob += pulp.lpSum(self.constraints[c].blocked_value * z[c] for c in z)
        prob += pulp.lpSum(self.levers[l].cost * x[l] for l in ids) <= budget

        for c in self.constraints:
            coverers = [l for l in ids if c in self.levers[l].relieves]
            prob += z[c] <= (pulp.lpSum(x[l] for l in coverers) if coverers else 0)
            for req in self.constraints[c].requires:     # prerequisite gating
                prob += z[c] <= z[req]

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        chosen = tuple(sorted(l for l in ids if x[l].value() and x[l].value() > 0.5))
        return self._build_bundle(chosen, budget)

    # ---- decision rights ----------------------------------------------

    def levers_for_persona(self, persona: dict) -> list[str]:
        cap = AUTHORITY_ORDER.get(persona.get("max_authority_level", "manager"), 1)
        return [l.id for l in self.levers.values()
                if l.id != "L0" and AUTHORITY_ORDER.get(l.authority_level, 9) <= cap]

    # ---- ranking + the mandated output schema --------------------------

    def rank(self, budget: float | None = None,
             allowed_levers: list[str] | None = None) -> list[Bundle]:
        bundles = [b for b in self.enumerate_bundles(budget, allowed_levers) if b.feasible]
        return sorted(bundles, key=lambda b: (-b.value_per_week, b.cost))

    def to_recommendation(self, bundle: Bundle, driver: str, confidence: float,
                          evidence_ref: str) -> list[dict]:
        """Emit the exact schema the Round-2 brief specifies:
        driver -> controllable lever -> action -> expected impact -> owner
               -> confidence -> monitoring plan
        """
        recs = []
        for lid in bundle.levers:
            l = self.levers[lid]
            realised_here = [c for c in l.relieves if c in bundle.realised]
            impact = sum(self.constraints[c].blocked_value for c in realised_here)
            recs.append({
                "driver": driver,
                "controllable_lever": l.id,
                "action": l.label,
                "expected_impact_inr_per_week": round(impact, 0),
                "expected_impact_note": (
                    "0 - prerequisite constraint not relieved in this bundle"
                    if impact == 0 else
                    f"relieves {', '.join(realised_here)}"),
                "cost_inr": l.cost,
                "owner": l.owner,
                "authority_required": l.authority_level,
                "reversibility": l.reversibility,
                "confidence": round(confidence, 3),
                "monitoring_plan": {
                    "metric": l.monitoring_metric,
                    "window_weeks": l.monitoring_window_weeks,
                    "success_threshold": l.success_threshold,
                    "review_owner": l.owner,
                },
                "evidence_ref": evidence_ref,
            })
        return recs


def budget_sweep(model: DecisionModel, budgets: list[float]) -> list[dict]:
    """The demo interaction: drag the budget and watch the recommendation
    change SHAPE - patch, then structural move, then full repair."""
    out = []
    for b in budgets:
        ranked = model.rank(budget=b)
        top = ranked[0] if ranked else None
        out.append({
            "budget_inr": b,
            "chosen_levers": top.levers if top else [],
            "chosen_labels": [l[:52] for l in top.labels] if top else [],
            "family": ("+".join(sorted(top.families)) if top and top.families else "none"),
            "cost_inr": top.cost if top else 0,
            "value_per_week_inr": top.value_per_week if top else 0,
            "constraints_realised": sorted(top.realised) if top else [],
            "payback_weeks": round(top.roi_weeks_to_payback, 1) if top else None,
        })
    return out
