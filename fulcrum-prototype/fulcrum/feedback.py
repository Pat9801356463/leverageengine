"""
S_FB - Analyst & business-user feedback loop.

Round-2 objective #7: "Mechanism to learn from analyst and business-user
feedback."

This module is deliberately small and legible. It does not silently reweight
model internals - that would make the causal engine's verdicts (S4b) harder
to audit, not easier. Instead it:

  1. Lets an analyst record a verdict on a hypothesis or a recommended
     action ("confirmed" | "rejected" | "acted_on" | "no_effect"), with an
     optional note, timestamp and attribution.
  2. Persists that feedback as an append-only JSONL log - the audit trail
     Round-2 explicitly asks for, and the natural format for something a
     dashboard can also export to.
  3. On the next run, surfaces any history relevant to today's hypotheses
     as context before the human sees today's verdict.
  4. Raises a new, explicit abstention trigger (FEEDBACK_CONFLICT) when an
     ACCEPTED hypothesis shares a cause_node with a hypothesis analysts
     rejected on a prior run - rather than silently trusting today's
     statistics over yesterday's correction, or silently overriding today's
     statistics because of it. It flags; a human decides.

This sits downstream of S4b the way a second reviewer sits downstream of a
diagnosis: it can raise a flag on the finding, not rewrite the test result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

VALID_ENTITY_TYPES = {"hypothesis", "action"}
VALID_VERDICTS = {"confirmed", "rejected", "acted_on", "no_effect"}


@dataclass
class FeedbackRecord:
    entity_type: str                # "hypothesis" | "action"
    entity_id: str                  # e.g. "E1" (hypothesis) or "L5" (lever)
    verdict: str                    # confirmed | rejected | acted_on | no_effect
    cause_node: str = ""            # hypotheses only - lets us match lookalikes
    note: str = ""
    user: str = "analyst"
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict:
        return asdict(self)


def record_feedback(path: Path, entity_type: str, entity_id: str, verdict: str,
                     cause_node: str = "", note: str = "",
                     user: str = "analyst") -> FeedbackRecord:
    """Append one feedback record to the JSONL log. Never rewrites history."""
    if entity_type not in VALID_ENTITY_TYPES:
        raise ValueError(f"entity_type must be one of {VALID_ENTITY_TYPES}, got {entity_type!r}")
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"verdict must be one of {VALID_VERDICTS}, got {verdict!r}")
    rec = FeedbackRecord(entity_type=entity_type, entity_id=entity_id,
                          cause_node=cause_node, verdict=verdict, note=note, user=user)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec.as_dict()) + "\n")
    return rec


def load_feedback(path: Path) -> list[dict]:
    """Load the full feedback history. A missing file means no history yet."""
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def history_for_cause_node(feedback: list[dict], cause_node: str) -> list[dict]:
    if not cause_node:
        return []
    return [r for r in feedback
            if r.get("entity_type") == "hypothesis" and r.get("cause_node") == cause_node]


def annotate_hypotheses(hyps: list, feedback: list[dict]) -> None:
    """Attach .feedback_history to each hypothesis in place.

    Does not touch h.verdict - S4b's statistical conclusion stays
    authoritative. This is context surfaced for a human (and for the
    feedback_conflict_trigger below), not a silent override.
    """
    for h in hyps:
        h.feedback_history = history_for_cause_node(feedback, h.proposed_cause_node)


def feedback_conflict_trigger(hyps: list):
    """
    Build an extra abstention trigger when an ACCEPTED hypothesis today
    shares a cause_node with a hypothesis analysts rejected on a prior run.
    Always returns a trigger (fired=False when there is nothing to flag) so
    the checklist can show the check ran, not just that it stayed quiet.
    """
    from .abstention import AbstentionTrigger  # local import avoids a cycle

    conflicts = []
    for h in hyps:
        if h.verdict != "accepted":
            continue
        rejections = [r for r in getattr(h, "feedback_history", [])
                      if r["verdict"] == "rejected"]
        if rejections:
            conflicts.append((h.hid, h.proposed_cause_node, len(rejections)))

    fired = bool(conflicts)
    detail = ("; ".join(f"{hid} (cause_node={node}) rejected by an analyst "
                         f"{n}x previously" for hid, node, n in conflicts)
               if fired else
               "no accepted hypothesis conflicts with prior analyst feedback")
    return AbstentionTrigger(
        code="FEEDBACK_CONFLICT",
        fired=fired,
        detail=detail,
        discriminating_test=(
            "Re-read the prior rejection note for this cause_node before "
            "acting on today's finding. If the source or definition changed "
            "since the rejection, record a new 'confirmed' feedback entry "
            "to clear this flag; if not, treat today's acceptance with "
            "extra scrutiny."
        ) if fired else "",
    )


def summarize(feedback: list[dict]) -> dict:
    by_verdict: dict[str, int] = {}
    for r in feedback:
        by_verdict[r["verdict"]] = by_verdict.get(r["verdict"], 0) + 1
    return {
        "n_records": len(feedback),
        "by_verdict": by_verdict,
        "entities_with_history": sorted({r["entity_id"] for r in feedback}),
    }
