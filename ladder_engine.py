"""The Ladder Engine — the technical moat.

Hard requirements (PRD 3.2):
  * No network calls. No LLM. No randomness.
  * Same input -> same output, always.
  * Every Verdict carries the source_url and verified_on of the ladder it used.
  * blocked_by is never empty when maintainable is False, and each entry maps to a
    plain-language message from the YAML.

Nothing in this file imports anything that touches the network. That is the point:
the refusal is a tested function, not a prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import yaml

LADDER_DIR = Path(__file__).parent / "data" / "ladders"


@dataclass(frozen=True)
class Citation:
    id: str
    text: str
    source_url: str
    verified_on: str


@dataclass(frozen=True)
class Deadline:
    id: str
    label: str
    due_on: date
    tier: int


@dataclass
class Facts:
    grievance_class: str
    tier1_written_complaint: bool | None = None
    tier1_filed_on: date | None = None
    last_communication_on: date | None = None
    pending_in_court: bool = False
    amount_inr: float = 0.0
    account_type: str | None = None
    institution: str | None = None
    # Set when the grievance is a commercial-decision style exclusion.
    commercial_decision: bool = False

    def missing(self, required: list[str]) -> list[str]:
        """Required fact names that are still unknown."""
        return [f for f in required if getattr(self, f, None) is None]


@dataclass
class Verdict:
    ladder_id: str
    ladder_version: str
    current_tier: int
    maintainable: bool
    blocked_by: list[str]
    blocked_messages: list[str]
    redirect_to: int | None
    deadlines: list[Deadline]
    next_action: str
    citations: list[Citation]
    haq_score: int
    source_url: str
    verified_on: str
    facts_pending: list[str] = field(default_factory=list)


def load_ladders(directory: Path | None = None) -> dict:
    """Load every ladder YAML into {ladder_id: spec}. Pure file I/O, no network."""
    directory = directory or LADDER_DIR
    ladders: dict = {}
    for path in sorted(directory.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text())
        ladders[spec["ladder_id"]] = spec
    return ladders


def _select_ladder(facts: Facts, ladders: dict) -> dict | None:
    """Pick the ladder whose grievance_classes cover this grievance. Deterministic:
    sorted by ladder_id so ties always resolve the same way."""
    for ladder_id in sorted(ladders):
        spec = ladders[ladder_id]
        for prefix in spec.get("grievance_classes", []):
            if facts.grievance_class == prefix or facts.grievance_class.startswith(prefix + "/"):
                return spec
    return None


def _as_date(value) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _evaluate_exclusion(exclusion: dict, facts: Facts, today: date) -> bool:
    """Return True when this exclusion BLOCKS the case. One branch per rule kind,
    so every blocking reason is auditable and none of it is inferred by a model."""
    rule = exclusion["rule"]

    if rule == "tier1_not_exhausted":
        return facts.tier1_written_complaint is not True

    if rule == "pending_in_court":
        return bool(facts.pending_in_court)

    if rule == "commercial_decision":
        return bool(facts.commercial_decision)

    if rule == "time_barred":
        start = _later_of(facts.tier1_filed_on, facts.last_communication_on)
        if start is None:
            return False
        return (today - start).days > exclusion["window_days"]

    if rule == "below_minimum_amount":
        return facts.amount_inr < exclusion["min_amount_inr"]

    if rule == "above_maximum_amount":
        return facts.amount_inr > exclusion["max_amount_inr"]

    if rule == "waiting_period_not_elapsed":
        # The entity gets N days to respond before the next tier opens.
        if facts.tier1_filed_on is None:
            return False
        return (today - facts.tier1_filed_on).days < exclusion["wait_days"]

    raise ValueError(f"unknown exclusion rule: {rule!r}")


def _later_of(a: date | None, b: date | None) -> date | None:
    """The 90-day clock runs from the LATER of the two dates (PRD 3.3)."""
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _haq_score(facts: Facts, blocked: list[str], ladder: dict) -> int:
    """Rule-based, not a model (PRD 1.2). Deterministic and explainable."""
    score = 50
    if not blocked:
        score += 30
    if facts.tier1_written_complaint:
        score += 10
    if facts.amount_inr >= ladder.get("scoring", {}).get("material_amount_inr", 10000):
        score += 5
    if facts.pending_in_court:
        score -= 40
    return max(0, min(100, score))


def resolve(facts: Facts, ladders: dict, today: date) -> Verdict:
    """The pure function. No network, no LLM, no randomness."""
    ladder = _select_ladder(facts, ladders)
    if ladder is None:
        return Verdict(
            ladder_id="none",
            ladder_version="-",
            current_tier=0,
            maintainable=False,
            blocked_by=["no_ladder_for_grievance"],
            blocked_messages=[
                "We do not yet have a verified escalation ladder for this kind of grievance."
            ],
            redirect_to=None,
            deadlines=[],
            next_action="Out of scope for the ladders currently loaded.",
            citations=[],
            haq_score=0,
            source_url="",
            verified_on="",
        )

    facts = Facts(**{**facts.__dict__})
    facts.tier1_filed_on = _as_date(facts.tier1_filed_on)
    facts.last_communication_on = _as_date(facts.last_communication_on)

    target_tier = ladder.get("target_tier", 2)
    tiers = {t["tier"]: t for t in ladder["tiers"]}

    pending = facts.missing(ladder.get("required_facts", []))

    blocked_by: list[str] = []
    blocked_messages: list[str] = []
    redirect_to: int | None = None

    for exclusion in ladder.get("exclusions", []):
        if _evaluate_exclusion(exclusion, facts, today):
            blocked_by.append(exclusion["id"])
            blocked_messages.append(exclusion["message"])
            if redirect_to is None and exclusion.get("redirect_to") is not None:
                redirect_to = exclusion["redirect_to"]

    maintainable = not blocked_by and not pending
    current_tier = redirect_to if redirect_to is not None else target_tier

    deadlines = _deadlines_for(facts, ladder, today, current_tier)

    citations = [
        Citation(
            id=c["id"],
            text=c["text"],
            source_url=c["source_url"],
            verified_on=c["verified_on"],
        )
        for c in ladder.get("citations", [])
    ]

    if pending:
        next_action = f"Need {len(pending)} more fact(s) before a verdict is possible."
    elif blocked_by:
        tier_name = tiers.get(current_tier, {}).get("name", f"tier {current_tier}")
        next_action = f"Not maintainable at tier {target_tier}. Go to {tier_name} first."
    else:
        tier_name = tiers.get(current_tier, {}).get("name", f"tier {current_tier}")
        next_action = f"Maintainable. File at {tier_name}."

    return Verdict(
        ladder_id=ladder["ladder_id"],
        ladder_version=str(ladder.get("version", "1")),
        current_tier=current_tier,
        maintainable=maintainable,
        blocked_by=blocked_by,
        blocked_messages=blocked_messages,
        redirect_to=redirect_to,
        deadlines=deadlines,
        next_action=next_action,
        citations=citations,
        haq_score=_haq_score(facts, blocked_by, ladder),
        source_url=ladder["source_url"],
        verified_on=str(ladder["verified_on"]),
        facts_pending=pending,
    )


def _deadlines_for(facts: Facts, ladder: dict, today: date, tier: int) -> list[Deadline]:
    """Deadlines are computed, never stored — so they are identical across runs."""
    out: list[Deadline] = []
    anchor = _later_of(facts.tier1_filed_on, facts.last_communication_on)

    for spec in ladder.get("deadlines", []):
        if spec["from"] == "later_of_filed_and_last_communication":
            if anchor is None:
                continue
            due = anchor + timedelta(days=spec["days"])
        elif spec["from"] == "tier1_filed_on":
            if facts.tier1_filed_on is None:
                continue
            due = facts.tier1_filed_on + timedelta(days=spec["days"])
        elif spec["from"] == "today":
            due = today + timedelta(days=spec["days"])
        else:
            raise ValueError(f"unknown deadline anchor: {spec['from']!r}")

        out.append(
            Deadline(id=spec["id"], label=spec["label"], due_on=due, tier=spec.get("tier", tier))
        )

    return sorted(out, key=lambda d: (d.due_on, d.id))
