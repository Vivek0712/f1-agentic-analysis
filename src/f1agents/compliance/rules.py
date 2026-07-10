"""Deterministic compliance gate for agent recommendations.

Design rule: the LLM never adjudicates compliance. Sporting-regulation
constraints are encoded as pure functions; every agent recommendation is
validated here before it reaches a human, and every verdict is written to
an immutable audit record. The Compliance Guardian agent's only job is to
explain a violation in plain language, citing the rule id.

The rule set below covers the strategy-relevant subset of the FIA Sporting
Regulations for a dry race. It is intentionally a code artifact: reviewable,
versioned, testable.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict

RULESET_VERSION = "2026.1"


@dataclass
class Verdict:
    rule_id: str
    passed: bool
    detail: str


@dataclass
class AuditRecord:
    agent: str
    recommendation: dict
    verdicts: list[Verdict]
    ruleset_version: str = RULESET_VERSION
    model_id: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def passed(self) -> bool:
        return all(v.passed for v in self.verdicts)

    def to_json(self) -> str:
        d = asdict(self)
        d["passed"] = self.passed
        d["record_hash"] = hashlib.sha256(
            json.dumps(d, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        return json.dumps(d, default=str)


# ------------------------------------------------------------------- rules

def rule_two_compounds(rec: dict) -> Verdict:
    """Art. 30.5(m): at least two dry compounds must be used in a dry race."""
    plan = rec.get("compound_plan", [])
    ok = len(set(plan)) >= 2 if plan else False
    return Verdict("SR-30.5m-two-compounds", ok,
                   f"compound plan {plan} uses {len(set(plan))} distinct compound(s)")


def rule_pit_window_bounds(rec: dict) -> Verdict:
    """Recommended pit lap must fall inside the race distance with margin."""
    lap = rec.get("recommended_pit_lap")
    total = rec.get("race_distance_laps", 0)
    ok = lap is not None and 1 < lap < total - 1
    return Verdict("OPS-pit-window-bounds", ok,
                   f"pit lap {lap} vs race distance {total}")


def rule_stint_life_limit(rec: dict, max_stint: int = 40) -> Verdict:
    """Team-side tyre-life safety bound: no modeled stint beyond max_stint laps."""
    stints = rec.get("stint_lengths", [])
    ok = all(s <= max_stint for s in stints) if stints else True
    return Verdict("TEAM-stint-life-limit", ok,
                   f"stint lengths {stints}, limit {max_stint}")


def rule_advisory_only(rec: dict) -> Verdict:
    """Agent outputs are advisory. Any recommendation flagged for automatic
    execution fails closed."""
    ok = not rec.get("auto_execute", False)
    return Verdict("GOV-advisory-only", ok, "auto_execute must be false")


ACTIVE_RULES = [rule_two_compounds, rule_pit_window_bounds,
                rule_stint_life_limit, rule_advisory_only]


def validate(agent: str, recommendation: dict, model_id: str = "") -> AuditRecord:
    verdicts = [rule(recommendation) for rule in ACTIVE_RULES]
    return AuditRecord(agent=agent, recommendation=recommendation,
                       verdicts=verdicts, model_id=model_id)
