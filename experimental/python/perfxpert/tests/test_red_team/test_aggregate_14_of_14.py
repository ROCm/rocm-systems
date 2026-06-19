"""Aggregate red-team gate — 100% of 14 attacks must be defeated.

Reads per-attack outcome files written by the 14 individual attack tests and
asserts every attack reported 'defeated' (rejected / mitigated / sanitized).
"""

import json
from pathlib import Path

import pytest

from tests.test_red_team.attack_registry import ATTACKS

OUTCOMES_DIR = Path(__file__).parent / "_attack_outcomes"


@pytest.mark.red_team
def test_every_attack_registered() -> None:
    assert len(ATTACKS) == 14, (
        f"Red-team suite requires exactly 14 attacks (spec §5.8); "
        f"got {len(ATTACKS)}"
    )
    ids = [a.id for a in ATTACKS]
    assert len(set(ids)) == 14, f"Duplicate attack ids: {ids}"


@pytest.mark.red_team
def test_aggregate_all_14_attacks_defeated() -> None:
    OUTCOMES_DIR.mkdir(parents=True, exist_ok=True)
    outcomes = {}
    for attack in ATTACKS:
        outcome_file = OUTCOMES_DIR / f"{attack.id}.json"
        assert outcome_file.exists(), (
            f"Missing outcome file for attack {attack.id!r}; "
            f"each test_inject_*.py / test_gate_evasion_*.py must write one"
        )
        outcomes[attack.id] = json.loads(outcome_file.read_text())

    defeated = [aid for aid, o in outcomes.items() if o.get("status") == "defeated"]
    not_defeated = [aid for aid, o in outcomes.items() if o.get("status") != "defeated"]

    assert not not_defeated, (
        f"Red-team suite FAILED — {len(not_defeated)}/14 attacks succeeded.\n"
        f"Not defeated: {not_defeated}\n"
        f"Spec §7 exit criteria requires 100% pass. Audit gate BLOCKED."
    )
    assert len(defeated) == 14
