"""Aggregate red-team gate — every registered attack must be defeated.

Reads per-attack outcome files written by the individual attack tests and
asserts every attack reported 'defeated' (rejected / mitigated / sanitized).

The count comes from the registry rather than a literal: a hard-coded total
turns adding an attack into a failing build, and the tempting fix is to bump
the number rather than wire the attack up.
"""

import json
from pathlib import Path

import pytest

from tests.test_red_team.attack_registry import ATTACKS

OUTCOMES_DIR = Path(__file__).parent / "_attack_outcomes"


@pytest.mark.red_team
def test_every_attack_registered() -> None:
    ids = [a.id for a in ATTACKS]
    assert len(set(ids)) == len(ids), f"Duplicate attack ids: {ids}"
    # Spec §5.8 defined the original inventory; the suite may only grow.
    assert len(ATTACKS) >= 14, (
        f"Red-team suite requires at least the 14 attacks of spec §5.8; "
        f"got {len(ATTACKS)}"
    )


@pytest.mark.red_team
def test_aggregate_all_attacks_defeated() -> None:
    total = len(ATTACKS)
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
        f"Red-team suite FAILED — {len(not_defeated)}/{total} attacks succeeded.\n"
        f"Not defeated: {not_defeated}\n"
        f"Spec §7 exit criteria requires 100% pass. Audit gate BLOCKED."
    )
    assert len(defeated) == total
