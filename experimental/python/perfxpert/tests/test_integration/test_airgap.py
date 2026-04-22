"""Airgap parity test — spec §7 exit criterion 3.

For EVERY parity fixture, every gate decision + every handoff target MUST be
identical whether LLM is available or not. Only narrative phrasing differs.

This test is distinct from the parity suite (Task 2-5): the parity suite
compares new-path vs old-path; this test compares new-path-with-LLM vs
new-path-without-LLM.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.test_parity.fixtures_inventory import available_parity_fixtures
from perfxpert.ai_analysis.api import analyze_database


AIRGAP_SNAPSHOT_DIR = Path(__file__).parent / "_airgap_snapshots"


@contextmanager
def airgap_env(on: bool):
    prev = os.environ.get("PERFXPERT_AIRGAP")
    if on:
        os.environ["PERFXPERT_AIRGAP"] = "1"
        os.environ.setdefault("PERFXPERT_NEW_PATH", "1")
    else:
        os.environ.pop("PERFXPERT_AIRGAP", None)
        os.environ.setdefault("PERFXPERT_NEW_PATH", "1")
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("PERFXPERT_AIRGAP", None)
        else:
            os.environ["PERFXPERT_AIRGAP"] = prev


def _extract_decisions(result) -> dict:
    """Extract the audit-relevant decision fingerprint from AnalysisResult.
    NARRATIVE (prose, LLM-specific phrasing) is EXCLUDED intentionally.
    """
    summary = getattr(result, "summary", None)
    recs = getattr(result, "recommendations", None)
    high = (recs.high_priority if recs and recs.high_priority else []) if recs else []
    return {
        "primary_bottleneck": getattr(summary, "primary_bottleneck", None) if summary else None,
        "handoff_targets": [getattr(r, "handoff_target", None) for r in high],
        "rec_categories": [getattr(r, "category", None) for r in high],
        "rec_techniques": [
            getattr(r, "technique_id", None) or getattr(r, "id", None) for r in high
        ],
        "gate_verdict": getattr(result, "gate_verdict", None),
    }


def pytest_generate_tests(metafunc):
    if "fx" in metafunc.fixturenames:
        fixtures = available_parity_fixtures()
        metafunc.parametrize("fx", fixtures, ids=[fx.id for fx in fixtures])


@pytest.mark.parity
def test_airgap_decision_fingerprint_identical_to_llm_mode(fx) -> None:
    with airgap_env(on=False):
        with_llm_result = analyze_database(database_path=str(fx.db_path), enable_llm=False)
    with airgap_env(on=True):
        no_llm_result = analyze_database(database_path=str(fx.db_path), enable_llm=False)

    on = _extract_decisions(with_llm_result)
    off = _extract_decisions(no_llm_result)

    AIRGAP_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    (AIRGAP_SNAPSHOT_DIR / f"{fx.id}.json").write_text(
        json.dumps({"with_llm": on, "airgap": off}, indent=2, default=str)
    )

    assert on == off, (
        f"Airgap parity violated on {fx.id}:\n"
        f"  with_llm={on}\n  airgap={off}\n"
        f"Spec §7 exit criterion 3 requires 100% identical gate decisions + handoffs."
    )
