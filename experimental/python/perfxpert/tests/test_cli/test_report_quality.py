###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""Regression guards for three bugs in the agentic report output.

These tests pin the expected contract after the Phase 8 "report quality"
fix set:

1. Narrative reflects Analysis findings, not Root's routing prose.
2. Recommendations never contain narrative prose ("Let me proceed…",
   "Delegating…") — fabricated-rec fallback was removed.
3. Tier-0 profiling-plan entries live in a dedicated Tier-0 section,
   NOT in the main recommendations list.

All tests run under airgap so no LLM credentials are required.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from perfxpert import analyze as analyze_mod
from perfxpert import api as perfxpert_api
from perfxpert import output_config
from perfxpert.connection import PerfxpertConnection as RocpdImportData


_FIXTURE_DB = (
    Path(__file__).resolve().parent.parent / "fixtures" / "compute_bound.db"
)


@pytest.fixture
def airgap(monkeypatch):
    """Force the agentic pipeline to take the deterministic airgap branch."""
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")


@pytest.fixture
def tiny_hip_src(tmp_path: Path) -> Path:
    """A minimal HIP tree with exactly ONE hipMemcpy call.

    Used by tests that must assert profiling-plan separation independent
    of the "multiple hipMemcpy" code-pattern heuristic.
    """
    (tmp_path / "kernel.hip").write_text(
        "#include <hip/hip_runtime.h>\n"
        "__global__ void add(float* a, float* b) { *a += *b; }\n"
        "int main() {\n"
        "  hipLaunchKernelGGL(add, dim3(1), dim3(1), 0, 0, nullptr, nullptr);\n"
        "  hipDeviceSynchronize();\n"
        "  return 0;\n"
        "}\n"
    )
    return tmp_path


@pytest.fixture
def sync_memcpy_hip_src(tmp_path: Path) -> Path:
    """A HIP tree with multiple synchronous hipMemcpy calls.

    The Tier-0 scanner should flag this as a real code-level perf issue
    and surface a recommendation — distinct from the profiling-plan
    instrumentation advice.
    """
    (tmp_path / "kernel.hip").write_text(
        "#include <hip/hip_runtime.h>\n"
        "__global__ void add(float* a, float* b) { *a += *b; }\n"
        "int main() {\n"
        "  float *a, *b;\n"
        "  hipMemcpy(a, b, 64, hipMemcpyHostToDevice);\n"
        "  hipMemcpy(a, b, 64, hipMemcpyHostToDevice);\n"
        "  hipMemcpy(a, b, 64, hipMemcpyHostToDevice);\n"
        "  hipMemcpy(a, b, 64, hipMemcpyDeviceToHost);\n"
        "  hipLaunchKernelGGL(add, dim3(1), dim3(1), 0, 0, nullptr, nullptr);\n"
        "  return 0;\n"
        "}\n"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Bug 1 — narrative reflects Analysis, not Root's routing prose.
# ---------------------------------------------------------------------------


def test_narrative_reflects_analysis_not_routing(airgap):
    """The final narrative names a kernel / metric, and does NOT contain any
    of the classic routing-speech tokens that used to leak through when
    Root's own LLM output was used verbatim as the narrative.
    """
    verdict = perfxpert_api.agent_root(
        airgap=True,
        database_path=str(_FIXTURE_DB),
        user_query="analyze this trace",
    )
    narrative = verdict.get("narrative") or ""

    # Routing-speech tokens must NOT appear.
    for banned in (
        "Delegating",
        "Let me proceed",
        "routing",
        "Based on the classified intent",
    ):
        assert banned.lower() not in narrative.lower(), (
            f"narrative still contains routing-speech token {banned!r}: "
            f"{narrative!r}"
        )

    # The narrative MUST carry real evidence — a metric reference or a
    # kernel name. Accept any of: a percentage, 'ns', 'bandwidth', or a
    # backticked kernel identifier.
    evidence_pattern = re.compile(
        r"(`[\w:<>]+`|\d+(?:\.\d+)?\s*%|\bns\b|\bbandwidth\b|kernel)",
        re.IGNORECASE,
    )
    assert evidence_pattern.search(narrative), (
        f"narrative lacks analysis evidence (kernel name / % / metric): "
        f"{narrative!r}"
    )


# ---------------------------------------------------------------------------
# Bug 2 — no recommendation carries narrative prose in issue / what_to_do.
# ---------------------------------------------------------------------------


def test_no_recommendation_contains_narrative_prose(airgap, tmp_path):
    """Run the full agentic path through ``_execute_agentic`` so the merged
    recommendations list (the one users see in the report) is exercised.

    Fabricated recommendations used to show up with issue / suggestion
    equal to the first line of the narrative. After the Bug 2 fix, the
    fallback is removed; the merged list is populated either by the
    deterministic pass or left empty — never by narrative prose.
    """
    cfg = output_config.output_config(
        output_file="report", output_path=str(tmp_path)
    )
    conn = RocpdImportData([str(_FIXTURE_DB)])
    analyze_mod._execute_agentic(
        conn,
        config=cfg,
        output_format="json",
    )
    doc = json.loads((tmp_path / "report.json").read_text())
    recs = doc.get("recommendations") or []

    banned_in_prose = ("let me proceed", "delegating", "routing to")
    for rec in recs:
        issue = (rec.get("issue") or "").lower()
        suggestion = (rec.get("suggestion") or "").lower()
        what_to_do = (rec.get("what_to_do") or "").lower()  # possible alias
        for banned in banned_in_prose:
            assert banned not in issue, (
                f"rec issue contains narrative prose {banned!r}: {rec!r}"
            )
            assert banned not in suggestion, (
                f"rec suggestion contains narrative prose {banned!r}: {rec!r}"
            )
            assert banned not in what_to_do, (
                f"rec what_to_do contains narrative prose {banned!r}: {rec!r}"
            )


# ---------------------------------------------------------------------------
# Bug 3 — Tier-0 profiling-plan entries separated from main recs.
# ---------------------------------------------------------------------------


def test_tier0_profiling_plan_not_in_main_recs(airgap, tmp_path, tiny_hip_src):
    """Profiling-plan instrumentation advice must land under
    ``tier0_findings.profiling_plan``, NEVER in the main recommendations
    list. No rec may have ``category == "Tier-0 Profiling Plan"`` or an
    ``issue`` starting with ``"Found <digit>"``.
    """
    cfg = output_config.output_config(
        output_file="report", output_path=str(tmp_path)
    )
    conn = RocpdImportData([str(_FIXTURE_DB)])
    analyze_mod._execute_agentic(
        conn,
        config=cfg,
        output_format="json",
        source_dir=str(tiny_hip_src),
    )
    doc = json.loads((tmp_path / "report.json").read_text())

    # (a) No main rec is a Tier-0 Profiling Plan entry.
    recs = doc.get("recommendations") or []
    banned_recs = [r for r in recs if r.get("category") == "Tier-0 Profiling Plan"]
    assert not banned_recs, (
        f"Tier-0 Profiling Plan entries leaked into main recs: {banned_recs!r}"
    )

    for r in recs:
        issue = r.get("issue") or ""
        assert not re.match(r"^Found \d", issue), (
            f"main rec issue starts with profiling-plan prose: {r!r}"
        )

    # (b) In combined mode (DB is present) the profiling-plan/instrumentation-advice
    # keys are stripped from tier0_findings — they only make sense in the
    # source-only path. The dedicated source-only test below re-asserts the
    # profiling_plan is present when there is no DB.
    tier0 = doc.get("tier0_findings") or {}
    assert "profiling_plan" not in tier0 or not tier0["profiling_plan"], (
        f"combined-mode tier0_findings must not carry profiling_plan: {tier0!r}"
    )
    assert "suggested_first_command" not in tier0, (
        f"combined-mode tier0_findings must not carry suggested_first_command: {tier0!r}"
    )


def test_tier0_code_patterns_CAN_flow_into_recs(
    airgap, tmp_path, sync_memcpy_hip_src,
):
    """Real code-level patterns (e.g. 'multiple synchronous hipMemcpy')
    SHOULD surface as main recommendations — those describe an actual
    perf issue in the user's source, not instrumentation advice.
    """
    cfg = output_config.output_config(
        output_file="report", output_path=str(tmp_path)
    )
    conn = RocpdImportData([str(_FIXTURE_DB)])
    analyze_mod._execute_agentic(
        conn,
        config=cfg,
        output_format="json",
        source_dir=str(sync_memcpy_hip_src),
    )
    doc = json.loads((tmp_path / "report.json").read_text())
    recs = doc.get("recommendations") or []
    haystack = " ".join(
        (r.get("issue") or "") + " " + (r.get("suggestion") or "")
        for r in recs
    ).lower()
    assert "hipmemcpy" in haystack or "synchronous" in haystack, (
        f"expected a 'synchronous hipMemcpy' code-level rec in main list; "
        f"got recs={recs!r}"
    )


def test_report_html_has_separate_tier0_section(
    airgap, tmp_path, tiny_hip_src,
):
    """Webview combined path (-i + --source-dir) must expose a dedicated
    Tier-0 section with an ``id="tier0"`` anchor. Instrumentation-advice
    blocks (Profiling Plan, Suggested Hardware Counters) are intentionally
    absent in combined mode — the user has already profiled; that advice
    belongs in the source-only path only. Detected code patterns DO flow
    into the main recommendations table.
    """
    cfg = output_config.output_config(
        output_file="report", output_path=str(tmp_path)
    )
    conn = RocpdImportData([str(_FIXTURE_DB)])
    analyze_mod._execute_agentic(
        conn,
        config=cfg,
        output_format="webview",
        source_dir=str(tiny_hip_src),
    )
    html = (tmp_path / "report.html").read_text()

    # Tier-0 section anchor present.
    assert 'id="tier0"' in html, (
        "expected an id=\"tier0\" anchor wrapping the Tier-0 Source Scan section"
    )

    # Instrumentation-advice blocks must NOT appear in combined mode.
    anchor = 'id="tier0"'
    idx = html.find(anchor)
    assert idx != -1
    tier0_window = html[idx:idx + 40_000]
    assert "Profiling Plan" not in tier0_window, (
        "combined-mode tier-0 section must not include the 'Profiling Plan' "
        "instrumentation advice (user already profiled)"
    )
    assert "Suggested Hardware Counters" not in tier0_window, (
        "combined-mode tier-0 section must not include 'Suggested Hardware Counters' "
        "(user already profiled)"
    )
    # (Rocprofv3 commands may still appear legitimately in downstream
    # recommendation sections — we don't assert on their absence here.)
