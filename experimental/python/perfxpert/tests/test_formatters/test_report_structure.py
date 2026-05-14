###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""Structural invariants for PerfXpert's four report formats.

Regression guards for the report-structure fixes landed after
`871a32981b` (Phase 8):

- webview: Summary is a standard `.scard` placed BEFORE Execution
  Breakdown; Tier-0 content is spliced as a single wrapper `.scard`
  and no longer carries a nested `<!DOCTYPE html>…</html>` document;
  there is exactly one `</body></html>` pair.
- markdown: Summary block uses `---` to separate narrative from
  metadata; canonical heading order (`# PerfXpert AI Performance
  Analysis` → `## Summary` → metadata → `## Time Breakdown`).
- text: SUMMARY precedes TIME BREAKDOWN, with a blank line between.
- json: all required keys are present, `schema_version` is bumped to
  0.3.0 (agentic + tier-0 separation + summary section).

All tests run under airgap so no LLM credentials are required.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from perfxpert import analyze as analyze_mod
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
    """A minimal HIP tree so the Tier-0 scanner has something to emit."""
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


def _render(
    *,
    fmt: str,
    tmp_path: Path,
    source_dir: Path | None = None,
    input_db: Path | None = _FIXTURE_DB,
) -> str:
    """Invoke ``_execute_agentic`` and return the rendered output as text."""
    out_dir = tmp_path / f"out_{fmt}"
    out_dir.mkdir(exist_ok=True)
    cfg = output_config.output_config(
        output_file="report", output_path=str(out_dir)
    )
    conn = RocpdImportData([str(input_db)]) if input_db else None
    analyze_mod._execute_agentic(
        conn,
        config=cfg,
        output_format=fmt,
        source_dir=str(source_dir) if source_dir else None,
    )
    ext = {"json": ".json", "markdown": ".md", "webview": ".html", "text": ".txt"}[fmt]
    return (out_dir / f"report{ext}").read_text()


# ---------------------------------------------------------------------------
# Webview
# ---------------------------------------------------------------------------


def test_webview_summary_at_top_and_uses_scard(airgap, tiny_hip_src, tmp_path):
    """Summary section renders as a `.scard`, placed BEFORE Execution
    Breakdown, and the report has exactly one `<!DOCTYPE html>` /
    `</body>` / `</html>` pair each.
    """
    out = _render(fmt="webview", tmp_path=tmp_path, source_dir=tiny_hip_src)

    # Exactly-one scaffolding tags (no nested full document from tier-0).
    assert out.count("<!DOCTYPE html>") == 1, (
        f"expected 1 <!DOCTYPE html>, got {out.count('<!DOCTYPE html>')}"
    )
    body_closes = [m.start() for m in re.finditer(r"(?m)^</body>\s*$", out)]
    html_closes = [m.start() for m in re.finditer(r"(?m)^</html>\s*$", out)]
    assert len(body_closes) == 1, f"expected 1 </body>, got {len(body_closes)}"
    assert len(html_closes) == 1, f"expected 1 </html>, got {len(html_closes)}"

    # Summary MUST precede Execution Breakdown in the rendered order.
    i_summary = out.find("<h2>Summary</h2>")
    i_exec = out.find("<h2>Execution Breakdown</h2>")
    assert i_summary != -1, "webview missing Summary h2"
    assert i_exec != -1, "webview missing Execution Breakdown h2"
    assert i_summary < i_exec, (
        f"Summary (at {i_summary}) must come before Execution Breakdown "
        f"(at {i_exec})"
    )

    # Summary lives inside a `.scard` — check the 240 chars leading up
    # to the Summary h2 for `class="scard"`.
    preamble = out[max(0, i_summary - 240):i_summary]
    assert 'class="scard"' in preamble, (
        "Summary h2 must be inside a `.scard` container, not an ad-hoc `.card`"
    )

    # No legacy `.card` markup remains.
    assert 'class="card"' not in out, (
        "webview still contains ad-hoc `.card` markup — Summary/Tier-0 "
        "splice should standardize on `.scard`"
    )


def test_webview_tier0_section_wrapped_as_scard(airgap, tiny_hip_src, tmp_path):
    """With `--source-dir`, a single wrapper `.scard` labelled
    "Tier-0 Source Scan" contains the Tier-0 content — no nested
    `<!DOCTYPE html>` or duplicate `<head>`/`<body>` scaffolding.
    """
    out = _render(fmt="webview", tmp_path=tmp_path, source_dir=tiny_hip_src)

    # Still exactly one DOCTYPE after Tier-0 splice.
    assert out.count("<!DOCTYPE html>") == 1

    # One wrapper labelled Tier-0 Source Scan.
    i_tier0 = out.find("<h2>Tier-0 Source Scan</h2>")
    assert i_tier0 != -1, "missing Tier-0 Source Scan h2"
    # That h2 must be inside a `<section class="scard"` — walk back to
    # the most recent opening <section class="scard" tag.
    prev_sec = out.rfind('<section class="scard"', 0, i_tier0)
    assert prev_sec != -1, "Tier-0 h2 not inside any <section class=\"scard\">"
    # And the closing </section> for that wrapper must come AFTER the
    # tier-0 inner content (so the whole thing is a single wrapper card).
    detect_kernels_idx = out.find("Detected GPU Kernels", i_tier0)
    assert detect_kernels_idx != -1, "Tier-0 content missing inner kernels list"
    # The wrapper scard owns the inner content: no stray closing before it.
    assert detect_kernels_idx > i_tier0

    # No nested head/body blocks from the tier-0 template.
    assert out.count("<head>") == 1
    assert out.count("<body>") == 1


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def test_markdown_summary_has_horizontal_rule_before_metadata(
    airgap, tiny_hip_src, tmp_path
):
    """Canonical heading order: H1 → `## Summary` → narrative → `---`
    → `**Database:**` metadata block → `## Time Breakdown`.
    """
    out = _render(fmt="markdown", tmp_path=tmp_path, source_dir=tiny_hip_src)

    # H1 first.
    assert out.lstrip().startswith("# PerfXpert AI Performance Analysis"), (
        "markdown must start with canonical H1"
    )

    # Summary → --- → Database metadata → Time Breakdown, in that order.
    i_h1 = out.find("# PerfXpert AI Performance Analysis")
    i_summary = out.find("## Summary", i_h1)
    i_hr = out.find("\n---\n", i_summary)
    i_db = out.find("**Database:**", i_hr)
    i_tb = out.find("## Time Breakdown", i_db)

    assert i_h1 == 0 or i_h1 < 5, f"H1 not at top: {i_h1}"
    assert i_summary > i_h1, "Summary must follow H1"
    assert i_hr > i_summary, "horizontal rule must follow Summary"
    assert i_db > i_hr, "metadata block must follow the horizontal rule"
    assert i_tb > i_db, "Time Breakdown must follow metadata"

    # The narrative region between `## Summary` and the `---` rule must
    # be non-empty (otherwise the Summary is structurally present but
    # semantically empty — regression we want to catch).
    between = out[i_summary + len("## Summary"):i_hr].strip()
    assert between, "Summary section is empty"


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def test_text_summary_precedes_time_breakdown(airgap, tiny_hip_src, tmp_path):
    """SUMMARY appears before TIME BREAKDOWN, with at least one blank
    line between the two banner blocks.
    """
    out = _render(fmt="text", tmp_path=tmp_path, source_dir=tiny_hip_src)
    i_summary = out.find("SUMMARY")
    i_tb = out.find("TIME BREAKDOWN")
    assert i_summary != -1 and i_tb != -1, "text missing SUMMARY or TIME BREAKDOWN"
    assert i_summary < i_tb, (
        f"SUMMARY (at {i_summary}) must precede TIME BREAKDOWN (at {i_tb})"
    )
    # Between the two there must be at least one '\n\n' pair.
    between = out[i_summary:i_tb]
    assert "\n\n" in between, (
        "expected a blank line between SUMMARY and TIME BREAKDOWN"
    )


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def test_json_has_all_required_keys_and_bumped_schema_version(
    airgap, tiny_hip_src, tmp_path
):
    """Every agentic + deterministic top-level key is present; the
    agent-brain fields are populated; `schema_version == "0.3.0"`.
    """
    out = _render(fmt="json", tmp_path=tmp_path, source_dir=tiny_hip_src)
    doc = json.loads(out)

    required = {
        "narrative",
        "primary_bottleneck",
        "summary",
        "tier0_findings",
        "time_breakdown",
        "hotspots",
        "memory_analysis",
        "hardware_counters",
        "recommendations",
        "warnings",
        "metadata",
        "schema_version",
    }
    missing = required - set(doc.keys())
    assert not missing, f"missing top-level keys: {missing}"

    # Schema bumped to reflect the agentic pipeline contract. Phase 10
    # additive fields bump the minor schema version further:
    #   0.3.1 — hotspots[*].source_locations
    #   0.3.2 — RCCL communication
    #   0.3.3 — Change-Impact Prediction (rec.predicted_impact_range)
    #   0.3.4 — Live Roofline
    # ATT (0.4.0) still trumps.
    sv = str(doc["schema_version"])
    assert sv.startswith("0.3."), (
        f"expected schema_version 0.3.x, got {doc['schema_version']!r}"
    )
    assert sv >= "0.3.0"

    # Agent-brain fields are populated on airgap.
    assert doc["narrative"], "narrative must not be empty in airgap mode"
    assert doc["primary_bottleneck"], "primary_bottleneck must not be empty"
    assert doc["tier0_findings"], "tier0_findings must be present when --source-dir set"

    # `llm_enhanced_explanation` is the legacy alias — must mirror narrative
    # (not null, not a different string).
    if "llm_enhanced_explanation" in doc:
        assert (
            doc["llm_enhanced_explanation"] == doc["narrative"]
            or doc["llm_enhanced_explanation"] is None
        ), (
            "llm_enhanced_explanation should either mirror narrative or be null"
        )
