###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""Tests for the ``--baseline`` splice in ``perfxpert analyze`` (Confluence row #7).

These exercise the splice helper directly (no full analyze run) so the
test stays fast AND does not depend on the live analysis pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from perfxpert.analyze import _splice_baseline_diff


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
BASELINE_DB = FIXTURES / "regression_baseline.db"
REGRESSED_DB = FIXTURES / "regression_tail_hurt.db"


@pytest.fixture(scope="module")
def _fixtures_exist():
    missing = [p for p in (BASELINE_DB, REGRESSED_DB) if not p.exists()]
    if missing:
        pytest.skip(f"fixtures missing: {missing}")
    return True


def test_analyze_baseline_splice_adds_diff_section_json(_fixtures_exist):
    """JSON splice adds a top-level ``trace_diff`` key."""
    base_report = {
        "schema_version": "0.3.0",
        "metadata": {"analysis_version": "0.3.0"},
        "summary": {"primary_bottleneck": "compute"},
    }
    spliced = _splice_baseline_diff(
        json.dumps(base_report),
        baseline_db=str(BASELINE_DB),
        new_db=str(REGRESSED_DB),
        output_format="json",
        top_kernels=10,
    )
    parsed = json.loads(spliced)
    assert "trace_diff" in parsed
    # Schema version got bumped to 0.3.1 because trace_diff is present.
    assert parsed["schema_version"] == "0.3.1"
    assert parsed["metadata"]["analysis_version"] == "0.3.1"
    # Preserved existing fields.
    assert parsed["summary"] == {"primary_bottleneck": "compute"}
    # trace_diff contract intact.
    td = parsed["trace_diff"]
    assert td["schema_version"] == "0.3.1"
    assert "wall_delta_pct" in td
    assert "per_kernel" in td


def test_analyze_baseline_splice_markdown_section(_fixtures_exist):
    """Markdown splice appends a ``## Changed vs baseline`` section."""
    base = "# PerfXpert Report\n\nSome body text.\n"
    spliced = _splice_baseline_diff(
        base,
        baseline_db=str(BASELINE_DB),
        new_db=str(REGRESSED_DB),
        output_format="markdown",
        top_kernels=10,
    )
    assert "# PerfXpert Report" in spliced
    assert "## Changed vs baseline" in spliced
    assert "Per-kernel deltas" in spliced


def test_analyze_baseline_splice_text_block(_fixtures_exist):
    base = "PerfXpert Report\n=====\n(body)\n"
    spliced = _splice_baseline_diff(
        base,
        baseline_db=str(BASELINE_DB),
        new_db=str(REGRESSED_DB),
        output_format="text",
        top_kernels=10,
    )
    assert "PerfXpert Report" in spliced
    assert "Changed vs baseline" in spliced


def test_analyze_baseline_splice_webview_before_body_close(_fixtures_exist):
    """Webview splice drops the ``.scard`` before ``</body>``."""
    base = (
        "<!DOCTYPE html><html><head><title>T</title></head>"
        "<body><div class='wrap'>existing</div></body></html>"
    )
    spliced = _splice_baseline_diff(
        base,
        baseline_db=str(BASELINE_DB),
        new_db=str(REGRESSED_DB),
        output_format="webview",
        top_kernels=10,
    )
    assert 'id="trace-diff"' in spliced
    assert spliced.count("</body>") == 1
    # The scard must appear BEFORE </body>.
    idx_scard = spliced.index('id="trace-diff"')
    idx_body = spliced.index("</body>")
    assert idx_scard < idx_body


def test_analyze_baseline_splice_is_defensive_on_bad_baseline():
    """Nonexistent baseline → original report returned unchanged."""
    base = "report"
    spliced = _splice_baseline_diff(
        base,
        baseline_db="/nonexistent/baseline.db",
        new_db="/nonexistent/new.db",
        output_format="text",
        top_kernels=10,
    )
    assert spliced == base
