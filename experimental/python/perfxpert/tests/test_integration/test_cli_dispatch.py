"""Tests for analyze.py CLI dispatch (agentic is the only path).

Regression guards: assert the legacy dispatch symbols stay removed and
that legacy env vars cannot revive them.
"""

from pathlib import Path
from unittest import mock

import pytest

from perfxpert import analyze as analyze_mod


@pytest.fixture
def fake_db(tmp_path):
    import sqlite3
    db = tmp_path / "fake.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE rocpd_kernel_dispatch (
            id INTEGER PRIMARY KEY, name TEXT, duration_ns INTEGER
        );
        INSERT INTO rocpd_kernel_dispatch VALUES (1, 'matmul', 1000);
    """)
    conn.commit()
    conn.close()
    return db


def test_cli_always_runs_agentic(fake_db, monkeypatch):
    """CLI always uses the agentic path; no feature-flag branching remains."""
    monkeypatch.delenv("PERFXPERT_LEGACY", raising=False)  # regression guard
    with mock.patch.object(analyze_mod, "_execute_agentic") as agentic:
        agentic.return_value = 0
        # Use the kwarg name the agentic layer actually reads. The legacy
        # `format=` kwarg was silently dropped in cycle-1 tests
        # (nitpick: misleading even though harmless).
        analyze_mod.execute(input=mock.MagicMock(), output_format="text")
        agentic.assert_called_once()


def test_cli_legacy_flag_is_no_op(fake_db, monkeypatch):
    """Regression guard: the removed PERFXPERT_LEGACY env var must still route agentic."""
    monkeypatch.setenv("PERFXPERT_LEGACY", "1")  # regression guard
    with mock.patch.object(analyze_mod, "_execute_agentic") as agentic:
        agentic.return_value = 0
        analyze_mod.execute(input=mock.MagicMock(), output_format="text")
        agentic.assert_called_once()


def test_legacy_symbols_are_absent():
    """Regression guard: removed legacy symbols must stay gone."""
    assert not hasattr(analyze_mod, "_execute_legacy"), (  # regression guard
        "_execute_legacy was removed during the agentic refactor and must stay gone"
    )
    import importlib
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("perfxpert.ai_analysis")


# -- Fix 2 — formatters wired on the agentic path ---------------------------


class _FakeRootOutput:
    """Minimal stand-in for agents.schemas.RootOutput for format tests."""

    def __init__(
        self,
        narrative: str = "",
        recommendations=None,
        primary_bottleneck: str = "mixed",
        warnings=None,
        metadata=None,
    ) -> None:
        self.narrative = narrative
        self.recommendations = list(recommendations or [])
        self.primary_bottleneck = primary_bottleneck
        self.warnings = list(warnings or [])
        self.metadata = dict(metadata or {})


def test_analyze_markdown_output_has_headings():
    """`--format markdown` must produce real Markdown with an H1 heading
    and at least one bullet list — NOT raw narrative prose."""
    out = analyze_mod._format_agentic_output(
        _FakeRootOutput(
            narrative="Kernel `matmul` dominates runtime.",
            recommendations=[
                {"type": "optimize", "summary": "Enable tensor cores", "priority": "HIGH"}
            ],
            primary_bottleneck="compute",
            warnings=["small sample size"],
        ),
        "markdown",
        database_path="/tmp/fake.db",
    )
    # Must start with an H1 heading (the legacy formatter puts
    # ``# PerfXpert AI Performance Analysis`` at the top).
    assert any(line.startswith("# ") for line in out.splitlines()), (
        f"Markdown output missing H1 heading; got first 200 chars:\n{out[:200]}"
    )
    # Must contain at least one list item (``- `` or ``* ``) — the
    # warnings / recommendations sections emit bullets.
    assert any(line.startswith(("- ", "* ")) for line in out.splitlines()), (
        "Markdown output missing any bullet lists"
    )
    # Narrative must be embedded as prose.
    assert "matmul" in out


def test_analyze_webview_output_is_html():
    """`--format webview` must emit a real HTML document with AMD branding
    in the title/footer, NOT a plaintext narrative."""
    out = analyze_mod._format_agentic_output(
        _FakeRootOutput(
            narrative="Memory-bound workload.",
            recommendations=[],
            primary_bottleneck="memory_transfer",
        ),
        "webview",
        database_path="/tmp/fake.db",
    )
    head = out[:200].lower()
    assert head.startswith("<!doctype") or "<html" in head, (
        f"Webview output is not HTML; first 200 chars:\n{out[:200]}"
    )
    # AMD branding appears in the title or template footer.
    assert "amd" in out.lower() or "AMD" in out
    # Narrative panel must be spliced in.
    assert "memory-bound workload" in out.lower() or "Memory-bound workload" in out


def test_analyze_text_output_is_structured():
    """`--format text` must be structured plaintext (section separators,
    bullets), NOT raw narrative prose."""
    out = analyze_mod._format_agentic_output(
        _FakeRootOutput(
            narrative="Narrative body goes here.",
            recommendations=[{"type": "x", "summary": "do thing"}],
            primary_bottleneck="latency",
            warnings=["tiny workload"],
        ),
        "text",
    )
    # The structured plaintext renderer uses `== section ==` headings
    # and long ``=`` bars.
    assert "== Summary ==" in out or "== Recommendations ==" in out, (
        f"Text output not structured; got:\n{out}"
    )
    assert "=" * 40 in out, "Text output missing horizontal section bars"
    assert "latency" in out.lower()


def test_analyze_json_output_has_schema_keys():
    """`--format json` must contain the documented schema keys, not be an
    ad-hoc dump of an internal object."""
    import json as _json
    out = analyze_mod._format_agentic_output(
        _FakeRootOutput(
            narrative="N",
            recommendations=[{"type": "x"}],
            primary_bottleneck="compute",
            warnings=["w"],
        ),
        "json",
    )
    parsed = _json.loads(out)
    for key in ("narrative", "recommendations", "primary_bottleneck", "warnings", "metadata"):
        assert key in parsed, f"JSON output missing schema key {key!r}: {parsed}"
