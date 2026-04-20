"""Tests for `perfxpert.tools.analyze_run.run_root_analysis` (Blocker 1).

The tool is a thin aggregator that wraps `agents.runtime.build_session
+ run_root` so every backend (opencode / claude / codex / gemini TUIs)
can invoke the same Root → Analysis → Recommendation decision hierarchy
the in-process `perfxpert analyze` path uses.

Tests here assert:

1. The tool is registered with `ToolClass.READ_ONLY` and discoverable
   via the MCP registry (so the TUI backends see it).
2. The tool returns a dict with the documented RootOutput schema keys
   (narrative, primary_bottleneck, recommendations, warnings, metadata).
3. Airgap mode is honored end-to-end (no provider call even when the
   caller passes a provider name).
4. The tool is callable without a real DB — which is important for the
   MCP tool schema being discoverable without runtime side-effects.
"""

from __future__ import annotations

from unittest import mock

import pytest

from perfxpert.tools import analyze_run
from perfxpert.tools._class import ToolClass


def test_tool_is_read_only() -> None:
    """Same-brain aggregators must still honor §5.8 threat-model:
    READ_ONLY means the MCP server can expose it.
    """
    assert analyze_run.run_root_analysis.__tool_class__ is ToolClass.READ_ONLY


def test_tool_is_in_mcp_registry() -> None:
    """Regression guard — Blocker 1 depends on this tool being visible
    via the MCP server so backends can discover + call it."""
    from mcp_server._registry import discover_read_only_tools

    reg = discover_read_only_tools()
    assert "analyze_run.run_root_analysis" in reg, sorted(reg)


def test_run_root_analysis_returns_schema_shaped_dict() -> None:
    """The happy path — mocked `run_root` returns a RootOutput stand-in;
    the wrapper reshapes it into a dict with the documented keys."""
    fake_out = mock.MagicMock()
    # Attribute access used by the fallback path.
    fake_out.model_dump.return_value = {
        "narrative": "The kernel `matmul` dominates runtime.",
        "recommendations": [{"type": "optimize", "summary": "enable FMA"}],
        "primary_bottleneck": "compute",
        "warnings": ["small sample size"],
        "metadata": {"db_path": "/tmp/fake.db"},
    }

    fake_session = mock.MagicMock()
    fake_session.session_id = "sess-unit-1"
    fake_session.run_root.return_value = fake_out

    with mock.patch.object(
        analyze_run,
        "_",
        create=True,  # noqa: SIM115 — sentinel so we patch runtime below
    ):
        pass
    # Patch the runtime.build_session + run_root path.
    with mock.patch("perfxpert.agents.runtime.build_session", return_value=fake_session):
        result = analyze_run.run_root_analysis(
            user_query="analyze this trace",
            database_path="/tmp/fake.db",
            airgap=True,
        )

    for key in ("narrative", "primary_bottleneck", "recommendations", "warnings", "metadata"):
        assert key in result, f"result missing {key!r}: {sorted(result)}"

    assert result["primary_bottleneck"] == "compute"
    assert isinstance(result["recommendations"], list)


def test_run_root_analysis_honors_airgap_flag() -> None:
    """airgap=True must flow through to build_session so the session
    never hits a live provider, even if ``provider`` is also set."""
    captured = {}

    def _fake_build(*, provider=None, session_id=None, airgap=None):
        captured["provider"] = provider
        captured["airgap"] = airgap
        sess = mock.MagicMock()
        sess.session_id = session_id or "fake-session"
        sess.run_root.return_value = mock.MagicMock(
            model_dump=lambda: {
                "narrative": "",
                "recommendations": [],
                "primary_bottleneck": "mixed",
                "warnings": [],
                "metadata": {},
            }
        )
        return sess

    with mock.patch("perfxpert.agents.runtime.build_session", side_effect=_fake_build):
        analyze_run.run_root_analysis(
            user_query="anything",
            provider="anthropic",
            airgap=True,
        )

    # When the user explicitly sets airgap=True we forward it as True; the
    # runtime then skips all provider resolution.
    assert captured["airgap"] is True


def test_run_root_analysis_without_db_or_source_still_runs() -> None:
    """A call with only a query (no db, no source) must not crash — the
    MCP tool schema is routinely discovered without a real trace on disk.
    """
    fake_session = mock.MagicMock()
    fake_session.session_id = "sess-empty"
    fake_session.run_root.return_value = mock.MagicMock(
        model_dump=lambda: {
            "narrative": "no data — airgap",
            "recommendations": [],
            "primary_bottleneck": "data_insufficient",
            "warnings": [],
            "metadata": {},
        }
    )
    with mock.patch("perfxpert.agents.runtime.build_session", return_value=fake_session):
        result = analyze_run.run_root_analysis(user_query="hello", airgap=True)
    assert result["primary_bottleneck"] == "data_insufficient"
