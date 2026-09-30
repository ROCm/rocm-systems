"""Phase 10 ``--advanced`` gate for pragma recommendations.

Default is OFF: ``subtype="pragma"`` recs never render.

Enable via either:

* ``perfxpert analyze --advanced ...``
* ``PERFXPERT_ADVANCED_RECS=1`` env var (CI-friendly)
"""

from __future__ import annotations

import argparse
import os

import pytest

from perfxpert import analyze
from perfxpert.analysis.recommendations import build_pragma_recommendation


# ---------------------------------------------------------------------------
# Argparse propagation
# ---------------------------------------------------------------------------


def _parse_and_process(argv):
    parser = argparse.ArgumentParser()
    process_args = analyze.add_args(parser)
    ns = parser.parse_args(argv)

    class _Conn:
        pass

    return process_args(_Conn(), ns)


def test_advanced_flag_default_off():
    """Default: --advanced is NOT present in kwargs."""
    kwargs = _parse_and_process([])
    assert "advanced" not in kwargs, (
        "--advanced unset should drop from kwargs to preserve "
        "the 'user did not set this' signal."
    )


def test_advanced_flag_explicit_on():
    kwargs = _parse_and_process(["--advanced"])
    assert kwargs.get("advanced") is True


# ---------------------------------------------------------------------------
# Rec filtering (the whole point of the gate)
# ---------------------------------------------------------------------------


def _make_recs():
    """Return a mixed rec list: one regular + one pragma."""
    regular = {
        "priority": "HIGH",
        "category": "Memory Transfer",
        "issue": "memcpy overhead",
        "suggestion": "batch transfers",
        "actions": [],
        "commands": [],
    }
    pragma_rec = build_pragma_recommendation(
        kernel_name="hot_gemm",
        pragma_entry={
            "pragma": "clang_loop_unroll_count",
            "syntax": "#pragma clang loop unroll_count(N)",
            "description": "Partially unroll",
            "factor_sweep": [2, 4, 8],
            "risk": "medium",
            "expected_impact": "1.1x-1.5x",
        },
        source_file="src/hot.hip",
        source_line=42,
    )
    return [regular, pragma_rec]


def test_advanced_default_off_filters_pragma_recs(monkeypatch):
    monkeypatch.delenv("PERFXPERT_ADVANCED_RECS", raising=False)
    recs = _make_recs()
    filtered = analyze._filter_advanced_recs(recs, advanced=False)
    assert len(filtered) == 1
    assert filtered[0]["category"] == "Memory Transfer"
    assert all(r.get("subtype") != "pragma" for r in filtered)


def test_advanced_on_keeps_pragma_recs(monkeypatch):
    monkeypatch.delenv("PERFXPERT_ADVANCED_RECS", raising=False)
    recs = _make_recs()
    filtered = analyze._filter_advanced_recs(recs, advanced=True)
    assert len(filtered) == 2
    subtypes = {r.get("subtype") for r in filtered}
    assert "pragma" in subtypes


# ---------------------------------------------------------------------------
# Env-var resolution (PERFXPERT_ADVANCED_RECS)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_val", ["1", "true", "TRUE", "yes", "on"])
def test_env_override_PERFXPERT_ADVANCED_RECS_on(monkeypatch, env_val):
    monkeypatch.setenv("PERFXPERT_ADVANCED_RECS", env_val)
    assert analyze._resolve_advanced_flag(None) is True
    # Explicit False from argparse should still allow the env var to win.
    # (Policy: env var means "user opted in once globally".)
    assert analyze._resolve_advanced_flag(False) is True


@pytest.mark.parametrize("env_val", ["0", "false", "", "no", "off"])
def test_env_override_PERFXPERT_ADVANCED_RECS_off(monkeypatch, env_val):
    monkeypatch.setenv("PERFXPERT_ADVANCED_RECS", env_val)
    assert analyze._resolve_advanced_flag(None) is False
    assert analyze._resolve_advanced_flag(False) is False


def test_cli_flag_wins_when_env_unset(monkeypatch):
    monkeypatch.delenv("PERFXPERT_ADVANCED_RECS", raising=False)
    assert analyze._resolve_advanced_flag(True) is True
    assert analyze._resolve_advanced_flag(False) is False
    assert analyze._resolve_advanced_flag(None) is False


# ---------------------------------------------------------------------------
# Rec shape — ensure Phase-10 fields are attached correctly
# ---------------------------------------------------------------------------


def test_build_pragma_recommendation_has_required_fields():
    rec = build_pragma_recommendation(
        kernel_name="k",
        pragma_entry={
            "pragma": "clang_loop_unroll_full",
            "syntax": "#pragma clang loop unroll(full)",
            "description": "Fully unroll",
            "factor_sweep": [],
            "risk": "low",
            "expected_impact": "1.05x-1.3x",
        },
        source_file="kernels.hip",
        source_line=17,
    )
    assert rec["subtype"] == "pragma"
    assert rec["source_file"] == "kernels.hip"
    assert rec["source_line"] == 17
    assert rec["pragma_id"] == "clang_loop_unroll_full"
    # Mandatory footer per fence rules.
    assert any("perfxpert diff" in a for a in rec["actions"]), (
        "every pragma card MUST carry the 'Verify with: perfxpert diff' line"
    )
