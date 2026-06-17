"""Tests for perfxpert.tools.pragma (Phase 10 advanced specialist).

Covers the three READ_ONLY MCP tools:
  * lookup_pragmas
  * explain_pragma
  * suggest_pragmas_for_kernel

Plus the Amdahl gate, Triton skip, YAML factor_sweep invariance.
"""

from __future__ import annotations

import pytest

from perfxpert.tools import pragma
from perfxpert.tools._class import ToolClass


# -- lookup_pragmas -------------------------------------------------------

def test_lookup_pragmas_gpu_only_returns_allowlist():
    entries = pragma.lookup_pragmas(gpu_only=True)
    assert len(entries) == 3, (
        "Phase 10 spec: exactly 3 allowlisted GPU-applicable pragmas."
    )
    for e in entries:
        assert e["gpu_applicable"] is True
        assert e["allowlist"] is True
    pragma_ids = {e["pragma"] for e in entries}
    assert pragma_ids == {
        "clang_loop_unroll_full",
        "clang_loop_unroll_count",
        "clang_loop_unroll_disable",
    }


def test_lookup_pragmas_not_gpu_only_returns_all():
    entries = pragma.lookup_pragmas(gpu_only=False)
    assert len(entries) >= 10, (
        "Spec: 3 allowlisted + 7 explicitly rejected = 10 entries total."
    )
    pragma_ids = {e["pragma"] for e in entries}
    # Explicitly rejected entries the fence must *see* to refuse.
    for rejected in (
        "clang_loop_vectorize",
        "clang_loop_vectorize_width",
        "clang_loop_interleave",
        "clang_loop_interleave_count",
        "clang_loop_distribute",
        "clang_loop_pipeline",
        "clang_loop_vectorize_predicate",
    ):
        assert rejected in pragma_ids, f"missing rejected entry: {rejected}"


# -- explain_pragma -------------------------------------------------------

def test_explain_pragma_known():
    info = pragma.explain_pragma("clang_loop_unroll_count")
    assert info["pragma"] == "clang_loop_unroll_count"
    assert info["factor_sweep"] == [2, 4, 8]
    assert info["gpu_applicable"] is True
    assert info["allowlist"] is True


def test_explain_pragma_unknown_raises_KeyError():
    with pytest.raises(KeyError):
        pragma.explain_pragma("clang_loop_totally_made_up")


# -- suggest_pragmas_for_kernel (Amdahl gate + Triton skip) ---------------

def test_suggest_pragmas_for_kernel_honors_amdahl():
    """Kernel with <5% total time → empty list."""
    signals = {"hotspot_pct": 2.0, "loop_trip_count": 16}
    assert pragma.suggest_pragmas_for_kernel("fused_add", signals) == []


def test_suggest_pragmas_skips_triton_kernel():
    """Source path contains .triton/ → tool skips the kernel."""
    signals = {
        "hotspot_pct": 25.0,
        "source_path": "/home/user/.triton/cache/fused_matmul.hip",
        "loop_trip_count": 16,
    }
    assert pragma.suggest_pragmas_for_kernel("fused_matmul", signals) == []


# -- suggest_pragmas_for_kernel (rule firing) -----------------------------

def test_suggest_pragmas_unroll_full_when_literal_trip_count():
    signals = {"loop_trip_count": 16, "hotspot_pct": 25}
    out = pragma.suggest_pragmas_for_kernel("gemm_kernel", signals)
    assert len(out) == 1
    rec = out[0]
    assert rec["pragma"] == "clang_loop_unroll_full"
    assert rec["kernel_name"] == "gemm_kernel"
    assert rec["triggered_by"] == "literal_trip_count_on_hotspot"
    assert rec["factor_sweep"] == []  # full-unroll has no factor


def test_suggest_pragmas_unroll_count_when_valu_stalled():
    signals = {
        "valu_util_pct": 0.6,
        "vgpr_per_thread": 48,
        "loop_body_size": 12,
        "hotspot_pct": 15,
    }
    out = pragma.suggest_pragmas_for_kernel("stencil", signals)
    ids = {r["pragma"] for r in out}
    assert "clang_loop_unroll_count" in ids
    unroll_count_rec = [r for r in out if r["pragma"] == "clang_loop_unroll_count"][0]
    assert unroll_count_rec["factor_sweep"] == [2, 4, 8], (
        "YAML-locked factor sweep — the agent must NEVER invent other values."
    )


def test_suggest_pragmas_unroll_disable_when_vgpr_pressure():
    # VGPR at 80% of arch max + scratch > 0 fires the disable rule.
    signals = {
        "hotspot_pct": 20,
        "vgpr_per_thread": 208,      # 208 / 256 = 0.8125 >= 0.80
        "arch_max_vgpr": 256,
        "scratch_size": 64,
        "waves_per_eu": 1,
    }
    out = pragma.suggest_pragmas_for_kernel("heavy_matmul", signals)
    ids = {r["pragma"] for r in out}
    assert "clang_loop_unroll_disable" in ids


# -- tool class enforcement -----------------------------------------------

def test_all_three_tools_are_read_only():
    assert pragma.lookup_pragmas.__tool_class__ == ToolClass.READ_ONLY
    assert pragma.explain_pragma.__tool_class__ == ToolClass.READ_ONLY
    assert pragma.suggest_pragmas_for_kernel.__tool_class__ == ToolClass.READ_ONLY
