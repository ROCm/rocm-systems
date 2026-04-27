###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""Regression guards for large PerfXpert implementation units.

The current tree still has legacy formatter and backend hot spots. Keep their
inventory explicit so future changes can shrink them deliberately without
allowing new 120+ line functions/classes or silent growth in the existing ones.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = (PACKAGE_ROOT / "perfxpert", PACKAGE_ROOT / "mcp_server")
MAX_UNIT_LINES = 120


@dataclass(frozen=True)
class Unit:
    path: str
    kind: str
    name: str
    line: int
    lines: int

    @property
    def key(self) -> tuple[str, str]:
        return (self.path, self.name)


KNOWN_HOTSPOT_BUDGETS = {
    ("perfxpert/__main__.py", "main"): 231,
    ("perfxpert/agents/analysis.py", "_extract_hw_metrics"): 147,
    ("perfxpert/agents/runtime.py", "AnalysisSession"): 141,
    ("perfxpert/analysis/att.py", "analyze_thread_trace"): 198,
    ("perfxpert/analysis/core.py", "analyze_kernel_resources"): 120,
    ("perfxpert/analysis/payload.py", "scan_tier0_sources"): 216,
    ("perfxpert/analysis/payload.py", "build_analysis_payload"): 229,
    ("perfxpert/analysis/pmc.py", "_split_pmc_into_passes"): 138,
    ("perfxpert/analysis/recommendations.py", "_filter_rec_commands"): 139,
    ("perfxpert/analysis/recommendations.py", "generate_recommendations"): 748,
    ("perfxpert/analyze.py", "add_args"): 275,
    ("perfxpert/analyze.py", "_format_agentic_output"): 446,
    ("perfxpert/analyze.py", "_execute_agentic"): 262,
    ("perfxpert/cli/_backend/claude.py", "ClaudeCodeAdapter"): 619,
    ("perfxpert/cli/_backend/codex.py", "CodexAdapter"): 1141,
    ("perfxpert/cli/_backend/codex.py", "install"): 150,
    ("perfxpert/cli/_backend/gemini.py", "GeminiAdapter"): 465,
    ("perfxpert/cli/_gate_hooks/claude.py", "ClaudeGateHook"): 140,
    ("perfxpert/cli/_gate_hooks/gemini.py", "GeminiGateHook"): 141,
    ("perfxpert/cli/_mcp_warmup.py", "warmup_perfxpert_mcp"): 140,
    ("perfxpert/cli/opencode_launcher.py", "main"): 125,
    ("perfxpert/connection.py", "PerfxpertConnection"): 294,
    ("perfxpert/connection.py", "_create_union_views"): 164,
    ("perfxpert/formatters/__init__.py", "format_analysis_output"): 651,
    ("perfxpert/formatters/_att_flamegraph.py", "render_att_flamegraph"): 150,
    ("perfxpert/formatters/_roofline_svg.py", "render_roofline_svg"): 221,
    ("perfxpert/formatters/json_fmt.py", "_format_as_json"): 171,
    ("perfxpert/formatters/markdown.py", "_format_as_markdown"): 338,
    ("perfxpert/formatters/markdown.py", "_format_tier0_markdown"): 178,
    ("perfxpert/formatters/webview.py", "_format_as_webview"): 1175,
    ("perfxpert/formatters/webview.py", "_format_diff_webview"): 234,
    ("perfxpert/formatters/webview.py", "_format_tier0_webview"): 480,
    ("perfxpert/runtime/gate_cascade.py", "evaluate"): 123,
    ("perfxpert/runtime/gate_cascade.py", "run_gate_cascade"): 225,
    ("perfxpert/tools/predict_impact.py", "predict_change_impact"): 164,
    ("perfxpert/tools/roofline.py", "plot_points"): 166,
    ("perfxpert/tools/tasks.py", "TaskStore"): 153,
    ("perfxpert/tools/trace_diff.py", "diff_runs"): 125,
}


def _iter_units() -> list[Unit]:
    units: list[Unit] = []
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(PACKAGE_ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            for node in ast.walk(tree):
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    continue
                if node.end_lineno is None:
                    continue
                units.append(
                    Unit(
                        path=rel,
                        kind=type(node).__name__,
                        name=node.name,
                        line=node.lineno,
                        lines=node.end_lineno - node.lineno + 1,
                    )
                )
    return units


def test_large_units_are_registered_for_refactor_tracking():
    large_units = [u for u in _iter_units() if u.lines >= MAX_UNIT_LINES]

    unregistered = [
        f"{u.path}:{u.line} {u.kind} {u.name} spans {u.lines} lines"
        for u in large_units
        if u.key not in KNOWN_HOTSPOT_BUDGETS
    ]

    assert not unregistered, (
        "New large functions/classes must either be split before merge or "
        "added here with an explicit refactor budget:\n" + "\n".join(unregistered)
    )


def test_registered_large_units_do_not_grow():
    by_key = {u.key: u for u in _iter_units()}
    grown = []
    missing = []
    for key, budget in KNOWN_HOTSPOT_BUDGETS.items():
        unit = by_key.get(key)
        if unit is None:
            missing.append(f"{key[0]}::{key[1]}")
            continue
        if unit.lines > budget:
            grown.append(
                f"{unit.path}:{unit.line} {unit.name} grew to "
                f"{unit.lines} lines (budget {budget})"
            )

    assert not missing, (
        "Remove renamed/split units from KNOWN_HOTSPOT_BUDGETS:\n"
        + "\n".join(missing)
    )
    assert not grown, (
        "Large implementation units grew; split helper sections or adjust the "
        "budget in the same review:\n" + "\n".join(grown)
    )
