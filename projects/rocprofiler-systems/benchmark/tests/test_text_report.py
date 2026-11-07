# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
import re
from typing import Any

from bench import text_report

ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _stats(median: float) -> dict[str, float]:
    return {
        "median": median,
        "p95": median * 1.1,
        "mad": median * 0.01,
        "n": 100,
        "sum": median * 100,
        "mean": median,
        "stddev": median * 0.02,
        "min": median * 0.9,
        "max": median * 1.2,
    }


def _doc() -> dict[str, Any]:
    keys = [
        "wall",
        "work",
        "init",
        "fini",
        "cpu_work",
        "cpu_init",
        "cpu_fini",
        "cpu",
        "max_rss_kb",
    ]
    runs = [{"error": None, "exit": 0}]
    return {
        "meta": {"env": {"cpu": "test"}, "repetitions": 1, "warmup": 0, "mode": "custom"},
        "commands": {
            "base": {"runs": runs, "command": ["a"]},
            "tool": {"runs": runs, "command": ["b"]},
        },
        "summary": {
            "base": {k: _stats(100.0) for k in keys},
            "tool": {k: _stats(120.0) for k in keys},
        },
        "errors": [{"command": "tool", "error": "boom"}],
        "checks": {"warnings": ["w1"]},
        "limitations": ["lim"],
    }


def test_color_only_adds_ansi_codes() -> None:
    doc = _doc()
    colored = text_report.format_text(doc, color=True)

    assert ANSI_RE.search(colored)
    assert ANSI_RE.sub("", colored) == text_report.format_text(doc)


def test_plain_output_has_no_ansi_codes() -> None:
    assert not ANSI_RE.search(text_report.format_text(_doc()))


def _section(text: str, title: str) -> str:
    """Body of the section whose banner starts with '══ <title> '."""
    return text.split(f"══ {title} ", 1)[1].split("\n══ ", 1)[0]


def test_statistics_section_has_all_stats_and_mean_overhead() -> None:
    body = _section(text_report.format_text(_doc()), "STATISTICS (mean-based)")
    header = body.splitlines()[2]
    for column in ("command", "count", "sum", "mean", "min", "max", "stddev", "Δ mean %"):
        assert column in header
    tool_wall = next(
        line for line in body.splitlines() if "│tool " in line and "+20.0%" in line
    )
    assert "100" in tool_wall  # count


def test_statistics_section_shows_na_for_undefined_stddev() -> None:
    def na_count(doc: dict[str, Any]) -> int:
        return _section(text_report.format_text(doc), "STATISTICS (mean-based)").count(
            "n/a"
        )

    doc = _doc()
    before = na_count(doc)  # Δ mean % is already n/a below the 1 ms floor
    doc["summary"]["tool"]["wall"]["stddev"] = math.nan
    assert na_count(doc) == before + 1


def test_statistics_section_without_tool_has_no_overhead_column() -> None:
    doc = _doc()
    del doc["summary"]["tool"], doc["commands"]["tool"]
    body = _section(text_report.format_text(doc), "STATISTICS (mean-based)")
    assert "stddev" in body and "Δ mean %" not in body
