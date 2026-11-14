# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Terminal report: every part is a section with the same banner, indentation and tables."""

from __future__ import annotations

import math
import re
import shlex
import textwrap
from typing import Any, List, Tuple

from bench.analysis import KNOWN_KEYS, MIN_PCT_BASE_NS, is_real, pct

Row = Tuple[str, str, float, str]
Section = Tuple[str, List[str]]

WALL_ROWS: list[Row] = [
    ("work  T1->T2", "work", 1e-6, "ms"),
    ("init  T0->T1", "init", 1e-6, "ms"),
    ("fini  T2->T3", "fini", 1e-6, "ms"),
    ("total T0->T3", "wall", 1e-6, "ms"),
]
CPU_ROWS: list[Row] = [
    ("work  T1->T2", "cpu_work", 1e-6, "ms"),
    ("init  T0->T1", "cpu_init", 1e-6, "ms"),
    ("fini  T2->T3", "cpu_fini", 1e-6, "ms"),
    ("total T0->T3", "cpu", 1e3, "ms"),
]
MEM_ROWS: list[Row] = [("peak RSS", "max_rss_kb", 1 / 1024, "MB")]
GLANCE_ROWS: list[Row] = [
    ("work  wall-clock", "work", 1e-6, "ms"),
    ("work  CPU time", "cpu_work", 1e-6, "ms"),
    ("total wall-clock", "wall", 1e-6, "ms"),
    ("total CPU time", "cpu", 1e3, "ms"),
    ("peak RSS", "max_rss_kb", 1 / 1024, "MB"),
]
UNIT_SUFFIXES = (
    ("_ms", "ms"),
    ("_us", "µs"),
    ("_ns", "ns"),
    ("_kb", "KB"),
    ("_mb", "MB"),
    ("_gb", "GB"),
    ("_pct", "%"),
    ("_bytes", "B"),
    ("_s", "s"),
)

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
YELLOW = "\033[33m"
RED_BOLD = "\033[1;31m"
VERDICT_STYLES = {"REAL": BOLD, "noise": DIM}
VERDICT_RE = re.compile(r"\b(REAL|noise)\b")
UNPAINTED_SECTIONS = ("REPORT", "LEGEND", "NOTES")

COL = 10
UNIT_W = 6
GLANCE_BASE_W = 14
STAT_COLS = ("median", "p95", "mad")
DELTA_COLS = ("Δ median", "Δ %", "Δ p95", "Δ %")
SUM_W = 14  # a sum over hundreds of runs outgrows COL
STATS_COLS = (
    ("count", COL),
    ("sum", SUM_W),
    ("mean", COL),
    ("min", COL),
    ("max", COL),
    ("stddev", COL),
)
STAT_W = 3 * COL
DELTA_W = 5 * COL + 2
GLANCE_TOOL_W = 4 * COL + 2
MIN_WIDTH = 100
TEXT_WRAP = 96

LEGEND = (
    ("wall-clock", "elapsed real time (what a stopwatch shows)"),
    ("CPU time", "time the process spent executing on CPU (user+sys), not waiting"),
    ("work", "T1->T2: the workload itself"),
    ("init", "T0->T1: process and tool startup"),
    ("fini", "T2->T3: tool shutdown and exit"),
    ("total", "T0->T3: the whole process"),
    (
        "median",
        "the middle value of all runs (half of the runs were faster, half slower)",
    ),
    (
        "p95",
        (
            "95th percentile: 95% of the runs were at or below this value, 5% above it; "
            "shows how bad it gets when luck runs out; stable from about 100 runs"
        ),
    ),
    (
        "mad",
        (
            "median absolute deviation: the median of |run - median|, the typical scatter "
            "of the runs; a few slow outliers hardly move it"
        ),
    ),
    ("count", "number of successful runs"),
    ("sum", "all runs added up"),
    (
        "mean",
        "arithmetic average of all runs; unlike the median, a few slow outliers move it",
    ),
    (
        "stddev",
        "sample standard deviation (n-1) around the mean; n/a with fewer than 2 runs",
    ),
    (
        "Δ",
        "value with the tool minus the baseline value; Δ % is that difference relative to the baseline",
    ),
    ("Δ mean %", "(tool mean - baseline mean) / baseline mean x 100"),
    (
        "verdict",
        "REAL if |median difference| > max(2 x standard error of the median difference, 1% of baseline median), otherwise noise",
    ),
)


def metric_unit(name: str) -> str:
    """Unit of an application-reported metric, taken from the suffix of its name."""
    return next((unit for suffix, unit in UNIT_SUFFIXES if name.endswith(suffix)), "")


def _num(value: float, width: int = COL) -> str:
    return f"{'n/a':>{width}}" if math.isnan(value) else f"{value:>{width}.2f}"


def _pct(diff: float, base: float, factor: float, unit: str) -> str:
    floor = MIN_PCT_BASE_NS * 1e-6 / factor if unit == "ms" else 0.0  # 1 ms in raw units
    value = pct(diff, base, floor)
    return f"{'n/a':>{COL}}" if math.isnan(value) else f"{value:>+{COL - 1}.1f}%"


def _verdict(base: dict[str, float], other: dict[str, float]) -> str:
    return f"{'REAL' if is_real(base, other) else 'noise':>{COL + 2}}"


def _joined(label: str, unit: str, groups: list[str], label_w: int) -> str:
    return (
        label.ljust(label_w) + unit.ljust(UNIT_W) + "".join("│" + g + " " for g in groups)
    )


def _frame_header(groups: list[tuple[str, str, int]], label_w: int) -> list[str]:
    """Title row, column-name row and rule for groups of (title, column names, width)."""
    titles = _joined("", "", [t.center(w) for t, _, w in groups], label_w)
    names = _joined("", "unit", [n for _, n, _ in groups], label_w)
    rule = "─" * (label_w + UNIT_W) + "".join("┼" + "─" * (w + 1) for _, _, w in groups)
    return [titles, names, rule]


def _stat_names() -> str:
    return "".join(f"{c:>{COL}}" for c in STAT_COLS)


def _delta_names() -> str:
    return "".join(f"{c:>{COL}}" for c in DELTA_COLS) + f"{'verdict':>{COL + 2}}"


def _stat_cells(stats: dict[str, float], factor: float) -> str:
    return "".join(_num(stats[k] * factor) for k in STAT_COLS)


def _delta_cells(
    b: dict[str, float], o: dict[str, float], factor: float, unit: str
) -> str:
    dm, dp = o["median"] - b["median"], o["p95"] - b["p95"]
    return (
        _num(dm * factor)
        + _pct(dm, b["median"], factor, unit)
        + _num(dp * factor)
        + _pct(dp, b["p95"], factor, unit)
        + _verdict(b, o)
    )


def _detail_row(
    row: Row, summary: dict[str, Any], base: str, tool: str | None, label_w: int
) -> str:
    label, metric, factor, unit = row
    b = summary[base][metric]
    groups = [_stat_cells(b, factor)]
    if tool:
        o = summary[tool][metric]
        groups += [_stat_cells(o, factor), _delta_cells(b, o, factor, unit)]
    return _joined(label, unit, groups, label_w)


def _detail_table(
    rows: list[Row], summary: dict[str, Any], base: str, tool: str | None = None
) -> list[str]:
    """Baseline stats; with a tool also its stats and the overhead."""
    names = [base, tool] if tool else [base]
    present = [r for r in rows if all(r[1] in summary[n] for n in names)]
    if not present:
        return []
    label_w = max(len(r[0]) for r in present) + 2
    groups = [(f"{base} (baseline)", _stat_names(), STAT_W)]
    if tool:
        groups += [
            (tool, _stat_names(), STAT_W),
            (f"{tool} - {base} (overhead)", _delta_names(), DELTA_W),
        ]
    return _frame_header(groups, label_w) + [
        _detail_row(r, summary, base, tool, label_w) for r in present
    ]


def _glance_row(
    row: Row, summary: dict[str, Any], base: str, tools: list[str], label_w: int
) -> str:
    label, metric, factor, unit = row
    b = summary[base][metric]
    groups = [_num(b["median"] * factor, GLANCE_BASE_W)]
    for tool in tools:
        o = summary[tool][metric]
        diff = o["median"] - b["median"]
        groups.append(
            _num(o["median"] * factor)
            + _num(diff * factor)
            + _pct(diff, b["median"], factor, unit)
            + _verdict(b, o)
        )
    return _joined(label, unit, groups, label_w)


def _app_rows(summary: dict[str, Any], base: str) -> list[Row]:
    return [(n, n, 1.0, metric_unit(n)) for n in summary[base] if n not in KNOWN_KEYS]


def _glance_table(summary: dict[str, Any], base: str, tools: list[str]) -> list[str]:
    rows = [
        r
        for r in GLANCE_ROWS + _app_rows(summary, base)
        if all(r[1] in summary[n] for n in [base] + tools)
    ]
    label_w = max(len(r[0]) for r in rows) + 2
    names = f"{'median':>{COL}}{'Δ':>{COL}}{'Δ %':>{COL}}{'verdict':>{COL + 2}}"
    groups = [(f"{base} (baseline)", f"{'median':>{GLANCE_BASE_W}}", GLANCE_BASE_W)]
    groups += [(f"{t} vs {base}", names, GLANCE_TOOL_W) for t in tools]
    return _frame_header(groups, label_w) + [
        _glance_row(r, summary, base, tools, label_w) for r in rows
    ]


def _stats_cells(s: dict[str, float], factor: float) -> str:
    values = (s["sum"], s["mean"], s["min"], s["max"], s["stddev"])
    return f"{s['n']:>{COL}d}" + "".join(
        _num(v * factor, w) for v, (_, w) in zip(values, STATS_COLS[1:])
    )


def _stats_table(summary: dict[str, Any], base: str, tools: list[str]) -> list[str]:
    """count/sum/mean/min/max/stddev, one line per command (baseline first); with tools Δ mean %."""
    names = [base] + tools
    rows = [
        r
        for r in GLANCE_ROWS + _app_rows(summary, base)
        if all(r[1] in summary[n] for n in names)
    ]
    label_w = max(len(r[0]) for r in rows) + 2
    cmd_w = max(len(n) for n in names + ["command"]) + 2
    columns = "command".ljust(cmd_w) + "".join(f"{c:>{w}}" for c, w in STATS_COLS)
    groups = [("per command", columns, cmd_w + sum(w for _, w in STATS_COLS))]
    if tools:
        groups.append((f"vs {base}", f"{'Δ mean %':>{COL}}", COL))
    lines = _frame_header(groups, label_w)
    for label, metric, factor, unit in rows:
        b = summary[base][metric]
        for i, name in enumerate(names):
            s = summary[name][metric]
            cells = [name.ljust(cmd_w) + _stats_cells(s, factor)]
            if tools:
                cells.append(
                    _pct(s["mean"] - b["mean"], b["mean"], factor, unit)
                    if i
                    else " " * COL
                )
            lines.append(
                _joined(label if i == 0 else "", unit if i == 0 else "", cells, label_w)
            )
    return lines


def _key_values(pairs: list[tuple[str, str]]) -> list[str]:
    width = max(len(k) for k, _ in pairs) + 3
    return [f"  {k.ljust(width)}{v}" for k, v in pairs]


def _ok_count(doc: dict[str, Any], name: str) -> tuple[int, int]:
    runs = doc["commands"][name]["runs"]
    ok = sum(1 for r in runs if r["error"] is None and r["exit"] == 0)
    return ok, len(runs)


def _setup_section(doc: dict[str, Any]) -> list[str]:
    meta = doc["meta"]
    pairs = [
        ("machine", str(meta["env"].get("cpu"))),
        (
            "runs",
            f"{meta['repetitions']} per command, {meta['warmup']} warm-up (mode: {meta['mode']})",
        ),
    ]
    for i, name in enumerate(doc["commands"]):
        ok, total = _ok_count(doc, name)
        role = "baseline" if i == 0 else "tool"
        pairs.append(
            (
                "commands" if i == 0 else "",
                (
                    f"{name}  {role:<8}  {ok}/{total} ok   "
                    f"{shlex.join(doc['commands'][name]['command'])}"
                ),
            )
        )
    for i, (name, version) in enumerate(meta.get("tool_versions", {}).items()):
        pairs.append(("versions" if i == 0 else "", f"{name}: {version or 'unknown'}"))
    return _key_values(pairs)


def _wrapped(items: list[tuple[str, str]], bullet: str) -> list[str]:
    lines: list[str] = []
    for term, text in items:
        head = f"  {bullet}{term.ljust(12) if term else ''}"
        parts = textwrap.wrap(text, TEXT_WRAP) or [""]
        lines.append(head + parts[0])
        lines += [" " * len(head) + p for p in parts[1:]]
    return lines


def _problem_lines(doc: dict[str, Any], skipped: list[str]) -> list[str]:
    counts: dict[tuple[str, str], int] = {}
    for e in doc["errors"]:
        counts[(e["command"], e["error"])] = counts.get((e["command"], e["error"]), 0) + 1
    lines = [f"! {w}" for w in doc["checks"]["warnings"]]
    lines += [f"✗ {cmd}: {n} run(s) failed: {msg}" for (cmd, msg), n in counts.items()]
    lines += [
        f"✗ '{name}' has no successful runs and is left out of the tables"
        for name in skipped
    ]
    return lines


def _metric_sections(
    usable: dict[str, Any], base: str, tools: list[str]
) -> list[Section]:
    """With tools: the glance table, then per tool one table per group; else baseline-only tables."""
    groups = (
        ("WALL-CLOCK TIME", WALL_ROWS),
        ("CPU TIME", CPU_ROWS),
        ("MEMORY", MEM_ROWS),
        ("APP-REPORTED METRICS", _app_rows(usable, base)),
    )
    sections: list[Section] = []
    if tools:
        sections.append(
            (
                "AT A GLANCE (median per command, tool vs baseline)",
                _glance_table(usable, base, tools),
            )
        )
    sections.append(("STATISTICS (mean-based)", _stats_table(usable, base, tools)))
    for tool in tools or [None]:
        suffix = f" · {tool} vs {base}" if len(tools) > 1 else ""
        sections += [
            (title + suffix, _detail_table(rows, usable, base, tool))
            for title, rows in groups
        ]
    return [(title, table) for title, table in sections if table]


def _sections(doc: dict[str, Any]) -> list[Section]:
    usable = {n: s for n, s in doc["summary"].items() if s}
    base = next(iter(doc["summary"]), None)
    sections: list[Section] = [("REPORT", _setup_section(doc))]
    skipped = [n for n in doc["summary"] if n not in usable and n != base]
    if base in usable:
        sections += _metric_sections(usable, base, [n for n in usable if n != base])
    else:
        sections.append(
            (
                "RESULT",
                [f"  No successful runs of the baseline '{base}': nothing to compare."],
            )
        )
    problems = _problem_lines(doc, skipped)
    if problems:
        sections.append(("WARNINGS", [f"  {p}" for p in problems]))
    sections.append(("LEGEND", _wrapped(list(LEGEND), "")))
    sections.append(("NOTES", _wrapped([("", n) for n in doc["limitations"]], "• ")))
    return sections


def banner(title: str, width: int) -> str:
    head = f"══ {title} "
    return head + "═" * max(3, width - len(head))


def _paint_line(title: str, line: str) -> str:
    """ANSI-style one line of a section; applied after layout so column widths stay exact."""
    if title == "WARNINGS":
        return f"{RED_BOLD if line.lstrip().startswith('✗') else YELLOW}{line}{RESET}"
    if title == "RESULT":
        return f"{RED_BOLD}{line}{RESET}"
    if title in UNPAINTED_SECTIONS:
        return line
    return VERDICT_RE.sub(
        lambda m: f"{VERDICT_STYLES[m.group(1)]}{m.group(1)}{RESET}", line
    )


def format_text(doc: dict[str, Any], color: bool = False) -> str:
    """Render the report for a terminal. color: paint problems (red ✗, yellow !) and verdicts (REAL bold, noise dim)."""
    sections = _sections(doc)
    width = max([MIN_WIDTH] + [len(line) for _, lines in sections for line in lines])
    out: list[str] = []
    for title, lines in sections:
        shown = [_paint_line(title, line) for line in lines] if color else lines
        out += [banner(title, width)] + shown + ["", ""]
    return "\n".join(out).rstrip("\n")
