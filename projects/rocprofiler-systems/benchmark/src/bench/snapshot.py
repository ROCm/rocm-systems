# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Snapshot = one directory with snapshot.json (machine + per-scenario summaries) and results/.

`bench-snapshot compare A B` shows what changed between two snapshots.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from bench.analysis import is_real, pct

SNAPSHOT_FILE = "snapshot.json"
RESULTS_DIR = "results"
SCHEMA_VERSION = 1
# (summary key, label, scale to display unit, unit)
COMPARED_METRICS = (
    ("wall", "wall", 1e-6, "ms"),
    ("cpu", "cpu", 1e3, "ms"),
    ("max_rss_kb", "peak RSS", 1 / 1024, "MB"),
)


def new_snapshot_dir(root: str, now: datetime) -> Path:
    """<root>/snapshot-<YYYYmmdd-HHMMSS>; fails if it exists so nothing is overwritten."""
    path = Path(root) / f"snapshot-{now:%Y%m%d-%H%M%S}"
    (path / RESULTS_DIR).mkdir(parents=True)
    return path


def scenario_entry(
    name: str, scenario_path: str, result_path: Path, exit_code: int, doc: dict[str, Any]
) -> dict[str, Any]:
    """Compact per-scenario record; the full measurements stay in the result file."""
    return {
        "name": name,
        "scenario": scenario_path,
        "result": f"{RESULTS_DIR}/{result_path.name}",
        "exit": exit_code,
        "reference": doc["meta"]["reference"],
        "tool_versions": doc["meta"].get("tool_versions", {}),
        "summary": doc["summary"],
        "comparison": doc["comparison"],
    }


def write_snapshot(
    directory: Path,
    now: datetime,
    machine: dict[str, Any],
    scenarios: list[dict[str, Any]],
) -> Path:
    doc = {
        "schema": SCHEMA_VERSION,
        "created": now.astimezone().isoformat(timespec="seconds"),
        "system": machine,
        "scenarios": scenarios,
    }
    path = directory / SNAPSHOT_FILE
    path.write_text(json.dumps(doc, indent=2))
    return path


def load_snapshot(path: str) -> dict[str, Any]:
    """Accept the snapshot directory or its snapshot.json."""
    file = Path(path) / SNAPSHOT_FILE if Path(path).is_dir() else Path(path)
    try:
        doc = json.loads(file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if doc.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported snapshot schema {doc.get('schema')!r}")
    return doc


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Nested dict/list -> {'cpu.model': ..., 'gpus.0.name': ...}."""
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, list):
        items = enumerate(value)
    else:
        return {prefix: value}
    out: dict[str, Any] = {}
    for key, item in items:
        out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    return out


def system_diff(a: Any, b: Any) -> list[tuple[str, Any, Any]]:
    fa, fb = flatten(a), flatten(b)
    return [
        (k, fa.get(k), fb.get(k))
        for k in sorted(set(fa) | set(fb))
        if fa.get(k) != fb.get(k)
    ]


def compare_command(
    name: str, sum_a: dict[str, Any], sum_b: dict[str, Any]
) -> list[dict[str, Any]]:
    """Rows for one command (same name in both snapshots) over COMPARED_METRICS."""
    rows = []
    for key, label, scale, unit in COMPARED_METRICS:
        if key not in sum_a or key not in sum_b:
            continue
        a, b = sum_a[key], sum_b[key]
        diff = b["median"] - a["median"]
        rows.append(
            {
                "command": name,
                "metric": label,
                "unit": unit,
                "a": a["median"] * scale,
                "b": b["median"] * scale,
                "pct": pct(diff, a["median"]),
                "real": is_real(a, b),
                "worse": diff > 0,
            }
        )
    return rows


def compare_snapshots(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    by_a = {s["name"]: s for s in a["scenarios"]}
    by_b = {s["name"]: s for s in b["scenarios"]}
    rows: dict[str, list[dict[str, Any]]] = {}
    for name in sorted(set(by_a) & set(by_b)):
        sa, sb = by_a[name]["summary"], by_b[name]["summary"]
        rows[name] = [
            row
            for cmd in sa
            if cmd in sb
            for row in compare_command(cmd, sa[cmd], sb[cmd])
        ]
    return {
        "system": system_diff(a["system"], b["system"]),
        "tool_versions": [
            (f"{s}.{k}", va, vb)
            for s in sorted(set(by_a) & set(by_b))
            for k, va, vb in system_diff(
                by_a[s]["tool_versions"], by_b[s]["tool_versions"]
            )
        ],
        "only_in_a": sorted(set(by_a) - set(by_b)),
        "only_in_b": sorted(set(by_b) - set(by_a)),
        "failed_in_b": [
            name
            for name in sorted(set(by_a) & set(by_b))
            # a command whose runs all failed has an empty summary, so it yields no rows
            if by_b[name]["exit"] != 0
            or any(
                summary and not by_b[name]["summary"].get(cmd)
                for cmd, summary in by_a[name]["summary"].items()
            )
        ],
        "scenarios": rows,
    }


def format_comparison(a: dict[str, Any], b: dict[str, Any], cmp: dict[str, Any]) -> str:
    lines = [f"A: {a['created']}   B: {b['created']}", ""]
    for title, diffs in (
        ("SYSTEM DIFFERENCES (results may not be comparable)", cmp["system"]),
        ("TOOL VERSION DIFFERENCES", cmp["tool_versions"]),
    ):
        if diffs:
            lines += [title, *(f"  {k}: {va} -> {vb}" for k, va, vb in diffs), ""]
    for label, names in (
        ("only in A", cmp["only_in_a"]),
        ("only in B", cmp["only_in_b"]),
        (
            "with failed runs or missing commands in B (counted as REGRESSION)",
            cmp["failed_in_b"],
        ),
    ):
        if names:
            lines += [f"Scenarios {label}: {', '.join(names)}", ""]
    for name, rows in cmp["scenarios"].items():
        lines.append(f"SCENARIO {name}")
        lines.append(
            f"  {'command':<16}{'metric':<10}{'A':>12}{'B':>12}{'Δ %':>9}  verdict"
        )
        for r in rows:
            verdict = (
                ("REGRESSION" if r["worse"] else "improved") if r["real"] else "noise"
            )
            pct_text = "n/a" if r["pct"] != r["pct"] else f"{r['pct']:+.1f}"
            lines.append(
                f"  {r['command']:<16}{r['metric']:<10}{r['a']:>9.2f} {r['unit']:<2}"
                f"{r['b']:>9.2f} {r['unit']:<2}{pct_text:>9}  {verdict}"
            )
        lines.append("")
    lines.append(
        "verdict: REAL change only if |Δ median| > max(2 x standard error, 1% of A); "
        "same rule as the single-session report"
    )
    return "\n".join(lines)


def has_regression(cmp: dict[str, Any]) -> bool:
    return bool(cmp["failed_in_b"]) or any(
        r["real"] and r["worse"] for rows in cmp["scenarios"].values() for r in rows
    )


def load_suite(path: str) -> list[str]:
    """Scenario paths of a suite file {"scenarios": ["a.json", ...]}, relative to the suite file."""
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    names = doc.get("scenarios") if isinstance(doc, dict) else None
    if not names or not all(isinstance(n, str) and n for n in names):
        raise ValueError(
            f'{path}: expected {{"scenarios": ["a.json", ...]}} with at least one entry'
        )
    return [str(Path(path).resolve().parent / n) for n in names]
