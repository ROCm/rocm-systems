#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Compare the 16 former SLOT_LIMIT metrics before vs after Phase 2.

Parses ``rocprof-compute analyze --view table`` logs and writes CSV + HTML.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

SLOT16: list[tuple[str, str]] = [
    ("2.1.0", "VALU FLOPs"),
    ("2.1.17", "vL1D Cache Hit Rate"),
    ("3.1.28", "VL1 Hit"),
    ("4.1.9", "HBM Bandwidth"),
    ("4.2.0", "AI HBM"),
    ("4.2.1", "AI L2"),
    ("4.2.2", "AI L1"),
    ("4.2.3", "AI LDS"),
    ("4.2.4", "Performance (GFLOPs)"),
    ("11.1.0", "VALU FLOPs"),
    ("11.2.1", "IPC (Issued)"),
    ("11.3.0", "FLOPs (Total)"),
    ("15.4.4", "Read Instructions"),
    ("16.1.0", "Hit rate"),
    ("16.3.5", "Cache Hit Rate"),
    ("16.3.7", "Cache Hits"),
]

ROW_RE = re.compile(
    r"│\s*(?P<id>\d+\.\d+\.\d+)\s*│\s*(?P<name>[^│]+?)\s*│\s*(?P<avg>[^│]+?)\s*│"
)


def parse_log(path: Path) -> dict[str, tuple[str, str]]:
    text = path.read_text(errors="replace")
    found: dict[str, tuple[str, str]] = {}
    for match in ROW_RE.finditer(text):
        mid = match.group("id")
        if mid not in found:
            found[mid] = (match.group("name").strip(), match.group("avg").strip())
    return found


def to_float(value: str) -> float | None:
    cleaned = value.strip().replace(",", "")
    if cleaned in {"", "N/A", "None", "-", "nan", "NaN", "?"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def rel_pct(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    if before == 0.0 and after == 0.0:
        return 0.0
    if before == 0.0:
        return None
    return (after - before) / abs(before) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-dir", type=Path, required=True)
    parser.add_argument("--after-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["vcopy", "nbody", "mega_kernel"],
    )
    parser.add_argument("--before-suffix", default="_spp.log")
    parser.add_argument("--after-suffix", default="_spp.log")
    parser.add_argument("--title", default="SLOT_LIMIT-16 before/after Phase 2")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for workload in args.workloads:
        before_path = args.before_dir / f"{workload}{args.before_suffix}"
        after_path = args.after_dir / f"{workload}{args.after_suffix}"
        before = parse_log(before_path) if before_path.exists() else {}
        after = parse_log(after_path) if after_path.exists() else {}
        for mid, expected_name in SLOT16:
            b_name, b_avg = before.get(mid, ("?", "N/A"))
            a_name, a_avg = after.get(mid, ("?", "N/A"))
            b_f = to_float(b_avg)
            a_f = to_float(a_avg)
            rel = rel_pct(b_f, a_f)
            rows.append({
                "workload": workload,
                "metric_id": mid,
                "name": expected_name,
                "before_avg": b_avg,
                "after_avg": a_avg,
                "before_f": b_f,
                "after_f": a_f,
                "rel_pct": rel,
                "before_log_name": b_name,
                "after_log_name": a_name,
                "before_log": str(before_path),
                "after_log": str(after_path),
            })

    csv_path = args.out_dir / "slot16_before_after_phase2.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("workload,metric_id,name,before_avg,after_avg,rel_pct\n")
        for row in rows:
            rel = row["rel_pct"]
            rel_s = f"{rel:.4f}" if isinstance(rel, float) else "N/A"
            handle.write(
                f"{row['workload']},{row['metric_id']},{row['name']},"
                f"{row['before_avg']},{row['after_avg']},{rel_s}\n"
            )

    changed = [
        row
        for row in rows
        if isinstance(row["rel_pct"], float) and abs(row["rel_pct"]) > 1e-6
    ]
    missing = [
        row for row in rows if row["before_avg"] == "N/A" or row["after_avg"] == "N/A"
    ]

    html_path = args.out_dir / "slot16_before_after_phase2.html"
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(args.title)}</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:1.5rem;}",
        "table{border-collapse:collapse;width:100%;font-size:14px;}",
        "th,td{border:1px solid #ccc;padding:4px 8px;text-align:left;}",
        "th{background:#f0f0f0;}",
        "tr.changed{background:#fff8e1;}",
        "tr.missing{background:#ffebee;}",
        ".pos{color:#1b5e20;} .neg{color:#b71c1c;}",
        "</style></head><body>",
        f"<h1>{html.escape(args.title)}</h1>",
        "<p>Before = Phase 1 SPP analyze (pre Phase 2 YAML). "
        "After = Phase 2 <code>COLLECT_SUM</code>/<code>COLLECT_RATIO</code> "
        "parents. Same 16 SLOT_LIMIT metric IDs from the gfx942 impact report.</p>",
        f"<p>Rows: {len(rows)}; changed (|rel|&gt;0): {len(changed)}; "
        f"missing either side: {len(missing)}</p>",
        "<table><thead><tr>",
        "<th>Workload</th><th>ID</th><th>Metric</th>",
        "<th>Before Avg</th><th>After Avg</th><th>Rel %</th>",
        "</tr></thead><tbody>",
    ]
    for row in rows:
        rel = row["rel_pct"]
        cls = ""
        if row["before_avg"] == "N/A" or row["after_avg"] == "N/A":
            cls = "missing"
        elif isinstance(rel, float) and abs(rel) > 1e-6:
            cls = "changed"
        if isinstance(rel, float):
            rel_html = f"<span class='{'pos' if rel >= 0 else 'neg'}'>{rel:.2f}%</span>"
        else:
            rel_html = "N/A"
        lines.append(
            f"<tr class='{cls}'>"
            f"<td>{html.escape(str(row['workload']))}</td>"
            f"<td>{html.escape(str(row['metric_id']))}</td>"
            f"<td>{html.escape(str(row['name']))}</td>"
            f"<td>{html.escape(str(row['before_avg']))}</td>"
            f"<td>{html.escape(str(row['after_avg']))}</td>"
            f"<td>{rel_html}</td></tr>"
        )
    lines.extend(["</tbody></table>", "</body></html>"])
    html_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {html_path}")
    print(f"changed={len(changed)} missing={len(missing)} total={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
