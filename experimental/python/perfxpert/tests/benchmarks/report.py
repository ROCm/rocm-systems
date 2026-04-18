"""Generate nightly benchmark report (HTML + markdown + JSON)."""

from datetime import datetime
from pathlib import Path
from typing import Iterable

from tests.benchmarks.geomean import filter_recommended, weighted_geomean
from tests.benchmarks.tritonbench_runner import RunResult


def write_json(results: Iterable[RunResult], out: Path) -> None:
    import json
    rs = list(results)
    out.write_text(json.dumps({
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "geomean_applied": weighted_geomean(filter_recommended(rs)),
        "n_kernels_total": len(rs),
        "n_kernels_recommended": sum(1 for r in rs if r.pr_applied),
        "per_kernel": [
            {"kernel": r.kernel_id, "baseline_ns": r.baseline_ns,
             "optimized_ns": r.optimized_ns, "speedup": r.speedup,
             "perfxpert_recommended": r.pr_applied}
            for r in rs
        ],
    }, indent=2))


def write_markdown(results: Iterable[RunResult], out: Path) -> None:
    rs = list(results)
    g = weighted_geomean(filter_recommended(rs))
    lines = [
        f"# perfxpert nightly — {datetime.utcnow().date()}",
        "",
        f"- Kernels total: **{len(rs)}**",
        f"- Kernels with perfxpert recommendation applied: **{sum(1 for r in rs if r.pr_applied)}**",
        f"- Weighted geomean speedup (applied only): **{g:.3f}×**",
        f"- Gate: {'PASS' if g >= 1.20 else 'FAIL'} (threshold 1.20×)",
        "",
        "## Per-kernel results",
        "",
        "| Kernel | Baseline (ns) | Optimized (ns) | Speedup | Applied |",
        "|--------|---------------|----------------|---------|---------|",
    ]
    for r in rs:
        flag = "✓" if r.pr_applied else " "
        lines.append(
            f"| {r.kernel_id} | {r.baseline_ns:,} | {r.optimized_ns:,} "
            f"| {r.speedup:.3f}× | {flag} |"
        )
    out.write_text("\n".join(lines))


def write_html(results: Iterable[RunResult], out: Path) -> None:
    # Minimal self-contained HTML table; nightly artifact viewable in browser
    import html as h
    rs = list(results)
    g = weighted_geomean(filter_recommended(rs))
    rows = "\n".join(
        f"<tr><td>{h.escape(r.kernel_id)}</td><td>{r.baseline_ns:,}</td>"
        f"<td>{r.optimized_ns:,}</td><td>{r.speedup:.3f}</td>"
        f"<td>{'yes' if r.pr_applied else 'no'}</td></tr>"
        for r in rs
    )
    out.write_text(f"""<!doctype html>
<html><head><meta charset='utf-8'><title>perfxpert nightly</title>
<style>body{{font-family:system-ui}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 8px}}</style></head>
<body>
<h1>perfxpert nightly {datetime.utcnow().date()}</h1>
<p>Weighted geomean: <b>{g:.3f}×</b> ({'PASS' if g>=1.2 else 'FAIL'})</p>
<table><thead><tr><th>Kernel</th><th>Baseline (ns)</th>
<th>Optimized (ns)</th><th>Speedup</th><th>Applied</th></tr></thead>
<tbody>{rows}</tbody></table>
</body></html>
""")
