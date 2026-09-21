"""Driver for KernelBench: iterates the suite (level1 only for nightly),
collects a baseline, and runs perfxpert analysis on the captured trace.
Emits analysis_summary.json.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-root", type=Path, required=True)
    ap.add_argument("--level", default="level1")
    ap.add_argument("--filter", default=".*")
    args = ap.parse_args()

    suite = args.suite_root
    level_dir = suite / args.level
    rx = re.compile(args.filter)
    results = []

    if not level_dir.exists():
        print(f"[error] {level_dir} does not exist", file=sys.stderr)
        return 1

    for kernel_dir in sorted(level_dir.glob("*")):
        if not kernel_dir.is_dir() or not rx.search(kernel_dir.name):
            continue
        try:
            baseline_ns = _run_kernel_once(kernel_dir, "baseline")
            analysis = _run_perfxpert_analysis(kernel_dir)
            results.append({
                "kernel": kernel_dir.name,
                "baseline_ns": baseline_ns,
                "analysis_succeeded": analysis["analysis_succeeded"],
                "recommendation_count": analysis["recommendation_count"],
                "report_path": analysis["report_path"],
            })
        except Exception as e:
            print(f"[warn] {kernel_dir.name} failed: {e}", file=sys.stderr)
            results.append({
                "kernel": kernel_dir.name,
                "baseline_ns": 0,
                "analysis_succeeded": False,
                "recommendation_count": 0,
                "report_path": None,
                "error": str(e),
            })

    (suite / "analysis_summary.json").write_text(json.dumps({
        "suite": f"kernelbench-rocm:{args.level}",
        "mode": "analysis_only",
        "results": results,
    }, indent=2))

    return 0


def _run_kernel_once(kdir: Path, label: str) -> int:
    out = subprocess.run(
        ["./run.sh"], cwd=kdir, capture_output=True, text=True, timeout=600
    )
    m = re.search(r"median_ns:\s*(\d+)", out.stdout)
    if not m:
        raise RuntimeError(f"no median_ns in {label} output:\n{out.stdout[-1000:]}")
    return int(m.group(1))


def _run_perfxpert_analysis(kdir: Path) -> dict:
    db = kdir / "rocprof.db"
    if not db.exists():
        return {
            "analysis_succeeded": False,
            "recommendation_count": 0,
            "report_path": None,
        }
    report_dir = kdir / ".perfxpert"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "analysis.json"
    if report_path.exists():
        report_path.unlink()
    r = subprocess.run([
        "perfxpert", "analyze",
        "-i", str(db),
        "--format", "json",
        "-d", str(report_dir),
        "-o", "analysis",
    ], capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not report_path.exists():
        return {
            "analysis_succeeded": False,
            "recommendation_count": 0,
            "report_path": str(report_path),
        }
    payload = json.loads(report_path.read_text())
    return {
        "analysis_succeeded": True,
        "recommendation_count": len(payload.get("recommendations", [])),
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    sys.exit(main())
