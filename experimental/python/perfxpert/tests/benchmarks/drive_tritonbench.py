"""Driver for TritonBench: iterates the suite, collects baseline + applies
perfxpert recommendations + re-runs. Emits results.json.

This is the process-level glue; the runner module just calls this as a
subprocess and parses results.json.
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
    ap.add_argument("--filter", default=".*")
    args = ap.parse_args()

    suite = args.suite_root
    rx = re.compile(args.filter)
    results = []
    for kernel_dir in sorted(suite.glob("benchmarks/*")):
        if not kernel_dir.is_dir() or not rx.search(kernel_dir.name):
            continue
        try:
            baseline_ns = _run_kernel_once(kernel_dir, "baseline")
            # Apply perfxpert recommendation (captured to .perfxpert.patch)
            applied = _apply_perfxpert(kernel_dir)
            optimized_ns = _run_kernel_once(kernel_dir, "optimized") if applied else baseline_ns
            results.append({
                "kernel": kernel_dir.name,
                "baseline_ns": baseline_ns,
                "optimized_ns": optimized_ns,
                "perfxpert_recommended": bool(applied),
            })
        except Exception as e:
            print(f"[warn] {kernel_dir.name} failed: {e}", file=sys.stderr)

    (suite / "results.json").write_text(json.dumps({
        "suite": "tritonbench-rocm",
        "version": "0.2.0",
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


def _apply_perfxpert(kdir: Path) -> bool:
    db = kdir / "rocprof.db"
    if not db.exists():
        return False
    patch = kdir / ".perfxpert.patch"
    r = subprocess.run([
        "perfxpert", "analyze",
        "-i", str(db),
        "--offline",
        "--emit-patch", str(patch),
    ], capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not patch.exists():
        return False
    subprocess.run(["git", "apply", str(patch)], cwd=kdir, check=True)
    return True


if __name__ == "__main__":
    sys.exit(main())
