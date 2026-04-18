"""Driver for KernelBench: iterates the suite (level1 only for nightly),
collects baseline + applies perfxpert recommendations + re-runs.
Emits results.csv.
"""

import argparse
import csv
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
            # Apply perfxpert recommendation
            applied = _apply_perfxpert(kernel_dir)
            optimized_ns = _run_kernel_once(kernel_dir, "optimized") if applied else baseline_ns
            results.append({
                "kernel": kernel_dir.name,
                "baseline_ns": baseline_ns,
                "optimized_ns": optimized_ns,
                "perfxpert_recommended": "True" if applied else "False",
            })
        except Exception as e:
            print(f"[warn] {kernel_dir.name} failed: {e}", file=sys.stderr)

    results_path = suite / "results.csv"
    with open(results_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kernel", "baseline_ns", "optimized_ns", "perfxpert_recommended"])
        w.writeheader()
        w.writerows(results)

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
