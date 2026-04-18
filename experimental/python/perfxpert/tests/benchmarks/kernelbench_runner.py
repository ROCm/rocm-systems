"""KernelBench-ROCm harness.

Similar to TritonBench but iterates level1/ (elementary kernels).
Parses CSV output rather than JSON.
"""

import csv
import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class RunResult:
    kernel_id: str
    baseline_ns: int
    optimized_ns: int
    pr_applied: bool

    @property
    def speedup(self) -> float:
        return self.baseline_ns / self.optimized_ns if self.optimized_ns else float("inf")


def parse_kernelbench_output(raw: str) -> List[RunResult]:
    """Parse the CSV produced by KernelBench into RunResults."""
    out: List[RunResult] = []
    reader = csv.DictReader(StringIO(raw))
    for row in reader:
        out.append(RunResult(
            kernel_id=row["kernel"],
            baseline_ns=int(row["baseline_ns"]),
            optimized_ns=int(row["optimized_ns"]),
            pr_applied=row["perfxpert_recommended"].lower() in ("true", "yes", "1"),
        ))
    return out


def run_kernelbench(
    suite_root: Path,
    *,
    level: str = "level1",
    kernel_filter: str = ".*",
    timeout_s: int = 3600,
) -> List[RunResult]:
    """Invoke the KernelBench-ROCm suite and return parsed RunResults.

    The suite outputs results.csv in its working dir.
    """
    env_driver = Path(__file__).parent / "drive_kernelbench.py"
    cmd = [
        "python3", str(env_driver),
        "--suite-root", str(suite_root),
        "--level", level,
        "--filter", kernel_filter,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if r.returncode != 0:
        raise RuntimeError(
            f"kernelbench driver exited {r.returncode}:\n{r.stderr[-2000:]}"
        )
    results_path = suite_root / "results.csv"
    return parse_kernelbench_output(results_path.read_text())
