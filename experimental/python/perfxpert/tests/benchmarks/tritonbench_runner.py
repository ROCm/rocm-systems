"""TritonBench-ROCm harness.

Runs the pinned upstream benchmark suite and captures perfxpert analysis
artifacts for each resulting trace database.
For each benchmark kernel:
  1. Collect baseline trace (rocprofv3 --sys-trace)
  2. Run `perfxpert analyze --format json` on the trace DB
  3. Archive the analysis report next to the benchmark data
  4. Emit RunResult(kernel_id, baseline_ns, analysis_succeeded,
     recommendation_count, report_path)

Design constraint: the runner wraps but does NOT fork perfxpert. It uses
subprocess-based CLI entry points exclusively.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class RunResult:
    kernel_id: str
    baseline_ns: int
    analysis_succeeded: bool
    recommendation_count: int
    report_path: str | None


def parse_tritonbench_output(raw: str) -> List[RunResult]:
    """Parse the JSON produced by the analysis-only TritonBench driver."""
    data = json.loads(raw)
    out: List[RunResult] = []
    for entry in data["results"]:
        out.append(RunResult(
            kernel_id=entry["kernel"],
            baseline_ns=int(entry["baseline_ns"]),
            analysis_succeeded=bool(entry["analysis_succeeded"]),
            recommendation_count=int(entry.get("recommendation_count", 0)),
            report_path=entry.get("report_path"),
        ))
    return out


def run_tritonbench(
    suite_root: Path,
    *,
    kernel_filter: str = ".*",
    timeout_s: int = 3600,
) -> List[RunResult]:
    """Invoke the TritonBench-ROCm suite and return parsed RunResults.

    The suite outputs `analysis_summary.json` in its working dir; we read it after
    the subprocess completes.
    """
    env_driver = Path(__file__).parent / "drive_tritonbench.py"
    cmd = [
        "python3", str(env_driver),
        "--suite-root", str(suite_root),
        "--filter", kernel_filter,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if r.returncode != 0:
        raise RuntimeError(
            f"tritonbench driver exited {r.returncode}:\n{r.stderr[-2000:]}"
        )
    results_path = suite_root / "analysis_summary.json"
    return parse_tritonbench_output(results_path.read_text())
