"""KernelBench-ROCm harness.

Similar to TritonBench but iterates level1/ (elementary kernels).
Parses the analysis-only JSON summary emitted by the driver.
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


def parse_kernelbench_output(raw: str) -> List[RunResult]:
    """Parse the JSON produced by the analysis-only KernelBench driver."""
    out: List[RunResult] = []
    data = json.loads(raw)
    for row in data["results"]:
        out.append(RunResult(
            kernel_id=row["kernel"],
            baseline_ns=int(row["baseline_ns"]),
            analysis_succeeded=bool(row["analysis_succeeded"]),
            recommendation_count=int(row.get("recommendation_count", 0)),
            report_path=row.get("report_path"),
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

    The suite outputs analysis_summary.json in its working dir.
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
    results_path = suite_root / "analysis_summary.json"
    return parse_kernelbench_output(results_path.read_text())
