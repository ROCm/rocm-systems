"""TritonBench-ROCm harness.

Runs the pinned upstream benchmark suite against perfxpert's recommendations.
For each benchmark kernel:
  1. Collect baseline trace (rocprofv3 --sys-trace)
  2. Pipe trace DB through `perfxpert analyze --llm=<provider>` (or `--offline`
     for nightly to avoid hitting LLM quotas) — captures recommendation(s)
  3. Apply the recommendation to a copy of the kernel source
  4. Re-run the benchmark on the modified kernel
  5. Emit RunResult(kernel_id, baseline_ns, optimized_ns, pr_applied)

For nightly: uses the `--offline` flag so LLM providers aren't rate-limited
(deterministic handoffs still run all gates).

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
    optimized_ns: int
    pr_applied: bool

    @property
    def speedup(self) -> float:
        return self.baseline_ns / self.optimized_ns if self.optimized_ns else float("inf")


def parse_tritonbench_output(raw: str) -> List[RunResult]:
    """Parse the JSON produced by TritonBench's CLI into RunResults."""
    data = json.loads(raw)
    out: List[RunResult] = []
    for entry in data["results"]:
        out.append(RunResult(
            kernel_id=entry["kernel"],
            baseline_ns=int(entry["baseline_ns"]),
            optimized_ns=int(entry["optimized_ns"]),
            pr_applied=bool(entry["perfxpert_recommended"]),
        ))
    return out


def run_tritonbench(
    suite_root: Path,
    *,
    kernel_filter: str = ".*",
    timeout_s: int = 3600,
) -> List[RunResult]:
    """Invoke the TritonBench-ROCm suite and return parsed RunResults.

    The suite outputs `results.json` in its working dir; we read it after
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
    results_path = suite_root / "results.json"
    return parse_tritonbench_output(results_path.read_text())
