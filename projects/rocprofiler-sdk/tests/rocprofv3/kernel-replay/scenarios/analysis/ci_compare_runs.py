#!/usr/bin/env python3
"""Compare failing tests across successive CI runs of one branch.

Repeated runs of near-identical code are repeated measurements: a test that fails in
some runs and not others is flaky, one that fails in all of them is a real defect.
"""

import collections
import json
import re
import subprocess
import sys

REPO = "ROCm/rocm-systems"
FAIL_LINE = re.compile(
    r"\s*(\d+) - (\S+) \((Failed|SEGFAULT|Not Run|Timeout|Subprocess aborted)\)"
)
SUMMARY = re.compile(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)")


def gh(*args):
    r = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    return r.stdout if r.returncode == 0 else None


def jobs(run_id):
    raw = gh("api", f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100")
    if not raw:
        return []
    return json.loads(raw).get("jobs", [])


def failing_tests(job_id):
    raw = gh("api", f"repos/{REPO}/actions/jobs/{job_id}/logs")
    if not raw:
        return None, None
    fails, summary = [], None
    for line in raw.splitlines():
        m = FAIL_LINE.search(line)
        if m:
            fails.append((m.group(2), m.group(3)))
        s = SUMMARY.search(line)
        if s and summary is None:
            summary = (int(s.group(2)), int(s.group(3)))
    return sorted(set(fails)), summary


def main():
    runs = [a for a in sys.argv[1:]]
    per_run = {}
    for run_id in runs:
        for job in jobs(run_id):
            name = job["name"]
            if not ("mi325" in name or "Test rocprofiler-sdk (shard" in name):
                continue
            if job["conclusion"] in (None, "skipped"):
                continue
            fails, summary = failing_tests(job["id"])
            if fails is None:
                continue
            key = (run_id, name)
            per_run[key] = {
                "conclusion": job["conclusion"],
                "fails": fails,
                "summary": summary,
            }
            print(
                f"  run {run_id} | {name[:44]:44s} | {job['conclusion']:8s} | "
                f"{len(fails)} failing | summary={summary}"
            )

    print("\n=== per-test failure counts across the collected jobs ===")
    counter = collections.Counter()
    jobs_seen = collections.Counter()
    for (run_id, name), info in per_run.items():
        fam = "TheRock" if "shard" in name else "Core"
        jobs_seen[fam] += 1
        for test, status in info["fails"]:
            counter[(fam, test, status)] += 1
    for (fam, test, status), count in sorted(counter.items(), key=lambda kv: -kv[1]):
        print(f"  {count:3d}/{jobs_seen[fam]:<3d} {fam:8s} {status:10s} {test[:74]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
