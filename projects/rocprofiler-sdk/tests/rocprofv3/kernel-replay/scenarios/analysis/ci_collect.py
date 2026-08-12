#!/usr/bin/env python3
"""Collect per-test measurements from completed CI runs and summarise them.

Pulls CTest output out of job logs and records, per (branch, run, job, test):
duration, status. Used to answer two questions with real data:

  Q1  how stable are the kernel-replay tests across OS images and repeated runs
  Q2  how often does the hip-streams-per-thread segfault reproduce, across
      independent jobs and independent branches

Read-only: uses the gh CLI, writes CSV locally.
"""

import csv
import json
import os
import re
import subprocess
import sys

REPO = "ROCm/rocm-systems"
BRANCHES = [
    "users/mkuriche/kernel-replay-callback-api",
    "users/vkale/remove-callbacks-counters",
    "users/vkale/remove-callbacks-spm",
    "users/vkale/remove-callbacks-thread-trace",
    "users/vkale/remove-callbacks-counter-collection-cursor-fixes",
    "users/vkale/remove-callbacks-pc-sampling",
]
JOB_PAT = re.compile(r"Core . mi325 . (\S+)|Test rocprofiler-sdk \(shard")
# CTest result lines: " 123/456 Test  #12: name .... Passed  1.23 sec"
TEST_LINE = re.compile(
    r"\s*\d+/\d+\s+Test\s+#(\d+):\s+(\S+)\s+\.+\**\s*(Passed|Failed|Skipped|Exception: SegFault|Not Run|Timeout)\s+([\d.]+)\s+sec"
)
INTEREST = ("kernel-replay", "kernel_replay", "hip-streams-per-thread")


def gh(*args):
    out = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        return None
    return out.stdout


def runs_for(branch, limit=100):
    raw = gh(
        "api",
        f"repos/{REPO}/actions/runs?branch={branch}&per_page={limit}&event=pull_request",
    )
    if not raw:
        return []
    data = json.loads(raw)
    return [
        r
        for r in data.get("workflow_runs", [])
        if r["name"] in ("rocprofiler-sdk Continuous Integration", "TheRock CI")
        and r["status"] == "completed"
    ]


def jobs_for(run_id):
    raw = gh("api", f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100")
    if not raw:
        return []
    return [j for j in json.loads(raw).get("jobs", []) if JOB_PAT.search(j["name"])]


def parse_log(job_id):
    raw = gh("api", f"repos/{REPO}/actions/jobs/{job_id}/logs")
    if not raw:
        return [], None
    rows = []
    summary = None
    seen = set()
    for line in raw.splitlines():
        m = TEST_LINE.search(line)
        if m:
            _num, name, status, secs = m.groups()
            if any(k in name for k in INTEREST):
                key = (name, status, secs)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"test": name, "status": status, "seconds": float(secs)})
        if "tests passed," in line and "out of" in line:
            mm = re.search(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)", line)
            if mm and summary is None:
                summary = {
                    "pct": int(mm.group(1)),
                    "failed": int(mm.group(2)),
                    "total": int(mm.group(3)),
                }
    return rows, summary


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/exp/out"
    os.makedirs(outdir, exist_ok=True)
    measurements = []
    jobs_meta = []
    for branch in BRANCHES:
        for run in runs_for(branch):
            for job in jobs_for(run["id"]):
                rows, summary = parse_log(job["id"])
                if not rows and not summary:
                    continue
                jobs_meta.append(
                    {
                        "branch": branch,
                        "workflow": run["name"],
                        "run_id": run["id"],
                        "job_id": job["id"],
                        "job": job["name"],
                        "conclusion": job["conclusion"],
                        "created": run["created_at"],
                        "pct_passed": (summary or {}).get("pct", ""),
                        "tests_failed": (summary or {}).get("failed", ""),
                        "tests_total": (summary or {}).get("total", ""),
                    }
                )
                for row in rows:
                    measurements.append(
                        {
                            "branch": branch,
                            "workflow": run["name"],
                            "run_id": run["id"],
                            "job_id": job["id"],
                            "job": job["name"],
                            "created": run["created_at"],
                            **row,
                        }
                    )
                print(
                    f"  {branch.split('/')[-1][:28]:28s} {job['name'][:42]:42s} "
                    f"{len(rows):3d} rows  {job['conclusion']}"
                )
    for name, data in (("ci_measurements.csv", measurements), ("ci_jobs.csv", jobs_meta)):
        if not data:
            continue
        with open(os.path.join(outdir, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"wrote {len(data)} rows to {outdir}/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
