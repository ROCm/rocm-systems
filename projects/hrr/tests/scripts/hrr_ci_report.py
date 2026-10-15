# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Write the HRR CI summary for a run and decide its overall result.

The summary has three parts:

  * why this run tested what it did, and which base build the GPU suite used;
  * the conclusion of every unit, build and GPU job, so a job that never
    produced results (cancelled, timed out, still queued) still has a row;
  * the per-suite results table from summarize_junit.py, with a row for every
    ci.yaml family, saying why when one did not run.

Exits 1 when the configure job did not succeed, any HRR job failed or was
cancelled, or a results file could not be read.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import summarize_junit

JOB_PATTERNS = (
    ("Generated sources", re.compile(r"^Generated sources$")),
    ("Script tests", re.compile(r"^HRR script tests$")),
    ("Unit", re.compile(r"(?:^|/ )Unit \((?P<name>[^)]+)\)$")),
    ("Build", re.compile(r"(?:^|/ )Build capture-enabled HIP and HRR$")),
    ("Integration", re.compile(r"(?:^|/ )Integration \((?P<name>[^)]+)\)$")),
)
BAD_CONCLUSIONS = {"failure", "cancelled", "timed_out", "startup_failure"}


def classify_jobs(jobs: list[dict]) -> list[dict]:
    """HRR jobs from the run's job list, as {kind, name, conclusion, url}."""
    rows = []
    for job in jobs:
        for kind, pattern in JOB_PATTERNS:
            match = pattern.search(job.get("name", ""))
            if match:
                rows.append(
                    {
                        "kind": kind,
                        "name": match.groupdict().get("name") or "",
                        "conclusion": job.get("conclusion") or job.get("status") or "unknown",
                        "url": job.get("html_url", ""),
                    }
                )
                break
    return rows


def missing_families(families: list[dict], rows: list[dict]) -> list[str]:
    seen = {r["name"] for r in rows if r["kind"] == "Integration"}
    return [f["family"] for f in families if f["family"] not in seen]


def render_header(
    reason: str, run_gpu: bool, families: list[dict], base: dict | None,
    configure_result: str = "success",
) -> str:
    if configure_result != "success":
        return (
            f"## HRR CI\n\n- Configure: {configure_result}, so nothing else ran;"
            " see the Configure job's log.\n\n"
        )
    lines = ["## HRR CI", "", f"- Scope: {reason}."]
    if run_gpu:
        names = ", ".join(
            f"{f['family']} (`{f['runner']}`)" if f.get("runner") else f["family"]
            for f in families
        ) or "none"
        lines.append(f"- GPU families and runners: {names}.")
        if base:
            lines.append(
                f"- Base ROCm artifacts: [{base['repo']} run {base['run_id']}]({base['run_url']})"
                f" at `{base['commit'][:12]}`; amdhip64 rebuilt from this commit with capture enabled."
            )
    else:
        lines.append("- GPU suite: not run.")
    return "\n".join(lines) + "\n\n"


def render_jobs(rows: list[dict], missing: list[str]) -> str:
    if not rows and not missing:
        return ""
    lines = ["### Jobs", "", "| Job | Target | Conclusion |", "| --- | --- | --- |"]
    for row in rows:
        conclusion = f"[{row['conclusion']}]({row['url']})" if row["url"] else row["conclusion"]
        lines.append(f"| {row['kind']} | {row['name'] or '—'} | {conclusion} |")
    for family in missing:
        lines.append(f"| Integration | {family} | _did not run_ |")
    return "\n".join(lines) + "\n\n"


def overall_ok(rows: list[dict], missing: list[str]) -> bool:
    return not missing and not any(r["conclusion"] in BAD_CONCLUSIONS for r in rows)


def pass_through_ok(rows: list[dict]) -> tuple[bool, str]:
    """Verdict on an earlier run's jobs. A job that has not finished or is
    something other than a plain success is not allowed to count as pass: a
    label run must never turn a failing HRR CI into a passing one."""
    if not rows:
        return False, "the run that tested this commit lists no HRR jobs"
    for row in rows:
        if row["conclusion"] not in ("success", "skipped", "neutral"):
            return False, f"{row['kind']} {row['name'] or row['conclusion']} is {row['conclusion']}"
    return True, ""


def list_jobs(repo: str, run_id: str, attempt: str) -> list[dict]:
    out = subprocess.run(
        [
            "gh", "api", "--paginate",
            f"repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs",
            "--jq", ".jobs[]",
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    decoder = json.JSONDecoder()
    jobs, pos = [], 0
    while pos < len(out):
        while pos < len(out) and out[pos].isspace():
            pos += 1
        if pos >= len(out):
            break
        job, pos = decoder.raw_decode(out, pos)
        jobs.append(job)
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configure-result", default="success")
    parser.add_argument("--reason", default="")
    parser.add_argument("--run-gpu", default="false")
    parser.add_argument("--families", default="[]")
    parser.add_argument("--skipped-families", default="[]")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--base-json", type=Path)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", ""))
    parser.add_argument("--run-attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT", "1"))
    parser.add_argument("--pass-through", action="store_true",
                        help="deliver the verdict of the run that tested this commit")
    parser.add_argument("--upstream-run", default="",
                        help="that run's id; empty when no earlier run was found")
    parser.add_argument("--upstream-attempt", default="")
    args = parser.parse_args(argv)

    run_gpu = args.run_gpu == "true"
    families = json.loads(args.families or "[]") if run_gpu else []
    skipped = json.loads(args.skipped_families or "[]") if run_gpu else []
    base = None
    if args.base_json and args.base_json.is_file():
        base = json.loads(args.base_json.read_text())

    job_run, job_attempt, exiting = args.run_id, args.run_attempt, False
    if args.pass_through:
        job_run = args.upstream_run or args.run_id
        job_attempt = args.upstream_attempt
        if not job_attempt:
            # Nothing before this run, or the lookup failed: leave with a red
            # summary, which a later re-run resolves, rather than green.
            print(f"::error::no earlier run of HRR CI found for this commit; "
                  f"a label event cannot verify it, so this summary fails for run {args.run_id}")
            return 1
        exiting = True

    try:
        jobs = list_jobs(args.repo, job_run, job_attempt)
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or "").strip() or f"gh api exited {e.returncode}"
        print(f"::error::could not list the jobs of run {job_run}: {detail}")
        return 1
    rows = classify_jobs(jobs)
    missing = missing_families(families, rows)

    if exiting:
        ok, why = pass_through_ok(rows)
        if not ok:
            print(f"::error::HRR CI on this commit reported {why}; "
                  f"a label event must not turn it into a pass")
            return 1

    configured = args.configure_result == "success"
    markdown = render_header(args.reason, run_gpu, families, base, args.configure_result)
    markdown += render_jobs(rows, missing)
    errors: list[str] = []
    args.results_dir.mkdir(parents=True, exist_ok=True)
    table_families = [(f["family"], "") for f in families]
    table_families += [(s["family"], s["reason"]) for s in skipped]
    markdown += summarize_junit.render(args.results_dir, errors, table_families)

    sys.stdout.write(markdown)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as out:
            out.write(markdown)
    for error in errors:
        print(f"::error::unreadable JUnit XML {error}")
    return 0 if configured and overall_ok(rows, missing) and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
