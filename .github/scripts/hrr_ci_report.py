# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Write the HRR CI summary for a run and decide its overall result.

The summary has three parts:

  * why this run tested what it did, and which base build the GPU suite used;
  * the conclusion of every unit, build and GPU job, so a job that never
    produced results (cancelled, timed out, still queued) still has a row;
  * the per-suite results table from projects/hrr's summarize_junit.py.

Exits 1 when any HRR job failed or was cancelled, or a results file could not
be read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

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


def render_header(reason: str, run_gpu: bool, families: list[dict], base: dict | None) -> str:
    lines = ["## HRR CI", "", f"- Scope: {reason}."]
    if run_gpu:
        names = ", ".join(f["family"] for f in families) or "none"
        lines.append(f"- GPU families: {names}.")
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


def load_summarize_junit(path: Path):
    spec = importlib.util.spec_from_file_location("summarize_junit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    parser.add_argument("--reason", default="")
    parser.add_argument("--run-gpu", default="false")
    parser.add_argument("--families", default="[]")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--base-json", type=Path)
    parser.add_argument(
        "--summarize-junit",
        type=Path,
        default=Path("projects/hrr/tests/scripts/summarize_junit.py"),
    )
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", ""))
    parser.add_argument("--run-attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT", "1"))
    args = parser.parse_args(argv)

    run_gpu = args.run_gpu == "true"
    families = json.loads(args.families or "[]") if run_gpu else []
    base = None
    if args.base_json and args.base_json.is_file():
        base = json.loads(args.base_json.read_text())

    rows = classify_jobs(list_jobs(args.repo, args.run_id, args.run_attempt))
    missing = missing_families(families, rows)

    markdown = render_header(args.reason, run_gpu, families, base) + render_jobs(rows, missing)
    errors: list[str] = []
    args.results_dir.mkdir(parents=True, exist_ok=True)
    markdown += load_summarize_junit(args.summarize_junit).render(args.results_dir, errors)

    sys.stdout.write(markdown)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as out:
            out.write(markdown)
    for error in errors:
        print(f"::error::unreadable JUnit XML {error}")
    return 0 if overall_ok(rows, missing) and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
