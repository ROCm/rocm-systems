# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Write the HRR CI summary for a run and decide its overall result.

The summary has three parts:

  * why this run tested what it did, and which base build the GPU suite used;
  * the conclusion of every unit, build and GPU job, so a job that never
    produced results (cancelled, timed out, still queued) still has a row;
  * the per-suite results table from summarize_junit.py, with a row for every
    ci.yaml family, saying why when one did not run.

Exits 1 when the configure job did not succeed, a needed job failed, was
cancelled, or was skipped after configure selected it, any listed HRR job is
not in a terminal success state, a JUnit suite has failed cases, or a results
file could not be read. Job conclusions from the Actions API are links in the
table. The verdict uses --needs-json (the workflow's toJSON(needs)) and the
JUnit files, because the jobs API can still say queued after those exist.

With --pass-through true (a label run that ran no HRR job) it reports the
newest other run on --head-sha that has a verdict instead, and exits 1 unless
that run's HRR jobs and its own summary succeeded, or when it cannot be read.
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
    # The simulator jobs download this one build of rocJitsu; its failure fails them too.
    ("Build", re.compile(r"(?:^|/ )Build rocJitsu$")),
    # The real-GPU job is "Integration (<family>)", the rocJitsu one "rocJitsu (<family>)".
    ("Integration", re.compile(r"(?:^|/ )(?:Integration|rocJitsu) \((?P<name>[^)]+)\)$")),
)
# Anything else (queued, in_progress, waiting, unknown) has not finished.
TERMINAL_OK = {"success", "neutral"}
BAD_NEEDS = {"failure", "cancelled"}
CPU_NEEDS = ("codegen", "scripts", "unit")


def classify_jobs(jobs: list[dict]) -> list[dict]:
    """HRR jobs from the run's job list, as {kind, name, conclusion, url}."""
    rows = []
    for job in jobs:
        # A skipped matrix job keeps its unexpanded name; it is no family.
        if "${{" in job.get("name", ""):
            continue
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
    """True when every requested family ran and every listed job finished
    successfully. A skipped, failed, or still-queued job is not a pass: the
    jobs API has reported queued while the suite had already failed."""
    if missing:
        return False
    return all(r["conclusion"] in TERMINAL_OK for r in rows)


def need_result(needs: dict, name: str) -> str:
    info = needs.get(name)
    if isinstance(info, dict) and info.get("result"):
        return str(info["result"])
    return "missing"


def check_needs(needs: dict, *, run_gpu: bool) -> list[str]:
    """Problems in GitHub's toJSON(needs). A failure or cancellation fails the
    run. On a run whose configure succeeded, the CPU jobs were selected, and
    the GPU workflow was selected when run_gpu is true; a skip there is a job
    that was supposed to run."""
    problems = []
    for name in needs:
        result = need_result(needs, name)
        if result in BAD_NEEDS:
            problems.append(f"{name} is {result}")
    selected = list(CPU_NEEDS)
    if run_gpu:
        selected.append("gpu")
    for name in selected:
        result = need_result(needs, name)
        if result != "success" and f"{name} is {result}" not in problems:
            problems.append(f"{name} is {result}")
    return problems


SUMMARY_JOB = "HRR CI Summary"


def tested_commit(jobs: list[dict], rows: list[dict]) -> bool:
    """A run whose configure selected nothing still lists its HRR jobs, as
    skipped, and tested nothing. One whose configure failed, or has not
    finished, skipped them too but has a verdict of its own to report. A run
    that lists no jobs yet is passed over; its own summary comes later and is
    the newer check."""
    configure = next((j for j in jobs if j.get("name") == "Configure"), {})
    if (configure.get("conclusion") or configure.get("status")) not in (None, "success", "skipped"):
        return True
    return any(r["conclusion"] != "skipped" for r in rows)


def pass_through_ok(rows: list[dict], summary_conclusion: str) -> tuple[bool, str]:
    """Verdict on an earlier run. Its own summary decides, since that is the
    check a label run would otherwise hide, and every HRR job must also have
    finished without failing: a label run must never turn a failing HRR CI
    into a passing one."""
    for row in rows:
        if row["conclusion"] not in ("success", "skipped", "neutral"):
            return False, f"{row['kind']} {row['name'] or row['conclusion']} is {row['conclusion']}"
    if summary_conclusion != "success":
        return False, f"{SUMMARY_JOB} is {summary_conclusion or 'missing'}"
    return True, ""


def gh_api(path: str, jq: str) -> list:
    """The values `gh api --paginate --jq` prints for path, one per jq result."""
    out = subprocess.run(
        ["gh", "api", "--paginate", path, "--jq", jq],
        check=True, capture_output=True, text=True,
    ).stdout
    decoder = json.JSONDecoder()
    values, pos = [], 0
    while pos < len(out):
        while pos < len(out) and out[pos].isspace():
            pos += 1
        if pos >= len(out):
            break
        value, pos = decoder.raw_decode(out, pos)
        values.append(value)
    return values


def list_jobs(repo: str, run_id: str, attempt: str) -> list[dict]:
    return gh_api(f"repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs", ".jobs[]")


def find_tested_run(repo: str, run_id: str, head_sha: str) -> tuple[dict, list[dict], str] | None:
    """The newest other pull_request run of this workflow on head_sha that
    tested it, with its HRR jobs and its summary's conclusion. Its latest
    attempt counts, as it does for the checks GitHub shows. Label runs that
    ran nothing themselves are passed over."""
    [workflow_id] = gh_api(f"repos/{repo}/actions/runs/{run_id}", ".workflow_id")
    runs = gh_api(
        f"repos/{repo}/actions/workflows/{workflow_id}/runs"
        f"?event=pull_request&head_sha={head_sha}&per_page=100",
        ".workflow_runs[]",
    )
    runs = [r for r in runs if str(r["id"]) != str(run_id)]
    # A re-run keeps created_at; run_started_at moves with the latest attempt,
    # whose check runs are the newest on the commit.
    runs.sort(key=lambda r: r.get("run_started_at") or r["created_at"], reverse=True)
    for run in runs:
        jobs = list_jobs(repo, str(run["id"]), str(run["run_attempt"]))
        rows = classify_jobs(jobs)
        if tested_commit(jobs, rows):
            summary = next((j for j in jobs if j.get("name") == SUMMARY_JOB), {})
            return run, rows, summary.get("conclusion") or summary.get("status") or ""
    return None


def pass_through(repo: str, run_id: str, head_sha: str) -> int:
    """Summary for a label run that ran no HRR job: the verdict of the run
    that tested the commit, and never greener than it. Passes when no run
    tested the commit, since there is then no failure to hide. Fails while
    that run is still in progress; its own summary, written later, is then
    the newer check."""
    try:
        found = find_tested_run(repo, run_id, head_sha)
    except (subprocess.CalledProcessError, ValueError) as e:
        detail = (getattr(e, "stderr", "") or "").strip() or str(e)
        print(f"::error::could not find the HRR CI run that tested {head_sha}: {detail}")
        return 1
    if found is None:
        print(f"No earlier HRR CI run ran HRR jobs on {head_sha}; nothing to report.")
        return 0
    run, rows, summary_conclusion = found
    markdown = (
        f"## HRR CI\n\n- This label changes nothing that runs; the verdict is that of "
        f"[run {run['id']} attempt {run['run_attempt']}]({run.get('html_url', '')}), "
        f"which tested this commit.\n\n"
    ) + render_jobs(rows, [])
    sys.stdout.write(markdown)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as out:
            out.write(markdown)
    ok, why = pass_through_ok(rows, summary_conclusion)
    if not ok:
        print(f"::error::HRR CI on this commit reported {why}; "
              f"a label event must not turn it into a pass")
        return 1
    return 0


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
    parser.add_argument("--needs-json", default="",
                        help="GitHub toJSON(needs). Empty skips the needs check")
    parser.add_argument("--pass-through", choices=("true", "false"), default="false",
                        help="true: report the earlier run that tested --head-sha instead")
    # Not GITHUB_SHA: on a pull request that is the merge commit, which no
    # earlier run is listed under.
    parser.add_argument("--head-sha", default="")
    args = parser.parse_args(argv)

    run_gpu = args.run_gpu == "true"
    families = json.loads(args.families or "[]") if run_gpu else []
    skipped = json.loads(args.skipped_families or "[]") if run_gpu else []
    base = None
    if args.base_json and args.base_json.is_file():
        base = json.loads(args.base_json.read_text())

    if args.pass_through == "true":
        if not args.head_sha:
            parser.error("--pass-through true needs --head-sha, the pull request head")
        return pass_through(args.repo, args.run_id, args.head_sha)

    try:
        jobs = list_jobs(args.repo, args.run_id, args.run_attempt)
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or "").strip() or f"gh api exited {e.returncode}"
        print(f"::error::could not list the jobs of run {args.run_id}: {detail}")
        return 1
    rows = classify_jobs(jobs)
    missing = missing_families(families, rows)

    configured = args.configure_result == "success"
    markdown = render_header(args.reason, run_gpu, families, base, args.configure_result)
    markdown += render_jobs(rows, missing)
    errors: list[str] = []
    failed_suites: list[str] = []
    args.results_dir.mkdir(parents=True, exist_ok=True)
    table_families = [(f["family"], "") for f in families]
    table_families += [(s["family"], s["reason"]) for s in skipped]
    markdown += summarize_junit.render(
        args.results_dir, errors, table_families, failures=failed_suites
    )

    sys.stdout.write(markdown)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as out:
            out.write(markdown)
    for error in errors:
        print(f"::error::unreadable JUnit XML {error}")
    for suite in failed_suites:
        print(f"::error::failed cases in {suite}")

    needs_problems: list[str] = []
    if args.needs_json and configured:
        try:
            needs = json.loads(args.needs_json)
        except json.JSONDecodeError as error:
            print(f"::error::could not read --needs-json: {error}")
            return 1
        if not isinstance(needs, dict):
            print("::error::--needs-json must be an object")
            return 1
        needs_problems = check_needs(needs, run_gpu=run_gpu)
        for problem in needs_problems:
            print(f"::error::{problem}")
    return 0 if (
        configured and not needs_problems and overall_ok(rows, missing)
        and not errors and not failed_suites
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
