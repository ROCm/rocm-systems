#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decision logic for .github/workflows/rccl_perf_regression.yml.

The untrusted event payload is read from GITHUB_EVENT_PATH, never from `${{ }}`
interpolation, so a crafted comment body cannot escape into a shell.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

API_TIMEOUT_SECONDS = 30
# GitHub rejects a longer commit-status description.
MAXIMUM_DESCRIPTION_LENGTH = 140
# GitHub rejects an issue-comment body over 65536 characters.
MAXIMUM_REPORT_LENGTH = 60000
RCCL_PATH_PREFIX = "projects/rccl/"
RUN_CONFIG_PREFIX = "ci_detect_run_"
RUN_CONFIG_SUFFIX = ".json"
# `git check-ref-format` accepts backticks, $(), | and &, and the ref reaches a
# shell in another repo via `sbatch --export`, so allowlist rather than deny.
REF_ALLOWED_RE = re.compile(r"[A-Za-z0-9._/-]+")
REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
# Loose so a short SHA still gets an explanation; fork_vouch_denial does the
# binding.
VOUCHED_SHA_RE = re.compile(r"[0-9a-fA-F]{7,40}")
# author_association is too coarse: COLLABORATOR includes read and triage.
WRITE_PERMISSIONS = frozenset({"admin", "maintain", "write"})


class PermissionReadError(Exception):
    pass


@dataclass(frozen=True)
class GateRequest:
    authorized: bool
    reason: str
    level: str = ""
    head_sha: str = ""
    base_ref: str = ""
    pr_number: str = ""
    requester: str = ""
    head_repo: str = ""
    vouched_sha: str = ""
    deny_reason: str = ""

    def outputs(self) -> dict[str, str]:
        return {
            "authorized": "true" if self.authorized else "false",
            "head_sha": self.head_sha,
            "base_ref": self.base_ref,
            "pr_number": self.pr_number,
            "requester": self.requester,
            "head_repo": self.head_repo,
            "vouched_sha": self.vouched_sha,
            "deny_reason": self.deny_reason,
        }


@dataclass(frozen=True)
class Verdict:
    state: str
    description: str

    def outputs(self) -> dict[str, str]:
        return {"state": self.state, "desc": self.description}


class GitHubApi:
    # Duplicates rocjitsu_corpus_translation_comment.py; sharing it is AIMVT-285.
    def __init__(self, repository: str, token: str, api_url: str) -> None:
        if REPOSITORY_RE.fullmatch(repository) is None:
            raise ValueError(f"invalid GitHub repository: {repository!r}")
        if not token:
            raise ValueError("GH_TOKEN must not be empty")
        if not api_url.startswith("https://"):
            raise ValueError("GITHUB_API_URL must use HTTPS")
        self.repository = repository
        self._token = token
        self._api_url = api_url.rstrip("/")

    def get(self, path: str) -> Any:
        return self._request("GET", path)

    def create(self, path: str, payload: dict[str, Any]) -> Any:
        return self._request("POST", path, payload)

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._api_url}{path}",
            data=body,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "User-Agent": "rccl-perf-gate",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
            response_body = response.read()
        return json.loads(response_body) if response_body else None


def collaborator_permission(api: GitHubApi, repository: str, login: str) -> str:
    """Read a user's current permission level, raising if it cannot be read."""
    quoted = urllib.parse.quote(login, safe="")
    try:
        payload = api.get(f"/repos/{repository}/collaborators/{quoted}/permission")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as error:
        raise PermissionReadError(str(error)) from error
    if not isinstance(payload, dict):
        raise PermissionReadError("permission response is not an object")
    return str(payload.get("permission") or "none")


def resolve_request(
    event_name: str,
    event: dict[str, Any],
    api: GitHubApi,
    repository: str,
    command: str,
    environ: dict[str, str],
) -> GateRequest:
    """Authorize the trigger and resolve the head SHA / base ref it applies to."""
    if event_name == "workflow_dispatch":
        # Dispatching already requires repository write access.
        return GateRequest(
            authorized=True,
            reason="manual dispatch",
            head_sha=environ.get("GITHUB_SHA", ""),
            base_ref=validate_ref(environ.get("GITHUB_REF_NAME", "")),
            pr_number="",
            requester=environ.get("GITHUB_ACTOR", ""),
            head_repo=repository,
        )

    if event_name != "issue_comment":
        return GateRequest(False, f"unsupported event {event_name!r}")

    issue = event.get("issue") or {}
    if not issue.get("pull_request"):
        return GateRequest(False, "comment is not on a pull request")

    comment = event.get("comment") or {}
    vouched = parse_gate_command(str(comment.get("body") or ""), command)
    if vouched is None:
        return GateRequest(False, f"comment is not {command!r}")

    actor = str((comment.get("user") or {}).get("login") or "")
    if not actor:
        return GateRequest(False, "comment has no author")

    # Fail closed either way, but an unreadable permission denies EVERYONE, so
    # it is an error while an expected deny is only a warning.
    try:
        permission = collaborator_permission(api, repository, actor)
    except PermissionReadError as error:
        return GateRequest(
            False,
            f"Could not read @{actor}'s permission level, so the gate failed "
            f"closed: {error}",
            level="error",
            deny_reason="perm_read_error",
        )
    if permission not in WRITE_PERMISSIONS:
        return GateRequest(
            False,
            f"@{actor} needs write access to `{repository}` to run the perf "
            f"gate (got {permission!r}).",
            level="warning",
            deny_reason="not_writer",
        )

    number = issue.get("number")
    pull = api.get(f"/repos/{repository}/pulls/{int(number)}")
    if not isinstance(pull, dict):
        raise ValueError(f"pull request {number} response is not an object")
    head = pull.get("head") or {}
    head_sha = str(head.get("sha") or "")
    head_repo = str((head.get("repo") or {}).get("full_name") or "")
    author = str((pull.get("user") or {}).get("login") or "")
    denial = fork_vouch_denial(
        command, actor, author, head_repo, head_sha, vouched, repository
    )
    if denial is not None:
        return denial
    return GateRequest(
        authorized=True,
        reason=f"authorized for @{actor}",
        head_sha=head_sha,
        base_ref=validate_ref(str((pull.get("base") or {}).get("ref") or "")),
        pr_number=str(number),
        requester=actor,
        head_repo=head_repo,
        vouched_sha=vouched,
    )


def fork_vouch_denial(
    command: str,
    actor: str,
    author: str,
    head_repo: str,
    head_sha: str,
    vouched: str,
    repository: str,
) -> GateRequest | None:
    """Refuse a fork run unless a second writer named the head SHA."""
    if not head_repo:
        return GateRequest(
            False,
            "The pull request's head repository no longer exists.",
            level="warning",
            deny_reason="head_repo_gone",
        )
    if head_repo != repository:
        if actor.lower() == author.lower():
            return GateRequest(
                False,
                f"A fork pull request has to be vouched for by someone other "
                f"than its author, so @{actor} cannot request the gate here.",
                level="warning",
                deny_reason="fork_author_self",
            )
        if not vouched:
            return GateRequest(
                False,
                f"This is a fork pull request. Review the diff, then re-comment "
                f"`{command} {head_sha}` to vouch for that exact commit.",
                level="warning",
                deny_reason="fork_needs_vouch",
            )
    # Exact 40 chars: at 7 the fork author can author two commits sharing a
    # prefix, so an abbreviation does not identify the commit that was read.
    if vouched and vouched != head_sha.lower():
        return GateRequest(
            False,
            f"`{vouched}` is not this pull request's full head SHA. Re-comment "
            f"`{command} {head_sha}` to vouch for the commit that is there now.",
            level="warning",
            deny_reason="sha_stale",
        )
    return None


def parse_gate_command(body: str, command: str) -> str | None:
    """None when the body is not the command, else the vouched SHA or ""."""
    stripped = body.replace("\r", "").strip()
    if "\n" in stripped:
        return None
    parts = stripped.split()
    if not parts or parts[0] != command:
        return None
    if len(parts) == 1:
        return ""
    if len(parts) > 2 or VOUCHED_SHA_RE.fullmatch(parts[1]) is None:
        return None
    return parts[1].lower()


def validate_ref(ref: str) -> str:
    """Reject a branch name git or `sbatch --export` could not carry safely."""
    if REF_ALLOWED_RE.fullmatch(ref) is None:
        raise ValueError(f"branch name is empty or not [A-Za-z0-9._/-]: {ref!r}")
    result = subprocess.run(
        ["git", "check-ref-format", f"refs/heads/{ref}"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"invalid branch name: {ref!r}")
    return ref


def set_commit_status(
    api: GitHubApi,
    repository: str,
    sha: str,
    context: str,
    state: str,
    description: str,
    target_url: str,
) -> None:
    payload = {
        "state": state,
        "context": context,
        "target_url": target_url,
        "description": description[:MAXIMUM_DESCRIPTION_LENGTH],
    }
    api.create(f"/repos/{repository}/statuses/{sha}", payload)


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], capture_output=True, check=True, text=True
    )
    return result.stdout.strip()


def cvs_revision(rccl_ci_root: str) -> str:
    """The ROCm/cvs working copy is not pinned, so record what produced this run."""
    try:
        return git_output("-C", f"{rccl_ci_root}/cvs", "rev-parse", "HEAD")
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def fetch_base(base_ref: str, remote: str) -> None:
    """Fetch by URL: after a fork checkout `origin` is the fork, not upstream.

    Anonymous, because every checkout sets `persist-credentials: false`; this
    therefore depends on the base repository staying public.
    """
    subprocess.run(
        [
            "git",
            "fetch",
            "--no-tags",
            remote,
            f"+refs/heads/{base_ref}:refs/remotes/origin/{base_ref}",
        ],
        check=True,
    )


def rccl_paths_changed(base_ref: str) -> bool:
    base = git_output("merge-base", f"origin/{base_ref}", "HEAD")
    names = git_output("diff", "--name-only", base, "HEAD")
    return any(line.startswith(RCCL_PATH_PREFIX) for line in names.splitlines())


def should_build_rccl(event_name: str, rccl_changed: str, requested: str) -> bool:
    """Comment-triggered runs build only when RCCL sources changed."""
    if event_name == "issue_comment":
        return rccl_changed == "1"
    return requested == "1"


def run_config_path(rccl_ci_root: str, run_id: str) -> Path:
    name = f"{RUN_CONFIG_PREFIX}{run_id}{RUN_CONFIG_SUFFIX}"
    return Path(rccl_ci_root) / "configs" / name


def cleanup_run_config(config_json: str, rccl_ci_root: str, run_id: str) -> bool:
    """Delete this run's own config copy, refusing every other path."""
    if not config_json:
        return False
    path = Path(config_json)
    if path != run_config_path(rccl_ci_root, run_id):
        return False
    path.unlink(missing_ok=True)
    return True


def compute_verdict(build_result: str, detect_result: str, detect_code: str) -> Verdict:
    """Map job results onto a commit-status state; detector exit 1 still passes."""
    # `cancelled` covers a human cancel, a concurrency eviction and a job-level
    # timeout alike: nothing was measured, so ask rather than blame this PR.
    if build_result == "cancelled":
        return Verdict("pending", "Build did not finish; re-request the gate")
    if build_result != "success":
        return Verdict("failure", f"Build job {build_result}")
    if detect_result == "cancelled":
        return Verdict("pending", "Detector cancelled (nodes busy); re-request")
    if detect_result == "skipped":
        return Verdict("error", "Detector skipped")
    if detect_code not in {"0", "1"}:
        code = detect_code or "unknown"
        return Verdict("failure", f"Detector infra error (exit {code})")
    return Verdict("success", "Perf gate passed")


def render_comment(
    head_sha: str, verdict: Verdict, run_url: str, report: str = ""
) -> str:
    lines = [
        "### RCCL Perf Regression Gate",
        "",
        f"- Commit: `{head_sha}`",
        f"- Result: **{verdict.state}** — {verdict.description}",
        f"- [Run log]({run_url})",
        "",
    ]
    if not report:
        lines.append("_No perf table: the detector produced no report artifact._")
    elif len(report) > MAXIMUM_REPORT_LENGTH:
        lines.append(report[:MAXIMUM_REPORT_LENGTH])
        lines.append("\n_Report truncated; see the run log for the full table._")
    else:
        lines.append(report)
    return "\n".join(lines) + "\n"


def render_deny_comment(request: GateRequest) -> str:
    return (
        "### RCCL Perf Regression Gate\n\n"
        f"🚫 {request.reason}\n\n"
        f"<sub>`{request.deny_reason}`</sub>\n"
    )


def write_outputs(output: Path | None, values: dict[str, str]) -> None:
    """Append key=value pairs to GITHUB_OUTPUT, or stdout when run locally."""
    rendered = "".join(
        f"{key}={_single_line(value)}\n" for key, value in values.items()
    )
    if output is None:
        sys.stdout.write(rendered)
        return
    with output.open("a", encoding="utf-8") as handle:
        handle.write(rendered)


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"{name} must be set")
    return value


def _api() -> GitHubApi:
    return GitHubApi(
        _environment("GITHUB_REPOSITORY"),
        _environment("GH_TOKEN"),
        os.environ.get("GITHUB_API_URL", "https://api.github.com"),
    )


def _command_resolve(args: argparse.Namespace) -> int:
    repository = _environment("GITHUB_REPOSITORY")
    event_name = _environment("GITHUB_EVENT_NAME")
    event: dict[str, Any] = {}
    if args.event is not None and args.event.exists():
        event = json.loads(args.event.read_text(encoding="utf-8"))
    try:
        request = resolve_request(
            event_name,
            event,
            _api(),
            repository,
            os.environ.get("PERF_COMMAND", "/perf-regression"),
            dict(os.environ),
        )
    except Exception as error:
        # Deliberately broad: an escaping exception would leave deny_reason
        # empty, which silently skips the step that explains this on the PR.
        request = GateRequest(
            False,
            f"The perf gate could not resolve this request: {error}",
            level="error",
            deny_reason="resolve_error",
        )
    write_outputs(args.output, request.outputs())
    if request.deny_reason and args.comment is not None:
        args.comment.write_text(render_deny_comment(request), encoding="utf-8")
    prefix = f"::{request.level}::" if request.level else ""
    print(f"{prefix}{request.reason}")
    # A permission read that fails closed denies everyone, so go red instead of
    # leaving the gate silently pending.
    return 1 if request.level == "error" else 0


def _command_set_status(args: argparse.Namespace) -> int:
    repository = _environment("GITHUB_REPOSITORY")
    set_commit_status(
        _api(),
        repository,
        _environment("HEAD_SHA"),
        _environment("STATUS_CONTEXT"),
        args.state,
        args.description,
        os.environ.get("TARGET_URL", ""),
    )
    return 0


def _command_changed_paths(args: argparse.Namespace) -> int:
    base_ref = validate_ref(_environment("BASE_REF"))
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    fetch_base(base_ref, f"{server}/{_environment('GITHUB_REPOSITORY')}")
    changed = rccl_paths_changed(base_ref)
    write_outputs(args.output, {"rccl_changed": "1" if changed else "0"})
    print(f"projects/rccl/ changed: {changed}")
    return 0


def _command_build(args: argparse.Namespace) -> int:
    rccl_ci_root = _environment("RCCL_CI_ROOT")
    source_config = _environment("SOURCE_CONFIG_JSON")
    print(f"ROCm/cvs revision: {cvs_revision(rccl_ci_root)}")
    build = should_build_rccl(
        _environment("GITHUB_EVENT_NAME"),
        os.environ.get("RCCL_CHANGED", "0"),
        os.environ.get("BUILD_RCCL", "0"),
    )
    if not build:
        write_outputs(args.output, {"config_json": source_config})
        print(f"Skipping the librccl build; using {source_config}")
        return 0

    # Own copy per run, so a concurrent build cannot clobber another PR's config.
    # Emit first: a copy that dies part-way still has to be cleanable.
    run_config = run_config_path(rccl_ci_root, _environment("GITHUB_RUN_ID"))
    write_outputs(args.output, {"config_json": str(run_config)})
    shutil.copyfile(source_config, run_config)
    exported = ",".join(
        [
            "ALL",
            f"CANDIDATE_SRC={_environment('CANDIDATE_SRC')}",
            f"BASE_REF=origin/{validate_ref(_environment('BASE_REF'))}",
            f"CONFIG_JSON={run_config}",
        ]
    )
    subprocess.run(
        [
            "sbatch",
            "--wait",
            f"--export={exported}",
            f"--reservation={_environment('SLURM_RESERVATION')}",
            f"{rccl_ci_root}/cvs/ci/rccl_perf_gate/sbatch/rccl_build.sbatch",
        ],
        check=True,
    )
    return 0


def _command_detect(args: argparse.Namespace) -> int:
    rccl_ci_root = _environment("RCCL_CI_ROOT")
    result = subprocess.run(
        [
            "bash",
            f"{rccl_ci_root}/cvs/ci/rccl_perf_gate/submit_and_poll.sh",
            _environment("CONFIG_JSON"),
        ],
        check=False,
    )
    # finalize reads this: 0 clean, 1 advisory regression, else an infra error.
    write_outputs(args.output, {"exit_code": str(result.returncode)})
    return result.returncode


def _command_cleanup_config(args: argparse.Namespace) -> int:
    removed = cleanup_run_config(
        os.environ.get("CONFIG_JSON", ""),
        _environment("RCCL_CI_ROOT"),
        _environment("GITHUB_RUN_ID"),
    )
    print("Removed the per-run config copy" if removed else "Nothing to clean up")
    return 0


def _command_verdict(args: argparse.Namespace) -> int:
    repository = _environment("GITHUB_REPOSITORY")
    verdict = compute_verdict(
        os.environ.get("BUILD_RESULT", ""),
        os.environ.get("DETECT_RESULT", ""),
        os.environ.get("DETECT_CODE", ""),
    )
    write_outputs(args.output, verdict.outputs())
    head_sha = os.environ.get("HEAD_SHA", "")
    run_url = os.environ.get("TARGET_URL", "")
    # Dispatch has no PR, so nothing here measured a PR's code: stamping the
    # required status from it would green the gate without evidence.
    if head_sha and os.environ.get("PR_NUMBER", ""):
        set_commit_status(
            _api(),
            repository,
            head_sha,
            _environment("STATUS_CONTEXT"),
            verdict.state,
            verdict.description,
            run_url,
        )
    report = ""
    if args.report is not None and args.report.exists():
        report = args.report.read_text(encoding="utf-8")
    args.comment.write_text(
        render_comment(head_sha, verdict, run_url, report), encoding="utf-8"
    )
    print(f"{verdict.state}: {verdict.description}")
    return 0


def _command_enforce(args: argparse.Namespace) -> int:
    state = os.environ.get("STATE", "")
    if state in {"failure", "error"}:
        print(
            f"::error::RCCL Perf Regression Gate {state} "
            "-- see the PR status/comment."
        )
        return 1
    return 0


COMMANDS = {
    "resolve": _command_resolve,
    "set-status": _command_set_status,
    "changed-paths": _command_changed_paths,
    "build": _command_build,
    "detect": _command_detect,
    "cleanup-config": _command_cleanup_config,
    "verdict": _command_verdict,
    "enforce": _command_enforce,
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add(
        name: str, help_text: str, *, output: bool = False
    ) -> argparse.ArgumentParser:
        subparser = subparsers.add_parser(name, help=help_text)
        if output:
            subparser.add_argument(
                "--output",
                type=Path,
                default=None,
                help="GITHUB_OUTPUT path (defaults to stdout)",
            )
        return subparser

    resolve = add("resolve", "Authorize and resolve PR context", output=True)
    resolve.add_argument("--event", type=Path, default=None, help="GITHUB_EVENT_PATH")
    resolve.add_argument("--comment", type=Path, default=None)

    status = add("set-status", "Set the gate's commit status on a SHA")
    status.add_argument("--state", required=True)
    status.add_argument("--description", required=True)

    add("changed-paths", "Fetch the base ref and detect RCCL changes", output=True)
    add("build", "Build the reference + candidate librccl via Slurm", output=True)
    add("detect", "Run the A/B detector and capture its exit code", output=True)
    add("cleanup-config", "Remove this run's per-run config copy")

    verdict = add("verdict", "Compute the verdict and publish it", output=True)
    verdict.add_argument("--report", type=Path, default=None)
    verdict.add_argument("--comment", type=Path, default=Path("comment.md"))

    add("enforce", "Exit non-zero when the verdict was a failure or error")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except subprocess.CalledProcessError as error:
        print(f"error: command failed: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
