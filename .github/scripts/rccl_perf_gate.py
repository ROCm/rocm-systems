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
RCCL_PATH_PREFIX = "projects/rccl/"
RUN_CONFIG_PREFIX = "ci_detect_run_"
RUN_CONFIG_SUFFIX = ".json"
REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
# author_association is too coarse: COLLABORATOR includes read and triage.
WRITE_PERMISSIONS = frozenset({"admin", "maintain", "write"})


class PermissionReadError(Exception):
    """The requester's permission level could not be read, so we fail closed."""


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

    def outputs(self) -> dict[str, str]:
        return {
            "authorized": "true" if self.authorized else "false",
            "head_sha": self.head_sha,
            "base_ref": self.base_ref,
            "pr_number": self.pr_number,
            "requester": self.requester,
            "head_repo": self.head_repo,
        }


@dataclass(frozen=True)
class Verdict:
    publish: bool
    state: str
    description: str

    def outputs(self) -> dict[str, str]:
        return {
            "publish": "true" if self.publish else "false",
            "state": self.state,
            "desc": self.description,
        }


class GitHubApi:
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
    if not is_gate_command(str(comment.get("body") or ""), command):
        return GateRequest(False, f"comment is not exactly {command!r}")

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
            f"could not read collaborator permission for @{actor} "
            f"(failing closed): {error}",
            level="error",
        )
    if permission not in WRITE_PERMISSIONS:
        return GateRequest(
            False,
            f"@{actor} lacks write permission (got {permission!r}); not triggering",
            level="warning",
        )

    number = issue.get("number")
    pull = api.get(f"/repos/{repository}/pulls/{int(number)}")
    if not isinstance(pull, dict):
        raise ValueError(f"pull request {number} response is not an object")
    head = pull.get("head") or {}
    head_repo = (head.get("repo") or {}).get("full_name") or ""
    return GateRequest(
        authorized=True,
        reason=f"authorized for @{actor}",
        head_sha=str(head.get("sha") or ""),
        base_ref=validate_ref(str((pull.get("base") or {}).get("ref") or "")),
        pr_number=str(number),
        requester=actor,
        head_repo=str(head_repo),
    )


def is_gate_command(body: str, command: str) -> bool:
    """Exact, trimmed, single-line match -- rejects substrings and quotations."""
    return body.replace("\r", "").strip() == command


def validate_ref(ref: str) -> str:
    """Reject a branch name git would not accept; the base ref is untrusted."""
    if not ref:
        raise ValueError("branch name is empty")
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
    dry_run: bool = False,
) -> None:
    payload = {
        "state": state,
        "context": context,
        "target_url": target_url,
        "description": description[:MAXIMUM_DESCRIPTION_LENGTH],
    }
    if dry_run:
        print(f"[dry-run] POST /repos/{repository}/statuses/{sha} {payload}")
        return
    api.create(f"/repos/{repository}/statuses/{sha}", payload)


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], capture_output=True, check=True, text=True
    )
    return result.stdout.strip()


def fetch_base(base_ref: str) -> None:
    subprocess.run(
        [
            "git",
            "fetch",
            "--no-tags",
            "origin",
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


def cleanup_run_config(config_json: str, rccl_ci_root: str) -> bool:
    """Delete a per-run config copy, refusing any other path."""
    if not config_json:
        return False
    path = Path(config_json)
    if path.parent != Path(rccl_ci_root) / "configs":
        return False
    if not path.name.startswith(RUN_CONFIG_PREFIX):
        return False
    if not path.name.endswith(RUN_CONFIG_SUFFIX):
        return False
    path.unlink(missing_ok=True)
    return True


def compute_verdict(build_result: str, detect_result: str, detect_code: str) -> Verdict:
    """Map job results onto a commit-status state.

    Advisory: a detected regression (detector exit 1) still passes; only build
    failures and detector/infra errors fail the gate.
    """
    if build_result == "cancelled":
        # Superseded by a newer run, which now owns this SHA's status/comment.
        return Verdict(publish=False, state="", description="")
    if build_result != "success":
        return Verdict(True, "failure", f"Build job {build_result}")
    if detect_result in {"cancelled", "skipped"}:
        return Verdict(True, "error", f"Detector {detect_result}")
    if detect_code not in {"0", "1"}:
        return Verdict(
            True, "failure", f"Detector infra error (exit {detect_code or 'unknown'})"
        )
    return Verdict(True, "success", "Perf gate passed")


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
    if report:
        lines.append(report)
    return "\n".join(lines) + "\n"


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
    request = resolve_request(
        event_name,
        event,
        _api(),
        repository,
        os.environ.get("PERF_COMMAND", "/perf-regression"),
        dict(os.environ),
    )
    write_outputs(args.output, request.outputs())
    prefix = f"::{request.level}::" if request.level else ""
    print(f"{prefix}{request.reason}")
    return 0


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
        dry_run=args.dry_run,
    )
    return 0


def _command_changed_paths(args: argparse.Namespace) -> int:
    base_ref = validate_ref(_environment("BASE_REF"))
    fetch_base(base_ref)
    changed = rccl_paths_changed(base_ref)
    write_outputs(args.output, {"rccl_changed": "1" if changed else "0"})
    print(f"projects/rccl/ changed: {changed}")
    return 0


def _command_build(args: argparse.Namespace) -> int:
    rccl_ci_root = _environment("RCCL_CI_ROOT")
    source_config = _environment("SOURCE_CONFIG_JSON")
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
    run_config = run_config_path(rccl_ci_root, _environment("GITHUB_RUN_ID"))
    shutil.copyfile(source_config, run_config)
    # Emit before sbatch so failure cleanup can still find the copy.
    write_outputs(args.output, {"config_json": str(run_config)})
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
        os.environ.get("CONFIG_JSON", ""), _environment("RCCL_CI_ROOT")
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
    if not verdict.publish:
        print("Build cancelled (superseded) -- not publishing status/comment.")
        return 0

    head_sha = os.environ.get("HEAD_SHA", "")
    run_url = os.environ.get("TARGET_URL", "")
    if head_sha:
        set_commit_status(
            _api(),
            repository,
            head_sha,
            _environment("STATUS_CONTEXT"),
            verdict.state,
            verdict.description,
            run_url,
            dry_run=args.dry_run,
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

    status = add("set-status", "Set the gate's commit status on a SHA")
    status.add_argument("--state", required=True)
    status.add_argument("--description", required=True)
    status.add_argument("--dry-run", action="store_true")

    add("changed-paths", "Fetch the base ref and detect RCCL changes", output=True)
    add("build", "Build the reference + candidate librccl via Slurm", output=True)
    add("detect", "Run the A/B detector and capture its exit code", output=True)
    add("cleanup-config", "Remove this run's per-run config copy")

    verdict = add("verdict", "Compute the verdict and publish it", output=True)
    verdict.add_argument("--report", type=Path, default=None)
    verdict.add_argument("--comment", type=Path, default=Path("comment.md"))
    verdict.add_argument("--dry-run", action="store_true")

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
