#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Plan a trusted corpus comparison and maintain its sticky PR comment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any
import urllib.parse
import urllib.request

BASELINE_ARTIFACT = "gfx1250-b0-a0-sha-pairs"
CANDIDATE_ARTIFACT = "gfx1250-b0-a0-sha-pairs-candidate"
COMMENT_MARKER = "<!-- rocjitsu-gfx1250-b0-a0-translation -->"
COMPARISON_TIMEOUT_SECONDS = 60
MAXIMUM_ARTIFACT_BYTES = 20 * 1024 * 1024
MAXIMUM_MANIFEST_BYTES = 10 * 1024 * 1024
GIT_SHA_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
TRUSTED_BASE = "develop"
WORKFLOW_FILE = "rocjitsu-corpus-tests.yml"
TRUSTED_WORKFLOW_PATH = f".github/workflows/{WORKFLOW_FILE}"


@dataclass(frozen=True)
class ComparisonPlan:
    ready: bool
    reason: str
    pull_number: int | None = None
    candidate_run_id: int | None = None
    baseline_run_id: int | None = None
    candidate_head_sha: str | None = None


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

    def paginate(self, path: str, key: str | None = None) -> list[Any]:
        items = []
        page = 1
        separator = "&" if "?" in path else "?"
        while True:
            payload = self.get(f"{path}{separator}per_page=100&page={page}")
            if key is None:
                batch = payload
            else:
                if not isinstance(payload, dict):
                    raise ValueError(f"GitHub response for {path!r} is not an object")
                batch = payload.get(key)
            if not isinstance(batch, list):
                raise ValueError(f"GitHub response for {path!r} is not a list")
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1

    def create(self, path: str, payload: dict[str, Any]) -> Any:
        return self._request("POST", path, payload)

    def update(self, path: str, payload: dict[str, Any]) -> Any:
        return self._request("PATCH", path, payload)

    def delete(self, path: str) -> Any:
        return self._request("DELETE", path)

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
                "User-Agent": "rocjitsu-corpus-translation-comment",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            response_body = response.read()
        return json.loads(response_body) if response_body else None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        repository = _environment("GITHUB_REPOSITORY")
        api = GitHubApi(
            repository,
            _environment("GH_TOKEN"),
            os.environ.get("GITHUB_API_URL", "https://api.github.com"),
        )
        if args.command == "plan":
            event = json.loads(args.event.read_text(encoding="utf-8"))
            plan = resolve_plan(event, api, repository)
            write_plan_outputs(args.output, plan)
            print(plan.reason)
            return 0
        if args.command == "comment":
            baseline_found = _environment_boolean("BASELINE_FOUND")
            candidate_head_sha = _environment_git_sha("CANDIDATE_HEAD_SHA")
            pull_number = _environment_integer("PR_NUMBER")
            source_run_id = _environment_integer("SOURCE_RUN_ID")
            status, report = compare_candidate(
                args.comparison_tool,
                args.baseline,
                args.candidate,
                baseline_found,
            )
            body = render_comment(
                repository,
                source_run_id,
                candidate_head_sha,
                baseline_found,
                status,
                report,
                os.environ.get("GITHUB_SERVER_URL", "https://github.com"),
            )
            synchronize_comment(api, repository, pull_number, candidate_head_sha, body)
            return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command: {args.command}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="resolve the PR and artifact run IDs")
    plan.add_argument("--event", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    comment = commands.add_parser("comment", help="compare and synchronize the comment")
    comment.add_argument("--candidate", type=Path, required=True)
    comment.add_argument("--baseline", type=Path, required=True)
    comment.add_argument("--comparison-tool", type=Path, required=True)
    return parser.parse_args(argv)


def resolve_plan(event: dict[str, Any], api: Any, repository: str) -> ComparisonPlan:
    run = event.get("workflow_run")
    if not isinstance(run, dict) or run.get("event") != "pull_request":
        return ComparisonPlan(False, "Source workflow was not a pull-request run.")
    if run.get("conclusion") != "success":
        return ComparisonPlan(False, "Source pull-request workflow did not succeed.")
    if run.get("path") != TRUSTED_WORKFLOW_PATH:
        return ComparisonPlan(
            False,
            f"Source workflow path was not {TRUSTED_WORKFLOW_PATH}.",
        )
    head_repository = run.get("head_repository")
    if (
        not isinstance(head_repository, dict)
        or head_repository.get("full_name") != repository
    ):
        return ComparisonPlan(
            False,
            f"Source workflow did not run from {repository}.",
        )
    head_sha = run.get("head_sha")
    if not isinstance(head_sha, str) or GIT_SHA_RE.fullmatch(head_sha) is None:
        return ComparisonPlan(False, "Source workflow has an invalid head SHA.")
    run_id = _positive_integer(run.get("id"), "workflow run ID")
    pull_numbers = {
        pull.get("number")
        for pull in run.get("pull_requests", [])
        if isinstance(pull, dict) and _is_positive_integer(pull.get("number"))
    }
    if not pull_numbers:
        associated = api.paginate(
            f"/repos/{repository}/commits/{urllib.parse.quote(head_sha, safe='')}/pulls"
        )
        pull_numbers = {
            pull.get("number")
            for pull in associated
            if isinstance(pull, dict)
            and pull.get("base", {}).get("repo", {}).get("full_name") == repository
            and _is_positive_integer(pull.get("number"))
        }
    if len(pull_numbers) != 1:
        return ComparisonPlan(
            False,
            f"Expected one pull request for workflow run {run_id}, found "
            f"{len(pull_numbers)}.",
        )
    pull_number = next(iter(pull_numbers))
    pull = api.get(f"/repos/{repository}/pulls/{pull_number}")
    if not isinstance(pull, dict):
        raise ValueError(f"GitHub response for PR #{pull_number} is not an object")
    head = pull.get("head")
    if (
        not isinstance(head, dict)
        or head.get("sha") != head_sha
        or not isinstance(head.get("repo"), dict)
        or head["repo"].get("full_name") != repository
    ):
        return ComparisonPlan(
            False,
            f"Workflow run {run_id} does not match the current same-repository "
            f"head of PR #{pull_number}.",
        )
    base = pull.get("base", {})
    if (
        base.get("repo", {}).get("full_name") != repository
        or base.get("ref") != TRUSTED_BASE
    ):
        return ComparisonPlan(
            False,
            f"PR #{pull_number} does not target {repository}:{TRUSTED_BASE}.",
        )

    artifacts = api.paginate(
        f"/repos/{repository}/actions/runs/{run_id}/artifacts", "artifacts"
    )
    candidates = _named_artifacts(artifacts, CANDIDATE_ARTIFACT)
    if len(candidates) != 1:
        return ComparisonPlan(
            False,
            f"Expected one candidate artifact for workflow run {run_id}, found "
            f"{len(candidates)}.",
        )
    _validate_artifact_size(candidates[0], "candidate")

    baseline_run_id = find_baseline_run(api, repository)
    message = f"Resolved PR #{pull_number} and candidate workflow run {run_id}."
    if baseline_run_id is None:
        message += " No canonical develop baseline is available."
    else:
        message += f" Baseline workflow run is {baseline_run_id}."
    return ComparisonPlan(
        ready=True,
        reason=message,
        pull_number=pull_number,
        candidate_run_id=run_id,
        baseline_run_id=baseline_run_id,
        candidate_head_sha=head_sha,
    )


def find_baseline_run(api: Any, repository: str) -> int | None:
    runs = api.paginate(
        f"/repos/{repository}/actions/workflows/{WORKFLOW_FILE}/runs"
        f"?branch={TRUSTED_BASE}&status=completed",
        "workflow_runs",
    )
    for run in runs:
        if (
            not isinstance(run, dict)
            or run.get("event") not in {"push", "workflow_dispatch"}
            or run.get("conclusion") != "success"
        ):
            continue
        run_id = _positive_integer(run.get("id"), "baseline workflow run ID")
        artifacts = api.paginate(
            f"/repos/{repository}/actions/runs/{run_id}/artifacts", "artifacts"
        )
        baselines = _named_artifacts(artifacts, BASELINE_ARTIFACT)
        if len(baselines) != 1:
            continue
        try:
            _validate_artifact_size(baselines[0], "baseline")
        except ValueError as error:
            print(f"Ignoring workflow run {run_id}: {error}")
            continue
        return run_id
    return None


def write_plan_outputs(path: Path, plan: ComparisonPlan) -> None:
    lines = [f"ready={'true' if plan.ready else 'false'}"]
    if plan.ready:
        if plan.candidate_head_sha is None:
            raise ValueError("ready comparison plan has no candidate head SHA")
        lines.extend(
            [
                f"pull-number={plan.pull_number}",
                f"candidate-run-id={plan.candidate_run_id}",
                f"candidate-head-sha={plan.candidate_head_sha}",
                f"baseline-found={'true' if plan.baseline_run_id else 'false'}",
            ]
        )
        if plan.baseline_run_id is not None:
            lines.append(f"baseline-run-id={plan.baseline_run_id}")
    with path.open("a", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n")


def compare_candidate(
    comparison_tool: Path,
    baseline: Path,
    candidate: Path,
    baseline_found: bool,
) -> tuple[int | None, str | None]:
    _validate_manifest_file(candidate, "candidate")
    if not baseline_found:
        return None, None
    _validate_manifest_file(baseline, "baseline")
    if not comparison_tool.is_file():
        raise ValueError(f"comparison tool is not a file: {comparison_tool}")
    with tempfile.TemporaryDirectory() as temporary_directory:
        report = Path(temporary_directory) / "comparison.md"
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(comparison_tool),
                    "compare",
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--markdown",
                    str(report),
                ],
                capture_output=True,
                text=True,
                timeout=COMPARISON_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise ValueError(
                "manifest comparison exceeded " f"{COMPARISON_TIMEOUT_SECONDS} seconds"
            ) from error
        if result.returncode not in {0, 2, 3}:
            diagnostic = result.stderr.strip() or result.stdout.strip() or "<empty>"
            raise ValueError(
                f"manifest comparison failed with status {result.returncode}: "
                f"{diagnostic[-4096:]}"
            )
        return result.returncode, report.read_text(encoding="utf-8")


def render_comment(
    repository: str,
    source_run_id: int,
    candidate_head_sha: str,
    baseline_found: bool,
    comparison_status: int | None,
    report: str | None,
    server_url: str,
) -> str | None:
    if not server_url.startswith("https://"):
        raise ValueError("GITHUB_SERVER_URL must use HTTPS")
    _validate_git_sha(candidate_head_sha, "candidate head SHA")
    footer = (
        f"<sub>Compared PR head [`{candidate_head_sha}`]"
        f"({server_url.rstrip('/')}/{repository}/commit/{candidate_head_sha}) using "
        f"[rocjitsu-test-corpus run {source_run_id}]"
        f"({server_url.rstrip('/')}/{repository}/actions/runs/{source_run_id}).</sub>"
    )
    if not baseline_found:
        return "\n".join(
            [
                COMMENT_MARKER,
                "## gfx1250 B0-to-A0 translation baseline",
                "",
                "> [!WARNING]",
                "> No canonical develop baseline is available, so this PR could not be compared.",
                "",
                footer,
            ]
        )
    if comparison_status == 0:
        return None
    if comparison_status not in {2, 3} or not report:
        raise ValueError("changed comparison must provide a warning report")
    return f"{COMMENT_MARKER}\n{report.strip()}\n\n{footer}"


def synchronize_comment(
    api: Any,
    repository: str,
    pull_number: int,
    candidate_head_sha: str,
    body: str | None,
) -> None:
    _validate_git_sha(candidate_head_sha, "candidate head SHA")
    comments = api.paginate(f"/repos/{repository}/issues/{pull_number}/comments")
    existing = [
        comment
        for comment in comments
        if isinstance(comment, dict)
        and isinstance(comment.get("user"), dict)
        and comment["user"].get("login") == "github-actions[bot]"
        and isinstance(comment.get("body"), str)
        and COMMENT_MARKER in comment["body"]
    ]
    if body is None:
        for comment in existing:
            if not _pull_head_matches(api, repository, pull_number, candidate_head_sha):
                return
            api.delete(f"/repos/{repository}/issues/comments/{comment['id']}")
        print(f"Removed {len(existing)} resolved translation comment(s).")
        return
    if not existing:
        if not _pull_head_matches(api, repository, pull_number, candidate_head_sha):
            return
        api.create(f"/repos/{repository}/issues/{pull_number}/comments", {"body": body})
        print(f"Created translation comment on PR #{pull_number}.")
        return
    if not _pull_head_matches(api, repository, pull_number, candidate_head_sha):
        return
    api.update(
        f"/repos/{repository}/issues/comments/{existing[0]['id']}", {"body": body}
    )
    for duplicate in existing[1:]:
        if not _pull_head_matches(api, repository, pull_number, candidate_head_sha):
            return
        api.delete(f"/repos/{repository}/issues/comments/{duplicate['id']}")
    print(f"Updated translation comment on PR #{pull_number}.")


def _pull_head_matches(
    api: Any, repository: str, pull_number: int, candidate_head_sha: str
) -> bool:
    pull = api.get(f"/repos/{repository}/pulls/{pull_number}")
    if not isinstance(pull, dict):
        raise ValueError(f"GitHub response for PR #{pull_number} is not an object")
    head = pull.get("head")
    current_head_sha = head.get("sha") if isinstance(head, dict) else None
    if (
        not isinstance(current_head_sha, str)
        or GIT_SHA_RE.fullmatch(current_head_sha) is None
    ):
        raise ValueError(
            f"GitHub response for PR #{pull_number} has an invalid head SHA"
        )
    if current_head_sha == candidate_head_sha:
        return True
    print(
        f"Skipped stale translation comment for PR #{pull_number}: "
        f"compared {candidate_head_sha}, current head is {current_head_sha}."
    )
    return False


def _named_artifacts(artifacts: list[Any], name: str) -> list[dict[str, Any]]:
    return [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict)
        and artifact.get("name") == name
        and not artifact.get("expired", False)
    ]


def _validate_artifact_size(artifact: dict[str, Any], description: str) -> None:
    size = artifact.get("size_in_bytes")
    if not _is_positive_integer(size):
        raise ValueError(f"{description} artifact has an invalid size")
    if size > MAXIMUM_ARTIFACT_BYTES:
        raise ValueError(
            f"{description} artifact is {size} bytes; limit is "
            f"{MAXIMUM_ARTIFACT_BYTES}"
        )


def _validate_manifest_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise ValueError(f"{description} manifest is not a file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > MAXIMUM_MANIFEST_BYTES:
        raise ValueError(
            f"{description} manifest is {size} bytes; expected 1 to "
            f"{MAXIMUM_MANIFEST_BYTES}"
        )


def _environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _environment_boolean(name: str) -> bool:
    value = _environment(name)
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be 'true' or 'false'")
    return value == "true"


def _environment_integer(name: str) -> int:
    value = _environment(name)
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    return _positive_integer(parsed, name)


def _environment_git_sha(name: str) -> str:
    value = _environment(name)
    _validate_git_sha(value, name)
    return value


def _validate_git_sha(value: str, description: str) -> None:
    if GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase Git SHA")


def _positive_integer(value: Any, description: str) -> int:
    if not _is_positive_integer(value):
        raise ValueError(f"{description} must be a positive integer")
    return value


def _is_positive_integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


if __name__ == "__main__":
    sys.exit(main())
