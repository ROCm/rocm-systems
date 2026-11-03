#!/usr/bin/env python3
"""Resolve and verify the immutable RCCL candidate used by scheduled CI."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FULL_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
ARTIFACT_BUCKET = "therock-ci-artifacts-external"


class ProvenanceError(RuntimeError):
    """The requested candidate cannot be tied to one immutable source commit."""


def exact_sha(value: str, name: str) -> str:
    value = value.strip().lower()
    if not FULL_SHA.fullmatch(value):
        raise ProvenanceError(f"{name} must be a full 40-character commit SHA, got {value!r}")
    return value


def resolve_candidate_sha(
    *,
    event_name: str,
    github_ref: str,
    github_sha: str,
    candidate_override: str = "",
    artifact_run_id: str = "",
) -> str:
    """Return the exact candidate SHA or fail rather than guessing an artifact.

    A normal scheduled/dispatch run builds the ``develop`` commit GitHub fixed
    in ``github.sha`` when the run started. Reusing a previous artifact is
    deliberately a two-part override: both its run ID and exact source SHA are
    required, so an old run cannot silently become tonight's candidate.
    """
    run_sha = exact_sha(github_sha, "github.sha")
    override = (
        exact_sha(candidate_override, "candidate_sha") if candidate_override.strip() else ""
    )

    if artifact_run_id.strip() and not override:
        raise ProvenanceError(
            "candidate_sha is required when artifact_run_id reuses a previous build"
        )
    if override:
        if override != run_sha and not artifact_run_id.strip():
            raise ProvenanceError(
                "candidate_sha differs from this workflow's source SHA; provide the "
                "artifact_run_id built from that exact candidate"
            )
        return override

    if event_name not in {"schedule", "workflow_dispatch"}:
        raise ProvenanceError(
            f"candidate resolution is only defined for schedule/workflow_dispatch, got {event_name!r}"
        )
    if github_ref != "refs/heads/develop":
        raise ProvenanceError(
            "without an explicit candidate_sha + artifact_run_id, dispatch this workflow "
            f"from develop (got {github_ref!r})"
        )
    return run_sha


def workflow_run_url(repository: str, run_id: str) -> str:
    if not re.fullmatch(r"[1-9][0-9]*", run_id):
        raise ProvenanceError(f"artifact_run_id must be a positive integer, got {run_id!r}")
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", repository):
        raise ProvenanceError(f"invalid GitHub repository {repository!r}")
    return f"https://api.github.com/repos/{repository}/actions/runs/{run_id}"


def fetch_workflow_run(repository: str, run_id: str, token: str) -> dict:
    req = urllib.request.Request(
        workflow_run_url(repository, run_id),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def fetch_artifact_manifest(repository: str, run_id: str) -> tuple[str, dict | None]:
    repo_prefix = repository.replace("/", "-")
    url = (
        f"https://{ARTIFACT_BUCKET}.s3.amazonaws.com/"
        f"{repo_prefix}/{run_id}-linux/manifests/therock_manifest.json"
    )
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return url, json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return url, None
        raise


def fetch_commit(repository: str, sha: str, token: str) -> dict:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/commits/{sha}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def verify_artifact_run(
    *,
    repository: str,
    run_id: str,
    expected_sha: str,
    token: str,
    run: dict | None = None,
    artifact_manifest: dict | None = None,
    build_commit: dict | None = None,
    manifest_lookup: bool = True,
) -> dict:
    """Verify an artifact run's source SHA and return a durable manifest."""
    expected = exact_sha(expected_sha, "candidate_sha")
    data = run if run is not None else fetch_workflow_run(repository, run_id, token)
    actual = exact_sha(str(data.get("head_sha", "")), "artifact run head_sha")
    if actual != expected:
        raise ProvenanceError(
            f"artifact run {run_id} source mismatch: expected {expected}, got {actual}; "
            "refusing a stale candidate"
        )
    manifest_url, fetched_manifest = ("", artifact_manifest)
    if artifact_manifest is None and manifest_lookup:
        manifest_url, fetched_manifest = fetch_artifact_manifest(repository, run_id)
    else:
        repo_prefix = repository.replace("/", "-")
        manifest_url = (
            f"https://{ARTIFACT_BUCKET}.s3.amazonaws.com/"
            f"{repo_prefix}/{run_id}-linux/manifests/therock_manifest.json"
        )

    event = str(data.get("event", ""))
    build_sha = actual
    relationship = "direct"
    if fetched_manifest is not None:
        if str(fetched_manifest.get("github_run_id", "")) != str(run_id):
            raise ProvenanceError(
                f"artifact manifest run mismatch for {run_id}: "
                f"{fetched_manifest.get('github_run_id')!r}"
            )
        matches = [
            entry for entry in fetched_manifest.get("submodules", [])
            if entry.get("submodule_name") == "rocm-systems"
        ]
        if len(matches) != 1:
            raise ProvenanceError(
                f"artifact manifest for run {run_id} has {len(matches)} "
                "rocm-systems source entries"
            )
        if (
            matches[0].get("submodule_path") != "rocm-systems"
            or matches[0].get("submodule_url")
            != "https://github.com/ROCm/rocm-systems.git"
        ):
            raise ProvenanceError(
                f"artifact manifest for run {run_id} has a non-canonical "
                "rocm-systems source entry"
            )
        build_sha = exact_sha(
            str(matches[0].get("pin_sha", "")),
            "artifact manifest rocm-systems pin_sha",
        )
    elif event == "pull_request":
        raise ProvenanceError(
            f"pull-request artifact run {run_id} has no TheRock source manifest"
        )

    if event == "pull_request":
        if build_sha == actual:
            raise ProvenanceError(
                f"pull-request artifact build {build_sha} is not a distinct merge"
            )
        commit = (
            build_commit
            if build_commit is not None
            else fetch_commit(repository, build_sha, token)
        )
        commit_sha = exact_sha(str(commit.get("sha", "")), "artifact build commit")
        parents = [
            exact_sha(str(parent.get("sha", "")), "artifact build parent")
            for parent in commit.get("parents", [])
        ]
        if (
            commit_sha != build_sha
            or len(parents) != 2
            or parents[1] != actual
            or parents[0] == actual
        ):
            raise ProvenanceError(
                f"artifact build {build_sha} is not a two-parent PR merge "
                f"with head {actual} as parent 2"
            )
        relationship = "pull_request_merge"
    elif event in {"push", "schedule", "workflow_dispatch"}:
        if build_sha != actual:
            raise ProvenanceError(
                f"artifact build source mismatch: run head {actual}, build {build_sha}"
            )
    else:
        raise ProvenanceError(f"unsupported artifact source event: {event!r}")

    return {
        "schema_version": 2,
        "candidate_sha": expected,
        "artifact_run_id": str(run_id),
        "artifact_source_sha": actual,
        "artifact_source_url": data.get("html_url", ""),
        "artifact_source_event": event,
        "artifact_source_status": data.get("status", ""),
        "artifact_source_conclusion": data.get("conclusion") or "",
        "artifact_build_sha": build_sha,
        "artifact_build_relationship": relationship,
        "artifact_manifest_url": manifest_url if fetched_manifest is not None else "",
        "artifact_manifest_the_rock_sha": (
            str(fetched_manifest.get("the_rock_commit", ""))
            if fetched_manifest is not None else ""
        ),
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    resolve = sub.add_parser("resolve")
    resolve.add_argument("--event-name", required=True)
    resolve.add_argument("--github-ref", required=True)
    resolve.add_argument("--github-sha", required=True)
    resolve.add_argument("--candidate-sha", default="")
    resolve.add_argument("--artifact-run-id", default="")

    verify = sub.add_parser("verify-run")
    verify.add_argument("--repository", required=True)
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--expected-sha", required=True)
    verify.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "resolve":
            print(
                resolve_candidate_sha(
                    event_name=args.event_name,
                    github_ref=args.github_ref,
                    github_sha=args.github_sha,
                    candidate_override=args.candidate_sha,
                    artifact_run_id=args.artifact_run_id,
                )
            )
            return

        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            raise ProvenanceError("GITHUB_TOKEN is required to verify artifact provenance")
        manifest = verify_artifact_run(
            repository=args.repository,
            run_id=args.run_id,
            expected_sha=args.expected_sha,
            token=token,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps(manifest, sort_keys=True))
    except (OSError, ValueError, ProvenanceError) as exc:
        parser.exit(1, f"candidate provenance error: {exc}\n")


if __name__ == "__main__":
    main()
