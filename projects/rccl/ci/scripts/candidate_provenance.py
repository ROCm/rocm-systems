#!/usr/bin/env python3
"""Resolve and verify the immutable RCCL candidate used by scheduled CI."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FULL_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


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


def verify_artifact_run(
    *,
    repository: str,
    run_id: str,
    expected_sha: str,
    token: str,
    run: dict | None = None,
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
    return {
        "schema_version": 1,
        "candidate_sha": expected,
        "artifact_run_id": str(run_id),
        "artifact_source_sha": actual,
        "artifact_source_url": data.get("html_url", ""),
        "artifact_source_event": data.get("event", ""),
        "artifact_source_status": data.get("status", ""),
        "artifact_source_conclusion": data.get("conclusion") or "",
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
