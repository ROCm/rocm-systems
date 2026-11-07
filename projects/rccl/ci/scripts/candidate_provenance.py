#!/usr/bin/env python3
"""Resolve and verify the immutable RCCL candidate used by scheduled CI."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

FULL_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
ARTIFACT_BUCKET = "therock-ci-artifacts-external"
ARTIFACT_GROUP_TARGETS = {
    "gfx94X-dcgpu": ("gfx942",),
    "gfx950-dcgpu": ("gfx950",),
}


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


def artifact_prefix(repository: str, run_id: str) -> str:
    repo_prefix = repository.replace("/", "-")
    return f"{repo_prefix}/{run_id}-linux/"


def artifact_object_url(repository: str, run_id: str, relative: str) -> str:
    key = artifact_prefix(repository, run_id) + relative
    return (
        f"https://{ARTIFACT_BUCKET}.s3.amazonaws.com/"
        + urllib.parse.quote(key, safe="/")
    )


def fetch_artifact_manifests(repository: str, run_id: str) -> list[dict]:
    prefix = artifact_prefix(repository, run_id)
    manifest_prefix = prefix + "manifests/"
    listing_url = (
        f"https://{ARTIFACT_BUCKET}.s3.amazonaws.com/?"
        + urllib.parse.urlencode({"list-type": "2", "prefix": manifest_prefix})
    )
    with urllib.request.urlopen(listing_url, timeout=60) as response:
        root = ET.fromstring(response.read())
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    if root.findtext("s3:IsTruncated", namespaces=namespace) == "true":
        raise ProvenanceError(
            f"artifact listing for run {run_id} is truncated"
        )
    keys = [
        entry.findtext("s3:Key", namespaces=namespace)
        for entry in root.findall("s3:Contents", namespace)
    ]
    manifests = []
    for key in sorted(
        key for key in keys
        if key and key.endswith("/therock_manifest.json")
    ):
        relative = key.removeprefix(prefix)
        url = artifact_object_url(repository, run_id, relative)
        with urllib.request.urlopen(url, timeout=60) as response:
            manifests.append({
                "path": relative,
                "url": url,
                "manifest": json.load(response),
            })
    return manifests


def artifact_object_exists(repository: str, run_id: str, relative: str) -> bool:
    request = urllib.request.Request(
        artifact_object_url(repository, run_id, relative), method="HEAD"
    )
    try:
        with urllib.request.urlopen(request, timeout=60):
            return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
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
    artifact_manifests: list[dict] | None = None,
    build_commit: dict | None = None,
    artifact_group: str = "",
    available_objects: set[str] | None = None,
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
    event = str(data.get("event", ""))
    workflow_path = str(data.get("path", ""))
    direct_workflow = ".github/workflows/therock-rccl-ci.yml"
    multiarch_workflow = ".github/workflows/therock-multi-arch-ci.yml"
    if workflow_path not in {direct_workflow, multiarch_workflow}:
        raise ProvenanceError(
            f"unsupported artifact producer workflow: {workflow_path!r}"
        )

    manifests = (
        artifact_manifests
        if artifact_manifests is not None
        else fetch_artifact_manifests(repository, run_id)
    )
    the_rock_shas = set()
    manifest_pins = set()
    for item in manifests:
        manifest = item.get("manifest")
        if not isinstance(manifest, dict):
            raise ProvenanceError("artifact manifest entry is missing its JSON object")
        if str(manifest.get("github_run_id", "")) != str(run_id):
            raise ProvenanceError(
                f"artifact manifest run mismatch for {run_id}: "
                f"{manifest.get('github_run_id')!r}"
            )
        the_rock_shas.add(exact_sha(
            str(manifest.get("the_rock_commit", "")),
            "artifact manifest TheRock SHA",
        ))
        matches = [
            entry for entry in manifest.get("submodules", [])
            if entry.get("submodule_name") == "rocm-systems"
        ]
        if len(matches) != 1:
            raise ProvenanceError(
                f"artifact manifest {item.get('path')!r} has {len(matches)} "
                "rocm-systems source entries"
            )
        if (
            matches[0].get("submodule_path") != "rocm-systems"
            or matches[0].get("submodule_url")
            != "https://github.com/ROCm/rocm-systems.git"
        ):
            raise ProvenanceError(
                f"artifact manifest {item.get('path')!r} has a non-canonical "
                "rocm-systems entry"
            )
        manifest_pins.add(exact_sha(
            str(matches[0].get("pin_sha", "")),
            "artifact manifest rocm-systems dependency pin",
        ))
    if len(the_rock_shas) > 1 or len(manifest_pins) > 1:
        raise ProvenanceError(
            "artifact manifests disagree across families: "
            f"TheRock={sorted(the_rock_shas)}, rocm-systems={sorted(manifest_pins)}"
        )

    build_sha = actual
    relationship = "direct"
    authority = "workflow_run_head"
    if workflow_path == multiarch_workflow:
        roots = [
            item for item in manifests
            if item.get("path") == "manifests/therock_manifest.json"
        ]
        if len(roots) != 1:
            raise ProvenanceError(
                f"multi-arch artifact run {run_id} has {len(roots)} "
                "authoritative root manifests"
            )
        build_sha = next(iter(manifest_pins))
        authority = "multiarch_root_manifest"
    else:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", artifact_group):
            raise ProvenanceError(
                f"invalid artifact_group for manifest selection: {artifact_group!r}"
            )
        targets = ARTIFACT_GROUP_TARGETS.get(artifact_group)
        if not targets:
            raise ProvenanceError(
                f"unsupported direct artifact_group: {artifact_group!r}"
            )
        if manifests:
            family_path = f"manifests/{artifact_group}/therock_manifest.json"
            matches = [item for item in manifests if item.get("path") == family_path]
            if len(matches) != 1:
                raise ProvenanceError(
                    f"direct artifact run {run_id} has {len(matches)} manifests "
                    f"for group {artifact_group!r}"
                )
        required_objects = {
            "rccl_lib_generic.tar.zst",
            "rccl_lib_generic.tar.zst.sha256sum",
        }
        for target in targets:
            required_objects.update({
                f"rccl_lib_{target}.tar.zst",
                f"rccl_lib_{target}.tar.zst.sha256sum",
            })
        if available_objects is None:
            missing_objects = sorted(
                relative for relative in required_objects
                if not artifact_object_exists(repository, run_id, relative)
            )
        else:
            missing_objects = sorted(required_objects - available_objects)
        if missing_objects:
            raise ProvenanceError(
                f"artifact run {run_id} is missing required RCCL objects: "
                f"{missing_objects}"
            )
        # This workflow checks out the requested rocm-systems ref directly.
        # Its per-family manifest reports TheRock's dependency pin, not the
        # workspace source that produced RCCL.
        build_sha = actual
        authority = "direct_workflow_head"

    if event == "pull_request":
        if workflow_path != multiarch_workflow:
            raise ProvenanceError(
                f"pull-request artifacts are unsupported from {workflow_path!r}"
            )
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
        "artifact_build_authority": authority,
        "artifact_manifest_urls": [str(item.get("url", "")) for item in manifests],
        "artifact_manifest_the_rock_sha": (
            next(iter(the_rock_shas)) if the_rock_shas else ""
        ),
        "artifact_manifest_rocm_systems_dependency_pin": (
            next(iter(manifest_pins)) if manifest_pins else ""
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
    verify.add_argument("--artifact-group", default="")
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
            artifact_group=args.artifact_group,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps(manifest, sort_keys=True))
    except (OSError, ValueError, ProvenanceError) as exc:
        parser.exit(1, f"candidate provenance error: {exc}\n")


if __name__ == "__main__":
    main()
