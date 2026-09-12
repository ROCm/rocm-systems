#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve the TheRock build that the OrchestrAI run should test.

The OrchestrAI nightly does not build anything. It picks up the artifacts of a
recent TheRock Multi-Arch CI run on this repo and points the test machines at
them.

A Multi-Arch CI run's overall conclusion is NOT a usable signal: the build jobs
routinely succeed while some downstream test job fails, which marks the whole
run "failure" even though every artifact was published. So runs are scanned
newest-first and the first one whose artifacts are actually present in S3 wins.

Resolution is per platform, because a run may have built only one of them.

Usage:
    orchestrai_resolve_build.py [--config .github/orchestrai-config.yml]
                               [--platforms linux,windows] [--print]

Env:
    GH_TOKEN            token for the GitHub API (github.token is enough)
    GITHUB_REPOSITORY   this repo; the default source of the builds
    ORCHESTRAI_SOURCE_REPOSITORY  read builds from another repo instead
                        (set to ROCm/rocm-systems when testing from a fork)
    SOURCE_RUN_ID       optional: pin an explicit run id instead of scanning
    PLATFORMS           optional: comma-separated platform filter

Outputs (to $GITHUB_OUTPUT):
    build_source={"linux": {...}, "windows": {...}}
    has_source=true|false
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable

import yaml

PLATFORMS = ("linux", "windows")

API_TIMEOUT = 60
PROBE_TIMEOUT = 30
PROBE_ATTEMPTS = 3


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def gh_api(path: str, token: str) -> dict[str, Any]:
    """GET a GitHub REST API path and return the decoded JSON body."""
    req = urllib.request.Request(f"https://api.github.com/{path.lstrip('/')}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def artifact_exists(url: str) -> bool:
    """True when the S3 object exists.

    A HEAD is used rather than a bucket listing: a Multi-Arch CI run publishes
    well over 1000 keys, so a listing would need pagination while the exact
    object names are already known.
    """
    for attempt in range(1, PROBE_ATTEMPTS + 1):
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT) as resp:
                return 200 <= resp.status < 300
        except urllib.error.HTTPError as exc:
            # 403 is what a public bucket returns for a missing key when the
            # caller may not ListBucket, so treat it as "absent" like a 404.
            if exc.code in (403, 404):
                return False
            if attempt == PROBE_ATTEMPTS:
                print(
                    f"::warning::HEAD {url} failed with HTTP {exc.code}; "
                    "treating the artifact as absent",
                    file=sys.stderr,
                )
                return False
        except Exception as exc:  # network hiccup — retry, then give up
            if attempt == PROBE_ATTEMPTS:
                print(
                    f"::warning::HEAD {url} failed ({exc}); "
                    "treating the artifact as absent",
                    file=sys.stderr,
                )
                return False
    return False


def artifact_prefix_url(cfg: dict[str, Any], run_id: str, platform: str) -> str:
    src = cfg["source_build"]
    prefix = src["artifact_prefix_template"].format(run_id=run_id, platform=platform)
    return f"{src['artifact_base_url'].rstrip('/')}/{prefix}"


def probe_run(
    cfg: dict[str, Any], run_id: str, platform: str
) -> tuple[bool, list[str]]:
    """Return (usable, families) for one run on one platform.

    `usable` means every required artifact for the platform is present.
    `families` are the AMDGPU families whose marker artifact is also present —
    i.e. what this run can actually be tested on.
    """
    src = cfg["source_build"]
    base = artifact_prefix_url(cfg, run_id, platform)

    required = (src.get("required_artifacts") or {}).get(platform) or []
    if not required:
        print(
            f"::warning::source_build.required_artifacts.{platform} is empty — "
            f"cannot verify run {run_id}",
            file=sys.stderr,
        )
        return False, []

    for name in required:
        if not artifact_exists(f"{base}/{name}"):
            return False, []

    families = [
        family
        for family, marker in (src.get("family_markers") or {}).items()
        if artifact_exists(f"{base}/{marker}")
    ]
    return True, sorted(families)


def candidate_runs(cfg: dict[str, Any], repo: str, token: str) -> list[dict[str, Any]]:
    """Runs of the source workflow, newest first.

    Deliberately NOT filtered on status. A Multi-Arch CI run publishes its
    artifacts when the build stages finish, but the run stays in_progress for
    many more hours while the downstream test jobs execute — recent runs took
    anywhere from 8h to 20h end to end. Waiting for `completed` would mean always
    testing yesterday's build. The artifact probe below is the real completeness
    check, and it is a sound one: the family marker comes from math-libs, the
    last build stage, so its presence implies the core runtime artifacts landed.

    Queued runs are skipped outright — they cannot have artifacts yet.
    """
    src = cfg["source_build"]
    limit = int(src.get("max_runs_to_scan", 40))
    query = urllib.parse.urlencode(
        {"branch": src["branch"], "per_page": min(limit, 100)}
    )
    workflow = urllib.parse.quote(src["workflow"], safe="")
    payload = gh_api(f"repos/{repo}/actions/workflows/{workflow}/runs?{query}", token)
    runs = payload.get("workflow_runs") or []
    return [run for run in runs if run.get("status") != "queued"][:limit]


def source_repository(cfg: dict[str, Any]) -> str:
    """Repo whose Multi-Arch CI runs we read.

    Normally this repo. A fork has no Multi-Arch CI history of its own, so a test
    run there sets ORCHESTRAI_SOURCE_REPOSITORY=ROCm/rocm-systems and reads the
    real builds. Env wins over config so the committed file never has to change.
    """
    return (
        os.environ.get("ORCHESTRAI_SOURCE_REPOSITORY")
        or (cfg.get("source_build") or {}).get("repository")
        or os.environ.get("GITHUB_REPOSITORY", "")
    )


def run_summary(
    cfg: dict[str, Any], run: dict[str, Any], platform: str, repo: str
) -> dict:
    return {
        "repository": repo,
        "run_id": str(run["id"]),
        "run_url": run.get("html_url", ""),
        "sha": run.get("head_sha", ""),
        "branch": run.get("head_branch", ""),
        "created_at": run.get("created_at", ""),
        "status": run.get("status", ""),
        "conclusion": run.get("conclusion") or "",
        "artifact_base_url": artifact_prefix_url(cfg, str(run["id"]), platform),
    }


def resolve(
    cfg: dict[str, Any], repo: str, token: str, platforms: Iterable[str]
) -> dict[str, Any]:
    pinned = (os.environ.get("SOURCE_RUN_ID") or "").strip()
    resolved: dict[str, Any] = {}

    if pinned:
        run = gh_api(f"repos/{repo}/actions/runs/{pinned}", token)
        for platform in platforms:
            usable, families = probe_run(cfg, pinned, platform)
            if not usable:
                print(
                    f"::warning::pinned run {pinned} has no usable {platform} "
                    "artifacts — skipping that platform",
                    file=sys.stderr,
                )
                continue
            resolved[platform] = run_summary(cfg, run, platform, repo) | {
                "families": families
            }
        return resolved

    runs = candidate_runs(cfg, repo, token)
    if not runs:
        print(
            "::warning::no completed runs found for "
            f"{cfg['source_build']['workflow']} on {cfg['source_build']['branch']}",
            file=sys.stderr,
        )
        return resolved

    for platform in platforms:
        for run in runs:
            usable, families = probe_run(cfg, str(run["id"]), platform)
            if not usable:
                continue
            if not families:
                print(
                    f"::warning::run {run['id']} has {platform} artifacts but no "
                    "known family markers — skipping",
                    file=sys.stderr,
                )
                continue
            resolved[platform] = run_summary(cfg, run, platform, repo) | {
                "families": families
            }
            break
        else:
            print(
                f"::warning::no {platform} build with usable artifacts in the last "
                f"{len(runs)} run(s) of {cfg['source_build']['workflow']}",
                file=sys.stderr,
            )
    return resolved


def check_age(cfg: dict[str, Any], resolved: dict[str, Any]) -> None:
    """Warn (do not fail) when the chosen build is older than the policy allows.

    Testing a stale build is still more useful than testing nothing, so this is
    surfaced as a warning in the job log rather than an error.
    """
    from datetime import datetime, timedelta, timezone

    max_age = cfg["source_build"].get("max_age_hours")
    if not max_age:
        return
    cutoff = datetime.now(timezone.utc) - timedelta(hours=float(max_age))
    for platform, info in resolved.items():
        created = info.get("created_at") or ""
        try:
            when = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except ValueError:
            continue
        if when < cutoff:
            age_h = (datetime.now(timezone.utc) - when).total_seconds() / 3600
            print(
                f"::warning::{platform} build {info['run_id']} is {age_h:.0f}h old "
                f"(policy: {max_age}h) — the source workflow may not be running",
                file=sys.stderr,
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=".github/orchestrai-config.yml")
    ap.add_argument("--platforms", default=os.environ.get("PLATFORMS", ""))
    ap.add_argument("--print", action="store_true", dest="do_print")
    args = ap.parse_args()

    wanted = (args.platforms or "").strip().lower()
    platforms = (
        list(PLATFORMS)
        if not wanted or wanted == "all"
        else [p for p in PLATFORMS if p in {x.strip() for x in wanted.split(",")}]
    )

    cfg = load_config(args.config)
    repo = source_repository(cfg)
    if not repo:
        print(
            "::error::no source repository — set ORCHESTRAI_SOURCE_REPOSITORY, "
            "source_build.repository, or GITHUB_REPOSITORY",
            file=sys.stderr,
        )
        sys.exit(1)

    resolved = resolve(cfg, repo, os.environ.get("GH_TOKEN", ""), platforms)
    check_age(cfg, resolved)

    for platform, info in resolved.items():
        print(
            f"{platform}: run {info['run_id']} ({info['sha'][:12]}, "
            f"{info['created_at']}) families={','.join(info['families'])}",
            file=sys.stderr,
        )

    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"build_source={json.dumps(resolved)}\n")
            f.write(f"has_source={'true' if resolved else 'false'}\n")

    if args.do_print:
        print(json.dumps(resolved, indent=2))

    if not resolved:
        print(
            "::error::No TheRock build with usable artifacts was found — nothing "
            "to test. Check that the source workflow is still producing builds.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
