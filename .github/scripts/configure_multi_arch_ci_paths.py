#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Supplement TheRock multi-arch CI path detection for rocm-systems.

TheRock's shared ``configure_external_repo_ci.py`` discovers subtree repos from
``repos-config.json``. rocm-systems also has CI-relevant directories that are
not standalone subtree repos. This script maps those paths to the build/test
selectors consumed later in the active multi-arch workflow.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _matches_prefix(path: str, prefix: str) -> bool:
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


def _load_config(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    paths = data.get("paths", [])
    if not isinstance(paths, list):
        raise ValueError("'paths' must be a list")
    return paths


def _get_modified_paths_api(github_repo: str, base_sha: str, head_sha: str) -> list[str]:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN or GITHUB_TOKEN is required")

    url = f"https://api.github.com/repos/{github_repo}/compare/{base_sha}...{head_sha}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GitHub compare API failed: HTTP {e.code}") from e

    files = data.get("files", [])
    if len(files) >= 300:
        raise RuntimeError("GitHub compare API returned a truncated file list")
    return [file["filename"] for file in files]


def compute_outputs(
    modified_paths: list[str],
    original_changed_projects: str,
    config_paths: list[dict],
) -> dict[str, str]:
    build_projects = _split_csv(original_changed_projects)
    test_projects = _split_csv(original_changed_projects)
    matched_platforms: set[str] = set()
    matched_any = bool(build_projects)
    build_all = False
    test_all = False

    for entry in config_paths:
        prefix = entry.get("prefix", "")
        if not prefix or not any(_matches_prefix(path, prefix) for path in modified_paths):
            continue

        matched_any = True
        matched_platforms.update(entry.get("platforms", []))
        build_all = build_all or bool(entry.get("build_all", False))
        test_all = test_all or bool(entry.get("test_all", False))
        build_projects.extend(entry.get("build_projects", []))
        test_projects.extend(entry.get("test_projects", []))

    if build_all:
        build_projects = []
    if test_all:
        test_projects = []

    linux_enabled = ""
    windows_enabled = ""
    if matched_any and not _split_csv(original_changed_projects) and matched_platforms:
        linux_enabled = str("linux" in matched_platforms).lower()
        windows_enabled = str("windows" in matched_platforms).lower()

    return {
        "build_changed_projects": ",".join(_dedupe(build_projects)),
        "test_changed_projects": ",".join(_dedupe(test_projects)),
        "build_run_all_tests": str(build_all).lower(),
        "test_run_all_tests": str(test_all).lower(),
        "linux_enabled": linux_enabled,
        "windows_enabled": windows_enabled,
    }


def set_github_output(outputs: dict[str, str]) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT")
    if not output_file:
        for key, value in outputs.items():
            print(f"{key}={value}")
        return
    with open(output_file, "a", encoding="utf-8") as f:
        for key, value in outputs.items():
            f.write(f"{key}={value}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-repo", default="")
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--head-sha", default="")
    parser.add_argument("--config-path", default=".github/multi-arch-ci-paths.json")
    parser.add_argument("--original-changed-projects", default="")
    parser.add_argument("--modified-path", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_paths = _load_config(Path(args.config_path))
    modified_paths = args.modified_path
    if not modified_paths:
        if not args.github_repo or not args.base_sha or not args.head_sha:
            print(
                "Either --modified-path or --github-repo/--base-sha/--head-sha is required",
                file=sys.stderr,
            )
            return 1
        modified_paths = _get_modified_paths_api(
            args.github_repo,
            args.base_sha,
            args.head_sha,
        )
    outputs = compute_outputs(
        modified_paths=modified_paths,
        original_changed_projects=args.original_changed_projects,
        config_paths=config_paths,
    )
    set_github_output(outputs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
