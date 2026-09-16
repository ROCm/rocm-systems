#!/usr/bin/env python3
"""Path freeze: block develop PRs that change frozen trees.

Driven by .github/workflows/path-freeze.yml, which holds the freeze list.
Reads PR metadata through the GitHub API and never checks out PR code.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "path-freeze.yml"
# GitHub's pull-files REST endpoint returns at most 3000 files.
MAX_LISTED_FILES = 3000
# PRs fetched per GraphQL page, and files fetched per PR within that page.
PR_PAGE_SIZE = 50
FILE_PAGE_SIZE = 100
CHECK_NAME = "path-freeze"
# Never frozen, so a repository-wide freeze can still be lifted by a PR.
EXEMPT_PATHS = (".github/workflows/path-freeze.yml", ".github/scripts/path_freeze.py")
LIFT_HINT = (
    "Lift or narrow a freeze by editing `FREEZES` in "
    "`.github/workflows/path-freeze.yml` and merging that change to develop."
)

# Sweeping the whole backlog one PR at a time would exceed GITHUB_TOKEN's
# hourly request limit, so file lists are batched into one read per page of PRs.
OPEN_PRS_QUERY = """
query($owner: String!, $name: String!, $base: String!, $prs: Int!, $files: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, baseRefName: $base, first: $prs, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        headRefOid
        changedFiles
        files(first: $files) {
          totalCount
          pageInfo { hasNextPage }
          nodes { path changeType }
        }
      }
    }
  }
}
"""


class ConfigError(Exception):
    """The freeze list is unusable, so the check fails closed."""


@dataclass(frozen=True)
class Freeze:
    name: str
    paths: tuple[str, ...]

    def match(self, path: str) -> bool:
        return any(
            (
                fnmatch.fnmatch(path, pattern)
                if any(char in pattern for char in "*?[")
                else path == pattern or path.startswith(pattern.rstrip("/") + "/")
            )
            for pattern in self.paths
        )


@dataclass(frozen=True)
class OpenPR:
    number: int
    head_sha: str
    paths: tuple[str, ...]
    # GraphQL omits the pre-rename path and caps files per page, so those PRs
    # are re-read over REST instead.
    needs_rest: bool


FALSE_WORDS = frozenset({"false", "off", "0", "no"})
TRUE_WORDS = frozenset({"true", "on", "1", "yes"})


def truthy(value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in FALSE_WORDS


def load_freezes(
    env: Mapping[str, str] | None = None, include_disabled: bool = False
) -> tuple[bool, list[Freeze]]:
    """Return (repo-wide freeze, freezes) from the workflow's env."""
    env = os.environ if env is None else env
    freeze_all = truthy(env.get("FREEZE_ENTIRE_REPO"), default=False)
    try:
        entries = json.loads(env.get("FREEZES", "").strip() or "[]")
    except json.JSONDecodeError as exc:
        raise ConfigError(f"FREEZES is not valid JSON: {exc}") from exc
    if not isinstance(entries, list):
        raise ConfigError("FREEZES must be a JSON array of freeze entries.")

    freezes = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigError(f"FREEZES[{index}] must be an object.")
        name = str(entry.get("name") or f"freeze-{index}")
        if not include_disabled and not truthy(str(entry.get("enabled", True)), True):
            continue
        paths = entry.get("paths")
        if (
            not isinstance(paths, list)
            or not paths
            or not all(isinstance(path, str) and path.strip() for path in paths)
        ):
            raise ConfigError(
                f"Freeze `{name}` needs a non-empty list of string paths."
            )
        freezes.append(Freeze(name=name, paths=tuple(paths)))
    return freeze_all, freezes


def load_workflow_env(workflow: Path) -> dict[str, str]:
    """Read the freeze config the way Actions exposes it, as string env vars."""
    import yaml  # only `validate` needs this; the gate itself stays stdlib-only

    config = yaml.safe_load(workflow.read_text())
    return {key: str(value) for key, value in (config.get("env") or {}).items()}


def git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args], capture_output=True, text=True
    )


def path_in_repo(repo_root: Path, path: str) -> bool:
    """True if the path is committed, staged, or present in the working tree.

    Covers all three so validation works in a sparse CI checkout, in a
    pre-commit hook, and on a branch that adds the path in the same change.
    """
    target = path.rstrip("/")
    for revision in (f"HEAD:{target}", f":{target}"):
        if git(repo_root, "cat-file", "-e", revision).returncode == 0:
            return True
    return (repo_root / target).exists()


def tracked_paths(repo_root: Path) -> list[str]:
    listings = (
        git(repo_root, "ls-tree", "-r", "--name-only", "HEAD").stdout,
        git(repo_root, "ls-files").stdout,
    )
    return sorted({line for listing in listings for line in listing.splitlines()})


def validate_config(workflow: Path, repo_root: Path) -> list[str]:
    """Lint the freeze config. Disabled entries are checked too, so a freeze
    is not broken on the day someone re-enables it."""
    problems: list[str] = []
    env = load_workflow_env(workflow)
    known_words = sorted(TRUE_WORDS | FALSE_WORDS)

    switch = env.get("FREEZE_ENTIRE_REPO", "").strip().lower()
    if switch not in TRUE_WORDS | FALSE_WORDS:
        problems.append(
            f"FREEZE_ENTIRE_REPO must be one of {known_words}, got {switch!r}."
        )

    try:
        _, freezes = load_freezes(env, include_disabled=True)
        entries = json.loads(env.get("FREEZES", "").strip() or "[]")
    except ConfigError as exc:
        return problems + [str(exc)]

    for entry in entries:
        if "enabled" not in entry:
            continue
        value = str(entry["enabled"]).strip().lower()
        if value not in TRUE_WORDS | FALSE_WORDS:
            problems.append(
                f"Freeze `{entry.get('name')}` has enabled={value!r}; "
                f"must be one of {known_words}."
            )

    names = [freeze.name for freeze in freezes]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        problems.append(f"Duplicate freeze names: {', '.join(duplicates)}.")

    tracked: list[str] | None = None
    for freeze in freezes:
        for pattern in freeze.paths:
            if pattern.startswith("/") or ".." in Path(pattern).parts:
                problems.append(
                    f"Freeze `{freeze.name}` path `{pattern}` must be repo-relative."
                )
            elif any(char in pattern for char in "*?["):
                if tracked is None:
                    tracked = tracked_paths(repo_root)
                if not fnmatch.filter(tracked, pattern):
                    problems.append(
                        f"Freeze `{freeze.name}` pattern `{pattern}` matches nothing."
                    )
            elif not path_in_repo(repo_root, pattern):
                problems.append(
                    f"Freeze `{freeze.name}` path `{pattern}` is not in the repository."
                )

    for path in EXEMPT_PATHS:
        if not path_in_repo(repo_root, path):
            problems.append(
                f"Exempt path `{path}` is missing, so a repository-wide "
                "freeze could not be lifted."
            )
    return problems


def gh(*args: str, stdin: str | None = None) -> str:
    proc = subprocess.run(
        ["gh", "api", *args], capture_output=True, text=True, input=stdin
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"gh api {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout


def gh_paginate(path: str) -> list[Any]:
    """--paginate emits one JSON array per page; flatten them into one list."""
    out = gh("--paginate", path)
    decoder = json.JSONDecoder()
    items: list[Any] = []
    while out := out.lstrip():
        page, end = decoder.raw_decode(out)
        items.extend(page)
        out = out[end:]
    return items


def gh_graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps({"query": query, "variables": variables})
    payload = json.loads(gh("graphql", "--input", "-", stdin=body))
    if payload.get("errors"):
        raise RuntimeError(f"GraphQL query failed: {payload['errors']}")
    return payload["data"]


def keep(paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(path for path in paths if path and path not in EXEMPT_PATHS)


def changed_paths(repo: str, pr: int) -> tuple[str, ...]:
    """Every path the PR adds, edits, deletes, or renames away from."""
    changed = json.loads(gh(f"repos/{repo}/pulls/{pr}")).get("changed_files")
    if changed is None:
        raise RuntimeError("PR metadata omitted changed_files; failing closed.")
    if changed > MAX_LISTED_FILES:
        raise RuntimeError(
            f"PR changes {changed} files; GitHub lists at most {MAX_LISTED_FILES}. Failing closed."
        )

    files = gh_paginate(f"repos/{repo}/pulls/{pr}/files")
    if len(files) < changed:
        raise RuntimeError(
            f"Listed {len(files)} files but PR reports {changed}. Failing closed."
        )

    return keep(
        path
        for entry in files
        for path in (entry.get("filename"), entry.get("previous_filename"))
    )


def open_prs(repo: str, base: str) -> Iterator[OpenPR]:
    owner, name = repo.split("/", 1)
    cursor = None
    while True:
        page = gh_graphql(
            OPEN_PRS_QUERY,
            {
                "owner": owner,
                "name": name,
                "base": base,
                "prs": PR_PAGE_SIZE,
                "files": FILE_PAGE_SIZE,
                "cursor": cursor,
            },
        )["repository"]["pullRequests"]

        for node in page["nodes"]:
            files = node["files"]
            renamed = any(entry["changeType"] == "RENAMED" for entry in files["nodes"])
            yield OpenPR(
                number=node["number"],
                head_sha=node["headRefOid"],
                paths=keep(entry["path"] for entry in files["nodes"]),
                needs_rest=(
                    renamed
                    or files["pageInfo"]["hasNextPage"]
                    or files["totalCount"] != node["changedFiles"]
                ),
            )

        if not page["pageInfo"]["hasNextPage"]:
            return
        cursor = page["pageInfo"]["endCursor"]


def verdict(
    paths: Iterable[str], freeze_all: bool, freezes: list[Freeze]
) -> tuple[bool, str]:
    if not freeze_all and not freezes:
        return True, "No freezes are active."

    paths = list(paths)
    blocked: list[str] = []
    if freeze_all:
        blocked += [f"- repository-wide freeze: `{path}`" for path in paths]
    for freeze in freezes:
        blocked += [
            f"- freeze `{freeze.name}`: `{path}`"
            for path in paths
            if freeze.match(path)
        ]

    if not blocked:
        active = (
            "the repository-wide freeze"
            if freeze_all
            else ", ".join(f.name for f in freezes)
        )
        return True, f"No frozen paths were changed (active: {active})."

    listing = "\n".join(dict.fromkeys(blocked))
    return (
        False,
        f"This PR cannot land: it changes frozen paths.\n\n{listing}\n\n{LIFT_HINT}",
    )


def evaluate(repo: str, pr: int) -> tuple[bool, str]:
    """Return (allowed, summary). Anything unverifiable fails closed."""
    freeze_all, freezes = load_freezes()
    if not freeze_all and not freezes:
        return True, "No freezes are active."
    return verdict(changed_paths(repo, pr), freeze_all, freezes)


def post_check(repo: str, head_sha: str, allowed: bool, summary: str) -> None:
    payload = {
        "name": CHECK_NAME,
        "head_sha": head_sha,
        "status": "completed",
        "conclusion": "success" if allowed else "failure",
        "output": {"title": CHECK_NAME, "summary": summary},
    }
    gh(
        "--method",
        "POST",
        f"repos/{repo}/check-runs",
        "--input",
        "-",
        stdin=json.dumps(payload),
    )


def report(text: str) -> None:
    print(text)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(text.rstrip() + "\n")


def cmd_evaluate(repo: str, pr: int) -> int:
    allowed, summary = evaluate(repo, pr)
    report(summary)
    if allowed:
        return 0
    print(f"::error::{summary.splitlines()[0]}")
    return 1


def cmd_reevaluate_open(repo: str, base: str) -> int:
    """Re-post the check on open PRs so idle ones follow the current config."""
    freeze_all, freezes = load_freezes()
    lines = [
        f"Repository-wide freeze: {freeze_all}. "
        f"Active freezes: {', '.join(f.name for f in freezes) or 'none'}."
    ]
    checked = blocked = 0
    for pull in open_prs(repo, base):
        try:
            if pull.needs_rest:
                allowed, summary = evaluate(repo, pull.number)
            else:
                allowed, summary = verdict(pull.paths, freeze_all, freezes)
        except Exception as exc:
            allowed, summary = (
                False,
                f"Failed to evaluate PR #{pull.number}; failing closed: {exc}",
            )
        try:
            post_check(repo, pull.head_sha, allowed, summary)
        except Exception as exc:
            lines.append(f"- #{pull.number}: could not post check ({exc})")
            continue
        checked += 1
        if not allowed:
            blocked += 1
            lines.append(f"- #{pull.number} (`{pull.head_sha[:12]}`): FAIL")

    lines.insert(1, f"Re-checked {checked} open `{base}` PR(s); {blocked} blocked.")
    # Blocked PRs carry their own failing check run, so this job still succeeds.
    report("\n".join(lines))
    return 0


def cmd_validate(workflow: Path, repo_root: Path) -> int:
    problems = validate_config(workflow, repo_root)
    if not problems:
        report(f"Freeze config in `{workflow.name}` is valid.")
        return 0
    for problem in problems:
        print(f"::error file={workflow}::{problem}")
    report("Freeze config is invalid:\n" + "\n".join(f"- {p}" for p in problems))
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("evaluate", "reevaluate-open", "validate"))
    parser.add_argument("--repo")
    parser.add_argument("--pr", type=int)
    parser.add_argument("--base", default="develop")
    parser.add_argument("--workflow", type=Path, default=WORKFLOW)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    if args.command == "validate":
        return cmd_validate(args.workflow, args.repo_root)
    if not args.repo:
        parser.error(f"{args.command} requires --repo")
    if args.command == "reevaluate-open":
        return cmd_reevaluate_open(args.repo, args.base)
    if args.pr is None:
        parser.error("evaluate requires --pr")
    return cmd_evaluate(args.repo, args.pr)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"::error::{exc}")
        sys.exit(1)
