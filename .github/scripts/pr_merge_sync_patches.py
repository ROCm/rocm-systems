#!/usr/bin/env python3

"""
Sync Patches to Subrepositories
-------------------------------

This script is part of the monorepo synchronization system. It runs after a monorepo pull request
is merged and applies relevant changes to the corresponding sub-repositories using Git patches.

- Uses the merge commit of the monorepo PR to extract subtree changes.
- Detects file-level changes including adds, deletes, and renames.
- Applies changes directly using file copy/move/delete as needed.
- Squashes all commits per subtree into one before pushing.
- Uses the repos-config.json file to map subtrees to sub-repos.
- Assumes this script is run from the root of the monorepo.

Arguments:
    --repo      : Full repository name (e.g., org/repo)
    --pr        : Pull request number
    --subtrees  : A newline-separated list of subtree paths in category/name format (e.g., projects/rocBLAS)
    --config    : OPTIONAL, path to the repos-config.json file
    --dry-run   : If set, will only log actions without making changes.
    --debug     : If set, enables detailed debug logging.

Example Usage:
    python pr_merge_sync_patches.py --repo ROCm/rocm-systems --pr 123 --subtrees "$(printf 'projects/rocBLAS\nprojects/hipBLASLt\nshared/rocSPARSE')" --dry-run --debug
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional, List, Tuple
from pathlib import Path
from github_cli_client import GitHubCLIClient
from config_loader import load_repo_config
from repo_config_model import RepoEntry

logger = logging.getLogger(__name__)

def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Apply subtree patches to sub-repositories.")
    parser.add_argument("--repo", required=True, help="Full repository name (e.g., org/repo)")
    parser.add_argument("--pr", required=True, type=int, help="Pull request number")
    parser.add_argument("--subtrees", required=True, help="Newline-separated list of changed subtrees (category/name)")
    parser.add_argument("--config", required=False, default=".github/repos-config.json", help="Path to the repos-config.json file")
    parser.add_argument("--dry-run", action="store_true", help="If set, only logs actions without making changes.")
    parser.add_argument("--debug", action="store_true", help="If set, enables detailed debug logging.")
    return parser.parse_args(argv)

def get_subtree_info(config: List[RepoEntry], subtrees: List[str]) -> List[RepoEntry]:
    """Return config entries matching the given subtrees in category/name format."""
    requested = set(subtrees)
    matched = [
        entry for entry in config
        if f"{entry.category}/{entry.name}" in requested
    ]
    missing = requested - {f"{e.category}/{e.name}" for e in matched}
    if missing:
        logger.warning(f"Some subtrees not found in config: {', '.join(sorted(missing))}")
    return matched

def _run_git(args: List[str], cwd: Optional[Path] = None) -> str:
    """Run a git command and return stdout."""
    cmd = ["git"] + args
    logger.debug(f"Running git command: {' '.join(cmd)} (cwd={cwd})")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"Git command failed: {' '.join(cmd)}\n{result.stderr}")
        raise RuntimeError(f"Git command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout.strip()

def _clone_subrepo(repo_url: str, branch: str, destination: Path) -> None:
    """Clone a specific branch from the given GitHub repository into the destination path."""
    _run_git([
        "clone",
        "--branch", branch,
        "--single-branch",
        f"https://github.com/{repo_url}",
        str(destination)
    ])
    logger.debug(f"Cloned {repo_url} into {destination}")

def _configure_git_user(repo_path: Path) -> None:
    """Configure git user.name and user.email for the given repository directory."""
    _run_git(["config", "user.name", "systems-assistant[bot]"], cwd=repo_path)
    _run_git(["config", "user.email", "systems-assistant[bot]@users.noreply.github.com"], cwd=repo_path)

def _apply_patch(repo_path: Path, patch_path: Path, rel_file_path: Path, monorepo_path: Path, prefix: str) -> None:
    """Try to apply a patch; if it fails, fallback to full file replacement."""
    try:
        _run_git(["am", str(patch_path)], cwd=repo_path)
        logger.info(f"Applied patch {patch_path.name} successfully")
    except RuntimeError as e:
        logger.warning(f"Patch {patch_path.name} failed to apply; falling back to full file copy")

        # Construct source and destination
        monorepo_file = monorepo_path / prefix / rel_file_path
        subrepo_file = repo_path / rel_file_path
        subrepo_file.parent.mkdir(parents=True, exist_ok=True)

        if not monorepo_file.exists():
            raise RuntimeError(f"Fallback failed: {monorepo_file} does not exist")

        shutil.copyfile(monorepo_file, subrepo_file)
        _run_git(["add", str(rel_file_path)], cwd=repo_path)
        logger.info(f"Copied {monorepo_file} -> {subrepo_file}")

def _set_authenticated_remote(repo_path: Path, repo_url: str) -> None:
    """Set the push URL to use the GitHub App token from GH_TOKEN env."""
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN environment variable is not set")
    remote_url = f"https://x-access-token:{token}@github.com/{repo_url}.git"
    _run_git(["remote", "set-url", "origin", remote_url], cwd=repo_path)

def _push_changes(repo_path: Path, branch: str) -> None:
    """Push the commit to origin of branch."""
    _run_git(["push", "origin", branch], cwd=repo_path)
    logger.debug(f"Pushed changes from {repo_path} to origin")

def generate_file_level_patches(prefix: str, merge_sha: str, output_dir: Path) -> tuple[list[str], list[str], list[tuple[str, str]], list[str], list[Path]]:
    """Generate one patch per modified file, and collect adds, deletes, and renames."""
    diff_output = _run_git([
        "diff", "--name-status", "-M", f"{merge_sha}^!", "--", prefix
    ])

    added_files = []
    deleted_files = []
    renamed_files = []
    modified_files = []
    patch_files = []

    for line in diff_output.splitlines():
        parts = line.split('\t')
        status = parts[0]
        if status == 'A':
            added_files.append(parts[1])
        elif status == 'M':
            file_path = parts[1]
            patch_path = output_dir / (file_path.replace("/", "_") + ".patch")
            _run_git([
                "format-patch",
                "-1", merge_sha,
                f"--relative={prefix}",
                "--output", str(patch_path),
                "--", file_path
            ])
            patch_files.append(patch_path)
            modified_files.append(file_path)
        elif status == 'D':
            deleted_files.append(parts[1])
        elif status.startswith('R'):
            renamed_files.append((parts[1], parts[2]))

    logger.debug(f"Generated {len(patch_files)} modified file patches, "
                 f"{len(added_files)} added, {len(deleted_files)} deleted, "
                 f"{len(renamed_files)} renamed under {prefix}")
    return added_files, deleted_files, renamed_files, modified_files, patch_files

def resolve_patch_author(client: GitHubCLIClient, repo: str, pr: int) -> tuple[str, str]:
    """Determine the appropriate author for the patch"""
    pr_data = client.get_pr_by_number(repo, pr)
    body = pr_data.get("body", "") or ""
    match = re.search(r"Originally authored by @([A-Za-z0-9_-]+)", body)
    if match:
        username = match.group(1)
        logger.debug(f"Found originally authored username in PR body: @{username}")
    else:
        username = pr_data["user"]["login"]
        logger.debug(f"No explicit original author, using PR author: @{username}")
    name, email = client.get_user(username)
    return name or username, email

def apply_patches_and_squash(entry: RepoEntry, monorepo_url: str, monorepo_pr: int,
                             added_files: list[str], deleted_files: list[str], renamed_files: list[tuple[str, str]],
                             modified_files: list[str], modified_patch_paths: list[Path],
                             author_name: str, author_email: str, merge_sha: str, dry_run: bool = False) -> None:
    """
    Clone the subrepo, apply file-level patches each as a commit,
    delete files with git rm, copy added files, rename files,
    then squash all new commits into one before pushing.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = f"{entry.category}/{entry.name}"
        if dry_run:
            logger.info(f"[Dry-run] Sync for {entry.name}:")
            prefix_path = Path(prefix)

            if added_files:
                logger.info("  Added files:")
                for f in added_files:
                    short_path = Path(f).relative_to(prefix_path)
                    logger.info(f"    {short_path}")
            if deleted_files:
                logger.info("  Deleted files:")
                for f in deleted_files:
                    short_path = Path(f).relative_to(prefix_path)
                    logger.info(f"    {short_path}")
            if renamed_files:
                logger.info("  Renamed files:")
                for old, new in renamed_files:
                    old_rel = Path(old).relative_to(prefix_path)
                    new_rel = Path(new).relative_to(prefix_path)
                    logger.info(f"    {old_rel} -> {new_rel}")
            if modified_files:
                logger.info("  Modified files (via patch):")
                for f in modified_files:
                    short_path = Path(f).relative_to(prefix_path)
                    logger.info(f"    {short_path}")
            if not (added_files or deleted_files or renamed_files or modified_files or modified_patch_paths):
                logger.info("  No changes detected.")
            return

        subrepo_path = Path(tmpdir) / entry.name
        _clone_subrepo(entry.url, entry.branch, subrepo_path)

        _configure_git_user(subrepo_path)

        # Get current HEAD commit (before applying patches)
        base_commit = _run_git(["rev-parse", "HEAD"], cwd=subrepo_path)

        # Handle deletes
        for file_path in deleted_files:
            rel_path = file_path[len(prefix)+1:] if file_path.startswith(prefix + "/") else file_path
            _run_git(["rm", rel_path], cwd=subrepo_path)

        # Handle renames
        for old, new in renamed_files:
            old_rel = old[len(prefix)+1:] if old.startswith(prefix + "/") else old
            new_rel = new[len(prefix)+1:] if new.startswith(prefix + "/") else new
            _run_git(["mv", old_rel, new_rel], cwd=subrepo_path)

        # Handle adds
        for file_path in added_files:
            rel_path = file_path[len(prefix)+1:] if file_path.startswith(prefix + "/") else file_path
            src = Path(prefix) / rel_path
            dst = subrepo_path / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

        # Handle modified files (apply patches one by one)
        for patch_path, full_file_path in zip(modified_patch_paths, modified_files):
            rel_path = full_file_path[len(prefix)+1:] if full_file_path.startswith(prefix + "/") else full_file_path
            logger.debug(f"Applying patch {patch_path.name} to {entry.name} at {rel_path}")
            _apply_patch(subrepo_path, patch_path, Path(rel_path), Path.cwd(), prefix)

        # Final squash
        commit_msg = f"[rocm-systems] {monorepo_url}#{monorepo_pr} (commit {merge_sha[:7]})\n\n" + \
                     _run_git(["log", "-1", "--pretty=%B", merge_sha])
        _run_git(["reset", "--soft", base_commit], cwd=subrepo_path)
        _run_git(["commit", "-m", commit_msg, "--author", f"{author_name} <{author_email}>"], cwd=subrepo_path)

        _set_authenticated_remote(subrepo_path, entry.url)
        _push_changes(subrepo_path, entry.branch)

def main(argv: Optional[List[str]] = None) -> None:
    """Main function to apply patches to sub-repositories."""
    args = parse_arguments(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO
    )
    client = GitHubCLIClient()
    config = load_repo_config(args.config)
    subtrees = [line.strip() for line in args.subtrees.splitlines() if line.strip()]
    relevant_subtrees = get_subtree_info(config, subtrees)
    merge_sha = client.get_squash_merge_commit(args.repo, args.pr)
    logger.debug(f"Merge commit for PR #{args.pr} in {args.repo}: {merge_sha}")
    _run_git(["checkout", merge_sha])
    logger.info(f"Checked out merge commit {merge_sha} for patch operations")
    for entry in relevant_subtrees:
        prefix = f"{entry.category}/{entry.name}"
        logger.debug(f"Processing subtree {prefix}")
        with tempfile.TemporaryDirectory() as tmpdir:
            patch_dir = Path(tmpdir)
            # Generate patches and lists of adds/deletes/renames
            added_files, deleted_files, renamed_files, modified_files, modified_patch_paths,  = generate_file_level_patches(prefix, merge_sha, patch_dir)
            if not (added_files or deleted_files or renamed_files or modified_files or modified_patch_paths):
                logger.info(f"No changes to apply for {prefix}")
                continue
            author_name, author_email = resolve_patch_author(client, args.repo, args.pr)
            apply_patches_and_squash(entry, args.repo, args.pr,
                                     added_files, deleted_files, renamed_files, modified_files, modified_patch_paths,
                                     author_name, author_email, merge_sha,
                                     args.dry_run)

if __name__ == "__main__":
    main()
