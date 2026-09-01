import os
import shlex
import sys
import subprocess
from urllib.parse import urlparse

from github import Github
from git import Repo


class InputValidationError(ValueError):
    """Raised when workflow input is unsafe or malformed."""


def run(cmd, **kwargs):
    # Require argv lists so fork-controlled values (branch names, URLs) are never
    # interpolated into a shell string. Reject str commands to prevent regressions
    # back to shell=True f-string call sites.
    if isinstance(cmd, str):
        raise TypeError("run() requires an argv list; shell=True is not supported")
    print(f">> {shlex.join(cmd)}")
    # do not use shell=True to avoid injection bugs
    subprocess.check_call(cmd, **kwargs)


# Run a git command but ignore non-zero exit codes (e.g. merge --abort when no
# merge is in progress).
def run_allow_failure(cmd, **kwargs):
    try:
        run(cmd, **kwargs)
    except subprocess.CalledProcessError:
        pass


# Validate Git branch syntax and reject option-like names starting with "-" so
# untrusted branch names cannot be interpreted as command-line options.
def validate_branch_name(branch):
    result = subprocess.run(
        ["git", "check-ref-format", "--branch", branch],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stdout.strip() != branch:
        error = result.stderr.strip() or "branch name does not resolve literally"
        raise InputValidationError(f"Invalid branch name {branch!r}: {error}")

    ret = branch
    return ret


# clone_url from the GitHub API is always HTTPS; reject other schemes (e.g. git@)
# before passing the URL to git subtree pull.
def validate_clone_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise InputValidationError(f"Unsupported clone URL: {url!r}")

    repo_path = parsed.path.removeprefix("/")
    if not repo_path.endswith(".git"):
        raise InputValidationError(f"Unexpected clone URL path: {url!r}")

    owner, repo = repo_path[:-4].split("/", 1)
    if not owner or not repo or "/" in repo:
        raise InputValidationError(f"Unexpected clone URL path: {url!r}")

    ret = url
    return ret


def main():
    # 1) Read and validate env vars
    token = os.getenv("GITHUB_TOKEN")
    repo_full = os.getenv("GITHUB_REPOSITORY")
    prefix = os.getenv("SUBPREFIX")
    subrepo = os.getenv("SUBREPO")
    upstream = os.getenv("UPSTREAM")
    target = os.getenv("TARGET", "develop")
    pr_list = os.getenv("PR_LIST", "")

    if not all([token, repo_full, prefix, subrepo, upstream, pr_list]):
        print("ERROR: Missing one or more required environment variables.")
        sys.exit(1)

    target = validate_branch_name(target)
    pr_numbers = [p.strip() for p in pr_list.split(",") if p.strip()]
    conflicted_prs = []  # Track PRs with merge conflicts

    # 2) Init local repo and configure Git user
    repo = Repo(os.getcwd())
    run(["git", "config", "user.name", "systems-assistant[bot]"])
    run(["git", "config", "user.email", "systems-assistant[bot]@users.noreply.github.com"])

    # 3) Init GitHub clients
    gh = Github(token)
    super_repo = gh.get_repo(repo_full)
    sub_repo = gh.get_repo(upstream)

    # 4) Ensure target branch is checked out
    run(["git", "fetch", "origin", target])
    try:
        run(["git", "checkout", target])
    except subprocess.CalledProcessError:
        run(["git", "checkout", "-b", target, f"origin/{target}"])

    # 5) Loop over each PR
    for pr_num in pr_numbers:
        print(f"\n=== Importing PR #{pr_num} ===")
        pr = sub_repo.get_pull(int(pr_num))

        title = pr.title
        body = pr.body or ""
        head_ref = validate_branch_name(pr.head.ref)
        # GitHub API clone_url is always HTTPS (e.g. https://github.com/o/r.git).
        # SSH remotes are in pr.head.repo.ssh_url (git@host:o/r.git); we use
        # clone_url here, so git@ URLs are never passed to git subtree pull.
        head_url = validate_clone_url(pr.head.repo.clone_url)
        is_draft = pr.draft
        author = pr.user.login

        tclean = target.replace("/", "_")
        src_clean = subrepo.replace("/", "_")
        branch = f"import/{tclean}/{src_clean}/pr-{pr_num}"

        try:
            run(["git", "checkout", "-b", branch])
        except subprocess.CalledProcessError:
            run(["git", "branch", "-D", branch])
            run(["git", "checkout", "-b", branch])

        try:
            run(["git", "subtree", "pull", f"--prefix={prefix}", head_url, head_ref])
        except subprocess.CalledProcessError:
            print(f"❌ Merge conflict: subtree pull failed for PR #{pr_num}, skipping.")
            conflicted_prs.append(pr_num)
            run_allow_failure(["git", "merge", "--abort"])
            run(["git", "reset", "--hard"])
            run(["git", "checkout", target])
            continue

        run(["git", "push", "origin", branch])

        footer = (
            "\n\n---\n"
            f"🔁 Imported from [{upstream}#{pr_num}](https://github.com/{upstream}/pull/{pr_num})\n"
            f"🧑‍💻 Originally authored by @{author}"
        )
        full_body = body + footer

        new_pr = super_repo.create_pull(
            title=title,
            body=full_body,
            head=branch,
            base=target,
            draft=is_draft,
        )
        new_pr.add_to_labels("imported pr")

        run(["git", "checkout", target])

    if conflicted_prs:
        print("\n⚠️ The following PRs failed due to merge conflicts:")
        for pr in conflicted_prs:
            print(f" - #{pr}")
    else:
        print("\n✅ All PRs imported successfully without conflicts.")

    print("\nAll imports complete.")


if __name__ == "__main__":
    try:
        main()
    except InputValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
