import os
import re
import sys
import subprocess
from github import Github
from git import Repo


def run(cmd, **kwargs):
    """Run a command as an argv list (no shell). cmd must be a list."""
    if isinstance(cmd, str):
        raise TypeError("run() requires a list, not a string (shell=True is banned)")
    print(f">> {' '.join(cmd)}")
    subprocess.check_call(cmd, **kwargs)


def validate_ref(value, label):
    """Reject values that could inject shell metacharacters.

    Git ref names and clone URLs occasionally contain characters like @, :, and
    . but never whitespace, semicolons, backticks, $, or |.  Reject anything
    that falls outside the expected set so a crafted branch name cannot be
    interpreted as shell syntax even if shell=True were ever re-introduced.
    """
    # Allow the characters that appear in real branch names and HTTPS clone URLs.
    if not re.fullmatch(r"[\w./+:\-@]{1,500}", value):
        print(f"ERROR: {label} contains unexpected characters: {value!r}")
        sys.exit(1)


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

    pr_numbers = [p.strip() for p in pr_list.split(",") if p.strip()]
    conflicted_prs = []  # 🔹 Track PRs with merge conflicts

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
        head_ref = pr.head.ref
        head_url = pr.head.repo.clone_url
        is_draft = pr.draft
        author = pr.user.login

        # Validate fork-controlled values before they touch any subprocess call.
        validate_ref(head_ref, "head_ref")
        validate_ref(head_url, "head_url")

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
            run(["git", "merge", "--abort"])
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

    # 🔹 Summary of failed PRs due to conflict
    if conflicted_prs:
        print("\n⚠️ The following PRs failed due to merge conflicts:")
        for pr in conflicted_prs:
            print(f" - #{pr}")
    else:
        print("\n✅ All PRs imported successfully without conflicts.")

    print("\nAll imports complete.")


if __name__ == "__main__":
    main()
