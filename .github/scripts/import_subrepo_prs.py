import os
import sys
import subprocess
from github import Github
from git import Repo

def run(cmd, **kwargs):
    print(f">> {cmd}")
    subprocess.check_call(cmd, shell=True, **kwargs)

def main():
    # 1) Read and validate env vars
    token    = os.getenv("GITHUB_TOKEN")
    repo_full= os.getenv("GITHUB_REPOSITORY")
    prefix   = os.getenv("SUBPREFIX")
    subrepo  = os.getenv("SUBREPO")
    upstream = os.getenv("UPSTREAM")
    target   = os.getenv("TARGET", "develop")
    pr_list  = os.getenv("PR_LIST", "")

    if not all([token, repo_full, prefix, subrepo, upstream, pr_list]):
        print("ERROR: Missing one or more required environment variables.")
        sys.exit(1)

    pr_numbers = [p.strip() for p in pr_list.split(",") if p.strip()]
    conflicted_prs = []  # 🔹 Track PRs with merge conflicts

    # 2) Init local repo and configure Git user
    repo = Repo(os.getcwd())
    run("git config user.name  'systems-assistant[bot]'")
    run("git config user.email 'systems-assistant[bot]@users.noreply.github.com'")

    # 3) Init GitHub clients
    gh = Github(token)
    super_repo = gh.get_repo(repo_full)
    sub_repo   = gh.get_repo(upstream)

    # 4) Ensure target branch is checked out
    run(f"git fetch origin {target}")
    try:
        run(f"git checkout {target}")
    except subprocess.CalledProcessError:
        run(f"git checkout -b {target} origin/{target}")

    # 5) Loop over each PR
    for pr_num in pr_numbers:
        print(f"\n=== Importing PR #{pr_num} ===")
        pr = sub_repo.get_pull(int(pr_num))

        title    = pr.title
        body     = pr.body or ""
        head_ref = pr.head.ref
        head_url = pr.head.repo.clone_url
        is_draft = pr.draft
        author   = pr.user.login

        tclean   = target.replace("/", "_")
        src_clean= subrepo.replace("/", "_")
        branch   = f"import/{tclean}/{src_clean}/pr-{pr_num}"

        try:
            run(f"git checkout -b {branch}")
        except subprocess.CalledProcessError:
            run(f"git branch -D {branch}")
            run(f"git checkout -b {branch}")

        try:
            run(f"git subtree pull --prefix={prefix} {head_url} {head_ref}")
        except subprocess.CalledProcessError:
            print(f"❌ Merge conflict: subtree pull failed for PR #{pr_num}, skipping.")
            conflicted_prs.append(pr_num)  # 🔹 Track the failed PR
            run(f"git merge --abort || true")  # Clean up merge state if needed
            run(f"git reset --hard")  # Ensure clean state
            run(f"git checkout {target}")
            continue

        run(f"git push origin {branch}")

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

        run(f"git checkout {target}")

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
