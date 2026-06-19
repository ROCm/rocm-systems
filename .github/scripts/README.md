# `.github/scripts`

Python helpers invoked by the GitHub Actions workflows in `.github/workflows`. They
automate the housekeeping for this **super-repo** (a monorepo that aggregates many
ROCm sub-projects under `projects/<name>` and `shared/<name>` via git subtrees) and
drive its CI.

Most scripts are designed to run inside a runner: they read configuration and inputs
from environment variables (`GH_TOKEN`, `GITHUB_OUTPUT`, `GITHUB_EVENT_NAME`, etc.) and
write results back to `$GITHUB_OUTPUT` for later workflow steps to consume.

The repo layout these scripts assume:

- `projects/<name>/…`, `shared/<name>/…` — vendored sub-project source (subtrees)
- `.github/repos-config.json` — the mapping of each subtree to its upstream sub-repo
- `.github/labels.yml` — the canonical label set for the org

## Shared building blocks

These are imported by the other scripts rather than run directly.

| File | What it does |
|------|--------------|
| `github_cli_client.py` | A thin `GitHubCLIClient` wrapper around the GitHub REST API (using `GH_TOKEN`). Handles auth, pagination, retries, and rate-limiting. Provides the PR/label/user/commit operations the other scripts need (fetch changed files, create/close PRs, sync labels, get merge commits, look up users). |
| `repo_config_model.py` | Pydantic models (`RepoConfig`, `RepoEntry`) describing the schema of `repos-config.json` — each subtree's name, upstream `url`, `branch`, `category`, and its auto-pull/auto-push/source-of-truth flags. |
| `config_loader.py` | Loads and validates `repos-config.json` into `RepoEntry` objects using the models above. |

## Subtree synchronization (super-repo ↔ sub-repos)

These keep the monorepo and the individual upstream repositories in sync.

| File | What it does |
|------|--------------|
| `pr_detect_changed_subtrees.py` | Given a PR, figures out which configured subtrees its changed files touch. Emits the matching `category/name` list to `$GITHUB_OUTPUT` so downstream steps know which sub-repos are affected. Can filter by the auto-pull / auto-push / source-of-truth flags. |
| `pr_merge_sync_patches.py` | Runs after a super-repo PR merges. For each changed subtree, it generates per-commit git patches from the merge commit, clones the upstream sub-repo, applies and re-commits them (preserving the original author), and pushes back. The "push changes out to the real project repos" half of the sync. |
| `import_subrepo_prs.py` | The inbound direction: pulls a list of PRs *from* an upstream sub-repo into the super-repo via `git subtree pull`, opening a corresponding super-repo PR for each (labeled `imported pr`) and reporting any merge conflicts. |
| `merge-submodules.py` | Combines each sub-project's `.gitmodules` into a single top-level `.gitmodules`, namespacing submodule paths/names under their subtree directory. |
| `merge-codeowners.py` | Combines each sub-project's `CODEOWNERS` into one top-level `.github/CODEOWNERS`, prefixing each rule with its subtree directory. |

## Labels

| File | What it does |
|------|--------------|
| `collect-labels.py` | Reads a JSON list of repos, fetches every label from each, de-dupes them, and writes a merged `.github/labels.yml`. |
| `apply-labels.py` | Reads `labels.yml` and creates/updates those labels on a target repo (skipping ones already up to date). The counterpart to `collect-labels.py`. |
| `pr_category_label.py` | Looks at a PR's changed files and derives `project: <name>` / `shared: <name>` category labels, writing the set to add to `$GITHUB_OUTPUT`. |

## PR validation

| File | What it does |
|------|--------------|
| `hip_validate_pr_description.py` | Validates a PR description (from `PR_DESCRIPTION`) against the template — checks each `##` section is either non-empty text or a checklist with at least one box ticked. Exits non-zero with messages on failure. |

## TheRock CI configuration

"TheRock" is the build system used for CI. These two decide what to build and test.

| File | What it does |
|------|--------------|
| `therock_matrix.py` | Static config tables: which subtree maps to which logical project, the cmake flags and test list per project, and which subtrees trigger Windows-only CI. |
| `therock_configure_ci.py` | The decision engine. Based on the GitHub event (push / pull_request / workflow_dispatch / nightly) and the changed paths, it works out which projects to build and test, merges their cmake flags, applies skip rules (docs-only changes, platform-specific subtrees), and emits the resulting build matrix to `$GITHUB_OUTPUT`. |

## Tests

`tests/` contains pytest coverage (`conftest.py`, `test_pr_merge_sync_patches.py`,
`therock_configure_ci_test.py`) for the more involved scripts.

## How it fits together

- **A PR is opened** → `pr_category_label.py` (via the labeler workflow) tags it by
  affected project; `hip_validate_pr_description.py` checks the description;
  `therock_configure_ci.py` (using `therock_matrix.py`) computes the CI build/test matrix.
- **A PR is merged** → `pr_detect_changed_subtrees.py` finds the touched subtrees and
  `pr_merge_sync_patches.py` pushes those changes back out to the upstream sub-repos.
- **Pulling upstream work in** → `import_subrepo_prs.py` brings sub-repo PRs into the
  monorepo.
- **Repo maintenance** → `collect-labels.py` / `apply-labels.py` keep labels consistent,
  and `merge-submodules.py` / `merge-codeowners.py` regenerate aggregated config files.
- Most of the above lean on `github_cli_client.py` for API access and on
  `config_loader.py` + `repo_config_model.py` to read `repos-config.json`.
