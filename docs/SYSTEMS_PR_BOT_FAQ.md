# Systems PR Bot — Policy FAQ Doc

**Systems PR Bot** is an automated Pull Request (PR) gatekeeper.
On every Pull Request, it runs a set of policy checks — PR description,
forbidden files, unit tests, and required CI checks — then posts a single
results table comment summarising what passed or failed.
PRs that fail key checks are flagged with a **`Not ready to Review`** label
until the issues are resolved.

It helps save time on the **first level of PR review** by automating the basic
checks a reviewer would otherwise perform manually, reducing first-pass review
time. Reviewers can filter out **`Not ready to Review`** PRs from their pending
review list and only start reviewing once that label has been removed — i.e.
once all policy checks have passed.

This document explains what each policy check means, why it exists, and how to fix a failure.

> **Note:** This is **NOT an AI Bot and does not use any LLMs**. It is a
> deterministic, rule-based checker driven entirely by `policy.yml`.

______________________________________________________________________

## 🙋 Wish to Override the Policy Process and get unblocked?

Contact CODEOWNERS or supporters channel - (DevOps - Support or Help)

## 🙋 For any policy related feedback?

please reach out to the **ROCm Policy Council**.

📧 **Drop a mail to:** `rocm-repo-policy@amd.com` (ROCm Policy Council DLL)

Include your PR link, the check(s) you want overridden, and a short
justification so the council can review your request.

## ✅ Skip the PR Bot entirely (`@skip-pr-bot`)

If you want to opt a PR **out of the bot completely**, add the tag
**`@skip-pr-bot`** anywhere in the PR description. When present:

- The bot runs **no policy checks** at all.
- Any existing **`Not ready to Review`** label is **removed**.
- The bot posts a short notice:
  *"Author chose to skip pr bot run hence removing label."*

This works both when the tag is present at PR creation **and** when it is added
later via a description edit. Removing the tag (and pushing/editing again)
re-enables the normal checks.

______________________________________________________________________

## 📄 PR Description

**What does it check?**
The PR body (description) must be at least **30 characters** long **and** reference a tracking item (JIRA ID or ISSUE ID).
An empty or one-line description makes it hard for reviewers to understand the context.

**Required tracking reference** — include **one** of the following. Type the line
**exactly as shown, without surrounding backticks**:

| Type             | Example                                                |
| ---------------- | ------------------------------------------------------ |
| JIRA ID          | JIRA ID : TESTAUTO-6039                                |
| JIRA ID          | JIRA ID - #330                                         |
| JIRA ID          | JIRA ID #330                                           |
| ISSUE ID         | ISSUE ID : TESTUTO-3334                                |
| ISSUE ID         | ISSUE ID - TESTAUTO-3433                               |
| ISSUE ID (link)  | ISSUE ID : https://github.com/abc/abc_repo/issues/1234 |
| Closing keyword  | Closes #10                                             |
| Closing keyword  | Fixes octo-org/octo-repo#100                           |
| Closing keyword  | Resolves: #123                                         |
| GitHub issue     | #123                                                   |
| GitHub issue URL | https://github.com/abc/abc_repo/issues/123             |

> **Note:** For `JIRA ID` / `ISSUE ID`, the separator is **optional** and may be `:` or `-` (`ISSUE ID #330`, `ISSUE ID : #330`, and `ISSUE ID - #330` all work). Each accepts a JIRA key (`PREFIX-<number>` — any project), a number (with or without `#`), or a link.
>
> **Closing keywords** (case-insensitive, optional colon): `close` / `closes` / `closed`, `fix` / `fixes` / `fixed`, `resolve` / `resolves` / `resolved` — followed by `#<number>` or `<org>/<repo>#<number>`.

**How to fix**
Edit the PR description and explain:

- *What* changed and *why*.
- A tracking reference from the table above (required) — e.g. a `JIRA ID :` / `ISSUE ID :` line, a `Closes #123`, or a plain `#123`.
- Testing steps if applicable.

______________________________________________________________________

## 📏 PR Size

**What does it check?**
Large PRs are hard to review thoroughly.

> **Note:** PR size limits are **not currently enforced** by `policy.yml`
> (there are no `max_files_changed` / `max_total_changes` /
> `max_single_file_changes` values configured). This section is guidance only
> and the bot does not fail a PR on size today.

**Recommended guidance**
Split your work into smaller, focused PRs. Each PR should ideally do one thing:

- One feature, one fix, or one refactor — not all three at once.
- Move large auto-generated or vendored file changes to a separate PR.

______________________________________________________________________

## ⛔ Forbidden Files

**What does it check?**
Certain file types must never be committed to the repository because they can expose secrets or introduce security risks.

> **⚠️ Warning-only (non-blocking):** The Forbidden Files check **never fails
> the workflow** and **never adds the `Not ready to Review` label**. If a
> forbidden file is present, the results table shows a **⚠️ Warning** row
> listing the offending file(s) — but the PR Bot check stays **green**. It is a
> reminder to remove the file, not a hard gate.

| Pattern                                                  | Reason                                                         |
| -------------------------------------------------------- | -------------------------------------------------------------- |
| `**/*.pem`                                               | TLS/SSL certificates — must not be stored in source control    |
| `**/*.key`                                               | Private keys — must not be stored in source control            |
| `**/.env`                                                | Environment files — often contain secrets/passwords            |
| `**/*.exe`                                               | Windows executables — binary blobs with no review value        |
| `**/*.crt`, `**/*.cer`, `**/*.der`                       | Certificates — must not be committed                           |
| `**/*.p12`, `**/*.pfx`                                   | Keystores / certificate bundles — contain secrets              |
| `**/*.csr`                                               | Certificate signing requests — should not be in source control |
| `**/id_rsa`, `**/id_dsa`, `**/id_ecdsa`, `**/id_ed25519` | SSH private keys                                               |
| `**/*.gpg`, `**/*.asc`                                   | GPG keys / signatures — must not be committed                  |

**How to fix**
Remove the file from your commit:

```bash
git rm --cached path/to/secret.pem
echo "*.pem" >> .gitignore
git commit --amend
```

If a secret was already committed, rotate it immediately and follow your organization's incident response process.

______________________________________________________________________

## 🧪 Unit Test

**What does it check?**
PRs that change real source code must include at least one accompanying unit test.

> **⚠️ Warning-only (non-blocking):** The Unit Test check **never fails the
> workflow** and **never adds the `Not ready to Review` label**. If a code
> change is missing a test, the results table shows a **⚠️ Warning** row
> explaining what is missing — but the PR Bot check stays **green**. It is a
> reminder, not a gate.

**Rules**

- **Doc / config-only PRs are exempt.** If your PR only touches files like
  `.md`, `.txt`, `.yml`, `.yaml`, `.ini`, the check **passes automatically** — no test required.
- **Code PRs require a test.** If your PR changes source files such as
  `.py`, `.cpp`, `.cc`, `.c`, `.h`, `.js`, `.ts`, `.go`, `.java`, it must also
  include changes to a test file (a new test, or edits to an existing one).

**What counts as a test file?**

- Basename matches one of: `test_*`, `testing_*`, `*_test.*`, `*_tests.*`, `*_gtest.*`, or `Test*`
  - ✅ `test_parser.py`, `testing_parser.py`, `parser_test.cpp`, `parser_tests.cpp`, `parser_gtest.cpp`, `TestUtils.cpp`
  - ❌ `test.py` (does NOT have the `test_` prefix)

| Pattern     | Example             |
| ----------- | ------------------- |
| `test_*`    | `test_parser.py`    |
| `testing_*` | `testing_parser.py` |
| `*_test.*`  | `parser_test.cpp`   |
| `*_tests.*` | `parser_tests.cpp`  |
| `*_gtest.*` | `parser_gtest.cpp`  |
| `Test*`     | `TestUtils.cpp`     |

**Path-based recognition**
Any file located under a `test/gtest/` directory is also treated as a unit
test, regardless of its filename — e.g.
`projects/miopen/test/gtest/unit_conv_solver_ConvWinoRageRxS.cpp`.

**How to fix**
Add a unit test for the code you changed, named `test_<something>`:

```bash
# example for Python
touch tests/test_my_feature.py
```

> Even though this is only a warning, adding the missing test clears the ⚠️
> from the table.

______________________________________________________________________

## 🔎 pre-commit

**What does it check?**
The bot requires the formatting workflows scheduled for the PR's target
branch. Each workflow publishes a distinct check name:
`pre-commit / runtimes`, `pre-commit` (RocJITsu), `pre-commit / cuid`, or
`pre-commit / rocprofiler-compute`. All four run on PRs targeting `develop`;
the runtimes workflow is restricted to that branch. The bot waits for every
scheduled check, including checks that have not appeared yet when polling starts.

These PR workflows have no trigger-level path filters. Each named job first
uses `.github/actions/scoped-formatting` to examine a complete local Git diff
between the PR's merge base and head. Its include/exclude expressions decide
whether to run the formatter. A job with no matching inputs succeeds after
that inspection, without installing formatters or checking out source and
submodules. The initial checkout fetches commit/tree history and the small
shared action; file contents are fetched when needed. Changes to the shared
action activate all four formatting scopes.

GitHub's server-side path filters and PR-files API can omit paths on large
PRs. Neither controls formatting scheduling or scope selection. Scope
inspection includes deleted paths and both sides of renames, uses NUL-delimited
filenames, and fails the job if Git cannot establish an unambiguous comparison.

The runtimes and rocjitsu workflows use the root `.pre-commit-config.yaml`.
The CUID and rocprofiler-compute workflows use their project configurations.
Their check names and branch filters are declared in
`tools/systems_pr_bot/policy.yml`; automated tests verify that the named jobs
are scheduled without path filters or job-level conditions. Workflow-local
include/exclude expressions control formatting work, not which checks the bot
requires. The bot's advisory file checks still use the PR-files API and show
an explicit warning when its file count differs from the PR's total; that
partial list never removes a required formatting check.

**How to fix**
Run the checks locally, let them auto-fix where possible, then commit the result:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files --show-diff-on-failure
git add -u
git commit -m "chore: apply pre-commit fixes"
```

For a project configuration, add `--config projects/<project>/.pre-commit-config.yaml`
to the `pre-commit run` command.

______________________________________________________________________

## 🔎 CodeQL

**What does it check?**
GitHub's [CodeQL](https://codeql.github.com/) static-analysis engine scans the code added in this PR for known security vulnerabilities.
CodeQL's own workflows report their results on the PR. The bot currently
requires formatting checks only; when a CodeQL check is added to
`checks.required_check_runs`, it uses that check's conclusion rather than
reading code-scanning alerts.

Common findings include:

| Alert                       | Meaning                                                   |
| --------------------------- | --------------------------------------------------------- |
| `py/command-line-injection` | User input passed unsanitised into a shell command        |
| `py/sql-injection`          | User input concatenated into a SQL query                  |
| `py/flask-debug`            | Flask app started with `debug=True` on a public interface |
| `js/xss`                    | User input rendered unescaped into HTML                   |

**How to fix**

1. Open **Security → Code scanning alerts** on the repository page.
1. Read the alert details and the suggested fix.
1. Apply the fix (validate/sanitise input, use parameterised queries, disable debug mode, etc.).
1. Push the fix — CodeQL will re-run and the alert will be resolved.

> **Tip:** GitHub Advanced Security AI comments directly on the offending line with a suggested code change.

______________________________________________________________________

## 🌿 Bump PRs (Automated Dependency Updates)

**What is a "Bump PR"?**

A **Bump PR** is an automated pull request that updates dependencies (e.g. from Dependabot or a bot like `assistant-librarian`). These PRs are routine, high-volume, and do not follow the standard PR conventions.

**Why did my Bump PR skip policy checks?**

When a PR is detected as a bump update from a configured bot account (e.g. `@assistant-librarian[bot]`), **all policy checks are auto-approved**. This includes:

- JIRA/ISSUE ID reference requirement in Description
- Unit test requirement
- And all other policies

This keeps automated bots from being blocked by human-oriented policy gates and prevents spam of "Not ready to Review" labels.

**How does the bot know it's a Bump PR?**

The PR author's login is checked against a configured list of bump bot accounts. Currently recognized:

- `assistant-librarian` (and `assistant-librarian[bot]`)
- `systems-assistant` (and `systems-assistant[bot]`)
- `dependabot` (and `dependabot[bot]`)

If a different bot opens dependency-bump PRs in your repo, request that the maintainers add it to `bump_bot_authors` in `policy.yml`.

______________________________________________________________________

## General Questions

**What is the "Not ready to Review" label?**

When the **JIRA/ISSUE ID reference** is missing from the PR description, the bot
adds a **`Not ready to Review`** label to the PR so it is clearly gated.
The label is removed automatically once that reference is added.
The **Unit Test** and **Forbidden Files** checks are (⚠️ warning-only). Other failures (
Draft PR, pre-commit, CodeQL) do **not** add the label.

**How are pre-commit and CodeQL shown?**

Required workflows appear in the table under their exact check names. The bot
selects the expected names by target branch from trusted base-branch policy
before it polls the GitHub API, then reads every page of current check runs. A missing or pending
required check keeps the bot waiting; a failure is reported immediately.
Checks outside the target branch's requirements do not satisfy or block them.
CodeQL is enforced by its own workflows; it is included in this table only if
its check name is declared in `checks.required_check_runs`. The bot does not
query code-scanning alerts.

**The bot timed out — what do I do?**

The bot times out after 15 minutes if an expected check is missing or still
running. Compare the required names in the PR's **base-commit**
`tools/systems_pr_bot/policy.yml` with the checks reported on the PR head. The
privileged bot executes that trusted base policy; policy changes in the PR
take effect only after merge. A renamed check cannot satisfy its old name,
even when it has passed. RocJITsu retains the `pre-commit` name so the base
policy can still find it while the other workflows gain distinct names.

For a missing check, inspect its name and the workflow's branch filters; required
formatting checks are scheduled independently of changed paths. For a running
check, inspect its job for queued runners or stalled steps. Once the cause is
resolved, push a commit to trigger a fresh policy run:

```bash
git commit --allow-empty -m "ci: retrigger policy check"
git push
```

**How do I re-run the bot after fixing issues?**

Push any commit (including `--allow-empty`) to the PR branch.
The `synchronize` event triggers a fresh policy check automatically.

______________________________________________________________________
