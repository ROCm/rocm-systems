---
name: amdsmi-commit-and-pr-conventions
description: "Use when writing or restructuring git commits or opening/updating a pull request for amd-smi — composing commit titles, commit message bodies, PR titles, or PR descriptions. Defines the Conventional Commits `type(amdsmi):` house title convention, the rocm-systems PR template sections, the Systems PR Bot's JIRA/ISSUE gate and warning-only checks, brevity caps, and the rule that tickets appear only in the PR Issue Tracking section, never in code comments or commit bodies."
---

# Commit & PR Conventions — amd-smi

The single source of truth for what a commit message and a pull request look
like in amd-smi. `amdsmi-restructure-commits` triggers this skill for message wording;
`amdsmi-changelog-automation` owns `CHANGELOG.md` entries.

**Core principle:** Write for an expert who already knows the codebase. Bullets,
not prose. State *what changed and why* — never narrate *how the code works*.

## Title Convention (commits AND PRs)

`type(amdsmi): imperative summary`

This is a house convention; no bot or workflow lints PR titles. PRs are squash
merged with the PR title as the `develop` commit subject, so write the PR title
in this form and use it for commit subjects too.

| Rule | Value |
|------|-------|
| Type | One of `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert` |
| Scope | `(amdsmi)` by default, not `(amd-smi)`; a narrower lowercase scope is fine (`fix(cli): …`, `feat(fabric): …`) |
| Mood | Imperative, verb-first ("add", "fix", "remove" — not "added"/"fixes"/"fixing", and not a noun phrase like "More cleanup" or "status-string tests") |
| Length | Keep PR titles and commit subjects ≤ 72 (git norm; the PR title becomes the squash subject) |
| Punctuation | No trailing period |
| Breaking | Append `!` before the colon (`feat(amdsmi)!: …`) for an ABI/behavior break |

Good: `fix(amdsmi): reject nullptr mode pointer in compute-partition getter`
Bad: `[AMD-SMI] Fixed the partition bug.` (legacy tag, past tense, period)

**No `[AMD-SMI]` / `[ROCM-NNNNN]` tag** — legacy form, not the house convention.
Ticket references live only in the PR's `Issue Tracking` section (below).

Self-check a subject (structure + length) before committing:

```bash
SUBJECT="fix(amdsmi): reject nullptr mode pointer in compute-partition getter"
printf '%s' "$SUBJECT" | grep -qP '^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([a-z][a-z0-9_/-]*\))?!?: .+[^.]$' \
  && [ "${#SUBJECT}" -le 72 ] && echo OK || echo 'FAIL: not type(scope): imperative, or ends in a period, or >72 chars'
```

The regex enforces the type list, a lowercase scope, no trailing period, and a
body after the colon; the length test enforces the ≤ 72 cap. It cannot judge
mood — a subject can pass the regex and still be past-tense or a noun phrase, so
read the verb by eye (see the Mood row).

## Commit Message Body

```
type(amdsmi): imperative summary

- Verb-first bullet: what changed and why it was needed
- One bullet per distinct logical change
- Group by file/layer when a commit spans the cascade

Signed-off-by: Full Name <email@amd.com>
```

Rules:

- Blank line after the subject; wrap body at 72 columns.
- Each bullet starts with a verb (Add, Fix, Remove, Refactor, Implement).
- Include the "why" only when it is not self-evident from the change.
- **No JIRA ticket references in the commit body.** Reference the code/behavior,
  not `ROCM-NNNNN`. Tickets live in the PR's `Issue Tracking` section only.
- **No internal labels** ("regression guard", sprint names) in the message.
- `Signed-off-by` is required (`git commit -s`); preserve original authors and
  add `Co-authored-by:` trailers when squashing multiple authors.
- If a single bullet needs a full paragraph to explain, the change is probably
  too large or the root cause is unclear — stop and reconsider the split.

## Pull Request Body

Follow the rocm-systems template
(`.github/pull_request_template.md`) — these exact sections, in order:

| Section | Content | Keep it |
|---------|---------|---------|
| `## Motivation` | Why this PR exists, the problem it solves | 1-3 sentences |
| `## Technical Details` | What changed, grouped **by file/layer** with bullets | Bulleted, no prose walls |
| `## Issue Tracking` | A plain `JIRA ID: ROCM-NNNNN` line (or `Closes #NNNN` for a GitHub issue) — the **only** place a ticket appears. No links. The line must start at column 0: not bulleted, not in backticks, not inside an HTML comment, or the bot does not see it | One line |
| `## Test Plan` | How it was verified (commands, hardware) | Bulleted |
| `## Test Result` | Outcome of those tests | Brief |
| `## Submission Checklist` | The template's single checkbox, ticked; don't replace it with custom items | Leave intact |

House style for `## Technical Details` (matches the project's best PRs):

```markdown
## Technical Details

**Header / docs** (`include/amd_smi/amdsmi.h`):
- Corrected the `@retval` list — setter returns `AMDSMI_STATUS_NO_PERM`, not `_PERMISSION`

**Public API** (`src/amd_smi/amd_smi.cc`):
- Getter now rejects a `nullptr` mode pointer with `AMDSMI_STATUS_INVAL`
```

Rules:

- Bullets, not paragraphs. No ticket numbers outside `Issue Tracking`.
- The body becomes the `develop` commit message on squash merge: delete leftover
  template comments and placeholders, and keep every claim accurate (including
  history claims such as "never in any CMake list").
- Assume the reviewer is an amd-smi developer — skip background explanations.
- Don't restate the diff line-by-line; summarize each logical change once.
- Mention the cascade layers touched (header → impl → wrapper → interface → CLI
  → docs) so reviewers can check coverage.

## PR Bot Policy Gate

Every PR to `develop` is checked by the *Systems PR Bot* (`tools/systems_pr_bot`).
Only a missing JIRA/ISSUE reference adds the **"Not ready to Review"** label;
every other check is a warning or advisory row in the bot's results comment.

| Check | Effect | Pass condition |
|-------|--------|----------------|
| **JIRA/ISSUE reference** | ❌ adds the label | Description has a `JIRA ID: <KEY>` / `ISSUE ID: <KEY>` line, a closing keyword (`Closes #N`), or a bare `#N`. A bare `Resolves ROCM-NNNNN` (no `#`) does **not** pass |
| Unit Test | ⚠️ warning only | Any changed source file (`.c/.cc/.cpp/.h/.hpp/.py/.go/.rs` and others; full list in `tools/systems_pr_bot/policy.yml`) needs a `test_*` / `*_test.*` file in the **same** PR. Doc/config-only PRs auto-pass. Fires on edits to existing files; pure deletions of files are skipped |
| Description length | advisory | ≥ 30 chars |
| Forbidden files | ⚠️ warning only | No `.pem` / `.key` / `.env` / `.crt` / private keys |

The unit-test warning does not mean a test is required. Add one when the change
has behavior to assert; for a pure removal of code that was never compiled, say
so in the Test Plan instead. Separately, `pre-commit` is a required CI status
check (must be green to merge) but is not part of the bot's label logic above.

## Brevity Caps

| Symptom | Cap |
|---------|-----|
| Comments longer than the code they explain | Explain only a non-obvious root cause; one or two lines max. A paragraph means stop and ask. |
| PR/commit body reads like a tutorial | Cut anything an amd-smi dev already knows |
| Comma-separated list of changes | Break into one bullet per item |
| Ticket number in code or commit | Move it to the PR `Issue Tracking` section |

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Legacy `[AMD-SMI]` / `[ROCM-NNNNN]` tag in the title | Conventional Commits `type(amdsmi): …` |
| `(amd-smi)` scope | `(amdsmi)` |
| Past-tense subject ("Fixed…", "Added…") | Imperative ("fix", "add") |
| Noun-phrase subject ("More cleanup", "status-string tests") | Start with a verb ("clean up…", "add…") |
| `ROCM-NNNNN` in commit body or code comment | PR `Issue Tracking` section only |
| Source change with no `test_*` / `*_test.*` file | Add a test in the same PR if there is behavior to assert (bot only warns) |
| `JIRA ID` only in an HTML comment or a bulleted line | Plain line at column 0 under `## Issue Tracking` |
| `Resolves ROCM-NNNNN` as the only ticket ref | Use `JIRA ID: ROCM-NNNNN` or `Closes #N` (bot needs the prefix or `#`) |
| Comma-list of changes in body or changelog | One bullet per change |
| Paragraph-length code comments | One-line root-cause note, or ask |
| PR body free-form, skipping template sections | Use all six template sections (`Issue Tracking`, not `JIRA ID`) |
| Skipping `-s` / dropping a co-author | `git commit -s`, preserve trailers |
