---
name: amdsmi-commit-and-pr-conventions
description: "Use when writing or restructuring git commits or opening/updating a pull request for amd-smi — composing commit titles, commit message bodies, PR titles, or PR descriptions. Defines the Conventional Commits `type(amdsmi):` title convention enforced by the Systems PR Bot, the rocm-systems PR template sections, the unit-test and JIRA/ISSUE-reference gates, brevity caps, and the rule that JIRA tickets appear only in the PR JIRA ID section, never in code comments or commit bodies."
---

# Commit & PR Conventions — amd-smi

The single source of truth for what a commit message and a pull request look
like in amd-smi. `amdsmi-restructure-commits` triggers this skill for message wording;
`amdsmi-changelog-automation` owns `CHANGELOG.md` entries.

**Core principle:** Write for an expert who already knows the codebase. Bullets,
not prose. State *what changed and why* — never narrate *how the code works*.

## Title Convention (commits AND PRs)

`type(amdsmi): imperative summary`

The PR bot (`tools/systems_pr_bot`) enforces Conventional Commits on PR titles.
Use the same form for commit subjects so a single-commit / squash PR auto-titles
correctly.

| Rule | Value |
|------|-------|
| Type | One of `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert` |
| Scope | `(amdsmi)` by default; a narrower lowercase scope is fine (`fix(cli): …`, `feat(fabric): …`) |
| Mood | Imperative ("add", "fix", "remove" — not "added"/"fixes"/"fixing") |
| Length | PR title 10–80 chars (bot limit); keep commit subjects ≤ 72 (git norm, safely under the cap) |
| Punctuation | No trailing period |
| Breaking | Append `!` before the colon (`feat(amdsmi)!: …`) for an ABI/behavior break |

Good: `fix(amdsmi): reject nullptr mode pointer in compute-partition getter`
Bad: `[AMD-SMI] Fixed the partition bug.` (legacy tag, past tense, period)

**No `[AMD-SMI]` / `[ROCM-NNNNN]` tag** — the `[…]` form fails the bot's title
regex. Ticket references live only in the PR's `JIRA ID` section (below).

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
  not `ROCM-NNNNN`. Tickets live in the PR's `JIRA ID` section only.
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
| `## JIRA ID` | `JIRA ID: ROCM-NNNNN` (or `Closes #NNNN` for a GitHub issue) — the **only** place a ticket appears. No links | One line |
| `## Test Plan` | How it was verified (commands, hardware) | Bulleted |
| `## Test Result` | Outcome of those tests | Brief |
| `## Submission Checklist` | The template checkboxes | Leave intact |

House style for `## Technical Details` (matches the project's best PRs):

```markdown
## Technical Details

**Header / docs** (`include/amd_smi/amdsmi.h`):
- Corrected the `@retval` list — setter returns `AMDSMI_STATUS_NO_PERM`, not `_PERMISSION`

**Public API** (`src/amd_smi/amd_smi.cc`):
- Getter now rejects a `nullptr` mode pointer with `AMDSMI_STATUS_INVAL`
```

Rules:

- Bullets, not paragraphs. No ticket numbers outside `JIRA ID`.
- Assume the reviewer is an amd-smi developer — skip background explanations.
- Don't restate the diff line-by-line; summarize each logical change once.
- Mention the cascade layers touched (header → impl → wrapper → interface → CLI
  → docs) so reviewers can check coverage.

## PR Bot Policy Gate

Every PR to `develop` is checked by the *Systems PR Bot* (`tools/systems_pr_bot`).
Two failures add the **"Not ready to Review"** label and block the PR; the rest
are advisory rows in the bot's results comment.

| Check | Blocking? | Pass condition |
|-------|-----------|----------------|
| **Unit Test** | ❌ yes | Any changed source file (the amd-smi-relevant set is `.c/.cc/.cpp/.h/.hpp/.py/.go/.rs`; full list in `tools/systems_pr_bot/policy.yml`) needs a `test_*` / `*_test.*` file in the **same** PR. Doc/config-only PRs auto-pass |
| **JIRA/ISSUE reference** | ❌ yes | Description has a `JIRA ID: <KEY>` / `ISSUE ID: <KEY>` line, a closing keyword (`Closes #N`), or a bare `#N`. A bare `Resolves ROCM-NNNNN` (no `#`) does **not** pass |
| Title (Conventional Commits) | advisory | `type(scope): …`, 10–80 chars |
| Description length | advisory | ≥ 30 chars |
| Forbidden files | advisory | No `.pem` / `.key` / `.env` / `.crt` / private keys |

A source change with no accompanying test is the most common block — pair every
code change with a test, or split a docs-only PR out, before opening. Separately,
`pre-commit` is a required CI status check (must be green to merge) but is not part
of the bot's label logic above.

## Brevity Caps

| Symptom | Cap |
|---------|-----|
| Comments longer than the code they explain | Explain only a non-obvious root cause; one or two lines max. A paragraph means stop and ask. |
| PR/commit body reads like a tutorial | Cut anything an amd-smi dev already knows |
| Comma-separated list of changes | Break into one bullet per item |
| Ticket number in code or commit | Move it to the PR `JIRA ID` section |

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Legacy `[AMD-SMI]` / `[ROCM-NNNNN]` tag in the title | Conventional Commits `type(amdsmi): …` (the `[…]` form fails the bot) |
| Past-tense subject ("Fixed…", "Added…") | Imperative ("fix", "add") |
| `ROCM-NNNNN` in commit body or code comment | PR `JIRA ID` section only |
| Source change with no `test_*` / `*_test.*` file | Add a test in the same PR (bot blocks otherwise) |
| `Resolves ROCM-NNNNN` as the only ticket ref | Use `JIRA ID: ROCM-NNNNN` or `Closes #N` (bot needs the prefix or `#`) |
| Comma-list of changes in body or changelog | One bullet per change |
| Paragraph-length code comments | One-line root-cause note, or ask |
| PR body free-form, skipping template sections | Use all six template sections |
| Skipping `-s` / dropping a co-author | `git commit -s`, preserve trailers |
