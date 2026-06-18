---
name: amdsmi-commit-and-pr-conventions
description: "Use when writing or restructuring git commits or opening/updating a pull request for amd-smi — composing commit titles, commit message bodies, PR titles, or PR descriptions. Defines the [AMD-SMI] title convention, the rocm-systems PR template sections, brevity caps, and the rule that JIRA tickets appear only in the PR JIRA ID section, never in code comments or commit bodies."
---

# Commit & PR Conventions — amd-smi

The single source of truth for what a commit message and a pull request look
like in amd-smi. `amdsmi-restructure-commits` triggers this skill for message wording;
`amdsmi-changelog-automation` owns `CHANGELOG.md` entries.

**Core principle:** Write for an expert who already knows the codebase. Bullets,
not prose. State *what changed and why* — never narrate *how the code works*.

## Title Convention (commits AND PRs)

`[TAG] Imperative summary`

| Rule | Value |
|------|-------|
| Tag | `[AMD-SMI]` by default. Use `[ROCM-NNNNN]` / `[SWDEV-NNNNNN]` / `[AILITOOLS-NNN]` **only** when the commit or PR is 1:1 with that ticket |
| Mood | Imperative ("Add", "Fix", "Remove" — not "Added"/"Fixes"/"Fixing") |
| Length | ≤ 72 characters including the tag |
| Punctuation | No trailing period |
| Casing | Exactly `[AMD-SMI]` — uppercase, hyphen, square brackets |

Good: `[AMD-SMI] Reject nullptr mode pointer in compute-partition getter`
Bad: `[amd-smi] fixed the partition bug.` (lowercase tag, past tense, period)

## Commit Message Body

```
[AMD-SMI] Imperative summary

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
| `## JIRA ID` | `Resolves ROCM-NNNNN` — the **only** place a ticket appears. No links | One line |
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
| Lowercase or reformatted tag (`[amd-smi]`, `(AMD-SMI)`) | Exactly `[AMD-SMI]` |
| Past-tense subject ("Fixed…", "Added…") | Imperative ("Fix", "Add") |
| `ROCM-NNNNN` in commit body or code comment | PR `JIRA ID` section only |
| Comma-list of changes in body or changelog | One bullet per change |
| Paragraph-length code comments | One-line root-cause note, or ask |
| PR body free-form, skipping template sections | Use all six template sections |
| Skipping `-s` / dropping a co-author | `git commit -s`, preserve trailers |
