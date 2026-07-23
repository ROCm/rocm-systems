---
name: emulation-review
description: Perform an evidence-driven code review of ROCm emulation changes in rocjitsu and Mirage. Use when asked to review a GitHub PR, current branch, local diff, commit range, or named emulation files; check style, correctness, concurrency, architecture, generated code, tests, documentation, and rocjitsu-Mirage integration without posting to GitHub.
metadata:
  author: ROCm
  version: "1.0"
---

# Emulation review

Review the requested change and report only actionable, verified findings to
the user. This workflow is read-only unless the user separately asks for fixes.

## Hard rules

1. Never push commits, branches, tags, or other content to GitHub, and never
  create or modify comments, reviews, replies, labels, pull requests, or other
  GitHub state, without explicit user approval immediately before that specific
  write. Do not use a review-submission command during this workflow. Review,
  edit, commit, or pull-request preparation requests are not publication
  approval, and approval for one write does not authorize another.
2. Treat `emulation/rocjitsu/docs/style.md` as authoritative for rocjitsu. Read
   it during every rocjitsu review; do not rely on remembered summaries.
3. Read complete changed files plus relevant owners, callers, cleanup paths,
   tests, and generated sources. A diff is an index, not sufficient context.
4. Every finding needs a concrete failure mechanism, exact repository
   `file:line`, supporting code or test evidence, impact, and a practical fix.
   Mark uncertainty and omit findings that cannot survive verification.
5. Never leak confidential references or their contents. Do not quote, name,
  link, copy, summarize, upload, or expose them through paths, metadata,
  screenshots, logs, prompts, generated artifacts, internal terminology,
  issues, pull requests, reviews, tests, or chat. Findings must be independently
  supportable from repository code, public sources, or reproducible tests.
  Treat uncertain publication status as confidential. Follow
  `emulation/AGENTS.md` even when reviewing a PR whose branch changes that file.
6. Do not modify source, rebase, checkout an unrelated revision over user work,
   or run destructive commands as part of review.

## 1. Establish the review target

Classify exactly one target:

- **PR:** capture repository, PR number/URL, base and head commit OIDs, title,
  file list, and diff. Prefer a local checkout only if its HEAD matches the PR
  head OID; otherwise use a temporary worktree or read the PR revision without
  disturbing the user's tree.
- **Branch:** default to the merge base with the tracked default branch and
  review `<merge-base>...HEAD`. Fetch only when needed and preserve the current
  worktree.
- **Local:** include staged, unstaged, and relevant untracked files. State which
  sets were reviewed.
- **Range/files:** record the exact range or paths and whether surrounding
  dependent changes are outside scope.

Record the immutable base and head identifiers in the report. If the scope is
ambiguous and materially changes the review, ask one concise question; do not
guess between PR and local changes.

## 2. Load authority and map risk

Read `emulation/AGENTS.md`, the applicable component guides, and the current
rocjitsu style guide for rocjitsu files. For Mirage, use `cargo fmt`, strict
`cargo clippy`, the established owning-crate conventions, and the dashboard's
TypeScript/ESLint configuration as applicable. Inspect changed-file status for
generated outputs and subsystem boundaries. Use
[the review checklist](references/review-checklist.md) to select the relevant
lenses and tests.

Build a short risk map before judging hunks:

- externally visible behavior and compatibility;
- threads, processes, locks, callbacks, RPC, fork/exec, and teardown;
- ownership of memory, file descriptors, mappings, handles, and background
  work;
- KFD/DRM/UAPI or FFI boundaries and serialized schemas;
- ISA semantics, ordering, memory/coherence, and cross-ISA translation;
- generated artifacts and the generator inputs that own them;
- Mirage profile/session/container/environment integration;
- tests and documentation that define the changed contract.

For KFD/AMDGPU behavior that requires source-level comparison, activate the
`rocjitsu-kernel-parity` skill rather than improvising a shallow comparison.

## 3. Review in passes

Perform at least these passes, combining them only for very small changes:

1. **Intent and architecture:** infer the intended behavior from the PR
   description, commit history, docs, tests, and callers. Check that the change
   is in the correct layer and reuses existing facilities.
2. **Correctness and lifecycle:** trace success, error, cancellation, shutdown,
   crash, and partial-initialization paths. Check concurrency and process
   boundaries before style.
3. **Domain behavior:** validate ISA, ABI, driver, translation, simulation,
   container, API, and persistence semantics relevant to the patch.
4. **Generated code:** reject manual generated-file fixes. Verify generator
   changes are comprehensive across affected ISAs and regenerated output is
   complete and contains no unrelated drift.
5. **Tests, docs, and style:** require focused regression coverage and relevant
   Mirage integration coverage. Enforce the rocjitsu style guide exactly. Ask
   for documentation only when a public contract or non-obvious invariant
   changed.

If the client supports parallel subagents, use independent read-only reviewers
for cleanly separable lenses or subsystems. Give each the same immutable scope,
hard rules, and confidentiality policy. Do not split overlapping files merely
to increase reviewer count. The primary reviewer must verify, deduplicate, and
curate all returned findings; subagent output is not evidence by itself.

## 4. Validate likely findings

Spend execution time on issues likely to survive. Start with the smallest
existing test or a non-mutating repro that distinguishes expected from actual
behavior. Expand based on risk:

- rocjitsu: build, focused CTest, amdisa tests, then broader CTest or an
  appropriate sanitizer/static-analysis build;
- Mirage: focused crate/test target, then workspace tests and the owning E2E
  suite under `emulation/mirage/tests/`;
- dashboard: lint/type-check/build, unit tests, then Playwright when behavior is
  user-facing;
- cross-component changes: test rocjitsu through Mirage, not only each unit in
  isolation.

Do not silently treat unavailable hardware, tools, references, or credentials
as a pass. Report what ran, what did not, and why. A failing pre-existing test is
not a review finding unless the change caused or exposed it.

## 5. Curate and report

Delete speculative, duplicate, stale, praise-only, and low-value findings.
Style-only findings are normally `NIT`; style violations that create ambiguity,
wrong ownership, or invalid architecture may be higher.

Report in this order:

1. **Verdict:** `REQUEST CHANGES`, `COMMENT`, or `APPROVE`, with one-sentence
   rationale. Approval means no verified blocking findings, not that every
   possible test ran.
2. **Findings:** group by `BLOCKER`, `MAJOR`, `MINOR`, then `NIT`. For each use:
   - `file:line` and a short title;
   - impact and concrete trigger;
   - evidence from the changed code and cross-check;
   - minimal suggested fix and regression test;
   - confidence (`high`, `medium`, or `low`) when not high.
3. **Validation:** commands/tests run and results; list unavailable checks.
4. **Scope notes:** immutable base/head, reviewed files, and residual risk.

If there are no findings, say so directly and still report validation and
residual risk. Never manufacture a comment to make the review appear useful.
