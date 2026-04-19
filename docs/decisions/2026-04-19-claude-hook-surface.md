# Decision: Claude Code pre-tool-call hook surface for perfxpert gate

Date: 2026-04-19
Status: **PENDING — live doc-fetch required before Task 4.6 (Claude portion) can start**
Owner: `users/aelwazir/perfxpert-code-multi-backend`
Blocks: Plan Task 4.6 Claude-portion (does NOT block opencode or Gemini
portions; does NOT block the rest of PR 1)
Related: `docs/superpowers/plans/2026-04-19-perfxpert-code-multi-backend-plan.md`
§Task 0.5, §Task 4.6; brainstorm §13 F1

## Context

Cycle-2 design review (B-N1) flagged that the plan's Task 4.6 listed the
Claude Code pre-tool-call hook surface as "research required / verify
via doc fetch" inline within the implementation task. That is a
**prerequisite decision**, not an implementation detail — Task 4.6
cannot begin the Claude-portion until the surface is picked, and if
Claude exposes no suitable surface we need to downgrade the scope
explicitly.

Cycle-2 resolution: move the decision UP to a dedicated Task 0.5 (in
the plan) with this file as the decision record. Task 4.6's
Claude-portion is gated on this decision landing.

## Candidate surfaces (to be verified live)

1. **Native PreToolUse hook**. A `.claude/hooks/pre-tool-use.<ext>`
   file (shell/JS/Python — TBD by doc) that receives the pending
   tool-call and may return a synthetic rejection with a retry
   message. Preferred if available because it matches opencode's
   `tool.execute.before` pattern most closely.
2. **`allowedTools` settings list**. A `.claude/settings.json` key
   that restricts which tools the model may call for the current
   session. We would write
   `allowedTools: ["mcp__perfxpert__*"]` at install time and clear it
   after `intent_classify` is observed. Lift-mechanism TBD
   (file-write on intent-classify? settings refresh cadence?).
3. **Neither** — Claude Code exposes no suitable surface. In that
   case Task 4.6 Claude-portion **degrades to prompt-layer
   enforcement only** (the rejection-language stanza from Task 4
   `_prompt_adapter.render_prompt(reject_language=True)`), ships in
   PR 1 with this explicit acceptance-of-partial-mitigation, and
   the gate-probe in `verify_mcp_live` is documented as
   "known-limited on Claude" rather than asserting mechanical
   enforcement.

## Decision

**PENDING.** Requires a live web-fetch of the Claude Code
documentation under <https://code.claude.com/docs> (in particular
plugin/hook pages, settings reference, and any `allowedTools`
documentation) to confirm which of the three options is available
as of late April 2026.

Outputs of the doc-fetch task:

- Chosen surface (1, 2, or 3).
- If (1): hook file path + invocation contract + return-shape for
  rejection.
- If (2): settings key name + update mechanism + lift trigger.
- If (3): the acceptance-of-partial-mitigation callout to add to
  Task 4.6 Claude bullet + PR-1 acceptance criterion 9 footnote +
  R-new-4 scope narrowing.

## Acceptance

This decision record is resolved when:

- Doc-fetch step in Plan Task 0.5 has been completed and cited here
  with URLs and retrieval date.
- Decision above is updated from **PENDING** to the chosen option.
- A one-paragraph change set for Task 4.6 (Claude portion) is noted
  below the decision so the implementer can mechanically apply it.

Until resolved, Task 4.6 opencode-portion and Gemini-portion can
proceed in parallel. Task 4.6 Claude-portion MUST wait for this
file to flip to a non-PENDING status.

## Session state location (forward-reference to plan I-N3)

Once the surface is chosen, record the session-state location for
the gate in the Plan Task 4.6 sub-table (I-N3). Candidates:

- If (1): the hook file persists across invocations; state lives
  in `.claude/.perfxpert-gate-state.json` (or similar), keyed by
  Claude session id when exposed by the hook runtime; invalidated
  at session end.
- If (2): `.claude/settings.json` itself is the state (restriction
  present = gate engaged, restriction cleared = gate lifted).
  Invalidated by clearing the list; new session starts by
  re-writing the restriction.

A NEW Claude Code session always starts with the gate engaged —
even in the same cwd.
