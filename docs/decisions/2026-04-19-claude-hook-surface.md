# Decision: Claude Code pre-tool-call hook surface for perfxpert gate

Date: 2026-04-19
Status: **DECIDED (native-pretooluse)**
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

## Candidate surfaces (verified live 2026-04-18)

1. **Native `PreToolUse` hook** — a command- or HTTP-invocable hook
   configured inside `.claude/settings.json` under the `hooks` key.
   Receives a JSON payload on stdin with `hook_event_name`,
   `tool_name`, `tool_input`, `session_id`, `transcript_path`, `cwd`.
   Writes a JSON response on stdout (with exit code 0 for "parse
   stdout") containing `hookSpecificOutput.permissionDecision`:
   `"allow" | "deny" | "ask" | "defer"` plus a
   `permissionDecisionReason` string shown to Claude as context when
   denied. **Exit code 2 is also a blocking error** (stderr surfaced
   to Claude). Non-blocking errors on any other non-zero exit.
2. **`permissions.allow` / `permissions.deny` settings** — static
   rule lists inside the `permissions` key of `.claude/settings.json`
   (e.g. `"deny": ["Bash(rm -rf *)"]`). Simple but static: the rule
   set is loaded once per session; there is no documented
   per-tool-call runtime toggle from a sidecar signal.
3. **Neither** — would force prompt-layer-only enforcement.

## Decision

**native-pretooluse.** Evidence below.

### Evidence (doc-fetch 2026-04-18)

Primary source: <https://code.claude.com/docs/en/hooks> (followed from
<https://docs.claude.com/en/docs/claude-code/hooks> via 301 redirect,
retrieved 2026-04-18).

Key confirmations:

- The `PreToolUse` event "runs after Claude creates tool parameters
  and before processing the tool call" — the exact blocking point
  the gate needs.
- Configuration lives in `.claude/settings.json` (shared, committed)
  or `.claude/settings.local.json` (gitignored) at project scope;
  also `~/.claude/settings.json` at user scope. Project scope matches
  our plan default.
- Contract: stdin JSON with `hook_event_name`, `tool_name`,
  `tool_input`, `session_id`, `transcript_path`. Stdout JSON with
  `hookSpecificOutput.permissionDecision` (`allow|deny|ask|defer`)
  and `permissionDecisionReason`. Exit 0 = parse stdout, exit 2 =
  blocking error (stderr → Claude).
- When the hook returns `deny`, **the tool call is prevented
  entirely AND the `permissionDecisionReason` is shown to Claude as
  context**. This matches opencode's `{ block: true, retryWith: ...}`
  semantic: Claude sees the rejection reason and "must decide
  independently how to proceed" — typically retrying with the
  recommended alternative (calling `mcp__perfxpert__intent_classify`
  first).

Secondary source: <https://code.claude.com/docs/en/settings> (followed
from <https://docs.claude.com/en/docs/claude-code/settings>).

- `permissions.allow` and `permissions.deny` exist and accept rules
  like `"Bash(git diff *)"` or `"WebFetch"`. But these are static
  per-session rule lists, not event-driven.
- The same `settings.json` hosts the `hooks` object, so both
  surfaces live in one file.

### Why native PreToolUse over `permissions.deny`

1. **Event-based lift (B-N3).** The gate must lift once
   `perfxpert_intent_classify` has returned in the current session.
   The native hook runs per-tool-call and can read a sidecar state
   file (written when `intent_classify` returns) to decide. A static
   `permissions.deny: ["Bash", "Read", ...]` list would require
   mid-session mutation of `settings.json` + a settings refresh —
   not a documented mechanism.
2. **Retry signal (B-N2 parity).** `permissionDecisionReason` is
   piped to Claude on deny. That is the exact counterpart to
   opencode's `retryWith` message (fork-only patch 0020), so both
   backends share the same UX: "call `intent_classify` first."
3. **Partial-state safety (I-N1).** Hook install is a single
   `settings.json` patch under `hooks.PreToolUse`. If it fails (e.g.
   file write race or invalid JSON merge), the adapter can detect
   pre-MCP-registration and raise `GateHookUnsupported` cleanly. A
   `permissions.deny` fallback would require a different code path
   and a different lift mechanism.

### One-paragraph change set for Task 4.6 (Claude portion)

Implement `ClaudeGateHook` in `perfxpert/cli/_gate_hooks/claude.py`
that, on `install()`, atomically patches `<cwd>/.claude/settings.json`
to add a `hooks.PreToolUse` entry invoking a shipped shell script
(`<cwd>/.claude/hooks/perfxpert-gate.sh`). The script reads stdin
JSON, checks for `<cwd>/.claude/.perfxpert-gate-state.<session-id>.json`
(created by a separate `PostToolUse` hook on `mcp__perfxpert__intent_classify`),
and emits the appropriate `permissionDecision` + reason on stdout.
Install **MUST** run BEFORE MCP registration (I-N1); if the
settings.json patch fails (pre-existing conflicting hook, invalid
JSON, etc.), raise `GateHookUnsupported` so no partial state is left
behind. Session state location per I-N3 sub-table:
`<cwd>/.claude/.perfxpert-gate-state.<session-id>.json`; a NEW
session (different `session_id`) always starts with the gate
engaged even in the same cwd, because the sidecar file is keyed on
`session_id`.

## Acceptance

This decision record is resolved when:

- [x] Doc-fetch step in Plan Task 0.5 has been completed and cited
      here with URLs and retrieval date.
- [x] Decision above is updated from **PENDING** to the chosen option.
- [x] A one-paragraph change set for Task 4.6 (Claude portion) is
      noted below the decision so the implementer can mechanically
      apply it.

Task 4.6 Claude-portion is now **UNBLOCKED**.

## Session state location (I-N3 table entry)

| Backend | Session state location | Invalidation |
|---|---|---|
| Claude Code | `<cwd>/.claude/.perfxpert-gate-state.<session_id>.json` (written by PostToolUse hook on `mcp__perfxpert__intent_classify` return; read by PreToolUse hook for every non-perfxpert tool call) | Session end (new `session_id` = fresh state file required) |

A NEW Claude Code session always starts with the gate engaged —
even in the same cwd. The sidecar file keyed on `session_id`
enforces this naturally; without a matching file, the hook defaults
to `deny` for any non-`mcp__perfxpert__*` tool.
