# Decision: OpenAI Codex CLI pre-tool-call hook surface for perfxpert gate

Date: 2026-04-19
Status: **DECIDED (prompt-layer-only)**
Owner: `users/aelwazir/perfxpert-phase8-opencode-fork`
Blocks: Plan Task 4.6 Codex-portion, Plan Task 10 Codex gate install.
Related: `docs/superpowers/plans/2026-04-19-perfxpert-code-multi-backend-plan.md`
§Task 4.6 Codex portion, §Task 10; brainstorm §13 F1

## Context

Mirrors `2026-04-19-claude-hook-surface.md` in intent: we need to
pick a pre-tool-call surface for the Codex adapter's server-side
gate (Task 4.6 Codex-portion). Plan originally assumed the Claude
decision ("native-pretooluse") would carry over to Codex with minor
translation. Research during PR 2 implementation invalidates that
assumption.

## Candidate surfaces (verified 2026-04-18 via doc-fetch)

1. **Native `PreToolUse` hook** — Codex exposes a hook surface with
   `SessionStart`, `PreToolUse`, `PostToolUse`, `UserPromptSubmit`,
   `Stop` events (same vocabulary as Claude Code's hook API). Enabled
   via `[features] codex_hooks = true` in `~/.codex/config.toml`;
   hook definitions live in `~/.codex/hooks.json` or
   `<repo>/.codex/hooks.json`. Hook scripts receive JSON on stdin
   (`session_id`, `tool_name`, `tool_input.command`, `turn_id`) and
   can return `{"hookSpecificOutput": {"permissionDecision": "deny",
   "permissionDecisionReason": "..."}}` OR exit code 2 + stderr to
   block.

   **CRITICAL LIMITATION.** Per Codex's own hooks docs (retrieved
   2026-04-18): *"Currently `PreToolUse` only supports Bash tool
   interception. ... this doesn't intercept MCP, Write, WebSearch,
   or other non-shell tool calls."* The `matcher` field currently
   only matches `Bash`; there are no examples of matching MCP or
   Write tools because the feature is not implemented yet
   ("Work in progress" per the same doc).
2. **`rules.prefix_rules` / `approval_policy` settings** — static
   command-rule lists (e.g. block `rm -rf`). Same rules-only surface
   Codex has had since 0.5.x. Static per-session, no runtime toggle
   for event-based lift.
3. **Neither** — force prompt-layer-only enforcement
   (rejection-language stanza in the staged AGENTS.md, no server-side
   mechanical backstop).

## Decision

**prompt-layer-only.** Evidence + rationale below.

### Evidence (doc-fetch 2026-04-18)

Primary sources:
- <https://developers.openai.com/codex/hooks> — full hooks API reference.
- <https://developers.openai.com/codex/config-reference> — `[features] codex_hooks` + `[projects."<path>"].trust_level` keys.
- <https://developers.openai.com/codex/cli/reference> — CLI subcommand list.
- <https://developers.openai.com/codex/cli> — top-level CLI docs.

Key findings:

1. **PreToolUse covers Bash only**, not MCP / Write / WebSearch /
   other tools. The perfxpert gate MUST intercept EVERY non-
   `perfxpert_*` tool call (including `mcp__*`, `Read`, `Write`,
   `WebFetch`, etc.) until `intent_classify` returns. A Bash-only
   hook cannot satisfy that contract — the gate would be trivially
   bypassable by calling any non-Bash tool first.
2. **Matcher limited to tool_name = "Bash"**. No documented examples
   of matching `mcp_perfxpert_*`, `mcp_*`, or tool-name globs on
   non-Bash tools. The docs explicitly call out this as "Work in
   progress."
3. **API drift from plan assumptions.** Plan's Task 10 assumed
   `codex projects list --json` and `codex projects trust .`
   commands existed. As of April 2026 they do NOT; trust is
   config-only via a `[projects."<abs-path>"]` table in
   `~/.codex/config.toml` with `trust_level = "trusted" | "untrusted"`.
   The adapter handles this correctly by reading the config file
   directly (stdlib tomllib for reads, lazy tomlkit for writes).

### Why prompt-layer-only over native-hook on Codex

1. **Completeness requirement.** The gate must block EVERY tool type,
   not just Bash. Codex's PreToolUse cannot do that today.
2. **False sense of security.** Installing a Bash-only hook would
   suggest the gate is enforced when most tool calls (MCP in
   particular) would bypass it silently. That's worse than no hook
   at all — it makes the failure mode harder to diagnose.
3. **Follows plan guardrail.** Plan's Task 10 + guardrail
   explicitly says: *"If Codex's hook surface probe returns
   ambiguous, default to `prompt-layer-only` and document in the
   decision record."* We treat "Bash-only" as fulfilling the
   "cannot enforce the gate" condition.
4. **Re-visit path is clean.** When Codex expands PreToolUse to MCP
   tool interception (it's marked WIP), a future adapter patch can
   flip this decision to `native-hook` by implementing
   `CodexGateHook.install()` to write the `hooks.json` entry and
   removing the `GateHookUnsupported` in `_gate_hooks/codex.py`.
   At that point, re-introduce a return-value dataclass modelled on
   `ClaudeGateInstallResult` (dropped in the PR 2 review cleanup
   because nothing consumed it).

### Change set for Task 4.6 Codex-portion

**Applied during PR 2 implementation.** The code in
`perfxpert/cli/_gate_hooks/codex.py` raises `GateHookUnsupported`
from `install()`, which `CodexAdapter.install()` catches BEFORE
MCP registration (I-N1 partial-state protection) and records as
`gate_hook_installed=False` in the subsequent `verify_mcp_live()`
report. No partial state is ever left behind.

`verify_mcp_live()`'s `_probe_gate_hook_installed()` returns
`False` by design on Codex — the documented-known-limit state per
`LiveCheckReport.gate_hook_installed`, NOT a failure.

Rejection-language stanza in the staged `AGENTS.md` (rendered by
`_prompt_adapter.REJECTION_LANGUAGE_STANZA`) is the ONLY
enforcement on Codex today. Model-size sensitivity per R-new-4 is
accepted for this backend.

### One-paragraph degraded-gate notice (for release notes)

On the Codex backend, perfxpert's tool-priority gate is enforced at
the **prompt layer only**, not by a server-side PreToolUse hook.
This is because Codex's native PreToolUse hook (available as of
April 2026 behind `[features] codex_hooks = true`) currently only
intercepts Bash tool calls — it cannot block MCP, Write, or other
tool types, so it cannot satisfy the perfxpert "block every
non-perfxpert tool call until `intent_classify` returns"
requirement. Smaller models may bypass the advisory rejection
language; the recommendation is to use `claude` or `opencode` (both
of which have mechanical enforcement) when model-size bypass is a
concern. When Codex expands PreToolUse to MCP interception, this
adapter will switch to native-hook enforcement in a follow-up.

## Acceptance

This decision record is resolved when:

- [x] Doc-fetch step completed and cited with URLs + retrieval date.
- [x] Decision updated from PENDING to chosen option (prompt-layer-only).
- [x] Change set for Task 4.6 Codex-portion noted.
- [x] API drift from plan assumptions documented.

Task 4.6 Codex-portion is now **RESOLVED (prompt-layer-only)**.

## Re-visit conditions

Flip to `native-hook` when ANY of:

1. Codex's hooks docs state PreToolUse intercepts MCP tool calls
   (verify via doc-fetch + live probe at the time).
2. A `matcher` pattern that matches `mcp_perfxpert_*` or
   equivalent is documented.
3. A user files a feature request against perfxpert asking for
   mechanical Codex enforcement AND Codex has shipped the
   interception expansion.

Re-visit implementation would: (a) remove the unconditional
`GateHookUnsupported` in `perfxpert/cli/_gate_hooks/codex.py`, (b)
write `hooks.json` + toggle `[features] codex_hooks = true`, and
(c) flip `_probe_gate_hook_installed()` to return `True` on success.

## Session state location (I-N3 table entry)

| Backend | Session state location | Invalidation |
|---|---|---|
| Codex   | N/A (prompt-layer-only today) — no persistent server-side state. Rejection-language stanza is re-emitted via the staged `.perfxpert/AGENTS.md` every session. | N/A |

Future (native-hook flip): session state would live at
`~/.codex/runtime/perfxpert-gate-<session_id>.json`, invalidated
at session end — mirroring the Gemini approach.
