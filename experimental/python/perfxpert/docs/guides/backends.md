# Multi-Backend Launcher (`perfxpert-code <backend>`)

`perfxpert-code` is multi-backend. The default `perfxpert-code`
invocation still launches the AMD-branded bundled opencode (Phase 7
deliverable, unchanged). In addition, each supported backend has a
subcommand that registers the `perfxpert-mcp` server in the backend's
native config, stages an `AGENTS.md`-equivalent rendered prompt,
installs a pre-tool-call gate hook, and then execs the backend's
native TUI with `perfxpert-mcp` already attached.

This guide covers the user-visible surface. For the architectural
contract every adapter satisfies, see
[../architecture/backend-adapter.md](../architecture/backend-adapter.md).
For the underlying MCP server (34 READ_ONLY tools), see
[../integration/mcp-server.md](../integration/mcp-server.md). The
rationale for the Claude PreToolUse choice is recorded in
[../../../../../docs/decisions/2026-04-19-claude-hook-surface.md](../../../../../docs/decisions/2026-04-19-claude-hook-surface.md).

## Why multi-backend?

Users arrive with a backend already chosen — Claude Code, Gemini CLI,
Codex CLI, or no preference. Forcing everyone through the bundled
opencode (Phase 7) blocked adoption for Claude Code / Gemini users
who had already invested in their native TUI, muscle memory, and
auth. The multi-backend launcher lets `perfxpert-code <backend>`
write the correct MCP registration + gate hook + prompt cache for
whichever backend the user chose, then exec the native binary — the
perfxpert tool discipline travels with the install, not the TUI.

The bundled opencode remains the recommended default: it ships
pre-patched (the 17 AMD patches + STRICT-TOOL-DISCIPLINE stanza), so
no user-side install is required for the tool-priority gate to work.

## Backend comparison

| Backend | Subcommand | LLM | Config location | Scope | Gate hook | MCP tool prefix |
|---------|-----------|-----|-----------------|-------|-----------|-----------------|
| **opencode** (default, bundled) | `perfxpert-code` | Any (via opencode provider) | `~/.cache/perfxpert/opencode/opencode.json` | Per-bundle | Patched system prompt + fork patches 0010, 0020 | `perfxpert_*` |
| **Claude Code** | `perfxpert-code claude` | Anthropic Claude | `./.mcp.json` + `./.claude/CLAUDE.md` + `./.claude/settings.json` | Project | Native `PreToolUse` hook (event-based lift) | `mcp__perfxpert__*` |
| **Gemini CLI** | `perfxpert-code gemini` | Google Gemini | `~/.gemini/settings.json` + `./.perfxpert/AGENTS.md` | User (MCP) + project (prompt) | `allowedTools` restriction (event-based lift) | `mcp_perfxpert_*` |
| **Codex CLI** (coming soon) | `perfxpert-code codex` | OpenAI | `~/.codex/config.toml` (TBD) | Project (TBD) | TBD | `mcp_perfxpert_*` (TBD) |

Consent is requested once per **(backend, cwd-hash, file-set-hash)**
tuple and persisted — re-running the same subcommand in the same
directory is silent. Changing the file set (e.g. adding
`--allow-agents-md-append`) re-prompts because the hash changes.

## Install recipes

### Default: bundled opencode

No subcommand, no extra install. Phase 7 patched the upstream opencode
with AMD branding + the STRICT-TOOL-DISCIPLINE stanza + AMD red
palette; the launcher spawns the bundled binary with `perfxpert-mcp`
already wired into its config.

```bash
# SKIP-SAMPLE — requires the bundled opencode binary on PATH
perfxpert-code
```

### Claude Code (`perfxpert-code claude`)

Registers perfxpert in the project `.mcp.json`, stages the rendered
prompt at `.perfxpert/AGENTS.md`, writes a pointer at
`.claude/CLAUDE.md`, and installs the native `PreToolUse` hook inside
`.claude/settings.json`.

```bash
# SKIP-SAMPLE — requires claude CLI ≥ 2.1.59 on PATH
perfxpert-code claude

# Equivalent explicit form (showing the MCP registration step):
claude mcp add perfxpert --scope project -- perfxpert-mcp
```

If your project already tracks `.claude/CLAUDE.md` (common for teams),
the adapter writes a pointer by default and leaves your tracked file
alone. To append the rendered prompt into your tracked file instead,
pass `--allow-agents-md-append` (re-prompts consent because the file
set changes):

```bash
# SKIP-SAMPLE — opt-in for appending to a tracked CLAUDE.md
perfxpert-code claude --allow-agents-md-append
```

### Gemini CLI (`perfxpert-code gemini`)

List-appends the staged prompt cache to `context.fileName` in
`~/.gemini/settings.json` (preserving any user entries) and registers
perfxpert under `mcpServers`. The adapter **never** touches the
user's `GEMINI.md` — list-append in `context.fileName` is the
supported extension point.

```bash
# SKIP-SAMPLE — requires gemini CLI ≥ 0.2.0 on PATH
perfxpert-code gemini
```

### Codex CLI — deferred

The Codex adapter ships in PR 2. Running the subcommand today prints
a "deferred" message and exits 42. Use `perfxpert-code claude` or
`perfxpert-code gemini` in the meantime.

```bash
# SKIP-SAMPLE — prints deferred notice and exits 42
perfxpert-code codex
```

## Uninstall recipes

`perfxpert-code uninstall <backend>` reverses the install.
Marker-block drift (content hand-edited inside a perfxpert-managed
block) is detected and the file is left untouched — the command
reports the skipped paths and exits non-zero.

```bash
# SKIP-SAMPLE — reverses a Claude install under the current cwd
perfxpert-code uninstall claude

# SKIP-SAMPLE — reverses a Gemini install
perfxpert-code uninstall gemini

# SKIP-SAMPLE — non-interactive: consent the uninstall in advance
perfxpert-code uninstall --yes claude
```

On a successful uninstall, all of the following are reverted: MCP
registration entry, `.perfxpert/AGENTS.md` cache, any pointer file
the adapter wrote, and the gate-hook settings block. Files the user
created (e.g. a pre-existing `.claude/CLAUDE.md`) are preserved.

## Advanced flags

All `perfxpert-code <backend>` subcommands accept these dispatcher-
owned flags before any backend-native args:

| Flag | Effect |
|------|--------|
| `--dry-run` | Run `adapter.plan()`, print the actions that would run, skip every write, skip `spawn()`. No consent prompt fires. |
| `--quiet` | Suppress the AMD banner and the per-step install progress log. Errors still go to stderr. |
| `--force` | Bypass the recursion guard (refuse-if-already-inside-a-perfxpert-session) and the clobber guard. |
| `--allow-agents-md-append` | Opt in to appending the rendered prompt into a tracked `CLAUDE.md` / `GEMINI.md` / `AGENTS.md`. Default is to write a separate cache file. |

Dispatcher flags are consumed **greedily from the front** of argv;
the first non-dispatcher token ends the consume and the remainder is
passed to the backend binary unchanged. Example:

```bash
# SKIP-SAMPLE — --dry-run consumed by perfxpert-code; 'hello' reaches the backend
perfxpert-code claude --dry-run hello

# SKIP-SAMPLE — --dry-run here is treated as a backend-native flag
perfxpert-code claude hello --dry-run
```

The uninstall subcommand accepts a separate short flag list:
`--yes` / `-y` (non-interactive consent) and `--quiet`.

## Environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `PERFXPERT_MCP_WARMUP_TIMEOUT_S` | `10` | Seconds the `perfxpert-mcp` warmup probe waits for the server to answer `tools/list` during install verification. |
| `PERFXPERT_MCP_RETRY_BUDGET_S` | `30` | Total retry budget (seconds) for the tool-name-match retry loop in `verify_mcp_live()`. |
| `PERFXPERT_SKIP_LIVE_CHECK` | unset | `1`/`true`/`yes` skips the post-install `verify_mcp_live()` step entirely. Useful in CI where the backend isn't reachable. |
| `PERFXPERT_ASSUME_CONSENT` | unset | `1`/`true`/`yes` bypasses the interactive consent prompt (install AND uninstall). Required for non-interactive stdin. |
| `PERFXPERT_IN_AGENT_SESSION` | set by dispatcher | Recursion guard — the dispatcher sets this to the backend name in the child env so a nested `perfxpert-code <backend>` refuses to launch. Override with `--force`. |

## Troubleshooting

### Gate never fires

Symptom: the backend runs `bash` / `read` / `edit` straight away
without first routing through `perfxpert_intent_classify`.

Check, in order:

1. Run `perfxpert-code <backend> --dry-run` and confirm the
   "Install PreToolUse gate hook" (Claude) or
   "Write allowedTools gate restriction" (Gemini) action is listed.
   If missing, the adapter refused to install the hook — look for
   a `GateHookUnsupported` warning in the install log.
2. Confirm the gate hook file is present:
   `cat .claude/settings.json | jq '.hooks.PreToolUse'` (Claude)
   or `cat ~/.gemini/settings.json | jq '.allowedTools'` (Gemini).
3. Start a fresh session. The lift is event-based: after
   `perfxpert_intent_classify` returns once, the gate lifts for the
   remainder of THAT session. A brand-new session (new `session_id`)
   always starts with the gate engaged again — by design.

### Consent didn't persist

Symptom: re-running `perfxpert-code <backend>` in the same cwd
re-prompts for consent.

Cause: the file set changed between runs. Consent is keyed on the
tuple **(backend, cwd-hash, file-set-hash)** — adding or removing
`--allow-agents-md-append`, or the user creating / deleting one of
the target files, rolls the hash and invalidates the cached consent.

Workaround: export `PERFXPERT_ASSUME_CONSENT=1` for the current shell
if you want to skip re-prompting entirely. The consent cache lives
under `~/.perfxpert/consent.json`.

### Backend not found

Symptom: `perfxpert-code claude: 'claude' not found on PATH.
Install via https://code.claude.com/docs/en/install` (or the
equivalent for `gemini`).

Cause: the dispatcher calls `adapter.check_available()` before any
writes; if the binary is missing or below `min_version`, no install
runs. The error message includes the adapter's `install_hint` with a
verified URL.

Fix: install the backend per the printed hint. Re-run with `--dry-run`
first to confirm the binary is now discovered:

```bash
# SKIP-SAMPLE — --dry-run still runs check_available + plan()
perfxpert-code claude --dry-run
```

### Decision record PENDING error

Symptom: an older build prints
`GateHookUnsupported: Claude hook surface pending decision`.

Cause: cycle-2 pre-PR-1 builds raised `GateHookUnsupported` if the
hook-surface decision record was missing. The decision is now
recorded — update to the PR 1 build (cycle-3 or later), or link to
the decision record at
[../../../../../docs/decisions/2026-04-19-claude-hook-surface.md](../../../../../docs/decisions/2026-04-19-claude-hook-surface.md)
and retry.

### MCP warmup times out

Symptom: `PartialInstall: perfxpert MCP registered but live-check
failed: warmup timeout after 10s`.

Cause: `perfxpert-mcp` took longer than `PERFXPERT_MCP_WARMUP_TIMEOUT_S`
to answer `tools/list`. On slow disks or first-boot cold caches this
can happen.

Fix: raise the timeout or skip live-check in CI:

```bash
# SKIP-SAMPLE — raise the warmup budget to 30s
export PERFXPERT_MCP_WARMUP_TIMEOUT_S=30
perfxpert-code claude

# SKIP-SAMPLE — CI path: skip live-check entirely
export PERFXPERT_SKIP_LIVE_CHECK=1
perfxpert-code claude
```

## See also

- [../architecture/backend-adapter.md](../architecture/backend-adapter.md)
  — the `BackendAdapter` protocol + lifecycle contract (contributors)
- [../integration/mcp-server.md](../integration/mcp-server.md) —
  underlying MCP server + 34 READ_ONLY tool list
- [../../../../../docs/decisions/2026-04-19-claude-hook-surface.md](../../../../../docs/decisions/2026-04-19-claude-hook-surface.md)
  — why Claude uses the native `PreToolUse` hook surface
- [getting-started.md](getting-started.md) — §"Choosing a backend"
  for the short recipe
