# Multi-Backend Launcher (`perfxpert-code <backend>`)

`perfxpert-code` is multi-backend. The default `perfxpert-code`
invocation still launches the AMD-branded bundled opencode (unchanged).
In addition, each supported backend has a
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
[../../../../../docs/decisions/2026-04-19-claude-hook-surface.md](../../../../../docs/decisions/2026-04-19-claude-hook-surface.md);
the (different) decision for Codex lives in
[../../../../../docs/decisions/2026-04-19-codex-hook-surface.md](../../../../../docs/decisions/2026-04-19-codex-hook-surface.md).

## Why multi-backend?

Users arrive with a backend already chosen — Claude Code, Gemini CLI,
Codex CLI, or no preference. Forcing everyone through the bundled
opencode blocked adoption for Claude Code / Gemini users
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
| **Codex CLI** | `perfxpert-code codex` | OpenAI | `~/.codex/config.toml` (TOML: trust + MCP) + `./.perfxpert/AGENTS.md` | User (MCP + trust) + project (prompt) | Prompt-layer-only (Codex `PreToolUse` is Bash-only — see decision record) | `mcp_perfxpert_*` |

Consent is requested once per **(backend, cwd-hash, file-set-hash)**
tuple and persisted — re-running the same subcommand in the same
directory is silent. Changing the file set (e.g. adding
`--allow-agents-md-append`) re-prompts because the hash changes.

## Install recipes

### Default: bundled opencode

No subcommand, no extra install. The bundled opencode ships with AMD
branding + the STRICT-TOOL-DISCIPLINE stanza + AMD red palette pre-applied;
the launcher spawns the bundled binary with `perfxpert-mcp`
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

### Codex CLI (`perfxpert-code codex`)

Registers perfxpert in `~/.codex/config.toml` under the
`[mcp_servers.perfxpert]` table, stages the rendered prompt at
`<cwd>/.perfxpert/AGENTS.md`, and marks the current project as
trusted via the `[projects."<abs-cwd>"]` TOML table (required — Codex
refuses to run agents in untrusted projects). Writes preserve
comments + key ordering via lazy-imported `tomlkit`.

```bash
# SKIP-SAMPLE — requires codex CLI ≥ 0.7.0 on PATH
perfxpert-code codex

# Equivalent explicit form (showing the MCP registration step Codex does natively):
codex mcp add perfxpert perfxpert-mcp
```

#### Trust gate

Codex requires `[projects."<abs-path>"].trust_level = "trusted"` in
`~/.codex/config.toml` before it will run agents in a given project
directory. The adapter handles this for you: if the cwd is not yet
trusted, it prompts `[y/N]` (interactive) or aborts with
`TrustRequired` (non-interactive). To auto-trust during bootstrap or
CI, set `PERFXPERT_AUTO_TRUST=1`:

```bash
# SKIP-SAMPLE — bootstrap path: auto-trust the cwd without prompting
PERFXPERT_AUTO_TRUST=1 perfxpert-code codex
```

**Security caveat.** When `PERFXPERT_AUTO_TRUST=1` is honored, the
adapter prints a `[WARN]` line to stderr naming the trusted cwd —
and that warning bypasses `--quiet`. This is intentional: silently
marking a directory as trusted is a security-relevant action and
deserves an audit trail even in non-interactive runs. If you need
completely silent installs and your cwd is already trusted from a
previous run, the warning never fires.

#### Prompt-layer-only gate (why Codex differs)

Unlike Claude (`PreToolUse`) and Gemini (`allowedTools`), the Codex
adapter does **not** install a server-side pre-tool-call gate hook.
Codex's native `PreToolUse` hook exists (behind `[features]
codex_hooks = true`) but currently intercepts Bash only — it cannot
block MCP, Write, or other tool calls. The perfxpert gate must
intercept every non-`perfxpert_*` tool call until
`intent_classify` returns, so a Bash-only hook cannot satisfy the
contract. Installing one anyway would give false confidence.

Instead, the Codex install relies on the rejection-language stanza
embedded in the staged `.perfxpert/AGENTS.md` (prompt-layer
enforcement). Smaller models may bypass advisory language; if
mechanical enforcement matters for your workflow, use
`perfxpert-code claude` or the bundled `opencode` default (both
have server-side mechanical gates). The full rationale + re-visit
conditions are in
[../../../../../docs/decisions/2026-04-19-codex-hook-surface.md](../../../../../docs/decisions/2026-04-19-codex-hook-surface.md).

#### Git-tracked config refused

Same rule as Claude / opencode: if `~/.codex/config.toml` is tracked
inside a git repo (unusual but possible in dotfile repos), the
adapter refuses to write and raises `ConfigClobber` with a
`git rm --cached ~/.codex/config.toml` remediation hint. Malformed
TOML is surfaced the same way — `ConfigClobber("...not valid
TOML...Fix the syntax error...")` rather than a raw `tomlkit`
traceback. During uninstall, a malformed file is recorded as
`skipped_due_to_drift` so you can clean the other backends and come
back to the broken file.

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

# SKIP-SAMPLE — reverses a Codex install (drops MCP table + trust entry + staged AGENTS.md)
perfxpert-code uninstall codex

# SKIP-SAMPLE — non-interactive: consent the uninstall in advance
perfxpert-code uninstall --yes claude
```

On a successful uninstall, all of the following are reverted: MCP
registration entry, `.perfxpert/AGENTS.md` cache, any pointer file
the adapter wrote, and the gate-hook settings block. Files the user
created (e.g. a pre-existing `.claude/CLAUDE.md`) are preserved. For
Codex specifically, the `[projects."<cwd>"]` trust entry that the
install added is also removed; any other `[projects.*]` entries are
preserved untouched.

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
| `PERFXPERT_AUTO_TRUST` | unset | **Codex-only.** `1` auto-marks the current cwd as `trust_level = "trusted"` in `~/.codex/config.toml` without prompting. Prints a `[WARN]` to stderr identifying the trusted cwd; this warning bypasses `--quiet` by design (security-relevant audit trail). Required for non-interactive Codex bootstraps; skipped for Claude / Gemini / opencode. |

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

### Codex refuses to run ("project not trusted")

Symptom: `TrustRequired: current project <cwd> is not trusted by
Codex. Pass PERFXPERT_AUTO_TRUST=1 to auto-trust, OR rerun
interactively to accept the prompt.`

Cause: `perfxpert-code codex` was run in a non-interactive context
(e.g. CI, piped stdin) and the cwd wasn't yet in the
`[projects."..."]` table of `~/.codex/config.toml`.

Fix — pick one:

```bash
# SKIP-SAMPLE — CI path: auto-trust the cwd; warning prints to stderr
PERFXPERT_AUTO_TRUST=1 perfxpert-code codex

# SKIP-SAMPLE — interactive path: accept the prompt once, cached afterwards
perfxpert-code codex
```

### Codex uninstall reports `skipped_due_to_drift`

Symptom: `perfxpert-code uninstall codex` completes but the report
lists `~/.codex/config.toml` under `skipped_due_to_drift` (non-zero
exit).

Cause: either the file is git-tracked (refuses to write, same rule
as Claude / opencode) or the TOML is malformed (parse failed, also
refused). Drift protection is deliberate — the uninstall does not
overwrite user state.

Fix: inspect the file manually. If it's git-tracked, `git rm --cached
~/.codex/config.toml` and re-run the uninstall. If it's malformed,
fix the syntax (`codex mcp list` will hard-error on parse failure
too) and re-run. Other backends in the same uninstall invocation
are unaffected — they already cleaned up.

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

## Gate-probe coverage (acceptance criterion 9a)

The `verify_mcp_live()` gate probe (invoked during `install()` unless
`PERFXPERT_SKIP_LIVE_CHECK=1`) confirms the gate actually holds at
runtime by running a canned query through one small model per
backend. Acceptance criterion 9a from the multi-backend plan requires
that probe to pass against at least one small model per backend —
the specific models used are:

| Backend  | Small model used for gate probe | Notes |
|----------|---------------------------------|-------|
| opencode | opencode-default                | patched `{block, retryWith}` gate (bundled patch 0020). |
| claude   | `claude-haiku-4-5`              | native `PreToolUse` hook. R-new-4 scope: verified on haiku-4-5; other small models require independent re-verification at acceptance time. |
| gemini   | `gemini-2.5-flash`              | `allowedTools` restriction + runtime-state file for event-based lift. |
| codex    | *not probed*                    | Gate is prompt-layer-only (Codex `PreToolUse` is Bash-only). `install()` emits a warning-level log (`codex gate hook unsupported on this backend`) and records `gate_hook_installed=False`; `verify_mcp_live` still runs its connectivity checks (e.g. `codex mcp list`) but skips the gate-probe canary. See the [Codex hook-surface decision record](../../../../../docs/decisions/2026-04-19-codex-hook-surface.md). |

If you re-verify against a different small model for a non-Codex
backend (for example `claude-haiku-5` when it ships, or
`gemini-3.0-flash`), update this table and the corresponding
`R-new-4` / `I-N2` entries in the multi-backend plan. Today the
probe-target model for each backend is hard-coded inside that
adapter's `verify_mcp_live()` — there is no runtime override
env var. If future work parameterises this, add the env var name
here.

## See also

- [../architecture/backend-adapter.md](../architecture/backend-adapter.md)
  — the `BackendAdapter` protocol + lifecycle contract (contributors)
- [../integration/mcp-server.md](../integration/mcp-server.md) —
  underlying MCP server + 34 READ_ONLY tool list
- [../../../../../docs/decisions/2026-04-19-claude-hook-surface.md](../../../../../docs/decisions/2026-04-19-claude-hook-surface.md)
  — why Claude uses the native `PreToolUse` hook surface
- [../../../../../docs/decisions/2026-04-19-codex-hook-surface.md](../../../../../docs/decisions/2026-04-19-codex-hook-surface.md)
  — why Codex uses prompt-layer-only enforcement (PreToolUse is
  Bash-only)
- [getting-started.md](getting-started.md) — §"Choosing a backend"
  for the short recipe
