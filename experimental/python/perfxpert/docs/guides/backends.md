# Multi-Backend Launcher (`perfxpert-code <backend>`)

`perfxpert-code` is multi-backend. Plain `perfxpert-code` launches the
patched opencode path: packaged installs prefer the bundled
AMD-branded binary, while source/editable checkouts prefer a locally
built patched binary from the pinned `opencode` submodule. In
addition, each supported backend has a
subcommand that registers the `perfxpert-mcp` server in the backend's
native config, stages an `AGENTS.md`-equivalent rendered prompt,
installs a pre-tool-call gate hook, and then execs the backend's
native TUI with `perfxpert-mcp` already attached.

This guide covers the user-visible surface. For the architectural
contract every adapter satisfies, see
[../architecture/backend-adapter.md](../architecture/backend-adapter.md).
For the underlying MCP server (56 READ_ONLY tools — 8 agent-hierarchy
entry points + 47 classifier/knowledge tools + 1 `trace_diff.diff_runs`),
see
[../integration/mcp-server.md](../integration/mcp-server.md). The
rationale for the Claude PreToolUse choice is captured in the local
Claude hook-surface decision record; the (different) decision for
Codex is captured in the local Codex hook-surface decision record.

**Backend LLM freely picks tools.** There is no
forced-call contract between `perfxpert_intent_classify` and any
aggregator tool. The backend LLM reads the agent hierarchy as
reference in `AGENTS.md` and calls whichever of the 56 MCP tools
match the user's intent — it may invoke any agent-hierarchy tool
(`perfxpert_agent_root`, `perfxpert_agent_analysis`,
`perfxpert_agent_diff_specialist`, etc.) directly,
or compose lower-level classifier tools (`bottleneck_*`,
`regression_*`, …) when it already knows the answer shape. The
tool-priority gate still requires `intent_classify` as the first
call, but nothing after that is mechanically forced.

## Why multi-backend?

Users arrive with a backend already chosen — Claude Code, Gemini CLI,
Codex CLI, or no preference. Forcing everyone through the bundled
opencode blocked adoption for Claude Code / Gemini users
who had already invested in their native TUI, muscle memory, and
auth. The multi-backend launcher lets `perfxpert-code <backend>`
write the correct MCP registration + gate hook + prompt cache for
whichever backend the user chose, then exec the native binary — the
perfxpert tool discipline travels with the install, not the TUI.

The patched opencode path remains the recommended default. In wheels it
ships bundled; in source checkouts it can be rebuilt from the pinned
submodule. That path carries the STRICT-TOOL-DISCIPLINE stanza and the
fork-only opencode gate hook without any extra backend install.

## Backend comparison

| Backend | Subcommand | LLM | Config location | Scope | Gate hook | MCP tool prefix |
|---------|-----------|-----|-----------------|-------|-----------|-----------------|
| **opencode** (default patched path) | `perfxpert-code` | Any (via opencode provider) | `~/.cache/perfxpert/opencode/opencode.json` | Per-bundle / per-checkout | Patched system prompt + fork patches 0010, 0020 | `perfxpert_*` |
| **Claude Code** | `perfxpert-code claude` | Anthropic Claude | `./.mcp.json` + `./CLAUDE.local.md` + `./.claude/settings.json` | Project | Native `PreToolUse` hook (event-based lift) | `mcp__perfxpert__*` |
| **Gemini CLI** | `perfxpert-code gemini` | Google Gemini | `./.gemini/settings.json` + `./.perfxpert/AGENTS.md` | Project | Native `BeforeTool` / `AfterTool` hooks (event-based lift) | `mcp_perfxpert_*` |
| **Codex CLI** | `perfxpert-code codex` | OpenAI | `~/.codex/config.toml` (trust) + `./.codex/config.toml` when trusted or fallback `~/.codex/config.toml` for MCP + `./AGENTS.override.md` | Trust in user config; MCP project-local when trusted | Prompt-layer-only (Codex `PreToolUse` is Bash-only — see decision record) | `mcp__perfxpert__*` |

Consent is requested once per **(backend, cwd-hash, file-set-hash)**
tuple and persisted — re-running the same subcommand in the same
directory is silent. Changing the file set (e.g. adding
`--allow-agents-md-append`) re-prompts because the hash changes.

## Install recipes

### Default: patched opencode path

![perfxpert-code](assets/gifs/14-perfxpert-code.gif)

*`perfxpert-code` on the default patched path, with the interactive TUI
and MCP wiring ready to go.*

No subcommand, no extra backend install. In packaged installs the
launcher uses the bundled AMD-branded patched binary. In source/editable
checkouts it prefers a locally built patched binary from
`experimental/python/perfxpert/opencode`. Only when neither patched
copy exists does it fall back to an upstream `opencode` on disk, with a
warning and without the fork-only gate behavior.

```bash
# SKIP-SAMPLE — requires a patched opencode path (repo-local build or bundled wheel)
perfxpert-code
```

### Claude Code (`perfxpert-code claude`)

Registers perfxpert in the project `.mcp.json`, stages the rendered
prompt at `.perfxpert/AGENTS.md`, writes a pointer at
`CLAUDE.local.md` (at the project root), and installs the native
`PreToolUse` hook inside `.claude/settings.json`.

Claude Code support is implemented in-tree and covered by the backend
adapter plus dispatcher tests.

```bash
# SKIP-SAMPLE — requires claude CLI ≥ 2.1.59 on PATH
perfxpert-code claude

# Equivalent explicit form (showing the MCP registration step):
claude mcp add perfxpert --scope project -- perfxpert-mcp
```

If your project already tracks `CLAUDE.local.md` (common for teams),
the adapter writes a pointer by default and leaves your tracked file
alone. To append the rendered prompt into your tracked file instead,
pass `--allow-agents-md-append` (re-prompts consent because the file
set changes):

```bash
# SKIP-SAMPLE — opt-in for appending to a tracked CLAUDE.md
perfxpert-code claude --allow-agents-md-append
```

### Gemini CLI (`perfxpert-code gemini`)

List-appends the staged prompt cache to `context.fileName` in the
project-local `.gemini/settings.json` (preserving any project entries),
registers perfxpert under `mcpServers`, and installs native
`BeforeTool` / `AfterTool` Gemini hooks in the same project settings.
The adapter **never** touches the user's `GEMINI.md` — list-append in
`context.fileName` is the supported extension point.

Gemini support is implemented in-tree and covered by the backend
adapter plus dispatcher tests.

```bash
# SKIP-SAMPLE — requires gemini CLI ≥ 0.2.0 on PATH
perfxpert-code gemini
```

### Codex CLI (`perfxpert-code codex`)

Marks the current project as trusted via the
`[projects."<abs-cwd>"]` table in `~/.codex/config.toml` (required —
Codex refuses to run agents in untrusted projects), registers
perfxpert under `[mcp_servers.perfxpert]` in `<cwd>/.codex/config.toml`
for trusted projects or falls back to `~/.codex/config.toml` when
project scope is unavailable, and writes a perfxpert-managed
`<cwd>/AGENTS.override.md` compatibility override so Codex actually
loads the rendered prompt. Writes preserve comments + key ordering via
lazy-imported `tomlkit`.

Codex support is implemented in-tree and covered by the backend
adapter plus dispatcher tests.

```bash
# SKIP-SAMPLE — requires codex CLI ≥ 0.7.0 on PATH
perfxpert-code codex

# Equivalent explicit form (showing the MCP registration step Codex does natively):
codex mcp add perfxpert -- perfxpert-mcp
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

Unlike Claude (`PreToolUse`) and Gemini (`BeforeTool` / `AfterTool`), the Codex
adapter does **not** install a server-side pre-tool-call gate hook.
Codex's native `PreToolUse` hook exists (behind `[features]
codex_hooks = true`) but currently intercepts Bash only — it cannot
block MCP, Write, or other tool calls. The perfxpert gate must
intercept every non-PerfXpert-prefixed tool call until
`intent_classify` returns, so a Bash-only hook cannot satisfy the
contract. Installing one anyway would give false confidence.

Instead, the Codex install relies on the rejection-language stanza
embedded in a perfxpert-managed `AGENTS.override.md`
(prompt-layer enforcement). When the repo already has a root
`AGENTS.md`, the adapter shadow-copies that file into the override and
appends a perfxpert-managed block so Codex sees both the repo guidance
and the perfxpert gate. `AGENTS.override.md` is a compatibility file
owned by the adapter, not a native Codex-only source of truth.
Codex can defer MCP tool metadata out of the initial model-visible tool
inventory when the session is crowded. In that case, the generated
Codex prompt allows exactly one pre-gate exception: use Codex's
metadata search tool (`tool_search` / `tool_search_tool`) to expose the
PerfXpert `intent_classify` and `workflow_next_step` MCP tools, then
call the gate immediately. If discovery does not expose the gate, the
agent must still stop with a PerfXpert configuration error instead of
using shell, SSH, build, edit, or profiling commands as a fallback.
Smaller models may bypass advisory language; if
mechanical enforcement matters for your workflow, use
`perfxpert-code claude` or the bundled `opencode` default (both
have server-side mechanical gates). The full rationale + re-visit
conditions are captured in the local Codex hook-surface decision
record.

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

# SKIP-SAMPLE — reverses a Codex install (drops MCP table + trust entry + staged AGENTS.override.md)
perfxpert-code uninstall codex

# SKIP-SAMPLE — non-interactive: consent the uninstall in advance
perfxpert-code uninstall --yes claude
```

On a successful uninstall, all of the following are reverted: MCP
registration entry, any discovered prompt file or managed prompt
block the adapter wrote, any pointer file the adapter wrote, and the
gate-hook settings block. Files the user
created (e.g. a pre-existing `CLAUDE.local.md`) are preserved. For
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
| `--allow-agents-md-append` | Opt in to appending the rendered prompt into a tracked prompt file when a backend supports that flow. Today this matters for Claude/Codex `AGENTS.md`-family files; Gemini always stages `.perfxpert/AGENTS.md` and never edits `GEMINI.md`. |

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
   "Install Gemini BeforeTool/AfterTool gate hooks" action is listed.
   If missing, the adapter refused to install the hook — look for
   a `GateHookUnsupported` warning in the install log.
2. Confirm the gate hook file is present:
   `cat .claude/settings.json | jq '.hooks.PreToolUse'` (Claude)
   or `cat .gemini/settings.json | jq '.hooks.BeforeTool'` (Gemini).
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
recorded (see the local Claude hook-surface decision record) —
update to the PR 1 build (cycle-3 or later) and retry.

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
| gemini   | `gemini-2.5-flash`              | Native `BeforeTool` / `AfterTool` hooks + runtime-state file for event-based lift. |
| codex    | *not probed*                    | Gate is prompt-layer-only (Codex `PreToolUse` is Bash-only). `install()` emits a warning-level log (`codex gate hook unsupported on this backend`) and records `gate_hook_installed=False`; `verify_mcp_live` still runs its connectivity checks (e.g. `codex mcp list`) but skips the gate-probe canary. Rationale is captured in the local Codex hook-surface decision record. |

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
  underlying MCP server + 56 READ_ONLY tool list (8 agent-hierarchy
  + 47 classifier/knowledge + 1 `trace_diff.diff_runs`)
- Local Claude hook-surface decision record — why Claude uses the
  native `PreToolUse` hook surface.
- Local Codex hook-surface decision record — why Codex uses
  prompt-layer-only enforcement (PreToolUse is Bash-only).
- [getting-started.md](getting-started.md) — §"Choosing a backend"
  for the short recipe
