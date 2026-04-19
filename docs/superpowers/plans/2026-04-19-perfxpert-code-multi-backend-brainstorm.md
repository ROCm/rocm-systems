# perfxpert-code Multi-Backend Mode — Brainstorm

Date: 2026-04-19
Status: DRAFT for user review (plan, not implementation) — **Cycle-2 revision**
Companion: `2026-04-19-perfxpert-code-multi-backend-plan.md`

---

## 1. User need (restated)

Today, `perfxpert-code` launches a bundled/patched `opencode` binary with an
AMD-branded config dir (`AGENTS.md` + `opencode.json` + `mcp.json` + theme).
The user wants the same guardrails — perfxpert MCP server auto-wired and
the perfxpert system prompt active — to be available when they'd rather
use a different agent front-end: **Claude Code CLI (`claude`)**, **Codex
CLI (`codex`)**, or **Gemini CLI (`gemini`)**.

Constraints:

- The third-party tools must stay branded as themselves (they already
  have their own identity, UX, and auth flow — don't rebrand them).
- Default behavior of `perfxpert-code` with no subcommand stays
  **unchanged**: AMD-branded bundled opencode TUI.
- The injection must be **non-destructive** to whatever the user already
  has in `.mcp.json`, `~/.codex/config.toml`, or
  `~/.gemini/settings.json` — merge, never clobber.
- **Never touch a git-tracked `CLAUDE.md` / `AGENTS.md` / `GEMINI.md`** by
  default. Stage all content under `.perfxpert/` and use import-only
  pointers instead. (See §11 below — added in cycle-2.)
- Prefer **project-scoped** config (cwd) uniformly — MCP *and* prompt
  land in the same scope so consent, uninstall, and `.gitignore` guidance
  stay coherent.

Out-of-scope (this phase):

- Building a unified "agent kernel" that runs all 4 backends under one
  roof — too invasive. Each backend remains native.
- Any change to what `perfxpert-mcp` exposes (still 34 READ_ONLY tools).
- **Windows / native Windows paths.** Cycle-2 decision: POSIX-only for
  the first cut (macOS + Linux + WSL). Tracking issue for Windows
  support (§12).

---

## 2. Current state (reference)

Code paths read:

- `experimental/python/perfxpert/perfxpert/cli/opencode_launcher.py` —
  single entry point (`main()`), subcommand router
  (`route_subcommand()`), binary resolver, runtime-config stager
  (`_prepare_runtime_config_dir()` → `~/.cache/perfxpert/opencode/`).
- `experimental/python/perfxpert/perfxpert/_bundled/opencode_config/`
  — `AGENTS.md` (opencode-flavored; 111 lines), `opencode.json`
  (configures `default_agent=perfxpert`, wires `perfxpert-mcp` as local
  MCP), `mcp.json` (client-neutral MCP config), `amd-theme.json`.
- **`experimental/python/perfxpert/docs/integration/mcp-server.md`** —
  NOT present in the phase-8 worktree; it exists on phase 9. Cycle-2
  resolution: **Task 0** ports this file up to phase 8 before Task 7
  extends it; see plan doc.

### Canonical backend config shapes (verified late April 2026, cycle-2)

| Backend | Version | System-prompt file | System-prompt scopes | MCP registration | MCP config format | CLI helper to add MCP | Project override? | **Tool-name wire format** |
|---|---|---|---|---|---|---|---|---|
| **opencode (bundled)** | v1.4.11 | `AGENTS.md` | cwd | `opencode.json` `mcp` key | JSON `{"perfxpert": {"type": "local", "command": [...]}}` | — (config file only) | yes (cwd) | `perfxpert_<tool>` (raw) |
| **Claude Code CLI** | ≥ 2.1.59 | `CLAUDE.md` | **project `./.mcp.json` + `./.claude/CLAUDE.md` (NOT `~/.claude.json`)**, `CLAUDE.local.md` (gitignored), `@import` supported | **`.mcp.json` in cwd (scope=project, default)** or `~/.claude.json` (scope=user) | JSON `mcpServers: {name: {command, args, env}}` | `claude mcp add perfxpert --scope project -- perfxpert-mcp` | yes — `.mcp.json` checked-in, or `--scope project` | **`mcp__perfxpert__<tool>`** |
| **Codex CLI** | ≥ 0.7 | `AGENTS.md` | user `$CODEX_HOME/AGENTS.md` (default `~/.codex/`), project-scoped from git root walking down to cwd. 32 KiB cap. **Project config only loaded if project is TRUSTED** (`codex projects trust .`). | `~/.codex/config.toml` or project `.codex/config.toml` | TOML `[mcp_servers.NAME]` with `command`, `args`, `env`, `enabled`, `startup_timeout_sec`, `tool_timeout_sec` | `codex mcp add perfxpert --env K=V -- perfxpert-mcp` | yes (project `.codex/config.toml`) | **Undocumented — probe at install time via `verify_mcp_live()`; current observed form `mcp_perfxpert_<tool>` but unstable.** |
| **Gemini CLI** | ≥ 0.2 | `GEMINI.md` (or any filename listed in `context.fileName`) | user `~/.gemini/`, project `.gemini/` in cwd-ancestor chain. | `~/.gemini/settings.json` (user) or `.gemini/settings.json` (project) | JSON `mcpServers: {name: {command, args, env, cwd, timeout, trust, includeTools, excludeTools}}` | — (config file only) | yes (project `.gemini/settings.json`) | **`mcp_perfxpert_<tool>`** |

All four backends offer project-scoped config — good. All four honor
MCP stdio servers that spawn `perfxpert-mcp`. All four rewrite the
MCP-exposed tool name per their own client convention — this is the
**tool-name rewrite problem** at the heart of cycle-2 (§10, B1).

### Shell/bash tool availability (for ACTION phase)

All four have a built-in `bash`/`shell` tool (opencode: `bash`;
Claude Code: `Bash`; Codex: `shell`; Gemini: `shell_command`). The
perfxpert ACTION phase (build/profile/patch) works on all four.

---

## 3. Three candidate approaches

### Approach A — **Subcommand dispatch (recommended)**

`perfxpert-code <backend>` picks the backend; no subcommand = bundled
opencode (today's behavior).

```
perfxpert-code                   # default: bundled opencode TUI (UNCHANGED)
perfxpert-code claude            # → claude CLI with perfxpert guardrails
perfxpert-code codex             # → codex CLI with perfxpert guardrails
perfxpert-code gemini            # → gemini CLI with perfxpert guardrails
perfxpert-code claude --help     # passthrough to `claude --help`
perfxpert-code codex exec "…"    # arg passthrough
```

- Single entry point already exists (`opencode_launcher.main`); add three
  names to a backend registry and dispatch.
- `perfxpert-code claude run "analyze my trace"` → passes `run "..."`
  through to the `claude` subcommand system, with perfxpert MCP + prompt
  already installed.

### Approach B — **`--backend=…` flag**

`perfxpert-code --backend=claude [args]`.

**Cons.** More typing; fragile with first-positional routing.

### Approach C — **Sibling wrapper binaries**

`perfxpert-claude`, `perfxpert-codex`, `perfxpert-gemini`.

**Cons.** Bigger footprint — 3 new `[project.scripts]` entries,
3 new tiny entry-point modules.

### Recommendation: **Approach A (subcommand)**

Same rationale as cycle-1 — smallest code footprint, zero default-path
impact.

### Cycle-2 approach-comparison with blockers as risk-weighted arguments

| Criterion | A (subcommand) | B (flag) | C (sibling binaries) |
|---|---|---|---|
| Lines changed (approx) | ~80 dispatcher + N adapters | ~100 flag parser + N adapters | ~40/ea × 3 + N adapters |
| **B1 tool-name rewrite blast radius** | 1 adapter-level template field (`tool_name_template`) applied uniformly | same | 3× copy of same logic — drift risk |
| **B2 scope uniformity** | dispatcher enforces project-scope for MCP + prompt together | same | each wrapper enforces independently — drift risk |
| **B3 Codex trust gate** | single `verify_trust()` call in dispatcher | same | 3 copies — drift risk |
| Help discoverability | `perfxpert-code --help` lists backends | `--help` lists flag values | `<TAB>` completes binary names |
| Tab-complete | requires completion script | requires flag-value completion | native |
| Default-path preserved | yes, bare invocation unchanged | yes, unchanged | yes, unchanged |
| **Net verdict** | ✅ wins on every blocker | tied on blockers, loses on syntax | loses on drift risk |

Approach A is reaffirmed. Per-backend blocker mitigations (B1..B4) are
centralized in the dispatcher, not duplicated across wrappers.

---

## 4. Payload-adaptation strategy

### Option X — **Single source of truth, N adapters (recommended)**

Keep `_bundled/opencode_config/AGENTS.md` as the canonical perfxpert
agent instructions, translate on install per backend:

| Backend | Adapter behavior |
|---|---|
| opencode | Copy verbatim (today's behavior; `AGENTS.md` → runtime cfg dir). |
| Claude Code | Copy `AGENTS.md` → `<cwd>/.perfxpert/AGENTS.md` (read-only staging). If `CLAUDE.md` is git-tracked → write `<cwd>/.claude/CLAUDE.md` with an `@.perfxpert/AGENTS.md` import line. Only append to cwd `CLAUDE.md` with explicit `--allow-agents-md-append`. |
| Codex | Copy `AGENTS.md` → `<cwd>/.perfxpert/AGENTS.md`. If `AGENTS.md` is git-tracked → redirect via `$CODEX_HOME` pointing at our cache dir. Only append to cwd `AGENTS.md` with `--allow-agents-md-append`. |
| Gemini | Copy `AGENTS.md` → `<cwd>/.perfxpert/AGENTS.md`. `.gemini/settings.json` `context.fileName` APPENDS `.perfxpert/AGENTS.md` — **never modifies `GEMINI.md` itself**. (This is why Gemini moves up to PR 1; it is the safest backend by construction.) |

**Cycle-2 refinement — tool-name rewrite pipeline.** `AGENTS.md` is
authored with the canonical tool names `perfxpert_<tool>` (matching
opencode's wire format). Every adapter carries a
`tool_name_template: str` (e.g. `"{tool}"`, `"mcp__perfxpert__{tool}"`,
`"mcp_perfxpert_{tool}"`). The shared `_prompt_adapter.py` renders the
staged file by substituting every `perfxpert_<X>` in `AGENTS.md` with
the rendered per-backend form before writing. Adds a golden-file
snapshot test per backend — if the rendering changes, the test fails
loudly. (See B1 in §10.)

### Option Y — **N parallel prompts**

Rejected (4× maintenance burden). Same as cycle-1.

### Recommendation: **Option X** with the cycle-2 rewrite pipeline.

Backend-specific conditional blocks use `<!--backend:opencode-->` /
`<!--/backend:opencode-->` sentinels stripped per-target. Adapters
additionally version the marker: `<!-- perfxpert-managed v1 cache=<sha8> -->`
(see §11 / I6).

---

## 5. Install / activation flow per backend (recommended)

### 5.1 Claude Code (`perfxpert-code claude …`)

1. **Binary + version check.** `shutil.which("claude")` + parse
   `claude --version`; compare to `min_version = "2.1.59"`. Missing →
   `BackendNotFound` error with install hint. Too old → `VersionTooOld`.
2. **Consent gate.** `has_consent(backend, cwd_hash, file_set_hash)`.
   If consent is for a different cwd OR the target file set changed
   (e.g. `CLAUDE.md` newly tracked vs last run) — re-prompt. (I10.)
3. **MCP registration (project scope by default, idempotent).**
   - **Primary path: shell out** to
     `claude mcp add perfxpert --scope project -- perfxpert-mcp`
     (writes `.mcp.json` in cwd). Per-step progress line. Timeout 15s.
   - **On shell-out failure**, do NOT whole-file-rewrite `~/.claude.json`
     (it is a multi-MB session-history file). Instead, print the
     command for the user to run manually. (I4.)
   - Fallback structured edit is LAST RESORT — only for `.mcp.json`
     (small, dedicated file), never for `~/.claude.json`.
   - Append `.mcp.json` to `.gitignore` if a `.gitignore` is present
     and doesn't already cover it. (B2.)
4. **Prompt staging (never-touch-tracked-files default, I3).**
   - Stage `<cwd>/.perfxpert/AGENTS.md` with the rendered prompt (with
     `mcp__perfxpert__<tool>` names substituted, backend-only markers
     stripped, versioned sentinel).
   - Detect `CLAUDE.md` tracking via
     `git ls-files --error-unmatch CLAUDE.md`.
     - **If tracked OR nonexistent**: write `<cwd>/.claude/CLAUDE.md`
       (untracked by default) with `@.perfxpert/AGENTS.md`. Add
       `.claude/CLAUDE.md` to `.gitignore`.
     - **If untracked and user passed `--allow-agents-md-append`**:
       append `<!-- BEGIN perfxpert-managed v1 cache=<sha8> -->\n@.perfxpert/AGENTS.md\n<!-- END perfxpert-managed v1 -->`.
       Marker is versioned + cache-hashed; stale cache triggers re-stage. (I6.)
5. **Post-install verification (`verify_mcp_live()`, new in cycle-2, B1 + I11).**
   - Shell out to `claude mcp list --json`, parse, confirm `perfxpert`
     is present AND healthy. If not, surface the actual error.
   - If `PERFXPERT_TELEMETRY=1` is set, additionally send a short
     structured prompt ("reply with only 'ack' after calling
     `mcp__perfxpert__intent_classify`") and assert the perfxpert-mcp
     server saw `intent_classify` before any other tool. This is the
     miss phase-8 had — explicit gate, not an assumption. (I11.)
6. **Exec.** `os.execvpe("claude", argv, env)` — replace the Python
   process. Rationale: claude takes over the TTY, signal forwarding
   must be native, RSS should be freed. (I1.)
   - Exception: opencode still uses `subprocess.run` (no change;
     existing behavior is intentional).

### 5.2 Codex CLI (`perfxpert-code codex …`) — **DEFERRED to PR 2** (I9)

1. Binary + version check. Missing → install hint.
2. **Trust gate (cycle-2, B3).** Detect project trust status via
   `codex projects list --json` (or equivalent). If the current cwd is
   not trusted:
   - Interactive path: prompt the user to run `codex projects trust .`
     and re-invoke, OR fall back to user-scope with an explicit
     `⚠ trust-not-granted — installing to user scope` warning.
   - Non-interactive path: fail with `TrustRequired` exception and a
     line of guidance. No silent success.
3. MCP registration:
   - Primary path: `codex mcp add perfxpert -- perfxpert-mcp`.
     Timeout 15s.
   - Fallback: TOML round-trip via `tomlkit`, **lazy-imported inside
     the adapter's fallback branch only** (never module-level; optional
     `[backends]` extra). (I7.)
   - On shell-out failure → print-for-human command before direct edit.
4. Prompt staging (never-touch-tracked-files default):
   - Primary: stage `$CODEX_HOME/.perfxpert/AGENTS.md` via a per-cwd
     `CODEX_HOME` override, preserving user's untouched
     `~/.codex/AGENTS.md`.
   - If `AGENTS.md` in cwd is tracked → refuse to append without
     `--allow-agents-md-append`.
   - 32 KiB cap: pre-check staged file size; error before writing.
5. `verify_mcp_live()`: probe tool-name wire format (Codex's format is
   undocumented — see §2 table). The adapter lists live tool names and
   fails if the rendered form in AGENTS.md doesn't match what Codex
   exposes. (B1.)
6. Exec: `os.execvpe("codex", argv, env)`.

### 5.3 Gemini CLI (`perfxpert-code gemini …`) — **PROMOTED to PR 1** (I9)

Cycle-2 rationale for promotion: Gemini is the **safest backend by
construction** — `context.fileName` APPEND never touches the user's
`GEMINI.md`. PR 1 ships opencode (unchanged) + Claude + **Gemini**.
Codex moves to PR 2 (carries the trust-gate subplot).

1. Binary + version check.
2. MCP registration: JSON round-trip edit of
   `<cwd>/.gemini/settings.json` (project scope). Timeout 15s. Primary
   path has no shell-out helper in Gemini — config file only — so the
   structured edit IS the primary path, but `.gemini/settings.json` is
   small and user-owned, so whole-file-rewrite concern (I4) does not
   apply.
3. **Prompt staging (list-merge semantics, new in cycle-2).**
   - Stage `<cwd>/.perfxpert/AGENTS.md`.
   - Read existing `.gemini/settings.json`. If `context.fileName` is
     present and is a list, **append** `.perfxpert/AGENTS.md` to the
     list (preserving user's order and other entries). Never overwrite.
     If absent, set to `["GEMINI.md", ".perfxpert/AGENTS.md"]`.
   - `GEMINI.md` is never touched.
4. `verify_mcp_live()`: `gemini mcp list --json` (if available) else
   structured prompt as described in I11.
5. Exec: `os.execvpe("gemini", argv, env)`.

### 5.4 opencode (unchanged — today's behavior)

No change. `_prepare_runtime_config_dir()` + `OPENCODE_CONFIG` env +
`cd` into the runtime dir. Approach A's dispatch only activates when
the user types `claude` / `codex` / `gemini` as the first positional.
Exec keeps `subprocess.run` — intentional (we need to clean up the
runtime dir on exit).

---

## 6. `install-patches` in the multi-backend world

Y: **backend-aware `install`** subcommand.

`perfxpert-code install-patches` remains a deprecation alias with a
yellow-stderr notice for 2 release cycles, then removed. `install`
promotes `--backend={opencode,claude,codex,gemini,all}`. `all` is
serial (not parallel) — opencode first, then each requested backend in
registration order.

---

## 7. Open questions (flagged for user review)

1. **Consent UX.** First-run `perfxpert-code claude` prompts
   interactively. Consent is keyed on
   `(backend, cwd-hash, file-set-hash)` — if the target file set
   changes between runs (e.g. user just started tracking `CLAUDE.md`),
   re-prompt. Cache in `~/.config/perfxpert/config.yaml`
   (XDG-compliant). `PERFXPERT_ASSUME_CONSENT=1` suppresses. (I10.)
2. **Scope default: project vs user?** Cycle-2 decision: **project for
   both MCP and prompt**. User-scope opt-in via
   `PERFXPERT_<BACKEND>_SCOPE=user` (both MCP + prompt move together
   — never split). (B2.)
3. **`perfxpert-code claude` cwd vs staged runtime dir?** Cwd, because
   that's where `.mcp.json` and `.claude/CLAUDE.md` live (project scope).
4. **Auto-remove staged config on uninstall?** Yes — per-backend
   `uninstall <name>`. Removes the marker block (checks version + cache
   hash; refuses to remove if drifted), removes the
   `perfxpert` MCP entry (leaves other entries untouched), removes
   `.perfxpert/` if empty.
5. **Schema drift.** Each adapter records
   `min_version` + `known_schema_versions`; version-floor warn at
   first-run. On hard schema mismatch, raise `SchemaUnknown`.
6. **Cut for first release.** See §8 (revised).

---

## 8. First-cut scope recommendation (cycle-2 REVISED)

Ship **opencode (unchanged) + Claude Code + Gemini** in PR 1. Defer
**Codex** to PR 2.

Rationale (cycle-2):
- Gemini is **safer** than Codex: `context.fileName` list-append means
  `GEMINI.md` is never touched. No trust gate, no 32 KiB cap, no
  user-visible write into a file they may already track.
- Codex requires the **trust-gate subplot** (B3): detect trust, handle
  untrusted-cwd interactively, fall back to user-scope with an explicit
  warning. Non-trivial new code surface.
- Codex tool-name wire format is **undocumented** (B1). Shipping it
  requires a live-probe pattern that lands well when de-risked
  separately.
- Claude is required in PR 1 by user demand (it's the most-requested
  backend).

**Previous cycle-1 cut was opencode + Claude + Codex.** Cycle-2 swaps
Codex ↔ Gemini.

---

## 9. Sources

- Claude Code CLAUDE.md: https://code.claude.com/docs/en/claude-md
- Claude Code MCP: https://code.claude.com/docs/en/mcp
- Claude Code settings: https://code.claude.com/docs/en/settings
- Codex config: https://github.com/openai/codex/blob/main/docs/config.md
- Codex AGENTS.md: https://developers.openai.com/codex/guides/agents-md
- Codex MCP: https://developers.openai.com/codex/mcp
- Codex projects trust: (feature reference — TBD once Task 0 lands)
- Gemini MCP: https://github.com/google-gemini/gemini-cli/blob/main/docs/tools/mcp-server.md
- Gemini GEMINI.md: https://geminicli.com/docs/cli/gemini-md/
- PerfXpert `mcp-server.md` (to be ported to phase 8 in Task 0):
  `experimental/python/perfxpert/docs/integration/mcp-server.md`

---

## 10. Blockers (cycle-2)

These are the four items that made cycle-1 request-changes. They are
resolved in this revision.

### B1 — MCP tool-name rewrite per backend

Claude Code exposes MCP tools as `mcp__perfxpert__<tool>`; Gemini as
`mcp_perfxpert_<tool>`; Codex's format is undocumented; opencode is
raw `perfxpert_<tool>`. AGENTS.md today says `perfxpert_<tool>` — that
text is wrong for 3 of 4 backends.

**Resolution.** Every adapter carries a `tool_name_template: str` field.
`_prompt_adapter.py` (Task 4) substitutes every `perfxpert_<X>` in
`AGENTS.md` with the rendered per-backend form **before** staging.
Golden-file snapshot test per backend. `verify_mcp_live()` (Task 4.5)
probes the running backend post-install, lists tool names as seen by
the LLM, and fails loudly if they don't match the rendered AGENTS.md.

### B2 — Scope uniformity (MCP + prompt together)

cycle-1 was inconsistent: MCP at user scope, prompt at project scope.

**Resolution.** Project scope for BOTH by default.
`claude mcp add perfxpert --scope project -- perfxpert-mcp` writes
`.mcp.json` in cwd (NOT `~/.claude.json`). Prompt writes to
`.claude/CLAUDE.md` in cwd. Opt-out: `PERFXPERT_<BACKEND>_SCOPE=user`
moves **both** together. `.gitignore` hint included.

### B3 — Codex project-trust gate

cycle-1 silently assumed Codex's project-scope config would be loaded.
It's only loaded if the user trusts the project.

**Resolution.** Adapter detects trust status via
`codex projects list` (or equivalent). If untrusted:
(a) interactive-prompt the user to run `codex projects trust .`, OR
(b) fall back to user-scope with an explicit `⚠` warning, OR
(c) non-interactive → `TrustRequired` exception.
No silent success in an untrusted cwd.

### B4 — Stale `mcp-server.md` reference

cycle-1 plan cited
`experimental/python/perfxpert/docs/integration/mcp-server.md` as
existing. It lives on phase 9, not phase 8.

**Resolution.** New **Task 0** (see plan): port `mcp-server.md` from
phase 9 up to phase 8 before any other task runs. Plan is internally
consistent once Task 0 lands.

---

## 11. Important findings (cycle-2)

### I1 — `os.execvpe` for TUIs

cycle-1 used `subprocess.run` for all backends. For claude/codex/gemini
this is wrong (double Ctrl-C, resident Python, signal forwarding
breaks).

**Resolution.** `os.execvpe` for claude, codex, gemini. `subprocess.run`
stays for opencode (needs post-exit cleanup of the runtime dir).
Documented in the adapter `spawn()` contract.

### I2 — Full BackendAdapter lifecycle in Task 1

cycle-1 retroactively mutated the Protocol in Task 6 (adding
`dry_run: bool`). This breaks Tasks 4/5 on commit.

**Resolution.** Task 1 defines the full Protocol from day one:
`install(dry_run=False) -> InstallReport`,
`uninstall() -> UninstallReport`, `plan(dry_run=True) -> Plan`,
`spawn(argv, env) -> int` (does not return when exec'd),
`verify_mcp_live() -> LiveCheckReport`, and error taxonomy:
`BackendNotFound`, `VersionTooOld`, `ConfigClobber`, `ConsentDenied`,
`PartialInstall`, `SchemaUnknown`, `TrustRequired`. Methods
`register_mcp`, `stage_prompt`, `build_exec_args` are PROTECTED
building blocks, not public Protocol members. `dry_run` is in the
Protocol from day one.

### I3 — Never-touch-tracked-files default

cycle-1 appended to `CLAUDE.md` / `AGENTS.md` / `GEMINI.md` even if
tracked by git.

**Resolution.** If any of these files is git-tracked, default behavior
writes to an UNTRACKED sibling path (`.claude/CLAUDE.md`,
`$CODEX_HOME`-redirected, `context.fileName` append) with import-only
pointers to `.perfxpert/`. Append-to-tracked requires an explicit
`--allow-agents-md-append` flag.

Git-tracking detection:
`subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=cwd)`
— exit-0 = tracked.

### I4 — No whole-file `~/.claude.json` round-trip

cycle-1 fallback edited `~/.claude.json` directly. That file can be
multi-MB (session history).

**Resolution.** Always prefer shell-out. On shell-out failure, print
the command for the user to run manually rather than whole-file-
rewriting. Structured edit is LAST RESORT, only for small, dedicated
files (`.mcp.json`, `.codex/config.toml`, `.gemini/settings.json`).
Same rule for any backend.

### I5 — Per-step progress output + timeouts

cycle-1 was silent during multi-step first-run install.

**Resolution.** Emit per-step status lines:
`[1/4] Registering perfxpert MCP in .mcp.json ... ok`. Each step has a
15-second timeout. `--quiet` flag suppresses for CI.

### I6 — Marker versioning + cache-hash

cycle-1 marker `<!-- perfxpert-managed -->` could drift silently.

**Resolution.** Marker is
`<!-- perfxpert-managed v1 cache=<sha8> -->`. On every run, adapter
verifies BOTH the marker and the pointed-to cache file; if either is
stale or missing, re-stage. Never silently trust a byte-match of the
marker alone.

### I7 — `tomlkit` lazy import

cycle-1 implied a module-level `tomlkit` import behind an optional
extra. Latent ImportError when the extra isn't installed and the user
runs Codex.

**Resolution.** Import `tomlkit` INSIDE the Codex adapter, inside the
direct-edit fallback branch only. Primary path (shell-out to
`codex mcp add`) does not need `tomlkit`. Documented in the adapter
docstring.

### I8 — Tool-name length lint

Claude Code has a reported 64-char MCP tool-name cap.
`mcp__perfxpert__<underscored_name>` eats 16 chars before the tool's
own name.

**Resolution.** CI lint: for every registered `perfxpert.tools.*` tool,
`len("mcp__perfxpert__" + underscored_name) <= 64`. Warn at 56.

### I9 — PR 1 / PR 2 swap (Gemini in PR 1, Codex in PR 2)

See §8 above. Rewritten in brainstorm + plan.

### I10 — Consent per-`(backend, cwd-hash, file-set-hash)`

cycle-1 used a single global `backends.claude.consented=true`. That's
wrong — the target file set can change between runs.

**Resolution.** Consent is keyed on three fields. If any changes,
re-prompt. Stored in `~/.config/perfxpert/config.yaml`
(XDG-compliant).

### I11 — Observability / telemetry hook

Phase 8 had `.mcp.json` written correctly but perfxpert never received
a request — because the tool-name rewrite rendered our prompt's tool
references invalid. Tests didn't catch it.

**Resolution.** `PERFXPERT_TELEMETRY=1` env gate. When set,
`perfxpert-mcp` emits a structured log line for every tool-call
received: `{tool, timestamp, client_id}`. Adapter `verify_mcp_live()`
uses a short structured probe ("reply with only 'ack' after calling
`<rendered_name>_intent_classify`") and asserts the MCP server saw
`intent_classify` before any other tool. Directly addresses the
phase-8 miss.

### I12 — Platform support boundary (POSIX only)

cycle-1 was silent on Windows support.

**Resolution.** Explicitly POSIX-only in first cut: macOS, Linux, WSL.
Windows-native (backslash paths, CRLF, `~` expansion in `@imports`,
`.exe` resolution) is out of scope. Tracked as a follow-up issue.

---

## 12. Nitpicks (documented with rationale for deferral)

### N1 — `typing.Protocol` vs `abc.ABC`

`typing.Protocol` is intentional: backends are independent, no shared
behavior, structural typing fits the adapter-registry pattern. `ABC`
would require import-time coupling for no gain. **Retained; not a
blocker.**

### N2 — Named logger per adapter

Adapters gain a `perfxpert.cli._backend_log` named logger. Deferred
formal scaffolding to a follow-up doc; per-step `stderr.write` is
sufficient for cycle-2 acceptance. **Deferred with tracking issue.**

### N3 — Windows support follow-up

Platform boundary (I12) notes this is out of scope. Tracked as a
separate issue. **Deferred with rationale.**

---

## 13. Post-cycle-1 E2E findings (new)

The cycle-1 design review was a **static** review — it examined the
plan and brainstorm on paper. It did NOT cover runtime behavior. A
**live end-to-end (E2E)** exercise against the bundled opencode +
Haiku-4-5 surfaced two additional findings AFTER the cycle-1-revision
plan landed (commit 541f5cb6d7). Both are folded in here so the plan
reflects them.

### F1 (HIGH) — prompt-layer tool-priority gate is insufficient for smaller LLMs

**Evidence** (cycle-5 live E2E, Haiku-4-5 against bundled opencode):
- Scenario H1 "optimize multi_gpu_demo.cpp": first tool
  `perfxpert_intent_classify` ✓, BUT `read`/`glob` fired at positions
  2–4 **before** `perfxpert_workflow_next_step` (which came at
  position 7). 40% non-perfxpert tool ratio in the first 10 calls.
- Scenario H2 "profile with rocprofv3": **zero** `perfxpert_*` tools
  called. Model went straight to `bash → hipcc → rocprofv3`. Gate
  completely skipped. No false §5.8 refusal either, so the
  **gate-LIFT** half of the prompt is not the problem — the
  **gate-ENFORCE** half is ineffective on Haiku.

**Root cause.** "Do NOT emit bash/read/edit before intent_classify"
reads as advisory to smaller models. Larger models (Opus, Sonnet)
respect it; Haiku does not. Prompt language under-penalizes
non-compliance. Prompt-layer gating is inherently model-size
sensitive.

**Resolution — two mitigations:**

1. **Server-side PreToolUse hook (primary).** Intercept tool-calls in
   the first N turns of a session and reject any non-`perfxpert_*`
   call with a synthetic tool_result instructing the model to call
   `perfxpert_intent_classify` first. Mechanical enforcement,
   model-size-insensitive. Each backend needs its own hook surface:
   - **opencode**: `plugin.trigger("tool.execute.before", ...)` with
     `{ block: true, retryWith: <msg> }` extension. Starting point
     exists in phase 8 patch `0020-perfxpert-tool-gate.patch` README.
   - **Claude Code**: research needed — does `claude` expose a
     pre-tool-call hook? (Verify via docs fetch before Task 4.6.)
   - **Gemini**: per-tool policies or `allowedTools` list may block
     non-perfxpert tools for the first N turns.
   - **Codex**: sandbox-level hooks (deferred to PR 2).
2. **Prompt-layer reinforcement (secondary).** Strengthen AGENTS.md
   with explicit rejection language — not "do not emit" but
   "your response WILL BE REJECTED if you call bash before
   `perfxpert_intent_classify` returns." Frame as **consequence**
   not advisory.

The `verify_mcp_live()` method (already in the plan from cycle-1
fixes) gains an additional probe: after the prompt is staged, run a
canned query and assert the backend's LLM actually gates. This
catches the phase-8 regression class where the plumbing was correct
but the model did not respect it.

### F2 (MEDIUM) — `perfxpert-code run` bootstrap hang (exit-124)

**Evidence** (cycle-5 E2E reviewer report):
- Reproducible exit-124 on the **first** `perfxpert-code run <query>`
  invocation after stale server state.
- Single-line log; no traceback.
- Retrying 1–2× clears it.
- Likely MCP handshake timeout OR sqlite init race during
  `perfxpert-mcp` spawn.

**Why it matters for multi-backend.** This is an opencode-specific
bug today, but it affects the multi-backend reliability story —
every backend spawns `perfxpert-mcp` on stdio similarly. Claude,
Gemini, and Codex will all hit the same handshake race on first run.

**Resolution — three layered mitigations:**

1. **Warmup during `install()`.** The adapter's `install()` starts
   `perfxpert-mcp` once (warm up sqlite, cache tool registry) and
   stops it before the real spawn. First-run initialization happens
   on install, not on first model turn.
2. **`verify_mcp_live()` retry.** Retry up to 3 attempts with 2-second
   backoff on the MCP handshake, matching the bootstrap race.
3. **User-facing escape hatch.** Document `PERFXPERT_MCP_WARMUP=1` as
   an env override (default on) that users can set to `0` to skip
   warmup if it becomes annoying on their platform.

### Context: design-review scope vs E2E scope

The cycle-1 review was **static**: 4 blockers + 12 important findings
from reading the plan. It did not exercise runtime behavior. F1 and
F2 are the first **live-run** findings. The §13 changelog below (was
§13 in cycle-1 revision, now §14) documents cycle-1 findings; this
section documents cycle-2 findings discovered AFTER the plan was
drafted. Net effect: the plan now addresses both static-review issues
and runtime issues.

---

## 14. Reviewer feedback incorporated (changelog)

**Cycle-1 reviewers**: design-critic + practical engineer reviewer.
Verdict: request-changes with 4 blockers + 12 important findings.
**Cycle-5 E2E reviewer**: added F1 (HIGH) + F2 (MEDIUM) after the
cycle-1-revision plan landed (commit 541f5cb6d7); see §13.

| # | Finding | Resolved in §/Task |
|---|---|---|
| B1 | MCP tool-name rewrite per backend | §10 B1 + Task 4 (`_prompt_adapter`) + Task 4.5 (`verify_mcp_live`) |
| B2 | Scope uniformity | §1 constraints + §5.1 step 3 + §11 I6 |
| B3 | Codex project-trust gate | §5.2 step 2 + Task 5 (PR 2) |
| B4 | Stale `mcp-server.md` reference | §2 + Task 0 |
| I1 | `os.execvpe` for TUIs | §5.1 step 6 + adapter contract (Task 1) |
| I2 | Full lifecycle in Task 1 | Task 1 rewritten |
| I3 | Never-touch-tracked-files default | §5.1 step 4 + §11 I3 |
| I4 | No whole-file `~/.claude.json` | §5.1 step 3 + §11 I4 |
| I5 | Progress + timeouts | §5.1 + §11 I5 |
| I6 | Marker versioning + cache-hash | §4 + §11 I6 |
| I7 | `tomlkit` lazy import | §5.2 step 3 + §11 I7 |
| I8 | Tool-name length lint | §11 I8 + Task 4 test |
| I9 | PR 1 / PR 2 swap | §8 + Task 10 |
| I10 | Consent per-(backend, cwd, file-set) | §7 Q1 + §11 I10 + Task 3 |
| I11 | Telemetry hook | §5.1 step 5 + §11 I11 + Task 4.5 |
| I12 | POSIX-only boundary | §1 out-of-scope + §11 I12 |
| N1 | Protocol vs ABC | §12 N1 |
| N2 | Named logger | §12 N2 |
| N3 | Windows follow-up | §12 N3 |
| **F1** | **Prompt-layer gate insufficient for smaller LLMs** | **§13 F1 + Plan Task 4.6 (server-side PreToolUse hook per backend) + Plan Task 4 `_prompt_adapter` rejection-language stanza + R-new-4** |
| **F2** | **`perfxpert-code run` bootstrap exit-124** | **§13 F2 + Plan Task 4.7 (adapter `install()` warms `perfxpert-mcp`) + Plan `verify_mcp_live()` 3-attempt retry + `PERFXPERT_MCP_WARMUP=1` env hatch** |

Additional findings from the practical reviewer that were
incorporated:
- Git-tracking detection mechanism (I3) — explicit `git ls-files`
  check.
- List-merge semantics for Gemini `context.fileName` (§5.3 step 3).
- `--help` passthrough invariant — if first arg after backend is
  `--help` / `-h`, do NOT run the installer (plan Task 4 explicit test).
- `PERFXPERT_OPENCODE_PATH` should be ignored when backend != opencode
  (plan Task 2 short-circuit).
- Single `PERFXPERT_IN_AGENT_SESSION=<backend>` recursion guard
  instead of multiplying per-backend env vars (plan Task 4/5/10).
- XDG-compliant consent cache (plan Task 3).
- Per-backend `uninstall <name>` (plan Task 8).

### Findings deliberately NOT addressed (with rationale)

- **Docker image with pre-installed claude/codex/gemini for CI**
  (practical §4.1 option a). Rejected: adds GB-scale image; skip-on-CI
  + manual recipe is sufficient for cycle-2. Revisit if the skip gate
  lets a regression slip through.
- **`perfxpert-code status` subcommand** (practical §12). Deferred to
  a follow-up; not blocking cycle-2 acceptance.
- **Security note about prompt-injection hygiene** (practical
  "Missing scope" §2). Accepted as a 3-line section added to
  AGENTS.md (tracked in Task 4), but not elevated to a blocker — the
  risk is equivalent across all agentic tools, not a multi-backend-
  specific regression.

---

## 15. Summary

**Approach A (subcommand) + Option X (single prompt, N adapters with
tool-name rewrite + verify_mcp_live)**: smallest code footprint,
preserves the default AMD-branded opencode behavior verbatim, reuses
the existing launcher plumbing, and keeps one `AGENTS.md` canonical.

PR 1 covers **opencode + claude + gemini** (cycle-2 cut). PR 2 adds
**codex** with trust-gate handling.

Cycle-2 resolution summary: 4 blockers resolved, 12 important findings
resolved, 3 nitpicks accepted as documented deferrals. Post-cycle-1
E2E findings F1 (HIGH — prompt-layer gate model-size sensitivity) and
F2 (MEDIUM — MCP handshake bootstrap exit-124) also folded in via
Plan Tasks 4.6 and 4.7. Ready for cycle-2 review.
