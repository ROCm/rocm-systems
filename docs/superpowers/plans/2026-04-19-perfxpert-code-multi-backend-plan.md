# perfxpert-code Multi-Backend Mode — Implementation Plan

Date: 2026-04-19 (cycle-2 + post-cycle-1 E2E addendum 2026-04-18;
cycle-2 **revision-2** 2026-04-18 after design-critic + practical
cycle-2 review returned REQUEST CHANGES minor)
Status: **Cycle-2 revision-2** — addresses cycle-1 static blockers,
the two live-E2E findings (F1 HIGH, F2 MEDIUM) from Haiku-4-5, AND
the cycle-2 review findings (3 blockers B-N1/B-N2/B-N3, 6 importants
I-N1..I-N6). See brainstorm §13 (E2E findings), §14 (cycle-1
changelog), and §15 (cycle-2 changelog). Commit baseline:
efc24bd56c.
Branch: `users/aelwazir/perfxpert-code-multi-backend` (to be created
from `users/aelwazir/perfxpert-phase8-opencode-fork`)
Worktree: `/home/aelwazir/work/ai-analysis-rocpd/.worktrees/perfxpert-phase8`
Brainstorm: `2026-04-19-perfxpert-code-multi-backend-brainstorm.md`
Convention: `superpowers:writing-plans`

## Summary

Add multi-backend dispatch to `perfxpert-code` so users can run their
agent of choice (Claude Code / Codex / Gemini) with perfxpert guardrails
(MCP server + AGENTS.md-equivalent prompt, with per-backend tool-name
rewrite and a post-install live probe) installed into each backend's
native config format. Default behavior (no subcommand = bundled opencode
TUI) is unchanged.

**Cycle-2 PR cut (revised from cycle-1):** PR 1 ships
**opencode + Claude + Gemini**; PR 2 adds **Codex** (with trust-gate
handling). Swap rationale: Gemini is the safest backend by construction
(`context.fileName` append never touches `GEMINI.md`); Codex requires
the trust-gate subplot and has an undocumented MCP tool-name wire
format that needs live-probe de-risking separately.

**Post-cycle-1 E2E addendum:** PR 1 ALSO ships (a) a server-side
PreToolUse tool-gate hook per backend (Task 4.6; mechanical
enforcement because prompt-layer gating is model-size sensitive —
Haiku-4-5 live E2E confirmed the regression) and (b) `perfxpert-mcp`
warmup during `install()` + MCP handshake retry in `verify_mcp_live()`
(Task 4.7; addresses observed first-run exit-124 bootstrap hang).
Escape hatches: `PERFXPERT_GATE_HOOK=0` and `PERFXPERT_MCP_WARMUP=0`.
Tunables: `PERFXPERT_MCP_WARMUP_TIMEOUT_S` (default 10s) and
`PERFXPERT_MCP_RETRY_BUDGET_S` (default 6s, exponential backoff
2/4/8 across 3 attempts). See brainstorm §13.

**Cycle-2 revision-2 (2026-04-18) headline changes:**

- **Task 0.5 (NEW, B-N1)**: doc-fetch + decision record for the
  Claude Code pre-tool-call hook surface. Blocks Task 4.6
  Claude-portion only. Decision stub lives at
  `docs/decisions/2026-04-19-claude-hook-surface.md`.
- **Task 4 split (I-N4)**: Task 4 (~500 LOC, 25 tests) broken into
  **4a** (`_prompt_adapter.py` + golden-file tests, ~150 LOC), **4b**
  (`ClaudeCodeAdapter` core: `check_available`, `plan`, `install`,
  `spawn`, `uninstall`, ~200 LOC), and **4c** (`verify_mcp_live` +
  telemetry probe, ~100 LOC). Tasks 4.6 (gate hook) and 4.7 (warmup
  + retry) keep their numbers.
- **Gate is purely event-based (B-N3)**: the `_FIRST_N_TURNS = 2`
  time-based rule is **dropped entirely**. Gate rejects any
  non-`perfxpert_*` tool call UNTIL `perfxpert_intent_classify` has
  been observed returning in the current session. No turn count,
  no clock.
- **opencode `{block, retryWith}` is fork-only (B-N2)**: the plan
  and brainstorm now call out that opencode's gate-hook return shape
  is a bundled-patch extension (phase-8 patch 0020), not upstream
  opencode API. If a user swaps the bundle for upstream opencode,
  the gate degrades to prompt-layer only.
- **`LiveCheckReport.gate_hook_installed` + `GateHookUnsupported`
  (I-N1)**: hook install raises `GateHookUnsupported` BEFORE MCP
  registration if the surface is unavailable, preventing partial
  state. `verify_mcp_live`'s gate-probe treats
  `gate_hook_installed=False` as documented-known-limit, not failure.
- **Multi-model acceptance probe (I-N2)**: PR-1 acceptance requires
  `verify_mcp_live` gate probe to pass against at least ONE small
  model per backend (haiku for Claude, flash for Gemini,
  opencode-default for opencode). Models tested are enumerated in
  the PR description. R-new-4 is now explicitly scoped to
  "verified on claude-haiku-4-5; other small models require
  independent re-verification at acceptance time."
- **Per-backend session-state table (I-N3)**: Task 4.6 gains an
  explicit table documenting where each backend records "gate
  lifted this session" + invalidation rules. NEW session always
  starts gate-engaged — even in the same cwd.
- **Warmup WAL/SHM cleanup test (I-N6)**: Task 4.7 adds an explicit
  test that warmup leaves no orphan `-wal`/`-shm` residue.

## Scope

- **In scope (PR 1).** Subcommand dispatch, `BackendAdapter` Protocol
  (full lifecycle from day one, no retroactive mutation),
  `ClaudeCodeAdapter`, `GeminiAdapter`, idempotent MCP registration via
  shell-out (primary) / print-for-human (fallback) / structured edit
  (last resort only for small dedicated files), project-scoped prompt
  staging with never-touch-tracked-files default, tool-name rewrite
  pipeline, `verify_mcp_live()` post-install probe (with handshake
  retry — F2), **server-side PreToolUse tool-gate hook per backend**
  (F1 — Task 4.6), **`perfxpert-mcp` warmup during `install()`**
  (F2 — Task 4.7), consent cache keyed on
  `(backend, cwd-hash, file-set-hash)`, version floor warning, unit
  tests, skip-on-CI integration tests, tool-name length lint,
  platform-guardrail (POSIX-only) explicit.
- **In scope (PR 2).** `CodexAdapter` + Codex project-trust detection +
  lazy `tomlkit` fallback.
- **Out of scope.** Changing the perfxpert MCP tool surface. Rebranding
  the third-party CLIs. Any change to opencode bundle behavior when the
  user invokes the default (no-subcommand) path. **Windows-native
  support** (POSIX only in first cut; tracking issue for Windows).

## Assumptions

- Users invoking `perfxpert-code claude` want the perfxpert guardrails
  active **in the current working directory only** (project-scoped).
  Global scope (`~/.claude.json` for MCP, `~/.claude/CLAUDE.md` for
  prompt) is opt-in via `PERFXPERT_<BACKEND>_SCOPE=user` — and when
  set, BOTH the MCP registration AND the prompt move together (never
  split scopes). (B2.)
- `perfxpert-mcp` is on the user's `PATH` (or a known install path);
  adapters do NOT absolute-path-resolve it — they register the literal
  string `perfxpert-mcp` in each backend's config so the backend
  resolves it via its own spawn PATH. `perfxpert-code doctor` verifies
  this is callable.
- Python 3.10+ for `typing.Protocol`, `match` statements acceptable.
- `tomlkit` is an OPTIONAL dependency behind the `[backends]` extra,
  lazy-imported inside the Codex adapter's fallback branch only. Users
  who never run `perfxpert-code codex` do not need it. (I7.)
- POSIX target (macOS + Linux + WSL). Windows-native is out of scope.
  (I12.)
- **Tool-priority gate is enforced mechanically AND is purely
  event-based** (B-N3). Prompt-layer gating is model-size sensitive
  (F1 — cycle-5 E2E on Haiku-4-5 confirmed). Primary enforcement is
  the server-side PreToolUse hook installed by Task 4.6 per backend.
  The hook rejects any non-`perfxpert_*` tool call UNTIL
  `perfxpert_intent_classify` has been observed returning in the
  current session. No "first N turns" clock — a legitimate `bash` on
  turn 2 after `intent_classify` on turn 1 passes through. A NEW
  session always starts with the gate engaged, even in the same
  cwd. Prompt-layer rejection-language (Task 4a `_prompt_adapter`)
  is secondary. `PERFXPERT_GATE_HOOK=0` env override disables the
  hook for reviewer debugging.
- **`perfxpert-mcp` handshake is occasionally racy on first run**
  (F2 — cycle-5 E2E observed exit-124 that retry cleared). Adapter
  `install()` warms `perfxpert-mcp` once to prime sqlite +
  tool-registry cache. `verify_mcp_live()` retries the MCP handshake
  up to 3 times with exponential backoff (2s, 4s, 8s) under a total
  retry budget (`PERFXPERT_MCP_RETRY_BUDGET_S`, default 6s — which
  permits attempts 1 and 2 on a typical first-run race; the third
  attempt is taken only if the budget has not been exhausted).
  Warmup is bounded by `PERFXPERT_MCP_WARMUP_TIMEOUT_S` (default
  10s, single attempt hard cap). `PERFXPERT_MCP_WARMUP=0` env
  override disables warmup if it becomes annoying on a platform.
- **opencode gate-hook relies on the bundled patched opencode**
  (phase-8 patch 0020). The `{ block: true, retryWith: <msg> }`
  return shape is a fork-only extension — upstream opencode does
  NOT expose it. If a user swaps `PERFXPERT_OPENCODE_PATH` to
  upstream opencode, the gate degrades to prompt-layer only. This
  is documented behavior, not a bug. (B-N2.)

---

## Task breakdown

Each task is a self-contained unit with explicit file paths, a test
strategy, and a commit message. Tasks are sized for review (≤ 200
diff LOC typical).

> **All file paths are relative to `/home/aelwazir/work/ai-analysis-rocpd/.worktrees/perfxpert-phase8/experimental/python/perfxpert/`** unless noted.

### Task 0 — Port `mcp-server.md` from phase 9 to phase 8 (B4)

**Why.** Cycle-1 plan cited
`experimental/python/perfxpert/docs/integration/mcp-server.md` as
existing, but that file lives on phase 9 (`users/aelwazir/perfxpert-phase9-*`),
not the phase-8 worktree. Cycle-2 requires it in phase 8 before any
later task touches it (B4).

**Files.**
- Copy: `docs/integration/mcp-server.md` from phase-9 worktree
  (`/home/aelwazir/work/ai-analysis-rocpd/.worktrees/perfxpert-phase9/experimental/python/perfxpert/docs/integration/mcp-server.md`)
  to this worktree at the same relative path.
- Create: `docs/integration/README.md` if not present (brief pointer to
  `mcp-server.md`). Check before creating.

**Tests.**
- `tests/test_integration/test_docs_links.py` (or grep equivalent): assert
  `docs/integration/mcp-server.md` exists; referenced by plan and
  brainstorm.

**Commit.** `docs(perfxpert): port mcp-server.md from phase 9 to phase 8 (B4)`

---

### Task 0.5 — Doc-fetch + decision record: Claude Code pre-tool-call hook surface (B-N1)

**Why.** Cycle-2 review B-N1: Task 4.6 inlined "research required /
verify via doc fetch" for the Claude Code hook surface. That is a
**prerequisite decision**, not an implementation detail. If Claude
Code has no suitable surface, Task 4.6's Claude-portion must
demote to prompt-layer only with explicit
acceptance-of-partial-mitigation. Extract this decision up-front
so implementation order is linear.

**Files.**
- Read (live web-fetch): Claude Code documentation under
  <https://code.claude.com/docs>. In particular:
  - plugin / hook pages (if any)
  - `settings.json` reference, `allowedTools` key
  - any `PreToolUse` / pre-tool-call surface documentation
- Modify: `docs/decisions/2026-04-19-claude-hook-surface.md`
  (stub created at cycle-2 revision-2 time — flip the `Status:`
  line from **PENDING** to the chosen surface, cite URLs +
  retrieval date, and fill in the "one-paragraph change set for
  Task 4.6" footer).
- If Claude exposes NO suitable surface, update:
  - Plan Task 4.6 Claude bullet → "prompt-layer enforcement +
    documented-gap"
  - PR 1 acceptance criterion 9 → add footnote
    "Claude gate-probe is prompt-layer only per decision record."
  - R-new-4 scope narrowing to reflect the Claude partial-mitigation.

**Tests.** None (pure docs/decision task).

**Blocks.** Task 4.6 Claude-portion ONLY. Task 4.6 opencode-portion
and Gemini-portion proceed in parallel. The rest of PR 1 is
independent.

**Commit.** `docs(perfxpert): decision record — Claude Code pre-tool-call hook surface (B-N1)`

---

### Task 1 — `BackendAdapter` Protocol (full lifecycle) + scaffold + error taxonomy (I2)

**Why.** Protocol must be complete from day one. Cycle-1 mutated it
retroactively in Task 6 (adding `dry_run: bool`) which breaks Tasks 4/5
on commit. Cycle-2: every method + kwarg is locked in Task 1.

**Files.**
- Read: `pyproject.toml` (check `[project]` dependencies).
- Modify: `pyproject.toml` — add optional extra `[project.optional-dependencies]`
  `backends = ["tomlkit>=0.12"]`. Document as Codex-only lazy import.
- Create: `perfxpert/cli/backends/__init__.py` — exports Protocol +
  dataclasses + exception taxonomy.
- Create: `perfxpert/cli/backends/_types.py`:
  - Frozen `@dataclass` definitions:
    - `InstallReport(backend: str, actions: tuple[str, ...], paths_written: tuple[Path, ...], duration_s: float)`
    - `UninstallReport(backend: str, actions: tuple[str, ...], paths_removed: tuple[Path, ...], skipped_due_to_drift: tuple[Path, ...])`
    - `Plan(backend: str, actions: tuple[str, ...], targets: tuple[Path, ...])`
    - `LiveCheckReport(backend: str, mcp_listed: bool, mcp_healthy: bool, observed_tool_names: tuple[str, ...], gate_hook_installed: bool | None, error: str | None)` — `gate_hook_installed=None` means "gate hook probe skipped" (e.g. `PERFXPERT_GATE_HOOK=0`); `False` means "surface unsupported / documented-known-limit"; `True` means "installed + effective". (I-N1.)
  - Exceptions (all subclass `PerfxpertBackendError`, a.k.a.
    `BackendAdapterError`):
    - `BackendNotFound` — binary missing
    - `VersionTooOld` — `claude --version` below `min_version`
    - `ConfigClobber` — existing config entry with different command
    - `ConsentDenied` — user declined; no write
    - `PartialInstall` — some steps succeeded, others failed
    - `SchemaUnknown` — config schema version not in `known_schema_versions`
    - `TrustRequired` — Codex project not trusted (B3)
    - `GateHookUnsupported` — backend exposes no suitable
      pre-tool-call surface (I-N1). Raised by Task 4.6 hook install
      BEFORE Task 4 Step 1 MCP registration so no partial state is
      left behind. `verify_mcp_live()` treats
      `gate_hook_installed=False` as documented-known-limit
      (warning-level), not failure.
- Create: `perfxpert/cli/backends/_protocol.py`:
  - `BackendAdapter` `Protocol` (`@typing.runtime_checkable`):
    - `name: str`
    - `binary_name: str`
    - `install_hint: str`
    - `min_version: str | None`
    - `known_schema_versions: tuple[str, ...]`
    - **`tool_name_template: str`** — e.g. `"{tool}"`, `"mcp__perfxpert__{tool}"`, `"mcp_perfxpert_{tool}"`. (B1.)
    - **`spawn_strategy: Literal["execvpe", "subprocess"]`** — I1.
    - Public lifecycle:
      - `check_available(self) -> tuple[bool, str]` — returns (found, path-or-reason)
      - `plan(self, cwd: Path, scope: Literal["project", "user"] = "project", dry_run: bool = True) -> Plan` — dry-run in Protocol from day one (I2)
      - `install(self, cwd: Path, scope: Literal["project", "user"] = "project", allow_agents_md_append: bool = False, dry_run: bool = False, quiet: bool = False) -> InstallReport`
      - `verify_mcp_live(self, cwd: Path, telemetry: bool = False) -> LiveCheckReport` — new in cycle-2 (B1 + I11)
      - `uninstall(self, cwd: Path, scope: Literal["project", "user"] = "project") -> UninstallReport`
      - `spawn(self, argv: list[str], env: dict[str, str], cwd: Path) -> int` — per `spawn_strategy`; returns exit code for subprocess, does-not-return for execvpe
    - PROTECTED (implementation detail; NOT on the public Protocol
      surface):
      - `_register_mcp`, `_stage_prompt`, `_build_exec_args`, `_build_exec_env` — adapter-internal only
- Create: `perfxpert/cli/_backend_log.py` — named `logging.Logger` per
  adapter (`perfxpert.backend.claude`, `perfxpert.backend.gemini`,
  `perfxpert.backend.codex`). Log level controlled by
  `PERFXPERT_LOG=debug|info` env var. (Nitpick N2 partial resolution.)

**Tests.**
- `tests/test_cli/test_backends/__init__.py`
- `tests/test_cli/test_backends/test_protocol.py`:
  - A no-op adapter implementing the full Protocol; assert
    `isinstance(adapter, BackendAdapter)` via `runtime_checkable`.
  - Assert Plan / InstallReport / UninstallReport / LiveCheckReport are
    frozen (attempting to set a field raises).
  - Exception taxonomy import smoke test.

**Commit.** `feat(perfxpert/cli): BackendAdapter Protocol (full lifecycle) + error taxonomy + tomlkit optional extra`

---

### Task 2 — Backend-subcommand routing + help banner + env-guard (B2 start, R6, practical §3/§10)

**Why.** Extend `route_subcommand()` to recognize `"backend"` without
touching the default path. Add help-banner section. Add the
`PERFXPERT_IN_AGENT_SESSION=<backend>` recursion guard (single env var,
four values — consolidated from cycle-1's per-backend env vars).

**Files.**
- Modify: `perfxpert/cli/opencode_launcher.py`
  - Add `_BACKEND_SUBCOMMANDS: frozenset[str] = {"claude", "codex", "gemini"}`
    — Codex included in dispatch from PR 1 but only *activated* (adapter
    present) from PR 2.
  - In `route_subcommand`, after perfxpert-dispatch check:
    ```python
    if first_positional in _BACKEND_SUBCOMMANDS:
        return ("backend", list(argv))
    ```
  - In `main`, after the `"perfxpert"` branch, add a `"backend"` branch
    that calls `_exec_backend(adapter_name, argv_out[1:])`.
  - **Short-circuit `PERFXPERT_OPENCODE_PATH` resolution** when
    `first_positional in _BACKEND_SUBCOMMANDS` (practical §10.3).
  - **Recursion guard**: if `os.environ.get("PERFXPERT_IN_AGENT_SESSION")`
    is set, refuse to launch (unless `--force` — parsed by Task 6) and
    print "already inside a perfxpert-<backend> session."
  - Update `_print_perfxpert_help` to add a NEW section header
    `Agent backends:` separate from `Perfxpert-owned subcommands:`,
    with one-line description per backend. (Practical §1.2.)
- Create: `perfxpert/cli/_backend_dispatch.py` — new module:
  - `_exec_backend(name: str, remaining_argv: list[str]) -> int`
  - Loads adapter by name from a registry
    (`{"claude": ClaudeCodeAdapter, "gemini": GeminiAdapter,
    "codex": CodexAdapter}`). Codex raises
    `NotImplementedError("Codex support ships in PR 2")` in PR 1.
  - **`--help` passthrough invariant (practical §1.2 + §10)**: if
    `remaining_argv[0] in ("--help", "-h")`, skip the installer — do
    NOT write files. Pass through to `backend --help`.
  - Parses dispatcher-owned flags (not forwarded to backend):
    `--dry-run`, `--quiet`, `--force`, `--allow-agents-md-append`.
  - Sets `PERFXPERT_IN_AGENT_SESSION=<backend>` in the child env
    before exec.

**Invariants.**
- Bare `perfxpert-code` still routes to `opencode_default`. Regression
  test: existing `test_main_default_invocation_stages_runtime_cfg_dir`
  (verify it exists; if not, write it in this task).
- `perfxpert-code claude --help` → forwarded, NO files written.

**Tests.**
- `tests/test_cli/test_launcher_subcommands.py`:
  - `test_route_claude_is_backend`
  - `test_route_codex_is_backend`
  - `test_route_gemini_is_backend`
  - `test_backends_shown_in_separate_help_section`
  - `test_help_passthrough_skips_installer` (invariant above)
  - `test_opencode_path_env_ignored_for_backend_subcommands`
  - `test_recursion_guard_refuses_when_PERFXPERT_IN_AGENT_SESSION_set`
- `tests/test_cli/test_backend_dispatch.py` — mock `_exec_backend` and
  assert `main(["claude", "--help"])` calls it with
  `("claude", ["--help"])`.

**Commit.** `feat(perfxpert/cli): route claude/codex/gemini subcommands + recursion guard + help passthrough`

---

### Task 3 — Consent cache keyed on (backend, cwd-hash, file-set-hash) (I10, practical §2.2)

**Why.** cycle-1 used a single global `backends.claude.consented=true`.
cycle-2: consent is keyed on `(backend, cwd-hash, file-set-hash)` — if
the target file set changes (e.g. user just started tracking
`CLAUDE.md`), re-prompt. XDG-compliant paths.

**Files.**
- Create: `perfxpert/cli/backends/_consent.py`:
  - XDG-compliant cache location: `$XDG_CONFIG_HOME/perfxpert/consent.yaml`
    (default `~/.config/perfxpert/consent.yaml`).
  - `_cwd_hash(cwd: Path) -> str` — SHA-8 of absolute-resolved cwd.
  - `_file_set_hash(backend: str, cwd: Path) -> str` — SHA-8 over a
    sorted tuple of `(path, exists, git_tracked)` for each file the
    adapter plans to touch.
  - `has_consent(backend: str, cwd: Path, file_set_hash: str) -> bool`
  - `grant_consent(backend: str, cwd: Path, file_set_hash: str) -> None`
  - `revoke_consent(backend: str, cwd: Path) -> None`
  - `prompt_consent_interactive(backend: str, cwd: Path, plan: Plan) -> bool`
    — displays plan line-by-line via `rich.table` or plain stderr.
  - `PERFXPERT_ASSUME_CONSENT=1` → auto-grant.
  - TTY check: if not a TTY and env not set → refuse + print guidance.

**Tests.**
- `tests/test_cli/test_backends/test_consent.py`:
  - `test_consent_round_trip_with_cwd_and_file_set_hash`
  - `test_consent_invalidated_when_file_set_changes`
  - `test_consent_invalidated_when_cwd_changes`
  - `test_env_override_skips_prompt`
  - `test_non_tty_refuses_without_env`
  - `test_xdg_config_home_honored` — `monkeypatch.setenv("XDG_CONFIG_HOME", tmp_path)`
  - `test_home_isolation` — set `HOME=tmp_path`, assert no leakage to
    developer's real home.

**Commit.** `feat(perfxpert/cli): consent cache (backend, cwd, file-set) keyed, XDG-compliant`

---

### Task 4 — `ClaudeCodeAdapter` (split into 4a/4b/4c per I-N4)

**Why.** First real backend. Exercises every cycle-2 invariant.
Cycle-2 revision-2 (I-N4): Task 4 was ~500 LOC + 25 tests in one
commit — too large to review. Split into three independently
reviewable commits: **4a** (`_prompt_adapter` shared helpers +
golden-file tests), **4b** (`ClaudeCodeAdapter` core lifecycle
methods), **4c** (`verify_mcp_live` + telemetry probe). Tasks 4.6
(gate hook) and 4.7 (warmup + retry) keep their existing numbers
— they operate on the same adapter surface and ship in PR 1.

**Total footprint across 4a/4b/4c: ~450 LOC + 25 tests** (matches
cycle-1 estimate; split is a reviewability improvement, not an
LOC reduction).

---

### Task 4a — `_prompt_adapter.py` shared helpers + golden-file tests (~150 LOC) — I-N4

**Why.** Cross-backend render/marker/git-tracking/atomic-write
helpers. Isolated so every adapter can depend on a stable,
individually-reviewed API.

**Files.**
- Create: `perfxpert/cli/backends/_prompt_adapter.py` — shared helpers:
  - `render_prompt(src: Path, *, tool_name_template: str, strip_backends: tuple[str, ...]) -> str`:
    - Load `AGENTS.md`.
    - Substitute every `perfxpert_<X>` (where X is a registered tool
      name from the perfxpert MCP manifest) with the rendered
      per-backend form using `tool_name_template`. (B1.)
    - Strip `<!--backend:NAME-->...<!--/backend:NAME-->` blocks not in
      the target backend.
    - **Emit rejection-language gate stanza (F1 — brainstorm §13).**
      The staged prompt MUST include a variant of the tool-priority
      gate framed as a **consequence**, not advisory, e.g.
      "Your response WILL BE REJECTED if you call bash/read/edit
      before `<rendered>_intent_classify` returns." Prompt-layer gate
      is the *secondary* enforcement; the primary is the server-side
      hook from Task 4.6. A `reject_language: bool = True` kwarg
      toggles the stanza (on by default).
  - `emit_marker_block(content_hash: str) -> tuple[str, str]`:
    - Returns `(begin, end)` sentinels with versioning + cache hash:
      `<!-- BEGIN perfxpert-managed v1 cache=<sha8> -->` /
      `<!-- END perfxpert-managed v1 -->`. (I6.)
  - `is_git_tracked(path: Path, cwd: Path) -> bool`:
    - `subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=cwd, capture_output=True)`
      — exit-0 = tracked. (I3.)
  - `atomic_write(path: Path, content: str) -> None` — tmp-write-rename
    + `.bak` retention.
  - `stage_cache_file(src: Path, dst: Path, rendered: str) -> str` —
    writes rendered content to dst, returns SHA-8 cache hash.
  - `retry_mcp_handshake(fn, *, attempts=3, backoff_s: tuple[float, ...] = (2.0, 4.0, 8.0), budget_s: float = 6.0) -> T` —
    shared retry helper used by every adapter's `verify_mcp_live()`
    (Task 4c / 5 / 10). Respects `PERFXPERT_MCP_RETRY_BUDGET_S`
    (default 6s) — attempts stop as soon as total elapsed time
    exceeds the budget, even if `attempts` remain. Backoff is
    exponential (2 / 4 / 8) per I-N5 (cycle-1 had flat 2 / 2 / 2).

**Tests.**
- `tests/test_cli/test_backends/test_prompt_adapter.py`:
  - `test_render_substitutes_tool_names_with_template` (B1 golden-file)
  - `test_render_strips_non_target_backend_blocks`
  - `test_render_includes_rejection_language_stanza_when_reject_language_true`
  - `test_render_omits_rejection_stanza_when_reject_language_false`
  - `test_marker_block_has_version_and_cache_hash` (I6)
  - `test_is_git_tracked_true_for_tracked_file`
  - `test_is_git_tracked_false_for_untracked`
  - `test_atomic_write_leaves_bak_on_rewrite`
  - `test_retry_helper_honors_exponential_backoff_2_4_8` (I-N5)
  - `test_retry_helper_exits_early_when_budget_exhausted` (I-N5)
- Golden-file fixtures under
  `tests/test_cli/test_backends/fixtures/prompts/`:
  `claude_rendered.md`, `gemini_rendered.md`, `codex_rendered.md`
  (Codex golden uses the initial-guess template; regenerated by
  Task 10 after live-probe).

**Commit.** `feat(perfxpert/cli): backends/_prompt_adapter.py — render + marker + git-tracking + retry helpers (Task 4a)`

---

### Task 4b — `ClaudeCodeAdapter` core methods (`check_available`, `plan`, `install`, `spawn`, `uninstall`) (~200 LOC) — I-N4

**Why.** The adapter proper, minus live-check. `install()` Step 4/4
calls into `verify_mcp_live` which lands in Task 4c. Until 4c ships,
Task 4b's `install()` is behind a feature flag (`PERFXPERT_SKIP_LIVE_CHECK=1`
for the 4b-only intermediate state) or — preferred — Task 4b lands
as a single commit immediately followed by Task 4c, so `install()`'s
final step is live from the first release.

**Files.**
- Create: `perfxpert/cli/backends/claude.py`:
  - `ClaudeCodeAdapter(BackendAdapter)`:
    - `name = "claude"`, `binary_name = "claude"`, `min_version = "2.1.59"`.
    - `tool_name_template = "mcp__perfxpert__{tool}"`. (B1.)
    - `known_schema_versions = ("1.x",)`.
    - `spawn_strategy = "execvpe"`. (I1.)
    - `install_hint = "Install via https://code.claude.com/docs/en/install"`.
  - Methods:
    - `check_available()`: `shutil.which("claude")` +
      `claude --version` parse, compare to `min_version`. Returns
      `VersionTooOld` if below.
    - `plan(cwd, scope="project", dry_run=True)`:
      - Compute file-set (`.mcp.json`, `.claude/CLAUDE.md` or
        `CLAUDE.md`, `.perfxpert/AGENTS.md`).
      - For each file: git-tracked? exists? action needed?
      - Return `Plan` with actions (e.g.
        `["Register perfxpert MCP in .mcp.json (project scope)",
          "Stage prompt at .perfxpert/AGENTS.md (new)",
          "Write pointer at .claude/CLAUDE.md (new, untracked)"]`).
    - `install(cwd, scope="project", allow_agents_md_append=False, dry_run=False, quiet=False)`:
      - **Consent gate** — `has_consent(...)` with file-set hash; if
        missing, `prompt_consent_interactive`. (I10.)
      - **Step 1/4: Register MCP (B2 + I4 + I5).**
        - Primary: `subprocess.run(["claude", "mcp", "add", "perfxpert", "--scope", scope, "--", "perfxpert-mcp"], timeout=15)`.
        - Idempotent: if `subprocess.run(["claude", "mcp", "get", "perfxpert"], timeout=10).returncode == 0`, skip.
        - On non-zero exit or timeout: print the exact command for the
          user to run manually — do NOT whole-file-rewrite
          `~/.claude.json`. (I4.)
        - Fallback structured edit: ONLY for project scope (small
          dedicated `.mcp.json`), NOT for user scope (`~/.claude.json`
          is multi-MB).
        - Append `.mcp.json` to `.gitignore` if `.gitignore` exists
          and does not already match. (B2 — `.gitignore` hint.)
        - stderr: `[1/4] Registering perfxpert MCP in .mcp.json ... ok`.
      - **Step 2/4: Stage prompt cache.**
        - `render_prompt(...)` with `tool_name_template`.
        - `stage_cache_file` → writes `<cwd>/.perfxpert/AGENTS.md`.
        - stderr: `[2/4] Staging rendered prompt ... ok`.
      - **Step 3/4: Write pointer (I3 — never-touch-tracked-files).**
        - Check `CLAUDE.md` tracking: `is_git_tracked(Path("CLAUDE.md"), cwd)`.
        - If tracked OR nonexistent → write `.claude/CLAUDE.md` with
          `@.perfxpert/AGENTS.md` (no marker block; dedicated file).
        - If untracked and `allow_agents_md_append=True` → append
          marker block to `CLAUDE.md`. (Explicit opt-in.)
        - If untracked and `allow_agents_md_append=False` → refuse
          with guidance ("`CLAUDE.md` exists and is not tracked. Pass
          `--allow-agents-md-append` to merge, or remove the file.").
        - stderr: `[3/4] Writing pointer at .claude/CLAUDE.md ... ok`.
      - **Step 4/4: Verify.** Call `verify_mcp_live(cwd)`. If the live
        check fails, raise `PartialInstall` with the observed tool
        names. (B1 + I11.)
        - stderr: `[4/4] Verifying perfxpert MCP is live ... ok`.
        - **Safe-intermediate-state flag (Practical-top — cycle-3
          followup).** Task 4b's initial commit sets
          `PERFXPERT_SKIP_LIVE_CHECK=1` as the default when running
          inside Task 4b's own test suite (not user-facing). Task 4c's
          commit removes the default. This makes the 4b→4c intermediate
          state (4b merged, 4c still in review) safe: `install()`
          Step 4/4 becomes a no-op until 4c lands. User-facing default
          for the installed package is `PERFXPERT_SKIP_LIVE_CHECK=0`
          (live-check runs).
      - `grant_consent(...)` on success.
      - Return `InstallReport`.
    - `verify_mcp_live(cwd, telemetry=False)` → **implemented in Task 4c**.
    - `uninstall(cwd, scope="project")`:
      - Remove marker block from `.claude/CLAUDE.md` (or `CLAUDE.md` if
        appended). **Refuse if block content drifted**
        (content-hash mismatch — user edited inside our block); list
        in `UninstallReport.skipped_due_to_drift`.
      - Remove `perfxpert` MCP entry: `claude mcp remove perfxpert`
        (shell-out).
      - Remove `<cwd>/.perfxpert/AGENTS.md`. Remove `.perfxpert/` if
        empty.
    - `spawn(argv, env, cwd)`:
      - `os.chdir(cwd)`; `os.execvpe("claude", ["claude"] + argv, env)`.
        Does NOT return on success. (I1.)

**Tests.**
- `tests/test_cli/test_backends/test_claude_adapter.py`:
  - `test_check_available_missing_binary`
  - `test_check_available_version_too_old_raises`
  - `test_tool_name_template_is_mcp_double_underscore`
  - `test_plan_lists_every_file_touched`
  - `test_install_shells_out_to_claude_mcp_add`
  - `test_install_timeout_prints_command_and_falls_back`
  - `test_install_fallback_structured_edit_only_for_project_scope`
  - `test_install_refuses_whole_file_edit_of_home_claude_json` (I4)
  - `test_install_preserves_existing_mcp_servers_in_dot_mcp_json`
  - `test_install_idempotent_on_second_run`
  - `test_install_adds_dot_mcp_json_to_gitignore_if_present`
  - `test_install_refuses_append_without_flag_when_claude_md_untracked` (I3)
  - `test_install_writes_dot_claude_claude_md_when_claude_md_tracked` (I3)
  - `test_install_appends_marker_with_allow_flag`
  - `test_install_marker_is_versioned_and_hashed` (I6)
  - `test_install_emits_per_step_progress` (I5)
  - `test_install_quiet_suppresses_progress`
  - `test_uninstall_refuses_on_block_drift`
  - `test_uninstall_removes_mcp_entry_preserving_others`
  - `test_spawn_uses_execvpe_not_subprocess_run` (I1 — monkeypatch
    `os.execvpe`, assert called, assert `subprocess.run` NOT called for
    the exec itself)

**Commit.** `feat(perfxpert/cli): ClaudeCodeAdapter core — check_available, plan, install, spawn, uninstall (Task 4b)`

---

### Task 4c — `verify_mcp_live` + telemetry probe + gate probe (~100 LOC) — I-N4

**Why.** Live-check is the load-bearing invariant (B1 + I11 + F1);
isolating it from Task 4b keeps each commit's blast radius scoped.
Shared helper lives in `perfxpert/cli/backends/_prompt_adapter.py`
(Task 4a — `retry_mcp_handshake`); the per-adapter `verify_mcp_live`
wires the helper to the backend-specific `mcp list` command.

**Files.**
- Modify: `perfxpert/cli/backends/claude.py` — implement
  `verify_mcp_live(cwd, telemetry=False) -> LiveCheckReport`:
  - `claude mcp list --json` (timeout 15s); parse output; assert
    `perfxpert` listed + healthy. Wrap the call via
    `retry_mcp_handshake(...)` (Task 4a) for the F2 bootstrap race.
  - **MCP handshake retry (F2 — brainstorm §13).** 3 attempts,
    **exponential backoff 2s / 4s / 8s** (I-N5 fix — cycle-1 had
    flat 2 / 2 / 2). Total retry budget governed by
    `PERFXPERT_MCP_RETRY_BUDGET_S` (default 6s). On budget
    exhaustion raise `PartialInstall` with the observed tool names
    and — if Task 4.7 warmup failed earlier — include the warmup
    failure reason in the error message so the user sees ONE
    coherent diagnostic (nitpick N-6 / resolved-in-place).
  - If `telemetry=True` (set via `PERFXPERT_TELEMETRY=1` in env):
    additionally send a structured probe prompt
    ("reply with only 'ack' after calling
    `mcp__perfxpert__intent_classify`") via `claude exec …` with
    a short timeout, assert perfxpert-mcp saw `intent_classify`
    before any other tool (reads telemetry log from
    `$XDG_CACHE_HOME/perfxpert/mcp-telemetry.log`). (I11.)
  - **Gate probe (F1 — brainstorm §13).** When the server-side
    hook from Task 4.6 is installed (`gate_hook_installed=True`),
    `verify_mcp_live` additionally runs a canned
    non-perfxpert-first query (e.g. "list files in this repo")
    against the backend and asserts the hook rejected the first
    non-`perfxpert_*` tool call. Fails loudly if the gate is
    installed but not effective.
    - **Multi-model coverage (I-N2).** The gate probe MUST be
      exercised against at least ONE small model per backend in
      PR 1 acceptance (haiku for Claude, flash for Gemini,
      opencode-default for opencode). The specific model used is
      captured in the PR description under "Models tested for
      gate probe". R-new-4 scope: verified on claude-haiku-4-5;
      other small models require independent re-verification at
      acceptance time.
    - **`GateHookUnsupported` handling (I-N1).** If the hook
      install step from Task 4.6 raised `GateHookUnsupported`
      (e.g. Claude Code exposes no suitable surface per Task 0.5
      decision), `gate_hook_installed=False` and the gate probe
      is SKIPPED; `verify_mcp_live` returns successfully with a
      warning-level log ("gate-hook unsupported on this backend;
      prompt-layer enforcement only"). This is a
      documented-known-limit, NOT a failure.
  - Return `LiveCheckReport(..., gate_hook_installed=<bool|None>)`.

**Tests.**
- `tests/test_cli/test_backends/test_claude_verify.py`:
  - `test_verify_mcp_live_detects_unhealthy_entry` (B1)
  - `test_verify_mcp_live_telemetry_probe` (I11)
  - `test_verify_mcp_live_retries_handshake_3x_exponential_2_4_8` (F2 + I-N5)
  - `test_verify_mcp_live_exits_early_when_retry_budget_exhausted` (I-N5)
  - `test_verify_mcp_live_forwards_warmup_failure_reason_into_error` (N-6)
  - `test_verify_mcp_live_skips_gate_probe_when_gate_hook_installed_false` (I-N1)
  - `test_verify_mcp_live_returns_gate_hook_installed_field` (I-N1)

**Commit.** `feat(perfxpert/cli): ClaudeCodeAdapter verify_mcp_live + gate + telemetry probes (Task 4c)`

---

### Task 4.6 — Server-side tool-gate hook per backend (F1; brainstorm §13)

**Why.** Cycle-5 live E2E with Haiku-4-5 proved the prompt-layer
tool-priority gate is insufficient for smaller LLMs: Haiku fired
`read`/`glob` at positions 2–4 before `workflow_next_step` (H1), and
skipped perfxpert tools entirely when profiling was asked for (H2).
Prompt text alone is advisory to smaller models. The fix is
**mechanical, server-side enforcement** — reject any non-`perfxpert_*`
tool call **UNTIL `perfxpert_intent_classify` has been observed
returning in the current session** (B-N3 — the gate is purely
event-based; there is no "first N turns" clock).

This is a **new sub-task of Task 4** because it operates on the same
`ClaudeCodeAdapter` and `GeminiAdapter` surfaces; the hook is shipped
as part of each adapter's `install()`.

**Gate semantics (B-N3 — event-based only).**

- Initial state on a NEW session: gate ENGAGED (reject any
  non-`perfxpert_*` tool call).
- Transition: once `perfxpert_intent_classify` returns in this
  session, gate is LIFTED for the remainder of the session.
- No turn counter. A legitimate `bash` on turn 2 AFTER a successful
  `intent_classify` on turn 1 passes through. A `bash` on turn 2
  BEFORE `intent_classify` is rejected — regardless of turn number.
- Cycle-1 had a `_FIRST_N_TURNS = 2` time-based rule; **dropped
  entirely** in cycle-2 revision-2 because it re-introduced the
  cycle-4 false-refusal class (any legitimate `bash` on turn ≤ 2
  after `intent_classify` got rejected).

**Per-backend research + implementation.**

- **opencode** — starting point exists in phase 8 patch
  `0020-perfxpert-tool-gate.patch` (and its README). The patch uses
  `plugin.trigger("tool.execute.before", ...)` with a
  `{ block: true, retryWith: <msg> }` extension. Port the pattern
  into `perfxpert/cli/backends/_gate_hooks/opencode.py` as shared
  content (staged at install time).

  **Callout (B-N2 — bundled-only extension).** The `{ block: true,
  retryWith: <msg> }` return shape is a **fork-only extension**
  provided by phase-8 patch `0020-perfxpert-tool-gate.patch`. It is
  NOT part of upstream opencode's `tool.execute.before` API. If a
  user sets `PERFXPERT_OPENCODE_PATH` to an unpatched upstream
  opencode binary, the hook falls back to prompt-layer enforcement
  only (the rejection-language stanza from Task 4a's
  `_prompt_adapter`). This is documented behavior, not a bug;
  `verify_mcp_live` records `gate_hook_installed=False` and emits a
  warning-level log.
- **Claude Code** — **BLOCKED on Task 0.5 decision record** (B-N1).
  Task 0.5 web-fetches the Claude Code documentation and picks
  among: (a) native PreToolUse hook
  (`.claude/hooks/pre-tool-use.<ext>`), (b) `allowedTools` settings
  restriction in `.claude/settings.json`, or (c) neither — in which
  case Task 4.6 Claude-portion demotes to prompt-layer enforcement
  only and `install()` records `gate_hook_installed=False` via the
  `GateHookUnsupported` path (I-N1). Implementation lands in
  `perfxpert/cli/backends/_gate_hooks/claude.py` AFTER Task 0.5
  flips `docs/decisions/2026-04-19-claude-hook-surface.md` from
  PENDING to a chosen surface.

  **Subagent precondition (Practical-3 — cycle-3 followup).** Refuse
  to execute Task 4.6 Claude-portion if
  `docs/decisions/2026-04-19-claude-hook-surface.md` still says
  `Status: PENDING`. The implementing subagent MUST grep the decision
  record for `Status: DECIDED` before proceeding.
- **Gemini** — use per-tool policies or `allowedTools` list in
  `.gemini/settings.json` to block non-perfxpert tools initially.
  Settings-only (no hook surface needed at the CLI layer). Lift
  logic writes the cleared `allowedTools` list back to settings
  once `intent_classify` is observed via Gemini's telemetry.
- **Codex** — **deferred to PR 2**. Codex sandbox-level hooks land
  alongside the `CodexAdapter` in Task 10.

**Files.**
- Create: `perfxpert/cli/backends/_gate_hooks/__init__.py` — shared
  rejection message template (referencing the rendered
  `_intent_classify` tool name per backend) and the
  `GATE_STATE_LIFTED_SENTINEL` value. **No `_FIRST_N_TURNS`
  constant** (B-N3 drop).
- Create: `perfxpert/cli/backends/_gate_hooks/opencode.py` — port of
  patch `0020-perfxpert-tool-gate.patch` shape. Include a module
  docstring spelling out the fork-only dependency (B-N2).
- Create: `perfxpert/cli/backends/_gate_hooks/claude.py` — body
  deferred until Task 0.5 decision lands; until then module is a
  placeholder raising `GateHookUnsupported("blocked on Task 0.5")`
  if called.
- Create: `perfxpert/cli/backends/_gate_hooks/gemini.py` —
  `allowedTools` restriction + event-based lift logic.
- Modify: `perfxpert/cli/backends/claude.py` `install(...)` — new
  Step between prompt-staging and verify: "Install PreToolUse gate
  hook at `<chosen surface>`". The chosen surface is read from the
  Task 0.5 decision record. Hook install step raises
  `GateHookUnsupported` BEFORE the MCP registration step runs
  (I-N1 — prevents partial state where MCP is registered but the
  gate is silently absent).
- Modify: `perfxpert/cli/backends/gemini.py` `install(...)` — write
  initial `allowedTools: ["mcp_perfxpert_*"]` to
  `.gemini/settings.json` (gate-ENFORCE) + emit the lift-logic
  companion config.
- Modify: `perfxpert/cli/backends/_prompt_adapter.py` — confirm the
  `reject_language=True` stanza (from Task 4a) is coherent with the
  hook's `retryWith` message (shared text source).

**Session-state location per backend (I-N3).**

The gate rule is simple ("lift when `intent_classify` observed") but
the *location* where the hook records this lifted state is
per-backend:

| Backend | Session state location | Invalidation |
|---|---|---|
| opencode | phase-8 patch 0020 in-memory session object (held by the patched opencode process) | session end (process exit) |
| Claude Code | TBD per Task 0.5 decision record — either a sidecar file under `.claude/.perfxpert-gate-state.<session-id>.json` (native-hook surface) OR cleared `allowedTools` key in `.claude/settings.json` (settings surface). Documented in the flipped decision record's "change set" footer. | session end |
| Gemini | `~/.gemini/runtime/perfxpert-gate-<session-id>.json` (or equivalent, scoped to Gemini's session-id if available; falls back to per-process PID sidecar) | TTL 1h / manual cleanup by `perfxpert-code uninstall gemini` |
| Codex | (deferred PR 2) | — |

**A NEW session always starts with the gate engaged — even in the
same cwd.** The state file from a prior session MUST NOT be reused.
Hook install writes a fresh state marker at install time (or the
hook itself lazy-initializes on first tool call when no state file
exists for the current session id).

**Invariants.**
- Gate is **event-based**, not time-based (B-N3). Lift trigger =
  `perfxpert_intent_classify` returned in THIS session.
- Gate must be LIFTABLE — once `intent_classify` is called, the hook
  permits any tool. Failure to lift = false §5.8-style refusal
  (cycle-4 regression class).
- New session always starts with the gate engaged. Recursion /
  resumed sessions: a resumed session with the same session-id MAY
  inherit the prior lift; a brand-new session-id always re-engages.
- `PERFXPERT_GATE_HOOK=0` env override disables the hook entirely
  (for reviewer debugging).
- Hook install raises `GateHookUnsupported` BEFORE MCP registration
  when the chosen surface is unavailable (I-N1).

**Tests.**
- `tests/test_cli/test_backends/test_gate_hooks.py`:
  - `test_opencode_gate_rejects_bash_before_intent_classify`
  - `test_opencode_gate_lifts_after_intent_classify_returns`
  - `test_opencode_gate_permits_bash_on_turn_2_after_intent_classify_on_turn_1` (B-N3 — directly asserts the false-refusal class is gone)
  - `test_opencode_documents_fork_only_dependency_in_docstring` (B-N2)
  - `test_claude_gate_rejects_non_perfxpert_tools_until_intent_classify` (requires Task 0.5 resolution)
  - `test_claude_gate_raises_GateHookUnsupported_when_surface_unavailable` (I-N1)
  - `test_gemini_gate_allowedtools_restricts_until_intent_classify_observed`
  - `test_gemini_gate_lifts_allowedtools_after_intent_classify_returns`
  - `test_new_session_always_starts_gate_engaged_even_same_cwd` (I-N3)
  - `test_env_override_PERFXPERT_GATE_HOOK_0_disables_hook`
  - `test_hook_install_raises_before_mcp_registration_on_unsupported_surface` (I-N1 — partial-state protection)

**Acceptance gate.** After install, a canned query that asks the
model to "list files" must receive a rejection + retry message on
the first tool call; and a canned query that asks the model to
"analyze this trace" must go through `perfxpert_intent_classify`
first. Both assertions surfaced via telemetry log (I11). The probe
runs against at least ONE small model per backend (I-N2): haiku for
Claude, flash for Gemini, opencode-default for opencode. Models
actually tested are listed in the PR description.

**Commit.** `feat(perfxpert/cli): server-side event-based gate hook per backend (F1, B-N3)`

---

### Task 4.7 — Adapter `install()` warms `perfxpert-mcp` (F2; brainstorm §13)

**Why.** Cycle-5 E2E reviewer reported a reproducible exit-124 on the
**first** `perfxpert-code run <query>` invocation after a stale
server state. Single-line log; no traceback; retrying 1–2× cleared
it. Root cause likely: MCP handshake timeout OR sqlite init race
during `perfxpert-mcp` spawn. This is opencode-specific today but
**every** backend will hit the same race on first run because they
all spawn `perfxpert-mcp` on stdio similarly.

Fix is three layered mitigations: warmup during `install()`, retry
in `verify_mcp_live()` (Task 4c) via the shared helper
`retry_mcp_handshake` (Task 4a), and a user-facing env hatch.

**Files.**
- Create: `perfxpert/cli/backends/_mcp_warmup.py`:
  - `warmup_perfxpert_mcp(timeout: float | None = None) -> WarmupReport`:
    - `timeout` defaults to
      `float(os.environ.get("PERFXPERT_MCP_WARMUP_TIMEOUT_S", "10"))`
      — single-warmup hard cap (I-N5). 10s default picked to survive
      CI/WSL cold-start latencies; 6s was deemed tight in cycle-2
      review.
    - Spawns `perfxpert-mcp` as a subprocess, sends an MCP `initialize`
      handshake, waits for the tool-registry reply, closes the
      subprocess cleanly.
    - Primes: sqlite DB open, tool registry cache, any per-session
      expensive imports.
    - **Cleanup invariant (I-N6).** On return (success OR failure),
      sqlite WAL/SHM state MUST be flushed: no orphan `<db>-wal` or
      `<db>-shm` files left behind, no leaked file descriptors. The
      warmup subprocess is torn down via a clean `close()` handshake
      followed by `wait()` (NOT a kill -9) so sqlite checkpoints
      before exit.
    - Returns `WarmupReport(success: bool, duration_s: float, error: str | None)`.
    - Idempotent — safe to call multiple times.
  - `PERFXPERT_MCP_WARMUP` env gate:
    - `=1` (default when unset): warmup runs.
    - `=0`: warmup skipped (escape hatch if it becomes annoying on a
      platform).
  - `PERFXPERT_MCP_WARMUP_TIMEOUT_S` env tunable (default `10`, I-N5):
    overrides the single-warmup hard cap. Useful on very slow CI
    runners or cold-container environments.
  - `PERFXPERT_MCP_RETRY_BUDGET_S` env tunable (default `6`, I-N5):
    overrides the total retry budget used by `retry_mcp_handshake`
    (Task 4a). Exponential backoff is 2 / 4 / 8 across up to 3
    attempts; attempts stop as soon as the budget is exhausted (so
    default 6s typically covers attempts 1 and 2; attempt 3 only if
    the first two returned quickly).
- Modify: `perfxpert/cli/backends/claude.py` `install(...)` — add
  **Step 0.5/4**: "Warm perfxpert-mcp (sqlite init, tool-registry
  cache) ... ok" via `warmup_perfxpert_mcp()`. Log the duration for
  observability; on warmup failure print guidance but do NOT abort
  install (warmup is an optimization, not a correctness gate).
  **Warmup failure reason (N-6)** is captured on the adapter
  instance and forwarded into `verify_mcp_live()`'s error message if
  the subsequent handshake also fails, so the user sees ONE coherent
  diagnostic rather than two separate log lines.
- Modify: `perfxpert/cli/backends/gemini.py` `install(...)` — same
  warmup step + same failure-reason forwarding.
- Modify: `perfxpert/cli/backends/codex.py` `install(...)` (PR 2) —
  same warmup step.
- Note: the shared `retry_mcp_handshake` helper (with **exponential
  backoff 2 / 4 / 8** and `PERFXPERT_MCP_RETRY_BUDGET_S`-governed
  early-exit) lives in `perfxpert/cli/backends/_prompt_adapter.py`
  and is introduced in Task 4a; each adapter's `verify_mcp_live`
  (Task 4c / 5 / 10) calls it.
- Document: `docs/integration/mcp-server.md` (ported in Task 0) gains
  a "Known issue: first-run exit-124" subsection with the
  `PERFXPERT_MCP_WARMUP=1` workaround note AND the two tunables
  `PERFXPERT_MCP_WARMUP_TIMEOUT_S` + `PERFXPERT_MCP_RETRY_BUDGET_S`
  (I-N5).

**Tests.**
- `tests/test_cli/test_backends/test_mcp_warmup.py`:
  - `test_warmup_spawns_and_closes_perfxpert_mcp_cleanly` (I-N6 —
    asserts no orphan fds, no `-wal` / `-shm` residue after warmup
    returns)
  - `test_warmup_reports_duration`
  - `test_warmup_idempotent_on_second_call`
  - `test_warmup_skipped_when_env_PERFXPERT_MCP_WARMUP_is_0`
  - `test_warmup_honors_PERFXPERT_MCP_WARMUP_TIMEOUT_S_override` (I-N5)
  - `test_install_does_not_abort_on_warmup_failure` (graceful
    degradation)
  - `test_install_captures_warmup_failure_reason_for_forwarding` (N-6)
  - `test_verify_mcp_live_retries_3x_with_exponential_backoff_2_4_8` (F2 + I-N5)
  - `test_verify_mcp_live_honors_PERFXPERT_MCP_RETRY_BUDGET_S` (I-N5)
  - `test_verify_mcp_live_succeeds_on_second_retry` (simulated race)
  - `test_verify_mcp_live_error_message_includes_prior_warmup_failure_reason` (N-6)

**Commit.** `feat(perfxpert/cli): warm perfxpert-mcp in install() + exponential-retry + env tunables (F2, I-N5, I-N6)`

---

### Task 5 — `GeminiAdapter` (PR 1, promoted from PR 2 per I9)

**Why.** Cycle-2 promotes Gemini from PR 2 → PR 1. Rationale: Gemini is
the safest backend by construction (`context.fileName` list-append
never touches `GEMINI.md`).

**Files.**
- Create: `perfxpert/cli/backends/gemini.py`:
  - `GeminiAdapter(BackendAdapter)`:
    - `name = "gemini"`, `binary_name = "gemini"`, `min_version = "0.2.0"`.
    - `tool_name_template = "mcp_perfxpert_{tool}"`. (B1.)
    - `spawn_strategy = "execvpe"`. (I1.)
  - Methods:
    - `check_available()`: similar to claude.
    - `plan(cwd, scope="project", dry_run=True)`: lists
      `.gemini/settings.json`, `.perfxpert/AGENTS.md`. GEMINI.md is
      NEVER in the file-set.
    - `install(...)`:
      - Step 0/4: **Warm `perfxpert-mcp`** via
        `warmup_perfxpert_mcp()` (F2 — see Task 4.7). Honors
        `PERFXPERT_MCP_WARMUP=0` override. Warmup failure is
        non-fatal.
      - Step 1/4: Register MCP by editing `<cwd>/.gemini/settings.json`
        (atomic write, preserve all existing keys). No shell-out for
        Gemini — config-file only. File is small + user-owned; I4
        concern does not apply.
      - Step 2/4: Stage rendered prompt at `.perfxpert/AGENTS.md`
        (rejection-language stanza on; see Task 4a
        `_prompt_adapter.render_prompt(reject_language=True)`).
      - Step 3/4: **List-merge `context.fileName`** (practical §3.3):
        read existing list; append `.perfxpert/AGENTS.md` only if not
        already present; preserve user order + other entries. Never
        overwrite.
      - Step 3.5/4: **Install server-side gate hook** (F1 — see
        Task 4.6; event-based lift only per B-N3): write initial
        `allowedTools: ["mcp_perfxpert_*"]` in
        `.gemini/settings.json` + companion lift-logic config keyed
        on `intent_classify`-returned-in-session. Honors
        `PERFXPERT_GATE_HOOK=0` override. Session-state location:
        `~/.gemini/runtime/perfxpert-gate-<session-id>.json` (I-N3).
      - `verify_mcp_live(cwd)` → Step 4/4 (with retry policy from
        Task 4.7 and gate-probe from Task 4.6).
    - `verify_mcp_live(cwd, telemetry=False)`:
      - Best-effort: `gemini mcp list --json` if available. If
        unavailable, fall back to telemetry-probe-only when
        `telemetry=True`. Return `LiveCheckReport`.
    - `uninstall(cwd, scope="project")`:
      - Remove `.perfxpert/AGENTS.md` entry from `context.fileName`
        list (preserving other entries). Remove `mcpServers.perfxpert`
        entry. Remove `<cwd>/.perfxpert/AGENTS.md`.
    - `spawn(...)`: `os.execvpe("gemini", ...)`.

**Tests.**
- `tests/test_cli/test_backends/test_gemini_adapter.py`:
  - `test_tool_name_template_is_single_underscore`
  - `test_install_never_touches_gemini_md` (I3 — the point of Gemini)
  - `test_context_filename_list_merge_preserves_existing` (practical §3.3)
  - `test_context_filename_set_when_absent`
  - `test_install_preserves_existing_mcp_servers`
  - `test_install_idempotent`
  - `test_install_emits_per_step_progress`
  - `test_verify_mcp_live_telemetry_probe` (I11)
  - `test_uninstall_list_merge_removal`
  - `test_spawn_uses_execvpe`

**Commit.** `feat(perfxpert/cli): GeminiAdapter (PR 1) — context.fileName list-append, never touches GEMINI.md`

---

### Task 6 — Dispatcher wiring + `--dry-run` + `--force` + `--allow-agents-md-append` + `--quiet` (I5)

**Why.** Tie adapters together via `_backend_dispatch.py`. Add flags
parsed by the dispatcher (not forwarded to backend).

Note: cycle-1 Task 6 was flagged for retroactively adding `dry_run` to
the Protocol. Cycle-2 already has `dry_run` in Task 1's Protocol; this
task only wires it at the dispatcher level.

**Files.**
- Modify: `perfxpert/cli/_backend_dispatch.py`:
  - Parse dispatcher-owned flags (leading flags before forwarded argv):
    `--dry-run`, `--force`, `--allow-agents-md-append`, `--quiet`.
  - On first run for a (backend, cwd, file-set): call adapter
    `plan(dry_run=True)` first, then `prompt_consent_interactive`, then
    `install(dry_run=dry_run)`.
  - Banner emission: print a single AMD PerfXpert line to stderr
    (re-use `print_banner`) before exec — unless `--quiet`.
  - On successful install, call `adapter.spawn(remaining_argv, env, cwd)`.
    If `dry_run`, print `[DRY-RUN] Would exec: <argv>` and exit 0.
  - On `TrustRequired` (Codex only; PR 2): surface as actionable
    message + exit 2.

**Tests.**
- `tests/test_cli/test_backend_dispatch.py`:
  - `test_first_run_shows_plan_then_prompts_for_consent`
  - `test_subsequent_run_skips_consent_when_file_set_unchanged`
  - `test_subsequent_run_prompts_again_when_file_set_changes`
  - `test_dry_run_does_not_write_files`
  - `test_dry_run_prints_plan_with_every_action`
  - `test_force_skips_consent_prompt_but_still_writes`
  - `test_quiet_suppresses_progress_and_banner`
  - `test_allow_agents_md_append_forwarded_to_adapter`

**Commit.** `feat(perfxpert/cli): dispatcher flags (--dry-run, --force, --allow-agents-md-append, --quiet)`

---

### Task 7 — `install` subcommand + docs (extends Task 0's `mcp-server.md`)

**Why.** Users need discoverability. Promote `install-patches` to a
backend-aware `install` subcommand with a 2-release deprecation window.

**Files.**
- Modify: `perfxpert/cli/opencode_launcher.py`:
  - Add `install` to `_PERFXPERT_SUBCOMMANDS`. Support
    `perfxpert-code install --backend={opencode,claude,codex,gemini,all}`
    (default `opencode`). `all` is **serial** (practical §6.2):
    opencode first, then claude, gemini, codex (in that order).
- Modify: `perfxpert/__main__.py` — add `install` handler that
  dispatches to adapter `install(dry_run=False)` (no exec).
- Modify: `install-patches` handler — emit yellow-stderr deprecation
  notice: `⚠ install-patches is deprecated; use 'install --backend=opencode'. Removal in v0.5.`.
- Modify: `docs/integration/mcp-server.md` (ported in Task 0) — add
  section "Launching via perfxpert-code" with:
  - `perfxpert-code claude` / `gemini` / `codex` invocation examples
  - `perfxpert-code install --backend=<name>` for prep-only
  - Consent + uninstall flow
  - `.gitignore` hints for `.mcp.json` and `.claude/CLAUDE.md`
  - Platform boundary note (POSIX only).
- Create: `docs/guides/multi-backend.md` — user-facing guide. Cross-link
  from README, `docs/integration/mcp-server.md`, and
  `docs/architecture.md`.

**Tests.**
- `tests/test_cli/test_launcher_subcommands.py`:
  - `test_install_subcommand_in_help`
  - `test_install_patches_emits_deprecation_warning`
  - `test_install_patches_forwards_to_install_backend_opencode`
  - `test_install_all_is_serial` (assert order: opencode → claude →
    gemini → codex if present)
- `tests/test_integration/test_docs_links.py` — verify
  `mcp-server.md` and `multi-backend.md` cross-links resolve.

**Commit.** `feat(perfxpert/cli): backend-aware install subcommand + deprecation of install-patches + user guide`

---

### Task 8 — Per-backend `uninstall <name>` (practical §1.5, §7)

**Why.** We wrote config; we need a clean per-backend undo path.

**Files.**
- Modify: `perfxpert/__main__.py` — add `uninstall` handler:
  `perfxpert-code uninstall <backend>` (positional; interactive pick if
  omitted — lists configured backends from consent cache).
- Adapter `uninstall(cwd, scope)` implementations already defined in
  Tasks 4 + 5 (Claude, Gemini). Codex uninstall lands in Task 10.
- Shared helpers live in `perfxpert/cli/backends/_prompt_adapter.py`:
  marker-block drift detection, atomic rewrite.
- Surface a dry-run/confirmation flow: `perfxpert-code uninstall <backend>`
  prints the list of files/entries to modify + line-by-line summary,
  then prompts `y/N` (suppressed by `--yes` / `PERFXPERT_ASSUME_CONSENT=1`).

**Tests.**
- `tests/test_cli/test_backends/test_uninstall.py` (per adapter):
  - `test_install_then_uninstall_round_trip_restores_files_byte_identical`
  - `test_uninstall_refuses_on_marker_block_drift`
  - `test_uninstall_preserves_other_mcp_servers`
  - `test_uninstall_shows_plan_before_action_and_prompts_confirm`
  - `test_uninstall_yes_flag_skips_prompt`

**Commit.** `feat(perfxpert/cli): per-backend uninstall with drift detection and confirmation`

---

### Task 9 — Tool-name length CI lint (I8)

**Why.** Claude Code has a reported 64-char MCP tool-name cap. Our
wire format for claude is `mcp__perfxpert__<underscored_name>`. That
eats 16 chars before the tool's own name. Adding a tool that blows the
cap breaks silently on Claude.

**Files.**
- Create: `tests/test_mcp/test_tool_name_length.py`:
  - Import the perfxpert MCP tool registry (`perfxpert.tools`).
  - For every tool name, assert
    `len("mcp__perfxpert__" + underscored_name) <= 64`. Warn at 56.
  - Same assertion for the gemini template
    `mcp_perfxpert_<underscored_name>`.
- Modify: `pyproject.toml` `[tool.pytest.ini_options]` — ensure this
  test runs on every CI invocation (should be picked up by `testpaths = ["tests"]`).

**Tests.** The file itself.

**Commit.** `test(perfxpert/mcp): lint MCP tool-name length against Claude 64-char cap`

---

### Task 10 — PR 2: `CodexAdapter` + project-trust gate + lazy `tomlkit` + live-probe tool-name discovery (B3 + I7 + B1)

**Why.** Deferred from PR 1 per cycle-2 §8. Carries the Codex-specific
subplots.

**Files.**
- Create: `perfxpert/cli/backends/codex.py`:
  - `CodexAdapter(BackendAdapter)`:
    - `name = "codex"`, `binary_name = "codex"`, `min_version = "0.7.0"`.
    - `tool_name_template`: **probed at first `verify_mcp_live()` run**
      and cached. Default initial guess `"mcp_perfxpert_{tool}"` (based
      on observed Codex behavior, late April 2026). Probe updates cache
      if Codex's actual format differs. (B1.)
    - `spawn_strategy = "execvpe"`.
    - `known_schema_versions = ("0.7.x",)`.
  - **Trust gate (B3).** New method `_check_trust(cwd: Path) -> bool`:
    - Primary: `codex projects list --json`. Parse; check if cwd is
      listed as trusted.
    - If untrusted and TTY:
      - Prompt: "This project is not trusted by Codex. Run
        `codex projects trust .`? (y/N/user-scope)"
      - `y` → shell out, re-check, proceed with project scope
      - `user-scope` → fall back to user scope with ⚠ warning
      - `N` → raise `TrustRequired`
    - If untrusted and non-TTY: raise `TrustRequired`.
  - `install(...)`:
    - Step 0/4: `_check_trust(cwd)`.
    - Step 0.5/4: **Warm `perfxpert-mcp`** via
      `warmup_perfxpert_mcp()` (F2 — Task 4.7). Honors
      `PERFXPERT_MCP_WARMUP=0` override.
    - Step 1/4: Register MCP.
      - Primary: `codex mcp add perfxpert -- perfxpert-mcp`.
      - Fallback: **lazy-import `tomlkit` INSIDE this branch only** (I7):
        ```python
        try:
            import tomlkit
        except ImportError:
            raise PerfxpertBackendError(
                "tomlkit required for codex direct-edit fallback. "
                "Install: pip install 'perfxpert[backends]'"
            )
        ```
        TOML round-trip edit of `$CODEX_HOME/config.toml` (default
        `~/.codex/config.toml`) with `tomlkit`. Add
        `[mcp_servers.perfxpert]` with `command`, `args=[]`,
        `enabled=true`, `startup_timeout_sec=10`.
      - On shell-out failure → print-for-human command before direct
        edit (I4).
    - Step 2/4: Stage rendered prompt at
      `$CODEX_HOME/.perfxpert/AGENTS.md` (NOT cwd — Codex's AGENTS.md
      discovery walks from git root down).
      - 32 KiB cap pre-check (practical §3.4). Raise if rendered
        prompt would exceed.
    - Step 3/4: Pointer / redirect.
      - If `cwd/AGENTS.md` is tracked OR `allow_agents_md_append=False`:
        use `$CODEX_HOME` redirect (per-cwd shell wrapper
        `export CODEX_HOME=.perfxpert/codex-home`).
      - If `allow_agents_md_append=True`: append marker block to
        `cwd/AGENTS.md`.
    - Step 3.5/4: **Install server-side gate hook** (F1 — Task 4.6
      Codex portion). Research required: identify Codex's
      sandbox-level hook surface (`codex` plugin API, allowed-tools
      list, or config.toml policy key) and install the
      event-based rejection logic (reject non-`perfxpert_*` calls
      UNTIL `intent_classify` returns in session; B-N3). If Codex
      does not expose a hook surface, raise `GateHookUnsupported`
      BEFORE Step 1 MCP registration (I-N1) and fall back to
      prompt-layer rejection-language only (accept the degraded
      model-size sensitivity on Codex). Record
      `gate_hook_installed=False` on the subsequent
      `verify_mcp_live()` report.
    - Step 4/4: `verify_mcp_live(cwd)` (with retry policy from
      Task 4.7 and gate-probe from Task 4.6).
      - Probe Codex's actual tool-name wire format, cache under
        `$XDG_CACHE_HOME/perfxpert/codex_tool_name_template.txt`.
      - If cached format differs from `tool_name_template`, refuse
        install and surface the delta — user must re-render the prompt.
  - `uninstall(...)`: remove `[mcp_servers.perfxpert]` via
    `codex mcp remove perfxpert` (shell-out primary), `tomlkit`
    fallback. Remove `$CODEX_HOME/.perfxpert/` if empty.
  - `spawn(...)`: `os.execvpe("codex", ...)`.
- Modify: `perfxpert/cli/_backend_dispatch.py` — unlock `codex` in
  registry (was raising `NotImplementedError` in PR 1).

**Tests.**
- `tests/test_cli/test_backends/test_codex_adapter.py`:
  - `test_trust_gate_detects_trusted_project`
  - `test_trust_gate_prompts_interactively_when_untrusted`
  - `test_trust_gate_raises_trust_required_on_non_tty`
  - `test_trust_gate_user_scope_fallback_warns`
  - `test_tomlkit_lazy_imported_only_in_fallback_branch` (I7 —
    monkeypatch `sys.modules["tomlkit"] = None` on primary path;
    assert install succeeds without ImportError)
  - `test_tomlkit_preserves_comments_on_round_trip`
  - `test_32_kib_cap_precheck`
  - `test_mcp_servers_idempotent`
  - `test_existing_mcp_servers_unchanged`
  - `test_codex_home_redirect_when_agents_md_tracked` (I3)
  - `test_verify_mcp_live_probes_and_caches_tool_name_template` (B1)
  - `test_verify_mcp_live_refuses_install_when_format_drifts`

**Commit.** `feat(perfxpert/cli): CodexAdapter (PR 2) — trust gate, lazy tomlkit, live-probe tool-name discovery`

---

### Task 11 — End-to-end skip-on-CI integration test + manual recipe

**Why.** Real backend binaries aren't on CI, but a documented manual
recipe closes the gap.

**Files.**
- Create: `tests/e2e/test_multi_backend_e2e.py`:
  - `@pytest.mark.skipif(shutil.which("<binary>") is None, reason=…)`
    around each backend test.
  - Each test: tmp HOME, `perfxpert-code <backend> --dry-run "hello"`,
    assert plan output shape, assert no side effects.
  - Tmp HOME pattern (practical §2.2):
    `monkeypatch.setenv("HOME", str(tmp_path)); monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))`.
- Create: `docs/superpowers/plans/2026-04-19-perfxpert-code-multi-backend-manual-test.md`
  — step-by-step recipe a reviewer can run locally.
- Create: `docs/integration/mcp-server.md` "Verification" subsection —
  `perfxpert-code <backend> --dry-run` + the telemetry probe recipe
  from I11.

**Tests.** The file itself.

**Commit.** `test(perfxpert): e2e multi-backend dry-run skip-on-CI + manual recipe + telemetry probe docs`

---

## Commit sequence (cycle-2 revision-2)

| # | Commit | PR |
|---|---|---|
| 0 | Port mcp-server.md from phase 9 to phase 8 (B4) | 1 |
| **0.5** | **Decision record — Claude Code pre-tool-call hook surface (B-N1)** | **1** |
| 1 | BackendAdapter Protocol (full lifecycle) + types (incl. `gate_hook_installed`, `GateHookUnsupported`) + logging | 1 |
| 2 | Route backend subcommands + recursion guard + help passthrough | 1 |
| 3 | Consent cache (backend, cwd, file-set) keyed | 1 |
| **4a** | **`_prompt_adapter.py` shared helpers + golden-file + retry helper (I-N4)** | **1** |
| **4b** | **`ClaudeCodeAdapter` core (check_available/plan/install/spawn/uninstall) (I-N4)** | **1** |
| **4c** | **`ClaudeCodeAdapter.verify_mcp_live` + gate + telemetry probes (I-N4)** | **1** |
| **4.6** | **Server-side event-based gate hook per backend (F1, B-N2, B-N3)** | **1** |
| **4.7** | **Warm perfxpert-mcp in install() + exponential-retry + env tunables (F2, I-N5, I-N6)** | **1** |
| 5 | GeminiAdapter — context.fileName list-append, promoted to PR 1 | 1 |
| 6 | Dispatcher flags (--dry-run / --force / --allow-agents-md-append / --quiet) | 1 |
| 7 | `install` subcommand + deprecation of install-patches + docs | 1 |
| 8 | Per-backend uninstall with drift detection | 1 |
| 9 | MCP tool-name length CI lint (I8) | 1 |
| 10 | CodexAdapter + trust gate + lazy tomlkit + live-probe + Codex hook | 2 |
| 11 | E2E skip-on-CI + manual recipe + telemetry probe docs | 1 (test) / 2 (codex portion) |

Sub-task naming:
- **0.5** is a **pure docs/decision** commit (no code); gates Task 4.6
  Claude-portion only.
- **4a / 4b / 4c** use letter-suffix (not 4.1/4.2/4.3) per cycle-2
  review guidance to avoid collision with the existing decimal
  sub-tasks 4.6 and 4.7.
- **4.6 / 4.7** retain their existing numbers — they operate on the
  same adapter surface as 4a/4b/4c and ship within PR 1.
- Tasks 5, 6, 7, 8, 9, 10, 11 retain their original numbers.

**4a/4b/4c sequencing vs Task 0.5 gate (N3-1 + Practical-3 — cycle-3 followup).**
Tasks 4a, 4b, 4c commit WITHOUT the Claude hook-install step. Task 4.6
inserts the Claude hook-install step into `install()` AFTER Task 0.5's
decision record flips from `Status: PENDING`. This lets 4a/4b/4c commit
early while 0.5 is still pending; the 4.6 Claude-portion is gated.

**4b → 4c batching (Practical-top — cycle-3 followup).** Task 4b and 4c
SHOULD be reviewed and merged in the same batch. If they land in separate
batches, 4b's `PERFXPERT_SKIP_LIVE_CHECK=1` default (see Task 4b Step 4/4)
avoids a broken intermediate state — `install()` Step 4/4 becomes a no-op
until 4c lands.

The PR 1 commit count grew from 12 to 14 (added 0.5 + split 4 into
4a/4b/4c). Average commit size dropped (biggest commit was ~500 LOC
Task 4; now biggest is ~200 LOC Task 4b). Review throughput
improves; total LOC unchanged-to-slightly-higher.

---

## Test strategy (cross-cutting)

### Unit (runs on every CI commit)
- Adapter Protocol conformance — each adapter asserts it implements
  `BackendAdapter` via `typing.runtime_checkable`.
- MCP-config merge: adversarial matrix (practical §4.2):
  - fresh install (no existing config)
  - existing config with no MCP section
  - existing config with MCP section, no perfxpert entry
  - existing config with DIFFERENT perfxpert entry → `ConfigClobber`
    raised unless `--force`
  - existing config with IDENTICAL perfxpert entry (idempotency)
  - malformed config (invalid TOML/JSON) → graceful error with message
  - config with wrong permissions (read-only) → graceful error
  - symlinked config → follow + log a notice
- Prompt staging: existing `CLAUDE.md` / `AGENTS.md` / `GEMINI.md`
  with pre-existing content; assert idempotent append, marker
  boundaries, versioned sentinel, cache-hash verification.
- Consent cache: round-trip, cwd-hash invalidation, file-set-hash
  invalidation, XDG-compliant paths, HOME isolation.
- Tool-name rewrite: golden-file per backend (B1).
- Tool-name length lint: runs in CI (I8 — Task 9).
- `os.execvpe` vs `subprocess.run`: assert each adapter uses the
  correct spawn strategy.
- Git-tracking detection: `git ls-files --error-unmatch` behavior on
  tracked + untracked + missing files.
- **Gate-hook enforcement (F1, B-N3 — Task 4.6):** per-backend unit
  tests that simulate non-perfxpert first tool call and assert
  rejection + `retryWith`. Gate-lift after `intent_classify` also
  asserted. **Event-based only**: test
  `test_permits_bash_on_turn_2_after_intent_classify_on_turn_1`
  directly asserts the cycle-4 false-refusal class (time-based
  rejection after lift) is gone. `GateHookUnsupported` raised
  BEFORE MCP registration when surface unavailable (I-N1).
- **MCP warmup + retry (F2 — Task 4.7):** warmup idempotence, env
  override honored, simulated handshake timeout retried 3× with
  **exponential backoff 2 / 4 / 8** (I-N5) under
  `PERFXPERT_MCP_RETRY_BUDGET_S` budget. Warmup cleanup leaves no
  orphan `-wal` / `-shm` residue (I-N6). Warmup failure reason is
  forwarded into `verify_mcp_live` error message (N-6).

### Integration (runs on CI; no backend binaries required)
- Mock `subprocess.run` for backend CLIs. Assert the launcher calls
  the backend with correct argv, cwd, env.
- Assert default `perfxpert-code` (no args) still stages the
  opencode runtime dir + sets `OPENCODE_CONFIG`. Regression guard.
- `perfxpert-code claude --help` → NO files written, passthrough only.
- Recursion guard: `PERFXPERT_IN_AGENT_SESSION=<any>` refuses to launch.
- `PERFXPERT_OPENCODE_PATH` ignored when backend != opencode.

### End-to-end (skipped on CI, documented for manual run)
- Install `claude`, `codex`, `gemini` in a clean VM or container.
- `perfxpert-code <backend> --dry-run` — print plan, no writes.
- `perfxpert-code <backend>` → exec, confirm interactive session
  loads perfxpert MCP tools (`/mcp list` inside each client).
- Telemetry probe (I11): `PERFXPERT_TELEMETRY=1 perfxpert-code <backend>`
  with a scripted "reply ack after calling intent_classify" prompt,
  verify perfxpert-mcp telemetry log shows `intent_classify` first.

---

## Risks + mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Backend CLI bumps config schema; adapter writes invalid config | Medium | High | Merge via lib (`tomlkit`/JSON round-trip) preserving unknown keys. Version-floor warn on first run. Prefer shell-out to `claude mcp add` / `codex mcp add`. `known_schema_versions` → raise `SchemaUnknown` on mismatch. |
| R2 | User had pre-existing `mcpServers.perfxpert` pointing elsewhere | Low | Medium | Before writing, read existing entry; if `command != "perfxpert-mcp"`, raise `ConfigClobber` unless `--force`. |
| R3 | First run succeeds locally; CI fails (no binaries) | Medium | Low | `shutil.which(...)` + `pytest.skip`. Unit tests use mocks. Manual recipe in Task 11. |
| R4 | Project-scoped `CLAUDE.md` / `AGENTS.md` write surprises user in shared repo | Medium | Medium | **Never-touch-tracked-files default (I3).** Interactive consent on first run keyed on file-set-hash (I10). Distinctive versioned marker block (I6). `--dry-run` shows the plan (I5). `perfxpert-code uninstall <backend>` reverses. |
| R5 | Recursion: user inside perfxpert session runs `perfxpert-code claude` | Low | Low | Single `PERFXPERT_IN_AGENT_SESSION=<backend>` env; dispatcher refuses unless `--force`. |
| R6 | `perfxpert-mcp` not on `PATH` when backend spawns it | Medium | High | `perfxpert-code doctor` verifies; MCP config writes bare `perfxpert-mcp` (resolved via backend's own spawn PATH). |
| R7 | Atomicity: crash mid-write corrupts config | Low | High | Atomic tmp-write-rename. `.bak` retention 7 days. |
| R8 | Whole-file `~/.claude.json` rewrite loses multi-MB session history | **(cycle-1 risk now neutralized)** | High | **Never whole-file-rewrite `~/.claude.json`.** Shell-out primary, print-for-human fallback, structured edit only for small dedicated files. (I4.) |
| R9 | **Tool-name rewrite mismatch** — AGENTS.md references `perfxpert_X` but backend exposes `mcp__perfxpert__X`; agent never finds the tool | **(cycle-1 unknown; cycle-2 resolved)** | High | `tool_name_template` per adapter; `_prompt_adapter` substitutes; `verify_mcp_live()` post-install probe asserts the rendered names match the backend's live tool list. (B1.) |
| R10 | **Codex cwd untrusted** — project-scope config silently ignored | **(cycle-1 unknown; cycle-2 resolved)** | High | `_check_trust(cwd)` in CodexAdapter; explicit error path or user-scope fallback with ⚠ warning. (B3.) |
| R11 | **MCP registered but backend can't spawn it** (the phase-8 miss) | **(cycle-1 untested)** | High | `verify_mcp_live()` probes live MCP connectivity post-install + optional telemetry probe asserts the MCP server received tool calls. (I11.) |
| R12 | Tool-name > 64 chars breaks silently on Claude | Low | Medium | CI lint (I8 — Task 9). |
| R13 | Windows user pip-installs perfxpert, hits CRLF / backslash issues | Low | Low | Explicit POSIX-only boundary documented (I12). Tracking issue for Windows. |
| R14 | `tomlkit` ImportError at runtime when user didn't install `[backends]` extra | Low | Low | Lazy-import inside CodexAdapter's fallback branch; actionable error message. (I7.) |
| R15 | Concurrency: two `perfxpert-code` processes race on `.mcp.json` | Low | Low | Atomic tmp-write-rename; worst case is latest wins. Document known issue. |
| **R-new-4** | **Prompt-layer gate is model-size sensitive; smaller models (Haiku-class) bypass the "do NOT emit bash" advisory and fire non-perfxpert tools in positions 2–4 OR skip perfxpert entirely** | **Medium (re-verified live on claude-haiku-4-5 only — I-N2 scoping; other small models require independent re-verification at acceptance time)** | **High — regresses to phase-8 observed behavior** | **Server-side event-based gate hook per backend (Task 4.6) intercepts non-`perfxpert_*` calls UNTIL `intent_classify` returns and rejects with `retryWith` (B-N3 — purely event-based, no turn counter). Prompt-layer rejection-language stanza (Task 4a `_prompt_adapter`) is secondary enforcement. `verify_mcp_live()` adds a gate-probe asserting the hook is effective. Multi-model acceptance (I-N2) requires probing against ≥ 1 small model per backend (haiku / flash / opencode-default). F1; brainstorm §13.** |
| **R-new-5** | **`perfxpert-mcp` first-run bootstrap exit-124 (MCP handshake or sqlite init race)** | **Medium** | **Medium — single retry clears; user experience regression** | **Warmup during `install()` (Task 4.7) bounded by `PERFXPERT_MCP_WARMUP_TIMEOUT_S` (10s default); `verify_mcp_live()` retries handshake up to 3× with **exponential backoff 2 / 4 / 8** under `PERFXPERT_MCP_RETRY_BUDGET_S` (6s default) early-exit; warmup cleanup leaves no orphan WAL/SHM residue (I-N6); warmup failure reason forwarded into `verify_mcp_live` error message (N-6); `PERFXPERT_MCP_WARMUP=1` default-on env hatch. F2, I-N5, I-N6; brainstorm §13.** |
| **R-new-6** | **opencode gate hook relies on bundled-only patch (phase-8 patch 0020); upstream opencode does not expose `{block, retryWith}`** | **Low (users who swap to upstream opencode)** | **Medium — gate silently degrades to prompt-layer only** | **Documented behavior (B-N2) in Task 4.6 opencode bullet + brainstorm §13 F1 + README. `verify_mcp_live` records `gate_hook_installed=False` + warning-level log when upstream opencode is detected. Not a bug; accepted tradeoff.** |
| **R-new-7** | **Claude Code exposes no suitable pre-tool-call hook surface; Task 4.6 Claude-portion demotes to prompt-layer only** | **Unknown (pending Task 0.5 doc-fetch)** | **Medium — Claude users get prompt-layer-only gate (model-size sensitive)** | **Task 0.5 doc-fetch + decision record (`docs/decisions/2026-04-19-claude-hook-surface.md`) resolves this before Task 4.6 Claude-portion lands. If surface unavailable, `GateHookUnsupported` raised BEFORE MCP registration (I-N1) — no partial state. `verify_mcp_live` treats `gate_hook_installed=False` as documented-known-limit, not failure. B-N1, I-N1.** |

---

## Open questions (cycle-2, copy of brainstorm §7)

1. **Project scope default?** Confirmed: project scope for BOTH MCP
   and prompt. Opt-out via `PERFXPERT_<BACKEND>_SCOPE=user` moves
   BOTH together. (B2.)
2. **Interactive consent or env-based?** Interactive on first run with
   `(backend, cwd, file-set)` cache + `PERFXPERT_ASSUME_CONSENT=1`
   override. (I10.)
3. **`install-patches` deprecation timeline?** 2-release grace with
   yellow-stderr notice; removed in v0.5. (Task 7.)
4. **`uninstall` in PR 1 or PR 2?** PR 1 (per-backend, for claude +
   gemini). Codex uninstall lands in PR 2 with the CodexAdapter.
5. **`tomlkit` hard dep or optional extra?** Optional extra
   `[backends]`, lazy-imported in Codex adapter's fallback only. (I7.)

---

## Acceptance criteria (per PR)

**PR 1 is done when:**
0. **Task 0.5 decision record flipped from `PENDING` (N3-2 — cycle-3
   followup).** `docs/decisions/2026-04-19-claude-hook-surface.md` has
   `Status: DECIDED (<surface>)` where `<surface>` is one of:
   `native-pretooluse` / `allowedTools` / `prompt-layer-only`. This gate
   is hard — PR 1 cannot be merged while PENDING.
1. `perfxpert-code` with no args = unchanged behavior (existing
   `test_main_default_invocation_stages_runtime_cfg_dir` passes).
2. `perfxpert-code claude --dry-run "hello"` prints a plan with every
   action, writes nothing, exits 0.
3. `perfxpert-code claude "hello"` with consent:
   - Registers perfxpert MCP in `<cwd>/.mcp.json` (project scope) via
     `claude mcp add ... --scope project`.
   - Stages `.perfxpert/AGENTS.md` with `mcp__perfxpert__<tool>`
     substituted names.
   - Writes `.claude/CLAUDE.md` with `@.perfxpert/AGENTS.md` (never
     touches git-tracked `CLAUDE.md` without the flag).
   - `verify_mcp_live()` passes.
   - Execs `claude` via `os.execvpe`; Python process exits.
   - Idempotent on re-run.
4. Same three checks pass for `gemini` (with Gemini's
   `context.fileName` list-append invariant).
5. `perfxpert-code claude --help` passes through WITHOUT writing files.
6. `perfxpert-code uninstall claude` reverses (3); refuses on block
   drift.
7. All unit + integration tests pass in CI. Tool-name length lint
   passes.
8. `docs/integration/mcp-server.md` (ported in Task 0) + `docs/guides/multi-backend.md`
   (new) exist with invocation examples + `.gitignore` hints +
   POSIX-only note + known-issue note for first-run exit-124 with
   `PERFXPERT_MCP_WARMUP=1` workaround (F2).
9. **Server-side gate hook (F1, B-N3) — manual recipe only; not
   gated by CI** (N-I5: requires a live backend binary). For claude
   + gemini, a canned "list files" query on a freshly installed
   session is rejected by the hook with a `retryWith` message
   directing the model to call `intent_classify` first. A canned
   "analyze this trace" query routes through `intent_classify`.
   Both assertions covered by `verify_mcp_live` gate-probe +
   telemetry log. **Event-based only**: a legitimate `bash` on
   turn 2 AFTER `intent_classify` on turn 1 passes through (no
   time-based rejection).
9a. **Multi-model gate-probe coverage (I-N2).** `verify_mcp_live`
   gate probe passes against at least ONE small model per backend:
   **haiku** for Claude, **flash** for Gemini, **opencode-default**
   for opencode. The specific models tested are enumerated in the
   PR 1 description under "Models tested for gate probe". R-new-4
   is scoped to: "verified on claude-haiku-4-5; other small models
   require independent re-verification at acceptance time."
9b. **`GateHookUnsupported` pathway (I-N1).** If any backend's hook
   surface is unavailable (e.g. Claude Code per Task 0.5 decision),
   the adapter raises `GateHookUnsupported` BEFORE MCP registration
   (no partial state), and `LiveCheckReport.gate_hook_installed =
   False`. This case is verified via a dedicated unit test, not a
   live acceptance run.
10. **`perfxpert-mcp` warmup (F2, I-N5, I-N6).** Adapter `install()`
    reports a successful `WarmupReport` (or a graceful-degradation
    log if `PERFXPERT_MCP_WARMUP=0`). Warmup hard-cap
    (`PERFXPERT_MCP_WARMUP_TIMEOUT_S`, default 10s) and retry
    budget (`PERFXPERT_MCP_RETRY_BUDGET_S`, default 6s) are both
    respected. `verify_mcp_live()` retries handshake up to 3× with
    **exponential backoff 2 / 4 / 8** (test-verified via simulated
    race). Warmup leaves no orphan `-wal`/`-shm` files (I-N6
    test). Warmup failure reason is forwarded into
    `verify_mcp_live` error message (N-6).

**PR 2 is done when:**
1. `perfxpert-code codex --dry-run` works end-to-end in a trusted
   project.
2. `perfxpert-code codex` in an untrusted cwd prompts for
   `codex projects trust .` OR falls back to user-scope with warning.
3. `tomlkit` lazy import: `perfxpert-code codex` in an environment
   without `[backends]` extra succeeds on the shell-out primary path
   (only fallback path demands `tomlkit`).
4. `verify_mcp_live()` probes and caches Codex's tool-name wire format;
   surfaces drift visibly.
5. `perfxpert-code uninstall codex` reverses install.
6. All Codex tests pass.

---

## Non-goals / explicit deferrals

- No attempt to unify the 4 backends' prompt formats into one "portable
  prompt". Each gets a rendered copy with its tool-name wire format.
- No attempt to surface backend-specific session history in `perfxpert
  analyze` workflows.
- No auto-migration from the opencode session state to another backend.
- No binary bundling of `claude` / `codex` / `gemini` into the perfxpert
  wheel. Users must install them.
- **No Windows-native support** (POSIX only in first cut; tracking
  issue). (I12.)
- **No `perfxpert-code status` subcommand** in cycle-2 (tracked as
  follow-up; good support-ticket signal but not blocking).
- **No docker/CI image with pre-installed backends** (skip-on-CI +
  manual recipe is sufficient for cycle-2).
- **No formal Python logging scaffold** beyond the named-logger stub in
  Task 1 (nitpick N2 — tracked as follow-up).

---

## Estimated size

- **PR 1: Medium-Large** (~**2300 LOC** across ~20 files incl. tests;
  N-7 per cycle-2 review — cycle-1 estimate of ~1900 was optimistic
  once Task 4 split + Task 0.5 docs + new session-state + env-var
  wiring are counted). Adds Task 4a/4b/4c + Task 4.6 gate-hook
  modules + Task 4.7 warmup module. ~16–18 work hours for a focused
  session; review in 3 passes (4a → 4b+4c → 4.6+4.7+5+6+7+8+9).
- **PR 2: Medium** (~700 LOC, parallels existing adapter pattern +
  trust gate + live-probe + Codex-specific hook research; 5–6 work
  hours).

---

## References

- Brainstorm: `docs/superpowers/plans/2026-04-19-perfxpert-code-multi-backend-brainstorm.md`
- Existing launcher: `experimental/python/perfxpert/perfxpert/cli/opencode_launcher.py`
- Existing bundle: `experimental/python/perfxpert/perfxpert/_bundled/opencode_config/`
- MCP integration doc (to be ported in Task 0):
  `experimental/python/perfxpert/docs/integration/mcp-server.md`
- Claude Code memory: https://code.claude.com/docs/en/claude-md
- Claude Code MCP: https://code.claude.com/docs/en/mcp
- Codex config: https://github.com/openai/codex/blob/main/docs/config.md
- Codex AGENTS.md: https://developers.openai.com/codex/guides/agents-md
- Gemini MCP: https://github.com/google-gemini/gemini-cli/blob/main/docs/tools/mcp-server.md
- Gemini GEMINI.md: https://geminicli.com/docs/cli/gemini-md/

---

## Cycle-2 changelog (reviewer findings → plan sections)

| # | Finding | Resolution in plan |
|---|---|---|
| B1 | MCP tool-name rewrite per backend | Task 1 (`tool_name_template` in Protocol), Task 4a (`_prompt_adapter.render_prompt`), Task 4a test (golden-file), Task 4c (`verify_mcp_live`), Task 10 (live-probe for Codex) |
| B2 | Scope uniformity | Assumptions + Task 2 (`scope` param), Task 4b Step 1/4 (both MCP + prompt project-scope), `.gitignore` hint |
| B3 | Codex project-trust gate | Task 10 `_check_trust()` + interactive prompt + `TrustRequired` exception (Task 1) |
| B4 | Stale `mcp-server.md` reference | Task 0 (port from phase 9) |
| I1 | `os.execvpe` for TUIs | Task 1 `spawn_strategy`, Task 4b `spawn()`, Task 5 `spawn()`, Task 10 `spawn()`, Task 4b test `test_spawn_uses_execvpe_not_subprocess_run` |
| I2 | Full Protocol lifecycle in Task 1 | Task 1 rewritten — dry_run, verify_mcp_live, all exceptions from day one |
| I3 | Never-touch-tracked-files default | Task 4b Step 3/4 (git-tracked check + `.claude/CLAUDE.md` fallback), Task 5 (Gemini never touches `GEMINI.md`), Task 10 (Codex `$CODEX_HOME` redirect), `_prompt_adapter.is_git_tracked` |
| I4 | No whole-file `~/.claude.json` round-trip | Task 4b Step 1/4 (primary shell-out; print-for-human fallback; structured edit only for small dedicated files) |
| I5 | Per-step progress + timeouts | Task 4b per-step stderr lines, 15s timeouts, Task 6 `--quiet` |
| I6 | Marker versioning + cache-hash | `_prompt_adapter.emit_marker_block` with `v1 cache=<sha8>`, Task 4a test, Task 8 drift detection |
| I7 | `tomlkit` lazy import | Task 1 optional extra `[backends]`, Task 10 lazy-import inside fallback branch only + test |
| I8 | Tool-name length lint | Task 9 |
| I9 | PR 1 / PR 2 swap (Gemini in PR 1) | Scope rewritten; Task 5 promoted; Task 10 (Codex) in PR 2 |
| I10 | Consent per-(backend, cwd, file-set) | Task 3 `_consent.py` with triple-key; tests for invalidation |
| I11 | Telemetry hook | Task 4c `verify_mcp_live(telemetry=True)`, `PERFXPERT_TELEMETRY=1`, Task 11 docs |
| I12 | POSIX-only boundary | Scope + Assumptions + Non-goals + R13 |
| N1 | Protocol vs ABC | Brainstorm §12 N1 (documented, retained) |
| N2 | Named logger per adapter | Task 1 `_backend_log.py` stub; full scaffold deferred |
| N3 | Windows follow-up | Non-goals + tracking issue |
| **F1** | **Prompt-layer gate insufficient for smaller LLMs (cycle-5 E2E)** | **Task 4.6 (server-side event-based gate hook per backend) + Task 4a `_prompt_adapter` rejection-language stanza + Task 4c `verify_mcp_live` gate-probe + R-new-4 + Assumptions clause** |
| **F2** | **`perfxpert-code run` bootstrap exit-124 (cycle-5 E2E)** | **Task 4.7 (warmup during `install()`) + Task 4c `verify_mcp_live` 3× handshake retry (exponential backoff) + `PERFXPERT_MCP_WARMUP=1` env hatch + R-new-5 + Assumptions clause + docs known-issue note** |

### Cycle-2 review-2 findings (added 2026-04-18; revision-2)

| # | Finding | Resolution in plan |
|---|---|---|
| **B-N1** | **Task 4.6 Claude hook surface flagged as "research required" inline — decision belongs up front** | **NEW Task 0.5 (doc-fetch + decision record) + stub at `docs/decisions/2026-04-19-claude-hook-surface.md`; Task 4.6 Claude-portion explicitly BLOCKED on 0.5; R-new-7 added** |
| **B-N2** | **opencode `{block, retryWith}` is fork-only (phase-8 patch 0020), not upstream opencode** | **Task 4.6 opencode bullet + Assumptions + brainstorm §13 F1 + R-new-6 all carry the explicit callout; Task 4.6 test `test_opencode_documents_fork_only_dependency_in_docstring`** |
| **B-N3** | **Dual gate-lift rules (`_FIRST_N_TURNS=2` AND event-based) re-introduce cycle-4 false-refusal class** | **Task 4.6 rewritten: gate is purely event-based (lift when `perfxpert_intent_classify` returns in session); `_FIRST_N_TURNS` constant DELETED from `_gate_hooks/__init__.py`; new test `test_permits_bash_on_turn_2_after_intent_classify_on_turn_1`** |
| **I-N1** | **`verify_mcp_live` / hook install circular dependency — silent hook no-op reports failure AFTER MCP is registered** | **Task 1 adds `LiveCheckReport.gate_hook_installed: bool \| None` field + `GateHookUnsupported(BackendAdapterError)`. Task 4.6 raises `GateHookUnsupported` BEFORE Task 4b Step 1 MCP registration runs (no partial state). Task 4c treats `gate_hook_installed=False` as documented-known-limit** |
| **I-N2** | **F1 gate-probe verified on Haiku-4-5 only; generalization to flash / gpt-4o-mini / other small models unproven** | **PR-1 acceptance criterion 9a: gate probe passes against ≥ 1 small model per backend (haiku / flash / opencode-default); R-new-4 scope narrowed to "verified on claude-haiku-4-5; other small models require independent re-verification at acceptance time"** |
| **I-N3** | **Session-state tracking per-backend under-specified** | **Task 4.6 sub-table (opencode: patch 0020 in-memory / Claude: TBD Task 0.5 / Gemini: `~/.gemini/runtime/perfxpert-gate-<session-id>.json` / Codex: PR 2). NEW session always starts gate-engaged, even same cwd** |
| **I-N4** | **Task 4 ballooning (~500 LOC + 25 tests)** | **Task 4 split into 4a (`_prompt_adapter.py` + golden-file, ~150 LOC) / 4b (`ClaudeCodeAdapter` core, ~200 LOC) / 4c (`verify_mcp_live` + probes, ~100 LOC). Letter-suffix used (not 4.1/4.2/4.3) to avoid collision with existing 4.6 / 4.7. Commit sequence table + LOC estimate updated** |
| **I-N5** | **Warmup total time budget unspecified; 6s retry budget tight for CI/WSL** | **`PERFXPERT_MCP_WARMUP_TIMEOUT_S` (default 10s, 1 warmup hard cap) + `PERFXPERT_MCP_RETRY_BUDGET_S` (default 6s, exponential backoff 2/4/8 across ≤ 3 attempts). Documented in Task 4.7 + docs/integration/mcp-server.md known-issue subsection** |
| **I-N6** | **Warmup leaves orphan sqlite `-wal` / `-shm` state** | **Task 4.7 adds `test_warmup_spawns_and_closes_perfxpert_mcp_cleanly` asserting no orphan fds, no `-wal` / `-shm` residue. Warmup cleanup uses clean close+wait (not kill -9) so sqlite checkpoints before exit** |
| N-5 | **I8 budget drift 48→64 chars** | **Documented in Task 9 / I8 resolution: cycle-1 finding I8 specified 48-char conservative target; plan uses 64-char Claude Code hard cap with warn-at-56 threshold. Both acceptable; 64 matches the actual cap.** |
| N-6 | **Warmup failure forwarding into `verify_mcp_live` error** | **Task 4c `verify_mcp_live` error message includes prior warmup failure reason; Task 4.7 `install()` captures reason on adapter instance; test `test_verify_mcp_live_error_message_includes_prior_warmup_failure_reason`** |
| N-7 | **PR 1 LOC estimate** | **Bumped ~1900 → ~2300 in Estimated size section; marked as "realistic, not optimistic"** |
| N-B-practical | **Commit-message length for Task 4** | **Resolved by I-N4 split (3 smaller commits)** |
| N-I3 | **Commit sequence grows after Task 4 split** | **Commit sequence table rewritten with 0.5 + 4a/4b/4c; reads cleanly** |
| N-I5 | **Acceptance criterion 9 is manual recipe (requires live backend)** | **Acceptance criterion 9 now explicitly labeled "manual recipe only; not gated by CI"** |

**Final count: 0 blockers, 0 importants, 4 documented-with-rationale
nitpicks** (N-5 / N-6 / N-7 / N-I5 — all inlined at the task they
relate to). Cycle-2 review-2 closed. F1 (HIGH), F2 (MEDIUM), and
cycle-2 blockers B-N1 / B-N2 / B-N3 resolved. Importants I-N1 through
I-N6 resolved. Ready for cycle-3 review.
