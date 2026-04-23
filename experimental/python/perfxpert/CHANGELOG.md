# Changelog

All notable changes to perfxpert will be documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — 2026-04-19 (release cut)

This is the first tagged perfxpert release. It consolidates the
multi-backend launcher (previously drafted as v0.3.0 below), the Codex
backend adapter (drafted as v0.3.1 below), and Phase 8's final cleanup
into a single v0.2.0 cut. The v0.3.0 and v0.3.1 headings below are
retained as implementation-cycle drafts for audit trail; the canonical
user-visible release is this v0.2.0 block.

### Changed
- **Version bump 0.1.0 → 0.2.0.** `pyproject.toml`, `perfxpert/__init__.py`,
  `CMakeLists.txt`, the `perfxpert doctor` banner fallback, and the
  `docs/guides/getting-started.md` footer marker all move together.
  JSON analysis output keeps `schema_version = "0.1.0"` by design —
  that field tracks the on-disk analysis schema, not the module
  version, and consumers key off it independently.
- **Claude Code pointer path** migrated from `.claude/CLAUDE.md` to
  `CLAUDE.local.md` at the project root. Claude Code auto-loads
  `CLAUDE.local.md` at session start, so the perfxpert prompt now
  reaches the model without requiring a manual `/read`. User-facing
  docs (`docs/guides/backends.md`, `docs/integration/mcp-server.md`,
  `docs/architecture/backend-adapter.md`) are refreshed to match.
- **`claude mcp list` parser** now handles Anthropic Claude CLI's
  plain-text output format. Recent Claude CLI releases dropped the
  `--json` flag; the adapter parses the plain-text table form instead.
  The JSON-specific helpers in `perfxpert/cli/_backend/claude.py`
  (`_extract_server_names`, `_extract_tool_names`) are removed.

### Added
- **Codex backend (`perfxpert-code codex`).** Full Codex CLI adapter
  with trust-gate workflow
  (`[projects."<cwd>"].trust_level = "trusted"` in
  `~/.codex/config.toml`) with interactive prompt + CI-mode
  `PERFXPERT_AUTO_TRUST=1` bypass (emits an always-on stderr warning
  even under `--quiet`). Gate enforcement is **prompt-layer-only**
  because Codex's native `PreToolUse` hook intercepts Bash only
  (not MCP tool calls) as of April 2026. `tomlkit` is promoted to a **required runtime
  dependency** so the comment-preserving TOML fallback path works
  out of the box without an `[extras]` install step. Full install /
  uninstall recipe in `docs/guides/backends.md`.
- **Multi-backend `perfxpert-code` launcher** — `claude`, `gemini`,
  `codex`, `uninstall` subcommands in one dispatcher. Per-`(backend,
  cwd-hash, file-set-hash)` consent; `--dry-run` and `--quiet`
  dispatcher flags; MCP warmup + retry with env-configurable
  timeouts (`PERFXPERT_MCP_WARMUP_TIMEOUT_S`,
  `PERFXPERT_MCP_RETRY_BUDGET_S`, `PERFXPERT_SKIP_LIVE_CHECK`).

### Fixed
- **Wordmark.** The `perfxpert-code` banner previously rendered
  mis-shaped glyphs for `r`, `f`, `p`, `t` — so users saw a string
  that did not read as "PerfXpert". Patch `0051-amd-wordmark-ui.patch`
  is rewritten with correctly-shaped 9-letter block-letter art
  (44 chars wide) and the bundled opencode binary is rebuilt so the
  fix reaches pip-install users automatically.

### Docs
- `docs/guides/getting-started.md` fully audited and refreshed for
  cycle-2/3 stack state (agentic-only, no `PERFXPERT_LEGACY`,
  multi-backend launcher recipes, Codex trust-gate section).
- Phase-9 docs-audit cleanup: `docs/canonical_facts/` + inventory
  snapshot artifacts dropped; docs-tooling test suite pared down and
  relocated under `experimental/python/perfxpert/tests/`.
- Red-team / ship-readiness gate consolidated into
  `tests/test_ship_readiness.py` (strict mode) — ships-as-tested
  evidence rather than CI-gate-only.

## v0.3.1 — Codex backend

### Added
- **`perfxpert-code codex` subcommand** — full Codex CLI adapter.
  Registers perfxpert under `[mcp_servers.perfxpert]` in
  `~/.codex/config.toml` (writes via lazy-imported `tomlkit`,
  comment-preserving), stages `.perfxpert/AGENTS.md`, and handles
  Codex's `[projects."<cwd>"].trust_level = "trusted"` requirement
  (interactive prompt, or `PERFXPERT_AUTO_TRUST=1` for CI with an
  always-on stderr warning bypassing `--quiet`). Gate enforcement is
  prompt-layer-only because Codex's native `PreToolUse` hook
  intercepts Bash only as of April 2026.
  `perfxpert-code uninstall codex` reverses the install (MCP table +
  trust entry + staged `AGENTS.md`), with `ConfigClobber` /
  `skipped_due_to_drift` protection against git-tracked or malformed
  TOML.

## v0.3.0 — multi-backend launcher

### Added
- **`perfxpert-code claude|gemini|codex|uninstall` subcommands.**
  Multi-backend launcher. `claude` / `gemini` adapters fully ship in
  PR 1; `codex` is a stub that prints a deferred notice and exits
  42. `uninstall <backend>` reverses an install.
- **`--dry-run`, `--quiet` dispatcher flags.** `--dry-run` runs
  `adapter.plan()`, prints the actions, skips every write + `spawn()`.
  `--quiet` suppresses the AMD banner + per-step progress log.
- **`BackendAdapter` Protocol for contributors.** Locked day-1
  interface under `perfxpert/cli/_backend/protocol.py` so
  fifth-backend work doesn't need Protocol churn. See
  `docs/architecture/backend-adapter.md`.
- **Per-backend gate hook with event-based lift.** Native
  `PreToolUse` for Claude Code, `allowedTools` restriction for
  Gemini. Lift fires once `perfxpert_intent_classify` returns in the
  current session; non-perfxpert tool calls before that are rejected
  with a retry-hint message.
- **MCP warmup + retry (env-configurable).**
  `PERFXPERT_MCP_WARMUP_TIMEOUT_S` (default 10s),
  `PERFXPERT_MCP_RETRY_BUDGET_S` (default 30s),
  `PERFXPERT_SKIP_LIVE_CHECK` to skip post-install verification in
  CI.
- **Consent model: per-`(backend, cwd-hash, file-set-hash)` tuple.**
  Persisted under `~/.perfxpert/consent.json`. Re-runs in the same
  directory are silent; changing the file set re-prompts.
  `PERFXPERT_ASSUME_CONSENT=1` bypasses the prompt (required for
  non-interactive stdin).

### Deferred
- Codex adapter. Lands in PR 2 (plan Task 10). The `codex`
  subcommand is wired as a stub in PR 1 for surface stability.

### Docs
- `docs/guides/getting-started.md` §"Choosing a backend" —
  short-recipe overview.
- `docs/guides/backends.md` — dedicated user guide (comparison
  table, install / uninstall recipes, env-var reference, four
  troubleshooting scenarios).
- `docs/architecture/backend-adapter.md` — contributor-facing
  Protocol + lifecycle + "how to add a new backend" steps.
- `docs/integration/mcp-server.md` — cross-link to
  `guides/backends.md` at the top (manual snippets remain for
  direct-CLI users).

## vNEXT

### Removed
- **BREAKING**: pre-rename API-key env vars from the old project name are
  no longer honored. Any environment variable prefixed with the old
  project name (including the reference-guide override previously
  referenced in migration docs) must be re-exported under the
  `PERFXPERT_*` namespace. The canonical names are
  `PERFXPERT_LLM_ANTHROPIC_KEY`, `PERFXPERT_LLM_OPENAI_KEY`, and the
  standard vendor `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` continue to be
  honored. A single pre-rename alias (the `ROCPD_LLM_*` prefix — the rest were removed in Phase 7.1) still works with a `DeprecationWarning` as a migration ramp.
- Legacy `ai_analysis` module (`perfxpert/ai_analysis/`) fully removed,
  along with all parity and feature-flag dispatch tests. The
  `PERFXPERT_LEGACY` environment variable is now unrecognized — setting it
  has no effect.
- `docs/deprecation/PERFXPERT_LEGACY.md` superseded; migration notes moved
  to `docs/archive/migration-to-agentic.md`.

### Changed
- `perfxpert.analyze.execute()` unconditionally delegates to the agentic
  runtime; the `_execute_legacy` fallback is gone.
- `perfxpert.agents.root` wires the real `tasks.*` tools (create / next /
  update / close) instead of no-op `lambda` stubs.

## v0.2.0 draft — 2026-04-18 (agentic-runtime landing, rolled into v0.2.0 above)

### Added
- **Agentic runtime**: `perfxpert/agents/` with 7 agent definitions
  (Root, Analysis, Recommendation, Correctness, 3 specialists) over OpenAI
  Agents SDK. See design spec for full architecture.
- **`perfxpert-code`**: new interactive TUI. Ships as part of the pip install
  (bundles opencode per-platform). Replaces the old conversational mode
  flag on `perfxpert analyze`.
- **MCP server**: `perfxpert/mcp_server/` exposes READ_ONLY tools to external
  clients (Claude Desktop, Cursor, etc.).
- **Split fence**: per-agent `agents/fence/*.md` (≤ 400 lines each;
  CI-enforced) replaces the monolithic `llm-reference-guide.md`.
- **22 knowledge YAMLs** in `perfxpert/knowledge/` replace structured data
  that used to be prose in the fence.
- **45 deterministic tools** in `perfxpert/tools/` — all unit-testable, no
  LLM calls.
- **5 provider adapters** in `perfxpert/providers/` (Anthropic, OpenAI,
  Ollama, private, opencode) — pluggable via Agents SDK `Model` protocol.
- **Deterministic 5-gate cascade** (`runtime/gate_cascade.py`) —
  anti-Sakana protection via SOL bound + bitwise/numeric correctness +
  regression gate + test anchors.
- `perfxpert doctor` now reports active mode (`agentic` | `legacy`).

### Changed
- `PERFXPERT_USE_AGENTS` was removed in Phase 7.1, along with the toggle code it kept alive.
- The single-call LLM entrypoint was subsequently removed; use `perfxpert.agents.runtime.build_session()` + `session.run_root(...)`.
- **CI matrix inverted**: agentic is the primary test matrix; the
  pre-v0.2.0 matrix was subsequently removed.
- README, CONTRIBUTING, and the Python API docs rewritten for the new
  architecture.

### Deprecated
- `PERFXPERT_LEGACY` was introduced as a one-minor-version fallback in v0.2.0 and was then subsequently removed.
- `LLMAnalyzer` was kept as a deprecation stub in v0.2.0 and was removed in Phase 7.1.

### Removed
- Pre-agentic interactive workflow module (~4000 LOC: InteractiveSession + WorkflowSession + 7-phase loop) subsequently removed. Replaced by `perfxpert-code` wrapping the bundled opencode TUI. (The 7-phase loop here refers to the user-facing workflow phases — profile → analyze → optimize → re-profile — not release phases.)
- Pre-agentic LLM conversation module (~600 LOC: streaming + auto-compaction) subsequently removed. Replaced by Agents SDK native sessions.
- Monolithic fence reference guide subsequently removed. Split into per-agent slices + structured knowledge YAMLs under `perfxpert/knowledge/`.
- `LLMAnalyzer.analyze_with_llm` and the `_call_<provider>()` private methods were removed in Phase 7.1.
- `tests/test_llm_conversation.py` (51 tests of a deleted module) subsequently removed.
- Conversational-session CLI flags subsequently removed. Users typing the old flags get a migration hint pointing to `perfxpert-code`.
- `load_reference_guide` export from the pre-agentic tree was subsequently removed (previously relocated to `perfxpert.providers._reference_guide`).

### Migration

See [docs/archive/migration-to-agentic.md](docs/archive/migration-to-agentic.md).

### Backwards-compatible stubs (v0.2.0 only, all subsequently removed)

- `LLMAnalyzer` stub class — still importable in v0.2.0, removed in Phase 7.1.
- `PERFXPERT_USE_AGENTS` env var — no-op in v0.2.0, removed in Phase 7.1.
- `PERFXPERT_LEGACY` — fallback toggle in v0.2.0, subsequently removed.
  A user-supplied reference-guide override env var (pre-rename name)
  was also required while `PERFXPERT_LEGACY=1` was active; both were
  removed along with it.

---

## [0.1.x] — 2026-0X-XX and earlier

See git history. v0.1.x ran the pre-agentic path by default. The
experimental opt-in `PERFXPERT_USE_AGENTS` (available in later v0.1.x releases during the agentic refactor) was removed in Phase 7.1.
