# Changelog

All notable changes to perfxpert will be documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Advanced specialist tools — 8 new READ_ONLY MCP tools.** The
  specialist fences now consume eight additional deterministic
  tools surfaced on the MCP wire:
  `kernel_fusion.find_fusion_candidates` (adjacent-short-kernel
  fusion),
  `gpu_runtime_monitor.parse_amd_smi_json` /
  `parse_rocm_smi_json` / `analyze_thermal` (thermal / throttle
  envelope),
  `unified_memory.analyze_paging` (MI300X paging + cross-die),
  `dependency_graph.reconstruct_dag` (critical-path + GPU bubbles),
  `pragma.lookup_pragmas` / `explain_pragma` /
  `suggest_pragmas_for_kernel` (LLVM loop-hint advanced
  recommendations, gated behind `--advanced`).
- **Change-Impact Prediction — 3 new MCP tools.**
  `predict_impact.predict_change_impact`,
  `predict_impact.list_supported_changes`,
  `predict_impact.explain_prediction`. Returns a conservative
  speedup bracket (`hi × 0.85`), confidence, rationale, and a
  `source_citation` back into `knowledge/proven_optimizations.yaml`.
  Amdahl + tier-2 gates enforced internally. Bumps
  `schema_version` to `0.3.3` when any rec fires a prediction.
- **Live Roofline — `roofline.plot_points` MCP tool.** Emits a
  per-kernel `(ai, achieved_flops_per_s, bottleneck_class,
  fp_type, confidence)` list + arch ridge point; webview renders
  an inline-SVG log-log plot. Populating the top-level `roofline`
  key bumps `schema_version` to `0.3.4`.
- **Communication / RCCL analysis —
  `rccl_analysis.analyze_collectives` +
  `interconnect.lookup_peaks` MCP tools.** Per-collective
  `effective_bw_gbps` / `efficiency_pct` / `efficiency_label` +
  rollup `summary`. Validated at the module boundary via the
  `CommunicationBlock` + `CollectiveEntry` Pydantic models
  (`perfxpert/agents/schemas.py`). Bumps `schema_version` to
  `0.3.2`.
- **ATT Flamegraph (webview-only pure-rendering derivative).**
  `perfxpert/formatters/_att_flamegraph.py` renders an inline-SVG
  flamegraph under the Thread Trace `.scard` when
  `thread_trace.has_att_data` is true. No new payload key — pure
  function of the existing `thread_trace.kernels` +
  `top_stalling_instructions` subfields.
- **Knowledge YAML conventions.** `applies_to_gfx` arch gate on
  `matrix_meter.yaml` + `attention_patterns.yaml` (plus selective
  entries in `proven_optimizations.yaml` /
  `compiler_pragmas.yaml`); `units` field on every
  `expected_impact` entry in `fusion_patterns.yaml`.
- **Fence template consolidation.** `diff_specialist.md` is now
  the canonical per-agent fence template; `root.md` and
  `recommendation.md` were re-aligned to that layout.
- **Agents-as-MCP-tools (8 READ_ONLY agent tools).** The MCP server now
  exposes every agent in the hierarchy as its own tool:
  `perfxpert_agent_root`, `perfxpert_agent_analysis`,
  `perfxpert_agent_recommendation`, `perfxpert_agent_correctness`,
  `perfxpert_agent_compute_specialist`,
  `perfxpert_agent_memory_specialist`,
  `perfxpert_agent_latency_specialist`,
  `perfxpert_agent_diff_specialist`. Backend TUIs (Claude Code,
  Gemini CLI, Codex CLI, opencode) read the agent hierarchy as
  reference in `AGENTS.md` and freely pick whichever of the **56**
  MCP tools matches the user's intent. Current MCP tool
  inventory: **56** (8 agent-hierarchy + 48 classifier / knowledge
  / advanced-specialist / trace_diff). Schemas remain frozen per
  `perfxpert/agents/schemas.py`.
- **Public Python API (`perfxpert.api`).** 1:1 mirror of the 8 agent
  MCP tools (`api.agent_root`, `api.agent_analysis`, etc.) for
  in-process embedding without standing up an MCP server. The
  `perfxpert analyze` CLI now calls `perfxpert.api.agent_root(...)`
  internally — same function the MCP tool wraps.
- **Scoped submodule-init fast-install path.**
  `scripts/pip-install-from-git.sh` (new wrapper) + the equivalent
  `GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=submodule.active
  GIT_CONFIG_VALUE_0=experimental/python/perfxpert/opencode pip
  install …` incantation drops the rocm-systems submodule-init time
  from ~141 sec (recursive init of ~34 unrelated submodules) down
  to ~30 ms. Detailed measurements + rationale in
  `docs/guides/getting-started.md` §1.2. The plain recursive
  one-liner still works for users who want it.

### Changed
- **`--llm` providers.** All five providers —
  `anthropic`, `openai`, `ollama`, `private`, `opencode` — are
  selectable from the CLI **and** from `perfxpert.api.agent_root(...,
  provider=<name>)`. `claude-code` stays removed (single-provider
  wrapper, superseded by the `opencode` + backend-adapter stack).
- **No forced aggregator tool.** The backend LLM no longer must
  call a single `perfxpert_run_root_analysis` after
  `perfxpert_intent_classify`. The "MUST call
  `perfxpert_run_root_analysis`" contract is gone; backends choose
  any of the 56 MCP tools based on intent. The tool-priority gate
  still requires `perfxpert_intent_classify` as the first call.
- **FallbackProvider exception taxonomy.** Documented the full
  typed-error set — `AuthError`, `RateLimitError`,
  `QuotaExceededError`, `TransientError`, `FatalError`,
  `TimeoutError`, `ProviderChainExhausted`. Only
  transient + rate-limit cascade; auth / quota / fatal surface
  immediately because retry is futile. See
  `docs/guides/agentic-mode.md` §"Fallback chain".

### Docs
- `README.md` — MCP count 34→43, 5-provider list restored, pip
  upgrade hint, fast-install wrapper surfaced.
- `docs/guides/getting-started.md` — MCP count refreshed in 3
  places, §10 LLM Providers rebuilt for all 5 providers with
  CLI/API parity, duplicate §3/4/5/6 heading numbering fixed
  (continued into §7-§14), §1.2 scoped-submodule content
  retained. Footer marker reintroduced.
- `docs/guides/backends.md` — MCP count refreshed, forced-tool-call
  language removed.
- `docs/guides/agentic-mode.md` — typed-error taxonomy table
  added.
- `docs/integration/README.md` — MCP count refreshed, agent-tools
  line added.
- `docs/integration/mcp-server.md` — MCP count refreshed in 5
  places, "Snapshot: agent-hierarchy tools" section added,
  `_safety` machinery reference rewritten as "private
  `_`-prefixed skip list".
- `docs/architecture.md` — providers documented as CLI- AND
  library-accessible.
- `docs/architecture/README.md` — `backend-adapter.md` index row
  added.
- `docs/architecture/agent-hierarchy.md` — agent MCP + API surface
  cross-reference added; "only reachable via Root" implication
  removed.
- **NEW** `docs/guides/python-api.md` — embedding guide with one
  example per agent, links back to
  `docs/integration/mcp-server.md`.

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
  landed in PR 2. Trust-gate workflow
  (`[projects."<cwd>"].trust_level = "trusted"` in
  `~/.codex/config.toml`) with interactive prompt + CI-mode
  `PERFXPERT_AUTO_TRUST=1` bypass (emits an always-on stderr warning
  even under `--quiet`). Gate enforcement is **prompt-layer-only**
  because Codex's native `PreToolUse` hook intercepts Bash only
  (not MCP tool calls) as of April 2026 — see the Codex hook-surface
  decision record. `tomlkit` is promoted to a **required runtime
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
  `<cwd>/.codex/config.toml` for trusted projects (falling back to
  `~/.codex/config.toml` when project scope is unavailable), stages
  the perfxpert-managed compatibility file `AGENTS.override.md`
  (shadow-copies root `AGENTS.md` when present), and handles
  Codex's `[projects."<cwd>"].trust_level = "trusted"` requirement
  (interactive prompt, or `PERFXPERT_AUTO_TRUST=1` for CI with an
  always-on stderr warning bypassing `--quiet`). Gate enforcement is
  prompt-layer-only because Codex's native `PreToolUse` hook
  intercepts Bash only as of April 2026 (rationale in the local Codex
  hook-surface decision record).
  `perfxpert-code uninstall codex` reverses the install (MCP table +
  trust entry + staged `AGENTS.override.md`), with `ConfigClobber` /
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
  `PreToolUse` for Claude Code, native `BeforeTool` / `AfterTool`
  hooks for Gemini CLI. Lift fires once
  `perfxpert_intent_classify` returns in the current session;
  non-perfxpert tool calls before that are rejected with a
  retry-hint message.
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

### Decisions
- Claude Code gate hook uses the native `PreToolUse` surface (not
  `permissions.deny`). Doc-fetch evidence + rationale are captured in
  the local Claude hook-surface decision record.

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
