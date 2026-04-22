# Changelog

All notable changes to perfxpert will be documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## vNEXT

### Removed
- **BREAKING**: pre-rename API-key env vars from the old project name are
  no longer honored. Any environment variable prefixed with the old
  project name (including the reference-guide override previously
  referenced in migration docs) must be re-exported under the
  `PERFXPERT_*` namespace. The canonical names are
  `PERFXPERT_LLM_ANTHROPIC_KEY`, `PERFXPERT_LLM_OPENAI_KEY`, and the
  standard vendor `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` continue to be
  honored. The only remaining `ROCPD_LLM_*` support is a temporary
  deprecation alias for API keys.
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

## [0.2.0] — 2026-04-18

### Added
- **Agentic runtime**: `perfxpert/agents/` with 7 agent definitions
  (Root, Analysis, Recommendation, Correctness, 3 specialists) over OpenAI
  Agents SDK. See design spec for full architecture.
- **`perfxpert-code`**: new interactive TUI. Ships as part of the pip install
  and launches the local `opencode` CLI when installed on the system.
  Replaces the old `perfxpert analyze
  --interactive` flag.
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
- **`PERFXPERT_USE_AGENTS=1` is now a no-op** (kept for backward compat).
  The agentic path is the default.
- **`analyze_database()` routes into agent runtime** instead of directly
  into a single LLM call. Signature unchanged.
- **CI matrix inverted**: agentic is the primary test matrix; legacy is
  secondary (see below).
- README, CONTRIBUTING, and AI_ANALYSIS_API.md rewritten for the new
  architecture.

### Deprecated
- **`PERFXPERT_LEGACY=1`**: new opt-in to the deprecated local-only path. LLM
  enhancement is unavailable there. One-minor-version safety net; removed in
  vX.Y+1. See `docs/deprecation/PERFXPERT_LEGACY.md`.
- **`LLMAnalyzer` class**: kept as a deprecation stub that emits
  `DeprecationWarning`. Removal target: vX.Y+2.

### Removed
- `perfxpert/ai_analysis/interactive.py` (~4000 LOC): InteractiveSession +
  WorkflowSession + 7-phase loop. Replaced by `perfxpert-code` wrapping the
  local opencode CLI.
- `perfxpert/ai_analysis/llm_conversation.py` (~600 LOC): LLMConversation +
  streaming + auto-compaction. Replaced by Agents SDK native sessions.
- `perfxpert/ai_analysis/share/llm-reference-guide.md` (monolithic fence).
  Split into per-agent slices + structured knowledge YAMLs.
- `LLMAnalyzer.analyze_with_llm()` method and all `_call_<provider>()`
  private methods.
- `tests/test_llm_conversation.py` (51 tests of a deleted module).
- `--interactive` and `--resume-session` CLI flags. Users typing the old
  flags get a migration hint pointing to `perfxpert-code`.
- Legacy `load_reference_guide` export from `perfxpert.ai_analysis`.
  Relocated to `perfxpert.providers._reference_guide` (legacy-only).

### Migration

See [docs/archive/migration-to-agentic.md](docs/archive/migration-to-agentic.md).

### Backwards-compatible stubs

- `LLMAnalyzer` class still importable, emits DeprecationWarning.
- `PERFXPERT_USE_AGENTS` env var still recognized (no-op).
- `PERFXPERT_LEGACY=1` rerouted to the deprecated local-only path with a
  warning in v0.2.0. LLM enhancement was unavailable in that mode, and the
  old pre-rename reference-guide override env var used there is no longer
  honored; see Removed above.

---

## [0.1.x] — 2026-0X-XX and earlier

See git history. v0.1.x ran the legacy (pre-agentic) path by default.
`PERFXPERT_USE_AGENTS=1` was the experimental opt-in in later v0.1.x
releases (during the agentic refactor).
