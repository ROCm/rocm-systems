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
  honored. A single `ROCPD_LLM_*` alias remains with a
  `DeprecationWarning` as a migration ramp.
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
- `PERFXPERT_USE_AGENTS` was removed in Phase 7.1 along with the toggle code it kept alive.
- The single-call LLM entrypoint was removed in Phase 7.1; use `perfxpert.agents.runtime.build_session()` + `session.run_root(...)`.
- **CI matrix inverted**: agentic is the primary test matrix; the
  pre-v0.2.0 matrix was removed in Phase 7.1.
- README, CONTRIBUTING, and the Python API docs rewritten for the new
  architecture.

### Deprecated
- `PERFXPERT_LEGACY` was introduced as a one-minor-version fallback in v0.2.0 and was then removed in Phase 7.1.
- `LLMAnalyzer` was kept as a deprecation stub in v0.2.0 and was removed in Phase 7.1.

### Removed
- Pre-agentic interactive workflow module (~4000 LOC: InteractiveSession + WorkflowSession + 7-phase loop) removed in Phase 7.1. Replaced by `perfxpert-code` wrapping the bundled opencode TUI.
- Pre-agentic LLM conversation module (~600 LOC: streaming + auto-compaction) removed in Phase 7.1. Replaced by Agents SDK native sessions.
- Monolithic fence reference guide removed in Phase 7.1. Split into per-agent slices + structured knowledge YAMLs under `perfxpert/knowledge/`.
- `LLMAnalyzer.analyze_with_llm` and the `_call_<provider>()` private methods were removed in Phase 7.1.
- `tests/test_llm_conversation.py` (51 tests of a deleted module) removed in Phase 7.1.
- Conversational-session CLI flags removed in Phase 7.1. Users typing the old flags get a migration hint pointing to `perfxpert-code`.
- `load_reference_guide` export from the pre-agentic tree was removed in Phase 7.1 (previously relocated to `perfxpert.providers._reference_guide`).

### Migration

See [docs/migration-to-agentic.md](docs/migration-to-agentic.md).

### Backwards-compatible stubs (v0.2.0 only, all subsequently removed)

- `LLMAnalyzer` stub class — still importable in v0.2.0, subsequently removed.
- `PERFXPERT_USE_AGENTS` env var — no-op in v0.2.0, subsequently removed.
- `PERFXPERT_LEGACY` — fallback toggle in v0.2.0, subsequently removed.
  A user-supplied reference-guide override env var (pre-rename name)
  was also required while `PERFXPERT_LEGACY=1` was active; both were
  removed along with it.

---

## [0.1.x] — 2026-0X-XX and earlier

See git history. v0.1.x ran the pre-agentic path by default. The
experimental opt-in `PERFXPERT_USE_AGENTS` was available in later v0.1.x
releases (during the agentic refactor) and was itself subsequently removed.
