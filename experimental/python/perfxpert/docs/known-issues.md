# PerfXpert — Known Issues

Active, living list. Entries that correspond to shipped fixes have been
removed; check `CHANGELOG.md` for history.

## Fixed in `6b03ae998a`: structural bugs across all 4 report formats

Four structural issues were visible in every airgap-run webview / markdown
/ text / json emitted from `perfxpert analyze`:

- **Summary used ad-hoc `.card` markup and a raw `<pre>`.** The agent
  narrative was emitted as `<section class="card" id="agentic-narrative">
  <h2>Summary</h2><pre style="white-space:pre-wrap;">…</pre></section>`,
  which did not match the `.scard / .shdr / .sbody` template every other
  section uses. Fix: the Summary renders through
  `_build_summary_scard`, a standard `.scard` with the primary
  bottleneck surfaced as a `sbadge-info` header pill and warnings as a
  `.findings` list.

- **Summary landed AFTER `</body></html>`.** The splice helper fell back
  to replacing `</body>` whenever the template's non-existent `</main>`
  anchor failed. Combined with the Tier-0 splice this produced
  `</body></html></section></body></html>` at EOF. Fix:
  `_splice_after_overview` now anchors on `<h2>Overview</h2>` and
  inserts the Summary as the FIRST `.scard` after the Overview card
  inside `<div class="wrap">`, so it always lands at the top of the
  report.

- **Tier-0 splice embedded a complete nested `<!DOCTYPE html>…</html>`
  document.** `_format_tier0_webview` produces a self-contained HTML
  page; splicing it verbatim gave browsers a nested `<head>` / `<body>`
  / `<style>` / `<script>` tree (~300 redundant lines). Fix:
  `_build_tier0_wrapper_scard` extracts only the child
  `<section class="scard">` elements from the tier-0 output and wraps
  them in one parent `.scard` labelled "Tier-0 Source Scan"
  (id `tier0-scan`). Nested scaffolding is stripped entirely — one
  `<!DOCTYPE html>`, one `</body>`, one `</html>`.

- **Markdown metadata mashup.** `**Database:** / **Analysis Date:** /
  **Analysis Tier:**` ran directly into the narrative without a visual
  break. Fix: the Summary block now emits
  `## Summary\n\n**Primary bottleneck:** …\n\n<narrative>\n\n- ⚠
  <warning>\n\n---\n\n**Database:** …`. Warnings keep the `- ` prefix
  so existing tests that assert bullet presence still pass.

The JSON formatter bumps `schema_version` to `"0.3.0"` on every
agentic run to reflect the combined agentic-pipeline + tier-0-separation
+ Summary-section contract. `narrative` is the canonical agent-brain
field; the legacy `llm_enhanced_explanation` mirrors it for
backwards compat.

Regression tests: `tests/test_formatters/test_report_structure.py`
(5 cases covering all 4 formats).

## Fixed in `ec1bb44b40` / `58b1cec4ed` / `871a32981b`: report narrative + Tier-0 leakage

Three related bugs in `perfxpert analyze` report output were fixed on
Phase 8 so the final report reflects real analysis findings and the
Tier-0 source scan stays in its own section.

**Bug 1 — narrative was routing prose (SHA `ec1bb44b40`).**
`perfxpert.agents.root.run_root` used its own LLM prompt as the
narrative. That prompt answers with routing-speech ("Let me proceed
with the analysis workflow directly…") so users saw routing text
instead of analysis findings. Fix: Root now actually calls
`run_analysis` / `run_recommendation` / `run_correctness` based on the
classified intent and synthesizes a findings-driven narrative
("The trace is dominated by kernel `X` at Y% of GPU time; the primary
bottleneck is `compute` based on …"). Root's own LLM output is kept
only as optional high-level framing prose, prepended rather than
replacing. The airgap branch produces the same deterministic narrative
without an LLM.

**Bug 2 — narrative prose leaked into recommendations (SHA `ec1bb44b40`).**
Root fabricated a recommendation whose `summary` was the first line of
the narrative when the LLM returned no recs, producing entries like
"Issue: Let me proceed with the analysis workflow…". Fix: the fallback
is deleted. Empty LLM rec lists are left empty; downstream
`merge_recommendations` + the deterministic pass supply real items,
and formatters already render a "no issues identified" cell when both
are empty.

**Bug 3 — Tier-0 profiling-plan entries leaked into the main
recommendations (SHA `58b1cec4ed` + `871a32981b`).**
`scan_tier0_sources` used to append a "Found N GPU kernel(s) … Start
with a `--sys-trace` baseline …" rec into `tier0_findings.recommendations`,
which merged into the main Recommendations table. Users saw
instrumentation advice mixed with real performance recommendations.
Fix: the scanner now returns two buckets — `profiling_plan_actions`
(instrumentation advice — NEVER in the main list) and `code_patterns`
(actual code-level perf issues — DO flow into main recs). A new
`tier0_findings.profiling_plan` dict exposes the suggested first
command + counters + description. Every formatter now renders a
dedicated "Tier-0: Source Scan" section with Profiling Plan and
Detected Code Patterns subsections side-by-side. The webview anchors
the section with `id="tier0"` so users and tests can target it
directly.

Regression tests: `tests/test_cli/test_report_quality.py` (5 cases).

## Fixed: `--llm <prov>` produced structurally-thin reports in all 4 formats

Before Phase 8 commit `78ad3257f8`, running

```
perfxpert analyze --llm <prov> -i trace.db --source-dir . \
    --format {text,json,markdown,webview} -d out
```

would leave a file on disk with only the agent narrative + a
recommendations list — no `time_breakdown`, no hotspot table, no
memory analysis, no hardware-counter section, no Tier-0 findings.
The root cause was `_format_agentic_output` building a minimal dict
from `RootOutput.metadata` while the actual deterministic analysis
pass (`compute_time_breakdown`, `identify_hotspots`,
`analyze_memory_copies`, `analyze_hardware_counters`,
`analyze_kernel_resources`, `analyze_api_overhead`,
`detect_warmup_issues`, `analyze_thread_trace`,
`generate_recommendations`) was never invoked in the agentic code path.

**Fix (Phase 8 SHA `78ad3257f8`)**: `_execute_agentic` now runs
`build_analysis_payload` alongside `perfxpert.api.agent_root()` and
forwards the rich deterministic dict to `_format_agentic_output`.
The agent brain (narrative + primary_bottleneck + recommendations
+ warnings) is merged with the deterministic sections via
`merge_recommendations` (dedupe by `(type, target)` or
`(category, issue)`; LLM verdict wins but deterministic citations
and code snippets are preserved). Every format — text / JSON /
markdown / webview — now populates every section of the report
contract documented in
[`getting-started.md`](guides/getting-started.md#report-contents-every-format).

The airgap path runs the deterministic pass identically; only the
LLM narrative is replaced by the airgap template. Tier-0-only
(`--source-dir` without `-i`) dispatches to the tier-0 formatters
with the agent narrative spliced in.

## Fixed: empty HTML on `--llm-api-key` passed a model name

Before Phase 8, running

```
perfxpert analyze --llm anthropic --llm-api-key opus-4-7 \
  --source-dir . --format webview -d out -i out1/*/*.db
```

would leave an empty `out/<stem>_results.html` on disk with no
error message. Two root causes are fixed:

1. The `--llm-api-key` CLI flag was silently dropped by the
   agentic runtime — it only read env vars. Fixed: the flag is now
   threaded through `perfxpert.api.agent_root(..., api_key=...)` →
   `build_session` → `_cascade`, which temporarily sets the
   canonical vendor env var (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`
   / …) for the duration of the call. When the flag and env var
   disagree, the flag wins and PerfXpert emits a one-line stderr
   WARNING.
2. An empty LLM response produced a blank formatter pass instead of
   an error. Fixed: `AnalysisSession.run_root` now raises
   `FatalError("provider returned empty response — check API key /
   quota / model availability")` when the non-airgap narrative is
   empty, and the CLI handler (`perfxpert.__main__`) deletes any
   zero-byte output file and exits with rc=2.

Accidentally passing a model name (`opus-4-7`) to `--llm-api-key`
now fails fast with a `FatalError` from the provider SDK (the key
is invalid); the CLI surfaces a clean one-liner on stderr and
never leaves an HTML file behind.

## Codex tool gate is prompt-layer only

`perfxpert-code codex` cannot reject `bash`/`read` calls via a native
hook because Codex's `PreToolUse` surface is Bash-only — the gate
lives in the installed `AGENTS.md` rejection-language stanza. An
adversarial or smaller model can still skip `perfxpert_intent_classify`;
every other backend (`opencode`, `claude`, `gemini`) enforces a
mechanical gate. Accepted trade-off. Disable the prompt gate per
workflow with `PERFXPERT_DISABLE_TOOL_GATE=1`.

## Docs-audit scanner scope

Documented here so contributors reading "zero violations" know what is
and isn't covered.

### `scripts/link-checker.py`

- External URLs (`http://` / `https://`) are not validated.
- Anchor fragments (`#section-id`) are stripped before the file-existence
  check — a link at a missing anchor inside a real file passes.
- `--strict` is output-format only (CSV vs human); same validator set.

Workaround: rely on Markdown preview in your IDE / GitHub for anchor
correctness; external URL health is covered nightly by a separate
link-health workflow.

### `scripts/lint.sh` — banned-string scanner

Excluded paths (see the `find -not -path` clauses in `scripts/lint.sh`):

- `**/.git/**`, `**/.pytest_cache/**` — git / test runner internals.
- `**/docs/confluence/**` — Confluence-amend audit artifacts that must
  reference old symbol names to describe what was scrubbed.
- `**/opencode/**`, `**/node_modules/**` — upstream opencode submodule
  (MIT) + its bun-installed JS dep tree.

The scanner also ignores individual hits on lines containing the
literal phrase `"removed in Phase 7.1"` so CHANGELOG / archive sections
can keep searchable records of removed flags + classes without
re-introducing live guidance.

## Ship state (2026-04-20)

- Test suite: **1424 passed / 3 skipped / 0 failed** on the Phase 9
  tip. Skips are environment-only (OpenAI quota, `gemini`/`codex`
  CLI missing, opencode submodule not initialised).
- All three docs scanners (`scripts/lint.sh`, `scripts/link-checker.py`,
  `scripts/test-samples.py`) green in `--strict` mode; enforced by
  `tests/test_docs_tooling/test_ship_readiness.py`.
- End-to-end Docker install validated in `rocm/dev-ubuntu-22.04:latest`
  with the **scoped submodule init** path
  (`scripts/pip-install-from-git.sh` or `GIT_CONFIG_COUNT=1 …`):
  completes in ~30 sec + build, auto-downloads bun if not on PATH,
  compiles bundled opencode, yields `perfxpert doctor` → **ALL
  CLEAN**. The plain recursive `pip install "perfxpert[all] @ git+…"`
  one-liner still works but pays a 3-6 min penalty initialising
  unrelated rocm-systems submodules — see the fast-install wrapper
  under `docs/guides/getting-started.md` §1.2.
