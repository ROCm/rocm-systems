# PerfXpert — Known Issues

Active, living list. Entries that correspond to shipped fixes have been
removed; check `CHANGELOG.md` for history.

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
