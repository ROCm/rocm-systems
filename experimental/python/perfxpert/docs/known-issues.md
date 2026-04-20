# PerfXpert — Known Issues

Active, living list. Entries that correspond to shipped fixes have been
removed; check `CHANGELOG.md` for history.

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

- Test suite: ~1390 passed / 0 failed / 3-5 env-skipped (OpenAI quota,
  `gemini`/`codex` CLI missing, opencode submodule not initialised).
  Phase 8 landed 3 LitellmModel wiring tests, 1 MCP tool-discovery test,
  1 consent-denied dispatcher test, 4 airgap+tier-0 format-cell tests,
  5 `perfxpert_run_root_analysis` unit tests.
- All three docs scanners (`scripts/lint.sh`, `scripts/link-checker.py`,
  `scripts/test-samples.py`) green in `--strict` mode; enforced by
  `tests/test_docs_tooling/test_ship_readiness.py`.
- End-to-end Docker install validated in `rocm/dev-ubuntu-22.04:latest`:
  single `pip install "perfxpert[all] @ git+…"` completes in ~3 min,
  auto-downloads bun if not on PATH, compiles bundled opencode, yields
  `perfxpert doctor` → **ALL CLEAN**.
