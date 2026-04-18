# Phase 8 PR 2 — User-Issue Brainstorm + Plan

Date: 2026-04-18
Branch: `users/aelwazir/perfxpert-phase8-opencode-fork`
Worktree: `/home/aelwazir/work/ai-analysis-rocpd/.worktrees/perfxpert-phase8`

## Issues from a real session

### Issue 1 — perfxpert tools are not called first

In session `ses_25e1.md`, a query of the form `optimize multi_gpu_demo.cpp`
triggered one call to `intent.classify` followed by 18 generic file-tool
calls (`read`/`glob`/`grep`). The workflow tool (`workflow.next_step`)
was never invoked. The opencode-default system prompt gives the model no
incentive to prefer perfxpert MCP tools over built-in file tools for
GPU-performance queries.

**Approach.** Modify the opencode default system prompt (shipped as a
`.txt` file inside the submodule) via a patch file to prepend a
"STRICT TOOL DISCIPLINE" stanza that names the `intent.classify` and
`workflow.next_step` tools explicitly and forbids file-spelunking until
the workflow directs it. Reinforce at the tool-schema layer by
prepending a "Call BEFORE file-search tools..." priority hint to each
perfxpert tool's MCP description.

### Issue 2 — `perfxpert-code doctor` tries to chdir

Running `perfxpert-code doctor` errors with
`Failed to change directory to …/demo-app/doctor`: opencode interprets
a single positional argument as a CWD. The launcher passes argv through
unchanged.

**Approach.** Whitelist known subcommands (`doctor`, `stats`, `run`,
`auth`, `models`, `config`, `debug`) in the launcher. `doctor` routes
to `python -m perfxpert doctor` (our own diagnostic). Everything else
is passed through to opencode as a subcommand, NOT as the CWD.
`--help` and `--version` short-circuit before the config dir is staged.

### Issue 3 — LLM provider rate-limit retries aren't dismissible

Transcript shows repeated `[retrying in 2s attempt #2]` lines. The user
had no way to bail out or override the client-side retry budget.

**Approach (two levels).**

- `PERFXPERT_DISABLE_RATE_LIMIT_RETRY=1` — patch opencode's
  `session/retry.ts` so `retryable()` returns `undefined` (= stop
  retrying) when the env var is `1`. Surfaces the error to the user
  immediately.
- `PERFXPERT_LLM_FALLBACK_CHAIN=openai,anthropic,claude-code` — a thin
  Python wrapper in `perfxpert/providers/_fallback.py` that on
  `RateLimitError` tries the next provider in the chain. Wired into
  `providers.__init__` as `FallbackProvider` / `get_fallback_provider`.

## Why patches, not forks

The opencode submodule is pinned to `v1.4.11`. Forking would make
upgrading hard (rebase per release). Patches in `.patches/` stay
small, reviewable, `git apply --check`-able, and are the standard way
the perfxpert build overlays customizations.

## Success metrics

- **Issue 1.** In a fresh session, asking "optimize X.cpp" triggers
  `intent.classify` FIRST, then `workflow.next_step`, before any
  generic file tool. (Hard to unit-test at the LLM level; surfaced via
  the prompt-patch diff + MCP description lint.)
- **Issue 2.** `perfxpert-code doctor` runs the perfxpert doctor and
  does NOT produce a "Failed to change directory" error. Unit-tested.
- **Issue 3.** Setting `PERFXPERT_DISABLE_RATE_LIMIT_RETRY=1` makes
  the retry policy short-circuit. `FallbackProvider.complete()`
  advances to the next entry on `RateLimitError`. Unit-tested.

## PR layout

| PR | Scope | Commit title |
|----|-------|--------------|
| 2a | Prompt patch + MCP description priority | `feat(perfxpert/prompt): force perfxpert tool priority in opencode default system prompt` |
| 2b | Launcher subcommand routing | `fix(perfxpert/cli): launcher routes subcommands (doctor/stats/run) instead of treating them as CWD` |
| 2c | Fallback chain + retry env override | `feat(perfxpert/providers): fallback-provider chain + rate-limit retry override env var` |
| 2d | AMD rebrand patch series | `feat(perfxpert/opencode): AMD rebrand patch series (banner/prompt/help/color/footer)` |
| 2e | apply-opencode-patches.sh + validation | `chore(perfxpert/phase8): patch apply script + validation` |
