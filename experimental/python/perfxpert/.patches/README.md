# PerfXpert OpenCode Patches

This directory contains AMD-specific patches applied to the opencode
submodule (pinned to `v1.4.11`) to turn stock opencode into the
**AMD ROCm PerfXpert** interactive CLI. Patches stay here (not in the
submodule) so that bumping opencode is `git submodule update` + rerun
of `scripts/apply-opencode-patches.sh`.

## Patch Series

Applied in lexicographic order by `scripts/apply-opencode-patches.sh`.
Every patch must pass `git apply --check` against its predecessor's
output; CI verifies this via `tests/test_patches/test_apply.py`
(pending) and by running the shell script.

### AMD Rebrand (Phase 8 PR 2d)

| # | File | What it does |
|---|------|--------------|
| 0001 | `0001-banner-amd-rebrand.patch` | `cli/logo.ts` — AMD ROCm PerfXpert attribution comment + exports `perfxpertBanner` tagline. |
| 0002 | `0002-prompt-system-perfxpert.patch` | `session/prompt/anthropic.txt` — prepends the 7-agent PerfXpert hierarchy (intent / workflow / bottleneck / roofline / counters / sol / regression). |
| 0003 | `0003-help-text-amd.patch` | `cli/cmd/tui/feature-plugins/home/tips-view.tsx` — adds three PerfXpert tips (how to phrase queries, list of MCP tools, fallback-chain env var). |
| 0004 | `0004-color-palette-amd.patch` | `cli/ui.ts` — shifts `TEXT_HIGHLIGHT` from cyan (`\x1b[96m`) to xterm-256 index 196 (AMD red ≈ `#ED1C24`); WCAG AA contrast preserved on light+dark terminal bg. |
| 0005 | `0005-footer-attribution.patch` | `cli/ui.ts` — exports `PERFXPERT_ATTRIBUTION = "AMD ROCm PerfXpert · opencode v1.4.11 (MIT)"` for the status-bar renderer. Applies after 0004 in the same file. |

### User-Issue Patches (Phase 8 PR 2a, 2c)

| # | File | What it does |
|---|------|--------------|
| 0010 | `0010-perfxpert-tool-priority.patch` | `session/prompt/default.txt` — prepends a "STRICT TOOL DISCIPLINE" stanza forcing `intent_classify` then `workflow_next_step` as the first two tool calls for GPU-performance queries. Paired with the MCP schema-layer priority hint in `mcp_server/server.py`. |
| 0011 | `0011-rate-limit-retry-override.patch` | `session/retry.ts` — `retryable()` returns `undefined` immediately when `PERFXPERT_DISABLE_RATE_LIMIT_RETRY=1`, so users can escape unending client-side rate-limit retries. |

## Application

```bash
cd experimental/python/perfxpert
bash scripts/apply-opencode-patches.sh
```

The script iterates `.patches/*.patch` in order, `git apply --check`s
each, then applies it. Any failure short-circuits.

## Build-hook status (Phase 8 note)

Ideally the wheel build runs `apply-opencode-patches.sh` automatically
before bundling the opencode binary. That hook requires `bun install`
inside the submodule, which in turn needs a working `bun` toolchain
(not available on CI runners without an extra install step). **Until
that toolchain is in place, the apply step is manual.** Document the
manual step in the README / developer runbook; do NOT ship a built
wheel without patches applied.

Tracking: `docs/known-issues.md` — "apply-opencode-patches.sh is not
yet wired into the wheel build".

## Reverting

Each patch is reversible via `git apply -R <patch>`. To completely
restore the pristine submodule:

```bash
cd experimental/python/perfxpert/opencode
git checkout HEAD -- .
git clean -fd
```
