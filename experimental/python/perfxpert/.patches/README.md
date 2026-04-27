# PerfXpert OpenCode Patches

This directory contains AMD-specific patches applied to the opencode
submodule (gitlink-pinned to upstream commit `a35b8a95c27d28e979a3826e1289d7ee87f40251`,
from the `v1.4.11` tree) to turn stock opencode into the
**AMD ROCm PerfXpert** interactive CLI. Patches stay here (not in the
submodule) so that bumping opencode is `git submodule update` + rerun
of `scripts/apply-opencode-patches.sh`.

## Patch Series

Applied in lexicographic order by `scripts/apply-opencode-patches.sh`.
Every patch must pass `git apply --check` against its predecessor's
output; CI verifies this via `tests/test_patches/test_apply.py`
(pending) and by running the shell script. `SHA256SUMS` is the allowlist
and integrity manifest for the series; the apply script refuses to run
if a patch is missing from the manifest, extra in the manifest, or fails
checksum verification.

### Naming convention

All patches follow a single naming rule:

```
<NNNN>-<lowercase-hyphenated-slug>.patch
```

- **`NNNN`** — 4-digit, zero-padded numeric prefix. Gaps are allowed
  so thematically-grouped patches can be numbered together (the
  current series uses `0001-0005` for AMD rebrand, `0010-0011` for
  user-issue patches, and `0012-0017` for per-prompt coverage).
- **slug** — lowercase, hyphenated, brief description. No
  underscores or uppercase letters. Example: `rate-limit-retry-override`.
- **extension** — always `.patch`.

**Why 4 digits?** Because `ls` / glob expansion in bash sorts
lexicographically, not numerically. A 4-digit zero-padded prefix
guarantees *lexicographic sort == numeric sort* for the current
range (`0001` through `9999`). Without padding,
`10-foo.patch` would sort BEFORE `2-foo.patch`, breaking apply order.

**Warning — 10,000th patch:** if the series ever approaches a
5-digit prefix, rename the whole catalogue to 5-digit prefixes at
once (`00001-...` etc.). Do **not** mix 4-digit and 5-digit
prefixes — mixing breaks the lexicographic-equals-numeric invariant
(`10000-foo.patch` would sort before `9999-foo.patch`).

### AMD Rebrand

| # | File | What it does |
|---|------|--------------|
| 0001 | `0001-banner-amd-rebrand.patch` | `cli/logo.ts` — AMD ROCm PerfXpert attribution comment + exports `perfxpertBanner` tagline. |
| 0002 | `0002-prompt-system-perfxpert.patch` | `session/prompt/anthropic.txt` — prepends the 7-agent PerfXpert hierarchy (intent / workflow / bottleneck / roofline / counters / sol / regression) AND the STRICT TOOL DISCIPLINE stanza. Both stanzas are injected into every model prompt (see 0010, 0012–0017) so users on any backend receive identical PerfXpert guidance. |
| 0003 | `0003-help-text-amd.patch` | `cli/cmd/tui/feature-plugins/home/tips-view.tsx` — adds three PerfXpert tips (how to phrase queries, list of MCP tools, fallback-chain env var). |
| 0004 | `0004-color-palette-amd.patch` | `cli/ui.ts` — shifts `TEXT_HIGHLIGHT` from cyan (`\x1b[96m`) to 24-bit truecolor `\x1b[38;2;237;28;36m` for the exact AMD red `#ED1C24`; WCAG AA contrast preserved on light+dark terminal bg. Modern terminals render the brand colour exactly; legacy xterm-256 terminals approximate to index 160. |
| 0005 | `0005-footer-attribution.patch` | `cli/ui.ts` — exports `PERFXPERT_ATTRIBUTION = "AMD ROCm PerfXpert · opencode v1.4.11 (MIT)"` for the status-bar renderer. Applies after 0004 in the same file. |

### User-Issue Patches

| # | File | What it does |
|---|------|--------------|
| 0010 | `0010-perfxpert-tool-priority.patch` | `session/prompt/default.txt` — prepends the 7-agent hierarchy AND the STRICT TOOL DISCIPLINE stanza forcing `intent_classify` then `workflow_next_step` as the first two tool calls. Paired with the MCP schema-layer priority hint in `mcp_server/server.py`. |
| 0011 | `0011-rate-limit-retry-override.patch` | `session/retry.ts` — `retryable()` returns `undefined` immediately when `PERFXPERT_DISABLE_RATE_LIMIT_RETRY=1`, so users can escape unending client-side rate-limit retries. |
| 0012 | `0012-prompt-gpt-perfxpert.patch` | `session/prompt/gpt.txt` — same 7-agent hierarchy + STRICT TOOL DISCIPLINE stanza prepended. Covers GPT-family models routed via opencode's `PROMPT_GPT`. |
| 0013 | `0013-prompt-gemini-perfxpert.patch` | `session/prompt/gemini.txt` — same preamble for Gemini models (routed via `PROMPT_GEMINI`). |
| 0014 | `0014-prompt-kimi-perfxpert.patch` | `session/prompt/kimi.txt` — same preamble for Kimi models (routed via `PROMPT_KIMI`). |
| 0015 | `0015-prompt-codex-perfxpert.patch` | `session/prompt/codex.txt` — same preamble for GPT-Codex subvariants (routed via `PROMPT_CODEX`). |
| 0016 | `0016-prompt-beast-perfxpert.patch` | `session/prompt/beast.txt` — same preamble for GPT-4/o1/o3 models (routed via `PROMPT_BEAST`). |
| 0017 | `0017-prompt-trinity-perfxpert.patch` | `session/prompt/trinity.txt` — same preamble for Trinity models (routed via `PROMPT_TRINITY`). |

### Cycle-4 Tool-Gate Patch (cycle-4 B1)

| # | File | What it does |
|---|------|--------------|
| 0020 | `0020-perfxpert-tool-gate.patch` | All 8 primary prompts — appends a **TOOL GATE ENFORCEMENT** block after the existing STRICT TOOL DISCIPLINE stanza. Mandates that the FIRST tool call for any GPU-perf query be a perfxpert MCP tool (not `bash`/`read`/`glob`/`grep`/`edit`/`todowrite`). Documents a self-correction retry protocol and the `PERFXPERT_DISABLE_TOOL_GATE=1` escape hatch. Paired with the strengthened `[MUST BE CALLED FIRST FOR GPU-PERF QUERIES]` priority bracket in `mcp_server/server.py::_fn_to_tool_schema`. |

**B1 approach — weaker-variant, known limitation.** The cycle-4 brief asked
for a *pre-turn* tool-availability hook that only exposes `perfxpert_*` tools
for the first 2 turns, OR a *post-turn* rejection hook that rewrites
non-perfxpert tool_calls into a retry. Both require patching opencode's
session/processor.ts message-flow machinery (`plugin.trigger(...)` is
currently fire-and-forget). Implementing a real blocking hook inside the
opencode TypeScript runtime is outside the 45-minute budget for this
cycle. The weaker-variant we ship instead:

  1. Strengthens the MCP tool descriptions with an ALL-CAPS
     `[MUST BE CALLED FIRST FOR GPU-PERF QUERIES]` bracketed imperative
     (LLMs respect this form more reliably than a plain sentence).
  2. Prepends a new **TOOL GATE ENFORCEMENT** stanza to the STRICT TOOL
     DISCIPLINE block in every primary prompt, including a self-correction
     retry protocol and an opt-out env var.

The **limitation**: neither of these measures *mechanically* blocks a
non-perfxpert tool call — an adversarial or ignorant LLM can still call
`bash` first. The live-scenario validation shows the prompt+bracket
combo moves the needle (perfxpert-ratio up from 0/17), but does not
guarantee 100% compliance. A real pre-/post-turn gate at the opencode
TypeScript layer is tracked as a follow-up (see `docs/known-issues.md`).

**Coverage note (I1 fix):** opencode ships 8 *imported* primary model prompts
(anthropic, default, beast, codex, gemini, gpt, kimi, trinity) selected by
`session/system.ts::provider()`. All 8 now receive the same 7-agent hierarchy
preamble + STRICT TOOL DISCIPLINE stanza. Users on any model family get
identical PerfXpert framing — no single-model bias. Auxiliary prompts
(`plan.txt`, `build-switch.txt`, `max-steps.txt`, `plan-reminder-anthropic.txt`,
`copilot-gpt-5.txt`) are not patched: the first three are tactical inserts, and
the last two are not imported by opencode v1.4.11.

### Deep Rebrand (follow-up)

Existing patches 0001/0003/0004/0005 only rebrand banner, tips, color, and
footer. Patches 0030–0040 extend the rebrand to every remaining user-visible
TUI/CLI surface — terminal titles, session window titles, command palette
menu entries, help dialog, status dialog, home-footer version line,
error-component text, uninstall + upgrade prompts, MCP server setup prompts,
pr/run/serve/web/attach command descriptions, and an explicit attribution
entry.

Legal/compat invariants respected: opencode `LICENSE`, `NOTICE`,
`package.json` fields, submodule tag, `opencode.ai` URLs (help links), and all
`@opencode-ai/*` / `@opencode/*` imports are untouched.

| # | File | What it does |
|---|------|--------------|
| 0030 | `0030-deep-rebrand-session-ui.patch` | `cli/cmd/tui/app.tsx` — terminal title `OpenCode` → `AMD ROCm PerfXpert`; session title prefix `OC |` → `PerfXpert |`. Upgrade-complete toast reads `AMD ROCm PerfXpert (opencode engine vX.Y.Z)`. |
| 0031 | `0031-deep-rebrand-menus-help.patch` | Command `describe:` strings for `tui`, `run`, `attach`, `serve`, `web`, plus the `DialogHelp` component header/body — now show PerfXpert branding with attribution line `Built on opencode (MIT) by Sam Stephenson · https://github.com/sst/opencode`. |
| 0032 | `0032-deep-rebrand-status-bar.patch` | `DialogStatus` header `Status` → `AMD ROCm PerfXpert — Status`; home-footer version pill shows `AMD ROCm PerfXpert · opencode vX.Y.Z (MIT)`; needs-auth hint suggests `perfxpert-code mcp auth`. |
| 0033 | `0033-deep-rebrand-errors-toasts.patch` | `ErrorComponent` report banner + fatal-error copy rebranded to PerfXpert. |
| 0034 | `0034-deep-rebrand-config-labels.patch` | `uninstall` + `upgrade` commands — prompt intros, warn/error log lines, and final thank-you message use PerfXpert branding. Underlying package-manager names (`opencode-ai`, `brew opencode`) are unchanged (real package IDs). |
| 0035 | `0035-deep-rebrand-mcp-display-names.patch` | `mcp list` / `mcp auth` / `mcp auth list` / `mcp logout` prompt intros all prefix with `PerfXpert — ` / `AMD ROCm PerfXpert — `. |
| 0036 | `0036-deep-rebrand-log-prefixes.patch` | `pr.ts` — `describe`, "Found opencode session", "Starting opencode..." rewritten to PerfXpert. The spawned `opencode` binary argv is unchanged. |
| 0040 | `0040-about-perfxpert-attribution.patch` | Adds an `About PerfXpert` entry (slash `/about`) to the command palette. Selecting it shows a 10-second info toast with the full attribution. |

**Deliberately NOT rebranded** (with reason):
- `opencode` binary name passed to `Process.spawn` / `execvp` — renaming breaks install/upgrade.
- npm package `opencode-ai`, brew formula `opencode`, scoop/choco IDs — real package IDs.
- `opencode.ai/docs` and `opencode.ai/zen` URLs — link to the actual opencode service.
- `OpenCode Zen` / `OpenCode Go` dialog copy — describes the commercial OpenCode products.
- `@opencode/` Effect service tags, all `Flag.OPENCODE_*` env-var names, `.opencode/` config dir, `opencode.json` config file — internal identifiers / on-disk compat.

## Application

```bash
cd experimental/python/perfxpert
bash scripts/apply-opencode-patches.sh
```

The script iterates `.patches/*.patch` in order, `git apply --check`s
each, then applies it. Any failure short-circuits.

## Build-hook status

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
