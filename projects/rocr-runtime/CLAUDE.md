# CLAUDE.md — ROCR Runtime Agent Guide

## Project Overview

ROCR Runtime combines the HSA Runtime (ROCr) and ROCt Thunk libraries:
- **ROCr** (`libhsa-runtime64.so`): HSA runtime implementing the HSA Foundation specification
- **ROCt** (`libhsakmt`): Thunk library providing user-mode interface to the AMDGPU kernel driver (ROCk)

## Critical Rules

1. **HSA API backward compatibility** — never break ABI without major version bump
2. **PRs target `develop`** branch (not `main`)
3. **libhsakmt is always static** — linked into `libhsa-runtime64`, never exposed directly
4. **HSA API functions must return `hsa_status_t`** — all public HSA functions follow this pattern
5. **Test suites are separate builds** — rocrtst and kfdtest have their own CMake configurations
6. **Formatting via clang-format** — use `_clang-format` config (Google style, 100 col, 2-space indent)
7. **HSA spec conformance** — changes to HSA API behavior must align with HSA Foundation specification

## Behavioral Guidelines

Bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

**When asking the user to decide:**

- Before surfacing a question, resolve what you can yourself (read the code, the docs, or dispatch a subagent). Only ask about things that genuinely need the user's judgment.
- Present choices as a lettered multiple-choice list (A / B / C) with a one-line tradeoff for each, and recommend one. Default to asking in chat; only write a `/tmp` doc if the user asks for one.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior HSA runtime engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove functions/variables that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add signal validation" → "Write tests for invalid signals, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

### 5. HSA Runtime Specific Constraints

- **Thread safety:** Signal operations are concurrent — consider race conditions
- **Performance critical paths:** Signal load/store/wait, AQL dispatch, memory copies
- **Error propagation:** ROCt errors → HSA status codes → application
- **Memory ordering:** Document any memory barriers or atomic operations
- **Kernel interface:** All kernel calls go through ROCt — never bypass the thunk layer

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## HSA API Design Principles

1. **All public HSA functions return `hsa_status_t`** — success/error indication
2. **Handles are opaque** — `hsa_agent_t`, `hsa_signal_t` are opaque to applications
3. **Output parameters via pointers** — `hsa_status_t hsa_foo(input, output*)`
4. **Thread-safe by default** — document any exceptions
5. **Backward compatible** — once published, HSA API cannot break existing apps

## Architecture Layers

Respect the layer boundaries:

```
Application
    ↓
HSA Runtime (ROCr) — runtime/hsa-runtime/
    ↓
ROCt Thunk (libhsakmt) — libhsakmt/
    ↓
AMDGPU Kernel Driver (ROCk)
```

Never bypass a layer. Runtime code should call ROCt, not ioctl directly.

## Test Requirements

- **New HSA API functions** → rocrtst tests required
- **ROCt changes** → kfdtest tests required
- **Bug fixes** → reproducer test required
- **Performance changes** → benchmark/measurement required

## Build Targets

- `make` — builds libhsa-runtime64 (includes libhsakmt statically)
- `make install` — installs headers, library, CMake config
- `make package` — creates RPM/DEB packages

Test suites build separately (see `.claude/rules/project-layout.md`).
