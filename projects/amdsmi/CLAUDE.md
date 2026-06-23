# CLAUDE.md — AMD SMI Agent Guide

## Project Overview

AMD SMI is a multi-language system management library for AMD GPUs and EPYC CPUs.
Core C++ library (`libamd_smi.so`) with Python bindings, CLI, Go shim, and Rust bindings.

## Critical Rules

1. **Never edit `py-interface/amdsmi_wrapper.py` manually** — it's auto-generated. Regenerate with `tools/update_wrapper.sh` or `cmake -DBUILD_WRAPPER=ON`.
2. **PRs target `develop`** branch (not `main`).
3. **Pre-commit must pass** before review: `pip install pre-commit && pre-commit install`
4. **Version** is defined in `include/amd_smi/amdsmi.h` (`AMDSMI_LIB_VERSION_MAJOR/MINOR/RELEASE`). CMake extracts it from there.
5. **Excluded from formatting/linting**: `docs/`, `build/`, `esmi_ib_library/`, `third_party/`, `*.md`, `*.rst`

## Behavioral Guidelines

Bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.