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

## Existing Agent Tooling

### Skills (`.claude/skills/`)
- **amdsmi-build-install** — Build, package, install, verify from source
- **amdsmi-python-style-guide** — ROCm Python style rules (type hints, pathlib, error handling)
- **amdsmi-test-runner** — Run C++ and Python tests, verify results
- **changelog-automation** — Check and generate CHANGELOG.md entries
- **restructure-commits** — Consolidate branch commits into logical groups
- **verification-before-completion** — Pre-completion checklist for fixes and refactors
- **personal-bash-deploy** — Personal `~/.bashrc` deploy helpers (`ship`, `amdsmi_install_package`)

### Claude Commands (`.claude/commands/`)
- `/amdsmi-review-branch` — Review current branch vs main
- `/amdsmi-review-pr` — Review GitHub PR by number

### Custom Agents (`.github/agents/`)
- **amdsmi-review** — Orchestrator: comprehensive or focused code review
- **amdsmi-review-architecture** — Design patterns, API consistency, layering
- **amdsmi-review-build** — CMake, packaging, install targets, dependencies
- **amdsmi-review-docs** — Documentation, comments, help text, docstrings
- **amdsmi-review-performance** — Efficiency, scaling, resource usage, hot paths
- **amdsmi-review-security** — Vulnerabilities, secrets, input validation
- **amdsmi-review-skeptic** — Questions necessity, challenges scope, finds simpler alternatives
- **amdsmi-review-style** — Formatting, naming, pre-commit compliance
- **amdsmi-review-tests** — Test coverage, quality, missing tests

### Prompts (`.github/prompts/`)
- `/amdsmi-review-branch` — VS Code prompt for branch review
- `/amdsmi-review-pr` — VS Code prompt for PR review

### Repo Memories
Project structure, API cascade path, build/packaging paths, test directories, and tools are stored as repo memories — surfaced contextually, not loaded into every conversation.

### Instructions (`.github/instructions/` + `.claude/rules/`)
On-demand reference files loaded only when relevant. Source of truth is `.github/instructions/`; `.claude/rules/` symlinks to the same files for Claude Code compatibility:
- **project-layout** — Project layout, API propagation path, per-layer verification checklist, tools/generators, test suites, and build/packaging
