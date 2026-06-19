---
name: systematic-debugging
description: "Use when encountering any bug, test failure, build failure, or unexpected behavior in amd-smi — before proposing any fix. Enforces root-cause investigation before symptom patching."
---

# Systematic Debugging — amd-smi

Random fixes waste time and create new bugs. Quick patches mask underlying issues.

**Core principle:** ALWAYS find the root cause before attempting a fix. Symptom fixes are failure.

**Violating the letter of this process is violating the spirit of debugging.**

## The Iron Law

```
NO FIXES WITHOUT ROOT-CAUSE INVESTIGATION FIRST
```

If you haven't completed Phase 1, you cannot propose a fix.

## When to Use

Use for any technical issue:

- Test failures (C++ GTest, Python unit, CLI)
- Bugs reported in `amdsmi_*` output
- Unexpected CLI output
- Build/CMake failures
- Packaging failures (RPM/DEB postinst)
- Wrapper-regen mismatches
- Performance regressions

**Especially when:**
- Under time pressure (emergencies make guessing tempting)
- "Just one quick fix" seems obvious
- You've already tried 2+ fixes
- The previous fix didn't work
- You don't fully understand the issue

## The Four Phases

Complete each phase before proceeding to the next.

### Phase 1: Root-Cause Investigation

1. **Read error messages carefully**
   - Read the full stack trace, every frame
   - Note exact line numbers, file paths, error codes
   - For `amdsmi_status_t` errors: which enum value? Where is it set?

2. **Reproduce consistently**
   - Exact command, exact environment
   - Same hardware? Same install context (system vs pip)?
   - If not reproducible → gather more data, don't guess

3. **Check recent changes**
   - `git log --oneline -20 -- <affected files>`
   - `git diff HEAD~5 -- <affected file>`
   - New CMake options? New dependency? Wrapper regenerated?

4. **Gather evidence across cascade layers**

   amd-smi bugs often span layers. Add instrumentation at every boundary:

   ```bash
   # Layer 1: CLI argparse
   echo "argparse args: $@"

   # Layer 2: Python interface
   python3 -c "import amdsmi; print(amdsmi.amdsmi_get_lib_version())"

   # Layer 3: C wrapper loaded
   python3 -c "from amdsmi import amdsmi_wrapper; print(amdsmi_wrapper.libamd_smi._name)"

   # Layer 4: Actual C call return value
   strace -e trace=openat python3 -c "import amdsmi; amdsmi.amdsmi_init()" 2>&1 | grep libamd_smi
   ```

   This reveals **which layer breaks**, not just that something breaks.

5. **Trace data flow backward**
   - Where does the bad value originate?
   - What called this with the bad value?
   - Keep tracing up the cascade until you find the source
   - Fix at the source, not at the symptom

### Phase 2: Pattern Analysis

1. **Find a working example** — locate the closest-working similar function/command/test
2. **Compare against the reference** — read it completely, line by line; don't skim
3. **List every difference** — however small. Don't assume "that can't matter"
4. **Understand dependencies** — what config, environment, build flags does the working version need?

### Phase 3: Hypothesis and Testing

1. **Form a single hypothesis** — "I think X is the root cause because Y." Write it down.
2. **Test minimally** — the smallest possible change. One variable at a time. No "while I'm here" fixes.
3. **Verify before continuing** — worked? Phase 4. Didn't? Form a NEW hypothesis. Don't pile fixes.
4. **When you don't know** — say "I don't understand X". Don't pretend. Ask the user or research more.

### Phase 4: Implementation

1. **Create a failing test** that reproduces the bug
   - Use the `test-driven-development` skill
   - The test MUST fail before the fix and pass after
2. **Implement a single fix** — address the root cause. One change. No bundled refactors.
3. **Verify**
   - Bug test passes
   - No other tests broken
   - Issue is actually resolved end-to-end (use `verification-before-completion`)
4. **If the fix doesn't work** — STOP. Count attempts.
   - < 3 attempts → return to Phase 1 with the new information
   - **≥ 3 attempts → STOP and question the architecture** (next item)
5. **If 3+ fixes failed: question the architecture**

   Pattern indicating an architectural problem:
   - Each fix reveals new shared state / coupling in a different place
   - Fixes require "massive refactoring" to implement
   - Each fix creates new symptoms elsewhere

   Stop. Discuss with the user. This is not a failed hypothesis — this is a wrong architecture.

## Red Flags — STOP and Follow Process

- "Quick fix for now, investigate later"
- "Just try changing X and see if it works"
- "Skip the test, I'll manually verify"
- "It's probably X, let me fix that"
- "I don't fully understand but this might work"
- "Pattern says X but I'll adapt it differently"
- Proposing solutions before tracing data flow
- **"One more fix attempt" after 2+ failures**
- **Each fix reveals a new problem in a different place**

**All of these mean: stop, return to Phase 1.**

If 3+ fixes failed: question the architecture (Phase 4 step 5).

## Common Rationalizations

| Excuse | Reality |
|--------|---------|
| "Issue is simple, no process needed" | Simple issues have root causes too. Process is fast for simple bugs. |
| "Emergency, no time for process" | Systematic debugging is FASTER than guess-and-check thrashing. |
| "Just try this, then investigate" | First fix sets the pattern. Do it right from the start. |
| "I'll write the test after the fix works" | Untested fixes don't stick. Test-first proves the fix actually addresses the bug. |
| "Multiple fixes at once saves time" | Can't isolate what worked. Causes new bugs. |
| "Reference is too long, I'll adapt the pattern" | Partial understanding guarantees more bugs. Read it fully. |
| "I see the problem, let me fix it" | Seeing the symptom ≠ understanding the cause. |
| "One more fix attempt" (after 2+ failures) | 3+ failures = architectural problem. Question the pattern. |

## Quick Reference

| Phase | Activities | Done When |
|-------|-----------|-----------|
| 1. Root cause | Read errors, reproduce, check changes, instrument layers | You understand WHAT and WHY |
| 2. Pattern | Find working examples, compare differences | You can list every relevant difference |
| 3. Hypothesis | State theory, test minimally | Confirmed, or new hypothesis formed |
| 4. Implementation | Failing test → single fix → verify | Bug resolved, tests green |

## amd-smi-Specific Investigation Tips

- **API cascade bugs:** Use the cascade Quick Check (`grep` across header + cc + wrapper + interface + CLI) — gaps are often the root cause
- **Wrapper mismatch:** If a function's Python signature differs from the C header, run `tools/update_wrapper.sh` — the wrapper drifted
- **Install context bugs:** Reproduce in BOTH system install and pip install before claiming the fix works (see `amdsmi-build-install` skill)
- **CMake bugs:** Always `rm -rf build` between attempts — stale CMake cache hides real causes
- **Test flakiness:** Run the test in isolation, then alone in the suite, then with the full suite — find the polluting test

## Required Companion Skills

- **`test-driven-development`** — Phase 4 Step 1 (failing test)
- **`verification-before-completion`** — Phase 4 Step 3 (verify the fix)
- **`dispatching-parallel-agents`** — when there are 2+ independent failures
