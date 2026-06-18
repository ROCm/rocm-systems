---
name: test-driven-development
description: "Use when implementing any amd-smi feature, bug fix, or behavior change — before writing implementation code. Enforces strict RED-GREEN-REFACTOR: failing test first, watch it fail, minimal code to pass, refactor."
---

# Test-Driven Development — amd-smi

Write the test first. Watch it fail. Write minimal code to make it pass.

**Core principle:** If you didn't watch the test fail, you don't know what it tests.

**Violating the letter of the rules is violating the spirit of the rules.**

## The Iron Law

```
NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST
```

Wrote code before the test? Delete it. Start over from the test.

**No exceptions:**
- Don't keep it as "reference"
- Don't "adapt" it while writing the test
- Don't peek at it
- Delete means delete

## When to Use

**Always:**
- New `amdsmi_*` API functions
- CLI command additions or behavior changes
- Bug fixes (write the test that reproduces the bug first)
- Refactors that change observable behavior

**Exceptions (ask first):**
- Pure auto-generated files (`amdsmi_wrapper.py`)
- Build/packaging-only changes with no testable behavior
- Throwaway diagnostic scripts

Thinking "this is too simple for a test"? Stop. That's rationalization.

## Red-Green-Refactor

```
RED  → write one minimal failing test
     ↓
verify it fails (and fails for the right reason)
     ↓
GREEN → minimal code to make it pass
     ↓
verify it passes; all other tests still pass
     ↓
REFACTOR → clean up while staying green
     ↓
repeat for the next behavior
```

### RED — Write the Failing Test

One behavior. Clear name. Real code (no mocks unless unavoidable).

**C++ (GTest) example:**
```cpp
TEST_F(AmdSmiTest, GetGpuPowerCapHandlesInvalidHandle) {
    uint64_t cap = 0;
    amdsmi_status_t status = amdsmi_get_power_cap(nullptr, 0, &cap);
    EXPECT_EQ(status, AMDSMI_STATUS_INVAL);
}
```

**Python example:**
```python
def test_get_gpu_power_cap_raises_on_invalid_handle():
    with pytest.raises(AmdSmiLibraryException) as exc:
        amdsmi.amdsmi_get_power_cap(None)
    assert exc.value.err_code == amdsmi.AmdSmiRetCode.STATUS_INVAL
```

### Verify RED — Watch It Fail

**MANDATORY. Never skip.**

```bash
# C++
cd build && make -j$(nproc) amdsmitst && \
  ./tests/amd_smi_test/amdsmitst --gtest_filter=AmdSmiTest.GetGpuPowerCapHandlesInvalidHandle

# Python
cd tests/python_unittest && python3 -m pytest -v -k test_get_gpu_power_cap_raises_on_invalid_handle
```

Confirm:
- Test **fails** (not errors out)
- Failure message matches what you expected
- It fails because the behavior is missing, not because of a typo

**Test passes immediately?** You're testing existing behavior. Fix the test.

### GREEN — Minimal Code

Simplest code that makes the failing test pass. No extra features, no flexibility flags, no "while I'm here" improvements.

### Verify GREEN

**MANDATORY.**

Re-run the same test. Confirm pass. Then run the relevant suite to confirm nothing else broke.

### REFACTOR — Clean Up

After green only. Remove duplication, improve names, extract helpers. Keep tests green. Don't add behavior.

### Repeat

One failing test for the next behavior.

## Good Tests

| Quality | Good | Bad |
|---------|------|-----|
| **Minimal** | One thing. "and" in the name? Split it. | `test('validates handle and reads register and formats output')` |
| **Clear name** | `test_get_power_cap_returns_inval_on_null_handle` | `test_power_1` |
| **Real code** | Calls the actual `amdsmi_*` function | Mocks the function being tested |
| **Observable** | Tests behavior the user sees | Tests internal state nobody else can read |

## Rationalizations — Reality

| Excuse | Reality |
|--------|---------|
| "I'll write tests after to verify it works" | Tests-after pass immediately and prove nothing. You never saw them catch the bug. |
| "I already manually tested all edge cases" | Manual testing is ad-hoc, undocumented, and doesn't re-run on CI. |
| "Deleting X hours of work is wasteful" | Sunk cost. Working code without real tests is technical debt. |
| "TDD is dogmatic, being pragmatic means adapting" | TDD IS pragmatic — finds bugs before commit, prevents regressions, enables refactoring. |
| "Tests-after achieve the same goal — it's about spirit" | No. Tests-after answer "what does this do?" Tests-first answer "what should this do?" |
| "This is just a one-line fix" | One-line fixes have caused production outages. Write the test. |
| "The CI will catch it" | CI catches what you tested. If you didn't write the test, CI is blind. |

## Red Flags — STOP and Start Over

- Code before test
- "I already verified it manually"
- "Tests after achieve the same purpose"
- "It's about the spirit, not the ritual"
- "This case is different because..."
- Watching the test pass without ever seeing it fail
- Adding production code while writing the test

**All of these mean: delete the code, start over with the test.**

## amd-smi-Specific Notes

- **Cascade testing:** A new API function needs tests at multiple layers — at minimum the C++ GTest and a Python unit test. CLI changes need a CLI test.
- **Hardware dependency:** Most amd-smi tests require an AMD GPU. If your test cannot run on the dev box, say so in the test plan in the spec; don't silently skip the RED step.
- **Auto-generated wrapper:** When adding a C function, the wrapper regen is part of the GREEN step, not a separate "fix later".

## Required Companion Skills

- **`systematic-debugging`** — when a test fails in a way you don't understand
- **`verification-before-completion`** — after every commit
