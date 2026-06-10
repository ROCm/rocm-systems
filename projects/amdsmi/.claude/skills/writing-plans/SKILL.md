---
name: writing-plans
description: "Use when an approved spec exists and you need a bite-sized, file-level implementation plan before any code is written. Produces a plan ready for executing-plans or subagent dispatch."
---

# Writing Plans — amd-smi

Translate an approved spec into a concrete plan an engineer (or fresh subagent) with zero project context can execute task-by-task. **DRY. YAGNI. TDD. Frequent commits.**

**Announce at start:** "I'm using the `writing-plans` skill to create the implementation plan."

**Save plans to:** `docs/dev/plans/YYYY-MM-DD-<feature-name>.md`

## Required Inputs

- An approved spec at `docs/dev/specs/YYYY-MM-DD-<topic>-design.md`
- Familiarity with the amd-smi project layout (see `project-layout` rule)

If no spec exists, STOP and invoke the `brainstorming` skill first.

## Scope Check

If the spec covers multiple independent subsystems, it should have been split during brainstorming. If it wasn't, push back: each plan must produce working, testable software on its own. Suggest splitting before writing the plan.

## File Structure (Map First)

Before writing tasks, map every file that will be created or modified and what each is responsible for. This locks in decomposition decisions.

For amd-smi work, this map almost always includes:

| Layer | Typical Files |
|-------|---------------|
| Public header | `include/amd_smi/amdsmi.h` |
| C++ impl | `src/amd_smi/amd_smi.cc`, `rocm_smi/src/*.cc` |
| Wrapper | `py-interface/amdsmi_wrapper.py` (auto-generated — task = run `tools/update_wrapper.sh`) |
| Python API | `py-interface/amdsmi_interface.py` |
| CLI | `amdsmi_cli/amdsmi_commands.py`, `amdsmi_parser.py`, `amdsmi_helpers.py` |
| Tests | `tests/amd_smi_test/`, `tests/python_unittest/` |
| Docs | `docs/`, `CHANGELOG.md` |
| Build | `CMakeLists.txt`, `RPM/`, `DEBIAN/` |

Tasks should follow the cascade order so each commit builds independently.

## Plan Document Header

Every plan starts with:

```markdown
# <Feature Name> Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL — use `executing-plans` (or the `amdsmi-development` agent's subagent dispatch) to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** <one sentence>

**Spec:** <link to spec file>

**Architecture:** <2-3 sentences>

**Cascade layers touched:** <list>

---
```

## Task Structure

Each task is a self-contained unit producing a single commit. Steps within a task are 2–5 minute actions.

````markdown
### Task N: <Component Name>

**Files:**
- Create: `exact/path/to/file.cc`
- Modify: `exact/path/to/existing.h:42-58`
- Test: `tests/amd_smi_test/test_file.cc`

- [ ] **Step 1: Write the failing test**

```cpp
TEST_F(AmdSmiTest, AmdsmiGetNewFeature) {
    amdsmi_status_t status = amdsmi_get_gpu_new_feature(handle, &out);
    EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
}
```

- [ ] **Step 2: Run test — verify it fails**

```bash
cd build && make -j$(nproc) amdsmitst
./tests/amd_smi_test/amdsmitst --gtest_filter=AmdSmiTest.AmdsmiGetNewFeature
```
Expected: FAIL — symbol not defined.

- [ ] **Step 3: Implement** (show full code, no `// ...existing...`)

- [ ] **Step 4: Run test — verify it passes**

- [ ] **Step 5: Commit**

```bash
git add include/amd_smi/amdsmi.h src/amd_smi/amd_smi.cc tests/amd_smi_test/test_file.cc
git commit -s -m "[AMD-SMI] Add amdsmi_get_gpu_new_feature"
```
````

## No Placeholders

The following are **plan failures**. Never write them:

- `TBD`, `TODO`, `implement later`, `fill in details`
- "Add appropriate error handling" / "handle edge cases" without showing the code
- "Write tests for the above" without the actual test code
- "Similar to Task N" — repeat the code; engineers may read tasks out of order
- References to types, functions, or methods not defined in any earlier task

Every step that changes code must show the code.

## Self-Review

After writing the complete plan, re-read it against the spec:

1. **Spec coverage** — every requirement maps to a task. List gaps; add tasks.
2. **Placeholder scan** — none of the patterns above present
3. **Type consistency** — function/type/field names match across tasks (a method called `clear_layers()` in Task 3 and `clearAllLayers()` in Task 7 is a bug)
4. **Cascade completeness** — header → impl → wrapper regen → interface → CLI → tests → docs → changelog all addressed
5. **Commit boundaries** — each commit compiles independently when possible

Fix inline. Move on.

## Execution Handoff

After saving the plan, offer the user a choice:

> Plan saved to `docs/dev/plans/<filename>.md`. Two execution options:
>
> 1. **Subagent-driven** (recommended for multi-task plans) — invoke the `amdsmi-development` agent or planning agent, fresh subagent per task with review between tasks
> 2. **Inline execution** — execute tasks in this session using the `executing-plans` skill, batched with checkpoints
>
> Which approach?

If the planning agent dispatched this skill, return the plan path and the recommended execution mode rather than offering interactively.
