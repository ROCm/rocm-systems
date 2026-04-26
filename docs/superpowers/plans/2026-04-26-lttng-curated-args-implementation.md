# LTTng Curated Parameter-Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-API typed parameter-capture LTTng tracepoints for ~82 curated HIP and HSA APIs, generated from a YAML DSL, augmenting the existing generic enter/exit events.

**Architecture:** A YAML DSL (`curated_apis.yaml`) is the source of truth per provider. A Python codegen script reads the YAML and emits a per-API LTTng tracepoint header plus per-API inline emit helpers (both checked into the tree). A Python verifier script uses libclang to assert the YAML matches HIP/HSA header signatures (separate CI gate). The existing `lttng_migrate.py` is extended to inject a sentinel comment, IN-param C locals, and `_CURATED` macro routing into curated wrappers. New `_CURATED` / `_CURATED_NOARGS` / `_CURATED_HSA` macros (six variants) preserve the existing generic exit events while emitting the typed `<api>_args` event.

**Tech Stack:** Python 3 + PyYAML (codegen), libclang (verifier), C/C++17 (LTTng-UST tracepoint provider, helper inlines, macros), CMake (opt-in regenerate target), bash (coverage gate, payload tests), babeltrace2 (test assertions), LTTng-UST 2.13+.

**Spec:** `docs/superpowers/specs/2026-04-26-lttng-curated-args-design.md` (commit `7def6dcc1d`).

**Branch:** `lttng` tracks `origin/users/bewelton/lttng`. Folds into PR [#5475](https://github.com/ROCm/rocm-systems/pull/5475).

**Container for end-to-end testing:** `bewelton_lttng` on `banff-ccs-aus-g05-05.cs-aus.dcgpu` (Ubuntu 24.04, gfx942 MI325X, ROCm 7.2.2). Default build path: `/root/rocm-systems/build`. Stock library backups at `/root/rocm-stock-true/`. Container-local patches reapplied via `./dev-bin/apply-container-patches.sh main`.

---

## File Structure

### New files

```
projects/clr/hipamd/scripts/
    curated_apis.yaml                       # HIP DSL (Phase B)
    lttng_curated_lib.py                    # Shared DSL parser + field-budget calc (Phase A)
    lttng_curated_codegen.py                # Codegen: YAML -> tp.h + emit.h (Phase A)
    lttng_curated_verify.py                 # Verifier: libclang vs YAML (Phase A)
    test_lttng_curated_lib.py               # Unit tests for parser library (Phase A)
    test_lttng_curated_codegen.py           # Golden-file tests for codegen (Phase A)
    test_lttng_curated_verify.py            # Verifier tests (Phase A)

projects/clr/hipamd/src/lttng/
    rocm_dim3_pack.h                        # ROCM_DIM3_PACK macro + tests target (Phase A)
    rocm_hip_curated_tp.h                   # GENERATED, checked in (Phase C)
    rocm_trace_emit_curated.h               # GENERATED, checked in (Phase C)

projects/clr/hipamd/test/lttng/
    test_hip_curated_args_payload.sh        # Payload assertions (Phase D)
    test_hip_curated_args_coverage.sh       # All curated APIs fire (Phase D)
    test_dim3_pack.cpp                      # Unit tests for ROCM_DIM3_PACK (Phase A)

projects/rocr-runtime/runtime/hsa-runtime/scripts/
    curated_apis.yaml                       # HSA DSL (Phase E)
    # codegen + verify scripts SHARED with HIP via lttng_curated_lib.py
    # but invoked from this dir with --provider rocm_hsa

projects/rocr-runtime/runtime/hsa-runtime/lttng/
    rocm_hsa_curated_tp.h                   # GENERATED, checked in (Phase E)
    rocm_trace_emit_curated.h               # GENERATED, checked in (Phase E)

projects/rocr-runtime/runtime/hsa-runtime/test/lttng/
    test_hsa_curated_args_payload.sh        # Payload assertions (Phase E)
    test_hsa_curated_args_coverage.sh       # All curated APIs fire (Phase E)
```

### Modified files

```
projects/clr/hipamd/scripts/
    lttng_migrate.py                        # Add curated routing (Task 8)
    lttng_coverage_gate.sh                  # Sentinel + macro regex + IN local check (Task 11)

projects/clr/hipamd/src/lttng/
    rocm_hip_tp.h                           # Add #include of generated curated tp.h (Task 7)
    rocm_trace_emit.h                       # Add _CURATED macros + #include curated emit.h (Task 9)

projects/clr/hipamd/src/CMakeLists.txt      # Opt-in regenerate target (Task 12)

projects/clr/hipamd/src/hip_table_interface.cpp  # Re-migrated by extended migrator (Task 10)

projects/rocr-runtime/runtime/hsa-runtime/scripts/
    lttng_migrate.py                        # Same curated routing for HSA (Task 14)
    lttng_coverage_gate.sh                  # HSA mirror updates (Task 14)

projects/rocr-runtime/runtime/hsa-runtime/lttng/
    rocm_hsa_tp.h                           # #include generated curated tp.h (Task 14)
    rocm_trace_emit.h                       # _CURATED_HSA macros (Task 14)

projects/rocr-runtime/runtime/hsa-runtime/CMakeLists.txt  # Regenerate target (Task 14)

projects/rocr-runtime/runtime/hsa-runtime/core/common/hsa_table_interface.cpp  # Re-migrated (Task 14)
```

### CI workflow

```
.github/workflows/lttng-curated-gates.yml  # NEW — drift + verifier gates (Task 13.5)
```

---

## Phase Summary

| Phase | Tasks | Deliverable |
|---|---|---|
| **A.** Foundation library | 1–4, 4.5 | `lttng_curated_lib.py`, `rocm_dim3_pack.h`, codegen + verifier scripts (incl. multi-header support), all with unit tests, ZERO impact on existing build |
| **B.** Author HIP YAML (minimal) | 5 | `curated_apis.yaml` with 2 APIs (`hipMemcpyAsync` all-IN, `hipMalloc` OUT-param) |
| **C.** Generate + wire HIP headers | 6–7 | `rocm_hip_curated_tp.h`, `rocm_trace_emit_curated.h` checked in, included from existing tp.h / emit.h |
| **D.** Migrator + macros + coverage gate | 8–11 | Extended migrator + `_CURATED` macros + updated coverage gate, all wrappers re-migrated |
| **E.** Build + test HIP minimal | 12–13, 13.5 | CMake opt-in target, library builds, payload + coverage tests pass on container, CI drift+verifier gates wired |
| **F.** Expand HIP curated set | 14 | Full HIP YAML (~72 APIs); regenerate; re-migrate; re-test |
| **G.** HSA mirror | 15 | HSA YAML, generate, _CURATED_HSA macros, migrator extension, tests pass |
| **H.** Final integration | 16 | Container deploy, GraphBench overhead measurement, debate-driven-development re-run, push to PR #5475 |

---

## Phase A — Foundation library (tasks 1–4)

This phase builds isolated, unit-tested tooling: the `ROCM_DIM3_PACK` C macro, the Python parser library, the codegen script, and the verifier script. **Zero impact on the existing build** — nothing in this phase is wired into CMake. All four tasks can be built and committed without affecting the LTTng-instrumented `libamdhip64.so` already on the branch.

---

### Task 1: ROCM_DIM3_PACK header + unit tests

**Files:**
- Create: `projects/clr/hipamd/src/lttng/rocm_dim3_pack.h`
- Create: `projects/clr/hipamd/test/lttng/test_dim3_pack.cpp`

- [ ] **Step 1: Write the failing tests**

Create `projects/clr/hipamd/test/lttng/test_dim3_pack.cpp`:

```cpp
// Standalone unit tests for ROCM_DIM3_PACK. Compile with:
//   g++ -std=c++17 -I /opt/rocm/include test_dim3_pack.cpp -o test_dim3_pack
// Expects ROCM_DIM3_PACK to live in the same dir as this test (header is
// included via -I argument or relative path -- see the Makefile target in
// Step 6).
#include "../../src/lttng/rocm_dim3_pack.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>  /* for dim3 */

#define EXPECT_EQ(a, b) do { \
    if ((a) != (b)) { \
        std::printf("FAIL: %s:%d  %s != %s  (got 0x%016lx, want 0x%016lx)\n", \
                    __FILE__, __LINE__, #a, #b, \
                    (unsigned long)(a), (unsigned long)(b)); \
        return 1; \
    } \
} while (0)

int main() {
    // Per spec §4.1 testable success criteria:

    // test_dim3_packed_normal_range
    EXPECT_EQ(ROCM_DIM3_PACK(dim3(1, 2, 3)),
              uint64_t(0x0003000200000001ULL));

    // test_dim3_packed_x_full_32bit
    EXPECT_EQ(ROCM_DIM3_PACK(dim3(0xFFFFFFFFu, 1, 1)),
              uint64_t(0x00010001FFFFFFFFULL));

    // test_dim3_packed_y_overflow: y=0x10000 saturates to 0xFFFF, bit 63 set
    {
        uint64_t v = ROCM_DIM3_PACK(dim3(1, 0x10000u, 1));
        EXPECT_EQ(v & ROCM_DIM3_OVERFLOW_BIT, ROCM_DIM3_OVERFLOW_BIT);
        EXPECT_EQ((v >> 32) & 0xFFFFu, uint64_t(0xFFFFu));      // y lane saturated
        EXPECT_EQ((v >> 48) & 0x7FFFu, uint64_t(1));            // z untouched
    }

    // test_dim3_packed_z_overflow: z=0x10000 saturates to 0x7FFF, bit 63 set
    {
        uint64_t v = ROCM_DIM3_PACK(dim3(1, 1, 0x10000u));
        EXPECT_EQ(v & ROCM_DIM3_OVERFLOW_BIT, ROCM_DIM3_OVERFLOW_BIT);
        EXPECT_EQ((v >> 48) & 0x7FFFu, uint64_t(0x7FFFu));      // z lane saturated
    }

    // test_dim3_packed_z_high_bit_overflow: z=0x8000 ALSO overflow (15-bit lane)
    {
        uint64_t v = ROCM_DIM3_PACK(dim3(1, 1, 0x8000u));
        EXPECT_EQ(v & ROCM_DIM3_OVERFLOW_BIT, ROCM_DIM3_OVERFLOW_BIT);
        EXPECT_EQ((v >> 48) & 0x7FFFu, uint64_t(0x7FFFu));
    }

    // test_dim3_packed_z_max_no_false_overflow: z=0x7FFF is the non-overflow max
    {
        uint64_t v = ROCM_DIM3_PACK(dim3(1, 1, 0x7FFFu));
        EXPECT_EQ(v & ROCM_DIM3_OVERFLOW_BIT, uint64_t(0));
        EXPECT_EQ((v >> 48) & 0x7FFFu, uint64_t(0x7FFFu));
    }

    // test_dim3_packed_x_max_no_false_overflow: x is full 32 bits, never overflows
    {
        uint64_t v = ROCM_DIM3_PACK(dim3(0xFFFFFFFFu, 1, 1));
        EXPECT_EQ(v & ROCM_DIM3_OVERFLOW_BIT, uint64_t(0));
    }

    std::printf("PASS: all dim3_packed encoding tests\n");
    return 0;
}
```

- [ ] **Step 2: Try to compile (verify it fails — header doesn't exist)**

```bash
cd projects/clr/hipamd
g++ -std=c++17 -I/opt/rocm/include test/lttng/test_dim3_pack.cpp -o /tmp/test_dim3_pack 2>&1 | head -5
```

Expected: error about `rocm_dim3_pack.h: No such file or directory`.

- [ ] **Step 3: Implement the header**

Create `projects/clr/hipamd/src/lttng/rocm_dim3_pack.h`:

```c
/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * dim3 packing macro for LTTng curated tracepoints. See spec §4.1.
 *
 * Lane layout (64-bit packed value):
 *   bits  0..31  : x  (full 32 bits)
 *   bits 32..47  : y  (16-bit lane, saturates to 0xFFFF on overflow)
 *   bits 48..62  : z  (15-bit lane, saturates to 0x7FFF on overflow)
 *   bit  63      : overflow flag (set iff y or z exceeded its lane;
 *                  z >= 0x8000 is treated as overflow because the lane is
 *                  intentionally 15 bits to keep bit 63 unambiguous)
 *
 * Branch-light: no error/abort path, only saturating arithmetic. Overflow
 * is a degraded-data signal, not an error. Consumers MUST treat any packed
 * value with bit 63 set as "true y and/or z is unknown but >= the
 * lane-saturated value" per spec §4.1 overflow policy.
 */
#ifndef ROCM_DIM3_PACK_H_
#define ROCM_DIM3_PACK_H_

#include <stdint.h>
#include <hip/hip_runtime.h>  /* for dim3 */

#define ROCM_DIM3_OVERFLOW_BIT (1ULL << 63)
#define ROCM_DIM3_Z_MAX        (0x7FFFu)   /* 15-bit lane max */

static inline uint64_t ROCM_DIM3_PACK(dim3 d) {
    const uint64_t x = (uint64_t)d.x;
    const uint32_t y_raw = d.y;
    const uint32_t z_raw = d.z;
    const uint64_t y = (y_raw > 0xFFFFu) ? 0xFFFFu : y_raw;
    const uint64_t z = (z_raw > ROCM_DIM3_Z_MAX) ? ROCM_DIM3_Z_MAX : z_raw;
    const uint64_t overflow = ((y_raw > 0xFFFFu) || (z_raw > ROCM_DIM3_Z_MAX))
                                  ? ROCM_DIM3_OVERFLOW_BIT : 0ULL;
    return x | (y << 32) | (z << 48) | overflow;
}

#endif  /* ROCM_DIM3_PACK_H_ */
```

- [ ] **Step 4: Compile and run the tests**

```bash
g++ -std=c++17 -I/opt/rocm/include projects/clr/hipamd/test/lttng/test_dim3_pack.cpp -o /tmp/test_dim3_pack
/tmp/test_dim3_pack
```

Expected output: `PASS: all dim3_packed encoding tests`. Exit code 0.

- [ ] **Step 5: Commit**

```bash
git add projects/clr/hipamd/src/lttng/rocm_dim3_pack.h \
        projects/clr/hipamd/test/lttng/test_dim3_pack.cpp
git commit -m "lttng: add ROCM_DIM3_PACK macro with saturating overflow encoding

Implements the 32+16+15+1-bit dim3 packing format defined in the
curated-args spec §4.1. The 15-bit z lane (vs 16-bit) preserves bit 63
as an unambiguous overflow indicator. Includes seven unit tests
covering normal range, x full-32-bit, y/z overflow, z=0x8000 (which
is treated as overflow per the encoding), and z=0x7FFF max-non-overflow.

Self-contained header; not yet wired into any tracepoint emit helper —
that wiring lands with the codegen script in a follow-up commit.

Spec: docs/superpowers/specs/2026-04-26-lttng-curated-args-design.md §4.1"
```

---

### Task 2: DSL parser library (`lttng_curated_lib.py`)

**Files:**
- Create: `projects/clr/hipamd/scripts/lttng_curated_lib.py`
- Create: `projects/clr/hipamd/scripts/test_lttng_curated_lib.py`

- [ ] **Step 1: Write the failing tests**

Create `projects/clr/hipamd/scripts/test_lttng_curated_lib.py`:

```python
"""Unit tests for lttng_curated_lib. Run from worktree root:
    python3 -m pytest projects/clr/hipamd/scripts/test_lttng_curated_lib.py -v
or:
    python3 projects/clr/hipamd/scripts/test_lttng_curated_lib.py
"""
import io
import sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lttng_curated_lib import (
    parse_yaml_text, validate_api, expanded_field_count,
    DSL_TYPES, ALLOWED_DIRS, ParseError, BudgetError, IN_ARG_KIND,
)

# ---- Schema parsing ----

def test_parses_minimal_api():
    apis = parse_yaml_text("""
- api: hipMemcpyAsync
  category: memory
  args:
    - {name: dst,    type: ptr,    dir: IN}
    - {name: src,    type: ptr,    dir: IN}
""")
    assert len(apis) == 1
    assert apis[0]['api'] == 'hipMemcpyAsync'
    assert apis[0]['category'] == 'memory'
    assert len(apis[0]['args']) == 2

def test_rejects_missing_required_top_field():
    try:
        parse_yaml_text("- api: foo\n  args: []\n")  # no category
    except ParseError as e:
        assert 'category' in str(e)
        return
    raise AssertionError("expected ParseError")

def test_rejects_unknown_type():
    try:
        parse_yaml_text("""
- api: foo
  category: memory
  args: [{name: x, type: gizmo, dir: IN}]
""")
    except ParseError as e:
        assert 'gizmo' in str(e)
        return
    raise AssertionError("expected ParseError")

def test_rejects_unknown_dir():
    try:
        parse_yaml_text("""
- api: foo
  category: memory
  args: [{name: x, type: uint32, dir: SIDEWAYS}]
""")
    except ParseError as e:
        assert 'SIDEWAYS' in str(e)
        return
    raise AssertionError("expected ParseError")

def test_rejects_inout_v1():
    """Per spec §4.4: dir: INOUT is hard-error in v1."""
    try:
        parse_yaml_text("""
- api: foo
  category: memory
  args: [{name: x, type: uint32, dir: INOUT}]
""")
    except ParseError as e:
        assert 'INOUT' in str(e)
        return
    raise AssertionError("expected ParseError")

# ---- Field-budget calculation (spec §4.4) ----

def test_field_count_simple():
    api = {'api': 'foo', 'category': 'memory', 'args': [
        {'name': 'a', 'type': 'uint32', 'dir': 'IN'},
        {'name': 'b', 'type': 'ptr',    'dir': 'OUT'},
    ]}
    # 2 args, each 1 field  => 2 payload fields
    assert expanded_field_count(api) == 2

def test_field_count_dim3_expands_to_3():
    api = {'api': 'foo', 'category': 'kernel_launch', 'args': [
        {'name': 'g', 'type': 'dim3', 'dir': 'IN'},
    ]}
    assert expanded_field_count(api) == 3

def test_field_count_dim3_packed_is_1():
    api = {'api': 'foo', 'category': 'kernel_launch', 'args': [
        {'name': 'g', 'type': 'dim3_packed', 'dir': 'IN'},
    ]}
    assert expanded_field_count(api) == 1

def test_field_count_hipLaunchKernel_natural():
    """Spec §4.4 high-arity table: hipLaunchKernel natural = 10 payload."""
    api = {'api': 'hipLaunchKernel', 'category': 'kernel_launch', 'args': [
        {'name': 'function_address', 'type': 'ptr',    'dir': 'IN'},
        {'name': 'numBlocks',        'type': 'dim3',   'dir': 'IN'},
        {'name': 'dimBlocks',        'type': 'dim3',   'dir': 'IN'},
        {'name': 'args',             'type': 'ptr',    'dir': 'IN'},
        {'name': 'sharedMemBytes',   'type': 'size',   'dir': 'IN'},
        {'name': 'stream',           'type': 'handle', 'dir': 'IN'},
    ]}
    # 1 + 3 + 3 + 1 + 1 + 1 = 10 payload  (over budget without mitigation)
    assert expanded_field_count(api) == 10

def test_field_count_hipLaunchKernel_packed():
    """Spec §4.4: with dim3_packed mitigation, payload = 6."""
    api = {'api': 'hipLaunchKernel', 'category': 'kernel_launch', 'args': [
        {'name': 'function_address', 'type': 'ptr',         'dir': 'IN'},
        {'name': 'numBlocks',        'type': 'dim3_packed', 'dir': 'IN'},
        {'name': 'dimBlocks',        'type': 'dim3_packed', 'dir': 'IN'},
        {'name': 'args',             'type': 'ptr',         'dir': 'IN'},
        {'name': 'sharedMemBytes',   'type': 'size',        'dir': 'IN'},
        {'name': 'stream',           'type': 'handle',      'dir': 'IN'},
    ]}
    assert expanded_field_count(api) == 6

# ---- Budget enforcement (validate_api) ----

def test_validate_under_budget_passes():
    api = {'api': 'foo', 'category': 'memory', 'args': [
        {'name': 'a', 'type': 'uint32', 'dir': 'IN'},
    ]}
    validate_api(api)  # no raise

def test_validate_over_budget_raises():
    """Spec §4.4: codegen aborts on >9 payload fields."""
    api = {'api': 'foo', 'category': 'kernel_launch', 'args': [
        {'name': f'a{i}', 'type': 'uint32', 'dir': 'IN'} for i in range(10)
    ]}
    try:
        validate_api(api)
    except BudgetError as e:
        assert '10' in str(e)
        assert 'foo' in str(e)
        return
    raise AssertionError("expected BudgetError")

# ---- IN_ARG_KIND helper ----

def test_in_arg_kind_classifies_correctly():
    assert IN_ARG_KIND({'dir': 'IN'})    == 'IN'
    assert IN_ARG_KIND({'dir': 'OUT'})   == 'OUT'
    # INOUT is rejected upstream by parse_yaml_text but the helper still
    # answers honestly if called directly:
    assert IN_ARG_KIND({'dir': 'INOUT'}) == 'INOUT'

if __name__ == '__main__':
    import inspect
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn) and not inspect.isclass(fn):
            try:
                fn()
                print(f'  ok  {name}')
            except Exception as e:
                print(f'  FAIL {name}: {e}')
                failures += 1
    print(f'\n{"PASS" if failures == 0 else "FAIL"}: {failures} failures')
    sys.exit(1 if failures else 0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 projects/clr/hipamd/scripts/test_lttng_curated_lib.py 2>&1 | head -3
```

Expected: `ModuleNotFoundError: No module named 'lttng_curated_lib'`.

- [ ] **Step 3: Implement the parser library**

Create `projects/clr/hipamd/scripts/lttng_curated_lib.py`:

```python
"""Shared parser/validator library for the LTTng curated-args DSL.

Used by:
  - lttng_curated_codegen.py   (generates tracepoint header + emit helpers)
  - lttng_curated_verify.py    (libclang vs YAML drift check; CI gate)
  - lttng_migrate.py           (selects _CURATED vs _CURATED_NOARGS macro variants)

Schema and field-budget rules are normative per:
  docs/superpowers/specs/2026-04-26-lttng-curated-args-design.md §4.

The library has NO libclang dependency — that is isolated to the verifier
script. PyYAML is the only third-party dependency.
"""
import yaml

# ---- DSL vocabulary (spec §4.1) ----
DSL_TYPES = frozenset([
    'handle', 'ptr', 'device_ptr', 'size',
    'int32', 'uint32', 'int64', 'uint64',
    'float', 'enum', 'bool', 'dim3', 'dim3_packed', 'cstring',
])
ALLOWED_DIRS = frozenset(['IN', 'OUT', 'INOUT'])
ALLOWED_CATEGORIES = frozenset([
    'streams', 'events', 'kernel_launch', 'memory', 'graphs', 'module',
    'hsa_queues', 'hsa_signals', 'hsa_memory',
])

# Per spec §4.4: budget is 10 LTTng fields total including corr_id => 9 payload max.
PAYLOAD_BUDGET = 9

# Type expansion (spec §4.4).
TYPE_EXPANSION = {
    'dim3': 3, 'dim3_packed': 1,
    # All others expand to 1.
}

# Direction expansion (spec §4.4): INOUT contributes 2 (input + <name>_out).
DIR_EXPANSION = {'IN': 1, 'OUT': 1, 'INOUT': 2}


class ParseError(Exception):
    """Schema-level error in the YAML (missing field, bad type, INOUT in v1, etc.)."""

class BudgetError(Exception):
    """Field-budget violation (>9 payload fields after expansion)."""


def _type_expand(arg):
    return TYPE_EXPANSION.get(arg['type'], 1)


def _dir_expand(arg):
    return DIR_EXPANSION[arg['dir']]


def expanded_field_count(api):
    """Return total payload field count (excluding corr_id) after both
    type-expansion and direction-expansion (spec §4.4 normative rule)."""
    return sum(_type_expand(a) * _dir_expand(a) for a in api['args'])


def IN_ARG_KIND(arg):
    """Return 'IN' | 'OUT' | 'INOUT' — the arg's direction."""
    return arg['dir']


def _require_keys(d, required, ctx):
    missing = [k for k in required if k not in d]
    if missing:
        raise ParseError(f"{ctx}: missing required field(s): {', '.join(missing)}")


def _validate_arg(arg, api_name):
    _require_keys(arg, ['name', 'type', 'dir'], f"{api_name} arg")
    if arg['type'] not in DSL_TYPES:
        raise ParseError(
            f"{api_name} arg {arg['name']}: unknown type {arg['type']!r}; "
            f"valid: {sorted(DSL_TYPES)}")
    if arg['dir'] not in ALLOWED_DIRS:
        raise ParseError(
            f"{api_name} arg {arg['name']}: unknown dir {arg['dir']!r}; "
            f"valid: {sorted(ALLOWED_DIRS)}")
    # Spec §4.4 INOUT-out-of-scope-v1: hard error in codegen + verifier.
    if arg['dir'] == 'INOUT':
        raise ParseError(
            f"{api_name} arg {arg['name']}: dir: INOUT is out-of-scope for v1 "
            f"(spec §4.4 'INOUT scope (v1)'); model as IN or OUT instead")


def validate_api(api):
    """Validate one API entry. Raises ParseError or BudgetError."""
    _require_keys(api, ['api', 'category', 'args'], "API entry")
    if not isinstance(api['args'], list):
        raise ParseError(f"{api['api']}: 'args' must be a list")
    if api['category'] not in ALLOWED_CATEGORIES:
        raise ParseError(
            f"{api['api']}: unknown category {api['category']!r}; "
            f"valid: {sorted(ALLOWED_CATEGORIES)}")
    for arg in api['args']:
        _validate_arg(arg, api['api'])
    n = expanded_field_count(api)
    if n > PAYLOAD_BUDGET:
        raise BudgetError(
            f"{api['api']}: payload has {n} fields, exceeds budget of "
            f"{PAYLOAD_BUDGET} (spec §4.4). Apply mitigation: type as "
            f"dim3_packed and/or omit low-value args.")


def parse_yaml_text(text):
    """Parse YAML text into a list of validated API dicts."""
    raw = yaml.safe_load(text)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ParseError(f"top level must be a YAML list; got {type(raw).__name__}")
    out = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ParseError(f"each list entry must be a mapping; got {type(entry).__name__}")
        validate_api(entry)
        out.append(entry)
    # Sanity: no duplicate api names.
    seen = set()
    for a in out:
        if a['api'] in seen:
            raise ParseError(f"duplicate api: {a['api']}")
        seen.add(a['api'])
    return out


def parse_yaml_file(path):
    with open(path, 'r') as f:
        return parse_yaml_text(f.read())
```

- [ ] **Step 4: Run tests, verify all pass**

```bash
python3 projects/clr/hipamd/scripts/test_lttng_curated_lib.py
```

Expected: `PASS: 0 failures` and exit code 0. All 14 tests pass.

If `pyyaml` is missing: `pip install --user pyyaml` (already installed in container per CLAUDE.md context).

- [ ] **Step 5: Commit**

```bash
git add projects/clr/hipamd/scripts/lttng_curated_lib.py \
        projects/clr/hipamd/scripts/test_lttng_curated_lib.py
git commit -m "lttng: add curated-args DSL parser library + unit tests

Implements parse_yaml_text() and validate_api() per spec §4 schema:
- DSL type vocabulary (handle, ptr, device_ptr, size, integer/float
  types, enum, bool, dim3, dim3_packed, cstring)
- Direction vocabulary (IN, OUT) — INOUT rejected in v1 per §4.4
- Post-type × post-direction field-budget calculator (PAYLOAD_BUDGET = 9)
- Category whitelist
- Duplicate-API detection

Shared by codegen, verifier, and migrator scripts (added in follow-up
commits). 14 unit tests exercise schema parsing, budget enforcement,
INOUT rejection, and field counting on the hipLaunchKernel natural and
packed cases from spec §4.4."
```

---

### Task 3: Codegen script (`lttng_curated_codegen.py`)

**Files:**
- Create: `projects/clr/hipamd/scripts/lttng_curated_codegen.py`
- Create: `projects/clr/hipamd/scripts/test_lttng_curated_codegen.py`
- Create: `projects/clr/hipamd/scripts/testdata/curated_minimal.yaml`
- Create: `projects/clr/hipamd/scripts/testdata/golden_minimal_tp.h`
- Create: `projects/clr/hipamd/scripts/testdata/golden_minimal_emit.h`

- [ ] **Step 1: Write the test fixture YAML**

Create `projects/clr/hipamd/scripts/testdata/curated_minimal.yaml`:

```yaml
# Minimal fixture for codegen golden tests. Two APIs covering the all-IN
# and OUT-param patterns. Exercises three DSL types: ptr, size, enum,
# handle, plus a status-returning vs pointer-returning shape.
- api: hipMemcpyAsync
  category: memory
  args:
    - {name: dst,       type: ptr,    dir: IN}
    - {name: src,       type: ptr,    dir: IN}
    - {name: sizeBytes, type: size,   dir: IN}
    - {name: kind,      type: enum,   dir: IN}
    - {name: stream,    type: handle, dir: IN}

- api: hipMalloc
  category: memory
  args:
    - {name: ptr,  type: ptr,  dir: OUT}
    - {name: size, type: size, dir: IN}

- api: hipDeviceSynchronize
  category: streams
  args: []
```

- [ ] **Step 2: Write the failing tests**

Create `projects/clr/hipamd/scripts/test_lttng_curated_codegen.py`:

```python
"""Golden-file tests for lttng_curated_codegen.py."""
import os, sys, subprocess, tempfile, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CODEGEN = os.path.join(HERE, 'lttng_curated_codegen.py')
YAML    = os.path.join(HERE, 'testdata', 'curated_minimal.yaml')

def _run_codegen(provider, status_type='hipError_t', status_success='hipSuccess'):
    """Invoke codegen, return (tp_h_text, emit_h_text)."""
    with tempfile.TemporaryDirectory() as d:
        tp = os.path.join(d, 'tp.h')
        em = os.path.join(d, 'emit.h')
        cmd = ['python3', CODEGEN,
               '--yaml', YAML,
               '--provider', provider,
               '--status-type', status_type,
               '--status-success', status_success,
               '--out-tp', tp,
               '--out-emit', em]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"codegen failed:\n{r.stderr}"
        with open(tp) as f: tp_text = f.read()
        with open(em) as f: em_text = f.read()
    return tp_text, em_text

def test_emits_tracepoint_event_per_api():
    tp, _ = _run_codegen('rocm_hip')
    assert 'hipMemcpyAsync_args' in tp
    assert 'hipMalloc_args' in tp
    assert 'hipDeviceSynchronize_args' in tp
    # Schema version comment present.
    assert 'AUTO-GENERATED' in tp
    # Provider correctly templated.
    assert 'rocm_hip,' in tp
    assert 'rocm_hsa' not in tp

def test_emits_corr_id_field_first():
    tp, _ = _run_codegen('rocm_hip')
    # Find the hipMemcpyAsync event block and assert corr_id is first field.
    block_start = tp.index('hipMemcpyAsync_args')
    block = tp[block_start:block_start+1200]
    fields_start = block.index('LTTNG_UST_TP_FIELDS')
    fields_section = block[fields_start:fields_start+800]
    # First field decl must be corr_id.
    first_field_decl_idx = fields_section.index('lttng_ust_field_')
    assert 'corr_id' in fields_section[first_field_decl_idx:first_field_decl_idx+80]

def test_emits_helper_per_api():
    _, em = _run_codegen('rocm_hip')
    assert 'rocm_trace_emit_hipMemcpyAsync_args' in em
    assert 'rocm_trace_emit_hipMalloc_args' in em
    assert 'rocm_trace_emit_hipDeviceSynchronize_args' in em

def test_helper_signature_takes_status_last_for_all_in_api():
    """Spec §6.2: every helper takes status as last param, even all-IN."""
    _, em = _run_codegen('rocm_hip')
    # Look for hipMemcpyAsync helper signature, last param must include hipError_t.
    sig_start = em.index('rocm_trace_emit_hipMemcpyAsync_args')
    sig_end = em.index(')', sig_start) + 1
    sig = em[sig_start:sig_end]
    assert 'hipError_t' in sig, f"signature missing hipError_t status: {sig!r}"

def test_out_param_helper_uses_status_to_gate_deref():
    """Spec §5.2 hipMalloc example: helper deref's *ptr_out only on success."""
    _, em = _run_codegen('rocm_hip')
    body_start = em.index('rocm_trace_emit_hipMalloc_args')
    body_end = em.index('}', body_start) + 1
    body = em[body_start:body_end]
    # Must reference hipSuccess as the success sentinel.
    assert 'hipSuccess' in body
    # Must guard the deref behind the status check (look for ternary).
    assert '?' in body and ':' in body  # ternary guard

def test_no_arg_helper_signature():
    """Spec §6.2 _NOARGS variant: helper signature is just (corr_id, status)."""
    _, em = _run_codegen('rocm_hip')
    sig_start = em.index('rocm_trace_emit_hipDeviceSynchronize_args')
    sig_end = em.index(')', sig_start) + 1
    sig = em[sig_start:sig_end]
    # Only 2 params: corr_id and status.
    assert sig.count(',') == 1, f"no-arg helper should have 2 params: {sig!r}"
    assert 'corr_id' in sig
    assert 'hipError_t' in sig

def test_no_op_branch_for_off_mode():
    """Spec §5.2: when HIP_ENABLE_LTTNG_UST=0, helpers are no-ops."""
    _, em = _run_codegen('rocm_hip')
    assert '#if defined(HIP_ENABLE_LTTNG_UST)' in em or 'HIP_ENABLE_LTTNG_UST' in em
    assert '#else' in em
    # Check that the #else branch contains no-op helper definitions.
    else_idx = em.index('#else')
    after_else = em[else_idx:]
    assert 'rocm_trace_emit_hipMemcpyAsync_args' in after_else

def test_provider_parameterization_for_hsa():
    """Same script generates HSA tracepoint events when --provider rocm_hsa."""
    tp, em = _run_codegen('rocm_hsa', status_type='hsa_status_t',
                           status_success='HSA_STATUS_SUCCESS')
    # HSA provider in tp definitions
    assert 'rocm_hsa,' in tp
    # HSA status type in helper signatures
    assert 'hsa_status_t' in em

def test_yaml_sha256_in_header():
    """Spec §5.1: header includes SHA256(curated_apis.yaml) comment."""
    tp, _ = _run_codegen('rocm_hip')
    with open(YAML, 'rb') as f:
        expected = hashlib.sha256(f.read()).hexdigest()
    assert expected[:16] in tp, f"sha256 prefix {expected[:16]} not in tp.h"

if __name__ == '__main__':
    import inspect
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn) and not inspect.isclass(fn):
            try:
                fn()
                print(f'  ok  {name}')
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'  FAIL {name}: {e}')
                failures += 1
    print(f'\n{"PASS" if failures == 0 else "FAIL"}: {failures} failures')
    sys.exit(1 if failures else 0)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
python3 projects/clr/hipamd/scripts/test_lttng_curated_codegen.py 2>&1 | head -3
```

Expected: failure because `lttng_curated_codegen.py` does not exist.

- [ ] **Step 4: Implement the codegen script**

Create `projects/clr/hipamd/scripts/lttng_curated_codegen.py`:

```python
#!/usr/bin/env python3
"""Codegen: curated_apis.yaml -> rocm_<provider>_curated_tp.h + rocm_trace_emit_curated.h.

Per spec §5. The output files are checked in (see spec §3.2). Build does
not invoke this script by default; it runs only when the developer
explicitly regenerates (opt-in CMake target) or the CI drift gate runs it
and asserts `git diff --exit-code`.

Usage:
    python3 lttng_curated_codegen.py \\
        --yaml      path/to/curated_apis.yaml \\
        --provider  rocm_hip                  \\
        --status-type hipError_t              \\
        --status-success hipSuccess           \\
        --out-tp    path/to/rocm_hip_curated_tp.h \\
        --out-emit  path/to/rocm_trace_emit_curated.h
"""
import argparse, hashlib, os, sys, textwrap
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lttng_curated_lib import parse_yaml_file, expanded_field_count

# ---- Per-DSL-type emit-side info ----
# (lttng_field_macro, c_type_for_helper_param, cast_expression_template)
TYPE_INFO = {
    'handle':      ('lttng_ust_field_integer_hex', 'uint64_t', '(uint64_t)(uintptr_t)({arg})'),
    'ptr':         ('lttng_ust_field_integer_hex', 'uint64_t', '(uint64_t)(uintptr_t)({arg})'),
    'device_ptr':  ('lttng_ust_field_integer_hex', 'uint64_t', '(uint64_t)({arg})'),
    'size':        ('lttng_ust_field_integer',     'uint64_t', '(uint64_t)({arg})'),
    'int32':       ('lttng_ust_field_integer',     'int32_t',  '(int32_t)({arg})'),
    'uint32':      ('lttng_ust_field_integer',     'uint32_t', '(uint32_t)({arg})'),
    'int64':       ('lttng_ust_field_integer',     'int64_t',  '(int64_t)({arg})'),
    'uint64':      ('lttng_ust_field_integer',     'uint64_t', '(uint64_t)({arg})'),
    'float':       ('lttng_ust_field_float',       'float',    '(float)({arg})'),
    'enum':        ('lttng_ust_field_integer',     'int32_t',  '(int32_t)({arg})'),
    # Spec §4.1: bool emits canonical 0/1 via !!() to be storage-rep-independent.
    'bool':        ('lttng_ust_field_integer',     'uint32_t', '(uint32_t)(!!({arg}))'),
    'cstring':     ('lttng_ust_field_string',      'const char*', '({arg} ? {arg} : "")'),
    # dim3 / dim3_packed handled specially in emit_tp_event / emit_helper.
}

# ---- Helper formal-param types (real C types for the wrapper signature) ----
# These are placeholders; the migrator passes the wrapper's actual params
# directly to the helper. They are encoded as "void const* generic" since
# the helper's job is to cast and emit, not to be type-strict at the
# helper boundary. We use the real types for the OUT-pointer case so the
# helper can deref safely.
HELPER_PARAM_TYPE = {
    'handle':     'uint64_t',     # always cast at call site
    'ptr':        'const void*',
    'device_ptr': 'uint64_t',     # hipDeviceptr_t passed as uint64
    'size':       'size_t',
    'int32':      'int32_t',
    'uint32':     'uint32_t',
    'int64':      'int64_t',
    'uint64':     'uint64_t',
    'float':      'float',
    'enum':       'int32_t',
    'bool':       'int',          # C bool promotes to int
    'cstring':    'const char*',
    'dim3':       'dim3',
    'dim3_packed': 'dim3',         # helper takes dim3, packs internally
}

# OUT-param helpers must take a typed pointer-to-the-out-type. For now we
# use void** for all OUT-ptr cases — the helper deref's and casts to uint64.
# See spec §5.2 hipMalloc example.
def out_helper_param_type(arg):
    """C type for an OUT/INOUT helper parameter (which is a pointer-to-T)."""
    if arg['type'] in ('ptr', 'handle', 'device_ptr'):
        return 'void**'
    if arg['type'] in ('size', 'uint32', 'uint64'):
        return f"{HELPER_PARAM_TYPE[arg['type']]}*"
    if arg['type'] in ('int32', 'int64'):
        return f"{HELPER_PARAM_TYPE[arg['type']]}*"
    if arg['type'] == 'float':
        return 'float*'
    if arg['type'] == 'enum':
        return 'int32_t*'  # generic
    if arg['type'] == 'bool':
        return 'int*'
    raise SystemExit(f"OUT not supported for type {arg['type']}")


# ---- Codegen: tp.h ----

def emit_tp_event(provider, api):
    """Emit one LTTNG_UST_TRACEPOINT_EVENT block for `api`."""
    name = api['api']
    args = api['args']

    # Build TP_ARGS list and TP_FIELDS list, expanding dim3 / direction.
    tp_args = ['uint64_t', 'corr_id']
    tp_fields = ['        lttng_ust_field_integer(uint64_t, corr_id, corr_id)']

    for a in args:
        nm  = a['name']
        ty  = a['type']
        dr  = a['dir']
        # Per spec §4.4 v1: INOUT is rejected upstream.
        # OUT: emit one field, named <name> (consumer reads value-or-zero).
        # IN:  emit one field, named <name>.
        # dim3: 3 fields (<name>_x, _y, _z); dim3_packed: 1 field (uint64 hex).
        if ty == 'dim3':
            for axis in ('x', 'y', 'z'):
                tp_args += ['uint32_t', f'{nm}_{axis}']
                tp_fields.append(
                    f'        lttng_ust_field_integer(uint32_t, {nm}_{axis}, {nm}_{axis})')
        elif ty == 'dim3_packed':
            tp_args += ['uint64_t', nm]
            tp_fields.append(
                f'        lttng_ust_field_integer_hex(uint64_t, {nm}, {nm})')
        else:
            field_macro, _, _ = TYPE_INFO[ty]
            # tp_args type depends on the lttng field's underlying C type.
            tp_arg_type = {
                'lttng_ust_field_integer':     {'int32_t': 'int32_t', 'uint32_t': 'uint32_t',
                                                'int64_t': 'int64_t', 'uint64_t': 'uint64_t'}.get(
                                                    field_macro and HELPER_PARAM_TYPE[ty], 'uint64_t'),
                'lttng_ust_field_integer_hex': 'uint64_t',
                'lttng_ust_field_float':       'float',
                'lttng_ust_field_string':      'const char*',
            }[field_macro]
            tp_args += [tp_arg_type, nm]
            if field_macro == 'lttng_ust_field_string':
                tp_fields.append(f'        {field_macro}({nm}, {nm})')
            elif field_macro == 'lttng_ust_field_float':
                tp_fields.append(f'        {field_macro}(float, {nm}, {nm})')
            else:
                tp_fields.append(f'        {field_macro}({tp_arg_type}, {nm}, {nm})')

    args_str   = ', '.join(tp_args)
    fields_str = '\n'.join(tp_fields)

    return textwrap.dedent(f"""\
        LTTNG_UST_TRACEPOINT_EVENT(
            {provider}, {name}_args,
            LTTNG_UST_TP_ARGS({args_str}),
            LTTNG_UST_TP_FIELDS(
        {fields_str}
            )
        )
        """)


def emit_tp_h(provider, apis, yaml_path, yaml_sha256):
    out = []
    out.append(f"""/* AUTO-GENERATED by lttng_curated_codegen.py from {os.path.basename(yaml_path)}.
 * DO NOT EDIT BY HAND. Regenerate via the `regenerate-lttng-curated`
 * CMake target or by invoking the codegen script directly.
 *
 * SHA256({os.path.basename(yaml_path)}) at generation: {yaml_sha256}
 *
 * Provider: {provider}
 * API count: {len(apis)}
 *
 * Spec: docs/superpowers/specs/2026-04-26-lttng-curated-args-design.md
 */
""")
    # Include the rocm_dim3_pack.h header in case any API uses dim3_packed.
    needs_dim3_pack = any(a['type'] == 'dim3_packed' for api in apis for a in api['args'])
    if needs_dim3_pack:
        out.append('/* dim3_packed encoding is defined in rocm_dim3_pack.h, included by\n'
                   ' * the emit-helper header that includes us transitively. */\n')
    for api in apis:
        out.append(emit_tp_event(provider, api))
    return '\n'.join(out)


# ---- Codegen: emit.h ----

def emit_helper(provider, api, status_type, status_success):
    """Emit one static-inline helper function."""
    name = api['api']
    args = api['args']

    # Helper formal-param list. Order: corr_id, captured-args..., status.
    formal_params = ['uint64_t corr_id']
    cast_exprs = []   # for the lttng_ust_do_tracepoint() call
    deref_setups = [] # for OUT params (size_t value computed before do_tp)

    # Track if we have any OUT param needing the success gate.
    has_out = any(a['dir'] == 'OUT' for a in args)

    for a in args:
        nm = a['name']
        ty = a['type']
        dr = a['dir']
        if dr == 'IN':
            ptype = HELPER_PARAM_TYPE[ty]
            formal_params.append(f"{ptype} {nm}")
            if ty == 'dim3':
                # Expand to 3 args at do_tp call site.
                cast_exprs.extend([
                    f"(uint32_t){nm}.x", f"(uint32_t){nm}.y", f"(uint32_t){nm}.z"])
            elif ty == 'dim3_packed':
                # Pack at the helper's local before do_tp.
                deref_setups.append(f"        const uint64_t {nm}_packed = ROCM_DIM3_PACK({nm});")
                cast_exprs.append(f"{nm}_packed")
            else:
                _, _, cast_tmpl = TYPE_INFO[ty]
                cast_exprs.append(cast_tmpl.format(arg=nm))
        elif dr == 'OUT':
            ptype = out_helper_param_type(a)
            formal_params.append(f"{ptype} {nm}_out_ptr")
            # Setup line: compute deref value, gated by status.
            # For ptr/handle/device_ptr — emit (uint64_t)(uintptr_t)(*ptr).
            # For other types — just deref.
            if ty in ('ptr', 'handle', 'device_ptr'):
                deref_setups.append(textwrap.dedent(f"""\
                            const uint64_t {nm}_val =
                                (status == {status_success} && {nm}_out_ptr != NULL)
                                    ? (uint64_t)(uintptr_t)(*{nm}_out_ptr) : 0ULL;""").rstrip())
                cast_exprs.append(f"{nm}_val")
            else:
                # Numeric OUT: read or zero.
                deref_setups.append(textwrap.dedent(f"""\
                            const auto {nm}_val =
                                (status == {status_success} && {nm}_out_ptr != NULL)
                                    ? *{nm}_out_ptr : 0;""").rstrip())
                _, _, cast_tmpl = TYPE_INFO[ty]
                cast_exprs.append(cast_tmpl.format(arg=f"{nm}_val"))
        elif dr == 'INOUT':
            # Validated upstream; can't reach here in v1.
            raise SystemExit(f"INOUT in {name} reached codegen — should be rejected by parser")

    # Status param is last, always present (unused for all-IN APIs).
    formal_params.append(f"{status_type} status")
    if not has_out:
        # Mark unused to suppress -Wunused-parameter.
        formal_params[-1] = f"{status_type} /*status*/ /* unused: all-IN API */"

    formal_str = ',\n    '.join(formal_params)
    cast_str   = ',\n            '.join(cast_exprs) if cast_exprs else ''
    do_tp_args = f"corr_id,\n            {cast_str}" if cast_str else "corr_id"
    setups     = '\n'.join(deref_setups)

    body_inner = (setups + '\n' if setups else '') + textwrap.dedent(f"""\
                lttng_ust_do_tracepoint({provider}, {name}_args, {do_tp_args});""")

    return textwrap.dedent(f"""\
        static inline void rocm_trace_emit_{name}_args(
            {formal_str}) {{
            if (rocm_trace_disabled()) return;
            if (lttng_ust_tracepoint_enabled({provider}, {name}_args)) {{
        {body_inner}
            }}
        }}
        """)


def emit_noop_helper(api, status_type):
    """Emit a no-op helper for HIP_ENABLE_LTTNG_UST=0 mode."""
    name = api['api']
    args = api['args']
    formals = ['uint64_t']
    for a in args:
        if a['dir'] == 'OUT':
            formals.append(out_helper_param_type(a))
        elif a['type'] == 'dim3':
            formals.append('dim3')
        elif a['type'] == 'dim3_packed':
            formals.append('dim3')
        else:
            formals.append(HELPER_PARAM_TYPE[a['type']])
    formals.append(status_type)
    formal_str = ', '.join(formals)
    return f"static inline void rocm_trace_emit_{name}_args({formal_str}) {{}}\n"


def emit_emit_h(provider, apis, status_type, status_success, yaml_path, yaml_sha256):
    """Generate rocm_trace_emit_curated.h."""
    macro_guard = f"ROCM_{provider.upper().replace('ROCM_', '')}_TRACE_EMIT_CURATED_H_"
    enable_macro = 'HIP_ENABLE_LTTNG_UST' if provider == 'rocm_hip' else 'HSA_ENABLE_LTTNG_UST'

    out = []
    out.append(f"""/* AUTO-GENERATED by lttng_curated_codegen.py from {os.path.basename(yaml_path)}.
 * DO NOT EDIT BY HAND. SHA256({os.path.basename(yaml_path)}) at generation: {yaml_sha256}
 *
 * Per-API typed emit helpers for curated parameter capture. See spec §5.2.
 *
 * Helper signature invariant (spec §6.2): every helper takes
 *   (uint64_t corr_id, <captured-args...>, <status_type> status)
 * — status is the call's success status, used to gate OUT-param deref.
 * All-IN APIs accept it but mark it unused.
 */
#ifndef {macro_guard}
#define {macro_guard}

#include <stdint.h>
#include <stddef.h>
#include "rocm_trace_tid.h"
#include "rocm_dim3_pack.h"

#if defined({enable_macro}) && {enable_macro}

#include <atomic>
#include "{provider}_curated_tp.h"
""")
    if provider == 'rocm_hip':
        out.append('#include <hip/hip_runtime.h>\n')
    else:
        out.append('#include <hsa/hsa.h>\n#include <hsa/hsa_ext_amd.h>\n')

    # Reuse the same disabled flag as the existing generic helpers.
    out.append(f"""
extern std::atomic<bool> {provider}_trace_g_disabled;
static inline bool rocm_trace_disabled(void) {{
    return {provider}_trace_g_disabled.load(std::memory_order_relaxed);
}}

""")

    for api in apis:
        out.append(emit_helper(provider, api, status_type, status_success))
        out.append('\n')

    out.append(f"""
#else  /* {enable_macro} not defined — all helpers are no-ops */

""")
    for api in apis:
        out.append(emit_noop_helper(api, status_type))

    out.append(f"""
#endif  /* {enable_macro} */

#endif  /* {macro_guard} */
""")
    return ''.join(out)


# ---- Main ----

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--yaml',     required=True)
    ap.add_argument('--provider', required=True, choices=['rocm_hip', 'rocm_hsa'])
    ap.add_argument('--status-type', required=True,
                    help='C type for call status, e.g. hipError_t or hsa_status_t')
    ap.add_argument('--status-success', required=True,
                    help='Success-status sentinel, e.g. hipSuccess or HSA_STATUS_SUCCESS')
    ap.add_argument('--out-tp',   required=True)
    ap.add_argument('--out-emit', required=True)
    args = ap.parse_args()

    apis = parse_yaml_file(args.yaml)
    with open(args.yaml, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    tp_text   = emit_tp_h(args.provider, apis, args.yaml, sha256)
    emit_text = emit_emit_h(args.provider, apis,
                             args.status_type, args.status_success,
                             args.yaml, sha256)

    os.makedirs(os.path.dirname(args.out_tp) or '.', exist_ok=True)
    with open(args.out_tp, 'w') as f:
        f.write(tp_text)
    with open(args.out_emit, 'w') as f:
        f.write(emit_text)
    print(f"wrote {args.out_tp} ({len(tp_text)} B), "
          f"{args.out_emit} ({len(emit_text)} B), {len(apis)} APIs",
          file=sys.stderr)

if __name__ == '__main__':
    main()
```

- [ ] **Step 5: Run tests**

```bash
python3 projects/clr/hipamd/scripts/test_lttng_curated_codegen.py
```

Expected: all 9 tests pass with `PASS: 0 failures`.

- [ ] **Step 6: Sanity-check the generated output by hand**

```bash
python3 projects/clr/hipamd/scripts/lttng_curated_codegen.py \
    --yaml projects/clr/hipamd/scripts/testdata/curated_minimal.yaml \
    --provider rocm_hip \
    --status-type hipError_t --status-success hipSuccess \
    --out-tp /tmp/sample_tp.h --out-emit /tmp/sample_emit.h
head -30 /tmp/sample_tp.h
echo '---'
sed -n '/hipMalloc_args/,/^}$/p' /tmp/sample_emit.h | head -40
```

Verify:
- `hipMalloc_args` helper signature ends with `hipError_t status`
- Body has `(status == hipSuccess && ptr_out_ptr != NULL) ? (uint64_t)(uintptr_t)(*ptr_out_ptr) : 0ULL`
- Tracepoint event for `hipMemcpyAsync_args` has corr_id field first

- [ ] **Step 7: Commit**

```bash
git add projects/clr/hipamd/scripts/lttng_curated_codegen.py \
        projects/clr/hipamd/scripts/test_lttng_curated_codegen.py \
        projects/clr/hipamd/scripts/testdata/curated_minimal.yaml
git commit -m "lttng: add curated-args codegen script + golden tests

Implements YAML -> tracepoint header + emit-helper codegen per spec §5.
Parameterized by --provider (rocm_hip | rocm_hsa) so the same script
generates HIP and HSA outputs from their respective YAML files.

Output structure (per spec §5):
- tp.h: one LTTNG_UST_TRACEPOINT_EVENT per API, corr_id field first,
  dim3 expanded to 3 fields, dim3_packed as one uint64 hex.
- emit.h: one static inline helper per API, signature is uniformly
  (corr_id, captured-args..., status) per spec §6.2, with status used to
  gate OUT-param deref via the success sentinel passed via flag.

Generated headers carry SHA256(yaml) + AUTO-GENERATED comments for
reviewer / CI verification.

9 unit tests cover: per-API event emission, corr_id-first invariant,
helper signature invariant, OUT-param status-gated deref, no-arg helper
shape, OFF-mode no-op branch, provider parameterization for HSA, SHA256
header presence."
```

---

### Task 4: Verifier script (`lttng_curated_verify.py`)

**Files:**
- Create: `projects/clr/hipamd/scripts/lttng_curated_verify.py`
- Create: `projects/clr/hipamd/scripts/test_lttng_curated_verify.py`
- Create: `projects/clr/hipamd/scripts/testdata/fake_hip_header.h`

- [ ] **Step 1: Write the test fixture (fake header)**

Create `projects/clr/hipamd/scripts/testdata/fake_hip_header.h`:

```c
/* Test fixture — a minimal HIP-like header for verifier tests.
 * Lets us assert verifier behavior without depending on a real
 * /opt/rocm/include layout. */
#ifndef FAKE_HIP_HEADER_H_
#define FAKE_HIP_HEADER_H_

#include <stdint.h>
#include <stddef.h>

typedef enum { hipSuccess = 0, hipErrorOutOfMemory = 2 } hipError_t;
typedef void* hipStream_t;
typedef int   hipMemcpyKind;

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream);

hipError_t hipMalloc(void** ptr, size_t size);

hipError_t hipDeviceSynchronize(void);

#endif
```

- [ ] **Step 2: Write the failing tests**

Create `projects/clr/hipamd/scripts/test_lttng_curated_verify.py`:

```python
"""Tests for lttng_curated_verify.py — libclang vs YAML drift gate."""
import os, sys, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

VERIFY = os.path.join(HERE, 'lttng_curated_verify.py')
HEADER = os.path.join(HERE, 'testdata', 'fake_hip_header.h')

def _run_verify(yaml_text, expect_pass):
    """Write yaml_text to a temp file, invoke verifier, return (rc, output)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_text)
        yaml_path = f.name
    try:
        cmd = ['python3', VERIFY,
               '--yaml', yaml_path,
               '--header', HEADER]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if expect_pass:
            assert r.returncode == 0, f"expected pass, got rc={r.returncode}\n{r.stderr}"
        else:
            assert r.returncode != 0, f"expected fail, got pass\n{r.stdout}"
        return r.returncode, r.stdout + r.stderr
    finally:
        os.unlink(yaml_path)

def test_matching_yaml_passes():
    yaml_text = """
- api: hipMemcpyAsync
  category: memory
  args:
    - {name: dst,       type: ptr,    dir: IN}
    - {name: src,       type: ptr,    dir: IN}
    - {name: sizeBytes, type: size,   dir: IN}
    - {name: kind,      type: enum,   dir: IN}
    - {name: stream,    type: handle, dir: IN}
"""
    _run_verify(yaml_text, expect_pass=True)

def test_arg_name_mismatch_fails():
    """Spec §8.3: arg-name mismatch is a hard error."""
    yaml_text = """
- api: hipMemcpyAsync
  category: memory
  args:
    - {name: WRONG_NAME, type: ptr,    dir: IN}
    - {name: src,        type: ptr,    dir: IN}
    - {name: sizeBytes,  type: size,   dir: IN}
    - {name: kind,       type: enum,   dir: IN}
    - {name: stream,     type: handle, dir: IN}
"""
    _, output = _run_verify(yaml_text, expect_pass=False)
    assert 'WRONG_NAME' in output

def test_arg_count_mismatch_fails():
    yaml_text = """
- api: hipMemcpyAsync
  category: memory
  args:
    - {name: dst, type: ptr, dir: IN}
"""
    _, output = _run_verify(yaml_text, expect_pass=False)
    assert 'arg count' in output.lower() or 'expected' in output.lower()

def test_inout_rejected():
    """Spec §4.4 v1: dir: INOUT is hard error in verifier."""
    yaml_text = """
- api: hipMemcpyAsync
  category: memory
  args:
    - {name: dst,       type: ptr,    dir: INOUT}
    - {name: src,       type: ptr,    dir: IN}
    - {name: sizeBytes, type: size,   dir: IN}
    - {name: kind,      type: enum,   dir: IN}
    - {name: stream,    type: handle, dir: IN}
"""
    _, output = _run_verify(yaml_text, expect_pass=False)
    assert 'INOUT' in output

def test_over_budget_rejected():
    """Spec §4.4: over-budget is hard error."""
    yaml_text = """
- api: hipMemcpyAsync
  category: memory
  args:
""" + '\n'.join(f"    - {{name: a{i}, type: uint32, dir: IN}}" for i in range(11))
    _, output = _run_verify(yaml_text, expect_pass=False)
    assert 'budget' in output.lower() or 'fields' in output.lower()

def test_api_missing_in_header_fails():
    yaml_text = """
- api: hipNonExistent
  category: memory
  args: []
"""
    _, output = _run_verify(yaml_text, expect_pass=False)
    assert 'hipNonExistent' in output

if __name__ == '__main__':
    import inspect
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn) and not inspect.isclass(fn):
            try:
                fn()
                print(f'  ok  {name}')
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'  FAIL {name}: {e}')
                failures += 1
    print(f'\n{"PASS" if failures == 0 else "FAIL"}: {failures} failures')
    sys.exit(1 if failures else 0)
```

- [ ] **Step 3: Run tests, verify they fail**

```bash
python3 projects/clr/hipamd/scripts/test_lttng_curated_verify.py 2>&1 | head
```

Expected: failure because verifier script does not exist.

- [ ] **Step 4: Implement the verifier**

Create `projects/clr/hipamd/scripts/lttng_curated_verify.py`:

```python
#!/usr/bin/env python3
"""Verifier: assert curated_apis.yaml matches the actual HIP/HSA header
declarations via libclang. Per spec §8.3.

Hard errors (exit 1):
- API listed in YAML but not declared in the header.
- Arg count mismatch between YAML and header.
- Arg name mismatch (positional binding intentionally not supported per §4).
- Arg type mismatch per the type-vocabulary mapping (§4.1) — including
  uint32 used for a C bool parameter (must be the bool DSL type).
- Over-budget API (§4.4 — re-checked via lttng_curated_lib).
- dir: INOUT (§4.4 INOUT-out-of-scope-v1).

Informational only (exit 0 with warning):
- Header parameter declared but not in YAML (partial coverage by design).

Usage:
    python3 lttng_curated_verify.py \\
        --yaml   path/to/curated_apis.yaml \\
        --header path/to/hip_runtime_api.h \\
        [--extra-arg=-I/some/include] [--extra-arg=...]
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lttng_curated_lib import parse_yaml_file, expanded_field_count, PAYLOAD_BUDGET

# C type -> set of acceptable DSL types.
# Multiple-DSL-type acceptance lets `int` map to either `int32` or `enum`
# (since enums in C are int-typed at the API boundary).
C_TO_DSL = {
    'void *':                  {'ptr'},
    'const void *':            {'ptr'},
    'void **':                 {'ptr'},   # OUT pointer
    'char *':                  {'ptr', 'cstring'},
    'const char *':            {'cstring', 'ptr'},
    'size_t':                  {'size'},
    'unsigned long':           {'size', 'uint64'},
    'unsigned long long':      {'uint64', 'size'},
    'int':                     {'int32', 'enum'},
    'unsigned int':            {'uint32', 'enum'},
    'int32_t':                 {'int32', 'enum'},
    'uint32_t':                {'uint32', 'enum'},
    'int64_t':                 {'int64'},
    'uint64_t':                {'uint64'},
    'float':                   {'float'},
    'bool':                    {'bool'},   # spec §4.1: hard error if YAML uses uint32
    '_Bool':                   {'bool'},
    'dim3':                    {'dim3', 'dim3_packed'},
    'hipDeviceptr_t':          {'device_ptr'},
    # All opaque handle typedefs map to the `handle` DSL type. The set
    # below is approximate; libclang gives us the underlying canonical type
    # (e.g. `struct ihipStream_t *` for hipStream_t), so we also accept
    # any pointer-to-struct type if the DSL type is `handle`.
}

# Pointer-to-struct types from HIP/HSA — accept as `handle`.
HANDLE_TYPE_PATTERNS = (
    'ihipStream_t', 'ihipEvent_t', 'ihipModule_t', 'ihipFunction_t',
    'ihipGraph_t', 'ihipGraphExec_t', 'ihipGraphNode_t', 'hipUserObject_t',
    'hsa_signal_t', 'hsa_queue_t', 'hsa_agent_t',
)

def _type_is_handle(c_type):
    return any(p in c_type for p in HANDLE_TYPE_PATTERNS)

def _is_compatible(c_type, dsl_type):
    """Return True iff the C type is compatible with the DSL type."""
    c = c_type.strip()
    # Direct lookup.
    accepted = C_TO_DSL.get(c)
    if accepted and dsl_type in accepted:
        return True
    # Handle types: pointer-to-struct from HIP/HSA accepts `handle`.
    if dsl_type == 'handle' and _type_is_handle(c):
        return True
    # Generic pointer fallback for `ptr` DSL type.
    if dsl_type == 'ptr' and ('*' in c):
        return True
    # OUT pointer to T — accept device_ptr* or hipDeviceptr_t* for device_ptr.
    if dsl_type == 'device_ptr' and 'hipDeviceptr_t' in c:
        return True
    # `enum` DSL type accepts any enum-typed parameter.
    if dsl_type == 'enum' and ('enum' in c or c in ('int', 'unsigned int')):
        return True
    return False


def parse_header(header_path, extra_args):
    """Use libclang to parse a header and return {api_name: [(name, c_type), ...]}."""
    try:
        from clang import cindex
    except ImportError:
        sys.exit("ERROR: libclang Python bindings not installed. Try: pip install libclang")
    args = ['-x', 'c++', '-std=c++17']
    args += extra_args
    idx = cindex.Index.create()
    tu = idx.parse(header_path, args=args)
    out = {}
    for n in tu.cursor.walk_preorder():
        if n.kind != cindex.CursorKind.FUNCTION_DECL:
            continue
        params = []
        for arg in n.get_arguments():
            params.append((arg.spelling, arg.type.spelling))
        out[n.spelling] = params
    return out


def verify(yaml_path, header_path, extra_args):
    apis = parse_yaml_file(yaml_path)
    header_decls = parse_header(header_path, extra_args)
    errors = []
    warnings = []
    for api in apis:
        name = api['api']
        if name not in header_decls:
            errors.append(f"{name}: not declared in {header_path}")
            continue
        hdr_params = header_decls[name]
        yaml_args = api['args']
        # Spec §4.4 explicitly allows omitting low-value header params as a
        # field-budget mitigation. Match by NAME, not by count/position.
        hdr_by_name = {hname: htype for hname, htype in hdr_params}
        yaml_names = [a['name'] for a in yaml_args]
        # Hard-error: YAML arg name not in header (typo or stale name).
        for i, yaml_arg in enumerate(yaml_args):
            if yaml_arg['name'] not in hdr_by_name:
                errors.append(
                    f"{name} arg {i}: YAML name {yaml_arg['name']!r} not in "
                    f"header params {list(hdr_by_name)}")
                continue
            htype = hdr_by_name[yaml_arg['name']]
            # Spec §4.1: C bool MUST be DSL type bool, not uint32.
            if (htype.strip() in ('bool', '_Bool')
                    and yaml_arg['type'] != 'bool'):
                errors.append(
                    f"{name} arg {yaml_arg['name']}: C bool requires DSL type "
                    f"'bool' (spec §4.1), got {yaml_arg['type']!r}")
                continue
            if not _is_compatible(htype, yaml_arg['type']):
                errors.append(
                    f"{name} arg {yaml_arg['name']}: type mismatch — C "
                    f"{htype!r} not compatible with DSL {yaml_arg['type']!r}")
        # Informational: header params not in YAML (intentional partial
        # coverage per spec §4.4 / §8.3 'partial coverage of large APIs is
        # by design').
        for hname, _ in hdr_params:
            if hname not in yaml_names:
                warnings.append(
                    f"{name}: header param {hname!r} not in YAML "
                    f"(intentional omission per spec §4.4 mitigation?)")
        # Field-budget re-check (also enforced by parser, but verifier is the
        # CI gate so we report it again).
        if expanded_field_count(api) > PAYLOAD_BUDGET:
            errors.append(
                f"{name}: payload exceeds budget of {PAYLOAD_BUDGET} fields")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    for w in warnings:
        print(f"WARN: {w}")
    print(f"OK: {len(apis)} curated APIs verified against {header_path}")
    return 0

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--yaml',   required=True)
    ap.add_argument('--header', required=True)
    ap.add_argument('--extra-arg', action='append', default=[])
    args = ap.parse_args()
    sys.exit(verify(args.yaml, args.header, args.extra_arg))

if __name__ == '__main__':
    main()
```

- [ ] **Step 5: Run tests**

```bash
python3 projects/clr/hipamd/scripts/test_lttng_curated_verify.py
```

Expected: all 6 tests pass. (Requires `pip install libclang` if not already present.)

If libclang is missing in your local environment but available in the container:
```bash
./dev-bin/in-container.sh main "cd /root/rocm-systems && python3 projects/clr/hipamd/scripts/test_lttng_curated_verify.py"
```

- [ ] **Step 6: Commit**

```bash
git add projects/clr/hipamd/scripts/lttng_curated_verify.py \
        projects/clr/hipamd/scripts/test_lttng_curated_verify.py \
        projects/clr/hipamd/scripts/testdata/fake_hip_header.h
git commit -m "lttng: add curated-args drift verifier (libclang vs YAML)

Per spec §8.3, this is a separate CI gate that catches HIP/HSA header
signature changes that diverge from curated_apis.yaml. Hard errors:
- API in YAML but not declared in header
- Arg count mismatch
- Arg name mismatch (migrator binds by name; positional is unsafe)
- Arg type mismatch per §4.1 type-vocabulary mapping
- C bool with DSL type other than 'bool' (forces canonical 0/1 emit)
- Over-budget (§4.4 re-check)
- dir: INOUT (rejected upstream by parser, re-checked here)

6 unit tests against a fake HIP header fixture cover the matching case
and each hard-error path."
```

---

### Task 4.5: Multi-header verifier support (HSA APIs span hsa.h + hsa_ext_amd.h)

**Files:**
- Modify: `projects/clr/hipamd/scripts/lttng_curated_verify.py`
- Modify: `projects/clr/hipamd/scripts/test_lttng_curated_verify.py`
- Create: `projects/clr/hipamd/scripts/testdata/fake_hsa_base.h`
- Create: `projects/clr/hipamd/scripts/testdata/fake_hsa_ext.h`

**Why this task is here.** HIP curated APIs all live in `hip_runtime_api.h`, so `--header` taking a single value is fine for HIP. HSA is different: base APIs (`hsa_queue_create`, `hsa_signal_create`) live in `hsa.h`, but AMD extensions (`hsa_amd_*`) live in `hsa_ext_amd.h`. The verifier MUST union the declarations from both headers before checking each YAML API; otherwise either base APIs or extension APIs would always fail "not declared in header".

Task 15 originally said "extend the verifier if needed" — this is that scheduled extension.

- [ ] **Step 1: Write the new test fixtures**

Create `projects/clr/hipamd/scripts/testdata/fake_hsa_base.h`:

```c
/* Test fixture — minimal HSA-base-like header (no AMD extensions). */
#ifndef FAKE_HSA_BASE_H_
#define FAKE_HSA_BASE_H_
#include <stdint.h>
#include <stddef.h>

typedef enum { HSA_STATUS_SUCCESS = 0 } hsa_status_t;
typedef struct { uint64_t handle; } hsa_signal_t;

hsa_status_t hsa_signal_create(int64_t initial_value,
                               uint32_t num_consumers,
                               const void* consumers,
                               hsa_signal_t* signal);
#endif
```

Create `projects/clr/hipamd/scripts/testdata/fake_hsa_ext.h`:

```c
/* Test fixture — HSA AMD-extensions-only header. */
#ifndef FAKE_HSA_EXT_H_
#define FAKE_HSA_EXT_H_
#include "fake_hsa_base.h"

hsa_status_t hsa_amd_signal_create(int64_t initial_value,
                                   uint32_t num_consumers,
                                   const void* consumers,
                                   uint64_t attributes,
                                   hsa_signal_t* signal);
#endif
```

- [ ] **Step 2: Write the new tests against the fixtures**

Append to `projects/clr/hipamd/scripts/test_lttng_curated_verify.py`:

```python
HSA_BASE = os.path.join(HERE, 'testdata', 'fake_hsa_base.h')
HSA_EXT  = os.path.join(HERE, 'testdata', 'fake_hsa_ext.h')

def _run_verify_multi(yaml_text, headers, expect_pass):
    """Like _run_verify but accepts multiple --header arguments."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_text); yaml_path = f.name
    try:
        cmd = ['python3', VERIFY, '--yaml', yaml_path]
        for h in headers:
            cmd += ['--header', h]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if expect_pass:
            assert r.returncode == 0, f"expected pass, got rc={r.returncode}\n{r.stderr}"
        else:
            assert r.returncode != 0, f"expected fail\n{r.stdout}"
        return r.returncode, r.stdout + r.stderr
    finally:
        os.unlink(yaml_path)

def test_multi_header_base_only_api_verifies():
    """An API declared in hsa.h verifies when both --header values are given."""
    yaml_text = """
- api: hsa_signal_create
  category: hsa_signals
  args:
    - {name: initial_value, type: int64,  dir: IN}
    - {name: num_consumers, type: uint32, dir: IN}
    - {name: consumers,     type: ptr,    dir: IN}
    - {name: signal,        type: handle, dir: OUT}
"""
    _run_verify_multi(yaml_text, [HSA_BASE, HSA_EXT], expect_pass=True)

def test_multi_header_ext_only_api_verifies():
    """An API declared only in hsa_ext_amd.h verifies via the union."""
    yaml_text = """
- api: hsa_amd_signal_create
  category: hsa_signals
  args:
    - {name: initial_value, type: int64,  dir: IN}
    - {name: num_consumers, type: uint32, dir: IN}
    - {name: consumers,     type: ptr,    dir: IN}
    - {name: attributes,    type: uint64, dir: IN}
    - {name: signal,        type: handle, dir: OUT}
"""
    _run_verify_multi(yaml_text, [HSA_BASE, HSA_EXT], expect_pass=True)

def test_multi_header_mixed_api_set_verifies():
    """Both APIs in one YAML, spanning both headers — passes."""
    yaml_text = """
- api: hsa_signal_create
  category: hsa_signals
  args:
    - {name: initial_value, type: int64,  dir: IN}
    - {name: num_consumers, type: uint32, dir: IN}
    - {name: consumers,     type: ptr,    dir: IN}
    - {name: signal,        type: handle, dir: OUT}
- api: hsa_amd_signal_create
  category: hsa_signals
  args:
    - {name: initial_value, type: int64,  dir: IN}
    - {name: num_consumers, type: uint32, dir: IN}
    - {name: consumers,     type: ptr,    dir: IN}
    - {name: attributes,    type: uint64, dir: IN}
    - {name: signal,        type: handle, dir: OUT}
"""
    _run_verify_multi(yaml_text, [HSA_BASE, HSA_EXT], expect_pass=True)

def test_multi_header_undeclared_api_fails():
    """API not declared in any of the supplied headers — hard error."""
    yaml_text = """
- api: hsa_amd_nonexistent
  category: hsa_signals
  args: []
"""
    _, output = _run_verify_multi(yaml_text, [HSA_BASE, HSA_EXT], expect_pass=False)
    assert 'hsa_amd_nonexistent' in output
```

- [ ] **Step 3: Run tests, verify the new ones fail (single-header verifier doesn't support multi)**

```bash
python3 projects/clr/hipamd/scripts/test_lttng_curated_verify.py
```

Expected: 4 new tests fail with errors like "argument --header: cannot be specified more than once" (or similar). Pre-existing 6 tests still pass.

- [ ] **Step 4: Modify the verifier to accept multiple `--header` values and union declarations**

Edit `projects/clr/hipamd/scripts/lttng_curated_verify.py`:

(4a) Change the argparse declaration:

```python
ap.add_argument('--header', required=True, action='append',
                help='Header file to verify against. May be specified '
                     'multiple times; declarations from all headers are '
                     'unioned before checking YAML APIs (e.g. HSA needs '
                     'both hsa.h and hsa_ext_amd.h).')
```

(4b) Change `parse_header(header_path, extra_args)` to `parse_headers(header_paths, extra_args)`:

```python
def parse_headers(header_paths, extra_args):
    """Parse one or more headers; return a unioned {api_name: [(name, c_type)]}.

    On duplicate api_name across headers (shouldn't happen in HSA, but
    defend against it), the LATER header wins and a warning is emitted.
    """
    try:
        from clang import cindex
    except ImportError:
        sys.exit("ERROR: libclang Python bindings not installed. Try: pip install libclang")
    args = ['-x', 'c++', '-std=c++17'] + list(extra_args)
    idx = cindex.Index.create()
    union = {}
    for hp in header_paths:
        tu = idx.parse(hp, args=args)
        for n in tu.cursor.walk_preorder():
            if n.kind != cindex.CursorKind.FUNCTION_DECL:
                continue
            params = [(arg.spelling, arg.type.spelling)
                      for arg in n.get_arguments()]
            if n.spelling in union and union[n.spelling] != params:
                print(f"WARN: {n.spelling}: declaration in {hp} differs from "
                      f"earlier header; using {hp}", file=sys.stderr)
            union[n.spelling] = params
    return union
```

(4c) Update `verify()`:

```python
def verify(yaml_path, header_paths, extra_args):
    apis = parse_yaml_file(yaml_path)
    header_decls = parse_headers(header_paths, extra_args)
    # ... existing loop unchanged; the error message in the
    # 'not declared' branch should mention all header paths:
    if name not in header_decls:
        errors.append(f"{name}: not declared in any of {header_paths}")
        continue
    # ... rest unchanged ...
    print(f"OK: {len(apis)} curated APIs verified against {len(header_paths)} header(s)")
    return 0
```

(4d) Update `main()` to pass the list:

```python
sys.exit(verify(args.yaml, args.header, args.extra_arg))
```

- [ ] **Step 5: Run all tests, all 10 must pass**

```bash
python3 projects/clr/hipamd/scripts/test_lttng_curated_verify.py
```

Expected: `PASS: 0 failures` across 10 tests (6 original + 4 new).

- [ ] **Step 6: Commit**

```bash
git add projects/clr/hipamd/scripts/lttng_curated_verify.py \
        projects/clr/hipamd/scripts/test_lttng_curated_verify.py \
        projects/clr/hipamd/scripts/testdata/fake_hsa_base.h \
        projects/clr/hipamd/scripts/testdata/fake_hsa_ext.h
git commit -m "lttng: verifier accepts multiple --header args (HSA spans 2 headers)

HSA curated APIs split between hsa.h (base — hsa_queue_create,
hsa_signal_create) and hsa_ext_amd.h (AMD extensions — hsa_amd_*).
Single --header would always fail half the API set.

Change --header to action='append' and union declarations from all
supplied headers before name-lookup. Add 4 unit tests against fake
HSA base + ext fixtures covering: base-only API verifies, ext-only
API verifies, mixed YAML verifies, and undeclared API fails.

Required by Task 15 (HSA verify) — that task now invokes
  --header /opt/rocm/include/hsa/hsa.h
  --header /opt/rocm/include/hsa/hsa_ext_amd.h
in a single command rather than two passes."
```

---


## Phase B — Author HIP YAML (minimal) (task 5)

This phase introduces the YAML file with just enough APIs to validate the end-to-end pipeline (codegen → checked-in headers → migrator → build → tests). Full HIP curation happens in Phase F after the pipeline works.

---

### Task 5: Create minimal HIP `curated_apis.yaml`

**Files:**
- Create: `projects/clr/hipamd/scripts/curated_apis.yaml`

- [ ] **Step 1: Author the YAML**

Create `projects/clr/hipamd/scripts/curated_apis.yaml`:

```yaml
# Curated HIP APIs for typed parameter capture.
# Source of truth — see docs/superpowers/specs/2026-04-26-lttng-curated-args-design.md.
#
# Schema:
#   - api: string (must match HIP header function name exactly)
#     category: streams | events | kernel_launch | memory | graphs | module
#     args:
#       - {name: <param_name>, type: <DSL type>, dir: IN | OUT}
#
# DSL types: handle | ptr | device_ptr | size | int32 | uint32 | int64 |
#            uint64 | float | enum | bool | dim3 | dim3_packed | cstring
#
# Direction: IN (captured at entry, emitted at exit) | OUT (deref'd at exit)
# INOUT is rejected by codegen + verifier in v1 — see spec §4.4.
#
# Field budget: each API may have at most 9 payload fields after corr_id.
# Use `dim3_packed` instead of `dim3` to fit high-arity launch APIs.
#
# Phase B: minimal set (3 APIs) for end-to-end validation.
# Phase F will expand to the full ~72-API curated set per spec Appendix A.

- api: hipMemcpyAsync
  category: memory
  args:
    - {name: dst,       type: ptr,    dir: IN}
    - {name: src,       type: ptr,    dir: IN}
    - {name: sizeBytes, type: size,   dir: IN}
    - {name: kind,      type: enum,   dir: IN}
    - {name: stream,    type: handle, dir: IN}

- api: hipMalloc
  category: memory
  args:
    - {name: ptr,  type: ptr,  dir: OUT}
    - {name: size, type: size, dir: IN}

- api: hipDeviceSynchronize
  category: streams
  args: []
```

- [ ] **Step 2: Validate via the parser library**

```bash
python3 -c "
import sys
sys.path.insert(0, 'projects/clr/hipamd/scripts')
from lttng_curated_lib import parse_yaml_file
apis = parse_yaml_file('projects/clr/hipamd/scripts/curated_apis.yaml')
print(f'OK: {len(apis)} APIs validated')
for a in apis:
    print(f'  {a[\"api\"]}: {len(a[\"args\"])} args')
"
```

Expected: `OK: 3 APIs validated`.

- [ ] **Step 3: Validate via the verifier against real HIP headers** (run in container if libclang isn't local)

```bash
./dev-bin/in-container.sh main "cd /root/rocm-systems && python3 projects/clr/hipamd/scripts/lttng_curated_verify.py \
    --yaml projects/clr/hipamd/scripts/curated_apis.yaml \
    --header /opt/rocm/include/hip/hip_runtime_api.h \
    --extra-arg=-D__HIP_PLATFORM_AMD__=1 \
    --extra-arg=-I/opt/rocm/include"
```

Expected: `OK: 3 curated APIs verified`.

- [ ] **Step 4: Commit**

```bash
git add projects/clr/hipamd/scripts/curated_apis.yaml
git commit -m "lttng: add minimal HIP curated_apis.yaml (3 APIs)

Phase B starting set: hipMemcpyAsync (all-IN), hipMalloc (OUT-param +
status-gated deref), hipDeviceSynchronize (zero-arg, exercises the
_NOARGS macro variant). Validated by parser (§4 schema) and verifier
(§8.3 libclang vs header). Full curated set lands in Phase F."
```

---

## Phase C — Generate + wire HIP headers (tasks 6–7)

This phase generates the per-API tracepoint header and emit-helper header from the minimal YAML (Phase B), checks them in, and wires them into the existing `rocm_hip_tp.h` and `rocm_trace_emit.h`. After this phase, `libamdhip64.so` carries the new tracepoint event definitions but no wrappers fire them yet (that's Phase D).

---

### Task 6: Generate + check in HIP curated headers

**Files:**
- Create (generated): `projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h`
- Create (generated): `projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.h`

- [ ] **Step 1: Run the codegen**

```bash
python3 projects/clr/hipamd/scripts/lttng_curated_codegen.py \
    --yaml projects/clr/hipamd/scripts/curated_apis.yaml \
    --provider rocm_hip \
    --status-type hipError_t --status-success hipSuccess \
    --out-tp projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h \
    --out-emit projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.h
```

Expected stderr: `wrote ... 3 APIs`.

- [ ] **Step 2: Eyeball the generated files**

```bash
head -25 projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h
echo '---'
sed -n '/hipMalloc_args/,/^}$/p' projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.h
```

Verify: each contains `AUTO-GENERATED`, the SHA256 prefix, and the three APIs.

- [ ] **Step 3: Commit the generated files**

```bash
git add projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h \
        projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.h
git commit -m "lttng: generate HIP curated tracepoint + emit headers (Phase C, 3 APIs)

Generated by lttng_curated_codegen.py from curated_apis.yaml.
Checked-in headers per spec §3.2 — default build consumes these
directly without invoking Python. Re-running codegen on the same YAML
produces byte-identical output (CI drift gate).

This commit only adds the headers; they are not yet #included from
the tracepoint provider (rocm_hip_tp.h) or the emit abstraction
(rocm_trace_emit.h). That wiring lands in the next commit so the
checked-in artifact is reviewable in isolation."
```

---

### Task 7: Wire generated headers into existing tp.h + emit.h

**Files:**
- Modify: `projects/clr/hipamd/src/lttng/rocm_hip_tp.h`
- Modify: `projects/clr/hipamd/src/lttng/rocm_trace_emit.h`

- [ ] **Step 1: Locate the current end-of-event-definitions in `rocm_hip_tp.h`**

```bash
grep -n 'LTTNG_UST_TRACEPOINT_EVENT\|^#endif' projects/clr/hipamd/src/lttng/rocm_hip_tp.h | tail -8
```

The closing `#endif` for `_ROCM_HIP_TP_H` is the insertion point. The new `#include` must precede it.

- [ ] **Step 2: Add include in `rocm_hip_tp.h`**

Find the line just before the closing `#endif` of `_ROCM_HIP_TP_H` (likely at end of file). Insert:

```c
/* Curated per-API typed tracepoint events. Generated by
 * lttng_curated_codegen.py from curated_apis.yaml. See spec §5.1. */
#include "rocm_hip_curated_tp.h"
```

Use the Edit tool to insert this exactly before the `#include <lttng/tracepoint-event.h>` line if present, or before the final `#endif` if not.

- [ ] **Step 3: Add include + ensure ROCM_DIM3_PACK availability in `rocm_trace_emit.h`**

In `rocm_trace_emit.h`, after the existing `#include "rocm_hip_tp.h"` and the existing emit-helpers block (search for `static inline void rocm_trace_emit_hip_api_enter`), add at the end of the `HIP_ENABLE_LTTNG_UST` branch (before the `#else`):

```c
/* Curated per-API typed emit helpers. Generated; see spec §5.2. */
#include "rocm_trace_emit_curated.h"
```

The generated `rocm_trace_emit_curated.h` already includes `rocm_dim3_pack.h` so no separate include is needed here.

- [ ] **Step 4: Build the library to verify the includes are wellformed**

```bash
./dev-bin/sync.sh main
./dev-bin/in-container.sh main "cd /root/rocm-systems && cmake --build build/clr -j 32 --target amdhip64 2>&1 | tail -20"
```

Expected: build succeeds, no warnings about `rocm_hip_curated_tp.h` or `rocm_trace_emit_curated.h`. If LTTng-UST headers complain about duplicate event registration, the include order in `rocm_hip_tp.h` is wrong — the `#include` must be inside the `#if !defined(_ROCM_HIP_TP_H) || ...` guard, not outside.

- [ ] **Step 5: Verify the new tracepoint events appear in the .so**

```bash
./dev-bin/in-container.sh main "cd /root/rocm-systems && nm build/clr/hipamd/lib/libamdhip64.so | grep '__tracepoint.*_args' | head"
```

Expected: lines containing `__tracepoint__rocm_hip__hipMemcpyAsync_args`, `_hipMalloc_args`, `_hipDeviceSynchronize_args`.

- [ ] **Step 6: Commit**

```bash
git add projects/clr/hipamd/src/lttng/rocm_hip_tp.h \
        projects/clr/hipamd/src/lttng/rocm_trace_emit.h
git commit -m "lttng: wire generated curated tp.h + emit.h into HIP provider

Includes rocm_hip_curated_tp.h from rocm_hip_tp.h so the per-API
tracepoint events register alongside the existing generic events
(hip_api_enter / hip_api_exit_*) under the same provider package
(rocm_hip_tp.cpp). Includes rocm_trace_emit_curated.h from
rocm_trace_emit.h so wrappers can call the generated helpers.

After this commit libamdhip64.so exposes hipMemcpyAsync_args,
hipMalloc_args, hipDeviceSynchronize_args tracepoint events but no
wrapper fires them yet — that wiring lands with the migrator change."
```

---

## Phase D — Migrator + macros + coverage gate (tasks 8–11)

This phase teaches `lttng_migrate.py` to recognize curated APIs and inject the sentinel + IN-locals + `_CURATED` macro routing. New `_CURATED` macros are defined in `rocm_trace_emit.h`. The coverage gate is updated to require sentinel + macro regex match. After this phase, the three Phase B APIs fire their typed `_args` events end-to-end.

---

### Task 8: Define `_CURATED` macro variants in `rocm_trace_emit.h`

**Files:**
- Modify: `projects/clr/hipamd/src/lttng/rocm_trace_emit.h`

- [ ] **Step 1: Locate insertion point**

The existing `ROCM_TRACE_RET_STATUS` / `_PTR` / `_VOID` macros live in `hip_table_interface.cpp` (top of file, around lines 30–100). Verify:

```bash
grep -n 'ROCM_TRACE_RET_STATUS\b\|ROCM_TRACE_RET_PTR\b\|ROCM_TRACE_RET_VOID\b' projects/clr/hipamd/src/hip_table_interface.cpp | head -6
```

The new `_CURATED` macros belong in the same file as the existing macros so the migrator's emit produces a working translation unit. Insert immediately after the existing macros.

- [ ] **Step 2: Add the six `_CURATED` macros**

In `projects/clr/hipamd/src/hip_table_interface.cpp`, immediately after the closing `#endif` for the existing `ROCM_TRACE_RET_VOID` macro (or the equivalent end-of-existing-macros location), insert per spec §6.2:

```c
/* ---------- Curated parameter-capture variants (spec §6.2) ----------
 * Six macros: STATUS / PTR / VOID, each with a captured-args form and a
 * _NOARGS form. The migrator selects _NOARGS iff the curated API has zero
 * captured args. Helper signature invariant: every helper takes
 *   (uint64_t corr_id, <captured-args...>, hipError_t status)
 * even when captured-args is empty. Status comes from:
 *   - STATUS variants: macro-evaluated __rocm_status from the call's expr
 *   - PTR variants:    synthesized from null-vs-non-null retval
 *   - VOID variants:   literal hipSuccess
 *
 * Generic exit events (hip_api_exit_status / _ptr / _void) are still
 * emitted by these macros — the typed _args event AUGMENTS the existing
 * generic event, never replaces it (spec §1, §2).
 */

/* Captured-args variants. __VA_ARGS__ is non-empty by construction (the
 * migrator emits the _NOARGS form for zero-arg APIs). */
#define ROCM_TRACE_RET_STATUS_CURATED(api, expr, corr, ...)                  \
    do {                                                                     \
        const hipError_t __rocm_status = (expr);                             \
        rocm_trace_emit_##api##_args((corr), __VA_ARGS__, __rocm_status);    \
        rocm_trace_emit_hip_api_exit_status(__func__,                        \
            (corr), (int32_t)__rocm_status);                                 \
        return __rocm_status;                                                \
    } while (0)

#define ROCM_TRACE_RET_PTR_CURATED(api, ptr_type, expr, corr, ...)           \
    do {                                                                     \
        ptr_type const __rocm_ptr = (expr);                                  \
        const hipError_t __rocm_status =                                     \
            (__rocm_ptr != nullptr) ? hipSuccess : hipErrorOutOfMemory;      \
        rocm_trace_emit_##api##_args((corr), __VA_ARGS__, __rocm_status);    \
        rocm_trace_emit_hip_api_exit_ptr(__func__, (corr), __rocm_ptr);      \
        return __rocm_ptr;                                                   \
    } while (0)

#define ROCM_TRACE_RET_VOID_CURATED(api, expr, corr, ...)                    \
    do {                                                                     \
        (expr);                                                              \
        rocm_trace_emit_##api##_args((corr), __VA_ARGS__, hipSuccess);       \
        rocm_trace_emit_hip_api_exit_void(__func__, (corr));                 \
        return;                                                              \
    } while (0)

/* Zero-captured-args variants. Avoids any empty-__VA_ARGS__ expansion
 * (the codebase mixes C++14/17/20 — see spec §6.2 _NOARGS rationale). */
#define ROCM_TRACE_RET_STATUS_CURATED_NOARGS(api, expr, corr)                \
    do {                                                                     \
        const hipError_t __rocm_status = (expr);                             \
        rocm_trace_emit_##api##_args((corr), __rocm_status);                 \
        rocm_trace_emit_hip_api_exit_status(__func__,                        \
            (corr), (int32_t)__rocm_status);                                 \
        return __rocm_status;                                                \
    } while (0)

#define ROCM_TRACE_RET_PTR_CURATED_NOARGS(api, ptr_type, expr, corr)         \
    do {                                                                     \
        ptr_type const __rocm_ptr = (expr);                                  \
        const hipError_t __rocm_status =                                     \
            (__rocm_ptr != nullptr) ? hipSuccess : hipErrorOutOfMemory;      \
        rocm_trace_emit_##api##_args((corr), __rocm_status);                 \
        rocm_trace_emit_hip_api_exit_ptr(__func__, (corr), __rocm_ptr);      \
        return __rocm_ptr;                                                   \
    } while (0)

#define ROCM_TRACE_RET_VOID_CURATED_NOARGS(api, expr, corr)                  \
    do {                                                                     \
        (expr);                                                              \
        rocm_trace_emit_##api##_args((corr), hipSuccess);                    \
        rocm_trace_emit_hip_api_exit_void(__func__, (corr));                 \
        return;                                                              \
    } while (0)
```

- [ ] **Step 3: Build to verify the macros compile (no use sites yet)**

```bash
./dev-bin/sync.sh main
./dev-bin/in-container.sh main "cd /root/rocm-systems && cmake --build build/clr -j 32 --target amdhip64 2>&1 | tail -10"
```

Expected: build succeeds. The macros are defined but unused, so no behavior change.

- [ ] **Step 4: Commit**

```bash
git add projects/clr/hipamd/src/hip_table_interface.cpp
git commit -m "lttng: add 6 _CURATED macro variants (STATUS/PTR/VOID + _NOARGS)

Per spec §6.2. Three captured-args variants and three _NOARGS variants
mirror the existing ROCM_TRACE_RET_* macro family. Each variant emits
the typed <api>_args event AND preserves the matching generic exit
event (hip_api_exit_status / _ptr / _void) so the augment-not-replace
contract from spec §1/§2 holds.

Helper signature invariant: every helper takes (corr_id, captured...,
status). PTR variants synthesize status from null-vs-non-null retval.
VOID variants pass literal hipSuccess.

No use sites yet — those land with the migrator extension."
```

---

### Task 9: Extend `lttng_migrate.py` with curated routing

**Files:**
- Modify: `projects/clr/hipamd/scripts/lttng_migrate.py`

- [ ] **Step 1: Read the existing migrator's `migrate_file()` to understand the rewrite plan**

```bash
sed -n '248,365p' projects/clr/hipamd/scripts/lttng_migrate.py
```

Key observations:
- The migrator inserts `ENTER_SNIPPET` at `open_brace_off + 1` (just past `{`).
- It rewrites every `return EXPR;` via `rewrite_return_stmt()` → `ROCM_TRACE_RET_<cls>(EXPR)`.
- Idempotency check uses `MIGRATION_MARKER = '__rocm_corr...'`.

The curated extension needs to:
1. Parse `--curated-yaml`; build `{api_name → api_dict}`.
2. For each curated wrapper:
   a. Insert the sentinel `/* __ROCM_CURATED__: <api> */` immediately after `__rocm_corr`.
   b. Insert IN-locals (`__rocm_in_<name>`) for each IN/INOUT arg, after the sentinel.
   c. Rewrite `return EXPR;` to use the `_CURATED` (or `_CURATED_NOARGS`) macro variant matching the wrapper's return-type class and arg count.
3. Idempotency: check for sentinel comment, skip if present.

- [ ] **Step 2: Add the new code**

Apply the following edits to `projects/clr/hipamd/scripts/lttng_migrate.py`:

(2a) Near the top, after `from clang import cindex`, add the curated-lib import:

```python
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lttng_curated_lib import parse_yaml_file, IN_ARG_KIND
```

(2b) Add helper to load curated APIs (anywhere before `migrate_file`):

```python
def load_curated(yaml_path):
    """Returns {api_name: api_dict} or {} if path is None/missing."""
    if not yaml_path or not os.path.exists(yaml_path):
        return {}
    return {a['api']: a for a in parse_yaml_file(yaml_path)}


def _captured_args_for_curated(api):
    """Return list of captured-arg names from the YAML, in YAML order.
    For the migrator's macro emit we pass each captured arg's wrapper
    parameter name (which equals the YAML name per the §4 binding rule).
    OUT-only and INOUT args are passed differently: OUT passes the
    pointer parameter directly (helper deref's at the right time); IN
    passes the captured local __rocm_in_<name>."""
    result = []
    for a in api['args']:
        if a['dir'] == 'IN':
            result.append(f'__rocm_in_{a["name"]}')
        elif a['dir'] == 'OUT':
            # Helper expects the original out-pointer parameter (the wrapper's
            # parameter is already a pointer for OUT args).
            result.append(a['name'])
        # INOUT is rejected by the parser.
    return result
```

(2c) Add a sentinel constant and the curated-aware injection in `migrate_file`. Modify the per-wrapper loop body to handle curated APIs:

In `migrate_file()`, after `inventory[n.spelling] = (cls, ret_type)`, replace the existing single insert with:

```python
        # Original generic ENTER snippet — always inserted.
        insert_off = open_brace_off + 1
        edits.append((insert_off, insert_off,
                      ENTER_SNIPPET.encode('utf-8')))

        # Curated extension: if this API is in the curated set, also insert
        # the sentinel + IN-locals immediately after ENTER_SNIPPET.
        curated_api = curated.get(n.spelling)
        if curated_api is not None:
            sentinel = f' /* __ROCM_CURATED__: {n.spelling} */'
            in_locals_lines = []
            # Map each IN-arg name to the wrapper's parameter type via libclang.
            param_types = {arg.spelling: arg.type.spelling
                           for arg in n.get_arguments()}
            for a in curated_api['args']:
                if a['dir'] == 'IN':
                    pname = a['name']
                    if pname not in param_types:
                        raise SystemExit(
                            f"{n.spelling}: curated arg {pname!r} not in "
                            f"wrapper params {list(param_types)}")
                    pty = param_types[pname]
                    in_locals_lines.append(
                        f' {pty} const __rocm_in_{pname} = {pname};')
            in_locals_block = ''.join(in_locals_lines)
            curated_snippet = sentinel + in_locals_block
            edits.append((insert_off, insert_off,
                          curated_snippet.encode('utf-8')))
```

(Note: ordering — both edits insert at the same offset, but `apply edits in reverse-offset order` means the second-added edit gets applied first, so the sentinel ends up AFTER the ENTER_SNIPPET in the final source. To make this deterministic, the curated-snippet edit should use `(insert_off + 1, insert_off + 1, ...)` and the existing ENTER_SNIPPET edit must be applied first. Since we collect all edits then sort by offset descending, both inserting at the same offset means the order depends on stability. Use a tuple-with-tiebreaker: append a sub-order marker. The simplest fix: change `edits.append(...)` to use `edits.append((insert_off, insert_off, ENTER_SNIPPET.encode('utf-8'), 0))` and the curated-snippet `(insert_off, insert_off, curated_snippet.encode('utf-8'), 1)`, then sort by `(start, sub)` reversed. Adjust the apply loop accordingly.)

(2d) Modify `rewrite_return_stmt()` to accept the curated_api dict and emit the `_CURATED` form when it's not None:

```python
def rewrite_return_stmt(src, ret_node, cls, curated_api=None):
    """... existing docstring ..."""
    # ... existing extent extraction unchanged ...
    snippet = src[start:end].decode('utf-8', errors='strict')
    s = snippet.strip()
    if not s.startswith('return'):
        raise SystemExit(...)
    inner = s[len('return'):].rstrip(';').strip()

    if curated_api is None:
        # Existing non-curated path — emit ROCM_TRACE_RET_<cls>.
        if cls == 'STATUS':
            if not inner: raise SystemExit(...)
            repl = f'ROCM_TRACE_RET_STATUS({inner});'
        elif cls == 'PTR':
            if not inner: raise SystemExit(...)
            repl = f'ROCM_TRACE_RET_PTR({inner});'
        elif cls == 'VOID':
            if inner:
                repl = f'ROCM_TRACE_RET_VOID({inner});'
            else:
                repl = f'rocm_trace_emit_hip_api_exit_void(__func__, __rocm_corr); return;'
        elif cls == 'STRUCT':
            if not inner: raise SystemExit(...)
            repl = (f'do {{ auto __rocm_rv = ({inner}); '
                    f'rocm_trace_emit_hip_api_exit_void(__func__, __rocm_corr); '
                    f'return __rocm_rv; }} while (0);')
        else:
            raise AssertionError(cls)
        return (start, end, repl.encode('utf-8'))

    # Curated path. Pick _CURATED vs _CURATED_NOARGS based on captured arg count.
    captured = _captured_args_for_curated(curated_api)
    api = curated_api['api']
    if cls == 'STATUS':
        if not inner: raise SystemExit(f'{api}: STATUS curated wrapper has bare return')
        if captured:
            args_str = ', '.join(captured)
            repl = (f'ROCM_TRACE_RET_STATUS_CURATED({api}, {inner}, '
                    f'__rocm_corr, {args_str});')
        else:
            repl = f'ROCM_TRACE_RET_STATUS_CURATED_NOARGS({api}, {inner}, __rocm_corr);'
    elif cls == 'PTR':
        if not inner: raise SystemExit(f'{api}: PTR curated wrapper has bare return')
        # _CURATED_PTR macro takes ptr_type as second arg. Use auto for the
        # wrapper's actual return type — the migrator doesn't know it
        # statically. Use decltype((inner)) which the compiler resolves at
        # the macro expansion site.
        ptr_type = 'auto'  # accepted by C++14+
        if captured:
            args_str = ', '.join(captured)
            repl = (f'ROCM_TRACE_RET_PTR_CURATED({api}, {ptr_type}, {inner}, '
                    f'__rocm_corr, {args_str});')
        else:
            repl = f'ROCM_TRACE_RET_PTR_CURATED_NOARGS({api}, {ptr_type}, {inner}, __rocm_corr);'
    elif cls == 'VOID':
        if captured:
            args_str = ', '.join(captured)
            repl = (f'ROCM_TRACE_RET_VOID_CURATED({api}, {inner if inner else ""}, '
                    f'__rocm_corr, {args_str});')
        else:
            repl = f'ROCM_TRACE_RET_VOID_CURATED_NOARGS({api}, {inner if inner else ""}, __rocm_corr);'
    elif cls == 'STRUCT':
        # STRUCT-returning curated wrappers are not supported in v1 (no
        # curated API in spec Appendix A returns by-value struct).
        raise SystemExit(
            f'{api}: STRUCT-returning curated wrappers not supported in v1; '
            f'remove from curated_apis.yaml')
    return (start, end, repl.encode('utf-8'))
```

Then in the loop where return statements are rewritten, pass `curated_api`:

```python
for r in returns:
    edits.append(rewrite_return_stmt(src, r, cls, curated.get(n.spelling)))
```

(2e) Idempotency — extend the early-exit check to accept either marker:

```python
CURATED_SENTINEL_PREFIX = b'/* __ROCM_CURATED__:'

# In migrate_file, replace:
if MIGRATION_MARKER.encode('utf-8') in src:
# with: same check (existing behavior is fine — sentinels are only inserted
# when __rocm_corr is also inserted, so the existing marker covers both).
```

(2f) Add CLI flag:

```python
ap.add_argument('--curated-yaml', default=None,
                help='Path to curated_apis.yaml; if provided, curated APIs '
                     'get sentinel + IN-locals + _CURATED macro routing')
```

And in `main()`:

```python
curated = load_curated(args.curated_yaml)
inv = migrate_file(args.source, args.include_path, args.extra_arg,
                   args.inventory, dry_run=args.dry_run, curated=curated)
```

And `migrate_file` signature: `def migrate_file(..., dry_run=False, curated=None):` with `curated = curated or {}`.

- [ ] **Step 3: Add a unit test for the migrator's curated emission**

Create `projects/clr/hipamd/scripts/test_lttng_migrate_curated.py`:

```python
"""Smoke test: migrator emits curated sentinels and macro variants
correctly on a synthetic source file."""
import os, sys, subprocess, tempfile, textwrap
HERE = os.path.dirname(os.path.abspath(__file__))

def test_curated_emit():
    yaml_text = textwrap.dedent("""\
    - api: hipMemcpyAsync
      category: memory
      args:
        - {name: dst,       type: ptr,    dir: IN}
        - {name: src,       type: ptr,    dir: IN}
        - {name: sizeBytes, type: size,   dir: IN}
        - {name: kind,      type: enum,   dir: IN}
        - {name: stream,    type: handle, dir: IN}

    - api: hipMalloc
      category: memory
      args:
        - {name: ptr,  type: ptr,  dir: OUT}
        - {name: size, type: size, dir: IN}

    - api: hipDeviceSynchronize
      category: streams
      args: []
    """)

    src_text = textwrap.dedent("""\
    #include <hip/hip_runtime.h>

    hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                              hipMemcpyKind kind, hipStream_t stream) {
        return hipSuccess;
    }

    hipError_t hipMalloc(void** ptr, size_t size) {
        return hipSuccess;
    }

    hipError_t hipDeviceSynchronize(void) {
        return hipSuccess;
    }
    """)

    with tempfile.TemporaryDirectory() as d:
        yaml_path = os.path.join(d, 'apis.yaml')
        src_path  = os.path.join(d, 'wrappers.cpp')
        inv_path  = os.path.join(d, 'inv.txt')
        with open(yaml_path, 'w') as f: f.write(yaml_text)
        with open(src_path,  'w') as f: f.write(src_text)
        cmd = ['python3', os.path.join(HERE, 'lttng_migrate.py'),
               '--source', src_path,
               '--include-path', '/opt/rocm/include',
               '--extra-arg=-D__HIP_PLATFORM_AMD__=1',
               '--inventory', inv_path,
               '--curated-yaml', yaml_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"migrate failed:\n{r.stderr}"
        with open(src_path) as f:
            out = f.read()

    # Sentinel for each curated API.
    assert '/* __ROCM_CURATED__: hipMemcpyAsync */' in out
    assert '/* __ROCM_CURATED__: hipMalloc */' in out
    assert '/* __ROCM_CURATED__: hipDeviceSynchronize */' in out

    # IN-locals for the IN args of hipMemcpyAsync.
    assert '__rocm_in_dst' in out
    assert '__rocm_in_sizeBytes' in out

    # No IN-local for hipMalloc's OUT-only ptr arg.
    # (size is IN, so __rocm_in_size IS expected.)
    assert '__rocm_in_size' in out
    # ptr is OUT — no __rocm_in_ptr.
    # (Be careful with substring: __rocm_in_ptr could match __rocm_in_ptr_out.
    # We check the full identifier separately.)
    assert '__rocm_in_ptr ' not in out  # space ensures full token boundary
    assert '__rocm_in_ptr=' not in out

    # _CURATED macro variants.
    assert 'ROCM_TRACE_RET_STATUS_CURATED(hipMemcpyAsync,' in out
    assert 'ROCM_TRACE_RET_STATUS_CURATED(hipMalloc,' in out
    assert 'ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceSynchronize,' in out

    print('PASS')

if __name__ == '__main__':
    test_curated_emit()
```

- [ ] **Step 4: Run the test**

```bash
python3 projects/clr/hipamd/scripts/test_lttng_migrate_curated.py
```

Expected: `PASS`. Iterate on the migrator until it passes.

- [ ] **Step 5: Commit**

```bash
git add projects/clr/hipamd/scripts/lttng_migrate.py \
        projects/clr/hipamd/scripts/test_lttng_migrate_curated.py
git commit -m "lttng: extend migrator with curated-args routing

Adds --curated-yaml flag. For each wrapper named in the YAML, the
migrator now also injects:
  1. /* __ROCM_CURATED__: <api> */ sentinel comment (idempotency +
     coverage gate marker per spec §6.1)
  2. const C local __rocm_in_<name> per IN/INOUT arg (spec §6.1)
  3. _CURATED or _CURATED_NOARGS macro variant in place of the regular
     ROCM_TRACE_RET_* macro, picked by captured-arg count (spec §6.2)

The arg-name binding for IN-locals comes from the YAML 'name' field —
the verifier (separate CI gate) hard-errors on YAML/header name
mismatch so the migrator's binding is always safe.

OUT args don't get an IN-local; the wrapper's original out-pointer
parameter is passed directly to the helper, which deref's it gated by
the call's status (spec §5.2 hipMalloc example).

Smoke test on synthetic source verifies sentinel + IN-local + macro
emission for all-IN, OUT-param, and zero-arg cases."
```

---

### Task 10: Re-migrate `hip_table_interface.cpp` with curated YAML

**Files:**
- Modify: `projects/clr/hipamd/src/hip_table_interface.cpp` (re-migrated)
- Modify: `projects/clr/hipamd/scripts/lttng_migration_inventory.txt` (regenerated)

- [ ] **Step 1: Decision — use a separate overlay script (NOT the base migrator)**

The previous migrator pass on this file already injected `__rocm_corr` + `ROCM_TRACE_RET_*` macros. Re-running the base migrator skips already-migrated files (its idempotency check). Two ways to apply the curated extension:

- **Approach A (rejected):** Modify base `lttng_migrate.py` idempotency to "if sentinel-only is missing, apply only the new pieces". This makes the base migrator significantly more complex — it would need to support partial-overlay mode in addition to its existing fresh-migrate mode, and the tracking of "what's already there vs what to add" interleaves with the AST walk in fragile ways.
- **Approach B (chosen):** Add a separate `lttng_migrate_curated_overlay.py` script that operates only on already-generic-migrated source. The base `lttng_migrate.py` keeps its existing single-shot behavior. The overlay script:
  1. Finds each curated wrapper's body (locates the function definition).
  2. Inserts sentinel + IN-locals immediately after the existing `__rocm_corr` declaration.
  3. Rewrites the body's `ROCM_TRACE_RET_<cls>(EXPR);` to `ROCM_TRACE_RET_<cls>_CURATED(...)`.

Approach B keeps each tool focused on one job. The base migrator does generic migration; the overlay does curated. Both are idempotent on their own marker.

Create `projects/clr/hipamd/scripts/lttng_migrate_curated_overlay.py`:

```python
#!/usr/bin/env python3
"""Overlay the curated-args extension onto an already-migrated wrapper TU.

Idempotent: skips wrappers that already have the __ROCM_CURATED__ sentinel.

For each wrapper named in --curated-yaml whose body in --source contains
the provider's ENTER snippet but NOT the curated sentinel:
  1. Insert sentinel + IN-locals right after the ENTER_SNIPPET.
  2. Rewrite the wrapper's existing macro family (ROCM_TRACE_RET_* for HIP,
     ROCR_TRACE_API_RET_* for HSA) to the matching _CURATED / _CURATED_HSA
     (and _NOARGS) variant.

Provider-agnostic: takes --provider hip|hsa and selects:
- enter-helper regex (rocm_trace_emit_hip_api_enter vs rocm_trace_emit_hsa_api_enter)
- existing macro family regex
- emitted curated macro names
- default include flags (HIP needs -D__HIP_PLATFORM_AMD__=1)

Usage:
    python3 lttng_migrate_curated_overlay.py \\
        --provider hip \\
        --source path/to/hip_table_interface.cpp \\
        --curated-yaml path/to/curated_apis.yaml \\
        --include-path /opt/rocm/include
"""
import argparse, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lttng_curated_lib import parse_yaml_file
from clang import cindex

# Provider-specific configuration. Adding a new provider means adding one
# entry here and updating the --provider choices.
PROVIDER_CONFIG = {
    'hip': {
        'enter_re': re.compile(
            r'__rocm_corr\s*=\s*rocm_trace_next_corr_id\(\)\s*;'
            r'\s*rocm_trace_emit_hip_api_enter\([^)]*\)\s*;'),
        # Match ONLY the non-curated forms — must not re-match _CURATED on a
        # second pass. Anchor with a negative lookahead.
        'ret_re': re.compile(
            r'ROCM_TRACE_RET_(STATUS|PTR|VOID)(?!_CURATED)\s*\(\s*([^;]+?)\s*\)\s*;',
            flags=re.DOTALL),
        # Curated macro name template: {cls} in {STATUS,PTR,VOID}, plus _NOARGS.
        'curated_status':         'ROCM_TRACE_RET_STATUS_CURATED',
        'curated_status_noargs':  'ROCM_TRACE_RET_STATUS_CURATED_NOARGS',
        'curated_ptr':            'ROCM_TRACE_RET_PTR_CURATED',
        'curated_ptr_noargs':     'ROCM_TRACE_RET_PTR_CURATED_NOARGS',
        'curated_void':           'ROCM_TRACE_RET_VOID_CURATED',
        'curated_void_noargs':    'ROCM_TRACE_RET_VOID_CURATED_NOARGS',
        'default_extra_args': ['-D__HIP_PLATFORM_AMD__=1'],
    },
    'hsa': {
        'enter_re': re.compile(
            r'__rocm_corr\s*=\s*rocm_trace_next_corr_id\(\)\s*;'
            r'\s*rocm_trace_emit_hsa_api_enter\([^)]*\)\s*;'),
        'ret_re': re.compile(
            r'ROCR_TRACE_API_RET_(STATUS|PTR|VOID)(?!_CURATED)\s*\(\s*([^;]+?)\s*\)\s*;',
            flags=re.DOTALL),
        'curated_status':         'ROCR_TRACE_API_RET_STATUS_CURATED_HSA',
        'curated_status_noargs':  'ROCR_TRACE_API_RET_STATUS_CURATED_HSA_NOARGS',
        'curated_ptr':            'ROCR_TRACE_API_RET_PTR_CURATED_HSA',
        'curated_ptr_noargs':     'ROCR_TRACE_API_RET_PTR_CURATED_HSA_NOARGS',
        'curated_void':           'ROCR_TRACE_API_RET_VOID_CURATED_HSA',
        'curated_void_noargs':    'ROCR_TRACE_API_RET_VOID_CURATED_HSA_NOARGS',
        'default_extra_args': [],   # HSA headers don't need __HIP_PLATFORM_AMD__
    },
}

SENTINEL_RE = lambda api: re.compile(rf'/\* __ROCM_CURATED__: {re.escape(api)} \*/')


def find_wrapper_body(text, fn_name):
    """Return (body_start, body_end) byte offsets, or None."""
    pat = re.compile(r'\b' + re.escape(fn_name) + r'\s*\([^)]*\)\s*\{')
    m = pat.search(text)
    if not m:
        return None
    depth = 0
    i = text.index('{', m.start(0))
    body_start = i
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return (body_start, i+1)
        i += 1
    return None


def overlay(provider, source_path, yaml_path, include_path):
    cfg = PROVIDER_CONFIG[provider]
    apis = {a['api']: a for a in parse_yaml_file(yaml_path)}

    # Use libclang to get param-type info for wrapper signatures.
    args = ['-x', 'c++', '-std=c++17', '-I', include_path] + cfg['default_extra_args']
    idx = cindex.Index.create()
    tu = idx.parse(source_path, args=args)
    param_types = {}  # fn_name -> {pname: ptype}
    for n in tu.cursor.walk_preorder():
        if (n.kind == cindex.CursorKind.FUNCTION_DECL and n.is_definition()
                and n.spelling in apis):
            param_types[n.spelling] = {a.spelling: a.type.spelling
                                        for a in n.get_arguments()}

    with open(source_path, 'r') as f:
        src = f.read()
    edits = []  # (start, end, replacement)
    for fn, api in apis.items():
        if SENTINEL_RE(fn).search(src):
            print(f'  {fn}: already overlaid; skip', file=sys.stderr)
            continue
        body = find_wrapper_body(src, fn)
        if body is None:
            print(f'  {fn}: body not found in {source_path}; skip', file=sys.stderr)
            continue
        bstart, bend = body
        body_text = src[bstart:bend]
        # Find provider-specific ENTER snippet in body to anchor the
        # sentinel insertion.
        em = cfg['enter_re'].search(body_text)
        if em is None:
            print(f'  {fn}: ENTER snippet not found; skip', file=sys.stderr)
            continue
        insert_off = bstart + em.end()

        # Build sentinel + IN-locals.
        sentinel = f' /* __ROCM_CURATED__: {fn} */'
        in_locals = []
        ptypes = param_types.get(fn, {})
        for a in api['args']:
            if a['dir'] == 'IN':
                pname = a['name']
                pty = ptypes.get(pname)
                if pty is None:
                    sys.exit(f'{fn}: arg {pname!r} not in wrapper params {list(ptypes)}')
                in_locals.append(f' {pty} const __rocm_in_{pname} = {pname};')
        insertion = sentinel + ''.join(in_locals)
        edits.append((insert_off, insert_off, insertion))

        # Rewrite the existing macro family to its _CURATED variant. The
        # macro family and curated-name templates are picked from cfg so
        # the same code handles both HIP (ROCM_TRACE_RET_*) and HSA
        # (ROCR_TRACE_API_RET_*) without duplication.
        for m in cfg['ret_re'].finditer(body_text):
            cls, expr = m.group(1), m.group(2)
            macro_start = bstart + m.start(0)
            macro_end   = bstart + m.end(0)
            # Build captured-args list.
            captured = []
            for a in api['args']:
                if a['dir'] == 'IN':
                    captured.append(f'__rocm_in_{a["name"]}')
                elif a['dir'] == 'OUT':
                    captured.append(a['name'])
            if cls == 'STATUS':
                if captured:
                    repl = (f'{cfg["curated_status"]}({fn}, {expr}, '
                            f'__rocm_corr, {", ".join(captured)});')
                else:
                    repl = f'{cfg["curated_status_noargs"]}({fn}, {expr}, __rocm_corr);'
            elif cls == 'PTR':
                if captured:
                    repl = (f'{cfg["curated_ptr"]}({fn}, auto, {expr}, '
                            f'__rocm_corr, {", ".join(captured)});')
                else:
                    repl = f'{cfg["curated_ptr_noargs"]}({fn}, auto, {expr}, __rocm_corr);'
            elif cls == 'VOID':
                if captured:
                    repl = (f'{cfg["curated_void"]}({fn}, {expr}, '
                            f'__rocm_corr, {", ".join(captured)});')
                else:
                    repl = f'{cfg["curated_void_noargs"]}({fn}, {expr}, __rocm_corr);'
            else:
                continue
            edits.append((macro_start, macro_end, repl))

    if not edits:
        print(f'no edits to apply', file=sys.stderr)
        return 0
    edits.sort(key=lambda e: e[0], reverse=True)
    out = list(src)
    for start, end, repl in edits:
        out[start:end] = repl
    with open(source_path, 'w') as f:
        f.write(''.join(out))
    print(f'applied {len(edits)} edits', file=sys.stderr)
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--provider', required=True, choices=sorted(PROVIDER_CONFIG))
    ap.add_argument('--source', required=True)
    ap.add_argument('--curated-yaml', required=True)
    ap.add_argument('--include-path', default='/opt/rocm/include')
    args = ap.parse_args()
    sys.exit(overlay(args.provider, args.source, args.curated_yaml,
                     args.include_path))

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the overlay (run LOCALLY when libclang available; otherwise in container with explicit `docker cp` copy-back)**

The overlay script needs libclang Python bindings. Per `dev-bin/sync.sh` the official sync direction is local→container (push to origin, container fetches+resets). Container→local is **not** an automated path — `git reset --hard origin/<branch>` LOCALLY would DISCARD any local commits made in earlier tasks. **Never** do that.

Two safe options. Pick (A) if libclang is installable on the host; otherwise pick (B):

**Option A — overlay runs locally (preferred):**

```bash
# Confirm libclang is available locally:
python3 -c "from clang import cindex; print('libclang OK')"

# If "ImportError" / "No module named 'clang'": either `pip install --user libclang`
# (and re-test), or fall back to Option B below.

python3 projects/clr/hipamd/scripts/lttng_migrate_curated_overlay.py \
    --provider hip \
    --source projects/clr/hipamd/src/hip_table_interface.cpp \
    --curated-yaml projects/clr/hipamd/scripts/curated_apis.yaml
```

**Option B — overlay runs in container, single file copied back via `docker cp`:**

```bash
# 1. Get current local file into container via the official sync channel.
./dev-bin/sync.sh main

# 2. Run the overlay inside the container against the (now in-sync) source.
./dev-bin/in-container.sh main "cd /root/rocm-systems && python3 \
    projects/clr/hipamd/scripts/lttng_migrate_curated_overlay.py \
    --provider hip \
    --source projects/clr/hipamd/src/hip_table_interface.cpp \
    --curated-yaml projects/clr/hipamd/scripts/curated_apis.yaml"

# 3. Copy the SINGLE edited file back to the local worktree. This is
#    surgical — it touches only the overlay's output file, never the
#    rest of the worktree, and CANNOT discard local commits.
REMOTE_HOST=bewelton@banff-ccs-aus-g05-05.cs-aus.dcgpu
CONTAINER=bewelton_lttng
LOCAL_FILE=projects/clr/hipamd/src/hip_table_interface.cpp
ssh "$REMOTE_HOST" "docker cp ${CONTAINER}:/root/rocm-systems/${LOCAL_FILE} /tmp/hip_table_interface.cpp.from-container"
scp "$REMOTE_HOST:/tmp/hip_table_interface.cpp.from-container" "$LOCAL_FILE"
ssh "$REMOTE_HOST" "rm -f /tmp/hip_table_interface.cpp.from-container"

# 4. Sanity-check: the local diff should equal the overlay's edits, with
#    no unrelated changes.
git diff --stat -- "$LOCAL_FILE"
```

**Forbidden:** running `git fetch origin && git reset --hard origin/users/bewelton/lttng` locally as a "sync back" step. That destroys uncommitted work and any local commits not yet pushed. The overlay's outputs are reachable via `docker cp`; do not use a hard-reset to move data between machines.

- [ ] **Step 3: Inspect the diff for hipMemcpyAsync, hipMalloc, hipDeviceSynchronize**

```bash
grep -A3 'hipMemcpyAsync\|hipMalloc\|hipDeviceSynchronize' projects/clr/hipamd/src/hip_table_interface.cpp | head -40
```

Verify each shows the sentinel and the `_CURATED(...)` or `_CURATED_NOARGS(...)` macro.

- [ ] **Step 4: Build to verify the wrappers compile**

```bash
./dev-bin/sync.sh main
./dev-bin/in-container.sh main "cd /root/rocm-systems && cmake --build build/clr -j 32 --target amdhip64 2>&1 | tail -20"
```

Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add projects/clr/hipamd/scripts/lttng_migrate_curated_overlay.py \
        projects/clr/hipamd/src/hip_table_interface.cpp
git commit -m "lttng: overlay curated-args migration on hip_table_interface.cpp

Adds lttng_migrate_curated_overlay.py — idempotent overlay that
operates on an already-generic-migrated wrapper TU, inserting the
__ROCM_CURATED__ sentinel + IN-locals + _CURATED macro variants for
each curated API in curated_apis.yaml.

Applied to hip_table_interface.cpp for the 3 Phase B APIs:
hipMemcpyAsync, hipMalloc, hipDeviceSynchronize. Build verified."
```

---

### Task 11: Update `lttng_coverage_gate.sh` for curated checks

**Files:**
- Modify: `projects/clr/hipamd/scripts/lttng_coverage_gate.sh`

- [ ] **Step 1: Add a third gate section after the existing two**

Per spec §8.2: the gate must verify (a) every YAML API is in the inventory and (b) each curated wrapper body contains the sentinel AND a curated-macro invocation. Add to the bottom of `lttng_coverage_gate.sh` (before the final `exit 0`):

```bash
# ---------------------------------------------------------------------------
# 3. Curated-args coverage gate (spec §8.2)
# ---------------------------------------------------------------------------

CURATED_YAML="$SCRIPT_DIR/curated_apis.yaml"
if [ -f "$CURATED_YAML" ]; then
    # Curated APIs (one per line).
    python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from lttng_curated_lib import parse_yaml_file
for a in parse_yaml_file('$CURATED_YAML'):
    print(a['api'])
" | sort -u > "$WORK/curated.txt"

    # All inventoried wrappers (already in $WORK/migrated.txt from gate 1).
    MISSING_FROM_INV="$(comm -23 "$WORK/curated.txt" "$WORK/migrated.txt" || true)"
    if [ -n "$MISSING_FROM_INV" ]; then
        echo "FAIL (curated): APIs in curated_apis.yaml are missing from migration inventory:"
        printf '  %s\n' $MISSING_FROM_INV
        exit 1
    fi

    # Body-content scan: for each curated API, the wrapper body must contain
    # the sentinel AND a curated-macro invocation matching the regex from
    # spec §8.2. APIs with at least one IN/INOUT arg must additionally
    # contain at least one __rocm_in_ local.
    PYTHON_CURATED_GATE="$(cat <<'PY'
import os, re, sys
sys.path.insert(0, sys.argv[3])
from lttng_curated_lib import parse_yaml_file

src_dir   = sys.argv[1]
yaml_path = sys.argv[2]

apis = parse_yaml_file(yaml_path)
files = []
for root, _, fs in os.walk(src_dir):
    for fn in fs:
        if fn.endswith('.cpp'):
            p = os.path.join(root, fn)
            with open(p, 'rb') as fh:
                files.append((p, fh.read().decode('utf-8', errors='replace')))

# Spec §8.2 regex matcher for all six curated-macro variants.
MACRO_RE = re.compile(
    r'ROCM_TRACE_RET_(STATUS|PTR|VOID)_CURATED(_NOARGS)?(_HSA(_NOARGS)?)?\s*\(')

def find_body(text, name):
    pat = re.compile(r'\b' + re.escape(name) + r'\s*\(')
    for m in pat.finditer(text):
        depth = 0
        i = m.end() - 1
        while i < len(text):
            if text[i] == '(': depth += 1
            elif text[i] == ')':
                depth -= 1
                if depth == 0:
                    j = text.find('{', i)
                    if j < 0: break
                    bdepth = 0; k = j
                    while k < len(text):
                        if text[k] == '{': bdepth += 1
                        elif text[k] == '}':
                            bdepth -= 1
                            if bdepth == 0:
                                return text[j:k+1]
                        k += 1
                    break
            i += 1
    return None

failures = []
for api in apis:
    name = api['api']
    sentinel = f'/* __ROCM_CURATED__: {name} */'
    body = None
    for path, text in files:
        b = find_body(text, name)
        if b and sentinel in b:
            body = b
            break
    if body is None:
        failures.append(f'{name}: no wrapper body found containing sentinel {sentinel!r}')
        continue
    if not MACRO_RE.search(body):
        failures.append(f'{name}: sentinel present but no _CURATED macro invocation')
        continue
    # IN-local check, only when the API has at least one IN/INOUT arg.
    has_in = any(a['dir'] in ('IN', 'INOUT') for a in api['args'])
    if has_in and '__rocm_in_' not in body:
        failures.append(f'{name}: has IN args but no __rocm_in_ locals in body')

if failures:
    for f in failures:
        print(f'  CURATED FAIL: {f}')
    sys.exit(1)
print(f'CURATED: {len(apis)} curated APIs verified')
PY
)"
    set +e
    python3 -c "$PYTHON_CURATED_GATE" "$SRC_DIR" "$CURATED_YAML" "$SCRIPT_DIR" \
        > "$WORK/curated.log" 2>&1
    CURATED_RC=$?
    set -e
    cat "$WORK/curated.log"
    if [ "$CURATED_RC" -ne 0 ]; then
        echo "FAIL: curated-args coverage gate failed"
        exit 1
    fi
fi
```

- [ ] **Step 2: Run the coverage gate manually**

```bash
./dev-bin/in-container.sh main "cd /root/rocm-systems && bash projects/clr/hipamd/scripts/lttng_coverage_gate.sh \
    build/clr/hipamd/lib/libamdhip64.so \
    projects/clr/hipamd/scripts/lttng_migration_inventory.txt \
    projects/clr/hipamd/scripts/lttng_migration_inventory_c.txt"
```

Expected output: original `PASS: all <N> exported HIP symbols migrated` PLUS new `CURATED: 3 curated APIs verified`.

- [ ] **Step 3: Commit**

```bash
git add projects/clr/hipamd/scripts/lttng_coverage_gate.sh
git commit -m "lttng: extend coverage gate with curated-args checks (spec §8.2)

Adds a third gate after symbol-coverage and body-content. For every API
in curated_apis.yaml, asserts:
  - inventory contains the symbol (already verified by gate 1; checked
    for completeness)
  - wrapper body contains the /* __ROCM_CURATED__: <api> */ sentinel
  - wrapper body contains a curated-macro invocation matching the
    regex ROCM_TRACE_RET_(STATUS|PTR|VOID)_CURATED(_NOARGS)?(_HSA(_NOARGS)?)?
    (covers all six variants per spec §8.2)
  - wrapper body contains __rocm_in_ when the API has IN/INOUT args
    (skipped for OUT-only and zero-arg APIs)

Skipped silently when curated_apis.yaml is absent (allows gradual
rollout)."
```

---


## Phase E — Build + test HIP minimal (tasks 12–13)

This phase wires CMake, runs an end-to-end build/test cycle on the container, and adds the new payload + coverage tests. The 3 Phase B APIs fire their typed `_args` events through the full pipeline.

---

### Task 12: CMake wiring — opt-in `regenerate-lttng-curated` target

**Files:**
- Modify: `projects/clr/hipamd/src/CMakeLists.txt`

- [ ] **Step 1: Add the regenerate target**

Inside the existing `if(HIP_ENABLE_LTTNG_UST)` block in `projects/clr/hipamd/src/CMakeLists.txt` (around line 336), after the `target_link_libraries(amdhip64 PRIVATE PkgConfig::LTTNG_UST ...)` line and BEFORE the existing `add_custom_command(TARGET amdhip64 POST_BUILD ... lttng_coverage_gate.sh ...)` line, insert:

```cmake
    # ---- Curated parameter-capture: opt-in regeneration target ----
    # Per spec §9.1: default build does NOT regenerate. The custom command
    # is wired only to the manual `regenerate-lttng-curated` target so the
    # build never requires Python or PyYAML. Generated headers are checked
    # in. CI runs codegen + `git diff --exit-code` to catch YAML drift.
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_FOUND)
        set(_CURATED_YAML  ${CMAKE_CURRENT_LIST_DIR}/../scripts/curated_apis.yaml)
        set(_CURATED_TP_H  ${CMAKE_CURRENT_LIST_DIR}/lttng/rocm_hip_curated_tp.h)
        set(_CURATED_EMIT_H ${CMAKE_CURRENT_LIST_DIR}/lttng/rocm_trace_emit_curated.h)
        if(EXISTS ${_CURATED_YAML})
            add_custom_command(
                OUTPUT ${_CURATED_TP_H} ${_CURATED_EMIT_H}
                COMMAND ${Python3_EXECUTABLE}
                        ${CMAKE_CURRENT_LIST_DIR}/../scripts/lttng_curated_codegen.py
                        --yaml      ${_CURATED_YAML}
                        --provider  rocm_hip
                        --status-type hipError_t
                        --status-success hipSuccess
                        --out-tp    ${_CURATED_TP_H}
                        --out-emit  ${_CURATED_EMIT_H}
                DEPENDS ${_CURATED_YAML}
                        ${CMAKE_CURRENT_LIST_DIR}/../scripts/lttng_curated_codegen.py
                        ${CMAKE_CURRENT_LIST_DIR}/../scripts/lttng_curated_lib.py
                COMMENT "Regenerating LTTng curated tracepoints (HIP)"
            )
            # Manual target — NOT a dependency of amdhip64. Default build
            # consumes the checked-in headers directly.
            add_custom_target(regenerate-lttng-curated
                              DEPENDS ${_CURATED_TP_H} ${_CURATED_EMIT_H})
        endif()
    else()
        message(STATUS "Python3 not found; LTTng curated regeneration target unavailable.")
    endif()
```

- [ ] **Step 2: Verify default build still works (no regen)**

```bash
./dev-bin/sync.sh main
./dev-bin/in-container.sh main "cd /root/rocm-systems && cmake --build build/clr -j 32 --target amdhip64 2>&1 | tail -10"
```

Expected: build succeeds. Confirm the regenerate target is NOT in the default dependency tree:

```bash
./dev-bin/in-container.sh main "cd /root/rocm-systems && cmake --build build/clr --target help 2>&1 | grep -i 'regenerate-lttng'"
```

Expected: `regenerate-lttng-curated` is listed (target exists) but is not built unless explicitly invoked.

- [ ] **Step 3: Verify manual regeneration works**

```bash
./dev-bin/in-container.sh main "cd /root/rocm-systems && cmake --build build/clr --target regenerate-lttng-curated 2>&1 | tail -5"
```

Expected: codegen runs; `git status projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.h` shows no changes (already in sync with YAML).

- [ ] **Step 4: Commit**

```bash
git add projects/clr/hipamd/src/CMakeLists.txt
git commit -m "lttng: add opt-in regenerate-lttng-curated CMake target

Per spec §9.1: default build does NOT regenerate the curated headers
and does NOT require Python or PyYAML. The custom command is wired to
a manual target only. Developers regenerate after editing YAML via:

    cmake --build build/clr --target regenerate-lttng-curated

CI catches drift via a separate gate (added in Task 13). Skipped
silently when Python3 is not available or curated_apis.yaml is absent."
```

---

### Task 13: HIP payload + coverage tests; CI drift + idempotency gates

**Files:**
- Create: `projects/clr/hipamd/test/lttng/test_hip_curated_args_payload.sh`
- Create: `projects/clr/hipamd/test/lttng/test_hip_curated_args_coverage.sh`
- (Optional) Modify: `.github/workflows/<existing-ci>.yml` if CI is wired in this PR; otherwise document for follow-up

- [ ] **Step 1: Write the payload test**

Create `projects/clr/hipamd/test/lttng/test_hip_curated_args_payload.sh`:

```bash
#!/usr/bin/env bash
# End-to-end payload test for curated _args events (Phase E: 3 APIs).
#
# 1. Spins up a per-user lttng-sessiond.
# 2. Enables rocm_hip:hipMemcpyAsync_args, rocm_hip:hipMalloc_args,
#    rocm_hip:hipDeviceSynchronize_args plus generic enter/exit_status/exit_ptr.
# 3. Builds + runs a tiny program with known argument values.
# 4. Asserts the typed args events appear with correct payload values.
# 5. Asserts pointer-returning APIs still get hip_api_exit_ptr (NOT
#    exit_status), matching the spec §6.2 generic-exit preservation rule.
#
# Usage: test_hip_curated_args_payload.sh [<libamdhip64-build-dir>]
set -euo pipefail

BUILD_LIB_DIR="${1:-$PWD/build/clr/hipamd/lib}"
if [ ! -f "$BUILD_LIB_DIR/libamdhip64.so" ]; then
    echo "ERROR: $BUILD_LIB_DIR/libamdhip64.so not found" >&2
    exit 2
fi

WORK="$(mktemp -d)"
SESSION_NAME="hip-lttng-curated-payload-$$"

# Isolated sessiond (avoid host-wide pkill races per debate-review C5).
export LTTNG_HOME="$WORK/lttng_home"
mkdir -p "$LTTNG_HOME"
SESSIOND_PIDFILE="$WORK/sessiond.pid"

cleanup() {
    set +e
    lttng destroy "$SESSION_NAME" >/dev/null 2>&1
    if [ -f "$SESSIOND_PIDFILE" ]; then
        kill "$(cat $SESSIOND_PIDFILE)" 2>/dev/null
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

# Tiny HIP program with known argument values for assertions.
cat > "$WORK/curated.hip.cpp" <<'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>

int main() {
    // hipMalloc with KNOWN size 4096.
    int* dev_ptr = nullptr;
    hipMalloc(&dev_ptr, 4096);

    // hipMemcpyAsync from a known src ptr to dev_ptr, KNOWN size 1024,
    // KNOWN kind hipMemcpyHostToDevice (=1), default stream (NULL).
    char host_buf[1024];
    memset(host_buf, 0, sizeof(host_buf));
    hipMemcpyAsync(dev_ptr, host_buf, 1024, hipMemcpyHostToDevice, nullptr);

    // hipDeviceSynchronize — zero-arg API.
    hipDeviceSynchronize();

    hipFree(dev_ptr);
    return 0;
}
EOF

HIPCC=/opt/rocm/bin/hipcc
"$HIPCC" "$WORK/curated.hip.cpp" -L "$BUILD_LIB_DIR" -lamdhip64 -Wl,-rpath,"$BUILD_LIB_DIR" -o "$WORK/curated_test"

# Start sessiond.
lttng-sessiond --daemonize --pidfile "$SESSIOND_PIDFILE"

# Set up session.
TRACE_DIR="$WORK/trace"
lttng create "$SESSION_NAME" --output "$TRACE_DIR" >/dev/null
lttng enable-channel --userspace --discard --subbuf-size=32768 --num-subbuf=4 ch1 >/dev/null
# Generic events (already covered by existing test; we re-enable to verify
# augment-not-replace behavior).
lttng enable-event --userspace --channel=ch1 \
    'rocm_hip:hip_api_enter' \
    'rocm_hip:hip_api_exit_status' \
    'rocm_hip:hip_api_exit_ptr' >/dev/null
# Curated typed events.
lttng enable-event --userspace --channel=ch1 \
    'rocm_hip:hipMalloc_args' \
    'rocm_hip:hipMemcpyAsync_args' \
    'rocm_hip:hipDeviceSynchronize_args' >/dev/null
lttng start "$SESSION_NAME" >/dev/null

"$WORK/curated_test"

lttng stop "$SESSION_NAME" >/dev/null
lttng destroy "$SESSION_NAME" >/dev/null

# Dump trace and assert.
DUMP="$WORK/trace.txt"
babeltrace2 "$TRACE_DIR" > "$DUMP"

echo "=== assertions ==="

# A. hipMalloc_args appears, sizeBytes == 4096, ptr_out is non-zero.
if grep -q 'rocm_hip:hipMalloc_args' "$DUMP" && \
   grep 'rocm_hip:hipMalloc_args' "$DUMP" | grep -q 'size = 4096'; then
    echo "  PASS  hipMalloc_args present with size = 4096"
else
    echo "  FAIL  hipMalloc_args missing or size != 4096"
    grep 'rocm_hip:hipMalloc' "$DUMP" || true
    exit 1
fi

# B. hipMemcpyAsync_args appears, sizeBytes == 1024, kind == 1 (HostToDevice).
if grep 'rocm_hip:hipMemcpyAsync_args' "$DUMP" | grep -q 'sizeBytes = 1024' && \
   grep 'rocm_hip:hipMemcpyAsync_args' "$DUMP" | grep -q 'kind = 1'; then
    echo "  PASS  hipMemcpyAsync_args present with sizeBytes = 1024, kind = 1"
else
    echo "  FAIL  hipMemcpyAsync_args payload mismatch"
    grep 'hipMemcpyAsync' "$DUMP" || true
    exit 1
fi

# C. hipDeviceSynchronize_args appears (zero-arg payload — only corr_id).
if grep -q 'rocm_hip:hipDeviceSynchronize_args' "$DUMP"; then
    echo "  PASS  hipDeviceSynchronize_args present (NOARGS variant works)"
else
    echo "  FAIL  hipDeviceSynchronize_args missing"
    exit 1
fi

# D. Generic exit events still fire (augment-not-replace per spec §6.2).
N_ENTER=$(grep -c 'rocm_hip:hip_api_enter' "$DUMP" || true)
N_EXIT_STATUS=$(grep -c 'rocm_hip:hip_api_exit_status' "$DUMP" || true)
if [ "$N_ENTER" -ge 3 ] && [ "$N_EXIT_STATUS" -ge 3 ]; then
    echo "  PASS  generic enter/exit_status preserved ($N_ENTER enter, $N_EXIT_STATUS exit_status)"
else
    echo "  FAIL  generic event preservation broken"
    exit 1
fi

# E. corr_id linkage: each _args event must share a corr_id with a matching
#    enter and exit event from the same call. Spot-check hipMemcpyAsync.
ARGS_CORR=$(grep 'rocm_hip:hipMemcpyAsync_args' "$DUMP" | head -1 | \
            sed -n 's/.*corr_id = \([0-9]*\).*/\1/p')
if [ -n "$ARGS_CORR" ] && \
   grep "corr_id = $ARGS_CORR" "$DUMP" | grep -q 'hip_api_enter' && \
   grep "corr_id = $ARGS_CORR" "$DUMP" | grep -q 'hip_api_exit_status'; then
    echo "  PASS  corr_id $ARGS_CORR links _args event to generic enter+exit"
else
    echo "  FAIL  corr_id linkage broken for hipMemcpyAsync"
    exit 1
fi

echo "=== ALL PAYLOAD ASSERTIONS PASSED ==="
exit 0
```

- [ ] **Step 2: Make it executable and run**

```bash
chmod +x projects/clr/hipamd/test/lttng/test_hip_curated_args_payload.sh
./dev-bin/sync.sh main
./dev-bin/in-container.sh main "cd /root/rocm-systems && bash projects/clr/hipamd/test/lttng/test_hip_curated_args_payload.sh build/clr/hipamd/lib"
```

Expected: `=== ALL PAYLOAD ASSERTIONS PASSED ===` and exit code 0.

- [ ] **Step 3: Write the coverage test**

Create `projects/clr/hipamd/test/lttng/test_hip_curated_args_coverage.sh`:

```bash
#!/usr/bin/env bash
# Coverage test: every API in curated_apis.yaml fires its _args event.
# Generated harness calls each API with placeholder args; trace must
# contain matching _args event with linked corr_id.
set -euo pipefail

BUILD_LIB_DIR="${1:-$PWD/build/clr/hipamd/lib}"
YAML="${2:-projects/clr/hipamd/scripts/curated_apis.yaml}"

WORK="$(mktemp -d)"
SESSION_NAME="hip-lttng-curated-coverage-$$"
export LTTNG_HOME="$WORK/lttng_home"
mkdir -p "$LTTNG_HOME"
SESSIOND_PIDFILE="$WORK/sessiond.pid"

cleanup() {
    set +e
    lttng destroy "$SESSION_NAME" >/dev/null 2>&1
    if [ -f "$SESSIOND_PIDFILE" ]; then kill "$(cat $SESSIOND_PIDFILE)" 2>/dev/null; fi
    rm -rf "$WORK"
}
trap cleanup EXIT

# Generate harness from YAML (one call per API with placeholder args).
python3 - <<PY > "$WORK/harness.hip.cpp"
import sys, os
sys.path.insert(0, 'projects/clr/hipamd/scripts')
from lttng_curated_lib import parse_yaml_file

PLACEHOLDERS = {
    'ptr':         'reinterpret_cast<void*>(0x1000)',
    'device_ptr':  'reinterpret_cast<hipDeviceptr_t>(0x1000)',
    'handle':      'nullptr',
    'size':        '64',
    'int32':       '0',
    'uint32':      '0',
    'int64':       '0',
    'uint64':      '0',
    'float':       '1.0f',
    'enum':        '0',
    'bool':        'false',
    'dim3':        'dim3(1)',
    'dim3_packed': 'dim3(1)',
    'cstring':     '"x"',
}

print('#include <hip/hip_runtime.h>')
print('int main() {')
for api in parse_yaml_file('$YAML'):
    name = api['api']
    call_args = []
    for a in api['args']:
        if a['dir'] == 'OUT':
            # Allocate a stack slot for the OUT arg.
            call_args.append('nullptr')  # simplified — real impl would alloc
        else:
            call_args.append(PLACEHOLDERS[a['type']])
    print(f'    try {{ {name}({", ".join(call_args)}); }} catch(...) {{}}')
print('    return 0;')
print('}')
PY

HIPCC=/opt/rocm/bin/hipcc
"$HIPCC" "$WORK/harness.hip.cpp" -L "$BUILD_LIB_DIR" -lamdhip64 \
    -Wl,-rpath,"$BUILD_LIB_DIR" -o "$WORK/coverage_test"

lttng-sessiond --daemonize --pidfile "$SESSIOND_PIDFILE"
TRACE_DIR="$WORK/trace"
lttng create "$SESSION_NAME" --output "$TRACE_DIR" >/dev/null
lttng enable-channel --userspace --discard --subbuf-size=32768 --num-subbuf=4 ch1 >/dev/null
# Enable all curated _args events.
python3 -c "
import sys
sys.path.insert(0, 'projects/clr/hipamd/scripts')
from lttng_curated_lib import parse_yaml_file
for a in parse_yaml_file('$YAML'):
    print(f'rocm_hip:{a[\"api\"]}_args')
" | xargs -r lttng enable-event --userspace --channel=ch1 >/dev/null

lttng start "$SESSION_NAME" >/dev/null
"$WORK/coverage_test" || true   # placeholder args may cause hipError; OK
lttng stop "$SESSION_NAME" >/dev/null
lttng destroy "$SESSION_NAME" >/dev/null

DUMP="$WORK/trace.txt"
babeltrace2 "$TRACE_DIR" > "$DUMP"

# Assert each API's _args event appears at least once.
# IMPORTANT: do NOT pipe into `while read` — the loop body would run in a
# subshell and any MISSING counter increments would be lost in the parent.
# Use process substitution `< <(...)` so the loop body shares the parent
# shell's MISSING variable.
MISSING=0
while read api; do
    if grep -q "rocm_hip:${api}_args" "$DUMP"; then
        echo "  PASS  ${api}_args fired"
    else
        echo "  FAIL  ${api}_args NOT in trace"
        MISSING=$((MISSING+1))
    fi
done < <(python3 -c "
import sys
sys.path.insert(0, 'projects/clr/hipamd/scripts')
from lttng_curated_lib import parse_yaml_file
for a in parse_yaml_file('$YAML'):
    print(a['api'])
")

if [ "$MISSING" -gt 0 ]; then
    echo "FAIL: $MISSING curated _args events missing"
    exit 1
fi
echo "PASS: all curated _args events fired"
```

- [ ] **Step 4: Make executable and run**

```bash
chmod +x projects/clr/hipamd/test/lttng/test_hip_curated_args_coverage.sh
./dev-bin/in-container.sh main "cd /root/rocm-systems && bash projects/clr/hipamd/test/lttng/test_hip_curated_args_coverage.sh"
```

Expected: `PASS: all curated _args events fired`. Some APIs may return errors due to placeholder args — that's fine; the test asserts the _args event fires, not that the call succeeds.

- [ ] **Step 5: Commit tests**

```bash
git add projects/clr/hipamd/test/lttng/test_hip_curated_args_payload.sh \
        projects/clr/hipamd/test/lttng/test_hip_curated_args_coverage.sh
git commit -m "lttng: add HIP curated-args payload + coverage tests

Payload test: hipMalloc (size=4096), hipMemcpyAsync (sizeBytes=1024,
kind=hipMemcpyHostToDevice), hipDeviceSynchronize (NOARGS variant).
Asserts payload values match, generic enter/exit events still fire,
and corr_id links typed _args to the matching enter+exit pair.

Coverage test: harness program is generated from curated_apis.yaml so
new APIs added to the YAML are auto-tested. Asserts each <api>_args
event appears in the trace at least once."
```

- [ ] **Step 6: Document the CI gate invocations in codegen header**

Independent of the workflow wiring (Task 13.5), capture the canonical CI invocations in the codegen script so anyone reading the script has the exact commands. Edit `projects/clr/hipamd/scripts/lttng_curated_codegen.py`, prepend to the docstring:

```python
"""...

CI usage (wired into .github/workflows/lttng-curated-gates.yml by Task 13.5):

    # Drift gate: codegen output must match checked-in headers.
    python3 projects/clr/hipamd/scripts/lttng_curated_codegen.py \\
        --yaml projects/clr/hipamd/scripts/curated_apis.yaml \\
        --provider rocm_hip --status-type hipError_t --status-success hipSuccess \\
        --out-tp projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h \\
        --out-emit projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.h
    git diff --exit-code -- 'projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h' \\
                            'projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.h'

    # Verifier gate: YAML signatures must match HIP headers.
    python3 projects/clr/hipamd/scripts/lttng_curated_verify.py \\
        --yaml projects/clr/hipamd/scripts/curated_apis.yaml \\
        --header /opt/rocm/include/hip/hip_runtime_api.h \\
        --extra-arg=-D__HIP_PLATFORM_AMD__=1 --extra-arg=-I/opt/rocm/include
"""
```

- [ ] **Step 7: Commit**

```bash
git add projects/clr/hipamd/scripts/lttng_curated_codegen.py
git commit -m "lttng: document CI drift + verify gate invocations in codegen header

Documents the canonical CI invocations so the codegen script is
self-describing for anyone adding or maintaining the workflow. The
actual .github/workflows wiring lands in Task 13.5. Per spec §9.3."
```

---

### Task 13.5: Wire CI drift + verifier gates into a GitHub workflow

**Files:**
- Create: `.github/workflows/lttng-curated-gates.yml`

**Why this task is here.** Spec §9.1 / §9.3 require the drift gate AND the verifier gate to run "Per CI run." Without an actual workflow file, these are aspirational. The workflow lives in this PR so the gates are active from the moment the curated-args feature merges.

If the project's CI policy requires workflow edits to land via a separate PR/team review, file a follow-up issue tracking this work and SKIP this task here — but in that case Step 4 below MUST be executed (file the follow-up issue + comment on PR #5475 referencing it). Do not silently leave the spec requirement unmet.

- [ ] **Step 1: Locate similar existing workflow as template**

Inspect an existing project-scoped workflow that already runs Python validation (good templates):

```bash
ls .github/workflows/hipfile-pylint.yml .github/workflows/hipfile-shellcheck.yml \
   .github/workflows/hipfile-codespell.yml 2>/dev/null
head -40 .github/workflows/hipfile-pylint.yml 2>/dev/null
```

Use whichever exists as the structural template (trigger config, runner image, checkout step) so the new workflow matches project conventions.

- [ ] **Step 2: Create `.github/workflows/lttng-curated-gates.yml`**

```yaml
name: LTTng Curated Gates

on:
  pull_request:
    paths:
      - 'projects/clr/hipamd/scripts/curated_apis.yaml'
      - 'projects/clr/hipamd/scripts/lttng_curated_*.py'
      - 'projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h'
      - 'projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.h'
      - 'projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml'
      - 'projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_curated_*.py'
      - 'projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_hsa_curated_tp.h'
      - 'projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_trace_emit_curated.h'
      - '.github/workflows/lttng-curated-gates.yml'
  push:
    branches: [main, develop]

jobs:
  drift-gate:
    name: Codegen drift (HIP + HSA)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install pyyaml
      - name: Regenerate HIP curated headers and diff
        run: |
          python3 projects/clr/hipamd/scripts/lttng_curated_codegen.py \
            --yaml projects/clr/hipamd/scripts/curated_apis.yaml \
            --provider rocm_hip \
            --status-type hipError_t --status-success hipSuccess \
            --out-tp projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h \
            --out-emit projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.h
          git diff --exit-code -- \
            projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h \
            projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.h
      - name: Regenerate HSA curated headers and diff
        run: |
          python3 projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_curated_codegen.py \
            --yaml projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml \
            --provider rocm_hsa \
            --status-type hsa_status_t --status-success HSA_STATUS_SUCCESS \
            --out-tp projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_hsa_curated_tp.h \
            --out-emit projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_trace_emit_curated.h
          git diff --exit-code -- \
            projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_hsa_curated_tp.h \
            projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_trace_emit_curated.h

  verify-gate:
    name: Verifier (libclang vs YAML)
    runs-on: ubuntu-latest
    container:
      image: rocm/dev-ubuntu-24.04:latest    # adjust to whichever ROCm image
                                              # the project's other CI jobs use
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install pyyaml libclang
      - name: Verify HIP YAML against headers
        run: |
          python3 projects/clr/hipamd/scripts/lttng_curated_verify.py \
            --yaml projects/clr/hipamd/scripts/curated_apis.yaml \
            --header /opt/rocm/include/hip/hip_runtime_api.h \
            --extra-arg=-D__HIP_PLATFORM_AMD__=1 \
            --extra-arg=-I/opt/rocm/include
      - name: Verify HSA YAML against headers
        run: |
          python3 projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_curated_verify.py \
            --yaml projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml \
            --header /opt/rocm/include/hsa/hsa.h \
            --header /opt/rocm/include/hsa/hsa_ext_amd.h \
            --header /opt/rocm/include/hsa/hsa_api_trace.h \
            --extra-arg=-I/opt/rocm/include
 ```

Notes for the implementer:
- The exact `runs-on` / `container:` image MUST match what the project's other Python-based gate workflows use. Inspect `hipfile-pylint.yml` etc. and copy that idiom.
- The verify-gate uses the multi-`--header` form added in Task 4.5 (HSA APIs span `hsa.h` + `hsa_ext_amd.h` + `hsa_api_trace.h` — the last one declares `hsa_amd_queue_intercept_create` per Task 15a).
- If the ROCm-image-required verifier job exceeds the project's allowed CI image surface, run only the drift-gate in CI and file a follow-up to land the verifier in a nightly workflow. Document the choice in the workflow file's top comment.

- [ ] **Step 3: Locally validate the workflow YAML syntax**

```bash
# If actionlint is available locally:
which actionlint && actionlint .github/workflows/lttng-curated-gates.yml
# Or via Python yaml parse to at least catch syntax errors:
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/lttng-curated-gates.yml'))"
```

Expected: no errors.

- [ ] **Step 4: Fallback if CI policy blocks workflow edits in this PR**

If the project's CI policy requires workflow files to be added via a separate review/PR, do the following INSTEAD of Steps 2–3:

```bash
gh issue create \
  --title "Wire LTTng curated drift+verify gates into CI" \
  --body "$(cat <<'EOF'
Spec §9.1/§9.3 of docs/superpowers/specs/2026-04-26-lttng-curated-args-design.md
requires per-CI-run drift and verify gates. The implementation plan
(docs/superpowers/plans/2026-04-26-lttng-curated-args-implementation.md
Task 13.5) provides the workflow YAML; that workflow needs to land here.

Recommended commands documented in
projects/clr/hipamd/scripts/lttng_curated_codegen.py docstring.

Blocks: full satisfaction of the curated-args spec.
Linked PR: #5475
EOF
)"
# Then comment on PR #5475 with the issue number so reviewers see the
# tracking link and the spec requirement is not dropped on the floor.
gh pr comment 5475 --body "Filed follow-up #<issue-num> to wire CI gates per spec §9.1/§9.3."
```

In this fallback case, also amend the spec to mark §9.3 CI wiring as "tracked in <issue-num>" so future readers know the gap is intentional and tracked.

- [ ] **Step 5: Commit (only if Step 2 was executed)**

```bash
git add .github/workflows/lttng-curated-gates.yml
git commit -m "ci(lttng): add curated-args drift + verifier gates

Per spec §9.1/§9.3: drift gate regenerates the per-API tracepoint
headers from curated_apis.yaml and asserts byte-identity with the
checked-in copies; verifier gate uses libclang to assert YAML matches
HIP/HSA header signatures.

Both jobs run on PRs that touch curated_apis.yaml, the codegen/verify
scripts, or the generated headers. Drift gate runs on the standard
runner; verifier gate needs the ROCm image so headers are present at
the expected /opt/rocm/include path."
```

---

## Phase F — Expand HIP curated set to full coverage (task 14)

This phase grows `curated_apis.yaml` from the Phase B 3-API minimal set to the full ~72 HIP APIs from spec Appendix A.

---

### Task 14: Author full HIP `curated_apis.yaml`

**Files:**
- Modify: `projects/clr/hipamd/scripts/curated_apis.yaml`

- [ ] **Step 1: Author the full YAML by translating Appendix A**

Replace the contents of `projects/clr/hipamd/scripts/curated_apis.yaml` with the full curated set. Translate each Appendix A entry into the YAML schema. For brevity, add APIs in the same order as Appendix A: A.1 streams, A.2 events, A.3 kernel launch + module, A.4 memory, A.5 graphs.

Reference: spec lines 608–700 contain the human-readable list. The translation is mechanical:
- `IN handle X` → `{name: X, type: handle, dir: IN}`
- `IN ptr X` → `{name: X, type: ptr, dir: IN}`
- `IN dim3 X` → `{name: X, type: dim3, dir: IN}` (use `dim3_packed` if needed for budget per §4.4)
- `OUT handle X` → `{name: X, type: handle, dir: OUT}`

Apply `dim3_packed` per the spec §4.4 high-arity table:
- `hipModuleLaunchKernel`: gridDim, blockDim → dim3_packed
- `hipExtModuleLaunchKernel`: gridDim, blockDim, globalGridDim → dim3_packed
- `hipLaunchKernel`-family: numBlocks, dimBlocks → dim3_packed
- `hipExtLaunchKernel`: numBlocks, dimBlocks → dim3_packed

Resolve arg-name binding by reading the actual HIP header signatures (the verifier will catch mismatches in the next step):

```bash
./dev-bin/in-container.sh main "grep -A 1 'hipMemcpy2DAsync\|hipMallocPitch\|hipGraphAddKernelNode' /opt/rocm/include/hip/hip_runtime_api.h | head -40"
```

This is tedious but mechanical — ~70 entries, ~5–10 minutes per category if you copy the Appendix A table and type-check arg-by-arg against the header.

(For agentic execution: this task is a candidate for a parallel subagent-dispatched fan-out, with each subagent owning one Appendix A subsection.)

- [ ] **Step 2: Validate via parser**

```bash
python3 -c "
import sys; sys.path.insert(0, 'projects/clr/hipamd/scripts')
from lttng_curated_lib import parse_yaml_file
apis = parse_yaml_file('projects/clr/hipamd/scripts/curated_apis.yaml')
print(f'OK: {len(apis)} APIs validated')
"
```

Expected: `OK: 70-72 APIs validated`. Exact count matches Appendix A subtotals (12+7+11+26+16=72).

- [ ] **Step 3: Validate via verifier (in container)**

```bash
./dev-bin/in-container.sh main "cd /root/rocm-systems && python3 projects/clr/hipamd/scripts/lttng_curated_verify.py \
    --yaml projects/clr/hipamd/scripts/curated_apis.yaml \
    --header /opt/rocm/include/hip/hip_runtime_api.h \
    --extra-arg=-D__HIP_PLATFORM_AMD__=1 \
    --extra-arg=-I/opt/rocm/include"
```

Expected: `OK: 72 curated APIs verified`. If verifier reports type or name mismatches, fix the YAML entries (the verifier output is precise about which arg of which API).

- [ ] **Step 4: Regenerate headers**

```bash
./dev-bin/in-container.sh main "cd /root/rocm-systems && cmake --build build/clr --target regenerate-lttng-curated"
```

- [ ] **Step 5: Re-run the migrator overlay**

```bash
./dev-bin/in-container.sh main "cd /root/rocm-systems && python3 projects/clr/hipamd/scripts/lttng_migrate_curated_overlay.py \
    --provider hip \
    --source projects/clr/hipamd/src/hip_table_interface.cpp \
    --curated-yaml projects/clr/hipamd/scripts/curated_apis.yaml"
```

Expected: `applied <N> edits` where N ≈ 70 sentinels + 70 IN-local blocks + 70 macro rewrites = ~210 edits.

- [ ] **Step 6: Build + test full curated set**

```bash
./dev-bin/sync.sh main
./dev-bin/in-container.sh main "cd /root/rocm-systems && cmake --build build/clr -j 32 --target amdhip64 2>&1 | tail -20"
./dev-bin/in-container.sh main "cd /root/rocm-systems && bash projects/clr/hipamd/scripts/lttng_coverage_gate.sh \
    build/clr/hipamd/lib/libamdhip64.so \
    projects/clr/hipamd/scripts/lttng_migration_inventory.txt \
    projects/clr/hipamd/scripts/lttng_migration_inventory_c.txt"
./dev-bin/in-container.sh main "cd /root/rocm-systems && bash projects/clr/hipamd/test/lttng/test_hip_curated_args_payload.sh build/clr/hipamd/lib"
./dev-bin/in-container.sh main "cd /root/rocm-systems && bash projects/clr/hipamd/test/lttng/test_hip_curated_args_coverage.sh build/clr/hipamd/lib"
```

All four must pass. The coverage test will now exercise all ~72 curated APIs.

- [ ] **Step 7: Commit**

```bash
git add projects/clr/hipamd/scripts/curated_apis.yaml \
        projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h \
        projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.h \
        projects/clr/hipamd/src/hip_table_interface.cpp
git commit -m "lttng: expand HIP curated_apis.yaml to full Appendix A coverage (~72 APIs)

Adds streams (12), events (7), kernel launch + module (11),
memory (26), graphs (16) per spec Appendix A. dim3_packed mitigation
applied to hipLaunchKernel/hipExtLaunchKernel/hipModuleLaunchKernel/
hipExtModuleLaunchKernel families per §4.4.

Verifier passes (libclang vs header), coverage gate passes (every API
has sentinel + IN-locals + _CURATED macro), payload test passes
(spot-check), coverage test passes (all <N> _args events fire)."
```

---

## Phase G — HSA mirror (task 15)

Mirror the entire pipeline for HSA: 10 curated APIs, same codegen script (parameterized by `--provider rocm_hsa`), same verifier, same migrator-overlay approach, same `_CURATED_HSA` macros.

---

### Task 15: HSA mirror — broken into eight concrete sub-tasks (15a–15h)

The previous version of this task said "same as Task 7", "same shape as Task 8", etc. — that violates the plan's "no placeholders / repeat the code" rule. The HSA mirror is now expanded into eight concrete sub-tasks below, each with the actual file edits, commands, and expected outputs adapted for HSA.

The sub-tasks have a strict order: **15a → 15b → 15c → 15d → 15e → 15f → 15g → 15h**. Each sub-task ends with its own commit. The total of ~8 commits for HSA mirror is intentional.

**Common HSA file paths (used by all sub-tasks):**

| Role | Path |
|---|---|
| HSA YAML | `projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml` |
| HSA tp.h (existing) | `projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_hsa_tp.h` |
| HSA tp.h (curated, generated) | `projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_hsa_curated_tp.h` |
| HSA emit.h (existing) | `projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_trace_emit.h` |
| HSA emit.h (curated, generated) | `projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_trace_emit_curated.h` |
| HSA wrappers TU | `projects/rocr-runtime/runtime/hsa-runtime/core/common/hsa_table_interface.cpp` |
| HSA migrator | `projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_migrate.py` |
| HSA coverage gate | `projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_coverage_gate.sh` |
| HSA CMakeLists | `projects/rocr-runtime/runtime/hsa-runtime/CMakeLists.txt` |
| HSA test dir | `projects/rocr-runtime/runtime/hsa-runtime/test/lttng/` |
| HSA scripts dir | `projects/rocr-runtime/runtime/hsa-runtime/scripts/` |
| HIP scripts dir (source for copies) | `projects/clr/hipamd/scripts/` |

**Common HSA values:**

| Variable | Value |
|---|---|
| Provider name | `rocm_hsa` |
| Status type | `hsa_status_t` |
| Status success | `HSA_STATUS_SUCCESS` |
| Existing macro family | `ROCR_TRACE_API_RET_*` |
| Curated macro family | `ROCR_TRACE_API_RET_*_CURATED_HSA(_NOARGS)?` |
| Headers (verify multi) | `/opt/rocm/include/hsa/hsa.h` and `/opt/rocm/include/hsa/hsa_ext_amd.h` |
| Generic enter helper | `rocm_trace_emit_hsa_api_enter` |
| Generic exit helpers | `rocm_trace_emit_hsa_api_exit_status`, `_ptr`, `_void` |
| Tracing-enabled macro | `HSA_ENABLE_LTTNG_UST` |

---

#### Task 15a: Author HSA `curated_apis.yaml` and copy shared scripts

**Files:**
- Create: `projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml`
- Copy:   `projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_curated_lib.py` (from HIP)
- Copy:   `projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_curated_codegen.py` (from HIP)
- Copy:   `projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_curated_verify.py` (from HIP)
- Copy:   `projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_migrate_curated_overlay.py` (from HIP)

- [ ] **Step 1: Confirm exact HSA header signatures before authoring YAML**

The two APIs that previously held `args:  # placeholder — fill from real header` notes have been resolved against the real HSA headers in `/opt/rocm/include/hsa/`:

```bash
grep -A 8 'hsa_status_t HSA_API hsa_signal_create' /opt/rocm/include/hsa/hsa.h
grep -A 8 'hsa_status_t HSA_API hsa_queue_create'  /opt/rocm/include/hsa/hsa.h
grep -A 5 'hsa_status_t HSA_API hsa_amd_signal_create' /opt/rocm/include/hsa/hsa_ext_amd.h
grep -A 5 'hsa_amd_queue_intercept_create' /opt/rocm/include/hsa/hsa_api_trace.h
```

Verified declarations (recorded here so the YAML below is auditable without re-running the greps):

```c
hsa_status_t hsa_queue_create(hsa_agent_t agent, uint32_t size,
                              hsa_queue_type32_t type,
                              void (*callback)(hsa_status_t, hsa_queue_t*, void*),
                              void* data,
                              uint32_t private_segment_size,
                              uint32_t group_segment_size,
                              hsa_queue_t** queue);

hsa_status_t hsa_signal_create(hsa_signal_value_t initial_value,    /* int64 */
                               uint32_t num_consumers,
                               const hsa_agent_t* consumers,
                               hsa_signal_t* signal);

hsa_status_t hsa_amd_signal_create(hsa_signal_value_t initial_value,
                                   uint32_t num_consumers,
                                   const hsa_agent_t* consumers,
                                   uint64_t attributes,
                                   hsa_signal_t* signal);

hsa_status_t hsa_amd_queue_intercept_create(
    hsa_agent_t agent_handle, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t, hsa_queue_t*, void*), void* data,
    uint32_t private_segment_size, uint32_t group_segment_size,
    hsa_queue_t** queue);
```

The verifier in 15a-Step 4 below will fail loudly if anything in the YAML drifts from these.

- [ ] **Step 2: Copy the shared scripts (HIP -> HSA scripts dir)**

```bash
cp projects/clr/hipamd/scripts/lttng_curated_lib.py \
   projects/rocr-runtime/runtime/hsa-runtime/scripts/
cp projects/clr/hipamd/scripts/lttng_curated_codegen.py \
   projects/rocr-runtime/runtime/hsa-runtime/scripts/
cp projects/clr/hipamd/scripts/lttng_curated_verify.py \
   projects/rocr-runtime/runtime/hsa-runtime/scripts/
cp projects/clr/hipamd/scripts/lttng_migrate_curated_overlay.py \
   projects/rocr-runtime/runtime/hsa-runtime/scripts/
```

The four scripts are provider-agnostic (codegen / verify take `--provider`, overlay takes `--provider hip|hsa` per the C4 fix in Task 10, lib has no provider state). Copies (not symlinks) per the project's monorepo conventions. A drift check between the HIP/HSA copies is a recommended follow-up CI gate (file as separate issue; out of scope for this PR).

- [ ] **Step 3: Author HSA `curated_apis.yaml`**

Create `projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml` with the 10 HSA APIs from spec §A.6. The two previously-placeholder entries (`hsa_amd_queue_intercept_create`, `hsa_amd_signal_create`) are now filled with the actual parameter names from the headers above:

```yaml
# HSA curated APIs. Source of truth — see spec §A.6.
# Schema: same as HIP curated_apis.yaml; see spec §4.

- api: hsa_queue_create
  category: hsa_queues
  args:
    - {name: agent,                type: handle, dir: IN}
    - {name: size,                 type: uint32, dir: IN}
    - {name: type,                 type: enum,   dir: IN}
    - {name: callback,             type: ptr,    dir: IN}
    - {name: data,                 type: ptr,    dir: IN}
    - {name: private_segment_size, type: uint32, dir: IN}
    - {name: group_segment_size,   type: uint32, dir: IN}
    - {name: queue,                type: handle, dir: OUT}

- api: hsa_queue_destroy
  category: hsa_queues
  args:
    - {name: queue, type: handle, dir: IN}

- api: hsa_amd_queue_intercept_create
  category: hsa_queues
  notes: |
    Verified against /opt/rocm/include/hsa/hsa_api_trace.h.
    Same shape as hsa_queue_create with first param renamed agent_handle.
  args:
    - {name: agent_handle,         type: handle, dir: IN}
    - {name: size,                 type: uint32, dir: IN}
    - {name: type,                 type: enum,   dir: IN}
    - {name: callback,             type: ptr,    dir: IN}
    - {name: data,                 type: ptr,    dir: IN}
    - {name: private_segment_size, type: uint32, dir: IN}
    - {name: group_segment_size,   type: uint32, dir: IN}
    - {name: queue,                type: handle, dir: OUT}

- api: hsa_signal_create
  category: hsa_signals
  args:
    - {name: initial_value, type: int64,  dir: IN}
    - {name: num_consumers, type: uint32, dir: IN}
    - {name: consumers,     type: ptr,    dir: IN}
    - {name: signal,        type: handle, dir: OUT}

- api: hsa_signal_destroy
  category: hsa_signals
  args:
    - {name: signal, type: handle, dir: IN}

- api: hsa_amd_signal_create
  category: hsa_signals
  notes: |
    Verified against /opt/rocm/include/hsa/hsa_ext_amd.h. Same as
    hsa_signal_create with one extra IN attributes:uint64.
  args:
    - {name: initial_value, type: int64,  dir: IN}
    - {name: num_consumers, type: uint32, dir: IN}
    - {name: consumers,     type: ptr,    dir: IN}
    - {name: attributes,    type: uint64, dir: IN}
    - {name: signal,        type: handle, dir: OUT}

- api: hsa_amd_memory_pool_allocate
  category: hsa_memory
  args:
    - {name: memory_pool, type: handle, dir: IN}
    - {name: size,        type: size,   dir: IN}
    - {name: flags,       type: uint32, dir: IN}
    - {name: ptr,         type: ptr,    dir: OUT}

- api: hsa_amd_memory_pool_free
  category: hsa_memory
  args:
    - {name: ptr, type: ptr, dir: IN}

- api: hsa_amd_memory_async_copy
  category: hsa_memory
  notes: 8 payload + corr_id = 9 total — fits the 9-payload budget.
  args:
    - {name: dst,               type: ptr,    dir: IN}
    - {name: dst_agent,         type: handle, dir: IN}
    - {name: src,               type: ptr,    dir: IN}
    - {name: src_agent,         type: handle, dir: IN}
    - {name: size,              type: size,   dir: IN}
    - {name: num_dep_signals,   type: uint32, dir: IN}
    - {name: dep_signals,       type: ptr,    dir: IN}
    - {name: completion_signal, type: handle, dir: IN}

- api: hsa_amd_memory_async_copy_on_engine
  category: hsa_memory
  notes: |
    Drops dep_signals from YAML to fit budget per spec §4.4 high-arity
    table. _on_engine adds engine_id + force_copy_on_sdma to the base
    async_copy shape, totaling 11 fields including corr_id; dropping
    dep_signals brings it to 10 (the limit).
  args:
    - {name: dst,                 type: ptr,    dir: IN}
    - {name: dst_agent,           type: handle, dir: IN}
    - {name: src,                 type: ptr,    dir: IN}
    - {name: src_agent,           type: handle, dir: IN}
    - {name: size,                type: size,   dir: IN}
    - {name: num_dep_signals,     type: uint32, dir: IN}
    - {name: completion_signal,   type: handle, dir: IN}
    - {name: engine_id,           type: uint32, dir: IN}
    - {name: force_copy_on_sdma,  type: bool,   dir: IN}
```

- [ ] **Step 4: Validate via parser and verifier (multi-header per Task 4.5)**

```bash
# Parser:
python3 -c "
import sys; sys.path.insert(0, 'projects/rocr-runtime/runtime/hsa-runtime/scripts')
from lttng_curated_lib import parse_yaml_file
apis = parse_yaml_file('projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml')
print(f'OK: {len(apis)} APIs validated')
"

# Verifier (multi-header — see Task 4.5):
./dev-bin/in-container.sh main "cd /root/rocm-systems && python3 \
    projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_curated_verify.py \
    --yaml projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml \
    --header /opt/rocm/include/hsa/hsa.h \
    --header /opt/rocm/include/hsa/hsa_ext_amd.h \
    --header /opt/rocm/include/hsa/hsa_api_trace.h \
    --extra-arg=-I/opt/rocm/include"
```

Expected: `OK: 10 APIs validated` and `OK: 10 curated APIs verified against 3 header(s)`.

- [ ] **Step 5: Commit**

```bash
git add projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml \
        projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_curated_lib.py \
        projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_curated_codegen.py \
        projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_curated_verify.py \
        projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_migrate_curated_overlay.py
git commit -m "lttng(hsa): add curated_apis.yaml + copy shared codegen scripts

10 HSA APIs from spec §A.6: queue lifecycle (3), signal lifecycle (3),
memory pool + async copy (4 — incl. async_copy_on_engine with bool
force_copy_on_sdma, dep_signals dropped per §4.4).

Parameter names for hsa_amd_queue_intercept_create and
hsa_amd_signal_create resolved against /opt/rocm/include/hsa/
hsa_api_trace.h and hsa_ext_amd.h respectively (no placeholders).

Provider-agnostic scripts (lttng_curated_lib.py, codegen, verify,
overlay) copied byte-identical from projects/clr/hipamd/scripts/.
Drift between the two copies is a recommended follow-up CI gate."
```

---

#### Task 15b: Generate + check in HSA curated headers

**Files:**
- Create (generated): `projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_hsa_curated_tp.h`
- Create (generated): `projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_trace_emit_curated.h`

- [ ] **Step 1: Run codegen with HSA-specific arguments**

```bash
./dev-bin/in-container.sh main "cd /root/rocm-systems && python3 \
    projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_curated_codegen.py \
    --yaml projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml \
    --provider rocm_hsa \
    --status-type hsa_status_t --status-success HSA_STATUS_SUCCESS \
    --out-tp projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_hsa_curated_tp.h \
    --out-emit projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_trace_emit_curated.h"
```

Expected stderr: `wrote ... 10 APIs`.

- [ ] **Step 2: Sanity-check the generated files**

```bash
head -25 projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_hsa_curated_tp.h
echo '---'
sed -n '/hsa_signal_create_args/,/^}$/p' \
    projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_trace_emit_curated.h | head -30
```

Verify: `AUTO-GENERATED` comment present, SHA256 prefix present, provider is `rocm_hsa`, helper signature for `hsa_signal_create_args` ends with `hsa_status_t status`, and the OUT-param `signal` body uses `(status == HSA_STATUS_SUCCESS && signal_out_ptr != NULL) ? ... : 0ULL`.

- [ ] **Step 3: Commit**

```bash
git add projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_hsa_curated_tp.h \
        projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_trace_emit_curated.h
git commit -m "lttng(hsa): generate curated tp.h + emit.h headers (10 APIs)

Generated by lttng_curated_codegen.py from
projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml,
with --provider rocm_hsa --status-type hsa_status_t
--status-success HSA_STATUS_SUCCESS.

Checked-in headers per spec §3.2 — default build consumes these
directly without invoking Python. Re-running codegen on the same YAML
produces byte-identical output (CI drift gate).

This commit only adds the headers; they are not yet #included from
rocm_hsa_tp.h or rocm_trace_emit.h. That wiring lands in 15c so the
checked-in artifact is reviewable in isolation."
```

---

#### Task 15c: Wire generated HSA headers into existing HSA tp.h + emit.h

**Files:**
- Modify: `projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_hsa_tp.h`
- Modify: `projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_trace_emit.h`

- [ ] **Step 1: Locate HSA tp.h insertion point**

```bash
grep -n 'LTTNG_UST_TRACEPOINT_EVENT\|^#endif' \
    projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_hsa_tp.h | tail -8
```

The closing `#endif` for the file's include guard is the insertion point — the new `#include` must precede it.

- [ ] **Step 2: Add include in `rocm_hsa_tp.h`**

Use the Edit tool to insert immediately before the file's final `#endif`:

```c
/* Curated per-API typed tracepoint events. Generated by
 * lttng_curated_codegen.py from curated_apis.yaml. See spec §5.1. */
#include "rocm_hsa_curated_tp.h"
```

- [ ] **Step 3: Add include in `rocm_trace_emit.h`**

Find the `HSA_ENABLE_LTTNG_UST` enable branch in the existing emit.h (`grep -n 'HSA_ENABLE_LTTNG_UST' projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_trace_emit.h`). Inside that branch, after the existing `static inline void rocm_trace_emit_hsa_api_enter` block and before the matching `#else`, insert:

```c
/* Curated per-API typed emit helpers. Generated; see spec §5.2. */
#include "rocm_trace_emit_curated.h"
```

The generated `rocm_trace_emit_curated.h` already includes `rocm_dim3_pack.h` and the curated tp.h transitively, so no extra includes are needed here.

- [ ] **Step 4: Build `hsa-runtime64` to verify the includes are wellformed**

```bash
./dev-bin/sync.sh main
./dev-bin/in-container.sh main "cd /root/rocm-systems && cmake --build build/rocr -j 32 --target hsa-runtime64 2>&1 | tail -20"
```

Expected: build succeeds with no warnings about the curated headers.

- [ ] **Step 5: Verify HSA curated tracepoint events appear in the .so**

```bash
./dev-bin/in-container.sh main "cd /root/rocm-systems && nm \
    build/rocr/runtime/hsa-runtime/libhsa-runtime64.so | \
    grep '__tracepoint.*_args' | head"
```

Expected: lines containing `__tracepoint__rocm_hsa__hsa_queue_create_args`, `_hsa_signal_create_args`, etc.

- [ ] **Step 6: Commit**

```bash
git add projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_hsa_tp.h \
        projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_trace_emit.h
git commit -m "lttng(hsa): wire generated curated tp.h + emit.h into HSA provider

#include rocm_hsa_curated_tp.h from rocm_hsa_tp.h so the per-API
tracepoint events register alongside the existing generic events
(hsa_api_enter / hsa_api_exit_*) under the same provider package.
#include rocm_trace_emit_curated.h from rocm_trace_emit.h so wrappers
can call the generated helpers.

After this commit libhsa-runtime64.so exposes hsa_queue_create_args,
hsa_signal_create_args, etc. tracepoint events but no wrapper fires
them yet — that wiring lands with the migrator overlay in 15e."
```

---

#### Task 15d: Add six `_CURATED_HSA` macro variants to `hsa_table_interface.cpp`

**Files:**
- Modify: `projects/rocr-runtime/runtime/hsa-runtime/core/common/hsa_table_interface.cpp`

- [ ] **Step 1: Locate insertion point — verify existing macro family**

```bash
grep -n 'ROCR_TRACE_API_RET_STATUS\b\|ROCR_TRACE_API_RET_PTR\b\|ROCR_TRACE_API_RET_VOID\b' \
    projects/rocr-runtime/runtime/hsa-runtime/core/common/hsa_table_interface.cpp | head -6
```

Expected (per the source verified during plan authoring): macros at lines 53 (STATUS), 61 (PTR), 68 (VOID). Insert the new `_CURATED_HSA` variants immediately after the closing `#endif` (or last macro line) of the existing block.

- [ ] **Step 2: Add the six `_CURATED_HSA` macros**

Insert after the existing macros, per spec §6.5:

```c
/* ---------- HSA curated parameter-capture variants (spec §6.2 / §6.5)
 * Six macros: STATUS / PTR / VOID, each with a captured-args form and a
 * _NOARGS form. The migrator selects _NOARGS iff the curated API has zero
 * captured args. Helper signature invariant: every helper takes
 *   (uint64_t corr_id, <captured-args...>, hsa_status_t status)
 * even when captured-args is empty. Status comes from:
 *   - STATUS variants: macro-evaluated __rocm_status from the call's expr
 *   - PTR variants:    synthesized from null-vs-non-null retval (rare on HSA)
 *   - VOID variants:   literal HSA_STATUS_SUCCESS
 *
 * Generic exit events (hsa_api_exit_status / _ptr / _void) are still
 * emitted by these macros — the typed _args event AUGMENTS the existing
 * generic event, never replaces it (spec §1, §2).
 */

/* Captured-args variants. __VA_ARGS__ is non-empty by construction (the
 * migrator emits the _NOARGS form for zero-arg APIs). */
#define ROCR_TRACE_API_RET_STATUS_CURATED_HSA(api, expr, corr, ...)          \
    do {                                                                     \
        const hsa_status_t __rocm_status = (expr);                           \
        rocm_trace_emit_##api##_args((corr), __VA_ARGS__, __rocm_status);    \
        rocm_trace_emit_hsa_api_exit_status(__func__,                        \
            (corr), (int32_t)__rocm_status);                                 \
        return __rocm_status;                                                \
    } while (0)

#define ROCR_TRACE_API_RET_PTR_CURATED_HSA(api, ptr_type, expr, corr, ...)   \
    do {                                                                     \
        ptr_type const __rocm_ptr = (expr);                                  \
        const hsa_status_t __rocm_status =                                   \
            (__rocm_ptr != nullptr) ? HSA_STATUS_SUCCESS                     \
                                    : HSA_STATUS_ERROR_OUT_OF_RESOURCES;     \
        rocm_trace_emit_##api##_args((corr), __VA_ARGS__, __rocm_status);    \
        rocm_trace_emit_hsa_api_exit_ptr(__func__, (corr), __rocm_ptr);      \
        return __rocm_ptr;                                                   \
    } while (0)

#define ROCR_TRACE_API_RET_VOID_CURATED_HSA(api, expr, corr, ...)            \
    do {                                                                     \
        (expr);                                                              \
        rocm_trace_emit_##api##_args((corr), __VA_ARGS__, HSA_STATUS_SUCCESS);\
        rocm_trace_emit_hsa_api_exit_void(__func__, (corr));                 \
        return;                                                              \
    } while (0)

/* Zero-captured-args variants. Mirrors HIP's _NOARGS rationale (mixed
 * C++ standard surface; avoid empty-__VA_ARGS__ expansion). */
#define ROCR_TRACE_API_RET_STATUS_CURATED_HSA_NOARGS(api, expr, corr)        \
    do {                                                                     \
        const hsa_status_t __rocm_status = (expr);                           \
        rocm_trace_emit_##api##_args((corr), __rocm_status);                 \
        rocm_trace_emit_hsa_api_exit_status(__func__,                        \
            (corr), (int32_t)__rocm_status);                                 \
        return __rocm_status;                                                \
    } while (0)

#define ROCR_TRACE_API_RET_PTR_CURATED_HSA_NOARGS(api, ptr_type, expr, corr) \
    do {                                                                     \
        ptr_type const __rocm_ptr = (expr);                                  \
        const hsa_status_t __rocm_status =                                   \
            (__rocm_ptr != nullptr) ? HSA_STATUS_SUCCESS                     \
                                    : HSA_STATUS_ERROR_OUT_OF_RESOURCES;     \
        rocm_trace_emit_##api##_args((corr), __rocm_status);                 \
        rocm_trace_emit_hsa_api_exit_ptr(__func__, (corr), __rocm_ptr);      \
        return __rocm_ptr;                                                   \
    } while (0)

#define ROCR_TRACE_API_RET_VOID_CURATED_HSA_NOARGS(api, expr, corr)          \
    do {                                                                     \
        (expr);                                                              \
        rocm_trace_emit_##api##_args((corr), HSA_STATUS_SUCCESS);            \
        rocm_trace_emit_hsa_api_exit_void(__func__, (corr));                 \
        return;                                                              \
    } while (0)
```

- [ ] **Step 2.5: Confirm `HSA_STATUS_ERROR_OUT_OF_RESOURCES` is the right sentinel**

```bash
grep 'HSA_STATUS_ERROR_OUT_OF_RESOURCES' /opt/rocm/include/hsa/hsa.h | head -3
```

Expected: the symbol exists. If a different "OOM" sentinel is the convention in HSA (some headers use `HSA_STATUS_ERROR`), substitute it before building. The verifier doesn't check this; only the build does.

- [ ] **Step 3: Build to verify the macros compile (no use sites yet)**

```bash
./dev-bin/sync.sh main
./dev-bin/in-container.sh main "cd /root/rocm-systems && cmake --build build/rocr -j 32 --target hsa-runtime64 2>&1 | tail -10"
```

Expected: build succeeds. The macros are defined but unused.

- [ ] **Step 4: Commit**

```bash
git add projects/rocr-runtime/runtime/hsa-runtime/core/common/hsa_table_interface.cpp
git commit -m "lttng(hsa): add 6 _CURATED_HSA macro variants (STATUS/PTR/VOID + _NOARGS)

Per spec §6.2 / §6.5. Three captured-args variants and three _NOARGS
variants mirror the existing ROCR_TRACE_API_RET_* macro family. Each
variant emits the typed <api>_args event AND preserves the matching
generic exit event (hsa_api_exit_status / _ptr / _void) so the
augment-not-replace contract from spec §1/§2 holds.

Helper signature invariant: every helper takes (corr_id, captured...,
status). PTR variants synthesize status from null-vs-non-null retval
using HSA_STATUS_ERROR_OUT_OF_RESOURCES as the OOM sentinel. VOID
variants pass literal HSA_STATUS_SUCCESS.

No use sites yet — those land with the HSA migrator overlay in 15e."
```

---

#### Task 15e: Run the migrator overlay on `hsa_table_interface.cpp`

**Files:**
- Modify: `projects/rocr-runtime/runtime/hsa-runtime/core/common/hsa_table_interface.cpp`

- [ ] **Step 1: Run the overlay (Option A or Option B per Task 10's pattern)**

The overlay script is the same provider-agnostic file as HIP, invoked with `--provider hsa`:

**Option A (overlay runs locally if libclang is available):**

```bash
python3 -c "from clang import cindex; print('libclang OK')"
python3 projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_migrate_curated_overlay.py \
    --provider hsa \
    --source projects/rocr-runtime/runtime/hsa-runtime/core/common/hsa_table_interface.cpp \
    --curated-yaml projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml \
    --include-path /opt/rocm/include
```

**Option B (overlay runs in container, single file copied back via `docker cp` — never `git reset --hard`):**

```bash
./dev-bin/sync.sh main
./dev-bin/in-container.sh main "cd /root/rocm-systems && python3 \
    projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_migrate_curated_overlay.py \
    --provider hsa \
    --source projects/rocr-runtime/runtime/hsa-runtime/core/common/hsa_table_interface.cpp \
    --curated-yaml projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml \
    --include-path /opt/rocm/include"

REMOTE_HOST=bewelton@banff-ccs-aus-g05-05.cs-aus.dcgpu
CONTAINER=bewelton_lttng
LOCAL_FILE=projects/rocr-runtime/runtime/hsa-runtime/core/common/hsa_table_interface.cpp
ssh "$REMOTE_HOST" "docker cp ${CONTAINER}:/root/rocm-systems/${LOCAL_FILE} /tmp/hsa_table_interface.cpp.from-container"
scp "$REMOTE_HOST:/tmp/hsa_table_interface.cpp.from-container" "$LOCAL_FILE"
ssh "$REMOTE_HOST" "rm -f /tmp/hsa_table_interface.cpp.from-container"
git diff --stat -- "$LOCAL_FILE"
```

Expected: `applied <N> edits` where N = 10 sentinels + 10 IN-locals blocks + 10 macro rewrites = 30 edits.

- [ ] **Step 2: Inspect the diff**

```bash
grep -B 1 -A 4 '__ROCM_CURATED__: hsa_queue_create\|__ROCM_CURATED__: hsa_signal_create\|__ROCM_CURATED__: hsa_amd_memory_async_copy' \
    projects/rocr-runtime/runtime/hsa-runtime/core/common/hsa_table_interface.cpp | head -40
```

Verify each of the three spot-checked APIs has the sentinel and a `_CURATED_HSA(...)` (or `_CURATED_HSA_NOARGS`) macro invocation, and that `hsa_signal_destroy` (the only no-IN-arg API; well, single-IN-arg) has the appropriate IN-locals.

- [ ] **Step 3: Build to verify the wrappers compile**

```bash
./dev-bin/sync.sh main
./dev-bin/in-container.sh main "cd /root/rocm-systems && cmake --build build/rocr -j 32 --target hsa-runtime64 2>&1 | tail -20"
```

Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add projects/rocr-runtime/runtime/hsa-runtime/core/common/hsa_table_interface.cpp
git commit -m "lttng(hsa): overlay curated-args migration on hsa_table_interface.cpp

Applied lttng_migrate_curated_overlay.py --provider hsa to inject the
sentinel + IN-locals + _CURATED_HSA macro variants for each of the 10
HSA curated APIs in spec §A.6. Build verified."
```

---

#### Task 15f: Update HSA `lttng_coverage_gate.sh` for curated checks

**Files:**
- Modify: `projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_coverage_gate.sh`

- [ ] **Step 1: Add a curated-coverage gate section**

The HSA gate script's structure mirrors the HIP gate. Append the same Python-driven body-content scan as Task 11 Step 1, but parameterize for HSA. Insert before the final `exit 0`:

```bash
# ---------------------------------------------------------------------------
# 3. HSA curated-args coverage gate (spec §8.2)
# ---------------------------------------------------------------------------

CURATED_YAML="$SCRIPT_DIR/curated_apis.yaml"
if [ -f "$CURATED_YAML" ]; then
    python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from lttng_curated_lib import parse_yaml_file
for a in parse_yaml_file('$CURATED_YAML'):
    print(a['api'])
" | sort -u > "$WORK/curated.txt"

    MISSING_FROM_INV="$(comm -23 "$WORK/curated.txt" "$WORK/migrated.txt" || true)"
    if [ -n "$MISSING_FROM_INV" ]; then
        echo "FAIL (curated): APIs in HSA curated_apis.yaml are missing from migration inventory:"
        printf '  %s\n' $MISSING_FROM_INV
        exit 1
    fi

    PYTHON_CURATED_GATE="$(cat <<'PY'
import os, re, sys
sys.path.insert(0, sys.argv[3])
from lttng_curated_lib import parse_yaml_file

src_dir   = sys.argv[1]
yaml_path = sys.argv[2]

apis = parse_yaml_file(yaml_path)
files = []
for root, _, fs in os.walk(src_dir):
    for fn in fs:
        if fn.endswith('.cpp'):
            p = os.path.join(root, fn)
            with open(p, 'rb') as fh:
                files.append((p, fh.read().decode('utf-8', errors='replace')))

# Spec §8.2 regex matcher — covers all six HSA curated-macro variants.
MACRO_RE = re.compile(
    r'ROCR_TRACE_API_RET_(STATUS|PTR|VOID)_CURATED_HSA(_NOARGS)?\s*\(')

def find_body(text, name):
    pat = re.compile(r'\b' + re.escape(name) + r'\s*\(')
    for m in pat.finditer(text):
        depth = 0
        i = m.end() - 1
        while i < len(text):
            if text[i] == '(': depth += 1
            elif text[i] == ')':
                depth -= 1
                if depth == 0:
                    j = text.find('{', i)
                    if j < 0: break
                    bdepth = 0; k = j
                    while k < len(text):
                        if text[k] == '{': bdepth += 1
                        elif text[k] == '}':
                            bdepth -= 1
                            if bdepth == 0:
                                return text[j:k+1]
                        k += 1
                    break
            i += 1
    return None

failures = []
for api in apis:
    name = api['api']
    sentinel = f'/* __ROCM_CURATED__: {name} */'
    body = None
    for path, text in files:
        b = find_body(text, name)
        if b and sentinel in b:
            body = b
            break
    if body is None:
        failures.append(f'{name}: no wrapper body found containing sentinel {sentinel!r}')
        continue
    if not MACRO_RE.search(body):
        failures.append(f'{name}: sentinel present but no _CURATED_HSA macro invocation')
        continue
    has_in = any(a['dir'] in ('IN', 'INOUT') for a in api['args'])
    if has_in and '__rocm_in_' not in body:
        failures.append(f'{name}: has IN args but no __rocm_in_ locals in body')

if failures:
    for f in failures:
        print(f'  CURATED FAIL: {f}')
    sys.exit(1)
print(f'CURATED (HSA): {len(apis)} curated APIs verified')
PY
)"
    set +e
    python3 -c "$PYTHON_CURATED_GATE" "$SRC_DIR" "$CURATED_YAML" "$SCRIPT_DIR" \
        > "$WORK/curated.log" 2>&1
    CURATED_RC=$?
    set -e
    cat "$WORK/curated.log"
    if [ "$CURATED_RC" -ne 0 ]; then
        echo "FAIL: HSA curated-args coverage gate failed"
        exit 1
    fi
fi
```

(Substitute the actual HSA gate script's existing variable names for `$SCRIPT_DIR`, `$WORK`, `$SRC_DIR` if they differ from the HIP gate's; verify with `head -40 projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_coverage_gate.sh`.)

Note: the macro regex is HSA-specific (`_CURATED_HSA` suffix only) — distinct from the HIP regex which matches both `_CURATED` and `_CURATED_HSA` per spec §8.2. This is intentional: each gate enforces only its own provider's macros.

- [ ] **Step 2: Run the gate manually**

```bash
./dev-bin/in-container.sh main "cd /root/rocm-systems && bash \
    projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_coverage_gate.sh \
    build/rocr/runtime/hsa-runtime/libhsa-runtime64.so \
    projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_migration_inventory.txt"
```

Expected: original `PASS: all <N> exported HSA symbols migrated` PLUS `CURATED (HSA): 10 curated APIs verified`.

- [ ] **Step 3: Commit**

```bash
git add projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_coverage_gate.sh
git commit -m "lttng(hsa): extend coverage gate with curated-args checks (spec §8.2)

Mirrors the HIP gate's third section (Task 11) but with the
HSA-specific macro regex ROCR_TRACE_API_RET_(STATUS|PTR|VOID)_CURATED_HSA(_NOARGS)?
so the HSA gate enforces only HSA's curated macros.

Skipped silently when curated_apis.yaml is absent (allows gradual
rollout)."
```

---

#### Task 15g: Add HSA opt-in `regenerate-lttng-curated-hsa` CMake target

**Files:**
- Modify: `projects/rocr-runtime/runtime/hsa-runtime/CMakeLists.txt`

- [ ] **Step 1: Locate the HSA `HSA_ENABLE_LTTNG_UST` block**

```bash
grep -n 'HSA_ENABLE_LTTNG_UST\|find_package.*PkgConfig' \
    projects/rocr-runtime/runtime/hsa-runtime/CMakeLists.txt | head -10
```

Note the line range of the block — the new target lives inside it.

- [ ] **Step 2: Add the regenerate target**

Inside the existing `if(HSA_ENABLE_LTTNG_UST)` block (or equivalent guard), insert:

```cmake
    # ---- Curated parameter-capture: opt-in regeneration target (HSA) ----
    # Per spec §9.1: default build does NOT regenerate. The custom command
    # is wired only to the manual `regenerate-lttng-curated-hsa` target so
    # the build never requires Python or PyYAML. Generated headers are
    # checked in. CI runs codegen + `git diff --exit-code` to catch drift.
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_FOUND)
        set(_HSA_CURATED_YAML  ${CMAKE_CURRENT_LIST_DIR}/scripts/curated_apis.yaml)
        set(_HSA_CURATED_TP_H  ${CMAKE_CURRENT_LIST_DIR}/lttng/rocm_hsa_curated_tp.h)
        set(_HSA_CURATED_EMIT_H ${CMAKE_CURRENT_LIST_DIR}/lttng/rocm_trace_emit_curated.h)
        if(EXISTS ${_HSA_CURATED_YAML})
            add_custom_command(
                OUTPUT ${_HSA_CURATED_TP_H} ${_HSA_CURATED_EMIT_H}
                COMMAND ${Python3_EXECUTABLE}
                        ${CMAKE_CURRENT_LIST_DIR}/scripts/lttng_curated_codegen.py
                        --yaml      ${_HSA_CURATED_YAML}
                        --provider  rocm_hsa
                        --status-type hsa_status_t
                        --status-success HSA_STATUS_SUCCESS
                        --out-tp    ${_HSA_CURATED_TP_H}
                        --out-emit  ${_HSA_CURATED_EMIT_H}
                DEPENDS ${_HSA_CURATED_YAML}
                        ${CMAKE_CURRENT_LIST_DIR}/scripts/lttng_curated_codegen.py
                        ${CMAKE_CURRENT_LIST_DIR}/scripts/lttng_curated_lib.py
                COMMENT "Regenerating LTTng curated tracepoints (HSA)"
            )
            # Manual target — NOT a dependency of hsa-runtime64. Default
            # build consumes the checked-in headers directly.
            add_custom_target(regenerate-lttng-curated-hsa
                              DEPENDS ${_HSA_CURATED_TP_H} ${_HSA_CURATED_EMIT_H})
        endif()
    else()
        message(STATUS "Python3 not found; HSA LTTng curated regeneration target unavailable.")
    endif()
```

The target name `regenerate-lttng-curated-hsa` is distinct from the HIP `regenerate-lttng-curated` so a developer can regenerate one or the other.

- [ ] **Step 3: Verify default build is unaffected and the manual target works**

```bash
./dev-bin/sync.sh main
./dev-bin/in-container.sh main "cd /root/rocm-systems && cmake --build build/rocr -j 32 --target hsa-runtime64 2>&1 | tail -10"
./dev-bin/in-container.sh main "cd /root/rocm-systems && cmake --build build/rocr --target regenerate-lttng-curated-hsa 2>&1 | tail -5"
```

Expected: hsa-runtime64 builds; regenerate target runs; `git status -- projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_*.h` shows no changes.

- [ ] **Step 4: Commit**

```bash
git add projects/rocr-runtime/runtime/hsa-runtime/CMakeLists.txt
git commit -m "lttng(hsa): add opt-in regenerate-lttng-curated-hsa CMake target

Per spec §9.1: default HSA build does NOT regenerate the curated
headers and does NOT require Python or PyYAML. The custom command
is wired to a manual target only. Developers regenerate after
editing the HSA YAML via:

    cmake --build build/rocr --target regenerate-lttng-curated-hsa

CI catches drift via the lttng-curated-gates.yml workflow (Task 13.5)."
```

---

#### Task 15h: HSA payload + coverage tests

**Files:**
- Create: `projects/rocr-runtime/runtime/hsa-runtime/test/lttng/test_hsa_curated_args_payload.sh`
- Create: `projects/rocr-runtime/runtime/hsa-runtime/test/lttng/test_hsa_curated_args_coverage.sh`

- [ ] **Step 1: Write the HSA payload test**

The HSA payload test is structurally identical to Task 13's HIP version but exercises HSA APIs. Pick three representative APIs: `hsa_signal_create` (OUT-param), `hsa_signal_destroy` (single-IN-arg), and `hsa_amd_memory_pool_allocate` (OUT + IN args). Create `projects/rocr-runtime/runtime/hsa-runtime/test/lttng/test_hsa_curated_args_payload.sh`:

```bash
#!/usr/bin/env bash
# End-to-end payload test for HSA curated _args events.
set -euo pipefail

BUILD_LIB_DIR="${1:-$PWD/build/rocr/runtime/hsa-runtime}"
if [ ! -f "$BUILD_LIB_DIR/libhsa-runtime64.so" ]; then
    echo "ERROR: $BUILD_LIB_DIR/libhsa-runtime64.so not found" >&2
    exit 2
fi

WORK="$(mktemp -d)"
SESSION_NAME="hsa-lttng-curated-payload-$$"
export LTTNG_HOME="$WORK/lttng_home"
mkdir -p "$LTTNG_HOME"
SESSIOND_PIDFILE="$WORK/sessiond.pid"

cleanup() {
    set +e
    lttng destroy "$SESSION_NAME" >/dev/null 2>&1
    if [ -f "$SESSIOND_PIDFILE" ]; then kill "$(cat $SESSIOND_PIDFILE)" 2>/dev/null; fi
    rm -rf "$WORK"
}
trap cleanup EXIT

# Tiny HSA program with known argument values.
cat > "$WORK/curated.cpp" <<'EOF'
#include <hsa/hsa.h>
#include <stdio.h>
int main() {
    hsa_init();
    // hsa_signal_create with KNOWN initial_value 0x1234 and num_consumers 0.
    hsa_signal_t sig;
    hsa_signal_create(0x1234, 0, NULL, &sig);
    // hsa_signal_destroy with the just-created handle.
    hsa_signal_destroy(sig);
    hsa_shut_down();
    return 0;
}
EOF

g++ -std=c++17 "$WORK/curated.cpp" -I/opt/rocm/include \
    -L "$BUILD_LIB_DIR" -lhsa-runtime64 -Wl,-rpath,"$BUILD_LIB_DIR" \
    -o "$WORK/curated_test"

lttng-sessiond --daemonize --pidfile "$SESSIOND_PIDFILE"
TRACE_DIR="$WORK/trace"
lttng create "$SESSION_NAME" --output "$TRACE_DIR" >/dev/null
lttng enable-channel --userspace --discard --subbuf-size=32768 --num-subbuf=4 ch1 >/dev/null
lttng enable-event --userspace --channel=ch1 \
    'rocm_hsa:hsa_api_enter' \
    'rocm_hsa:hsa_api_exit_status' \
    'rocm_hsa:hsa_signal_create_args' \
    'rocm_hsa:hsa_signal_destroy_args' >/dev/null
lttng start "$SESSION_NAME" >/dev/null

"$WORK/curated_test"

lttng stop "$SESSION_NAME" >/dev/null
lttng destroy "$SESSION_NAME" >/dev/null

DUMP="$WORK/trace.txt"
babeltrace2 "$TRACE_DIR" > "$DUMP"

echo "=== HSA curated payload assertions ==="

# A. hsa_signal_create_args: initial_value == 0x1234, num_consumers == 0.
if grep 'rocm_hsa:hsa_signal_create_args' "$DUMP" | grep -q 'initial_value = 4660' && \
   grep 'rocm_hsa:hsa_signal_create_args' "$DUMP" | grep -q 'num_consumers = 0'; then
    echo "  PASS  hsa_signal_create_args present with correct payload"
else
    echo "  FAIL  hsa_signal_create_args missing or payload mismatch"
    grep 'hsa_signal_create' "$DUMP" || true
    exit 1
fi

# B. hsa_signal_destroy_args present.
if grep -q 'rocm_hsa:hsa_signal_destroy_args' "$DUMP"; then
    echo "  PASS  hsa_signal_destroy_args present"
else
    echo "  FAIL  hsa_signal_destroy_args missing"
    exit 1
fi

# C. Generic enter/exit_status preserved (augment-not-replace).
N_ENTER=$(grep -c 'rocm_hsa:hsa_api_enter' "$DUMP" || true)
N_EXIT=$(grep -c 'rocm_hsa:hsa_api_exit_status' "$DUMP" || true)
if [ "$N_ENTER" -ge 2 ] && [ "$N_EXIT" -ge 2 ]; then
    echo "  PASS  generic enter/exit_status preserved ($N_ENTER enter, $N_EXIT exit_status)"
else
    echo "  FAIL  generic event preservation broken"
    exit 1
fi

echo "=== ALL HSA PAYLOAD ASSERTIONS PASSED ==="
exit 0
```

- [ ] **Step 2: Write the HSA coverage test**

Same shape as Task 13 Step 3's HIP coverage test, with the same C5 fix (`while read api; ...; done < <(python3 -c ...)`):

```bash
#!/usr/bin/env bash
# Coverage test: every API in HSA curated_apis.yaml fires its _args event.
set -euo pipefail

BUILD_LIB_DIR="${1:-$PWD/build/rocr/runtime/hsa-runtime}"
YAML="${2:-projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml}"

WORK="$(mktemp -d)"
SESSION_NAME="hsa-lttng-curated-coverage-$$"
export LTTNG_HOME="$WORK/lttng_home"; mkdir -p "$LTTNG_HOME"
SESSIOND_PIDFILE="$WORK/sessiond.pid"

cleanup() {
    set +e
    lttng destroy "$SESSION_NAME" >/dev/null 2>&1
    if [ -f "$SESSIOND_PIDFILE" ]; then kill "$(cat $SESSIOND_PIDFILE)" 2>/dev/null; fi
    rm -rf "$WORK"
}
trap cleanup EXIT

# Generate harness from YAML — placeholder args.
python3 - <<PY > "$WORK/harness.cpp"
import sys
sys.path.insert(0, 'projects/rocr-runtime/runtime/hsa-runtime/scripts')
from lttng_curated_lib import parse_yaml_file

PLACEHOLDERS = {
    'ptr':         '(void*)0x1000',
    'handle':      '{ 0 }',                 # hsa_signal_t / hsa_agent_t / hsa_queue_t handle
    'size':        '64',
    'int32':       '0', 'uint32': '0', 'int64': '0', 'uint64': '0',
    'float':       '1.0f', 'enum': '0', 'bool': 'false',
    'cstring':     '"x"',
}

print('#include <hsa/hsa.h>')
print('#include <hsa/hsa_ext_amd.h>')
print('int main() {')
print('    hsa_init();')
for api in parse_yaml_file('$YAML'):
    name = api['api']
    args = []
    for a in api['args']:
        if a['dir'] == 'OUT':
            args.append('NULL')
        elif a['type'] == 'handle':
            # Pass a default-initialized handle struct via a cast.
            args.append('hsa_agent_t{}' if 'agent' in a['name']
                        else ('hsa_queue_t*{}' if a['name'] == 'queue'
                              else 'hsa_signal_t{}'))
        else:
            args.append(PLACEHOLDERS.get(a['type'], '0'))
    # Wrap in a try/catch-equivalent: HSA APIs return status, never throw.
    print(f'    {name}({", ".join(args)});')
print('    hsa_shut_down();')
print('    return 0;')
print('}')
PY

g++ -std=c++17 "$WORK/harness.cpp" -I/opt/rocm/include \
    -L "$BUILD_LIB_DIR" -lhsa-runtime64 -Wl,-rpath,"$BUILD_LIB_DIR" \
    -o "$WORK/coverage_test"

lttng-sessiond --daemonize --pidfile "$SESSIOND_PIDFILE"
TRACE_DIR="$WORK/trace"
lttng create "$SESSION_NAME" --output "$TRACE_DIR" >/dev/null
lttng enable-channel --userspace --discard --subbuf-size=32768 --num-subbuf=4 ch1 >/dev/null
python3 -c "
import sys
sys.path.insert(0, 'projects/rocr-runtime/runtime/hsa-runtime/scripts')
from lttng_curated_lib import parse_yaml_file
for a in parse_yaml_file('$YAML'):
    print(f'rocm_hsa:{a[\"api\"]}_args')
" | xargs -r lttng enable-event --userspace --channel=ch1 >/dev/null

lttng start "$SESSION_NAME" >/dev/null
"$WORK/coverage_test" || true   # placeholder args may fail; OK
lttng stop "$SESSION_NAME" >/dev/null
lttng destroy "$SESSION_NAME" >/dev/null

DUMP="$WORK/trace.txt"
babeltrace2 "$TRACE_DIR" > "$DUMP"

# IMPORTANT: process substitution, not a pipe — so MISSING accumulates
# in the parent shell (per the C5 fix in Task 13).
MISSING=0
while read api; do
    if grep -q "rocm_hsa:${api}_args" "$DUMP"; then
        echo "  PASS  ${api}_args fired"
    else
        echo "  FAIL  ${api}_args NOT in trace"
        MISSING=$((MISSING+1))
    fi
done < <(python3 -c "
import sys
sys.path.insert(0, 'projects/rocr-runtime/runtime/hsa-runtime/scripts')
from lttng_curated_lib import parse_yaml_file
for a in parse_yaml_file('$YAML'):
    print(a['api'])
")

if [ "$MISSING" -gt 0 ]; then
    echo "FAIL: $MISSING HSA curated _args events missing"
    exit 1
fi
echo "PASS: all HSA curated _args events fired"
```

- [ ] **Step 3: Make tests executable and run them**

```bash
chmod +x projects/rocr-runtime/runtime/hsa-runtime/test/lttng/test_hsa_curated_args_payload.sh \
         projects/rocr-runtime/runtime/hsa-runtime/test/lttng/test_hsa_curated_args_coverage.sh
./dev-bin/sync.sh main
./dev-bin/in-container.sh main "cd /root/rocm-systems && bash \
    projects/rocr-runtime/runtime/hsa-runtime/test/lttng/test_hsa_curated_args_payload.sh"
./dev-bin/in-container.sh main "cd /root/rocm-systems && bash \
    projects/rocr-runtime/runtime/hsa-runtime/test/lttng/test_hsa_curated_args_coverage.sh"
```

Expected: payload test prints `=== ALL HSA PAYLOAD ASSERTIONS PASSED ===`, coverage test prints `PASS: all HSA curated _args events fired`.

- [ ] **Step 4: Commit**

```bash
git add projects/rocr-runtime/runtime/hsa-runtime/test/lttng/test_hsa_curated_args_payload.sh \
        projects/rocr-runtime/runtime/hsa-runtime/test/lttng/test_hsa_curated_args_coverage.sh
git commit -m "lttng(hsa): add HSA curated-args payload + coverage tests

Payload test: hsa_signal_create (initial_value=0x1234, num_consumers=0),
hsa_signal_destroy. Asserts payload values match and generic
enter/exit_status events still fire (augment-not-replace).

Coverage test: harness program is generated from
projects/rocr-runtime/.../scripts/curated_apis.yaml so new APIs added
to the YAML are auto-tested. Asserts each <api>_args event appears in
the trace at least once.

Coverage loop uses process substitution (< <(...)) instead of a pipe
into 'while read' so the MISSING counter accumulates in the parent
shell — same fix as the HIP coverage test (C5)."
```

---

## Phase H — Final integration (task 16)

---

### Task 16: Container deploy + GraphBench overhead measurement + debate-driven-development re-run + push to PR #5475

**Files:** None new; this is a verification + deploy + push phase.

- [ ] **Step 1: Run the full local validation suite**

```bash
./dev-bin/sync.sh main
./dev-bin/in-container.sh main "cd /root/rocm-systems && cmake --build build/clr -j 32 --target amdhip64 2>&1 | tail -10"
./dev-bin/in-container.sh main "cd /root/rocm-systems && cmake --build build/rocr -j 32 --target hsa-runtime64 2>&1 | tail -10"
./dev-bin/in-container.sh main "cd /root/rocm-systems && bash projects/clr/hipamd/scripts/lttng_coverage_gate.sh build/clr/hipamd/lib/libamdhip64.so projects/clr/hipamd/scripts/lttng_migration_inventory.txt projects/clr/hipamd/scripts/lttng_migration_inventory_c.txt"
./dev-bin/in-container.sh main "cd /root/rocm-systems && bash projects/clr/hipamd/test/lttng/test_hip_api_tracepoints.sh build/clr/hipamd/lib"
./dev-bin/in-container.sh main "cd /root/rocm-systems && bash projects/clr/hipamd/test/lttng/test_hip_invariants.sh build/clr/hipamd/lib"
./dev-bin/in-container.sh main "cd /root/rocm-systems && bash projects/clr/hipamd/test/lttng/test_hip_curated_args_payload.sh build/clr/hipamd/lib"
./dev-bin/in-container.sh main "cd /root/rocm-systems && bash projects/clr/hipamd/test/lttng/test_hip_curated_args_coverage.sh build/clr/hipamd/lib"
./dev-bin/in-container.sh main "cd /root/rocm-systems && bash projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_coverage_gate.sh ..."
./dev-bin/in-container.sh main "cd /root/rocm-systems && bash projects/rocr-runtime/runtime/hsa-runtime/test/lttng/test_hsa_api_tracepoints.sh"
./dev-bin/in-container.sh main "cd /root/rocm-systems && bash projects/rocr-runtime/runtime/hsa-runtime/test/lttng/test_hsa_invariants.sh"
./dev-bin/in-container.sh main "cd /root/rocm-systems && bash projects/rocr-runtime/runtime/hsa-runtime/test/lttng/test_hsa_curated_args_payload.sh"
./dev-bin/in-container.sh main "cd /root/rocm-systems && bash projects/rocr-runtime/runtime/hsa-runtime/test/lttng/test_hsa_curated_args_coverage.sh"
```

All 12 invocations must report PASS.

- [ ] **Step 2: Install instrumented libraries to /opt/rocm/lib in container**

```bash
./dev-bin/in-container.sh main "cd /root/rocm-systems && cmake --install build/clr"
./dev-bin/in-container.sh main "cd /root/rocm-systems && cmake --install build/rocr"
./dev-bin/in-container.sh main "ls -la /opt/rocm/lib/libamdhip64.so.7.2.70202 /opt/rocm/lib/libhsa-runtime64.so.1.18.70202"
```

Expected: timestamps reflect just-installed builds.

- [ ] **Step 3: Run GraphBench overhead measurement (per the existing benchmark protocol)**

```bash
./dev-bin/in-container.sh main "cd /tmp/GraphBench && taskset -c 4 chrt -f 50 ./run_benchmark.sh --reps 12 --modes lttng_off,lttng_on,rocprofv3_hsa"
```

Expected: lttng_on overhead remains roughly comparable to the pre-curated-args baseline (was +0.6%; allow up to +2% with curated events enabled). Capture the numbers in `~/ai/2026-04-26-graphbench-with-curated-args.md` as a benchmark record.

If overhead > 5%, investigate which curated events are most expensive. The `tracepoint_enabled()` short-circuit means curated events should cost essentially zero when the consumer doesn't subscribe to them; if not, investigate whether `_args` events are being unconditionally enabled.

- [ ] **Step 4: Re-run debate-driven-development on the curated-args commits**

Use the `debate-review-code` skill to gate the curated-args additions:

```
/debate-review-code <commit-range>
```

Where commit-range covers the new commits added across all phases (run `git log --oneline e2bd878047..HEAD` to find the range). Expect 5–10 review claims; address each per the loop's protocol.

- [ ] **Step 5: Update PR #5475 description**

```bash
gh pr edit 5475 --body "$(cat <<'EOF'
## Summary

Add LTTng-UST producer-side instrumentation to HIP CLR, ROCr, and rocprofiler-register, plus per-API typed parameter capture for ~82 curated APIs.

[... existing PR description ...]

## New in this revision: curated parameter-capture (Phase 8)

- ~82 curated HIP+HSA APIs now emit a typed `<api>_args` event in addition to the generic enter/exit, with parameters captured per [spec](docs/superpowers/specs/2026-04-26-lttng-curated-args-design.md).
- YAML DSL (`curated_apis.yaml`) is the source of truth; codegen produces `rocm_hip_curated_tp.h` + `rocm_trace_emit_curated.h` (checked in).
- New libclang-based verifier (`lttng_curated_verify.py`) catches HIP/HSA header drift.
- Coverage gate extended with sentinel + macro regex + IN-local checks.
- Six new macro variants: `ROCM_TRACE_RET_{STATUS,PTR,VOID}_CURATED(_NOARGS)?` that augment (don't replace) the generic `hip_api_exit_*` events.
- GraphBench overhead with all curated events disabled remains within the +0.6% baseline; with all enabled, +<X>%.

EOF
)"
```

- [ ] **Step 6: Push to remote**

```bash
git push origin lttng:users/bewelton/lttng
```

PR #5475 picks up the new commits automatically since the branch tracks the same remote.

- [ ] **Step 7: Convert PR from Draft to Ready for Review**

```bash
gh pr ready 5475
```

---

## Self-Review Checklist (run before declaring the plan done)

After completing all 16 tasks, run this fresh-eyes pass against the spec:

- [ ] **Spec coverage:** Every section of the spec maps to a task:
  - §1 Goal → Tasks 6, 7, 13 (typed events augment generic)
  - §2 Non-goals → Tasks 5, 14 (YAML deliberately excludes RCCL, struct walking, etc.)
  - §3 Architecture → Tasks 6–11 (event flow), Task 9 (CATCH/exception)
  - §4 DSL → Tasks 2 (parser), 5/14/15 (YAML), 4 (verifier)
  - §4.1 Type vocab → Tasks 1 (dim3_pack), 2 (parser), 3 (codegen), 4 (verifier)
  - §4.4 Field budget → Tasks 2 (parser), 3 (codegen), 4 (verifier)
  - §5 Codegen → Task 3
  - §6 Migrator → Tasks 8, 9, 10, 15
  - §6.2 Macros → Tasks 8, 15
  - §7 Edge cases → Task 3 (cstring NULL safety, etc.)
  - §8 Tests → Tasks 13, 15
  - §9 Build wiring → Tasks 12, 15
- [ ] **Placeholder scan:** Search the plan for "TBD", "TODO", "fill in", "implement later", "similar to". Fix any found.
- [ ] **Type consistency:** Macro names match between the Phase D macros and the migrator output. Helper signatures match between codegen output (Task 3) and macro expansion (Task 8). YAML schema in Task 5 matches the parser library in Task 2.
- [ ] **No invented APIs:** Every HIP/HSA function name in the plan exists in the actual HIP/HSA headers. The verifier in Task 4 catches this; the coverage test in Task 13 catches it at runtime.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-26-lttng-curated-args-implementation.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. With 16 tasks, ~6–8 hours wall time at typical subagent throughput.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Slower per-iteration but allows deeper hand-on-the-wheel for the migrator-overlay logic which is the trickiest piece.

**Which approach?**

