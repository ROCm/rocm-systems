# gfx950 FP16 Issues and Fixes Summary

## Context

This summary captures the key issues encountered while bringing up RCCL on MI350 (`gfx950`) and the corresponding fixes. It is focused on root cause and remediation only.

## Issue 1: Device Link Failure With Debug Info Enabled

- **Symptom**: Debug builds failed during split-specialized linking with relocation errors in debug sections (for example in `.debug_frame` / `.debug_line`) when producing the intermediate combined device object for `gfx950`.
- **Impact**: Prevented use of line-level debug information in the new build flow.
- **Fix**:
  - Kept device debug information intact.
  - Removed the fragile intermediate relocatable link stage for per-arch device objects.
  - Linked the final per-arch shared code object directly from the per-kernel device object list via a response file (`.rsp`).
- **Resulting behavior**: Debug-enabled builds no longer depend on the problematic intermediate combined relocatable object path.

## Issue 2: FP16 False Mismatches (Prominent on gfx950)

- **Symptom**: Several FP16 unit tests (especially `min` reductions, including graph variants) reported mismatches even when transport/runtime behavior was otherwise consistent.
- **Observed platform difference**: The problematic behavior reproduced on `gfx950` while corresponding coverage on `gfx942` passed.
- **Root cause**:
  - Host-side unit-test helper logic in `test/common/PtrUnion.cpp` used `__half` host intrinsics (`__float2half`, `__half2float`) in data prep/reduction/compare code paths.
  - On this build/target/toolchain path, these host conversions produced incorrect FP16 expectations for some cases, causing false negatives in validation.
- **Fix**:
  - Replaced host-side FP16 helper operations in `PtrUnion` with explicit bit-accurate FP16 conversion helpers.
  - Updated FP16-related paths (`Set`, `Get`, `Scale`, `Reduce`, `DivideByInt`, string/debug formatting, equality checks) to use deterministic bit conversion.
  - Kept NaN handling deterministic by requiring bitwise equality when both values decode as NaN.
- **Resulting behavior**: FP16 expected-value generation and comparison became deterministic and stable across the affected test paths on `gfx950`.

## Files Changed for These Fixes

- `cmake/SplitSpecializedCompile.cmake` (device link flow update)
- `test/common/PtrUnion.cpp` (host-side FP16 deterministic conversion and comparison logic)
