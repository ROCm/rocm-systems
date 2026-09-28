# PR #12302 Kuhar Review Fixes

## Scope

Address the three review requests on the existing
`users/lialan/rocjitsu-avx512-smfmac` branch. Do not rebase, push, resolve GitHub
threads, or post replies as part of this work.

## Approaches considered

### 1. Narrow conditional fallback and focused decoded tests

Keep the existing integer-dot implementations and make only their widened
clamp branches conditional on working native 64-bit SIMD masks. Keep the
unclamped 32-bit branches available on every supported SIMD build. Extend the
existing SMFMAC decoded suite for the two missing architectural contracts.

This is the selected approach because it directly matches the review requests,
keeps the change local, and avoids changing established SIMD data flow.

### 2. Split clamped and unclamped dot products into separate helpers

Introduce compile-time-specialized helper functions and dispatch between them
from each integer-dot probe. This would make the type boundary explicit, but it
would duplicate control flow across signed, unsigned, and mixed-sign forms and
create a larger review surface than the fix requires.

### 3. Emulate all native 64-bit mask operations lane by lane

Extend the compiler-compatibility layer so the widened clamp kernels also run
on the affected Clang/libstdc++ combination. This would retain more SIMD
coverage, but it broadens a focused SMFMAC PR into another compatibility
implementation and adds risk unrelated to Kuhar's request.

## Implementation design

### Integer-dot fallback

Remove the whole-probe `UTIL_SIMD_BROKEN_NATIVE_64BIT_MASKS` guards from
`ROCJITSU_TRY_SIMD_VOP3P_DOT_INT` and
`ROCJITSU_TRY_SIMD_VOP3P_DOT_INT_MIXED`.

Within `try_execute_vop3p_dot_int_simd` and
`try_execute_vop3p_dot_int_mixed_simd`:

- Compute the effective architectural clamp condition as today.
- On affected builds, return `false` when that condition is true so the
  instruction body executes the scalar implementation.
- Exclude the widened 64-bit clamp code from affected builds at preprocessing
  time, so the broken native-mask expressions are never instantiated.
- Leave the unclamped reduction in 32-bit SIMD lanes unchanged, including its
  architectural wrapping behavior.

This preserves SIMD for unclamped `v_dot4_i32_i8`, `v_dot8_*`, and mixed-sign
forms while retaining exact scalar saturation for clamped operations.

### Fusion regression coverage

Remove the AVX-512 availability and 16-wide-native guards from
`SmfmacSimdExact.Bf16OverflowCancellationRequiresFusion`. Its explicit expected
words validate the scalar `matrix_fma` change even when the default execution
also selects the scalar path. AVX-512 builds will continue comparing forced
scalar and SIMD execution through the existing helper.

### Partial-EXEC SMFMAC coverage

Add a decoded CDNA4 test using a representative F16 16x16x64 SMFMAC form. Seed
all packed A and B elements to `1.0`, legal sparse selectors, and zero
accumulators; each output must therefore be `32.0` after the 32 selected
products. Set EXEC to an alternating partial mask before execution.

Run the same decoded instruction once with forced-scalar execution and once
with default execution. In both runs, assert that every word of the destination
window equals `32.0`, including lanes disabled in EXEC, and that the
architectural EXEC value remains unchanged. The test runs on non-AVX-512 builds
for scalar coverage and exercises the AVX-512 path when available.

## Validation

- Build and link the RocJITsu test executable with the existing GCC 13 and
  AMDClang configurations.
- Run the full `SmfmacSimdExact.*` suite under both compilers.
- Run the relevant integer-dot SIMD correctness tests under AMDClang, where the
  broken-native-64-bit-mask compatibility path is active.
- Run the fusion and partial-EXEC tests in a non-AVX-512 build to confirm they
  are no longer skipped and validate scalar behavior.
- Run `git diff --check` and repository pre-commit hooks over the PR diff.

## Acceptance criteria

- All three Kuhar comments have a corresponding local code or test change.
- Unclamped integer-dot probes remain enabled on affected SIMD builds.
- Clamped integer-dot operations safely fall back to scalar on those builds.
- Partial EXEC does not suppress any SMFMAC destination lane update.
- Scalar BF16 overflow cancellation is tested without requiring AVX-512.
- No rebase, push, GitHub reply, or thread-resolution action occurs.
