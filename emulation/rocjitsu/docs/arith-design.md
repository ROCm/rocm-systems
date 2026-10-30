# Rocjitsu Arithmetic Library design

This document describes the Rocjitsu Arithmetic Library (`rj-arith`), located in
[`lib/arith/`](../lib/arith/). It implements GPU arithmetic on a CPU. Its interface follows GPU
instructions and register layouts; its implementation uses portable C++ and
CPU-specific SIMD arithmetic.

## Scope and contract

The library is a collection of C entry points mapping one-to-one to GPU instructions.
Each aims to reproduce the instruction's arithmetic, synchronously on the calling
CPU thread. Operation is stateless: operands and controls arrive as arguments,
and results are written through destination pointers supplied by the caller.
The user explicitly chooses between loose and bit-exact semantics through the
[semantics control](#one-semantics-control).

Entry points are reentrant. Concurrent calls may share read-only operands, but
execution is not guaranteed to be correct if one call writes storage that another
concurrent call reads or writes.

Calls use no threading or synchronization primitives, allocate no dynamic memory,
and make no system calls. They never explicitly change the CPU FP environment,
including rounding modes, denormal flushing, or exception controls. Ordinary
arithmetic may raise sticky exception flags.

The library assumes the CPU's default floating-point environment: nearest-even
rounding, denormal flushing disabled, and exceptions masked. Results are not
guaranteed to be correct under a nondefault environment.

The caller explicitly selects the CPU feature level through each entry point's
`flags` argument, enabling internal SIMD paths such as AVX-512. Instruction calls
never detect CPU features. Only the convenience function
`rj_arith_init_cpu_flags()` performs detection; callers can cache its result.

## Public API

### C interface and instruction names

The public interface is usable from C23 and C++17, through
[`rj_arith/rj_arith.h`](../lib/arith/include/rj_arith/rj_arith.h). Names follow
`rj_arith_<mnemonic>[_w32|_w64][_<architecture>]`: wave width precedes architecture.
Names retain ISA mnemonics. Where a corresponding Clang AMDGPU builtin exists,
its architecture and wave-width suffixes guide ours; suffixes do not encode
implementation order. For example, gfx11 WMMA is unsuffixed, while the gfx12
forms use `_w32_gfx12` or `_w64_gfx12`, matching those builtins.
Vector entry points retain explicit wave widths even when a lane-wise builtin
has none. Where no builtin distinguishes different instruction contracts,
architecture suffixes disambiguate them. Scalar instructions omit wave width.

The C interface constrains the API to a narrow, hermetic surface rather than
making Rocjitsu a C project. Typed `constexpr` constants work in both languages.
C++17 header checks always run; C23 checks run only when a compiler probe succeeds.
Skipping C checks on older toolchains is intentional, including current CI.
The project itself does not require C23.

### A representative instruction

`rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12` computes `D = A × B + C`
with 16×16 FP16 factors and FP32 accumulation:

```c
int rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12(
    uint64_t flags,
    uint64_t instruction_flags,
    uint32_t *const *d,
    const uint32_t *const *a,
    const uint32_t *const *b,
    const uint32_t *const *c);
```

A and B each occupy four VGPRs; C and D each occupy eight, with 32 lane words
per VGPR. A/B words pack two FP16 elements; C/D words contain FP32 bit patterns.
The matrices are distributed across the wave, not replicated per lane.
WMMA ignores EXEC, so the API has no execution-mask parameter.
The [README example](../lib/arith/README.md#api-example) constructs these operands and decodes the result.

Sparse SWMMAC instead reads the initial accumulator from `d`; its fourth pointer
operand is `index`, not a separate `c`. See the
[sparse instruction contracts](../lib/arith/include/rj_arith/detail/rj_arith_wmma.h)
for operand layouts and metadata selectors.

### Operands and register storage

Scalar inputs are values; scalar outputs are pointers. Vector operands are
arrays of VGPR pointers, with `uint32_t` words carrying raw bits in ISA layout.
Each VGPR contains contiguous lane words in lane order (VGPR-major layout).
Input words are const-qualified. VGPRs need not be adjacent in CPU memory,
so callers can use their own register storage without copying operand data.
Raw integer words also avoid requiring FP16 or other nonstandard floating types
in the public ABI. Explicit operands follow assembly order; implicit scalar
inputs follow the explicit operands.

### Aliasing and execution masks

Whole-VGPR aliases are supported: reads observe the original sources before
writes change destinations. Distinct VGPRs must not partially overlap.
Instruction-specific contracts define the write order when destination pointers
alias one another or scalar outputs overlap vector storage.

Instructions affected by EXEC take `uint32_t exec_mask` for Wave32 or
`uint64_t exec_mask` for Wave64 and preserve inactive vector destinations.
Instructions that ignore EXEC omit the parameter. Scalar comparison and carry
masks follow their own inactive-bit rules.

### Controls and errors

`flags` combines CPU selection, semantics, and guest FP state;
`instruction_flags` carries decoded modifiers such as sign negation (NEG),
absolute value (ABS), clamping (CLAMP), and cross-lane source selection (DPP).
CPU levels and semantics are enums within `flags`, not independent feature bits.
The [common header](../lib/arith/include/rj_arith/detail/rj_arith_common.h) defines these fields;
[family headers](../lib/arith/include/rj_arith/detail/) specify each instruction's accepted controls.

Calls return `RJ_ARITH_SUCCESS` or an error distinguishing invalid controls,
unsupported numerical semantics, unsupported FP status, and unsupported register
effects. Errors leave all outputs unchanged, including FP status storage.
Reserved semantics are rejected even for empty EXEC. Valid operand storage is
an unchecked caller precondition unless the instruction explicitly permits null.

## Numerical semantics and FP state

### One semantics control

Modes 0–3 form an ordered choice: each higher mode requires more faithful behavior.
The constants include their bit position in `flags` and can be OR-ed with CPU
and guest FP controls.

| Mode | Constant | Requirement |
|---|---|---|
| 0 | `RJ_ARITH_SEMANTICS_LOOSE` | Loose arithmetic; no FP status reporting |
| 1 | `RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE` | Prefer empirical bit-exact arithmetic, allow loose fallback; no FP status reporting |
| 2 | `RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS` | Require empirical bit-exact arithmetic; no FP status reporting |
| 3 | `RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL` | Require empirical bit-exact arithmetic and FP status reporting |
| 4–6 | Reserved | Rejected |
| 7 | `RJ_ARITH_SEMANTICS_COMPATIBILITY` | Instruction-specific policy, outside the ordering |

Loose semantics permit numerical approximations, not wrong operands, missing
operations, or incorrect integer and comparison results. Native floating-point
arithmetic may overflow intermediate values or use a different accumulation order,
including returning infinity where the GPU returns a finite value. Loose paths
need not reorder terms or check operand ranges to prevent these discrepancies.
Modes 2 and 3 reject unsupported behavior rather than silently falling back to
loose arithmetic.

### Compatibility mode

Mode 7 is a temporary, instruction-specific policy outside the ordered modes.
Each entry point selects the arithmetic and FP status policy matching Rocjitsu's
previous behavior. This can retain instruction-specific details beyond the
other modes, such as rounding FP16 results before applying OMOD.
FP32 MUL selects numeric-only exact semantics in every configuration. Its
integration retains Rocjitsu's existing exception classifier for Compatibility;
explicit mode 3 uses the library's qualified reporting domain instead.
Compatibility may retain demonstrated hardware-fidelity improvements over the
previous interpreter; instruction contracts document such exceptions.
Rocjitsu's [`arithmetic.semantics`](configuration.md#cpu-arithmetic-policy)
setting maps directly to these modes and defaults to compatibility. Its string
values are the lowercase constant names without `RJ_ARITH_SEMANTICS_`.
The migrated families are dense and sparse WMMA, fused floating-point arithmetic, basic
floating-point arithmetic and selection and division. Other handlers retain their
existing implementations.

### Guest FP controls

Guest rounding, denormal handling, and NaN policies are emulated explicitly,
without changing the host environment. Guest controls are instruction-specific:
a zero flag field selects the instruction's default policy, which can include
mandatory denormal flushing. Loose paths may ignore guest rounding where their
contract permits it. Instruction-local CPU rounding controls are usable because
they do not change the host FP environment.

### FP status and other instruction results

FP status outputs are trailing register pointers, such as `excp_flag_user` or
`trapsts`. Modes 0–2 leave them unused; mode 3 requires valid storage without a
null check. A pointer does not select reporting: null is allowed in modes 0–2,
and non-null storage is left unchanged. Instructions without FP status outputs
have no additional pointer requirement in mode 3.

Reporting accumulates participating-lane exception causes with bitwise OR,
preserving existing register bits. Each applicable register has one trailing
output parameter of its native width. Guest status is computed from the GPU
model, not by reading the host CPU's exception flags.

Comparison masks, SCC, EXEC, and explicit register writes remain required results
in every mode. In particular, an instruction that explicitly writes TRAPSTS is
not performing optional FP status reporting just because it names that register.

### What bit-exact support means

A supported bit-exact implementation covers all input values under its supported
guest policy, including signed zeros, subnormals, infinities, and NaNs.
The target is the GPU's behavior, including its approximations and rounding steps,
not necessarily the correctly rounded mathematical result. Missing required support returns an error.

## Implementation strategy

Conversion helpers belong to `lib/arith/`, independently of `util/`: GPU rounding
and special-value behavior can differ by architecture, and CPU SIMD paths must
implement the same instruction contracts. This small duplication keeps those
hardware-specific policies under the arithmetic library's ownership.


Exact arithmetic follows validated Rocjitsu models or simple ISA-defined behavior.
Sources and evidence stay beside the code and tests; optimizing a model does not
establish that it matches hardware.

SIMD and scalar paths implement the same contract. Scalar fallbacks handle values
outside a fast path without unnecessarily discarding SIMD work for other lanes.
Reporting and non-reporting modes share arithmetic, with FP status calculations
performed only when requested. EXEC masks generally control stores, not computation.

## Testing

### Explicit tests with shared family helpers

Every entry point has a named test that calls it explicitly. Related operations,
element types, and wave sizes share source files and helpers where useful.
Tests sample essential differences in modifiers, EXEC, aliases, CPU paths, and
semantics rather than multiplying every combination into an exhaustive suite.

### Deterministic inputs and trustworthy references

Tests use default-seeded `std::minstd_rand` and its high bits, with explicit
special values and boundary cases. The engine and default seed are fully specified,
so input sequences are reproducible across standard-library implementations, not
just repeated runs. Input generation avoids implementation-dependent choices such
as `std::default_random_engine` and the algorithms used by `std::uniform_int_distribution`.

Loose-semantics validation uses fuzzy comparisons with empirically calibrated
`atol` and `rtol` tolerances against independent references:
`abs(actual - expected) <= atol + rtol * abs(expected)`.
These bounds apply to the tested input domain, not to arbitrary floating-point
inputs. Within that domain, NaNs match NaNs and infinities must match in sign.
Cases requiring GPU-specific cancellation or overflow behavior belong in exact
tests; loose tests must not impose those guarantees on native arithmetic. Masks
and preserved storage are checked exactly. Bounds live beside the instruction
tests or family helpers.

Exact-semantics validation uses golden results or computed references whose
bit-exact equivalence is established, rather than assumed from mathematical
plausibility. Fixtures record their model or hardware provenance. Production
output is never its own correctness oracle; scalar/SIMD agreement alone cannot
establish that both match the GPU.

### Matrix tests

Each matrix entry point has one arithmetic test with full logical operands,
rounding-sensitive values, and a small set of NaN and infinity inputs where representable.
The same data serves semantics, CPU-path, and aliasing checks. Floating goldens
use hexadecimal literals; instruction layouts and expected results remain easy to find.

### State, errors, and validation

Tests check preserved storage, inactive lanes, aliases, status accumulation, and
unchanged outputs on errors, alongside numerical results. Header checks and
benchmark smoke tests complement these checks. GCC, Clang, portable builds, and
sanitizers provide broader validation; ordinary tests need no GPU.
Each `*_test.cpp` builds as a separate executable, with individual GTest cases
exposed to CTest. Public headers are compiled independently as C23 and C++17.

## Benchmarking (planned)

No benchmark executable is built at this stage. The following describes its intended interface.

### Workloads and reported results

The benchmark covers WMMA, DOT, FP, and math entry points across supported CPU
levels and semantics. Floating workloads include finite inputs and a single-NaN
variant to expose fallbacks triggered by a nonfinite input. Workloads use full
EXEC and default modifiers and guest FP controls. CSV contains successful
measurements only; unsupported modes are not timed as if they performed arithmetic.
Speedups compare the same workload and semantics against the scalar path.

Operations with an implicit accumulator restore it before each call to prevent
repeated execution from changing the input distribution. That copy is included
in the timing and identified in the `Variant` column. CSV goes to stdout;
progress and errors go to stderr.

### Measurement method

Each workload runs a fixed number of repetitions. The outer loop advances the
repetition and the inner loop visits every workload, spacing out samples of the
same workload. Its call count is doubled until the first repetition reaches the
minimum duration, then reused for subsequent repetitions.

Results report median nanoseconds per wave. Defaults are five repetitions and
one millisecond, controlled by `--repetitions` and `--min-ms`. CPU affinity and
other external controls address timing noise; the benchmark does not adapt its
repetition count to measured noise.
