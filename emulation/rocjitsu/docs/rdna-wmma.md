# Bit-exact RDNA WMMA execution

F32-output F16/BF16 K16 WMMA has integer SIMD implementations for RDNA3,
RDNA3.5 and RDNA4, in wave32 and wave64. SIMD lanes represent independent
outputs. GFX11 retains its eight DOT2 steps, accumulator sign frame and
subnormal flushing; GFX12 retains four DOT4 steps and its different alignment
and underflow rules. The unchanged scalar helpers remain the reference and
fallback. Packed-output WMMA retains its existing implementation.

The portable backend uses `std::experimental::simd` through `util::stdx`.
It is enabled for native widths of eight or sixteen 32-bit lanes. All arithmetic,
including special values, is vectorized with integer operations. Narrower native
SIMD widths retain scalar execution. The arithmetic primitives are also tested
at four lanes; this does not enable the production fast path at that width.
This follows the codebase's existing
`util::stdx` and `RJ_FORCE_SCALAR` conventions without a new backend selector.

The implementation acquires observed register regions, gathers operands with
the architecture's wave layout, and stages every output before writing any
register. Source/destination aliasing, sign modifiers and inline accumulators
retain the scalar behavior. Integer arithmetic does not depend on host rounding
modes or modify floating-point exception flags.

`RJ_FORCE_SCALAR=1` retains the original scalar execution, consistent with the
other SIMD instruction paths. It is read once per linked module at load time.

## Async interaction

Both wave sizes are eligible for optional async execution of these WMMA shapes.
The ordinary full-EXEC, register-footprint, plugin and execution-mode checks
still apply. Helpers execute the same portable SIMD implementation. The
arithmetic backend does not change admission policy: an independent instruction
must be available on the issuer, and unavailable helper capacity falls back to
inline execution.
Use the JSON thread controls documented in [asynchronous MMA](async-instructions.md).

Faster individual instructions can reduce the benefit of offloading them.
Thread allocation is measured with SIMD enabled. The gfx1100 preset selects
32 dispatch threads and no helpers at a budget of 32 or more: this improves
Gluon while the FP16 IREE matmul is essentially unchanged. gfx1201 selects
24 dispatch threads and eight helpers at that budget: Tensile benefits while
IREE and Gluon remain close to their synchronous timings. This keeps the
existing 24 dispatch threads and uses the extra eight slots for helpers.
Compared with 32 dispatch threads and no helpers at the same budget, it favors
Tensile; 32+0 was slightly faster for IREE and matched Gluon, and leaves all
workers available for general CU work. The measured 16+16 split is faster for
Tensile but reduces dispatch capacity below the existing 24-thread allocation.
Both alternatives remain explicit workload-specific options. The shipped
presets retain their existing synchronous entries through 24 threads. gfx1151
keeps its 24-thread ceiling pending application performance measurements.

These choices come from three interleaved repetitions on fixed physical CPUs,
using whole-process time including validation. The RDNA3 FP16 IREE workload
is a 1024-cubed matmul with disassembled wave32 WMMA; unlike the existing FP32
controls, it exercises the optimized arithmetic. Helpers can issue substantial
fractions of WMMA without improving end-to-end time, so issue counts alone do
not justify reserving helper threads. Custom JSON allocations remain available.

## Qualification

`RdnaWmmaSimd` compares raw arithmetic and full decoded register results with the
scalar implementation. It covers F16/BF16, both wave sizes, all sign modifiers,
inline C, D overlapping A/B/C, partial EXEC, arbitrary raw words, nonfinite values
and host rounding/exception state. `Gfx11Dot2`, `Gfx12Dot` and `PackedWmma` retain
the existing hardware-derived arithmetic fixtures. Async tests compare issue
against synchronous execution, including dependent registers and synchronous
fallbacks on RDNA3, RDNA3.5 and RDNA4. These are simulator comparisons and replay
of existing hardware fixtures, not new physical-GPU qualification.
