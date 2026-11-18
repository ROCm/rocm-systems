# Vectorised exact gfx12 FP8 WMMA

Measurement and correctness notes for `src/wmma/x86_64/wmma_gfx12_fp8_exact.h`.

## What this changes

Rocjitsu's default `compatibility` semantics resolve every gfx12 WMMA to the empirical
bit-exact model, so `v_wmma_f32_16x16x16_fp8_fp8` ran the scalar DOT8 model once per output
element per stage: 256 outputs, two stages, sixteen products. This adds AVX2 and AVX-512
kernels for E4M3 x E4M3 only. BF8 introduces infinities and invalid products and stays on the
scalar model, as do all other matrix forms.

## Why it is the shape it is

The vector lanes are output **columns** of one row. Every step of the model is then per-lane:
A contributes a broadcast, B a contiguous load, and C and D contiguous accesses. There is no
horizontal reduction, shuffle or gather in the kernel. Vectorising across the eight products
instead would make every group sum and grid selection a horizontal operation.

Three properties of the model were established before the kernels were written, each removing
work that would otherwise force a scalar fallback:

| Property | Consequence |
|---|---|
| `max(-126, c_leading)` equals `max(exponent_field, 1) - 127` for every FP32 encoding | no bit scan; AVX2 has no leading-zero count |
| a zero factor given alignment exponent -137 puts an all-zero DOT8 on product grid -153, below the accumulator floor of -152 | the empty-exponent sentinel disappears from grid selection |
| a variable arithmetic right shift saturates its count at 31 and fills with the sign bit | that is exactly the floor a term below the grid needs, so alignment carries no clamp and no mask |

The first was checked exhaustively over the FP32 encoding classes. For the second, the valid
window is any value at or below -137; the weaker "never outranks a nonzero product" bound alone
is not sufficient, and a kernel using it is caught by the tests below.

Every aligned term and the running total stay inside `int32`: a product needs at most 26 bits on
its group grid, four of them 28, two groups plus the accumulator under 30. FP32 cannot hold this;
an otherwise identical kernel that sums the floored terms in FP32 differs from the model on about
3.7% of outputs.

The result is rounded by converting the grid-unit count to FP32 and scaling by `2^grid`. The
scale is exact, so the conversion is the model's single round-to-nearest-even. Lanes whose result
is subnormal or overflows, and grids below the FP32 normal floor, take the integer pack instead.
That same test makes the kernel independent of MXCSR flush-to-zero: a flushed result reads back
with a zero exponent field and is repaired.

## Correctness

`rj_arith.WmmaF32FP8FP8W32Gfx12.ExactSimdAgreement` and its Wave64 twin require every supported
CPU level to reproduce the baseline integer model bit for bit.

The directed tiles in that test are not decoration. Random factors and accumulators never drive a
result into the FP32 subnormal range or to overflow, and never leave every product zero, so they
never reach the repair path at all: over a 4000-tile random check, it was taken 0 times out of
84 million outputs. The directed tiles take it 1,160,192 times.

Mutants, against random tiles only and against the directed tiles:

| mutant | random tiles | directed tiles |
|---|---|---|
| zero-factor sentinel -137 changed to -21 | not caught | caught |
| scalar repair path removed | not caught | caught |
| negative-accumulator vanish rule dropped | caught | caught |
| accumulator grid off by one | caught | caught |
| one grid shared by both groups of four | caught | not caught |

Neither layer alone is sufficient, which is why the test interleaves directed and random tiles.

Separately, output hashes over 3000 tiles per form through the public entry points are identical
to the pre-change library for all six FP8/BF8 gfx12 WMMA forms, Wave32 and Wave64, with half the
tiles drawn from arbitrary bit patterns. `wmma_gfx12_fp8_exact_bench.cpp` reproduces that.

## Measurements

shark-mkm-1, Ryzen Threadripper PRO 9985WX, pinned, machine idle.

Through the public entry point, 16x16x16 tile, median ns per call:

| form | before | after |
|---|---|---|
| `fp8_fp8_w32` | 15588 | 288 |
| `fp8_fp8_w64` | 14770 | 294 |
| `bf8_*` (unchanged, scalar) | 12708-14208 | same |

Against the loose kernel, which is AVX2 and not bit-exact, on one core in one process:

| kernel | ns/tile | vs exact | vs loose |
|---|---|---|---|
| library exact, before | 19600 | 1.0x | 117x slower |
| scalar, int32 carrier | 4100-5020 | 3.9-4.8x | 25-30x slower |
| AVX2 | 531 | 36.9x | 3.2x slower |
| AVX-512 | 280 | 70.0x | 1.7x slower |
| library loose (not bit-exact) | 167 | 117x | 1.0x |

The scalar ladder that led to the vector form, each step bit-exact, measured as a ratio against
the library in the same run:

| step | ratio |
|---|---|
| verbatim model | 1.01x |
| DOT8 as signed integers and arithmetic shifts | 1.28x |
| decode each factor once per tile, NaN as row/column masks, branch-free alignment | 2.85x |
| FP pack replacing the integer pack | 3.18x |
| DOT8 inlined into the tile loop | 3.50x |
| int32 carrier | 3.90x |

Latency levers did nothing at any point: stage-major ordering, splitting the stage chain and
unrolling four columns all measured at or below the variant they were applied to. After the
structural rewrite the kernel is throughput-bound near 4.5 instructions per cycle.

Ablations on the scalar kernel: the per-factor NaN test was 27% of the verbatim baseline, the
accumulator side 18% of the rewritten kernel, and the pack 9%.
