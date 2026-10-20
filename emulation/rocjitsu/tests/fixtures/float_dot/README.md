# DOT2 and WMMA hardware fixtures

These raw results were captured on Radeon Pro W7900 (`gfx1100`) and Radeon AI
Pro R9700 (`gfx1201`) using TheRock `10.2.0a20260916`. The decoded regression
tests consume the captured bits directly and require no GPU.

| Header | Contents |
| --- | --- |
| `rdna_f16_samples.h` | Selected RDNA3 FP16 high-half WMMA output words. |
| `rdna_sparse_float_samples.h` | Selected RDNA4 sparse FP16 matrix output words. |
| `rdna_f8_samples.h` | Selected RDNA4 FP8/BF8 WMMA output words. |
| `rdna_integer_samples.h` | Selected RDNA integer WMMA/SWMMAC output words. |
| `gfx1100_cases.h` | GFX11 FP32-output DOT2 cases and a WMMA matrix. |
| `gfx1201_cases.h` | GFX12 FP32-output DOT2 cases and WMMA matrices. |
| `packed_operand_cases.h` | 440 packed WMMA instruction/operand captures for modifiers, selectors and inline constants. |
| `packed_wmma_cases.h` | Seven F16/BF16 matrix inputs with packed outputs for both architectures and wave sizes. |

DOT2 records store packed A/B words, FP32 C and the expected FP32 result.
WMMA inputs use row-major matrices. Packed matrix expectations are indexed by
`[gfx12][wave64][bf16][fixture][element]`, with row-major elements. Operand
fixtures store raw instruction words, uniform A/B/C words and the two
architecture-specific expected destination words. Their initial D value is
`0x56781234`, allowing the tests to check preservation of the unused half.

Coverage includes signed zeros, NaN payloads and precedence, infinities,
alignment and rounding boundaries, subnormals, cancellation, both packed
half selections, wave32/wave64 and overlapping input/output registers. Separate
regressions retain intermediate F16 overflow-mode and cancellation witnesses.
Comparisons include every output bit, including NaNs and preserved halves.

The GFX11 fixtures also test RDNA3.5 dispatch, without claiming a physical
RDNA3.5 measurement. The captures qualify dense F16/BF16 DOT2/WMMA arithmetic
on these two cards; they do not establish behavior for other matrix formats.

## RDNA matrix regressions

The `rdna_*_test.cpp` tests retain a few raw results captured on W7900
(`gfx1100`) and R9700 (`gfx1201`). They run without a GPU. The arithmetic models
are inferred from these observations, not specifications of hardware internals.

Integer cases distinguish intermediate saturation, sparse index selection, and
operand overlap. Inputs use the original capture seed and xorshift sequence.

FP8/BF8 cases distinguish native-format arithmetic and NaN precedence.

Sparse floating-point cases exercise index keys, source signs, and packed outputs.

FP16/BF16 dense cases exercise packed destination halves and accumulator handling.

Tests sample destination words and check that other registers remain unchanged.
This deliberately small corpus is regression coverage, not exhaustive validation.

## Inline DOT captures

`valu_fp_mode_test.cpp` and `gfx12_dot_test.cpp` also hold DOT lanes copied
from five separate probe batches. The VALU and DOT probes compiled one HIP
kernel per instruction and modifier form, set MODE with `s_setreg` around the
instruction, and wrote the raw destination bits for 8,192 fixed-seed lanes
(special-value cross products, random values and, for integer DOT, operands
near the accumulator limits). The same binaries were replayed under rocjitsu.
The WMMA and witness probes are described in their rows below. The probe
sources are not part of this repository.

| Card | Host software |
| --- | --- |
| Radeon RX 9070 XT (`gfx1201`) | ROCm 7.1.0; in-kernel `amdgpu` on Linux 7.0 for the DOT and WMMA probes |
| Radeon PRO W7900 (`gfx1100`) | ROCm 7.2.0; `amdgpu` DKMS 6.16.13 on Linux 6.17 for the DOT probes |

The VALU probe used the same ROCm installs; its kernel and driver versions were
not recorded. On `gfx1100` the VALU probe cleared MODE.IEEE and the DOT probes
kept it set.

| Batch | Contents | MODEs |
| --- | --- | --- |
| VALU probe | True16 `V_DOT2_F16_F16` and `V_DOT2_BF16_BF16`, plain, ABS/NEG on C and high halves | Plain F16: 84 (every rounding and denormal combination, split rounding, FP16_OVFL); F16 modifier forms: 7; BF16: `0xf0` |
| First DOT probe | VOP3P `V_DOT2_F32_F16/BF16`, `V_DOT4_F32_{FP8,BF8}_{FP8,BF8}` (`gfx1201` only), integer DOT4/DOT8, `V_DOT2ACC_F32_F16` (`gfx1100` only), each with one modifier per form | 12 for plain forms; modifier forms `0xf0`, `0x00`, `0x30`, `0xc0`, `0xff`, so their rounding cases change both fields at once; integer forms `0xf0`, `0x00` |
| DOT follow-up probe | ABS+NEG on C for every floating DOT form, including true16 BF16; integer OP_SEL combined with NEG and CLAMP near the limits | Plain forms: 15, including one denormal input or output bit at a time (`0x70`, `0xb0`, `0xd0`, `0xe0`), one rounding field at a time (`0xf3`, `0xfc`) and FP16_OVFL; modifier forms: 9 |
| WMMA C-floor probe (`gfx1201`, wave32) | `V_WMMA_F32_16X16X16_BF16` and `V_WMMA_BF16_16X16X16_BF16` with only k=0 of A nonzero, small BF16 products and subnormal C; 16,384 cases per opcode | kernel default |
| Witness probe | Single lanes chosen so that one rule change alters the result: the negative-C floor unit at both floors, FP8/BF8 one-grid alignment, and a zero-product accumulator | `0xf0`, `0x30`, `0xc0`, `0x00`, `0xf3`, `0xfc` |

### Measured and extrapolated behaviour

Measured on both cards: true16 DOT2, VOP3P DOT2 F32 including NEG_HI on C as
ABS, and integer DOT4/DOT8 half selection with NEG and CLAMP. Measured on
`gfx1201` only: FP8/BF8 DOT4 and the WMMA C-floor cases. RDNA3.5 reuses the
RDNA3 rules. RDNA2, CDNA1–4 and CDNA5 share the generated DOT bodies (NEG_HI on
C as ABS, integer half selection) without a capture.

### The negative-C floor unit

On `gfx1201`, a negative C whose bits all fall below the alignment grid still
subtracts one unit. `gfx12_dot_terms` floors C's alignment exponent at 2^-14
for packed F16 and 2^-126 otherwise, and the unit applies at or below that
floor, so the two output formats reach it from different directions:

- **At 2^-126**, only BF16 products are small enough. `gfx1201` applies the
  unit for the F32 and packed-BF16 outputs; `gfx1100` does not.
- **At 2^-14**, F16 products alone reach it. `gfx1201` applies the unit and
  `gfx1100` does not: `{0x6400 x 0x3c00, 0x3e00 x 0x3c00}` with C=`0x8001`
  gives `0x6401` on gfx1201 and `0x6402` on gfx1100.

| Caller | Floor | Evidence |
| --- | --- | --- |
| VOP3P `V_DOT2_F32_BF16` | 2^-126 | First DOT probe lanes that only this rule matches |
| `V_WMMA_F32_16X16X16_BF16`, `V_WMMA_BF16_16X16X16_BF16` | 2^-126 | WMMA C-floor probe; `Gfx12Dot.WmmaTinyNegativeAccumulatorSubtractsFloorUnit` |
| True16 `V_DOT2_BF16_BF16` | 2^-126 | Witness probe; `Bf16TinyNegativeAccumulator` |
| VOPD `V_DUAL_DOT2ACC_F32_BF16` | 2^-126 | Witness probe: `gfx1201` applies the unit, `gfx1100` matches the gfx11 rule without it. No decoded case; there is no VOPD test harness |
| True16 `V_DOT2_F16_F16`, `V_WMMA_F16_16X16X16_F16` | 2^-14 | Witness probe; `F16TinyNegativeAccumulator`. The WMMA form shares the step but has no lane of its own |

The broad corpora do not reach either boundary: disabling the unit at 2^-14
leaves every captured variant matching, and the lanes above were constructed
for it.
