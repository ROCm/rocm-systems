# DOT2, WMMA and SWMMAC local observation fixtures

The arithmetic models described here are best-effort inferences based solely on
local input/output observations from the installed W7900 (`gfx1100`) and R9700
(`gfx1201`). The captured bits record what those devices returned for the saved
inputs and settings. Product groups, intermediate steps, alignment grids and
precedence rules describe the simulator model inferred from those observations;
they do not establish how the hardware computes its results. We do not know the
hardware's internal implementation. These models are not vendor specifications
and may be incomplete or wrong for untested inputs, flags, modes or devices.
Matching the finite corpus bit for bit does not guarantee a match elsewhere.

The original GFX11/GFX12 and packed floating-point fixtures were captured on
Radeon Pro W7900 (`gfx1100`) and Radeon AI Pro R9700 (`gfx1201`) using
TheRock `10.2.0a20260916`. The decoded regression
tests consume the captured bits directly and require no GPU. Later per-opcode
fixtures record their SDK version and date in the sections below.

The headers in this directory contain compact raw hardware expectations and the
inputs needed to reconstruct them. The decoded tests run without a GPU; after a
normal host build, run them with:

```sh
ctest --test-dir build --output-on-failure -R 'RdnaWmma|RdnaSwmmac|PackedWmma|Gfx11Dot2|Gfx12Dot'
```

The larger physical sweeps described below are additional qualification results,
not files or commands supplied by this checkout. Their raw input/output pairs,
code objects, hashes and replay tools are retained separately in the
`rdna-wmma-captures-20261010.tar.zst` qualification archive. The archive README
provides its GPU-free replay instructions. The compact headers are sufficient
for the repository regression tests; they are a subset of that larger corpus.

| Header | Contents |
| --- | --- |
| `gfx1100_cases.h` | GFX11 FP32-output DOT2 cases and a WMMA matrix. |
| `gfx1201_cases.h` | GFX12 FP32-output DOT2 cases and WMMA matrices. |
| `packed_operand_cases.h` | 440 packed WMMA instruction/operand captures for modifiers, selectors and inline constants. |
| `packed_wmma_cases.h` | Seven F16/BF16 matrix inputs with packed outputs for both architectures and wave sizes. |
| `rdna_wmma_iu8_cases.h` | Four IU8 matrices for signed saturation, both architectures and wave sizes. |
| `rdna_wmma_iu4_cases.h` | Four IU4 matrices for signed saturation, both architectures and wave sizes. |
| `rdna_wmma_iu4_k32_cases.h` | Two IU4 K32 matrices for signed saturation on gfx1201, both wave sizes. |
| `rdna_swmmac_iu8_cases.h` | Sparse IU8 saturation/index/overlap matrices on gfx1201, both wave sizes. |
| `rdna_swmmac_iu4_cases.h` | Sparse IU4 K32 saturation/index/overlap matrices on gfx1201, both wave sizes. |
| `rdna_swmmac_iu4_k64_cases.h` | Sparse IU4 K64 saturation/index/overlap matrices on gfx1201, both wave sizes. |
| `rdna_wmma_f32_fp8_fp8_cases.h` | FP8/FP8 F32 WMMA raw matrices and operand witnesses on gfx1201, both wave sizes. |
| `rdna_wmma_f32_fp8_bf8_cases.h` | FP8/BF8 F32 WMMA raw matrices and operand witnesses on gfx1201, both wave sizes. |
| `rdna_wmma_f32_bf8_fp8_cases.h` | BF8/FP8 F32 WMMA raw matrices and operand witnesses on gfx1201, both wave sizes. |
| `rdna_wmma_f32_bf8_bf8_cases.h` | BF8/BF8 F32 WMMA raw matrices and operand witnesses on gfx1201, both wave sizes. |
| `rdna_wmma_f32_f16_cases.h` | F16 F32 WMMA raw matrices and operand witnesses on gfx1100 and gfx1201, both wave sizes. |
| `rdna_wmma_f32_bf16_cases.h` | BF16 F32 WMMA raw matrices and operand witnesses on gfx1100 and gfx1201, both wave sizes. |
| `rdna_wmma_f16_f16_cases.h` | F16 F16 WMMA raw matrices and operand witnesses on gfx1100 and gfx1201, both wave sizes. |
| `rdna_wmma_bf16_bf16_cases.h` | BF16 BF16 WMMA raw matrices and operand witnesses on gfx1100 and gfx1201, both wave sizes. |
| `rdna_swmmac_f32_f16_cases.h` | F16 F32 sparse SWMMAC raw matrices and operand witnesses on gfx1201, both wave sizes. |
| `rdna_swmmac_f32_bf16_cases.h` | BF16 F32 sparse SWMMAC raw matrices and operand witnesses on gfx1201, both wave sizes. |
| `rdna_swmmac_f16_f16_cases.h` | F16 F16 sparse SWMMAC raw matrices and operand witnesses on gfx1201, both wave sizes. |
| `rdna_swmmac_bf16_bf16_cases.h` | BF16 BF16 sparse SWMMAC raw matrices and operand witnesses on gfx1201, both wave sizes. |
| `rdna_swmmac_f32_fp8_fp8_cases.h` | FP8/FP8 F32 sparse SWMMAC raw matrices and operand witnesses on gfx1201, both wave sizes. |
| `rdna_swmmac_f32_fp8_bf8_cases.h` | FP8/BF8 F32 sparse SWMMAC raw matrices and operand witnesses on gfx1201, both wave sizes. |
| `rdna_swmmac_f32_bf8_fp8_cases.h` | BF8/FP8 F32 sparse SWMMAC raw matrices and operand witnesses on gfx1201, both wave sizes. |
| `rdna_swmmac_f32_bf8_bf8_cases.h` | BF8/BF8 F32 sparse SWMMAC raw matrices and operand witnesses on gfx1201, both wave sizes. |

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

The GFX11 fixtures also test RDNA3.5 decoded dispatch, including the four dense
16-bit opcodes in the shared float runner. This reuses gfx1100 observations and
does not claim a physical RDNA3.5 measurement. The original floating-point
fixtures check the dense F16/BF16 DOT2/WMMA models against local observations;
later sections describe the additional integer and native 8-bit formats.

## Integer IU8 WMMA captures

`rdna_wmma_iu8_cases.h` contains physical `V_WMMA_I32_16X16X16_IU8`
outputs captured on the same W7900/gfx1100 and R9700/gfx1201 devices on
2026-10-10, using TheRock 10.1.0a20260822. Each entry holds 256 raw destination
words in register-major, then lane-major order. Entries are ordered by
architecture (gfx1100, gfx1201), then wave size (32, 64). Both sources are
signed and CLAMP is enabled. MODE reads back as `0xf0`.

`RdnaWmmaI32.HardwareSaturation` reconstructs the input registers from a
32-bit xorshift seed `0x12056101` and the eight listed accumulator boundary
values. GFX11 source lanes 0-15 are replicated across the remaining lane
groups. The probe loads A at v0, B at v8, C at v16, and D at v24, executes one
WMMA, and stores the complete raw register words. The captured instructions
use the canonical assembler encoding `cc44c018 7c421100` on both targets.
Disassembly verifies
the opcode/modifiers, and code-object metadata verifies each wave width.

These captures reject final-sum-only saturation and are consistent with a model
that saturates after two products on GFX11 and eight on GFX12, with a different
K order for GFX12 wave64. Those intermediate groups and their order are inferred,
not directly observed inside the hardware. Separate physical probes cover all
signedness/clamp
combinations, inline C (including integer -1), destination overlaps, and
MODE readback. RDNA3.5 selects the GFX11 implementation but has no local
physical measurement. Finite hardware corpora do not prove all possible
matrix inputs.

## Integer IU4 WMMA captures

`rdna_wmma_iu4_cases.h` records `V_WMMA_I32_16X16X16_IU4` on the same
two devices, SDK and date as IU8, using the same seeded input registers and
entry ordering. The canonical signed/clamped instruction is
`cc45c018 7c421100`. The best-effort IU4 model clamps after four products on
GFX11 and eight on GFX12; both wave sizes use the existing IU4 input order.
This grouping is consistent with the local captures. The parameterized
integer tests replay both corpora and distinguish saturation from wraparound,
all signedness combinations, destination overlap, and inline C=-1.

## Integer IU4 K32 WMMA captures

`rdna_wmma_iu4_k32_cases.h` records `V_WMMA_I32_16X16X32_IU4` on
R9700/gfx1201 with the same SDK, date and seeded input reconstruction as
IU4 K16. Entries are wave32 then wave64. The canonical signed/clamped
instruction is `cc4ac018 7c421100`, with MODE readback `0xf0`.
The best-effort model clamps after each group of sixteen products. Its wave32
order consumes both registers of one lane group before the other lane group,
interchanging K bits 3 and 4 relative to the generic layout. Its wave64 order is
the generic order. These choices reproduce the local captures without identifying
the hardware's internal order. The decoded tests replay the raw outputs and cover
signedness, wraparound, overlapping destinations and inline C=-1 for this shape.

## Sparse integer IU8 SWMMAC captures

`rdna_swmmac_iu8_cases.h` records `V_SWMMAC_I32_16X16X32_IU8` on
R9700/gfx1201 on 2026-10-10 using TheRock 10.1.0a20260822. Both inputs
are signed and CLAMP is enabled. The base instruction is
`cc54c018 7c821100`, with the key in OPSEL and the destination changed
as recorded in each fixture. Controlled MODE fields read back as `0xf0`.
Each record holds 256 raw destination words in register/lane order.

`RdnaSwmmacI32.HardwareSaturationAndOverlap` reconstructs the same
xorshift seed and accumulator boundary words as the dense integer suite.
Each nibble in index register v32 is replaced with
`{4, 8, 12, 9, 13, 14}[nibble % 6]`, retaining only supported ordered
2-of-4 index pairs. A and B start at v0 and v8; D is also C.
Destinations v0/v8 overlap A/B at both widths. Additional destinations
v1/v9 retain shifted placements, while v25 in wave32 and v29 in wave64
include the index register. Every source is read before destination writes.
The tests compare all 33 registers, including unchanged values.

The best-effort model clamps after each group of eight compressed products.
Its wave32 index mapping uses sixteen bits per lane with OPSEL[0] selecting keys
0-1 and OPSEL[1] ignored; wave64 uses eight bits per lane and OPSEL[1:0] selects
keys 0-3. Local captures are consistent with this mapping and show no change from
the extra wave32 bit in the saved cases. The decoded tests replay those aliases
against the same expected words.
LLVM rejects spelling those extra wave32 keys, so this alias check is
additional raw-encoding coverage outside the supported operand range.
Separate uniform physical captures
supply the signedness and wrap/clamp expectations in the decoded tests.

## Sparse integer IU4 K32 SWMMAC captures

`rdna_swmmac_iu4_cases.h` records `V_SWMMAC_I32_16X16X32_IU4` on the
same R9700, SDK and date, using the sparse IU8 input reconstruction.
Destinations v0/v8 overlap A/B at both widths, while v25 in wave32 and
v29 in wave64 overlap the index register. Additional destinations retain
the previous IU8 probe placements without implying source overlap for IU4.
The base signed/clamped instruction is
`cc55c018 7c821100`. The best-effort model uses sixteen index bits per lane at
both widths, with OPSEL[0] selecting the set and saturation after each group of
eight compressed products. This mapping and grouping fit the local captures.
Physical uniform captures supply the signedness and wraparound expectations.
Additional raw captures show unchanged outputs when OPSEL[1] is set, including
all overlapping destinations in the fixture table.

## Sparse integer IU4 K64 SWMMAC captures

`rdna_swmmac_iu4_k64_cases.h` records `V_SWMMAC_I32_16X16X64_IU4`
with the same device, SDK, date and input reconstruction. Wave32 A occupies
v0-v1 and B v8-v11; wave64 A occupies v0 and B v8-v9. Destinations v0 and v8 overlap
all A and B registers respectively. In wave32, v1 also overlaps A v1 and B v8,
and v9 overlaps B v9-v11. The control is v24; the index-overlap
destinations are v25 in wave32 and v29 in wave64.
The base signed/clamped instruction is `cc56c018 7c821100`.
In the best-effort model, wave32 uses all 32 index bits per lane and ignores
the index key; wave64 uses sixteen bits per lane and OPSEL[0] selects the set.
The model clamps after each group of sixteen compressed products. As with dense
IU4 K32, its wave32 order consumes both registers of one lane group before the
other lane group, interchanging compressed K bits 3 and 4 relative to the generic
layout. Its wave64 order is the generic order. These are inferences from local
outputs, not observations of internal hardware steps. The decoded tests retain
all measured raw destination words, signedness/wrap expectations and key aliases.

## Native FP8/BF8 WMMA captures

The `V_WMMA_F32_16X16X16_*` fixtures below were captured on R9700/gfx1201
on 2026-10-10 using TheRock 10.1.0a20260822, for wave32 and wave64. Matrix
records retain the exact instruction words, MODE, input seed, input pattern
and a counted span of raw destination words. The shared `RdnaWmmaFloat` test
reconstructs all 33 registers and checks every register after decoded execution
in both default and forced-scalar modes. Destinations v0 and v8 directly overlap
A and B; v24 is a control. The xorshift seed is `0x12056f10`; the finite pattern
bounds each factor to its format's finite range and selects C boundary words.

The best-effort model uses two steps of eight products. Within each modeled
step, it aligns groups of four products independently, then aligns their sums
with C. It interchanges generic K bits 2 and 3 for wave64 and uses the generic
order for wave32. Native FP8/BF8 subnormal exponent floors participate in this
alignment model. Factor NaNs and invalid operations involving BF8 infinities
produce `0xffc00a3d` in the model; otherwise it preserves and quiets C's NaN
payload. Its invalid products take precedence over a NaN C, while a valid
infinite product preserves it. These choices fit the local captures but do not
identify hardware internals. FP8 has no infinities. Saved uniform witnesses
include negative-zero A and C with positive-one B producing positive zero, and
a negative subnormal C surviving zero products.

Captures with BF8 in A or B constrain special-value precedence in the model:
opposing infinities within one modeled step override a NaN C, while opposite
infinities in separate modeled steps preserve C's NaN payload. In that model,
each step receives a NaN accumulator, which takes precedence over its valid
infinite products. The captures expose only final outputs, not these steps.

Uniform operand records retain A/B/C/D/index words and physical expected bits.
Inline 0, 0.5 and integer -1 fixtures exercise constant C, including raw
`0xffffffff`. Compact matrix fixtures also record MODE `0x0f` and `0x8000ff`;
zero-product cases keep subnormal C observable under preserve/flush and rounding
settings. The ISA lists no ABS/NEG support for FP8/BF8 WMMA. LLVM accepts the
C ABS/NEG bit encodings; additional local gfx1201 captures returned unchanged
outputs for those bits. The model ignores them based on those cases. These are
extra encoding observations, not defined modifiers.

For a factor made of repeated `0x7c` bytes, with the other factor +1 and C
zero, the measured D word is `0x45c00000` for FP8 and `0x7f800000` for BF8.
Positive one is encoded as `0x38` in FP8 and `0x3c` in BF8. Both operand
positions and both wave widths are captured.

The per-format headers select these operand interpretations:

- `rdna_wmma_f32_fp8_fp8_cases.h`: FP8 A and B; neither has infinities.
  Compact operands cover C-bit encodings, inline C and zero-product MODE cases.
  Additional archived observations cover direct special-value witnesses. The
  remaining headers below embed their special-value witnesses as compact operands.
- `rdna_wmma_f32_fp8_bf8_cases.h`: FP8 A and BF8 B. The `0x7c` witnesses
  distinguish finite A from infinite B.
- `rdna_wmma_f32_bf8_fp8_cases.h`: BF8 A and FP8 B. The `0x7c` witnesses
  distinguish infinite A from finite B, independently of the reverse format order.
- `rdna_wmma_f32_bf8_bf8_cases.h`: BF8 A and B. Both `0x7c` factors
  are infinite; invalid-product and accumulator-NaN witnesses cover both sides.
  Twenty-four nonuniform operand records distinguish infinities within and across
  steps, with quiet/signaling NaN or zero C, at both wave widths. These records
  optionally replace 128 lane words starting at VGPR `factor_base`: each of the
  eight `factor_groups` words repeats across sixteen consecutive lanes, in
  register-major order. The span covers four VGPRs at wave32 or two at wave64.
  All other input registers retain the uniform words.

The larger retained local corpus includes all 65,536 raw byte pairs per format,
finite and raw matrices, every inline C selector, MODE sweeps, destination
overlaps, cancellation/alignment witnesses and the extra C-bit encodings.
These finite observations qualify the model but do not prove all possible
matrix inputs.

## F16/BF16 sparse SWMMAC captures

The per-opcode fixtures below were captured on R9700/gfx1201
on 2026-10-10 using TheRock 10.1.0a20260822, for wave32 and wave64. They use
the same seed, raw/finite input patterns and all-register decoded comparison
described above. Destinations v0 and v8 directly overlap A and B; v24 is a
control. Compact matrices retain non-default MODE settings, and zero-product
operands make subnormal C observable.
Sparse C is tied to D. Fixtures also overlap D with the index VGPR using
the actual output span for each width. They retain every legal index key and
ordered, distinct two-of-four indices. The best-effort model uses sixteen index
bits per lane in wave32 and eight in wave64, with contiguous compressed K at
both widths and four steps of four compressed products. Operand records cover
every supported source modifier. In the model, A negation follows its packed
half; B negation follows compressed-product parity after sparse selection.
These rules are inferred solely from the local input/output observations.

Additional compact operands retain raw sparse C ABS/NEG encodings rejected
by the local LLVM assembler. Local gfx1201 results are consistent with applying
ABS before NEG to the tied accumulator, as in the dense WMMA model. This does
not establish an internal hardware ordering. These extra encoding observations
are excluded from supported-operand counts. Each qualified header retains six
such operands,
using destination v24 and MODE `0xf0` at both wave widths. Separate archived
qualification covers 192 additional raw C-bit captures per opcode, including
clamp combinations, with zero decoded replay mismatches in both execution settings.

Qualified per-opcode headers:

- `rdna_swmmac_f32_f16_cases.h`: `V_SWMMAC_F32_16X16X32_F16`.
- `rdna_swmmac_f32_bf16_cases.h`: `V_SWMMAC_F32_16X16X32_BF16`.
- `rdna_swmmac_f16_f16_cases.h`: `V_SWMMAC_F16_16X16X32_F16`.
  The model rounds each step directly to F16, including its inferred handling
  of F16 overflow mode.
- `rdna_swmmac_bf16_bf16_cases.h`: `V_SWMMAC_BF16_16X16X32_BF16`.
  The model truncates each step's rounded F32 result to BF16.

The retained corpus covers finite and raw matrices, MODE sweeps and shifted
overlaps. These finite observations qualify the model but do not prove all
possible matrix inputs.

## Native FP8/BF8 sparse SWMMAC captures

The per-opcode fixtures below were captured on R9700/gfx1201
on 2026-10-10 using TheRock 10.1.0a20260822, for wave32 and wave64. They use
the same seed, raw/finite input patterns and all-register decoded comparison
described above. Destinations v0 and v8 directly overlap A and B; v24 is a
control. Compact matrices retain non-default MODE settings, and zero-product
operands make subnormal C observable.
Sparse C is tied to D. Fixtures also overlap D with the index VGPR using
the actual output span for each width. They retain every legal index key and
ordered, distinct two-of-four indices. The best-effort model uses sixteen index
bits per lane in wave32 and eight in wave64, with contiguous compressed K at
both widths and two steps of eight compressed products. It uses the same
inferred alignment, zero and NaN rules as the dense FP8/BF8 model above. The ISA
defines no source sign modifiers for this family. Separate archived local
gfx1201 captures returned unchanged outputs for raw C ABS/NEG and clamp
encodings: all 16 flag batches per opcode match decoded replay in both execution
settings with the model ignoring those bits. These observations are
outside the supported operand scope. Raw A/B sign encodings have not been
measured. Each qualified header
also retains raw C-bit witnesses at destination v24 and MODE `0xf0`, with zero
and nonzero products. The latter distinguish ignored modifiers from a missing
destination write. Uniform special-value records retain
the format-sensitive `0x7c` witnesses, factor/accumulator NaNs, and negative-zero
A/C with positive-one B.

Qualified per-opcode headers:

- `rdna_swmmac_f32_fp8_fp8_cases.h`: `V_SWMMAC_F32_16X16X32_FP8_FP8`;
  both factors are FP8, with no infinities.
- `rdna_swmmac_f32_fp8_bf8_cases.h`: `V_SWMMAC_F32_16X16X32_FP8_BF8`;
  A is FP8 and B is BF8.
  Twelve nonuniform operand records distinguish infinities within and across
  eight-product steps, with quiet/signaling NaN or zero C. They patch B at
  `factor_base` 8 using `factor_groups`.
- `rdna_swmmac_f32_bf8_fp8_cases.h`: `V_SWMMAC_F32_16X16X32_BF8_FP8`;
  A is BF8 and B is FP8.
  Twelve nonuniform operand records distinguish infinities within and across
  eight-product steps, with quiet/signaling NaN or zero C. They patch A at
  `factor_base` 0 using `factor_groups`.
- `rdna_swmmac_f32_bf8_bf8_cases.h`: `V_SWMMAC_F32_16X16X32_BF8_BF8`;
  both factors are BF8.
  Twenty-four nonuniform operand records distinguish infinities within and across
  eight-product steps, with quiet/signaling NaN or zero C. They patch A or B at
  `factor_base` 0 or 8 using `factor_groups`.

The retained corpus covers finite and raw matrices, MODE sweeps and shifted
overlaps. Larger local sweeps outside the compact headers retain all 65,536 raw
byte pairs per format in `results/fp8-pairs-op*`, plus cancellation/alignment
witnesses in `results/float-cancellation-op*`. The pair sweep uses 256 matrices,
each testing the Cartesian product of 16 A bytes and 16 B bytes. These finite
observations qualify the model but do not prove all possible matrix inputs.

## F16/BF16 WMMA captures

The per-opcode fixtures below were captured on W7900/gfx1100 and R9700/gfx1201
on 2026-10-10 using TheRock 10.1.0a20260822, for wave32 and wave64. They use
the same seed, raw/finite input patterns and all-register decoded comparison
described above. Destinations v0 and v8 directly overlap A and B; v24 is a
control. Compact matrices retain non-default MODE settings, and zero-product
operands probe subnormal C. The observations below distinguish output formats.
These observations are consistent with the existing GFX11 eight-DOT2-step and
GFX12 four-DOT4-step models. The per-opcode decoded fixtures supplement the
older DOT/WMMA arithmetic fixtures listed above; both corpora remain regression
coverage when the model changes. Operand records cover every supported
source modifier. Inline 0, 0.5 and integer -1 witnesses retain constant-C raw
results.

Qualified per-opcode headers:

- `rdna_wmma_f32_f16_cases.h`: `V_WMMA_F32_16X16X16_F16`.
  The recorded zero-product F32 subnormal C flushes on gfx1100 and survives
  on gfx1201.
  Separate archived CLAMP observations are outside supported operand forms
  and match decoded replay in both execution settings.
  The gfx1100 CLAMP captures also use a noncanonical high selector; they are
  retained as adjacent observations and excluded from supported coverage.
- `rdna_wmma_f32_bf16_cases.h`: `V_WMMA_F32_16X16X16_BF16`.
  The recorded zero-product F32 subnormal C flushes on gfx1100 and survives
  on gfx1201.
  Uniform factors have the same numerical values as the F16 witnesses, with
  distinct BF16 bit patterns; both operand and matrix records distinguish the
  formats. Four additional matrices bound factor exponents from -6 through 3
  so finite accumulation and the C boundary words remain observable. These
  use `bounded_bf16` and retain the full-range overflow witnesses as well.
  Separate archived C=0 controls demonstrate sensitivity to the accumulator.
  Separate archived CLAMP observations are outside supported operand forms
  and match decoded replay in both execution settings.
  The gfx1100 CLAMP captures also use a noncanonical high selector; they are
  retained as adjacent observations and excluded from supported coverage.
- `rdna_wmma_f16_f16_cases.h`: `V_WMMA_F16_16X16X16_F16`;
  operand witnesses include both RDNA3 packed destination half selections.
  The model narrows packed output after every inferred arithmetic step.
  The full-range finite
  matrices retain F32-shaped C/D words and primarily qualify overflow and layout;
  the finite operand records below supply evidence for narrowing and packed
  accumulation. Zero-product F16 subnormal C survives in the selected output
  halves on gfx1100 and gfx1201 for the three retained MODE settings.
  Forty-eight finite nonuniform operand records pair nonzero C with C=0, using
  four factor patterns at `factor_base` 0. Each of the eight `factor_groups` words
  repeats across sixteen consecutive lanes in register-major order, patching
  128 lane words. Groups repeat within each RDNA3 register to preserve input
  replication. These records cover gfx1100 and gfx1201 at both wave widths,
  including both RDNA3 destination halves. Sixteen additional RDNA3 high-half matrix
  records cross raw/finite inputs with v0/v8 overlaps and non-default MODE,
  including FP16_OVFL. Six high-half zero-product records cover the same three
  MODE settings as their low-half counterparts.
  Separate archived CLAMP observations are outside supported operand forms
  and match decoded replay in both execution settings.
  The gfx1100 CLAMP captures also use a noncanonical high selector; they are
  retained as adjacent observations and excluded from supported coverage.
- `rdna_wmma_bf16_bf16_cases.h`: `V_WMMA_BF16_16X16X16_BF16`;
  operand witnesses include both RDNA3 packed destination half selections.
  The model narrows packed output after every inferred arithmetic step.
  Zero-product BF16 subnormal C flushes on gfx1100 and survives on gfx1201
  in the selected output halves for all three retained MODE settings.
  Full-range matrices retain F32-shaped C/D words. Six records set
  `bounded_bf16`, selecting bounded factors and packed BF16 C/D values to keep
  finite accumulation observable. Four records, one per architecture and wave
  width, have separate archived physical C=0 controls. Two RDNA3 high-half
  counterparts cover destination placement and
  preservation. The C halves are equal in these matrices; uniform operand
  records with distinct C halves qualify which half is read.
  Eighteen high-half RDNA3 matrices and six zero-product operand records
  cross that selection with MODE, raw/finite inputs
  and destination overlaps. These are counterparts of all retained low-half
  RDNA3 matrix and subnormal-C cases.
  Separate archived CLAMP observations are outside supported operand forms
  and match decoded replay in both execution settings.
  The gfx1100 CLAMP captures also use a noncanonical high selector; they are
  retained as adjacent observations and excluded from supported coverage.

The retained corpus covers finite and raw matrices, MODE
sweeps and shifted overlaps, plus every inline C selector. These finite observations
qualify the model but do not prove all possible matrix inputs.
