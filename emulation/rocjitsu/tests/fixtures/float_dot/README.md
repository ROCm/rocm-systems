# DOT2 and WMMA hardware fixtures

The floating-point raw results below were captured on Radeon Pro W7900 (`gfx1100`) and Radeon AI
Pro R9700 (`gfx1201`) using TheRock `10.2.0a20260916`. The decoded regression
tests consume the captured bits directly and require no GPU.

| Header | Contents |
| --- | --- |
| `gfx1100_cases.h` | GFX11 FP32-output DOT2 cases and a WMMA matrix. |
| `gfx1201_cases.h` | GFX12 FP32-output DOT2 cases and WMMA matrices. |
| `packed_operand_cases.h` | 440 packed WMMA instruction/operand captures for modifiers, selectors and inline constants. |
| `rdna_swmmac_iu4_cases.h` | Sparse IU4 K32 saturation/index/overlap matrices on gfx1201, both wave sizes. |
| `rdna_swmmac_iu4_k64_cases.h` | Sparse IU4 K64 saturation/index/overlap matrices on gfx1201, both wave sizes. |
| `rdna_swmmac_iu8_cases.h` | Sparse IU8 saturation/index/overlap matrices on gfx1201, both wave sizes. |
| `rdna_wmma_iu4_k32_cases.h` | Two IU4 K32 matrices for signed saturation on gfx1201, both wave sizes. |
| `rdna_wmma_iu4_cases.h` | Four IU4 matrices for signed saturation, both architectures and wave sizes. |
| `rdna_wmma_iu8_cases.h` | Four IU8 matrices for signed saturation, both architectures and wave sizes. |
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
RDNA3.5 measurement. The floating-point captures qualify dense F16/BF16 DOT2/WMMA arithmetic
on these two cards; they do not establish behavior for other matrix formats.

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
