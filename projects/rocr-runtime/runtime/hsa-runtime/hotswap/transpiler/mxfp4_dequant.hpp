#ifndef HOTSWAP_TRANSPILER_MXFP4_DEQUANT_HPP
#define HOTSWAP_TRANSPILER_MXFP4_DEQUANT_HPP

// see hotswap/docs/matrix-translation.md §7.4 — MXFP4 dequant primitive,
// cross-target lowering of v_cvt_scale_pk8_bf16_fp4 (gfx1250-only VOP3)
// into per-nibble bit-algebra expansion on any target that lacks the
// primitive (gfx942/gfx950).
//
// Two reference implementations live here so the unit test in
// `tests/mxfp4_dequant_test.cpp` can cross-check them against each
// other across the full 16 * 256 = 4096 input space:
//
//   * `mxfp4BitAlgebraBf16Bits` — the bit-algebra implementation
//     whose structure the IR emitter in handle_valu.cpp mirrors
//     step-for-step.  The algorithm is: decompose FP4 E2M1 into
//     (sign, exp_fp4, mant_fp4); synthesise BF16 normal-vs-subnormal
//     field values directly; add `scale_byte - 127` to the BF16
//     exponent; handle NaN scale / fp4 ±0 / overflow / underflow
//     branches.
//
//   * `mxfp4LutBf16Bits` — a slower, more obvious LUT-based
//     implementation: a 16-entry table of FP4 -> BF16 bit patterns
//     plus an explicit "multiply by 2^(scale_byte - 127)" step
//     that manipulates the BF16 exponent field via integer ops.
//     Different algorithm; same output on every input.
//
// The gtest asserts that mxfp4BitAlgebraBf16Bits(n, s) ==
// mxfp4LutBf16Bits(n, s) for every (n, s) in [0,16) * [0,256), giving
// us confidence that the bit-algebra lowering is semantically
// equivalent to the canonical OCP-spec-derived table expansion.  The
// canary `canary_cvt_scale_pk8_bf16_fp4.hip` independently verifies
// the bit-algebra output against the hardware primitive on gfx1250.
//
// Both functions encode the `scale_sel == 0` semantics only — i.e.
// the scale byte is taken from the low byte of the 32-bit scale
// register.  `scale_sel != 0` is outside the declared support set;
// the handler refuses loudly.  See matrix-translation.md §7.4 for
// the corpus-provenance argument (both `_matmul_ogs_*` blobs use
// only scale_sel=0 across 128 instances combined).

#include <cstdint>

namespace transpiler {
namespace mxfp4 {

// Compute BF16 bit pattern for `fp4_e2m1_to_bf16(nibble) * 2^(scale_byte - 127)`
// using per-field bit arithmetic (no fp multiply, no LUT).  Matches the
// IR emission in handle_valu.cpp::V_CVT_SCALE_PK8_BF16_FP4 step-for-step.
//
// Arguments
//   nibble     : FP4 E2M1 encoding carried in the low 4 bits.  The
//                function internally masks with 0xF (same as the IR
//                emitter's `and i32 %lane_nibble, 15`), so callers
//                may pass any uint8; the upper 4 bits are ignored.
//                Layout: bit 3 = sign, bits 2..1 = exp (bias=1), bit
//                0 = mantissa.
//   scale_byte : E8M0 scale byte.  0xFF is NaN; 0x00..0xFE encodes
//                2^(scale_byte - 127).  Since the parameter is a
//                uint8 all 8 bits are consumed; the caller is
//                responsible for choosing which byte of the 32-bit
//                hardware scale register to pass (scale_sel == 0
//                means the low byte; the handler refuses other
//                scale_sel values per the declared support set).
//
// Output corners
//   scale_byte == 0xFF        -> BF16 canonical qNaN (0x7FC0)
//   fp4 encodes ±0            -> BF16 ±0 preserving sign (finite scale)
//   new_exp >= 0xFF            -> BF16 ±Inf preserving sign
//   new_exp in [1, 254]        -> BF16 normal with mantissa preserved
//   new_exp <= 0               -> BF16 subnormal via right-shift of the
//                                implicit-1.mant; zero when all bits fall
//                                off past shift count 7.
uint16_t mxfp4BitAlgebraBf16Bits(uint8_t nibble, uint8_t scale_byte);

// Compute BF16 bit pattern via the canonical 16-entry LUT + explicit
// exponent-field manipulation.  Different algorithmic shape from
// `mxfp4BitAlgebraBf16Bits`; the unit test asserts they agree on every
// input, which gives us two-implementation confidence that neither
// side silently encoded an OCP-spec miscount.
uint16_t mxfp4LutBf16Bits(uint8_t nibble, uint8_t scale_byte);

// 16-entry FP4 E2M1 -> BF16 table, exposed for the unit test to pin
// the constants against the OCP MXFP spec explicitly.  Indexed by the
// raw 4-bit nibble value:
//
//   0b0000 = +0     0b0001 = +0.5   0b0010 = +1     0b0011 = +1.5
//   0b0100 = +2     0b0101 = +3     0b0110 = +4     0b0111 = +6
//   0b1000 = -0     0b1001 = -0.5   0b1010 = -1     0b1011 = -1.5
//   0b1100 = -2     0b1101 = -3     0b1110 = -4     0b1111 = -6
//
// These are the bit-exact BF16 encodings of the 16 FP4 E2M1 values at
// identity scale (scale_byte = 0x7F = 2^0).
extern const uint16_t kMxfp4ToBf16Table[16];

} // namespace mxfp4
} // namespace transpiler

#endif // HOTSWAP_TRANSPILER_MXFP4_DEQUANT_HPP
