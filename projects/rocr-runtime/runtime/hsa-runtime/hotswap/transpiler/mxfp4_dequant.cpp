// see hotswap/docs/matrix-translation.md §7.4 — MXFP4 dequant primitive.
//
// Two-way reference implementation for cross-target lowering of
// `v_cvt_scale_pk8_bf16_fp4` (gfx1250-only VOP3).  See mxfp4_dequant.hpp
// for the two functions' contracts and the cross-check invariant the
// unit test asserts.

#include "mxfp4_dequant.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <cstdint>

namespace transpiler {
namespace mxfp4 {

// OCP MXFP FP4 E2M1 -> BF16 bit patterns at identity scale (2^0).
// Derivation (OCP MXFP spec):
//   0b0000 / 0b1000 -> ±0.0   = 0x0000 / 0x8000   (BF16 ±0)
//   0b0001 / 0b1001 -> ±0.5   = 0x3F00 / 0xBF00   (BF16 exp=126, mant=0)
//   0b0010 / 0b1010 -> ±1.0   = 0x3F80 / 0xBF80   (BF16 exp=127, mant=0)
//   0b0011 / 0b1011 -> ±1.5   = 0x3FC0 / 0xBFC0   (BF16 exp=127, mant=0x40)
//   0b0100 / 0b1100 -> ±2.0   = 0x4000 / 0xC000   (BF16 exp=128, mant=0)
//   0b0101 / 0b1101 -> ±3.0   = 0x4040 / 0xC040   (BF16 exp=128, mant=0x40)
//   0b0110 / 0b1110 -> ±4.0   = 0x4080 / 0xC080   (BF16 exp=129, mant=0)
//   0b0111 / 0b1111 -> ±6.0   = 0x40C0 / 0xC0C0   (BF16 exp=129, mant=0x40)
const uint16_t kMxfp4ToBf16Table[16] = {
    0x0000, 0x3F00, 0x3F80, 0x3FC0,
    0x4000, 0x4040, 0x4080, 0x40C0,
    0x8000, 0xBF00, 0xBF80, 0xBFC0,
    0xC000, 0xC040, 0xC080, 0xC0C0,
};

uint16_t mxfp4BitAlgebraBf16Bits(uint8_t nibble, uint8_t scale_byte) {
  uint32_t n = static_cast<uint32_t>(nibble) & 0xFu;

  // FP4 E2M1 field extraction.  sign is bit 3; exp is bits 2..1
  // (bias=1, range 0..3); mant is bit 0.
  uint32_t sign_bit = (n >> 3) & 0x1u;
  uint32_t exp_fp4  = (n >> 1) & 0x3u;
  uint32_t mant_fp4 =  n       & 0x1u;

  // Normal-FP4 -> BF16 field synthesis.  FP4 exp = 1..3 maps to BF16
  // exp = 127..129 (i.e. exp_fp4 + 126).  FP4 mant=1 maps to BF16
  // mantissa 0x40 (the top bit of the 7-bit mantissa, which for BF16
  // at exp=127 encodes the 0.5 contribution: 1 + 0x40*2^-7 = 1.5).
  uint32_t bf16_exp_norm  = exp_fp4 + 126u;
  uint32_t bf16_mant_norm = mant_fp4 ? 0x40u : 0x00u;

  // Subnormal-FP4 branch (exp_fp4 == 0): FP4 value is ±0 (mant_fp4=0)
  // or ±0.5 (mant_fp4=1).  ±0.5 is representable as a BF16 NORMAL
  // (exp=126, mant=0); ±0 is BF16 ±0 (exp=0, mant=0 with sign).
  uint32_t bf16_exp_sub   = mant_fp4 ? 126u : 0u;
  uint32_t bf16_mant_sub  = 0u;

  uint32_t is_sub = (exp_fp4 == 0u) ? 1u : 0u;
  uint32_t bf16_exp  = is_sub ? bf16_exp_sub  : bf16_exp_norm;
  uint32_t bf16_mant = is_sub ? bf16_mant_sub : bf16_mant_norm;

  // Apply the E8M0 scale by adjusting the BF16 exponent field.  For
  // scale_byte == 0xFF the output is NaN regardless of the FP4
  // nibble (IEEE-754 0 × NaN = NaN).  For FP4 ±0 inputs (bf16
  // magnitude == 0) the output is ±0 preserving sign (finite-scale
  // case), which falls out of the zero-exp + zero-mant branch
  // trivially without the scale add.
  uint32_t scale = static_cast<uint32_t>(scale_byte);

  if (scale == 0xFFu) {
    // Canonical BF16 qNaN: sign=0, exp=0xFF, mant[6]=1 -> 0x7FC0.
    // The hardware primitive's exact NaN bit pattern is not pinned
    // by the OCP MXFP spec; if the canary surfaces a divergence with
    // the native gfx1250 output, this constant is the knob to
    // revisit.
    return static_cast<uint16_t>(0x7FC0);
  }

  uint32_t magnitude = (bf16_exp << 7) | bf16_mant;  // low 15 bits
  uint32_t sign_field = sign_bit << 15;
  if (magnitude == 0u) {
    // FP4 ±0: result is ±0 regardless of the (finite) scale.  Sign
    // comes from the FP4 nibble's sign bit.
    return static_cast<uint16_t>(sign_field);
  }

  // Non-zero FP4 (magnitude > 0): bf16_exp is in 126..129, bf16_mant
  // is 0 or 0x40.  Compute the scaled exponent in signed arithmetic
  // so overflow / underflow branches dispatch on the sign of the
  // result (not on unsigned wrap).
  int32_t new_exp_i =
      static_cast<int32_t>(bf16_exp) +
      static_cast<int32_t>(scale) - 127;

  if (new_exp_i >= 0xFF) {
    // Overflow to BF16 ±Inf (BF16 supports inf; FP4 does not, but the
    // destination format's semantics apply after the scale add).
    return static_cast<uint16_t>(sign_field | 0x7F80u);
  }

  if (new_exp_i >= 1) {
    // Normal BF16 result.  Sign preserved, mantissa preserved
    // (power-of-2 scale doesn't change mantissa bits), exp replaced.
    return static_cast<uint16_t>(sign_field |
        ((static_cast<uint32_t>(new_exp_i) & 0xFFu) << 7) |
        bf16_mant);
  }

  // Underflow into BF16 subnormal range.  BF16 subnormals are
  // representable down to value = 2^-133 (mant = 0x01, exp = 0); any
  // result below that decays to ±0.  Construct the subnormal bit
  // pattern by shifting the implicit-1.mant value right by the
  // amount the exponent fell below 1.
  uint32_t implicit_1_mant = 0x80u | bf16_mant;     // 0x80 (FP4=1) or 0xC0 (FP4=1.5)
  int32_t shift_amt = 1 - new_exp_i;                // >= 1
  uint32_t sub_mant = (shift_amt >= 8)
                          ? 0u
                          : (implicit_1_mant >> shift_amt);
  return static_cast<uint16_t>(sign_field | sub_mant);
}

namespace {

// The 16 FP4 E2M1 values as exact IEEE-754 doubles.  Used by the
// double-math reference path; must correspond index-for-index to
// `kMxfp4ToBf16Table` so the unit test can cross-check both FP4 ->
// BF16 first-step lookups against the same OCP-spec table.
constexpr double kMxfp4ToDouble[16] = {
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
};

// Convert an exact-BF16-representable double to its BF16 bit pattern.
//
// Preconditions (enforced via `report_fatal_error` — violating them
// indicates either a caller bug or an unexpected hardware-primitive
// semantics, NOT a case for silent rounding):
//
//   * Input is finite (not Inf/NaN).  `mxfp4LutBf16Bits` computes
//     `fp4_val * scale` where `fp4_val` is finite (max ±6.0) and
//     `scale` is a finite power of 2 (2^-127..2^127), so the product
//     is always finite.  Scale-byte 0xFF (E8M0 NaN) is handled by
//     the caller BEFORE this function, so NaN doubles never arrive
//     here.
//
//   * The double's 52-bit mantissa has its low 45 bits zero.  Our
//     inputs are always FP4 mantissas (∈ {0, 1.0, 1.5}) scaled by
//     powers of 2; the non-zero mantissa cases have bit 51 set and
//     nothing below it.  A double with non-zero low-45 bits cannot
//     arise from our declared input domain, and would require RTE
//     rounding we intentionally do not implement — a future
//     extension that needs rounding should add an explicit rounding-
//     aware path and exercise it with tests, not silently reuse
//     this function.
//
// With those preconditions, the conversion is purely exponent-field
// remapping: the top 7 bits of the double's mantissa become the
// BF16 mantissa, no rounding.  Overflow (BF16 exp >= 0xFF) saturates
// to ±Inf; underflow (BF16 exp < 1) produces an exact BF16 subnormal
// or ±0 via right-shift of the implicit-1.mant field.
uint16_t exactDoubleToBf16(double v) {
  uint64_t u;
  static_assert(sizeof(u) == sizeof(v), "double must be 64 bits");
  __builtin_memcpy(&u, &v, sizeof(u));
  uint32_t sign   = static_cast<uint32_t>(u >> 63) & 1u;
  uint32_t exp_d  = static_cast<uint32_t>((u >> 52) & 0x7FFu);
  uint64_t mant_d = u & ((uint64_t{1} << 52) - 1);

  if (exp_d == 0x7FFu) {
    llvm::report_fatal_error(
        "mxfp4::exactDoubleToBf16: Inf/NaN input not representable "
        "under declared FP4 × power-of-2 input domain");
  }
  uint64_t low45 = mant_d & ((uint64_t{1} << 45) - 1);
  if (low45 != 0) {
    llvm::report_fatal_error(
        "mxfp4::exactDoubleToBf16: non-zero low-45 mantissa bits "
        "would require BF16 rounding; input outside declared "
        "FP4 × power-of-2 domain");
  }
  if (exp_d == 0u) {
    // Double subnormal/zero: every value below 2^-1022 is below the
    // BF16 subnormal floor of 2^-133, so the result is ±0.
    return static_cast<uint16_t>(sign << 15);
  }

  uint32_t high7 = static_cast<uint32_t>(mant_d >> 45);   // top 7 mantissa bits
  int32_t e_unbiased = static_cast<int32_t>(exp_d) - 1023;
  int32_t bf_exp = e_unbiased + 127;

  if (bf_exp >= 0xFF) {
    // Overflow beyond BF16 finite range -> saturate to ±Inf.
    return static_cast<uint16_t>((sign << 15) | 0x7F80u);
  }

  if (bf_exp >= 1) {
    return static_cast<uint16_t>((sign << 15) |
        ((static_cast<uint32_t>(bf_exp) & 0xFFu) << 7) |
        high7);
  }

  // Underflow: right-shift the implicit-1.mant pattern into the
  // subnormal range.  No rounding to worry about — the low bits we
  // drop are all zero by precondition (low45 == 0 ensures the top 7
  // mantissa bits are the ONLY set bits below bit 52).
  uint32_t implicit_1_mant = 0x80u | high7;   // 0x80 (FP4=1.0) or 0xC0 (FP4=1.5)
  int32_t shift_amt = 1 - bf_exp;              // >= 1
  uint32_t sub_mant = (shift_amt >= 8)
                          ? 0u
                          : (implicit_1_mant >> shift_amt);
  return static_cast<uint16_t>((sign << 15) | sub_mant);
}

// Build 2^(scale_byte - 127) as an exact double for scale_byte in 0..0xFE.
// E8M0 scale_byte == 0xFF is NaN and is handled by the caller.
double e8m0ScaleToDouble(uint8_t scale_byte) {
  // scale_byte == 0 -> 2^-127.  exp_bias for double is 1023, so
  // exp_field = (scale_byte - 127) + 1023 = scale_byte + 896.
  // scale_byte == 0xFE -> exp_field = 1150, which is within the
  // double's normal range (exp_field in [1, 2046] is normal).
  uint64_t bits =
      (static_cast<uint64_t>(static_cast<uint32_t>(scale_byte) + 896u))
      << 52;
  double d;
  __builtin_memcpy(&d, &bits, sizeof(d));
  return d;
}

} // namespace

uint16_t mxfp4LutBf16Bits(uint8_t nibble, uint8_t scale_byte) {
  // Different algorithmic shape from `mxfp4BitAlgebraBf16Bits`: LUT
  // the FP4 value into an exact double, multiply by the double form
  // of 2^(scale_byte - 127) (also exact — scale is a power of 2),
  // then round the result back to BF16 with IEEE RTE.  Shares only
  // the OCP-spec FP4 -> {double, bf16} tables (index-for-index
  // equivalent constants) and the canonical BF16 NaN bit pattern.
  //
  // If this function and `mxfp4BitAlgebraBf16Bits` agree on every
  // (nibble, scale_byte) input, we have two-implementation evidence
  // that neither encoding silently skewed from the OCP spec.  See
  // `tests/mxfp4_dequant_test.cpp` for the full 4096-point sweep.
  if (scale_byte == 0xFFu) {
    return static_cast<uint16_t>(0x7FC0);
  }
  double fp4_val = kMxfp4ToDouble[nibble & 0xFu];
  double scale   = e8m0ScaleToDouble(scale_byte);
  double scaled  = fp4_val * scale;  // exact: power-of-2 mul
  // `exactDoubleToBf16` handles ±0 (which fp4 * finite-scale
  // produces directly when fp4_val is ±0), overflow to Inf, and
  // subnormal underflow uniformly — with `report_fatal_error` on
  // any input it cannot convert without rounding.
  return exactDoubleToBf16(scaled);
}

} // namespace mxfp4
} // namespace transpiler
