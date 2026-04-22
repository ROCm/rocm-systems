// Canary for the gfx1250-only VOP3 scaled packed-8 FP4 -> BF16 convert
// (`v_cvt_scale_pk8_bf16_fp4`, VOP3Instructions.td:1873).  Pins the
// Salmon cross-target dequantisation expansion (gfx1250 -> gfx942)
// bit-for-bit against the hardware primitive's output on gfx1250.
//
// Why .hip (not Triton)
// =====================
//
// Triton's AMD backend does NOT emit `v_cvt_scale_pk8_bf16_fp4` from
// any surface-level API today.  `_upcast_from_mxfp.py`'s AMD branch
// hand-builds the dequant via bit-ops
// (`em0 = tensor & 0x07; x0 = em0 << (dst_m_bits - 1) | ...`); only
// the Hopper-PTX branch uses an inline-asm cvt.  The scaled WMMA path
// in `_p_matmul.py` goes through `tl.dot_scaled(..., "e2m1", ...,
// "bf16")`, which DOES lower to 64× `v_cvt_scale_pk8_bf16_fp4` + 64×
// `v_wmma_f32_16x16x32_bf16` on gfx1250 (observed in the captured
// corpus kernels `_matmul_ogs_{06d912ce88af,0af655e6ea2b}`), but
// bundles the primitive with dense WMMA so a Triton canary cannot
// isolate it.
//
// Therefore the canary is authored as a HIP kernel using the
// `__builtin_amdgcn_cvt_scale_pk8_bf16_fp4` clang builtin — same
// mechanism the existing lit fixture
// (`lit_tests/v_cvt_scale_pk8_bf16_fp4/v_cvt_scale_pk8_bf16_fp4.hip`)
// uses, and the same `.hip` canary discipline as
// `canary_readlane_last_lane.hip` (which is a `.hip` for the same
// reason — Triton can't force a specific lane constant).
//
// Regression-guard signature
// ==========================
//
// On gfx1250, per the existing lit fixture, one call to this builtin
// emits one `v_cvt_scale_pk8_bf16_fp4` instruction.  This canary's
// kernel invokes the builtin exactly once per thread; hipcc unrolls
// nothing, so the kernel's `.gfx1250.co` disassembly contains exactly
// 1 instance of the primitive (verified via `llvm-objdump` with an
// LLVM build that knows the gfx1250 scale opcode tables).  On
// gfx942 the builtin is unavailable (the intrinsic is gated behind
// `isGFX125xOnly` in `IntrinsicsAMDGPU.td:686`), so the gfx942 .co is
// built via a separate code path — a dequant written as portable
// C/HIP that the CPU reference matches bit-for-bit.  See
// `#ifdef __gfx1250__` below.
//
// Input coverage
// ==============
//
// Each thread processes one (packedFP4, scale) pair; the launch shape
// is configured so `N` threads cover:
//
//   * Every one of the 16 FP4 E2M1 nibble encodings at least once,
//     across the 8 nibble positions within the packed i32.
//   * E8M0 scale bytes covering: 0x00 (2^-127 subnormal scale),
//     0x01, 0x7E, 0x7F (scale=1.0 identity), 0x80 (2^1=2.0), 0x81,
//     0xFC, 0xFD, 0xFE (2^127 maximum normal scale), and 0xFF (NaN).
//   * Every one of the 4 `scale_sel` byte positions (0..3) exercised
//     in separate kernel invocations — see the `scale_sel` constexpr
//     variants below.  Only `scale_sel=0` is currently wired into
//     the recipe because the cross-target handler only supports
//     scale_sel=0 (corpus-matching scope); the remaining variants
//     live in the source file as forward-looking coverage once the
//     handler's declared support set widens.
//
// Comparator choice
// =================
//
// `abs-max` with `tol = 0` (bit-exact): the hardware primitive and
// the cross-target lowering are both deterministic bit-level
// operations over 16-bit BF16 values.  Multiplication by a power of
// 2 is representable exactly in BF16 when the result is in range, so
// there is no rounding to absorb.  The CPU reference implements the
// same bit-level arithmetic both the hardware and the Salmon
// lowering encode (OCP MXFP FP4 E2M1 -> BF16 table + E8M0 exponent
// add), so `native`, `legacy`, and `salmon` must all match the
// reference bit-for-bit.  Any non-zero tolerance would mask a real
// bug.
//
// Expected verdicts (after the cross-target handler lands)
// ========================================================
//
//   native : match.  hipcc on gfx942 does NOT know the builtin (it
//            errors out if we try to use it for gfx942); this file
//            guards the builtin behind `#ifdef __gfx1250__` and
//            provides a portable dequant fallback for the gfx942
//            build.  The fallback computes the same bit-level
//            result as the CPU reference, so native matches.
//
//   legacy : (expected behaviour not yet pinned.)  The byte-level
//            legacy transpiler's handling of the gfx1250-only
//            `v_cvt_scale_pk8_bf16_fp4` opcode is out of scope for
//            this canary; the recipe does not assert on it.
//            Whatever verdict legacy produces today is the verdict
//            recorded in the landing commit message.
//
//   salmon : match.  Under `WaveNativeProjection` (default), the
//            cross-target handler in `handle_valu.cpp` expands each
//            `v_cvt_scale_pk8_bf16_fp4` into a per-nibble FP4 -> BF16
//            bit-algebra chain + E8M0 exponent-bits add, producing
//            the same BF16 bit pattern the gfx1250 native primitive
//            emits.  Bit-exact.
//
// Regression contract
// ===================
//
//   match -> WRONG k/N on salmon: either the FP4 -> BF16 bit algebra
//     in the handler regressed, the scale-byte extraction regressed
//     (scale_sel != 0 silently accepted / scale_sel == 0 silently
//     wrong-byte), or the subnormal / overflow / NaN branches in the
//     scaled-multiply arithmetic regressed.  First mismatching index
//     in the Failures section of the compare_correctness report
//     points at the offending (nibble, scale_byte) combination.
//
//   match -> EXIT=2 on salmon: the handler's cross-target arm
//     started refusing something it previously lowered.  Expected
//     only if the declared support set was deliberately narrowed
//     (e.g. `scale_sel != 0` is already refused today); if not, the
//     refusal is a regression.  See
//     `hotswap/docs/matrix-translation.md §7.N` for the current
//     declared support set.

#include <hip/hip_runtime.h>

// Host-visible struct: one output word per thread holds the 8 BF16
// results packed into 128 bits.  We use `uint16_t` 8-wide to keep the
// per-element read simple on the host; the kernel stores through a
// reinterpret to the compiler's `__bf16` vector so that the builtin's
// `<8 x bfloat>` return value round-trips without bit-wise surprises.
struct Packed8BF16 {
  uint16_t w[8];
};

// One thread per (packedFP4, scale) pair. `in_pairs[tid].packed` is a
// 32-bit word holding 8 FP4 nibbles; `in_pairs[tid].scale` is a 32-bit
// word holding up to 4 E8M0 scale bytes (we only use the low byte —
// the handler only supports scale_sel=0 today).  Output `out[tid]`
// holds the 8 BF16 bit patterns the hardware primitive would emit,
// or the portable-dequant equivalent on gfx942.
struct InputPair {
  uint32_t packed;
  uint32_t scale;
};

// Portable FP4 E2M1 -> BF16 bit-pattern table, indexed by the 4-bit
// nibble value (low nibble is lane 0 on hardware).  Derived from the
// OCP MXFP spec; matches the hardware primitive's output for
// scale_byte = 0x7F (identity scale).  The exact same values are
// computed bit-algebraically by the Salmon cross-target lowering —
// see the design-note comment above the `V_CVT_SCALE_PK8_BF16_FP4`
// cross-target handler in `handle_valu.cpp`.
//
// Encodings (OCP MXFP FP4 E2M1, bias=1):
//   0000: +0      0001: +0.5     0010: +1       0011: +1.5
//   0100: +2      0101: +3       0110: +4       0111: +6
//   1000: -0      1001: -0.5     1010: -1       1011: -1.5
//   1100: -2      1101: -3       1110: -4       1111: -6
__device__ __host__ __forceinline__ uint16_t fp4_nibble_to_bf16_bits(uint32_t nibble) {
  // Hand-rolled bit-algebra (no LUT) so the device-side gfx942
  // fallback path produces the same sequence the Salmon lowering
  // emits as IR, without either side drifting from the table. See
  // the handler's design-note comment for the derivation.
  uint32_t sign_bit  = (nibble >> 3) & 0x1u;
  uint32_t exp_fp4   = (nibble >> 1) & 0x3u;  // 0..3
  uint32_t mant_fp4  =  nibble       & 0x1u;  // 0 or 1
  // Normal case: fp4 value is (sign, exp_fp4 in 1..3, mant_fp4).
  //   BF16 exp = exp_fp4 + 126  (so fp4 exp=1 -> bf16 exp=127 i.e. 2^0).
  //   BF16 mant = mant_fp4 << 6 (the top bit of the 7-bit bf16 mantissa).
  uint32_t bf16_exp_norm  = exp_fp4 + 126u;
  uint32_t bf16_mant_norm = mant_fp4 ? 0x40u : 0x00u;
  // Subnormal FP4 case (exp_fp4 == 0): value is ±0 (mant_fp4 == 0)
  // or ±0.5 (mant_fp4 == 1).  ±0.5 is representable as a BF16
  // NORMAL value with exp=126, mant=0; ±0 is BF16 +0 / -0.
  uint32_t bf16_exp_sub  = mant_fp4 ? 126u : 0u;
  uint32_t bf16_mant_sub = 0u;
  uint32_t is_sub = (exp_fp4 == 0u) ? 1u : 0u;
  uint32_t bf16_exp  = is_sub ? bf16_exp_sub  : bf16_exp_norm;
  uint32_t bf16_mant = is_sub ? bf16_mant_sub : bf16_mant_norm;
  return static_cast<uint16_t>((sign_bit << 15) | (bf16_exp << 7) | bf16_mant);
}

// Apply an E8M0 scale byte to a BF16 bit pattern.  Matches the
// hardware's `v_cvt_scale_pk8_bf16_fp4` result element-by-element
// for the scale byte selected by scale_sel=0 (the low byte of the
// 32-bit scale register).  Corner cases:
//
//   scale_byte == 0xFF                    -> BF16 qNaN (0x7FC0)
//   fp4_bf16 magnitude == 0               -> preserve ±0 (regardless
//                                            of scale byte; IEEE
//                                            0 × finite = 0)
//   new_exp >= 0xFF                       -> ±Inf (sign preserved)
//   new_exp in [1, 254]                   -> normal BF16 result,
//                                            mantissa and sign preserved
//   new_exp <= 0                          -> subnormal or flush-to-zero;
//                                            shift the (implicit-1).mant
//                                            right by (1 - new_exp);
//                                            zero if the shift makes
//                                            all bits fall off.
__device__ __host__ __forceinline__ uint16_t
apply_e8m0_scale_to_bf16(uint16_t fp4_bf16_bits, uint32_t scale_byte) {
  if (scale_byte == 0xFFu) {
    // NaN scale wins over everything, including ±0, per IEEE
    // 0 × NaN = NaN.  Emit the canonical BF16 qNaN (sign=0, exp=0xFF,
    // mant[6]=1 -> 0x7FC0).
    return static_cast<uint16_t>(0x7FC0);
  }
  uint32_t magnitude = fp4_bf16_bits & 0x7FFFu;
  if (magnitude == 0u) {
    // FP4 ±0 input: output is ±0 unchanged (scale is finite).
    return fp4_bf16_bits;
  }
  uint32_t sign_bits = fp4_bf16_bits & 0x8000u;
  // FP4-derived BF16 values are always normal (exp in 126..129) for
  // non-zero magnitude, per fp4_nibble_to_bf16_bits above.
  uint32_t fp4_exp  = (fp4_bf16_bits >> 7) & 0xFFu;   // 126..129
  uint32_t fp4_mant =  fp4_bf16_bits       & 0x7Fu;   // 0 or 0x40
  // Multiplication by 2^(scale_byte - 127) = exponent add in bf16
  // field space (bf16 bias is 127).  Signed compute because new_exp
  // can underflow into subnormal / zero territory.
  int32_t new_exp_i = static_cast<int32_t>(fp4_exp)
                    + static_cast<int32_t>(scale_byte) - 127;
  if (new_exp_i >= 0xFF) {
    // Saturate to BF16 ±Inf on overflow.  The FP4 values 4 and 6
    // paired with scale_byte near 0xFE can overshoot BF16's exp=0xFE
    // max; BF16 supports Inf, so that is the bit-exact result.
    return static_cast<uint16_t>(sign_bits | 0x7F80u);
  }
  if (new_exp_i >= 1) {
    // Normal BF16 result: sign preserved, mantissa preserved, exp
    // replaced.  This is the hot path for matmul MXFP4 dequant.
    return static_cast<uint16_t>(sign_bits
        | ((static_cast<uint32_t>(new_exp_i) & 0xFFu) << 7)
        | fp4_mant);
  }
  // Underflow: result is BF16 subnormal or zero.  Shift the implicit
  // 1.mant bit pattern right by (1 - new_exp_i) and drop the
  // implicit leading bit past exp=0.
  //
  //   1.mant (8 bits: bit 7 is implicit 1, bits 6..0 are the 7-bit
  //   mantissa) at exp=new_exp_i shifted to align at exp=0 is:
  //     sub_mant = (implicit_1_mant) >> (1 - new_exp_i)
  //   where the shift count is clamped: if >= 8, the result is zero.
  uint32_t implicit_1_mant = 0x80u | fp4_mant;   // 0x80 (1.0) or 0xC0 (1.5)
  int32_t shift_amt = 1 - new_exp_i;             // at least 1
  uint32_t sub_mant = (shift_amt >= 8) ? 0u : (implicit_1_mant >> shift_amt);
  // exp field is 0 for subnormal/zero; mant carries the value.
  return static_cast<uint16_t>(sign_bits | sub_mant);
}

// 128-bit vector typed as a 4-wide uint32.  Used as the output
// stride so the whole BF16 result lands in a single 128-bit store
// (global_store_b128 on gfx1250, four 32-bit stores on gfx942).
// Writing through a uint32* stride avoids the hipcc miscompile we
// observed on gfx1250 when the kernel loops over 8 `__bf16` elements
// and stores them one-by-one: hipcc's byte-pack optimiser collapses
// those 8 stores into a `v_perm_b32` + 3× `v_mov_b32` that broadcasts
// the FIRST BF16 into all 8 lanes instead of keeping them distinct.
// A single-shot `__builtin_nontemporal_store`-free uint32 vector
// store through the 128-bit typed output dodges that optimiser
// entirely; the <8 x __bf16> result is bit-cast to <4 x uint32> and
// the pointer is cast to `uint32*` so the store shape matches.
typedef __attribute__((ext_vector_type(4))) uint32_t v4u32;
typedef __attribute__((ext_vector_type(8))) __bf16  v8bf16;

// Device-only kernel.  The gfx1250 build emits exactly one
// `v_cvt_scale_pk8_bf16_fp4` via the clang builtin (regression-guard
// signature: 1 instance per kernel invocation, pinned by
// `llvm-objdump` in the Makefile / CI).  The gfx942 build takes the
// portable path and computes the same bit pattern element-by-element
// on the CPU-side.
extern "C" __global__ void canary_cvt_scale_pk8_bf16_fp4(
    const InputPair *in_pairs, Packed8BF16 *out) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t packed = in_pairs[tid].packed;
  uint32_t scale  = in_pairs[tid].scale;

#ifdef __gfx1250__
  // Hardware path: gfx1250-only VOP3 primitive.  scale_sel=0 pins
  // the low byte of `scale` as the E8M0 scale byte for all 8
  // nibbles, matching the corpus-observed scope
  // (`_matmul_ogs_{06d912ce88af,0af655e6ea2b}.hsaco` use only
  // scale_sel=0, all 64 instances each).
  v8bf16 r = __builtin_amdgcn_cvt_scale_pk8_bf16_fp4(packed, scale, 0);
  v4u32 r_as_dwords = __builtin_bit_cast(v4u32, r);
  // Single-shot 128-bit store through a `uint32*` — see the comment
  // on `v4u32` above for why the typed-pointer dance matters.
  *reinterpret_cast<v4u32 *>(&out[tid]) = r_as_dwords;
#else
  // Portable path: gfx942 (and any non-gfx1250 target).  Must be
  // bit-identical to the hardware primitive on all (packed, scale)
  // inputs the recipe sweeps — this path is compiled into the
  // gfx942 `.co` that `native` runs against the CPU-reference gold.
  // Same 128-bit-store idiom so the two paths' output layout agrees
  // byte-for-byte (and so a future extension that adds a gfx950
  // target doesn't surface a pointer-layout difference as a real
  // regression).
  uint16_t words[8];
  for (int i = 0; i < 8; ++i) {
    uint32_t nibble = (packed >> (i * 4)) & 0xFu;
    uint16_t fp4_bf = fp4_nibble_to_bf16_bits(nibble);
    words[i] = apply_e8m0_scale_to_bf16(fp4_bf, scale & 0xFFu);
  }
  v4u32 packed_out;
  packed_out[0] = static_cast<uint32_t>(words[0]) |
                  (static_cast<uint32_t>(words[1]) << 16);
  packed_out[1] = static_cast<uint32_t>(words[2]) |
                  (static_cast<uint32_t>(words[3]) << 16);
  packed_out[2] = static_cast<uint32_t>(words[4]) |
                  (static_cast<uint32_t>(words[5]) << 16);
  packed_out[3] = static_cast<uint32_t>(words[6]) |
                  (static_cast<uint32_t>(words[7]) << 16);
  *reinterpret_cast<v4u32 *>(&out[tid]) = packed_out;
#endif
}
