// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_COMMON_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_COMMON_H_

#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/arch/amdgpu/shared/comparison.h"
#include "rocjitsu/isa/arch/amdgpu/shared/division.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "simdojo/components/vector_reg.h"
#include "util/simd.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace rocjitsu::cdna5 {
struct Isa;
}

namespace rocjitsu::amdgpu {

/// Explicit-width alias for IEEE-754 binary32. C++23 has std::float32_t
/// in <stdfloat>; rocjitsu is on C++20 so a local alias.
using float32_t = float;

/// Process-wide, immutable override that disables the SIMD fast path in
/// kernels that have one. Forwards to util::force_scalar() (read once from
/// RJ_FORCE_SCALAR); returned by value, so there is no mutable global to
/// flip at runtime.
inline bool simd_force_scalar() { return util::force_scalar(); }

/// True when a source-selector field names one of the nine inline float
/// constants (0.5 .. 1/(2*pi)). A 32-bit literal is selector 255, so it never
/// matches here even when its value lands in the same range.
inline bool is_inline_float_src(uint32_t selector) { return selector >= 240u && selector <= 248u; }

/// True when a packed 16-bit body has to re-narrow this source itself. An
/// inline float constant reaches a 32-bit source in single precision, so the
/// packed halves would take the low half of e.g. 0x3F800000. A source declared
/// 16 bits wide was already resolved through the half-precision inline table --
/// Operand::read_lane branches on the same `size_bits_ == 16` -- and narrowing
/// that again would read the 16-bit pattern as an f32 denormal and flush it to
/// zero. CDNA2 builds every packed f16 source 16 bits wide, CDNA3 does for
/// v_pk_min_f16 / v_pk_max_f16.
inline bool pk16_src_needs_narrowing(uint32_t selector, int src_size_bits) {
  return is_inline_float_src(selector) && src_size_bits != 16;
}

/// @brief True16 DOT2 replicates integer and floating inline constants into both halves.
inline bool dot2_src_needs_half_replication(uint32_t selector) {
  return (selector >= 128u && selector <= 208u) || is_inline_float_src(selector);
}

/// @brief Return whether floating CLAMP converts a NaN result to positive zero.
/// @details GFX12 and gfx1250 always convert NaN. Earlier profiles require MODE.DX10_CLAMP.
inline bool floating_clamp_nan_to_zero(rj_code_arch_t arch, bool dx10_clamp) {
  return arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5 || dx10_clamp;
}

/// @brief Return the floating NaN CLAMP policy for one wavefront.
inline bool floating_clamp_nan_to_zero(const Wavefront &wf) {
  return floating_clamp_nan_to_zero(wf.cu().arch(), wf.dx10_clamp());
}

/// @brief Clamp one floating result with an explicit NaN conversion policy.
template <typename T> inline T clamp_floating_result(T value, bool clamp_nan_to_zero) {
  static_assert(std::is_floating_point_v<T>);
  using U = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
  constexpr U sign = U(1) << (sizeof(T) * 8 - 1);
  const U bits = std::bit_cast<U>(value);
  const U magnitude = bits & ~sign;
  if (magnitude > std::bit_cast<U>(std::numeric_limits<T>::infinity()))
    return clamp_nan_to_zero ? T(0) : value;
  if ((bits & sign) || magnitude == 0)
    return T(0);
  if (magnitude > std::bit_cast<U>(T(1)))
    return T(1);
  return value;
}

/// @brief Clamp one floating result according to the wave's ISA and MODE state.
template <typename T> inline T clamp_floating_result(T value, const Wavefront &wf) {
  return clamp_floating_result(value, floating_clamp_nan_to_zero(wf));
}

/// @brief Execute VOP3 integer addition, saturating when CLAMP is set.
template <typename T>
inline std::make_unsigned_t<T> vop3_integer_add(std::make_unsigned_t<T> lhs,
                                                std::make_unsigned_t<T> rhs, bool clamp) {
  static_assert(std::is_integral_v<T> && (sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8));
  using U = std::make_unsigned_t<T>;
  const U lhs_bits = lhs;
  const U rhs_bits = rhs;
  if (!clamp)
    return static_cast<U>(lhs_bits + rhs_bits);
  if constexpr (std::is_signed_v<T>) {
    static_assert(sizeof(T) <= 4, "signed 64-bit VOP3 saturation is not implemented");
    using Wide = std::conditional_t<sizeof(T) == 2, int32_t, int64_t>;
    const Wide wide = static_cast<Wide>(std::bit_cast<T>(lhs_bits)) +
                      static_cast<Wide>(std::bit_cast<T>(rhs_bits));
    const Wide saturated = std::clamp(wide, static_cast<Wide>(std::numeric_limits<T>::min()),
                                      static_cast<Wide>(std::numeric_limits<T>::max()));
    return static_cast<U>(saturated);
  } else {
    const U max = std::numeric_limits<U>::max();
    return rhs_bits > max - lhs_bits ? max : static_cast<U>(lhs_bits + rhs_bits);
  }
}

/// @brief Add with intrinsic saturation, then select the maximum or minimum operand.
template <typename T, bool SelectMax>
inline std::make_unsigned_t<T> vop3_integer_add_minmax(std::make_unsigned_t<T> lhs,
                                                       std::make_unsigned_t<T> rhs,
                                                       std::make_unsigned_t<T> bound) {
  static_assert(std::is_integral_v<T> && sizeof(T) == 4);
  using U = std::make_unsigned_t<T>;
  const U sum_bits = vop3_integer_add<T>(lhs, rhs, true);
  if constexpr (std::is_signed_v<T>) {
    const T sum = std::bit_cast<T>(sum_bits);
    const T typed_bound = std::bit_cast<T>(bound);
    return static_cast<U>(SelectMax ? std::max(sum, typed_bound) : std::min(sum, typed_bound));
  } else {
    return SelectMax ? std::max(sum_bits, bound) : std::min(sum_bits, bound);
  }
}

/// @brief Execute an integer multiply-add with an exact intermediate.
template <typename T, unsigned SourceBits>
inline std::make_unsigned_t<T> vop3_integer_mad(uint32_t lhs, uint32_t rhs,
                                                std::make_unsigned_t<T> addend, bool clamp) {
  static_assert(std::is_integral_v<T> && sizeof(T) <= 4);
  static_assert(SourceBits == 16 || SourceBits == 24);
  using U = std::make_unsigned_t<T>;
  if constexpr (std::is_signed_v<T>) {
    const auto extend = [](uint32_t value) -> int64_t {
      constexpr uint32_t shift = 32 - SourceBits;
      return static_cast<int64_t>(static_cast<int32_t>(value << shift) >> shift);
    };
    const int64_t wide = extend(lhs) * extend(rhs) + static_cast<int64_t>(std::bit_cast<T>(addend));
    if (!clamp)
      return static_cast<U>(wide);
    return static_cast<U>(std::clamp(wide, static_cast<int64_t>(std::numeric_limits<T>::min()),
                                     static_cast<int64_t>(std::numeric_limits<T>::max())));
  } else {
    constexpr uint32_t source_mask = (uint32_t{1} << SourceBits) - 1u;
    const uint64_t wide = static_cast<uint64_t>(lhs & source_mask) * (rhs & source_mask) + addend;
    if (clamp && wide > std::numeric_limits<U>::max())
      return std::numeric_limits<U>::max();
    return static_cast<U>(wide);
  }
}

template <typename T, unsigned SourceBits>
inline std::make_unsigned_t<T> vop3_integer_mul(uint32_t lhs, uint32_t rhs, bool clamp) {
  return vop3_integer_mad<T, SourceBits>(lhs, rhs, std::make_unsigned_t<T>{0}, clamp);
}

inline uint32_t vop3_integer_sad_u8(uint32_t lhs, uint32_t rhs, uint32_t addend, bool clamp) {
  uint32_t difference = 0;
  for (unsigned i = 0; i < 4; ++i) {
    const uint32_t a = (lhs >> (i * 8)) & 0xffu;
    const uint32_t b = (rhs >> (i * 8)) & 0xffu;
    difference += a > b ? a - b : b - a;
  }
  return vop3_integer_add<uint32_t>(difference, addend, clamp);
}

/// @brief Execute shifted byte SAD plus accumulation, saturating when CLAMP is set.
inline uint32_t vop3_integer_sad_hi_u8(uint32_t lhs, uint32_t rhs, uint32_t addend, bool clamp) {
  uint32_t difference = 0;
  for (unsigned i = 0; i < 4; ++i) {
    const uint32_t a = (lhs >> (i * 8)) & 0xffu;
    const uint32_t b = (rhs >> (i * 8)) & 0xffu;
    difference += a > b ? a - b : b - a;
  }
  const uint64_t wide = (static_cast<uint64_t>(difference) << 16) + addend;
  if (clamp && wide > UINT32_MAX)
    return UINT32_MAX;
  return static_cast<uint32_t>(wide);
}

inline uint32_t vop3_integer_sad_u16(uint32_t lhs, uint32_t rhs, uint32_t addend, bool clamp) {
  uint32_t difference = 0;
  for (unsigned i = 0; i < 2; ++i) {
    const uint32_t a = (lhs >> (i * 16)) & 0xffffu;
    const uint32_t b = (rhs >> (i * 16)) & 0xffffu;
    difference += a > b ? a - b : b - a;
  }
  return vop3_integer_add<uint32_t>(difference, addend, clamp);
}

inline uint32_t vop3_integer_sad_u32(uint32_t lhs, uint32_t rhs, uint32_t addend, bool clamp) {
  const uint32_t difference = lhs > rhs ? lhs - rhs : rhs - lhs;
  return vop3_integer_add<uint32_t>(difference, addend, clamp);
}

inline uint32_t vop3_integer_msad_u8(uint32_t lhs, uint32_t rhs, uint32_t addend, bool clamp) {
  uint32_t difference = 0;
  for (unsigned i = 0; i < 4; ++i) {
    const uint32_t a = (lhs >> (i * 8)) & 0xffu;
    const uint32_t b = (rhs >> (i * 8)) & 0xffu;
    if (b != 0)
      difference += a > b ? a - b : b - a;
  }
  return vop3_integer_add<uint32_t>(difference, addend, clamp);
}

/// @brief Execute VOP3 integer subtraction, saturating when CLAMP is set.
template <typename T>
inline std::make_unsigned_t<T> vop3_integer_sub(std::make_unsigned_t<T> lhs,
                                                std::make_unsigned_t<T> rhs, bool clamp) {
  static_assert(std::is_integral_v<T> && (sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8));
  using U = std::make_unsigned_t<T>;
  const U lhs_bits = lhs;
  const U rhs_bits = rhs;
  if (!clamp)
    return static_cast<U>(lhs_bits - rhs_bits);
  if constexpr (std::is_signed_v<T>) {
    static_assert(sizeof(T) <= 4, "signed 64-bit VOP3 saturation is not implemented");
    using Wide = std::conditional_t<sizeof(T) == 2, int32_t, int64_t>;
    const Wide wide = static_cast<Wide>(std::bit_cast<T>(lhs_bits)) -
                      static_cast<Wide>(std::bit_cast<T>(rhs_bits));
    const Wide saturated = std::clamp(wide, static_cast<Wide>(std::numeric_limits<T>::min()),
                                      static_cast<Wide>(std::numeric_limits<T>::max()));
    return static_cast<U>(saturated);
  } else {
    return lhs_bits < rhs_bits ? U{0} : static_cast<U>(lhs_bits - rhs_bits);
  }
}

/// @brief Return whether V_DOT4 integer instructions honor the encoded CLAMP bit.
/// @details The RDNA4 ISA manual says V_DOT4_I32_IU8 and V_DOT4_U32_U8 ignore
/// CLAMP, but experiments with those instructions on gfx1201 hardware show
/// saturation when CLAMP is set. The CDNA5-backed gfx1250 profile still ignores
/// it. Keep this policy shared by scalar and SIMD execution.
inline bool dot4_clamp_supported(const Wavefront &wf) {
  const rj_code_arch_t arch = wf.cu().arch();
  return arch != ROCJITSU_CODE_ARCH_CDNA5;
}

inline uint32_t sign_extend_u32(uint32_t value, unsigned bits) {
  assert(bits >= 1 && bits <= 32 && "sign_extend_u32 requires a 1..32 bit width");
  const uint32_t sign = uint32_t{1} << (bits - 1);
  const uint32_t mask = bits == 32 ? ~uint32_t{0} : ((uint32_t{1} << bits) - uint32_t{1});
  return ((value & mask) ^ sign) - sign;
}

inline uint32_t lshl_masked(uint32_t value, uint32_t count) { return value << (count & 31u); }

inline uint64_t lshl_masked(uint64_t value, uint64_t count) { return value << (count & 63u); }

inline uint32_t bfm_b32(uint32_t width, uint32_t offset) {
  const uint32_t w = width & 31u;
  const uint32_t off = offset & 31u;
  return w == 0 ? uint32_t{0} : ((uint32_t{1} << w) - uint32_t{1}) << off;
}

inline uint32_t mul_i24_u32(uint32_t lhs, uint32_t rhs) {
  return sign_extend_u32(lhs, 24) * sign_extend_u32(rhs, 24);
}

inline uint32_t mad_i24_u32(uint32_t lhs, uint32_t rhs, uint32_t addend) {
  return mul_i24_u32(lhs, rhs) + addend;
}

inline uint32_t mad_lo_u16(uint32_t lhs, uint32_t rhs, uint32_t addend) {
  const uint32_t a = lhs & 0xffffu;
  const uint32_t b = rhs & 0xffffu;
  const uint32_t c = addend & 0xffffu;
  return (a * b + c) & 0xffffu;
}

/// Return whether signed integer add/sub overflows. The unsigned parameter type
/// is deduced, so the same helper serves 32- and 64-bit scalar add/sub.
template <typename U> inline bool signed_add_overflows(U a, U b) {
  static_assert(std::is_unsigned_v<U>, "operands must be unsigned");
  const U sum = static_cast<U>(a + b);
  constexpr U sign_bit = U{1} << (sizeof(U) * 8 - 1);
  // Overflow iff the operands share a sign that differs from the result's.
  return ((a ^ sum) & (b ^ sum) & sign_bit) != 0;
}

template <typename U> inline bool signed_sub_overflows(U a, U b) {
  static_assert(std::is_unsigned_v<U>, "operands must be unsigned");
  const U diff = static_cast<U>(a - b);
  constexpr U sign_bit = U{1} << (sizeof(U) * 8 - 1);
  // Overflow iff the operands differ in sign and a's sign differs from the
  // result's.
  return ((a ^ b) & (a ^ diff) & sign_bit) != 0;
}

/// Write an explicit SGPR lane-mask destination. Wave32 targets use the low
/// dword only; writing a pair would clobber the next SGPR, which codegen may
/// legally use for unrelated scalar state.
template <typename Operand>
inline void write_explicit_lane_mask(const Operand &dst, Wavefront &wf, uint64_t mask) {
  if (wf.wf_size() <= 32)
    amdgpu::RegisterAccess(wf).write_scalar(dst, static_cast<uint32_t>(mask));
  else
    amdgpu::RegisterAccess(wf).write_scalar64(dst, mask);
}

inline void write_explicit_lane_mask(uint32_t physical_dst, Wavefront &wf, uint64_t mask) {
  amdgpu::RegisterAccess regs(wf);
  if (wf.wf_size() <= 32)
    regs.write_sgpr(physical_dst, static_cast<uint32_t>(mask));
  else
    regs.write_sgpr64(physical_dst, mask);
}

template <typename Op>
inline void write_wave_mask_scalar(const Op &op, Wavefront &wf, uint64_t mask) {
  write_explicit_lane_mask(op, wf, mask);
}

/// @brief Read a wave-sized mask from an explicit scalar-register operand.
/// @details Reads one SGPR for a Wave32 wavefront and an SGPR pair for Wave64.
template <typename Op> inline uint64_t read_wave_mask_scalar(const Op &op, Wavefront &wf) {
  RegisterAccess regs(wf);
  return wf.wf_size() <= 32 ? static_cast<uint64_t>(regs.read_scalar(op)) : regs.read_scalar64(op);
}

template <typename T>
inline T apply_vop3_b32_src_mod(T value, uint32_t abs, uint32_t neg, uint32_t src_idx) {
  if (abs & (1u << src_idx))
    value &= T(0x7fffffffu);
  if (neg & (1u << src_idx))
    value ^= T(0x80000000u);
  return value;
}

template <typename Inst> inline bool vop3_fp8_decode_e5m3(const Inst &inst) {
  if constexpr (requires { typename Inst::IsaType; }) {
    if constexpr (std::is_same_v<typename Inst::IsaType, ::rocjitsu::cdna5::Isa> &&
                  requires { inst.inst_.clamp; })
      return inst.inst_.clamp;
  }
  return false;
}

template <typename Operand>
inline uint32_t read_vop3_true16_src(const Operand &src, Wavefront &wf, uint32_t lane,
                                     uint32_t opsel, uint32_t src_idx) {
  uint32_t value = amdgpu::RegisterAccess(wf).read_lane(src, lane);
  if (opsel & (1u << src_idx))
    value >>= 16;
  return value & 0xffffu;
}

inline bool cdna_vop3_low_dst_zeroes_high(const Wavefront &wf) {
  switch (wf.cu().arch()) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return true;
  default:
    return false;
  }
}

template <typename Operand>
inline void write_vop3_true16_dst(const Operand &dst, Wavefront &wf, uint32_t lane, uint32_t opsel,
                                  uint32_t value, bool cdna_low_dst_zeroes_high = false) {
  uint32_t src_half = value & 0xffffu;
  const bool dst_hi = (opsel & 0x8u) != 0;
  const bool low_dst_zeroes_high =
      !dst_hi && cdna_low_dst_zeroes_high && cdna_vop3_low_dst_zeroes_high(wf);
  auto reg = dst.to_register_ref();
  if (reg && reg->cls == RegClass::VGPR) {
    // Real VOP3 OP_SEL true16 destinations select the destination half with
    // op_sel[3]. CDNA low-half OP_SEL writes zero the upper half; fixed
    // MIXLO/MIXHI-style half writes and RDNA/gfx true16 writes preserve it.
    uint32_t off = reg->index + (wf.vgpr_msb_for_role(dst.vgpr_msb_role()) << 8);
    uint32_t voff = wf.gpr_idx_en() ? apply_gpr_idx(wf, off, dst.vgpr_msb_role()) : off;
    uint32_t idx = wf.vgpr_alloc().base + voff;
    RegisterAccess regs(wf);
    auto dst_region = regs.readwrite_vgpr_region(idx, 1, uint64_t{1} << lane);
    uint32_t old_dst = dst_region.read().lane(0, lane);
    uint32_t merged = dst_hi
                          ? ((old_dst & 0x0000ffffu) | (src_half << 16))
                          : (low_dst_zeroes_high ? src_half : ((old_dst & 0xffff0000u) | src_half));
    dst_region.write().set_lane(0, lane, merged);
    return;
  }
  amdgpu::RegisterAccess(wf).write_lane(dst, lane, dst_hi ? (src_half << 16) : src_half);
}

inline uint32_t effective_vop3_omod_f32(const Wavefront &wf, uint32_t omod,
                                        bool force_output_flush = false) {
  return fp_mode::effective_omod(wf.cu().arch(), force_output_flush ? 0 : wf.fp_denorm_mode_f32(),
                                 wf.ieee_mode(), omod);
}

inline uint32_t effective_vop3_omod_f16(const Wavefront &wf, uint32_t omod) {
  return fp_mode::effective_f16_omod(wf.cu().arch(), wf.fp_denorm_mode_f16_f64(), wf.ieee_mode(),
                                     false, omod);
}

inline uint32_t effective_vop3_omod_f64(const Wavefront &wf, uint32_t omod) {
  return fp_mode::effective_omod(wf.cu().arch(), wf.fp_denorm_mode_f16_f64(), wf.ieee_mode(), omod);
}

/// Per-half f32 reader for the packed-f32 VOP3P family (v_pk_add/mul/fma_f32).
/// In a VGPR pair {N, N+1}, register N holds the LO f32 of every lane and N+1
/// the HI f32, so each half is a native-width native<float> read of one
/// register (no 64-bit-lane / narrow32 detour). The pair view carries the
/// architecture's scalar-source policy: CDNA5 replicates one scalar DWORD;
/// earlier CDNA preserves scalar register pairs.
struct PkF32Halves {
  util::native<float> lo;
  util::native<float> hi;
};

/// Result of a carry-bearing VOP2 functor: the 32-bit per-lane result and the
/// per-lane carry/borrow as a `simd_mask`. A class template (not a fixed type)
/// so it never names `native<uint32_t>::mask_type` outside the SIMD build —
/// the carry functors below build it through `make_simd_carry`, whose return
/// type is deduced and only instantiated on the constrained code path.
template <typename Value, typename Mask> struct SimdCarry {
  Value value;
  Mask carry;
};

/// Deduce-and-wrap helper for the carry functors. Keeps each functor a single
/// expression while leaving the mask type implicit.
template <typename Value, typename Mask>
SimdCarry<Value, Mask> make_simd_carry(Value value, Mask carry) {
  return {value, carry};
}

enum class F16Vop2FmaShape { AddLiteral, MultiplyLiteral, Accumulate };

/// Both VOP3P encoding families use the same selector bits under different names.
template <typename Encoding> inline uint32_t packed_opsel(const Encoding &enc) {
  if constexpr (requires { enc.opsel; })
    return enc.opsel;
  else
    return enc.op_sel;
}

template <typename Encoding> inline uint32_t packed_opsel_hi(const Encoding &enc) {
  if constexpr (requires { enc.opsel_hi; })
    return enc.opsel_hi;
  else
    return enc.op_sel_hi;
}

template <typename Encoding> inline uint32_t packed_opsel_hi_2(const Encoding &enc) {
  if constexpr (requires { enc.opsel_hi_2; })
    return enc.opsel_hi_2;
  else
    return enc.op_sel_hi_2;
}

template <typename V>
inline V select_packed_halves(V value, uint32_t low, uint32_t high, unsigned operand) {
  return ((value >> (((low >> operand) & 1u) * 16)) & V(0xffffu)) |
         ((value >> (((high >> operand) & 1u) * 16)) << 16);
}

/// Destination shape for the VOP3P fma_mix / mad_mix family. F32 writes a
/// full 32-bit float into vdst; F16_LO writes the f16-narrowed result into
/// the low half of vdst (high half preserved); F16_HI writes it into the
/// high half (low half preserved). Selected by the per-mnemonic glue probe.
enum class FmaMixDst { F32, F16_LO, F16_HI };

enum class PackedFloatOp { ADD, MUL, FMA, MIN, MAX, MINIMUM, MAXIMUM };

/// Whether a VOP3P dot widens its 16-bit float halves as F16 or BF16. The two
/// formats share the entire dot2 structure (op_sel packing and neg/neg_hi,
/// left-to-right accumulate) and differ only in how each half is widened to f32.
enum class Vop3pDotHalfFormat { F16, BF16 };

/// Apply a three-input truth table to all bits of a scalar or SIMD word.
template <typename U> inline U bitop3_words(U a, U b, U c, uint8_t table) {
  const auto choose = [](U mask, U yes, U no) { return (mask & yes) | (~mask & no); };
  const auto bit = [table](unsigned i) { return U(0) - U((table >> i) & 1u); };
  return choose(a, choose(b, choose(c, bit(7), bit(6)), choose(c, bit(5), bit(4))),
                choose(b, choose(c, bit(3), bit(2)), choose(c, bit(1), bit(0))));
}

// Source operand kinds admit both scalar and vector selectors. is_vgpr()
// describes that capability, not whether this particular selector names a VGPR.
template <typename Op> inline bool e32_packed_vgpr_operand(const Op &op) {
  using Kind = decltype(op.opr_type_);
  return op.size_bits() == 16 && !op.delegate() &&
         (op.opr_type_ == Kind::OPR_VGPR || (op.is_vgpr() && op.encoding_value() >= 256));
}

/// e32 true16 selectors encode the half in VGPR index bit 7. Construct a
/// full-word operand for the same physical register; keep staged DPP operands
/// intact because their half selection has already been applied.
template <typename Op> inline Op e32_word_operand(const Op &op) {
  if (!e32_packed_vgpr_operand(op))
    return op;
  using Kind = decltype(op.opr_type_);
  const IsaExecutionBackend backend{.operand_backend = Op::full_execution_backend()};
  ScopedIsaExecutionBackend scope(&backend);
  Op word(32, Kind::OPR_VGPR, op.encoding_value() & 0x7f);
  word.set_vgpr_msb_role(op.vgpr_msb_role());
  return word;
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_COMMON_H_
