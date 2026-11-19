// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Register selection, source/destination modifiers and lane-mask bindings
// shared by generated instruction handlers and VM helpers.

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_SIMD_GLUE_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_SIMD_GLUE_H_

#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/arch/amdgpu/shared/comparison.h"
#include "rocjitsu/isa/arch/amdgpu/shared/denormal_value.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/floating_operation.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fract.h"
#include "rocjitsu/isa/arch/amdgpu/shared/input_denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/arch/amdgpu/shared/ldexp.h"
#include "rocjitsu/isa/arch/amdgpu/shared/minmax.h"
#include "rocjitsu/isa/arch/amdgpu/shared/output_denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/output_modifier.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_static_resolve.h"
#include "rocjitsu/isa/arch/amdgpu/shared/source_modifier.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "simdojo/components/vector_reg.h"
#include "util/host_target.h"
#include "util/simd.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>

namespace rocjitsu::cdna5 {
struct Isa;
} // namespace rocjitsu::cdna5

namespace rocjitsu {
namespace amdgpu {

/// Explicit-width alias for IEEE-754 binary32. C++23 has std::float32_t
/// in <stdfloat>; rocjitsu is on C++20 so a local alias.
using float32_t = float;

/// Check the CU policy and the module-local RJ_FORCE_SCALAR override.
inline bool simd_force_scalar(const Wavefront &wf) { return wf.cu().force_scalar(); }

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
inline bool floating_clamp_nan_to_zero(rocjitsu::IsaFamily arch, bool dx10_clamp) {
  return arch == rocjitsu::IsaFamily::Rdna4 || arch == rocjitsu::IsaFamily::Cdna5 || dx10_clamp;
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
  if (magnitude > std::bit_cast<U>(std::numeric_limits<T>::infinity())) {
    return clamp_nan_to_zero ? T(0) : value;
  }
  if ((bits & sign) || magnitude == 0) {
    return T(0);
  }
  if (magnitude > std::bit_cast<U>(T(1))) {
    return T(1);
  }
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
  if (!clamp) {
    return static_cast<U>(lhs_bits + rhs_bits);
  }
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

/// @brief Add with saturation, select the maximum or minimum, then apply signed ReLU if enabled.
template <typename T, bool SelectMax>
inline std::make_unsigned_t<T> vop3_integer_add_minmax(std::make_unsigned_t<T> lhs,
                                                       std::make_unsigned_t<T> rhs,
                                                       std::make_unsigned_t<T> bound, bool clamp) {
  static_assert(std::is_integral_v<T> && sizeof(T) == 4);
  using U = std::make_unsigned_t<T>;
  const U sum_bits = vop3_integer_add<T>(lhs, rhs, true);
  if constexpr (std::is_signed_v<T>) {
    const T sum = std::bit_cast<T>(sum_bits);
    const T typed_bound = std::bit_cast<T>(bound);
    const T result = SelectMax ? std::max(sum, typed_bound) : std::min(sum, typed_bound);
    return static_cast<U>(clamp ? std::max<T>(result, 0) : result);
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
    if (!clamp) {
      return static_cast<U>(wide);
    }
    return static_cast<U>(std::clamp(wide, static_cast<int64_t>(std::numeric_limits<T>::min()),
                                     static_cast<int64_t>(std::numeric_limits<T>::max())));
  } else {
    constexpr uint32_t source_mask = (uint32_t{1} << SourceBits) - 1u;
    const uint64_t wide = static_cast<uint64_t>(lhs & source_mask) * (rhs & source_mask) + addend;
    if (clamp && wide > std::numeric_limits<U>::max()) {
      return std::numeric_limits<U>::max();
    }
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
  if (clamp && wide > UINT32_MAX) {
    return UINT32_MAX;
  }
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
    if (b != 0) {
      difference += a > b ? a - b : b - a;
    }
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
  if (!clamp) {
    return static_cast<U>(lhs_bits - rhs_bits);
  }
  if constexpr (std::is_signed_v<T>) {
    static_assert(sizeof(T) <= 4, "signed 64-bit VOP3 saturation is not implemented");
    using Wide = std::conditional_t<sizeof(T) == 2, int32_t, int64_t>;
    const Wide wide = static_cast<Wide>(std::bit_cast<T>(lhs_bits)) -
                      static_cast<Wide>(std::bit_cast<T>(rhs_bits));
    const Wide saturated = std::clamp(wide, static_cast<Wide>(std::numeric_limits<T>::min()),
                                      static_cast<Wide>(std::numeric_limits<T>::max()));
    return static_cast<U>(saturated);
  } else {
    return static_cast<U>(std::max(lhs_bits, rhs_bits) - rhs_bits);
  }
}

inline uint32_t sign_extend_u32(uint32_t value, unsigned bits) {
  assert(bits >= 1 && bits <= 32 && "sign_extend_u32 requires a 1..32 bit width");
  const uint32_t sign = uint32_t{1} << (bits - 1);
  const uint32_t mask = bits == 32 ? ~uint32_t{0} : ((uint32_t{1} << bits) - uint32_t{1});
  return ((value & mask) ^ sign) - sign;
}

inline uint32_t lshl_masked(uint32_t value, uint32_t count) { return value << (count & 31u); }

inline uint64_t lshl_masked(uint64_t value, uint64_t count) { return value << (count & 63u); }

// V_LSHL_ADD_U64 uses S1[2:0]. Encoded shifts 5 through 7 select a shift of zero.
inline uint64_t lshl_for_add_u64(uint64_t value, uint64_t count) {
  const uint64_t shift = count & 7u;
  return value << (shift <= 4u ? shift : 0u);
}

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
  if (wf.wf_size() <= 32) {
    amdgpu::RegisterAccess(wf).write_scalar(dst, static_cast<uint32_t>(mask));
  } else {
    amdgpu::RegisterAccess(wf).write_scalar64(dst, mask);
  }
}

inline void write_explicit_lane_mask(uint32_t physical_dst, Wavefront &wf, uint64_t mask) {
  amdgpu::RegisterAccess regs(wf);
  if (wf.wf_size() <= 32) {
    if (regs.write_sgpr(physical_dst, static_cast<uint32_t>(mask)).failed()) {
      return;
    }
  } else {
    if (regs.write_sgpr64(physical_dst, mask).failed()) {
      return;
    }
  }
}

inline util::native<uint32_t> simd_sign_extend_u32(util::native<uint32_t> v, unsigned bits) {
  assert(bits >= 1 && bits <= 32 && "simd_sign_extend_u32 requires a 1..32 bit width");
  const uint32_t sign = uint32_t{1} << (bits - 1);
  const uint32_t mask = bits == 32 ? ~uint32_t{0} : ((uint32_t{1} << bits) - uint32_t{1});
  return ((v & mask) ^ sign) - sign;
}

inline util::native<uint32_t> simd_lshl_u32(util::native<uint32_t> v, util::native<uint32_t> sh) {
  return util::map_native_scalar<uint32_t>(
      v, sh, [](uint32_t value, uint32_t count) { return value << (count & 31u); });
}

inline util::native<uint32_t> simd_lshr_u32(util::native<uint32_t> v, util::native<uint32_t> sh) {
  return util::map_native_scalar<uint32_t>(
      v, sh, [](uint32_t value, uint32_t count) { return value >> (count & 31u); });
}

/// @brief Convert f32 to a zero-extended packed i16 lane.
/// @details NaN converts to zero and out-of-range values saturate before the
/// signed 16-bit result is returned in the low half of a uint32_t.
inline uint32_t cvt_i16_f32(float32_t value) {
  if (std::isnan(value)) {
    return uint32_t{0};
  }
  if (value >= 32768.0f) {
    return static_cast<uint32_t>(static_cast<uint16_t>(std::numeric_limits<int16_t>::max()));
  }
  if (value < -32768.0f) {
    return static_cast<uint32_t>(static_cast<uint16_t>(std::numeric_limits<int16_t>::min()));
  }
  return static_cast<uint16_t>(static_cast<int16_t>(value));
}

/// @brief Convert f32 to a zero-extended packed u16 lane.
/// @details NaN and negative inputs convert to zero; values above the u16 range
/// saturate to UINT16_MAX.
inline uint32_t cvt_u16_f32(float32_t value) {
  if (std::isnan(value) || value < 0.0f) {
    return uint32_t{0};
  }
  if (value >= 65536.0f) {
    return static_cast<uint32_t>(std::numeric_limits<uint16_t>::max());
  }
  return static_cast<uint16_t>(value);
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

template <typename Inst> inline bool vop3_fp8_decode_e5m3(const Inst &inst) {
  if constexpr (requires { typename Inst::IsaType; }) {
    if constexpr (std::is_same_v<typename Inst::IsaType, ::rocjitsu::cdna5::Isa> &&
                  requires { inst.inst_.clamp; }) {
      return inst.inst_.clamp;
    }
  }
  return false;
}

// ALIGNBIT/BYTE select a byte of the count on GFX9/10 and a half on GFX11+.
inline uint32_t vop3_align_count_offset(uint32_t opsel, int count_bits) {
  assert(count_bits == 8 || count_bits == 16);
  return count_bits == 8 ? 8u * (opsel & 3u) : 16u * ((opsel >> 2) & 1u);
}

template <typename Operand>
inline uint32_t read_vop3_true16_src(const Operand &src, Wavefront &wf, uint32_t lane,
                                     uint32_t opsel, uint32_t src_idx) {
  uint32_t value = amdgpu::RegisterAccess(wf).read_lane(src, lane);
  if (opsel & (1u << src_idx)) {
    value >>= 16;
  }
  return value & 0xffffu;
}

inline util::native<uint32_t> select_vop3_true16_src(util::native<uint32_t> value, uint32_t opsel,
                                                     uint32_t src_idx) {
  if (opsel & (1u << src_idx)) {
    value >>= 16;
  }
  return value & util::broadcast<uint32_t>(0xffffu);
}

inline bool cdna_vop3_low_dst_zeroes_high(const Wavefront &wf) {
  switch (wf.cu().arch()) {
  case rocjitsu::IsaFamily::Cdna1:
  case rocjitsu::IsaFamily::Cdna2:
  case rocjitsu::IsaFamily::Cdna3:
  case rocjitsu::IsaFamily::Cdna4:
    return true;
  default:
    return false;
  }
}

inline uint8_t vop3_true16_dst_byte_mask(const Wavefront &wf, const rocjitsu::Operand &dst,
                                         bool dst_hi, bool cdna_low_dst_zeroes_high) {
  if (dst_hi) {
    return 0xc;
  }
  if (cdna_low_dst_zeroes_high && cdna_vop3_low_dst_zeroes_high(wf) &&
      (!dst.preserve_unwritten_bytes() || dst.register_byte_mask() == 0xf)) {
    return 0xf;
  }
  return 0x3;
}

template <typename Operand>
inline void write_vop3_true16_dst(const Operand &dst, Wavefront &wf, uint32_t lane, uint32_t opsel,
                                  uint32_t value, bool cdna_low_dst_zeroes_high = false) {
  uint32_t src_half = value & 0xffffu;
  const bool dst_hi = (opsel & 0x8u) != 0;
  auto reg = dst.to_register_ref();
  if (reg && reg->cls == RegClass::VGPR) {
    // OP_SEL[3] selects the destination half. On CDNA1–4, ordinary MAD,
    // FMA_F16 and DIV_FIXUP_F16 preserve the other half. Their legacy forms
    // clear the upper half on a low-half write.
    uint32_t off = reg->index + (wf.vgpr_msb_for_role(dst.vgpr_msb_role()) << 8);
    uint32_t voff = wf.gpr_idx_en() ? apply_gpr_idx(wf, off, dst.vgpr_msb_role()) : off;
    uint32_t idx = wf.vgpr_alloc().base + voff;
    RegisterAccess regs(wf);
    const uint8_t bytes = vop3_true16_dst_byte_mask(wf, dst, dst_hi, cdna_low_dst_zeroes_high);
    regs.write_vgpr_region(idx, 1, uint64_t{1} << lane, bytes)
        .set_lane(0, lane, dst_hi ? src_half << 16 : src_half);
    return;
  }
  amdgpu::RegisterAccess(wf).write_lane(dst, lane, dst_hi ? (src_half << 16) : src_half);
}

/// @brief Apply the shared ABS/NEG rules to floating-point SIMD lanes.
/// @details Only the representation changes around the raw-bit helper.
template <unsigned SrcIdx>
util::native<float> apply_vop3_src_mod_f32(util::native<float> value, uint32_t abs, uint32_t neg) {
  const auto bits = std::bit_cast<util::native<uint32_t>>(value);
  return std::bit_cast<util::native<float>>(
      source_modifier::apply<fp_format::F32>(bits, SrcIdx, abs, neg));
}

template <unsigned SrcIdx>
util::native<double> apply_vop3_src_mod_f64(util::native<double> value, uint32_t abs,
                                            uint32_t neg) {
  const auto bits = std::bit_cast<util::native<uint64_t>>(value);
  return std::bit_cast<util::native<double>>(
      source_modifier::apply<fp_format::F64>(bits, SrcIdx, abs, neg));
}

/// @brief Apply OMOD, then CLAMP, to host floating-point SIMD lanes.
/// @details Live helper outside the shared raw-bit stage; see output_modifier.h.
/// The caller establishes rounding and resolves effective OMOD and NaN clamping.
/// Power-of-two scaling is exact for normal finite results except at format limits.
template <typename T>
util::native<T> apply_vop3_dst_mod(util::native<T> v, uint32_t omod, uint32_t clamp,
                                   bool clamp_nan_to_zero) {
  if constexpr (std::is_same_v<T, float>) {
    if (omod != 0) {
      using U = util::native<uint32_t>;
      U bits = std::bit_cast<U>(v);
      util::stdx::where((bits & U(0x7fffffffu)) < U(0x00800000u), bits) = U(0);
      v = std::bit_cast<util::native<float>>(bits);
    }
  }
  const auto unscaled = v;
  if (omod == 1) {
    v = v * T(2);
  } else if (omod == 2) {
    v = v * T(4);
  } else if (omod == 3) {
    v = v * T(0.5);
  }
  if constexpr (std::is_same_v<T, float>) {
    using U = util::native<uint32_t>;
    U bits = std::bit_cast<U>(v);
    const U original = std::bit_cast<U>(unscaled);
    util::stdx::where((original & U(0x7fffffffu)) > U(0x7f800000u), bits) = original;
    if (omod == 3) {
      util::stdx::where((original & U(0x7fffffffu)) < U(0x01000000u), bits) =
          original & U(0x80000000u);
    }
    v = std::bit_cast<util::native<float>>(bits);
  }
  if (clamp) {
#if RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS
    if constexpr (std::is_same_v<T, double>) {
      v = util::map_native64_scalar<double>(v, [clamp_nan_to_zero](double x) {
        if (std::isnan(x)) {
          return clamp_nan_to_zero ? 0.0 : x;
        }
        if (x <= 0.0) {
          return 0.0;
        }
        if (x > 1.0) {
          return 1.0;
        }
        return x;
      });
    } else
#endif
    {
      if (clamp_nan_to_zero) {
        util::stdx::where(util::stdx::isnan(v), v) = T(0);
      }
      util::stdx::where(v <= T(0), v) = T(0);
      util::stdx::where(v > T(1), v) = T(1);
    }
  }
  if (omod != 0) {
    if constexpr (std::is_same_v<T, float>) {
      using U = util::native<uint32_t>;
      U bits = std::bit_cast<U>(v);
      const auto subnormal = ((bits & U(0x7f800000u)) == U(0)) && ((bits & U(0x007fffffu)) != U(0));
      // Existing zeros were canonicalized before scaling. Preserve the sign
      // of a newly underflowed result.
      util::stdx::where(subnormal, bits) = bits & U(0x80000000u);
      v = std::bit_cast<util::native<float>>(bits);
    } else {
      using U = util::native<uint64_t>;
      U bits = std::bit_cast<U>(v);
#if RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS
      bits = util::map_native64_scalar<uint64_t>(bits, [](uint64_t lane) {
        const bool subnormal =
            (lane & 0x7ff0000000000000ULL) == 0 && (lane & 0x000fffffffffffffULL) != 0;
        if (subnormal) {
          lane &= 0x8000000000000000ULL;
        }
        return (lane & 0x7fffffffffffffffULL) == 0 ? 0 : lane;
      });
#else
      const auto subnormal = ((bits & U(0x7ff0000000000000ULL)) == U(0)) &&
                             ((bits & U(0x000fffffffffffffULL)) != U(0));
      util::stdx::where(subnormal, bits) = bits & U(0x8000000000000000ULL);
      util::stdx::where((bits & U(0x7fffffffffffffffULL)) == U(0), bits) = U(0);
#endif
      v = std::bit_cast<util::native<double>>(bits);
    }
  }
  return v;
}

inline util::native<double> apply_vop3_dst_mod_f64(util::native<double> v, uint32_t omod,
                                                   uint32_t clamp, bool clamp_nan_to_zero) {
  return apply_vop3_dst_mod<double>(v, omod, clamp, clamp_nan_to_zero);
}

/// @brief Apply F32 OMOD, then CLAMP, through the shared raw-bit output stage.
/// @details Scaling and clamping use only integer operations, so neither host
/// rounding nor host exception state affects or observes them. Overflow rounds by
/// `round_mode` (MODE.FP_ROUND); TRANS callers pass nearest-even.
inline util::native<float> apply_vop3_dst_mod_f32(util::native<float> v, uint32_t omod,
                                                  uint32_t clamp, bool clamp_nan_to_zero,
                                                  uint32_t round_mode) {
  if (omod == 0 && clamp == 0) {
    return v;
  }
  output_modifier::Policy policy;
  policy.omod = omod;
  policy.clamp = clamp != 0;
  policy.clamp_nan_to_zero = clamp_nan_to_zero;
  policy.round_mode = round_mode;
  using U = util::native<uint32_t>;
  return std::bit_cast<util::native<float>>(
      output_modifier::apply<fp_format::F32>(std::bit_cast<U>(v), policy));
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

/// @brief Resolve instruction fields and wave MODE into an output policy for `Fmt`.
/// @details Architecture rules decide whether OMOD is active and CLAMP clears NaNs.
/// Callers of output_modifier::apply supply result bits already rounded to `Fmt`.
/// See output_modifier.h for migration scope and hardware evidence per target.
/// @param omod VOP3 OMOD field.
/// @param clamp VOP3 CLAMP field.
template <typename Fmt>
inline output_modifier::Policy output_modifier_policy(const Wavefront &wf, uint32_t omod,
                                                      uint32_t clamp) {
  output_modifier::Policy policy;
  static_assert(std::is_same_v<Fmt, fp_format::F16> || std::is_same_v<Fmt, fp_format::F32> ||
                    std::is_same_v<Fmt, fp_format::F64>,
                "output modifiers are defined for F16, F32 and F64 results");
  if constexpr (std::is_same_v<Fmt, fp_format::F32>) {
    policy.omod = effective_vop3_omod_f32(wf, omod);
    policy.round_mode = wf.fp_round_mode_f32();
  } else if constexpr (std::is_same_v<Fmt, fp_format::F16>) {
    policy.omod = effective_vop3_omod_f16(wf, omod);
    policy.round_mode = wf.fp_round_mode_f16_f64();
  } else {
    policy.omod = effective_vop3_omod_f64(wf, omod);
    policy.round_mode = wf.fp_round_mode_f16_f64();
  }
  policy.clamp = clamp != 0;
  policy.clamp_nan_to_zero = floating_clamp_nan_to_zero(wf);
  policy.fp16_ovfl = wf.fp16_ovfl();
  return policy;
}

/// @brief Output policy for a transcendental-unit result.
/// @details The TRANS unit rounds OMOD overflow to nearest in every MODE, so
/// overflow gives infinity unless FP16_OVFL saturates an F16 result. gfx1201
/// captures show infinity under round-toward-zero for F16 and F32 results.
/// ISA discrepancy: the ISA expects OMOD overflow to follow the MODE round
/// mode, but transcendental results overflow to infinity in every mode, as
/// gfx1201 does. This policy therefore forces `round_mode = 0`.
template <typename Fmt>
inline output_modifier::Policy
transcendental_output_modifier_policy(const Wavefront &wf, uint32_t omod, uint32_t clamp) {
  output_modifier::Policy policy = output_modifier_policy<Fmt>(wf, omod, clamp);
  policy.round_mode = 0;
  return policy;
}

/// @brief Flush MODE input denormals in the source, then run a unary operation.
/// @details Floating lanes are flushed on their encoding, so host DAZ cannot
/// interfere. F16 sources must still be raw half encodings: a subnormal half
/// becomes a normal F32 once widened.
template <typename Fmt, typename Op>
inline auto flush_input_then(input_denormal::Policy policy, Op operation) {
  return [policy, operation](auto value) {
    using V = decltype(value);
    if constexpr (std::is_floating_point_v<typename V::value_type>) {
      using Bits = util::native<typename Fmt::Lane>;
      const Bits bits = std::bit_cast<Bits>(value);
      return operation(std::bit_cast<V>(input_denormal::flush_input<Fmt>(bits, policy)));
    } else {
      return operation(input_denormal::flush_input<Fmt>(value, policy));
    }
  };
}

/// @brief Wrap a raw-bit operation with the instruction's source and output modifiers.
/// @details Resolve MODE once, before the SIMD lane loop. The operation itself
/// only needs to implement input flushing and produce destination-format bits.
/// @param floating_sources Bit i set when source i is floating and takes ABS/NEG;
/// LDEXP's integer exponent does not.
template <typename Fmt, typename Inst, typename Op>
inline auto vop3_float_operation(const Inst &inst, const Wavefront &wf, Op operation,
                                 uint32_t floating_sources = ~0u) {
  return floating_operation::WithModifiers<Fmt, Op>{
      {inst.inst_.abs & floating_sources, inst.inst_.neg & floating_sources},
      output_modifier_policy<Fmt>(wf, inst.inst_.omod, inst.inst_.clamp),
      operation};
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
  if (!e32_packed_vgpr_operand(op)) {
    return op;
  }
  using Kind = decltype(op.opr_type_);
  const IsaExecutionBackend backend{.operand_backend = Op::full_execution_backend()};
  ScopedIsaExecutionBackend scope(&backend);
  Op word(32, Kind::OPR_VGPR, op.encoding_value() & 0x7f);
  word.set_vgpr_msb_role(op.vgpr_msb_role());
  return word;
}

// gfx1100 and gfx1201 apply NEG_LO/NEG_HI to the selected integer source halves.
// RDNA3.5 follows RDNA3; CDNA5 follows RDNA4 for their common packed operations.
template <typename Encoding>
inline uint32_t packed_integer_negations(rocjitsu::IsaFamily arch, const Encoding &enc) {
  if (arch == rocjitsu::IsaFamily::Rdna3 || arch == rocjitsu::IsaFamily::Rdna3_5 ||
      arch == rocjitsu::IsaFamily::Rdna4 || arch == rocjitsu::IsaFamily::Cdna5) {
    return static_cast<uint32_t>(enc.neg) | (static_cast<uint32_t>(enc.neg_hi) << 16);
  }
  return 0;
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_SIMD_GLUE_H_
