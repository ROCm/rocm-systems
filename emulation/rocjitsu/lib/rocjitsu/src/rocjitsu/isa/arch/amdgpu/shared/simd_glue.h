// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Operand-aware SIMD glue for the auto-generated execute_<mnemonic>
// kernels in execute_shared.h. Hand-maintained; lives separately from
// the generated header so the same code is not duplicated in
// simd_codegen.py (raw string) and execute_shared.h (emitted output).
//
// Layering: this header sees rocjitsu types (Wavefront, plus the Op /
// Inst template parameters) and bridges to the generic util SIMD
// primitives in util/simd.h. The generic util layer never depends on
// rocjitsu; only this direction is permitted.

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_SIMD_GLUE_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_SIMD_GLUE_H_

#include "rocjitsu/base/rj_compiler.h"
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
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "simdojo/components/vector_reg.h"
#include "util/host_target.h"
#include "util/simd.h"

#include <algorithm>
#include <array>
#include <atomic>
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

/// Module-local override that disables SIMD fast paths. Initialized from
/// RJ_FORCE_SCALAR at module load; test hooks may override the local copy.
inline bool simd_force_scalar() { return util::force_scalar(); }

/// SIMD families exposed to test-only execution-path observation.
enum class SimdFastPath : uint8_t {
  VOP3_LDEXP_RAW,
  VOP3_TERNARY_FP16,
  VOP3_TERNARY_RAW_FP,
  VOP_WORDS,
  VOP1_UNARY,
  VOP2_BINARY,
  VOP3_BINARY,
  VOP3_BINARY_TRUE16_SRC,
  VOP3_TERNARY,
  CVT_F64_TO_B32,
  VOPD,
  VOPC_COMPARE_STATUS,
};

namespace detail {

/// Number of live observers in this module. The disabled path performs a relaxed
/// atomic load and branch, without resolving thread-local storage. Observers on
/// other threads can enable the TLS check but cannot receive this thread's events.
inline std::atomic<unsigned> simd_fast_path_observer_count{0};
inline thread_local uint64_t *active_simd_fast_path_bits = nullptr;

// Keep TLS resolution out of line: the compiler can otherwise hoist it above
// the observer-count check even though the source short-circuits the access.
RJ_NOINLINE inline void record_simd_fast_path_slow(SimdFastPath path) {
  if (active_simd_fast_path_bits != nullptr)
    *active_simd_fast_path_bits |= uint64_t{1} << static_cast<uint8_t>(path);
}

inline void record_simd_fast_path(SimdFastPath path) {
  if (simd_fast_path_observer_count.load(std::memory_order_relaxed) != 0) [[unlikely]]
    record_simd_fast_path_slow(path);
}

} // namespace detail

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
  if (wf.wf_size() <= 32)
    amdgpu::RegisterAccess(wf).write_scalar(dst, static_cast<uint32_t>(mask));
  else
    amdgpu::RegisterAccess(wf).write_scalar64(dst, mask);
}

inline void write_explicit_lane_mask(uint32_t physical_dst, Wavefront &wf, uint64_t mask) {
  amdgpu::RegisterAccess regs(wf);
  if (wf.wf_size() <= 32) {
    if (regs.write_sgpr(physical_dst, static_cast<uint32_t>(mask)).failed())
      return;
  } else {
    if (regs.write_sgpr64(physical_dst, mask).failed())
      return;
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
  if (std::isnan(value))
    return uint32_t{0};
  if (value >= 32768.0f)
    return static_cast<uint32_t>(static_cast<uint16_t>(std::numeric_limits<int16_t>::max()));
  if (value < -32768.0f)
    return static_cast<uint32_t>(static_cast<uint16_t>(std::numeric_limits<int16_t>::min()));
  return static_cast<uint16_t>(static_cast<int16_t>(value));
}

/// @brief Convert f32 to a zero-extended packed u16 lane.
/// @details NaN and negative inputs convert to zero; values above the u16 range
/// saturate to UINT16_MAX.
inline uint32_t cvt_u16_f32(float32_t value) {
  if (std::isnan(value) || value < 0.0f)
    return uint32_t{0};
  if (value >= 65536.0f)
    return static_cast<uint32_t>(std::numeric_limits<uint16_t>::max());
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
                  requires { inst.inst_.clamp; })
      return inst.inst_.clamp;
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
  if (opsel & (1u << src_idx))
    value >>= 16;
  return value & 0xffffu;
}

inline util::native<uint32_t> select_vop3_true16_src(util::native<uint32_t> value, uint32_t opsel,
                                                     uint32_t src_idx) {
  if (opsel & (1u << src_idx))
    value >>= 16;
  return value & util::broadcast<uint32_t>(0xffffu);
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

inline uint8_t vop3_true16_dst_byte_mask(const Wavefront &wf, const rocjitsu::Operand &dst,
                                         bool dst_hi, bool cdna_low_dst_zeroes_high) {
  if (dst_hi)
    return 0xc;
  if (cdna_low_dst_zeroes_high && cdna_vop3_low_dst_zeroes_high(wf) &&
      (!dst.preserve_unwritten_bytes() || dst.register_byte_mask() == 0xf))
    return 0xf;
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
  if (omod == 1)
    v = v * T(2);
  else if (omod == 2)
    v = v * T(4);
  else if (omod == 3)
    v = v * T(0.5);
  if constexpr (std::is_same_v<T, float>) {
    using U = util::native<uint32_t>;
    U bits = std::bit_cast<U>(v);
    const U original = std::bit_cast<U>(unscaled);
    util::stdx::where((original & U(0x7fffffffu)) > U(0x7f800000u), bits) = original;
    if (omod == 3)
      util::stdx::where((original & U(0x7fffffffu)) < U(0x01000000u), bits) =
          original & U(0x80000000u);
    v = std::bit_cast<util::native<float>>(bits);
  }
  if (clamp) {
#if RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS
    if constexpr (std::is_same_v<T, double>) {
      v = util::map_native64_scalar<double>(v, [clamp_nan_to_zero](double x) {
        if (std::isnan(x))
          return clamp_nan_to_zero ? 0.0 : x;
        if (x <= 0.0)
          return 0.0;
        if (x > 1.0)
          return 1.0;
        return x;
      });
    } else
#endif
    {
      if (clamp_nan_to_zero)
        util::stdx::where(util::stdx::isnan(v), v) = T(0);
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
        if (subnormal)
          lane &= 0x8000000000000000ULL;
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
  if (omod == 0 && clamp == 0)
    return v;
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

inline util::native<uint32_t> finalize_omod_f16_bits_simd(util::native<uint32_t> value,
                                                          uint32_t omod) {
  if (omod == 0)
    return value;
  using U = util::native<uint32_t>;
  const auto subnormal = ((value & U(0x7c00u)) == U(0)) && ((value & U(0x03ffu)) != U(0));
  util::stdx::where(subnormal, value) = value & U(0x8000u);
  util::stdx::where((value & U(0x7fffu)) == U(0), value) = U(0);
  return value;
}

/// @brief F32 FMA under MODE denormal controls, in the caller's rounding environment.
/// @details The caller owns that environment: MODE.FP_ROUND rounding, and containing the
/// flags this arithmetic raises. Exceptional lanes use scalar NaN selection and
/// minimum-normal boundary handling.
/// force_denorm_flush overrides MODE to flush subnormal a, b, c and results.
/// Nonzero effective OMOD independently forces output flushing before destination scaling.
inline util::native<float> fma_f32_simd_unscoped(util::native<float> a, util::native<float> b,
                                                 util::native<float> c, const Wavefront &wf,
                                                 uint32_t omod = 0,
                                                 bool force_denorm_flush = false) {
  using U = util::native<uint32_t>;
  const uint32_t denorm_mode = force_denorm_flush ? 0 : wf.fp_denorm_mode_f32();
  if (!(denorm_mode & 1u)) {
    a = denormal::flush_value(a);
    b = denormal::flush_value(b);
    c = denormal::flush_value(c);
  }
  const bool flush_output = omod || !(denorm_mode & 2u);
  auto exceptional = (std::bit_cast<U>(a) & U(0x7fffffffu)) >= U(0x7f800000u) ||
                     (std::bit_cast<U>(b) & U(0x7fffffffu)) >= U(0x7f800000u) ||
                     (std::bit_cast<U>(c) & U(0x7fffffffu)) >= U(0x7f800000u);
  auto result = util::stdx::fma(a, b, c);
  if (flush_output)
    exceptional |= (std::bit_cast<U>(result) & U(0x7fffffffu)) == U(0x00800000u);
  if (util::stdx::any_of(exceptional))
    for (std::size_t i = 0; i < U::size(); ++i)
      if (exceptional[i])
        result[i] = fp_mode::fma_f32(a[i], b[i], c[i], wf.cu().arch(), wf.ieee_mode(), denorm_mode,
                                     omod != 0);
  // FMA's subnormal intermediate is flushed before OMOD can scale it normal.
  return flush_output ? denormal::flush_value(result) : result;
}

/// @brief F32 ADD and MUL with FMA's NaN and pre-packing tininess policy.
/// @details Callers pass SUB as ADD of the negated subtrahend. The caller owns the
/// guest rounding environment, as for fma_f32_simd_unscoped.
template <fp_mode::Arithmetic operation>
inline util::native<float> binary_f32_simd_unscoped(util::native<float> a, util::native<float> b,
                                                    const Wavefront &wf, uint32_t omod = 0) {
  static_assert(operation == fp_mode::Arithmetic::ADD || operation == fp_mode::Arithmetic::MUL);
  if constexpr (operation == fp_mode::Arithmetic::ADD)
    return fma_f32_simd_unscoped(a, util::native<float>(1.0f), b, wf, omod);
  else {
    using U = util::native<uint32_t>;
    auto zero = std::bit_cast<util::native<float>>((std::bit_cast<U>(a) ^ std::bit_cast<U>(b)) &
                                                   U(0x80000000u));
    return fma_f32_simd_unscoped(a, b, zero, wf, omod);
  }
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
template <typename = void>
  requires(util::has_stdx_simd)
inline PkF32Halves read_pkf32_halves(const RegisterAccess::OperandReadPair32View &op,
                                     uint32_t lane_base) {
  return {op.template load_lo_native<float>(lane_base),
          op.template load_hi_native<float>(lane_base)};
}

/// In-vector f32 sign flip (neg modifier): XOR the sign bit. Bit-exact to the
/// scalar `x = -x` for all values incl. ±0 / ±Inf / NaN (payload preserved).
inline util::native<float> pkf32_neg(util::native<float> v, bool do_neg) {
  if (!do_neg)
    return v;
  return std::bit_cast<util::native<float>>(std::bit_cast<util::native<uint32_t>>(v) ^
                                            util::native<uint32_t>(0x80000000u));
}

/// @brief Run an instruction's chunk loop, optionally under one guest FP environment.
/// @details Call after the register views exist, so callbacks fired while
/// acquiring those views stay outside the scope. Returns false when the destination
/// has no VGPR storage: that store path notifies plugins per chunk and must not
/// run inside the scope, so the caller leaves the instruction to the scalar body.
/// Amortizes one save/set/restore over every chunk instead of one per chunk.
/// `Scoped` false constructs nothing, leaving host-float functors' code generation
/// and their host NaN selection untouched.
///
/// VOPD in simd_coverage.h reaches the same policy without this helper: its two
/// slots are separate computation regions, so each scopes its own chunk loop.
/// The VOP3P packed-float and FMA MIX helpers keep their existing instruction-wide
/// scopes, which predate this helper and still span view acquisition.
template <bool Scoped, typename View, typename Body>
[[nodiscard]] inline bool run_instruction_chunks(const View &dst, uint32_t round_mode, Body body) {
  if constexpr (Scoped) {
    if (!dst.has_storage())
      return false;
    fp_mode::ScopedEnvironment environment(round_mode);
    body();
  } else {
    body();
  }
  return true;
}

/// VOP2 binary SIMD fast path. Returns true when the SIMD path executed
/// and the caller should skip its scalar per-lane loop; false on the
/// `force_scalar` override or when any operand reports `!simd_capable()`.
/// Constrained on `util::has_stdx_simd`; an unconstrained overload
/// below returns false unconditionally when the constraint cannot be
/// satisfied, so callers can write the probe without an `if constexpr`
/// guard.
/// `Scoped` takes one guest FP environment for the whole instruction through
/// run_instruction_chunks, for functors that use the unscoped arithmetic
/// cores instead of scoping each chunk.
template <typename T, bool Scoped = false, typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop2_simd(Inst &inst, Wavefront &wf, BinOp bin_op,
                                                       uint32_t round_mode = 0) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  // Resolve operand views once. Read-view acquisition observes plugin-visible
  // VGPR reads; write-view acquisition is write-only. Operands that are not
  // contiguous VGPR storage carry their scalar broadcast/fallback behavior in
  // the view object.
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.vsrc1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  if (!run_instruction_chunks<Scoped>(dst, round_mode, [&] {
        for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
          const uint64_t chunk = (exec >> base) & chunk_full;
          if (chunk == 0)
            continue;
          const auto a = src0.template load_native<T>(base);
          const auto b = src1.template load_native<T>(base);
          const auto r = bin_op(a, b);
          if (dst.template store_native<T>(base, r, chunk).failed())
            return;
        }
      }))
    return false;
  detail::record_simd_fast_path(SimdFastPath::VOP2_BINARY);
  return true;
}

/// Unconstrained fallback selected when `util::has_stdx_simd` is false.
/// Trivially inlined to `return false;` so the generated probe at the
/// call site costs nothing on toolchains without `<experimental/simd>`.
template <typename T, bool Scoped = false, typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop2_simd(Inst &, Wavefront &, BinOp, uint32_t = 0) {
  return false;
}

/// Execute the paired integer operations found in streaming-store kernels.
/// Resolve checked views once and load both slot results before either write,
/// since the two destinations may alias the other slot's sources. Opcode IDs
/// come from the generated ISA rather than being duplicated in this helper.
template <uint16_t kMovOp, uint16_t kAddOp, uint16_t kLshlOp, typename Slot>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopd_integer_pair_simd(Wavefront &wf, const Slot &x,
                                                             const Slot &y) {
  if (wf.wf_size() != 32 || simd_force_scalar())
    return false;
  const auto is_integer_slot = [](const Slot &slot) {
    if (slot.neg != 0 || slot.has_src2_operand || slot.src2_is_imm)
      return false;
    if (slot.op != kMovOp && slot.op != kAddOp && slot.op != kLshlOp)
      return false;
    return slot.dst->simd_capable() && slot.src0->simd_capable() &&
           (slot.op == kMovOp || slot.src1->simd_capable());
  };
  if (!is_integer_slot(x) || !is_integer_slot(y))
    return false;
  const uint64_t exec = static_cast<uint32_t>(wf.exec());
  if (exec == 0)
    return true;
  RegisterAccess registers(wf);
  const auto x_src0 = registers.read_operand(*x.src0, exec);
  const auto y_src0 = registers.read_operand(*y.src0, exec);
  std::optional<RegisterAccess::OperandReadView> x_src1;
  std::optional<RegisterAccess::OperandReadView> y_src1;
  if (x.op != kMovOp)
    x_src1.emplace(registers.read_operand(*x.src1, exec));
  if (y.op != kMovOp)
    y_src1.emplace(registers.read_operand(*y.src1, exec));
  const auto x_dst = registers.write_operand(*x.dst, exec);
  const auto y_dst = registers.write_operand(*y.dst, exec);
  const auto evaluate = [](const Slot &slot, const auto &src0_view, const auto &src1_view,
                           uint32_t lane_base) {
    const auto src0 = src0_view.template load_native<uint32_t>(lane_base);
    switch (slot.op) {
    case kMovOp:
      return src0;
    case kAddOp:
      return src0 + src1_view->template load_native<uint32_t>(lane_base);
    case kLshlOp:
      return simd_lshl_u32(src1_view->template load_native<uint32_t>(lane_base), src0);
    default:
      assert(false && "integer slot was validated before SIMD execution");
      return decltype(src0)(0);
    }
  };
  constexpr uint32_t kWidth = util::native_width_v<uint32_t>;
  for (uint32_t lane_base = 0; lane_base < wf.wf_size(); lane_base += kWidth) {
    const uint64_t lane_mask = (exec >> lane_base) & util::mask<uint64_t>(kWidth);
    if (lane_mask == 0)
      continue;
    const auto x_result = evaluate(x, x_src0, x_src1, lane_base);
    const auto y_result = evaluate(y, y_src0, y_src1, lane_base);
    if (x_dst.template store_native<uint32_t>(lane_base, x_result, lane_mask).failed())
      return true;
    if (y_dst.template store_native<uint32_t>(lane_base, y_result, lane_mask).failed())
      return true;
  }
  return true;
}

template <uint16_t kMovOp, uint16_t kAddOp, uint16_t kLshlOp, typename Slot>
[[nodiscard]] bool try_execute_vopd_integer_pair_simd(Wavefront &, const Slot &, const Slot &) {
  return false;
}

/// Re-type a `simd_mask` (e.g. the result of a float comparison) to the mask
/// type of `native<To>`, so it can drive `util::stdx::where` on a `native<To>`
/// value. Needed by the clamp/NaN cvt-to-int functors, which compute masks in
/// the float domain but blend into an int result. Wraps the libstdc++
/// `__proposed` mask cast in one place; `<experimental/simd>` is libstdc++-only
/// so the dependency is acceptable.
template <typename To, typename Mask>
  requires(util::has_stdx_simd)
inline auto simd_mask_as(const Mask &m) {
  return util::stdx::__proposed::static_simd_cast<util::native<To>>(m);
}

template <typename T, typename CmpOp>
  requires(util::has_stdx_simd)
inline uint64_t cmp_bits64(util::native<T> a, util::native<T> b, CmpOp cmp_op) {
#if RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS
  constexpr std::size_t W = util::native_width64;
  alignas(64) T abuf[W];
  alignas(64) T bbuf[W];
  a.copy_to(abuf, util::stdx::element_aligned);
  b.copy_to(bbuf, util::stdx::element_aligned);
  uint64_t bits = 0;
  for (std::size_t i = 0; i < W; ++i) {
    if constexpr (std::is_floating_point_v<T>) {
      using One = util::stdx::fixed_size_simd<T, 1>;
      const One av(&abuf[i], util::stdx::element_aligned);
      const One bv(&bbuf[i], util::stdx::element_aligned);
      if (cmp_op(av, bv)[0])
        bits |= uint64_t{1} << i;
    } else if (cmp_op(abuf[i], bbuf[i])) {
      bits |= uint64_t{1} << i;
    }
  }
  return bits;
#else
  return util::simd_mask_to_bits(cmp_op(a, b));
#endif
}

/// VOP1 unary SIMD fast path. Reads `src0` as `Tin`, applies `un_op`
/// (`native<Tin> -> native<Tout>`), masked-stores the result to `vdst` as
/// `Tout`. `Tin` and `Tout` are both 32-bit lane types (possibly different,
/// e.g. int32->float32 for v_cvt_f32_i32). Same contract as the VOP2 path:
/// returns true when the SIMD path executed; false on the `force_scalar`
/// override or when either operand reports `!simd_capable()`.
template <typename Tin, typename Tout, typename Inst, typename UnOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_unary_vop1_simd(Inst &inst, Wavefront &wf, UnOp un_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<Tout>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<Tin>(base);
    const auto r = un_op(a);
    if (dst.template store_native<Tout>(base, r, chunk).failed())
      return true;
  }
  return true;
}

/// Unconstrained fallback for the unary path; see the binary-path note above.
template <typename Tin, typename Tout, typename Inst, typename UnOp>
[[nodiscard]] bool try_execute_unary_vop1_simd(Inst &, Wavefront &, UnOp) {
  return false;
}

/// VOP2 ternary (fused multiply-add) SIMD fast path for literal-addend and
/// literal-multiplier forms. Reads src0/vsrc1 and writes vdst; it deliberately
/// does not read vdst. The accumulator-shaped VOP2 FMA/MAC forms use
/// try_execute_ternary_vop2_acc_simd below so register observation follows the
/// instruction-visible read set exactly.
///
/// F32 operations use fma_f32_simd_unscoped to preserve architectural NaN
/// payloads. `Scoped` takes one guest FP environment for the instruction through
/// run_instruction_chunks, for functors using the unscoped arithmetic cores.
template <typename T, bool Scoped = false, typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop2_simd(Inst &inst, Wavefront &wf,
                                                        util::native<T> k, FmaOp fma_op,
                                                        uint32_t round_mode = 0) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.vsrc1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  return run_instruction_chunks<Scoped>(dst, round_mode, [&] {
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      const auto a = src0.template load_native<T>(base);
      const auto b = src1.template load_native<T>(base);
      const auto d = util::native<T>{};
      const auto r = fma_op(a, b, d, k);
      if (dst.template store_native<T>(base, r, chunk).failed())
        return;
    }
  });
}

/// Unconstrained fallback for the ternary path; see the binary-path note above.
template <typename T, bool Scoped = false, typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_ternary_vop2_simd(Inst &, Wavefront &, util::native<T>, FmaOp,
                                                 uint32_t = 0) {
  return false;
}

/// VOP2 ternary SIMD fast path for dst-accumulate FMA/MAC forms:
/// `dst = fma(src0, vsrc1, dst)`. The destination is acquired as a read-write
/// view so read observation and write resolution use the same write-role VGPR
/// mapping.
/// `Scoped` takes one guest FP environment for the instruction through
/// run_instruction_chunks, for functors using the unscoped arithmetic cores.
template <typename T, bool Scoped = false, typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop2_acc_simd(Inst &inst, Wavefront &wf,
                                                            util::native<T> k, FmaOp fma_op,
                                                            uint32_t round_mode = 0) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.vsrc1, exec);
  auto acc = regs.readwrite_operand(inst.vdst, exec);
  return run_instruction_chunks<Scoped>(acc, round_mode, [&] {
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      const auto a = src0.template load_native<T>(base);
      const auto b = src1.template load_native<T>(base);
      const auto d = acc.template load_native<T>(base);
      const auto r = fma_op(a, b, d, k);
      if (acc.template store_native<T>(base, r, chunk).failed())
        return;
    }
  });
}

template <typename T, bool Scoped = false, typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_ternary_vop2_acc_simd(Inst &, Wavefront &, util::native<T>, FmaOp,
                                                     uint32_t = 0) {
  return false;
}

/// 64-bit-lane VOP2 binary SIMD fast path (v_add_f64 / v_mul_f64 /
/// v_max_num_f64 / v_min_num_f64). VOP2 has no abs/neg/omod/clamp fields and
/// reads its second source as `vsrc1` (not `src1`), using split lo/hi
/// VGPR pairs and masked stores. `T` is `double` for
/// host arithmetic, or `uint64_t` for raw encodings: min/max callers pass
/// those to minmax::evaluate, the same evaluation as the scalar path.
template <typename T = double, typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop2_f64_simd(Inst &inst, Wavefront &wf,
                                                           BinOp bin_op) {
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, uint64_t>);
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto src1 = regs.read_operand64(inst.vsrc1, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    if (dst.template store_native<T>(base, bin_op(a, b), chunk).failed())
      return true;
  }
  return true;
}

/// Unconstrained fallback for the f64 vop2 binary path.
template <typename T = double, typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop2_f64_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// 64-bit-lane VOP1 unary SIMD fast path. The 64-bit counterpart of
/// try_execute_unary_vop1_simd: reads src0 as `native<T>` through a 64-bit
/// RegisterAccess operand view, applies `un_op` (`native<T> ->
/// native<T>`), and masked-stores the result to vdst. `T` is `double` for the
/// f64 math ops (ceil/floor/trunc/rndne/fract/rcp/rsq/sqrt) and `uint64_t` for
/// the pure 64-bit move (v_mov_b64). Same contract as the other paths: returns
/// true when the SIMD path executed; false on the `force_scalar` override or
/// when either operand reports `!simd_capable()`.
///
/// The rounding ops map to `vroundpd`, sqrt to `vsqrtpd`, and `1.0 / x` to
/// `vdivpd` — all correctly-rounded IEEE operations, bit-identical to the scalar
/// `std::ceil`/`std::sqrt`/... for every finite and infinite input. NaN-*input*
/// lanes may differ in propagated NaN payload (accepted — the result is a NaN
/// either way), guarded by the UtilSimd.*F64*_VectorMatchesScalar_BitExact tests.
template <typename T, typename Inst, typename UnOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_unary_vop1_f64_simd(Inst &inst, Wavefront &wf, UnOp un_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    if (dst.template store_native<T>(base, un_op(a), chunk).failed())
      return true;
  }
  return true;
}

/// Unconstrained fallback for the f64 unary path; see the binary-path note.
template <typename T, typename Inst, typename UnOp>
[[nodiscard]] bool try_execute_unary_vop1_f64_simd(Inst &, Wavefront &, UnOp) {
  return false;
}

// Evaluate FP64 inputs and store one 32-bit result per lane, preserving inactive
// destinations. FREXP exponent extraction still uses this mixed-width bridge.
template <typename Tout, typename Inst, typename CvtOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_cvt_f64_to_b32_simd(Inst &inst, Wavefront &wf, CvtOp cvt_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto s = src0.template load_native<double>(base);
    const util::narrow32<Tout> r = cvt_op(s);
    if (dst.template store_narrow<Tout>(base, r, chunk).failed())
      return true;
  }
  return true;
}

/// Unconstrained fallback for the f64->b32 cvt path; see the binary-path note.
template <typename Tout, typename Inst, typename CvtOp>
[[nodiscard]] bool try_execute_cvt_f64_to_b32_simd(Inst &, Wavefront &, CvtOp) {
  return false;
}

/// @brief Apply V_CVT_F32_F16 policy before widening a SIMD batch.
inline util::native<float> cvt_f32_f16_mode_simd(util::native<uint32_t> raw, const Wavefront &wf) {
  using U = util::native<uint32_t>;
  raw = input_denormal::prepare<fp_format::F16>(
      raw, input_denormal::Policy::make(wf.fp_denorm_mode_f16_f64()));
  const U magnitude = raw & U(0x7fffu);
  if (fp_mode::quiets_nan(wf.cu().arch(), wf.ieee_mode()))
    util::stdx::where(magnitude > U(0x7c00u), raw) = raw | U(0x0200u);
  return util::f16_to_f32_simd(raw);
}

/// VOP3 v_cvt_f32_f16 fast path. The generic form reads the low f16 half and
/// writes the full f32 dword, matching non-true16 scalar write_lane semantics.
/// The true16 form selects src0 with op_sel[0] before widening; the destination
/// is f32 in both cases, so no destination half merge is needed.
/// Widening is exact and the output stage is raw-bit, so neither needs a guest
/// FP environment.
template <bool True16, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_cvt_f32_f16_vop3_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t opsel = vop3_opsel(inst.inst_);
  const uint32_t omod = effective_vop3_omod_f32(wf, inst.inst_.omod);
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto raw = src0.template load_native<T>(base);
    if constexpr (True16)
      raw = select_vop3_true16_src(raw, opsel, 0);
    else
      raw = raw & util::broadcast<T>(0xffffu);
    const auto src = apply_vop3_src_mod_f32<0>(cvt_f32_f16_mode_simd(raw, wf), abs, neg);
    const auto result = apply_vop3_dst_mod_f32(src, omod, clamp, floating_clamp_nan_to_zero(wf),
                                               wf.fp_round_mode_f32());
    if (dst.template store_native<T>(base, std::bit_cast<util::native<T>>(result), chunk).failed())
      return true;
  }
  return true;
}

template <bool True16, typename Inst>
[[nodiscard]] bool try_execute_cvt_f32_f16_vop3_simd(Inst &, Wavefront &) {
  return false;
}

/// VOPC compare SIMD fast path: per active EXEC lane, `cmp_op(src0, vsrc1)`
/// produces a `simd_mask` whose bit is packed into VCC at the lane position;
/// inactive-lane VCC bits are preserved (mirroring the scalar body, which
/// flips only active-lane bits). VOPC writes VCC only — there is no vdst
/// operand and CDNA4 has no v_cmpx (EXEC-writing) form, so this single shape
/// covers every compare. `T` is the 32-bit lane read type (uint32_t raw
/// encodings for the f16/f32 relations, int32_t/uint32_t for the integer ones);
/// the f16 and 16-bit integer relations read as 32-bit lanes and narrow/convert
/// inside the functor. The VCC merge is identical to the carry path's.
///
/// Float relations never see host floats: their functors call
/// comparison::evaluate on the raw encodings with the captured MODE
/// input-denormal policy, the same evaluation the scalar body uses, so host
/// DAZ cannot alter a lane and the compares are bit-exact for every input
/// including NaN/Inf/±0/subnormals. Accepted divergences in legacy host
/// fmin/fmax paths do not apply to these compares or to minmax::evaluate.
template <typename T, typename Inst, typename CmpOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op,
                                                WriteResult write_result) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  uint64_t vcc = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.vsrc1, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const uint64_t cmp_bits = util::simd_mask_to_bits(cmp_op(a, b));
    vcc = (vcc & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  write_result(vcc);
  return true;
}

/// Unconstrained fallback for the VOPC path; see the binary-path note above.
template <typename T, typename Inst, typename CmpOp, typename WriteResult>
[[nodiscard]] bool try_execute_vopc_simd(Inst &, Wavefront &, CmpOp, WriteResult) {
  return false;
}

template <typename T, typename Inst, typename CmpOp>
[[nodiscard]] inline bool try_execute_vopc_simd(Inst &inst, Wavefront &wf, CmpOp op) {
  return try_execute_vopc_simd<T>(inst, wf, op, [&wf](uint64_t value) { wf.set_vcc_mask(value); });
}

/// 64-bit-lane VOPC compare SIMD fast path (f64/i64/u64 relations). Identical to
/// try_execute_vopc_simd but reads each operand as `native<T>` (T = uint64_t
/// raw encodings for f64, int64_t / uint64_t for the integer relations) through
/// 64-bit RegisterAccess operand views, so it processes `native_width64` lanes
/// per chunk. f64 functors evaluate through comparison::evaluate as above.
/// Same VCC merge.
template <typename T, typename Inst, typename CmpOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc64_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op,
                                                  WriteResult write_result) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  uint64_t vcc = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto src1 = regs.read_operand64(inst.vsrc1, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const uint64_t cmp_bits = cmp_bits64<T>(a, b, cmp_op);
    vcc = (vcc & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  write_result(vcc);
  return true;
}

/// Unconstrained fallback for the 64-bit VOPC path; see the binary-path note.
template <typename T, typename Inst, typename CmpOp, typename WriteResult>
[[nodiscard]] bool try_execute_vopc64_simd(Inst &, Wavefront &, CmpOp, WriteResult) {
  return false;
}

template <typename T, typename Inst, typename CmpOp>
[[nodiscard]] inline bool try_execute_vopc64_simd(Inst &inst, Wavefront &wf, CmpOp op) {
  return try_execute_vopc64_simd<T>(inst, wf, op,
                                    [&wf](uint64_t value) { wf.set_vcc_mask(value); });
}

/// VOP3 integer/bitwise binary SIMD fast path. Same shape as
/// try_execute_binary_vop2_simd but reads the VOP3 operands `src0`/`src1`
/// (instead of `src0`/`vsrc1`). The generated integer/bitwise VOP3 bodies apply
/// no source/result modifiers other than integer CLAMP. Saturating integer
/// arithmetic falls back to the scalar body; the plain op is bit-identical to
/// that body when CLAMP is clear. T is a 32-bit integer lane type.
///
/// Raw floating-point operations use WithModifiers to supply ABS/NEG, OMOD
/// and CLAMP around the operation. CLAMP then stays on the SIMD path.
template <typename T, typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_simd(Inst &inst, Wavefront &wf, BinOp bin_op) {
  if ((!floating_operation::applies_modifiers_v<BinOp> && inst.inst_.clamp != 0u) ||
      simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const auto r = bin_op(a, b);
    if (dst.template store_native<T>(base, r, chunk).failed())
      return true;
  }
  return true;
}

/// Unconstrained fallback for the VOP3 integer binary path; see the VOP2
/// binary-path note above.
template <typename T, typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop3_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// VOP3 binary operations whose operands are encoded as true16 sources but
/// whose destination is a full b32 pack result.
template <typename T, typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_true16_src_simd(Inst &inst, Wavefront &wf,
                                                                  BinOp bin_op) {
  static_assert(std::is_same_v<T, uint32_t>);
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint32_t opsel = vop3_opsel(inst.inst_);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = select_vop3_true16_src(src0.template load_native<T>(base), opsel, 0);
    const auto b = select_vop3_true16_src(src1.template load_native<T>(base), opsel, 1);
    const auto r = bin_op(a, b);
    if (dst.template store_native<T>(base, r, chunk).failed())
      return true;
  }
  return true;
}

template <typename T, typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop3_true16_src_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// VOP3 f32 binary SIMD fast path. Reads `src0`/`src1`, applies the per-source
/// abs/neg modifiers, runs `bin_op`, then applies the result omod/clamp — the
/// exact order of the generated scalar body (abs->neg per source, op,
/// omod->clamp on the result). The modifier helpers are bit-exact, so unlike the
/// VOP2 path this fast path stays correct even when modifiers are set; no bail.
/// `Scoped` takes one guest FP environment for the instruction through
/// run_instruction_chunks, for functors using the unscoped arithmetic cores.
template <typename T, bool Scoped = false, typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_fp_simd(Inst &inst, Wavefront &wf, BinOp bin_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = effective_vop3_omod_f32(wf, inst.inst_.omod);
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  return run_instruction_chunks<Scoped>(dst, wf.fp_round_mode_f32(), [&] {
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      const auto a = apply_vop3_src_mod_f32<0>(src0.template load_native<T>(base), abs, neg);
      const auto b = apply_vop3_src_mod_f32<1>(src1.template load_native<T>(base), abs, neg);
      const auto r = apply_vop3_dst_mod_f32(bin_op(a, b), omod, clamp,
                                            floating_clamp_nan_to_zero(wf), wf.fp_round_mode_f32());
      if (dst.template store_native<T>(base, r, chunk).failed())
        return;
    }
  });
}

/// Unconstrained fallback for the VOP3 f32 binary path.
template <typename T, bool Scoped = false, typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop3_fp_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// VOP3 raw-lane VOPC compare SIMD fast path (32-bit lane). The VOP3 form
/// of v_cmp_<rel>_<i16|u16|i32|u32|f16|f32> reads src0/src1 (not src0/vsrc1) and
/// writes the per-lane compare result into an arbitrary wave-mask scalar
/// destination through a caller-provided architectural commit. The functor
/// receives the raw source encodings: integer relations use no modifiers, and
/// float relations apply their captured abs/neg fields in shared/comparison.h.
/// The raw result contains active EXEC lanes only. Returns true when the SIMD
/// path executed.
template <typename T, bool True16 = false, typename Inst, typename CmpOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc_vop3_int_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op,
                                                         WriteResult write_result) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  uint64_t dst = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto a = src0.template load_native<T>(base);
    auto b = src1.template load_native<T>(base);
    if constexpr (True16) {
      a = select_vop3_true16_src(a, vop3_opsel(inst.inst_), 0);
      b = select_vop3_true16_src(b, vop3_opsel(inst.inst_), 1);
    }
    const uint64_t cmp_bits = util::simd_mask_to_bits(cmp_op(a, b));
    dst = (dst & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  write_result(dst);
  return true;
}

/// Unconstrained fallback for the VOP3 integer VOPC path; see the binary-path note.
template <typename T, bool True16 = false, typename Inst, typename CmpOp, typename WriteResult>
[[nodiscard]] bool try_execute_vopc_vop3_int_simd(Inst &, Wavefront &, CmpOp, WriteResult) {
  return false;
}

/// 64-bit-lane VOP3 raw-lane VOPC compare SIMD fast path (i64/u64/f64).
/// Identical to try_execute_vopc_vop3_int_simd but reads each operand as
/// `native<T>` (T = int64_t / uint64_t) through 64-bit RegisterAccess operand
/// views, so it processes `native_width64` lanes per chunk. The caller performs
/// the result commit; any float modifiers are applied by the functor.
template <typename T, typename Inst, typename CmpOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc64_vop3_int_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op,
                                                           WriteResult write_result) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  uint64_t dst = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto src1 = regs.read_operand64(inst.src1, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const uint64_t cmp_bits = cmp_bits64<T>(a, b, cmp_op);
    dst = (dst & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  write_result(dst);
  return true;
}

/// Unconstrained fallback for the 64-bit VOP3 integer VOPC path; see the binary-path note.
template <typename T, typename Inst, typename CmpOp, typename WriteResult>
[[nodiscard]] bool try_execute_vopc64_vop3_int_simd(Inst &, Wavefront &, CmpOp, WriteResult) {
  return false;
}

/// VOP3 binary SIMD fast path on raw 64-bit encodings. Reads src0/src1 as
/// `native<uint64_t>` and leaves every VOP3 modifier to `bin_op`, which applies
/// them to the bits in the same order as the scalar body.
template <typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_raw64_simd(Inst &inst, Wavefront &wf,
                                                             BinOp bin_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint64_t;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto src1 = regs.read_operand64(inst.src1, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    if (dst.template store_native<T>(base, bin_op(a, b), chunk).failed())
      return true;
  }
  return true;
}

/// Unconstrained fallback for the raw 64-bit VOP3 binary path.
template <typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop3_raw64_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// @brief Run a unary F32/F64 operation that rounds before OMOD/CLAMP.
/// @details Adapt the existing float functor to the common raw-bit modifier
/// wrapper. The VOP1 helpers supply the same src0/vdst access and EXEC masking
/// needed here; all VOP3 modifiers come from the wrapper.
template <typename Float, typename Inst, typename UnOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_unary_vop3_rounded_f32_simd(Inst &inst, Wavefront &wf,
                                                                  UnOp un_op) {
  const auto raw_operation = [un_op](auto bits) {
    const auto value = std::bit_cast<util::native<float>>(bits);
    return std::bit_cast<decltype(bits)>(un_op(value));
  };
  return try_execute_unary_vop1_simd<uint32_t, uint32_t>(
      inst, wf, vop3_float_operation<fp_format::F32>(inst, wf, raw_operation));
}

template <typename Inst, typename UnOp>
[[nodiscard]] bool try_execute_unary_vop3_rounded_f32_simd(Inst &, Wavefront &, UnOp) {
  return false;
}

/// VOP3 f64 unary SIMD fast path. 64-bit-lane counterpart of
/// try_execute_unary_vop3_fp_simd: reads `src0` as `native<double>`, applies
/// the src0 abs/neg modifiers (apply_vop3_src_mod_f64), runs `un_op`, applies
/// the result omod/clamp (apply_vop3_dst_mod_f64). All bit-exact.
template <typename Inst, typename UnOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_unary_vop3_fp64_simd(Inst &inst, Wavefront &wf, UnOp un_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = double;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = effective_vop3_omod_f64(wf, inst.inst_.omod);
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(src0.template load_native<T>(base), abs, neg);
    const auto r = apply_vop3_dst_mod_f64(un_op(a), omod, clamp, floating_clamp_nan_to_zero(wf));
    if (dst.template store_native<T>(base, r, chunk).failed())
      return true;
  }
  return true;
}

/// Unconstrained fallback for the VOP3 f64 unary path; see the binary-path note.
template <typename Inst, typename UnOp>
[[nodiscard]] bool try_execute_unary_vop3_fp64_simd(Inst &, Wavefront &, UnOp) {
  return false;
}

/// VOP3 integer/bitwise ternary SIMD fast path. Reads `src0`/`src1`/`src2`,
/// runs `tern_op(a, b, c)`, and masked-stores the result. The generated scalar
/// bodies for these ternary integer ops apply no source/result modifiers except
/// integer CLAMP. Saturating arithmetic falls back to the scalar body; the plain
/// functor is bit-identical to that body when CLAMP is clear. T is a 32-bit
/// integer lane type (typically uint32_t). Same SIMD-capable / EXEC-chunk loop
/// as the binary VOP3 path. For raw floating-point operations, WithModifiers
/// supplies ABS/NEG, OMOD and CLAMP, so CLAMP stays on the SIMD path.
template <typename T, typename Inst, typename TernOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop3_simd(Inst &inst, Wavefront &wf, TernOp tern_op) {
  if ((!floating_operation::applies_modifiers_v<TernOp> && inst.inst_.clamp != 0u) ||
      simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand(inst.src2, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const auto c = src2.template load_native<T>(base);
    const auto r = tern_op(a, b, c);
    if (dst.template store_native<T>(base, r, chunk).failed())
      return true;
  }
  return true;
}

/// Unconstrained fallback for the VOP3 integer ternary path; see binary-path note.
template <typename T, typename Inst, typename TernOp>
[[nodiscard]] bool try_execute_ternary_vop3_simd(Inst &, Wavefront &, TernOp) {
  return false;
}

/// VOP3 ternary operations whose three sources and destination are true16,
/// such as min3/max3/med3 i16/u16 on RDNA true16 encodings. The glue applies
/// no modifier; F16 min/max functors apply every VOP3 modifier to the raw bits.
template <typename T, typename Inst, typename TernOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop3_true16_simd(Inst &inst, Wavefront &wf,
                                                               TernOp tern_op) {
  static_assert(std::is_same_v<T, uint32_t>);
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint32_t opsel = vop3_opsel(inst.inst_);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand(inst.src2, exec);
  auto dst = regs.readwrite_operand(inst.vdst, exec);
  if (!dst.has_storage())
    return false;
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = select_vop3_true16_src(src0.template load_native<T>(base), opsel, 0);
    const auto b = select_vop3_true16_src(src1.template load_native<T>(base), opsel, 1);
    const auto c = select_vop3_true16_src(src2.template load_native<T>(base), opsel, 2);
    const auto r = tern_op(a, b, c);
    const auto out_half = r & util::broadcast<T>(0xffffu);
    auto prev = dst.template load_native<T>(base);
    auto out = (opsel & 0x8u) ? ((prev & util::broadcast<T>(0x0000ffffu)) | (out_half << 16))
                              : ((prev & util::broadcast<T>(0xffff0000u)) | out_half);
    if (!(opsel & 0x8u) && cdna_vop3_low_dst_zeroes_high(wf))
      out = out_half;
    if (dst.template store_native<T>(base, out, chunk).failed())
      return true;
  }
  return true;
}

template <typename T, typename Inst, typename TernOp>
[[nodiscard]] bool try_execute_ternary_vop3_true16_simd(Inst &, Wavefront &, TernOp) {
  return false;
}

/// VOP3 f32 ternary SIMD fast path (FMA / MAD family). Reads src0/src1/src2 as
/// `native<float>`, applies the per-source abs/neg VOP3 modifiers, runs
/// `tern_op(a, b, c)`, applies CLAMP and optionally OMOD. FIXUP handles OMOD
/// inside its operation using guest rounding. FMA uses the shared hardware
/// NaN selection policy in both scalar and SIMD execution.
/// `Scoped` takes one guest FP environment for the instruction through
/// run_instruction_chunks, for functors using the unscoped arithmetic cores.
template <bool Scoped = false, typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop3_fp_simd(Inst &inst, Wavefront &wf, FmaOp tern_op,
                                                           bool apply_omod = true,
                                                           bool force_output_flush = false) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = float32_t;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod =
      apply_omod ? effective_vop3_omod_f32(wf, inst.inst_.omod, force_output_flush) : 0;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand(inst.src2, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  return run_instruction_chunks<Scoped>(dst, wf.fp_round_mode_f32(), [&] {
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      const auto a = apply_vop3_src_mod_f32<0>(src0.template load_native<T>(base), abs, neg);
      const auto b = apply_vop3_src_mod_f32<1>(src1.template load_native<T>(base), abs, neg);
      const auto c = apply_vop3_src_mod_f32<2>(src2.template load_native<T>(base), abs, neg);
      const auto r = apply_vop3_dst_mod_f32(tern_op(a, b, c), omod, clamp,
                                            floating_clamp_nan_to_zero(wf), wf.fp_round_mode_f32());
      if (dst.template store_native<T>(base, r, chunk).failed())
        return;
    }
  });
}

template <bool Scoped = false, typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_ternary_vop3_fp_simd(Inst &, Wavefront &, FmaOp, bool = true,
                                                    bool = false) {
  return false;
}

/// @brief Execute VOP3 F16 ternary SIMD with instruction-specific output ordering.
/// @details MAD and older min/max use the destination-format output stage:
/// widen -> ABS/NEG -> evaluate -> round F16 -> OMOD -> CLAMP.
/// Storage: generic zero-extends the destination dword; true16 selects source
/// and destination halves per OP_SEL and ISA storage policy.
template <bool True16, typename Inst, typename TernOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop3_fp16_simd(Inst &inst, Wavefront &wf,
                                                             TernOp tern_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  const uint32_t opsel = vop3_opsel(inst.inst_);
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const auto output_policy =
      output_modifier_policy<fp_format::F16>(wf, inst.inst_.omod, inst.inst_.clamp);
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const auto finish = [&](util::native<float> value) {
    return output_modifier::apply<fp_format::F16>(util::f32_to_f16_mode_simd(value, wf.fp16_ovfl()),
                                                  output_policy);
  };
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand(inst.src2, exec);
  if constexpr (True16) {
    const bool dst_hi = (opsel & 0x8u) != 0;
    const uint8_t bytes = vop3_true16_dst_byte_mask(wf, inst.vdst, dst_hi, true);
    auto dst = regs.write_operand(inst.vdst, exec, bytes);
    if (!dst.has_storage())
      return false;
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      auto a_raw = src0.template load_native<T>(base);
      auto b_raw = src1.template load_native<T>(base);
      auto c_raw = src2.template load_native<T>(base);
      a_raw = select_vop3_true16_src(a_raw, opsel, 0);
      b_raw = select_vop3_true16_src(b_raw, opsel, 1);
      c_raw = select_vop3_true16_src(c_raw, opsel, 2);
      const auto a = apply_vop3_src_mod_f32<0>(util::f16_to_f32_simd(a_raw), abs, neg);
      const auto b = apply_vop3_src_mod_f32<1>(util::f16_to_f32_simd(b_raw), abs, neg);
      const auto c = apply_vop3_src_mod_f32<2>(util::f16_to_f32_simd(c_raw), abs, neg);
      const auto out_half = finish(tern_op(a, b, c));
      const auto out = dst_hi ? out_half << 16 : out_half;
      if (dst.template store_native<T>(base, out, chunk).failed())
        return true;
    }
  } else {
    auto dst = regs.write_operand(inst.vdst, exec);
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      auto a_raw = src0.template load_native<T>(base) & util::broadcast<T>(0xffffu);
      auto b_raw = src1.template load_native<T>(base) & util::broadcast<T>(0xffffu);
      auto c_raw = src2.template load_native<T>(base) & util::broadcast<T>(0xffffu);
      const auto a = apply_vop3_src_mod_f32<0>(util::f16_to_f32_simd(a_raw), abs, neg);
      const auto b = apply_vop3_src_mod_f32<1>(util::f16_to_f32_simd(b_raw), abs, neg);
      const auto c = apply_vop3_src_mod_f32<2>(util::f16_to_f32_simd(c_raw), abs, neg);
      const auto out = finish(tern_op(a, b, c)) & util::broadcast<T>(0xffffu);
      if (dst.template store_native<T>(base, out, chunk).failed())
        return true;
    }
  }
  detail::record_simd_fast_path(SimdFastPath::VOP3_TERNARY_FP16);
  return true;
}

template <bool True16, typename Inst, typename TernOp>
[[nodiscard]] bool try_execute_ternary_vop3_fp16_simd(Inst &, Wavefront &, TernOp) {
  return false;
}

/// VOP3 f64 ternary SIMD fast path. 64-bit-lane counterpart: read src0/src1/src2
/// through 64-bit RegisterAccess operand views, apply abs/neg in f64, run
/// `tern_op`, apply CLAMP and optionally OMOD, and store through a 64-bit RegisterAccess write
/// view.
template <typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop3_fp64_simd(Inst &inst, Wavefront &wf,
                                                             FmaOp tern_op,
                                                             bool apply_omod = true) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = double;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = apply_omod ? effective_vop3_omod_f64(wf, inst.inst_.omod) : 0;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto src1 = regs.read_operand64(inst.src1, exec);
  auto src2 = regs.read_operand64(inst.src2, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(src0.template load_native<T>(base), abs, neg);
    const auto b = apply_vop3_src_mod_f64<1>(src1.template load_native<T>(base), abs, neg);
    const auto c = apply_vop3_src_mod_f64<2>(src2.template load_native<T>(base), abs, neg);
    const auto r =
        apply_vop3_dst_mod_f64(tern_op(a, b, c), omod, clamp, floating_clamp_nan_to_zero(wf));
    if (dst.template store_native<T>(base, r, chunk).failed())
      return true;
  }
  return true;
}

template <typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_ternary_vop3_fp64_simd(Inst &, Wavefront &, FmaOp, bool = true) {
  return false;
}

/// VOP3 dst-accumulate FMA fast path (f32). Counterpart of
/// try_execute_ternary_vop3_fp_simd for the v_fmac / v_mac family, where the
/// third FMA operand IS the destination register (no separate src2 Operand
/// in the per-isa codegen class). The scalar body reads vdst as the
/// accumulator without applying abs/neg to it (per scalar; verified for
/// v_fmac_f32_vop3 + v_mac_f32_vop3). SIMD mirrors: read inst.vdst as the
/// third operand, apply abs/neg to src0/src1 only, run `tern_op`, apply
/// result omod/clamp, masked-store back to inst.vdst (overwriting accumulator).
/// `Scoped` takes one guest FP environment for the instruction through
/// run_instruction_chunks, for functors using the unscoped arithmetic cores.
template <bool Scoped = false, typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_fmac_vop3_fp_simd(Inst &inst, Wavefront &wf, FmaOp tern_op,
                                                        bool force_output_flush = false) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = float32_t;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = effective_vop3_omod_f32(wf, inst.inst_.omod, force_output_flush);
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto acc = regs.readwrite_operand(inst.vdst, exec);
  return run_instruction_chunks<Scoped>(acc, wf.fp_round_mode_f32(), [&] {
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      const auto a = apply_vop3_src_mod_f32<0>(src0.template load_native<T>(base), abs, neg);
      const auto b = apply_vop3_src_mod_f32<1>(src1.template load_native<T>(base), abs, neg);
      const auto c = acc.template load_native<T>(base); // accumulator, NO modifier
      const auto r = apply_vop3_dst_mod_f32(tern_op(a, b, c), omod, clamp,
                                            floating_clamp_nan_to_zero(wf), wf.fp_round_mode_f32());
      if (acc.template store_native<T>(base, r, chunk).failed())
        return;
    }
  });
}

template <bool Scoped = false, typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_fmac_vop3_fp_simd(Inst &, Wavefront &, FmaOp, bool = false) {
  return false;
}

enum class MadVop2 { Accumulate, MultiplyLiteral, AddLiteral };

/// VOP3 f32 unary SIMD fast path. Reads `src0` as f32, applies the src0 abs/neg
/// modifiers, runs `un_op` (`native<float> -> native<float>`), then applies the
/// result omod/clamp — the scalar body's order (abs->neg, op, omod->clamp). Tin
/// and Tout are both float32_t (the plain int/cvt unary VOP3 forms apply no
/// modifiers and reuse the VOP1 unary path directly). The modifier helpers are
/// bit-exact, so the fast path stays correct with modifiers set; no bail.
/// `round_mode` rounds OMOD overflow. Callers pass MODE.FP_ROUND, or nearest-even
/// for the TRANS results that ignore it; both come from the generator's policy sets.
template <typename Tin, typename Tout, typename Inst, typename UnOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_unary_vop3_fp_simd(Inst &inst, Wavefront &wf, UnOp un_op,
                                                         bool force_output_flush,
                                                         uint32_t round_mode) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = effective_vop3_omod_f32(wf, inst.inst_.omod, force_output_flush);
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<Tout>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(src0.template load_native<Tin>(base), abs, neg);
    const auto r =
        apply_vop3_dst_mod_f32(un_op(a), omod, clamp, floating_clamp_nan_to_zero(wf), round_mode);
    if (dst.template store_native<Tout>(base, r, chunk).failed())
      return true;
  }
  return true;
}

/// Unconstrained fallback for the VOP3 f32 unary path.
template <typename Tin, typename Tout, typename Inst, typename UnOp>
[[nodiscard]] bool try_execute_unary_vop3_fp_simd(Inst &, Wavefront &, UnOp, bool, uint32_t) {
  return false;
}

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

// gfx1100 and gfx1201 apply NEG_LO/NEG_HI to the selected integer source halves.
// RDNA3.5 follows RDNA3; CDNA5 follows RDNA4 for their common packed operations.
template <typename Encoding>
inline uint32_t packed_integer_negations(rj_code_arch_t arch, const Encoding &enc) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
      arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5)
    return static_cast<uint32_t>(enc.neg) | (static_cast<uint32_t>(enc.neg_hi) << 16);
  return 0;
}

/// VOP3P packed-f32 binary fast path (v_pk_add_f32 / v_pk_mul_f32). In a VGPR
/// pair {N, N+1} register N is the LO f32 of every lane, N+1 the HI f32, so each
/// half is a native-width native<float> read of one register (read_pkf32_halves)
/// and the per-half arithmetic runs at full native width. OP_SEL chooses each
/// source half independently, including broadcasts and swapped halves.
/// neg/neg_hi bits 0/1 sign-flip the respective half. MODE and CLAMP match
/// the scalar helper. CDNA5 scalar sources replicate one DWORD independently
/// of OP_SEL; earlier CDNA preserves scalar register pairs.
template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_pk_binary_f32_simd(Inst &inst, Wavefront &wf,
                                                               uint32_t op_sel, uint32_t op_sel_hi,
                                                               Op op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  fp_mode::ScopedEnvironment environment(wf.fp_round_mode_f32());
  auto flush_input = [&wf](util::native<float> value) {
    return (wf.fp_denorm_mode_f32() & 1u) ? value : denormal::flush_value(value);
  };
  auto binary = [&op](util::native<float> first, util::native<float> second) {
    util::native<float> result = op(first, second);
    // Preserve the first input's quieted NaN whenever that input is NaN.
    // Host SIMD arithmetic can otherwise choose the second input's payload.
    util::stdx::where(util::stdx::isnan(first), result) = std::bit_cast<util::native<float>>(
        std::bit_cast<util::native<uint32_t>>(first) | 0x00400000u);
    return result;
  };
  auto flush_output = [&wf, &inst](util::native<float> value) {
    if (inst.inst_.clamp)
      value = apply_vop3_dst_mod_f32(value, 0, 1, floating_clamp_nan_to_zero(wf),
                                     wf.fp_round_mode_f32());
    return (wf.fp_denorm_mode_f32() & 2u) ? value : denormal::flush_value(value);
  };
  constexpr std::size_t W = util::native_width_v<float>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const bool neg0_lo = inst.inst_.neg & 1u;
  const bool neg1_lo = inst.inst_.neg & 2u;
  const bool neg0_hi = inst.inst_.neg_hi & 1u;
  const bool neg1_hi = inst.inst_.neg_hi & 2u;
  RegisterAccess regs(wf);
  const auto scalar_mode = wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA5 ? ScalarPairMode::Replicate32
                                                                      : ScalarPairMode::Preserve;
  auto src0 = regs.read_operand_pair32(inst.src0, exec, scalar_mode);
  auto src1 = regs.read_operand_pair32(inst.src1, exec, scalar_mode);
  auto dst = regs.write_operand_pair32(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const PkF32Halves a = read_pkf32_halves(src0, base);
    const PkF32Halves b = read_pkf32_halves(src1, base);
    const util::native<float> r_lo =
        binary(flush_input(pkf32_neg((op_sel & 1u) ? a.hi : a.lo, neg0_lo)),
               flush_input(pkf32_neg((op_sel & 2u) ? b.hi : b.lo, neg1_lo)));
    const util::native<float> r_hi =
        binary(flush_input(pkf32_neg((op_sel_hi & 1u) ? a.hi : a.lo, neg0_hi)),
               flush_input(pkf32_neg((op_sel_hi & 2u) ? b.hi : b.lo, neg1_hi)));
    if (dst.template store_native_pair<float>(base, flush_output(r_lo), flush_output(r_hi), chunk)
            .failed())
      return true;
  }
  return true;
}

template <typename Inst, typename Op>
[[nodiscard]] bool try_execute_vop3p_pk_binary_f32_simd(Inst &, Wavefront &, uint32_t, uint32_t,
                                                        Op) {
  return false;
}

template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_pk_binary_f32_simd(Inst &inst, Wavefront &wf, Op op) {
  return try_execute_vop3p_pk_binary_f32_simd(inst, wf, inst.inst_.op_sel, inst.inst_.op_sel_hi,
                                              op);
}

template <typename Inst, typename Op>
[[nodiscard]] bool try_execute_vop3p_pk_binary_f32_simd(Inst &, Wavefront &, Op) {
  return false;
}

/// VOP3P packed-f32 ternary fast path (v_pk_fma_f32). 3-source FMA per half;
/// same per-register native<float> read/write as the binary form. OP_SEL
/// chooses each source half independently; op_sel_hi_2 selects src2-hi.
/// neg/neg_hi bits 0/1/2 sign-flip the respective half. MODE and CLAMP match
/// the scalar helper, including FMA NaN payload selection. The helper installs
/// one guest FP environment for the instruction, so `op` must not scope itself:
/// use fma_f32_simd_unscoped.
template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_pk_ternary_f32_simd(Inst &inst, Wavefront &wf,
                                                                uint32_t op_sel, uint32_t op_sel_hi,
                                                                uint32_t op_sel_hi_2, Op op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  fp_mode::ScopedEnvironment environment(wf.fp_round_mode_f32());
  auto flush_input = [&wf](util::native<float> value) {
    return (wf.fp_denorm_mode_f32() & 1u) ? value : denormal::flush_value(value);
  };
  auto flush_output = [&wf, &inst](util::native<float> value) {
    if (inst.inst_.clamp)
      value = apply_vop3_dst_mod_f32(value, 0, 1, floating_clamp_nan_to_zero(wf),
                                     wf.fp_round_mode_f32());
    return (wf.fp_denorm_mode_f32() & 2u) ? value : denormal::flush_value(value);
  };
  constexpr std::size_t W = util::native_width_v<float>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const bool neg0_lo = inst.inst_.neg & 1u;
  const bool neg1_lo = inst.inst_.neg & 2u;
  const bool neg2_lo = inst.inst_.neg & 4u;
  const bool neg0_hi = inst.inst_.neg_hi & 1u;
  const bool neg1_hi = inst.inst_.neg_hi & 2u;
  const bool neg2_hi = inst.inst_.neg_hi & 4u;
  RegisterAccess regs(wf);
  const auto scalar_mode = wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA5 ? ScalarPairMode::Replicate32
                                                                      : ScalarPairMode::Preserve;
  auto src0 = regs.read_operand_pair32(inst.src0, exec, scalar_mode);
  auto src1 = regs.read_operand_pair32(inst.src1, exec, scalar_mode);
  auto src2 = regs.read_operand_pair32(inst.src2, exec, scalar_mode);
  auto dst = regs.write_operand_pair32(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const PkF32Halves a = read_pkf32_halves(src0, base);
    const PkF32Halves b = read_pkf32_halves(src1, base);
    const PkF32Halves c = read_pkf32_halves(src2, base);
    const util::native<float> r_lo =
        op(flush_input(pkf32_neg((op_sel & 1u) ? a.hi : a.lo, neg0_lo)),
           flush_input(pkf32_neg((op_sel & 2u) ? b.hi : b.lo, neg1_lo)),
           flush_input(pkf32_neg((op_sel & 4u) ? c.hi : c.lo, neg2_lo)));
    const util::native<float> r_hi =
        op(flush_input(pkf32_neg((op_sel_hi & 1u) ? a.hi : a.lo, neg0_hi)),
           flush_input(pkf32_neg((op_sel_hi & 2u) ? b.hi : b.lo, neg1_hi)),
           flush_input(pkf32_neg(op_sel_hi_2 ? c.hi : c.lo, neg2_hi)));
    if (dst.template store_native_pair<float>(base, flush_output(r_lo), flush_output(r_hi), chunk)
            .failed())
      return true;
  }
  return true;
}

template <typename Inst, typename Op>
[[nodiscard]] bool try_execute_vop3p_pk_ternary_f32_simd(Inst &, Wavefront &, uint32_t, uint32_t,
                                                         uint32_t, Op) {
  return false;
}

template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_pk_ternary_f32_simd(Inst &inst, Wavefront &wf, Op op) {
  return try_execute_vop3p_pk_ternary_f32_simd(inst, wf, inst.inst_.op_sel, inst.inst_.op_sel_hi,
                                               inst.inst_.op_sel_hi_2, op);
}

template <typename Inst, typename Op>
[[nodiscard]] bool try_execute_vop3p_pk_ternary_f32_simd(Inst &, Wavefront &, Op) {
  return false;
}

/// Whether a VOP3P dot widens its 16-bit float halves as F16 or BF16. The two
/// formats share the entire dot2 structure (op_sel packing and neg/neg_hi,
/// left-to-right accumulate) and differ only in how each half is widened to f32.
enum class Vop3pDotHalfFormat { F16, BF16 };

/// VOP3P v_dot2_f32_{f16,bf16} SIMD fast path. Two half-precision products plus
/// an f32 accumulator collapse into a single f32 lane: result =
/// src0_lo*src1_lo + src0_hi*src1_hi + acc (plain `*` / `+`, left-to-right —
/// matching the scalar, NOT a contracted fma). op_sel / op_sel_hi pick the
/// halves of src0/src1 (gated to the default packing op_sel == 0 &&
/// op_sel_hi == 3); neg / neg_hi flip the src0/src1 product-operand signs; on
/// the accumulator neg_hi bit 2 is ABS, applied before neg bit 2, as in the
/// scalar body. The encoded CLAMP field is ignored for
/// floating DOT instructions. NaN-input payload divergence is accepted, as for
/// the pk_fma slices.
/// @tparam Fmt Selects F16 or BF16 widening of the half-precision sources.
template <Vop3pDotHalfFormat Fmt, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_dot_f16_simd(Inst &inst, Wavefront &wf) {
  // Route integer accumulation policies through the exact scalar instruction body.
  if (isa_properties(wf.cu().arch()).float_dot_accumulation != FloatDotAccumulation::HostF32)
    return false;
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  if (packed_opsel(inst.inst_) != 0u || packed_opsel_hi(inst.inst_) != 3u)
    return false;
  // Decline exactly the sources the scalar body re-narrows, so the two paths
  // cannot disagree. src2 is a genuine f32 accumulator, so an inline constant
  // there is already correct.
  if (pk16_src_needs_narrowing(inst.inst_.src0, inst.src0.size_bits()) ||
      pk16_src_needs_narrowing(inst.inst_.src1, inst.src1.size_bits()))
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  using F = util::native<float>;
  using U = util::native<uint32_t>;
  const bool neg_src0_lo = inst.inst_.neg & 1u;
  const bool neg_src1_lo = inst.inst_.neg & 2u;
  const bool neg_src0_hi = inst.inst_.neg_hi & 1u;
  const bool neg_src1_hi = inst.inst_.neg_hi & 2u;
  // Widen one half, then flip its sign (NEG only; the product sources have no ABS).
  const auto half = [](U raw, bool negate) {
    F value;
    if constexpr (Fmt == Vop3pDotHalfFormat::BF16)
      value = util::bf16_to_f32_simd(raw);
    else
      value = util::f16_to_f32_simd(raw);
    return std::bit_cast<F>(
        source_modifier::apply<fp_format::F32>(std::bit_cast<U>(value), false, negate));
  };
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand(inst.src2, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const U raw0 = src0.template load_native<uint32_t>(base);
    const U raw1 = src1.template load_native<uint32_t>(base);
    const F acc = std::bit_cast<F>(source_modifier::apply<fp_format::F32>(
        src2.template load_native<uint32_t>(base), 2, inst.inst_.neg_hi, inst.inst_.neg));
    const F src0_lo = half(raw0 & 0xFFFFu, neg_src0_lo);
    const F src0_hi = half(raw0 >> 16, neg_src0_hi);
    const F src1_lo = half(raw1 & 0xFFFFu, neg_src1_lo);
    const F src1_hi = half(raw1 >> 16, neg_src1_hi);
    F result = src0_lo * src1_lo + src0_hi * src1_hi + acc;
    if (dst.template store_native<float>(base, result, chunk).failed())
      return true;
  }
  return true;
}

template <Vop3pDotHalfFormat Fmt, typename Inst>
[[nodiscard]] bool try_execute_vop3p_dot_f16_simd(Inst &, Wavefront &) {
  return false;
}

} // namespace amdgpu
} // namespace rocjitsu

#include "rocjitsu/isa/arch/amdgpu/shared/simd_coverage.h"

/// Probe macro emitted at the top of each SIMD-eligible execute_<mnemonic>
/// kernel by simd_codegen.py. Expands to the binary-VOP2 fast-path call and
/// an early `return` on success, keeping each generated body to a single
/// line. Variadic in the operator argument so functor lambdas (which contain
/// commas) pass through as one token sequence. Relies on the kernel's `inst`
/// and `wf` parameters being in scope.
#define ROCJITSU_TRY_SIMD_VOP2_BINARY(T, ...)                                                      \
  if (::rocjitsu::amdgpu::try_execute_binary_vop2_simd<T>(inst, wf, __VA_ARGS__)) {                \
    ::rocjitsu::amdgpu::detail::record_simd_fast_path(                                             \
        ::rocjitsu::amdgpu::SimdFastPath::VOP2_BINARY);                                            \
    return;                                                                                        \
  }

/// VOP2 binary counterpart taking one guest FP environment for the instruction,
/// for functors built on the unscoped arithmetic cores. The trailing argument is
/// its rounding: MODE.FP_ROUND for F32, nearest-even for the F16 helper.
#define ROCJITSU_TRY_SIMD_VOP2_BINARY_SCOPED(T, ...)                                               \
  if (::rocjitsu::amdgpu::try_execute_binary_vop2_simd<T, true>(inst, wf, __VA_ARGS__))            \
  return

/// VOP1 unary counterpart of ROCJITSU_TRY_SIMD_VOP2_BINARY. Variadic in the
/// operator argument so functor lambdas pass through as one token sequence.
#define ROCJITSU_TRY_SIMD_VOP1_UNARY(Tin, Tout, ...)                                               \
  if (::rocjitsu::amdgpu::try_execute_unary_vop1_simd<Tin, Tout>(inst, wf, __VA_ARGS__)) {         \
    ::rocjitsu::amdgpu::detail::record_simd_fast_path(                                             \
        ::rocjitsu::amdgpu::SimdFastPath::VOP1_UNARY);                                             \
    return;                                                                                        \
  }

/// Literal FMA/MAD VOP2 counterpart. `KEXPR` is the inline-literal bits
/// broadcast to every lane before the call. Variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOP2_TERNARY(T, KEXPR, ...)                                              \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop2_simd<T>(inst, wf, ::util::broadcast<T>(KEXPR),  \
                                                           __VA_ARGS__))                           \
  return

/// Dst-accumulate FMA/MAC VOP2 counterpart. The helper acquires vdst as a
/// read-write register view, so only these forms observe a vdst read.
#define ROCJITSU_TRY_SIMD_VOP2_TERNARY_ACC(T, KEXPR, ...)                                          \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop2_acc_simd<T>(                                    \
          inst, wf, ::util::broadcast<T>(KEXPR), __VA_ARGS__))                                     \
  return

/// Counterpart taking one guest FP environment for the instruction, for functors
/// built on the unscoped arithmetic cores. The trailing argument is its rounding.
#define ROCJITSU_TRY_SIMD_VOP2_TERNARY_SCOPED(T, KEXPR, ...)                                       \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop2_simd<T, true>(                                  \
          inst, wf, ::util::broadcast<T>(KEXPR), __VA_ARGS__))                                     \
  return

/// Accumulate counterpart of ROCJITSU_TRY_SIMD_VOP2_TERNARY_SCOPED, with the
/// same unscoped-functor contract and trailing rounding argument.
#define ROCJITSU_TRY_SIMD_VOP2_TERNARY_ACC_SCOPED(T, KEXPR, ...)                                   \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop2_acc_simd<T, true>(                              \
          inst, wf, ::util::broadcast<T>(KEXPR), __VA_ARGS__))                                     \
  return

/// 64-bit-lane VOP2 binary counterpart (v_add/mul/max_num/min_num_f64). Lane
/// type fixed to double, read/written through the split lo/hi VGPR-pair path
/// (vsrc1 as the second source). Variadic in the functor.
#if RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS
#define ROCJITSU_TRY_SIMD_VOP2_BINARY_FP64(...) static_cast<void>(inst)
#else
#define ROCJITSU_TRY_SIMD_VOP2_BINARY_FP64(...)                                                    \
  if (::rocjitsu::amdgpu::try_execute_binary_vop2_f64_simd(inst, wf, __VA_ARGS__))                 \
  return
#endif

/// 64-bit VOP2 binary counterpart on raw floating-point encodings (min/max).
/// The functor reads `native<uint64_t>` lanes. Variadic in the functor.
#if RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS
#define ROCJITSU_TRY_SIMD_VOP2_BINARY_RAW_FP64(...) static_cast<void>(inst)
#else
#define ROCJITSU_TRY_SIMD_VOP2_BINARY_RAW_FP64(...)                                                \
  if (::rocjitsu::amdgpu::try_execute_binary_vop2_f64_simd<uint64_t>(inst, wf, __VA_ARGS__))       \
  return
#endif

/// 64-bit-lane VOP1 unary counterpart. `T` is the 64-bit lane type (`double`
/// for the f64 math ops, `uint64_t` for v_mov_b64). Variadic in the functor so
/// its commas pass through as one token sequence.
#if RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS
#define ROCJITSU_TRY_SIMD_VOP1_UNARY_F64(T, ...) static_cast<void>(inst)
#else
#define ROCJITSU_TRY_SIMD_VOP1_UNARY_F64(T, ...)                                                   \
  if (::rocjitsu::amdgpu::try_execute_unary_vop1_f64_simd<T>(inst, wf, __VA_ARGS__))               \
  return
#endif

/// Mixed-width cvt counterpart, f64 source -> 32-bit dst. `Tout` is the 32-bit
/// result lane type; the functor (`native<double> -> narrow32<Tout>`) is variadic
/// so its commas pass through as one token sequence.
#if RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS
#define ROCJITSU_TRY_SIMD_CVT_F64_TO_B32(Tout, ...) static_cast<void>(inst)
#else
#define ROCJITSU_TRY_SIMD_CVT_F64_TO_B32(Tout, ...)                                                \
  if (::rocjitsu::amdgpu::try_execute_cvt_f64_to_b32_simd<Tout>(inst, wf, __VA_ARGS__)) {          \
    ::rocjitsu::amdgpu::detail::record_simd_fast_path(                                             \
        ::rocjitsu::amdgpu::SimdFastPath::CVT_F64_TO_B32);                                         \
    return;                                                                                        \
  }
#endif

/// VOP3 v_cvt_f32_f16 counterpart. Generic form reads src0.l; TRUE16 form
/// selects src0 via op_sel[0]. Both write a full f32 dword.
#define ROCJITSU_TRY_SIMD_CVT_F32_F16_VOP3()                                                       \
  if (::rocjitsu::amdgpu::try_execute_cvt_f32_f16_vop3_simd<false>(inst, wf))                      \
  return

#define ROCJITSU_TRY_SIMD_CVT_F32_F16_VOP3_TRUE16()                                                \
  if (::rocjitsu::amdgpu::try_execute_cvt_f32_f16_vop3_simd<true>(inst, wf))                       \
  return

/// VOPC compare counterpart. `T` is the 32-bit lane read type; the comparison
/// functor (which may convert/narrow inside) is variadic so its commas pass
/// through as one token sequence.
#define ROCJITSU_TRY_SIMD_VOPC(T, ...)                                                             \
  if (::rocjitsu::amdgpu::try_execute_vopc_simd<T>(inst, wf, __VA_ARGS__))                         \
  return

/// 64-bit-lane VOPC compare counterpart (f64/i64/u64). `T` is the 64-bit lane
/// read type; the comparison functor is variadic so its commas pass through.
#if RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS
#define ROCJITSU_TRY_SIMD_VOPC64(T, ...) static_cast<void>(inst)
#else
#define ROCJITSU_TRY_SIMD_VOPC64(T, ...)                                                           \
  if (::rocjitsu::amdgpu::try_execute_vopc64_simd<T>(inst, wf, __VA_ARGS__))                       \
  return
#endif

/// VOP3 integer/bitwise binary counterpart (reads src0/src1, no modifiers).
/// `T` is the 32-bit integer lane type; variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOP3_BINARY_INT(T, ...)                                                  \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_simd<T>(inst, wf, __VA_ARGS__)) {                \
    ::rocjitsu::amdgpu::detail::record_simd_fast_path(                                             \
        ::rocjitsu::amdgpu::SimdFastPath::VOP3_BINARY);                                            \
    return;                                                                                        \
  }

/// VOP3 binary counterpart for true16 source selectors with a full b32 result,
/// such as v_pack_b32_f16.
#define ROCJITSU_TRY_SIMD_VOP3_BINARY_TRUE16_SRC(T, ...)                                           \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_true16_src_simd<T>(inst, wf, __VA_ARGS__)) {     \
    ::rocjitsu::amdgpu::detail::record_simd_fast_path(                                             \
        ::rocjitsu::amdgpu::SimdFastPath::VOP3_BINARY_TRUE16_SRC);                                 \
    return;                                                                                        \
  }

/// VOP3 f32 binary counterpart (reads src0/src1, applies abs/neg/omod/clamp).
/// `T` is the 32-bit float lane type; variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOP3_BINARY_FP(T, ...)                                                   \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_fp_simd<T>(inst, wf, __VA_ARGS__))               \
  return

/// Counterpart taking one guest FP environment for the instruction, for functors
/// built on the unscoped arithmetic cores.
#define ROCJITSU_TRY_SIMD_VOP3_BINARY_FP_SCOPED(T, ...)                                            \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_fp_simd<T, true>(inst, wf, __VA_ARGS__))         \
  return

/// Raw floating-point paths apply source and output modifiers around the operation.
/// The operation handles input flushing and returns raw destination-format bits.
#define ROCJITSU_TRY_SIMD_VOP3_BINARY_RAW_FP(Fmt, ...)                                             \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_simd<uint32_t>(                                  \
          inst, wf, ::rocjitsu::amdgpu::vop3_float_operation<Fmt>(inst, wf, __VA_ARGS__)))         \
  return
#if RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS
#define ROCJITSU_TRY_SIMD_VOP3_BINARY_RAW_FP64(Fmt, ...) static_cast<void>(inst)
#else
#define ROCJITSU_TRY_SIMD_VOP3_BINARY_RAW_FP64(Fmt, ...)                                           \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_raw64_simd(                                      \
          inst, wf, ::rocjitsu::amdgpu::vop3_float_operation<Fmt>(inst, wf, __VA_ARGS__)))         \
  return
#endif

/// Unary operations whose F32/F64 result is ready for the shared output modifiers.
#define ROCJITSU_TRY_SIMD_VOP3_UNARY_ROUNDED_FP32(...)                                             \
  if (::rocjitsu::amdgpu::try_execute_unary_vop3_rounded_f32_simd(inst, wf, __VA_ARGS__))          \
  return

/// VOP3 f32 unary counterpart (reads src0, applies abs/neg/omod/clamp). `Tin`
/// and `Tout` are both float32_t; variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOP3_UNARY_FP(Tin, Tout, ...)                                            \
  if (::rocjitsu::amdgpu::try_execute_unary_vop3_fp_simd<Tin, Tout>(inst, wf, __VA_ARGS__))        \
  return

/// VOP3 raw-lane VOPC compare counterpart (32-bit lane, SGPR-pair dst). Despite
/// the _INT name, f16/f32 relations use it too: the glue applies no modifiers,
/// and a float functor applies the captured ABS/NEG fields and MODE input
/// flush itself via comparison::evaluate. `T` is the 32-bit raw lane read type;
/// variadic in the functor so its commas pass through as one token sequence.
#define ROCJITSU_TRY_SIMD_VOPC_VOP3_INT(T, ...)                                                    \
  if (::rocjitsu::amdgpu::try_execute_vopc_vop3_int_simd<T>(                                       \
          inst, wf, __VA_ARGS__, [&](uint64_t result) {                                            \
            ::rocjitsu::amdgpu::write_explicit_lane_mask(inst.vdst, wf, result);                   \
          }))                                                                                      \
  return
#define ROCJITSU_TRY_SIMD_VOPC_VOP3_INT_RESULT(WRITE_RESULT, T, ...)                               \
  if (::rocjitsu::amdgpu::try_execute_vopc_vop3_int_simd<T>(inst, wf, __VA_ARGS__, WRITE_RESULT))  \
  return

/// VOP3 raw-lane compare counterpart whose 16-bit sources select their half
/// with true16 OPSEL.
#define ROCJITSU_TRY_SIMD_VOPC_VOP3_TRUE16_INT(T, ...)                                             \
  if (::rocjitsu::amdgpu::try_execute_vopc_vop3_int_simd<T, true>(                                 \
          inst, wf, __VA_ARGS__, [&](uint64_t result) {                                            \
            ::rocjitsu::amdgpu::write_explicit_lane_mask(inst.vdst, wf, result);                   \
          }))                                                                                      \
  return
#define ROCJITSU_TRY_SIMD_VOPC_VOP3_TRUE16_INT_RESULT(WRITE_RESULT, T, ...)                        \
  if (::rocjitsu::amdgpu::try_execute_vopc_vop3_int_simd<T, true>(inst, wf, __VA_ARGS__,           \
                                                                  WRITE_RESULT))                   \
  return

/// 64-bit-lane VOP3 raw-lane VOPC compare counterpart (i64/u64/f64, SGPR-pair
/// dst). As with the 32-bit macro, f64 callers use it too, with the functor
/// applying ABS/NEG and MODE input flush via comparison::evaluate. `T` is the
/// 64-bit raw lane read type; variadic in the functor.
#if RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS
#define ROCJITSU_TRY_SIMD_VOPC64_VOP3_INT(T, ...) static_cast<void>(inst)
#define ROCJITSU_TRY_SIMD_VOPC64_VOP3_INT_RESULT(WRITE_RESULT, T, ...) static_cast<void>(inst)
#else
#define ROCJITSU_TRY_SIMD_VOPC64_VOP3_INT(T, ...)                                                  \
  if (::rocjitsu::amdgpu::try_execute_vopc64_vop3_int_simd<T>(                                     \
          inst, wf, __VA_ARGS__, [&](uint64_t result) {                                            \
            ::rocjitsu::amdgpu::write_explicit_lane_mask(inst.vdst, wf, result);                   \
          }))                                                                                      \
  return
#define ROCJITSU_TRY_SIMD_VOPC64_VOP3_INT_RESULT(WRITE_RESULT, T, ...)                             \
  if (::rocjitsu::amdgpu::try_execute_vopc64_vop3_int_simd<T>(inst, wf, __VA_ARGS__,               \
                                                              WRITE_RESULT))                       \
  return
#endif

/// VOP3 integer/bitwise ternary counterpart (reads src0/src1/src2, no
/// modifiers). `T` is the 32-bit integer lane type; variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_INT(T, ...)                                                 \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop3_simd<T>(inst, wf, __VA_ARGS__)) {               \
    ::rocjitsu::amdgpu::detail::record_simd_fast_path(                                             \
        ::rocjitsu::amdgpu::SimdFastPath::VOP3_TERNARY);                                           \
    return;                                                                                        \
  }

/// VOP3 ternary counterpart for true16 SRC0/SRC1 plus full-width SRC2/result.

/// VOP3 ternary counterpart for true16 SRC0/SRC1/SRC2 plus selected-half dst.

/// VOP3 ternary counterparts on raw floating-point encodings (min/max), with
/// the same functor contract as ROCJITSU_TRY_SIMD_VOP3_BINARY_RAW_FP.
#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_RAW_FP(Fmt, ...)                                            \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop3_simd<uint32_t>(                                 \
          inst, wf, ::rocjitsu::amdgpu::vop3_float_operation<Fmt>(inst, wf, __VA_ARGS__))) {       \
    ::rocjitsu::amdgpu::detail::record_simd_fast_path(                                             \
        ::rocjitsu::amdgpu::SimdFastPath::VOP3_TERNARY_RAW_FP);                                    \
    return;                                                                                        \
  }

/// VOP3 f32 ternary counterpart (per-source abs/neg, result omod/clamp).
/// Functor takes already-modified `native<float>` arguments; variadic.
#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP32(...)                                                   \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop3_fp_simd(inst, wf, __VA_ARGS__))                 \
  return

/// Counterpart taking one guest FP environment for the instruction, for functors
/// built on the unscoped arithmetic cores.
#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP32_SCOPED(...)                                            \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop3_fp_simd<true>(inst, wf, __VA_ARGS__))           \
  return

/// VOP3 F16 ternary: widen -> ABS/NEG -> evaluate -> round F16 -> OMOD/CLAMP.
/// DIV_FIXUP explicitly selects modifiers before narrowing. Variadic.
#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP16(...)                                                   \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop3_fp16_simd<false>(inst, wf, __VA_ARGS__))        \
  return

/// VOP3 f16 ternary counterpart for true16 source and destination halves.
#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_TRUE16_FP16(...)                                            \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop3_fp16_simd<true>(inst, wf, __VA_ARGS__))         \
  return

/// VOP3 f64 ternary counterpart (64-bit RegisterAccess reads, per-source
/// abs/neg, omod/clamp).
/// Variadic.
#if RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS
#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP64(...) static_cast<void>(inst)
#else
#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP64(...)                                                   \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop3_fp64_simd(inst, wf, __VA_ARGS__))               \
  return
#endif

/// VOP3 dst-accumulate FMA counterpart (f32). Per-isa class has no src2;
/// vdst is the accumulator. abs/neg apply to src0/src1 only.
#define ROCJITSU_TRY_SIMD_FMAC_VOP3_FP32(...)                                                      \
  if (::rocjitsu::amdgpu::try_execute_fmac_vop3_fp_simd(inst, wf, __VA_ARGS__))                    \
  return

/// Counterpart taking one guest FP environment for the instruction, for functors
/// built on the unscoped arithmetic cores.
#define ROCJITSU_TRY_SIMD_FMAC_VOP3_FP32_SCOPED(...)                                               \
  if (::rocjitsu::amdgpu::try_execute_fmac_vop3_fp_simd<true>(inst, wf, __VA_ARGS__))              \
  return

/// VOP3 f64 unary counterpart (64-bit RegisterAccess read, src0 abs/neg,
/// omod/clamp on the result). Variadic in the functor.
#if RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS
#define ROCJITSU_TRY_SIMD_VOP3_UNARY_FP64(...) static_cast<void>(inst)
#else
#define ROCJITSU_TRY_SIMD_VOP3_UNARY_FP64(...)                                                     \
  if (::rocjitsu::amdgpu::try_execute_unary_vop3_fp64_simd(inst, wf, __VA_ARGS__))                 \
  return
#endif

/// VOP3P packed-16 integer binary probe. Functor takes two u32 simd vectors
/// (each holding {low16, high16} packed) and returns the same shape with
/// the per-half op applied. The glue applies half selectors and retains
/// scalar execution for integer CLAMP.

/// VOP3P packed-16 integer ternary probe (3-source pk_mad family).

/// VOP3P packed-f32 binary probe (v_pk_add_f32 / v_pk_mul_f32). Functor takes
/// (a, b) as native<float> (neg-applied) and returns native<float>.
#define ROCJITSU_TRY_SIMD_VOP3P_PK_BINARY_F32(...)                                                 \
  if (::rocjitsu::amdgpu::try_execute_vop3p_pk_binary_f32_simd(inst, wf, __VA_ARGS__))             \
  return

/// Profile-aware form for ISA-local VOP3P bodies whose selector fields do not
/// use the canonical op_sel/op_sel_hi member names.
#define ROCJITSU_TRY_SIMD_VOP3P_PK_BINARY_F32_SELECTORS(OpSel, OpSelHi, ...)                       \
  if (::rocjitsu::amdgpu::try_execute_vop3p_pk_binary_f32_simd(                                    \
          inst, wf, static_cast<uint32_t>(OpSel), static_cast<uint32_t>(OpSelHi), __VA_ARGS__))    \
  return

/// VOP3P packed-f32 ternary probe (v_pk_fma_f32). Functor takes (a, b, c) as
/// native<float>; selectors choose each source half, including broadcasts.
#define ROCJITSU_TRY_SIMD_VOP3P_PK_TERNARY_F32(...)                                                \
  if (::rocjitsu::amdgpu::try_execute_vop3p_pk_ternary_f32_simd(inst, wf, __VA_ARGS__))            \
  return

/// Profile-aware form for ISA-local ternary VOP3P bodies whose selector fields
/// do not use the canonical op_sel/op_sel_hi/op_sel_hi_2 member names.
#define ROCJITSU_TRY_SIMD_VOP3P_PK_TERNARY_F32_SELECTORS(OpSel, OpSelHi, OpSelHi2, ...)            \
  if (::rocjitsu::amdgpu::try_execute_vop3p_pk_ternary_f32_simd(                                   \
          inst, wf, static_cast<uint32_t>(OpSel), static_cast<uint32_t>(OpSelHi),                  \
          static_cast<uint32_t>(OpSelHi2), __VA_ARGS__))                                           \
  return

/// VOP3P v_dot2_f32_{f16,bf16} SIMD probe. Arg: the half-precision widening
/// format (F16 or BF16) as a ::rocjitsu::amdgpu::Vop3pDotHalfFormat enumerator.
#define ROCJITSU_TRY_SIMD_VOP3P_DOT_F16(Fmt)                                                       \
  if (::rocjitsu::amdgpu::try_execute_vop3p_dot_f16_simd<                                          \
          ::rocjitsu::amdgpu::Vop3pDotHalfFormat::Fmt>(inst, wf))                                  \
  return

/// VOP2/VOP3 dst-accumulate f16 dot probe (v_dot2c_f32_f16). Arg: Vop3 bool.

#endif // ROCJITSU_ISA_AMDGPU_SHARED_SIMD_GLUE_H_
