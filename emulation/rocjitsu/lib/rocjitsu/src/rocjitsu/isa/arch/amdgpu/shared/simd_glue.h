// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Raw operand selection, lane-mask writes and packed-U64 shifts for instruction
// adapters. CLAMP helpers remain for gfx10 interpolation.

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_SIMD_GLUE_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_SIMD_GLUE_H_

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_static_resolve.h"
#include "rocjitsu/isa/execution_backend.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd_control.h"

#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace rocjitsu {
namespace amdgpu {

// These numerical modifiers remain live in gfx10 interpolation.
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

/// Module-local override that disables SIMD fast paths. Initialized from
/// RJ_FORCE_SCALAR at module load; test hooks may override the local copy.
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

// Packed-U64 fallback shifts use only the low six count bits.
inline uint64_t lshl_masked(uint64_t value, uint64_t count) { return value << (count & 63u); }

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
