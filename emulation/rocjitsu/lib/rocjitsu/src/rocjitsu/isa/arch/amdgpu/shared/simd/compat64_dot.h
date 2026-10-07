// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_COMPAT64_DOT_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_COMPAT64_DOT_H_

#include "rocjitsu/isa/arch/amdgpu/shared/simd/common.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd/portable_math.h"

namespace rocjitsu::amdgpu {

/// VOP3P integer dot-product SIMD fast path (v_dot4_i32_i8 / v_dot4_u32_u8 /
/// v_dot8_i32_i4 / v_dot8_u32_u4 / v_dot2_i32_i16 / v_dot2_u32_u16). Unlike
/// the packed-16 family the destination is a single 32-bit lane (NOT packed)
/// and src2 is a per-lane accumulator; the dot reduction happens *within*
/// each lane, so the fast path vectorizes across lanes (each lane computes
/// its own reduction). ElemBits selects the sub-word width (16/8/4 -> 2/4/8
/// products per lane); Signed selects sign- vs zero-extension and, for the
/// signed forms when inst.clamp is set, saturation to [INT_MIN, INT_MAX]. Unsigned
/// clamp saturates to UINT_MAX. Clamped accumulation is widened so overflow is
/// detected before narrowing; unclamped accumulation remains in uint32_t bits
/// and wraps. For the 16-bit forms op_sel /
/// op_sel_hi pick the source halves,
/// so the fast path gates on the default packing (op_sel == 0, op_sel_hi == 3)
/// and bails otherwise; the 8/4-bit scalar bodies ignore op_sel so no gate is
/// needed there.
template <int ElemBits, bool Signed, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_dot_int_simd(Inst &inst, Wavefront &wf) {
  static_assert(ElemBits == 16 || ElemBits == 8 || ElemBits == 4, "dot ElemBits must be 16/8/4");
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  if constexpr (ElemBits == 16) {
    if (packed_opsel(inst.inst_) != 0u || packed_opsel_hi(inst.inst_) != 3u)
      return false;
  }
  const bool clamp = inst.inst_.clamp && (ElemBits != 8 || dot4_clamp_supported(wf));
  if (clamp)
    return false;
  constexpr int N = 32 / ElemBits;
  constexpr uint32_t kElemMask = (ElemBits == 16) ? 0xFFFFu : (ElemBits == 8) ? 0xFFu : 0xFu;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  using U = util::native<uint32_t>;
  using I = util::native<int32_t>;
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
    const U acc = src2.template load_native<uint32_t>(base);
    if constexpr (Signed) {
      U sum = acc;
      for (int i = 0; i < N; ++i) {
        const U ea = (raw0 >> (i * ElemBits)) & kElemMask;
        const U eb = (raw1 >> (i * ElemBits)) & kElemMask;
        const I a = std::bit_cast<I>(simd_sign_extend_u32(ea, ElemBits));
        const I b = std::bit_cast<I>(simd_sign_extend_u32(eb, ElemBits));
        sum += std::bit_cast<U>(a * b);
      }
      dst.template store_native<uint32_t>(base, sum, chunk);
    } else {
      U sum = acc;
      for (int i = 0; i < N; ++i) {
        const U ea = (raw0 >> (i * ElemBits)) & kElemMask;
        const U eb = (raw1 >> (i * ElemBits)) & kElemMask;
        sum += ea * eb;
      }
      dst.template store_native<uint32_t>(base, sum, chunk);
    }
  }
  return true;
}

/// VOP3P mixed-sign integer dot product (v_dot4_i32_iu8 / v_dot8_i32_iu4).
/// Same structure as try_execute_vop3p_dot_int_simd but the per-operand
/// signedness is chosen at RUNTIME from inst.neg (bit 0 -> src0 signed, bit 1
/// -> src1 signed) — hoisted out of the chunk loop. src2 is the int32
/// accumulator seed; clamp (when set) saturates to [INT_MIN, INT_MAX] before
/// narrowing. Unclamped accumulation wraps in uint32_t bits. The 8/4-bit scalar
/// bodies read no op_sel/neg_hi, so no gate.
template <int ElemBits, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_dot_int_mixed_simd(Inst &inst, Wavefront &wf) {
  static_assert(ElemBits == 8 || ElemBits == 4, "iu dot ElemBits must be 8/4");
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr int N = 32 / ElemBits;
  constexpr uint32_t kElemMask = (ElemBits == 8) ? 0xFFu : 0xFu;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const bool clamp = inst.inst_.clamp && (ElemBits != 8 || dot4_clamp_supported(wf));
  if (clamp)
    return false;
  const bool src0_signed = (inst.inst_.neg & 0x1u) != 0;
  const bool src1_signed = (inst.inst_.neg & 0x2u) != 0;
  using U = util::native<uint32_t>;
  using I = util::native<int32_t>;
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
    const U acc = src2.template load_native<uint32_t>(base);
    U sum = acc;
    for (int i = 0; i < N; ++i) {
      const U ea = (raw0 >> (i * ElemBits)) & kElemMask;
      const U eb = (raw1 >> (i * ElemBits)) & kElemMask;
      const I a = src0_signed ? std::bit_cast<I>(simd_sign_extend_u32(ea, ElemBits))
                              : util::stdx::static_simd_cast<I>(ea);
      const I b = src1_signed ? std::bit_cast<I>(simd_sign_extend_u32(eb, ElemBits))
                              : util::stdx::static_simd_cast<I>(eb);
      sum += std::bit_cast<U>(a * b);
    }
    dst.template store_native<uint32_t>(base, sum, chunk);
  }
  return true;
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_COMPAT64_DOT_H_
