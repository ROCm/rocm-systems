// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/hwfloat/mul_f32.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <type_traits>

namespace rocjitsu::rdna3 {
struct Isa;
} // namespace rocjitsu::rdna3

namespace rocjitsu::rdna4 {
struct Isa;
} // namespace rocjitsu::rdna4

namespace rocjitsu::amdgpu::hwfloat {
namespace detail {

/// Concrete target whose plain V_MUL_F32 status was measured for this ISA.
template <typename Inst> constexpr rj_code_target_id_t qualified_mul_f32_target() {
  if constexpr (requires { typename Inst::IsaType; }) {
    if constexpr (std::is_same_v<typename Inst::IsaType, ::rocjitsu::rdna3::Isa>)
      return ROCJITSU_CODE_TARGET_GFX1100;
    else if constexpr (std::is_same_v<typename Inst::IsaType, ::rocjitsu::rdna4::Isa>)
      return ROCJITSU_CODE_TARGET_GFX1201;
  }
  return ROCJITSU_CODE_TARGET_INVALID;
}

/// Whether src0 selects DPP, DPP8, or SDWA instead of a plain operand.
inline bool has_src0_modifier(uint32_t src0) {
  return src0 == SRC_DPP || src0 == SRC_SDWA || dpp::is_src_dpp8(src0);
}

template <typename Fields> uint32_t vop3_op_sel(const Fields &fields) {
  if constexpr (requires { fields.op_sel; })
    return fields.op_sel;
  else if constexpr (requires { fields.opsel; })
    return fields.opsel;
  else
    return ~0u;
}

/// Whether an operand resolves without a delegate to one 32-bit value per lane
/// that RegisterAccess views read and write directly. VGPRs must lie inside
/// the wave's allocation; other VGPRs keep the generic denied-access handling.
template <typename Op> bool is_plain_mul_f32_operand(const Op &op, const Wavefront &wf) {
  if (op.delegate() || !op.simd_capable() || op.size_bits() != 32 ||
      op.validate_encoding().failed())
    return false;
  // Mixed source categories can select a scalar or inline value. Their
  // is_vgpr() capability does not identify the selected register class.
  const auto reg = op.to_register_ref();
  return !reg || reg->cls != RegClass::VGPR ||
         wf.vgpr_alloc().contains(wf.vgpr_alloc().base + reg->index);
}

template <typename Inst> bool has_qualified_mul_f32_policy(const Wavefront &wf) {
  constexpr rj_code_target_id_t target = qualified_mul_f32_target<Inst>();
  return target != ROCJITSU_CODE_TARGET_INVALID && wf.cu().target() == target && !wf.dx10_clamp() &&
         (wf.wf_size() == 32 || wf.wf_size() == 64);
}

inline MulF32Policy mul_f32_policy(const Wavefront &wf) {
  return {.round = wf.fp_round_mode_f32(),
          .denorm = wf.fp_denorm_mode_f32(),
          .quiet_nan = fp_mode::quiets_nan(wf.cu().arch(), wf.ieee_mode())};
}

template <typename Lhs, typename Rhs, typename Dst>
bool has_plain_mul_f32_operands(const Lhs &lhs, const Rhs &rhs, const Dst &dst,
                                const Wavefront &wf) {
  return is_plain_mul_f32_operand(lhs, wf) && is_plain_mul_f32_operand(rhs, wf) &&
         is_plain_mul_f32_operand(dst, wf) && dst.is_vgpr() && dst.is_writable();
}

// Policy qualification is independent of host SIMD and register observation.
// Staged execution applies its separate observer/debug gate after this check.
template <typename Inst>
bool has_qualified_mul_f32_vop2_policy(const Inst &inst, const Wavefront &wf) {
  return has_qualified_mul_f32_policy<Inst>(wf) && !has_src0_modifier(inst.inst_.src0) &&
         has_plain_mul_f32_operands(inst.src0, inst.vsrc1, inst.vdst, wf);
}

template <typename Inst>
bool has_qualified_mul_f32_vop3_policy(const Inst &inst, const Wavefront &wf) {
  const auto &fields = inst.inst_;
  return has_qualified_mul_f32_policy<Inst>(wf) && !has_src0_modifier(fields.src0) &&
         fields.omod == 0 && fields.clamp == 0 && vop3_op_sel(fields) == 0 &&
         ((fields.abs | fields.neg) & ~3u) == 0 &&
         has_plain_mul_f32_operands(inst.src0, inst.src1, inst.vdst, wf);
}

} // namespace detail
} // namespace rocjitsu::amdgpu::hwfloat
