// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/hwfloat/dx9_mul_f32.h"
#include "rocjitsu/isa/arch/amdgpu/shared/hwfloat/mul_f32_policy.h"
#include "rocjitsu/vm/amdgpu/register_access.h"

namespace rocjitsu::amdgpu::hwfloat {
namespace detail {

template <typename Inst> bool has_qualified_dx9_mul_f32_target(const Wavefront &wf) {
  constexpr rj_code_target_id_t target = qualified_mul_f32_target<Inst>();
  return target != ROCJITSU_CODE_TARGET_INVALID && wf.cu().target() == target &&
         (wf.wf_size() == 32 || wf.wf_size() == 64);
}

inline Dx9MulF32Policy dx9_mul_f32_policy(const Wavefront &wf, uint32_t omod = 0,
                                          bool clamp = false) {
  return {.round = wf.fp_round_mode_f32(),
          .denorm = wf.fp_denorm_mode_f32(),
          .rdna4 = wf.cu().target() == ROCJITSU_CODE_TARGET_GFX1201,
          .ieee = wf.ieee_mode(),
          .dx10_clamp = wf.dx10_clamp(),
          .omod = omod,
          .clamp = clamp};
}

inline uint32_t dx9_mul_f32_source(uint32_t bits, uint32_t abs, uint32_t neg, uint32_t index) {
  if (abs & (1u << index))
    bits &= 0x7fffffffu;
  if (neg & (1u << index))
    bits ^= 0x80000000u;
  return bits;
}

inline void publish_dx9_mul_f32_causes(Wavefront &wf, uint32_t causes) {
  wf.raise_alu_causes(causes);
  wf.set_trapsts(wf.trapsts() | causes);
}

} // namespace detail

/// Correctness path independent of host SIMD, debugging and register observers.
/// Preserve the original active-lane read/read/write sequence, adding causes
/// before each destination. Unsupported forms decline before register access.
template <typename Inst> bool try_execute_qualified_dx9_mul_f32_vop2(Inst &inst, Wavefront &wf) {
  if (!detail::has_qualified_dx9_mul_f32_target<Inst>(wf) ||
      detail::has_src0_modifier(inst.inst_.src0) ||
      !detail::has_plain_mul_f32_operands(inst.src0, inst.vsrc1, inst.vdst, wf))
    return false;
  RegisterAccess regs(wf);
  const uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (uint64_t{1} << lane)))
      continue;
    const auto result =
        evaluate_dx9_mul_f32(regs.read_lane(inst.src0, lane), regs.read_lane(inst.vsrc1, lane),
                             detail::dx9_mul_f32_policy(wf));
    detail::publish_dx9_mul_f32_causes(wf, result.causes);
    regs.write_lane(inst.vdst, lane, result.bits);
  }
  return true;
}

template <typename Inst> bool try_execute_qualified_dx9_mul_f32_vop3(Inst &inst, Wavefront &wf) {
  const auto &fields = inst.inst_;
  if (!detail::has_qualified_dx9_mul_f32_target<Inst>(wf) ||
      detail::has_src0_modifier(fields.src0) || detail::vop3_op_sel(fields) != 0 ||
      ((fields.abs | fields.neg) & ~3u) ||
      !detail::has_plain_mul_f32_operands(inst.src0, inst.src1, inst.vdst, wf))
    return false;
  RegisterAccess regs(wf);
  const uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (uint64_t{1} << lane)))
      continue;
    const auto result = [&]() {
      // VOP3's original generated body establishes both scopes before its
      // source callbacks. Keep that callback environment and restore it before
      // destination first-touch/allocation; integer arithmetic needs no scope.
      fp_mode::detail::ScopedFenv outer(wf.fp_round_mode_f32());
      fp_mode::detail::ScopedFenv inner(wf.fp_round_mode_f32());
      return evaluate_dx9_mul_f32(
          detail::dx9_mul_f32_source(regs.read_lane(inst.src0, lane), fields.abs, fields.neg, 0),
          detail::dx9_mul_f32_source(regs.read_lane(inst.src1, lane), fields.abs, fields.neg, 1),
          detail::dx9_mul_f32_policy(wf, fields.omod, fields.clamp));
    }();
    detail::publish_dx9_mul_f32_causes(wf, result.causes);
    regs.write_lane(inst.vdst, lane, result.bits);
  }
  return true;
}

} // namespace rocjitsu::amdgpu::hwfloat
