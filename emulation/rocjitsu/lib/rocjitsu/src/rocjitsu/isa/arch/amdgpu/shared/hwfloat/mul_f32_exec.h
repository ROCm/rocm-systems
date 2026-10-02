// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/hwfloat/mul_f32_policy.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd.h"

#include <algorithm>
#include <cstdint>

namespace rocjitsu::amdgpu::hwfloat {

namespace detail {

// Register staging is an optimization gate, separate from numerical policy.
inline bool admits_mul_f32_staging(const Wavefront &wf) {
  const InstructionComputeUnitView &cu = wf.cu();
  return !util::force_scalar() && !cu.observes_register_access() && !cu.debug_active();
}

template <typename View>
void stage_mul_f32_source(const View &view, uint32_t lanes, bool abs, bool neg, uint32_t *out) {
  constexpr uint32_t W = static_cast<uint32_t>(util::native_width_v<uint32_t>);
  for (uint32_t base = 0; base < lanes; base += W) {
    alignas(util::native<uint32_t>) uint32_t chunk[W];
    util::blit_to_buffer<uint32_t>(chunk, view.template load_native<uint32_t>(base));
    std::copy_n(chunk, W, out + base);
  }
  if (abs || neg) {
    const uint32_t keep = abs ? 0x7fffffffu : 0xffffffffu;
    const uint32_t flip = neg ? 0x80000000u : 0u;
    for (uint32_t lane = 0; lane < lanes; ++lane)
      out[lane] = (out[lane] & keep) ^ flip;
  }
}

/// Stage every source lane, compute results and causes for the active lanes,
/// report the causes, then write the destination. The caller qualifies the
/// operands first; zero EXEC returns before any source read or state change.
template <typename Lhs, typename Rhs, typename Dst>
bool execute_qualified_mul_f32(Wavefront &wf, const Lhs &lhs, const Rhs &rhs, const Dst &dst,
                               uint32_t abs, uint32_t neg) {
  const uint64_t exec = wf.exec();
  if (!exec)
    return false;
  const uint32_t lanes = wf.wf_size();
  RegisterAccess regs(wf);
  alignas(64) uint32_t a[kMulF32MaxLanes];
  alignas(64) uint32_t b[kMulF32MaxLanes];
  alignas(64) uint32_t result[kMulF32MaxLanes] = {};
  stage_mul_f32_source(regs.read_operand(lhs, exec), lanes, (abs & 1u) != 0, (neg & 1u) != 0, a);
  stage_mul_f32_source(regs.read_operand(rhs, exec), lanes, (abs & 2u) != 0, (neg & 2u) != 0, b);
  const MulF32Policy policy = mul_f32_policy(wf);
  const uint32_t causes = multiply_f32_wave(a, b, result, exec, lanes, policy);
  wf.raise_alu_causes(causes);
  wf.set_trapsts(wf.trapsts() | causes);

  const auto destination = regs.write_operand(dst, exec);
  constexpr uint32_t W = static_cast<uint32_t>(util::native_width_v<uint32_t>);
  for (uint32_t base = 0; base < lanes; base += W) {
    const uint64_t chunk = (exec >> base) & ((uint64_t{1} << W) - 1);
    if (chunk)
      destination.template store_native<uint32_t>(base, util::load<uint32_t>(result + base), chunk);
  }
  return true;
}

} // namespace detail

/// Execute a plain V_MUL_F32 VOP2 on gfx1100 or gfx1201 with integer-exact
/// results and measured TRAPSTS causes. Returns false, before any source read
/// or state change, for every other target, form, or wave state; the generic
/// implementation then executes the instruction.
template <typename Inst> bool try_execute_qualified_mul_f32_vop2(Inst &inst, Wavefront &wf) {
  constexpr rj_code_target_id_t target = detail::qualified_mul_f32_target<Inst>();
  if constexpr (!util::has_stdx_simd || target == ROCJITSU_CODE_TARGET_INVALID) {
    (void)inst;
    (void)wf;
    return false;
  } else {
    if (!detail::admits_mul_f32_staging(wf) || !detail::has_qualified_mul_f32_vop2_policy(inst, wf))
      return false;
    return detail::execute_qualified_mul_f32(wf, inst.src0, inst.vsrc1, inst.vdst, 0, 0);
  }
}

/// VOP3 counterpart of try_execute_qualified_mul_f32_vop2(). ABS and NEG on
/// the two sources are supported; OMOD, CLAMP, and OP_SEL are declined.
template <typename Inst> bool try_execute_qualified_mul_f32_vop3(Inst &inst, Wavefront &wf) {
  constexpr rj_code_target_id_t target = detail::qualified_mul_f32_target<Inst>();
  if constexpr (!util::has_stdx_simd || target == ROCJITSU_CODE_TARGET_INVALID) {
    (void)inst;
    (void)wf;
    return false;
  } else {
    const auto &fields = inst.inst_;
    if (!detail::admits_mul_f32_staging(wf) || !detail::has_qualified_mul_f32_vop3_policy(inst, wf))
      return false;
    return detail::execute_qualified_mul_f32(wf, inst.src0, inst.src1, inst.vdst, fields.abs,
                                             fields.neg);
  }
}

} // namespace rocjitsu::amdgpu::hwfloat
