// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "rj_arith/rj_arith.h"

#include <bit>
#include <cstdint>

namespace rocjitsu::amdgpu {

enum class Gfx10Interp { Mov, P1, P2, P1LL, P1LV, P2F16 };

// Fetch LDS coefficients and stage operands before delegating arithmetic to rj-arith.
// M0's primitive mask assigns each quad to one primitive. Attributes contain
// one 12-dword block per primitive, with interleaved P0/P10 then the four P20s.
template <Gfx10Interp Op, typename Inst> void execute_gfx10_interp(Inst &inst, Wavefront &wf) {
  if (!wf.exec()) {
    return;
  }
  constexpr bool extended = requires { inst.inst_.src1; };
  constexpr bool half =
      Op == Gfx10Interp::P1LL || Op == Gfx10Interp::P1LV || Op == Gfx10Interp::P2F16;
  const uint32_t packed = [&] {
    if constexpr (extended) {
      return uint32_t(inst.inst_.src0);
    } else {
      return uint32_t(inst.inst_.attr | (inst.inst_.attrchan << 6));
    }
  }();
  const unsigned attr = packed & 63, channel = (packed >> 6) & 3;
  const unsigned half_shift = (packed >> 8) * 16;
  const uint32_t m0 = wf.m0();
  if ((m0 >> 31) || attr > 32) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  const Operand &coordinate = [&]() -> const Operand & {
    if constexpr (extended) {
      return inst.src1;
    } else {
      return inst.vsrc;
    }
  }();
  uint64_t mode = half && half_shift ? RJ_ARITH_ALU_HIGH_A : 0;
  if constexpr (extended) {
    if (inst.inst_.clamp) {
      mode |= RJ_ARITH_ALU_CLAMP;
    }
    mode |= uint64_t(inst.inst_.omod) << 6;
    if (inst.inst_.abs & 2) {
      mode |= RJ_ARITH_ALU_ABS_B;
    }
    if (inst.inst_.abs & 4) {
      mode |= RJ_ARITH_ALU_ABS_C;
    }
    if (inst.inst_.neg & 2) {
      mode |= RJ_ARITH_ALU_NEG_B;
    }
    if (inst.inst_.neg & 4) {
      mode |= RJ_ARITH_ALU_NEG_C;
    }
    if constexpr (Op == Gfx10Interp::P2F16) {
      if (vop3_opsel(inst.inst_) & 8u) {
        mode |= RJ_ARITH_ALU_HIGH_D;
      }
    }
  }
  uint64_t flags = wf.cu().arithmetic_flags();
  constexpr bool half_output = Op == Gfx10Interp::P2F16;
  if (half_output && wf.fp16_ovfl()) {
    flags |= RJ_ARITH_FP16_OVFL;
  }
  flags |= uint64_t(half_output ? wf.fp_round_mode_f16_f64() : wf.fp_round_mode_f32()) << 24;
  if (!(wf.fp_denorm_mode_f32() & 1)) {
    flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  }
  if (!((half_output ? wf.fp_denorm_mode_f16_f64() : wf.fp_denorm_mode_f32()) & 2)) {
    flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  }
  flags |= (wf.fp_denorm_mode_f16_f64() & 1) ? RJ_ARITH_FP_PRESERVE_INPUT_DENORMALS_F16
                                             : RJ_ARITH_FP_FLUSH_INPUT_DENORMALS_F16;
  if (wf.ieee_mode()) {
    flags |= RJ_ARITH_FP_IEEE_MODE;
  }
  if (!wf.dx10_clamp()) {
    flags |= RJ_ARITH_FP_DX10_CLAMP_OFF;
  }
  alignas(64) uint32_t coefficients[64]{}, coordinates[64]{}, addends[64]{}, result[64]{};
  RegisterAccess regs(wf);
  const unsigned quads = wf.wf_size() / 4;
  const uint32_t starts = (m0 >> 16) & ((1u << (quads - 1)) - 1);
  const unsigned primitives = 1 + std::popcount(starts);
  for (unsigned quad = 0; quad < quads; ++quad) {
    const uint64_t mask = (wf.exec() >> (4 * quad)) & 15;
    if (!mask) {
      continue;
    }
    const unsigned primitive = std::popcount(starts & ((1u << quad) - 1));
    const unsigned offset = (m0 & 0xffffu) + 48 * (attr * primitives + primitive);
    auto coefficient = [&](unsigned slot) -> uint32_t {
      const unsigned address = offset + 4 * slot;
      const uint32_t physical = wf.lds_base() + address;
      if (address + 4 > wf.lds_size() || uint64_t{physical} + 4 > wf.lds().size_bytes()) {
        return 0;
      }
      wf.observe_lds_direct_read(physical, 4, quad * 4 + std::countr_zero(mask));
      return wf.lds().read32(physical);
    };
    // RocJITsu graphics frontends supply 12 LDS DWORDs per primitive and attribute:
    // P0/P10 interleaved for XYZW, followed by all four P20 coefficients.
    constexpr bool second = Op == Gfx10Interp::P2 || Op == Gfx10Interp::P2F16;
    unsigned slot = second ? 8 + channel : 2 * channel + 1;
    if constexpr (Op == Gfx10Interp::Mov) {
      constexpr unsigned p10 = 0, p20 = 1;
      const unsigned selector = coordinate.encoding_value();
      slot = selector == p10 ? 2 * channel + 1 : selector == p20 ? 8 + channel : 2 * channel;
    }
    const uint32_t slope = coefficient(slot);
    uint32_t origin = 0;
    if constexpr (Op == Gfx10Interp::P1 || Op == Gfx10Interp::P1LL) {
      origin = coefficient(2 * channel);
    }
    for (unsigned lane_in_quad = 0; lane_in_quad < 4; ++lane_in_quad) {
      if (!(mask & (1u << lane_in_quad))) {
        continue;
      }
      const unsigned lane = quad * 4 + lane_in_quad;
      coefficients[lane] = slope;
      if constexpr (Op != Gfx10Interp::Mov) {
        coordinates[lane] = regs.read_lane(coordinate, lane);
        if constexpr (Op == Gfx10Interp::P1LV || Op == Gfx10Interp::P2F16) {
          addends[lane] = regs.read_lane(inst.src2, lane);
        } else if constexpr (Op == Gfx10Interp::P2) {
          addends[lane] = regs.read_lane(inst.vdst, lane);
        } else {
          addends[lane] = origin;
        }
      }
    }
  }
  const uint32_t *a = coefficients, *b = coordinates, *c = addends;
  uint32_t *d = result;
  const auto invoke = [&]<int Lanes>() {
    if constexpr (Op == Gfx10Interp::Mov) {
      if constexpr (Lanes == 32) {
        return rj_arith_v_interp_mov_f32_w32_gfx10(flags, uint32_t(wf.exec()), mode, &d, &a, &b,
                                                   &c);
      } else {
        return rj_arith_v_interp_mov_f32_w64_gfx10(flags, wf.exec(), mode, &d, &a, &b, &c);
      }
    }
    if constexpr (Op == Gfx10Interp::P1) {
      if constexpr (Lanes == 32) {
        return rj_arith_v_interp_p1_f32_w32_gfx10(flags, uint32_t(wf.exec()), mode, &d, &a, &b, &c);
      } else {
        return rj_arith_v_interp_p1_f32_w64_gfx10(flags, wf.exec(), mode, &d, &a, &b, &c);
      }
    }
    if constexpr (Op == Gfx10Interp::P2) {
      if constexpr (Lanes == 32) {
        return rj_arith_v_interp_p2_f32_w32_gfx10(flags, uint32_t(wf.exec()), mode, &d, &a, &b, &c);
      } else {
        return rj_arith_v_interp_p2_f32_w64_gfx10(flags, wf.exec(), mode, &d, &a, &b, &c);
      }
    }
    if constexpr (Op == Gfx10Interp::P1LL) {
      if constexpr (Lanes == 32) {
        return rj_arith_v_interp_p1ll_f16_w32_gfx10(flags, uint32_t(wf.exec()), mode, &d, &a, &b,
                                                    &c);
      } else {
        return rj_arith_v_interp_p1ll_f16_w64_gfx10(flags, wf.exec(), mode, &d, &a, &b, &c);
      }
    }
    if constexpr (Op == Gfx10Interp::P1LV) {
      if constexpr (Lanes == 32) {
        return rj_arith_v_interp_p1lv_f16_w32_gfx10(flags, uint32_t(wf.exec()), mode, &d, &a, &b,
                                                    &c);
      } else {
        return rj_arith_v_interp_p1lv_f16_w64_gfx10(flags, wf.exec(), mode, &d, &a, &b, &c);
      }
    }
    if constexpr (Op == Gfx10Interp::P2F16) {
      if constexpr (Lanes == 32) {
        return rj_arith_v_interp_p2_f16_w32_gfx10(flags, uint32_t(wf.exec()), mode, &d, &a, &b, &c);
      } else {
        return rj_arith_v_interp_p2_f16_w64_gfx10(flags, wf.exec(), mode, &d, &a, &b, &c);
      }
    }
  };
  const int status =
      wf.wf_size() == 32 ? invoke.template operator()<32>() : invoke.template operator()<64>();
  if (status != RJ_ARITH_SUCCESS) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  for (unsigned lane = 0; lane < wf.wf_size(); ++lane) {
    if (!((wf.exec() >> lane) & 1)) {
      continue;
    }
    if constexpr (half_output) {
      regs.write_lane_masked(inst.vdst, lane, result[lane], mode & RJ_ARITH_ALU_HIGH_D ? 12 : 3);
    } else {
      regs.write_lane(inst.vdst, lane, result[lane]);
    }
  }
}

} // namespace rocjitsu::amdgpu
