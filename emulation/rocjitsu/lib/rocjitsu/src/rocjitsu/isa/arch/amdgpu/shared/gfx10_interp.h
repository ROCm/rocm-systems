// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"
#include "rocjitsu/vm/amdgpu/lds.h"

namespace rocjitsu::amdgpu {

enum class Gfx10Interp { Mov, P1, P2, P1LL, P1LV, P2F16 };

// M0's primitive mask assigns each quad to one primitive. Attributes contain
// one 12-dword block per primitive, with interleaved P0/P10 then the four P20s.
template <Gfx10Interp Op, typename Inst> void execute_gfx10_interp(Inst &inst, Wavefront &wf) {
  if (!wf.exec())
    return;
  constexpr bool extended = requires { inst.inst_.src1; };
  constexpr bool half =
      Op == Gfx10Interp::P1LL || Op == Gfx10Interp::P1LV || Op == Gfx10Interp::P2F16;
  const uint32_t packed = [&] {
    if constexpr (extended)
      return uint32_t(inst.inst_.src0);
    else
      return uint32_t(inst.inst_.attr | (inst.inst_.attrchan << 6));
  }();
  const unsigned attr = packed & 63, channel = (packed >> 6) & 3;
  const unsigned half_shift = (packed >> 8) * 16;
  const uint32_t m0 = wf.m0();
  if ((m0 >> 31) || attr > 32) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  const Operand &coordinate = [&]() -> const Operand & {
    if constexpr (extended)
      return inst.src1;
    else
      return inst.vsrc;
  }();
  unsigned omod = 0;
  bool clamp = false;
  if constexpr (extended) {
    clamp = inst.inst_.clamp;
    omod = fp_mode::effective_omod(wf.cu().arch(), wf.fp_denorm_mode_f32(), wf.ieee_mode(),
                                   inst.inst_.omod);
  }
  RegisterAccess regs(wf);
  const unsigned quads = wf.wf_size() / 4;
  const uint32_t starts = (m0 >> 16) & ((1u << (quads - 1)) - 1);
  const unsigned primitives = 1 + std::popcount(starts);
  auto half_value = [&](uint32_t value) {
    return util::f16_to_f32(fp_mode::detail::flush_input_f16(uint16_t(value >> half_shift),
                                                             wf.fp_denorm_mode_f16_f64()));
  };
  auto modify = [&](float value, unsigned index) {
    if constexpr (extended)
      return source_modifier::apply_to_float(value, index, inst.inst_.abs, inst.inst_.neg);
    else
      return value;
  };
  auto finish = [&](float value) {
    if (omod)
      value = fp_mode::apply_omod_f32(value, omod, wf.fp_round_mode_f32());
    return std::bit_cast<uint32_t>(clamp ? clamp_floating_result(value, wf) : value);
  };

  for (unsigned quad = 0; quad < quads; ++quad) {
    const uint64_t mask = (wf.exec() >> (4 * quad)) & 15;
    if (!mask)
      continue;
    const unsigned primitive = std::popcount(starts & ((1u << quad) - 1));
    const unsigned offset = (m0 & 0xffffu) + 48 * (attr * primitives + primitive);
    auto coefficient = [&](unsigned slot) -> uint32_t {
      const unsigned address = offset + 4 * slot;
      return address + 4 <= wf.lds_size() ? wf.lds().read32(wf.lds_base() + address) : 0;
    };
    constexpr bool second = Op == Gfx10Interp::P2 || Op == Gfx10Interp::P2F16;
    unsigned slot = second ? 8 + channel : 2 * channel + 1;
    if constexpr (Op == Gfx10Interp::Mov) {
      constexpr unsigned p10 = 0, p20 = 1;
      const unsigned selector = coordinate.encoding_value();
      slot = selector == p10 ? 2 * channel + 1 : selector == p20 ? 8 + channel : 2 * channel;
    }
    const uint32_t slope = coefficient(slot);
    uint32_t origin = 0;
    if constexpr (Op == Gfx10Interp::P1 || Op == Gfx10Interp::P1LL)
      origin = coefficient(2 * channel);
    for (unsigned i = 0; i < 4; ++i) {
      if (!(mask & (1u << i)))
        continue;
      const unsigned lane = quad * 4 + i;
      if constexpr (Op == Gfx10Interp::Mov) {
        regs.write_lane(inst.vdst, lane, finish(std::bit_cast<float>(slope)));
      } else {
        const float a = half ? half_value(slope) : std::bit_cast<float>(slope);
        const float b = modify(std::bit_cast<float>(regs.read_lane(coordinate, lane)), 1);
        float c;
        if constexpr (Op == Gfx10Interp::P1LV)
          c = modify(half_value(regs.read_lane(inst.src2, lane)), 2);
        else if constexpr (Op == Gfx10Interp::P2F16)
          c = modify(std::bit_cast<float>(regs.read_lane(inst.src2, lane)), 2);
        else if constexpr (Op == Gfx10Interp::P2)
          c = std::bit_cast<float>(regs.read_lane(inst.vdst, lane));
        else
          c = half ? half_value(origin) : std::bit_cast<float>(origin);
        if constexpr (Op == Gfx10Interp::P2F16) {
          fp_mode::ScopedEnvironment environment(0);
          const bool flush = !(wf.fp_denorm_mode_f32() & 1u);
          uint16_t result = fp_mode::detail::fma_f32_to_f16_nearest_environment(
              a, flush ? denormal::flush_value(b) : b, flush ? denormal::flush_value(c) : c,
              wf.fp_round_mode_f16_f64(), clamp, wf.fp16_ovfl(), floating_clamp_nan_to_zero(wf));
          if (!(wf.fp_denorm_mode_f16_f64() & 2u) && (result & 0x7c00u) == 0)
            result &= 0x8000u;
          const bool high = vop3_opsel(inst.inst_) & 8u;
          regs.write_lane_masked(inst.vdst, lane, uint32_t(result) << (high ? 16 : 0),
                                 high ? 12 : 3);
        } else {
          const auto result = fp_mode::arithmetic<fp_mode::Arithmetic::FMA>(
              a, b, c, wf.fp_round_mode_f32(), wf.fp_denorm_mode_f32(), wf.cu().arch(),
              wf.ieee_mode());
          regs.write_lane(inst.vdst, lane, finish(result));
        }
      }
    }
  }
}

} // namespace rocjitsu::amdgpu
