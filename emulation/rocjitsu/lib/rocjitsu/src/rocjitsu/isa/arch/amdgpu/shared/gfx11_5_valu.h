// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/fp_status.h"
#include "rocjitsu/isa/arch/amdgpu/shared/source_modifier.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

namespace rocjitsu::amdgpu::gfx11_5_valu {

inline void record(Wavefront &wave, uint32_t causes) {
  wave.set_trapsts(wave.trapsts() | causes);
  wave.raise_alu_causes(causes);
}

// GFX11's arithmetic dual slots reduce to one F32 FMA. Put the operands in
// architectural order so both NaN priority and the exception flags agree.
struct FmaInputs {
  uint32_t a, b, c;
};

inline FmaInputs dual_fma_inputs(uint32_t op, uint32_t a, uint32_t b, uint32_t c) {
  constexpr uint32_t sign = fp_format::F32::kSign;
  switch (op) {
  case 0: // FMAC
  case 1: // FMAAK
    return {a, b, c};
  case 2: // FMAMK
    return {a, c, b};
  case 3: // MUL
    return {a, b, (a ^ b) & sign};
  case 4: // ADD
    return {a, 0x3f800000u, b};
  case 5: // SUB
    return {a, 0x3f800000u, b ^ sign};
  default: // SUBREV (6)
    assert(op == 6);
    return {a ^ sign, 0x3f800000u, b};
  }
}

inline uint32_t evaluate(FmaInputs in, Wavefront &wave, bool force_flush = false) {
  return std::bit_cast<uint32_t>(fp_mode::arithmetic<fp_mode::Arithmetic::FMA>(
      std::bit_cast<float>(in.a), std::bit_cast<float>(in.b), std::bit_cast<float>(in.c),
      wave.fp_round_mode_f32(), wave.fp_denorm_mode_f32(), ROCJITSU_CODE_ARCH_RDNA3_5,
      wave.ieee_mode(), force_flush));
}

inline uint32_t flags(FmaInputs in, uint32_t result, Wavefront &wave) {
  return fp_status::fma<fp_format::F32>(in.a, in.b, in.c, result, wave.fp_round_mode_f32(),
                                        wave.fp_denorm_mode_f32());
}

// Compute causes before destination clamping. An effective output scaling can
// raise a second overflow or underflow even when the arithmetic itself was exact.
template <bool Multiply, typename Inst>
uint32_t binary_flags(const Inst &inst, Wavefront &wave, const Operand &rhs) {
  if (wave.dx10_clamp())
    return 0;
  uint32_t absolute = 0, negate = 0, omod = 0;
  if constexpr (requires { inst.inst_.omod; }) {
    absolute = inst.inst_.abs;
    negate = inst.inst_.neg;
    omod = fp_mode::effective_omod(ROCJITSU_CODE_ARCH_RDNA3_5, wave.fp_denorm_mode_f32(),
                                   wave.ieee_mode(), inst.inst_.omod);
  }
  RegisterAccess regs(wave);
  const uint64_t active = wave.exec() & wave.vgpr_write_mask();
  uint32_t causes = 0;
  for (uint64_t mask = active; mask; mask &= mask - 1) {
    const unsigned lane = std::countr_zero(mask);
    auto in = dual_fma_inputs(
        Multiply ? 3 : 4,
        source_modifier::apply<fp_format::F32>(regs.read_lane(inst.src0, lane), 0, absolute,
                                               negate),
        source_modifier::apply<fp_format::F32>(regs.read_lane(rhs, lane), 1, absolute, negate), 0);
    const uint32_t result = evaluate(in, wave, omod != 0);
    causes |= flags(in, result, wave);
    if (omod) {
      const uint32_t scale = omod == 1 ? 0x40000000u : omod == 2 ? 0x40800000u : 0x3f000000u;
      const auto scaled = dual_fma_inputs(3, result, scale, 0);
      causes |= flags(scaled, evaluate(scaled, wave, true), wave);
    }
  }
  return causes;
}

} // namespace rocjitsu::amdgpu::gfx11_5_valu
