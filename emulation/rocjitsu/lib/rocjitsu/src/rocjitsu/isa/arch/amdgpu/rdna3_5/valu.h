// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/fp_status.h"
#include "rocjitsu/isa/arch/amdgpu/shared/source_modifier.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

namespace rocjitsu::rdna3_5 {

inline void record_alu_causes(amdgpu::Wavefront &wave, uint32_t causes) {
  wave.set_trapsts(wave.trapsts() | causes);
  wave.raise_alu_causes(causes);
}

// GFX11's arithmetic dual slots reduce to one F32 FMA. Put the operands in
// architectural order so both NaN priority and the exception flags agree.
struct FmaInputs {
  uint32_t a, b, c;
};

inline FmaInputs dual_fma_inputs(uint32_t op, uint32_t a, uint32_t b, uint32_t c) {
  constexpr uint32_t sign = amdgpu::fp_format::F32::kSign;
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

inline uint32_t evaluate_fma(FmaInputs in, amdgpu::Wavefront &wave, bool force_flush = false) {
  return std::bit_cast<uint32_t>(amdgpu::fp_mode::arithmetic<amdgpu::fp_mode::Arithmetic::FMA>(
      std::bit_cast<float>(in.a), std::bit_cast<float>(in.b), std::bit_cast<float>(in.c),
      wave.fp_round_mode_f32(), wave.fp_denorm_mode_f32(), ROCJITSU_CODE_ARCH_RDNA3_5,
      wave.ieee_mode(), force_flush));
}

inline uint32_t fma_flags(FmaInputs in, uint32_t result, amdgpu::Wavefront &wave) {
  return amdgpu::fp_status::fma<amdgpu::fp_format::F32>(
      in.a, in.b, in.c, result, wave.fp_round_mode_f32(), wave.fp_denorm_mode_f32());
}

// Compute causes before destination clamping. An effective output scaling can
// raise a second overflow or underflow even when the arithmetic itself was exact.
enum class FmaOperation : uint32_t {
  Fma = 0,
  Multiply = 3,
  Add = 4,
  Subtract = 5,
  ReverseSubtract = 6
};

template <FmaOperation operation, typename Inst>
uint32_t arithmetic_flags(const Inst &inst, amdgpu::Wavefront &wave, const rocjitsu::Operand &rhs) {
  if (wave.dx10_clamp())
    return 0;
  uint32_t absolute = 0, negate = 0, omod = 0;
  if constexpr (requires { inst.inst_.omod; }) {
    absolute = inst.inst_.abs;
    negate = inst.inst_.neg;
    omod = amdgpu::fp_mode::effective_omod(ROCJITSU_CODE_ARCH_RDNA3_5, wave.fp_denorm_mode_f32(),
                                           wave.ieee_mode(), inst.inst_.omod);
  }
  amdgpu::RegisterAccess regs(wave);
  const uint64_t active = wave.exec() & wave.vgpr_write_mask();
  uint32_t causes = 0;
  for (uint64_t mask = active; mask; mask &= mask - 1) {
    const unsigned lane = std::countr_zero(mask);
    const auto read = [&](const rocjitsu::Operand &operand, unsigned index) {
      return amdgpu::source_modifier::apply<amdgpu::fp_format::F32>(regs.read_lane(operand, lane),
                                                                    index, absolute, negate);
    };
    const uint32_t lhs = read(inst.src0, 0);
    const uint32_t second = read(rhs, 1);
    uint32_t addend = 0;
    if constexpr (operation == FmaOperation::Fma)
      addend = read(inst.src2, 2);
    const auto in = dual_fma_inputs(static_cast<uint32_t>(operation), lhs, second, addend);
    const uint32_t result = evaluate_fma(in, wave, omod != 0);
    causes |= fma_flags(in, result, wave);
    if (omod) {
      const uint32_t scale = omod == 1 ? 0x40000000u : omod == 2 ? 0x40800000u : 0x3f000000u;
      const auto scaled = dual_fma_inputs(3, result, scale, 0);
      causes |= fma_flags(scaled, evaluate_fma(scaled, wave, true), wave);
    }
  }
  return causes;
}

} // namespace rocjitsu::rdna3_5
