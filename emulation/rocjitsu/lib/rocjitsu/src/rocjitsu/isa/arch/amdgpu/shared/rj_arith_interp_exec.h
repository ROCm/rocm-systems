// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_vector_exec.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cstdint>

namespace rocjitsu::amdgpu::arith {

// GFX11.5 interpolation is one mixed FMA with fixed quad broadcasts. Reuse
// the GFX11 exact kernel for NaN priority, guest rounding, and packed writes.
// VINTERP never raises FP exceptions, including under the strictest policy.
template <bool P2, bool Half, bool Rtz, typename Mask>
inline int gfx115_interp(uint64_t flags, Mask exec, uint64_t modifiers, uint32_t *const *dst,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  constexpr unsigned width = sizeof(Mask) * 8;
  std::array<uint32_t, width> coefficient{}, origin{};
  for (Mask active = exec; active; active &= active - 1) {
    const unsigned lane = std::countr_zero(active);
    const unsigned quad = lane & ~3u;
    coefficient[lane] = a[0][quad + (P2 ? 2 : 1)];
    origin[lane] = c[0][P2 ? lane : quad];
  }
  const uint32_t *ap = coefficient.data(), *cp = origin.data();
  flags = (flags & ~RJ_ARITH_SEMANTICS_MASK) | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
  if constexpr (Rtz)
    flags = (flags & ~RJ_ARITH_FP_ROUND_MASK) | RJ_ARITH_FP_ROUND_ZERO;
  if constexpr (Half)
    modifiers |= RJ_ARITH_MIX_F16_A | (P2 ? 0 : RJ_ARITH_MIX_F16_C);
  const bool high = modifiers & RJ_ARITH_ALU_HIGH_D;
  modifiers &= ~RJ_ARITH_ALU_HIGH_D;
  if constexpr (width == 32) {
    const auto fma = Half && P2
                         ? (high ? rj_arith_v_fma_mixhi_f16_w32 : rj_arith_v_fma_mixlo_f16_w32)
                         : rj_arith_v_fma_mix_f32_w32;
    return fma(flags, exec, modifiers, dst, &ap, b, &cp, nullptr);
  } else {
    const auto fma = Half && P2
                         ? (high ? rj_arith_v_fma_mixhi_f16_w64 : rj_arith_v_fma_mixlo_f16_w64)
                         : rj_arith_v_fma_mix_f32_w64;
    return fma(flags, exec, modifiers, dst, &ap, b, &cp, nullptr);
  }
}

// Observe the quad-broadcast source lanes even when they are inactive in EXEC.
// Acquire every input before writes, and preserve the other half for FP16 P2.
// Interpolation uses guest FP controls and has no FP-status output.
template <bool P2, bool Half, typename Wave32, typename Wave64>
inline void execute_interp(Wavefront &wave, Wave32 wave32, Wave64 wave64, uint64_t flags,
                           uint64_t modifiers, const Operand &destination, const Operand &lhs,
                           const Operand &rhs, const Operand &third) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!exec_mask)
    return;
  const uint64_t quads =
      (exec_mask | (exec_mask >> 1) | (exec_mask >> 2) | (exec_mask >> 3)) & 0x1111111111111111ULL;
  if (Half && wave.fp16_ovfl())
    flags |= RJ_ARITH_FP16_OVFL;
  const auto denorm32 = wave.fp_denorm_mode_f32();
  const auto denorm16 = wave.fp_denorm_mode_f16_f64();
  flags |= uint64_t(Half && P2 ? wave.fp_round_mode_f16_f64() : wave.fp_round_mode_f32()) << 24;
  if (!(denorm32 & 1))
    flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  if (Half && ((denorm16 ^ denorm32) & 1))
    flags |= denorm16 & 1 ? RJ_ARITH_FP_PRESERVE_INPUT_DENORMALS_F16
                          : RJ_ARITH_FP_FLUSH_INPUT_DENORMALS_F16;
  if (!((Half && P2 ? denorm16 : denorm32) & 2))
    flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, lhs, quads << (P2 ? 2 : 1),
                  Half ? (modifiers & RJ_ARITH_ALU_HIGH_A ? 0xc : 0x3) : 0xf);
  VectorInput32 b(registers, wave, rhs, exec_mask);
  VectorInput32 c(registers, wave, third, P2 ? exec_mask : quads,
                  Half && !P2 ? (modifiers & RJ_ARITH_ALU_HIGH_C ? 0xc : 0x3) : 0xf);
  write_vector32_result(
      wave, registers, destination, exec_mask,
      [&](uint32_t *const *d, uint64_t write_mask) {
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, &c.data)
                   : wave64(flags, write_mask, modifiers, d, &a.data, &b.data, &c.data);
      },
      /*allow_direct=*/!(Half && P2),
      Half && P2 ? (modifiers & RJ_ARITH_ALU_HIGH_D ? 0xc : 0x3) : 0xf);
}

} // namespace rocjitsu::amdgpu::arith
