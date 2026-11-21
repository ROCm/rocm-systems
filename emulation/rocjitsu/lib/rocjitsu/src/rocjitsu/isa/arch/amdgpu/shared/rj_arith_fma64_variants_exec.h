// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Observe FP64 accumulating/literal operands and publish masked rj-arith results.
#pragma once

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_float_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_vector_exec.h"

namespace rocjitsu::amdgpu::arith {

// Capture the implicit accumulator before publication, including staged destinations.
template <bool OldCdna>
inline void execute_fmac64(Wavefront &wave, uint64_t modifiers, const Operand &destination,
                           const Operand &lhs, const Operand &rhs) {
  if (!wave.exec()) {
    return;
  }
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  const uint64_t flags = float16_f64_flags(wave, wave.cu().arithmetic_flags(), OldCdna);
  RegisterAccess registers(wave);
  VectorInput64 a(registers, wave, lhs, wave.exec());
  VectorInput64 b(registers, wave, rhs, wave.exec());
  VectorInput64 c(registers, wave, destination, wave.exec());
  uint32_t state = 0;
  write_vector64_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t mask) {
        // Direct output already contains C; staged output needs the same initial value.
        // Copying C to its own direct storage leaves aliased source operands unchanged.
        for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
          if ((mask >> lane) & 1) {
            d[0][lane] = c.data[0][lane];
            d[1][lane] = c.data[1][lane];
          }
        }
        if constexpr (OldCdna) {
          return wave.wf_size() == 32
                     ? rj_arith_v_fmac_f64_w32_gfx942(flags, uint32_t(mask), modifiers, d,
                                                      a.data.data(), b.data.data(), &state)
                     : rj_arith_v_fmac_f64_w64_gfx942(flags, mask, modifiers, d, a.data.data(),
                                                      b.data.data(), &state);
        } else {
          return rj_arith_v_fmac_f64_w32_gfx1250(flags, uint32_t(mask), modifiers, d, a.data.data(),
                                                 b.data.data(), &state);
        }
      });
}

// The encoded literal is uniform FP64 data, independent of vector-register layout.
template <bool Multiply>
inline void execute_literal_fma64(Wavefront &wave, const Operand &destination, const Operand &lhs,
                                  const Operand &rhs, const Operand &literal) {
  if (!wave.exec()) {
    return;
  }
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  const uint64_t flags = float16_f64_flags(wave, wave.cu().arithmetic_flags(), false);
  RegisterAccess registers(wave);
  VectorInput64 a(registers, wave, lhs, wave.exec());
  VectorInput64 b(registers, wave, rhs, wave.exec());
  const uint64_t constant = registers.read_lane64(literal, 0);
  uint32_t state = 0;
  write_vector64_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t mask) {
        if constexpr (Multiply) {
          return rj_arith_v_fmamk_f64_w32_gfx1250(flags, uint32_t(mask), 0, d, a.data.data(),
                                                  constant, b.data.data(), &state);
        } else {
          return rj_arith_v_fmaak_f64_w32_gfx1250(flags, uint32_t(mask), 0, d, a.data.data(),
                                                  b.data.data(), constant, &state);
        }
      });
}

} // namespace rocjitsu::amdgpu::arith
