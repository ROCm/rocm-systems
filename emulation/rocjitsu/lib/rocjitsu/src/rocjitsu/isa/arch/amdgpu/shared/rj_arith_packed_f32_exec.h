// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Adapt packed FP32 word pairs to rj-arith while preserving register observers
// and guest FP32 controls; inline constants use pair32 replication.

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_float_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_vector_exec.h"

namespace rocjitsu::amdgpu::arith {

// Acquire all paired inputs before publication, preserving source/destination aliases.
inline void execute_packed_fma32(Wavefront &wave, FloatTernaryWave32Function wave32,
                                 FloatTernaryWave64Function wave64, uint64_t flags,
                                 uint64_t modifiers, const Operand &destination, const Operand &lhs,
                                 const Operand &rhs, const Operand &third, bool old_cdna) {
  if (!wave.exec()) {
    return;
  }
  flags = float32_flags(wave, flags, old_cdna);
  RegisterAccess registers(wave);
  // gfx1250 replicates scalar words; older CDNA consumes a scalar register pair.
  const auto scalar_mode = old_cdna ? ScalarPairMode::Preserve : ScalarPairMode::Replicate32;
  VectorInput64 a(registers, wave, lhs, wave.exec(), /*packed_words=*/true, scalar_mode);
  VectorInput64 b(registers, wave, rhs, wave.exec(), /*packed_words=*/true, scalar_mode);
  VectorInput64 c(registers, wave, third, wave.exec(), /*packed_words=*/true, scalar_mode);
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  write_vector64_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        return wave.wf_size() == 32 ? wave32(flags, uint32_t(write_mask), modifiers, d,
                                             a.data.data(), b.data.data(), c.data.data(), nullptr)
                                    : wave64(flags, write_mask, modifiers, d, a.data.data(),
                                             b.data.data(), c.data.data(), nullptr);
      });
}

// Execute two independent FP32 binary operations with the same pair-source policy.
inline void execute_packed_binary32(Wavefront &wave, FloatBinary32Function wave32,
                                    FloatBinary64Function wave64, uint64_t flags,
                                    uint64_t modifiers, const Operand &destination,
                                    const Operand &lhs, const Operand &rhs, bool old_cdna) {
  if (!wave.exec()) {
    return;
  }
  flags = float32_flags(wave, flags, old_cdna);
  RegisterAccess registers(wave);
  const auto scalar_mode = old_cdna ? ScalarPairMode::Preserve : ScalarPairMode::Replicate32;
  VectorInput64 a(registers, wave, lhs, wave.exec(), /*packed_words=*/true, scalar_mode);
  VectorInput64 b(registers, wave, rhs, wave.exec(), /*packed_words=*/true, scalar_mode);
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  write_vector64_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, a.data.data(), b.data.data(),
                            nullptr)
                   : wave64(flags, write_mask, modifiers, d, a.data.data(), b.data.data(), nullptr);
      });
}

} // namespace rocjitsu::amdgpu::arith
