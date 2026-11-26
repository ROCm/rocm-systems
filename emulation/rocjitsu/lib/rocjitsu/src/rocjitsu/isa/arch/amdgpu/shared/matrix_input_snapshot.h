// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file matrix_input_snapshot.h
/// @brief Matrix A/B operands copied when an asynchronous MMA was published.

#pragma once

#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief Published copies of an MMA's factor registers.
/// @details The issuer may overwrite the live A/B registers, or the scale
/// registers of a block-scaled MFMA, while a helper still executes the MMA
/// (write-after-read). Accumulator and result registers are never
/// snapshotted: C is read and D written in the live register file.
struct MatrixInputSnapshot {
  struct Operand {
    uint32_t base = 0; ///< Physical VGPR index as resolved by the handler.
    uint32_t regs = 0; ///< Zero when the operand reads live registers.
    const uint32_t *lanes = nullptr; ///< `regs` registers of wave-size lanes each.
  };
  Operand a, b, scale_a, scale_b;
};

/// Set by a helper around each instruction it executes; null on every issuer.
inline thread_local const MatrixInputSnapshot *matrix_input_snapshot = nullptr;

/// Set by the issuer around a renamed load so LdsInlineAccess and
/// memory_pipeline writebacks store directly into the wave's scratch VGPR slot
/// instead of VgprFile.
inline thread_local uint8_t *rename_dst_override = nullptr;
inline thread_local uint32_t rename_dst_arch_phys = 0;
inline thread_local uint32_t rename_dst_regs = 0;

/// Resolve a physical destination VGPR range against the active rename override.
/// Returns the byte pointer into the wave's scratch slot if the full range lies
/// within the active rename target, or nullptr to write architectural VgprFile.
inline uint8_t *resolve_rename_dst(uint32_t dst_phys, uint32_t num_regs, uint32_t wf_size) {
  uint8_t *base = rename_dst_override;
  if (!base) [[likely]]
    return nullptr;
  if (dst_phys < rename_dst_arch_phys ||
      dst_phys + num_regs > rename_dst_arch_phys + rename_dst_regs)
    return nullptr;
  return base + size_t(dst_phys - rename_dst_arch_phys) * wf_size * sizeof(uint32_t);
}

} // namespace rocjitsu::amdgpu
