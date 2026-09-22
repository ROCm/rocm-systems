// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_MMA_EXEC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_MMA_EXEC_H_

/// @file CDNA4 MFMA — thin wrapper around the shared MFMA implementation.
///
/// CDNA4 uses AccMode::Unified (same GFX9 register layout as CDNA3).

#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"

namespace rocjitsu {
namespace cdna4 {

/// CDNA4 resolve_acc defaults to Unified mode.
template <typename F>
inline uint32_t resolve_acc(uint32_t vb, uint32_t dst, int src2_ev, uint32_t &const_acc,
                            F &&get_const) {
  return amdgpu::resolve_acc<amdgpu::AccMode::Unified>(vb, dst, src2_ev, const_acc,
                                                       std::forward<F>(get_const));
}

} // namespace cdna4
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_MMA_EXEC_H_
