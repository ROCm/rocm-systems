// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA1_MMA_EXEC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA1_MMA_EXEC_H_

/// @file CDNA1 MFMA — thin wrapper around the shared MFMA implementation.
///
/// CDNA1 (GFX908) uses AccMode::VgprOnly — there is no dedicated AccVGPR
/// register file. MFMA outputs go to regular VGPRs. The src2 accumulator
/// can be a VGPR (encoding 256-511) or an inline constant (0-255).

#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"

namespace rocjitsu {
namespace cdna1 {

/// CDNA1 resolve_acc defaults to VgprOnly mode (no AccVGPR file).
template <typename F>
inline uint32_t resolve_acc(uint32_t vb, uint32_t dst, int src2_ev, uint32_t &const_acc,
                            F &&get_const) {
  return amdgpu::resolve_acc<amdgpu::AccMode::VgprOnly>(vb, dst, src2_ev, const_acc,
                                                        std::forward<F>(get_const));
}

} // namespace cdna1
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA1_MMA_EXEC_H_
