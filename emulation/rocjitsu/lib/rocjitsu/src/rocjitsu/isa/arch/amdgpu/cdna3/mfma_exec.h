// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_MFMA_EXEC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_MFMA_EXEC_H_

/// @file CDNA3 MFMA — thin wrapper around the shared MFMA implementation.
///
/// CDNA3 uses AccMode::Unified (VGPR and AccVGPR in a single unified file).

#include "rocjitsu/isa/arch/amdgpu/shared/mfma_exec.h"

namespace rocjitsu {
namespace cdna3 {

/// CDNA3 resolve_acc defaults to Unified mode.
template <typename F>
inline uint32_t resolve_acc(uint32_t vb, uint32_t dst, int src2_ev, uint32_t &const_acc,
                            F &&get_const) {
  return amdgpu::resolve_acc<amdgpu::AccMode::Unified>(vb, dst, src2_ev, const_acc,
                                                       std::forward<F>(get_const));
}

} // namespace cdna3
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_MFMA_EXEC_H_
