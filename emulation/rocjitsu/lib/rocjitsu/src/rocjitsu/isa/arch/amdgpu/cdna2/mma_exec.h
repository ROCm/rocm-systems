// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA2_MMA_EXEC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA2_MMA_EXEC_H_

/// @file CDNA2 MFMA — thin wrapper around the shared MFMA implementation.
///
/// CDNA2 (GFX90A) uses AccMode::Separate — a dedicated AccVGPR register file
/// at encoding offset 256 for destinations (OPR_ACCVGPR), and 768 for src2
/// (OPR_SRC_ACCVGPR_OR_CONST). The register layout math is identical to
/// CDNA3/4 (GFX9 family) but the encoding ranges differ.

#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"

namespace rocjitsu {
namespace cdna2 {

/// CDNA2 resolve_acc defaults to Separate mode (dedicated AccVGPR file).
template <typename F>
inline uint32_t resolve_acc(uint32_t vb, uint32_t dst, int src2_ev, uint32_t &const_acc,
                            F &&get_const) {
  return amdgpu::resolve_acc<amdgpu::AccMode::Separate>(vb, dst, src2_ev, const_acc,
                                                        std::forward<F>(get_const));
}

} // namespace cdna2
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA2_MMA_EXEC_H_
