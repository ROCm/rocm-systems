// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_MFMA_EXEC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_MFMA_EXEC_H_

/// @file WMMA execution stubs for rdna3_5 (reuses shared MFMA register math).
///
/// RDNA WMMA uses Wave32 matrix tiles with a different register layout than
/// CDNA MFMA. The shared mfma_exec.h register mapping functions assume
/// Wave64; RDNA-specific register layout is a Phase C.8 deliverable.

#include "rocjitsu/isa/arch/amdgpu/shared/mfma_exec.h"

namespace rocjitsu {
namespace rdna3_5 {
namespace mfma {

using namespace amdgpu::mfma; // NOLINT(google-build-using-namespace)

/// RDNA WMMA resolve_acc — uses Unified mode (no separate AccVGPR file).
template <typename F>
inline uint32_t resolve_acc(uint32_t vb, uint32_t dst, int src2_ev, uint32_t &const_acc,
                            F &&get_const) {
  return amdgpu::mfma::resolve_acc<amdgpu::mfma::AccMode::Unified>(vb, dst, src2_ev, const_acc,
                                                                   std::forward<F>(get_const));
}

} // namespace mfma
} // namespace rdna3_5
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_MFMA_EXEC_H_
