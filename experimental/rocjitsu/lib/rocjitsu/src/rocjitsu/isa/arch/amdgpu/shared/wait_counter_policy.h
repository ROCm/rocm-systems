// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Automatically generated — do not modify.
// Regenerate via: python -m amdisa --multi ... --gen-all

/// @file wait_counter_policy.h
/// @brief Arch-specific wait counter type selection for AMDGPU memory ops.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_WAIT_COUNTER_POLICY_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_WAIT_COUNTER_POLICY_H_

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/vm/amdgpu/wait_counters.h"

namespace rocjitsu {
namespace amdgpu {

/// @brief Return true if @p arch uses a dedicated counter for vector stores.
inline bool uses_split_vector_store_counter(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA4:
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return true;
  default:
    return false;
  }
}

/// @brief Return the wait counter type for a store-only vector memory op.
inline WaitCounterType vector_store_counter_type(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
    return WaitCounterType::VSCNT;
  case ROCJITSU_CODE_ARCH_CDNA4:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return WaitCounterType::STORECNT;
  default:
    return WaitCounterType::VMCNT;
  }
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_WAIT_COUNTER_POLICY_H_
