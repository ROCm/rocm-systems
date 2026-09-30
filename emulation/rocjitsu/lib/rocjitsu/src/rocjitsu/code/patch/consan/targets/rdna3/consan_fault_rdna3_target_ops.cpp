// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_fault_rdna3_target_ops.cpp
/// @brief RDNA3 fault-classification recipes.

#include "rocjitsu/code/patch/consan/targets/consan_fault_target_ops_internal.h"

#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/machine_insts.h"

namespace rocjitsu::consan::fault_target_detail {

AtomicFaultEncoding classify_rdna3_atomic_fault_encoding(std::string_view mnemonic, uint32_t size) {
  if ((mnemonic.starts_with("flat_atomic") || mnemonic.starts_with("global_atomic")) &&
      size == sizeof(rdna3::FlatMachineInst)) {
    return AtomicFaultEncoding::Rdna3Flat;
  }
  return AtomicFaultEncoding::Unsupported;
}

} // namespace rocjitsu::consan::fault_target_detail
