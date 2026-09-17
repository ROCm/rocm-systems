// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_fault_cdna3_cdna4_target_ops.cpp
/// @brief CDNA3/CDNA4 fault-classification recipes.

#include "rocjitsu/code/patch/consan/targets/consan_fault_target_ops_internal.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"

namespace rocjitsu::consan::fault_target_detail {

AtomicFaultEncoding classify_cdna3_cdna4_atomic_fault_encoding(std::string_view mnemonic,
                                                               uint32_t size) {
  if ((mnemonic.starts_with("flat_atomic") && size == sizeof(cdna4::FlatMachineInst)) ||
      (mnemonic.starts_with("global_atomic") && size == sizeof(cdna4::FlatGlblMachineInst))) {
    return AtomicFaultEncoding::CdnaFlat;
  }
  return AtomicFaultEncoding::Unsupported;
}

} // namespace rocjitsu::consan::fault_target_detail
