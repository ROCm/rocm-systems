// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_fault_gfx9_target_ops.cpp
/// @brief GFX9 CDNA fault-classification recipes.

#include "rocjitsu/code/patch/consan/consan_fault_target_ops_internal.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"

namespace rocjitsu::consan_fault_target_detail {

ConSanAtomicFaultEncoding classify_gfx9_atomic_fault_encoding(std::string_view mnemonic,
                                                              uint32_t size) {
  if ((mnemonic.starts_with("flat_atomic") && size == sizeof(cdna4::FlatMachineInst)) ||
      (mnemonic.starts_with("global_atomic") && size == sizeof(cdna4::FlatGlblMachineInst))) {
    return ConSanAtomicFaultEncoding::CdnaFlat;
  }
  return ConSanAtomicFaultEncoding::Unsupported;
}

} // namespace rocjitsu::consan_fault_target_detail
