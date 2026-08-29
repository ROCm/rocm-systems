// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_fault_target_ops.h"

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_semantic_classifiers.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"

namespace rocjitsu {

bool consan_lds_address_fault_arch_supported(rj_code_arch_t arch) {
  return consan_uses_gfx9_cdna_encoding(arch) || consan_uses_gfx12_encoding(arch);
}

ConSanAtomicFaultEncoding classify_consan_atomic_fault_encoding(std::string_view mnemonic,
                                                                uint32_t size,
                                                                rj_code_arch_t arch) {
  if (consan_uses_gfx9_cdna_encoding(arch) &&
      ((mnemonic.starts_with("flat_atomic") && size == sizeof(cdna4::FlatMachineInst)) ||
       (mnemonic.starts_with("global_atomic") && size == sizeof(cdna4::FlatGlblMachineInst)))) {
    return ConSanAtomicFaultEncoding::CdnaFlat;
  }
  if (consan_uses_gfx12_encoding(arch) &&
      (mnemonic.starts_with("flat_atomic") || mnemonic.starts_with("global_atomic")) &&
      size == sizeof(rdna4::VflatMachineInst)) {
    return ConSanAtomicFaultEncoding::FlatLike;
  }
  if (consan_uses_gfx12_encoding(arch) && mnemonic.starts_with("buffer_atomic") &&
      size == sizeof(rdna4::VbufferMachineInst)) {
    return ConSanAtomicFaultEncoding::Buffer;
  }
  if (consan_uses_gfx12_encoding(arch) && mnemonic.starts_with("ds_") && is_ds_atomic(mnemonic) &&
      lds_width_bits(mnemonic) == 32u && size == sizeof(rdna4::VdsMachineInst)) {
    return ConSanAtomicFaultEncoding::Ds;
  }
  return ConSanAtomicFaultEncoding::Unsupported;
}

bool consan_atomic_fault_supports_scope(ConSanAtomicFaultEncoding encoding) {
  return encoding == ConSanAtomicFaultEncoding::FlatLike ||
         encoding == ConSanAtomicFaultEncoding::Buffer;
}

bool consan_atomic_fault_supports_order(ConSanAtomicFaultEncoding encoding) {
  return encoding == ConSanAtomicFaultEncoding::CdnaFlat ||
         encoding == ConSanAtomicFaultEncoding::FlatLike ||
         encoding == ConSanAtomicFaultEncoding::Buffer;
}

bool consan_atomic_fault_supports_address(ConSanAtomicFaultEncoding encoding) {
  return encoding == ConSanAtomicFaultEncoding::FlatLike ||
         encoding == ConSanAtomicFaultEncoding::Buffer || encoding == ConSanAtomicFaultEncoding::Ds;
}

} // namespace rocjitsu
