// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_fault_target_ops.h"

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_fault_target_ops_internal.h"

namespace rocjitsu {

bool consan_lds_address_fault_arch_supported(rj_code_arch_t arch) {
  return consan_uses_gfx9_cdna_encoding(arch) || consan_uses_gfx12_encoding(arch);
}

ConSanAtomicFaultEncoding classify_consan_atomic_fault_encoding(std::string_view mnemonic,
                                                                uint32_t size,
                                                                rj_code_arch_t arch) {
  if (consan_uses_gfx9_cdna_encoding(arch))
    return consan_fault_target_detail::classify_gfx9_atomic_fault_encoding(mnemonic, size);
  if (consan_uses_gfx12_encoding(arch))
    return consan_fault_target_detail::classify_gfx12_atomic_fault_encoding(mnemonic, size);
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
