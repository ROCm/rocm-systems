// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/targets/consan_fault_target_ops.h"

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/targets/consan_fault_target_ops_internal.h"
#include "rocjitsu/code/patch/consan/targets/consan_target_profiles.h"

namespace rocjitsu {

bool consan_lds_address_fault_arch_supported(rj_code_arch_t arch) {
  return consan_arch_is_cdna3_or_cdna4(arch) || consan_arch_is_rdna4_or_cdna5(arch);
}

ConSanAtomicFaultEncoding classify_consan_atomic_fault_encoding(std::string_view mnemonic,
                                                                uint32_t size,
                                                                rj_code_arch_t arch) {
  if (consan_arch_is_cdna3_or_cdna4(arch))
    return consan_fault_target_detail::classify_cdna3_cdna4_atomic_fault_encoding(mnemonic, size);
  if (consan_arch_is_rdna4_or_cdna5(arch))
    return consan_fault_target_detail::classify_rdna4_cdna5_atomic_fault_encoding(mnemonic, size);
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

ConSanAtomicFaultRewriteResult
rewrite_consan_atomic_fault_address(std::span<uint8_t> instruction,
                                    ConSanAtomicFaultEncoding encoding, uint32_t width_bits,
                                    uint32_t address_delta) {
  return consan_fault_target_detail::rewrite_rdna4_cdna5_atomic_fault_address(
      instruction, encoding, width_bits, address_delta);
}

ConSanAtomicFaultRewriteResult
rewrite_consan_atomic_fault_scope_to_wave(std::span<uint8_t> instruction,
                                          ConSanAtomicFaultEncoding encoding) {
  return consan_fault_target_detail::rewrite_rdna4_cdna5_atomic_fault_scope_to_wave(instruction,
                                                                                    encoding);
}

} // namespace rocjitsu
