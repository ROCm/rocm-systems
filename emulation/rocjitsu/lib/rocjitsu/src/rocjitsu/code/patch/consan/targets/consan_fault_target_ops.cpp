// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/targets/consan_fault_target_ops.h"

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/targets/consan_fault_target_ops_internal.h"
#include "rocjitsu/code/patch/consan/targets/consan_target_profiles.h"

namespace rocjitsu::consan {

bool lds_address_fault_arch_supported(rj_code_arch_t arch) {
  return arch_is_cdna3_or_cdna4(arch) || arch_is_rdna4_or_cdna5(arch);
}

AtomicFaultEncoding classify_atomic_fault_encoding(std::string_view mnemonic, uint32_t size,
                                                   rj_code_arch_t arch) {
  if (arch_is_cdna3_or_cdna4(arch))
    return fault_target_detail::classify_cdna3_cdna4_atomic_fault_encoding(mnemonic, size);
  if (arch_is_rdna4_or_cdna5(arch))
    return fault_target_detail::classify_rdna4_cdna5_atomic_fault_encoding(mnemonic, size);
  return AtomicFaultEncoding::Unsupported;
}

bool atomic_fault_supports_scope(AtomicFaultEncoding encoding) {
  return encoding == AtomicFaultEncoding::FlatLike || encoding == AtomicFaultEncoding::Buffer;
}

bool atomic_fault_supports_order(AtomicFaultEncoding encoding) {
  return encoding == AtomicFaultEncoding::CdnaFlat || encoding == AtomicFaultEncoding::FlatLike ||
         encoding == AtomicFaultEncoding::Buffer;
}

bool atomic_fault_supports_address(AtomicFaultEncoding encoding) {
  return encoding == AtomicFaultEncoding::FlatLike || encoding == AtomicFaultEncoding::Buffer ||
         encoding == AtomicFaultEncoding::Ds;
}

AtomicFaultRewriteResult rewrite_atomic_fault_address(std::span<uint8_t> instruction,
                                                      AtomicFaultEncoding encoding,
                                                      uint32_t width_bits, uint32_t address_delta) {
  return fault_target_detail::rewrite_rdna4_cdna5_atomic_fault_address(instruction, encoding,
                                                                       width_bits, address_delta);
}

AtomicFaultRewriteResult rewrite_atomic_fault_scope_to_wave(std::span<uint8_t> instruction,
                                                            AtomicFaultEncoding encoding) {
  return fault_target_detail::rewrite_rdna4_cdna5_atomic_fault_scope_to_wave(instruction, encoding);
}

} // namespace rocjitsu::consan
