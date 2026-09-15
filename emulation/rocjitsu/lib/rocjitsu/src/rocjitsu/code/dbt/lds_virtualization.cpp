// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file lds_virtualization.cpp
/// @brief Architecture-pair dispatch for virtual-LDS translation.

#include "rocjitsu/code/dbt/lds_virtualization.h"

#include "rocjitsu/code/dbt/semantic/cdna4_to_cdna3_virtual_lds.h"

namespace rocjitsu {
namespace {

/// @brief Return true for the architecture pair implemented by this build.
[[nodiscard]] bool is_cdna4_to_cdna3(rj_code_arch_t guest_arch, rj_code_arch_t host_arch) {
  return guest_arch == ROCJITSU_CODE_ARCH_CDNA4 && host_arch == ROCJITSU_CODE_ARCH_CDNA3;
}

} // namespace

bool supports_virtual_lds_sidecars(rj_code_arch_t guest_arch, rj_code_arch_t host_arch) {
  return is_cdna4_to_cdna3(guest_arch, host_arch);
}

bool source_instruction_uses_virtualizable_lds(const Instruction &inst, rj_code_arch_t guest_arch,
                                               rj_code_arch_t host_arch) {
  if (is_cdna4_to_cdna3(guest_arch, host_arch))
    return cdna4_to_cdna3_source_instruction_uses_virtualizable_lds(inst);
  return false;
}

std::optional<VirtualLdsBaseSgprReservation>
reserve_virtual_lds_base_sgpr_pair(TranslationContext &context, KernelBlockScope blocks,
                                   const KdTranslation &translation, rj_code_arch_t guest_arch,
                                   rj_code_arch_t host_arch) {
  if (is_cdna4_to_cdna3(guest_arch, host_arch)) {
    return reserve_cdna4_to_cdna3_virtual_lds_base_sgpr_pair(context, blocks, translation);
  }
  return std::nullopt;
}

bool append_virtual_lds_entry_prologue(KdTranslation &translation, rj_code_arch_t guest_arch,
                                       rj_code_arch_t host_arch) {
  if (is_cdna4_to_cdna3(guest_arch, host_arch))
    return append_cdna4_to_cdna3_virtual_lds_entry_prologue(translation);
  return false;
}

ExpandResult lower_virtual_lds_instruction(const Instruction &inst, TranslationContext &context,
                                           rj_code_arch_t guest_arch, rj_code_arch_t host_arch) {
  if (is_cdna4_to_cdna3(guest_arch, host_arch))
    return lower_cdna4_to_cdna3_virtual_lds_instruction(inst, context);
  return ExpandResult::not_handled();
}

} // namespace rocjitsu
