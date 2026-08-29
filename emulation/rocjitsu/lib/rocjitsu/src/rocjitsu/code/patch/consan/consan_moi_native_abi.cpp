// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"

#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

namespace rocjitsu::consan_moi_impl {

std::optional<uint16_t> gfx1250_vgpr_msb_mode_at(std::span<const uint8_t> bytes,
                                                 uint64_t text_file_offset,
                                                 uint64_t container_entry_text_offset,
                                                 uint64_t site_file_offset) {
  return consan_gfx1250_vgpr_msb_mode_at(bytes, text_file_offset, container_entry_text_offset,
                                         site_file_offset);
}

bool append_moi_flat_load_wait(std::vector<uint32_t> &words, rj_code_arch_t arch) {
  const auto wait = instrumentation::build_s_wait_flat_load0(arch);
  if (!wait)
    return false;
  words.push_back(*wait);
  return true;
}

bool append_moi_global_atomic_wait(std::vector<uint32_t> &words, rj_code_arch_t arch) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  return target != nullptr && consan_detail::append_moi_global_atomic_completion(words, *target);
}

bool append_moi_lds_wait(std::vector<uint32_t> &words, rj_code_arch_t arch) {
  const auto wait = instrumentation::build_s_wait_lds0(arch);
  if (!wait)
    return false;
  words.push_back(*wait);
  return true;
}

std::optional<MoiWorkitemOwnerDerivation>
build_moi_workitem_owner_derivation(const consan_detail::MoiWorkitemOwnerDerivationPlan &plan,
                                    uint16_t value_vgpr, rj_code_arch_t arch,
                                    std::string_view consumer, std::vector<std::string> &errors) {
  MoiWorkitemOwnerDerivation derivation{.vgpr = value_vgpr, .words = {}};
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr ||
      !consan_detail::append_moi_workitem_owner_derivation(
          derivation.words, {.plan = plan, .result_vgpr = value_vgpr}, *target)) {
    errors.emplace_back("ConSan MOI " + std::string(consumer) +
                        " could not encode owner derivation");
    return std::nullopt;
  }
  return derivation;
}

bool append_save_moi_special_state(
    std::vector<uint32_t> &words,
    const std::optional<consan_detail::MoiSpecialStateSgprs> &registers, rj_code_arch_t arch) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  return registers && target &&
         consan_detail::append_save_moi_special_state(words, *registers, *target);
}

bool append_restore_moi_special_state(
    std::vector<uint32_t> &words,
    const std::optional<consan_detail::MoiSpecialStateSgprs> &registers, rj_code_arch_t arch) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  return registers && target &&
         consan_detail::append_restore_moi_special_state(words, *registers, *target);
}

} // namespace rocjitsu::consan_moi_impl
