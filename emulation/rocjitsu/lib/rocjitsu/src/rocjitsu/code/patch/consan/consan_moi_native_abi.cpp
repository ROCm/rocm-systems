// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <limits>

namespace rocjitsu::consan_moi_impl {

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

bool append_moi_delay_words(std::vector<uint32_t> &words, rj_code_arch_t arch,
                            const ConSanRequest &request, std::vector<std::string> &errors,
                            std::string_view context) {
  if (request.delay_nops == 0)
    return true;

  switch (request.delay_mode) {
  case ConSanDelayMode::Nop:
    for (uint32_t i = 0; i < request.delay_nops; ++i)
      words.push_back(build_s_nop(0, arch));
    return true;
  case ConSanDelayMode::Sleep:
    if (request.delay_nops > std::numeric_limits<uint16_t>::max()) {
      errors.emplace_back(std::string(context) +
                          " sleep delay immediate exceeds the 16-bit s_sleep field");
      return false;
    }
    words.push_back(build_s_sleep(static_cast<uint16_t>(request.delay_nops), arch));
    return true;
  case ConSanDelayMode::SleepVar:
    if (request.delay_var_ssrc > std::numeric_limits<uint8_t>::max()) {
      errors.emplace_back(std::string(context) +
                          " sleep_var source exceeds the 8-bit scalar source field");
      return false;
    }
    words.push_back(build_s_sleep_var(request.delay_var_ssrc, arch));
    return true;
  }

  errors.emplace_back(std::string(context) + " has unknown delay mode '" +
                      consan_delay_mode_name(request.delay_mode) + "'");
  return false;
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
