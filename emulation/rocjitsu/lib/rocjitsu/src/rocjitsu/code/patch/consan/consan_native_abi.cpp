// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_native_abi.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_internal.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <limits>

namespace rocjitsu::consan::detail {

bool append_flat_load_wait(std::vector<uint32_t> &words, rj_code_arch_t arch) {
  const auto wait = instrumentation::build_s_wait_flat_load0(arch);
  if (!wait)
    return false;
  words.push_back(*wait);
  return true;
}

bool append_global_atomic_wait(std::vector<uint32_t> &words, rj_code_arch_t arch) {
  const TargetProfile *target = target_profile(arch);
  return target != nullptr && detail::append_global_atomic_completion(words, *target);
}

bool append_lds_wait(std::vector<uint32_t> &words, rj_code_arch_t arch) {
  const auto wait = instrumentation::build_s_wait_lds0(arch);
  if (!wait)
    return false;
  words.push_back(*wait);
  return true;
}

bool append_delay_words(std::vector<uint32_t> &words, rj_code_arch_t arch, const DelayPlan &plan,
                        std::vector<std::string> &errors, std::string_view context) {
  if (plan.count == 0)
    return true;

  switch (plan.mode) {
  case SuperColliderDelayMode::Nop:
    for (uint32_t i = 0; i < plan.count; ++i)
      words.push_back(build_s_nop(0, arch));
    return true;
  case SuperColliderDelayMode::Sleep:
    if (plan.count > std::numeric_limits<uint16_t>::max()) {
      errors.emplace_back(std::string(context) +
                          " sleep delay immediate exceeds the 16-bit s_sleep field");
      return false;
    }
    words.push_back(build_s_sleep(static_cast<uint16_t>(plan.count), arch));
    return true;
  case SuperColliderDelayMode::SleepWave:
    errors.emplace_back(std::string(context) + " sleep_wave requires SuperCollider replay scratch");
    return false;
  case SuperColliderDelayMode::SleepVar:
    if (plan.variable_source > std::numeric_limits<uint8_t>::max()) {
      errors.emplace_back(std::string(context) +
                          " sleep_var source exceeds the 8-bit scalar source field");
      return false;
    }
    words.push_back(build_s_sleep_var(plan.variable_source, arch));
    return true;
  }

  errors.emplace_back(std::string(context) + " has unknown delay mode '" +
                      delay_mode_name(plan.mode) + "'");
  return false;
}

std::optional<WorkitemOwnerDerivation>
build_workitem_owner_derivation(const detail::WorkitemOwnerDerivationPlan &plan,
                                uint16_t value_vgpr, rj_code_arch_t arch, std::string_view consumer,
                                std::vector<std::string> &errors) {
  WorkitemOwnerDerivation derivation{.vgpr = value_vgpr, .words = {}};
  const TargetProfile *target = target_profile(arch);
  if (target == nullptr ||
      !detail::append_workitem_owner_derivation(
          derivation.words, {.plan = plan, .result_vgpr = value_vgpr}, *target)) {
    errors.emplace_back("ConSan " + std::string(consumer) + " could not encode owner derivation");
    return std::nullopt;
  }
  return derivation;
}

bool append_save_special_state(std::vector<uint32_t> &words,
                               const std::optional<detail::SpecialStateSgprs> &registers,
                               rj_code_arch_t arch) {
  const TargetProfile *target = target_profile(arch);
  return registers && target && detail::append_save_special_state(words, *registers, *target);
}

bool append_restore_special_state(std::vector<uint32_t> &words,
                                  const std::optional<detail::SpecialStateSgprs> &registers,
                                  rj_code_arch_t arch) {
  const TargetProfile *target = target_profile(arch);
  return registers && target && detail::append_restore_special_state(words, *registers, *target);
}

} // namespace rocjitsu::consan::detail
