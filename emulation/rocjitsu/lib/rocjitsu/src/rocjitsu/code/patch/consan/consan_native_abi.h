// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_internal.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu::consan::detail {

inline constexpr uint16_t kScalarInlineNegativeOneOperand = 193u;
inline constexpr uint16_t kScalarOperandTtmpBase = 108u;
inline constexpr uint32_t kMaxVgprs = kOrdinaryVgprLimit;
inline constexpr uint32_t kMaxSgprs = 106u;

[[nodiscard]] constexpr uint16_t ttmp_scalar_operand(uint16_t ttmp) {
  return static_cast<uint16_t>(kScalarOperandTtmpBase + ttmp);
}

using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
namespace kd = rocr::llvm::amdhsa;

[[nodiscard]] bool append_flat_load_wait(std::vector<uint32_t> &words, rj_code_arch_t arch);
[[nodiscard]] bool append_global_atomic_wait(std::vector<uint32_t> &words, rj_code_arch_t arch);
[[nodiscard]] bool append_lds_wait(std::vector<uint32_t> &words, rj_code_arch_t arch);
struct DelayPlan {
  SuperColliderDelayMode mode = SuperColliderDelayMode::Nop;
  uint32_t count = 0;
  uint32_t variable_source = 0;
};
[[nodiscard]] bool append_delay_words(std::vector<uint32_t> &words, rj_code_arch_t arch,
                                      const DelayPlan &plan, std::vector<std::string> &errors,
                                      std::string_view context);

struct WorkitemOwnerDerivation {
  uint16_t vgpr = 0;
  std::vector<uint32_t> words;
};

[[nodiscard]] std::optional<WorkitemOwnerDerivation>
build_workitem_owner_derivation(const detail::WorkitemOwnerDerivationPlan &plan,
                                uint16_t value_vgpr, rj_code_arch_t arch, std::string_view consumer,
                                std::vector<std::string> &errors);

[[nodiscard]] bool
append_save_special_state(std::vector<uint32_t> &words,
                          const std::optional<detail::SpecialStateSgprs> &registers,
                          rj_code_arch_t arch);

[[nodiscard]] bool
append_restore_special_state(std::vector<uint32_t> &words,
                             const std::optional<detail::SpecialStateSgprs> &registers,
                             rj_code_arch_t arch);

} // namespace rocjitsu::consan::detail
