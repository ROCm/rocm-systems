// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

[[nodiscard]] inline std::string
consan_patched_image_growth_policy_description(const ConSanPatchedImageGrowthLimit &policy,
                                               size_t input_image_bytes) {
  switch (policy.kind) {
  case ConSanPatchedImageGrowthLimitKind::AbsoluteBytes:
    return "absolute-bytes=" + std::to_string(policy.absolute_bytes);
  case ConSanPatchedImageGrowthLimitKind::InputPercent:
    return "input-percent=" + std::to_string(policy.input_percent) +
           ", original-input-image-bytes=" + std::to_string(input_image_bytes);
  }
  return "invalid-kind=" + std::to_string(static_cast<unsigned int>(policy.kind));
}

[[nodiscard]] inline size_t consan_saturating_size_add(size_t lhs, size_t rhs) {
  return rhs > std::numeric_limits<size_t>::max() - lhs ? std::numeric_limits<size_t>::max()
                                                        : lhs + rhs;
}

/// Apply the common ConSan patched-image growth policy and report an exact
/// limit rejection. Other transactional patcher failures remain distinct so a
/// malformed ELF or allocation failure is not mislabeled as a policy decision.
[[nodiscard]] inline bool replace_consan_text(CodeObjectPatcher &patcher,
                                              std::span<const uint8_t> new_text,
                                              const ConSanOptions &options,
                                              std::string_view operation, ConSanResult &result) {
  const size_t current_image_bytes = patcher.image_bytes().size();
  const size_t input_image_bytes =
      options.patched_image_growth_input_bytes.value_or(current_image_bytes);
  const auto budget = consan_patched_image_growth_budget(options.patched_image_growth_limit,
                                                         input_image_bytes, current_image_bytes);
  const std::string policy = consan_patched_image_growth_policy_description(
      options.patched_image_growth_limit, input_image_bytes);
  if (!budget) {
    result.errors.emplace_back("ConSan " + std::string(operation) +
                               " has an invalid patched-image growth policy (" + policy + ")");
    return false;
  }
  if (budget->already_exceeded) {
    result.patched_image_growth_rejections.push_back(
        {.operation = std::string(operation),
         .policy = options.patched_image_growth_limit,
         .input_image_bytes = input_image_bytes,
         .existing_growth_bytes = budget->existing_growth_bytes,
         .transaction_growth_bytes = 0,
         .required_total_growth_bytes = budget->existing_growth_bytes,
         .limit_bytes = budget->total_limit_bytes});
    result.errors.emplace_back("ConSan " + std::string(operation) +
                               " rejected patched-image file growth: required total " +
                               std::to_string(budget->existing_growth_bytes) + " bytes, limit " +
                               std::to_string(budget->total_limit_bytes) + " bytes (policy " +
                               policy + ")");
    return false;
  }

  TextReplacementInfo info;
  if (patcher.replace_text(new_text, budget->remaining_growth_bytes, &info))
    return true;

  if (info.file_growth_limit_exceeded && info.required_file_growth) {
    const size_t required_total =
        consan_saturating_size_add(budget->existing_growth_bytes, *info.required_file_growth);
    result.patched_image_growth_rejections.push_back(
        {.operation = std::string(operation),
         .policy = options.patched_image_growth_limit,
         .input_image_bytes = input_image_bytes,
         .existing_growth_bytes = budget->existing_growth_bytes,
         .transaction_growth_bytes = *info.required_file_growth,
         .required_total_growth_bytes = required_total,
         .limit_bytes = budget->total_limit_bytes});
    result.errors.emplace_back("ConSan " + std::string(operation) +
                               " rejected patched-image file growth: required total " +
                               std::to_string(required_total) + " bytes, limit " +
                               std::to_string(budget->total_limit_bytes) + " bytes (policy " +
                               policy + ")");
    return false;
  }

  std::string detail = "patched-image remaining file growth limit " +
                       std::to_string(budget->remaining_growth_bytes) + " bytes, total limit " +
                       std::to_string(budget->total_limit_bytes) + " bytes (policy " + policy + ")";
  if (info.required_file_growth)
    detail += ", required file growth " + std::to_string(*info.required_file_growth) + " bytes";
  result.errors.emplace_back("ConSan " + std::string(operation) +
                             " could not replace executable text (" + detail + ")");
  return false;
}

} // namespace rocjitsu
