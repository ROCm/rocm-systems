// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_dispatch_preload.h
/// @brief Target-neutral AMDHSA dispatch-ID preload transformation contract.

#pragma once

#include <cstdint>
#include <optional>

namespace rocjitsu {

/// AMDHSA initializes at most 16 user SGPRs. Dispatch-ID insertion is allowed
/// only when all original preloads, the new two-SGPR ID, and the following
/// system SGPRs can be mapped without truncation or register overflow.
enum class ConSanMoiDispatchIdPreloadSupport : uint8_t {
  SupportedAlreadyEnabled,
  SupportedInsert,
  InvalidDispatchPosition,
  UserSgprInitializationLimit,
  SgprAllocationLimit,
};

struct ConSanMoiDispatchIdPreloadPlan {
  ConSanMoiDispatchIdPreloadSupport support =
      ConSanMoiDispatchIdPreloadSupport::InvalidDispatchPosition;
  uint16_t dispatch_id_sgpr = 0;
  uint16_t original_user_sgpr_count = 0;
  uint16_t expanded_user_sgpr_count = 0;
  uint16_t system_sgpr_count = 0;
  uint16_t first_shifted_guest_sgpr = 0;
  uint16_t shifted_guest_sgpr_count = 0;
  /// Full user-SGPR windows can make room for dispatch ID by shortening the
  /// hardware kernarg preload and reloading its final two dwords in software.
  uint16_t kernarg_reload_sgpr = 0;
  uint16_t kernarg_reload_base_sgpr = 0;
  uint16_t kernarg_reload_offset_dwords = 0;
  uint16_t kernarg_reload_count = 0;
  uint16_t shifted_system_sgpr_count = 0;
  uint16_t system_sgpr_shift = 0;
  uint16_t original_kernarg_preload_length = 0;
  uint16_t replacement_kernarg_preload_length = 0;
  uint16_t required_sgpr_count = 0;

  [[nodiscard]] constexpr bool supported() const {
    return support == ConSanMoiDispatchIdPreloadSupport::SupportedAlreadyEnabled ||
           support == ConSanMoiDispatchIdPreloadSupport::SupportedInsert;
  }
  [[nodiscard]] constexpr bool descriptor_change_required() const {
    return support == ConSanMoiDispatchIdPreloadSupport::SupportedInsert;
  }
  [[nodiscard]] constexpr bool requires_kernarg_reload() const { return kernarg_reload_count != 0; }
};

[[nodiscard]] constexpr uint16_t
consan_moi_amdhsa_dispatch_id_prefix_sgpr_count(bool private_segment_buffer, bool dispatch_ptr,
                                                bool queue_ptr, bool kernarg_segment_ptr) {
  return static_cast<uint16_t>((private_segment_buffer ? 4u : 0u) + (dispatch_ptr ? 2u : 0u) +
                               (queue_ptr ? 2u : 0u) + (kernarg_segment_ptr ? 2u : 0u));
}

/// Reference contract for adding AMDHSA's 64-bit dispatch-ID preload without
/// changing the SGPR view seen by guest instructions. The generated prologue
/// first captures dispatch_id_sgpr:{+1} into persistent instrumentation state,
/// then copies every shifted guest preload from high layout index `dest + 2`
/// back to original index `dest`, in increasing destination order.
[[nodiscard]] constexpr ConSanMoiDispatchIdPreloadPlan
consan_moi_plan_dispatch_id_preload(uint16_t original_user_sgpr_count, uint16_t system_sgpr_count,
                                    uint16_t dispatch_id_prefix_sgpr_count,
                                    bool dispatch_id_already_enabled, uint16_t sgpr_limit = 106,
                                    uint16_t user_sgpr_initialization_limit = 16) {
  ConSanMoiDispatchIdPreloadPlan plan;
  plan.dispatch_id_sgpr = dispatch_id_prefix_sgpr_count;
  plan.original_user_sgpr_count = original_user_sgpr_count;
  plan.expanded_user_sgpr_count = original_user_sgpr_count;
  plan.system_sgpr_count = system_sgpr_count;
  const uint32_t original_required_count =
      static_cast<uint32_t>(original_user_sgpr_count) + system_sgpr_count;

  const uint32_t dispatch_end = static_cast<uint32_t>(dispatch_id_prefix_sgpr_count) + 2u;
  if (dispatch_id_prefix_sgpr_count > original_user_sgpr_count ||
      (dispatch_id_already_enabled && dispatch_end > original_user_sgpr_count)) {
    plan.support = ConSanMoiDispatchIdPreloadSupport::InvalidDispatchPosition;
    return plan;
  }
  if (original_user_sgpr_count > user_sgpr_initialization_limit) {
    plan.support = ConSanMoiDispatchIdPreloadSupport::UserSgprInitializationLimit;
    return plan;
  }
  if (dispatch_id_already_enabled) {
    if (original_required_count > sgpr_limit) {
      plan.support = ConSanMoiDispatchIdPreloadSupport::SgprAllocationLimit;
      return plan;
    }
    plan.required_sgpr_count = static_cast<uint16_t>(original_required_count);
    plan.support = ConSanMoiDispatchIdPreloadSupport::SupportedAlreadyEnabled;
    return plan;
  }

  const uint32_t expanded_user_count = static_cast<uint32_t>(original_user_sgpr_count) + 2u;
  const uint32_t required_count = expanded_user_count + system_sgpr_count;
  if (expanded_user_count > user_sgpr_initialization_limit) {
    plan.support = ConSanMoiDispatchIdPreloadSupport::UserSgprInitializationLimit;
    return plan;
  }
  if (required_count > sgpr_limit) {
    plan.support = ConSanMoiDispatchIdPreloadSupport::SgprAllocationLimit;
    return plan;
  }
  plan.expanded_user_sgpr_count = static_cast<uint16_t>(expanded_user_count);
  plan.first_shifted_guest_sgpr = dispatch_id_prefix_sgpr_count;
  plan.shifted_guest_sgpr_count = static_cast<uint16_t>(
      original_user_sgpr_count + system_sgpr_count - dispatch_id_prefix_sgpr_count);
  plan.required_sgpr_count = static_cast<uint16_t>(required_count);
  plan.support = ConSanMoiDispatchIdPreloadSupport::SupportedInsert;
  return plan;
}

[[nodiscard]] constexpr std::optional<uint16_t>
consan_moi_dispatch_id_restore_source(const ConSanMoiDispatchIdPreloadPlan &plan,
                                      uint16_t guest_destination_sgpr) {
  if (!plan.supported() || !plan.descriptor_change_required())
    return std::nullopt;
  const uint32_t shifted_end =
      static_cast<uint32_t>(plan.first_shifted_guest_sgpr) + plan.shifted_guest_sgpr_count;
  if (guest_destination_sgpr < plan.first_shifted_guest_sgpr ||
      guest_destination_sgpr >= shifted_end)
    return std::nullopt;
  return static_cast<uint16_t>(guest_destination_sgpr + 2u);
}

} // namespace rocjitsu
