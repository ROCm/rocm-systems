// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_resource.h
/// @brief DBT/DBI-neutral register planning for ConSan probes.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/isa/register_set.h"

#include <cstdint>
#include <optional>
#include <span>

namespace rocjitsu {

/// @brief One contiguous temporary-register request at an instruction.
struct ConSanRegisterRequest {
  RegClass reg_class = RegClass::VGPR;
  uint16_t count = 0;
  uint16_t alignment = 1;
  uint16_t search_start = 0;
  uint16_t current_allocation_count = 0;
  uint16_t max_referenced_count = 0;
  uint16_t architecture_limit = 0;
  std::optional<uint16_t> explicit_base;
  RegisterSet forbidden;
  bool allow_spill = true;
  bool force_spill = false;
  bool allow_spill_descriptor_growth = false;
};

/// @brief Read-only result of planning one register request.
struct ConSanRegisterPlan {
  ConSanRegisterAllocationSource source = ConSanRegisterAllocationSource::Unsupported;
  ConSanRegisterPlanReason reason = ConSanRegisterPlanReason::None;
  std::optional<uint16_t> base;
  uint16_t count = 0;
  uint16_t required_descriptor_count = 0;
};

/// @brief Plan explicit, dead, fresh, or spill-backed registers without mutation.
[[nodiscard]] ConSanRegisterPlan plan_consan_registers(const ConSanRegisterRequest &request,
                                                       const RegisterSet &live_before);

/// Summarizes resource decisions and emitted spill mechanics. This derived
/// reporting view is not transform state: callers calculate it when needed
/// instead of storing mutable data that can become stale as inputs change.
[[nodiscard]] ConSanResourcePlanSummary
summarize_consan_resource_plans(std::span<const ConSanCandidateResourcePlan> plans,
                                std::span<const ConSanPatchInfo> patches);

} // namespace rocjitsu
