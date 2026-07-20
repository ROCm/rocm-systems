// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_resource.h"

#include <algorithm>
#include <cstdint>

namespace rocjitsu {

namespace {

[[nodiscard]] uint32_t align_up(uint32_t value, uint16_t alignment) {
  const uint32_t effective_alignment = std::max<uint32_t>(alignment, 1u);
  return ((value + effective_alignment - 1u) / effective_alignment) * effective_alignment;
}

[[nodiscard]] bool range_intersects(const RegisterSet &set, RegClass reg_class, uint16_t base,
                                    uint16_t count) {
  for (uint16_t i = 0; i < count; ++i) {
    if (set.contains({reg_class, static_cast<uint16_t>(base + i), 1}))
      return true;
  }
  return false;
}

[[nodiscard]] bool range_fits(uint32_t base, uint16_t count, uint16_t limit) {
  return base <= limit && count <= static_cast<uint32_t>(limit) - base;
}

[[nodiscard]] std::optional<uint16_t> find_window(const ConSanRegisterRequest &request,
                                                  const RegisterSet &live_before, uint16_t limit,
                                                  bool require_dead) {
  const uint32_t first = align_up(request.search_start, request.alignment);
  const uint32_t step = std::max<uint32_t>(request.alignment, 1u);
  for (uint32_t base = first; range_fits(base, request.count, limit); base += step) {
    const auto candidate = static_cast<uint16_t>(base);
    if (range_intersects(request.forbidden, request.reg_class, candidate, request.count))
      continue;
    if (require_dead && range_intersects(live_before, request.reg_class, candidate, request.count))
      continue;
    return candidate;
  }
  return std::nullopt;
}

} // namespace

ConSanRegisterPlan plan_consan_registers(const ConSanRegisterRequest &request,
                                         const RegisterSet &live_before) {
  ConSanRegisterPlan plan;
  plan.count = request.count;
  plan.required_descriptor_count = request.current_allocation_count;
  if (request.count == 0 || request.alignment == 0 || request.architecture_limit == 0 ||
      request.current_allocation_count > request.architecture_limit ||
      request.max_referenced_count > request.architecture_limit) {
    plan.reason = ConSanRegisterPlanReason::InvalidRequest;
    return plan;
  }

  if (request.explicit_base) {
    const uint16_t base = *request.explicit_base;
    if (base % request.alignment != 0) {
      plan.reason = ConSanRegisterPlanReason::ExplicitMisaligned;
      return plan;
    }
    if (!range_fits(base, request.count, request.architecture_limit)) {
      plan.reason = ConSanRegisterPlanReason::ExplicitOutOfRange;
      return plan;
    }
    if (range_intersects(request.forbidden, request.reg_class, base, request.count)) {
      plan.reason = ConSanRegisterPlanReason::ForbiddenOverlap;
      return plan;
    }
    if (static_cast<uint32_t>(base) < request.max_referenced_count &&
        range_intersects(live_before, request.reg_class, base, request.count)) {
      plan.reason = ConSanRegisterPlanReason::ExplicitLive;
      return plan;
    }
    plan.source = ConSanRegisterAllocationSource::Explicit;
    plan.base = base;
    plan.required_descriptor_count = static_cast<uint16_t>(std::max<uint32_t>(
        request.current_allocation_count, static_cast<uint32_t>(base) + request.count));
    return plan;
  }

  if (request.force_spill) {
    if (request.allow_spill) {
      if (auto victim = find_window(request, live_before, request.current_allocation_count,
                                    /*require_dead=*/false)) {
        plan.source = ConSanRegisterAllocationSource::SpillRequired;
        plan.base = victim;
        return plan;
      }
    }
    plan.reason = ConSanRegisterPlanReason::NoLegalWindow;
    return plan;
  }

  if (auto dead = find_window(request, live_before, request.current_allocation_count,
                              /*require_dead=*/true)) {
    plan.source = ConSanRegisterAllocationSource::LivenessDead;
    plan.base = dead;
    return plan;
  }

  const uint32_t fresh_start = align_up(
      std::max<uint16_t>(request.search_start, std::max<uint16_t>(request.current_allocation_count,
                                                                  request.max_referenced_count)),
      request.alignment);
  const uint32_t fresh_step = std::max<uint32_t>(request.alignment, 1u);
  for (uint32_t base = fresh_start; range_fits(base, request.count, request.architecture_limit);
       base += fresh_step) {
    if (range_intersects(request.forbidden, request.reg_class, static_cast<uint16_t>(base),
                         request.count)) {
      continue;
    }
    plan.source = ConSanRegisterAllocationSource::DescriptorGrowth;
    plan.base = static_cast<uint16_t>(base);
    plan.required_descriptor_count = static_cast<uint16_t>(base + request.count);
    return plan;
  }

  if (request.allow_spill) {
    if (auto victim = find_window(request, live_before, request.current_allocation_count,
                                  /*require_dead=*/false)) {
      plan.source = ConSanRegisterAllocationSource::SpillRequired;
      plan.base = victim;
      return plan;
    }
  }

  plan.reason = ConSanRegisterPlanReason::NoLegalWindow;
  return plan;
}

} // namespace rocjitsu
