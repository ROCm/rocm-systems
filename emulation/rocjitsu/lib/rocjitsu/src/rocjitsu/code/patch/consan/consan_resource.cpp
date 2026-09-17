// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/spill_manager.h"

#include <algorithm>
#include <array>
#include <cstdint>

namespace rocjitsu::consan {

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

[[nodiscard]] std::optional<uint16_t> find_window(const RegisterRequest &request,
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

[[nodiscard]] uint16_t required_descriptor_count(const RegisterRequest &request, uint16_t base) {
  return static_cast<uint16_t>(std::max<uint32_t>(request.current_allocation_count,
                                                  static_cast<uint32_t>(base) + request.count));
}

[[nodiscard]] std::optional<RegisterPlan>
plan_spill_window(const RegisterRequest &request, const RegisterSet &live_before, uint16_t limit) {
  const auto victim = find_window(request, live_before, limit, /*require_dead=*/false);
  if (!victim)
    return std::nullopt;
  RegisterPlan plan;
  plan.source = RegisterAllocationSource::SpillRequired;
  plan.base = victim;
  plan.count = request.count;
  plan.required_descriptor_count = required_descriptor_count(request, *victim);
  return plan;
}

} // namespace

RegisterPlan plan_registers(const RegisterRequest &request, const RegisterSet &live_before) {
  RegisterPlan plan;
  plan.count = request.count;
  plan.required_descriptor_count = request.current_allocation_count;
  if (request.count == 0 || request.alignment == 0 || request.architecture_limit == 0 ||
      request.current_allocation_count > request.architecture_limit ||
      request.max_referenced_count > request.architecture_limit) {
    plan.reason = RegisterPlanReason::InvalidRequest;
    return plan;
  }

  if (request.explicit_base) {
    const uint16_t base = *request.explicit_base;
    if (base % request.alignment != 0) {
      plan.reason = RegisterPlanReason::ExplicitMisaligned;
      return plan;
    }
    if (!range_fits(base, request.count, request.architecture_limit)) {
      plan.reason = RegisterPlanReason::ExplicitOutOfRange;
      return plan;
    }
    if (range_intersects(request.forbidden, request.reg_class, base, request.count)) {
      plan.reason = RegisterPlanReason::ForbiddenOverlap;
      return plan;
    }
    if (static_cast<uint32_t>(base) < request.max_referenced_count &&
        range_intersects(live_before, request.reg_class, base, request.count)) {
      plan.reason = RegisterPlanReason::ExplicitLive;
      return plan;
    }
    plan.source = RegisterAllocationSource::Explicit;
    plan.base = base;
    plan.required_descriptor_count = required_descriptor_count(request, base);
    return plan;
  }

  if (request.force_spill) {
    if (request.allow_spill) {
      const uint16_t spill_limit = request.allow_spill_descriptor_growth
                                       ? request.architecture_limit
                                       : request.current_allocation_count;
      if (auto spill = plan_spill_window(request, live_before, spill_limit))
        return *spill;
    }
    plan.reason = RegisterPlanReason::NoLegalWindow;
    return plan;
  }

  if (auto dead = find_window(request, live_before, request.current_allocation_count,
                              /*require_dead=*/true)) {
    plan.source = RegisterAllocationSource::LivenessDead;
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
    plan.source = RegisterAllocationSource::DescriptorGrowth;
    plan.base = static_cast<uint16_t>(base);
    plan.required_descriptor_count = static_cast<uint16_t>(base + request.count);
    return plan;
  }

  // Shared physical code can have owners with different descriptor allocations:
  // max_referenced_count reflects the largest owner, while current_allocation_count
  // is the largest window available to every owner without descriptor growth. If
  // no fresh window exists above the largest guest reference, a liveness-dead
  // window below it is still valid after growing the smaller descriptors.
  if (auto dead = find_window(request, live_before, request.architecture_limit,
                              /*require_dead=*/true)) {
    plan.source = RegisterAllocationSource::DescriptorGrowth;
    plan.base = dead;
    plan.required_descriptor_count = required_descriptor_count(request, *dead);
    return plan;
  }

  if (request.allow_spill) {
    // Descriptor-growing spill is an opt-in force-spill fallback. Ordinary
    // allocation may only spill inside the currently declared allocation.
    if (auto spill = plan_spill_window(request, live_before, request.current_allocation_count))
      return *spill;
  }

  plan.reason = RegisterPlanReason::NoLegalWindow;
  return plan;
}

ResourcePlanSummary summarize_resource_plans(std::span<const CandidateResourcePlan> plans) {
  ResourcePlanSummary summary;
  const std::array alternative_counts = {
      &summary.alternative_selected,   &summary.alternative_rejected,
      &summary.alternative_superseded, &summary.alternative_contributed,
      &summary.alternative_vetoed,
  };
  static_assert(static_cast<size_t>(ResourcePlanAlternativeOutcome::Vetoed) + 1u == 5u);
  const std::array source_counts = {
      &summary.unsupported_plans,       &summary.explicit_plans, &summary.dead_plans,
      &summary.descriptor_growth_plans, &summary.spill_plans,
  };
  static_assert(static_cast<size_t>(RegisterAllocationSource::SpillRequired) + 1u == 5u);
  for (const CandidateResourcePlan &plan : plans) {
    for (const ResourcePlanAlternative &alternative : plan.alternatives) {
      ++summary.alternative_attempts;
      const auto outcome = resource_plan_alternative_outcome(plan, alternative);
      ++*alternative_counts[static_cast<size_t>(outcome)];
    }
    ++*source_counts[static_cast<size_t>(plan.source)];
    if (plan.source == RegisterAllocationSource::SpillRequired) {
      summary.planned_spill_slot_bytes +=
          static_cast<size_t>(plan.scratch_vgpr_count) * SpillManager::kSlotBytes;
    }
  }
  return summary;
}

void accumulate_emitted_spill(ResourcePlanSummary &summary, const PatchAbiEffects &effects) {
  if (effects.spilled_vgpr_count == 0)
    return;
  ++summary.emitted_spill_patches;
  summary.emitted_spill_slot_bytes +=
      static_cast<size_t>(effects.spilled_vgpr_count) * SpillManager::kSlotBytes;
}

} // namespace rocjitsu::consan
