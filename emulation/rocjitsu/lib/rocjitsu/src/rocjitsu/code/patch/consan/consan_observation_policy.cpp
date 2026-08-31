// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_observation_policy.cpp
/// @brief Authoritative composition of ConSan semantic-policy products.

#include "rocjitsu/code/patch/consan/consan.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>

namespace rocjitsu {
namespace {

[[nodiscard]] std::string quoted_aliases(std::span<const std::string> names) {
  std::string aliases;
  for (const std::string &name : names) {
    if (!aliases.empty())
      aliases += "', '";
    aliases += name;
  }
  return aliases;
}

void render_observation_diagnostics(const ProgramInventory &inventory,
                                    ConSanObservationProduct &product) {
  const bool supercollider = product.plan().engine == ConSanCapabilityEngine::SuperCollider;
  const std::string engine_name =
      supercollider ? "SuperCollider"
                    : std::string(consan_capability_engine_name(product.plan().engine));

  for (const ConSanSiteDecision &decision : product.plan().site_decisions) {
    if (decision.reason != ConSanAccessPolicyReason::ConflictingPhysicalAliases)
      continue;
    if (supercollider) {
      const auto access =
          std::ranges::find_if(inventory.access_sites(), [&](const auto &candidate) {
            return candidate.physical_id == decision.semantic_site.physical;
          });
      const std::string_view access_kind =
          access != inventory.access_sites().end() && access->origin == ConSanAccessOrigin::Flat
              ? "FLAT"
              : "native-LDS";
      product.diagnostics.emplace_back(
          "ConSan SuperCollider " + std::string(access_kind) + " access at original text offset " +
          std::to_string(decision.semantic_site.physical.original_text_offset) +
          " was decoded inconsistently through aliases '" +
          quoted_aliases(decision.source_containers) + "'");
    } else {
      product.diagnostics.emplace_back(
          "ConSan MOI physical access at original text offset " +
          std::to_string(decision.semantic_site.physical.original_text_offset) +
          " was decoded inconsistently through aliases '" +
          quoted_aliases(decision.source_containers) + "'");
    }
  }
  for (const ConSanBarrierSiteDecision &decision : product.plan().barrier_site_decisions) {
    if (decision.reason != ConSanBarrierPolicyReason::ConflictingPhysicalAliases)
      continue;
    std::string message = "ConSan " + engine_name + " ";
    if (!supercollider)
      message = "ConSan MOI physical ";
    message += "barrier at original text offset " +
               std::to_string(decision.semantic_site.physical.original_text_offset) +
               " was decoded inconsistently through ";
    message += supercollider ? "physical aliases"
                             : "aliases '" + quoted_aliases(decision.source_containers) + "'";
    product.diagnostics.push_back(std::move(message));
  }
  for (const ConSanAtomicSiteDecision &decision : product.plan().atomic_site_decisions) {
    if (decision.reason != ConSanAtomicPolicyReason::ConflictingPhysicalAliases)
      continue;
    product.diagnostics.emplace_back(
        "ConSan MOI physical atomic at original text offset " +
        std::to_string(decision.semantic_site.physical.original_text_offset) +
        " was decoded inconsistently through aliases '" +
        quoted_aliases(decision.source_containers) + "'");
  }
  for (const ConSanFenceSiteDecision &decision : product.plan().fence_site_decisions) {
    if (decision.reason != ConSanFencePolicyReason::ConflictingPhysicalAliases)
      continue;
    product.diagnostics.emplace_back(
        "ConSan MOI physical fence at original text offset " +
        std::to_string(decision.semantic_site.physical.original_text_offset) +
        " was decoded inconsistently through aliases '" +
        quoted_aliases(decision.source_containers) + "'");
  }
}

[[nodiscard]] std::vector<PhysicalSiteId>
ordinary_synchronization_reservations(const ProgramInventory &inventory) {
  std::unordered_map<std::string_view, const ConSanSyncEvent *> events;
  events.reserve(inventory.sync().sync_events.size());
  for (const ConSanSyncEvent &event : inventory.sync().sync_events)
    events.emplace(event.identity, &event);

  std::vector<PhysicalSiteId> reservations;
  for (const ConSanSyncSequence &sequence : inventory.sync().sync_sequences) {
    if (sequence.kind != ConSanSyncSequenceKind::OrdinaryMemory ||
        (sequence.memory_role != ConSanSyncMemoryRole::Acquire &&
         sequence.memory_role != ConSanSyncMemoryRole::Release) ||
        !consan_sync_confidence_meets(sequence.confidence,
                                      ConSanSemanticConfidence::Conservative) ||
        !consan_sync_confidence_meets(sequence.memory_role_confidence,
                                      ConSanSemanticConfidence::Conservative)) {
      continue;
    }
    const ConSanSyncEvent *communication = nullptr;
    for (const std::string &identity : sequence.member_event_identities) {
      const auto found = events.find(identity);
      if (found == events.end() || found->second->kind != ConSanSyncEventKind::OrdinaryMemory)
        continue;
      if (communication != nullptr) {
        communication = nullptr;
        break;
      }
      communication = found->second;
    }
    if (communication == nullptr || communication->container_name != sequence.container_name ||
        communication->in_kernel != sequence.in_kernel) {
      continue;
    }
    for (const ConSanAccessInventorySite &access : inventory.access_sites()) {
      const bool in_kernel = access.container.kind == ConSanProgramContainerKind::Kernel;
      if (in_kernel != communication->in_kernel ||
          access.container.name != communication->container_name ||
          access.physical_id.original_text_offset != communication->text_offset ||
          std::ranges::find(reservations, access.physical_id) != reservations.end()) {
        continue;
      }
      reservations.push_back(access.physical_id);
    }
  }
  return reservations;
}

} // namespace

bool consan_site_matches_kernel_allowlist(const ProgramInventory &inventory,
                                          std::span<const uint64_t> owner_descriptor_file_offsets,
                                          std::span<const std::string> source_containers,
                                          std::span<const std::string> kernel_name_allowlist) {
  if (kernel_name_allowlist.empty())
    return true;
  const auto selected_name = [&](std::string_view candidate) {
    const std::string_view normalized = consan_normalize_kernel_name(candidate);
    return std::ranges::any_of(kernel_name_allowlist, [&](std::string_view allowed) {
      return consan_normalize_kernel_name(allowed) == normalized;
    });
  };
  if (!owner_descriptor_file_offsets.empty()) {
    return std::ranges::all_of(owner_descriptor_file_offsets, [&](uint64_t descriptor) {
      const ConSanKernelInfo *kernel = inventory.find_kernel_by_descriptor(descriptor);
      return kernel != nullptr && selected_name(kernel->name);
    });
  }
  return !source_containers.empty() && std::ranges::all_of(source_containers, selected_name);
}

bool consan_site_matches_kernel_allowlist(const ProgramInventory &inventory,
                                          std::span<const ConSanExecutionOwner> execution_owners,
                                          std::span<const std::string> source_containers,
                                          std::span<const std::string> kernel_name_allowlist) {
  std::vector<uint64_t> descriptors;
  descriptors.reserve(execution_owners.size());
  for (const ConSanExecutionOwner &owner : execution_owners)
    descriptors.push_back(owner.descriptor_file_offset);
  return consan_site_matches_kernel_allowlist(inventory, descriptors, source_containers,
                                              kernel_name_allowlist);
}

ConSanObservationProduct
assemble_consan_observation_product(const ProgramInventory &inventory,
                                    const ConSanObservationPolicyRequest &request) {
  ConSanObservationProduct product;
  product.atomic_fence_fragment_required = request.include_atomic_fence_policy;
  ConSanObservationPlan plan;

  ConSanAccessPolicyResult access = plan_consan_access_observation(
      inventory, {.engine = request.engine,
                  .native_lds_enabled = request.native_lds_enabled,
                  .group_flat_enabled = request.group_flat_enabled,
                  .flat_provenance_mode = request.flat_provenance_mode,
                  .container_filter = request.container_filter,
                  .kernel_name_allowlist = request.kernel_name_allowlist,
                  .reserved_for_synchronization = request.reserved_for_synchronization});
  plan = std::move(access.plan);
  product.access_errors = std::move(access.errors);

  ConSanBarrierPolicyResult barrier = plan_consan_barrier_observation(
      inventory, {.engine = request.engine,
                  .tracking_enabled = request.barrier_tracking_enabled,
                  .container_filter = request.container_filter,
                  .kernel_name_allowlist = request.kernel_name_allowlist});
  product.barrier_errors = std::move(barrier.errors);
  product.barrier_fragment_appended = plan.append(barrier.plan);

  if (request.include_atomic_fence_policy) {
    const bool sampled_access_window_available =
        std::ranges::any_of(plan.probe_intents, [](const ConSanProbeIntent &intent) {
          return intent.kind == ConSanProbeIntentKind::SampledAccess;
        });
    ConSanAtomicFencePolicyResult atomic_fence = plan_consan_atomic_fence_observation(
        inventory, {.engine = request.engine,
                    .tracking_enabled = request.atomic_fence_tracking_enabled,
                    .sampled_access_window_available = sampled_access_window_available,
                    .container_filter = request.container_filter,
                    .kernel_name_allowlist = request.kernel_name_allowlist});
    product.atomic_errors = std::move(atomic_fence.atomic_errors);
    product.fence_errors = std::move(atomic_fence.fence_errors);
    product.atomic_fence_fragment_appended = plan.append(atomic_fence.plan);
  }

  const bool structurally_valid =
      product.access_errors.empty() && product.barrier_errors.empty() &&
      product.atomic_errors.empty() && product.fence_errors.empty() &&
      product.barrier_fragment_appended &&
      (!product.atomic_fence_fragment_required || product.atomic_fence_fragment_appended) &&
      plan.valid();
  product.initial_coverage = ConSanCoverageLedger(std::move(plan));
  if (!structurally_valid) {
    render_observation_diagnostics(inventory, product);
    if (product.diagnostics.empty()) {
      product.diagnostics.emplace_back(
          request.engine == ConSanCapabilityEngine::SuperCollider
              ? "ConSan SuperCollider access policy produced an invalid observation plan"
              : "ConSan MOI policy produced an invalid observation plan");
    }
  }
  return product;
}

ConSanObservationProduct assemble_consan_observation_product(const ProgramInventory &inventory,
                                                             const ConSanRequest &request,
                                                             const ConSanDebugOverrides &debug) {
  const ConSanFlavor flavor = request.flavor.value_or(ConSanFlavor::None);
  const std::optional<ConSanCapabilityEngine> engine =
      consan_capability_engine(flavor, request.moi_engine);
  if (!engine) {
    ConSanObservationProduct product;
    product.diagnostics.emplace_back("ConSan observation policy received an invalid engine");
    return product;
  }

  const bool moi = flavor == ConSanFlavor::Moi;
  const std::vector<PhysicalSiteId> synchronization_reservations =
      moi ? ordinary_synchronization_reservations(inventory) : std::vector<PhysicalSiteId>{};
  return assemble_consan_observation_product(
      inventory, {
                     .engine = *engine,
                     .native_lds_enabled = moi || request.probe_lds_check_trap,
                     .group_flat_enabled = moi || request.probe_flat_check_trap,
                     .flat_provenance_mode = request.flat_provenance_mode,
                     .barrier_tracking_enabled = moi ? request.moi_track_barriers : true,
                     .include_atomic_fence_policy = moi,
                     .atomic_fence_tracking_enabled = moi && request.moi_track_atomics,
                     // SuperCollider's diagnostic selection filter must not shrink its
                     // physical coverage denominator. MOI's candidate filter remains an
                     // explicit diagnostic-only policy input during this migration.
                     .container_filter =
                         moi ? std::string_view(debug.test_kernel_name_filter) : std::string_view{},
                     .kernel_name_allowlist = request.kernel_name_allowlist,
                     .reserved_for_synchronization = synchronization_reservations,
                 });
}

} // namespace rocjitsu
