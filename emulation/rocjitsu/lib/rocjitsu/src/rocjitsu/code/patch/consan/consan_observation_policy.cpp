// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_observation_policy.cpp
/// @brief Authoritative composition of ConSan semantic-policy products.

#include "rocjitsu/code/patch/consan/consan.h"

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>

namespace rocjitsu::consan {
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
                                    ObservationProduct &product) {
  const bool supercollider = product.plan().mode == Mode::SuperCollider;
  const std::string mode_name =
      supercollider ? "SuperCollider" : std::string(mode_label(product.plan().mode));

  for (const SiteDecision &decision : product.plan().site_decisions) {
    if (decision.reason != AccessPolicyReason::ConflictingPhysicalAliases)
      continue;
    if (supercollider) {
      const auto access =
          std::ranges::find_if(inventory.access_sites(), [&](const auto &candidate) {
            return candidate.physical_id == decision.semantic_site.physical;
          });
      const std::string_view access_kind =
          access != inventory.access_sites().end() && access->origin == AccessOrigin::Flat
              ? "FLAT"
              : "native-LDS";
      product.diagnostics.emplace_back(
          "ConSan SuperCollider " + std::string(access_kind) + " access at original text offset " +
          std::to_string(decision.semantic_site.physical.original_text_offset) +
          " was decoded inconsistently through aliases '" +
          quoted_aliases(inventory.source_container_names(decision.semantic_site.physical)) + "'");
    } else {
      product.diagnostics.emplace_back(
          "ConSan physical access at original text offset " +
          std::to_string(decision.semantic_site.physical.original_text_offset) +
          " was decoded inconsistently through aliases '" +
          quoted_aliases(inventory.source_container_names(decision.semantic_site.physical)) + "'");
    }
  }
  for (const BarrierSiteDecision &decision : product.plan().barrier_site_decisions) {
    if (decision.reason != BarrierPolicyReason::ConflictingPhysicalAliases)
      continue;
    std::string message = "ConSan " + mode_name + " ";
    if (!supercollider)
      message = "ConSan physical ";
    message += "barrier at original text offset " +
               std::to_string(decision.semantic_site.physical.original_text_offset) +
               " was decoded inconsistently through ";
    message += supercollider ? "physical aliases"
                             : "aliases '" +
                                   quoted_aliases(inventory.source_container_names(
                                       decision.semantic_site.physical)) +
                                   "'";
    product.diagnostics.push_back(std::move(message));
  }
  for (const AtomicSiteDecision &decision : product.plan().atomic_site_decisions) {
    if (decision.reason != AtomicPolicyReason::ConflictingPhysicalAliases)
      continue;
    product.diagnostics.emplace_back(
        "ConSan physical atomic at original text offset " +
        std::to_string(decision.semantic_site.physical.original_text_offset) +
        " was decoded inconsistently through aliases '" +
        quoted_aliases(inventory.source_container_names(decision.semantic_site.physical)) + "'");
  }
  for (const FenceSiteDecision &decision : product.plan().fence_site_decisions) {
    if (decision.reason != FencePolicyReason::ConflictingPhysicalAliases)
      continue;
    product.diagnostics.emplace_back(
        "ConSan physical fence at original text offset " +
        std::to_string(decision.semantic_site.physical.original_text_offset) +
        " was decoded inconsistently through aliases '" +
        quoted_aliases(inventory.source_container_names(decision.semantic_site.physical)) + "'");
  }
}

[[nodiscard]] std::vector<PhysicalSiteId>
ordinary_synchronization_reservations(const ProgramInventory &inventory) {
  std::vector<PhysicalSiteId> reservations;
  const SynchronizationInventoryView sync = inventory.sync();
  for (const SyncSequence &sequence : sync.sync_sequences) {
    if (sequence.kind != SyncKind::OrdinaryMemory ||
        (sequence.memory_role != SyncMemoryRole::Acquire &&
         sequence.memory_role != SyncMemoryRole::Release) ||
        !sync_confidence_meets(sequence.confidence, SemanticConfidence::Conservative) ||
        !sync_confidence_meets(sequence.memory_role_confidence, SemanticConfidence::Conservative)) {
      continue;
    }
    const SyncEvent *communication = nullptr;
    for (SyncEventId identity : sequence.member_event_ids) {
      const SyncEvent *event = sync.find_event(identity);
      if (event == nullptr || event->kind != SyncKind::OrdinaryMemory)
        continue;
      if (communication != nullptr) {
        communication = nullptr;
        break;
      }
      communication = event;
    }
    const ProgramSite *communication_source =
        communication == nullptr ? nullptr : sync.source(*communication);
    const ProgramContainer *sequence_container = sync.container(sequence);
    if (communication_source == nullptr || sequence_container == nullptr ||
        communication_source->container != sequence_container->id) {
      continue;
    }
    for (const ProgramSite &access : inventory.access_sites()) {
      if (access.container != communication_source->container ||
          access.physical_id.original_text_offset != communication->text_offset() ||
          std::ranges::find(reservations, access.physical_id) != reservations.end()) {
        continue;
      }
      reservations.push_back(access.physical_id);
    }
  }
  return reservations;
}

} // namespace

bool site_matches_kernel_allowlist(const ProgramInventory &inventory,
                                   std::span<const uint64_t> owner_descriptor_file_offsets,
                                   std::span<const std::string> source_containers,
                                   std::span<const std::string> kernel_name_allowlist) {
  if (kernel_name_allowlist.empty())
    return true;
  const auto selected_name = [&](std::string_view candidate) {
    return std::ranges::any_of(kernel_name_allowlist, [&](std::string_view allowed) {
      return kernel_name_matches(allowed, candidate);
    });
  };
  if (!owner_descriptor_file_offsets.empty()) {
    return std::ranges::all_of(owner_descriptor_file_offsets, [&](uint64_t descriptor) {
      const ProgramContainer *kernel = inventory.find_kernel_by_descriptor(descriptor);
      return kernel != nullptr && selected_name(kernel->name);
    });
  }
  return !source_containers.empty() && std::ranges::all_of(source_containers, selected_name);
}

bool site_matches_kernel_allowlist(const ProgramInventory &inventory,
                                   std::span<const ExecutionOwner> execution_owners,
                                   std::span<const std::string> source_containers,
                                   std::span<const std::string> kernel_name_allowlist) {
  std::vector<uint64_t> descriptors;
  descriptors.reserve(execution_owners.size());
  for (const ExecutionOwner &owner : execution_owners) {
    if (const ProgramContainer *kernel = inventory.kernel(owner))
      descriptors.push_back(kernel->descriptor_file_offset);
  }
  return site_matches_kernel_allowlist(inventory, descriptors, source_containers,
                                       kernel_name_allowlist);
}

ObservationProduct assemble_observation_product(const ProgramInventory &inventory,
                                                const ObservationPolicyRequest &request) {
  ObservationProduct product;
  product.atomic_fence_fragment_required = request.include_atomic_fence_policy;
  ObservationPlan plan;

  AccessPolicyResult access = plan_access_observation(
      inventory, {.mode = request.mode,
                  .native_lds_enabled = request.native_lds_enabled,
                  .group_flat_enabled = request.group_flat_enabled,
                  .flat_provenance_mode = request.flat_provenance_mode,
                  .container_filter = request.container_filter,
                  .kernel_name_allowlist = request.kernel_name_allowlist,
                  .reserved_for_synchronization = request.reserved_for_synchronization});
  plan = std::move(access.plan);
  product.access_errors = std::move(access.errors);

  BarrierPolicyResult barrier =
      plan_barrier_observation(inventory, {.mode = request.mode,
                                           .tracking_enabled = request.barrier_tracking_enabled,
                                           .container_filter = request.container_filter,
                                           .kernel_name_allowlist = request.kernel_name_allowlist});
  product.barrier_errors = std::move(barrier.errors);
  product.barrier_fragment_appended = plan.append(barrier.plan);

  if (request.include_atomic_fence_policy) {
    std::vector<DirectionalAccessAvailability> directional_access_windows;
    const ModeProbeVocabulary *vocabulary = mode_probe_vocabulary(request.mode);
    if (vocabulary && vocabulary->ordering_requires_directional_access_window) {
      for (const ProbeIntent &intent : plan.probe_intents) {
        if (intent.kind != vocabulary->access)
          continue;
        for (const ProgramSite &site : inventory.access_sites()) {
          if (site.physical_id != intent.physical_site)
            continue;
          for (ProgramContainerId owner : inventory.execution_owner_kernels(site)) {
            auto availability = std::ranges::find(directional_access_windows, owner,
                                                  &DirectionalAccessAvailability::owner);
            if (availability == directional_access_windows.end()) {
              directional_access_windows.push_back({.owner = owner});
              availability = std::prev(directional_access_windows.end());
            }
            availability->read |=
                site.kind == LdsAccessKind::Read || site.kind == LdsAccessKind::Atomic;
            availability->write |=
                site.kind == LdsAccessKind::Write || site.kind == LdsAccessKind::Atomic;
          }
        }
      }
    }
    AtomicFencePolicyResult atomic_fence = plan_atomic_fence_observation(
        inventory, {.mode = request.mode,
                    .tracking_enabled = request.atomic_fence_tracking_enabled,
                    .directional_access_windows = directional_access_windows,
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
  product.initial_coverage = CoverageLedger(std::move(plan));
  if (!structurally_valid) {
    render_observation_diagnostics(inventory, product);
    if (product.diagnostics.empty()) {
      product.diagnostics.emplace_back(
          request.mode == Mode::SuperCollider
              ? "ConSan SuperCollider access policy produced an invalid observation plan"
              : "ConSan policy produced an invalid observation plan");
    }
  }
  return product;
}

ObservationProduct assemble_observation_product(const ProgramInventory &inventory,
                                                const Request &request,
                                                const DebugOverrides &debug) {
  const Mode mode = request.mode.value_or(Mode::None);
  const std::optional<Mode> enabled_mode = consan::enabled_mode(mode);
  if (!enabled_mode) {
    ObservationProduct product;
    product.diagnostics.emplace_back("ConSan observation policy received an invalid mode");
    return product;
  }

  const bool uses_reports = mode == Mode::Default;
  const std::vector<PhysicalSiteId> synchronization_reservations =
      uses_reports ? ordinary_synchronization_reservations(inventory)
                   : std::vector<PhysicalSiteId>{};
  return assemble_observation_product(
      inventory,
      {
          .mode = *enabled_mode,
          .native_lds_enabled = uses_reports || request.probe_lds_check_trap,
          .group_flat_enabled = uses_reports || request.probe_flat_check_trap,
          .flat_provenance_mode = request.flat_provenance_mode,
          .barrier_tracking_enabled = uses_reports ? request.track_barriers : true,
          .include_atomic_fence_policy = uses_reports,
          .atomic_fence_tracking_enabled = uses_reports && request.track_atomics,
          // SuperCollider's diagnostic selection filter must not shrink its
          // physical coverage denominator. ConSan's candidate filter remains an
          // explicit diagnostic-only policy input during this migration.
          .container_filter =
              uses_reports ? std::string_view(debug.test_kernel_name_filter) : std::string_view{},
          .kernel_name_allowlist = request.kernel_name_allowlist,
          .reserved_for_synchronization = synchronization_reservations,
      });
}

} // namespace rocjitsu::consan
