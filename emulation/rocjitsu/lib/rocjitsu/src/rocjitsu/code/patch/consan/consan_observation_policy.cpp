// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_observation_policy.cpp
/// @brief Authoritative composition of ConSan semantic-policy products.

#include "rocjitsu/code/patch/consan/consan.h"

#include <algorithm>
#include <string>
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
  const bool supercollider = product.plan.engine == ConSanCapabilityEngine::SuperCollider;
  const std::string engine_name =
      supercollider ? "SuperCollider"
                    : std::string(consan_capability_engine_name(product.plan.engine));

  for (const ConSanSiteDecision &decision : product.plan.site_decisions) {
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
  for (const ConSanBarrierSiteDecision &decision : product.plan.barrier_site_decisions) {
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
  for (const ConSanAtomicSiteDecision &decision : product.plan.atomic_site_decisions) {
    if (decision.reason != ConSanAtomicPolicyReason::ConflictingPhysicalAliases)
      continue;
    product.diagnostics.emplace_back(
        "ConSan MOI physical atomic at original text offset " +
        std::to_string(decision.semantic_site.physical.original_text_offset) +
        " was decoded inconsistently through aliases '" +
        quoted_aliases(decision.source_containers) + "'");
  }
  for (const ConSanFenceSiteDecision &decision : product.plan.fence_site_decisions) {
    if (decision.reason != ConSanFencePolicyReason::ConflictingPhysicalAliases)
      continue;
    product.diagnostics.emplace_back(
        "ConSan MOI physical fence at original text offset " +
        std::to_string(decision.semantic_site.physical.original_text_offset) +
        " was decoded inconsistently through aliases '" +
        quoted_aliases(decision.source_containers) + "'");
  }
}

} // namespace

ConSanObservationProduct
assemble_consan_observation_product(const ProgramInventory &inventory,
                                    const ConSanObservationPolicyRequest &request) {
  ConSanObservationProduct product;
  product.atomic_fence_fragment_required = request.include_atomic_fence_policy;

  ConSanAccessPolicyResult access = plan_consan_access_observation(
      inventory, {.engine = request.engine,
                  .native_lds_enabled = request.native_lds_enabled,
                  .group_flat_enabled = request.group_flat_enabled,
                  .flat_provenance_mode = request.flat_provenance_mode,
                  .container_filter = request.container_filter,
                  .reserved_for_synchronization = request.reserved_for_synchronization});
  product.plan = std::move(access.plan);
  product.access_errors = std::move(access.errors);

  ConSanBarrierPolicyResult barrier = plan_consan_barrier_observation(
      inventory, {.engine = request.engine,
                  .tracking_enabled = request.barrier_tracking_enabled,
                  .container_filter = request.container_filter});
  product.barrier_errors = std::move(barrier.errors);
  product.barrier_fragment_appended = product.plan.append(barrier.plan);

  if (request.include_atomic_fence_policy) {
    const bool sampled_access_window_available =
        std::ranges::any_of(product.plan.probe_intents, [](const ConSanProbeIntent &intent) {
          return intent.kind == ConSanProbeIntentKind::SampledAccess;
        });
    ConSanAtomicFencePolicyResult atomic_fence = plan_consan_atomic_fence_observation(
        inventory, {.engine = request.engine,
                    .tracking_enabled = request.atomic_fence_tracking_enabled,
                    .sampled_access_window_available = sampled_access_window_available,
                    .container_filter = request.container_filter});
    product.atomic_errors = std::move(atomic_fence.atomic_errors);
    product.fence_errors = std::move(atomic_fence.fence_errors);
    product.atomic_fence_fragment_appended = product.plan.append(atomic_fence.plan);
  }

  product.initial_coverage = ConSanCoverageLedger(product.plan);
  const bool structurally_valid =
      product.access_errors.empty() && product.barrier_errors.empty() &&
      product.atomic_errors.empty() && product.fence_errors.empty() &&
      product.barrier_fragment_appended &&
      (!product.atomic_fence_fragment_required || product.atomic_fence_fragment_appended) &&
      product.plan.valid();
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

} // namespace rocjitsu
