// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_perturbation.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_perturbation_policy.h"
#include "rocjitsu/code/patch/consan/consan_placement.h"
#include "rocjitsu/code/patch/consan/consan_sync_event_index.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/code/patch/trampoline_builder.h"
#include "rocjitsu/isa/decoder.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <ranges>
#include <utility>

namespace rocjitsu {

void build_perturbation_candidate_inventory(const ProgramInventory &program_inventory,
                                            ConSanPerturbationPlanningState &planning) {
  planning.candidates.clear();
  const SyncEventSemanticIndex events =
      build_sync_event_semantic_index(program_inventory.sync().sync_events);
  for (const ConSanSyncSequence &sequence : program_inventory.sync().sync_sequences) {
    const ConSanPerturbationKind kind =
        sequence.kind == ConSanSyncSequenceKind::Barrier  ? ConSanPerturbationKind::Barrier
        : sequence.kind == ConSanSyncSequenceKind::Atomic ? ConSanPerturbationKind::Atomic
                                                          : ConSanPerturbationKind::None;
    if (kind == ConSanPerturbationKind::None)
      continue;
    if (sequence.member_semantic_ids.empty())
      continue;
    for (const ConSanPerturbationEdge edge :
         {ConSanPerturbationEdge::Release, ConSanPerturbationEdge::Acquire}) {
      const ConSanPerturbationRejectionReason rejection =
          perturbation_rejection_reason(events, sequence, kind, edge);
      const SemanticSiteId &anchor_member = edge == ConSanPerturbationEdge::Release
                                                ? sequence.member_semantic_ids.front()
                                                : sequence.member_semantic_ids.back();
      const ConSanSyncEvent *anchor = find_sequence_member_event(events, anchor_member);
      ConSanPerturbationCandidate candidate;
      candidate.kind = kind;
      candidate.edge = edge;
      candidate.identity =
          sequence.identity + (edge == ConSanPerturbationEdge::Release ? "|perturb-edge=release"
                                                                       : "|perturb-edge=acquire");
      candidate.sequence_identity = sequence.identity;
      candidate.container_name = sequence.container_name;
      candidate.in_kernel = sequence.in_kernel;
      candidate.basic_block_index = sequence.basic_block_index.value_or(0);
      candidate.anchor_event_identity = anchor == nullptr ? std::string{} : anchor->identity;
      candidate.anchor_text_offset = anchor == nullptr ? 0 : anchor->text_offset;
      candidate.anchor_size = anchor == nullptr ? 0 : anchor->size;
      candidate.ordered_member_identities = sequence.member_event_identities;
      candidate.eligible =
          rejection == ConSanPerturbationRejectionReason::None && anchor != nullptr;
      candidate.rejection_reason =
          anchor == nullptr ? ConSanPerturbationRejectionReason::MissingAnchorEvent : rejection;
      planning.candidates.push_back(std::move(candidate));
    }
  }
}

void build_perturbation_plan(const ConSanOptions &options,
                             ConSanPerturbationPlanningState &planning,
                             ConSanTransformArtifacts &result,
                             std::span<const CarriedPerturbationPlan> carried_plans) {
  result.mutation.perturbation.requested =
      options.sc_perturb_kind == ConSanPerturbationKind::None ? 0u : options.sc_perturb_max;
  if (options.sc_perturb_kind == ConSanPerturbationKind::None)
    return;
  if (options.flavor != ConSanFlavor::SuperCollider) {
    result.errors.emplace_back("ConSan SC perturbation requires the SuperCollider flavor");
    return;
  }
  if (options.sc_perturb_max == 0u || options.sc_perturb_max > 2u) {
    result.errors.emplace_back("ConSan SC perturb maximum must be from 1 through 2");
    return;
  }
  if (options.sc_perturb_sleep == 0u || options.sc_perturb_sleep > 15u) {
    result.errors.emplace_back("ConSan SC perturb sleep immediate must be from 1 through 15");
    return;
  }
  if (options.sc_perturb_required_count > options.sc_perturb_max) {
    result.errors.emplace_back("ConSan SC perturb required count exceeds the selected maximum");
    return;
  }

  if (!carried_plans.empty()) {
    if (carried_plans.size() > options.sc_perturb_max) {
      result.errors.emplace_back(
          "ConSan carried SC perturbation plans exceed the selected maximum");
      return;
    }
    for (const CarriedPerturbationPlan &carried : carried_plans) {
      if (carried.source_candidate_identity.empty() || carried.source_sequence_identity.empty() ||
          carried.container_name.empty() || carried.anchor_event_identity.empty() ||
          carried.ordered_member_identities.empty() ||
          carried.source_anchor_text_offset >
              std::numeric_limits<uint64_t>::max() - carried.anchor_size ||
          carried.anchor_size == 0u ||
          (carried.removed_cache_boundary && !carried.overlaps_atomic_mutation)) {
        result.errors.emplace_back("ConSan carried perturbation identity is incomplete");
        continue;
      }
      const auto matches_carried = [&](const ConSanPerturbationCandidate &candidate) {
        return candidate.eligible && candidate.kind == carried.kind &&
               candidate.edge == carried.edge &&
               candidate.container_name == carried.container_name &&
               candidate.in_kernel == carried.in_kernel &&
               candidate.anchor_text_offset == carried.translated_anchor_text_offset &&
               candidate.anchor_size == carried.anchor_size;
      };
      const size_t match_count =
          static_cast<size_t>(std::ranges::count_if(planning.candidates, matches_carried));
      auto candidate = std::ranges::find_if(planning.candidates, matches_carried);
      if ((carried.removed_cache_boundary || carried.sequence_semantics_weakened) &&
          match_count == 0u) {
        planning.candidates.push_back(
            {.kind = carried.kind,
             .edge = carried.edge,
             .identity = carried.source_candidate_identity,
             .sequence_identity = carried.source_sequence_identity,
             .container_name = carried.container_name,
             .in_kernel = carried.in_kernel,
             .basic_block_index = carried.basic_block_index,
             .anchor_event_identity = carried.anchor_event_identity,
             .anchor_text_offset = carried.translated_anchor_text_offset,
             .anchor_size = carried.anchor_size,
             .ordered_member_identities = carried.ordered_member_identities,
             .eligible = true,
             .rejection_reason = {}});
        candidate = std::prev(planning.candidates.end());
      } else if (match_count != 1u || candidate == planning.candidates.end()) {
        result.errors.emplace_back(
            "ConSan could not recover exactly one carried perturbation edge with its owner");
        continue;
      }
      planning.plans.push_back(
          {.candidate_identity = candidate->identity,
           .sequence_identity = candidate->sequence_identity,
           .kind = candidate->kind,
           .edge = candidate->edge,
           .anchor_event_identity = candidate->anchor_event_identity,
           .anchor_text_offset = candidate->anchor_text_offset,
           .anchor_size = candidate->anchor_size,
           .sleep_imm = carried.sleep_imm,
           .source_candidate_identity = carried.source_candidate_identity,
           .source_sequence_identity = carried.source_sequence_identity,
           .source_container_name = carried.container_name,
           .source_anchor_event_identity = carried.anchor_event_identity,
           .source_in_kernel = carried.in_kernel,
           .source_anchor_text_offset = carried.source_anchor_text_offset,
           .source_anchor_size = carried.anchor_size,
           .source_owner_descriptor_file_offset = carried.source_owner_descriptor_file_offset,
           .overlaps_atomic_mutation = carried.overlaps_atomic_mutation,
           .removed_cache_boundary = carried.removed_cache_boundary});
    }
    result.mutation.perturbation.planned = planning.plans.size();
    if (options.sc_perturb_required_count != 0u &&
        result.mutation.perturbation.planned != options.sc_perturb_required_count) {
      result.errors.emplace_back(
          "ConSan SC perturb required " + std::to_string(options.sc_perturb_required_count) +
          " plans, got " + std::to_string(result.mutation.perturbation.planned));
    }
    return;
  }

  std::vector<const ConSanPerturbationCandidate *> matching;
  for (const ConSanPerturbationCandidate &candidate : planning.candidates) {
    if (!candidate.eligible || candidate.kind != options.sc_perturb_kind ||
        candidate.edge != options.sc_perturb_edge ||
        (!options.test_kernel_name_filter.empty() &&
         candidate.container_name.find(options.test_kernel_name_filter) == std::string::npos))
      continue;
    if (!options.sc_perturb_identity.empty() && candidate.identity != options.sc_perturb_identity)
      continue;
    matching.push_back(&candidate);
  }

  const size_t begin = options.sc_perturb_identity.empty() ? options.sc_perturb_index : 0u;
  for (size_t i = begin; i < matching.size() && planning.plans.size() < options.sc_perturb_max;
       ++i) {
    const ConSanPerturbationCandidate &candidate = *matching[i];
    planning.plans.push_back({.candidate_identity = candidate.identity,
                              .sequence_identity = candidate.sequence_identity,
                              .kind = candidate.kind,
                              .edge = candidate.edge,
                              .anchor_event_identity = candidate.anchor_event_identity,
                              .anchor_text_offset = candidate.anchor_text_offset,
                              .anchor_size = candidate.anchor_size,
                              .sleep_imm = options.sc_perturb_sleep,
                              .source_candidate_identity = {},
                              .source_sequence_identity = {},
                              .source_container_name = {},
                              .source_anchor_event_identity = {},
                              .source_in_kernel = true,
                              .source_anchor_text_offset = 0,
                              .source_anchor_size = 0,
                              .source_owner_descriptor_file_offset = std::nullopt,
                              .overlaps_atomic_mutation = false,
                              .removed_cache_boundary = false});
  }
  result.mutation.perturbation.planned = planning.plans.size();
  if (options.sc_perturb_required_count != 0u &&
      result.mutation.perturbation.planned != options.sc_perturb_required_count) {
    result.errors.emplace_back("ConSan SC perturb required " +
                               std::to_string(options.sc_perturb_required_count) + " plans, got " +
                               std::to_string(result.mutation.perturbation.planned));
  }
}

void try_apply_perturbation_patches(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                    const ConSanOptions &options,
                                    const ConSanPerturbationPlanningState &planning,
                                    ConSanTransformArtifacts &result) {
  if (planning.plans.empty())
    return;
  if (code_object.text_sections().size() != 1u) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan SC perturbation requires exactly one executable text section");
    return;
  }

  std::unique_ptr<AmdGpuCodeObject> composed_code_object;
  const AmdGpuCodeObject *emission_code_object = &code_object;
  if (!result.replacement.empty()) {
    composed_code_object =
        std::make_unique<AmdGpuCodeObject>(result.replacement.data(), result.replacement.size());
    if (!composed_code_object->is_valid() || composed_code_object->text_sections().size() != 1u) {
      result.errors.emplace_back(
          "ConSan SC perturbation could not parse the composed redundant-access image");
      return;
    }
    emission_code_object = composed_code_object.get();
  }
  CodeObjectPatcher patcher(*emission_code_object);
  const std::span<const uint8_t> text = patcher.text_bytes();
  const std::vector<LocalNopCave> caves =
      find_uncovered_nop_caves(code_object, result.program_inventory, arch);
  DbiPatchPlacementPlanner placement_planner(arch, text.size());
  for (const ConSanCommittedPatchGeometry &existing : result.patches) {
    std::string reservation_error;
    if (!placement_planner.reserve_existing_range(existing.anchor_offset, existing.original_size,
                                                  &reservation_error) ||
        (existing.trampoline_size != 0u &&
         !placement_planner.reserve_existing_range(existing.trampoline_offset,
                                                   existing.trampoline_size, &reservation_error))) {
      result.outcome = ConSanTransformOutcome::Unsupported;
      result.warnings.emplace_back(
          "ConSan SC perturbation could not reserve the composed patch image: " +
          reservation_error);
      return;
    }
  }
  struct PlannedPatch {
    const ConSanPerturbationPlan *plan = nullptr;
    const ConSanPerturbationCandidate *candidate = nullptr;
    DbiPatchPlacement placement;
    std::vector<uint32_t> body_words;
  };
  std::vector<PlannedPatch> planned;
  planned.reserve(planning.plans.size());
  const uint32_t nop = build_s_nop(0, arch);

  for (const ConSanPerturbationPlan &plan : planning.plans) {
    const auto candidate = std::ranges::find(planning.candidates, plan.candidate_identity,
                                             &ConSanPerturbationCandidate::identity);
    if (candidate == planning.candidates.end() || !candidate->eligible ||
        candidate->sequence_identity != plan.sequence_identity ||
        candidate->anchor_event_identity != plan.anchor_event_identity ||
        candidate->anchor_text_offset != plan.anchor_text_offset ||
        candidate->anchor_size != plan.anchor_size) {
      result.errors.emplace_back(
          "ConSan SC perturbation plan no longer matches its admitted candidate");
      return;
    }
    if (plan.anchor_size != sizeof(uint32_t) && plan.anchor_size != 2u * sizeof(uint32_t) &&
        plan.anchor_size != 3u * sizeof(uint32_t)) {
      result.outcome = ConSanTransformOutcome::Unsupported;
      result.warnings.emplace_back(
          "ConSan SC perturbation supports only 4-, 8-, and 12-byte boundary instructions");
      return;
    }
    if (plan.anchor_text_offset > text.size() ||
        plan.anchor_size > text.size() - plan.anchor_text_offset) {
      result.errors.emplace_back("ConSan SC perturbation anchor exceeds executable text");
      return;
    }
    std::array<uint32_t, 3> original_words{};
    std::memcpy(original_words.data(), text.data() + plan.anchor_text_offset, plan.anchor_size);
    if (plan.removed_cache_boundary) {
      const bool all_nops = std::ranges::all_of(
          std::span<const uint32_t>(original_words.data(), plan.anchor_size / sizeof(uint32_t)),
          [&](uint32_t word) { return word == nop; });
      if (!plan.overlaps_atomic_mutation || !all_nops) {
        result.errors.emplace_back(
            "ConSan carried removed-cache perturbation boundary is not the staged NOP range");
        return;
      }
    } else {
      std::unique_ptr<Instruction> original;
      std::unique_ptr<Decoder> decoder = Decoder::create(arch);
      if (decoder)
        original = decode_bounded_instruction(
            *decoder,
            std::span<const uint32_t>(original_words).first(plan.anchor_size / sizeof(uint32_t)),
            plan.anchor_text_offset);
      std::string relocatable_error;
      if (!original || static_cast<uint32_t>(original->size()) != plan.anchor_size ||
          !is_relocatable_consan_barrier_destination(*original, plan.anchor_text_offset, text, arch,
                                                     &relocatable_error)) {
        result.outcome = ConSanTransformOutcome::Unsupported;
        result.warnings.emplace_back("ConSan SC perturbation boundary is not relocatable: " +
                                     relocatable_error);
        return;
      }
    }

    DbiPatchPlacementRequest request;
    request.anchor_offset = plan.anchor_text_offset;
    request.original_size = plan.anchor_size;
    request.body_size = sizeof(uint32_t) + plan.anchor_size;
    request.allow_appended_cave = false;
    std::optional<DbiPatchPlacement> placement;
    for (const LocalNopCave &cave : caves) {
      const uint64_t cave_end =
          cave.text_offset + static_cast<uint64_t>(cave.word_count) * sizeof(uint32_t);
      const uint64_t reservation_size = request.body_size + sizeof(uint32_t);
      for (uint64_t offset = cave.text_offset;
           offset <= cave_end && reservation_size <= cave_end - offset;
           offset += sizeof(uint32_t)) {
        request.local_cave = DbiPatchLocalCave{offset, cave_end - offset};
        placement = placement_planner.plan(request);
        if (placement)
          break;
      }
      if (placement)
        break;
    }
    if (!placement) {
      request.local_cave.reset();
      request.allow_appended_cave = true;
      std::string placement_error;
      placement = placement_planner.plan(request, &placement_error);
      if (!placement) {
        result.outcome = ConSanTransformOutcome::Unsupported;
        result.warnings.emplace_back(
            "ConSan SC perturbation has no reachable local or appended cave: " + placement_error);
        return;
      }
    }

    const auto ret =
        compute_sopp_branch_simm16(placement->return_branch_offset, placement->return_target);
    if (!ret) {
      result.errors.emplace_back("ConSan SC perturbation return branch exceeds s_branch simm16");
      return;
    }
    std::vector<uint32_t> body_words;
    body_words.reserve(plan.anchor_size / sizeof(uint32_t) + 2u);
    if (plan.edge == ConSanPerturbationEdge::Release)
      body_words.push_back(build_s_sleep(static_cast<uint16_t>(plan.sleep_imm), arch));
    body_words.insert(body_words.end(), original_words.begin(),
                      original_words.begin() + plan.anchor_size / sizeof(uint32_t));
    if (plan.edge == ConSanPerturbationEdge::Acquire)
      body_words.push_back(build_s_sleep(static_cast<uint16_t>(plan.sleep_imm), arch));
    body_words.push_back(build_s_branch(*ret, arch));
    planned.push_back({.plan = &plan,
                       .candidate = &*candidate,
                       .placement = *placement,
                       .body_words = std::move(body_words)});
  }

  std::vector<uint8_t> new_text(text.begin(), text.end());
  for (const PlannedPatch &patch : planned) {
    const auto forward =
        compute_sopp_branch_simm16(patch.placement.anchor_offset, patch.placement.body_offset);
    if (!forward) {
      result.errors.emplace_back("ConSan SC perturbation forward branch exceeds s_branch simm16");
      return;
    }
    const uint32_t branch = build_s_branch(*forward, arch);
    std::memcpy(new_text.data() + patch.placement.anchor_offset, &branch, sizeof(branch));
    for (uint32_t offset = sizeof(uint32_t); offset < patch.plan->anchor_size;
         offset += sizeof(uint32_t)) {
      std::memcpy(new_text.data() + patch.placement.anchor_offset + offset, &nop, sizeof(nop));
    }
    const uint32_t trampoline_size =
        static_cast<uint32_t>(patch.body_words.size() * sizeof(uint32_t));
    if (patch.placement.kind == DbiPatchPlacementKind::AppendedCave) {
      if (new_text.size() != patch.placement.body_offset) {
        result.errors.emplace_back("ConSan SC perturbation emitted a stale appended-cave mapping");
        return;
      }
      append_consan_patch_words(new_text, patch.body_words);
    } else {
      std::memcpy(new_text.data() + patch.placement.body_offset, patch.body_words.data(),
                  trampoline_size);
    }
  }
  if (!replace_consan_text(patcher, new_text, options.patched_image_growth_limit, "SC perturbation",
                           result.program_inventory.code_object_id(), result.errors,
                           &result.transform_failure_cause)) {
    return;
  }
  result.replacement = std::move(patcher).emit();
  for (const PlannedPatch &patch : planned) {
    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineScPerturbation;
    info.anchor_offset = patch.placement.anchor_offset;
    info.trampoline_offset = patch.placement.body_offset;
    info.original_size = patch.plan->anchor_size;
    info.trampoline_size = static_cast<uint32_t>(patch.body_words.size() * sizeof(uint32_t));
    info.perturbation_edge = patch.plan->edge;
    info.perturbation_sequence_identity = patch.plan->sequence_identity;
    info.perturbation_source_candidate_identity = patch.plan->source_candidate_identity;
    info.perturbation_source_sequence_identity = patch.plan->source_sequence_identity;
    info.perturbation_source_container_name = patch.plan->source_container_name;
    info.perturbation_source_anchor_identity = patch.plan->source_anchor_event_identity.empty()
                                                   ? patch.plan->anchor_event_identity
                                                   : patch.plan->source_anchor_event_identity;
    info.perturbation_source_in_kernel = patch.plan->source_in_kernel;
    info.perturbation_source_anchor_offset = patch.plan->source_anchor_text_offset;
    info.perturbation_source_anchor_size = patch.plan->source_anchor_size;
    info.perturbation_source_owner_descriptor_file_offset =
        patch.plan->source_owner_descriptor_file_offset;
    info.perturbation_composite_atomic_overlap = patch.plan->overlaps_atomic_mutation;
    info.perturbation_composite_removed_boundary = patch.plan->removed_cache_boundary;
    if (patch.candidate->in_kernel) {
      const ConSanKernelInfo *owner =
          result.program_inventory.find_kernel_by_name(patch.candidate->container_name);
      if (owner == nullptr) {
        result.errors.emplace_back("ConSan carried perturbation lost its exact kernel owner");
        result.replacement.clear();
        return;
      }
      info.owner_descriptor_file_offsets.push_back(owner->descriptor_file_offset);
    }
    result.patches.push_back(std::move(info));
  }
  result.mutation.perturbation.applied = planned.size();
  if (!result.patches.empty())
    result.mark_modified();
}

} // namespace rocjitsu
