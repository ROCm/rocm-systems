// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_perturbation.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_perturbation_policy.h"
#include "rocjitsu/code/patch/consan/consan_placement.h"
#include "rocjitsu/code/patch/consan/consan_sync_event_index.h"
#include "rocjitsu/code/patch/consan/consan_text_relocation.h"
#include "rocjitsu/code/patch/rdna4_instrumentation_builder.h"
#include "rocjitsu/isa/decoder.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <ranges>
#include <utility>

namespace rocjitsu {

namespace {

struct PerturbationCandidateFacts {
  const ConSanSyncSequence *sequence = nullptr;
  const ConSanSyncEvent *anchor = nullptr;
  const ConSanProgramSite *anchor_source = nullptr;
  const ConSanProgramContainer *anchor_container = nullptr;

  [[nodiscard]] bool complete() const {
    return sequence != nullptr && anchor != nullptr && anchor_source != nullptr &&
           anchor_container != nullptr;
  }
};

[[nodiscard]] PerturbationCandidateFacts
perturbation_candidate_facts(const ProgramInventory &inventory,
                             const ConSanPerturbationCandidate &candidate) {
  const SynchronizationInventoryView sync = inventory.sync();
  const ConSanSyncSequence *sequence = sync.find_sequence(candidate.sequence);
  const ConSanSyncEvent *anchor = sync.find_event(candidate.anchor_event);
  const ConSanProgramSite *anchor_source =
      anchor == nullptr ? nullptr : inventory.program_site(anchor->source_site);
  return {
      .sequence = sequence,
      .anchor = anchor,
      .anchor_source = anchor_source,
      .anchor_container =
          anchor_source == nullptr ? nullptr : inventory.container(anchor_source->container.id),
  };
}

[[nodiscard]] std::optional<ConSanPerturbationPlan>
materialize_perturbation_plan(const ProgramInventory &inventory,
                              const ConSanPerturbationCandidate &candidate, uint32_t sleep_imm) {
  const PerturbationCandidateFacts facts = perturbation_candidate_facts(inventory, candidate);
  if (!facts.complete())
    return std::nullopt;
  return ConSanPerturbationPlan{
      .candidate = candidate,
      .sequence_identity = facts.sequence->identity,
      .container_name = facts.anchor_container->name,
      .in_kernel = facts.anchor_container->is_kernel(),
      .anchor_text_offset = facts.anchor->text_offset(),
      .anchor_size = facts.anchor_source->size(),
      .sleep_imm = sleep_imm,
      .source_sequence_identity = std::nullopt,
      .overlaps_atomic_mutation = false,
      .removed_cache_boundary = false,
      .source_sequence_semantics_weakened = false,
  };
}

} // namespace

std::string consan_perturbation_candidate_identity(const ProgramInventory &program_inventory,
                                                   const ConSanPerturbationCandidate &candidate) {
  const ConSanSyncSequence *sequence = program_inventory.sync().find_sequence(candidate.sequence);
  if (sequence == nullptr)
    return {};
  return sequence->identity + (candidate.edge == ConSanPerturbationEdge::Release
                                   ? "|perturb-edge=release"
                                   : "|perturb-edge=acquire");
}

void build_perturbation_candidate_inventory(const ProgramInventory &program_inventory,
                                            ConSanPerturbationPlanningState &planning) {
  planning.candidates.clear();
  const SynchronizationInventoryView events = program_inventory.sync();
  for (const ConSanSyncSequence &sequence : program_inventory.sync().sync_sequences) {
    const ConSanPerturbationKind kind =
        sequence.kind == ConSanSyncKind::Barrier  ? ConSanPerturbationKind::Barrier
        : sequence.kind == ConSanSyncKind::Atomic ? ConSanPerturbationKind::Atomic
                                                  : ConSanPerturbationKind::None;
    if (kind == ConSanPerturbationKind::None)
      continue;
    if (sequence.member_event_ids.empty())
      continue;
    const bool members_resolve =
        std::ranges::all_of(sequence.member_event_ids, [&](ConSanSyncEventId member) {
          return events.find_event(member) != nullptr;
        });
    for (const ConSanPerturbationEdge edge :
         {ConSanPerturbationEdge::Release, ConSanPerturbationEdge::Acquire}) {
      const ConSanPerturbationRejectionReason rejection =
          perturbation_rejection_reason(events, sequence, kind, edge);
      const ConSanSyncEventId anchor_member = edge == ConSanPerturbationEdge::Release
                                                  ? sequence.member_event_ids.front()
                                                  : sequence.member_event_ids.back();
      const ConSanSyncEvent *anchor = events.find_event(anchor_member);
      ConSanPerturbationCandidate candidate;
      candidate.kind = kind;
      candidate.edge = edge;
      candidate.sequence = events.sequence_id(sequence);
      candidate.anchor_event = anchor_member;
      candidate.eligible = rejection == ConSanPerturbationRejectionReason::None &&
                           anchor != nullptr && members_resolve;
      candidate.rejection_reason = anchor == nullptr || !members_resolve
                                       ? ConSanPerturbationRejectionReason::MissingAnchorEvent
                                       : rejection;
      planning.candidates.push_back(std::move(candidate));
    }
  }
}

void build_perturbation_plan(const ProgramInventory &program_inventory,
                             const ConSanRequest &request, const MutationRequest &mutation,
                             const ConSanDebugOverrides &debug,
                             ConSanPerturbationPlanningState &planning, ConSanMutationTally &tally,
                             std::vector<std::string> &errors,
                             std::span<const ConSanPerturbationPlan> carried_plans) {
  tally.requested =
      mutation.sc_perturb_kind == ConSanPerturbationKind::None ? 0u : mutation.sc_perturb_max;
  if (mutation.sc_perturb_kind == ConSanPerturbationKind::None)
    return;
  if (request.flavor != ConSanFlavor::SuperCollider) {
    errors.emplace_back("ConSan SC perturbation requires the SuperCollider flavor");
    return;
  }
  if (mutation.sc_perturb_max == 0u || mutation.sc_perturb_max > 2u) {
    errors.emplace_back("ConSan SC perturb maximum must be from 1 through 2");
    return;
  }
  if (mutation.sc_perturb_sleep == 0u || mutation.sc_perturb_sleep > 15u) {
    errors.emplace_back("ConSan SC perturb sleep immediate must be from 1 through 15");
    return;
  }
  if (mutation.sc_perturb_required_count > mutation.sc_perturb_max) {
    errors.emplace_back("ConSan SC perturb required count exceeds the selected maximum");
    return;
  }

  if (!carried_plans.empty()) {
    if (carried_plans.size() > mutation.sc_perturb_max) {
      errors.emplace_back("ConSan carried SC perturbation plans exceed the selected maximum");
      return;
    }
    for (const ConSanPerturbationPlan &carried : carried_plans) {
      const ConSanPerturbationCandidate &translated = carried.candidate;
      if (!carried.source_sequence_identity || carried.source_sequence_identity->empty() ||
          carried.anchor_text_offset > std::numeric_limits<uint64_t>::max() - carried.anchor_size ||
          carried.anchor_size == 0u ||
          (carried.removed_cache_boundary && !carried.overlaps_atomic_mutation)) {
        errors.emplace_back("ConSan carried perturbation identity is incomplete");
        continue;
      }
      const auto matches_carried = [&](const ConSanPerturbationCandidate &candidate) {
        const PerturbationCandidateFacts facts =
            perturbation_candidate_facts(program_inventory, candidate);
        return candidate.eligible && candidate.kind == translated.kind &&
               candidate.edge == translated.edge && facts.complete() &&
               facts.anchor_container->name == carried.container_name &&
               facts.anchor_container->is_kernel() == carried.in_kernel &&
               facts.anchor->text_offset() == carried.anchor_text_offset &&
               facts.anchor_source->size() == carried.anchor_size;
      };
      const size_t match_count =
          static_cast<size_t>(std::ranges::count_if(planning.candidates, matches_carried));
      auto candidate = std::ranges::find_if(planning.candidates, matches_carried);
      if ((carried.removed_cache_boundary || carried.source_sequence_semantics_weakened) &&
          match_count == 0u) {
        ConSanPerturbationCandidate synthetic = translated;
        synthetic.sequence = {};
        synthetic.anchor_event = {};
        synthetic.eligible = true;
        synthetic.rejection_reason = {};
        planning.candidates.push_back(std::move(synthetic));
        candidate = std::prev(planning.candidates.end());
      } else if (match_count != 1u || candidate == planning.candidates.end()) {
        errors.emplace_back(
            "ConSan could not recover exactly one carried perturbation edge with its owner");
        continue;
      }
      ConSanPerturbationPlan plan = carried;
      plan.candidate = *candidate;
      if (const PerturbationCandidateFacts facts =
              perturbation_candidate_facts(program_inventory, *candidate);
          facts.complete()) {
        plan.sequence_identity = facts.sequence->identity;
        plan.container_name = facts.anchor_container->name;
        plan.in_kernel = facts.anchor_container->is_kernel();
        plan.anchor_text_offset = facts.anchor->text_offset();
        plan.anchor_size = facts.anchor_source->size();
      }
      planning.plans.push_back(std::move(plan));
    }
    tally.planned = planning.plans.size();
    if (mutation.sc_perturb_required_count != 0u &&
        tally.planned != mutation.sc_perturb_required_count) {
      errors.emplace_back("ConSan SC perturb required " +
                          std::to_string(mutation.sc_perturb_required_count) + " plans, got " +
                          std::to_string(tally.planned));
    }
    return;
  }

  std::vector<const ConSanPerturbationCandidate *> matching;
  for (const ConSanPerturbationCandidate &candidate : planning.candidates) {
    const PerturbationCandidateFacts facts =
        perturbation_candidate_facts(program_inventory, candidate);
    if (!candidate.eligible || candidate.kind != mutation.sc_perturb_kind ||
        candidate.edge != mutation.sc_perturb_edge || !facts.complete() ||
        (!debug.test_kernel_name_filter.empty() &&
         facts.anchor_container->name.find(debug.test_kernel_name_filter) == std::string::npos))
      continue;
    if (!mutation.sc_perturb_identity.empty() &&
        consan_perturbation_candidate_identity(program_inventory, candidate) !=
            mutation.sc_perturb_identity)
      continue;
    matching.push_back(&candidate);
  }

  const size_t begin = mutation.sc_perturb_identity.empty() ? mutation.sc_perturb_index : 0u;
  for (size_t i = begin; i < matching.size() && planning.plans.size() < mutation.sc_perturb_max;
       ++i) {
    const ConSanPerturbationCandidate &candidate = *matching[i];
    if (auto plan =
            materialize_perturbation_plan(program_inventory, candidate, mutation.sc_perturb_sleep))
      planning.plans.push_back(std::move(*plan));
  }
  tally.planned = planning.plans.size();
  if (mutation.sc_perturb_required_count != 0u &&
      tally.planned != mutation.sc_perturb_required_count) {
    errors.emplace_back("ConSan SC perturb required " +
                        std::to_string(mutation.sc_perturb_required_count) + " plans, got " +
                        std::to_string(tally.planned));
  }
}

bool stage_perturbation_patches(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                const ConSanPerturbationPlanningState &planning,
                                ConSanTransformArtifacts &result) {
  if (planning.plans.empty())
    return true;
  if (code_object.text_sections().size() != 1u) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan SC perturbation requires exactly one executable text section");
    return false;
  }

  std::unique_ptr<AmdGpuCodeObject> composed_code_object;
  const AmdGpuCodeObject *emission_code_object = &code_object;
  if (!result.replacement.empty()) {
    composed_code_object =
        std::make_unique<AmdGpuCodeObject>(result.replacement.data(), result.replacement.size());
    if (!composed_code_object->is_valid() || composed_code_object->text_sections().size() != 1u) {
      result.errors.emplace_back(
          "ConSan SC perturbation could not parse the composed redundant-access image");
      return false;
    }
    emission_code_object = composed_code_object.get();
  }
  const Section &text_section = *emission_code_object->text_sections().front();
  const std::span<const uint8_t> text(reinterpret_cast<const uint8_t *>(text_section.data()),
                                      text_section.size());
  std::vector<ConSanTextFragment> fragments;
  fragments.reserve(planning.plans.size());
  const uint32_t nop = build_s_nop(0, arch);

  for (const ConSanPerturbationPlan &plan : planning.plans) {
    const ConSanPerturbationCandidate &candidate = plan.candidate;
    if (!candidate.eligible) {
      result.errors.emplace_back("ConSan SC perturbation plan retained a rejected candidate");
      return false;
    }
    if (plan.anchor_size != sizeof(uint32_t) && plan.anchor_size != 2u * sizeof(uint32_t) &&
        plan.anchor_size != 3u * sizeof(uint32_t)) {
      result.outcome = ConSanTransformOutcome::Unsupported;
      result.warnings.emplace_back(
          "ConSan SC perturbation supports only 4-, 8-, and 12-byte boundary instructions");
      return false;
    }
    if (plan.anchor_text_offset > text.size() ||
        plan.anchor_size > text.size() - plan.anchor_text_offset) {
      result.errors.emplace_back("ConSan SC perturbation anchor exceeds executable text");
      return false;
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
        return false;
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
        return false;
      }
    }

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineScPerturbation;
    info.anchor_offset = plan.anchor_text_offset;
    info.original_size = plan.anchor_size;
    info.perturbation_edge = candidate.edge;
    info.perturbation_sequence_identity = plan.sequence_identity;
    info.perturbation_source_sequence_identity = plan.source_sequence_identity.value_or("");
    info.perturbation_composite_atomic_overlap = plan.overlaps_atomic_mutation;
    info.perturbation_composite_removed_boundary = plan.removed_cache_boundary;
    if (plan.in_kernel) {
      const ConSanProgramContainer *owner =
          result.program_inventory.find_kernel_by_name(plan.container_name);
      if (owner == nullptr) {
        result.errors.emplace_back("ConSan carried perturbation lost its exact kernel owner");
        return false;
      }
      info.owner_descriptor_file_offsets.push_back(owner->descriptor_file_offset);
    }
    ConSanTextFragment fragment;
    fragment.kind = ConSanTextFragmentKind::Around;
    const uint32_t sleep = build_s_sleep(static_cast<uint16_t>(plan.sleep_imm), arch);
    if (candidate.edge == ConSanPerturbationEdge::Release)
      fragment.before_words.push_back(sleep);
    else
      fragment.after_words.push_back(sleep);
    fragment.patch = std::move(info);
    fragments.push_back(std::move(fragment));
  }
  return stage_consan_text_fragments(std::move(fragments), result);
}

} // namespace rocjitsu
