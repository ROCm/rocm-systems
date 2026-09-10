// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_sync_emission.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"
#include "rocjitsu/code/patch/consan/consan_moi_dynamic_record_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_engine_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_memory_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_prologue.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_runtime_workgroup_gate.h"
#include "rocjitsu/code/patch/consan/consan_placement.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace rocjitsu {

using consan_detail::MoiAtomicEvidenceSitePlan;
using consan_detail::MoiBarrierEvidenceSitePlan;
using consan_detail::MoiFenceEvidenceSitePlan;
using consan_detail::MoiSpecialStateSgprs;
using consan_detail::MoiWorkgroupKeyRegisterPlan;
using consan_detail::range_overlaps;
using consan_detail::reject_optional_scratch_range_overlap;
using consan_moi_detail::append_atomic_fetch_add_one_u32;
using consan_moi_detail::append_atomic_load_u32;
using consan_moi_detail::append_compare_moi_report_dispatch_id_word;
using consan_moi_detail::append_load_u32_vgpr_at_offset;
using consan_moi_detail::append_store_u32_vgpr_at_offset;
using consan_moi_detail::moi_has_runtime_hardware_dispatch_id;

namespace consan_moi_impl {

/// Project admitted Record/Replay fence evidence directly into the common MOI
/// lowering contract.
///
/// The `FenceRecord` intent is the only starting point. This join verifies it
/// against the immutable graph association, resolves the one address-bearing
/// instruction and its paired capture intent, and computes the exact guest
/// replacement range. Lowering therefore receives no candidate that still
/// needs semantic filtering.
[[nodiscard]] std::vector<MoiFenceEvidenceSitePlan>
build_moi_fence_evidence_site_plans(const ProgramInventory &inventory,
                                    const ConSanObservationPlan &observation,
                                    std::vector<std::string> &errors) {
  std::vector<MoiFenceEvidenceSitePlan> plans;
  const SynchronizationInventoryView graph = inventory.sync();

  for (const ConSanProbeIntent &intent : observation.probe_intents) {
    if (intent.kind != ConSanProbeIntentKind::FenceRecord)
      continue;
    const ConSanSyncEvent *fence_event = graph.find_event(intent.source_site);
    if (!intent.synchronization_association || fence_event == nullptr ||
        intent.physical_site != fence_event->semantic_id.physical ||
        fence_event->kind != ConSanSyncKind::Fence) {
      errors.emplace_back("ConSan MOI admitted fence record lost its synchronization event");
      return {};
    }

    const ConSanMoiFenceCandidate *association = nullptr;
    for (const ConSanMoiFenceCandidate &candidate : graph.moi_fence_candidates) {
      const ConSanSyncEvent *candidate_event = graph.find_event(candidate.fence_event);
      const ConSanSyncSequence *candidate_sequence = graph.find_sequence(candidate.sequence);
      if (candidate_event == nullptr || candidate_event->semantic_id != fence_event->semantic_id ||
          candidate_sequence == nullptr || !candidate.eligible() ||
          candidate_sequence->identity != intent.synchronization_association->value)
        continue;
      if (association != nullptr) {
        errors.emplace_back("ConSan MOI admitted fence record has ambiguous graph associations");
        return {};
      }
      association = &candidate;
    }
    const ConSanSyncSequence *sequence =
        association == nullptr ? nullptr : graph.find_sequence(association->sequence);
    const ConSanSyncEvent *communication =
        association != nullptr && association->communication_event
            ? graph.find_event(*association->communication_event)
            : nullptr;
    if (association == nullptr || sequence == nullptr || communication == nullptr ||
        (communication->kind != ConSanSyncKind::Atomic &&
         communication->kind != ConSanSyncKind::OrdinaryMemory) ||
        !graph.same_container(*communication, *fence_event) ||
        graph.execution_owners(*communication).empty()) {
      errors.emplace_back("ConSan MOI admitted fence record lost its communication sequence");
      return {};
    }
    const ConSanProbeIntent *capture = nullptr;
    for (const ConSanProbeIntent &candidate : observation.probe_intents) {
      if (candidate.kind != ConSanProbeIntentKind::AtomicAddressCapture ||
          candidate.synchronization_association != intent.synchronization_association ||
          std::ranges::find(candidate.covered_semantic_sites, communication->semantic_id) ==
              candidate.covered_semantic_sites.end())
        continue;
      if (capture != nullptr) {
        errors.emplace_back("ConSan MOI admitted fence record has ambiguous capture intents");
        return {};
      }
      capture = &candidate;
    }
    if (capture == nullptr || !capture->atomic_lowering_form) {
      errors.emplace_back("ConSan MOI admitted fence record lost its capture or lowering form");
      return {};
    }
    MoiFenceEvidenceSitePlan plan;
    plan.event = graph.event_id(*fence_event);
    plan.sequence = graph.sequence_id(*sequence);
    plan.source_site = communication->source_site;
    plan.evidence_intent = intent.id;
    plan.address_capture_intent = capture->id;
    plan.communication_lowering_form = *capture->atomic_lowering_form;
    if (!plan.is_well_formed() || !resolve_moi_fence_evidence_source(inventory, plan)) {
      errors.emplace_back("ConSan MOI admitted fence record lost its decoded lowering site");
      return {};
    }
    plans.push_back(std::move(plan));
  }

  std::ranges::sort(plans, {}, [&](const MoiFenceEvidenceSitePlan &plan) {
    return graph.find_event(plan.event)->text_offset();
  });
  return plans;
}

[[nodiscard]] uint16_t inline_atomic_scratch_count(const ConSanAtomicLoweringForm &form) {
  if (form.kind == ConSanAtomicLoweringFormKind::LdsVectorOffset ||
      form.kind == ConSanAtomicLoweringFormKind::BufferResourceVectorOffset ||
      form.kind == ConSanAtomicLoweringFormKind::FlatScalarVectorAddress ||
      form.kind == ConSanAtomicLoweringFormKind::GlobalScalarVectorAddress)
    return 5u;
  const bool returned_value_aliases_address =
      form.returns_old_value && form.destination_vgpr &&
      static_cast<uint32_t>(*form.destination_vgpr) + form.destination_register_count >
          form.address_vgpr &&
      static_cast<uint32_t>(form.address_vgpr) + form.address_vgpr_count > *form.destination_vgpr;
  return returned_value_aliases_address || form.signed_byte_offset != 0 ? 5u : 3u;
}

[[nodiscard]] uint16_t fence_record_scratch_count(const ConSanAtomicLoweringForm &form) {
  // Fence records are dynamically reserved once per executing wave on every
  // architecture. The common layout keeps five indexed-store temporaries,
  // one derived-owner temporary, and a preserved 64-bit communication token.
  return std::max<uint16_t>(inline_atomic_scratch_count(form), 8u);
}

[[nodiscard]] uint16_t atomic_record_scratch_count(const ConSanAtomicLoweringForm &form) {
  // Dynamic Record/Replay publication needs seven VGPRs for the slot,
  // address, value, and preserved guest atomic address. Compare-exchange also
  // retains the pre-linearization event index while its success mask is
  // captured and the common emitter publishes the completed record.
  const uint16_t record_minimum = form.compare_exchange ? 8u : 7u;
  return std::max<uint16_t>(inline_atomic_scratch_count(form), record_minimum);
}

/// Project admitted atomic evidence directly into the common MOI lowering
/// contract.
///
/// This is the sole join between synchronization inventory, flavor policy,
/// and operand-rich guest decode. It starts directly from admitted evidence
/// intents and their paired address-capture intent. Record/Replay ordinary
/// sequences owned by a fence have no
/// `AtomicRecord` intent and therefore do not enter the atomic-record path;
/// Sampled and InlineShadow select their own explicit evidence intent kinds.
[[nodiscard]] std::vector<MoiAtomicEvidenceSitePlan> build_moi_atomic_evidence_site_plans(
    const ProgramInventory &inventory, const ConSanObservationPlan &observation,
    ConSanProbeIntentKind evidence_kind, std::vector<std::string> &errors) {
  std::vector<MoiAtomicEvidenceSitePlan> plans;
  const SynchronizationInventoryView graph = inventory.sync();

  for (const ConSanProbeIntent &intent : observation.probe_intents) {
    if (intent.kind != evidence_kind)
      continue;
    if (!intent.synchronization_association) {
      errors.emplace_back("ConSan MOI admitted atomic evidence lost its capture or association");
      return {};
    }
    const ConSanSyncEvent *event = graph.find_event(intent.source_site);
    const ConSanSyncSequence *sequence =
        graph.find_unique_sequence(intent.synchronization_association->value);
    if (event == nullptr || sequence == nullptr ||
        graph.find_unique_sequence_containing(event->semantic_id) != sequence ||
        (event->kind != ConSanSyncKind::Atomic && event->kind != ConSanSyncKind::OrdinaryMemory)) {
      errors.emplace_back("ConSan MOI admitted atomic evidence lost its synchronization graph");
      return {};
    }
    const ConSanProbeIntent *capture = nullptr;
    for (const ConSanProbeIntent &candidate : observation.probe_intents) {
      if (candidate.kind != ConSanProbeIntentKind::AtomicAddressCapture ||
          candidate.synchronization_association != intent.synchronization_association ||
          std::ranges::find(candidate.covered_semantic_sites, event->semantic_id) ==
              candidate.covered_semantic_sites.end())
        continue;
      if (capture != nullptr) {
        errors.emplace_back("ConSan MOI admitted atomic evidence has ambiguous capture intents");
        return {};
      }
      capture = &candidate;
    }
    if (capture == nullptr || !capture->atomic_lowering_form) {
      errors.emplace_back("ConSan MOI admitted atomic evidence lost its decoded lowering site");
      return {};
    }
    MoiAtomicEvidenceSitePlan plan;
    plan.event = graph.event_id(*event);
    plan.sequence = graph.sequence_id(*sequence);
    plan.source_site = event->source_site;
    plan.address_capture_intent = capture->id;
    plan.evidence_intent = intent.id;
    plan.lowering_form = *capture->atomic_lowering_form;
    if (!resolve_moi_atomic_evidence_source(inventory, plan) || !plan.is_well_formed()) {
      errors.emplace_back("ConSan MOI admitted atomic evidence lost its decoded lowering site");
      return {};
    }
    plans.push_back(std::move(plan));
  }

  std::ranges::sort(plans, {}, [&](const MoiAtomicEvidenceSitePlan &plan) {
    return inventory.program_site(plan.source_site)->text_offset();
  });
  return plans;
}

[[nodiscard]] bool append_atomic_scalar_clause_patch(std::span<const uint8_t> text,
                                                     const ConSanAtomicSite &site,
                                                     std::optional<uint64_t> scalar_clause_offset,
                                                     rj_code_arch_t arch,
                                                     std::vector<ConSanPatchInfo> &patches,
                                                     std::vector<std::string> &errors) {
  if (!scalar_clause_offset)
    return true;
  const uint64_t offset = *scalar_clause_offset;
  if (offset > text.size() || sizeof(uint32_t) > text.size() - offset ||
      offset >= site.text_offset) {
    errors.emplace_back("ConSan MOI atomic scalar-clause offset is outside its text prefix");
    return false;
  }
  uint32_t word = 0;
  std::memcpy(&word, text.data() + offset, sizeof(word));
  const uint32_t nop = build_s_nop(0, arch);
  if (word == nop || std::ranges::any_of(patches, [&](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::InlineScalarClauseNopRewrite &&
               patch.anchor_offset == offset;
      }))
    return true;

  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder) {
    errors.emplace_back("ConSan MOI atomic scalar-clause validation could not create a decoder");
    return false;
  }
  std::array<uint32_t, 4> words{};
  const size_t available = std::min(sizeof(words), text.size() - offset);
  std::memcpy(words.data(), text.data() + offset, available);
  std::unique_ptr<Instruction> instruction = decode_bounded_instruction(
      *decoder, std::span<const uint32_t>(words).first(available / sizeof(uint32_t)), offset);
  if (!instruction || instruction->mnemonic() != "s_clause") {
    errors.emplace_back("ConSan MOI atomic scalar-clause prefix is no longer an s_clause");
    return false;
  }
  ConSanPatchInfo info;
  info.kind = ConSanPatchKind::InlineScalarClauseNopRewrite;
  info.anchor_offset = offset;
  info.trampoline_offset = offset;
  info.original_size = sizeof(nop);
  patches.push_back(std::move(info));
  return true;
}

} // namespace consan_moi_impl
} // namespace rocjitsu
