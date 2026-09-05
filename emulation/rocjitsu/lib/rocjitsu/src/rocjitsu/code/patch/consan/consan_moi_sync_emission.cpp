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
#include "rocjitsu/code/patch/consan/consan_moi_record_event_emission.h"
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

[[nodiscard]] bool append_moi_sync_intent_lowering_commit(
    const ConSanObservationPlan &observation, std::vector<std::string> &errors,
    std::span<const ConSanProbeIntentId> intent_ids, const ConSanCommittedPatchGeometry &patch,
    std::string_view probe_name, std::vector<ConSanCommittedLowering> &commits) {
  auto commit = make_consan_instrumented_patch_lowering(observation, intent_ids, patch);
  if (!commit) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " produced an invalid intent-bound lowering");
    return false;
  }
  commits.push_back(std::move(*commit));
  return true;
}

[[nodiscard]] SampledAtomicSemanticsResult
sampled_atomic_semantics_for_source(const MoiAtomicEvidenceSourceView &source) {
  using Reason = SampledAtomicSemanticsReason;
  const auto reject = [](Reason reason) {
    return SampledAtomicSemanticsResult{.semantics = std::nullopt, .reason = reason};
  };
  const ConSanSyncSequence &sequence = *source.sequence;
  const ConSanAtomicSite &site = source.site;
  if (!consan_sync_confidence_meets(sequence.confidence, ConSanSemanticConfidence::Conservative) ||
      !consan_sync_confidence_meets(sequence.memory_role_confidence,
                                    ConSanSemanticConfidence::Conservative)) {
    return reject(Reason::UnqualifiedSharedSyncSequence);
  }

  consan_detail::SampledAtomicSemantics semantics;
  switch (sequence.memory_role) {
  case ConSanSyncMemoryRole::Release:
    semantics.role = ConSanMoiSampledSyncRole::RmwRelease;
    break;
  case ConSanSyncMemoryRole::Acquire:
    semantics.role = ConSanMoiSampledSyncRole::RmwAcquire;
    break;
  case ConSanSyncMemoryRole::AcquireRelease:
    semantics.role = ConSanMoiSampledSyncRole::RmwAcquireRelease;
    break;
  case ConSanSyncMemoryRole::Unknown:
  case ConSanSyncMemoryRole::None:
  case ConSanSyncMemoryRole::SequentiallyConsistent:
    return reject(Reason::UnsupportedQualifiedMemoryRole);
  }
  const std::optional<ConSanMemoryScope> semantic_scope = sequence.scope;
  if (!semantic_scope) {
    return reject(Reason::MissingQualifiedScope);
  }
  const auto sampled_scope = consan_moi_sampled_sync_scope(*semantic_scope);
  if (!sampled_scope)
    return reject(Reason::UnsupportedQualifiedScope);
  semantics.scope = *sampled_scope;
  if (site.width_bits == 0 || site.width_bits % 8u != 0 ||
      site.width_bits / 8u > std::numeric_limits<uint32_t>::max()) {
    return reject(Reason::UnsupportedQualifiedByteRange);
  }
  semantics.byte_count = site.width_bits / 8u;
  switch (sequence.rmw_outcome) {
  case ConSanSyncRmwOutcome::NoReturn:
    semantics.outcome = ConSanMoiSampledSyncOutcome::RmwNoReturn;
    break;
  case ConSanSyncRmwOutcome::ReturnsOldValue:
    semantics.outcome = ConSanMoiSampledSyncOutcome::RmwReturnsOld;
    break;
  case ConSanSyncRmwOutcome::CompareExchange:
    if (!site.returns_old_value.value_or(false) || !site.data_vgpr ||
        !site.destination_vgpr) {
      return reject(Reason::CompareExchangeDynamicOutcomeUnavailable);
    }
    semantics.outcome = ConSanMoiSampledSyncOutcome::CasSuccess;
    break;
  case ConSanSyncRmwOutcome::NotApplicable:
  case ConSanSyncRmwOutcome::Unknown:
    return reject(Reason::UnsupportedQualifiedRmwOutcome);
  }
  const ConSanMoiSampledSyncEncodeResult encoded = encode_consan_moi_sampled_sync_metadata({
      .address = 1,
      .byte_count = semantics.byte_count,
      .kind = ConSanMoiSampledSyncKind::Atomic,
      .role = semantics.role,
      .scope = semantics.scope,
      .outcome = semantics.outcome,
  });
  if (encoded.classification != ConSanMoiSampledSyncClassification::Valid) {
    return reject(Reason::SampledSyncAbiRejectedQualifiedSequence);
  }
  semantics.descriptor = encoded.packed.descriptor;
  if (sequence.rmw_outcome == ConSanSyncRmwOutcome::CompareExchange) {
    ConSanMoiSampledSyncMetadata failure_metadata{
        .address = 1,
        .byte_count = semantics.byte_count,
        .kind = ConSanMoiSampledSyncKind::Atomic,
        .role = semantics.role,
        .scope = semantics.scope,
        .outcome = ConSanMoiSampledSyncOutcome::CasFailure,
    };
    const ConSanMoiSampledSyncEncodeResult failure =
        encode_consan_moi_sampled_sync_metadata(failure_metadata);
    if (failure.classification != ConSanMoiSampledSyncClassification::Valid) {
      return reject(Reason::SampledSyncAbiRejectedCasFailure);
    }
    semantics.cas_failure_descriptor = failure.packed.descriptor;
  }
  return {.semantics = semantics, .reason = Reason::None};
}

std::string_view sampled_atomic_semantics_reason_name(SampledAtomicSemanticsReason reason) {
  using Reason = SampledAtomicSemanticsReason;
  constexpr auto vocabulary = make_consan_enum_vocabulary(
      "invalid-sampled-atomic-semantics-reason", consan_enum(Reason::None, ""),
      consan_enum(Reason::UnqualifiedSharedSyncSequence, "unqualified-shared-sync-sequence"),
      consan_enum(Reason::UnsupportedQualifiedMemoryRole, "unsupported-qualified-memory-role"),
      consan_enum(Reason::MissingQualifiedScope, "missing-qualified-scope"),
      consan_enum(Reason::UnsupportedQualifiedScope, "unsupported-qualified-scope"),
      consan_enum(Reason::UnsupportedQualifiedByteRange, "unsupported-qualified-byte-range"),
      consan_enum(Reason::CompareExchangeDynamicOutcomeUnavailable,
                  "compare-exchange-dynamic-outcome-unavailable"),
      consan_enum(Reason::UnsupportedQualifiedRmwOutcome, "unsupported-qualified-rmw-outcome"),
      consan_enum(Reason::SampledSyncAbiRejectedQualifiedSequence,
                  "sampled-sync-abi-rejected-qualified-sequence"),
      consan_enum(Reason::SampledSyncAbiRejectedCasFailure,
                  "sampled-sync-abi-rejected-cas-failure"));
  return vocabulary.name(reason);
}

/// Project admitted Record/Replay fence evidence directly into the common MOI
/// lowering contract.
///
/// The observation decision is the only starting point. This join verifies
/// its explicit `FenceRecord` intent against the immutable graph association,
/// resolves the one address-bearing instruction, and computes the exact guest
/// replacement range. Lowering therefore receives no candidate that still
/// needs semantic filtering.
[[nodiscard]] std::vector<MoiFenceEvidenceSitePlan>
build_moi_fence_evidence_site_plans(const ProgramInventory &inventory,
                                    const ConSanObservationPlan &observation,
                                    std::vector<std::string> &errors) {
  std::vector<MoiFenceEvidenceSitePlan> plans;
  const SynchronizationInventoryView graph = inventory.sync();

  for (const ConSanFenceSiteDecision &decision : observation.fence_site_decisions) {
    if (decision.kind != ConSanSiteDecisionKind::Admitted)
      continue;

    ConSanProbeIntentId address_capture;
    ConSanProbeIntentId evidence;
    for (ConSanProbeIntentId id : decision.intent_ids) {
      const ConSanProbeIntent *intent = observation.intent(id);
      if (intent == nullptr)
        continue;
      if (intent->kind == ConSanProbeIntentKind::AtomicAddressCapture)
        address_capture = id;
      if (intent->kind != ConSanProbeIntentKind::FenceRecord)
        continue;
      if (evidence.valid()) {
        errors.emplace_back("ConSan MOI admitted fence has multiple record intents");
        return {};
      }
      evidence = id;
    }
    if (!evidence.valid())
      continue;
    if (!address_capture.valid() || !decision.association) {
      errors.emplace_back("ConSan MOI admitted fence record lost its capture or graph association");
      return {};
    }
    const ConSanProbeIntent *intent = observation.intent(evidence);
    const ConSanSyncEvent *fence_event = graph.find_event(decision.semantic_site);
    if (intent == nullptr || intent->physical_site != decision.semantic_site.physical ||
        intent->synchronization_association != decision.association || fence_event == nullptr ||
        fence_event->kind != ConSanSyncKind::Fence) {
      errors.emplace_back("ConSan MOI admitted fence record lost its synchronization event");
      return {};
    }

    const ConSanMoiFenceCandidate *association = nullptr;
    for (const ConSanMoiFenceCandidate &candidate : graph.moi_fence_candidates) {
      const ConSanSyncEvent *candidate_event = graph.find_event(candidate.fence_event);
      const ConSanSyncSequence *candidate_sequence = graph.find_sequence(candidate.sequence);
      if (candidate_event == nullptr || candidate_event->semantic_id != decision.semantic_site ||
          candidate_sequence == nullptr || !candidate.eligible() ||
          candidate_sequence->identity != decision.association->value)
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
    const ConSanFenceSite *fence_source =
        inventory.program_site<ConSanFenceSite>(fence_event->source_site);
    if (fence_source == nullptr) {
      errors.emplace_back("ConSan MOI admitted fence record lost its decoded source site");
      return {};
    }

    MoiFenceEvidenceSitePlan plan;
    plan.event = graph.event_id(*fence_event);
    plan.sequence = graph.sequence_id(*sequence);
    plan.source_site = communication->source_site;
    plan.evidence_intent = evidence;
    plan.address_capture_intent = address_capture;
    plan.memory_role = association->memory_role;
    plan.patch_text_offset = fence_event->text_offset();
    plan.patch_file_offset = fence_source->file_offset;
    plan.patch_size = fence_source->size;
    const ConSanProgramContainer *container =
        resolve_moi_evidence_container(inventory, fence_event->source_site);
    if (!container || !decision.communication_lowering_form) {
      errors.emplace_back("ConSan MOI admitted fence record lost its decoded lowering site");
      return {};
    }
    const std::optional<ConSanAtomicSite> communication_site =
        materialize_moi_communication_site(inventory, communication->source_site,
                                           plan.sequence);
    if (!communication_site) {
      errors.emplace_back("ConSan MOI admitted fence record lost its decoded lowering site");
      return {};
    }
    if (communication->kind == ConSanSyncKind::OrdinaryMemory) {
      const ConSanOrdinaryMemorySite *site =
          inventory.program_site<ConSanOrdinaryMemorySite>(communication->source_site);
      if (site == nullptr ||
          (site->support_reason != ConSanOrdinaryMemorySupportReason::Supported &&
           site->support_reason !=
               ConSanOrdinaryMemorySupportReason::SupportedSynchronizationOnly)) {
        errors.emplace_back("ConSan MOI admitted fence record lost its decoded lowering site");
        return {};
      }
      if (association->memory_role == ConSanSyncMemoryRole::Acquire) {
        if (sequence->kind != ConSanSyncKind::OrdinaryMemory ||
            sequence->memory_role != ConSanSyncMemoryRole::Acquire ||
            sequence->begin_text_offset != site->text_offset ||
            sequence->end_text_offset <= sequence->begin_text_offset ||
            sequence->end_text_offset < fence_event->text_offset() + fence_source->size ||
            sequence->end_text_offset - sequence->begin_text_offset >
                std::numeric_limits<uint32_t>::max()) {
          errors.emplace_back("ConSan MOI admitted fence record lost its decoded lowering site");
          return {};
        }
        plan.patch_text_offset = sequence->begin_text_offset;
        plan.patch_file_offset = site->file_offset;
        plan.patch_size =
            static_cast<uint32_t>(sequence->end_text_offset - sequence->begin_text_offset);
        plan.capture_address_before_guest = true;
        plan.scalar_clause_text_offset = sequence->scalar_clause_text_offset;
      }
    }
    plan.communication_lowering_form = *decision.communication_lowering_form;
    if (!plan.is_well_formed()) {
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
/// and operand-rich guest decode. It intentionally starts from admitted
/// decisions rather than rediscovering candidates and filtering them after
/// the fact. Record/Replay ordinary sequences owned by a fence have no
/// `AtomicRecord` intent and therefore do not enter the atomic-record path;
/// Sampled and InlineShadow select their own explicit evidence intent kinds.
[[nodiscard]] std::vector<MoiAtomicEvidenceSitePlan> build_moi_atomic_evidence_site_plans(
    const ProgramInventory &inventory, const ConSanObservationPlan &observation,
    ConSanProbeIntentKind evidence_kind, std::vector<std::string> &errors) {
  std::vector<MoiAtomicEvidenceSitePlan> plans;
  const SynchronizationInventoryView graph = inventory.sync();

  for (const ConSanAtomicSiteDecision &decision : observation.atomic_site_decisions) {
    if (decision.kind != ConSanSiteDecisionKind::Admitted)
      continue;

    ConSanProbeIntentId address_capture;
    ConSanProbeIntentId evidence;
    for (ConSanProbeIntentId id : decision.intent_ids) {
      const ConSanProbeIntent *intent = observation.intent(id);
      if (intent == nullptr)
        continue;
      if (intent->kind == ConSanProbeIntentKind::AtomicAddressCapture)
        address_capture = id;
      if (intent->kind == evidence_kind)
        evidence = id;
    }
    if (!evidence.valid())
      continue;
    if (!address_capture.valid() || !decision.association) {
      errors.emplace_back("ConSan MOI admitted atomic evidence lost its capture or association");
      return {};
    }

    const ConSanSyncEvent *event = graph.find_event(decision.semantic_site);
    const ConSanSyncSequence *sequence = graph.find_unique_sequence(decision.association->value);
    if (event == nullptr || sequence == nullptr ||
        graph.find_unique_sequence_containing(event->semantic_id) != sequence ||
        (event->kind != ConSanSyncKind::Atomic && event->kind != ConSanSyncKind::OrdinaryMemory)) {
      errors.emplace_back("ConSan MOI admitted atomic evidence lost its synchronization graph");
      return {};
    }
    MoiAtomicEvidenceSitePlan plan;
    plan.event = graph.event_id(*event);
    plan.sequence = graph.sequence_id(*sequence);
    plan.source_site = event->source_site;
    plan.address_capture_intent = address_capture;
    plan.evidence_intent = evidence;
    if (!decision.lowering_form) {
      errors.emplace_back("ConSan MOI admitted atomic evidence lost its decoded lowering site");
      return {};
    }
    plan.lowering_form = *decision.lowering_form;
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
