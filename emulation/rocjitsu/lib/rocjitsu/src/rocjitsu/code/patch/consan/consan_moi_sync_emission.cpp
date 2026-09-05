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
sampled_atomic_semantics_for_plan(const SynchronizationInventoryView &graph,
                                  const MoiAtomicEvidenceSitePlan &plan) {
  using Reason = SampledAtomicSemanticsReason;
  const auto reject = [](Reason reason) {
    return SampledAtomicSemanticsResult{.semantics = std::nullopt, .reason = reason};
  };
  const ConSanSyncSequence *sequence = graph.find_unique_sequence(plan.association.value);
  if (sequence == nullptr ||
      !consan_sync_confidence_meets(sequence->confidence, ConSanSemanticConfidence::Conservative) ||
      !consan_sync_confidence_meets(sequence->memory_role_confidence,
                                    ConSanSemanticConfidence::Conservative)) {
    return reject(Reason::UnqualifiedSharedSyncSequence);
  }

  consan_detail::SampledAtomicSemantics semantics;
  switch (sequence->memory_role) {
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
  const std::optional<ConSanMemoryScope> semantic_scope = sequence->scope;
  if (!semantic_scope) {
    return reject(Reason::MissingQualifiedScope);
  }
  const auto sampled_scope = consan_moi_sampled_sync_scope(*semantic_scope);
  if (!sampled_scope)
    return reject(Reason::UnsupportedQualifiedScope);
  semantics.scope = *sampled_scope;
  if (sequence->width_bits == 0 || sequence->width_bits % 8u != 0 ||
      sequence->width_bits / 8u > std::numeric_limits<uint32_t>::max()) {
    return reject(Reason::UnsupportedQualifiedByteRange);
  }
  semantics.byte_count = sequence->width_bits / 8u;
  switch (sequence->rmw_outcome) {
  case ConSanSyncRmwOutcome::NoReturn:
    semantics.outcome = ConSanMoiSampledSyncOutcome::RmwNoReturn;
    break;
  case ConSanSyncRmwOutcome::ReturnsOldValue:
    semantics.outcome = ConSanMoiSampledSyncOutcome::RmwReturnsOld;
    break;
  case ConSanSyncRmwOutcome::CompareExchange:
    if (!plan.site.returns_old_value.value_or(false) || !plan.site.data_vgpr ||
        !plan.site.dst_vgpr) {
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
  if (sequence->rmw_outcome == ConSanSyncRmwOutcome::CompareExchange) {
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

[[nodiscard]] ConSanAtomicSite
normalize_ordinary_fence_communication(const ConSanOrdinaryMemorySite &site) {
  ConSanAtomicSite normalized;
  if (site.flat_address_space_hint == ConSanFlatAddressSpaceHint::Group)
    normalized.address_space_hint = ConSanAtomicAddressSpaceHint::FlatGroup;
  normalized.text_offset = site.text_offset;
  normalized.file_offset = site.file_offset;
  normalized.size = site.size;
  normalized.width_bits = site.width_bits;
  normalized.dst_vgpr = site.destination_vgpr;
  normalized.addr_vgpr = site.address_vgpr;
  normalized.data_vgpr = site.operation == ConSanOrdinaryMemoryOperation::Load
                             ? site.destination_vgpr
                             : site.value_vgpr;
  normalized.saddr_sgpr = site.address_sgpr;
  normalized.raw_vaddr = site.raw_vaddr;
  normalized.raw_vdata = site.operation == ConSanOrdinaryMemoryOperation::Load
                             ? std::optional<uint32_t>(site.destination_vgpr)
                             : std::optional<uint32_t>(site.value_vgpr);
  normalized.raw_rsrc = site.raw_rsrc;
  normalized.raw_soffset = site.raw_soffset;
  normalized.raw_offen = site.raw_offen;
  normalized.raw_idxen = site.raw_idxen;
  normalized.raw_saddr = site.raw_saddr;
  normalized.raw_scale_offset = site.raw_scale_offset;
  normalized.raw_ioffset = site.raw_ioffset;
  normalized.raw_scope = site.raw_scope;
  normalized.scope = site.scope;
  normalized.raw_th = site.raw_th;
  normalized.returns_old_value = false;
  normalized.mnemonic = site.mnemonic;
  return normalized;
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
        fence_event->kind != ConSanSyncEventKind::Fence) {
      errors.emplace_back("ConSan MOI admitted fence record lost its synchronization event");
      return {};
    }

    const ConSanMoiFenceCandidate *association = nullptr;
    for (const ConSanMoiFenceCandidate &candidate : graph.moi_fence_candidates) {
      if (candidate.fence_event != decision.semantic_site || !candidate.eligible() ||
          candidate.sequence_identity != decision.association->value)
        continue;
      if (association != nullptr) {
        errors.emplace_back("ConSan MOI admitted fence record has ambiguous graph associations");
        return {};
      }
      association = &candidate;
    }
    const ConSanSyncSequence *sequence = graph.find_unique_sequence(decision.association->value);
    const ConSanSyncEvent *communication =
        association != nullptr && association->communication_event
            ? graph.find_event(*association->communication_event)
            : nullptr;
    if (association == nullptr || sequence == nullptr || communication == nullptr ||
        (communication->kind != ConSanSyncEventKind::Atomic &&
         communication->kind != ConSanSyncEventKind::OrdinaryMemory) ||
        communication->container_name != fence_event->container_name ||
        communication->in_kernel != fence_event->in_kernel ||
        communication->execution_owners.empty()) {
      errors.emplace_back("ConSan MOI admitted fence record lost its communication sequence");
      return {};
    }

    MoiFenceEvidenceSitePlan plan;
    plan.semantic_site = decision.semantic_site;
    plan.association = *decision.association;
    plan.evidence_intent = evidence;
    plan.address_capture_intent = address_capture;
    plan.memory_role = association->memory_role;
    plan.patch_text_offset = fence_event->text_offset;
    plan.patch_file_offset = fence_event->file_offset;
    plan.patch_size = fence_event->size;
    auto container = resolve_moi_evidence_container(inventory, fence_event->in_kernel,
                                                    fence_event->container_name,
                                                    communication->execution_owners);
    if (!container || !decision.communication_lowering_form) {
      errors.emplace_back("ConSan MOI admitted fence record lost its decoded lowering site");
      return {};
    }
    if (communication->kind == ConSanSyncEventKind::Atomic) {
      const ConSanAtomicSite *site =
          inventory.decoded_site<ConSanAtomicSite>(communication->source_site);
      if (site == nullptr) {
        errors.emplace_back("ConSan MOI admitted fence record lost its decoded lowering site");
        return {};
      }
      plan.communication_site = *site;
    } else {
      const ConSanOrdinaryMemorySite *site =
          inventory.decoded_site<ConSanOrdinaryMemorySite>(communication->source_site);
      if (site == nullptr ||
          (site->support_reason != ConSanOrdinaryMemorySupportReason::Supported &&
           site->support_reason !=
               ConSanOrdinaryMemorySupportReason::SupportedSynchronizationOnly)) {
        errors.emplace_back("ConSan MOI admitted fence record lost its decoded lowering site");
        return {};
      }
      plan.communication_site = normalize_ordinary_fence_communication(*site);
      if (association->memory_role == ConSanSyncMemoryRole::Acquire) {
        if (sequence->kind != ConSanSyncSequenceKind::OrdinaryMemory ||
            sequence->memory_role != ConSanSyncMemoryRole::Acquire ||
            sequence->begin_text_offset != site->text_offset ||
            sequence->end_text_offset <= sequence->begin_text_offset ||
            sequence->end_text_offset < fence_event->text_offset + fence_event->size ||
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
    if (sequence->scope)
      plan.communication_site.scope = sequence->scope;
    plan.communication_lowering_form = *decision.communication_lowering_form;
    plan.kernel_descriptor_file_offset = container->kernel_descriptor_file_offset;
    plan.container_name = std::move(container->qualified_name);
    plan.container_entry_text_offset = container->entry_text_offset;
    plan.text_file_offset = container->text_file_offset;
    if (!plan.is_well_formed()) {
      errors.emplace_back("ConSan MOI admitted fence record lost its decoded lowering site");
      return {};
    }
    plans.push_back(std::move(plan));
  }

  std::ranges::sort(plans, {}, [](const MoiFenceEvidenceSitePlan &plan) {
    return plan.semantic_site.physical.original_text_offset;
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

[[nodiscard]] std::optional<ConSanMoiAtomicEventKind>
moi_atomic_event_kind(ConSanSyncMemoryRole role) {
  switch (role) {
  case ConSanSyncMemoryRole::Release:
    return ConSanMoiAtomicEventKind::Release;
  case ConSanSyncMemoryRole::Acquire:
    return ConSanMoiAtomicEventKind::Acquire;
  case ConSanSyncMemoryRole::AcquireRelease:
    return ConSanMoiAtomicEventKind::AcquireRelease;
  case ConSanSyncMemoryRole::Unknown:
  case ConSanSyncMemoryRole::None:
  case ConSanSyncMemoryRole::SequentiallyConsistent:
    return std::nullopt;
  }
  return std::nullopt;
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
        (event->kind != ConSanSyncEventKind::Atomic &&
         event->kind != ConSanSyncEventKind::OrdinaryMemory)) {
      errors.emplace_back("ConSan MOI admitted atomic evidence lost its synchronization graph");
      return {};
    }
    const auto event_kind = moi_atomic_event_kind(sequence->memory_role);
    if (!event_kind) {
      errors.emplace_back("ConSan MOI admitted atomic evidence has no supported ordering role");
      return {};
    }

    MoiAtomicEvidenceSitePlan plan;
    plan.semantic_site = decision.semantic_site;
    plan.association = *decision.association;
    plan.address_capture_intent = address_capture;
    plan.evidence_intent = evidence;
    plan.event_kind = *event_kind;
    plan.is_rmw = event->kind == ConSanSyncEventKind::Atomic;
    plan.ordered_sequence_end_text_offset = sequence->end_text_offset;
    plan.scalar_clause_text_offset = sequence->scalar_clause_text_offset;
    auto container = resolve_moi_evidence_container(inventory, event->in_kernel,
                                                    event->container_name, event->execution_owners);
    if (!container || !decision.lowering_form) {
      errors.emplace_back("ConSan MOI admitted atomic evidence lost its decoded lowering site");
      return {};
    }
    if (event->kind == ConSanSyncEventKind::Atomic) {
      const ConSanAtomicSite *site = inventory.decoded_site<ConSanAtomicSite>(event->source_site);
      if (site == nullptr) {
        errors.emplace_back("ConSan MOI admitted atomic evidence lost its decoded lowering site");
        return {};
      }
      plan.site = *site;
    } else {
      const ConSanOrdinaryMemorySite *site =
          inventory.decoded_site<ConSanOrdinaryMemorySite>(event->source_site);
      if (site == nullptr) {
        errors.emplace_back("ConSan MOI admitted atomic evidence lost its decoded lowering site");
        return {};
      }
      plan.site = normalize_ordinary_fence_communication(*site);
    }
    if (sequence->scope)
      plan.site.scope = sequence->scope;
    plan.lowering_form = *decision.lowering_form;
    plan.kernel_descriptor_file_offset = container->kernel_descriptor_file_offset;
    plan.container_name = std::move(container->qualified_name);
    plan.uses_cluster_workgroup_id = container->uses_cluster_workgroup_id;
    if (!plan.is_well_formed()) {
      errors.emplace_back("ConSan MOI admitted atomic evidence lost its decoded lowering site");
      return {};
    }
    plans.push_back(std::move(plan));
  }

  std::ranges::sort(plans, {},
                    [](const MoiAtomicEvidenceSitePlan &plan) { return plan.site.text_offset; });
  return plans;
}

[[nodiscard]] bool neutralize_atomic_scalar_clause(std::vector<uint8_t> &text,
                                                   const MoiAtomicEvidenceSitePlan &candidate,
                                                   rj_code_arch_t arch,
                                                   std::vector<ConSanPatchInfo> &patches,
                                                   std::vector<std::string> &errors) {
  if (!candidate.scalar_clause_text_offset)
    return true;
  const uint64_t offset = *candidate.scalar_clause_text_offset;
  if (offset > text.size() || sizeof(uint32_t) > text.size() - offset ||
      offset >= candidate.site.text_offset) {
    errors.emplace_back("ConSan MOI atomic scalar-clause offset is outside its text prefix");
    return false;
  }
  uint32_t word = 0;
  std::memcpy(&word, text.data() + offset, sizeof(word));
  const uint32_t nop = build_s_nop(0, arch);
  if (word == nop)
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
  std::memcpy(text.data() + offset, &nop, sizeof(nop));
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
