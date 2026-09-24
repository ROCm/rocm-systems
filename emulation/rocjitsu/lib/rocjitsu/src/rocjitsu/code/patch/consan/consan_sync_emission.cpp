// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_sync_emission.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_access_target.h"
#include "rocjitsu/code/patch/consan/consan_device_primitives.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_identity_contracts.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_memory_emission.h"
#include "rocjitsu/code/patch/consan/consan_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_placement.h"
#include "rocjitsu/code/patch/consan/consan_probe_contracts.h"
#include "rocjitsu/code/patch/consan/consan_prologue.h"
#include "rocjitsu/code/patch/consan/consan_register_allocation.h"
#include "rocjitsu/code/patch/consan/consan_relocation.h"
#include "rocjitsu/code/patch/consan/consan_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/consan/consan_runtime_workgroup_gate.h"
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

namespace rocjitsu::consan {

using detail::append_atomic_fetch_add_one_u32;
using detail::append_atomic_load_u32;
using detail::append_compare_report_dispatch_id_word;
using detail::append_load_u32_vgpr_at_offset;
using detail::append_store_u32_vgpr_at_offset;
using detail::AtomicEvidenceSitePlan;
using detail::BarrierEvidenceSitePlan;
using detail::has_runtime_hardware_dispatch_id;
using detail::range_overlaps;
using detail::reject_optional_scratch_range_overlap;
using detail::SpecialStateSgprs;

namespace detail {

/// Project admitted atomic evidence directly into the common ConSan lowering
/// contract.
///
/// This is the sole join between synchronization inventory, mode policy,
/// and operand-rich guest decode. It starts directly from admitted evidence
/// intents and their paired address-capture intent.
[[nodiscard]] std::vector<AtomicEvidenceSitePlan>
build_atomic_evidence_site_plans(const ProgramInventory &inventory,
                                 const ObservationPlan &observation, ProbeIntentKind evidence_kind,
                                 std::vector<std::string> &errors) {
  std::vector<AtomicEvidenceSitePlan> plans;
  const SynchronizationInventoryView graph = inventory.sync();

  for (const ProbeIntent &intent : observation.probe_intents) {
    if (intent.kind != evidence_kind)
      continue;
    if (evidence_kind == ProbeIntentKind::PublicationModification) {
      const ProbeIntent *capture = nullptr;
      for (const auto &candidate : observation.probe_intents) {
        if (candidate.kind != ProbeIntentKind::PublicationAddressCapture ||
            candidate.source_site != intent.source_site)
          continue;
        if (capture) {
          errors.emplace_back("ConSan publication modification has ambiguous address capture");
          return {};
        }
        capture = &candidate;
      }
      if (!capture || !capture->atomic_lowering_form) {
        errors.emplace_back("ConSan publication modification lost its address capture");
        return {};
      }
      AtomicEvidenceSitePlan plan;
      plan.publication_modification = true;
      plan.source_site = intent.source_site;
      plan.address_capture_intent = capture->id;
      plan.evidence_intent = intent.id;
      plan.lowering_form = *capture->atomic_lowering_form;
      if (!plan.is_well_formed() || !resolve_atomic_evidence_source(inventory, plan)) {
        errors.emplace_back("ConSan publication modification lost its decoded source");
        return {};
      }
      plans.push_back(std::move(plan));
      continue;
    }
    if (!intent.synchronization_association) {
      errors.emplace_back("ConSan admitted atomic evidence lost its capture or association");
      return {};
    }
    const SyncEvent *event = graph.find_event(intent.source_site);
    const SyncSequence *sequence =
        graph.find_unique_sequence(intent.synchronization_association->value);
    if (event == nullptr || sequence == nullptr ||
        graph.find_unique_sequence_containing(event->semantic_id) != sequence ||
        (event->kind != SyncKind::Atomic && event->kind != SyncKind::OrdinaryMemory)) {
      errors.emplace_back("ConSan admitted atomic evidence lost its synchronization graph");
      return {};
    }
    const ProbeIntent *capture = nullptr;
    for (const ProbeIntent &candidate : observation.probe_intents) {
      if (candidate.kind != ProbeIntentKind::AtomicAddressCapture ||
          candidate.synchronization_association != intent.synchronization_association ||
          std::ranges::find(candidate.covered_semantic_sites, event->semantic_id) ==
              candidate.covered_semantic_sites.end())
        continue;
      if (capture != nullptr) {
        errors.emplace_back("ConSan admitted atomic evidence has ambiguous capture intents");
        return {};
      }
      capture = &candidate;
    }
    if (capture == nullptr || !capture->atomic_lowering_form) {
      errors.emplace_back("ConSan admitted atomic evidence lost its decoded lowering site");
      return {};
    }
    AtomicEvidenceSitePlan plan;
    plan.event = graph.event_id(*event);
    plan.sequence = graph.sequence_id(*sequence);
    plan.source_site = event->source_site;
    plan.address_capture_intent = capture->id;
    plan.evidence_intent = intent.id;
    plan.lowering_form = *capture->atomic_lowering_form;
    if (!resolve_atomic_evidence_source(inventory, plan) || !plan.is_well_formed()) {
      errors.emplace_back("ConSan admitted atomic evidence lost its decoded lowering site");
      return {};
    }
    plans.push_back(std::move(plan));
  }

  std::ranges::sort(plans, {}, [&](const AtomicEvidenceSitePlan &plan) {
    return inventory.program_site(plan.source_site)->text_offset();
  });
  return plans;
}

[[nodiscard]] bool append_atomic_scalar_clause_patch(std::span<const uint8_t> text,
                                                     const AtomicSite &site,
                                                     std::optional<uint64_t> scalar_clause_offset,
                                                     rj_code_arch_t arch,
                                                     std::vector<PatchInfo> &patches,
                                                     std::vector<std::string> &errors) {
  if (!scalar_clause_offset)
    return true;
  const uint64_t offset = *scalar_clause_offset;
  if (offset > text.size() || sizeof(uint32_t) > text.size() - offset ||
      offset >= site.text_offset) {
    errors.emplace_back("ConSan atomic scalar-clause offset is outside its text prefix");
    return false;
  }
  uint32_t word = 0;
  std::memcpy(&word, text.data() + offset, sizeof(word));
  const uint32_t nop = build_s_nop(0, arch);
  if (word == nop || std::ranges::any_of(patches, [&](const PatchInfo &patch) {
        return patch.kind == PatchKind::InlineScalarClauseNopRewrite &&
               patch.anchor_offset == offset;
      }))
    return true;

  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder) {
    errors.emplace_back("ConSan atomic scalar-clause validation could not create a decoder");
    return false;
  }
  std::array<uint32_t, 4> words{};
  const size_t available = std::min(sizeof(words), text.size() - offset);
  std::memcpy(words.data(), text.data() + offset, available);
  std::unique_ptr<Instruction> instruction = decode_bounded_instruction(
      *decoder, std::span<const uint32_t>(words).first(available / sizeof(uint32_t)), offset);
  if (!instruction || instruction->mnemonic() != "s_clause") {
    errors.emplace_back("ConSan atomic scalar-clause prefix is no longer an s_clause");
    return false;
  }
  PatchInfo info;
  info.kind = PatchKind::InlineScalarClauseNopRewrite;
  info.anchor_offset = offset;
  info.trampoline_offset = offset;
  info.original_size = sizeof(nop);
  patches.push_back(std::move(info));
  return true;
}

} // namespace detail
} // namespace rocjitsu::consan
