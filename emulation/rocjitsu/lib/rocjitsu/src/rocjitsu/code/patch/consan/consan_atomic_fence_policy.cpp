// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan.h"

#include <algorithm>
#include <map>
#include <ranges>
#include <tuple>
#include <unordered_map>

namespace rocjitsu {
namespace {

/// Unique logical sequence claiming one synchronization-event identity.
///
/// The pointer borrows immutable inventory for the duration of a pure policy
/// call. `ambiguous` is distinct from a missing pointer so policy can preserve
/// whether no sequence or multiple incompatible sequences caused rejection.
struct SequenceMembership {
  const ConSanSyncSequence *sequence = nullptr;
  bool ambiguous = false;
};

/// Read-only join from stable event identity to its sequence-membership fact.
using SequenceMembershipIndex = std::unordered_map<std::string_view, SequenceMembership>;

[[nodiscard]] bool valid_engine(ConSanCapabilityEngine engine) {
  return static_cast<uint8_t>(engine) < static_cast<uint8_t>(ConSanCapabilityEngine::Count);
}

[[nodiscard]] SequenceMembershipIndex
build_sequence_membership_index(std::span<const ConSanSyncSequence> sequences) {
  SequenceMembershipIndex result;
  size_t member_count = 0;
  for (const ConSanSyncSequence &sequence : sequences)
    member_count += sequence.member_event_identities.size();
  result.reserve(member_count);
  for (const ConSanSyncSequence &sequence : sequences) {
    for (const std::string &identity : sequence.member_event_identities) {
      const auto [entry, inserted] =
          result.try_emplace(identity, SequenceMembership{&sequence, false});
      if (!inserted) {
        entry->second.sequence = nullptr;
        entry->second.ambiguous = true;
      }
    }
  }
  return result;
}

[[nodiscard]] std::unordered_map<std::string_view, const ConSanSyncEvent *>
build_event_index(std::span<const ConSanSyncEvent> events) {
  std::unordered_map<std::string_view, const ConSanSyncEvent *> result;
  result.reserve(events.size());
  for (const ConSanSyncEvent &event : events) {
    const auto [entry, inserted] = result.emplace(event.identity, &event);
    if (!inserted)
      entry->second = nullptr;
  }
  return result;
}

[[nodiscard]] bool owner_semantics_equal(std::span<const ConSanExecutionOwner> lhs,
                                         std::span<const ConSanExecutionOwner> rhs) {
  return std::ranges::equal(lhs, rhs, [](const auto &left, const auto &right) {
    return left.descriptor_file_offset == right.descriptor_file_offset && left.proof == right.proof;
  });
}

[[nodiscard]] bool event_policy_semantics_equal(const ConSanSyncEvent &lhs,
                                                const ConSanSyncEvent &rhs) {
  return std::tie(lhs.kind, lhs.operation, lhs.address_source, lhs.memory_role, lhs.rmw_outcome,
                  lhs.confidence, lhs.memory_role_confidence, lhs.text_offset, lhs.file_offset,
                  lhs.size, lhs.width_bits, lhs.mnemonic, lhs.static_byte_offset, lhs.raw_scope) ==
             std::tie(rhs.kind, rhs.operation, rhs.address_source, rhs.memory_role, rhs.rmw_outcome,
                      rhs.confidence, rhs.memory_role_confidence, rhs.text_offset, rhs.file_offset,
                      rhs.size, rhs.width_bits, rhs.mnemonic, rhs.static_byte_offset,
                      rhs.raw_scope) &&
         owner_semantics_equal(lhs.execution_owners, rhs.execution_owners);
}

[[nodiscard]] bool fence_policy_semantics_equal(const ConSanMoiFenceCandidate &lhs,
                                                const ConSanMoiFenceCandidate &rhs) {
  return std::tie(lhs.in_kernel, lhs.text_offset, lhs.file_offset, lhs.size, lhs.memory_role,
                  lhs.communication_address_source, lhs.communication_static_byte_offset,
                  lhs.raw_scope, lhs.association) ==
         std::tie(rhs.in_kernel, rhs.text_offset, rhs.file_offset, rhs.size, rhs.memory_role,
                  rhs.communication_address_source, rhs.communication_static_byte_offset,
                  rhs.raw_scope, rhs.association);
}

[[nodiscard]] std::vector<std::string>
fence_source_container_names(std::span<const ConSanMoiFenceCandidate *const> aliases) {
  std::vector<std::string> result;
  result.reserve(aliases.size());
  for (const ConSanMoiFenceCandidate *fence : aliases) {
    if (std::ranges::find(result, fence->container_name) == result.end())
      result.push_back(fence->container_name);
  }
  std::ranges::sort(result);
  return result;
}

[[nodiscard]] bool fence_filter_matches(std::span<const ConSanMoiFenceCandidate *const> aliases,
                                        std::string_view filter) {
  return filter.empty() || std::ranges::any_of(aliases, [&](const ConSanMoiFenceCandidate *fence) {
           return fence->container_name.find(filter) != std::string::npos;
         });
}

[[nodiscard]] std::vector<std::string>
source_container_names(std::span<const ConSanSyncEvent *const> aliases) {
  std::vector<std::string> result;
  result.reserve(aliases.size());
  for (const ConSanSyncEvent *event : aliases) {
    const std::span<const std::string> sources = event->source_containers.empty()
                                                     ? std::span(&event->container_name, 1u)
                                                     : std::span(event->source_containers);
    for (const std::string &source : sources) {
      if (std::ranges::find(result, source) == result.end())
        result.push_back(source);
    }
  }
  std::ranges::sort(result);
  return result;
}

[[nodiscard]] bool filter_matches(std::span<const ConSanSyncEvent *const> aliases,
                                  std::string_view filter) {
  return filter.empty() || std::ranges::any_of(aliases, [&](const ConSanSyncEvent *event) {
           if (event->container_name.find(filter) != std::string::npos)
             return true;
           return std::ranges::any_of(event->source_containers, [&](const std::string &source) {
             return source.find(filter) != std::string::npos;
           });
         });
}

[[nodiscard]] SemanticSiteId event_semantic_id(const ProgramInventory &inventory,
                                               const ConSanSyncEvent &event) {
  if (event.semantic_id.valid())
    return event.semantic_id;
  return {
      .physical = {.code_object = inventory.code_object_id(),
                   .original_text_offset = event.text_offset},
      .domain = ConSanSemanticSiteDomain::SynchronizationEvent,
  };
}

template <typename Container>
[[nodiscard]] const ConSanAtomicSite *atomic_site_in_container(const Container &container,
                                                               uint64_t text_offset) {
  const auto site =
      std::ranges::find(container.atomic_sites, text_offset, &ConSanAtomicSite::text_offset);
  if (site == container.atomic_sites.end() ||
      std::ranges::count(container.atomic_sites, text_offset, &ConSanAtomicSite::text_offset) !=
          1) {
    return nullptr;
  }
  return &*site;
}

[[nodiscard]] const ConSanAtomicSite *find_atomic_site(const ProgramInventory &inventory,
                                                       const ConSanSyncEvent &event) {
  if (event.in_kernel) {
    const auto kernels = inventory.kernels();
    const auto container =
        std::ranges::find(kernels, event.container_name, &ConSanKernelInfo::name);
    if (container != kernels.end())
      return atomic_site_in_container(*container, event.text_offset);
  } else {
    const auto functions = inventory.functions();
    const auto container =
        std::ranges::find(functions, event.container_name, &ConSanFunctionInfo::name);
    if (container != functions.end())
      return atomic_site_in_container(*container, event.text_offset);
  }
  return nullptr;
}

template <typename Container>
[[nodiscard]] const ConSanOrdinaryMemorySite *ordinary_site_in_container(const Container &container,
                                                                         uint64_t text_offset) {
  const auto site = std::ranges::find(container.ordinary_memory_sites, text_offset,
                                      &ConSanOrdinaryMemorySite::text_offset);
  if (site == container.ordinary_memory_sites.end() ||
      std::ranges::count(container.ordinary_memory_sites, text_offset,
                         &ConSanOrdinaryMemorySite::text_offset) != 1) {
    return nullptr;
  }
  return &*site;
}

[[nodiscard]] const ConSanOrdinaryMemorySite *find_ordinary_site(const ProgramInventory &inventory,
                                                                 const ConSanSyncEvent &event) {
  if (event.in_kernel) {
    const auto kernels = inventory.kernels();
    const auto container =
        std::ranges::find(kernels, event.container_name, &ConSanKernelInfo::name);
    if (container != kernels.end())
      return ordinary_site_in_container(*container, event.text_offset);
  } else {
    const auto functions = inventory.functions();
    const auto container =
        std::ranges::find(functions, event.container_name, &ConSanFunctionInfo::name);
    if (container != functions.end())
      return ordinary_site_in_container(*container, event.text_offset);
  }
  return nullptr;
}

[[nodiscard]] ConSanAtomicSite normalize_ordinary_site(const ConSanOrdinaryMemorySite &site) {
  ConSanAtomicSite result;
  if (site.flat_address_space_hint == ConSanFlatAddressSpaceHint::Group)
    result.address_space_hint = ConSanAtomicAddressSpaceHint::FlatGroup;
  result.text_offset = site.text_offset;
  result.file_offset = site.file_offset;
  result.size = site.size;
  result.width_bits = site.width_bits;
  result.dst_vgpr = site.destination_vgpr;
  result.addr_vgpr = site.address_vgpr;
  result.data_vgpr = site.operation == ConSanOrdinaryMemoryOperation::Load ? site.destination_vgpr
                                                                           : site.value_vgpr;
  result.saddr_sgpr = site.address_sgpr;
  result.raw_saddr = site.raw_saddr;
  result.raw_scale_offset = site.raw_scale_offset;
  result.raw_vaddr = site.raw_vaddr;
  result.raw_vsrc = site.raw_vsrc;
  result.raw_vdst = site.raw_vdst;
  result.raw_vdata = site.operation == ConSanOrdinaryMemoryOperation::Load
                         ? std::optional<uint32_t>(site.destination_vgpr)
                         : std::optional<uint32_t>(site.value_vgpr);
  result.raw_rsrc = site.raw_rsrc;
  result.raw_soffset = site.raw_soffset;
  result.raw_offen = site.raw_offen;
  result.raw_idxen = site.raw_idxen;
  result.raw_ioffset = site.raw_ioffset;
  result.raw_scope = site.raw_scope;
  result.raw_th = site.raw_th;
  result.returns_old_value = false;
  result.mnemonic = site.mnemonic;
  return result;
}

[[nodiscard]] ConSanAtomicSite normalize_native_site(const ConSanAtomicSite &site,
                                                     const ProgramInventory &inventory) {
  ConSanAtomicSite result = site;
  if (result.address_space_hint == ConSanAtomicAddressSpaceHint::FlatGroup ||
      (result.mnemonic.starts_with("ds_") &&
       consan_arch_supports_capability_form(inventory.arch(),
                                            ConSanCapabilityForm::OrderedLdsAtomic))) {
    result.raw_scope = 1u;
  }
  return result;
}

[[nodiscard]] std::optional<ConSanCapabilityForm>
atomic_capability_form(const ConSanSyncEvent &event) {
  if (event.kind == ConSanSyncEventKind::OrdinaryMemory)
    return ConSanCapabilityForm::AddressedOrdinaryFence;
  if (event.kind != ConSanSyncEventKind::Atomic)
    return std::nullopt;
  if (event.address_source == ConSanSyncAddressSource::LdsVector ||
      event.mnemonic.starts_with("ds_")) {
    return ConSanCapabilityForm::OrderedLdsAtomic;
  }
  if (event.address_source == ConSanSyncAddressSource::FlatVector)
    return ConSanCapabilityForm::OrderedFlatAtomic;
  if (event.address_source == ConSanSyncAddressSource::GlobalScalarVector)
    return ConSanCapabilityForm::OrderedVglobalAtomic;
  return std::nullopt;
}

[[nodiscard]] ConSanDynamicResultRequirement dynamic_requirement(const ConSanSyncEvent &event) {
  switch (event.rmw_outcome) {
  case ConSanSyncRmwOutcome::NotApplicable:
  case ConSanSyncRmwOutcome::NoReturn:
    return ConSanDynamicResultRequirement::None;
  case ConSanSyncRmwOutcome::ReturnsOldValue:
    return ConSanDynamicResultRequirement::ReturnedOldValue;
  case ConSanSyncRmwOutcome::CompareExchange:
    return ConSanDynamicResultRequirement::CompareExchangeSuccess;
  case ConSanSyncRmwOutcome::Unknown:
    return ConSanDynamicResultRequirement::Count;
  }
  return ConSanDynamicResultRequirement::Count;
}

[[nodiscard]] ConSanAtomicPolicyReason
classify_exact_atomic_encoding(const ConSanAtomicSite &site, rj_code_arch_t arch, bool is_rmw) {
  const bool is_flat = site.mnemonic.starts_with("flat_atomic") ||
                       (!is_rmw && (site.mnemonic.starts_with("flat_load") ||
                                    site.mnemonic.starts_with("flat_store")));
  const bool is_vglobal = site.mnemonic.starts_with("global_atomic") ||
                          (!is_rmw && (site.mnemonic.starts_with("global_load") ||
                                       site.mnemonic.starts_with("global_store")));
  if (!is_flat && !is_vglobal)
    return ConSanAtomicPolicyReason::UnsupportedAddressSource;
  if (site.width_bits != 32u)
    return ConSanAtomicPolicyReason::InvalidAccessWidth;

  const bool is_cdna = consan_uses_gfx9_cdna_encoding(arch);
  const bool is_rdna3 = consan_uses_gfx11_encoding(arch);
  const bool is_gfx12 = consan_uses_gfx12_encoding(arch);
  const bool gfx12_flat_saddr =
      site.raw_saddr &&
      (*site.raw_saddr == 0x7cu || (site.saddr_sgpr && *site.raw_saddr == *site.saddr_sgpr &&
                                    *site.saddr_sgpr <= 104u && (*site.saddr_sgpr & 1u) == 0u));
  const bool gfx12_flat_encoding =
      is_flat && is_gfx12 && site.size == 3u * sizeof(uint32_t) && gfx12_flat_saddr;
  const bool cdna_flat_encoding = is_flat && is_cdna && site.size == 2u * sizeof(uint32_t) &&
                                  site.raw_saddr && *site.raw_saddr == 0u;
  const bool rdna3_flat_encoding = is_flat && is_rdna3 && site.size == 2u * sizeof(uint32_t) &&
                                   site.raw_saddr && *site.raw_saddr == 0x7cu;
  const bool vglobal_saddr =
      site.raw_saddr && (*site.raw_saddr == (is_cdna ? 0x7fu : 0x7cu) ||
                         (site.saddr_sgpr && *site.raw_saddr == *site.saddr_sgpr &&
                          *site.saddr_sgpr <= 104u && (*site.saddr_sgpr & 1u) == 0u));
  const bool pre_gfx12_vglobal_encoding =
      is_vglobal && (is_cdna || is_rdna3) && site.size == 2u * sizeof(uint32_t) && vglobal_saddr;
  const bool gfx12_vglobal_encoding = is_vglobal && is_gfx12 && site.size == 3u * sizeof(uint32_t);
  if (!gfx12_flat_encoding && !cdna_flat_encoding && !rdna3_flat_encoding &&
      !pre_gfx12_vglobal_encoding && !gfx12_vglobal_encoding) {
    return ConSanAtomicPolicyReason::UnsupportedEncoding;
  }
  if (!site.raw_ioffset || (is_flat && *site.raw_ioffset != 0))
    return ConSanAtomicPolicyReason::UnsupportedEncoding;
  const bool compare_exchange = is_rmw && consan_atomic_is_compare_exchange(site);
  if (!site.addr_vgpr || *site.addr_vgpr >= 255u || !site.data_vgpr ||
      (compare_exchange && *site.data_vgpr >= 255u)) {
    return ConSanAtomicPolicyReason::MissingOperands;
  }
  if (compare_exchange && site.returns_old_value && !*site.returns_old_value) {
    return ConSanAtomicPolicyReason::CompareExchangeOutcomeUnavailable;
  }
  if (compare_exchange && !site.dst_vgpr)
    return ConSanAtomicPolicyReason::MissingOperands;
  if (!site.raw_scope || !site.raw_th || !site.returns_old_value)
    return ConSanAtomicPolicyReason::MissingScope;
  if (*site.raw_scope < 1u || *site.raw_scope > 3u)
    return ConSanAtomicPolicyReason::UnsupportedScope;
  return ConSanAtomicPolicyReason::None;
}

[[nodiscard]] ConSanAtomicPolicyReason
classify_cdna5_buffer_ordinary_encoding(const ConSanAtomicSite &site, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA5 || !site.mnemonic.starts_with("buffer_"))
    return ConSanAtomicPolicyReason::UnsupportedAddressSource;
  if (site.width_bits == 0u || site.width_bits > 128u)
    return ConSanAtomicPolicyReason::InvalidAccessWidth;
  if (site.size != 3u * sizeof(uint32_t) || !site.raw_rsrc || !site.raw_soffset ||
      !site.raw_vaddr || !site.raw_ioffset || !site.raw_scope || !site.raw_offen ||
      !site.raw_idxen || !*site.raw_offen || *site.raw_idxen) {
    return ConSanAtomicPolicyReason::UnsupportedEncoding;
  }
  if (!site.addr_vgpr || !site.saddr_sgpr || !site.data_vgpr ||
      *site.raw_vaddr != *site.addr_vgpr || *site.raw_rsrc != *site.saddr_sgpr ||
      (*site.saddr_sgpr & 3u) != 0u || *site.saddr_sgpr > 124u ||
      (*site.raw_soffset != 0x7cu && *site.raw_soffset > 127u) || *site.addr_vgpr > 255u) {
    return ConSanAtomicPolicyReason::MissingOperands;
  }
  constexpr int32_t kSigned24Min = -(1 << 23);
  constexpr int32_t kSigned24Max = (1 << 23) - 1;
  if (*site.raw_ioffset < kSigned24Min || *site.raw_ioffset > kSigned24Max)
    return ConSanAtomicPolicyReason::UnsupportedEncoding;
  if (*site.raw_scope < 1u || *site.raw_scope > 3u)
    return ConSanAtomicPolicyReason::UnsupportedScope;
  return ConSanAtomicPolicyReason::None;
}

[[nodiscard]] ConSanAtomicPolicyReason classify_atomic_encoding(const ProgramInventory &inventory,
                                                                const ConSanSyncEvent &event,
                                                                const ConSanSyncSequence &sequence,
                                                                ConSanCapabilityEngine engine) {
  const bool ordinary = event.kind == ConSanSyncEventKind::OrdinaryMemory;
  const ConSanAtomicSite *native = ordinary ? nullptr : find_atomic_site(inventory, event);
  const ConSanOrdinaryMemorySite *ordinary_site =
      ordinary ? find_ordinary_site(inventory, event) : nullptr;
  if ((ordinary && ordinary_site == nullptr) || (!ordinary && native == nullptr))
    return ConSanAtomicPolicyReason::MissingOperands;
  ConSanAtomicSite site = ordinary ? normalize_ordinary_site(*ordinary_site)
                                   : normalize_native_site(*native, inventory);

  // Synchronization analysis may prove a stronger semantic scope than the
  // raw instruction bits. In particular, CDNA5 group-aperture FLAT loads use
  // raw scope zero while an exact adjacent zero-wait proves workgroup acquire.
  // Lowering already consumes the sequence scope; policy must classify the
  // same normalized contract.
  if (ordinary)
    site.raw_scope = sequence.raw_scope;

  // CDNA5 buffer ordinary communication is a Record/Replay-only address form.
  // Its associated fence owns the after-operation evidence, while the common
  // address planner materializes the resource descriptor and vector offset.
  if (ordinary && engine == ConSanCapabilityEngine::RecordReplay &&
      site.mnemonic.starts_with("buffer_")) {
    return classify_cdna5_buffer_ordinary_encoding(site, inventory.arch());
  }

  if (engine == ConSanCapabilityEngine::InlineShadow || ordinary)
    return classify_exact_atomic_encoding(site, inventory.arch(), !ordinary);

  const bool allowed_lds = site.mnemonic.starts_with("ds_") &&
                           consan_arch_supports_capability_form(
                               inventory.arch(), ConSanCapabilityForm::OrderedLdsAtomic);
  if (!site.mnemonic.starts_with("flat_atomic") && !site.mnemonic.starts_with("global_atomic") &&
      !allowed_lds) {
    return ConSanAtomicPolicyReason::UnsupportedAddressSource;
  }
  if (!site.addr_vgpr || !site.data_vgpr || *site.addr_vgpr >= 255u)
    return ConSanAtomicPolicyReason::MissingOperands;
  if (!site.raw_scope)
    return ConSanAtomicPolicyReason::MissingScope;
  if (*site.raw_scope < 1u || *site.raw_scope > 3u)
    return ConSanAtomicPolicyReason::UnsupportedScope;
  if (consan_atomic_is_compare_exchange(site) &&
      (!site.returns_old_value.value_or(false) || !site.dst_vgpr)) {
    return ConSanAtomicPolicyReason::CompareExchangeOutcomeUnavailable;
  }
  return ConSanAtomicPolicyReason::None;
}

[[nodiscard]] bool sequence_is_atomic_contract(const ConSanSyncEvent &event,
                                               const ConSanSyncSequence &sequence) {
  if (event.kind == ConSanSyncEventKind::Atomic)
    return sequence.kind == ConSanSyncSequenceKind::Atomic;
  return event.kind == ConSanSyncEventKind::OrdinaryMemory &&
         sequence.kind == ConSanSyncSequenceKind::OrdinaryMemory;
}

[[nodiscard]] ConSanAtomicPolicyReason
classify_atomic_semantics(const ProgramInventory &inventory, const ConSanSyncEvent &event,
                          const SequenceMembership &membership) {
  if (membership.ambiguous)
    return ConSanAtomicPolicyReason::AmbiguousSequenceMembership;
  const ConSanSyncSequence *sequence = membership.sequence;
  if (sequence == nullptr || !sequence_is_atomic_contract(event, *sequence) ||
      !consan_sync_confidence_meets(sequence->confidence, ConSanSemanticConfidence::Conservative) ||
      !consan_sync_confidence_meets(sequence->memory_role_confidence,
                                    ConSanSemanticConfidence::Conservative)) {
    return ConSanAtomicPolicyReason::UnqualifiedSyncSequence;
  }
  switch (sequence->memory_role) {
  case ConSanSyncMemoryRole::Release:
  case ConSanSyncMemoryRole::Acquire:
  case ConSanSyncMemoryRole::AcquireRelease:
    break;
  case ConSanSyncMemoryRole::Unknown:
  case ConSanSyncMemoryRole::None:
  case ConSanSyncMemoryRole::SequentiallyConsistent:
    return ConSanAtomicPolicyReason::UnsupportedMemoryRole;
  }
  const std::optional<uint32_t> semantic_scope =
      sequence->raw_scope ? sequence->raw_scope
                          : (inventory.arch() == ROCJITSU_CODE_ARCH_CDNA5 &&
                                     event.address_source == ConSanSyncAddressSource::LdsVector
                                 ? std::optional<uint32_t>(1u)
                                 : std::nullopt);
  if (!semantic_scope)
    return ConSanAtomicPolicyReason::MissingScope;
  if (*semantic_scope < 1u || *semantic_scope > 3u)
    return ConSanAtomicPolicyReason::UnsupportedScope;
  if (sequence->width_bits == 0u || sequence->width_bits % 8u != 0u)
    return ConSanAtomicPolicyReason::InvalidAccessWidth;
  if (event.kind == ConSanSyncEventKind::Atomic &&
      dynamic_requirement(event) == ConSanDynamicResultRequirement::Count) {
    return ConSanAtomicPolicyReason::UnsupportedDynamicOutcome;
  }
  return ConSanAtomicPolicyReason::None;
}

[[nodiscard]] ConSanProbeIntentKind atomic_evidence_kind(ConSanCapabilityEngine engine) {
  switch (engine) {
  case ConSanCapabilityEngine::RecordReplay:
    return ConSanProbeIntentKind::AtomicRecord;
  case ConSanCapabilityEngine::Sampled:
    return ConSanProbeIntentKind::SampledAtomicOrdering;
  case ConSanCapabilityEngine::InlineShadow:
    return ConSanProbeIntentKind::ExactAtomicOrdering;
  case ConSanCapabilityEngine::SuperCollider:
  case ConSanCapabilityEngine::Count:
    break;
  }
  return ConSanProbeIntentKind::Count;
}

[[nodiscard]] ConSanProbeIntentId
add_intent(ConSanObservationPlan &plan, const PhysicalSiteId &physical_site,
           std::vector<SemanticSiteId> covered_sites, ConSanProbeIntentKind kind,
           ConSanProbePosition position, const ConSanSynchronizationAssociationId &association,
           ConSanDynamicResultRequirement dynamic_result) {
  const ConSanProbeIntentId id{static_cast<uint32_t>(plan.probe_intents.size())};
  plan.probe_intents.push_back({
      .id = id,
      .engine = plan.engine,
      .physical_site = physical_site,
      .covered_semantic_sites = std::move(covered_sites),
      .kind = kind,
      .position = position,
      .lane_mask = ConSanLaneMaskPolicy::ActiveExecutionMask,
      .requirement = ConSanProbeRequirement::Required,
      .synchronization_association = association,
      .dynamic_result = dynamic_result,
  });
  return id;
}

void add_covered_site(ConSanObservationPlan &plan, ConSanProbeIntentId id,
                      const SemanticSiteId &site) {
  if (id.value >= plan.probe_intents.size())
    return;
  std::vector<SemanticSiteId> &covered = plan.probe_intents[id.value].covered_semantic_sites;
  if (std::ranges::find(covered, site) == covered.end())
    covered.push_back(site);
}

[[nodiscard]] bool has_qualified_fence_for(const SynchronizationInventoryView &inventory,
                                           std::string_view communication_identity) {
  return std::ranges::any_of(
      inventory.moi_fence_candidates, [&](const ConSanMoiFenceCandidate &fence) {
        return fence.eligible() && fence.communication_event_identity == communication_identity;
      });
}

} // namespace

ConSanAtomicFencePolicyResult
plan_consan_atomic_fence_observation(const ProgramInventory &inventory,
                                     const ConSanAtomicFencePolicyRequest &request) {
  ConSanAtomicFencePolicyResult result;
  result.plan.engine = request.engine;
  if (!valid_engine(request.engine) || inventory.empty())
    return result;

  const SynchronizationInventoryView synchronization = inventory.sync();
  const SequenceMembershipIndex memberships =
      build_sequence_membership_index(synchronization.sync_sequences);
  const auto events_by_identity = build_event_index(synchronization.sync_events);

  std::map<std::pair<ConSanSyncEventKind, uint64_t>, std::vector<const ConSanSyncEvent *>>
      aliases_by_site;
  for (const ConSanSyncEvent &event : synchronization.sync_events) {
    if (event.kind == ConSanSyncEventKind::Atomic ||
        event.kind == ConSanSyncEventKind::OrdinaryMemory) {
      aliases_by_site[{event.kind, event.text_offset}].push_back(&event);
    }
  }

  for (const auto &[site_key, aliases] : aliases_by_site) {
    (void)site_key;
    const ConSanSyncEvent &event = *aliases.front();
    const auto membership_entry = memberships.find(event.identity);
    const SequenceMembership membership =
        membership_entry == memberships.end() ? SequenceMembership{} : membership_entry->second;
    // Ordinary memory belongs to access policy unless synchronization
    // analysis associated an acquire/release sequence around it.
    if (event.kind == ConSanSyncEventKind::OrdinaryMemory &&
        (membership.sequence == nullptr ||
         membership.sequence->kind != ConSanSyncSequenceKind::OrdinaryMemory ||
         membership.sequence->memory_role == ConSanSyncMemoryRole::Unknown ||
         membership.sequence->memory_role == ConSanSyncMemoryRole::None)) {
      continue;
    }

    const SemanticSiteId semantic_id = event_semantic_id(inventory, event);
    const std::vector<std::string> names = source_container_names(aliases);
    const bool conflicting_alias = std::ranges::any_of(aliases, [&](const ConSanSyncEvent *alias) {
      return !event_policy_semantics_equal(event, *alias);
    });
    const std::optional<ConSanCapabilityForm> form = atomic_capability_form(event);
    const ConSanCapabilityDisposition capability =
        form ? consan_capability_disposition(inventory.target(), request.engine, *form)
             : ConSanCapabilityDisposition::OutOfContract;
    ConSanSiteDecisionKind kind = ConSanSiteDecisionKind::NotApplicable;
    ConSanAtomicPolicyReason reason = ConSanAtomicPolicyReason::TrackingDisabled;

    if (!request.tracking_enabled) {
      reason = ConSanAtomicPolicyReason::TrackingDisabled;
    } else if (request.engine == ConSanCapabilityEngine::SuperCollider) {
      reason = ConSanAtomicPolicyReason::EngineMutationOnly;
    } else if (!filter_matches(aliases, request.container_filter)) {
      reason = ConSanAtomicPolicyReason::ContainerFilterExcluded;
    } else if (event.execution_owners.empty()) {
      reason = ConSanAtomicPolicyReason::MissingExecutionOwner;
    } else if (request.engine == ConSanCapabilityEngine::Sampled &&
               !request.sampled_access_window_available) {
      reason = ConSanAtomicPolicyReason::MissingSampledAccessWindow;
    } else if (!form || (capability != ConSanCapabilityDisposition::Supported &&
                         capability != ConSanCapabilityDisposition::AssociatedOnly)) {
      reason = ConSanAtomicPolicyReason::TargetCapabilityUnavailable;
    } else if (conflicting_alias) {
      kind = ConSanSiteDecisionKind::Unsupported;
      reason = ConSanAtomicPolicyReason::ConflictingPhysicalAliases;
      result.atomic_errors.push_back(reason);
    } else {
      reason = classify_atomic_semantics(inventory, event, membership);
      if (reason == ConSanAtomicPolicyReason::None)
        reason = classify_atomic_encoding(inventory, event, *membership.sequence, request.engine);
      const bool semantic_not_applicable =
          reason == ConSanAtomicPolicyReason::UnqualifiedSyncSequence ||
          (reason == ConSanAtomicPolicyReason::UnsupportedMemoryRole && membership.sequence &&
           (membership.sequence->memory_role == ConSanSyncMemoryRole::Unknown ||
            membership.sequence->memory_role == ConSanSyncMemoryRole::None)) ||
          (reason == ConSanAtomicPolicyReason::UnsupportedScope && membership.sequence &&
           membership.sequence->raw_scope == 0u);
      kind = reason == ConSanAtomicPolicyReason::None ? ConSanSiteDecisionKind::Admitted
             : semantic_not_applicable                ? ConSanSiteDecisionKind::NotApplicable
                                                      : ConSanSiteDecisionKind::Unsupported;
    }

    ConSanDynamicResultRequirement required_dynamic_result =
        event.kind == ConSanSyncEventKind::Atomic ? dynamic_requirement(event)
                                                  : ConSanDynamicResultRequirement::None;
    if (required_dynamic_result == ConSanDynamicResultRequirement::Count)
      required_dynamic_result = ConSanDynamicResultRequirement::None;
    ConSanAtomicSiteDecision decision{
        .engine = request.engine,
        .semantic_site = semantic_id,
        .kind = kind,
        .capability = capability,
        .reason = reason,
        .association = std::nullopt,
        .dynamic_result = required_dynamic_result,
        .intent_ids = {},
        .source_containers = names,
    };
    if (membership.sequence != nullptr)
      decision.association = ConSanSynchronizationAssociationId{membership.sequence->identity};

    if (decision.kind == ConSanSiteDecisionKind::Admitted && decision.association) {
      const ConSanProbeIntentId capture =
          add_intent(result.plan, semantic_id.physical, {semantic_id},
                     ConSanProbeIntentKind::AtomicAddressCapture, ConSanProbePosition::Before,
                     *decision.association, ConSanDynamicResultRequirement::None);
      decision.intent_ids.push_back(capture);
      const bool record_replay_fence_owns_ordinary =
          request.engine == ConSanCapabilityEngine::RecordReplay &&
          event.kind == ConSanSyncEventKind::OrdinaryMemory &&
          has_qualified_fence_for(synchronization, event.identity);
      if (!record_replay_fence_owns_ordinary) {
        decision.intent_ids.push_back(add_intent(
            result.plan, semantic_id.physical, {semantic_id}, atomic_evidence_kind(request.engine),
            ConSanProbePosition::After, *decision.association, decision.dynamic_result));
      }
    }
    result.plan.atomic_site_decisions.push_back(std::move(decision));
  }

  std::map<uint64_t, std::vector<const ConSanMoiFenceCandidate *>> fence_aliases_by_site;
  for (const ConSanMoiFenceCandidate &fence : synchronization.moi_fence_candidates)
    fence_aliases_by_site[fence.text_offset].push_back(&fence);

  for (const auto &[fence_offset, aliases] : fence_aliases_by_site) {
    const ConSanMoiFenceCandidate &fence = *aliases.front();
    const auto fence_event_entry = events_by_identity.find(fence.fence_event_identity);
    const ConSanSyncEvent *fence_event =
        fence_event_entry == events_by_identity.end() ? nullptr : fence_event_entry->second;
    const SemanticSiteId fence_id =
        fence_event != nullptr ? event_semantic_id(inventory, *fence_event)
                               : SemanticSiteId{
                                     .physical = {.code_object = inventory.code_object_id(),
                                                  .original_text_offset = fence_offset},
                                     .domain = ConSanSemanticSiteDomain::SynchronizationEvent,
                                 };
    const ConSanCapabilityDisposition capability = consan_capability_disposition(
        inventory.target(), request.engine, ConSanCapabilityForm::AddressedOrdinaryFence);
    const bool conflicting_alias =
        std::ranges::any_of(aliases, [&](const ConSanMoiFenceCandidate *alias) {
          return !fence_policy_semantics_equal(fence, *alias);
        });
    ConSanFenceSiteDecision decision{
        .engine = request.engine,
        .semantic_site = fence_id,
        .kind = ConSanSiteDecisionKind::NotApplicable,
        .capability = capability,
        .reason = ConSanFencePolicyReason::TrackingDisabled,
        .inventory_association = fence.association,
        .association = std::nullopt,
        .intent_ids = {},
        .source_containers = fence_source_container_names(aliases),
    };
    if (!fence.sequence_identity.empty())
      decision.association = ConSanSynchronizationAssociationId{fence.sequence_identity};

    if (!request.tracking_enabled) {
      decision.reason = ConSanFencePolicyReason::TrackingDisabled;
    } else if (request.engine == ConSanCapabilityEngine::SuperCollider) {
      decision.reason = ConSanFencePolicyReason::EngineMutationOnly;
    } else if (!fence_filter_matches(aliases, request.container_filter)) {
      decision.reason = ConSanFencePolicyReason::ContainerFilterExcluded;
    } else if (conflicting_alias) {
      decision.kind = ConSanSiteDecisionKind::Unsupported;
      decision.reason = ConSanFencePolicyReason::ConflictingPhysicalAliases;
      result.fence_errors.push_back(decision.reason);
    } else if (!fence.eligible()) {
      decision.reason = ConSanFencePolicyReason::AssociationUnavailable;
    } else if (capability != ConSanCapabilityDisposition::Supported &&
               capability != ConSanCapabilityDisposition::AssociatedOnly) {
      decision.reason = ConSanFencePolicyReason::TargetCapabilityUnavailable;
    } else {
      const auto communication_entry = events_by_identity.find(fence.communication_event_identity);
      const ConSanSyncEvent *communication =
          communication_entry == events_by_identity.end() ? nullptr : communication_entry->second;
      const auto atomic_decision =
          communication == nullptr
              ? result.plan.atomic_site_decisions.end()
              : std::ranges::find_if(
                    result.plan.atomic_site_decisions,
                    [&](const ConSanAtomicSiteDecision &candidate) {
                      return candidate.semantic_site.physical ==
                                 event_semantic_id(inventory, *communication).physical &&
                             candidate.kind == ConSanSiteDecisionKind::Admitted &&
                             candidate.association == decision.association;
                    });
      if (fence_event != nullptr && communication != nullptr &&
          (fence_event->execution_owners.empty() || communication->execution_owners.empty())) {
        decision.reason = ConSanFencePolicyReason::MissingExecutionOwner;
      } else if (fence_event == nullptr || communication == nullptr ||
                 atomic_decision == result.plan.atomic_site_decisions.end()) {
        decision.kind = ConSanSiteDecisionKind::Unsupported;
        decision.reason = ConSanFencePolicyReason::MissingCommunicationEvent;
      } else {
        decision.kind = ConSanSiteDecisionKind::Admitted;
        decision.reason = ConSanFencePolicyReason::None;
        if (request.engine == ConSanCapabilityEngine::RecordReplay) {
          if (!atomic_decision->intent_ids.empty()) {
            const ConSanProbeIntentId capture = atomic_decision->intent_ids.front();
            add_covered_site(result.plan, capture, fence_id);
            decision.intent_ids.push_back(capture);
          }
          const ConSanProbeIntentId record =
              add_intent(result.plan, fence_id.physical, {atomic_decision->semantic_site, fence_id},
                         ConSanProbeIntentKind::FenceRecord, ConSanProbePosition::After,
                         *decision.association, ConSanDynamicResultRequirement::None);
          decision.intent_ids.push_back(record);
          atomic_decision->intent_ids.push_back(record);
        } else {
          decision.intent_ids = atomic_decision->intent_ids;
          for (ConSanProbeIntentId id : decision.intent_ids)
            add_covered_site(result.plan, id, fence_id);
        }
      }
    }
    result.plan.fence_site_decisions.push_back(std::move(decision));
  }

  // A qualified ordinary Record/Replay sequence normally delegates its after
  // evidence to a fence. If a corrupt inventory claimed a qualified candidate
  // but no usable fence decision survived, fall back to the direct atomic
  // record rather than publishing a before-only admitted contract.
  if (request.engine == ConSanCapabilityEngine::RecordReplay) {
    for (ConSanAtomicSiteDecision &decision : result.plan.atomic_site_decisions) {
      if (decision.kind != ConSanSiteDecisionKind::Admitted || !decision.association ||
          decision.intent_ids.size() != 1u)
        continue;
      decision.intent_ids.push_back(
          add_intent(result.plan, decision.semantic_site.physical, {decision.semantic_site},
                     ConSanProbeIntentKind::AtomicRecord, ConSanProbePosition::After,
                     *decision.association, decision.dynamic_result));
    }
  }

  return result;
}

} // namespace rocjitsu
