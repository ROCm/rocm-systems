// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"

#include <algorithm>
#include <map>
#include <ranges>
#include <tuple>

namespace rocjitsu {
namespace {

[[nodiscard]] bool valid_engine(ConSanCapabilityEngine engine) {
  return static_cast<uint8_t>(engine) < static_cast<uint8_t>(ConSanCapabilityEngine::Count);
}

[[nodiscard]] bool valid_decision_kind(ConSanSiteDecisionKind kind) {
  return static_cast<uint8_t>(kind) < static_cast<uint8_t>(ConSanSiteDecisionKind::Count);
}

[[nodiscard]] bool valid_access_reason(ConSanAccessPolicyReason reason) {
  return static_cast<uint8_t>(reason) < static_cast<uint8_t>(ConSanAccessPolicyReason::Count);
}

[[nodiscard]] bool valid_barrier_reason(ConSanBarrierPolicyReason reason) {
  return static_cast<uint8_t>(reason) < static_cast<uint8_t>(ConSanBarrierPolicyReason::Count);
}

[[nodiscard]] bool valid_atomic_reason(ConSanAtomicPolicyReason reason) {
  return static_cast<uint8_t>(reason) < static_cast<uint8_t>(ConSanAtomicPolicyReason::Count);
}

[[nodiscard]] bool valid_fence_reason(ConSanFencePolicyReason reason) {
  return static_cast<uint8_t>(reason) < static_cast<uint8_t>(ConSanFencePolicyReason::Count);
}

[[nodiscard]] bool valid_fence_association(ConSanFenceAssociation association) {
  return static_cast<uint8_t>(association) < static_cast<uint8_t>(ConSanFenceAssociation::Count);
}

[[nodiscard]] bool valid_capability_disposition(ConSanCapabilityDisposition disposition) {
  switch (disposition) {
  case ConSanCapabilityDisposition::OutOfContract:
  case ConSanCapabilityDisposition::NotApplicable:
  case ConSanCapabilityDisposition::Supported:
  case ConSanCapabilityDisposition::MutationOnly:
  case ConSanCapabilityDisposition::AccessOnly:
  case ConSanCapabilityDisposition::AssociatedOnly:
    return true;
  }
  return false;
}

[[nodiscard]] bool valid_intent_kind(ConSanProbeIntentKind kind) {
  return static_cast<uint8_t>(kind) < static_cast<uint8_t>(ConSanProbeIntentKind::Count);
}

[[nodiscard]] bool valid_position(ConSanProbePosition position) {
  return static_cast<uint8_t>(position) < static_cast<uint8_t>(ConSanProbePosition::Count);
}

[[nodiscard]] bool valid_dynamic_result(ConSanDynamicResultRequirement requirement) {
  return static_cast<uint8_t>(requirement) <
         static_cast<uint8_t>(ConSanDynamicResultRequirement::Count);
}

[[nodiscard]] bool atomic_or_fence_intent(ConSanProbeIntentKind kind) {
  switch (kind) {
  case ConSanProbeIntentKind::AtomicAddressCapture:
  case ConSanProbeIntentKind::AtomicRecord:
  case ConSanProbeIntentKind::SampledAtomicOrdering:
  case ConSanProbeIntentKind::ExactAtomicOrdering:
  case ConSanProbeIntentKind::FenceRecord:
    return true;
  default:
    return false;
  }
}

[[nodiscard]] bool valid_lowering_outcome(ConSanLoweringOutcomeKind outcome) {
  return static_cast<uint8_t>(outcome) < static_cast<uint8_t>(ConSanLoweringOutcomeKind::Count);
}

[[nodiscard]] bool contains_physical_site(std::span<const PhysicalSiteId> sites,
                                          const PhysicalSiteId &site) {
  return std::ranges::find(sites, site) != sites.end();
}

[[nodiscard]] bool contains_substring(std::span<const ConSanAccessInventorySite *const> aliases,
                                      std::string_view filter) {
  return filter.empty() || std::ranges::any_of(aliases, [&](const auto *access) {
           return access->container.name.find(filter) != std::string::npos;
         });
}

[[nodiscard]] bool access_alias_semantics_equal(const ConSanAccessInventorySite &lhs,
                                                const ConSanAccessInventorySite &rhs) {
  return std::tie(lhs.container.kind, lhs.container.entry_text_offset, lhs.origin, lhs.kind,
                  lhs.address_space, lhs.provenance, lhs.confidence, lhs.supported_mvp,
                  lhs.file_offset, lhs.instruction_size, lhs.decoded_width_bits, lhs.mnemonic,
                  lhs.flat_address_space_hint, lhs.operands, lhs.ranges, lhs.exclusions) ==
         std::tie(rhs.container.kind, rhs.container.entry_text_offset, rhs.origin, rhs.kind,
                  rhs.address_space, rhs.provenance, rhs.confidence, rhs.supported_mvp,
                  rhs.file_offset, rhs.instruction_size, rhs.decoded_width_bits, rhs.mnemonic,
                  rhs.flat_address_space_hint, rhs.operands, rhs.ranges, rhs.exclusions);
}

[[nodiscard]] ConSanAccessPolicyReason
inventory_exclusion_reason(const ConSanAccessInventorySite &access) {
  if (access.exclusions.empty() && !access.ranges.empty())
    return ConSanAccessPolicyReason::None;
  if (access.exclusions.empty())
    return ConSanAccessPolicyReason::RangeEncodingUnavailable;
  switch (access.exclusions.front().reason) {
  case ConSanInventoryExclusionReason::NonAccessInstruction:
    return ConSanAccessPolicyReason::NonAccessInstruction;
  case ConSanInventoryExclusionReason::InvalidInstructionSize:
    return ConSanAccessPolicyReason::InvalidInstructionSize;
  case ConSanInventoryExclusionReason::InvalidAccessWidth:
    return ConSanAccessPolicyReason::InvalidAccessWidth;
  case ConSanInventoryExclusionReason::MissingAddressOperand:
    return ConSanAccessPolicyReason::MissingAddressOperand;
  case ConSanInventoryExclusionReason::RangeEncodingUnavailable:
    return ConSanAccessPolicyReason::RangeEncodingUnavailable;
  case ConSanInventoryExclusionReason::Count:
    break;
  }
  return ConSanAccessPolicyReason::RangeEncodingUnavailable;
}

[[nodiscard]] bool is_supported_relaxed_lds_atomic(std::string_view mnemonic) {
  return mnemonic == "ds_add_f32" || mnemonic == "ds_add_f64" || mnemonic == "ds_add_u32" ||
         mnemonic == "ds_add_u64" || mnemonic == "ds_cmpstore_rtn_b32" ||
         mnemonic == "ds_cmpst_rtn_b32";
}

[[nodiscard]] bool is_supported_single_range_native_lds(std::string_view mnemonic,
                                                        rj_code_arch_t arch) {
  if ((arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
       is_rdna4_family_arch(arch)) &&
      (mnemonic == "ds_load_b96" || mnemonic == "ds_store_b96"))
    return true;
  if (consan_uses_gfx9_cdna_encoding(arch) &&
      (mnemonic == "ds_read_b96" || mnemonic == "ds_write_b96"))
    return true;
  constexpr std::array always = {
      "ds_load_i8",         "ds_load_u8",          "ds_load_i16",       "ds_load_u16",
      "ds_load_u8_d16",     "ds_load_u8_d16_hi",   "ds_load_i8_d16",    "ds_load_i8_d16_hi",
      "ds_load_u16_d16",    "ds_load_u16_d16_hi",  "ds_store_b8",       "ds_store_b16",
      "ds_store_b8_d16_hi", "ds_store_b16_d16_hi", "ds_read_u8",        "ds_read_u16",
      "ds_write_b8",        "ds_write_b16",        "ds_load_b32",       "ds_load_b64",
      "ds_load_b128",       "ds_load_tr8_b64",     "ds_load_tr16_b128", "ds_read_b32",
      "ds_read_b64",        "ds_read_b64_tr_b16",  "ds_read_b128",      "ds_store_b32",
      "ds_store_b64",       "ds_store_b128",       "ds_write_b32",      "ds_write_b64",
      "ds_write_b128",      "ds_add_f32",          "ds_add_f64",        "ds_add_u32",
      "ds_add_u64",         "ds_cmpstore_rtn_b32", "ds_cmpst_rtn_b32",
  };
  if (std::ranges::find(always, mnemonic) != always.end())
    return true;
  constexpr std::array gfx9 = {
      "ds_read_i8",         "ds_read_u8_d16",     "ds_read_u8_d16_hi",
      "ds_read_i8_d16",     "ds_read_i8_d16_hi",  "ds_read_u16_d16",
      "ds_read_u16_d16_hi", "ds_write_b8_d16_hi", "ds_write_b16_d16_hi",
  };
  return consan_uses_gfx9_cdna_encoding(arch) && std::ranges::find(gfx9, mnemonic) != gfx9.end();
}

[[nodiscard]] bool is_supported_two_range_native_lds(std::string_view mnemonic) {
  constexpr std::array forms = {
      "ds_load_2addr_b32",
      "ds_store_2addr_b32",
      "ds_read2_b32",
      "ds_write2_b32",
      "ds_load_2addr_b64",
      "ds_store_2addr_b64",
      "ds_read2_b64",
      "ds_write2_b64",
      "ds_load_2addr_stride64_b32",
      "ds_store_2addr_stride64_b32",
      "ds_read2st64_b32",
      "ds_write2st64_b32",
      "ds_load_2addr_stride64_b64",
      "ds_store_2addr_stride64_b64",
      "ds_read2st64_b64",
      "ds_write2st64_b64",
  };
  return std::ranges::find(forms, mnemonic) != forms.end();
}

[[nodiscard]] bool is_supported_flat_access(std::string_view mnemonic) {
  if (consan_flat_load_subword_semantics(mnemonic) || consan_flat_store_subword_semantics(mnemonic))
    return true;
  constexpr std::array forms = {
      "flat_load_b32",     "flat_load_b64",    "flat_load_b128",     "flat_store_b32",
      "flat_store_b64",    "flat_store_b128",  "flat_load_dword",    "flat_load_dwordx2",
      "flat_load_dwordx4", "flat_store_dword", "flat_store_dwordx2", "flat_store_dwordx4",
      "flat_load_ushort",  "flat_store_short", "flat_load_u16",      "flat_store_b16",
  };
  return std::ranges::find(forms, mnemonic) != forms.end();
}

[[nodiscard]] uint32_t vector_flat_no_saddr(rj_code_arch_t arch) {
  if (consan_uses_gfx9_cdna_encoding(arch))
    return 0u;
  if (arch == ROCJITSU_CODE_ARCH_CDNA5)
    return static_cast<uint32_t>(cdna5::OPR_SREG_NULL);
  return static_cast<uint32_t>(rdna4::OPR_SREG_NULL);
}

[[nodiscard]] ConSanAccessPolicyReason
classify_moi_access_support(const ConSanAccessInventorySite &access, rj_code_arch_t arch) {
  const ConSanAccessPolicyReason inventory_reason = inventory_exclusion_reason(access);
  if (inventory_reason != ConSanAccessPolicyReason::None)
    return inventory_reason;
  if (access.file_offset > access.physical_id.code_object.byte_size ||
      access.instruction_size > access.physical_id.code_object.byte_size - access.file_offset)
    return ConSanAccessPolicyReason::InstructionOutOfBounds;

  if (access.origin != ConSanAccessOrigin::Flat) {
    if (access.origin == ConSanAccessOrigin::DirectToLds)
      return ConSanAccessPolicyReason::None;
    return is_supported_single_range_native_lds(access.mnemonic, arch) ||
                   is_supported_two_range_native_lds(access.mnemonic)
               ? ConSanAccessPolicyReason::None
               : ConSanAccessPolicyReason::UnsupportedMnemonic;
  }

  const bool gfx12_encoding = access.instruction_size == 3u * sizeof(uint32_t);
  const bool cdna4_encoding =
      access.instruction_size == 2u * sizeof(uint32_t) && access.operands.raw_segment == 0u;
  if (!gfx12_encoding && !cdna4_encoding)
    return ConSanAccessPolicyReason::UnsupportedFlatEncoding;
  if (!access.operands.raw_ioffset)
    return ConSanAccessPolicyReason::UnsupportedFlatEncoding;
  if (*access.operands.raw_ioffset != 0 && !is_rdna4_family_arch(arch))
    return ConSanAccessPolicyReason::NonzeroFlatOffset;
  if (is_rdna4_family_arch(arch) &&
      (!access.operands.raw_saddr || !access.operands.raw_scale_offset))
    return ConSanAccessPolicyReason::UnsupportedFlatEncoding;
  const bool scalar_vector_address = is_rdna4_family_arch(arch) && access.operands.raw_saddr &&
                                     *access.operands.raw_saddr != vector_flat_no_saddr(arch);
  if (access.operands.address_vgpr && *access.operands.address_vgpr >= 255u &&
      !scalar_vector_address)
    return ConSanAccessPolicyReason::ReservedFlatAddressRegister;
  return is_supported_flat_access(access.mnemonic) ? ConSanAccessPolicyReason::None
                                                   : ConSanAccessPolicyReason::UnsupportedMnemonic;
}

[[nodiscard]] ConSanAccessPolicyReason
classify_supercollider_access_support(const ConSanAccessInventorySite &access,
                                      ConSanFlatProvenanceMode provenance_mode,
                                      rj_code_arch_t arch) {
  const ConSanAccessPolicyReason inventory_reason = inventory_exclusion_reason(access);
  if (inventory_reason != ConSanAccessPolicyReason::None)
    return inventory_reason;
  if (access.file_offset > access.physical_id.code_object.byte_size ||
      access.instruction_size > access.physical_id.code_object.byte_size - access.file_offset)
    return ConSanAccessPolicyReason::InstructionOutOfBounds;
  const bool supported = consan_supercollider_supports_access(access, provenance_mode, arch);
  return supported ? ConSanAccessPolicyReason::None : ConSanAccessPolicyReason::UnsupportedMnemonic;
}

[[nodiscard]] ConSanProbeIntentKind intent_kind(ConSanCapabilityEngine engine) {
  switch (engine) {
  case ConSanCapabilityEngine::SuperCollider:
    return ConSanProbeIntentKind::RedundantAccessObservation;
  case ConSanCapabilityEngine::RecordReplay:
    return ConSanProbeIntentKind::AccessRecord;
  case ConSanCapabilityEngine::Sampled:
    return ConSanProbeIntentKind::SampledAccess;
  case ConSanCapabilityEngine::InlineShadow:
    return ConSanProbeIntentKind::ExactShadowAccess;
  case ConSanCapabilityEngine::Count:
    break;
  }
  return ConSanProbeIntentKind::Count;
}

[[nodiscard]] SemanticSiteId fallback_access_id(const ConSanAccessInventorySite &access) {
  return {
      .physical = access.physical_id,
      .domain = ConSanSemanticSiteDomain::Access,
      .member_ordinal = 0,
      .range_ordinal = 0,
  };
}

[[nodiscard]] std::vector<SemanticSiteId> semantic_ids(const ConSanAccessInventorySite &access) {
  std::vector<SemanticSiteId> result;
  result.reserve(std::max<size_t>(access.ranges.size(), 1u));
  for (const ConSanAccessRange &range : access.ranges)
    result.push_back(range.id.valid() ? range.id : fallback_access_id(access));
  if (result.empty())
    result.push_back(fallback_access_id(access));
  return result;
}

[[nodiscard]] std::vector<std::string>
container_names(std::span<const ConSanAccessInventorySite *const> aliases) {
  std::vector<std::string> result;
  result.reserve(aliases.size());
  for (const ConSanAccessInventorySite *access : aliases) {
    if (std::ranges::find(result, access->container.name) == result.end())
      result.push_back(access->container.name);
  }
  std::ranges::sort(result);
  return result;
}

} // namespace

const ConSanProbeIntent *ConSanObservationPlan::intent(ConSanProbeIntentId id) const {
  if (!id.valid() || id.value >= probe_intents.size())
    return nullptr;
  const ConSanProbeIntent &candidate = probe_intents[id.value];
  return candidate.id == id ? &candidate : nullptr;
}

bool ConSanObservationPlan::valid() const {
  if (!valid_engine(engine))
    return false;
  for (size_t index = 0; index < probe_intents.size(); ++index) {
    const ConSanProbeIntent &probe = probe_intents[index];
    if (probe.id.value != index || probe.engine != engine || !probe.physical_site.valid() ||
        probe.covered_semantic_sites.empty() || !valid_intent_kind(probe.kind) ||
        !valid_position(probe.position) || !valid_dynamic_result(probe.dynamic_result) ||
        std::ranges::any_of(probe.covered_semantic_sites,
                            [](const SemanticSiteId &site) { return !site.valid(); })) {
      return false;
    }
    const bool synchronization_intent = atomic_or_fence_intent(probe.kind);
    if (synchronization_intent != probe.synchronization_association.has_value() ||
        (probe.synchronization_association && !probe.synchronization_association->valid())) {
      return false;
    }
    if (probe.kind == ConSanProbeIntentKind::AtomicAddressCapture) {
      if (probe.position != ConSanProbePosition::Before ||
          probe.dynamic_result != ConSanDynamicResultRequirement::None) {
        return false;
      }
    } else if (synchronization_intent && probe.position != ConSanProbePosition::After) {
      return false;
    } else if (!synchronization_intent &&
               probe.dynamic_result != ConSanDynamicResultRequirement::None) {
      return false;
    }
  }
  for (const ConSanSiteDecision &decision : site_decisions) {
    if (decision.engine != engine || !decision.semantic_site.valid() ||
        !valid_decision_kind(decision.kind) || !valid_access_reason(decision.reason) ||
        decision.source_containers.empty()) {
      return false;
    }
    const bool admitted = decision.kind == ConSanSiteDecisionKind::Admitted;
    if (admitted != (decision.reason == ConSanAccessPolicyReason::None) ||
        admitted != !decision.intent_ids.empty()) {
      return false;
    }
    for (ConSanProbeIntentId id : decision.intent_ids) {
      const ConSanProbeIntent *probe = intent(id);
      if (probe == nullptr ||
          std::ranges::find(probe->covered_semantic_sites, decision.semantic_site) ==
              probe->covered_semantic_sites.end()) {
        return false;
      }
    }
  }
  for (const ConSanBarrierSiteDecision &decision : barrier_site_decisions) {
    if (decision.engine != engine || !decision.semantic_site.valid() ||
        decision.semantic_site.domain != ConSanSemanticSiteDomain::SynchronizationEvent ||
        !valid_decision_kind(decision.kind) || !valid_barrier_reason(decision.reason) ||
        decision.source_containers.empty()) {
      return false;
    }
    const bool admitted = decision.kind == ConSanSiteDecisionKind::Admitted;
    if (admitted != (decision.reason == ConSanBarrierPolicyReason::None) ||
        admitted != !decision.intent_ids.empty()) {
      return false;
    }
    for (ConSanProbeIntentId id : decision.intent_ids) {
      const ConSanProbeIntent *probe = intent(id);
      if (probe == nullptr ||
          std::ranges::find(probe->covered_semantic_sites, decision.semantic_site) ==
              probe->covered_semantic_sites.end()) {
        return false;
      }
    }
  }
  for (const ConSanAtomicSiteDecision &decision : atomic_site_decisions) {
    if (decision.engine != engine || !decision.semantic_site.valid() ||
        decision.semantic_site.domain != ConSanSemanticSiteDomain::SynchronizationEvent ||
        !valid_decision_kind(decision.kind) || !valid_capability_disposition(decision.capability) ||
        !valid_atomic_reason(decision.reason) || !valid_dynamic_result(decision.dynamic_result) ||
        decision.source_containers.empty() ||
        (decision.association && !decision.association->valid())) {
      return false;
    }
    const bool admitted = decision.kind == ConSanSiteDecisionKind::Admitted;
    if (admitted != (decision.reason == ConSanAtomicPolicyReason::None) ||
        admitted != !decision.intent_ids.empty() ||
        (admitted && (!decision.association ||
                      (decision.capability != ConSanCapabilityDisposition::Supported &&
                       decision.capability != ConSanCapabilityDisposition::AssociatedOnly)))) {
      return false;
    }
    for (ConSanProbeIntentId id : decision.intent_ids) {
      const ConSanProbeIntent *probe = intent(id);
      if (probe == nullptr || probe->synchronization_association != decision.association ||
          std::ranges::find(probe->covered_semantic_sites, decision.semantic_site) ==
              probe->covered_semantic_sites.end()) {
        return false;
      }
    }
  }
  for (const ConSanFenceSiteDecision &decision : fence_site_decisions) {
    if (decision.engine != engine || !decision.semantic_site.valid() ||
        decision.semantic_site.domain != ConSanSemanticSiteDomain::SynchronizationEvent ||
        !valid_decision_kind(decision.kind) || !valid_capability_disposition(decision.capability) ||
        !valid_fence_reason(decision.reason) ||
        !valid_fence_association(decision.inventory_association) ||
        decision.source_containers.empty() ||
        (decision.association && !decision.association->valid())) {
      return false;
    }
    const bool admitted = decision.kind == ConSanSiteDecisionKind::Admitted;
    if (admitted != (decision.reason == ConSanFencePolicyReason::None) ||
        admitted != !decision.intent_ids.empty() ||
        (admitted && (decision.inventory_association != ConSanFenceAssociation::Qualified ||
                      !decision.association ||
                      (decision.capability != ConSanCapabilityDisposition::Supported &&
                       decision.capability != ConSanCapabilityDisposition::AssociatedOnly)))) {
      return false;
    }
    for (ConSanProbeIntentId id : decision.intent_ids) {
      const ConSanProbeIntent *probe = intent(id);
      if (probe == nullptr || probe->synchronization_association != decision.association ||
          std::ranges::find(probe->covered_semantic_sites, decision.semantic_site) ==
              probe->covered_semantic_sites.end()) {
        return false;
      }
    }
  }
  return true;
}

bool ConSanObservationPlan::append(const ConSanObservationPlan &fragment) {
  if (!valid() || !fragment.valid() || engine != fragment.engine)
    return false;
  if (probe_intents.size() > ConSanProbeIntentId::invalid_value - fragment.probe_intents.size())
    return false;

  ConSanObservationPlan combined = *this;
  const uint32_t intent_base = static_cast<uint32_t>(combined.probe_intents.size());
  for (ConSanProbeIntent probe : fragment.probe_intents) {
    probe.id.value += intent_base;
    combined.probe_intents.push_back(std::move(probe));
  }
  for (ConSanSiteDecision decision : fragment.site_decisions) {
    for (ConSanProbeIntentId &id : decision.intent_ids)
      id.value += intent_base;
    combined.site_decisions.push_back(std::move(decision));
  }
  for (ConSanBarrierSiteDecision decision : fragment.barrier_site_decisions) {
    for (ConSanProbeIntentId &id : decision.intent_ids)
      id.value += intent_base;
    combined.barrier_site_decisions.push_back(std::move(decision));
  }
  for (ConSanAtomicSiteDecision decision : fragment.atomic_site_decisions) {
    for (ConSanProbeIntentId &id : decision.intent_ids)
      id.value += intent_base;
    combined.atomic_site_decisions.push_back(std::move(decision));
  }
  for (ConSanFenceSiteDecision decision : fragment.fence_site_decisions) {
    for (ConSanProbeIntentId &id : decision.intent_ids)
      id.value += intent_base;
    combined.fence_site_decisions.push_back(std::move(decision));
  }
  if (!combined.valid())
    return false;
  *this = std::move(combined);
  return true;
}

ConSanCoverageLedger::ConSanCoverageLedger(const ConSanObservationPlan &plan)
    : site_decisions_(plan.site_decisions), barrier_site_decisions_(plan.barrier_site_decisions),
      atomic_site_decisions_(plan.atomic_site_decisions),
      fence_site_decisions_(plan.fence_site_decisions) {
  intent_entries_.reserve(plan.probe_intents.size());
  for (const ConSanProbeIntent &intent : plan.probe_intents)
    intent_entries_.push_back({
        .intent = intent,
        .lowering = ConSanLoweringOutcomeKind::Pending,
        .detail = {},
    });
}

const ConSanIntentCoverageEntry *ConSanCoverageLedger::intent_entry(ConSanProbeIntentId id) const {
  if (!id.valid() || id.value >= intent_entries_.size())
    return nullptr;
  const ConSanIntentCoverageEntry &entry = intent_entries_[id.value];
  return entry.intent.id == id ? &entry : nullptr;
}

bool ConSanCoverageLedger::set_lowering_outcome(ConSanProbeIntentId id,
                                                ConSanLoweringOutcomeKind outcome,
                                                std::string detail) {
  if (!valid_lowering_outcome(outcome) || !id.valid() || id.value >= intent_entries_.size() ||
      intent_entries_[id.value].intent.id != id) {
    return false;
  }
  intent_entries_[id.value].lowering = outcome;
  intent_entries_[id.value].detail = std::move(detail);
  return true;
}

size_t ConSanCoverageLedger::set_physical_lowering_outcome(const PhysicalSiteId &physical_site,
                                                           ConSanLoweringOutcomeKind outcome,
                                                           std::string detail) {
  if (!physical_site.valid() || !valid_lowering_outcome(outcome))
    return 0;
  size_t updated = 0;
  for (ConSanIntentCoverageEntry &entry : intent_entries_) {
    if (entry.intent.physical_site != physical_site)
      continue;
    entry.lowering = outcome;
    entry.detail = detail;
    ++updated;
  }
  return updated;
}

bool ConSanCoverageLedger::all_intents_instrumented() const {
  return std::ranges::all_of(intent_entries_, [](const ConSanIntentCoverageEntry &entry) {
    return entry.lowering == ConSanLoweringOutcomeKind::Instrumented;
  });
}

ConSanAccessPolicyResult plan_consan_access_observation(const ProgramInventory &inventory,
                                                        const ConSanAccessPolicyRequest &request) {
  ConSanAccessPolicyResult result;
  result.plan.engine = request.engine;
  if (!valid_engine(request.engine) || inventory.empty())
    return result;

  std::map<uint64_t, std::vector<const ConSanAccessInventorySite *>> aliases_by_offset;
  for (const ConSanAccessInventorySite &access : inventory.access_sites())
    aliases_by_offset[access.physical_id.original_text_offset].push_back(&access);

  for (const auto &[offset, aliases] : aliases_by_offset) {
    (void)offset;
    const ConSanAccessInventorySite &access = *aliases.front();
    const std::vector<SemanticSiteId> ids = semantic_ids(access);
    const std::vector<std::string> names = container_names(aliases);
    ConSanSiteDecisionKind decision_kind = ConSanSiteDecisionKind::NotApplicable;
    ConSanAccessPolicyReason reason = ConSanAccessPolicyReason::AccessFamilyDisabled;

    const bool conflicting_alias = std::ranges::any_of(
        aliases, [&](const auto *alias) { return !access_alias_semantics_equal(access, *alias); });
    const bool flat = access.origin == ConSanAccessOrigin::Flat;
    const ConSanCapabilityForm form =
        flat ? ConSanCapabilityForm::GroupFlatAccess : ConSanCapabilityForm::NativeLdsAccess;
    const bool enabled = flat ? request.group_flat_enabled : request.native_lds_enabled;

    if (!contains_substring(aliases, request.container_filter)) {
      reason = ConSanAccessPolicyReason::ContainerFilterExcluded;
    } else if (contains_physical_site(request.reserved_for_synchronization, access.physical_id)) {
      reason = ConSanAccessPolicyReason::ReservedForSynchronizationPolicy;
    } else if (!enabled) {
      reason = ConSanAccessPolicyReason::AccessFamilyDisabled;
    } else if (!consan_arch_supports_capability_form(inventory.arch(), form) ||
               consan_capability_disposition(inventory.target(), request.engine, form) !=
                   ConSanCapabilityDisposition::Supported) {
      reason = ConSanAccessPolicyReason::TargetCapabilityUnavailable;
    } else if (access.kind != ConSanLdsAccessKind::Read &&
               access.kind != ConSanLdsAccessKind::Write &&
               !(request.engine != ConSanCapabilityEngine::SuperCollider && !flat &&
                 access.kind == ConSanLdsAccessKind::Atomic &&
                 consan_arch_supports_capability_form(
                     inventory.arch(), ConSanCapabilityForm::RelaxedLdsAtomicAccess) &&
                 is_supported_relaxed_lds_atomic(access.mnemonic))) {
      reason = ConSanAccessPolicyReason::OperationKindExcluded;
    } else if (flat && access.address_space == ConSanAccessAddressSpace::NonGroup) {
      reason = ConSanAccessPolicyReason::NonGroupAddressSpace;
    } else if (flat && access.address_space == ConSanAccessAddressSpace::Unresolved) {
      reason = ConSanAccessPolicyReason::FlatProvenancePolicyExcluded;
    } else if (flat && access.confidence != ConSanSemanticConfidence::Exact &&
               request.flat_provenance_mode == ConSanFlatProvenanceMode::Strict) {
      reason = ConSanAccessPolicyReason::FlatProvenancePolicyExcluded;
    } else if (conflicting_alias) {
      decision_kind = ConSanSiteDecisionKind::Unsupported;
      reason = ConSanAccessPolicyReason::ConflictingPhysicalAliases;
      result.errors.push_back(reason);
    } else {
      reason = request.engine == ConSanCapabilityEngine::SuperCollider
                   ? classify_supercollider_access_support(access, request.flat_provenance_mode,
                                                           inventory.arch())
                   : classify_moi_access_support(access, inventory.arch());
      decision_kind = reason == ConSanAccessPolicyReason::None
                          ? ConSanSiteDecisionKind::Admitted
                          : ConSanSiteDecisionKind::Unsupported;
    }

    std::optional<ConSanProbeIntentId> intent_id;
    if (decision_kind == ConSanSiteDecisionKind::Admitted) {
      intent_id = ConSanProbeIntentId{static_cast<uint32_t>(result.plan.probe_intents.size())};
      result.plan.probe_intents.push_back({
          .id = *intent_id,
          .engine = request.engine,
          .physical_site = access.physical_id,
          .covered_semantic_sites = ids,
          .kind = intent_kind(request.engine),
          .position = ConSanProbePosition::Before,
          .synchronization_association = std::nullopt,
          .dynamic_result = ConSanDynamicResultRequirement::None,
      });
    }
    for (const SemanticSiteId &id : ids) {
      ConSanSiteDecision decision{
          .engine = request.engine,
          .semantic_site = id,
          .kind = decision_kind,
          .reason = reason,
          .intent_ids = {},
          .source_containers = names,
      };
      if (intent_id)
        decision.intent_ids.push_back(*intent_id);
      result.plan.site_decisions.push_back(std::move(decision));
    }
  }
  return result;
}

} // namespace rocjitsu
