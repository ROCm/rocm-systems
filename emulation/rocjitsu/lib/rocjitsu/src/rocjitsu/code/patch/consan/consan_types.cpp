// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/kernel_scope.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_moi.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/code/patch/spill_manager.h"
#include "rocjitsu/code/patch/trampoline_builder.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx12_cache_flags.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/bit.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rocjitsu {

namespace {

[[nodiscard]] bool ascii_iequals(std::string_view lhs, std::string_view rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    const auto l = static_cast<unsigned char>(lhs[i]);
    const auto r = static_cast<unsigned char>(rhs[i]);
    if (std::tolower(l) != std::tolower(r))
      return false;
  }
  return true;
}

} // namespace

std::optional<uint16_t> consan_gfx1250_vgpr_msb_mode_at(std::span<const uint8_t> bytes,
                                                        uint64_t text_file_offset,
                                                        uint64_t container_entry_text_offset,
                                                        uint64_t site_file_offset) {
  if (text_file_offset > bytes.size() ||
      container_entry_text_offset > bytes.size() - text_file_offset)
    return std::nullopt;
  const uint64_t container_file_offset = text_file_offset + container_entry_text_offset;
  if (container_file_offset > site_file_offset || site_file_offset > bytes.size())
    return std::nullopt;
  uint16_t mode = 0;
  for (uint64_t offset = container_file_offset; offset + sizeof(uint32_t) <= site_file_offset;
       offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + offset, sizeof(word));
    if ((word & 0xFFFF0000u) == 0xBF860000u)
      // The low byte is the mode established by this instruction.  The high
      // byte records the previous mode and must not become persistent state.
      mode = static_cast<uint16_t>(word & 0xFFu);
  }
  return mode;
}

const char *consan_flavor_name(ConSanFlavor flavor) {
  switch (flavor) {
  case ConSanFlavor::None:
    return "none";
  case ConSanFlavor::SuperCollider:
    return "supercollider";
  case ConSanFlavor::Moi:
    return "moi";
  }
  return "unknown";
}

const char *consan_moi_engine_name(ConSanMoiEngine engine) {
  switch (engine) {
  case ConSanMoiEngine::RecordReplay:
    return "record_replay";
  case ConSanMoiEngine::InlineShadow:
    return "inline_shadow";
  case ConSanMoiEngine::Sampled:
    return "sampled";
  }
  return "unknown";
}

const char *consan_transform_outcome_name(ConSanTransformOutcome outcome) {
  switch (outcome) {
  case ConSanTransformOutcome::Unchanged:
    return "unchanged";
  case ConSanTransformOutcome::ModifiedValid:
    return "modified-valid";
  case ConSanTransformOutcome::Unsupported:
    return "unsupported";
  case ConSanTransformOutcome::Invalid:
    return "invalid";
  }
  return "unknown";
}

const char *consan_resource_site_kind_name(ConSanResourceSiteKind kind) {
  switch (kind) {
  case ConSanResourceSiteKind::Access:
    return "access";
  case ConSanResourceSiteKind::Barrier:
    return "barrier";
  case ConSanResourceSiteKind::Atomic:
    return "atomic";
  case ConSanResourceSiteKind::Fence:
    return "fence";
  }
  return "unknown";
}

const char *consan_register_allocation_source_name(ConSanRegisterAllocationSource source) {
  switch (source) {
  case ConSanRegisterAllocationSource::Unsupported:
    return "unsupported";
  case ConSanRegisterAllocationSource::Explicit:
    return "explicit";
  case ConSanRegisterAllocationSource::LivenessDead:
    return "dead";
  case ConSanRegisterAllocationSource::DescriptorGrowth:
    return "descriptor-growth";
  case ConSanRegisterAllocationSource::SpillRequired:
    return "spill";
  }
  return "unknown";
}

const char *consan_delay_mode_name(ConSanDelayMode mode) {
  switch (mode) {
  case ConSanDelayMode::Nop:
    return "nop";
  case ConSanDelayMode::Sleep:
    return "sleep";
  case ConSanDelayMode::SleepVar:
    return "sleep_var";
  }
  return "unknown";
}

const char *consan_barrier_operand_source_name(ConSanBarrierSite::OperandSource source) {
  switch (source) {
  case ConSanBarrierSite::OperandSource::Unknown:
    return "unknown";
  case ConSanBarrierSite::OperandSource::Immediate:
    return "immediate";
  case ConSanBarrierSite::OperandSource::DynamicM0:
    return "dynamic-m0";
  case ConSanBarrierSite::OperandSource::StaticM0Literal32:
    return "static-m0-literal32";
  case ConSanBarrierSite::OperandSource::Literal32:
    return "literal32";
  case ConSanBarrierSite::OperandSource::Literal64:
    return "literal64";
  }
  return "unknown";
}

const char *consan_barrier_scope_name(ConSanBarrierSite::Scope scope) {
  switch (scope) {
  case ConSanBarrierSite::Scope::Unknown:
    return "unknown";
  case ConSanBarrierSite::Scope::Workgroup:
    return "workgroup";
  case ConSanBarrierSite::Scope::Cluster:
    return "cluster";
  }
  return "unknown";
}

const char *consan_moi_candidate_source_name(ConSanMoiCandidateSource source) {
  switch (source) {
  case ConSanMoiCandidateSource::NativeLds:
    return "native-lds";
  case ConSanMoiCandidateSource::FlatGroup:
    return "flat-group";
  case ConSanMoiCandidateSource::FlatMaybeGroup:
    return "flat-maybe-group";
  }
  return "unknown";
}

const char *consan_lds_access_kind_name(ConSanLdsAccessKind kind) {
  switch (kind) {
  case ConSanLdsAccessKind::Read:
    return "read";
  case ConSanLdsAccessKind::Write:
    return "write";
  case ConSanLdsAccessKind::Atomic:
    return "atomic";
  case ConSanLdsAccessKind::Other:
    return "other";
  }
  return "unknown";
}

const char *consan_flat_address_space_hint_name(ConSanFlatAddressSpaceHint hint) {
  switch (hint) {
  case ConSanFlatAddressSpaceHint::Unknown:
    return "unknown";
  case ConSanFlatAddressSpaceHint::Group:
    return "group";
  case ConSanFlatAddressSpaceHint::Private:
    return "private";
  case ConSanFlatAddressSpaceHint::MaybeGroup:
    return "maybe-group";
  case ConSanFlatAddressSpaceHint::MaybePrivate:
    return "maybe-private";
  case ConSanFlatAddressSpaceHint::Global:
    return "global";
  }
  return "unknown";
}

const char *consan_register_plan_reason_name(ConSanRegisterPlanReason reason) {
  switch (reason) {
  case ConSanRegisterPlanReason::None:
    return "none";
  case ConSanRegisterPlanReason::InvalidRequest:
    return "invalid_request";
  case ConSanRegisterPlanReason::ExplicitMisaligned:
    return "explicit_misaligned";
  case ConSanRegisterPlanReason::ExplicitOutOfRange:
    return "explicit_out_of_range";
  case ConSanRegisterPlanReason::ExplicitLive:
    return "explicit_live";
  case ConSanRegisterPlanReason::ForbiddenOverlap:
    return "forbidden_overlap";
  case ConSanRegisterPlanReason::MissingInstruction:
    return "missing_instruction";
  case ConSanRegisterPlanReason::MissingOwner:
    return "missing_owner";
  case ConSanRegisterPlanReason::AmbiguousOwners:
    return "ambiguous_owners";
  case ConSanRegisterPlanReason::InvalidDescriptor:
    return "invalid_descriptor";
  case ConSanRegisterPlanReason::NoLegalWindow:
    return "no_legal_window";
  case ConSanRegisterPlanReason::DynamicStack:
    return "dynamic_stack";
  }
  return "unknown";
}

ConSanResourcePlanAlternativeOutcome
consan_resource_plan_alternative_outcome(const ConSanCandidateResourcePlan &plan,
                                         const ConSanResourcePlanAlternative &alternative) {
  if (alternative.outcome == ConSanResourcePlanAlternativeOutcome::Selected &&
      plan.source == ConSanRegisterAllocationSource::Unsupported)
    return ConSanResourcePlanAlternativeOutcome::Vetoed;
  return alternative.outcome;
}

const char *consan_resource_plan_alternative_kind_name(ConSanResourcePlanAlternativeKind kind) {
  switch (kind) {
  case ConSanResourcePlanAlternativeKind::GuestOperandOverlapSpill:
    return "guest_operand_overlap_spill";
  case ConSanResourcePlanAlternativeKind::SpillBackedOperandRecovery:
    return "spill_backed_operand_recovery";
  }
  return "unknown";
}

const char *
consan_resource_plan_alternative_outcome_name(ConSanResourcePlanAlternativeOutcome outcome) {
  switch (outcome) {
  case ConSanResourcePlanAlternativeOutcome::Selected:
    return "selected";
  case ConSanResourcePlanAlternativeOutcome::Rejected:
    return "rejected";
  case ConSanResourcePlanAlternativeOutcome::Superseded:
    return "superseded";
  case ConSanResourcePlanAlternativeOutcome::Contributed:
    return "contributed";
  case ConSanResourcePlanAlternativeOutcome::Vetoed:
    return "vetoed";
  }
  return "unknown";
}

const char *consan_site_disposition_name(ConSanSiteDisposition disposition) {
  switch (disposition) {
  case ConSanSiteDisposition::NotApplicable:
    return "not_applicable";
  case ConSanSiteDisposition::Supported:
    return "supported";
  case ConSanSiteDisposition::Unsupported:
    return "unsupported";
  }
  return "unknown";
}

const char *consan_site_lowering_outcome_name(ConSanSiteLoweringOutcome outcome) {
  switch (outcome) {
  case ConSanSiteLoweringOutcome::NotApplicable:
    return "not_applicable";
  case ConSanSiteLoweringOutcome::Unsupported:
    return "unsupported";
  case ConSanSiteLoweringOutcome::Pending:
    return "pending";
  case ConSanSiteLoweringOutcome::Patched:
    return "patched";
  case ConSanSiteLoweringOutcome::ResourceFailed:
    return "resource_failed";
  case ConSanSiteLoweringOutcome::PlacementOrLoweringFailed:
    return "placement_or_lowering_failed";
  }
  return "unknown";
}

const char *consan_site_lowering_reason_name(ConSanSiteLoweringReason reason) {
  switch (reason) {
  case ConSanSiteLoweringReason::None:
    return "none";
  case ConSanSiteLoweringReason::SemanticNotApplicable:
    return "semantic_not_applicable";
  case ConSanSiteLoweringReason::SemanticUnsupported:
    return "semantic_unsupported";
  case ConSanSiteLoweringReason::UnsupportedResourcePlan:
    return "unsupported_resource_plan";
  case ConSanSiteLoweringReason::InstrumentationPatchMissing:
    return "instrumentation_patch_missing";
  }
  return "unknown";
}

const char *consan_site_disposition_reason_name(ConSanSiteDispositionReason reason) {
  switch (reason) {
  case ConSanSiteDispositionReason::None:
    return "none";
  case ConSanSiteDispositionReason::NonAccessInstruction:
    return "non_access_instruction";
  case ConSanSiteDispositionReason::UnsupportedMnemonic:
    return "unsupported_mnemonic";
  case ConSanSiteDispositionReason::InvalidInstructionSize:
    return "invalid_instruction_size";
  case ConSanSiteDispositionReason::InvalidAccessWidth:
    return "invalid_access_width";
  case ConSanSiteDispositionReason::MissingAddressOperand:
    return "missing_address_operand";
  case ConSanSiteDispositionReason::InstructionOutOfBounds:
    return "instruction_out_of_bounds";
  case ConSanSiteDispositionReason::UnsupportedFlatEncoding:
    return "unsupported_flat_encoding";
  case ConSanSiteDispositionReason::NonzeroFlatOffset:
    return "nonzero_flat_offset";
  case ConSanSiteDispositionReason::ReservedFlatAddressRegister:
    return "reserved_flat_address_register";
  case ConSanSiteDispositionReason::NonGroupAddressSpace:
    return "non_group_address_space";
  case ConSanSiteDispositionReason::FlatProvenancePolicyExcluded:
    return "flat_provenance_policy_excluded";
  case ConSanSiteDispositionReason::RuntimeKernelExcluded:
    return "runtime_kernel_excluded";
  case ConSanSiteDispositionReason::NonFlatAtomicAddress:
    return "non_flat_atomic_address";
  case ConSanSiteDispositionReason::MissingAtomicOperands:
    return "missing_atomic_operands";
  case ConSanSiteDispositionReason::MissingOrderingMetadata:
    return "missing_ordering_metadata";
  case ConSanSiteDispositionReason::UnqualifiedSyncSequence:
    return "unqualified_sync_sequence";
  case ConSanSiteDispositionReason::NoAtomicAcquireConsumer:
    return "no_atomic_acquire_consumer";
  case ConSanSiteDispositionReason::NoPrecedingSampledWindow:
    return "no_preceding_sampled_window";
  case ConSanSiteDispositionReason::UnsupportedMemoryRole:
    return "unsupported_memory_role";
  case ConSanSiteDispositionReason::MissingScope:
    return "missing_scope";
  case ConSanSiteDispositionReason::UnsupportedScope:
    return "unsupported_scope";
  case ConSanSiteDispositionReason::CompareExchangeOutcomeUnavailable:
    return "compare_exchange_outcome_unavailable";
  case ConSanSiteDispositionReason::UnsupportedAtomicOutcome:
    return "unsupported_atomic_outcome";
  case ConSanSiteDispositionReason::UnsupportedAtomicEncoding:
    return "unsupported_atomic_encoding";
  case ConSanSiteDispositionReason::UnsupportedMetadataEncoding:
    return "unsupported_metadata_encoding";
  case ConSanSiteDispositionReason::IneligibleFence:
    return "ineligible_fence";
  case ConSanSiteDispositionReason::MissingCommunicationEvent:
    return "missing_communication_event";
  case ConSanSiteDispositionReason::InvalidBarrierEncoding:
    return "invalid_barrier_encoding";
  }
  return "unknown";
}

std::optional<ConSanFlavor> parse_consan_flavor(std::string_view value) {
  if (ascii_iequals(value, "supercollider"))
    return ConSanFlavor::SuperCollider;
  if (ascii_iequals(value, "moi"))
    return ConSanFlavor::Moi;
  return std::nullopt;
}

std::optional<ConSanMoiEngine> parse_consan_moi_engine(std::string_view value) {
  if (ascii_iequals(value, "record_replay") || ascii_iequals(value, "record-replay") ||
      ascii_iequals(value, "context"))
    return ConSanMoiEngine::RecordReplay;
  if (ascii_iequals(value, "inline_shadow") || ascii_iequals(value, "inline-shadow"))
    return ConSanMoiEngine::InlineShadow;
  if (ascii_iequals(value, "sampled_watchpoint") || ascii_iequals(value, "sampled-watchpoint") ||
      ascii_iequals(value, "sampled"))
    return ConSanMoiEngine::Sampled;
  return std::nullopt;
}

} // namespace rocjitsu
