
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This retained `.inc` path is the private implementation of one compiled
// semantic-analysis component. It is not an include-order fragment.

#include "rocjitsu/code/patch/consan/consan_sync_analysis.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/analysis/kernel_scope.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_barrier_move_proof.h"
#include "rocjitsu/code/patch/consan/consan_cfg.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_inventory_diagnostics.h"
#include "rocjitsu/code/patch/consan/consan_perturbation.h"
#include "rocjitsu/code/patch/consan/consan_perturbation_policy.h"
#include "rocjitsu/code/patch/consan/consan_physical_site_alias.h"
#include "rocjitsu/code/patch/consan/consan_placement.h"
#include "rocjitsu/code/patch/consan/consan_program_analysis.h"
#include "rocjitsu/code/patch/consan/consan_runtime_kernel.h"
#include "rocjitsu/code/patch/consan/consan_semantic_classifiers.h"
#include "rocjitsu/code/patch/consan/consan_sync_event_index.h"
#include "rocjitsu/code/patch/consan/consan_sync_metadata.h"
#include "rocjitsu/code/patch/consan/targets/consan_fault_target_ops.h"
#include "rocjitsu/code/patch/consan/targets/consan_program_analysis_target_ops.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

[[nodiscard]] std::vector<size_t>
ordered_program_site_indices(std::span<const ConSanProgramSite> sites) {
  std::vector<size_t> order(sites.size());
  std::iota(order.begin(), order.end(), 0u);
  std::ranges::stable_sort(order, [&](size_t lhs, size_t rhs) {
    return std::pair{sites[lhs].container, sites[lhs].text_offset()} <
           std::pair{sites[rhs].container, sites[rhs].text_offset()};
  });
  return order;
}

[[nodiscard]] bool lds_address_fault_site_supported(const ConSanProgramSite &site,
                                                    rj_code_arch_t arch) {
  if (site.origin != ConSanAccessOrigin::NativeLds ||
      !consan_lds_address_fault_arch_supported(arch) ||
      !site.lowering.replay_guest_access.available() || !site.operands.address_vgpr ||
      site.size() != 2u * sizeof(uint32_t) ||
      (site.kind != ConSanLdsAccessKind::Read && site.kind != ConSanLdsAccessKind::Write)) {
    return false;
  }
  // A target that splits relocated two-address accesses has no single guest
  // instruction against which to prove this exact rewrite.
  const ConSanTargetProfile *target = consan_target_profile(arch);
  return target != nullptr &&
         (!target->requires_split_two_address_lds_relocation || !site.lowering.form ||
          site.lowering.form->kind != ConSanAccessLoweringFormKind::NativeTwoRange);
}

void append_decoded_fault_sites(std::string_view fingerprint, rj_code_arch_t arch,
                                const ProgramInventory &inventory,
                                std::vector<ConSanFaultSite> &sites) {
  const std::span<const ConSanProgramSite> program_sites = inventory.program_sites();
  std::map<std::pair<ConSanProgramContainerId, ConSanFaultSiteKind>, uint32_t> occurrences;
  for (size_t decoded_index : ordered_program_site_indices(program_sites)) {
    const ConSanProgramSite &decoded = program_sites[decoded_index];
    const ConSanProgramContainer *container = inventory.container(decoded.container);
    if (container == nullptr ||
        (container->is_kernel() && is_rocclr_runtime_kernel_name(container->name)))
      continue;

    ConSanFaultSite site;
    std::string_view site_kind;
    if (const ConSanBarrierSite *barrier = decoded.get_if<ConSanBarrierSite>()) {
      if (barrier->size != sizeof(uint32_t))
        continue;
      site.kind = ConSanFaultSiteKind::Barrier;
      site_kind = "barrier";
    } else if (const ConSanAtomicSite *atomic = decoded.get_if<ConSanAtomicSite>()) {
      if (classify_consan_atomic_fault_encoding(atomic->mnemonic, atomic->size, arch) ==
          AtomicFaultEncoding::Unsupported)
        continue;
      site.kind = ConSanFaultSiteKind::Atomic;
      site_kind = "atomic";
    } else if (decoded.get_if<ConSanOrdinaryMemorySite>() != nullptr) {
      site.kind = ConSanFaultSiteKind::OrdinaryMemory;
      site_kind = "ordinary-memory";
    } else {
      continue;
    }

    site.source_site = {static_cast<uint32_t>(decoded_index)};
    site.occurrence = occurrences[{decoded.container, site.kind}]++;
    site.identity = std::string(fingerprint) +
                    (container->is_kernel() ? "|kernel=" : "|function=") + container->name +
                    "|kind=" + std::string(site_kind) + "|pc=0x" +
                    consan_fixed_hex(decoded.text_offset(), 16) +
                    "|mnemonic=" + std::string(decoded.mnemonic_view()) +
                    "|occurrence=" + std::to_string(site.occurrence);
    sites.push_back(std::move(site));
  }
}

void append_lds_address_fault_sites(std::span<const uint8_t> bytes, std::string_view fingerprint,
                                    rj_code_arch_t arch, const ProgramInventory &inventory,
                                    std::span<const ConSanProgramSite> program_sites,
                                    std::vector<ConSanFaultSite> &sites) {
  std::optional<ConSanProgramContainerId> previous_container;
  uint32_t occurrence = 0;
  for (size_t source_index = 0; source_index < program_sites.size(); ++source_index) {
    const ConSanProgramSite &access = program_sites[source_index];
    const ConSanProgramContainer *container = inventory.container(access.container);
    if (container == nullptr || !lds_address_fault_site_supported(access, arch) ||
        (container->is_kernel() && is_rocclr_runtime_kernel_name(container->name)))
      continue;
    if (!previous_container || *previous_container != access.container) {
      previous_container = access.container;
      occurrence = 0;
    }
    ConSanFaultSite site;
    site.kind = ConSanFaultSiteKind::LdsAccess;
    site.source_site = {static_cast<uint32_t>(source_index)};
    site.occurrence = occurrence++;
    if (consan_arch_has_selectable_vgpr_bank(arch) &&
        access.decoded_file_offset() >= access.text_offset()) {
      site.selectable_vgpr_bank_mode = consan_selectable_vgpr_bank_mode_at(
          arch, bytes, container->text_file_offset, container->entry_text_offset,
          access.decoded_file_offset());
    }
    site.identity = std::string(fingerprint) +
                    (container->is_kernel() ? "|kernel=" : "|function=") + container->name +
                    "|kind=lds-access|pc=0x" + consan_fixed_hex(access.text_offset(), 16) +
                    "|mnemonic=" + std::string(access.mnemonic_view()) +
                    "|occurrence=" + std::to_string(site.occurrence);
    sites.push_back(std::move(site));
  }
}

void build_fault_site_inventory(std::span<const uint8_t> bytes, rj_code_arch_t arch,
                                ConSanProgramAnalysisResult &result) {
  const std::string fingerprint = make_consan_code_object_id(bytes).fingerprint;
  result.fault_sites.clear();
  append_decoded_fault_sites(fingerprint, arch, result.program_inventory, result.fault_sites);
  std::ranges::stable_sort(result.fault_sites, [&](const ConSanFaultSite &lhs,
                                                   const ConSanFaultSite &rhs) {
    const auto kind_order = [](ConSanFaultSiteKind kind) {
      return kind == ConSanFaultSiteKind::Barrier  ? 0u
             : kind == ConSanFaultSiteKind::Atomic ? 1u
                                                   : 2u;
    };
    const ConSanProgramSite *lhs_source = result.program_inventory.program_site(lhs.source_site);
    const ConSanProgramSite *rhs_source = result.program_inventory.program_site(rhs.source_site);
    return std::tuple{lhs_source->container, kind_order(lhs.kind), lhs_source->text_offset()} <
           std::tuple{rhs_source->container, kind_order(rhs.kind), rhs_source->text_offset()};
  });
  append_lds_address_fault_sites(bytes, fingerprint, arch, result.program_inventory,
                                 result.program_inventory.program_sites(), result.fault_sites);
}

[[nodiscard]] ConSanSyncAddressSource sync_address_source(ConSanAtomicAddressSpaceHint hint) {
  switch (hint) {
  case ConSanAtomicAddressSpaceHint::Lds:
    return ConSanSyncAddressSource::LdsVector;
  case ConSanAtomicAddressSpaceHint::FlatGroup:
  case ConSanAtomicAddressSpaceHint::FlatPrivate:
  case ConSanAtomicAddressSpaceHint::FlatMaybeGroup:
  case ConSanAtomicAddressSpaceHint::FlatMaybePrivate:
  case ConSanAtomicAddressSpaceHint::FlatGlobal:
  case ConSanAtomicAddressSpaceHint::FlatUnknown:
    return ConSanSyncAddressSource::FlatVector;
  case ConSanAtomicAddressSpaceHint::Global:
    return ConSanSyncAddressSource::GlobalScalarVector;
  case ConSanAtomicAddressSpaceHint::Scratch:
    return ConSanSyncAddressSource::ScratchVector;
  case ConSanAtomicAddressSpaceHint::Buffer:
    return ConSanSyncAddressSource::BufferResource;
  case ConSanAtomicAddressSpaceHint::Scalar:
  case ConSanAtomicAddressSpaceHint::Unknown:
    return ConSanSyncAddressSource::Unknown;
  }
  return ConSanSyncAddressSource::Unknown;
}

void classify_atomic_sync_confidence(const ConSanAtomicSite &site, ConSanSyncEvent &event) {
  switch (site.address_space_hint) {
  case ConSanAtomicAddressSpaceHint::FlatMaybeGroup:
  case ConSanAtomicAddressSpaceHint::FlatMaybePrivate:
    event.confidence = ConSanSemanticConfidence::Ambiguous;
    event.confidence_reason =
        "flat atomic address space is not statically distinguishable; ordering requires "
        "sequence association";
    return;
  case ConSanAtomicAddressSpaceHint::FlatUnknown:
    event.confidence = ConSanSemanticConfidence::Unsupported;
    event.confidence_reason =
        "flat atomic address space has no usable provenance; ordering requires sequence "
        "association";
    return;
  case ConSanAtomicAddressSpaceHint::Scalar:
  case ConSanAtomicAddressSpaceHint::Unknown:
    event.confidence = ConSanSemanticConfidence::Unsupported;
    event.confidence_reason =
        "atomic address encoding is not represented; ordering requires sequence association";
    return;
  default:
    event.confidence = ConSanSemanticConfidence::Conservative;
    event.confidence_reason =
        "instruction shape is decoded; ordering requires surrounding fence/cache sequence "
        "association";
    return;
  }
}

void build_sync_events(const ConSanCodeObjectId &code_object_id, rj_code_arch_t arch,
                       const ProgramInventory &inventory, std::vector<ConSanSyncEvent> &events) {
  const std::string_view fingerprint = code_object_id.fingerprint;
  const std::span<const ConSanProgramSite> program_sites = inventory.program_sites();
  const auto finish_event = [&](ConSanSyncEvent &event, std::string_view kind, uint32_t occurrence,
                                ConSanProgramSiteId source_site) {
    const ConSanProgramSite &source = program_sites[source_site.ordinal];
    const ConSanProgramContainer *container = inventory.container(source.container);
    if (container == nullptr)
      return;
    const bool in_kernel = container->is_kernel();
    const std::string_view container_kind = in_kernel ? "kernel" : "function";
    event.semantic_id.physical.code_object = code_object_id;
    event.semantic_id.physical.original_text_offset = source.text_offset();
    event.semantic_id.domain = ConSanSemanticSiteDomain::SynchronizationEvent;
    event.source_site = source_site;
    event.identity = std::string(fingerprint) + "|" + std::string(container_kind) + "=" +
                     container->name + "|event=" + std::string(kind) + "|pc=0x" +
                     consan_fixed_hex(event.text_offset(), 16) +
                     "|mnemonic=" + std::string(source.mnemonic_view()) +
                     "|occurrence=" + std::to_string(occurrence);
    events.push_back(std::move(event));
  };

  std::map<std::pair<ConSanProgramContainerId, size_t>, uint32_t> occurrences;
  for (size_t decoded_index : ordered_program_site_indices(program_sites)) {
    const ConSanProgramSite &decoded = program_sites[decoded_index];
    const ConSanProgramContainer *container = inventory.container(decoded.container);
    if (container == nullptr ||
        (container->is_kernel() && is_rocclr_runtime_kernel_name(container->name)))
      continue;

    ConSanSyncEvent event;
    std::string_view event_kind;
    size_t occurrence_kind = 0;
    if (const ConSanBarrierSite *site = decoded.get_if<ConSanBarrierSite>()) {
      event.kind = ConSanSyncKind::Barrier;
      event.address_source = ConSanSyncAddressSource::NotApplicable;
      event.rmw_outcome = ConSanSyncRmwOutcome::NotApplicable;
      const bool lifecycle_operation = site->operation == ConSanBarrierSite::Operation::Init ||
                                       site->operation == ConSanBarrierSite::Operation::Join ||
                                       site->operation == ConSanBarrierSite::Operation::Leave ||
                                       site->operation == ConSanBarrierSite::Operation::Wakeup ||
                                       site->operation == ConSanBarrierSite::Operation::StateQuery;
      if (lifecycle_operation) {
        event.confidence = ConSanSemanticConfidence::Unsupported;
        event.confidence_reason =
            "barrier lifecycle operation is inventoried but participant semantics are unavailable";
      } else if ((site->operand_source == ConSanBarrierSite::OperandSource::Immediate ||
                  site->operand_source == ConSanBarrierSite::OperandSource::Literal32) &&
                 site->barrier_id && site->scope != ConSanBarrierSite::Scope::Unknown) {
        event.confidence = ConSanSemanticConfidence::Conservative;
        event.confidence_reason =
            "barrier ID and scope are decoded but participant set is unavailable";
      } else if (site->operand_source == ConSanBarrierSite::OperandSource::DynamicM0) {
        event.confidence = ConSanSemanticConfidence::Unsupported;
        event.confidence_reason = "dynamic M0 barrier ID and participant state are unavailable";
      } else {
        event.confidence = ConSanSemanticConfidence::Unsupported;
        event.confidence_reason = "barrier ID or scope is not a supported decoded form";
      }
      switch (site->operation) {
      case ConSanBarrierSite::Operation::Signal:
        event.operation = ConSanSyncOperation::BarrierSignal;
        event.memory_role = ConSanSyncMemoryRole::Release;
        event.memory_role_confidence = ConSanSemanticConfidence::Conservative;
        break;
      case ConSanBarrierSite::Operation::Wait:
        event.operation = ConSanSyncOperation::BarrierWait;
        event.memory_role = ConSanSyncMemoryRole::Acquire;
        event.memory_role_confidence = ConSanSemanticConfidence::Conservative;
        break;
      case ConSanBarrierSite::Operation::Full:
        event.operation = ConSanSyncOperation::BarrierFull;
        event.memory_role = ConSanSyncMemoryRole::AcquireRelease;
        event.memory_role_confidence = ConSanSemanticConfidence::Conservative;
        break;
      case ConSanBarrierSite::Operation::Init:
        event.operation = ConSanSyncOperation::BarrierInit;
        break;
      case ConSanBarrierSite::Operation::Join:
        event.operation = ConSanSyncOperation::BarrierJoin;
        break;
      case ConSanBarrierSite::Operation::Leave:
        event.operation = ConSanSyncOperation::BarrierLeave;
        break;
      case ConSanBarrierSite::Operation::Wakeup:
        event.operation = ConSanSyncOperation::BarrierWakeup;
        break;
      case ConSanBarrierSite::Operation::StateQuery:
        event.operation = ConSanSyncOperation::BarrierStateQuery;
        break;
      case ConSanBarrierSite::Operation::Unknown:
        event.operation = ConSanSyncOperation::Unknown;
        break;
      }
      if (lifecycle_operation || site->operation == ConSanBarrierSite::Operation::Unknown) {
        event.memory_role = ConSanSyncMemoryRole::Unknown;
        event.memory_role_confidence = ConSanSemanticConfidence::Unsupported;
      }
      event_kind = "barrier";
      occurrence_kind = 0;
    } else if (decoded.get_if<ConSanFenceSite>() != nullptr) {
      event.kind = ConSanSyncKind::Fence;
      event.operation = ConSanSyncOperation::Fence;
      event.address_source = ConSanSyncAddressSource::NotApplicable;
      event.memory_role = ConSanSyncMemoryRole::Unknown;
      event.rmw_outcome = ConSanSyncRmwOutcome::NotApplicable;
      event.confidence = ConSanSemanticConfidence::Conservative;
      event.confidence_reason =
          "cache operation is decoded but its high-level memory role requires "
          "sequence association";
      event_kind = "fence";
      occurrence_kind = 1;
    } else if (const ConSanOrdinaryMemorySite *site = decoded.get_if<ConSanOrdinaryMemorySite>()) {
      if (site->support_reason != ConSanOrdinaryMemorySupportReason::Supported &&
          site->support_reason != ConSanOrdinaryMemorySupportReason::SupportedSynchronizationOnly)
        continue;
      event.kind = ConSanSyncKind::OrdinaryMemory;
      event.operation = site->operation == ConSanOrdinaryMemoryOperation::Load
                            ? ConSanSyncOperation::OrdinaryLoad
                            : ConSanSyncOperation::OrdinaryStore;
      event.address_source =
          site->mnemonic.starts_with("global_")
              ? ConSanSyncAddressSource::GlobalScalarVector
              : (site->mnemonic.starts_with("buffer_") ? ConSanSyncAddressSource::BufferResource
                                                       : ConSanSyncAddressSource::FlatVector);
      event.memory_role = ConSanSyncMemoryRole::Unknown;
      event.rmw_outcome = ConSanSyncRmwOutcome::NotApplicable;
      event.confidence = ConSanSemanticConfidence::Conservative;
      event.confidence_reason = "bounded ordinary-memory encoding is decoded; ordering requires "
                                "same-block cache-sequence association";
      event.scope = site->scope;
      event_kind = "ordinary-memory";
      occurrence_kind = 2;
    } else if (const ConSanAtomicSite *site = decoded.get_if<ConSanAtomicSite>()) {
      event.kind = ConSanSyncKind::Atomic;
      const bool is_compare_exchange = consan_atomic_is_compare_exchange(*site);
      event.operation = is_compare_exchange ? ConSanSyncOperation::AtomicCompareExchange
                                            : ConSanSyncOperation::AtomicRmw;
      event.address_source = sync_address_source(site->address_space_hint);
      event.memory_role = ConSanSyncMemoryRole::Unknown;
      if (is_compare_exchange) {
        event.rmw_outcome = ConSanSyncRmwOutcome::CompareExchange;
      } else if (site->returns_old_value) {
        event.rmw_outcome = *site->returns_old_value ? ConSanSyncRmwOutcome::ReturnsOldValue
                                                     : ConSanSyncRmwOutcome::NoReturn;
      } else {
        event.rmw_outcome = ConSanSyncRmwOutcome::Unknown;
      }
      classify_atomic_sync_confidence(*site, event);
      const bool implicit_workgroup_scope =
          site->address_space_hint == ConSanAtomicAddressSpaceHint::FlatGroup ||
          (site->mnemonic.starts_with("ds_") &&
           consan_arch_supports_capability_form(arch, ConSanCapabilityForm::OrderedLdsAtomic));
      event.scope =
          implicit_workgroup_scope ? std::optional{ConSanMemoryScope::Workgroup} : site->scope;
      event_kind = "atomic";
      occurrence_kind = 3;
    } else {
      continue;
    }
    auto &occurrence = occurrences[{decoded.container, occurrence_kind}];
    finish_event(event, event_kind, occurrence++, {static_cast<uint32_t>(decoded_index)});
  }
}

void build_sync_event_inventory(std::span<const uint8_t> bytes,
                                SynchronizationInventoryBuildView inventory,
                                const ProgramInventory &program_inventory, rj_code_arch_t arch) {
  const ConSanCodeObjectId code_object_id = make_consan_code_object_id(bytes);
  inventory.sync_events.clear();
  build_sync_events(code_object_id, arch, program_inventory, inventory.sync_events);

  std::ranges::stable_sort(
      inventory.sync_events, [&](const ConSanSyncEvent &lhs, const ConSanSyncEvent &rhs) {
        const ConSanProgramContainerId lhs_container =
            program_inventory.program_sites()[lhs.source_site.ordinal].container;
        const ConSanProgramContainerId rhs_container =
            program_inventory.program_sites()[rhs.source_site.ordinal].container;
        return std::pair{lhs_container, lhs.text_offset()} <
               std::pair{rhs_container, rhs.text_offset()};
      });
}

[[nodiscard]] bool sync_event_semantics_equal(const ProgramInventory &inventory,
                                              const ConSanSyncEvent &lhs,
                                              const ConSanSyncEvent &rhs) {
  const ConSanProgramSite *lhs_source = inventory.program_site(lhs.source_site);
  const ConSanProgramSite *rhs_source = inventory.program_site(rhs.source_site);
  const ConSanProgramContainer *lhs_container =
      lhs_source == nullptr ? nullptr : inventory.container(lhs_source->container);
  const ConSanProgramContainer *rhs_container =
      rhs_source == nullptr ? nullptr : inventory.container(rhs_source->container);
  if (lhs_source == nullptr || rhs_source == nullptr || lhs_container == nullptr ||
      rhs_container == nullptr || lhs_container->kind != rhs_container->kind ||
      !lhs_source->same_payload(*rhs_source))
    return false;
  return lhs.kind == rhs.kind && static_cast<const ConSanSyncSemantics &>(lhs) == rhs;
}

[[nodiscard]] bool
canonicalize_sync_events_by_physical_site(SynchronizationInventoryBuildView inventory,
                                          ConSanProgramAnalysisResult &result) {
  const std::span<const ConSanProgramSite> program_sites = inventory.program_sites;
  const auto container_entry = [&](const ConSanSyncEvent &event) -> std::optional<uint64_t> {
    const ConSanProgramSite *source = consan_program_site(program_sites, event);
    const ConSanProgramContainer *container =
        source == nullptr ? nullptr : result.program_inventory.container(source->container);
    return container == nullptr ? std::nullopt
                                : std::optional<uint64_t>(container->entry_text_offset);
  };

  return consan_detail::canonicalize_physical_site_aliases(
      inventory.sync_events, result.errors, "ConSan synchronization site",
      [&](const ConSanSyncEvent &event) {
        const ConSanProgramSite *source = consan_program_site(program_sites, event);
        return source == nullptr ? uint64_t{0} : source->decoded_file_offset();
      },
      [&](const ConSanSyncEvent &event) -> std::string_view {
        const ConSanProgramSite *source = consan_program_site(program_sites, event);
        const ConSanProgramContainer *container =
            source == nullptr ? nullptr : result.program_inventory.container(source->container);
        return container == nullptr ? std::string_view{} : std::string_view(container->name);
      },
      [&](const ConSanSyncEvent &lhs, const ConSanSyncEvent &rhs) {
        return container_entry(lhs) == container_entry(rhs) &&
               sync_event_semantics_equal(result.program_inventory, lhs, rhs);
      },
      [](ConSanSyncEvent &, const ConSanSyncEvent &) {});
}

[[nodiscard]] std::vector<std::unique_ptr<BasicBlock>>
build_sync_basic_blocks(const AmdGpuCodeObject &code_object, Decoder &decoder, rj_code_arch_t arch,
                        const ProgramInventory &program_inventory) {
  const consan_detail::ConSanCfgBuildInputs cfg =
      consan_detail::build_consan_cfg_inputs(code_object, program_inventory.containers());
  return BasicBlock::build(code_object, decoder, arch, cfg.leaders, cfg.code_ranges);
}

[[nodiscard]] bool block_is_in_cycle(const BasicBlock *start) {
  if (start == nullptr)
    return false;
  std::vector<const BasicBlock *> pending(start->successors().begin(), start->successors().end());
  std::unordered_set<const BasicBlock *> visited;
  while (!pending.empty()) {
    const BasicBlock *block = pending.back();
    pending.pop_back();
    if (block == start)
      return true;
    if (block == nullptr || !visited.insert(block).second)
      continue;
    pending.insert(pending.end(), block->successors().begin(), block->successors().end());
  }
  return false;
}

[[nodiscard]] std::optional<uint64_t> scalar_clause_covering_instruction(BasicBlock &block,
                                                                         uint64_t text_offset) {
  uint32_t clause_remaining = 0;
  std::optional<uint64_t> clause_offset;
  for (const Instruction &instruction : block.instructions()) {
    const bool inside_clause = clause_remaining > 0;
    if (clause_remaining > 0)
      --clause_remaining;
    if (instruction.src_loc() == text_offset)
      return inside_clause ? clause_offset : std::nullopt;
    if (is_s_clause(instruction)) {
      clause_remaining = s_clause_following_instruction_count(instruction);
      clause_offset = instruction.src_loc();
    } else if (clause_remaining == 0) {
      clause_offset.reset();
    }
  }
  return std::nullopt;
}

void build_singleton_sync_sequences(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                                    SynchronizationInventoryBuildView inventory) {
  const BlockOffsetIndex block_offset_index = build_block_offset_index(blocks);
  std::unordered_map<const BasicBlock *, uint32_t> block_indices;
  block_indices.reserve(blocks.size());
  for (size_t index = 0; index < blocks.size(); ++index) {
    if (blocks[index] != nullptr)
      block_indices.emplace(blocks[index].get(), static_cast<uint32_t>(index));
  }
  std::vector<std::optional<bool>> cyclic_block_cache(blocks.size());
  inventory.sync_sequences.clear();
  inventory.sync_sequences.reserve(inventory.sync_events.size());
  for (size_t event_ordinal = 0; event_ordinal < inventory.sync_events.size(); ++event_ordinal) {
    const ConSanSyncEvent &event = inventory.sync_events[event_ordinal];
    if (!event.source_site.valid() || event.source_site.ordinal >= inventory.program_sites.size())
      continue;
    const ConSanProgramSite &source = inventory.program_sites[event.source_site.ordinal];
    ConSanSyncSequence sequence;
    static_cast<ConSanSyncSemantics &>(sequence) = event;
    sequence.identity = event.identity + "|sequence=singleton";
    sequence.begin_text_offset = event.text_offset();
    sequence.end_text_offset = event.text_offset() + source.size();
    sequence.member_event_ids.push_back({static_cast<uint32_t>(event_ordinal)});
    if (const ConSanBarrierSite *site = source.get_if<ConSanBarrierSite>()) {
      sequence.barrier_id = site->barrier_id;
      sequence.barrier_operand_source = site->operand_source;
      sequence.barrier_scope = site->scope;
      sequence.participant_count = site->participant_count;
      sequence.participant_mask = site->participant_mask;
    }

    BasicBlock *block = block_for_offset(block_offset_index, event.text_offset());
    if (block != nullptr) {
      const auto instruction =
          std::ranges::find_if(block->instructions(), [&](const Instruction &candidate) {
            return candidate.src_loc() == event.text_offset() &&
                   static_cast<uint32_t>(candidate.size()) == source.size();
          });
      if (instruction != block->instructions().end()) {
        const auto block_index = block_indices.find(block);
        if (block_index != block_indices.end()) {
          sequence.basic_block_index = block_index->second;
          std::optional<bool> &in_cycle = cyclic_block_cache[block_index->second];
          if (!in_cycle)
            in_cycle = block_is_in_cycle(block);
          sequence.in_cyclic_cfg_component = *in_cycle;
          sequence.scalar_clause_text_offset =
              scalar_clause_covering_instruction(*block, event.text_offset());
          sequence.inside_scalar_clause = sequence.scalar_clause_text_offset.has_value();
        }
      }
    }
    if (!sequence.basic_block_index) {
      sequence.confidence = combine_consan_sync_confidence(sequence.confidence,
                                                           ConSanSemanticConfidence::Unsupported);
      if (!sequence.confidence_reason.empty())
        sequence.confidence_reason += "; ";
      sequence.confidence_reason += "event is not an exact instruction in a decoded basic block";
    }
    inventory.sync_sequences.push_back(std::move(sequence));
  }
}

[[nodiscard]] bool compatible_barrier_participants(const ConSanSyncSequence &signal,
                                                   const ConSanSyncSequence &wait) {
  const auto is_static_id = [](ConSanBarrierSite::OperandSource source) {
    return source == ConSanBarrierSite::OperandSource::Immediate ||
           source == ConSanBarrierSite::OperandSource::Literal32;
  };
  if (!is_static_id(signal.barrier_operand_source) || !is_static_id(wait.barrier_operand_source) ||
      !signal.barrier_id || !wait.barrier_id || signal.barrier_id != wait.barrier_id ||
      signal.barrier_scope == ConSanBarrierSite::Scope::Unknown ||
      wait.barrier_scope == ConSanBarrierSite::Scope::Unknown ||
      signal.barrier_scope != wait.barrier_scope) {
    return false;
  }
  if (signal.participant_count && wait.participant_count &&
      signal.participant_count != wait.participant_count) {
    return false;
  }
  if (signal.participant_mask && wait.participant_mask &&
      signal.participant_mask != wait.participant_mask) {
    return false;
  }
  return true;
}

[[nodiscard]] bool is_bounded_barrier_bookkeeping(const Instruction &instruction) {
  const std::string_view mnemonic = instruction.mnemonic();
  // These are exactly the four pointer/index bookkeeping instructions in the
  // retained RDNA4 hip-moi barrier handoff. Keep this list closed: accepting
  // arbitrary move/add families would include special-register and EXEC
  // updates that can change which lanes execute the wait.
  return mnemonic == "v_add_nc_u32_e32" || mnemonic == "s_add_nc_u64" || mnemonic == "s_mov_b32";
}

[[nodiscard]] bool compatible_barrier_pair_identity(const SynchronizationInventoryView &events,
                                                    const ConSanSyncSequence &signal,
                                                    const ConSanSyncSequence &wait) {
  return signal.kind == ConSanSyncKind::Barrier &&
         signal.operation == ConSanSyncOperation::BarrierSignal &&
         wait.kind == ConSanSyncKind::Barrier &&
         wait.operation == ConSanSyncOperation::BarrierWait &&
         events.same_container(signal, wait) && signal.basic_block_index &&
         wait.basic_block_index && !signal.inside_scalar_clause && !wait.inside_scalar_clause &&
         compatible_barrier_participants(signal, wait);
}

[[nodiscard]] bool compatible_barrier_pair_metadata(const SynchronizationInventoryView &events,
                                                    const ConSanSyncSequence &signal,
                                                    const ConSanSyncSequence &wait) {
  return compatible_barrier_pair_identity(events, signal, wait) &&
         signal.basic_block_index == wait.basic_block_index;
}

[[nodiscard]] bool
is_safe_bounded_barrier_pair(const SynchronizationInventoryView &events,
                             const ConSanSyncSequence &signal, const ConSanSyncSequence &wait,
                             const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                             bool allow_extended_pair) {
  constexpr uint32_t kMaximumBookkeepingInstructions = 4u;
  constexpr uint64_t kMaximumBookkeepingBytes = 24u;
  // Exact whole-pair fault injection and Sampled sequence metadata may span
  // the useful work between a signal and its completing wait. Keep that
  // relaxation local to those consumers: other synchronization semantics
  // retain the closed bookkeeping pattern above.
  // Extended consumers already require the signal and wait to be consecutive
  // synchronization events in one exact decoded basic block, with the same
  // static ID/scope and no call edge or intervening barrier.  Those structural
  // checks prove the pair independently of how much straight-line useful work
  // the compiler schedules between its members.  Do not impose a corpus-sized
  // byte/instruction cap on that otherwise exact shape.
  const uint32_t maximum_instructions =
      allow_extended_pair ? std::numeric_limits<uint32_t>::max() : kMaximumBookkeepingInstructions;
  const uint64_t maximum_bytes =
      allow_extended_pair ? std::numeric_limits<uint64_t>::max() : kMaximumBookkeepingBytes;
  if (!compatible_barrier_pair_metadata(events, signal, wait) ||
      signal.end_text_offset > wait.begin_text_offset ||
      wait.begin_text_offset - signal.end_text_offset > maximum_bytes) {
    return false;
  }
  const size_t block_index = *signal.basic_block_index;
  if (block_index >= blocks.size() || blocks[block_index] == nullptr ||
      !blocks[block_index]->call_edges().empty()) {
    return false;
  }
  uint64_t expected_offset = signal.end_text_offset;
  uint32_t instruction_count = 0;
  for (const Instruction &instruction : blocks[block_index]->instructions()) {
    const uint64_t offset = instruction.src_loc();
    if (offset < signal.end_text_offset)
      continue;
    if (offset >= wait.begin_text_offset)
      break;
    if (offset != expected_offset || instruction.size() <= 0 ||
        ++instruction_count > maximum_instructions || is_barrier_instruction(instruction) ||
        (!allow_extended_pair && !is_bounded_barrier_bookkeeping(instruction))) {
      return false;
    }
    expected_offset += static_cast<uint64_t>(instruction.size());
  }
  return expected_offset == wait.begin_text_offset;
}

[[nodiscard]] bool
is_safe_cluster_triangle_barrier_pair(const ConSanSyncSequence &signal,
                                      const ConSanSyncSequence &wait,
                                      const SynchronizationInventoryView &events,
                                      const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  // Cluster arrive lowers to a conditional diamond: one path executes the
  // signal and the other bypasses it, then both paths join at the completing
  // wait. Admit only that exact local CFG shape. This is deliberately narrower
  // than general cross-block pairing: the signal block must be the complete
  // fallthrough arm, and the join may contain only a short straight-line prefix
  // before the wait.
  constexpr uint32_t kMaximumJoinPrefixInstructions = 4u;
  constexpr uint64_t kMaximumJoinPrefixBytes = 24u;
  if (!compatible_barrier_pair_identity(events, signal, wait) ||
      signal.barrier_scope != ConSanBarrierSite::Scope::Cluster ||
      signal.basic_block_index == wait.basic_block_index ||
      signal.end_text_offset > wait.begin_text_offset ||
      wait.begin_text_offset - signal.end_text_offset > kMaximumJoinPrefixBytes) {
    return false;
  }

  const size_t signal_index = *signal.basic_block_index;
  const size_t wait_index = *wait.basic_block_index;
  if (signal_index >= blocks.size() || wait_index >= blocks.size() ||
      blocks[signal_index] == nullptr || blocks[wait_index] == nullptr) {
    return false;
  }
  BasicBlock *signal_block = blocks[signal_index].get();
  BasicBlock *wait_block = blocks[wait_index].get();
  if (!signal_block->call_edges().empty() || !wait_block->call_edges().empty() ||
      signal_block->start_offset() != signal.begin_text_offset ||
      signal_block->end_offset() != signal.end_text_offset ||
      wait_block->start_offset() > wait.begin_text_offset ||
      signal_block->predecessors().size() != 1u || signal_block->successors().size() != 1u ||
      signal_block->successors().front() != wait_block) {
    return false;
  }

  BasicBlock *guard = signal_block->predecessors().front();
  if (guard == nullptr || !guard->call_edges().empty() || guard->successors().size() != 2u ||
      std::ranges::find(guard->successors(), signal_block) == guard->successors().end() ||
      std::ranges::find(guard->successors(), wait_block) == guard->successors().end() ||
      wait_block->predecessors().size() != 2u ||
      std::ranges::find(wait_block->predecessors(), guard) == wait_block->predecessors().end() ||
      std::ranges::find(wait_block->predecessors(), signal_block) ==
          wait_block->predecessors().end()) {
    return false;
  }
  const Instruction *guard_terminator = guard->terminator();
  if (guard_terminator == nullptr || (guard_terminator->flags() & COND_BRANCH) == 0u) {
    return false;
  }

  uint64_t expected_offset = wait_block->start_offset();
  uint32_t instruction_count = 0;
  for (const Instruction &instruction : wait_block->instructions()) {
    const uint64_t offset = instruction.src_loc();
    if (offset >= wait.begin_text_offset)
      break;
    if (offset != expected_offset || instruction.size() <= 0 ||
        ++instruction_count > kMaximumJoinPrefixInstructions ||
        (instruction.flags() &
         (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR)) != 0u ||
        is_barrier_instruction(instruction)) {
      return false;
    }
    expected_offset += static_cast<uint64_t>(instruction.size());
  }
  return expected_offset == wait.begin_text_offset;
}

void associate_barrier_sync_sequences(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                                      bool allow_extended_pairs,
                                      SynchronizationInventoryBuildView inventory) {
  std::vector<ConSanSyncSequence> &sync_sequences = inventory.sync_sequences;
  const SynchronizationInventoryView events = inventory.view();
  const auto has_unresolved_preceding_signal = [&](size_t signal_index) {
    for (size_t cursor = signal_index; cursor > 0; --cursor) {
      const ConSanSyncSequence &prior = sync_sequences[cursor - 1u];
      if (prior.kind != ConSanSyncKind::Barrier)
        continue;
      if (prior.operation == ConSanSyncOperation::BarrierSignal) {
        const ConSanSyncSequence &signal = sync_sequences[signal_index];
        return events.same_container(prior, signal) &&
               prior.basic_block_index == signal.basic_block_index;
      }
      // Any non-signal barrier terminates this ambiguity search. In
      // particular, a proven init/join lifecycle may immediately precede its
      // completing signal/wait pair.
      return false;
    }
    return false;
  };
  std::vector<bool> unresolved_preceding_signal(sync_sequences.size(), false);
  for (size_t index = 0; index < sync_sequences.size(); ++index)
    unresolved_preceding_signal[index] = has_unresolved_preceding_signal(index);
  std::vector<ConSanSyncSequence> associated;
  associated.reserve(sync_sequences.size());
  for (size_t i = 0; i < sync_sequences.size();) {
    if (i + 1 < sync_sequences.size() &&
        (is_safe_bounded_barrier_pair(events, sync_sequences[i], sync_sequences[i + 1], blocks,
                                      allow_extended_pairs) ||
         is_safe_cluster_triangle_barrier_pair(sync_sequences[i], sync_sequences[i + 1], events,
                                               blocks)) &&
        !unresolved_preceding_signal[i]) {
      ConSanSyncSequence sequence = std::move(sync_sequences[i]);
      ConSanSyncSequence &wait = sync_sequences[i + 1];
      sequence.operation = ConSanSyncOperation::BarrierFull;
      sequence.memory_role = ConSanSyncMemoryRole::AcquireRelease;
      sequence.confidence = combine_consan_sync_confidence(sequence.confidence, wait.confidence);
      sequence.memory_role_confidence = combine_consan_sync_confidence(
          sequence.memory_role_confidence, wait.memory_role_confidence);
      const bool cross_block = sequence.basic_block_index != wait.basic_block_index;
      sequence.confidence_reason =
          cross_block            ? "bounded cluster barrier signal/wait pair in a "
                                   "single conditional CFG triangle with matching decoded "
                                   "ID and scope; participant set is unavailable"
          : allow_extended_pairs ? "bounded same-owner same-block barrier signal/wait pair "
                                   "with matching decoded ID and scope, no intervening "
                                   "barrier, and bounded intervening instruction run; "
                                   "participant set is unavailable"
                                 : "bounded same-block barrier signal/wait pair with "
                                   "matching decoded ID and scope and only the retained "
                                   "RDNA4 register bookkeeping pattern; participant set "
                                   "is unavailable";
      sequence.identity += "|associated-with=" + wait.identity;
      sequence.end_text_offset = wait.end_text_offset;
      sequence.in_cyclic_cfg_component |= wait.in_cyclic_cfg_component;
      sequence.inside_scalar_clause |= wait.inside_scalar_clause;
      sequence.member_event_ids.insert(sequence.member_event_ids.end(),
                                       wait.member_event_ids.begin(), wait.member_event_ids.end());
      if (!sequence.participant_count)
        sequence.participant_count = wait.participant_count;
      if (!sequence.participant_mask)
        sequence.participant_mask = wait.participant_mask;
      associated.push_back(std::move(sequence));
      i += 2;
      continue;
    }

    ConSanSyncSequence sequence = std::move(sync_sequences[i]);
    if (sequence.kind == ConSanSyncKind::Barrier &&
        (sequence.operation == ConSanSyncOperation::BarrierSignal ||
         sequence.operation == ConSanSyncOperation::BarrierWait)) {
      sequence.confidence =
          combine_consan_sync_confidence(sequence.confidence, ConSanSemanticConfidence::Ambiguous);
      sequence.confidence_reason =
          "barrier signal/wait component has no safe bounded same-block partner";
    }
    associated.push_back(std::move(sequence));
    ++i;
  }
  sync_sequences = std::move(associated);
}

[[nodiscard]] const ConSanSyncEvent *
first_sequence_event(const SynchronizationInventoryView &events,
                     const ConSanSyncSequence &sequence);

void build_barrier_lifecycle_group_inventory(SynchronizationInventoryBuildView inventory,
                                             const SynchronizationInventoryView &events) {
  std::vector<ConSanSyncSequence> &sync_sequences = inventory.sync_sequences;
  std::vector<ConSanBarrierLifecycleGroup> &groups = inventory.barrier_lifecycle_groups;
  groups.clear();
  for (size_t begin = 0; begin < sync_sequences.size(); ++begin) {
    const ConSanSyncSequence &init = sync_sequences[begin];
    if (init.operation != ConSanSyncOperation::BarrierInit)
      continue;

    ConSanBarrierLifecycleGroup group;
    group.member_event_ids = init.member_event_ids;
    const std::optional<uint32_t> basic_block_index = init.basic_block_index;
    uint64_t end_text_offset = init.end_text_offset;

    const auto append_member = [&](const ConSanSyncSequence &sequence) {
      end_text_offset = sequence.end_text_offset;
      group.member_event_ids.insert(group.member_event_ids.end(), sequence.member_event_ids.begin(),
                                    sequence.member_event_ids.end());
    };
    const auto is_contiguous = [&](const ConSanSyncSequence &sequence) {
      return basic_block_index && sequence.basic_block_index == basic_block_index &&
             events.same_container(sequence, init) && sequence.begin_text_offset == end_text_offset;
    };
    const auto has_matching_static_id = [&](const ConSanSyncSequence &sequence) {
      return init.barrier_id && sequence.barrier_id == init.barrier_id &&
             init.barrier_scope != ConSanBarrierSite::Scope::Unknown &&
             sequence.barrier_scope == init.barrier_scope &&
             (sequence.barrier_operand_source == ConSanBarrierSite::OperandSource::Immediate ||
              sequence.barrier_operand_source == ConSanBarrierSite::OperandSource::Literal32);
    };

    if (!basic_block_index || !init.barrier_id ||
        init.barrier_scope == ConSanBarrierSite::Scope::Unknown ||
        (init.barrier_operand_source != ConSanBarrierSite::OperandSource::Immediate &&
         init.barrier_operand_source != ConSanBarrierSite::OperandSource::StaticM0Literal32 &&
         init.barrier_operand_source != ConSanBarrierSite::OperandSource::Literal32)) {
      group.issue = ConSanBarrierLifecycleIssue::InitMissingStaticIdOrScope;
      groups.push_back(std::move(group));
      continue;
    }

    size_t cursor = begin + 1u;
    size_t join_count = 0;
    bool mismatch = false;
    while (cursor < sync_sequences.size() &&
           sync_sequences[cursor].operation == ConSanSyncOperation::BarrierJoin) {
      const ConSanSyncSequence &join = sync_sequences[cursor];
      if (!is_contiguous(join)) {
        group.issue = ConSanBarrierLifecycleIssue::NonContiguousRun;
        mismatch = true;
        break;
      }
      append_member(join);
      if (!has_matching_static_id(join)) {
        group.issue = ConSanBarrierLifecycleIssue::MemberIdOrScopeMismatch;
        mismatch = true;
        break;
      }
      ++join_count;
      ++cursor;
    }
    if (mismatch) {
      groups.push_back(std::move(group));
      continue;
    }
    if (join_count == 0) {
      group.issue = ConSanBarrierLifecycleIssue::MissingJoin;
      groups.push_back(std::move(group));
      continue;
    }
    if (cursor >= sync_sequences.size() ||
        sync_sequences[cursor].operation != ConSanSyncOperation::BarrierFull ||
        !is_contiguous(sync_sequences[cursor])) {
      group.issue = ConSanBarrierLifecycleIssue::MissingCompletingBarrier;
      groups.push_back(std::move(group));
      continue;
    }
    const ConSanSyncSequence &barrier = sync_sequences[cursor++];
    append_member(barrier);
    if (!has_matching_static_id(barrier)) {
      group.issue = ConSanBarrierLifecycleIssue::MemberIdOrScopeMismatch;
      groups.push_back(std::move(group));
      continue;
    }
    if (cursor >= sync_sequences.size() ||
        sync_sequences[cursor].operation != ConSanSyncOperation::BarrierLeave ||
        !is_contiguous(sync_sequences[cursor])) {
      group.issue = ConSanBarrierLifecycleIssue::MissingLeave;
      groups.push_back(std::move(group));
      continue;
    }
    const ConSanSyncSequence &leave = sync_sequences[cursor];
    append_member(leave);
    // LLVM's RDNA4 ISA model defines S_BARRIER_LEAVE as a zero-operand SOPP
    // with a fixed-zero simm16. Its execution-synchronization specification
    // defines it as dropping the last named barrier joined by this thread.
    // Therefore the matching static join, rather than the fixed encoding,
    // supplies the leave association for this lifecycle.
    const ConSanSyncEvent *leave_event = first_sequence_event(events, leave);
    const ConSanProgramSite *leave_decoded =
        leave_event == nullptr ? nullptr
                               : consan_program_site(inventory.program_sites, *leave_event);
    const ConSanBarrierSite *leave_source =
        leave_decoded == nullptr ? nullptr : leave_decoded->get_if<ConSanBarrierSite>();
    if (leave_source == nullptr || leave_source->size != sizeof(uint32_t) ||
        leave_source->raw_simm16 != 0u) {
      group.issue = ConSanBarrierLifecycleIssue::InvalidLeaveEncoding;
      groups.push_back(std::move(group));
      continue;
    }
    group.issue = ConSanBarrierLifecycleIssue::None;
    groups.push_back(std::move(group));
  }
}

void build_barrier_move_destination_inventory(
    const AmdGpuCodeObject &code_object, const std::vector<std::unique_ptr<BasicBlock>> &blocks,
    const SynchronizationInventoryView &events, rj_code_arch_t arch,
    std::span<const ConSanSyncSequence> sync_sequences, ConSanProgramAnalysisResult &result) {
  result.barrier_move_destinations.clear();
  if (code_object.text_sections().size() != 1u)
    return;
  const Section *text_section = code_object.text_sections().front();
  const auto text = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(text_section->data()), text_section->size());
  std::unordered_set<std::string> seen;

  for (const ConSanSyncSequence &sequence : sync_sequences) {
    if (sequence.kind != ConSanSyncKind::Barrier ||
        sequence.operation != ConSanSyncOperation::BarrierFull ||
        !consan_sync_confidence_meets(sequence.confidence,
                                      ConSanSemanticConfidence::Conservative) ||
        sequence.member_event_ids.size() != 2u) {
      continue;
    }
    const ConSanSyncEvent *event = first_sequence_event(events, sequence);
    if (event == nullptr)
      continue;

    const ConSanProgramSite *source = result.program_inventory.program_site(event->source_site);
    const ConSanProgramContainer *container =
        source == nullptr ? nullptr : result.program_inventory.container(source->container);
    if (container == nullptr || (container->is_kernel() && !container->has_text_range))
      continue;
    const uint64_t container_begin = container->entry_text_offset;
    const uint64_t container_end = container_begin + container->code_size;
    const uint64_t text_file_offset = container->text_file_offset;

    for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
      BasicBlock *block = blocks[block_index].get();
      if (block == nullptr || block->start_offset() < container_begin ||
          block->start_offset() >= container_end) {
        continue;
      }
      uint32_t clause_remaining = 0;
      for (const Instruction &instruction : block->instructions()) {
        const bool inside_clause = clause_remaining > 0;
        if (clause_remaining > 0)
          --clause_remaining;
        if (is_s_clause(instruction))
          clause_remaining = s_clause_following_instruction_count(instruction);
        if (instruction.src_loc() < container_begin || instruction.src_loc() >= container_end)
          continue;

        ConSanBarrierMoveDestination destination;
        destination.container_name = container->name;
        destination.in_kernel = container->is_kernel();
        destination.basic_block_index = static_cast<uint32_t>(block_index);
        destination.text_offset = instruction.src_loc();
        destination.file_offset = text_file_offset + instruction.src_loc();
        destination.size = static_cast<uint32_t>(instruction.size());
        destination.mnemonic = std::string(instruction.mnemonic());
        destination.memory_operation = instruction.is_memory_op();
        destination.identity = event->semantic_id.physical.code_object.fingerprint + "|" +
                               std::string(destination.in_kernel ? "kernel=" : "function=") +
                               destination.container_name + "|destination=instruction|pc=0x" +
                               consan_fixed_hex(destination.text_offset, 16) +
                               "|mnemonic=" + destination.mnemonic +
                               "|size=" + std::to_string(destination.size);
        if (!seen.insert(destination.identity).second)
          continue;

        std::string relocatable_error;
        if (inside_clause) {
          destination.issue = ConSanBarrierMoveDestinationIssue::InsideScalarClause;
        } else if (is_barrier_instruction(instruction)) {
          destination.issue = ConSanBarrierMoveDestinationIssue::BarrierSourceOrLifecycle;
        } else if (is_fence_like(instruction.mnemonic())) {
          destination.issue = ConSanBarrierMoveDestinationIssue::FenceOperation;
        } else if (!instruction.is_memory_op()) {
          destination.issue = ConSanBarrierMoveDestinationIssue::NotMemoryOperation;
        } else if (!is_relocatable_consan_barrier_destination(instruction, instruction.src_loc(),
                                                              text, arch, &relocatable_error)) {
          destination.issue = ConSanBarrierMoveDestinationIssue::NotRelocatable;
          destination.issue_detail = std::move(relocatable_error);
        }
        if (destination.suitable() && sequence.basic_block_index &&
            destination.basic_block_index != *sequence.basic_block_index) {
          if (const auto proof = prove_completing_structured_diamond(
                  blocks, *sequence.basic_block_index, destination.basic_block_index,
                  sequence.begin_text_offset, destination.text_offset)) {
            destination.cfg_contract = ConSanBarrierMoveCfgContract::CompletingStructuredDiamond;
            destination.structured_guard_block_index = proof->guard_block_index;
            destination.structured_source_block_index = proof->source_block_index;
            destination.structured_guard_offset = proof->guard_offset;
            destination.structured_source_offset = proof->source_offset;
          } else if (const auto proof = prove_structured_exec_diamond(
                         blocks, *sequence.basic_block_index, destination.basic_block_index,
                         sequence.begin_text_offset)) {
            destination.cfg_contract =
                ConSanBarrierMoveCfgContract::DestructiveStructuredExecDiamond;
            destination.structured_guard_block_index = proof->guard_block_index;
            destination.structured_source_block_index = proof->source_block_index;
            destination.structured_guard_offset = proof->guard_offset;
            destination.structured_source_offset = proof->source_offset;
          }
        }
        result.barrier_move_destinations.push_back(std::move(destination));
      }
    }
  }
}

[[nodiscard]] const ConSanSyncEvent *
first_sequence_event(const SynchronizationInventoryView &events,
                     const ConSanSyncSequence &sequence) {
  if (sequence.member_event_ids.empty())
    return nullptr;
  return events.find_event(sequence.member_event_ids.front());
}

[[nodiscard]] bool has_exact_group_flat_communication_site(const ProgramInventory &inventory,
                                                           const ConSanSyncEvent &event) {
  if (event.kind == ConSanSyncKind::Atomic) {
    const ConSanAtomicSite *site = inventory.program_site<ConSanAtomicSite>(event.source_site);
    return site != nullptr && site->address_space_hint == ConSanAtomicAddressSpaceHint::FlatGroup;
  }
  if (event.kind == ConSanSyncKind::OrdinaryMemory) {
    const ConSanOrdinaryMemorySite *site =
        inventory.program_site<ConSanOrdinaryMemorySite>(event.source_site);
    return site != nullptr && site->flat_address_space_hint == ConSanFlatAddressSpaceHint::Group;
  }
  return false;
}

[[nodiscard]] bool is_workgroup_flat_communication(const ProgramInventory &program_inventory,
                                                   const ConSanSyncEvent &event) {
  if (event.address_source != ConSanSyncAddressSource::FlatVector)
    return false;
  // RDNA4 carries workgroup scope explicitly even when pointer tracking
  // cannot reconstruct a complete generic address. Other targets spell the
  // same semantic through an exact src_shared_base provenance proof.
  if (event.scope == ConSanMemoryScope::Workgroup)
    return true;
  return has_exact_group_flat_communication_site(program_inventory, event);
}

[[nodiscard]] bool exact_workgroup_flat_acquire_encoding(const ProgramInventory &program_inventory,
                                                         const ConSanSyncEvent &event) {
  if (event.kind != ConSanSyncKind::OrdinaryMemory)
    return false;
  const ConSanOrdinaryMemorySite *site =
      program_inventory.program_site<ConSanOrdinaryMemorySite>(event.source_site);
  if (site == nullptr || site->operation != ConSanOrdinaryMemoryOperation::Load ||
      site->flat_address_space_hint != ConSanFlatAddressSpaceHint::Group)
    return false;
  return site->workgroup_acquire_ordering;
}

[[nodiscard]] bool is_bounded_acquire_bookkeeping(const Instruction &instruction);

[[nodiscard]] std::vector<const Instruction *>
instructions_preceding(const ConSanSyncSequence &sequence,
                       const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  std::vector<const Instruction *> result;
  if (!sequence.basic_block_index || *sequence.basic_block_index >= blocks.size() ||
      blocks[*sequence.basic_block_index] == nullptr)
    return result;
  for (const Instruction &instruction : blocks[*sequence.basic_block_index]->instructions()) {
    if (instruction.src_loc() >= sequence.begin_text_offset)
      break;
    result.push_back(&instruction);
  }
  return result;
}

[[nodiscard]] std::optional<uint64_t>
exact_workgroup_release_wait_boundary(const ConSanSyncSequence &sequence,
                                      const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                                      rj_code_arch_t arch) {
  const std::vector<const Instruction *> preceding = instructions_preceding(sequence, blocks);
  uint64_t boundary = sequence.begin_text_offset;
  bool saw_lds_zero = false;
  bool admitted_leading_clause = false;
  for (auto instruction = preceding.rbegin(); instruction != preceding.rend(); ++instruction) {
    const int size = (*instruction)->size();
    if (size != static_cast<int>(sizeof(uint32_t)) ||
        (*instruction)->src_loc() + sizeof(uint32_t) != boundary ||
        (*instruction)->raw_encoding() == nullptr)
      break;
    // LLVM may group a release atomic with the immediately following relaxed
    // atomic. The release is still ordered by the exact zero-wait prefix, but
    // the scalar scheduling hint sits between that prefix and the first
    // communication instruction. Admit only that first covered instruction;
    // an atomic later in the clause has an intervening memory operation and
    // therefore cannot inherit this release proof.
    if (!admitted_leading_clause && boundary == sequence.begin_text_offset &&
        is_s_clause(**instruction) && sequence.scalar_clause_text_offset &&
        *sequence.scalar_clause_text_offset == (*instruction)->src_loc() &&
        s_clause_following_instruction_count(**instruction) > 0u) {
      admitted_leading_clause = true;
      boundary = (*instruction)->src_loc();
      continue;
    }
    const ConSanWaitInstructionEncoding wait = classify_consan_wait_instruction(
        (*instruction)->mnemonic(), (*instruction)->raw_encoding()[0], arch);
    if (!wait.drains_lds && !wait.release_boundary)
      break;
    saw_lds_zero |= wait.drains_lds;
    boundary = (*instruction)->src_loc();
  }
  return saw_lds_zero ? std::optional<uint64_t>(boundary) : std::nullopt;
}

[[nodiscard]] std::optional<uint64_t>
exact_workgroup_acquire_wait_end(const ConSanSyncSequence &sequence,
                                 const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                                 rj_code_arch_t arch) {
  if (!sequence.basic_block_index || *sequence.basic_block_index >= blocks.size())
    return std::nullopt;
  BasicBlock *block = blocks[*sequence.basic_block_index].get();
  if (block == nullptr)
    return std::nullopt;
  constexpr uint64_t kMaximumAcquireSuffixBytes = 64u;
  uint64_t expected_offset = sequence.end_text_offset;
  bool saw_load_zero = false;
  bool saw_lds_zero = false;
  for (const Instruction &instruction : block->instructions()) {
    if (instruction.src_loc() < expected_offset)
      continue;
    if (instruction.src_loc() != expected_offset ||
        expected_offset - sequence.end_text_offset >= kMaximumAcquireSuffixBytes ||
        !is_bounded_acquire_bookkeeping(instruction)) {
      return std::nullopt;
    }
    const int size = instruction.size();
    if (size <= 0)
      return std::nullopt;
    expected_offset += static_cast<uint64_t>(size);
    if (size != static_cast<int>(sizeof(uint32_t)) || instruction.raw_encoding() == nullptr)
      continue;
    const ConSanWaitInstructionEncoding wait = classify_consan_wait_instruction(
        instruction.mnemonic(), instruction.raw_encoding()[0], arch);
    if (wait.drains_load && wait.drains_lds)
      return expected_offset;
    saw_load_zero |= wait.drains_load;
    saw_lds_zero |= wait.drains_lds;
    if (saw_load_zero && saw_lds_zero)
      return expected_offset;
  }
  return std::nullopt;
}

[[nodiscard]] bool has_only_waits_between(const SynchronizationInventoryView &events,
                                          const ConSanSyncSequence &before,
                                          const ConSanSyncSequence &after,
                                          const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  if (!before.basic_block_index || before.basic_block_index != after.basic_block_index ||
      !events.same_container(before, after) || before.end_text_offset > after.begin_text_offset ||
      *before.basic_block_index >= blocks.size()) {
    return false;
  }
  BasicBlock *block = blocks[*before.basic_block_index].get();
  if (block == nullptr)
    return false;
  uint64_t expected_offset = before.end_text_offset;
  for (const Instruction &instruction : block->instructions()) {
    const uint64_t offset = instruction.src_loc();
    if (offset < before.end_text_offset)
      continue;
    if (offset >= after.begin_text_offset)
      break;
    if (offset != expected_offset || !instruction.mnemonic().starts_with("s_wait"))
      return false;
    expected_offset += static_cast<uint64_t>(instruction.size());
  }
  return expected_offset == after.begin_text_offset;
}

[[nodiscard]] bool
is_exact_acquire_cache_pair(const ConSanSyncSequence &first, const ConSanSyncSequence &second,
                            const SynchronizationInventoryView &events,
                            const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  const ConSanSyncEvent *first_event = first_sequence_event(events, first);
  const ConSanSyncEvent *second_event = first_sequence_event(events, second);
  const ConSanProgramSite *first_decoded =
      first_event == nullptr ? nullptr : consan_program_site(events.program_sites, *first_event);
  const ConSanProgramSite *second_decoded =
      second_event == nullptr ? nullptr : consan_program_site(events.program_sites, *second_event);
  const ConSanFenceSite *first_source =
      first_decoded == nullptr ? nullptr : first_decoded->get_if<ConSanFenceSite>();
  const ConSanFenceSite *second_source =
      second_decoded == nullptr ? nullptr : second_decoded->get_if<ConSanFenceSite>();
  return first_source != nullptr && second_source != nullptr &&
         first_source->cache_operation == ConSanCacheOperation::AcquirePairPrefix &&
         second_source->cache_operation == ConSanCacheOperation::AcquirePairCompletion &&
         has_only_waits_between(events, first, second, blocks);
}

enum class BoundedAcquirePathKind { SameBlock, ExactFallthroughJoin, ExactSelfLoopExit };

[[nodiscard]] bool is_bounded_acquire_bookkeeping(const Instruction &instruction) {
  constexpr uint64_t kControlFlowFlags =
      BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR;
  return instruction.size() > 0 &&
         (!instruction.is_memory_op() || instruction.mnemonic().starts_with("s_wait")) &&
         !is_barrier_instruction(instruction) && !is_fence_like(instruction.mnemonic()) &&
         !is_s_clause(instruction) && (instruction.flags() & kControlFlowFlags) == 0u &&
         !instruction.branch_offset_bytes();
}

[[nodiscard]] bool scan_bounded_acquire_bookkeeping(BasicBlock &block, uint64_t begin, uint64_t end,
                                                    bool &saw_wait) {
  uint64_t expected_offset = begin;
  for (const Instruction &instruction : block.instructions()) {
    const uint64_t offset = instruction.src_loc();
    if (offset < begin)
      continue;
    if (offset >= end)
      break;
    if (offset != expected_offset || !is_bounded_acquire_bookkeeping(instruction))
      return false;
    saw_wait |= instruction.mnemonic().starts_with("s_wait");
    expected_offset += static_cast<uint64_t>(instruction.size());
  }
  return expected_offset == end;
}

[[nodiscard]] std::optional<BoundedAcquirePathKind>
prove_bounded_acquire_path(const ConSanSyncSequence &before, const ConSanSyncSequence &after,
                           const SynchronizationInventoryView &events,
                           const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                           bool require_wait_for_same_block = false) {
  constexpr uint64_t kMaximumBookkeepingBytes = 64u;
  constexpr uint64_t kMaximumPollingLoopPrefixBytes = 128u;
  if (!before.basic_block_index || !after.basic_block_index ||
      !events.same_container(before, after) || before.end_text_offset > after.begin_text_offset ||
      after.begin_text_offset - before.end_text_offset > kMaximumBookkeepingBytes ||
      before.inside_scalar_clause || after.inside_scalar_clause ||
      *before.basic_block_index >= blocks.size() || *after.basic_block_index >= blocks.size()) {
    return std::nullopt;
  }
  BasicBlock *source = blocks[*before.basic_block_index].get();
  BasicBlock *boundary = blocks[*after.basic_block_index].get();
  if (source == nullptr || boundary == nullptr || !source->call_edges().empty() ||
      !boundary->call_edges().empty()) {
    return std::nullopt;
  }

  bool saw_wait = false;
  if (source == boundary) {
    if (!scan_bounded_acquire_bookkeeping(*source, before.end_text_offset, after.begin_text_offset,
                                          saw_wait) ||
        (!saw_wait &&
         (require_wait_for_same_block || before.end_text_offset != after.begin_text_offset))) {
      return std::nullopt;
    }
    return BoundedAcquirePathKind::SameBlock;
  }

  // Cache operations can be direct branch targets even when the atomic path
  // reaches them by ordinary fallthrough. This is the shape emitted for an
  // acquire fence after a conditionally executed RMW: an inactive path may
  // enter at the cache operation, while the active RMW path has one exact,
  // bounded, non-branching route to it. Prove that route locally rather than
  // requiring the join block to have a unique predecessor. Other predecessors
  // do not invalidate the atomic-to-cache edge established by this fallthrough
  // path.
  if (!source->has_terminator() && source->end_offset() == boundary->start_offset() &&
      source->successors().size() == 1u && source->successors().front() == boundary) {
    if (!scan_bounded_acquire_bookkeeping(*source, before.end_text_offset, source->end_offset(),
                                          saw_wait) ||
        !scan_bounded_acquire_bookkeeping(*boundary, boundary->start_offset(),
                                          after.begin_text_offset, saw_wait) ||
        !saw_wait) {
      return std::nullopt;
    }
    return BoundedAcquirePathKind::ExactFallthroughJoin;
  }

  // Admit one exact compiler-lowered polling loop: the communication event is
  // the loop header, the conditional terminator either returns to that header
  // or falls through to the unique boundary block, and no other block can
  // enter the boundary. This is deliberately narrower than general dominance.
  const Instruction *terminator = source->terminator();
  if (terminator == nullptr || (terminator->flags() & COND_BRANCH) == 0u ||
      !terminator->branch_offset_bytes() || before.begin_text_offset < source->start_offset() ||
      before.begin_text_offset - source->start_offset() > kMaximumPollingLoopPrefixBytes ||
      source->end_offset() != boundary->start_offset() || source->successors().size() != 2u ||
      std::ranges::find(source->successors(), source) == source->successors().end() ||
      std::ranges::find(source->successors(), boundary) == source->successors().end() ||
      boundary->predecessors().size() != 1u || boundary->predecessors().front() != source) {
    return std::nullopt;
  }
  const int64_t branch_target = static_cast<int64_t>(source->end_offset()) +
                                static_cast<int64_t>(*terminator->branch_offset_bytes());
  if (branch_target < 0 || static_cast<uint64_t>(branch_target) != source->start_offset() ||
      terminator->src_loc() < before.end_text_offset ||
      terminator->src_loc() + static_cast<uint64_t>(terminator->size()) != source->end_offset()) {
    return std::nullopt;
  }
  bool prefix_wait = false;
  if (!scan_bounded_acquire_bookkeeping(*source, source->start_offset(), before.begin_text_offset,
                                        prefix_wait) ||
      !scan_bounded_acquire_bookkeeping(*source, before.end_text_offset, terminator->src_loc(),
                                        saw_wait)) {
    return std::nullopt;
  }
  saw_wait |= terminator->mnemonic().starts_with("s_wait");
  if (!scan_bounded_acquire_bookkeeping(*boundary, boundary->start_offset(),
                                        after.begin_text_offset, saw_wait) ||
      !saw_wait) {
    return std::nullopt;
  }
  return BoundedAcquirePathKind::ExactSelfLoopExit;
}

[[nodiscard]] ConSanSyncSequence
associate_atomic_cache_sequence(const ConSanSyncSequence *release, const ConSanSyncSequence &atomic,
                                const ConSanSyncSequence *acquire,
                                const ConSanSyncSequence *acquire_tail = nullptr) {
  ConSanSyncSequence sequence = atomic;
  sequence.memory_role = release != nullptr && acquire != nullptr
                             ? ConSanSyncMemoryRole::AcquireRelease
                         : release != nullptr ? ConSanSyncMemoryRole::Release
                                              : ConSanSyncMemoryRole::Acquire;
  sequence.memory_role_confidence = ConSanSemanticConfidence::Conservative;
  // A standalone flat atomic with no static address-space provenance is kept
  // unsupported because its ordering cannot be inferred from the instruction
  // alone. Reaching this helper supplies that missing evidence through an
  // exact release pattern or a bounded acquire-side compiler-bookkeeping
  // proof. Preserve the decoded address source for the later provenance gate,
  // but let the proven sequence carry conservative semantic confidence.
  sequence.confidence = ConSanSemanticConfidence::Conservative;
  sequence.confidence_reason = acquire != nullptr
                                   ? "bounded atomic/acquire-cache compiler-bookkeeping proof"
                                   : "same-block RDNA4 release-cache/atomic waits-only proof";
  std::vector<ConSanSyncEventId> semantic_members;
  if (release != nullptr) {
    sequence.identity += "|release-cache=" + release->identity;
    sequence.begin_text_offset = release->begin_text_offset;
    sequence.in_cyclic_cfg_component |= release->in_cyclic_cfg_component;
    sequence.inside_scalar_clause |= release->inside_scalar_clause;
    semantic_members.insert(semantic_members.end(), release->member_event_ids.begin(),
                            release->member_event_ids.end());
  }
  semantic_members.insert(semantic_members.end(), atomic.member_event_ids.begin(),
                          atomic.member_event_ids.end());
  if (acquire != nullptr) {
    sequence.identity += "|acquire-cache=" + acquire->identity;
    sequence.in_cyclic_cfg_component |= acquire->in_cyclic_cfg_component;
    sequence.inside_scalar_clause |= acquire->inside_scalar_clause;
    semantic_members.insert(semantic_members.end(), acquire->member_event_ids.begin(),
                            acquire->member_event_ids.end());
  }
  if (acquire_tail != nullptr) {
    sequence.identity += "|acquire-cache-tail=" + acquire_tail->identity;
    sequence.in_cyclic_cfg_component |= acquire_tail->in_cyclic_cfg_component;
    sequence.inside_scalar_clause |= acquire_tail->inside_scalar_clause;
    semantic_members.insert(semantic_members.end(), acquire_tail->member_event_ids.begin(),
                            acquire_tail->member_event_ids.end());
  }
  if (acquire_tail != nullptr)
    sequence.end_text_offset = acquire_tail->end_text_offset;
  else if (acquire != nullptr)
    sequence.end_text_offset = acquire->end_text_offset;
  sequence.member_event_ids = std::move(semantic_members);
  return sequence;
}

[[nodiscard]] std::optional<uint64_t>
exact_release_wait_boundary(const ConSanSyncSequence &sequence,
                            const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                            rj_code_arch_t arch) {
  const std::vector<const Instruction *> preceding = instructions_preceding(sequence, blocks);
  uint64_t suffix_begin = sequence.begin_text_offset;
  for (auto instruction = preceding.rbegin(); instruction != preceding.rend(); ++instruction) {
    const int size = (*instruction)->size();
    if (size <= 0 || (*instruction)->src_loc() + static_cast<uint64_t>(size) != suffix_begin ||
        !(*instruction)->mnemonic().starts_with("s_wait"))
      break;
    suffix_begin = (*instruction)->src_loc();
    if (size != static_cast<int>(sizeof(uint32_t)) || (*instruction)->raw_encoding() == nullptr)
      return std::nullopt;
    const ConSanWaitInstructionEncoding wait = classify_consan_wait_instruction(
        (*instruction)->mnemonic(), (*instruction)->raw_encoding()[0], arch);
    if (wait.bounded_release_counter_form && !wait.drains_load && !wait.drains_store &&
        !wait.drains_lds)
      return std::nullopt;
    if (wait.release_boundary)
      return (*instruction)->src_loc();
  }
  return std::nullopt;
}

void associate_atomic_sync_sequences(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                                     rj_code_arch_t arch,
                                     const SynchronizationInventoryView &events,
                                     SynchronizationInventoryBuildView inventory,
                                     const ProgramInventory &program_inventory) {
  std::vector<ConSanSyncSequence> &sync_sequences = inventory.sync_sequences;
  std::vector<ConSanSyncSequence> associated;
  associated.reserve(sync_sequences.size());
  for (size_t i = 0; i < sync_sequences.size();) {
    const ConSanSyncSequence &current = sync_sequences[i];
    if (current.kind == ConSanSyncKind::Fence &&
        is_release_cache_event(inventory.program_sites, first_sequence_event(events, current)) &&
        i + 1 < sync_sequences.size()) {
      const ConSanSyncSequence &atomic = sync_sequences[i + 1];
      if (atomic.kind == ConSanSyncKind::Atomic &&
          has_only_waits_between(events, current, atomic, blocks)) {
        const ConSanSyncSequence *acquire = nullptr;
        const ConSanSyncSequence *acquire_tail = nullptr;
        if (i + 2 < sync_sequences.size()) {
          const ConSanSyncSequence &candidate = sync_sequences[i + 2];
          if (candidate.kind == ConSanSyncKind::Fence &&
              is_acquire_cache_event(inventory.program_sites,
                                     first_sequence_event(events, candidate)) &&
              prove_bounded_acquire_path(atomic, candidate, events, blocks)) {
            acquire = &candidate;
          } else if (i + 3 < sync_sequences.size() && candidate.kind == ConSanSyncKind::Fence &&
                     sync_sequences[i + 3].kind == ConSanSyncKind::Fence &&
                     prove_bounded_acquire_path(atomic, candidate, events, blocks) &&
                     is_exact_acquire_cache_pair(candidate, sync_sequences[i + 3], events,
                                                 blocks)) {
            acquire = &candidate;
            acquire_tail = &sync_sequences[i + 3];
          }
        }
        associated.push_back(
            associate_atomic_cache_sequence(&current, atomic, acquire, acquire_tail));
        i += acquire_tail != nullptr ? 4 : (acquire != nullptr ? 3 : 2);
        continue;
      }
    }
    if (current.kind == ConSanSyncKind::Atomic && i + 1 < sync_sequences.size()) {
      const ConSanSyncSequence &acquire = sync_sequences[i + 1];
      if (acquire.kind == ConSanSyncKind::Fence &&
          is_acquire_cache_event(inventory.program_sites, first_sequence_event(events, acquire)) &&
          prove_bounded_acquire_path(current, acquire, events, blocks)) {
        associated.push_back(associate_atomic_cache_sequence(nullptr, current, &acquire));
        i += 2;
        continue;
      }
      if (i + 2 < sync_sequences.size() && acquire.kind == ConSanSyncKind::Fence &&
          sync_sequences[i + 2].kind == ConSanSyncKind::Fence &&
          prove_bounded_acquire_path(current, acquire, events, blocks) &&
          is_exact_acquire_cache_pair(acquire, sync_sequences[i + 2], events, blocks)) {
        associated.push_back(
            associate_atomic_cache_sequence(nullptr, current, &acquire, &sync_sequences[i + 2]));
        i += 3;
        continue;
      }
    }
    associated.push_back(current);
    ++i;
  }
  sync_sequences = std::move(associated);

  // RDNA4/CDNA5 lower the release half of an atomic to an immediately preceding
  // store-count-zero wait, either combined or inside a contiguous suffix of
  // counter-specific waits, without a separate release-cache event. This applies
  // both to release-only no-return atomics and to returning acquire-release
  // atomics followed by an acquire invalidate. Admit only that exact,
  // same-block encoding. The wait is a boundary of the addressed atomic
  // sequence, not a synchronization operation on its own.
  for (ConSanSyncSequence &sequence : sync_sequences) {
    if (sequence.kind != ConSanSyncKind::Atomic ||
        (sequence.memory_role != ConSanSyncMemoryRole::Unknown &&
         sequence.memory_role != ConSanSyncMemoryRole::Acquire)) {
      continue;
    }
    const auto boundary = exact_release_wait_boundary(sequence, blocks, arch);
    if (!boundary)
      continue;
    const bool already_acquire = sequence.memory_role == ConSanSyncMemoryRole::Acquire;
    sequence.memory_role =
        already_acquire ? ConSanSyncMemoryRole::AcquireRelease : ConSanSyncMemoryRole::Release;
    sequence.memory_role_confidence = ConSanSemanticConfidence::Conservative;
    sequence.confidence = ConSanSemanticConfidence::Conservative;
    sequence.confidence_reason =
        already_acquire ? "exact same-block RDNA4/CDNA5 release wait immediately before "
                          "bounded atomic/acquire-cache sequence"
                        : "exact same-block RDNA4/CDNA5 s_wait_storecnt_dscnt 0 "
                          "immediately before atomic release";
    sequence.release_wait_text_offset = *boundary;
    sequence.identity += "|release-wait=pc=0x" + consan_fixed_hex(*boundary, 16);
    sequence.begin_text_offset = *boundary;
  }

  // The language-level workgroup release used by RCCL has no cache event:
  // LLVM drains LDS immediately before a no-return group-FLAT atomic. Admit
  // only the exact target-specific zero-wait suffix. This complements the
  // device/system cache sequence above and deliberately leaves a missing,
  // nonzero, or noncontiguous wait unqualified.
  for (ConSanSyncSequence &sequence : sync_sequences) {
    if (sequence.kind != ConSanSyncKind::Atomic)
      continue;
    const ConSanSyncEvent *communication = first_sequence_event(events, sequence);
    if (communication == nullptr ||
        !is_workgroup_flat_communication(program_inventory, *communication))
      continue;
    sequence.scope = ConSanMemoryScope::Workgroup;
    if (sequence.memory_role != ConSanSyncMemoryRole::Unknown &&
        sequence.memory_role != ConSanSyncMemoryRole::Acquire)
      continue;
    const auto boundary = exact_workgroup_release_wait_boundary(sequence, blocks, arch);
    if (!boundary)
      continue;
    const bool already_acquire = sequence.memory_role == ConSanSyncMemoryRole::Acquire;
    sequence.memory_role =
        already_acquire ? ConSanSyncMemoryRole::AcquireRelease : ConSanSyncMemoryRole::Release;
    sequence.memory_role_confidence = ConSanSemanticConfidence::Conservative;
    sequence.confidence = ConSanSemanticConfidence::Conservative;
    sequence.confidence_reason =
        already_acquire
            ? "exact same-block LDS-zero wait before workgroup group-FLAT acquire-release"
            : "exact same-block LDS-zero wait before workgroup group-FLAT release";
    sequence.release_wait_text_offset = *boundary;
    sequence.identity += "|workgroup-release-wait=pc=0x" + consan_fixed_hex(*boundary, 16);
    sequence.begin_text_offset = *boundary;
  }
}

void build_moi_fence_candidate_inventory(SynchronizationInventoryBuildView inventory) {
  const std::vector<ConSanSyncEvent> &sync_events = inventory.sync_events;
  std::vector<ConSanMoiFenceCandidate> &candidates = inventory.moi_fence_candidates;
  candidates.clear();
  const auto fence_source = [&](const ConSanSyncEvent &event) -> const ConSanFenceSite * {
    const ConSanProgramSite *source = consan_program_site(inventory.program_sites, event);
    return source == nullptr ? nullptr : source->get_if<ConSanFenceSite>();
  };

  const SynchronizationInventoryView event_inventory = inventory.view();

  for (size_t fence_ordinal = 0; fence_ordinal < sync_events.size(); ++fence_ordinal) {
    const ConSanSyncEvent &fence = sync_events[fence_ordinal];
    if (fence.kind != ConSanSyncKind::Fence)
      continue;

    ConSanMoiFenceCandidate candidate;
    candidate.fence_event = {static_cast<uint32_t>(fence_ordinal)};

    const ConSanSyncSequence *sequence =
        event_inventory.find_unique_sequence_containing(candidate.fence_event);
    if (sequence == nullptr) {
      candidate.association = ConSanFenceAssociation::MissingOrAmbiguousSequence;
      candidates.push_back(std::move(candidate));
      continue;
    }
    candidate.sequence = event_inventory.sequence_id(*sequence);

    const bool addressed_atomic =
        sequence->kind == ConSanSyncKind::Atomic &&
        (sequence->operation == ConSanSyncOperation::AtomicRmw ||
         sequence->operation == ConSanSyncOperation::AtomicCompareExchange);
    const bool addressed_ordinary = sequence->kind == ConSanSyncKind::OrdinaryMemory &&
                                    (sequence->operation == ConSanSyncOperation::OrdinaryLoad ||
                                     sequence->operation == ConSanSyncOperation::OrdinaryStore);
    if (!addressed_atomic && !addressed_ordinary) {
      candidate.association = ConSanFenceAssociation::NotAddressedCommunication;
      candidates.push_back(std::move(candidate));
      continue;
    }
    if (!consan_sync_confidence_meets(sequence->confidence,
                                      ConSanSemanticConfidence::Conservative) ||
        !consan_sync_confidence_meets(sequence->memory_role_confidence,
                                      ConSanSemanticConfidence::Conservative)) {
      candidate.association = ConSanFenceAssociation::InsufficientConfidence;
      candidates.push_back(std::move(candidate));
      continue;
    }

    const auto sequence_has_cache_member = [&](ConSanCacheOperation operation) {
      return std::ranges::any_of(sequence->member_event_ids, [&](const ConSanSyncEventId identity) {
        const ConSanSyncEvent *event = event_inventory.find_event(identity);
        if (event == nullptr)
          return false;
        const ConSanFenceSite *source = fence_source(*event);
        return source != nullptr && source->cache_operation == operation;
      });
    };
    const ConSanFenceSite *fence_site = fence_source(fence);
    if (fence_site == nullptr) {
      candidate.association = ConSanFenceAssociation::UnsupportedCacheOperation;
      candidates.push_back(std::move(candidate));
      continue;
    }
    const bool acquire_pair_prefix =
        addressed_ordinary && sequence->memory_role == ConSanSyncMemoryRole::Acquire &&
        fence_site->cache_operation == ConSanCacheOperation::AcquirePairPrefix &&
        sequence_has_cache_member(ConSanCacheOperation::AcquirePairCompletion);
    const bool acquire_pair_completion =
        addressed_ordinary && sequence->memory_role == ConSanSyncMemoryRole::Acquire &&
        fence_site->cache_operation == ConSanCacheOperation::AcquirePairCompletion &&
        sequence_has_cache_member(ConSanCacheOperation::AcquirePairPrefix);
    if (acquire_pair_prefix) {
      candidate.association = ConSanFenceAssociation::AcquirePairPrefixCoveredByTail;
      candidates.push_back(std::move(candidate));
      continue;
    }
    if (is_release_cache_event(inventory.program_sites, &fence))
      candidate.memory_role = ConSanSyncMemoryRole::Release;
    else if (is_acquire_cache_event(inventory.program_sites, &fence) || acquire_pair_completion)
      candidate.memory_role = ConSanSyncMemoryRole::Acquire;
    else {
      candidate.association = ConSanFenceAssociation::UnsupportedCacheOperation;
      candidates.push_back(std::move(candidate));
      continue;
    }
    const bool role_is_carried =
        sequence->memory_role == candidate.memory_role ||
        sequence->memory_role == ConSanSyncMemoryRole::AcquireRelease ||
        sequence->memory_role == ConSanSyncMemoryRole::SequentiallyConsistent;
    if (!role_is_carried) {
      candidate.association = ConSanFenceAssociation::MemoryRoleMismatch;
      candidates.push_back(std::move(candidate));
      continue;
    }

    const ConSanSyncEvent *communication = nullptr;
    bool malformed_member = false;
    for (const ConSanSyncEventId identity : sequence->member_event_ids) {
      const ConSanSyncEvent *member = event_inventory.find_event(identity);
      if (member == nullptr) {
        malformed_member = true;
        break;
      }
      if (member->kind != ConSanSyncKind::Atomic && member->kind != ConSanSyncKind::OrdinaryMemory)
        continue;
      if (communication != nullptr) {
        malformed_member = true;
        break;
      }
      communication = member;
    }
    if (malformed_member || communication == nullptr || communication->identity.empty()) {
      candidate.association = ConSanFenceAssociation::MissingOrAmbiguousCommunicationEvent;
      candidates.push_back(std::move(candidate));
      continue;
    }
    if (communication->address_source != ConSanSyncAddressSource::GlobalScalarVector &&
        communication->address_source != ConSanSyncAddressSource::BufferResource &&
        communication->address_source != ConSanSyncAddressSource::FlatVector) {
      candidate.association = ConSanFenceAssociation::UnsupportedAddressSource;
      candidates.push_back(std::move(candidate));
      continue;
    }
    if (!communication->scope || !consan_memory_scope_is_supported(*communication->scope) ||
        *communication->scope == ConSanMemoryScope::Wavefront) {
      candidate.association = ConSanFenceAssociation::UnsupportedScope;
      candidates.push_back(std::move(candidate));
      continue;
    }

    candidate.communication_event = {static_cast<uint32_t>(communication - sync_events.data())};
    candidate.association = ConSanFenceAssociation::Qualified;
    candidates.push_back(std::move(candidate));
  }
}

struct OrdinaryReleaseBoundaries {
  std::vector<uint64_t> offsets;
  bool saw_wait = false;
};

[[nodiscard]] bool is_permitted_ordinary_release_bookkeeping(const Instruction &instruction) {
  const std::string_view mnemonic = instruction.mnemonic();
  return mnemonic.starts_with("s_wait") || mnemonic == "s_delay_alu" || mnemonic == "s_nop";
}

[[nodiscard]] bool is_ordinary_release_cache_instruction(const Instruction &instruction,
                                                         rj_code_arch_t arch) {
  return classify_consan_cache_operation(instruction.mnemonic(), arch).operation ==
         ConSanCacheOperation::Release;
}

[[nodiscard]] OrdinaryReleaseBoundaries
ordinary_release_cache_boundaries(const ConSanSyncSequence &store,
                                  const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                                  rj_code_arch_t arch) {
  OrdinaryReleaseBoundaries boundaries;
  if (!store.basic_block_index || *store.basic_block_index >= blocks.size())
    return boundaries;
  BasicBlock *block = blocks[*store.basic_block_index].get();
  if (block == nullptr)
    return boundaries;
  std::optional<uint64_t> prior_end;
  for (const Instruction &instruction : block->instructions()) {
    const uint64_t offset = instruction.src_loc();
    if (offset >= store.begin_text_offset)
      break;
    const int size = instruction.size();
    if (size <= 0 || (prior_end && offset != *prior_end)) {
      boundaries = {};
      prior_end.reset();
      continue;
    }
    prior_end = offset + static_cast<uint64_t>(size);
    if (is_ordinary_release_cache_instruction(instruction, arch)) {
      boundaries.offsets.push_back(offset);
    } else if (!is_permitted_ordinary_release_bookkeeping(instruction)) {
      boundaries = {};
    } else if (!boundaries.offsets.empty() && instruction.mnemonic().starts_with("s_wait")) {
      boundaries.saw_wait = true;
    }
  }
  if (!prior_end || *prior_end != store.begin_text_offset)
    boundaries = {};
  return boundaries;
}

[[nodiscard]] std::optional<uint64_t>
ordinary_release_cache_after_store(const ConSanSyncSequence &store,
                                   const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                                   rj_code_arch_t arch) {
  if (!store.basic_block_index || *store.basic_block_index >= blocks.size())
    return std::nullopt;
  BasicBlock *block = blocks[*store.basic_block_index].get();
  if (block == nullptr)
    return std::nullopt;
  uint64_t expected = store.end_text_offset;
  bool saw_wait = false;
  for (const Instruction &instruction : block->instructions()) {
    const uint64_t offset = instruction.src_loc();
    if (offset < expected)
      continue;
    const int size = instruction.size();
    if (offset != expected || size <= 0)
      return std::nullopt;
    expected += static_cast<uint64_t>(size);
    if (is_ordinary_release_cache_instruction(instruction, arch))
      return saw_wait ? std::optional<uint64_t>(offset) : std::nullopt;
    if (!is_permitted_ordinary_release_bookkeeping(instruction))
      return std::nullopt;
    saw_wait |= instruction.mnemonic().starts_with("s_wait");
  }
  return std::nullopt;
}

[[nodiscard]] std::vector<size_t>
fence_sequence_indices_by_begin(std::span<const ConSanSyncSequence> sync_sequences) {
  std::vector<size_t> indices;
  indices.reserve(sync_sequences.size());
  for (size_t index = 0; index < sync_sequences.size(); ++index) {
    if (sync_sequences[index].kind == ConSanSyncKind::Fence)
      indices.push_back(index);
  }
  std::ranges::sort(indices, [&](size_t lhs, size_t rhs) {
    const uint64_t lhs_offset = sync_sequences[lhs].begin_text_offset;
    const uint64_t rhs_offset = sync_sequences[rhs].begin_text_offset;
    return lhs_offset != rhs_offset ? lhs_offset < rhs_offset : lhs < rhs;
  });
  return indices;
}

[[nodiscard]] auto fence_sequence_lower_bound(std::span<const ConSanSyncSequence> sync_sequences,
                                              const std::vector<size_t> &indices,
                                              uint64_t text_offset) {
  return std::lower_bound(indices.begin(), indices.end(), text_offset,
                          [&](size_t index, uint64_t offset) {
                            return sync_sequences[index].begin_text_offset < offset;
                          });
}

void associate_ordinary_release_sync_sequences(
    const std::vector<std::unique_ptr<BasicBlock>> &blocks, rj_code_arch_t arch,
    const SynchronizationInventoryView &events, SynchronizationInventoryBuildView inventory) {
  std::vector<ConSanSyncSequence> &sync_sequences = inventory.sync_sequences;
  const std::vector<size_t> fence_indices = fence_sequence_indices_by_begin(sync_sequences);
  std::unordered_map<size_t, size_t> associations;
  std::unordered_set<size_t> claimed_cache_sequences;
  for (size_t store_index = 0; store_index < sync_sequences.size(); ++store_index) {
    const ConSanSyncSequence &store_sequence = sync_sequences[store_index];
    if (store_sequence.kind != ConSanSyncKind::OrdinaryMemory ||
        store_sequence.operation != ConSanSyncOperation::OrdinaryStore) {
      continue;
    }
    const ConSanSyncEvent *store = first_sequence_event(events, store_sequence);
    if (store == nullptr)
      continue;
    const OrdinaryReleaseBoundaries boundaries =
        ordinary_release_cache_boundaries(store_sequence, blocks, arch);
    std::optional<uint64_t> cache_offset;
    if (boundaries.saw_wait && boundaries.offsets.size() == 1u)
      cache_offset = boundaries.offsets.front();
    if (!cache_offset)
      cache_offset = ordinary_release_cache_after_store(store_sequence, blocks, arch);
    if (!cache_offset)
      continue;
    std::vector<size_t> cache_matches;
    for (auto it = fence_sequence_lower_bound(sync_sequences, fence_indices, *cache_offset);
         it != fence_indices.end(); ++it) {
      const size_t cache_index = *it;
      const ConSanSyncSequence &cache_sequence = sync_sequences[cache_index];
      if (cache_sequence.begin_text_offset != *cache_offset)
        break;
      const ConSanSyncEvent *cache = first_sequence_event(events, cache_sequence);
      if (cache != nullptr &&
          consan_ordinary_release_metadata_compatible(inventory.program_sites, *cache,
                                                      cache_sequence, *store, store_sequence)) {
        cache_matches.push_back(cache_index);
      }
    }
    if (cache_matches.size() != 1u || claimed_cache_sequences.contains(cache_matches.front()))
      continue;
    associations.emplace(store_index, cache_matches.front());
    claimed_cache_sequences.insert(cache_matches.front());
  }

  std::vector<ConSanSyncSequence> associated;
  associated.reserve(sync_sequences.size());
  for (size_t index = 0; index < sync_sequences.size(); ++index) {
    if (claimed_cache_sequences.contains(index))
      continue;
    const auto association = associations.find(index);
    if (association == associations.end()) {
      associated.push_back(std::move(sync_sequences[index]));
      continue;
    }
    ConSanSyncSequence sequence = std::move(sync_sequences[index]);
    ConSanSyncSequence &cache = sync_sequences[association->second];
    sequence.memory_role = ConSanSyncMemoryRole::Release;
    sequence.memory_role_confidence = ConSanSemanticConfidence::Conservative;
    sequence.confidence = combine_consan_sync_confidence(sequence.confidence, cache.confidence);
    sequence.confidence_reason =
        "exact same-block release-cache/waits/ordinary store pattern with only "
        "permitted bookkeeping between members";
    sequence.identity += "|release-cache=" + cache.identity;
    sequence.begin_text_offset = cache.begin_text_offset;
    sequence.in_cyclic_cfg_component |= cache.in_cyclic_cfg_component;
    sequence.inside_scalar_clause |= cache.inside_scalar_clause;
    std::vector<ConSanSyncEventId> semantic_members = cache.member_event_ids;
    semantic_members.insert(semantic_members.end(), sequence.member_event_ids.begin(),
                            sequence.member_event_ids.end());
    sequence.member_event_ids = std::move(semantic_members);
    associated.push_back(std::move(sequence));
  }
  sync_sequences = std::move(associated);

  // Optimized language-level releases can omit a separate writeback event and
  // carry their release half in the architecture-specific store-count-zero
  // wait immediately before the device/system-scoped store. Keep this
  // admission separate from the broader cache-associated form above: a
  // missing or nonzero wait, intervening code, or weaker scope stays unknown.
  for (ConSanSyncSequence &sequence : sync_sequences) {
    if (sequence.kind != ConSanSyncKind::OrdinaryMemory ||
        sequence.operation != ConSanSyncOperation::OrdinaryStore ||
        sequence.memory_role != ConSanSyncMemoryRole::Unknown) {
      continue;
    }
    const ConSanSyncEvent *store = first_sequence_event(events, sequence);
    const ConSanProgramSite *store_decoded =
        store == nullptr ? nullptr : consan_program_site(inventory.program_sites, *store);
    const ConSanOrdinaryMemorySite *store_source =
        store_decoded == nullptr ? nullptr : store_decoded->get_if<ConSanOrdinaryMemorySite>();
    if (store_source == nullptr || store_source->width_bits != 32u || !store->scope ||
        !consan_memory_scope_is_agent_or_system(*store->scope) ||
        store->confidence != ConSanSemanticConfidence::Conservative ||
        store_decoded->execution_owners.empty()) {
      continue;
    }
    const auto boundary = exact_release_wait_boundary(sequence, blocks, arch);
    if (!boundary)
      continue;
    sequence.memory_role = ConSanSyncMemoryRole::Release;
    sequence.memory_role_confidence = ConSanSemanticConfidence::Conservative;
    sequence.confidence = ConSanSemanticConfidence::Conservative;
    sequence.confidence_reason =
        "exact same-block architecture-specific store-count-zero wait immediately before "
        "scoped ordinary release store";
    sequence.release_wait_text_offset = *boundary;
    sequence.identity += "|release-waits=pc=0x" + consan_fixed_hex(*boundary, 16);
    sequence.begin_text_offset = *boundary;
  }
}

void associate_ordinary_acquire_sync_sequences(
    const std::vector<std::unique_ptr<BasicBlock>> &blocks,
    const SynchronizationInventoryView &events, rj_code_arch_t arch,
    SynchronizationInventoryBuildView inventory, const ProgramInventory &program_inventory) {
  std::vector<ConSanSyncSequence> &sync_sequences = inventory.sync_sequences;
  struct Association {
    size_t cache_index = 0;
    std::optional<size_t> cache_tail_index;
    std::optional<uint64_t> polling_loop_header_text_offset;
  };
  const std::vector<size_t> fence_indices = fence_sequence_indices_by_begin(sync_sequences);
  std::unordered_map<size_t, Association> associations;
  std::unordered_set<size_t> claimed_cache_sequences;
  for (size_t load_index = 0; load_index < sync_sequences.size(); ++load_index) {
    const ConSanSyncSequence &load_sequence = sync_sequences[load_index];
    if (load_sequence.kind != ConSanSyncKind::OrdinaryMemory ||
        load_sequence.operation != ConSanSyncOperation::OrdinaryLoad) {
      continue;
    }
    const ConSanSyncEvent *load = first_sequence_event(events, load_sequence);
    if (load == nullptr)
      continue;
    if (is_workgroup_flat_communication(program_inventory, *load) &&
        !exact_workgroup_flat_acquire_encoding(program_inventory, *load))
      continue;
    const auto make_association = [&](size_t cache_index, std::optional<size_t> cache_tail_index,
                                      BoundedAcquirePathKind path) {
      std::optional<uint64_t> polling_loop_header_text_offset;
      if (path == BoundedAcquirePathKind::ExactSelfLoopExit && load_sequence.basic_block_index &&
          *load_sequence.basic_block_index < blocks.size() &&
          blocks[*load_sequence.basic_block_index] != nullptr) {
        polling_loop_header_text_offset = blocks[*load_sequence.basic_block_index]->start_offset();
      }
      return Association{cache_index, cache_tail_index, polling_loop_header_text_offset};
    };
    std::vector<Association> cache_matches;
    std::vector<size_t> bounded_metadata_matches;
    std::vector<Association> cache_pair_matches;
    const uint64_t maximum_cache_offset = load_sequence.end_text_offset > UINT64_MAX - 64u
                                              ? UINT64_MAX
                                              : load_sequence.end_text_offset + 64u;
    for (auto it = fence_sequence_lower_bound(sync_sequences, fence_indices,
                                              load_sequence.end_text_offset);
         it != fence_indices.end(); ++it) {
      const size_t cache_index = *it;
      const ConSanSyncSequence &cache_sequence = sync_sequences[cache_index];
      if (cache_sequence.begin_text_offset > maximum_cache_offset)
        break;
      const ConSanSyncEvent *cache = first_sequence_event(events, cache_sequence);
      if (cache == nullptr ||
          !consan_ordinary_acquire_metadata_compatible(
              inventory.program_sites, *load, load_sequence, *cache, cache_sequence,
              ConSanOrdinaryAcquireMetadataPolicy::BoundedPathSingleFence) ||
          load_sequence.end_text_offset > cache_sequence.begin_text_offset ||
          cache_sequence.begin_text_offset - load_sequence.end_text_offset > 64u) {
        continue;
      }
      bounded_metadata_matches.push_back(cache_index);
      const auto path = prove_bounded_acquire_path(load_sequence, cache_sequence, events, blocks,
                                                   /*require_wait_for_same_block=*/true);
      if (!path)
        continue;
      const bool same_block = *path == BoundedAcquirePathKind::SameBlock;
      const auto metadata_policy =
          same_block ? ConSanOrdinaryAcquireMetadataPolicy::SameBlockSingleFence
                     : ConSanOrdinaryAcquireMetadataPolicy::BoundedPathSingleFence;
      if (consan_ordinary_acquire_metadata_compatible(inventory.program_sites, *load, load_sequence,
                                                      *cache, cache_sequence, metadata_policy))
        cache_matches.push_back(make_association(cache_index, std::nullopt, *path));
    }

    if (bounded_metadata_matches.empty()) {
      for (auto it = fence_sequence_lower_bound(sync_sequences, fence_indices,
                                                load_sequence.end_text_offset);
           it != fence_indices.end(); ++it) {
        const size_t first_index = *it;
        const ConSanSyncSequence &first = sync_sequences[first_index];
        if (first.begin_text_offset > maximum_cache_offset)
          break;
        const auto tail_it = std::next(it);
        if (tail_it == fence_indices.end())
          break;
        const size_t tail_index = *tail_it;
        const ConSanSyncSequence &tail = sync_sequences[tail_index];
        const ConSanSyncEvent *first_event = first_sequence_event(events, first);
        const ConSanSyncEvent *tail_event = first_sequence_event(events, tail);
        if (first_event == nullptr || tail_event == nullptr ||
            !consan_ordinary_acquire_metadata_compatible(
                inventory.program_sites, *load, load_sequence, *first_event, first,
                ConSanOrdinaryAcquireMetadataPolicy::BoundedPathCachePairMember) ||
            !consan_ordinary_acquire_metadata_compatible(
                inventory.program_sites, *load, load_sequence, *tail_event, tail,
                ConSanOrdinaryAcquireMetadataPolicy::BoundedPathCachePairMember) ||
            !is_exact_acquire_cache_pair(first, tail, events, blocks)) {
          continue;
        }
        const auto path = prove_bounded_acquire_path(load_sequence, first, events, blocks,
                                                     /*require_wait_for_same_block=*/true);
        if (!path)
          continue;
        const bool same_block = *path == BoundedAcquirePathKind::SameBlock;
        const auto metadata_policy =
            same_block ? ConSanOrdinaryAcquireMetadataPolicy::SameBlockCachePairMember
                       : ConSanOrdinaryAcquireMetadataPolicy::BoundedPathCachePairMember;
        if (!consan_ordinary_acquire_metadata_compatible(inventory.program_sites, *load,
                                                         load_sequence, *first_event, first,
                                                         metadata_policy) ||
            !consan_ordinary_acquire_metadata_compatible(inventory.program_sites, *load,
                                                         load_sequence, *tail_event, tail,
                                                         metadata_policy)) {
          continue;
        }
        cache_pair_matches.push_back(make_association(first_index, tail_index, *path));
      }
    }

    std::optional<Association> association;
    if (cache_matches.size() == 1u) {
      association = cache_matches.front();
    } else if (bounded_metadata_matches.empty() && cache_matches.empty() &&
               cache_pair_matches.size() == 1u) {
      association = cache_pair_matches.front();
    }
    if (!association || claimed_cache_sequences.contains(association->cache_index) ||
        (association->cache_tail_index &&
         claimed_cache_sequences.contains(*association->cache_tail_index)))
      continue;
    associations.emplace(load_index, *association);
    claimed_cache_sequences.insert(association->cache_index);
    if (association->cache_tail_index)
      claimed_cache_sequences.insert(*association->cache_tail_index);
  }

  std::vector<ConSanSyncSequence> associated;
  associated.reserve(sync_sequences.size());
  for (size_t index = 0; index < sync_sequences.size(); ++index) {
    if (claimed_cache_sequences.contains(index))
      continue;
    const auto association = associations.find(index);
    if (association == associations.end()) {
      associated.push_back(std::move(sync_sequences[index]));
      continue;
    }
    ConSanSyncSequence sequence = std::move(sync_sequences[index]);
    ConSanSyncSequence &cache = sync_sequences[association->second.cache_index];
    ConSanSyncSequence *cache_tail = association->second.cache_tail_index
                                         ? &sync_sequences[*association->second.cache_tail_index]
                                         : nullptr;
    sequence.memory_role = ConSanSyncMemoryRole::Acquire;
    sequence.memory_role_confidence = ConSanSemanticConfidence::Conservative;
    sequence.confidence = combine_consan_sync_confidence(sequence.confidence, cache.confidence);
    if (cache_tail != nullptr)
      sequence.confidence =
          combine_consan_sync_confidence(sequence.confidence, cache_tail->confidence);
    sequence.confidence_reason = "bounded ordinary-load/acquire-cache compiler-bookkeeping proof";
    sequence.acquire_polling_loop_header_text_offset =
        association->second.polling_loop_header_text_offset;
    sequence.identity += "|acquire-cache=" + cache.identity;
    if (cache_tail != nullptr)
      sequence.identity += "|acquire-cache-tail=" + cache_tail->identity;
    sequence.end_text_offset =
        cache_tail != nullptr ? cache_tail->end_text_offset : cache.end_text_offset;
    sequence.in_cyclic_cfg_component |= cache.in_cyclic_cfg_component;
    sequence.inside_scalar_clause |= cache.inside_scalar_clause;
    sequence.member_event_ids.insert(sequence.member_event_ids.end(),
                                     cache.member_event_ids.begin(), cache.member_event_ids.end());
    if (cache_tail != nullptr) {
      sequence.in_cyclic_cfg_component |= cache_tail->in_cyclic_cfg_component;
      sequence.inside_scalar_clause |= cache_tail->inside_scalar_clause;
      sequence.member_event_ids.insert(sequence.member_event_ids.end(),
                                       cache_tail->member_event_ids.begin(),
                                       cache_tail->member_event_ids.end());
    }
    associated.push_back(std::move(sequence));
  }
  sync_sequences = std::move(associated);

  // Some targets lower an LDS acquire load to group-FLAT ordering plus an
  // exact combined load/LDS zero wait, without a cache-invalidate operation.
  // Keep that target-owned fallback separate from the cache-associated paths
  // above.
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr || !target->synchronization.workgroup_flat_acquire_wait_fallback)
    return;
  for (ConSanSyncSequence &sequence : sync_sequences) {
    if (sequence.kind != ConSanSyncKind::OrdinaryMemory ||
        sequence.operation != ConSanSyncOperation::OrdinaryLoad ||
        sequence.memory_role != ConSanSyncMemoryRole::Unknown || sequence.inside_scalar_clause)
      continue;
    const ConSanSyncEvent *load = first_sequence_event(events, sequence);
    const ConSanProgramSite *load_decoded =
        load == nullptr ? nullptr : program_inventory.program_site(load->source_site);
    const ConSanOrdinaryMemorySite *load_source =
        load_decoded == nullptr ? nullptr : load_decoded->get_if<ConSanOrdinaryMemorySite>();
    if (load_source == nullptr || load_source->width_bits != 32u ||
        load_decoded->execution_owners.empty() ||
        !is_workgroup_flat_communication(program_inventory, *load))
      continue;
    if (!exact_workgroup_flat_acquire_encoding(program_inventory, *load))
      continue;
    const auto wait_end = exact_workgroup_acquire_wait_end(sequence, blocks, arch);
    if (!wait_end)
      continue;
    sequence.memory_role = ConSanSyncMemoryRole::Acquire;
    sequence.memory_role_confidence = ConSanSemanticConfidence::Conservative;
    sequence.confidence = ConSanSemanticConfidence::Conservative;
    sequence.confidence_reason =
        "exact same-block workgroup group-FLAT load/acquire-zero-wait sequence";
    sequence.scope = ConSanMemoryScope::Workgroup;
    sequence.identity += "|workgroup-acquire-wait-end=pc=0x" + consan_fixed_hex(*wait_end, 16);
    sequence.end_text_offset = *wait_end;
  }
}

[[nodiscard]] ConSanOwnerProofKind owner_proof_kind(KernelCfgOwnerProofKind kind) {
  switch (kind) {
  case KernelCfgOwnerProofKind::KernelLocal:
    return ConSanOwnerProofKind::KernelLocal;
  case KernelCfgOwnerProofKind::DirectCall:
    return ConSanOwnerProofKind::DirectCall;
  case KernelCfgOwnerProofKind::RecoveredIndirectCall:
    return ConSanOwnerProofKind::RecoveredIndirectCall;
  }
  return ConSanOwnerProofKind::RecoveredIndirectCall;
}

[[nodiscard]] std::vector<ConSanExecutionOwner> execution_owners_for_offset(
    const std::unordered_map<const BasicBlock *, std::vector<ConSanExecutionOwner>>
        &owners_by_block,
    const BlockOffsetIndex &block_index, uint64_t offset) {
  const BasicBlock *block = block_for_offset(block_index, offset);
  if (block == nullptr)
    return {};
  const auto owners = owners_by_block.find(block);
  return owners == owners_by_block.end() ? std::vector<ConSanExecutionOwner>{} : owners->second;
}

void annotate_execution_owners(const AmdGpuCodeObject &code_object, Decoder &decoder,
                               rj_code_arch_t arch,
                               const std::vector<std::unique_ptr<BasicBlock>> *reusable_blocks,
                               SynchronizationInventoryBuildView inventory,
                               ConSanProgramAnalysisResult &result) {
  const std::span<const ConSanPreappliedCodeRange> preapplied_ranges =
      result.program_inventory.preapplied_mutation().code_ranges;
  const consan_detail::ConSanCfgBuildInputs cfg = consan_detail::build_consan_cfg_inputs(
      code_object, result.program_inventory.containers(), preapplied_ranges);
  std::vector<std::unique_ptr<BasicBlock>> rebuilt_blocks;
  if (reusable_blocks == nullptr || !preapplied_ranges.empty()) {
    rebuilt_blocks = BasicBlock::build(code_object, decoder, arch, cfg.leaders, cfg.code_ranges);
    reusable_blocks = &rebuilt_blocks;
  }
  const auto &blocks = *reusable_blocks;
  const BlockOffsetIndex block_index = build_block_offset_index(blocks);
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> text = patcher.text_bytes();
  std::unordered_map<const BasicBlock *, std::vector<ConSanExecutionOwner>> owners_by_block;
  owners_by_block.reserve(blocks.size());
  for (const ConSanProgramContainer &kernel : result.program_inventory.kernels()) {
    if (!kernel.has_text_range)
      continue;
    std::vector<uint64_t> additional_entry_offsets;
    for (const ConSanPreappliedCodeRange &range : preapplied_ranges) {
      if (range.kernel_name != kernel.name)
        continue;
      if (block_for_offset(block_index, range.text_offset) != nullptr)
        additional_entry_offsets.push_back(range.text_offset);
      if (range.continuation_text_offset &&
          block_for_offset(block_index, *range.continuation_text_offset) != nullptr)
        additional_entry_offsets.push_back(*range.continuation_text_offset);
    }
    auto scope = build_kernel_cfg_scope(
        blocks, block_index,
        KernelScopeRequest{.entry_offset = kernel.entry_text_offset,
                           .additional_entry_offsets = additional_entry_offsets},
        cfg.kernel_entries, text);
    if (scope) {
      for (const auto &[block, proof] : scope->owner_proofs) {
        owners_by_block[block].push_back({.kernel = kernel.id, .proof = owner_proof_kind(proof)});
      }
    }
  }

  for (ConSanBarrierMoveDestination &destination : result.barrier_move_destinations) {
    destination.execution_owners =
        execution_owners_for_offset(owners_by_block, block_index, destination.text_offset);
  }
  for (ConSanProgramSite &site : inventory.program_sites)
    site.execution_owners = execution_owners_for_offset(owners_by_block, block_index,
                                                        site.physical_id.original_text_offset);
}

} // namespace

bool analyze_consan_semantic_inventory(std::span<const uint8_t> code_object_bytes,
                                       const AmdGpuCodeObject &code_object, Decoder &decoder,
                                       rj_code_arch_t arch, const ConSanRequest &request,
                                       const ConSanDebugOverrides &debug,
                                       const MutationRequest &mutation,
                                       ProgramInventoryBuilder &inventory_builder,
                                       ConSanPerturbationPlanningState &perturbation,
                                       ConSanProgramAnalysisResult &result) {
  SynchronizationInventoryBuildView synchronization_inventory = inventory_builder.synchronization();
  const bool needs_semantic_inventory = request.flavor == ConSanFlavor::Moi ||
                                        mutation.fault_dry_run || mutation.has_fault_mutation() ||
                                        mutation.sc_perturb_kind != ConSanPerturbationKind::None ||
                                        debug.abort_unmatched_barrier_wait;
  if (!needs_semantic_inventory) {
    if (std::ranges::any_of(result.program_inventory.access_sites(),
                            [&](const ConSanProgramSite &site) {
                              const ConSanProgramContainer *container =
                                  result.program_inventory.container(site.container);
                              return container != nullptr && !container->is_kernel();
                            })) {
      // SuperCollider's ordinary clean transform does not consume
      // synchronization semantics. Shared-function sites still need kernel
      // ownership, while kernel-local sites carry their descriptor directly.
      annotate_execution_owners(code_object, decoder, arch, nullptr, synchronization_inventory,
                                result);
    }
    result.program_inventory = inventory_builder.view();
    return true;
  }

  build_fault_site_inventory(code_object_bytes, arch, result);
  build_sync_event_inventory(code_object_bytes, synchronization_inventory, result.program_inventory,
                             arch);
  if (!canonicalize_sync_events_by_physical_site(synchronization_inventory, result)) {
    result.program_inventory = inventory_builder.view();
    return false;
  }

  const SynchronizationInventoryView sync_events = synchronization_inventory.view();
  const auto sync_blocks =
      build_sync_basic_blocks(code_object, decoder, arch, result.program_inventory);
  build_singleton_sync_sequences(sync_blocks, synchronization_inventory);
  associate_barrier_sync_sequences(sync_blocks,
                                   consan_requires_extended_barrier_pairs(request, debug, mutation),
                                   synchronization_inventory);
  associate_atomic_sync_sequences(sync_blocks, arch, sync_events, synchronization_inventory,
                                  result.program_inventory);
  build_barrier_lifecycle_group_inventory(synchronization_inventory, sync_events);
  if (mutation.fault_move_barrier) {
    build_barrier_move_destination_inventory(code_object, sync_blocks, sync_events, arch,
                                             synchronization_inventory.sync_sequences, result);
  }
  annotate_execution_owners(code_object, decoder, arch, &sync_blocks, synchronization_inventory,
                            result);
  associate_ordinary_release_sync_sequences(sync_blocks, arch, sync_events,
                                            synchronization_inventory);
  associate_ordinary_acquire_sync_sequences(sync_blocks, sync_events, arch,
                                            synchronization_inventory, result.program_inventory);
  synchronization_inventory.sequence_membership_by_event = build_sync_sequence_membership_index(
      synchronization_inventory.sync_sequences, synchronization_inventory.sync_events.size());
  build_moi_fence_candidate_inventory(synchronization_inventory);
  result.program_inventory = inventory_builder.view();
  build_perturbation_candidate_inventory(result.program_inventory, perturbation);
  return true;
}

} // namespace rocjitsu
