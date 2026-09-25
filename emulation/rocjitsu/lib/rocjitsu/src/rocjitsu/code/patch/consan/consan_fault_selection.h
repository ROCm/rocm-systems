// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_fault_selection.h
/// @brief Exact, ownership-aware selection of ConSan mutation sites.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace rocjitsu::consan {

/// Immutable selection facts shared by mutation planning and proof
/// rederivation. This deliberately excludes mutation kind, payload, lowering
/// options, and diagnostic state.
struct FaultSelection {
  std::string_view primary_site_identity;
  std::string_view primary_sequence_identity;
  std::string_view companion_site_identity;
  std::string_view companion_sequence_identity;
  std::string_view kernel_name_filter;
  uint32_t ordinal = 0;
};

/// Narrow immutable input used to resolve semantic fault selections.
///
/// Selection needs the normalized program inventory and the exact fault sites
/// derived from it. It must not inspect mutation state, resource planning,
/// emitted patches, diagnostics, or any other transformation-transaction
/// field.
struct FaultSelectionView {
  const ProgramInventory &program_inventory;
  std::span<const FaultSite> fault_sites;

  [[nodiscard]] const ProgramSite *source(const FaultSite &site) const {
    return program_inventory.program_site(site);
  }
  [[nodiscard]] std::span<const ExecutionOwner> execution_owners(const FaultSite &site) const {
    const ProgramSite *program_site = source(site);
    return program_site == nullptr
               ? std::span<const ExecutionOwner>{}
               : std::span<const ExecutionOwner>(program_site->execution_owners);
  }
};

/// Fully rederived ordinary-acquire mutation target. Every pointer refers into
/// the immutable inventory supplied to the selector.
struct OrdinaryAcquireMutationTarget {
  const FaultSite *site = nullptr;
  const SyncEvent *load = nullptr;
  const SyncEvent *cache = nullptr;
  const SyncSequence *sequence = nullptr;
};

/// A complete exact logical barrier. A singleton full barrier, admitted only
/// within an explicit group, has no companion site.
struct ExactBarrierDropPair {
  const SyncSequence *sequence = nullptr;
  const FaultSite *primary = nullptr;
  const FaultSite *companion = nullptr;
};

/// Two ordered, disjoint exact barriers selected as one logical mutation.
struct ExactBarrierDropGroup {
  ExactBarrierDropPair first;
  ExactBarrierDropPair second;
};

/// Stable semantic reason why an exact logical barrier could not be resolved
/// to its two physical mutation sites.
enum class ExactBarrierDropPairIssue : uint8_t {
  None,
  MissingExactIdentity,
  SequenceNotFound,
  SequenceNotQualified,
  PrimaryNotMember,
  MemberSiteMissing,
  MemberSiteAmbiguous,
  InvalidPairGeometry,
  Count,
};

/// The exact pair, or the typed reason why selection rejected it.
struct ExactBarrierDropPairResolution {
  std::optional<ExactBarrierDropPair> pair;
  ExactBarrierDropPairIssue issue = ExactBarrierDropPairIssue::None;
};

/// Stable semantic reason why two exact logical barriers could not be resolved
/// as one ordered mutation group. Pair-specific detail stays typed separately.
enum class ExactBarrierDropGroupIssue : uint8_t {
  None,
  MissingCompanionIdentity,
  FirstPairRejected,
  SecondPairRejected,
  PairsOverlapOrUnordered,
  PairsHaveDifferentOwners,
  Count,
};

/// The exact group, or its typed group and optional member-pair rejection.
struct ExactBarrierDropGroupResolution {
  std::optional<ExactBarrierDropGroup> group;
  ExactBarrierDropGroupIssue issue = ExactBarrierDropGroupIssue::None;
  ExactBarrierDropPairIssue member_issue = ExactBarrierDropPairIssue::None;
};

/// Decide whether a cross-block destination carries the exact structured-CFG
/// proof and explicit request opt-in required by barrier-move mutation.
[[nodiscard]] bool fault_admits_cross_block_barrier_move(const BarrierMoveDestination &destination,
                                                         const MutationRequest &mutation);

[[nodiscard]] const FaultSite *find_fault_site_by_identity(const FaultSelectionView &inventory,
                                                           std::string_view identity,
                                                           FaultSiteKind kind);

[[nodiscard]] bool execution_owners_include_requested_kernel(std::span<const ExecutionOwner> owners,
                                                             const FaultSelectionView &inventory,
                                                             std::string_view kernel_name_filter);

[[nodiscard]] const FaultSite *select_fault_site_for_plan(const FaultSelectionView &inventory,
                                                          const FaultSelection &selection,
                                                          FaultSiteKind kind);

[[nodiscard]] std::optional<OrdinaryAcquireMutationTarget>
select_ordinary_acquire_mutation_target(const FaultSelectionView &inventory,
                                        const FaultSelection &selection);

[[nodiscard]] ExactBarrierDropPairResolution
resolve_exact_barrier_drop_pair(const FaultSelectionView &inventory,
                                const FaultSelection &selection, bool allow_full_singleton = false);

[[nodiscard]] ExactBarrierDropGroupResolution
resolve_exact_barrier_drop_group(const FaultSelectionView &inventory,
                                 const FaultSelection &selection);

[[nodiscard]] std::string_view
exact_barrier_drop_pair_issue_message(ExactBarrierDropPairIssue issue);

[[nodiscard]] std::string exact_barrier_drop_group_issue_message(
    ExactBarrierDropGroupIssue issue,
    ExactBarrierDropPairIssue member_issue = ExactBarrierDropPairIssue::None);

[[nodiscard]] std::string exact_barrier_drop_group_identity(const ExactBarrierDropGroup &group);

} // namespace rocjitsu::consan
