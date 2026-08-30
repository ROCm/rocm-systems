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

namespace rocjitsu {

/// Immutable selection facts shared by mutation planning and proof
/// rederivation. This deliberately excludes mutation kind, payload, lowering
/// options, and diagnostic state.
struct ConSanFaultSelection {
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
struct ConSanFaultSelectionView {
  const ProgramInventory &program_inventory;
  std::span<const ConSanFaultSite> fault_sites;
};

/// Fully rederived ordinary-acquire mutation target. Every pointer refers into
/// the immutable inventory supplied to the selector.
struct OrdinaryAcquireMutationTarget {
  const ConSanFaultSite *site = nullptr;
  const ConSanSyncEvent *load = nullptr;
  const ConSanSyncEvent *cache = nullptr;
  const ConSanSyncSequence *sequence = nullptr;
};

/// A complete exact two-member logical barrier and its two physical sites.
struct ExactBarrierDropPair {
  const ConSanSyncSequence *sequence = nullptr;
  const ConSanFaultSite *primary = nullptr;
  const ConSanFaultSite *companion = nullptr;
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
[[nodiscard]] bool
consan_fault_admits_cross_block_barrier_move(const ConSanBarrierMoveDestination &destination,
                                             const ConSanOptions &options);

[[nodiscard]] const ConSanFaultSite *
find_fault_site_by_identity(const ConSanFaultSelectionView &inventory, std::string_view identity,
                            ConSanFaultSiteKind kind);

[[nodiscard]] bool
consan_execution_owners_include_requested_kernel(std::span<const ConSanExecutionOwner> owners,
                                                 const ConSanFaultSelectionView &inventory,
                                                 std::string_view kernel_name_filter);

[[nodiscard]] const ConSanFaultSite *
select_fault_site_for_plan(const ConSanFaultSelectionView &inventory,
                           const ConSanFaultSelection &selection, ConSanFaultSiteKind kind);

[[nodiscard]] std::optional<OrdinaryAcquireMutationTarget>
select_ordinary_acquire_mutation_target(const ConSanFaultSelectionView &inventory,
                                        const ConSanFaultSelection &selection);

[[nodiscard]] ExactBarrierDropPairResolution
resolve_exact_barrier_drop_pair(const ConSanFaultSelectionView &inventory,
                                const ConSanFaultSelection &selection);

[[nodiscard]] ExactBarrierDropGroupResolution
resolve_exact_barrier_drop_group(const ConSanFaultSelectionView &inventory,
                                 const ConSanFaultSelection &selection);

[[nodiscard]] std::string_view
exact_barrier_drop_pair_issue_message(ExactBarrierDropPairIssue issue);

[[nodiscard]] std::string exact_barrier_drop_group_issue_message(
    ExactBarrierDropGroupIssue issue,
    ExactBarrierDropPairIssue member_issue = ExactBarrierDropPairIssue::None);

[[nodiscard]] std::string exact_barrier_drop_group_identity(const ExactBarrierDropGroup &group);

} // namespace rocjitsu
