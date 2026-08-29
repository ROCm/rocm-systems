// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_fault_selection.h
/// @brief Exact, ownership-aware selection of ConSan mutation sites.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstdint>
#include <optional>
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

/// Fully rederived ordinary-acquire mutation target. Every pointer refers into
/// the immutable artifacts supplied to the selector.
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

[[nodiscard]] const ConSanFaultSite *
find_fault_site_by_identity(const ConSanTransformArtifacts &result, std::string_view identity,
                            ConSanFaultSiteKind kind);

[[nodiscard]] bool
consan_execution_owners_include_requested_kernel(std::span<const ConSanExecutionOwner> owners,
                                                 const ConSanTransformArtifacts &result,
                                                 std::string_view kernel_name_filter);

[[nodiscard]] const ConSanFaultSite *
select_fault_site_for_plan(const ConSanTransformArtifacts &result,
                           const ConSanFaultSelection &selection, ConSanFaultSiteKind kind);

[[nodiscard]] std::optional<OrdinaryAcquireMutationTarget>
select_ordinary_acquire_mutation_target(const ConSanTransformArtifacts &result,
                                        const ConSanFaultSelection &selection);

[[nodiscard]] std::optional<ExactBarrierDropPair>
resolve_exact_barrier_drop_pair(const ConSanTransformArtifacts &result,
                                const ConSanFaultSelection &selection, std::string *reason);

[[nodiscard]] std::optional<ExactBarrierDropGroup>
resolve_exact_barrier_drop_group(const ConSanTransformArtifacts &result,
                                 const ConSanFaultSelection &selection, std::string *reason);

[[nodiscard]] std::string exact_barrier_drop_group_identity(const ExactBarrierDropGroup &group);

} // namespace rocjitsu
