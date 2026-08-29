// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_pipeline.h"

namespace rocjitsu {

/// Read-only diagnostic projection of lowerer-private proof artifacts.
///
/// This internal opt-in interface supports verbose development logs and
/// invariant tests. It is deliberately absent from TransformResult's public
/// production interface: no install, attribution, coverage, or retry decision
/// may depend on this projection.
struct ConSanTransformDebugReport {
  std::span<const ConSanFaultSite> fault_sites;
  std::span<const ConSanBarrierMoveDestination> barrier_move_destinations;
  std::span<const ConSanFaultMutationPlan> fault_plans;
  std::span<const ConSanCandidateResourcePlan> resource_plans;
  std::span<const ConSanCommittedLowering> committed_lowerings;
  std::span<const ConSanPatchInfo> patches;
};

[[nodiscard]] ConSanTransformDebugReport
consan_transform_debug_report(const TransformResult &result);

} // namespace rocjitsu
