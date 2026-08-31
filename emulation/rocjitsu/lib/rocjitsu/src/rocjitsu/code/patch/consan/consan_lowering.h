// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_lowering.h
/// @brief Typed output boundary for ConSan's internal native lowerer.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_lowering_types.h"

namespace rocjitsu {

/// Immutable observation input consumed by native resource solving.
///
/// The transaction assembles this value after inventory and before runtime
/// binding. A lowerer may consume it but cannot append policy fragments or
/// initialize a different coverage ledger.
struct ConSanLoweringObservation {
  ConSanCoverageLedger initial_coverage;

  [[nodiscard]] const ConSanObservationPlan &plan() const {
    return initial_coverage.observation_plan();
  }

  [[nodiscard]] bool well_formed() const {
    return plan().valid() && initial_coverage == ConSanCoverageLedger(plan());
  }
};

/// Lower one code object to production-owned static artifacts. Mutable working
/// state cannot cross this boundary while the option input is decomposed.
[[nodiscard]] ConSanTransformArtifacts
lower_consan(std::span<const uint8_t> code_object_bytes, const ConSanOptions &options,
             ConSanLoweringExecution *execution = nullptr,
             ConSanLoweringExtent extent = ConSanLoweringExtent::Complete,
             const ConSanLoweringObservation *observation = nullptr);

} // namespace rocjitsu
