// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_lowering_types.h
/// @brief Narrow execution contract shared by ConSan orchestration and lowering.

#pragma once

#include <cstdint>

namespace rocjitsu {

/// Select how far one native lowering pass may execute.
enum class ConSanLoweringExtent : uint8_t {
  Complete,
  ThroughProgramInventory,
  ThroughObservationPlan,
};

/// Native-stage execution facts returned to the authoritative transaction.
///
/// Counts, rather than booleans, preserve composite and resumed mutation
/// passes and make it impossible for the public pipeline to invent a stage
/// from the final artifact shape. Internal validation may re-inventory
/// pristine bytes for an independent proof; only top-level lowering passes
/// call these methods.
struct ConSanLoweringExecution {
  uint32_t program_inventory_passes = 0;
  uint32_t observation_plan_passes = 0;
  uint32_t resource_solving_and_lowering_passes = 0;
  uint32_t final_validation_passes = 0;

  void note_program_inventory() { ++program_inventory_passes; }
  void note_observation_plan() { ++observation_plan_passes; }
  void note_resource_solving_and_lowering() { ++resource_solving_and_lowering_passes; }
  void note_final_validation() { ++final_validation_passes; }

  void append(const ConSanLoweringExecution &other) {
    program_inventory_passes += other.program_inventory_passes;
    observation_plan_passes += other.observation_plan_passes;
    resource_solving_and_lowering_passes += other.resource_solving_and_lowering_passes;
    final_validation_passes += other.final_validation_passes;
  }
};

} // namespace rocjitsu
