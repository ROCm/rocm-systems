// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_lowering.h
/// @brief Typed output boundary for ConSan's internal native lowerer.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

namespace rocjitsu {

/// Native-stage execution facts returned to the authoritative transaction.
///
/// Counts, rather than booleans, preserve composite mutation passes and make
/// it impossible for the public pipeline to invent a stage from the final
/// artifact shape. Internal validation may re-inventory pristine bytes for an
/// independent proof; only top-level lowering passes call these methods.
struct ConSanLoweringExecution {
  uint32_t program_inventory_passes = 0;
  uint32_t observation_plan_passes = 0;
  uint32_t resource_solving_and_lowering_passes = 0;
  uint32_t final_validation_passes = 0;

  void note_program_inventory() { ++program_inventory_passes; }
  void note_observation_plan() { ++observation_plan_passes; }
  void note_resource_solving_and_lowering() { ++resource_solving_and_lowering_passes; }
  void note_final_validation() { ++final_validation_passes; }
};

/// Lower one code object to production-owned static artifacts. Mutable working
/// state cannot cross this boundary while the option input is decomposed.
[[nodiscard]] ConSanTransformArtifacts lower_consan(std::span<const uint8_t> code_object_bytes,
                                                    const ConSanOptions &options,
                                                    ConSanLoweringExecution *execution = nullptr);

} // namespace rocjitsu
