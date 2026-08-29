// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_sync_analysis.h
/// @brief Semantic synchronization, fault-site, and owner-association boundary.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstdint>
#include <span>

namespace rocjitsu {

class AmdGpuCodeObject;
class Decoder;

/// Build all synchronization semantics needed by the selected request and
/// publish one immutable inventory view. Internal CFGs, indexes, association
/// passes, and target-form screening do not cross this boundary.
[[nodiscard]] bool analyze_consan_semantic_inventory(std::span<const uint8_t> code_object_bytes,
                                                     const AmdGpuCodeObject &code_object,
                                                     Decoder &decoder, rj_code_arch_t arch,
                                                     const ConSanOptions &options,
                                                     ProgramInventoryBuilder &inventory_builder,
                                                     ConSanPerturbationPlanningState &perturbation,
                                                     ConSanTransformArtifacts &result);

} // namespace rocjitsu
