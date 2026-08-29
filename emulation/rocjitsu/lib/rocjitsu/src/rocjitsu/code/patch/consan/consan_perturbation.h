// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_perturbation.h
/// @brief Typed SuperCollider perturbation planning and mutation boundary.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

class AmdGpuCodeObject;

struct CarriedPerturbationPlan {
  std::string source_candidate_identity;
  std::string source_sequence_identity;
  std::string container_name;
  bool in_kernel = true;
  uint32_t basic_block_index = 0;
  std::string anchor_event_identity;
  std::vector<std::string> ordered_member_identities;
  uint64_t source_anchor_text_offset = 0;
  uint64_t translated_anchor_text_offset = 0;
  uint32_t anchor_size = 0;
  std::optional<uint64_t> source_owner_descriptor_file_offset;
  ConSanPerturbationKind kind = ConSanPerturbationKind::None;
  ConSanPerturbationEdge edge = ConSanPerturbationEdge::Release;
  uint32_t sleep_imm = 0;
  bool overlaps_atomic_mutation = false;
  bool removed_cache_boundary = false;
  bool sequence_semantics_weakened = false;
};

void build_perturbation_candidate_inventory(const ProgramInventory &program_inventory,
                                            ConSanPerturbationPlanningState &planning);

void build_perturbation_plan(const ConSanOptions &options,
                             ConSanPerturbationPlanningState &planning,
                             ConSanTransformArtifacts &result,
                             std::span<const CarriedPerturbationPlan> carried_plans = {});

void try_apply_perturbation_patches(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                    const ConSanOptions &options,
                                    const ConSanPerturbationPlanningState &planning,
                                    ConSanTransformArtifacts &result);

} // namespace rocjitsu
