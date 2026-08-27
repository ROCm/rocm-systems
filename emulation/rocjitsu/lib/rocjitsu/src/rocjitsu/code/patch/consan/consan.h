// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan.h
/// @brief Shared ConSan contracts and prototype mechanism data.
///
/// Production transformation enters through `consan_pipeline.h`. No library
/// header exposes the mutable compatibility result entry used by explicitly
/// mechanism-level tests while the lowerer is decomposed.

#pragma once

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "rocjitsu/code/patch/consan/consan_access_shape.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/planning_work.h"
#include "rocjitsu/code/rj_code.h"

#include "rocjitsu/code/patch/consan/consan_options.h.inc"

#include "rocjitsu/code/patch/consan/consan_request_contract.h.inc"

#include "rocjitsu/code/patch/consan/consan_code_object_types.h.inc"

#include "rocjitsu/code/patch/consan/consan_site_identity.h.inc"

#include "rocjitsu/code/patch/consan/consan_resource_types.h.inc"

#include "rocjitsu/code/patch/consan/consan_fault_sync_types.h.inc"

#include "rocjitsu/code/patch/consan/consan_program_inventory.h.inc"

#include "rocjitsu/code/patch/consan/consan_observation_plan.h.inc"

#include "rocjitsu/code/patch/consan/consan_result.h.inc"

/// Bridge runtime option identities to the flattened public capability
/// matrix. Invalid or disabled option combinations are outside that contract.
[[nodiscard]] constexpr std::optional<ConSanCapabilityEngine>
consan_capability_engine(ConSanFlavor flavor, ConSanMoiEngine moi_engine) {
  switch (flavor) {
  case ConSanFlavor::SuperCollider:
    return ConSanCapabilityEngine::SuperCollider;
  case ConSanFlavor::Moi:
    switch (moi_engine) {
    case ConSanMoiEngine::RecordReplay:
      return ConSanCapabilityEngine::RecordReplay;
    case ConSanMoiEngine::Sampled:
      return ConSanCapabilityEngine::Sampled;
    case ConSanMoiEngine::InlineShadow:
      return ConSanCapabilityEngine::InlineShadow;
    }
    return std::nullopt;
  case ConSanFlavor::None:
    return std::nullopt;
  }
  return std::nullopt;
}

/// Return the active VGPR-bank role mask at an instruction in a gfx1250
/// container. The scan is bounded by the owning container entry.
[[nodiscard]] std::optional<uint16_t>
consan_gfx1250_vgpr_msb_mode_at(std::span<const uint8_t> bytes, uint64_t text_file_offset,
                                uint64_t container_entry_text_offset, uint64_t site_file_offset);

/// Return whether SuperCollider's current lowerer can implement one normalized
/// access selected from the shared program inventory. This is a lowering
/// capability query, not semantic policy: callers still decide address-space
/// provenance, aliasing, and whether the access family was requested.
[[nodiscard]] bool consan_supercollider_supports_access(const ConSanAccessInventorySite &access,
                                                        ConSanFlatProvenanceMode mode,
                                                        rj_code_arch_t arch);

} // namespace rocjitsu
