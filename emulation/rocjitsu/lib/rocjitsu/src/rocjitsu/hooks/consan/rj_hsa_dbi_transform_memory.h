// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace rocjitsu::consan_hook {

enum class ConSanTransformOwnershipPhase : uint8_t {
  IncrementalPatch,
  CompositeIncrementalPatch,
  FinalValidation,
};

struct ConSanTransformOwnership {
  ConSanTransformOwnershipPhase phase;
  uint64_t input_image_copies;
  uint64_t maximum_image_copies;
};

/// Major-image ownership at each supported transform peak.
///
/// IncrementalPatch owns the hook's pristine staging image, the previous
/// result, a parser image plus section-header storage, bounded section payload,
/// and bounded section names, the patcher image, replacement text, and the
/// transactional grown image. The composite variant additionally retains the
/// independently validated mutation image while instrumenting it.
/// FinalValidation owns pristine staging plus those four original-parser
/// units, and the result plus four replacement-parser units and a descriptor
/// probe. ConSan releases its outer inventory parser before entering the
/// incremental MOI pipeline, and every patch stage moves its emitted image.
///
/// AmdGpuCodeObject rejects aggregate copied section payload and aggregate
/// section-name bytes larger than its backing image. Smaller analysis metadata
/// and allocator overhead are not represented by these major-image units.
inline constexpr std::array<ConSanTransformOwnership, 3> kConSanTransformOwnershipPhases = {{
    {ConSanTransformOwnershipPhase::IncrementalPatch, 1, 8},
    {ConSanTransformOwnershipPhase::CompositeIncrementalPatch, 1, 9},
    {ConSanTransformOwnershipPhase::FinalValidation, 5, 6},
}};

[[nodiscard]] inline std::optional<uint64_t>
consan_transform_phase_reservation_bytes(const ConSanTransformOwnership &ownership,
                                         uint64_t input_image_bytes, uint64_t maximum_image_bytes) {
  if ((ownership.input_image_copies != 0 &&
       input_image_bytes > std::numeric_limits<uint64_t>::max() / ownership.input_image_copies) ||
      (ownership.maximum_image_copies != 0 &&
       maximum_image_bytes >
           std::numeric_limits<uint64_t>::max() / ownership.maximum_image_copies)) {
    return std::nullopt;
  }
  const uint64_t input_reservation = input_image_bytes * ownership.input_image_copies;
  const uint64_t maximum_image_reservation = maximum_image_bytes * ownership.maximum_image_copies;
  if (input_reservation > std::numeric_limits<uint64_t>::max() - maximum_image_reservation)
    return std::nullopt;
  return input_reservation + maximum_image_reservation;
}

[[nodiscard]] inline std::optional<uint64_t> consan_transform_major_image_reservation_bytes(
    size_t input_image_bytes, const rocjitsu::ConSanPatchedImageGrowthLimit &growth_policy) {
  static_assert(sizeof(size_t) == sizeof(uint64_t),
                "ConSan process admission requires a 64-bit size_t");
  const std::optional<size_t> maximum_growth =
      rocjitsu::consan_patched_image_growth_limit_bytes(growth_policy, input_image_bytes);
  if (!maximum_growth || *maximum_growth > std::numeric_limits<size_t>::max() - input_image_bytes)
    return std::nullopt;
  const uint64_t maximum_image_bytes = static_cast<uint64_t>(input_image_bytes + *maximum_growth);
  uint64_t peak_reservation = 0;
  for (const ConSanTransformOwnership &ownership : kConSanTransformOwnershipPhases) {
    const std::optional<uint64_t> phase_reservation = consan_transform_phase_reservation_bytes(
        ownership, static_cast<uint64_t>(input_image_bytes), maximum_image_bytes);
    if (!phase_reservation)
      return std::nullopt;
    if (*phase_reservation > peak_reservation)
      peak_reservation = *phase_reservation;
  }
  return peak_reservation;
}

} // namespace rocjitsu::consan_hook
