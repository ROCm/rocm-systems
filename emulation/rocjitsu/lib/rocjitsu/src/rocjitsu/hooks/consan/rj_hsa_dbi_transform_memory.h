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

struct ConSanTransformReservationEstimate {
  ConSanTransformOwnership ownership;
  uint64_t maximum_image_bytes;
  uint64_t reservation_bytes;
};

[[nodiscard]] inline constexpr const char *
consan_transform_ownership_phase_name(ConSanTransformOwnershipPhase phase) {
  switch (phase) {
  case ConSanTransformOwnershipPhase::IncrementalPatch:
    return "incremental-patch";
  case ConSanTransformOwnershipPhase::CompositeIncrementalPatch:
    return "composite-incremental-patch";
  case ConSanTransformOwnershipPhase::FinalValidation:
    return "final-validation";
  }
  return "unknown";
}

/// Major-image ownership at each supported transform peak.
///
/// IncrementalPatch owns the hook's pristine staging image, the previous
/// result, a parser's six units, the patcher image, replacement text, and the
/// transactional grown image. Each parser owns its image, section objects and
/// headers, bounded payload, bounded section names, at most two retained copies
/// of bounded symbol names. Section headers, transient symbol names, and
/// metadata names are views into the parser image so temporary full collections
/// do not overlap the retained representations.
///
/// The composite variant additionally retains the independently validated
/// mutation image while instrumenting it. That extra phase exists because the
/// SuperCollider flat tail currently overlaps its outer parser with the parser
/// for the mutated image; reducing both sides to compact name/descriptor maps
/// would be the place to remove it. FinalValidation owns pristine staging plus
/// the six original-parser units, and the result plus six replacement-
/// parser units and a descriptor probe. ConSan releases its outer inventory
/// parser before entering the incremental MOI pipeline, and every patch stage
/// moves its emitted image.
///
/// AmdGpuCodeObject rejects aggregate copied section payload and aggregate
/// section-name or symbol-name bytes larger than its backing image. Smaller
/// non-string analysis metadata and allocator overhead are not represented by
/// these major-image units.
inline constexpr std::array<ConSanTransformOwnership, 3> kConSanTransformOwnershipPhases = {{
    {ConSanTransformOwnershipPhase::IncrementalPatch, 1, 10},
    {ConSanTransformOwnershipPhase::CompositeIncrementalPatch, 1, 11},
    {ConSanTransformOwnershipPhase::FinalValidation, 7, 8},
}};

[[nodiscard]] inline constexpr uint64_t consan_transform_max_maximum_image_copies() {
  uint64_t maximum = 0;
  for (const ConSanTransformOwnership &ownership : kConSanTransformOwnershipPhases)
    maximum = ownership.maximum_image_copies > maximum ? ownership.maximum_image_copies : maximum;
  return maximum;
}

[[nodiscard]] inline constexpr uint64_t consan_transform_max_total_copies() {
  uint64_t maximum = 0;
  for (const ConSanTransformOwnership &ownership : kConSanTransformOwnershipPhases) {
    const uint64_t total = ownership.input_image_copies + ownership.maximum_image_copies;
    maximum = total > maximum ? total : maximum;
  }
  return maximum;
}

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

[[nodiscard]] inline std::optional<ConSanTransformReservationEstimate>
consan_transform_major_image_reservation(
    size_t input_image_bytes, const rocjitsu::ConSanPatchedImageGrowthLimit &growth_policy) {
  static_assert(sizeof(size_t) == sizeof(uint64_t),
                "ConSan process admission requires a 64-bit size_t");
  const std::optional<size_t> maximum_growth =
      rocjitsu::consan_patched_image_growth_limit_bytes(growth_policy, input_image_bytes);
  if (!maximum_growth || *maximum_growth > std::numeric_limits<size_t>::max() - input_image_bytes)
    return std::nullopt;
  const uint64_t maximum_image_bytes = static_cast<uint64_t>(input_image_bytes + *maximum_growth);
  ConSanTransformReservationEstimate peak = {
      .ownership = kConSanTransformOwnershipPhases.front(),
      .maximum_image_bytes = maximum_image_bytes,
      .reservation_bytes = 0,
  };
  for (const ConSanTransformOwnership &ownership : kConSanTransformOwnershipPhases) {
    const std::optional<uint64_t> phase_reservation = consan_transform_phase_reservation_bytes(
        ownership, static_cast<uint64_t>(input_image_bytes), maximum_image_bytes);
    if (!phase_reservation)
      return std::nullopt;
    if (*phase_reservation > peak.reservation_bytes) {
      peak.ownership = ownership;
      peak.reservation_bytes = *phase_reservation;
    }
  }
  return peak;
}

[[nodiscard]] inline std::optional<uint64_t> consan_transform_major_image_reservation_bytes(
    size_t input_image_bytes, const rocjitsu::ConSanPatchedImageGrowthLimit &growth_policy) {
  const std::optional<ConSanTransformReservationEstimate> estimate =
      consan_transform_major_image_reservation(input_image_bytes, growth_policy);
  if (!estimate)
    return std::nullopt;
  return estimate->reservation_bytes;
}

} // namespace rocjitsu::consan_hook
