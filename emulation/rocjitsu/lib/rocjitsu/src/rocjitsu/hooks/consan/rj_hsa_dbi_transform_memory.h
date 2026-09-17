// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/checked_byte_budget.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/patch/consan/consan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace rocjitsu::consan::hook {

enum class TransformOwnershipPhase : uint8_t {
  IncrementalPatch,
  CompositeIncrementalPatch,
  FinalValidation,
};

[[nodiscard]] inline constexpr const char *
transform_ownership_phase_name(TransformOwnershipPhase phase);

struct TransformOwnership {
  TransformOwnershipPhase phase;
  uint64_t input_image_copies;
  uint64_t maximum_image_copies;
};

struct TransformReservationEstimate {
  std::optional<TransformOwnership> ownership;
  uint64_t maximum_image_bytes;
  uint64_t reservation_bytes;

  [[nodiscard]] constexpr const char *phase_name() const {
    return ownership ? transform_ownership_phase_name(ownership->phase) : "none";
  }
  [[nodiscard]] constexpr uint64_t input_image_copies() const {
    return ownership ? ownership->input_image_copies : 0;
  }
  [[nodiscard]] constexpr uint64_t maximum_image_copies() const {
    return ownership ? ownership->maximum_image_copies : 0;
  }
};

[[nodiscard]] inline constexpr const char *
transform_ownership_phase_name(TransformOwnershipPhase phase) {
  using E = TransformOwnershipPhase;
  constexpr auto vocabulary =
      make_enum_vocabulary("unknown", enum_entry(E::IncrementalPatch, "incremental-patch"),
                           enum_entry(E::CompositeIncrementalPatch, "composite-incremental-patch"),
                           enum_entry(E::FinalValidation, "final-validation"));
  return vocabulary.name(phase).data();
}

/// Major-image ownership at each supported transform peak.
///
/// IncrementalPatch owns the hook's pristine staging image, the previous
/// result, a parser's exported ownership units, the patcher image, replacement
/// text, and the transactional grown image. Each parser owns its image, section
/// objects and vector slots, bounded section names, and conservatively charged
/// symbol- and metadata-derived state. Section payloads, section headers,
/// transient symbol-name characters, and metadata names are views into the
/// parser image.
///
/// The composite variant additionally retains the independently validated
/// mutation image while instrumenting it. That extra phase exists because the
/// SuperCollider's staged mutation image remains live while independently
/// parsing and instrumenting that image. Descriptor normalization sequences its
/// pristine and staged parsers and retains only compact name/offset maps between
/// them. FinalValidation owns pristine staging plus
/// the original parser's `kAmdGpuCodeObjectRetainedMajorImageUnits` units, and
/// the result plus the replacement parser's units and a descriptor probe.
/// ConSan releases its outer inventory parser before entering the incremental
/// ConSan pipeline, and every patch stage moves its emitted image.
///
/// AmdGpuCodeObject rejects aggregate viewed section payload extents or copied
/// section names larger than its backing image and symbol- and
/// metadata-derived state larger than its exported multi-unit budget.
/// Allocator bookkeeping is not represented by these major-image units. The
/// non-parser terms below keep the phase table mechanically coupled to the
/// parser's exported ownership bound.
inline constexpr uint64_t kIncrementalNonParserMaximumImageUnits = 4;
inline constexpr uint64_t kCompositeNonParserMaximumImageUnits = 5;
inline constexpr uint64_t kFinalOriginalNonParserInputImageUnits = 1;
inline constexpr uint64_t kFinalReplacementNonParserMaximumImageUnits = 2;

inline constexpr std::array<TransformOwnership, 3> kTransformOwnershipPhases = {{
    {TransformOwnershipPhase::IncrementalPatch, 1,
     kAmdGpuCodeObjectRetainedMajorImageUnits + kIncrementalNonParserMaximumImageUnits},
    {TransformOwnershipPhase::CompositeIncrementalPatch, 1,
     kAmdGpuCodeObjectRetainedMajorImageUnits + kCompositeNonParserMaximumImageUnits},
    {TransformOwnershipPhase::FinalValidation,
     kAmdGpuCodeObjectRetainedMajorImageUnits + kFinalOriginalNonParserInputImageUnits,
     kAmdGpuCodeObjectRetainedMajorImageUnits + kFinalReplacementNonParserMaximumImageUnits},
}};

/// Derived phase maxima used to place overflow-boundary tests at the table's
/// current limits without duplicating the selection algorithm.
[[nodiscard]] inline constexpr uint64_t transform_max_maximum_image_copies() {
  uint64_t maximum = 0;
  for (const TransformOwnership &ownership : kTransformOwnershipPhases)
    maximum = ownership.maximum_image_copies > maximum ? ownership.maximum_image_copies : maximum;
  return maximum;
}

[[nodiscard]] inline constexpr uint64_t transform_max_total_copies() {
  uint64_t maximum = 0;
  for (const TransformOwnership &ownership : kTransformOwnershipPhases) {
    const uint64_t total = ownership.input_image_copies + ownership.maximum_image_copies;
    maximum = total > maximum ? total : maximum;
  }
  return maximum;
}

[[nodiscard]] inline std::optional<uint64_t>
transform_phase_reservation_bytes(const TransformOwnership &ownership, uint64_t input_image_bytes,
                                  uint64_t maximum_image_bytes) {
  const auto input_reservation = byte_accounting::checked_allocation_charge(
      0, ownership.input_image_copies, input_image_bytes);
  if (!input_reservation)
    return std::nullopt;
  return byte_accounting::checked_allocation_charge(
      *input_reservation, ownership.maximum_image_copies, maximum_image_bytes);
}

[[nodiscard]] inline std::optional<TransformReservationEstimate>
transform_major_image_reservation(size_t input_image_bytes,
                                  const PatchedImageGrowthLimit &growth_policy) {
  static_assert(sizeof(size_t) == sizeof(uint64_t),
                "ConSan process admission requires a 64-bit size_t");
  const std::optional<size_t> maximum_growth =
      patched_image_growth_limit_bytes(growth_policy, input_image_bytes);
  if (!maximum_growth)
    return std::nullopt;
  const auto maximum_image_bytes = util::checked_add(input_image_bytes, *maximum_growth);
  if (!maximum_image_bytes)
    return std::nullopt;
  TransformReservationEstimate peak = {
      .ownership = std::nullopt,
      .maximum_image_bytes = static_cast<uint64_t>(*maximum_image_bytes),
      .reservation_bytes = 0,
  };
  for (const TransformOwnership &ownership : kTransformOwnershipPhases) {
    const std::optional<uint64_t> phase_reservation =
        transform_phase_reservation_bytes(ownership, static_cast<uint64_t>(input_image_bytes),
                                          static_cast<uint64_t>(*maximum_image_bytes));
    if (!phase_reservation)
      return std::nullopt;
    if (*phase_reservation > peak.reservation_bytes) {
      peak.ownership = ownership;
      peak.reservation_bytes = *phase_reservation;
    }
  }
  return peak;
}

} // namespace rocjitsu::consan::hook
