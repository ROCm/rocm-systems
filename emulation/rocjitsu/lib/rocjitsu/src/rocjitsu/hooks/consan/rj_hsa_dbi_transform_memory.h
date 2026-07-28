// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace rocjitsu::consan_hook {

/// Input-sized ownership at the largest supported ConSan overlap.
///
/// This covers the runtime reader, staged pristine storage, input parser image
/// and its bounded aggregate section storage, plus one additional pristine
/// validation/staging owner.
inline constexpr uint64_t kConSanTransformInputImageCopies = 5;

/// Maximum-image ownership at the largest supported ConSan overlap.
///
/// This covers the result or staged image, its parser/section storage, and
/// patcher or final-validation storage. AmdGpuCodeObject rejects aggregate
/// copied section payload larger than its backing image, so each parser and its
/// sections fit the image-equivalent terms above. Smaller analysis metadata and
/// allocator overhead are not represented by this admission unit.
inline constexpr uint64_t kConSanTransformMaximumImageCopies = 3;

[[nodiscard]] inline std::optional<uint64_t> consan_transform_major_image_reservation_bytes(
    size_t input_image_bytes, const rocjitsu::ConSanPatchedImageGrowthLimit &growth_policy) {
  static_assert(sizeof(size_t) == sizeof(uint64_t),
                "ConSan process admission requires a 64-bit size_t");
  const std::optional<size_t> maximum_growth =
      rocjitsu::consan_patched_image_growth_limit_bytes(growth_policy, input_image_bytes);
  if (!maximum_growth || *maximum_growth > std::numeric_limits<size_t>::max() - input_image_bytes)
    return std::nullopt;
  const size_t maximum_image_bytes = input_image_bytes + *maximum_growth;
  if (input_image_bytes > std::numeric_limits<uint64_t>::max() / kConSanTransformInputImageCopies ||
      maximum_image_bytes >
          std::numeric_limits<uint64_t>::max() / kConSanTransformMaximumImageCopies) {
    return std::nullopt;
  }
  const uint64_t input_reservation =
      static_cast<uint64_t>(input_image_bytes) * kConSanTransformInputImageCopies;
  const uint64_t maximum_image_reservation =
      static_cast<uint64_t>(maximum_image_bytes) * kConSanTransformMaximumImageCopies;
  if (input_reservation > std::numeric_limits<uint64_t>::max() - maximum_image_reservation) {
    return std::nullopt;
  }
  return input_reservation + maximum_image_reservation;
}

} // namespace rocjitsu::consan_hook
