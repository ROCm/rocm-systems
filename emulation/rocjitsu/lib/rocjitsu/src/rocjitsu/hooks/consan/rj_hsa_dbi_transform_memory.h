// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace rocjitsu::consan_hook {

/// Whole-image-equivalent multiplier for the largest supported ConSan
/// ownership overlap.
///
/// The final-validation peak can retain the runtime reader, staged input,
/// result image, both parser images and their section copies, plus one
/// validation copy. The transform peak instead retains reader/staged storage,
/// one parser image and its sections, patcher storage, replacement text,
/// transactional storage, and insertion padding. Reserving eight copies of
/// the maximum admitted image covers both ownership shapes. Smaller analysis
/// metadata and allocator overhead are not represented by this admission unit.
inline constexpr uint64_t kConSanTransformMajorImageCopies = 8;

[[nodiscard]] inline std::optional<uint64_t> consan_transform_major_image_reservation_bytes(
    size_t input_image_bytes, const rocjitsu::ConSanPatchedImageGrowthLimit &growth_policy) {
  static_assert(sizeof(size_t) == sizeof(uint64_t),
                "ConSan process admission requires a 64-bit size_t");
  const std::optional<size_t> maximum_growth =
      rocjitsu::consan_patched_image_growth_limit_bytes(growth_policy, input_image_bytes);
  if (!maximum_growth || *maximum_growth > std::numeric_limits<size_t>::max() - input_image_bytes)
    return std::nullopt;
  const size_t maximum_image_bytes = input_image_bytes + *maximum_growth;
  if (maximum_image_bytes >
      std::numeric_limits<uint64_t>::max() / kConSanTransformMajorImageCopies) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(maximum_image_bytes) * kConSanTransformMajorImageCopies;
}

} // namespace rocjitsu::consan_hook
