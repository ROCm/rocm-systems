// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rocjitsu::amdgpu {

/// @brief Surface metadata needed to materialize image clears in the memory pipeline.
struct ImageMetadataAccess {
  uint64_t base = 0, metadata = 0, slice_size = 0;
  uint32_t width = 0, height = 0, swizzle = 0;
  bool pipe_aligned = true;
  bool depth = false;
  uint32_t mip_levels = 1;
  /// Absolute mip selected by each lane, or by each filter tap when sampling.
  std::array<uint8_t, 64> levels{};
  std::vector<std::array<uint8_t, 64>> tap_levels;
  /// Per-lane x in bits 0-15 and y in bits 16-31.
  std::array<uint32_t, 64> coordinates{};
  /// Per-lane absolute array layer, including the descriptor view start.
  std::array<uint32_t, 64> layers{};
};

/// @brief Texel requests and interpolation weights captured at sample issue time.
struct ImageSampleAccess {
  struct Comparison {
    std::array<float, 64> references{};
    uint32_t function = 0;
  };
  std::unique_ptr<Comparison> comparison;
  bool gather = false;
  bool horizontal = false;
  static constexpr size_t kMaxFilters = 16;
  /// Two mips, four bilinear taps, and three texels at a cube corner.
  static constexpr size_t kMaxTaps = kMaxFilters * 2 * 4 * 3;
  struct Tap {
    std::array<uint64_t, 64> addresses{};
    /// Per-lane x in bits 0-15 and y in bits 16-31.
    std::array<uint32_t, 64> coordinates{};
    /// Per-lane absolute array layer, including the descriptor view start.
    std::array<uint32_t, 64> layers{};
    uint64_t lane_mask = 0;
  };
  struct Filter {
    /// XY interpolation fractions indexed by lane, mip, then axis.
    std::array<std::array<std::array<float, 2>, 2>, 64> fractions{};
    /// Bits identify bilinear taps averaged from three cube corner texels.
    std::array<std::array<uint8_t, 2>, 64> cube_corners{};
  };
  /// Reserve two mips of four bilinear taps; larger footprints grow on demand.
  std::vector<Tap> taps{8};
  /// Ordinary bilinear and trilinear requests retain one filter.
  std::vector<Filter> filters{1};
  /// Number of anisotropic footprint samples requested by each lane.
  std::array<uint32_t, 64> filter_counts{};
  std::array<float, 64> mip_fractions{};
  uint32_t tap_count = 1;
  /// Texel requests per footprint sample, including both mips and cube corners.
  uint32_t taps_per_filter = 1;
  /// One texel normally, or three when a bilinear tap crosses a cube corner.
  uint32_t texels_per_tap = 1;
  uint32_t border_color = 0;
};

} // namespace rocjitsu::amdgpu
