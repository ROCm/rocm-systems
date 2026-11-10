// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/image_address.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rocjitsu::amdgpu {

/// @brief Select the encoded border color for a descriptor channel or literal.
inline bool image_border_component_is_one(uint32_t color, uint32_t selector) {
  return selector == 1 || (selector >= 4 && (color == 2 || (color == 1 && selector == 7)));
}

/// @brief Depth comparison before spatial or mip filtering, with ISA compare-function order.
inline bool image_depth_comparison(uint32_t function, float reference, float depth) {
  switch (function) {
  case 1:
    return reference < depth;
  case 2:
    return reference == depth;
  case 3:
    return reference <= depth;
  case 4:
    return reference > depth;
  case 5:
    return reference != depth;
  case 6:
    return reference >= depth;
  case 7:
    return true;
  default:
    return false;
  }
}

/// @brief Ray inputs and resource controls captured before a BVH memory request.
struct BvhIntersectionAccess {
  struct Ray {
    std::array<float, 3> origin{}, direction{}, inverse{};
    float extent = 0;
    uint16_t flags = 0;
    Gfx11BvhNodeType node_type = Gfx11BvhNodeType::Triangle0;
  };
  std::array<Ray, 64> rays{};
  uint8_t grow = 0, sort = 0;
  bool sort_enabled = false, barycentrics = false, pointer_flags = false;
};

/// A partially resident texture records the requested mip and fetch outcome per lane.
/// A resident access leaves the shader-initialized feedback register untouched.
class ImageFeedback {
public:
  std::array<uint8_t, 64> levels{};
  uint32_t warning_threshold = 0;

  uint64_t warnings = 0;
  uint64_t misses = 0;
  bool tfe = false;
  bool lwe = false;
  bool checked = false;
  bool blend_prt = true;
  bool zero_sample_result(uint32_t lane) const {
    return !blend_prt && (misses & (uint64_t{1} << lane));
  }

  bool returns_status() const { return tfe || lwe; }
  uint64_t written_misses() const { return tfe ? misses : 0; }
  uint64_t written_warnings() const { return lwe ? warnings & ~misses : 0; }

  void record_sample(uint32_t lane, bool mipmapped, double requested_lod, uint8_t returned_level) {
    // Both causes encode the first selected mip. GFX11/12 point sampling
    // reports level 1 for a 0.75 request, while linear mip filtering reports level 0.
    levels[lane] = returned_level;
    if (lwe && mipmapped && requested_lod * 256 < warning_threshold) {
      warnings |= uint64_t{1} << lane;
    }
  }
};

/// @brief Surface metadata needed to materialize image clears in the memory pipeline.
struct ImageMetadataAccess {
  /// First unfinished tap/lane pair after a temporarily unavailable VM access.
  uint32_t next_transfer = 0;
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
struct GraphicsSampleAccess {
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
    uint64_t missing_lanes = 0;
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
