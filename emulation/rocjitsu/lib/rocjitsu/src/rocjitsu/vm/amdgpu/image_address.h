// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu {
/// @brief Tiling identity shared with the device configuration exposed to ROCr.
struct ImageAddressConfig {
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID; ///< Addressing generation.
  uint32_t gb_addr_config = 0; ///< GB_ADDR_CONFIG value shared with KFD and DRM.
};

/// @brief Addressing shape of an image and its mip chain.
struct ImageLayout {
  uint32_t width = 1;  ///< Level-zero width in texels.
  uint32_t height = 1; ///< Level-zero height; one for 1D images.
  uint32_t depth = 1;  ///< Level-zero depth or the number of array layers.
  uint32_t pitch =
      0; ///< Encoded texel pitch for single-level linear/GFX9 tiled images.
         ///< Zero means no encoded pitch; mip chains and GFX10+ tiled modes use native packing.
  uint32_t element_bytes = 0; ///< Storage bytes per texel.
  uint32_t swizzle_mode = 0;  ///< Generation-specific descriptor swizzle encoding.
  uint32_t dimensions = 2;    ///< Spatial dimensions, excluding array layers.
  uint32_t mip_levels = 1;    ///< Number of levels in the backing allocation.
};

/// @brief Default tiling identity for a standalone CU without a device description.
ImageAddressConfig default_image_address_config(rj_code_arch_t arch);

/// @brief Resolve a single-sample thin-image view once before processing its texels.
class ImageAddressing {
public:
  /// @brief Return no value when the allocation layout is unsupported.
  static std::optional<ImageAddressing> prepare(const ImageAddressConfig &config,
                                                const ImageLayout &layout, uint32_t mip = 0);
  /// @brief Check coordinates against the selected mip view, preserving array layers.
  bool contains(uint32_t x, uint32_t y, uint32_t z) const;
  /// @brief Address a texel whose coordinates satisfy contains().
  uint64_t byte_offset(uint32_t x, uint32_t y, uint32_t z) const;

private:
  ImageAddressing() = default;

  ImageAddressConfig config_{};
  ImageLayout layout_{};
  uint32_t width_ = 1;  ///< Selected mip width in texels.
  uint32_t height_ = 1; ///< Selected mip height in texels.
  uint32_t depth_ = 1;  ///< Selected mip depth, or unchanged array layer count.
  uint32_t block_bits_ = 0;
  uint32_t block_width_ = 1;
  uint32_t block_height_ = 1;
  uint32_t pitch_ = 0; ///< Resolved texel row pitch.
  uint32_t tail_x_ = 0;
  uint32_t tail_y_ = 0;
  uint32_t tail_bits_ = 0;
  uint64_t offset_ = 0; ///< Selected mip storage-block offset in bytes.
  uint64_t slice_ = 0;  ///< Byte stride between thin slices or array layers.
};

} // namespace rocjitsu::amdgpu
