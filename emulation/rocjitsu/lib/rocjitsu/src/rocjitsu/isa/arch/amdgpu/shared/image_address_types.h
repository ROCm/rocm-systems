// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief LOD input mode of a qualified sampling instruction.
enum class ImageSampleMode { Implicit, Zero, Explicit, Bias, Derivatives, Derivatives16 };

/// Encoded controls selecting active and derivative coordinates for RDNA sampler operations.
struct ImageSampleSources {
  uint16_t sampler;
  uint8_t dim;
  ImageSampleMode mode;
  bool query;
  bool linear_candidate;
  bool unorm;
  bool offset;
  bool compare;
  bool clamp;
  bool gather;
  bool lod_warning = false;
  bool filtered_feedback_candidate =
      false;                // Full resource with a data result and a supported form.
  bool texel_fault = false; ///< Encoded TFE bit.
};

/// @brief Candidate wave-relative VGPR selectors for image coordinates.
/// @details Printing and dependency metadata consume the prefix selected by
/// the instruction's DIM and A16 fields. Linear execution may consume a
/// shorter prefix selected by the resource type after checking that DIM
/// supplies enough coordinates. Entries outside a caller's prefix need not
/// name valid registers.
using ImageAddressRegisters = std::array<uint32_t, 13>;

} // namespace rocjitsu::amdgpu
