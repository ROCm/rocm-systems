// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/result.h"

#include <span>
#include <string>

namespace rocjitsu::consan {

struct MetadataResources {
  uint32_t kernarg_bytes = 0;
  uint32_t kernarg_alignment = 0;
  uint32_t sgprs = 0;
  uint32_t vgprs = 0;
  uint32_t maximum_workgroup_size = 0;
};

/// @brief Shared image-level validation of the MVP AMDHSA note and argument ABI.
[[nodiscard]] FailureOr<MetadataResources> inspect_metadata(std::span<const uint8_t> image,
                                                            const KernelDescriptorInfo &kernel,
                                                            std::string *error = nullptr);

/// @brief Patch numeric resource values without moving allocated ELF storage.
/// @details Reject values that cannot fit the note's existing integer encoding.
[[nodiscard]] Result update_metadata(std::span<uint8_t> image, const KernelDescriptorInfo &kernel,
                                     const MetadataResources &resources,
                                     std::string *error = nullptr);

} // namespace rocjitsu::consan
