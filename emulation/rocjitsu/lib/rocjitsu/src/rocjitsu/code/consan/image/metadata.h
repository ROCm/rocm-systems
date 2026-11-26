// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/result.h"

#include <array>
#include <span>
#include <string>

namespace rocjitsu::consan {

struct MetadataResources {
  uint32_t sgprs = 0;
  uint32_t vgprs = 0;
  uint32_t maximum_workgroup_size = 0;
  uint32_t kernarg_alignment = 0;
  std::array<uint32_t, 3> required_workgroup_size{};
  bool uses_dynamic_stack = false;
};

enum class MetadataPolicy { Selected, Native };

/// @brief Read structurally consistent original AMDHSA argument/resource metadata.
/// @details Instrumentation eligibility is checked by the selected-entry consumer.
[[nodiscard]] FailureOr<MetadataResources>
inspect_metadata(std::span<const uint8_t> image, const KernelDescriptorInfo &kernel,
                 std::string *error = nullptr, MetadataPolicy policy = MetadataPolicy::Selected);

} // namespace rocjitsu::consan
