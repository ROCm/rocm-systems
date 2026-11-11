// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace rocjitsu::amdgpu {

/// A contiguous run of LDS bytes. Ranges can end partway through an element.
struct LdsByteRange {
  uint32_t address = 0; ///< Absolute CU LDS byte address.
  uint32_t bytes = 0;

  bool operator==(const LdsByteRange &) const = default;
};

} // namespace rocjitsu::amdgpu
