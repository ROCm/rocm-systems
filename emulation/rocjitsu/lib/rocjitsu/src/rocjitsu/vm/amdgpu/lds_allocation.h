// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// The VGPR and instruction offset form a 32-bit byte address within the DS allocation.
constexpr uint64_t ds_address(uint32_t allocation_base, uint32_t vgpr, uint32_t offset) {
  return uint64_t{allocation_base} + (vgpr + offset);
}

/// Complete elements within both the requested length and an exclusive bound.
/// Use element_bytes == 1 for byte-granular ranges that may retain element tails.
constexpr uint64_t whole_element_bytes_in_range(uint64_t address, uint64_t bytes,
                                                uint64_t element_bytes, uint64_t end) {
  if (element_bytes == 0)
    return 0;
  const uint64_t available = address < end ? end - address : 0;
  return std::min(bytes, available) / element_bytes * element_bytes;
}

/// Workgroup byte range captured when an instruction is routed to its LDS backing.
class LdsAllocationRange {
public:
  uint32_t begin;
  uint64_t end;

  static constexpr LdsAllocationRange from_size(uint32_t base, uint32_t size) {
    return {base, uint64_t{base} + size};
  }

  constexpr uint64_t retained_bytes(uint64_t address, uint64_t bytes,
                                    uint32_t element_bytes) const {
    return address < begin ? 0 : whole_element_bytes_in_range(address, bytes, element_bytes, end);
  }
};

} // namespace rocjitsu::amdgpu
