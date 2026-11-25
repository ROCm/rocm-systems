// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/generated/shared/operations.h"

#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu {

/// Addressing shared by both elements of a two-address DS instruction.
struct Ds2Addressing {
  uint32_t element_bytes;
  bool stride64;

  /// The scaled offset of an eight-bit element index can exceed 16 bits.
  [[nodiscard]] constexpr uint32_t byte_offset(uint8_t index) const {
    return index * element_bytes * (stride64 ? 64u : 1u);
  }
};

/// @brief Describe the address units of a decoded two-address DS operation.
[[nodiscard]] constexpr std::optional<Ds2Addressing> ds2_addressing(Operation operation) {
  switch (operation) {
  case Operation::DS_READ2_B32:
  case Operation::DS_WRITE2_B32:
  case Operation::DS_WRXCHG2_RTN_B32:
  case Operation::DS_LOAD_2ADDR_B32:
  case Operation::DS_STORE_2ADDR_B32:
  case Operation::DS_STOREXCHG_2ADDR_RTN_B32:
    return Ds2Addressing{.element_bytes = 4, .stride64 = false};
  case Operation::DS_READ2ST64_B32:
  case Operation::DS_WRITE2ST64_B32:
  case Operation::DS_WRXCHG2ST64_RTN_B32:
  case Operation::DS_LOAD_2ADDR_STRIDE64_B32:
  case Operation::DS_STORE_2ADDR_STRIDE64_B32:
  case Operation::DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32:
    return Ds2Addressing{.element_bytes = 4, .stride64 = true};
  case Operation::DS_READ2_B64:
  case Operation::DS_WRITE2_B64:
  case Operation::DS_WRXCHG2_RTN_B64:
  case Operation::DS_LOAD_2ADDR_B64:
  case Operation::DS_STORE_2ADDR_B64:
  case Operation::DS_STOREXCHG_2ADDR_RTN_B64:
    return Ds2Addressing{.element_bytes = 8, .stride64 = false};
  case Operation::DS_READ2ST64_B64:
  case Operation::DS_WRITE2ST64_B64:
  case Operation::DS_WRXCHG2ST64_RTN_B64:
  case Operation::DS_LOAD_2ADDR_STRIDE64_B64:
  case Operation::DS_STORE_2ADDR_STRIDE64_B64:
  case Operation::DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64:
    return Ds2Addressing{.element_bytes = 8, .stride64 = true};
  default:
    return std::nullopt;
  }
}

} // namespace rocjitsu::amdgpu
