// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_BUFFER_ADDRESS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_BUFFER_ADDRESS_H_

/// @file Shared canonical addressing for 48-bit scalar and vector buffers.

#include <cstdint>

namespace rocjitsu::amdgpu::addr_calc {

/// Buffer descriptors store 48 address bits. Restore the canonical GPU virtual
/// address before consulting mappings installed by the kernel driver.
constexpr uint64_t buffer_virtual_address(uint64_t address) {
  constexpr uint64_t mask = (uint64_t{1} << 48) - 1;
  address &= mask;
  return (address & (uint64_t{1} << 47)) ? address | ~mask : address;
}

} // namespace rocjitsu::amdgpu::addr_calc

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_BUFFER_ADDRESS_H_
