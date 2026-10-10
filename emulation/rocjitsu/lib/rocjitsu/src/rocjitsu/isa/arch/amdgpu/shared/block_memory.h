// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_BLOCK_MEMORY_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_BLOCK_MEMORY_H_

#include <cstdint>
#include <limits>

namespace rocjitsu::amdgpu {

inline constexpr uint32_t kBlockDwordCount = 32;
inline constexpr uint32_t kUnmaskedBlockDwordMask = std::numeric_limits<uint32_t>::max();

/// M0 selects block DWORDs; ordinary results beyond that span are not masked.
constexpr bool block_dword_enabled(uint32_t mask, uint32_t word) {
  return word >= kBlockDwordCount || (mask & (uint32_t{1} << word)) != 0;
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_BLOCK_MEMORY_H_
