// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Simulator-neutral helpers for encoding and decoding AMDGPU wait-counter
// immediates. Field placement follows the hardware layout, so a combined
// s_wait_loadcnt_dscnt immediate carries the memory counter in the high bits
// and dscnt in the low bits, matching how the emulator's wavefront reads it.

#pragma once

#include <cstdint>

namespace data_hazard_waitcnt {

constexpr uint32_t LEGACY_VMCNT_LOW_MASK = 0xF;
constexpr uint32_t LEGACY_VMCNT_HIGH_MASK = 0x30;
constexpr uint32_t LEGACY_VMCNT_HIGH_SHIFT = 10;
constexpr uint32_t LEGACY_LGKMCNT_SHIFT = 8;
constexpr uint32_t LEGACY_LGKMCNT_4BIT_MASK = 0xF;
constexpr uint32_t LEGACY_LGKMCNT_5BIT_MASK = 0x1F;
constexpr uint32_t SPLIT_WAIT_COUNTER_MASK = 0x3F;
constexpr uint32_t SPLIT_WAIT_DSCNT_SHIFT = 0;
constexpr uint32_t SPLIT_WAIT_MEM_SHIFT = 8;

struct LegacyWaitcntFields {
  uint32_t vmcnt;
  uint32_t lgkmcnt;
};

struct SplitWaitcntFields {
  uint32_t primary;
  uint32_t dscnt;
};

constexpr LegacyWaitcntFields decode_legacy_waitcnt(uint32_t simm16, uint32_t lgkmcnt_mask) {
  return {
      (simm16 & LEGACY_VMCNT_LOW_MASK) |
          ((simm16 >> LEGACY_VMCNT_HIGH_SHIFT) & LEGACY_VMCNT_HIGH_MASK),
      (simm16 >> LEGACY_LGKMCNT_SHIFT) & lgkmcnt_mask,
  };
}

constexpr SplitWaitcntFields decode_split_waitcnt(uint32_t simm16) {
  return {
      (simm16 >> SPLIT_WAIT_MEM_SHIFT) & SPLIT_WAIT_COUNTER_MASK,
      (simm16 >> SPLIT_WAIT_DSCNT_SHIFT) & SPLIT_WAIT_COUNTER_MASK,
  };
}

constexpr uint32_t encode_legacy_waitcnt(uint32_t vmcnt, uint32_t lgkmcnt, uint32_t lgkmcnt_mask) {
  return (vmcnt & LEGACY_VMCNT_LOW_MASK) |
         ((vmcnt & LEGACY_VMCNT_HIGH_MASK) << LEGACY_VMCNT_HIGH_SHIFT) |
         ((lgkmcnt & lgkmcnt_mask) << LEGACY_LGKMCNT_SHIFT);
}

constexpr uint32_t encode_split_waitcnt(uint32_t primary, uint32_t dscnt) {
  return ((primary & SPLIT_WAIT_COUNTER_MASK) << SPLIT_WAIT_MEM_SHIFT) |
         ((dscnt & SPLIT_WAIT_COUNTER_MASK) << SPLIT_WAIT_DSCNT_SHIFT);
}

} // namespace data_hazard_waitcnt
