// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file waitcnt_translator.h
/// @brief Decode and encode wait-counter instructions for semantic lowering.

#pragma once

#include <cstdint>
#include <vector>

namespace rocjitsu {

/// @brief Decoded wait-counter values from a GFX9-style s_waitcnt simm16 field.
struct WaitcntValues {
  uint8_t vmcnt = 0x3F;   ///< VM count (loads + stores on GFX9). Sentinel: 0x3F.
  uint8_t lgkmcnt = 0x0F; ///< LDS/GDS/Kmem count. Sentinel: 0x0F.
  uint8_t expcnt = 0x07;  ///< Export count. Sentinel: 0x07.
};

/// @brief Decode a GFX9-style s_waitcnt simm16 field into individual counters.
[[nodiscard]] WaitcntValues decode_waitcnt_gfx9(uint16_t simm16);

/// @brief Encode wait-counter values as GFX12 split s_wait_* instruction words.
[[nodiscard]] std::vector<uint32_t> encode_waitcnt_gfx12(const WaitcntValues &vals);

} // namespace rocjitsu
