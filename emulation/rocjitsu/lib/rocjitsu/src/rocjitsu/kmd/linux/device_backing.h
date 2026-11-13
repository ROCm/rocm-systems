// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file device_backing.h
/// @brief Identifiers for GPU device memory in native and shared-RAM backing.

#include <cstdint>
#include <string_view>

namespace rocjitsu::kmd {

inline constexpr std::string_view kDeviceBackingNamePrefix = "rocjitsu_alloc_renderD";

/// @brief Persistent GPU classification in System V VMA names after IPC_RMID.
/// @details The low slot bits select a transient creation key, released after
/// the keeper attaches and marks the segment for removal. Exhausting these keys
/// falls back to IPC_PRIVATE: token metadata retains device classification, but
/// the VMA name can no longer identify the segment for ROCr GPU core dumps.
inline constexpr uint32_t kDeviceTokenKeyPrefix = 0x524a4700;
inline constexpr unsigned kDeviceTokenSlotBits = 8;
inline constexpr uint32_t kDeviceTokenSlotCount = uint32_t{1} << kDeviceTokenSlotBits;
inline constexpr unsigned kDeviceTokenSlotHexDigits = kDeviceTokenSlotBits / 4;
static_assert(kDeviceTokenSlotBits % 4 == 0);
static_assert((kDeviceTokenKeyPrefix & (kDeviceTokenSlotCount - 1)) == 0);

} // namespace rocjitsu::kmd
