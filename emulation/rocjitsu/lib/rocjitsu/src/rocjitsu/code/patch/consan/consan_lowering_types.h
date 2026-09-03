// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_lowering_types.h
/// @brief Narrow extent contract shared by ConSan orchestration and lowering.

#pragma once

#include <cstdint>

namespace rocjitsu {

/// Select how far one native lowering pass may execute.
enum class ConSanLoweringExtent : uint8_t {
  Complete,
  ThroughProgramInventory,
};

} // namespace rocjitsu
