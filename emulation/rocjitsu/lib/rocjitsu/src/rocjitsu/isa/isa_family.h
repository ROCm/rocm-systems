// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iosfwd>

namespace rocjitsu {

/// @brief Shared ISA implementation selected internally by a concrete target.
/// @details A family selects common traits and translation tables. It does not
/// establish instruction legality or compatibility between concrete targets.
enum class IsaFamily : int32_t {
  Cdna1 = 0,
  Cdna2 = 1,
  Cdna3 = 2,
  Cdna4 = 3,
  Rdna1 = 4,
  Rdna2 = 5,
  Rdna3 = 6,
  Rdna3_5 = 7,
  Rdna4 = 8,
  Rv32i = 9,
  Rv64i = 10,
  Cdna5 = 11,
  Count = 12,
  Invalid = Count,
};

std::ostream &operator<<(std::ostream &stream, IsaFamily family);

} // namespace rocjitsu
