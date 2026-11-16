// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Wave-width mask types shared by dispatch and ISA-specific backend declarations.
#pragma once

#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith {

// EXEC has one bit per lane in the instruction wave.
template <int Lanes>
  requires(Lanes == 32 || Lanes == 64)
using ExecMask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;

} // namespace rocjitsu::arith
