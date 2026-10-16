// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared enumeration of supported semantics modes, including compatibility.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>
#include <vector>

namespace rocjitsu::arith::test {

// Return all defined semantics modes, including temporary compatibility mode.
inline std::vector<uint64_t> semantics_modes() { return {RJ_ARITH_SEMANTICS_LOOSE}; }

} // namespace rocjitsu::arith::test
