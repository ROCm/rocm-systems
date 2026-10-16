// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16 add/subtract/multiply, including packed forms and guest FP status and min/max selection.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include "fp/binary.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate FP16 binary arithmetic. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Binary Op, bool Packed = false>
void half_binary_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                           const uint32_t *a, const uint32_t *b);

} // namespace rocjitsu::arith
