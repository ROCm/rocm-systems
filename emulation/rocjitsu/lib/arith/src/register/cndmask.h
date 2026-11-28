// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Per-lane conditional selection between two source registers.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include "common/exec_mask.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Select between two sources by lane condition. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Half, int Lanes = 32>
void cndmask_x86_64_v4(ExecMask<Lanes> exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                       const uint32_t *b, ExecMask<Lanes> condition);

// Select between two sources by lane condition. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Half, int Lanes = 32>
void cndmask_x86_64_v3(ExecMask<Lanes> exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                       const uint32_t *b, ExecMask<Lanes> condition);

} // namespace rocjitsu::arith
