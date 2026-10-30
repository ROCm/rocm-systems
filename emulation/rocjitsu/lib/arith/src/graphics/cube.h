// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Cube-map face selection, coordinates, and major-axis magnitude.
// Operation selector and cross-backend declarations.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith {

enum class Cube { Id, Sc, Tc, Ma };

// Evaluate cube-map coordinates or face selection. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Cube Op>
void cube_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                    const uint32_t *b, const uint32_t *c, bool quiet_nan, bool clamp_nan,
                    unsigned round_mode);

// Evaluate cube-map coordinates or face selection. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Cube Op>
void cube_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                    const uint32_t *b, const uint32_t *c, bool quiet_nan, bool clamp_nan,
                    unsigned round_mode);

} // namespace rocjitsu::arith
