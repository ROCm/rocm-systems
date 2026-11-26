// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Internal bridge for VM-routed Compatibility operands; not an instruction C API.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate already-routed FP32 MUL with separate participating and written lanes.
// Requires Compatibility semantics, 32 or 64 lanes, valid status storage and
// write_mask contained in exec_mask. Sources may alias D; errors preserve outputs.
// The gfx12 INVALID classifier observes written lanes; inherited MUL range/status
// classification observes all participating lanes, including suppressed stores.
// gfx1250 retains range-only reporting without gfx12 INVALID causes.
int mul_f32_compatibility_staged(uint64_t flags, uint64_t exec_mask, uint64_t write_mask, int lanes,
                                 bool rdna3, uint32_t modifiers, uint32_t *const *d,
                                 const uint32_t *const *a, const uint32_t *const *b,
                                 uint32_t *state, bool gfx1250 = false);

} // namespace rocjitsu::arith
