// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Backend declarations for FP32-intermediate half MAD.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate selected lanes, retaining the opposite half; requires x86-64-v3 and validated controls.
void half_mad_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                        const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Evaluate selected lanes, retaining the opposite half; requires x86-64-v4 and validated controls.
void half_mad_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                        const uint32_t *a, const uint32_t *b, const uint32_t *c);

} // namespace rocjitsu::arith
