// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Directed FP32 interpolation backend, with lane-local exceptional repair.
#pragma once
#include <stdint.h>
namespace rocjitsu::arith {
// Compute staged mixed FMA at FP32 precision for the selected lanes. Return
// lanes requiring baseline handling; their destination values are unspecified.
uint32_t interp_gfx10_directed_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                         uint32_t *d, const uint32_t *a, const uint32_t *b,
                                         const uint32_t *c);
} // namespace rocjitsu::arith
