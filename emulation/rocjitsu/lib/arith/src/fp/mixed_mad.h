// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited CDNA mixed MAD: FP32 intermediate, then optional FP16 narrowing.
#pragma once

#include "fp/mixed_fma.h"
#include <stdint.h>

namespace rocjitsu::arith {

// Return a raw FP32 or low-positioned FP16 result for validated source controls.
uint32_t mixed_mad_value(MixedFma destination, uint64_t flags, uint32_t mode, uint32_t a,
                         uint32_t b, uint32_t c);

// Require the named ISA and validated controls; preserve inactive/unselected data.
void mixed_mad_x86_64_v3(MixedFma destination, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                         uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c);
void mixed_mad_x86_64_v4(MixedFma destination, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                         uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c);

} // namespace rocjitsu::arith
