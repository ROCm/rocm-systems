// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16/BF16 conversion operations and ISA-specific backends. No guest FP status is produced.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

enum class Float16Format { F16, Bf16 };

enum class Float16Conversion { Widen, Pack, Stochastic, StochasticPack };

// Convert one Wave32 using validated controls, preserving inactive lanes.
void float16_conversion_x86_64_v3(Float16Format format, Float16Conversion op, uint64_t flags,
                                  uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                                  const uint32_t *b, const uint32_t *c);
void float16_conversion_x86_64_v4(Float16Format format, Float16Conversion op, uint64_t flags,
                                  uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                                  const uint32_t *b, const uint32_t *c);

} // namespace rocjitsu::arith
