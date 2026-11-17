// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// BF16 packing around existing whole-wave FP32 math backends.
#pragma once
#include <stdint.h>
namespace rocjitsu::arith {
using Bf16MathCore = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *,
                             const uint32_t *const *, uint32_t *);
// Widen selected halves, evaluate core without FP status, then scale/truncate and mask stores.
// Requires the named ISA, valid controls, and an FP32 core supporting mode 2.
int bf16_math_x86_64_v3(Bf16MathCore core, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                        uint32_t *d, const uint32_t *a);
int bf16_math_x86_64_v4(Bf16MathCore core, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                        uint32_t *d, const uint32_t *a);
} // namespace rocjitsu::arith
