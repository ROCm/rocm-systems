// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Baseline-ISA FMA repair entry points for target-specific SIMD backends.
// Keep scalar model definitions out of SIMD translation units so linker
// COMDAT selection cannot redirect baseline calls to a higher ISA.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Return one FP32 FMA lane under validated instruction and guest controls.
uint32_t repair_fma_f32(uint32_t a, uint32_t b, uint32_t c, uint64_t flags,
                        uint32_t instruction_flags, bool rdna3);

// Return one selected FP16 FMA lane, accumulating status when requested.
uint16_t repair_fma_f16(uint16_t a, uint16_t b, uint16_t c, uint64_t flags,
                        uint32_t instruction_flags, bool rdna3, uint32_t *status = nullptr);

} // namespace rocjitsu::arith
