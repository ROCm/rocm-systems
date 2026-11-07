// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited CDNA legacy multiply policy and CPU backend declarations.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate one raw lane with validated, already architecture-gated modifiers.
// Baseline compiled; used by scalar execution and nonfinite SIMD lane repair.
uint32_t legacy_mul_value(uint64_t flags, uint32_t mode, uint32_t a, uint32_t b);

// Require validated controls and the named ISA; preserve lanes outside EXEC.
// One 32-lane VGPR per operand; sources may alias the destination.
void legacy_mul_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                          const uint32_t *a, const uint32_t *b);
void legacy_mul_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                          const uint32_t *a, const uint32_t *b);

} // namespace rocjitsu::arith
