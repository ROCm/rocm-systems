// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Integer add/subtract with per-lane carry/borrow input and output masks.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

// Carry/borrow semantics adapted from rocjitsu's RDNA4 arithmetic handlers.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith {

enum class CarryOp { Add, Sub, Subrev };

// Add/subtract with carry or borrow masks. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <CarryOp Op, bool WithCarry>
uint32_t carry_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b, uint32_t input_carry);

// Add/subtract with carry or borrow masks. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <CarryOp Op, bool WithCarry>
uint32_t carry_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b, uint32_t input_carry);

} // namespace rocjitsu::arith
