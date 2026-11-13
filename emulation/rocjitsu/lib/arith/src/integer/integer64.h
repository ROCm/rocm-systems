// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Backend declarations for split-word 64-bit integer arithmetic.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

enum class Integer64 { Add, Sub, Mul, MinU, MaxU, MinI, MaxI, ShiftAdd };

// Evaluate validated wide arithmetic, preserving inactive lanes and aliased input words.
template <Integer64 Op>
void integer64_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c);

// Evaluate validated wide arithmetic, preserving inactive lanes and aliased input words.
template <Integer64 Op>
void integer64_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c);

} // namespace rocjitsu::arith
