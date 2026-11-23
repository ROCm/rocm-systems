// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed U64 arithmetic selectors and ISA-specific backend declarations.
#pragma once
#include <stdint.h>
namespace rocjitsu::arith {
enum class PackedU64 { Add, Sub, ShiftAdd };
// Compute two U64 elements per lane, preserving inactive lanes and arbitrary register overlap.
// ShiftAdd requires both U32 counts to be in [0,4] on active lanes.
template <PackedU64 Op>
void packed_u64_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c);
template <PackedU64 Op>
void packed_u64_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c);
} // namespace rocjitsu::arith
