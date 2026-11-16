// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared compact coverage table and ISA-specific screen-partition backends.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

extern const uint32_t screen_partition_table[32];
// Transform all source lanes before masked stores; preserve overlapping storage.
void screen_partition_x86_64_v3(uint64_t exec_mask, uint32_t *d, const uint32_t *a);
void screen_partition_x86_64_v4(uint64_t exec_mask, uint32_t *d, const uint32_t *a);
} // namespace rocjitsu::arith
