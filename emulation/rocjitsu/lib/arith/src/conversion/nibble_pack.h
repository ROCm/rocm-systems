// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Byte-to-nibble saturation kernels, preserving inactive lanes and the other halfword.
#pragma once
#include <stdint.h>
namespace rocjitsu::arith {
// Saturate four bytes into the selected halfword; requires the named CPU features.
void nibble_pack_x86_64_v3(bool signed_values, int lanes, uint64_t exec_mask, uint32_t mode,
                           uint32_t *d, const uint32_t *a);
void nibble_pack_x86_64_v4(bool signed_values, int lanes, uint64_t exec_mask, uint32_t mode,
                           uint32_t *d, const uint32_t *a);
} // namespace rocjitsu::arith
