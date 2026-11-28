// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Loose TANH with vector input/output modifiers and masked destination publication.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate a validated TANH request at the named ISA; all float encodings are supported.
// Half selects FP16 sources/output halves. Preserve aliases, inactive lanes and the other half.
template <bool Half>
void tanh_loose_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a);
template <bool Half>
void tanh_loose_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a);

} // namespace rocjitsu::arith
