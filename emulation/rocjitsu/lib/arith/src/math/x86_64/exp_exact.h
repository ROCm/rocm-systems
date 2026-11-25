// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-specific implementations of the inherited integer EXP model.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate EXP with effective modifiers and the selected NaN policy. Preserve inactive lanes.
// The caller validates controls and guarantees the named CPU features.
void exp_exact_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a, bool quiet_snan);
void exp_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a, bool quiet_snan);

} // namespace rocjitsu::arith
