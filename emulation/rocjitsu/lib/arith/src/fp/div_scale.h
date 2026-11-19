// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Cross-ISA backend declarations for division scale.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Rescale division operands and produce scaling conditions. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Width>
uint32_t div_scale_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, bool quiet_nan, uint64_t flags);

// Rescale division operands and produce scaling conditions. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Width>
uint32_t div_scale_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, bool quiet_nan, uint64_t flags);

} // namespace rocjitsu::arith
