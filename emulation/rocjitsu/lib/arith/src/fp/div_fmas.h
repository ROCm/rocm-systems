// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Cross-ISA backend declarations for division fmas.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate division FMA with conditional scaling. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Width>
void div_fmas_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t condition, bool flush_output,
                        uint32_t *state);

// Evaluate division FMA with conditional scaling. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Width>
void div_fmas_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t condition, bool flush_output,
                        uint32_t *state);

} // namespace rocjitsu::arith
