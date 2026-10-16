// SPDX-License-Identifier: MIT

// Per-lane conditional selection between two source registers.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Select between two sources by lane condition. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Half>
void cndmask_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                       const uint32_t *b, uint32_t condition);

// Select between two sources by lane condition. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void cndmask_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                       const uint32_t *b, uint32_t condition);

} // namespace rocjitsu::arith
