// SPDX-License-Identifier: MIT

// Division of 2/pi into exponent-selected pieces for trigonometric range reduction.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include <array>
#include <stdint.h>

namespace rocjitsu::arith {

// Two exponent ranges, each indexed by a clamped table-bit offset.
extern const std::array<uint64_t, 2404> trig_preop_table;
extern const std::array<uint64_t, 2370> trig_preop_table_rdna4;

// Select an exponent-dependent piece of 2/pi. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Rdna3>
void trig_preop_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                          const uint32_t *a_hi, const uint32_t *b);

// Select an exponent-dependent piece of 2/pi. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Rdna3>
void trig_preop_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                          const uint32_t *a_hi, const uint32_t *b);

} // namespace rocjitsu::arith
