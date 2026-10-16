// SPDX-License-Identifier: MIT

// FP16 three-input minimum, maximum, median, and combined selections.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Select among three FP16 values. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool FirstMaximum, bool SecondMaximum, bool Propagate, bool Median = false,
          bool Rdna3 = false>
void half_minmax3_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a, const uint32_t *b, const uint32_t *c);

} // namespace rocjitsu::arith
