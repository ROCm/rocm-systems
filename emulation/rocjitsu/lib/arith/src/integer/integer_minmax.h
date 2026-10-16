// SPDX-License-Identifier: MIT

// Signed/unsigned two-input and packed 16-bit min/max.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

enum class IntegerMinmax { Min, Max, Min3, Max3, Minmax, Maxmin, Median };

// Select integer minima or maxima. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <IntegerMinmax Op, bool Signed>
void integer_minmax_x86_64_v3(uint32_t exec_mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                              const uint32_t *c);

// Select integer minima or maxima. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <IntegerMinmax Op, bool Signed>
void integer_minmax_x86_64_v4(uint32_t exec_mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                              const uint32_t *c);

} // namespace rocjitsu::arith
