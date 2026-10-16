// SPDX-License-Identifier: MIT

// 32-bit add/subtract and three-input addition, with optional saturation.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

enum class IntegerAdd { Add, Sub, Subrev, Add3 };

// Add or subtract integers. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <IntegerAdd Op, bool Signed>
void integer_add_sat_x86_64_v3(uint32_t exec_mask, uint32_t *d, const uint32_t *a,
                               const uint32_t *b);

// Add or subtract integers. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <IntegerAdd Op, bool Signed>
void integer_add_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                           const uint32_t *b, const uint32_t *c);

// Add or subtract integers. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <IntegerAdd Op>
void integer_add_wrap_x86_64_v3(uint32_t exec_mask, uint32_t *d, const uint32_t *a,
                                const uint32_t *b, const uint32_t *c);

} // namespace rocjitsu::arith
