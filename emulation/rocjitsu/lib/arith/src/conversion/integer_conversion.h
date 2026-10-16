// SPDX-License-Identifier: MIT

// Saturating integer narrowing and signed/unsigned halfword extension.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Narrow or extend integer halfwords. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Unsigned, bool Packed>
void integer_conversion_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                                  const uint32_t *b);

// Narrow or extend integer halfwords. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Unsigned, bool Packed>
void integer_conversion_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                                  const uint32_t *b);

} // namespace rocjitsu::arith
