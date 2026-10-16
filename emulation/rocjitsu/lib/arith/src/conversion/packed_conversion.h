// SPDX-License-Identifier: MIT

// Pairwise FP32 packing into FP16/BF16 with format-specific rounding.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

enum class PackedConversion { HalfRtz, Signed, Unsigned };

// Pack two FP32 conversions into one word. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <PackedConversion Op>
void packed_conversion_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                                 const uint32_t *b);

// Pack two FP32 conversions into one word. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <PackedConversion Op>
void packed_conversion_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                                 const uint32_t *b);

} // namespace rocjitsu::arith
