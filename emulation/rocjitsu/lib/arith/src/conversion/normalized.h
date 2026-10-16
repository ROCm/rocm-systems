// SPDX-License-Identifier: MIT

// FP16/FP32 conversion to signed/unsigned normalized 16-bit encodings.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

enum class NormalizedForm { PackedFloat, PackedHalf, Half };

// Convert to normalized 16-bit values. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Unsigned, NormalizedForm Form>
void normalized_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                          const uint32_t *b, bool flush_input);

// Convert to normalized 16-bit values. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Unsigned, NormalizedForm Form>
void normalized_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                          const uint32_t *b, bool flush_input);

} // namespace rocjitsu::arith
