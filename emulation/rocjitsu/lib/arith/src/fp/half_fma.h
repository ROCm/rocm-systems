// SPDX-License-Identifier: MIT

// FP16 FMA/FMAC with guest rounding and optional FP status.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include "fp/fma.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate FP16 fused multiply-add. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <FmaOperands Operands = FmaOperands::Registers>
void half_fma_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                        const uint32_t *a, const uint32_t *b, const uint32_t *c,
                        uint16_t literal = 0);

} // namespace rocjitsu::arith
