// SPDX-License-Identifier: MIT

// Exponent extraction from FP16, FP32, and FP64 encodings.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#ifndef RJ_ARITH_FREXP_EXP_H_
#define RJ_ARITH_FREXP_EXP_H_

#include <stdint.h>

namespace rocjitsu::arith {

// Extract floating-point exponents. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void frexp_exp_x86_64_v3(bool fp64, uint32_t exec_mask, uint32_t *d, const uint32_t *const *a,
                         bool flush_input);

} // namespace rocjitsu::arith

#endif
