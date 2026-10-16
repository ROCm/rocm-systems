// SPDX-License-Identifier: MIT

// FP32 add/subtract/multiply and DX9-zero multiplication.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

enum class Binary {
  Add,
  Sub,
  Subrev,
  Mul,
  MulDx9Zero,
  MinNum,
  MaxNum,
  Minimum,
  Maximum,
  MinLegacy,
  MaxLegacy
};

#if defined(RJ_ARITH_HAVE_X86_64_V3)
// Evaluate FP32 binary arithmetic. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void binary_x86_64_v3(Binary op, uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                      const uint32_t *a, const uint32_t *b);

// Attempts finite nearest-even ADD/SUB/SUBREV; leaves outputs untouched on decline.
bool binary_exact_x86_64_v3(Binary op, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                            uint32_t *d, const uint32_t *a, const uint32_t *b);
#endif

} // namespace rocjitsu::arith
