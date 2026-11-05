// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP64 arithmetic with split-word VGPR operands and guest FP policies.
// Also implements legacy, number-preferring and NaN-propagating min/max selection.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

enum class Fp64 {
  Add,
  Mul,
  Fma,
  MinLegacy,
  MaxLegacy,
  MinNum,
  MaxNum,
  Minimum,
  Maximum,
  Trunc,
  Ceil,
  Rndne,
  Floor,
  Fract,
  Sqrt,
  Rcp,
  FrexpMant,
  Rsq
};

#if defined(RJ_ARITH_HAVE_X86_64_V3)
// Evaluate FP64 arithmetic. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void fp64_x86_64_v3(Fp64 op, uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);
#endif

#if defined(RJ_ARITH_HAVE_X86_64_V4)
// Evaluate FP64 ADD/MUL/FMA with instruction-local guest rounding and masked destination stores.
// Requires validated flags and a nonzero guest rounding mode.
void fp64_directed_x86_64_v4(Fp64 op, uint64_t flags, uint32_t exec_mask,
                             uint32_t instruction_flags, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c);
#endif

} // namespace rocjitsu::arith
