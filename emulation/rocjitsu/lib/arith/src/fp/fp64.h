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
// Requires the named CPU features and validated instruction controls. ADD/MUL/FMA select NaNs
// in source order; NaN sources are quieted unless quiet_snan is clear.
void fp64_x86_64_v3(Fp64 op, uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                    bool quiet_snan = true, bool native_math = false, bool cdna = false);
#endif

#if defined(RJ_ARITH_HAVE_X86_64_V4)
// Evaluate FP64 ADD/MUL/FMA, integral rounding, mantissa or CDNA reciprocal/root arithmetic.
// Requires validated controls; guest rounding is instruction-local, writes are EXEC-masked.
void fp64_x86_64_v4(Fp64 op, uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                    const uint32_t *const *c, bool quiet_snan = true, bool native_math = false,
                    bool cdna = false);
#endif

} // namespace rocjitsu::arith
