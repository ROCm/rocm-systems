// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared FP16/FP32 arithmetic and min/max operation selector.
// Declares FP32 backends; FP16 backends are declared in half_binary.h.

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
// Evaluate FP32 binary arithmetic or selection. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void binary_x86_64_v3(Binary op, uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                      uint32_t *d, const uint32_t *a, const uint32_t *b);

// Attempts finite nearest-even ADD/SUB/SUBREV; leaves outputs untouched on decline.
bool binary_exact_x86_64_v3(Binary op, uint64_t flags, uint32_t exec_mask,
                            uint32_t instruction_flags, uint32_t *d, const uint32_t *a,
                            const uint32_t *b);
#endif

#if defined(RJ_ARITH_HAVE_X86_64_V4)
// Evaluate loose ADD/SUB/SUBREV/MUL/DX9-MUL with validated controls and masked writes.
void binary_x86_64_v4(Binary op, uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                      uint32_t *d, const uint32_t *a, const uint32_t *b);

// Attempt finite MUL or MUL_DX9_ZERO domains without writes on decline; uses default host FP
// state. FP status reporting follows the resolved semantics.
bool mul_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                         uint32_t *d, const uint32_t *a, const uint32_t *b, uint32_t *state,
                         bool dx9_zero);
#endif

} // namespace rocjitsu::arith
