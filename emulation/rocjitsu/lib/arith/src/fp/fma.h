// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Operand-placement choices shared by register and literal FMA implementations.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Placement of the scalar literal in a fused multiply-add.
enum class FmaOperands { Registers, MultiplyLiteral, AddLiteral };

#if defined(RJ_ARITH_HAVE_X86_64_V3)
// Evaluate FP32 MAD with separate product rounding and caller-selected modifiers.
// Requires the named ISA and validated flags; preserve inactive lanes and aliases.
void mad_f32_x86_64_v3(bool legacy, uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                       uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c);
#endif

#if defined(RJ_ARITH_HAVE_X86_64_V4)
// Evaluate FP32 MAD with separate product rounding and caller-selected modifiers.
// Requires the named ISA and validated flags; preserve inactive lanes and aliases.
void mad_f32_x86_64_v4(bool legacy, uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                       uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c);
#endif

} // namespace rocjitsu::arith
