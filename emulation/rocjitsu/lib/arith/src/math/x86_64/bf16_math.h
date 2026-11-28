// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// BF16 native arithmetic and packing around inherited FP32 math backends.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Operation identities shared by the BF16 entry points and native CPU backends.
enum class Bf16MathOp { Rcp, Rsq, Sqrt, Exp, Log, Sin, Cos, Tanh };

// FP32 core used by Compatibility and staged Loose sine/cosine.
using Bf16MathCore = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *,
                             const uint32_t *const *, uint32_t *);
// Widen selected halves, evaluate core without FP status, then scale/truncate and mask stores.
// Requires the named ISA and valid controls; flags select Loose or the inherited core.
int bf16_math_x86_64_v3(Bf16MathCore core, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                        uint32_t *d, const uint32_t *a);
int bf16_math_x86_64_v4(Bf16MathCore core, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                        uint32_t *d, const uint32_t *a);

// Evaluate Loose BF16 reciprocal/roots, EXP/LOG or TANH with shared FP32 arithmetic.
// Preserve selected halves, masks and whole-VGPR aliases; produce no FP status.
template <Bf16MathOp Op>
void bf16_math_loose_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a);
template <Bf16MathOp Op>
void bf16_math_loose_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a);

} // namespace rocjitsu::arith
