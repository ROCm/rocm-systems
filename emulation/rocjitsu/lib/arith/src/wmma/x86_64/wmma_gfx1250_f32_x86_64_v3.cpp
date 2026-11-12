// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86-64-v3 FP32 gfx1250 WMMA, using the ISA-isolated row reduction.
#include "wmma/gfx1250_backends.h"
#include "wmma/gfx1250_f32.h"
namespace rocjitsu::arith {
// Evaluate the full Wave32 instruction in this ISA domain.
void wmma_gfx1250_f32_x86_64_v3(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  RJ_ARITH_IMPL_NAMESPACE::wmma_gfx1250_f32_kernel(controls, d, a, b, c);
}
} // namespace rocjitsu::arith
