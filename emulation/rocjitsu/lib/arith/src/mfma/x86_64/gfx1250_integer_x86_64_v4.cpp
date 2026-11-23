// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 dense and sparse IU8 public backend for x86-64-v4.
#include "mfma/backends.h"
#include "mfma/x86_64/gfx1250_integer.h"
namespace rocjitsu::arith {
template <bool Sparse>
void gfx1250_integer_x86_64_v4(uint64_t controls, bool clamp, uint32_t *const *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               const uint32_t *const *c, const uint32_t *const *index) {
  gfx1250_i8::compute<Sparse>(controls, clamp, d, a, b, c, index);
}
template Gfx1250IntegerBackend gfx1250_integer_x86_64_v4<false>;
template Gfx1250IntegerBackend gfx1250_integer_x86_64_v4<true>;
} // namespace rocjitsu::arith
