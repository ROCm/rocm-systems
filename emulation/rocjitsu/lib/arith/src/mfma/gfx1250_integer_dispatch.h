// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Select the gfx1250 byte matrix kernel after validating flags and normalizing metadata.
#pragma once
#include "common/validation.h"
#include "mfma/backends.h"
#include "mfma/sparse_integer.h"
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
template <bool Sparse>
void dispatch_gfx1250_integer([[maybe_unused]] uint64_t flags, uint64_t controls, bool clamp,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              const uint32_t *const *index) {
#if defined(RJ_ARITH_HAVE_AVX512VNNI)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_ZEN4)) {
    gfx1250_integer_avx512vnni<Sparse>(controls, clamp, d, a, b, c, index);
    return;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    gfx1250_integer_x86_64_v4<Sparse>(controls, clamp, d, a, b, c, index);
    return;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    gfx1250_integer_x86_64_v3<Sparse>(controls, clamp, d, a, b, c, index);
    return;
  }
#endif
  if constexpr (Sparse) {
    sparse_matrix_i8_kernel<16, 128, true>(controls, clamp, d, a, b, index);
  } else {
    matrix_i8_kernel<16, 64, 1, true>(controls, clamp, d, a, b, c);
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
