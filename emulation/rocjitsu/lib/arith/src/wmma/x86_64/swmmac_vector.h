// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-local packed-register vector loads for sparse WMMA.
#pragma once

#include "common/bits.h"
#include <cstring>
#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::sparse_x86 {
#if defined(__AVX512F__)
constexpr int Width = 16;
#else
constexpr int Width = 8;
#endif
using U32 [[gnu::vector_size(Width * 4)]] = uint32_t;
using F32 [[gnu::vector_size(Width * 4)]] = float;

inline U32 load(const uint32_t *p) {
  U32 x;
  std::memcpy(&x, p, sizeof(x));
  return x;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::sparse_x86
