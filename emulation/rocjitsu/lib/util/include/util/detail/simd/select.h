// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DETAIL_SIMD_SELECT_H_
#define UTIL_DETAIL_SIMD_SELECT_H_

#include "util/detail/simd/config.h"

#if __has_include(<experimental/simd>)
#if defined(__AVX512F__)
#include "util/detail/simd/x86_avx512.h"
#else
#include "util/detail/simd/portable.h"
#endif
#else
#include "util/detail/simd/no_stdx.h"
#endif

#endif // UTIL_DETAIL_SIMD_SELECT_H_
