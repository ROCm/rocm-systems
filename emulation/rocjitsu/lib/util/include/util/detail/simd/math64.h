// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DETAIL_SIMD_MATH64_H_
#define UTIL_DETAIL_SIMD_MATH64_H_

#include "util/detail/simd/config.h"

#if UTIL_SIMD_BROKEN_NATIVE_64BIT_MASKS
#include "util/detail/simd/compat64_math.h"
#else
#include "util/detail/simd/native64_math.h"
#endif

#endif // UTIL_DETAIL_SIMD_MATH64_H_
