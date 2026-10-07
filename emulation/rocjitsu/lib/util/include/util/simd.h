// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_SIMD_H_
#define UTIL_SIMD_H_

#include "util/detail/simd/common.h"
#include "util/detail/simd/control.h"

#if __has_include(<experimental/simd>)
#include "util/detail/simd/math64.h"
#endif

#endif // UTIL_SIMD_H_
