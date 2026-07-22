// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_UTIL_SIMD_TEST_HOOKS_H_
#define ROCJITSU_UTIL_SIMD_TEST_HOOKS_H_

#include "util/simd.h"

namespace util {

/// Test-only: override the process force-scalar gate in-process so a single
/// test can drive both the scalar and SIMD execute paths and compare results.
/// Defined in util/src/simd.cpp. Production code never includes this header.
void set_force_scalar_for_testing(bool v);

} // namespace util

#endif // ROCJITSU_UTIL_SIMD_TEST_HOOKS_H_
