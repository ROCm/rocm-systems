/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <catch2/catch_test_macros.hpp>
#include <hip/hip_runtime.h>

// Local HRR test harness.
//
// The HRR project owns its behavior tests and must not depend on the hip-tests
// Catch2 infrastructure (the hip-tests Catch2 helper headers). These macros
// replace only the small pieces the migrated HRR tests actually use.

// HRR_TEST_CASE mirrors the hip-tests test-case macro but with a fixed [hrr] tag so
// the HRR project's tests do not rely on the hip-tests GET_TAGS() config lookup.
#define HRR_TEST_CASE(name) TEST_CASE(#name, "[hrr]")

// HRR_HIP_CHECK replaces the hip-tests HIP error check: fail the current Catch2 test if a HIP call
// does not return hipSuccess, reporting the error code and string.
#define HRR_HIP_CHECK(expr)                                                     \
  do {                                                                          \
    hipError_t _hrr_err = (expr);                                              \
    INFO("HIP call failed: " #expr);                                           \
    INFO("hipError_t: " << static_cast<int>(_hrr_err));                        \
    INFO("hipGetErrorString: " << hipGetErrorString(_hrr_err));                \
    REQUIRE(_hrr_err == hipSuccess);                                           \
  } while (0)

// HRR_SKIP replaces the hip-tests skip macro: emit a warning describing why the current
// case is being skipped and return early.
#define HRR_SKIP(message)                                                       \
  do {                                                                          \
    WARN(message);                                                             \
    return;                                                                    \
  } while (0)
