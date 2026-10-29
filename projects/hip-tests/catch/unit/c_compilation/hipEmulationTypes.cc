/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
/**
 * Test Description
 * ------------------------
 *    - Uses the floating-point emulation control types from hip/library_types.h
 *      in a C (not C++) compilation unit. This exercises the `#ifndef __cplusplus`
 *      typedef block that makes the *_t names available in C, and checks the
 *      enumerator values with _Static_assert.

 * Test source
 * ------------------------
 *    - catch/unit/c_compilation/hipEmulationTypes.cc
 *    - catch/unit/c_compilation/hipEmulationTypes.c
 * Test requirements
 * ------------------------
 *    - AMD platform (the C typedef block exists only on the AMD path)
 */

extern "C" int hipEmulationTypesC();

// @asserts: hipEmulationStrategy_t - emulation *_t types compile and hold documented values in C
HIP_TEST_CASE(Unit_hipEmulationTypes_ctest) { REQUIRE(hipEmulationTypesC() == 1); }
