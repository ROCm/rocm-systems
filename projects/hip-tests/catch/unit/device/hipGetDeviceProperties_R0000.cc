/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// This test doesn't test the ABI, it tests if the API is alias to the R0000 when HIP_FORCE_API_VERSION is set.
// HIP Nvidia doesn't support this mechanism, so we can't test this.

#if HT_AMD

#undef HIP_ABI_IMPL
#define HIP_FORCE_API_VERSION 500
#include <hip/hip_runtime_api.h>
#include <catch2/catch_all.hpp>

#ifdef ENABLE_YAML_TAGS
#include "hip_test_config.hh"

#define SECOND_ARG(a, b, ...) b
#define GET_TAGS(...) SECOND_ARG(__VA_ARGS__)
#define HIP_TEST_CASE(name) TEST_CASE(#name, GET_TAGS(name))
#define HIP_TEMPLATE_TEST_CASE(name, ...) TEMPLATE_TEST_CASE(#name, GET_TAGS(name), __VA_ARGS__)

#else
#define GET_TAGS(...)
#define HIP_TEST_CASE(name) TEST_CASE(#name, "")
#define HIP_TEMPLATE_TEST_CASE(name, ...) TEMPLATE_TEST_CASE(#name, "", __VA_ARGS__)
#endif

/**
 * Test Description
 * ------------------------
 *  - Test if R0000 is used for hipDeviceProp_t
 * Test source
 * ------------------------
 *  - unit/device/hipGetDeviceProperties_R0000.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION <= 6.0
 */
HIP_TEST_CASE(Unit_hipGetDeviceProperties_R0000) {
  REQUIRE(sizeof(hipDeviceProp_t) == sizeof(hipDeviceProp_tR0000));
}

#endif
