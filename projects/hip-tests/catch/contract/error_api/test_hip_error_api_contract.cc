/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstring>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
void RequireNonEmptyString(const char* value) {
  REQUIRE(value != nullptr);
  REQUIRE(std::strlen(value) > 0);
}

void RequireNonNullString(const char* value) { REQUIRE(value != nullptr); }
}

// @asserts: hipGetErrorName - returns a non-null, non-empty name string for hipSuccess
HIP_TEST_CASE(Contract_ErrorApi_HipGetErrorName_Success_ReturnsNonEmptyString) {
  RequireNonEmptyString(hipGetErrorName(hipSuccess));
}

// @asserts: hipGetErrorString - returns a non-null, non-empty description string for hipSuccess
HIP_TEST_CASE(Contract_ErrorApi_HipGetErrorString_Success_ReturnsNonEmptyString) {
  RequireNonEmptyString(hipGetErrorString(hipSuccess));
}

// @asserts: hipGetErrorName - returns a non-null, non-empty name string for hipErrorInvalidValue
HIP_TEST_CASE(Contract_ErrorApi_HipGetErrorName_InvalidValue_ReturnsNonEmptyString) {
  RequireNonEmptyString(hipGetErrorName(hipErrorInvalidValue));
}

// @asserts: hipGetErrorString - returns a non-null, non-empty description string for hipErrorInvalidValue
HIP_TEST_CASE(Contract_ErrorApi_HipGetErrorString_InvalidValue_ReturnsNonEmptyString) {
  RequireNonEmptyString(hipGetErrorString(hipErrorInvalidValue));
}

// @asserts: hipGetErrorName - returns a non-null string even for an unknown/out-of-range error code
HIP_TEST_CASE(Contract_ErrorApi_HipGetErrorName_UnknownError_ReturnsNonNullString) {
  RequireNonNullString(hipGetErrorName(static_cast<hipError_t>(-1)));
}

// @asserts: hipGetErrorString - returns a non-null string even for an unknown/out-of-range error code
HIP_TEST_CASE(Contract_ErrorApi_HipGetErrorString_UnknownError_ReturnsNonNullString) {
  RequireNonNullString(hipGetErrorString(static_cast<hipError_t>(-1)));
}
