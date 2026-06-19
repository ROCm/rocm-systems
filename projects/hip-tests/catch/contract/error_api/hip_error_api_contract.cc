/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
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

HIP_TEST_CASE(Contract_ErrorApi_GetErrorNameSuccess_ReturnsNonEmptyString) {
  RequireNonEmptyString(hipGetErrorName(hipSuccess));
}

HIP_TEST_CASE(Contract_ErrorApi_GetErrorStringSuccess_ReturnsNonEmptyString) {
  RequireNonEmptyString(hipGetErrorString(hipSuccess));
}

HIP_TEST_CASE(Contract_ErrorApi_GetErrorNameInvalidValue_ReturnsNonEmptyString) {
  RequireNonEmptyString(hipGetErrorName(hipErrorInvalidValue));
}

HIP_TEST_CASE(Contract_ErrorApi_GetErrorStringInvalidValue_ReturnsNonEmptyString) {
  RequireNonEmptyString(hipGetErrorString(hipErrorInvalidValue));
}

HIP_TEST_CASE(Contract_ErrorApi_GetErrorNameUnknownError_ReturnsNonNullString) {
  RequireNonNullString(hipGetErrorName(static_cast<hipError_t>(-1)));
}

HIP_TEST_CASE(Contract_ErrorApi_GetErrorStringUnknownError_ReturnsNonNullString) {
  RequireNonNullString(hipGetErrorString(static_cast<hipError_t>(-1)));
}
