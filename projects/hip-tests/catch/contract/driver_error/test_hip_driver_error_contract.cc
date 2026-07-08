/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstring>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
constexpr hipError_t kKnownErrors[] = {hipSuccess, hipErrorInvalidValue, hipErrorNotSupported};

void RequireNonEmptyString(const char* value) {
  REQUIRE(value != nullptr);
  REQUIRE(std::strlen(value) > 0);
}
}  // namespace

HIP_TEST_CASE(Contract_DriverError_GetErrorName_KnownCode_ReturnsNonEmptyString) {
  for (hipError_t error : kKnownErrors) {
    const char* name = nullptr;
    HIP_CHECK(hipDrvGetErrorName(error, &name));
    RequireNonEmptyString(name);
  }
}

HIP_TEST_CASE(Contract_DriverError_GetErrorString_KnownCode_ReturnsNonEmptyString) {
  for (hipError_t error : kKnownErrors) {
    const char* message = nullptr;
    HIP_CHECK(hipDrvGetErrorString(error, &message));
    RequireNonEmptyString(message);
  }
}

HIP_TEST_CASE(Contract_DriverError_GetErrorName_RepeatedQueryIsStable) {
  for (hipError_t error : kKnownErrors) {
    const char* first_name = nullptr;
    const char* second_name = nullptr;
    HIP_CHECK(hipDrvGetErrorName(error, &first_name));
    HIP_CHECK(hipDrvGetErrorName(error, &second_name));

    RequireNonEmptyString(first_name);
    RequireNonEmptyString(second_name);
    REQUIRE(std::strcmp(first_name, second_name) == 0);
  }
}

HIP_TEST_CASE(Contract_DriverError_GetErrorString_RepeatedQueryIsStable) {
  for (hipError_t error : kKnownErrors) {
    const char* first_message = nullptr;
    const char* second_message = nullptr;
    HIP_CHECK(hipDrvGetErrorString(error, &first_message));
    HIP_CHECK(hipDrvGetErrorString(error, &second_message));

    RequireNonEmptyString(first_message);
    RequireNonEmptyString(second_message);
    REQUIRE(std::strcmp(first_message, second_message) == 0);
  }
}

HIP_TEST_CASE(Contract_DriverError_GetErrorName_InvalidCode_IsRejected) {
  const char* name = nullptr;
  REQUIRE(hipDrvGetErrorName(static_cast<hipError_t>(-1), &name) != hipSuccess);
}

HIP_TEST_CASE(Contract_DriverError_GetErrorString_InvalidCode_IsRejected) {
  const char* message = nullptr;
  REQUIRE(hipDrvGetErrorString(static_cast<hipError_t>(-1), &message) != hipSuccess);
}
