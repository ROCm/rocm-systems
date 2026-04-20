/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <chrono>
#include <hip_test_common.hh>
#include <utils.hh>

namespace hipStreamCreateWithFlagsTests {

HIP_TEST_CASE(Unit_hipStreamCreateWithFlags_Negative_NullStream) {
  HIP_CHECK_ERROR(hipStreamCreateWithFlags(nullptr, hipStreamDefault), hipErrorInvalidValue);
}

HIP_TEST_CASE(Unit_hipStreamCreateWithFlags_Negative_InvalidFlag) {
  hipStream_t stream{};
  unsigned int flag = 0xFF;
  REQUIRE(flag != hipStreamDefault);
  REQUIRE(flag != hipStreamNonBlocking);
  HIP_CHECK_ERROR(hipStreamCreateWithFlags(&stream, flag), hipErrorInvalidValue);
}

// create a stream and check the properties are correctly set
HIP_TEST_CASE(Unit_hipStreamCreateWithFlags_Default) {
  const unsigned int flagUnderTest = GENERATE(hipStreamDefault, hipStreamNonBlocking);
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreateWithFlags(&stream, flagUnderTest));

  unsigned int flag{};
  HIP_CHECK(hipStreamGetFlags(stream, &flag));
  REQUIRE(flag == flagUnderTest);

  int priority{};
  HIP_CHECK(hipStreamGetPriority(stream, &priority));
  // zero is considered default priority
  REQUIRE(priority == 0);

  HIP_CHECK(hipStreamDestroy(stream));
}

// a stream will default to blocking the null stream, but will not block the null stream when
// created with hipStreamNonBlocking
#if HT_AMD /* Disabled because frequency based wait is timing out on nvidia platforms */
HIP_TEST_CASE(Unit_hipStreamCreateWithFlags_DefaultStreamInteraction) {
  const hipStream_t defaultStream = GENERATE(static_cast<hipStream_t>(nullptr), hipStreamPerThread);
  const unsigned int flagUnderTest = GENERATE(hipStreamDefault, hipStreamNonBlocking);
  CAPTURE(defaultStream, flagUnderTest);

  hipStream_t stream{};
  HIP_CHECK(hipStreamCreateWithFlags(&stream, flagUnderTest));

  constexpr auto delay = std::chrono::milliseconds(500);

  SECTION("default stream waiting for created stream") {
    const hipError_t expectedError =
        (flagUnderTest == hipStreamDefault) && (defaultStream == nullptr) ? hipErrorNotReady
                                                                          : hipSuccess;
    LaunchDelayKernel(delay, stream);
    REQUIRE(hipStreamQuery(defaultStream) == expectedError);
  }
  SECTION("created stream waiting for default stream") {
    LaunchDelayKernel(delay, defaultStream);
    REQUIRE(hipStreamQuery(stream) == hipSuccess);
  }

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipStreamDestroy(stream));
}
#endif

}  // namespace hipStreamCreateWithFlagsTests
