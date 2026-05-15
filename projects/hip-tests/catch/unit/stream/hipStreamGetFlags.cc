/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
Testcase Scenarios :
1) Test flag value of stream created with hipStreamCreateWithFlags/
 /hipStreamCreate/hipStreamCreateWithPriority.
2) Negative tests for hipStreamGetFlags api.
3) Test flag value when streams created with CUMask.
*/

#include <hip_test_common.hh>

/**
 * @brief Check that hipStreamGetFlags returns the same flags that were used to create the stream.
 *
 */
HIP_TEST_CASE(Unit_hipStreamGetFlags_Basic) {
  unsigned int expectedFlag = GENERATE(hipStreamDefault, hipStreamNonBlocking);
  unsigned int returnedFlags;
  hipStream_t stream;

  HIP_CHECK(hipStreamCreateWithFlags(&stream, expectedFlag));
  HIP_CHECK(hipStreamGetFlags(stream, &returnedFlags));
  REQUIRE((returnedFlags & expectedFlag) == expectedFlag);
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * @brief Negative scenarios for hipStreamGetFlags.
 *
 */
HIP_TEST_CASE(Unit_hipStreamGetFlags_Negative) {
  hipStream_t validStream;
  unsigned int flags;

  HIP_CHECK(hipStreamCreate(&validStream));

  SECTION("Nullptr Stream && Valid Flags") { /* EXSWCPHIPT-17 */
#if HT_AMD
    HIP_CHECK_ERROR(hipStreamGetFlags(nullptr, &flags), hipErrorInvalidValue);
#elif HT_NVIDIA
    HIP_CHECK(hipStreamGetFlags(nullptr, &flags));
#endif
  }

  SECTION("Valid Stream && Nullptr Flags") {
    HIP_CHECK_ERROR(hipStreamGetFlags(validStream, nullptr), hipErrorInvalidValue);
  }

  HIP_CHECK(hipStreamDestroy(validStream));
}

#if HT_AMD
/**
 * Test flag value when streams created with CUMask.
 */
HIP_TEST_CASE(Unit_hipStreamGetFlags_StreamsCreatedWithCUMask) {
  hipStream_t stream;
  unsigned int flags;
  const uint32_t cuMask = 0xffffffff;
  HIP_CHECK(hipExtStreamCreateWithCUMask(&stream, 1, &cuMask));
  HIP_CHECK(hipStreamGetFlags(stream, &flags));
  REQUIRE(flags == hipStreamDefault);
  HIP_CHECK(hipStreamDestroy(stream));
}
#endif
