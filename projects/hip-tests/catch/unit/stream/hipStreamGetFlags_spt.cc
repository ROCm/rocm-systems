/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
/**
 * @addtogroup hipStreamGetFlags_spt hipStreamGetFlags_spt
 * @{
 * @ingroup StreamTest
 * `hipStreamGetFlags_spt(hipStream_t stream, unsigned int* flags)` -
 * Return flags associated with a stream on stream per thread
 */

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify that hipStreamGetFlags_spt returns the same
 * flags that were used to create the stream.
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamGetFlags_spt.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamGetFlags_spt_Basic") {
  unsigned int expectedFlag = GENERATE(hipStreamDefault, hipStreamNonBlocking);
  unsigned int returnedFlags;
  hipStream_t stream;

  HIP_CHECK(hipStreamCreateWithFlags(&stream, expectedFlag));
  HIP_CHECK(hipStreamGetFlags_spt(stream, &returnedFlags));
  REQUIRE((returnedFlags & expectedFlag) == expectedFlag);
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * Test Description
 * ------------------------
 *  - Tests to verify Negative scenarios for the hipStreamGetFlags_spt API
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamGetFlags_spt.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */

TEST_CASE("Unit_hipStreamGetFlags_spt_Negative") {
  hipStream_t validStream;
  unsigned int flags;

  HIP_CHECK(hipStreamCreate(&validStream));

  SECTION("Nullptr Stream && Valid Flags") {
#if HT_AMD
    HIP_CHECK_ERROR(hipStreamGetFlags_spt(nullptr, &flags), hipSuccess);
#elif HT_NVIDIA
    HIP_CHECK(hipStreamGetFlags_spt(nullptr, &flags));
#endif
  }

  SECTION("Valid Stream && Nullptr Flags") {
    HIP_CHECK_ERROR(hipStreamGetFlags_spt(validStream, nullptr), hipErrorInvalidValue);
  }

  HIP_CHECK(hipStreamDestroy(validStream));
}
/**
 * Test Description
 * ------------------------
 *  - Test flag value when streams created with CUMask
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamGetFlags_spt.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */

#if HT_AMD
TEST_CASE("Unit_hipStreamGetFlags_spt_StreamsCreatedWithCUMask") {
  hipStream_t stream;
  unsigned int flags;
  const uint32_t cuMask = 0xffffffff;
  HIP_CHECK(hipExtStreamCreateWithCUMask(&stream, 1, &cuMask));
  HIP_CHECK(hipStreamGetFlags_spt(stream, &flags));
  REQUIRE(flags == hipStreamDefault);
  HIP_CHECK(hipStreamDestroy(stream));
}
#endif

/**
 * End doxygen group StreamTest.
 * @}
 */
