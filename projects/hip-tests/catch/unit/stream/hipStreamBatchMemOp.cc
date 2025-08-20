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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
/**
 * @addtogroup hipStreamBatchMemOp hipStreamBatchMemOp
 * @{
 * @ingroup StreamTest
 * `hipError_t hipStreamBatchMemOp(hipStream_t stream, unsigned int count,
                               hipStreamBatchMemOpParams* paramArray, unsigned
 int flags);` -
 * Enqueues an array of stream memory operations in the stream.
 */
/**
 * Test Description
 * ------------------------
 * - Verify the Negative cases of the hipStreamBatchMemOp API.
 * Test source
 * ------------------------
 *    - unit/stream/hipStreamBatchMemOp.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipStreamBatchMemOp_Negative_Tests") {
  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));
  REQUIRE(stream != nullptr);
  int totalOps = 2;
  static hipStreamBatchMemOpParams paramArray[2], invalidParamArray[2];
  std::vector<hipDeviceptr_t> opsArray(1);
  HIP_CHECK(hipMalloc((void**)&opsArray[0], sizeof(uint32_t)));

  paramArray[0].operation = hipStreamMemOpWriteValue32;
  paramArray[0].writeValue.address = opsArray[0];
  paramArray[0].writeValue.value = 1000;
  paramArray[0].writeValue.flags = 0x0;
  paramArray[0].writeValue.alias = 0;

  paramArray[1].operation = hipStreamMemOpWaitValue32;
  paramArray[1].waitValue.address = opsArray[0];
  paramArray[1].waitValue.value = 1000;
  paramArray[1].waitValue.flags = hipStreamWaitValueEq;

  invalidParamArray[0].operation = hipStreamMemOpBarrier;
  invalidParamArray[0].writeValue.address = opsArray[0];
  invalidParamArray[0].writeValue.value = 1000;
  invalidParamArray[0].writeValue.flags = 32;
  invalidParamArray[0].writeValue.alias = 0;

  invalidParamArray[1].operation = hipStreamMemOpBarrier;
  invalidParamArray[1].waitValue.address = opsArray[0];
  invalidParamArray[1].waitValue.value = 1000;
  invalidParamArray[1].waitValue.flags = hipStreamWaitValueEq;

  SECTION("Stream as a nullptr") {
    HIP_CHECK_ERROR(hipStreamBatchMemOp(nullptr, totalOps, paramArray, 0), hipErrorInvalidValue);
  }

  SECTION("Invalid Stream") {
    HIP_CHECK_ERROR(hipStreamBatchMemOp(reinterpret_cast<hipStream_t>(-1), totalOps, paramArray, 0),
                    hipErrorContextIsDestroyed);
  }

  SECTION("Parameter Array as a nullptr") {
    HIP_CHECK_ERROR(hipStreamBatchMemOp(stream, totalOps, nullptr, 0), hipErrorInvalidValue);
  }

  SECTION("More than 256 Total Operations") {
    HIP_CHECK_ERROR(hipStreamBatchMemOp(stream, 1000, paramArray, 0), hipErrorInvalidValue);
  }

  SECTION("Total Operations less than 0") {
    HIP_CHECK_ERROR(hipStreamBatchMemOp(stream, -4, paramArray, 0), hipErrorInvalidValue);
  }
  SECTION("Total Operations Zero") {
    HIP_CHECK_ERROR(hipStreamBatchMemOp(stream, 0, paramArray, 0), hipErrorInvalidValue);
  }

  SECTION("Flag value not Zero") {
    HIP_CHECK_ERROR(hipStreamBatchMemOp(stream, totalOps, paramArray, -6), hipErrorInvalidValue);
  }

// Disabled due to defect SWDEV-502219
#if 0
  SECTION("InValid Parameter Array") {
    HIP_CHECK_ERROR(hipStreamBatchMemOp(stream, totalOps, invalidParamArray, 0),
                    hipErrorInvalidValue);
  }
#endif
  HIP_CHECK(hipFree((void*)opsArray[0]));
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * End doxygen group StreamTest.
 * @}
 */
