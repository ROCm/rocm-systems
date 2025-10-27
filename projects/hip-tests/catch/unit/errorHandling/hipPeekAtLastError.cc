/*
Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip_test_process.hh>
#include <threaded_zig_zag_test.hh>

/**
 * @addtogroup hipPeekAtLastError hipPeekAtLastError
 * @{
 * @ingroup ErrorTest
 * `hipPeekAtLastError(void)` -
 * Return last error returned by any HIP runtime API call.
 */

/**
 * Test Description
 * ------------------------
 *  - Validate that `hipErrorInvalidValue` is returned after invalid `hipMalloc`
 * call.
 *  - Validate that `hipSuccess` is returned again for getting the last error.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipPeekAtLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipPeekAtLastError_Positive_Basic") {
  HIP_CHECK(hipPeekAtLastError());
  HIP_CHECK_ERROR(hipMalloc(nullptr, 1), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipPeekAtLastError(), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
  HIP_CHECK(hipPeekAtLastError());
}

/**
 * Test Description
 * ------------------------
 *  - Validate that appropriate error is returned when working with multiple
 * threads.
 *  - Validate that appropriate error is returned for getting the last error
 * when working with multiple threads.
 *  - Cause error on purpose within one of the threads.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipPeekAtLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipPeekAtLastError_Positive_Threaded") {
  class HipPeekAtLastErrorTest : public ThreadedZigZagTest<HipPeekAtLastErrorTest> {
   public:
    void TestPart2() { REQUIRE_THREAD(hipMalloc(nullptr, 1) == hipErrorInvalidValue); }
    void TestPart3() {
      HIP_CHECK(hipPeekAtLastError());
      HIP_CHECK(hipGetLastError());
    }
    void TestPart4() { REQUIRE_THREAD(hipPeekAtLastError() == hipErrorInvalidValue); }
  };

  HipPeekAtLastErrorTest test;
  test.run();
}
/**
 * Test Description
 * ------------------------
 *  - Verify hipPeekAtLastError status with hipMalloc api invalid arg call.
 *    Status should be last Error reported in the thread/Runtime.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipPeekAtLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipPeekAtLastError_Positive") {
    HIP_CHECK_ERROR(hipMalloc(nullptr, 1), hipErrorInvalidValue);
    int* A_d;
    HIP_CHECK(hipMalloc(&A_d, 1024));
    HIP_CHECK_ERROR(hipPeekAtLastError(), hipErrorInvalidValue);
    HIP_CHECK(hipFree(A_d));
    HIP_CHECK_ERROR(hipPeekAtLastError(), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipPeekAtLastError(), hipErrorInvalidValue);
}
/**
 * Test Description
 * ------------------------
 *  - Verify hipPeekAtLastError status with an Error - Success calls
 *    Each time status should return the corresponding Error when it called.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipPeekAtLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipPeekAtLastError_Chk_Updated_Status") {
    hipGraph_t graph;
    int value = 0;
    HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipPeekAtLastError(), hipErrorInvalidValue);
    int* C_d;
    HIP_CHECK(hipMalloc(&C_d, 1024));
    HIP_CHECK_ERROR(hipDeviceGetGraphMemAttribute(-1, hipGraphMemAttrUsedMemCurrent, &value),
                    hipErrorInvalidDevice);
    HIP_CHECK(hipFree(C_d));
    HIP_CHECK_ERROR(hipPeekAtLastError(), hipErrorInvalidDevice);
}
/**
 * Test Description
 * ------------------------
 *  - Verify hipPeekAtLastError status along with the hipGetLastError API call.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipPeekAtLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipPeekAtLastError_Chk_Along_hipGetLastError") {
    hipGraph_t graph;
    HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipGetLastError(), hipSuccess);
    int* C_d;
    HIP_CHECK_ERROR(hipPeekAtLastError(), hipSuccess);
    HIP_CHECK(hipMalloc(&C_d, 1024));
    HIP_CHECK(hipFree(C_d));
    HIP_CHECK_ERROR(hipPeekAtLastError(), hipSuccess);
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipPeekAtLastError status
 *    with different Error and Success combinations.
 *    Each time status should return the corresponding Error when it called.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipPeekAtLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipPeekAtLastError_Error_Combinations") {
  int value = 0;
  hipGraph_t graph;
  SECTION("A case with Error-Error-Success-Success") {
      HIP_CHECK(hipPeekAtLastError());
      HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipDeviceGetGraphMemAttribute(-1, hipGraphMemAttrUsedMemCurrent, &value),
                      hipErrorInvalidDevice);
      HIP_CHECK_ERROR(hipPeekAtLastError(), hipErrorInvalidDevice);
      int* A_d;
      HIP_CHECK(hipMalloc(&A_d, 1024));
      HIP_CHECK(hipFree(A_d));
      HIP_CHECK_ERROR(hipPeekAtLastError(), hipErrorInvalidDevice);
  }
  SECTION("A case with Error-Success-Error-Success") {
      HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
      int* A_d;
      HIP_CHECK(hipMalloc(&A_d, 1024));
      HIP_CHECK_ERROR(hipDeviceGetGraphMemAttribute(-1, hipGraphMemAttrUsedMemCurrent, &value),
                      hipErrorInvalidDevice);
      HIP_CHECK(hipFree(A_d));
      HIP_CHECK_ERROR(hipPeekAtLastError(), hipErrorInvalidDevice);
  }
  SECTION("A case with Success-Error-Error-Success") {
      int *A_d;
      HIP_CHECK(hipMalloc(&A_d, 1024));
      HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipDeviceGetGraphMemAttribute(-1, hipGraphMemAttrUsedMemCurrent, &value),
                      hipErrorInvalidDevice);
      HIP_CHECK(hipFree(A_d));
      HIP_CHECK_ERROR(hipPeekAtLastError(), hipErrorInvalidDevice);
  }
  SECTION("A Case with Success-Error-Success-Error") {
      int *A_d;
      HIP_CHECK(hipMalloc(&A_d, 1024));
      HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
      HIP_CHECK(hipFree(A_d));
      HIP_CHECK_ERROR(hipDeviceGetGraphMemAttribute(-1, hipGraphMemAttrUsedMemCurrent, &value),
                      hipErrorInvalidDevice);
      HIP_CHECK_ERROR(hipPeekAtLastError(), hipErrorInvalidDevice);
  }
}

static void thread_func() {
  HIP_CHECK_ERROR(hipPeekAtLastError(), hipSuccess);
  HIP_CHECK_ERROR(hipMalloc(nullptr, 1), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipPeekAtLastError(), hipErrorInvalidValue);
  int* A_d;
  HIP_CHECK(hipMalloc(&A_d, 1024));
  HIP_CHECK(hipFree(A_d));
}
/**
 * Test Description
 * ------------------------
 *  - Verify hipPeekAtLastError status with a runtime api invalid arg call.
 *    Check in other thread this error should not report by hipPeekAtLastError()
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipPeekAtLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipPeekAtLastError_With_Thread") {
  hipGraph_t graph;
    int *A_d;
    HIP_CHECK(hipMalloc(&A_d, 1024));
    HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
    std::thread t(thread_func);
    t.join();
    HIP_CHECK(hipFree(A_d));
    HIP_CHECK_ERROR(hipPeekAtLastError(), hipErrorInvalidValue);
}
/**
 * Test Description
 * ------------------------
 *  - Verify hipPeekAtLastError status in the multiple processes.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipPeekAtLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipPeekAtLastError_MultiProcess") {
  hipGraph_t graph;
    int *A_d;
    HIP_CHECK(hipMalloc(&A_d, 1024));
    HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
    hip::SpawnProc proc("hipPeekAtLastErrorEnv_Exe", true);
    HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipPeekAtLastError(), hipErrorInvalidValue);
    REQUIRE(proc.run() == 1);
    HIP_CHECK(hipFree(A_d));
    HIP_CHECK_ERROR(hipPeekAtLastError(), hipErrorInvalidValue);
}
static void __global__ emptyKernl() {}
/**
 * Test Description
 * ------------------------
 *  - Verify hipPeekAtLastError status with Invalid Configuration in kernel call.
 *    Kernel call invalid configuration- blocks=0 & threadsPerBlock=0
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipPeekAtLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
// Below test failed on NVIDIA due to error mismatch produced by the Invalid Kernel config.
// For more details please check the ticket SWDEV-501896 comments.
#if HT_AMD
TEST_CASE("Unit_hipPeekAtLastError_Kernel_Invalid_Config") {
  hipError_t ret;
    hipLaunchKernelGGL(emptyKernl, dim3(0), dim3(0), 0, 0);
    int* A_d;
    HIP_CHECK(hipMalloc(&A_d, 1024));
    ret = hipPeekAtLastError();
    REQUIRE(ret == hipErrorInvalidConfiguration);
    HIP_CHECK(hipFree(A_d));
}
#endif
/**
 * End doxygen group ErrorTest.
 * @}
 */
