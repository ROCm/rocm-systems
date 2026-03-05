/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

/*
This testfile verifies the following scenarios of hipMemcpyParam2DAsync API
1. Negative Scenarios
2. Extent Validation Scenarios
3. D2D copy for different datatypes
4. H2D and D2H copy for different datatypes
5. Device context change scenario where memory allocated in one GPU
   stream created in another GPU
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

static constexpr size_t NUM_W{10};
static constexpr size_t NUM_H{10};
/*
 * This testcase verifies D2D functionality of hipMemcpyParam2DAsync API
 * Where Memory is allocated in GPU-0 and stream is created in GPU-1
 *
 * Input: Intializing "A_d" device variable with "C_h" host variable
 * Output: "A_d" device variable to "E_d" device variable
 *
 * Validating the result by copying "E_d" to "A_h" and checking
 * it with the initalized data "C_h".
 *
 */
TEMPLATE_TEST_CASE("Unit_hipMemcpyParam2DAsync_multiDevice_StreamOnDiffDevice",
                   "[hipMemcpyParam2DAsync][multigpu]", char, float, int,
                   double, long double) {
  CHECK_IMAGE_SUPPORT

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    // Allocating and Initializing the data
    HIP_CHECK(hipSetDevice(0));
    TestType *A_h{nullptr}, *C_h{nullptr}, *A_d{nullptr};
    size_t pitch_A;
    size_t width{NUM_W * sizeof(TestType)};
    HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width, NUM_H));
    HipTest::initArrays<TestType>(nullptr, nullptr, nullptr, &A_h, nullptr, &C_h, width * NUM_H,
                                  false);
    HipTest::setDefaultData<TestType>(NUM_W * NUM_H, A_h, nullptr, C_h);
    int peerAccess = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&peerAccess, 1, 0));
    if (!peerAccess) {
      SUCCEED("Skipped the test as there is no peer access");
    } else {
      TestType* E_d{nullptr};
      size_t pitch_E;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&E_d), &pitch_E, width, NUM_H));

      // Initalizing A_d with C_h
      HIP_CHECK(hipSetDevice(1));
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      HIP_CHECK(hipMemcpy2DAsync(A_d, pitch_A, C_h, width, NUM_W * sizeof(TestType), NUM_H,
                                 hipMemcpyHostToDevice, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      // Device to Device
      hip_Memcpy2D desc = {};
      desc.srcMemoryType = hipMemoryTypeDevice;
      desc.srcHost = A_d;
      desc.srcDevice = hipDeviceptr_t(A_d);
      desc.srcPitch = pitch_A;
      desc.dstMemoryType = hipMemoryTypeDevice;
      desc.dstHost = E_d;
      desc.dstDevice = hipDeviceptr_t(E_d);
      desc.dstPitch = pitch_E;
      desc.WidthInBytes = NUM_W * sizeof(TestType);
      desc.Height = NUM_H;
      REQUIRE(hipMemcpyParam2DAsync(&desc, stream) == hipSuccess);
      HIP_CHECK(hipStreamSynchronize(stream));

      // Copying the result E_d to A_h host variable
      HIP_CHECK(hipMemcpy2D(A_h, width, E_d, pitch_E, NUM_W * sizeof(TestType), NUM_H,
                            hipMemcpyDeviceToHost));
      HIP_CHECK(hipDeviceSynchronize());
      // Validating the result
      REQUIRE(HipTest::checkArray<TestType>(A_h, C_h, NUM_W, NUM_H) == true);

      // DeAllocating the memory
      HIP_CHECK(hipFree(E_d));
      HIP_CHECK(hipFree(A_d));
      HIP_CHECK(hipStreamDestroy(stream));
      HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr, A_h, nullptr, C_h, false);
    }
  } else {
    SUCCEED("skipping the testcases as numDevices < 2");
  }
}
