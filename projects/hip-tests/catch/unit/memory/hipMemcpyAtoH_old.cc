/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
/*
 * Test Scenarios:
 * 1. Perform host and Pinned host Memory
 * 2. Perform bytecount 0  validation for hipMemcpyAtoH API
 * 3. Allocate Memory from one GPU device and call hipMemcpyAtoH from Peer
 *    GPU device
 * 4. Perform hipMemcpyAtoH Negative Scenarios
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>


static constexpr auto NUM_W{10};
static constexpr auto NUM_H{1};
static constexpr auto copy_bytes{2};

/*
This testcase performs the basic and pinned host memory scenarios
of hipMemcpyAtoH API
Memory is allocated in GPU-0 and the API is triggered from GPU-1
Input: "A_d" initialized with "hData" Pi value
Output:"B_h" host variable output of hipMemcpyAtoH API
        is then validated with "hData"
*/
#if HT_AMD
TEMPLATE_TEST_CASE(Unit_hipMemcpyAtoH_multiDevice_PeerDeviceContext, char, int, float) {
  CHECK_IMAGE_SUPPORT

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    int peerAccess = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&peerAccess, 1, 0));
    if (!peerAccess) {
      SUCCEED("Skipped the test as there is no peer access");
    } else {
      HIP_CHECK(hipSetDevice(0));
      hipArray_t A_d;
      TestType *hData{nullptr}, *B_h{nullptr};
      size_t width{NUM_W * sizeof(TestType)};

      // Initialization of data
      HipTest::initArrays<TestType>(nullptr, nullptr, nullptr, &hData, &B_h, nullptr, NUM_W);
      HipTest::setDefaultData<TestType>(NUM_W, hData, B_h, nullptr);
      hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();
      HIP_CHECK(hipMallocArray(&A_d, &desc, NUM_W, NUM_H, hipArrayDefault));
      HIP_CHECK(hipMemcpy2DToArray(A_d, 0, 0, hData, width, width, NUM_H, hipMemcpyHostToDevice));

      HIP_CHECK(hipDeviceSynchronize());
      // Changing the device context
      HIP_CHECK(hipSetDevice(1));

      // Performing API call
      REQUIRE(hipMemcpyAtoH(B_h, A_d, 0, copy_bytes * sizeof(TestType)) == hipSuccess);
      // Validating the result
      REQUIRE(HipTest::checkArray(B_h, hData, copy_bytes, NUM_H) == true);

      // DeAllocating the memory
      HIP_CHECK(hipFreeArray(A_d));
      REQUIRE(HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr, hData, B_h, nullptr,
                                            false) == true);
    }
  } else {
    SUCCEED("skipping the testcases as numDevices < 2");
  }
}
#endif
