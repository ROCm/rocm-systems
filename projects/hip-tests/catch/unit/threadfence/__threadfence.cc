/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include <cmd_options.hh>
#include <hip_test_common.hh>
#include <resource_guards.hh>

#include "threadfence_common.hh"

/**
 * @addtogroup __threadfence __threadfence
 * @{
 * @ingroup ThreadfenceTest
 */

/**
 * Test Description
 * ------------------------
 *    - Basic test for a device-wide memory fence on shared memory.
 *
 * Test source
 * ------------------------
 *    - unit/threadfence/__threadfence.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___threadfence_Positive_Basic_Shared") {
  LinearAllocGuard<int> in_dev(LinearAllocs::hipMalloc, 2 * sizeof(int));
  LinearAllocGuard<int> out_dev(LinearAllocs::hipMalloc, 2 * sizeof(int));

  LinearAllocGuard<int> out_host(LinearAllocs::hipHostMalloc, 2 * sizeof(int));

  for (int i = 0; i < cmd_options.iterations; ++i) {
    HIP_CHECK(hipMemsetD32((hipDeviceptr_t)(&(in_dev.ptr()[0])), kInitVal1, 1));
    HIP_CHECK(hipMemsetD32((hipDeviceptr_t)(&(in_dev.ptr()[1])), kInitVal2, 1));

    HipTest::launchKernel(ThreadfenceTestKernel<ThreadfenceScope::kDevice, true>, 1, 2,
                          4 * sizeof(int), nullptr, out_dev.ptr(), in_dev.ptr());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(out_host.host_ptr(), out_dev.ptr(), 2 * sizeof(int), hipMemcpyDefault));

    REQUIRE(!(out_host.ptr()[0] == kInitVal1 && out_host.ptr()[1] == kSetVal2));
  }
}

/**
 * Test Description
 * ------------------------
 *    - Basic test for a device-wide memory fence on global memory.
 *
 * Test source
 * ------------------------
 *    - unit/threadfence/__threadfence.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___threadfence_Positive_Basic_Global") {
  LinearAllocGuard<int> in_dev(LinearAllocs::hipMalloc, 2 * sizeof(int));
  LinearAllocGuard<int> out_dev(LinearAllocs::hipMalloc, 2 * sizeof(int));

  LinearAllocGuard<int> out_host(LinearAllocs::hipHostMalloc, 2 * sizeof(int));

  for (int i = 0; i < cmd_options.iterations; ++i) {
    HIP_CHECK(hipMemsetD32((hipDeviceptr_t)(&(in_dev.ptr()[0])), kInitVal1, 1));
    HIP_CHECK(hipMemsetD32((hipDeviceptr_t)(&(in_dev.ptr()[1])), kInitVal2, 1));

    HipTest::launchKernel(ThreadfenceTestKernel<ThreadfenceScope::kDevice, false>, 2, 1, 0, nullptr,
                          out_dev.ptr(), in_dev.ptr());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(out_host.host_ptr(), out_dev.ptr(), 2 * sizeof(int), hipMemcpyDefault));

    REQUIRE(!(out_host.ptr()[0] == kInitVal1 && out_host.ptr()[1] == kSetVal2));
  }
}

/**
 * Test Description
 * ------------------------
 *    - Basic test for a device-wide memory fence on page-locked host memory.
 *
 * Test source
 * ------------------------
 *    - unit/threadfence/__threadfence.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___threadfence_Positive_Basic_Pinned") {
  LinearAllocGuard<int> in_host(LinearAllocs::hipHostMalloc, 2 * sizeof(int));
  LinearAllocGuard<int> out_host(LinearAllocs::hipHostMalloc, 2 * sizeof(int));

  for (int i = 0; i < cmd_options.iterations; ++i) {
    in_host.host_ptr()[0] = kInitVal1;
    in_host.host_ptr()[1] = kInitVal2;

    HipTest::launchKernel(ThreadfenceTestKernel<ThreadfenceScope::kDevice, false>, 2, 1, 0, nullptr,
                          out_host.host_ptr(), in_host.host_ptr());
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(!(out_host.host_ptr()[0] == kInitVal1 && out_host.ptr()[1] == kSetVal2));
  }
}

/**
 * Test Description
 * ------------------------
 *    - Basic test for a device-wide memory fence on managed memory.
 *
 * Test source
 * ------------------------
 *    - unit/threadfence/__threadfence.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___threadfence_Positive_Basic_Managed") {
  LinearAllocGuard<int> in_host(LinearAllocs::hipMallocManaged, 2 * sizeof(int));
  LinearAllocGuard<int> out_host(LinearAllocs::hipMallocManaged, 2 * sizeof(int));

  for (int i = 0; i < cmd_options.iterations; ++i) {
    in_host.host_ptr()[0] = kInitVal1;
    in_host.host_ptr()[1] = kInitVal2;

    HipTest::launchKernel(ThreadfenceTestKernel<ThreadfenceScope::kDevice, false>, 2, 1, 0, nullptr,
                          out_host.ptr(), in_host.ptr());
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(!(out_host.host_ptr()[0] == kInitVal1 && out_host.ptr()[1] == kSetVal2));
  }
}

/**
 * Test Description
 * ------------------------
 *    - Basic test for a device-wide memory fence on global peer device memory.
 *
 * Test source
 * ------------------------
 *    - unit/threadfence/__threadfence.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___threadfence_Positive_Basic_Peer") {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("At least 2 devices are required");
    return;
  }

  int can_access_peer = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 0, 1));
  if (!can_access_peer) {
    std::string msg = "Skipped as peer access cannot be enabled between devices";
    HipTest::HIP_SKIP_TEST(msg.c_str());
    return;
  }

  HIP_CHECK(hipSetDevice(0));

  LinearAllocGuard<int> in_dev(LinearAllocs::hipMalloc, 2 * sizeof(int));
  LinearAllocGuard<int> out_dev(LinearAllocs::hipMalloc, 2 * sizeof(int));

  LinearAllocGuard<int> out_host(LinearAllocs::hipHostMalloc, 2 * sizeof(int));

  for (int i = 0; i < cmd_options.iterations; ++i) {
    HIP_CHECK(hipMemsetD32((hipDeviceptr_t)(&(in_dev.ptr()[0])), kInitVal1, 1));
    HIP_CHECK(hipMemsetD32((hipDeviceptr_t)(&(in_dev.ptr()[1])), kInitVal2, 1));

    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipDeviceEnablePeerAccess(0, 0));
    HipTest::launchKernel(ThreadfenceTestKernel<ThreadfenceScope::kDevice, false>, 2, 1, 0, nullptr,
                          out_dev.ptr(), in_dev.ptr());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipSetDevice(0));

    HIP_CHECK(hipMemcpy(out_host.host_ptr(), out_dev.ptr(), 2 * sizeof(int), hipMemcpyDefault));

    REQUIRE(!(out_host.ptr()[0] == kInitVal1 && out_host.ptr()[1] == kSetVal2));
  }
}

/**
 * End doxygen group ThreadfenceTest.
 * @}
 */
