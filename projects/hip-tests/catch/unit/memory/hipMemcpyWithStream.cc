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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <memcpy1d_tests_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

TEST_CASE("Unit_hipMemcpy_Positive_Basic") { MemcpyWithDirectionCommonTests<false>(hipMemcpy); }

TEST_CASE("Unit_hipMemcpy_Positive_Synchronization_Behavior") {
  using namespace std::placeholders;
  HIP_CHECK(hipDeviceSynchronize());

  // For transfers from device to either pageable or pinned host memory, the function returns only
  // once the copy has completed
  SECTION("Device memory to host memory") {
    const auto f = std::bind(hipMemcpy, _1, _2, _3, hipMemcpyDeviceToHost);
    MemcpyDtoHPageableSyncBehavior(f, true);
    MemcpyDtoHPinnedSyncBehavior(f, true);
  }

  // For transfers from device memory to device memory, no host-side synchronization is performed.
  SECTION("Device memory to device memory") {
    MemcpyDtoDSyncBehavior(std::bind(hipMemcpy, _1, _2, _3, hipMemcpyDeviceToDevice), false);
  }

  // For transfers from any host memory to any host memory, the function is fully synchronous with
  // respect to the host
  SECTION("Host memory to host memory") {
    MemcpyHtoHSyncBehavior(std::bind(hipMemcpy, _1, _2, _3, hipMemcpyHostToHost), true);
  }
}

TEST_CASE("Unit_hipMemcpy_Negative_Parameters") {
  using namespace std::placeholders;

  SECTION("Host to device") {
    LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);
    MemcpyWithDirectionCommonNegativeTests(hipMemcpy, device_alloc.ptr(), host_alloc.ptr(),
                                           kPageSize, hipMemcpyHostToDevice);
  }

  SECTION("Device to host") {
    LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);
    MemcpyWithDirectionCommonNegativeTests(hipMemcpy, host_alloc.ptr(), device_alloc.ptr(),
                                           kPageSize, hipMemcpyDeviceToHost);
  }

  SECTION("Host to host") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipHostMalloc, kPageSize);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipHostMalloc, kPageSize);
    MemcpyWithDirectionCommonNegativeTests(hipMemcpy, dst_alloc.ptr(), src_alloc.ptr(), kPageSize,
                                           hipMemcpyHostToHost);
  }

  SECTION("Device to device") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);
    MemcpyWithDirectionCommonNegativeTests(hipMemcpy, dst_alloc.ptr(), src_alloc.ptr(), kPageSize,
                                           hipMemcpyDeviceToDevice);
  }
}

TEST_CASE("Unit_hipMemcpyWithStream_Capture") {
  constexpr size_t kNumElements = 1024;
  constexpr int kInitialValue = 42;

  std::vector<int> host_src(kNumElements, kInitialValue);
  std::vector<int> host_dst(kNumElements, 0);

  LinearAllocGuard<int> device_src(LinearAllocs::hipMalloc, kNumElements * sizeof(int));
  LinearAllocGuard<int> device_dst(LinearAllocs::hipMalloc, kNumElements * sizeof(int));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);

  HIP_CHECK(hipMemcpyWithStream(device_src.ptr(), host_src.data(), kNumElements * sizeof(int),
                                hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyWithStream(device_dst.ptr(), device_src.ptr(), kNumElements * sizeof(int),
                                hipMemcpyDeviceToDevice, stream));
  HIP_CHECK(hipMemcpyWithStream(host_dst.data(), device_dst.ptr(), kNumElements * sizeof(int),
                                hipMemcpyDeviceToHost, stream));

  END_CAPTURE(stream);

  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));

  REQUIRE(std::equal(host_src.begin(), host_src.end(), host_dst.begin()));
}
