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
#include <numeric>

TEST_CASE("Unit_hipMemcpyDtoHAsync_Positive_Basic") {
  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);

  const auto f = [stream = stream_guard.stream()](void* dst, void* src, size_t count) {
    return hipMemcpyDtoHAsync(dst, reinterpret_cast<hipDeviceptr_t>(src), count, stream);
  };
  MemcpyDeviceToHostShell<true>(f, stream_guard.stream());
}

TEST_CASE("Unit_hipMemcpyDtoHAsync_Positive_Synchronization_Behavior") {
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Device memory to pageable host memory") {
    MemcpyDtoHPageableSyncBehavior(
        [](void* dst, void* src, size_t count) {
          return hipMemcpyDtoHAsync(dst, reinterpret_cast<hipDeviceptr_t>(src), count, nullptr);
        },
        true);
  }

  SECTION("Device memory to pinned host memory") {
    MemcpyDtoHPinnedSyncBehavior(
        [](void* dst, void* src, size_t count) {
          return hipMemcpyDtoHAsync(dst, reinterpret_cast<hipDeviceptr_t>(src), count, nullptr);
        },
        false);
  }
}

TEST_CASE("Unit_hipMemcpyDtoHAsync_Negative_Parameters") {
  using namespace std::placeholders;
  LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoHAsync(dst, reinterpret_cast<hipDeviceptr_t>(src), count, nullptr);
      },
      host_alloc.ptr(), device_alloc.ptr(), kPageSize);
}

TEST_CASE("Unit_hipMemcpyHtoDAsync_Positive_Basic") {
  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);

  const auto f = [stream = stream_guard.stream()](void* dst, void* src, size_t count) {
    return hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst), src, count, stream);
  };
  MemcpyHostToDeviceShell<true>(f, stream_guard.stream());
}

TEST_CASE("Unit_hipMemcpyHtoDAsync_Positive_Synchronization_Behavior") {
  // This behavior differs on NVIDIA and AMD, on AMD the hipMemcpy calls is synchronous with
  // respect to the host
#if HT_AMD
  HipTest::HIP_SKIP_TEST(
      "EXSWCPHIPT-127 - MemcpyAsync from host to device memory behavior differs on AMD and "
      "Nvidia");
  return;
#endif
  MemcpyHPinnedtoDSyncBehavior(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst), src, count, nullptr);
      },
      false);
}

TEST_CASE("Unit_hipMemcpyHtoDAsync_Negative_Parameters") {
  using namespace std::placeholders;
  LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst), src, count, nullptr);
      },
      device_alloc.ptr(), host_alloc.ptr(), kPageSize);
}

TEST_CASE("Unit_hipMemcpyDtoDAsync_Positive_Basic") {
  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);

  SECTION("Device to device") {
    SECTION("Peer access enabled") {
      MemcpyDeviceToDeviceShell<true, true>(
          [stream = stream_guard.stream()](void* dst, void* src, size_t count) {
            return hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst),
                                      reinterpret_cast<hipDeviceptr_t>(src), count, stream);
          },
          stream_guard.stream());
    }
    SECTION("Peer access disabled") {
      MemcpyDeviceToDeviceShell<true, false>(
          [stream = stream_guard.stream()](void* dst, void* src, size_t count) {
            return hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst),
                                      reinterpret_cast<hipDeviceptr_t>(src), count, stream);
          },
          stream_guard.stream());
    }
  }
}

TEST_CASE("Unit_hipMemcpyDtoDAsync_Positive_Synchronization_Behavior") {
  MemcpyDtoDSyncBehavior(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst),
                                  reinterpret_cast<hipDeviceptr_t>(src), count, nullptr);
      },
      false);
}

TEST_CASE("Unit_hipMemcpyDtoDAsync_Negative_Parameters") {
  using namespace std::placeholders;
  LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst),
                                  reinterpret_cast<hipDeviceptr_t>(src), count, nullptr);
      },
      dst_alloc.ptr(), src_alloc.ptr(), kPageSize);
}

/**
 * Test Description
 * ------------------------
 *  - Basic functional testcase to trigger capturehipMemcpyDtoHAsync internal api
 *  to improve code coverage.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyAsync_derivatives.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipMemcpyDtoHAsync_Capture") {
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  auto host_dst = std::make_unique<int[]>(kPageSize);
  auto host_src = std::make_unique<int[]>(kPageSize);
  int* device_src = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_src), sizeof(int) * kPageSize));

  std::iota(host_src.get(), host_src.get() + kPageSize, 0);

  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(device_src), host_src.get(),
                          sizeof(int) * kPageSize));

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipMemcpyDtoHAsync(host_dst.get(), reinterpret_cast<hipDeviceptr_t>(device_src),
                               sizeof(int) * kPageSize, stream));
  END_CAPTURE(stream);

  HIP_CHECK(hipStreamSynchronize(stream));

  for (int i = 0; i < kPageSize; ++i) {
    REQUIRE(host_dst[i] == host_src[i]);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(device_src));
}

/**
 * Test Description
 * ------------------------
 *  - Basic functional testcase to trigger capturehipMemcpyHtoDAsync internal api
 *  to improve code coverage.
 * Test source
 * ------------------------
 */
TEST_CASE("Unit_hipMemcpyHtoDAsync_Capture") {
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  auto host_src = std::make_unique<int[]>(kPageSize);
  auto host_dst = std::make_unique<int[]>(kPageSize);
  int* device_ptr = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_ptr), sizeof(int) * kPageSize));

  std::iota(host_src.get(), host_src.get() + kPageSize, 0);

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(device_ptr), host_src.get(),
                               sizeof(int) * kPageSize, stream));
  END_CAPTURE(stream);

  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipMemcpyDtoH(host_dst.get(), reinterpret_cast<hipDeviceptr_t>(device_ptr),
                          sizeof(int) * kPageSize));
  for (int i = 0; i < kPageSize; ++i) {
    REQUIRE(host_dst[i] == host_src[i]);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(device_ptr));
}
