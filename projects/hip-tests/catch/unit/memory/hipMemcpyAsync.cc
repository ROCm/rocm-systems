/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <memcpy1d_tests_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

TEST_CASE(Unit_hipMemcpyAsync_Positive_Basic) {
  using namespace std::placeholders;
  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);
  const hipStream_t stream = stream_guard.stream();

  MemcpyWithDirectionCommonTests<true>(std::bind(hipMemcpyAsync, _1, _2, _3, _4, stream), stream);
}

TEST_CASE(Unit_hipMemcpyAsync_Positive_Synchronization_Behavior) {
  using namespace std::placeholders;
  HIP_CHECK(hipDeviceSynchronize());

  // This behavior differs on NVIDIA and AMD, on AMD the hipMemcpy calls is synchronous with
  // respect to the host
  SECTION("Host pageable memory to device memory") {
#if HT_NVIDIA
    MemcpyHPageabletoDSyncBehavior(
        std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyHostToDevice, nullptr), false);
#endif
  }

  SECTION("Host pinned memory to device memory") {
    MemcpyHPinnedtoDSyncBehavior(
        std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyHostToDevice, nullptr), false);
  }

  SECTION("Device memory to pageable host memory") {
    MemcpyDtoHPageableSyncBehavior(
        std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToHost, nullptr), true);
  }

  SECTION("Device memory to pinned host memory") {
    MemcpyDtoHPinnedSyncBehavior(
        std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToHost, nullptr), false);
  }

  SECTION("Device memory to device memory") {
    MemcpyDtoDSyncBehavior(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToDevice, nullptr),
                           false);
  }

  SECTION("Device memory to device Memory No CU") {
    MemcpyDtoDSyncBehavior(
        std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToDeviceNoCU, nullptr), false);
  }

  SECTION("Host memory to host memory") {
    MemcpyHtoHSyncBehavior(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyHostToHost, nullptr),
                           true);
  }
}

TEST_CASE(Unit_hipMemcpyAsync_Negative_Parameters) {
  using namespace std::placeholders;

  SECTION("Host to device") {
    LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyHostToDevice, nullptr),
                              device_alloc.ptr(), host_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyAsync(device_alloc.ptr(), host_alloc.ptr(), kPageSize,
                                     static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidMemcpyDirection);
    }
  }

  SECTION("Device to host") {
    LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToHost, nullptr),
                              host_alloc.ptr(), device_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyAsync(host_alloc.ptr(), device_alloc.ptr(), kPageSize,
                                     static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidMemcpyDirection);
    }
  }

  SECTION("Host to host") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipHostMalloc, kPageSize);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyHostToHost, nullptr),
                              dst_alloc.ptr(), src_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyAsync(dst_alloc.ptr(), src_alloc.ptr(), kPageSize,
                                     static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidMemcpyDirection);
    }
  }

  SECTION("Device to device") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);

    MemcpyCommonNegativeTests(
        std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToDevice, nullptr), dst_alloc.ptr(),
        src_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyAsync(src_alloc.ptr(), dst_alloc.ptr(), kPageSize,
                                     static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidMemcpyDirection);
    }
  }
}

TEST_CASE(Unit_hipMemcpyAsync_Capture) {
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(
      hipMemcpyAsync(dst_alloc.ptr(), src_alloc.ptr(), kPageSize, hipMemcpyDeviceToDevice, stream));
  END_CAPTURE(stream);

  HIP_CHECK(hipStreamDestroy(stream));
}
