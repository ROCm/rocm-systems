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

#include <cstring>

HIP_TEST_CASE(Unit_hipMemcpyAsync_Positive_Basic) {
  using namespace std::placeholders;
  const auto stream_type = isQuickLevel()
      ? GENERATE(Streams::created)
      : GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);
  const hipStream_t stream = stream_guard.stream();

  MemcpyWithDirectionCommonTests<true>(std::bind(hipMemcpyAsync, _1, _2, _3, _4, stream), stream);
}

HIP_TEST_CASE(Unit_hipMemcpyAsync_Positive_Synchronization_Behavior) {
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

HIP_TEST_CASE(Unit_hipMemcpyAsync_Negative_Parameters) {
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

HIP_TEST_CASE(Unit_hipMemcpyAsync_Capture) {
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

/*
 * Regression test for a silently lost host-to-device copy.
 *
 * A small async H2D copy from pageable memory is staged: the runtime memcpy's the source into a
 * pinned staging slot, then a blit shader reads that slot and writes the destination. On gfx11
 * slots are packed at the 64-byte cache line size the agent reports, so two consecutive copies
 * land in one 128-byte L2 line.
 *
 * The first copy of a pair is what loses the second one's data. Its shader pulls in the whole line
 * before the CPU fills the neighboring slot, and because staging is coarse-grained the CPU's store
 * never invalidates it. The second shader needed a system-scope acquire to drop that stale line
 * but got agent scope, which reaches L1 and not L2, so it read zeros and wrote them to the
 * destination.
 *
 * All four of these are needed; drop any one and the failure disappears:
 *   - the source is pageable host memory; a pinned source is read directly, with no staging
 *   - the copy is asynchronous
 *   - the copy is smaller than the 128-byte granule, so that two of them share a line; larger
 *     transfers get a line each, and past 16 KiB they leave the blit path for SDMA entirely
 *   - the stream is freshly created on every iteration, which is what keeps handing out pools whose
 *     untouched pages read back as the zeros that make a lost copy recognizable
 *
 * Due to the nature of the regression, the test needs to be probabilistic.
 */
HIP_TEST_CASE(Unit_hipMemcpyAsync_Regression_PageableStaging_ConsecutiveSmallCopies) {
  // Two copies have to share one 128-byte line for the stale read to be reachable; at 128 bytes
  // and above each gets a line of its own and nothing is ever lost.
  constexpr size_t kCopyBytes = 64;

  // Slots are handed out consecutively, so every second copy inherits a line its predecessor
  // already pulled into L2, and only those odd-numbered copies can be lost.
  constexpr int kCopiesPerStream = 32;

  constexpr int kIterations = 500;

  constexpr size_t kBlockBytes = kCopyBytes * kCopiesPerStream;

  LinearAllocGuard<unsigned char> pattern(LinearAllocs::malloc, kCopyBytes);
  LinearAllocGuard<unsigned char> expected(LinearAllocs::malloc, kBlockBytes);
  LinearAllocGuard<unsigned char> readback(LinearAllocs::malloc, kBlockBytes);
  for (size_t i = 0; i < kCopyBytes; ++i) {
    pattern.ptr()[i] = static_cast<unsigned char>(0xA5 + i);
  }
  for (int copy = 0; copy < kCopiesPerStream; ++copy) {
    std::memcpy(expected.ptr() + copy * kCopyBytes, pattern.ptr(), kCopyBytes);
  }

  LinearAllocGuard<unsigned char> device(LinearAllocs::hipMalloc, kBlockBytes);

  for (int iteration = 0; iteration < kIterations; ++iteration) {
    const StreamGuard stream_guard(Streams::created);
    const hipStream_t stream = stream_guard.stream();

    for (int copy = 0; copy < kCopiesPerStream; ++copy) {
      HIP_CHECK(hipMemcpyAsync(device.ptr() + copy * kCopyBytes, pattern.ptr(), kCopyBytes,
                               hipMemcpyDefault, stream));
    }

    // Poison the landing area so a zero readback can only have come from the device.
    std::memset(readback.ptr(), 0xFF, kBlockBytes);
    HIP_CHECK(hipMemcpyAsync(readback.ptr(), device.ptr(), kBlockBytes, hipMemcpyDefault, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    if (std::memcmp(readback.ptr(), expected.ptr(), kBlockBytes) == 0) {
      continue;
    }

    for (size_t byte = 0; byte < kBlockBytes; ++byte) {
      INFO("Iteration " << iteration << ", copy " << byte / kCopyBytes << " of "
                        << kCopiesPerStream << ", byte " << byte % kCopyBytes << ", wrote 0x"
                        << std::hex << static_cast<int>(expected.ptr()[byte]) << ", read 0x"
                        << static_cast<int>(readback.ptr()[byte]));
      REQUIRE(readback.ptr()[byte] == expected.ptr()[byte]);
    }
  }
}
