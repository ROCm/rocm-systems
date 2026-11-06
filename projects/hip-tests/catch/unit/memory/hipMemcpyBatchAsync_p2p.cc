/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <limits>
#include <utility>
#include <vector>

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <resource_guards.hh>

#include "memcpyBatchAsync_common.hh"

#if HT_AMD

/**
 * Batched buffer exchange between device 0 and device 1: for each batch entry, one allocation on
 * each GPU is swapped via hipMemcpyFlagExtOpSwap with mutual peer access enabled; the stream and
 * hipMemcpyBatchAsync run on device 0.
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_P2P_Swap) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST("Skipping because fewer than 2 devices are available");
  }

  const int device_for_a = 0;
  const int device_for_b = 1;
  int can_access_peer_a_to_b = 0;
  int can_access_peer_b_to_a = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer_a_to_b, device_for_a, device_for_b));
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer_b_to_a, device_for_b, device_for_a));

  if (!can_access_peer_a_to_b || !can_access_peer_b_to_a) {
    HIP_SKIP_TEST(
        "Skipping because peer access is not supported in "
        "both directions between device 0 and "
        "device 1");
  }

  const size_t count = GENERATE(2, 3, 8);
  const size_t size_in_bytes = GENERATE(as<size_t>{}, 1, 63, 4096);
  const hipError_t expectedError = getSwapExpectedReturn(
      LinearAllocs::hipMalloc, LinearAllocs::hipMalloc, device_for_a, device_for_b);

  std::vector<std::vector<unsigned char>> initial_values_a(
      count, std::vector<unsigned char>(size_in_bytes, 10));
  std::vector<std::vector<unsigned char>> initial_values_b(
      count, std::vector<unsigned char>(size_in_bytes, 4));
  std::vector<void*> swap_ptrs_a(count);
  std::vector<void*> swap_ptrs_b(count);
  std::vector<LinearAllocGuard<unsigned char>> allocations;

  EnablePeerAccess({{device_for_a, device_for_b}, {device_for_b, device_for_a}});

  HIP_CHECK(hipSetDevice(device_for_a));
  StreamGuard stream_guard(Streams::created);

  for (size_t i = 0; i < count; ++i) {
    HIP_CHECK(hipSetDevice(device_for_b));
    LinearAllocGuard<unsigned char> alloc_b(LinearAllocs::hipMalloc, size_in_bytes);
    swap_ptrs_b[i] = alloc_b.ptr();
    allocations.push_back(std::move(alloc_b));
    fillBuffer(swap_ptrs_b[i], initial_values_b[i], LinearAllocs::hipMalloc);

    HIP_CHECK(hipSetDevice(device_for_a));
    LinearAllocGuard<unsigned char> alloc_a(LinearAllocs::hipMalloc, size_in_bytes);
    swap_ptrs_a[i] = alloc_a.ptr();
    allocations.push_back(std::move(alloc_a));
    fillBuffer(swap_ptrs_a[i], initial_values_a[i], LinearAllocs::hipMalloc);
  }

  HIP_CHECK(hipSetDevice(device_for_a));
  std::vector<size_t> sizes(count, size_in_bytes);
  hipMemcpyAttributes attr{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagExtOpSwap};
  size_t attrs_idxs[1] = {0};
  size_t fail_index = 0;

  HIP_CHECK_ERROR(hipMemcpyBatchAsync(swap_ptrs_a.data(), swap_ptrs_b.data(), sizes.data(), count,
                                      &attr, attrs_idxs, 1, &fail_index, stream_guard.stream()),
                  expectedError);
  if (expectedError == hipSuccess) {
    HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

    HIP_CHECK(hipSetDevice(device_for_a));
    for (size_t i = 0; i < count; ++i) {
      requireBufferEquals(swap_ptrs_a[i], initial_values_b[i], LinearAllocs::hipMalloc);
    }
    HIP_CHECK(hipSetDevice(device_for_b));
    for (size_t i = 0; i < count; ++i) {
      requireBufferEquals(swap_ptrs_b[i], initial_values_a[i], LinearAllocs::hipMalloc);
    }
  }

  DisablePeerAccess({{device_for_a, device_for_b}, {device_for_b, device_for_a}});
}

// Cross-GPU batched multicast: all entries share one source allocation on `device_for_src` and
// each uses a distinct destination on `device_for_dst`. Peer access is enabled from
// `device_for_dst` to `device_for_src`; the stream is created on `device_for_dst`.
static void RunMulticastCopyP2pTest(size_t count, size_t size_in_bytes, int device_for_src,
                                    int device_for_dst) {
  std::vector<unsigned char> initial_values(size_in_bytes, 10);
  std::vector<void*> src_ptrs(count);
  std::vector<void*> dst_ptrs(count);
  std::vector<LinearAllocGuard<unsigned char>> allocations;

  EnablePeerAccess({{device_for_dst, device_for_src}});

  HIP_CHECK(hipSetDevice(device_for_src));
  LinearAllocGuard<unsigned char> src_alloc(LinearAllocs::hipMalloc, size_in_bytes);
  void* src_mem = src_alloc.ptr();
  fillBuffer(src_mem, initial_values, LinearAllocs::hipMalloc);
  allocations.push_back(std::move(src_alloc));

  HIP_CHECK(hipSetDevice(device_for_dst));
  StreamGuard stream_guard(Streams::created);

  for (size_t i = 0; i < count; ++i) {
    src_ptrs[i] = src_mem;
    LinearAllocGuard<unsigned char> dst_alloc(LinearAllocs::hipMalloc, size_in_bytes);
    dst_ptrs[i] = dst_alloc.ptr();
    allocations.push_back(std::move(dst_alloc));
  }

  std::vector<size_t> sizes(count, size_in_bytes);
  hipMemcpyAttributes attr{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagDefault};
  size_t attrs_idxs[1] = {0};
  size_t fail_index = 0;
  HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), count, &attr,
                                attrs_idxs, 1, &fail_index, stream_guard.stream()));
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  for (size_t i = 0; i < count; ++i) {
    requireBufferEquals(dst_ptrs[i], initial_values, LinearAllocs::hipMalloc);
  }

  DisablePeerAccess({{device_for_dst, device_for_src}});
}

/**
 * Cross-GPU batched multicast: one shared source on the peer GPU, multiple destinations on the
 * local GPU.
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_P2P_Multicast) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST("Skipping because fewer than 2 devices are available");
  }

  int can_access_peer = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 1, 0));

  if (!can_access_peer) {
    HIP_SKIP_TEST("Skipping because device 1 cannot access peer memory on device 0");
  }

  const size_t count = GENERATE(2, 3, 8);
  const size_t size_in_bytes = GENERATE(as<size_t>{}, 1, 63, 4096);

  RunMulticastCopyP2pTest(count, size_in_bytes, 0, 1);
}

/**
 * Cross-GPU batched multicast with large per-copy size.
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_P2P_Multicast_Large) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST("Skipping because fewer than 2 devices are available");
  }

  int can_access_peer = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 1, 0));

  if (!can_access_peer) {
    HIP_SKIP_TEST("Skipping because device 1 cannot access peer memory on device 0");
  }

  const size_t count = GENERATE(2, 3, 8);
  const size_t size_in_bytes = 1024 * 1024;

  RunMulticastCopyP2pTest(count, size_in_bytes, 0, 1);
}

static void RunIndirectCopyP2pTest(unsigned int flags, size_t count, size_t size_in_bytes,
                                   int device_for_src, int device_for_dst) {
  const bool is_indirect_src = flags & hipMemcpyFlagExtOpIndirectSrc;
  const bool is_indirect_dst = flags & hipMemcpyFlagExtOpIndirectDst;
  const hipError_t expected_error =
      getIndirectExpectedReturn(LinearAllocs::hipMalloc, LinearAllocs::hipMalloc, device_for_src,
                                device_for_dst, device_for_dst);

  EnablePeerAccess({{device_for_dst, device_for_src}});

  IndirectCopyBuffers buffers =
      makeIndirectCopyBuffers(count, size_in_bytes, LinearAllocs::hipMalloc,
                              LinearAllocs::hipMalloc, device_for_src, device_for_dst);

  for (size_t i = 0; i < count; ++i) {
    if (is_indirect_src) {
      HIP_CHECK(hipSetDevice(device_for_src));
      buffers.batch_src_ptrs[i] =
          addPointerSlot(buffers.slots, buffers.src_ptrs[i], LinearAllocs::hipMalloc);
    }

    if (is_indirect_dst) {
      HIP_CHECK(hipSetDevice(device_for_dst));
      buffers.batch_dst_ptrs[i] =
          addPointerSlot(buffers.slots, buffers.dst_ptrs[i], LinearAllocs::hipMalloc);
    }
  }

  HIP_CHECK(hipSetDevice(device_for_dst));
  StreamGuard stream_guard(Streams::created);
  std::vector<size_t> sizes(count, size_in_bytes);
  hipMemcpyAttributes attr{hipMemcpySrcAccessOrderStream, {}, {}, flags};
  size_t attrs_idxs[1] = {0};

  HIP_CHECK_ERROR(hipMemcpyBatchAsync(buffers.batch_dst_ptrs.data(), buffers.batch_src_ptrs.data(),
                                      sizes.data(), count, &attr, attrs_idxs, 1, nullptr,
                                      stream_guard.stream()),
                  expected_error);

  if (expected_error == hipSuccess) {
    HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
    for (size_t i = 0; i < count; ++i) {
      requireBufferEquals(buffers.dst_ptrs[i], buffers.initial_values[i], LinearAllocs::hipMalloc);
    }
  }

  DisablePeerAccess({{device_for_dst, device_for_src}});
}

/**
 * Cross-GPU batched copy that reaches its buffers through pointer slots, on the peer GPU for an
 * indirect source and on the local GPU for an indirect destination.
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_P2P_IndirectCopy) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  int can_access_peer = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 1, 0));

  if (!can_access_peer) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
  }

  const unsigned int flags =
      GENERATE(as<unsigned int>{}, hipMemcpyFlagExtOpIndirectSrc, hipMemcpyFlagExtOpIndirectDst,
               hipMemcpyFlagExtOpIndirectSrc | hipMemcpyFlagExtOpIndirectDst);
  const size_t count = GENERATE(1, 3, 8);
  const size_t size_in_bytes = GENERATE(as<size_t>{}, 1, 63, 4096);
  CAPTURE(flags, count, size_in_bytes);

  RunIndirectCopyP2pTest(flags, count, size_in_bytes, 0, 1);
}

// Devices, starting from device 0, that can access each other's memory in both directions.
static std::vector<int> GetMutuallyPeeredDevices() {
  std::vector<int> devices{0};
  for (int candidate = 1; candidate < HipTest::getDeviceCount(); ++candidate) {
    bool peered = true;
    for (const int device : devices) {
      int can_access_to = 0;
      int can_access_from = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can_access_to, device, candidate));
      HIP_CHECK(hipDeviceCanAccessPeer(&can_access_from, candidate, device));
      peered = peered && can_access_to && can_access_from;
    }
    if (peered) {
      devices.push_back(candidate);
    }
  }
  return devices;
}

// Allgather-style batches issued back to back on one stream with no synchronization in between,
// over the first `max_devices` mutually peered GPUs: every GPU's buffer is copied to two
// destinations on every other GPU, so each source feeds several destinations and one batch becomes
// several SDMA operations that all wait on the stream's prior work. With `with_intra_device_copy`
// each batch also carries an intra-device copy, which runs as a shader copy waiting on the same
// prior work. The sources alternate between two patterns from batch to batch, so the final
// contents show that the last batch landed after all the others. The data moved grows with the
// square of the number of GPUs.
static void RunAllgatherBackToBack(size_t max_devices, bool with_intra_device_copy) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  std::vector<int> devices = GetMutuallyPeeredDevices();
  if (devices.size() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
  }
  if (devices.size() > max_devices) {
    devices.resize(max_devices);
  }
  CAPTURE(devices.size(), with_intra_device_copy);

  constexpr size_t kNumPatterns = 2;
  constexpr size_t kDestinationsPerPeer = 2;
  const int stream_device = devices.front();
  const size_t num_devices = devices.size();
  const auto pattern = [](size_t device_idx, size_t pattern_idx) {
    return std::vector<unsigned char>(
        kBackToBackCopySize,
        static_cast<unsigned char>(1 + device_idx * kNumPatterns + pattern_idx));
  };

  std::vector<std::pair<int, int>> peer_pairs;
  for (const int device_a : devices) {
    for (const int device_b : devices) {
      if (device_a != device_b) {
        peer_pairs.emplace_back(device_a, device_b);
      }
    }
  }
  EnablePeerAccess(peer_pairs);

  std::vector<LinearAllocGuard<unsigned char>> allocations;
  // srcs[device_idx][pattern_idx]
  std::vector<std::array<void*, kNumPatterns>> srcs(num_devices);
  for (size_t device_idx = 0; device_idx < num_devices; ++device_idx) {
    HIP_CHECK(hipSetDevice(devices[device_idx]));
    for (size_t pattern_idx = 0; pattern_idx < kNumPatterns; ++pattern_idx) {
      srcs[device_idx][pattern_idx] =
          addBuffer(allocations, pattern(device_idx, pattern_idx), LinearAllocs::hipMalloc);
    }
  }

  // The destinations of one batch, each with the device it lives on and the source device it
  // receives from, and the sources of each batch variant in the same order
  struct Destination {
    void* ptr;
    int device;
    size_t src_device_idx;
  };
  std::vector<Destination> dsts;
  std::array<std::vector<void*>, kNumPatterns> batch_srcs;
  const std::vector<unsigned char> zeros(kBackToBackCopySize, 0);
  for (size_t src_idx = 0; src_idx < num_devices; ++src_idx) {
    for (size_t dst_idx = 0; dst_idx < num_devices; ++dst_idx) {
      if (src_idx == dst_idx) {
        continue;
      }
      HIP_CHECK(hipSetDevice(devices[dst_idx]));
      for (size_t i = 0; i < kDestinationsPerPeer; ++i) {
        dsts.push_back(
            {addBuffer(allocations, zeros, LinearAllocs::hipMalloc), devices[dst_idx], src_idx});
      }
    }
  }
  if (with_intra_device_copy) {
    HIP_CHECK(hipSetDevice(stream_device));
    dsts.push_back({addBuffer(allocations, zeros, LinearAllocs::hipMalloc), stream_device, 0});
  }

  std::vector<void*> dst_ptrs;
  for (const Destination& dst : dsts) {
    dst_ptrs.push_back(dst.ptr);
    for (size_t pattern_idx = 0; pattern_idx < kNumPatterns; ++pattern_idx) {
      batch_srcs[pattern_idx].push_back(srcs[dst.src_device_idx][pattern_idx]);
    }
  }

  HIP_CHECK(hipSetDevice(stream_device));
  StreamGuard stream_guard(Streams::created);
  std::vector<size_t> sizes(dst_ptrs.size(), kBackToBackCopySize);
  hipMemcpyAttributes attr{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagDefault};
  size_t attrs_idxs[1] = {0};
  HIP_CHECK(hipDeviceSynchronize());

  for (size_t iteration = 0; iteration < kBackToBackIterations; ++iteration) {
    std::vector<void*>& iteration_srcs = batch_srcs[iteration % kNumPatterns];
    HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), iteration_srcs.data(), sizes.data(),
                                  sizes.size(), &attr, attrs_idxs, 1, nullptr,
                                  stream_guard.stream()));
  }
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  const size_t last_pattern_idx = (kBackToBackIterations - 1) % kNumPatterns;
  for (size_t i = 0; i < dsts.size(); ++i) {
    INFO("Destination " << i << " on device " << dsts[i].device);
    HIP_CHECK(hipSetDevice(dsts[i].device));
    requireBufferEquals(dsts[i].ptr, pattern(dsts[i].src_device_idx, last_pattern_idx),
                        LinearAllocs::hipMalloc);
  }

  DisablePeerAccess(peer_pairs);
  HIP_CHECK(hipSetDevice(stream_device));
}

/**
 * Allgather-style batches issued back to back on one stream between two peered GPUs, optionally
 * with an intra-device shader copy in each batch.
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_P2P_Allgather_BackToBack) {
  RunAllgatherBackToBack(2, GENERATE(false, true));
}

/**
 * Allgather-style batches issued back to back on one stream across every mutually peered GPU,
 * optionally with an intra-device shader copy in each batch.
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_P2P_Allgather_BackToBack_AllDevices) {
  RunAllgatherBackToBack(std::numeric_limits<size_t>::max(), GENERATE(false, true));
}
#endif
