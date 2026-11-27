/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
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

// How many batches RunAllgatherBackToBack submits on `num_devices` GPUs. Only the number of
// batches shrinks on large systems; every batch still copies between every pair of GPUs.
//
// Up to 8 GPUs this is kBackToBackIterations (400), which already runs in a few seconds. Above 8
// GPUs, the 8-GPU run is the budget: 400 batches * 112 peer copies per batch (8 * 7 * 2) = 44,800
// peer copies. Divide the budget by the peer copies per batch on this system, then round down to
// whole A/B rounds (a multiple of `num_patterns`), with at least one round. An odd count would end
// on A, the same bytes the first batch wrote, and a run where only the first batch landed would
// still pass the final check.
//
//   GPUs   peer copies per batch   batches
//   <= 8   <= 112                  400
//   16     16 * 15 * 2 = 480       44,800 / 480 = 93 -> 92
//   64     64 * 63 * 2 = 8,064     44,800 / 8,064 = 5 -> 4 (A, B, A, B)
//
// Without this, 400 batches on 64 GPUs took about 277 s per child process, past its deadline.
static size_t AllgatherIterations(size_t num_devices, size_t destinations_per_peer,
                                  size_t num_patterns) {
  constexpr size_t kReferenceDevices = 8;
  const auto peer_copies = [destinations_per_peer](size_t devices) {
    return devices * (devices - 1) * destinations_per_peer;
  };
  const size_t iterations = std::clamp<size_t>(
      kBackToBackIterations * peer_copies(kReferenceDevices) / peer_copies(num_devices),
      num_patterns, kBackToBackIterations);
  return iterations - iterations % num_patterns;
}

// Checks that hipMemcpyBatchAsync batches queued back to back on one stream, with no
// synchronization in between, run in stream order. Uses the first `max_devices` mutually peered
// GPUs. The example below uses two GPUs.
//
// 1. Buffers (set up once). Each GPU has two 2 MiB source buffers, A and B, each filled with one
//    byte value unique to that GPU and pattern (1 + gpu * 2 + pattern):
//
//            A    B
//      GPU0  1    2
//      GPU1  3    4
//
//    For every ordered pair (src GPU, dst GPU), two zeroed destination buffers live on the dst GPU:
//      d0, d1 on GPU1 receive from GPU0;  d2, d3 on GPU0 receive from GPU1.
//    With `with_intra_device_copy`, d4 on GPU0 also receives from GPU0 itself.
//
// 2. One batch = one allgather: one hipMemcpyBatchAsync that writes every destination once.
//      Batch A: GPU0.A -> d0, d1;  GPU1.A -> d2, d3;  (GPU0.A -> d4)
//      Batch B: GPU0.B -> d0, d1;  GPU1.B -> d2, d3;  (GPU0.B -> d4)
//    A and B write the same destinations, so each batch overwrites the one before it. The 2 MiB
//    peer copies run on SDMA, split into several engine operations per batch; the d4 copy runs as
//    a shader (blit kernel). All of them must wait for the previous batch to finish.
//
// 3. Submit A, B, A, B, ... on one stream without waiting, then synchronize once. The batch count
//    comes from AllgatherIterations and is always even, so the last batch is B.
//
// 4. Read every destination back and check every byte against the last batch, B:
//      d0, d1 == 2 (GPU0.B);  d2, d3 == 4 (GPU1.B);  d4 == 2 (GPU0.B)
//    A destination holding A's value (1 or 3) means an earlier A batch finished after the last B
//    batch: stream order was broken. A broken dependency that never resolves shows up as a hang,
//    which the signal-tracker child deadline catches.
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
  constexpr size_t kNumPatterns = 2;
  constexpr size_t kDestinationsPerPeer = 2;
  const int stream_device = devices.front();
  const size_t num_devices = devices.size();
  const size_t iterations = AllgatherIterations(num_devices, kDestinationsPerPeer, kNumPatterns);
  CAPTURE(num_devices, with_intra_device_copy, iterations);
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

  for (size_t iteration = 0; iteration < iterations; ++iteration) {
    std::vector<void*>& iteration_srcs = batch_srcs[iteration % kNumPatterns];
    HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), iteration_srcs.data(), sizes.data(),
                                  sizes.size(), &attr, attrs_idxs, 1, nullptr,
                                  stream_guard.stream()));
  }
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  const size_t last_pattern_idx = (iterations - 1) % kNumPatterns;
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
  if (RunInEachSignalTracker("Unit_hipMemcpyBatchAsync_P2P_Allgather_BackToBack")) {
    return;
  }
  RunAllgatherBackToBack(2, GENERATE(false, true));
}

/**
 * Allgather-style batches issued back to back on one stream across every mutually peered GPU,
 * optionally with an intra-device shader copy in each batch.
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_P2P_Allgather_BackToBack_AllDevices) {
  if (RunInEachSignalTracker("Unit_hipMemcpyBatchAsync_P2P_Allgather_BackToBack_AllDevices",
                             kSignalTrackerChildLongDeadlineSeconds)) {
    return;
  }
  RunAllgatherBackToBack(std::numeric_limits<size_t>::max(), GENERATE(false, true));
}
#endif
