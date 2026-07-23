/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <resource_guards.hh>

#include "memcpyBatchAsync_common.hh"

#if HT_AMD

/**
 * Batched D2D copies from device 0 to device 1: each entry has its own src allocation on
 * device 0 and a distinct dst allocation on device 1. The stream and hipMemcpyBatchAsync run on
 * device 1 with peer access enabled from device 1 to device 0.
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_P2P_Basic) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST("Skipping because fewer than 2 devices are available");
  }

  int can_access_peer = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 1, 0));

  if (!can_access_peer) {
    HIP_SKIP_TEST("Skipping because device 1 cannot access peer memory on device 0");
  }

  const size_t count = GENERATE(1, 3, 8);
  const size_t size_in_bytes = GENERATE(as<size_t>{}, 1, 63, 4096);
  const int device_for_src = 0;
  const int device_for_dst = 1;
  std::vector<unsigned char> initial_values(size_in_bytes, 10);
  std::vector<void*> src_ptrs(count);
  std::vector<void*> dst_ptrs(count);
  std::vector<LinearAllocGuard<unsigned char>> allocations;

  EnablePeerAccess({{device_for_dst, device_for_src}});
  HIP_CHECK(hipSetDevice(device_for_dst));
  StreamGuard stream_guard(Streams::created);

  for (size_t i = 0; i < count; ++i) {
    HIP_CHECK(hipSetDevice(device_for_src));
    LinearAllocGuard<unsigned char> src_alloc(LinearAllocs::hipMalloc, size_in_bytes);
    src_ptrs[i] = src_alloc.ptr();
    fillBuffer(src_ptrs[i], initial_values, LinearAllocs::hipMalloc);
    allocations.push_back(std::move(src_alloc));

    HIP_CHECK(hipSetDevice(device_for_dst));
    LinearAllocGuard<unsigned char> dst_alloc(LinearAllocs::hipMalloc, size_in_bytes);
    dst_ptrs[i] = dst_alloc.ptr();
    HIP_CHECK(hipMemset(dst_ptrs[i], 0, size_in_bytes));
    allocations.push_back(std::move(dst_alloc));
  }

  HIP_CHECK(hipSetDevice(device_for_dst));
  std::vector<size_t> sizes(count, size_in_bytes);
  size_t attrs_idxs[1] = {0};
  size_t fail_index = 0;
  HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), count, nullptr,
                                attrs_idxs, 0, &fail_index, stream_guard.stream()));
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  for (size_t i = 0; i < count; ++i) {
    requireBufferEquals(dst_ptrs[i], initial_values, LinearAllocs::hipMalloc);
  }

  DisablePeerAccess({{device_for_dst, device_for_src}});
}

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
#endif
