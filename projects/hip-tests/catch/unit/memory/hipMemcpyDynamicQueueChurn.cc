/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

namespace {

struct Transfer {
  int src_gpu;
  int dma_gpu;
  bool dst_is_gpu;
  int dst_index;
};

constexpr Transfer kTransfers[] = {
    {0, 0, false, 0}, {1, 1, false, 0}, {2, 2, false, 0}, {3, 3, false, 0},
    {4, 4, false, 1}, {5, 5, false, 1}, {6, 6, false, 1}, {7, 7, false, 1},
    {0, 0, false, 1}, {1, 1, false, 1}, {2, 2, false, 1}, {3, 3, false, 1},
    {4, 4, false, 0}, {5, 5, false, 0}, {6, 6, false, 0}, {7, 7, false, 0},
    {0, 0, true, 0}, {0, 0, true, 1}, {0, 0, true, 2}, {0, 0, true, 3},
    {0, 0, true, 4}, {0, 0, true, 5}, {0, 0, true, 6}, {0, 0, true, 7},
    {1, 1, true, 1}, {1, 1, true, 2}, {1, 1, true, 3}, {1, 1, true, 4},
    {1, 1, true, 5}, {1, 1, true, 6}, {1, 1, true, 7},
    {2, 2, true, 3}, {2, 2, true, 4}, {2, 2, true, 5}, {2, 2, true, 6},
    {2, 2, true, 7},
    {3, 3, true, 3}, {3, 3, true, 4}, {3, 3, true, 5}, {3, 3, true, 6},
    {3, 3, true, 7},
    {4, 4, true, 4}, {4, 4, true, 5}, {4, 4, true, 6}, {4, 4, true, 7},
    {5, 5, true, 5}, {5, 5, true, 6}, {5, 5, true, 7},
    {6, 6, true, 6}, {6, 6, true, 7},
    {7, 7, true, 7},
};

}  // namespace

HIP_TEST_CASE(Unit_hipMemcpy_DynamicQueueChurn_StagedCopies) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < 8) {
    HIP_SKIP_TEST("Requires at least 8 GPUs");
  }

  const size_t kSize = isQuickLevel() ? 128ULL * 1024ULL * 1024ULL
                                      : 1ULL * 1024ULL * 1024ULL * 1024ULL;
  void* device_buffers[8] = {};
  void* host_buffers[2] = {};

  for (int dev = 0; dev < 8; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    for (int peer = 0; peer < 8; ++peer) {
      if (peer != dev) {
        int can_access_peer = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, dev, peer));
        if (can_access_peer != 0) {
          hipError_t peer_access = hipDeviceEnablePeerAccess(peer, 0);
          if (peer_access != hipSuccess && peer_access != hipErrorPeerAccessAlreadyEnabled) {
            HIP_CHECK(peer_access);
          }
        }
      }
    }
    HIP_CHECK(hipMalloc(&device_buffers[dev], kSize));
    HIP_CHECK(hipMemset(device_buffers[dev], 0, kSize));
    HIP_CHECK(hipDeviceSynchronize());
  }

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipHostMalloc(&host_buffers[0], kSize));
  HIP_CHECK(hipHostMalloc(&host_buffers[1], kSize));
  std::memset(host_buffers[0], 0, kSize);
  std::memset(host_buffers[1], 0, kSize);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<hipStream_t> streams(std::size(kTransfers));
  for (size_t i = 0; i < std::size(kTransfers); ++i) {
    HIP_CHECK(hipSetDevice(kTransfers[i].dma_gpu));
    HIP_CHECK(hipStreamCreate(&streams[i]));
  }

  std::atomic<bool> failed{false};
  std::vector<std::thread> threads;
  threads.reserve(std::size(kTransfers));

  for (size_t i = 0; i < std::size(kTransfers); ++i) {
    threads.emplace_back([&, i]() {
      const Transfer& transfer = kTransfers[i];
      if (hipSetDevice(transfer.dma_gpu) != hipSuccess) {
        failed.store(true, std::memory_order_relaxed);
        return;
      }
      void* dst = transfer.dst_is_gpu ? device_buffers[transfer.dst_index]
                                      : host_buffers[transfer.dst_index];
      void* src = device_buffers[transfer.src_gpu];
      if (hipMemcpyAsync(dst, src, kSize, hipMemcpyDefault, streams[i]) != hipSuccess) {
        failed.store(true, std::memory_order_relaxed);
        return;
      }
      if (hipStreamSynchronize(streams[i]) != hipSuccess) {
        failed.store(true, std::memory_order_relaxed);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
  const bool transfer_failed = failed.load(std::memory_order_relaxed);
  HIP_CHECK(hipDeviceSynchronize());

  for (size_t i = 0; i < std::size(kTransfers); ++i) {
    HIP_CHECK(hipSetDevice(kTransfers[i].dma_gpu));
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }

  HIP_CHECK(hipHostFree(host_buffers[0]));
  HIP_CHECK(hipHostFree(host_buffers[1]));
  for (int dev = 0; dev < 8; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipFree(device_buffers[dev]));
  }
  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE_FALSE(transfer_failed);
}
