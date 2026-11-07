/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-played peers for single-GPU DDA LL kernel tests: coherent host scratch per rank behind a device peer table, a
// host-visible abort flag, and LL line writers that publish the payload before the flags.

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

#include <hip/hip_runtime.h>

#include "gtest/gtest.h"

namespace RcclUnitTesting {

class DdaLLHostPeers {
 public:
  // nRanks zeroed scratch buffers of scratchBytes each, the device table of their addresses, and a cleared abort flag.
  void Init(int nRanks, size_t scratchBytes) {
    host_.assign(nRanks, nullptr);
    std::vector<void*> dev(nRanks, nullptr);
    for (int r = 0; r < nRanks; ++r) {
      ASSERT_EQ(hipHostMalloc(&host_[r], scratchBytes, hipHostMallocMapped | hipHostMallocCoherent), hipSuccess);
      std::memset(host_[r], 0, scratchBytes);
      ASSERT_EQ(hipHostGetDevicePointer(&dev[r], host_[r], 0), hipSuccess);
    }
    ASSERT_EQ(hipHostMalloc(reinterpret_cast<void**>(&abortFlag_), sizeof(uint32_t),
                            hipHostMallocMapped | hipHostMallocCoherent),
              hipSuccess);
    *abortFlag_ = 0;
    ASSERT_EQ(hipMalloc(&devPeers_, nRanks * sizeof(void*)), hipSuccess);
    ASSERT_EQ(hipMemcpy(devPeers_, dev.data(), nRanks * sizeof(void*), hipMemcpyHostToDevice), hipSuccess);
  }

  void Release() {
    EXPECT_EQ(hipFree(devPeers_), hipSuccess);
    EXPECT_EQ(hipHostFree(abortFlag_), hipSuccess);
    for (void* p : host_) {
      EXPECT_EQ(hipHostFree(p), hipSuccess);
    }
    host_.clear();
  }

  void* Host(int rank) const { return host_[rank]; }

  template <typename T>
  T** DevPeers() const {
    return static_cast<T**>(devPeers_);
  }

  uint32_t* AbortFlag() const { return abortFlag_; }

  // Writes one LL line's payload words (0 and 2) before its flag words (1 and 3), so the poller never sees fresh flags
  // over stale payload.
  static void PublishLine(volatile uint32_t* w, uint32_t lo, uint32_t hi, uint32_t flag) {
    w[0] = lo;
    w[2] = hi;
    std::atomic_thread_fence(std::memory_order_release);
    w[1] = flag;
    w[3] = flag;
  }

  // Writes one 8B half of an LL line: half 0 is payload word 0 and flag word 1, half 1 is word 2 and flag 3.
  static void PublishHalf(volatile uint32_t* w, int half, uint32_t value, uint32_t flag) {
    w[2 * half] = value;
    std::atomic_thread_fence(std::memory_order_release);
    w[2 * half + 1] = flag;
  }

  // Waits for the stream; past the watchdog raises abort and runs publishAll, so a lost wake-up or an ignored abort
  // fails rather than hangs.
  bool Finishes(hipStream_t stream, const std::function<void()>& publishAll, std::chrono::seconds watchdog) {
    const auto deadline = std::chrono::steady_clock::now() + watchdog;
    while (hipStreamQuery(stream) == hipErrorNotReady) {
      if (std::chrono::steady_clock::now() > deadline) {
        *abortFlag_ = 1;
        publishAll();
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        return false;
      }
      std::this_thread::yield();
    }
    return hipStreamSynchronize(stream) == hipSuccess;
  }

 private:
  std::vector<void*> host_;
  void* devPeers_ = nullptr;
  uint32_t* abortFlag_ = nullptr;
};

}  // namespace RcclUnitTesting
