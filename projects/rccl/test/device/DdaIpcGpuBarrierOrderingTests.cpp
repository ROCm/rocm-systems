/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// 8-GPU litmus for ipc_gpu_barrier.h fences; rounds repeat inside one kernel so kernel boundaries cannot hide them.

#include <hip/hip_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"

#include "../common/ProcessIsolatedTestRunner.hpp"
#include "../common/TestChecks.hpp"
#include "algorithms/dda/ipc/ipc_gpu_barrier.h"

namespace RcclUnitTesting {

namespace {

using dda::common::DeviceMailbox;
using dda::common::IpcGpuBarrier;

constexpr int kRanks = 8;
constexpr int kBlocks = 64;
constexpr int kMpWords = 16;
constexpr int kMpIters = 100000;
constexpr int kFlatIters = 20000;
constexpr int kFlatThreads = 512;  // the DDA kernels' __launch_bounds__
constexpr int kFlatSlice = kFlatThreads;  // one word per thread, so no lane idles
constexpr int kSkewSleeps = 4;  // ~16 us of s_sleep 127, well above an XGMI flag round trip

// No host constructor works without a bootstrap, so build the objects bitwise; the asserts catch size changes only.
struct MailboxImage {
  int nBlocks;
  uint32_t* flags;
};
struct BarrierImage {
  int nBlocks;
  int selfRank;
  std::array<DeviceMailbox, kRanks> mailboxes;
};
static_assert(sizeof(MailboxImage) == sizeof(DeviceMailbox), "DeviceMailbox layout changed");
static_assert(sizeof(BarrierImage) == sizeof(IpcGpuBarrier), "IpcGpuBarrier layout changed");
static_assert(std::is_trivially_copyable<DeviceMailbox>::value, "DeviceMailbox must be bit-copyable");
static_assert(std::is_trivially_copyable<IpcGpuBarrier>::value, "IpcGpuBarrier must be bit-copyable");

DeviceMailbox makeMailbox(uint32_t* flags) {
  MailboxImage img{kBlocks, flags};
  DeviceMailbox mb;
  std::memcpy(static_cast<void*>(&mb), &img, sizeof(mb));
  return mb;
}

IpcGpuBarrier makeBarrier(int selfRank, const std::vector<DeviceMailbox>& mailboxes) {
  BarrierImage img{kBlocks, selfRank, {}};
  for (int r = 0; r < kRanks; ++r) {
    img.mailboxes[r] = mailboxes[r];
  }
  IpcGpuBarrier bar;
  std::memcpy(static_cast<void*>(&bar), &img, sizeof(bar));
  return bar;
}

__device__ __forceinline__ uint32_t payloadValue(int rank, int iter, int block, int idx) {
  return static_cast<uint32_t>(iter + 1) * 0x9E3779B1u ^ (static_cast<uint32_t>(rank) << 24) ^
         (static_cast<uint32_t>(block) << 12) ^ static_cast<uint32_t>(idx);
}

// Block b sends to rank self+1 and receives from self-1; the ack mailbox keeps the sender from overwriting unread data.
template <bool kPush>
__global__ void mailboxMessagePassing(DeviceMailbox* dataMb, DeviceMailbox* ackMb, uint32_t* const* bufs, int self,
                                      int iters, unsigned long long* bad) {
  if (threadIdx.x != 0) {
    return;
  }
  const int b = blockIdx.x;
  const int dst = (self + 1) % kRanks;
  const int src = (self + kRanks - 1) % kRanks;
  uint32_t* outSlot = (kPush ? bufs[dst] : bufs[self]) + static_cast<size_t>(b) * kMpWords;
  const uint32_t* inSlot = (kPush ? bufs[self] : bufs[src]) + static_cast<size_t>(b) * kMpWords;
  unsigned long long mismatches = 0;
  for (int it = 0; it < iters; ++it) {
    if (it > 0) {
      ackMb[self].waitFlagWithMemFence(dst, b);
    }
    for (int i = 0; i < kMpWords; ++i) {
      outSlot[i] = payloadValue(self, it, b, i);
    }
    dataMb[dst].setFlagWithMemFence(self, b);
    dataMb[self].waitFlagWithMemFence(src, b);
    for (int i = 0; i < kMpWords; ++i) {
      if (inSlot[i] != payloadValue(src, it, b, i)) {
        mismatches++;
      }
    }
    if (it + 1 < iters) {
      ackMb[src].setFlagWithMemFence(self, b);
    }
  }
  if (mismatches != 0) {
    atomicAdd(bad, mismatches);
  }
}

// The DDA flat IPC shape: write own slice, barrier, read every peer slice, release-only barrier.
// kAcquireOnly opens with the <false, true> entry barrier that follows a prior kernel or memcpy in production.
template <bool kAcquireOnly>
__global__ void flatSync(IpcGpuBarrier bar, uint32_t* const* bufs, int self, int iters, int slice,
                         unsigned long long* bad) {
  const int b = blockIdx.x;
  unsigned long long mismatches = 0;
  for (int it = 0; it < iters; ++it) {
    uint32_t* mine = bufs[self] + static_cast<size_t>(b) * slice;
    // Half the ranks delay later waves so an early flag shows unwritten data; uniform test as s_sleep ignores EXEC.
    if ((self + it) % 2 == 1 && __builtin_amdgcn_readfirstlane(threadIdx.x) >= warpSize) {
      for (int k = 0; k < kSkewSleeps; ++k) {
        __builtin_amdgcn_s_sleep(127);
      }
    }
    for (int i = threadIdx.x; i < slice; i += blockDim.x) {
      mine[i] = payloadValue(self, it, b, i);
    }
    if constexpr (kAcquireOnly) {
      // Release-only stand-in for the prior kernel end; with no acquire here, only the barrier orders the reads.
      __syncthreads();
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
      bar.syncOnSameBlockIdx<false, true>();
    } else {
      bar.syncOnSameBlockIdx<true, true>();
    }
    for (int p = 0; p < kRanks; ++p) {
      if (p == self) {
        continue;
      }
      const uint32_t* peer = bufs[p] + static_cast<size_t>(b) * slice;
      for (int i = threadIdx.x; i < slice; i += blockDim.x) {
        if (peer[i] != payloadValue(p, it, b, i)) {
          mismatches++;
        }
      }
    }
    bar.syncOnSameBlockIdx<true, false>();
    // Stands in for the kernel end that follows <true, false> in production.
    __syncthreads();
  }
  if (mismatches != 0) {
    atomicAdd(bad, mismatches);
  }
}

// Uncached is the production scratch; Coarse (cached) is the only allocation where a relaxed release is observable.
enum class Alloc {
  kUncached,
  kCoarse
};
enum class Shape {
  kPull,
  kPush,
  kFlat,
  kFlatAcquireOnly
};

bool haveEightPeerGpus() {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess || n < kRanks) {
    return false;
  }
  for (int i = 0; i < kRanks; ++i) {
    for (int j = 0; j < kRanks; ++j) {
      int can = 0;
      if (i != j && (hipDeviceCanAccessPeer(&can, i, j) != hipSuccess || !can)) {
        return false;
      }
    }
  }
  return true;
}

// Gate in the parent only: a skipped isolated child is scored as a pass.
bool shouldSkip() {
  return std::getenv(ProcessIsolatedTestRunner::kReexecMarkerEnvVar) == nullptr && !haveEightPeerGpus();
}

void* allocZeroed(size_t bytes, Alloc alloc) {
  void* p = nullptr;
  if (alloc == Alloc::kUncached) {
    HIP_EXPECT(hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached));
  } else {
    HIP_EXPECT(hipMalloc(&p, bytes));
  }
  HIP_EXPECT(hipMemset(p, 0, bytes));
  return p;
}

void runOrderingTest(Shape shape, Alloc alloc, int iters, int slice) {
  for (int i = 0; i < kRanks; ++i) {
    HIP_CHECK(hipSetDevice(i));
    for (int j = 0; j < kRanks; ++j) {
      if (i != j) {
        hipError_t e = hipDeviceEnablePeerAccess(j, 0);
        ASSERT_TRUE(e == hipSuccess || e == hipErrorPeerAccessAlreadyEnabled) << hipGetErrorString(e);
      }
    }
  }
  const bool flat = shape == Shape::kFlat || shape == Shape::kFlatAcquireOnly;
  const int words = flat ? slice : kMpWords;
  const size_t payloadBytes = static_cast<size_t>(kBlocks) * words * sizeof(uint32_t);
  const size_t flagBytes = static_cast<size_t>(kRanks) * kBlocks * sizeof(uint32_t);

  std::vector<void*> allocations;
  std::vector<uint32_t*> bufs(kRanks);
  std::vector<DeviceMailbox> dataMb(kRanks);
  std::vector<DeviceMailbox> ackMb(kRanks);
  for (int r = 0; r < kRanks; ++r) {
    HIP_CHECK(hipSetDevice(r));
    bufs[r] = static_cast<uint32_t*>(allocZeroed(payloadBytes, alloc));
    // Mailbox flags always use the production DeviceBuffer allocation.
    void* d = allocZeroed(flagBytes, Alloc::kUncached);
    void* a = allocZeroed(flagBytes, Alloc::kUncached);
    allocations.insert(allocations.end(), {bufs[r], d, a});
    dataMb[r] = makeMailbox(static_cast<uint32_t*>(d));
    ackMb[r] = makeMailbox(static_cast<uint32_t*>(a));
  }

  std::vector<uint32_t**> dBufs(kRanks);
  std::vector<DeviceMailbox*> dData(kRanks);
  std::vector<DeviceMailbox*> dAck(kRanks);
  std::vector<unsigned long long*> dBad(kRanks);
  std::vector<hipStream_t> streams(kRanks);
  for (int r = 0; r < kRanks; ++r) {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipMalloc(&dBufs[r], kRanks * sizeof(uint32_t*)));
    HIP_CHECK(hipMemcpy(dBufs[r], bufs.data(), kRanks * sizeof(uint32_t*), hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&dData[r], kRanks * sizeof(DeviceMailbox)));
    HIP_CHECK(hipMemcpy(dData[r], dataMb.data(), kRanks * sizeof(DeviceMailbox), hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&dAck[r], kRanks * sizeof(DeviceMailbox)));
    HIP_CHECK(hipMemcpy(dAck[r], ackMb.data(), kRanks * sizeof(DeviceMailbox), hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&dBad[r], sizeof(unsigned long long)));
    HIP_CHECK(hipMemset(dBad[r], 0, sizeof(unsigned long long)));
    HIP_CHECK(hipStreamCreateWithFlags(&streams[r], hipStreamNonBlocking));
  }

  // All eight kernels spin on each other, so every launch precedes the first synchronize.
  for (int r = 0; r < kRanks; ++r) {
    HIP_CHECK(hipSetDevice(r));
    if (shape == Shape::kFlat) {
      hipLaunchKernelGGL(flatSync<false>, dim3(kBlocks), dim3(kFlatThreads), 0, streams[r], makeBarrier(r, dataMb),
                         dBufs[r], r, iters, slice, dBad[r]);
    } else if (shape == Shape::kFlatAcquireOnly) {
      hipLaunchKernelGGL(flatSync<true>, dim3(kBlocks), dim3(kFlatThreads), 0, streams[r], makeBarrier(r, dataMb),
                         dBufs[r], r, iters, slice, dBad[r]);
    } else if (shape == Shape::kPull) {
      hipLaunchKernelGGL(mailboxMessagePassing<false>, dim3(kBlocks), dim3(64), 0, streams[r], dData[r], dAck[r],
                         dBufs[r], r, iters, dBad[r]);
    } else {
      hipLaunchKernelGGL(mailboxMessagePassing<true>, dim3(kBlocks), dim3(64), 0, streams[r], dData[r], dAck[r],
                         dBufs[r], r, iters, dBad[r]);
    }
    HIP_CHECK(hipGetLastError());
  }

  unsigned long long totalBad = 0;
  for (int r = 0; r < kRanks; ++r) {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipStreamSynchronize(streams[r]));
    unsigned long long bad = 0;
    HIP_CHECK(hipMemcpy(&bad, dBad[r], sizeof(bad), hipMemcpyDeviceToHost));
    totalBad += bad;
  }
  EXPECT_EQ(totalBad, 0ull) << "receivers observed payload words older than the flag they acquired";

  for (int r = 0; r < kRanks; ++r) {
    HIP_CHECK(hipSetDevice(r));
    HIP_EXPECT(hipStreamDestroy(streams[r]));
    HIP_EXPECT(hipFree(dBufs[r]));
    HIP_EXPECT(hipFree(dData[r]));
    HIP_EXPECT(hipFree(dAck[r]));
    HIP_EXPECT(hipFree(dBad[r]));
  }
  for (void* p : allocations) {
    HIP_EXPECT(hipFree(p));
  }
}

ProcessIsolatedTestRunner::TestConfig makeConfig(const char* name, Shape shape, Alloc alloc, int iters, int slice = 0) {
  return ProcessIsolatedTestRunner::TestConfig(name, [=]() { runOrderingTest(shape, alloc, iters, slice); })
    .withTimeout(std::chrono::seconds(60))
    .withNumGpus(kRanks);
}

constexpr const char* kSkipReason = "needs 8 GPUs with full peer access";

} // namespace

TEST(DdaIpcGpuBarrierOrdering, MailboxPull_Uncached) {
  if (shouldSkip()) {
    GTEST_SKIP() << kSkipReason;
  }
  RUN_ISOLATED_TESTS(makeConfig("DdaIpcGpuBarrierOrdering.MailboxPull_Uncached", Shape::kPull, Alloc::kUncached,
                                kMpIters));
}

TEST(DdaIpcGpuBarrierOrdering, MailboxPull_Coarse) {
  if (shouldSkip()) {
    GTEST_SKIP() << kSkipReason;
  }
  RUN_ISOLATED_TESTS(makeConfig("DdaIpcGpuBarrierOrdering.MailboxPull_Coarse", Shape::kPull, Alloc::kCoarse, kMpIters));
}

TEST(DdaIpcGpuBarrierOrdering, MailboxPush_Coarse) {
  if (shouldSkip()) {
    GTEST_SKIP() << kSkipReason;
  }
  RUN_ISOLATED_TESTS(makeConfig("DdaIpcGpuBarrierOrdering.MailboxPush_Coarse", Shape::kPush, Alloc::kCoarse, kMpIters));
}

TEST(DdaIpcGpuBarrierOrdering, FlatSync_Uncached) {
  if (shouldSkip()) {
    GTEST_SKIP() << kSkipReason;
  }
  RUN_ISOLATED_TESTS(makeConfig("DdaIpcGpuBarrierOrdering.FlatSync_Uncached", Shape::kFlat, Alloc::kUncached,
                                kFlatIters, kFlatSlice));
}

TEST(DdaIpcGpuBarrierOrdering, FlatSync_Coarse) {
  if (shouldSkip()) {
    GTEST_SKIP() << kSkipReason;
  }
  RUN_ISOLATED_TESTS(makeConfig("DdaIpcGpuBarrierOrdering.FlatSync_Coarse", Shape::kFlat, Alloc::kCoarse, kFlatIters,
                                kFlatSlice));
}

TEST(DdaIpcGpuBarrierOrdering, FlatAcquireOnly_Uncached) {
  if (shouldSkip()) {
    GTEST_SKIP() << kSkipReason;
  }
  RUN_ISOLATED_TESTS(makeConfig("DdaIpcGpuBarrierOrdering.FlatAcquireOnly_Uncached", Shape::kFlatAcquireOnly,
                                Alloc::kUncached, kFlatIters, kFlatSlice));
}

TEST(DdaIpcGpuBarrierOrdering, FlatAcquireOnly_Coarse) {
  if (shouldSkip()) {
    GTEST_SKIP() << kSkipReason;
  }
  RUN_ISOLATED_TESTS(makeConfig("DdaIpcGpuBarrierOrdering.FlatAcquireOnly_Coarse", Shape::kFlatAcquireOnly,
                                Alloc::kCoarse, kFlatIters, kFlatSlice));
}

} // namespace RcclUnitTesting
