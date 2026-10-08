/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Cross-GPU message-passing tests for the DDA IPC barrier fences in
// algorithms/dda/ipc/ipc_gpu_barrier.h. Eight GPUs in one process, peer access over XGMI.
//
//   MailboxPull_*  sender stores a payload in its own buffer, setFlagWithMemFence; receiver
//                  waitFlagWithMemFence then loads the payload over XGMI. One lane per block.
//   MailboxPush_*  same, but the sender stores the payload into the receiver's buffer.
//   FlatSync_*     the DDA flat IPC shape: write own slice, syncOnSameBlockIdx<true, true>,
//                  read all peer slices, syncOnSameBlockIdx<true, false>.
//
// What each arm detects on gfx942 (OCI, 8x MI300X):
//   *_Uncached  uses the production scratch allocation. Detects a relaxed waitFlagWithMemFence
//               (stale vL1 hits without buffer_inv sc0 sc1). Cannot detect a relaxed
//               setFlagWithMemFence: uncached stores leave nothing in L2 for buffer_wbl2 to flush.
//   *_Coarse    cached payload. Detects both halves of the release/acquire pair.

#include <hip/hip_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"

#include "../common/ProcessIsolatedTestRunner.hpp"
#include "algorithms/dda/ipc/ipc_gpu_barrier.h"

namespace RcclUnitTesting {

namespace {

using dda::common::DeviceMailbox;
using dda::common::IpcGpuBarrier;

constexpr int kRanks = 8;
constexpr int kBlocks = 64;
constexpr int kMpWords = 16;

// DeviceMailbox and IpcGpuBarrier have no host constructor usable without a bootstrap, so the
// test builds them bitwise from these mirrors; the static_asserts fail the build if the layout drifts.
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

// Block b of rank r sends to rank r+1 and receives from rank r-1 (mod 8). An ack mailbox stops the
// sender from overwriting the payload before the receiver has read it.
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

__global__ void flatSync(IpcGpuBarrier bar, uint32_t* const* bufs, int self, int iters, int slice,
                         unsigned long long* bad) {
  const int b = blockIdx.x;
  unsigned long long mismatches = 0;
  for (int it = 0; it < iters; ++it) {
    uint32_t* mine = bufs[self] + static_cast<size_t>(b) * slice;
    for (int i = threadIdx.x; i < slice; i += blockDim.x) {
      mine[i] = payloadValue(self, it, b, i);
    }
    bar.syncOnSameBlockIdx<true, true>();
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
    // Production ends the kernel here; the boundary orders the next write after the flag waits.
    __syncthreads();
  }
  if (mismatches != 0) {
    atomicAdd(bad, mismatches);
  }
}

enum class Alloc { kUncached, kCoarse };
enum class Shape { kPull, kPush, kFlat };

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

void* allocZeroed(size_t bytes, Alloc alloc) {
  void* p = nullptr;
  if (alloc == Alloc::kUncached) {
    EXPECT_EQ(hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached), hipSuccess);
  } else {
    EXPECT_EQ(hipMalloc(&p, bytes), hipSuccess);
  }
  EXPECT_EQ(hipMemset(p, 0, bytes), hipSuccess);
  return p;
}

void runOrderingTest(Shape shape, Alloc alloc, int iters, int slice) {
  for (int i = 0; i < kRanks; ++i) {
    ASSERT_EQ(hipSetDevice(i), hipSuccess);
    for (int j = 0; j < kRanks; ++j) {
      if (i != j) {
        hipError_t e = hipDeviceEnablePeerAccess(j, 0);
        ASSERT_TRUE(e == hipSuccess || e == hipErrorPeerAccessAlreadyEnabled) << hipGetErrorString(e);
      }
    }
  }
  const int words = shape == Shape::kFlat ? slice : kMpWords;
  const size_t payloadBytes = static_cast<size_t>(kBlocks) * words * sizeof(uint32_t);
  const size_t flagBytes = static_cast<size_t>(kRanks) * kBlocks * sizeof(uint32_t);

  std::vector<void*> allocations;
  std::vector<uint32_t*> bufs(kRanks);
  std::vector<DeviceMailbox> dataMb(kRanks);
  std::vector<DeviceMailbox> ackMb(kRanks);
  for (int r = 0; r < kRanks; ++r) {
    ASSERT_EQ(hipSetDevice(r), hipSuccess);
    bufs[r] = static_cast<uint32_t*>(allocZeroed(payloadBytes, alloc));
    // Mailboxes always use the production allocation (DeviceBuffer: uncached).
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
    ASSERT_EQ(hipSetDevice(r), hipSuccess);
    ASSERT_EQ(hipMalloc(&dBufs[r], kRanks * sizeof(uint32_t*)), hipSuccess);
    ASSERT_EQ(hipMemcpy(dBufs[r], bufs.data(), kRanks * sizeof(uint32_t*), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMalloc(&dData[r], kRanks * sizeof(DeviceMailbox)), hipSuccess);
    ASSERT_EQ(hipMemcpy(dData[r], dataMb.data(), kRanks * sizeof(DeviceMailbox), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMalloc(&dAck[r], kRanks * sizeof(DeviceMailbox)), hipSuccess);
    ASSERT_EQ(hipMemcpy(dAck[r], ackMb.data(), kRanks * sizeof(DeviceMailbox), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMalloc(&dBad[r], sizeof(unsigned long long)), hipSuccess);
    ASSERT_EQ(hipMemset(dBad[r], 0, sizeof(unsigned long long)), hipSuccess);
    ASSERT_EQ(hipStreamCreateWithFlags(&streams[r], hipStreamNonBlocking), hipSuccess);
  }

  // All eight kernels spin on each other, so they are launched before any is synchronized.
  for (int r = 0; r < kRanks; ++r) {
    ASSERT_EQ(hipSetDevice(r), hipSuccess);
    if (shape == Shape::kFlat) {
      hipLaunchKernelGGL(flatSync, dim3(kBlocks), dim3(512), 0, streams[r], makeBarrier(r, dataMb), dBufs[r], r, iters,
                         slice, dBad[r]);
    } else if (shape == Shape::kPull) {
      hipLaunchKernelGGL(mailboxMessagePassing<false>, dim3(kBlocks), dim3(64), 0, streams[r], dData[r], dAck[r],
                         dBufs[r], r, iters, dBad[r]);
    } else {
      hipLaunchKernelGGL(mailboxMessagePassing<true>, dim3(kBlocks), dim3(64), 0, streams[r], dData[r], dAck[r],
                         dBufs[r], r, iters, dBad[r]);
    }
    ASSERT_EQ(hipGetLastError(), hipSuccess);
  }

  unsigned long long totalBad = 0;
  for (int r = 0; r < kRanks; ++r) {
    ASSERT_EQ(hipSetDevice(r), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(streams[r]), hipSuccess);
    unsigned long long bad = 0;
    ASSERT_EQ(hipMemcpy(&bad, dBad[r], sizeof(bad), hipMemcpyDeviceToHost), hipSuccess);
    totalBad += bad;
  }
  EXPECT_EQ(totalBad, 0ull) << "receivers observed payload words older than the flag they acquired";

  for (int r = 0; r < kRanks; ++r) {
    ASSERT_EQ(hipSetDevice(r), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(streams[r]), hipSuccess);
    EXPECT_EQ(hipFree(dBufs[r]), hipSuccess);
    EXPECT_EQ(hipFree(dData[r]), hipSuccess);
    EXPECT_EQ(hipFree(dAck[r]), hipSuccess);
    EXPECT_EQ(hipFree(dBad[r]), hipSuccess);
  }
  for (void* p : allocations) {
    EXPECT_EQ(hipFree(p), hipSuccess);
  }
}

ProcessIsolatedTestRunner::TestConfig makeConfig(const char* name, Shape shape, Alloc alloc, int iters,
                                                 int slice = 0) {
  return ProcessIsolatedTestRunner::TestConfig(name, [=]() { runOrderingTest(shape, alloc, iters, slice); })
    .withTimeout(std::chrono::seconds(300))
    .withNumGpus(kRanks);
}

constexpr int kMpIters = 100000;
constexpr int kFlatIters = 20000;
constexpr int kFlatSlice = 256;

} // namespace

TEST(DdaIpcGpuBarrierOrdering, MailboxPull_Uncached) {
  if (!haveEightPeerGpus()) {
    GTEST_SKIP() << "needs 8 GPUs with full peer access";
  }
  RUN_ISOLATED_TESTS(makeConfig("DdaIpcGpuBarrierOrdering.MailboxPull_Uncached", Shape::kPull, Alloc::kUncached, kMpIters));
}

TEST(DdaIpcGpuBarrierOrdering, MailboxPull_Coarse) {
  if (!haveEightPeerGpus()) {
    GTEST_SKIP() << "needs 8 GPUs with full peer access";
  }
  RUN_ISOLATED_TESTS(makeConfig("DdaIpcGpuBarrierOrdering.MailboxPull_Coarse", Shape::kPull, Alloc::kCoarse, kMpIters));
}

TEST(DdaIpcGpuBarrierOrdering, MailboxPush_Coarse) {
  if (!haveEightPeerGpus()) {
    GTEST_SKIP() << "needs 8 GPUs with full peer access";
  }
  RUN_ISOLATED_TESTS(makeConfig("DdaIpcGpuBarrierOrdering.MailboxPush_Coarse", Shape::kPush, Alloc::kCoarse, kMpIters));
}

TEST(DdaIpcGpuBarrierOrdering, FlatSync_Uncached) {
  if (!haveEightPeerGpus()) {
    GTEST_SKIP() << "needs 8 GPUs with full peer access";
  }
  RUN_ISOLATED_TESTS(makeConfig("DdaIpcGpuBarrierOrdering.FlatSync_Uncached", Shape::kFlat, Alloc::kUncached, kFlatIters, kFlatSlice));
}

TEST(DdaIpcGpuBarrierOrdering, FlatSync_Coarse) {
  if (!haveEightPeerGpus()) {
    GTEST_SKIP() << "needs 8 GPUs with full peer access";
  }
  RUN_ISOLATED_TESTS(makeConfig("DdaIpcGpuBarrierOrdering.FlatSync_Coarse", Shape::kFlat, Alloc::kCoarse, kFlatIters, kFlatSlice));
}

} // namespace RcclUnitTesting
