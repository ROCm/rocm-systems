/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// 8-GPU litmus for ipc_gpu_barrier.h fences; rounds repeat inside one kernel so kernel boundaries cannot hide them.
// FlatSync_FourRanks reruns the flat shape on a 4-GPU clique whose barriers carry nRanks = 4.
// AbortedBarrierStaysSilent needs one GPU and checks that a barrier gives up for good once it sees the abort flag;
// its _FourRanks variant repeats that with nRanks = 4 and checks the four unused lanes are never signaled.

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

using dda::common::DdaKernelParams;
using dda::common::DeviceMailbox;
using dda::common::IpcGpuBarrier;

constexpr int kRanks = 8;
constexpr int kPartialRanks = 4;  // a relaxed clique below the eight mailboxes the barrier is strided for
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
  int nRanks;  // live peer count; the barrier signals only lanes below it
  std::array<DeviceMailbox, kRanks> mailboxes;
  DdaKernelParams params;
  bool aborted;
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

// mailboxes always holds kRanks entries; with nRanks < kRanks the caller decides what the unused tail points at.
IpcGpuBarrier makeBarrier(int selfRank, const std::vector<DeviceMailbox>& mailboxes, DdaKernelParams params = {},
                          int nRanks = kRanks) {
  BarrierImage img{kBlocks, selfRank, nRanks, {}, params, false};
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
      ackMb[self].waitFlagWithMemFence(dst, b, DdaKernelParams{});
    }
    for (int i = 0; i < kMpWords; ++i) {
      outSlot[i] = payloadValue(self, it, b, i);
    }
    dataMb[dst].setFlagWithMemFence(self, b, DdaKernelParams{});
    dataMb[self].waitFlagWithMemFence(src, b, DdaKernelParams{});
    for (int i = 0; i < kMpWords; ++i) {
      if (inSlot[i] != payloadValue(src, it, b, i)) {
        mismatches++;
      }
    }
    if (it + 1 < iters) {
      ackMb[src].setFlagWithMemFence(self, b, DdaKernelParams{});
    }
  }
  if (mismatches != 0) {
    atomicAdd(bad, mismatches);
  }
}

// The DDA flat IPC shape: write own slice, barrier, read every peer slice, release-only barrier.
// kAcquireOnly opens with the <false, true> entry barrier that follows a prior kernel or memcpy in production.
template <bool kAcquireOnly>
__global__ void flatSync(IpcGpuBarrier bar, uint32_t* const* bufs, int self, int nRanks, int iters, int slice,
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
    for (int p = 0; p < nRanks; ++p) {
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

// One rank with its abort flag raised and no live peers: barrier 1 gives up, then late peers drain its flags.
// Barrier 2 must signal nothing, or a late peer would pair with it and read a shard this rank never produced.
// Only the nRanks live peers drain, so a signal on a lane at or above nRanks survives for the host to see.
__global__ void abortedBarrierStaysSilent(IpcGpuBarrier bar, uint32_t* const* flags, int self, int nRanks) {
  bar.syncOnSameBlockIdx<true, true>();
  if (threadIdx.x < static_cast<unsigned>(nRanks)) {
    __atomic_store_n(flags[threadIdx.x] + blockIdx.x * kRanks + self, 0u, __ATOMIC_RELAXED);
  }
  __syncthreads();
  bar.syncOnSameBlockIdx<true, false>();
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

bool havePeerGpus(int nGpus) {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess || n < nGpus) {
    return false;
  }
  for (int i = 0; i < nGpus; ++i) {
    for (int j = 0; j < nGpus; ++j) {
      int can = 0;
      if (i != j && (hipDeviceCanAccessPeer(&can, i, j) != hipSuccess || !can)) {
        return false;
      }
    }
  }
  return true;
}

// Gate in the parent only: a skipped isolated child is scored as a pass.
bool shouldSkip(int nGpus = kRanks) {
  return std::getenv(ProcessIsolatedTestRunner::kReexecMarkerEnvVar) == nullptr && !havePeerGpus(nGpus);
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

// nRanks < kRanks runs the clique on GPUs 0..nRanks-1 with barriers built for nRanks live peers.
void runOrderingTest(Shape shape, Alloc alloc, int iters, int slice, int nRanks = kRanks) {
  const bool flat = shape == Shape::kFlat || shape == Shape::kFlatAcquireOnly;
  // The mailbox kernels hardcode the 8-rank ring; only the barrier shapes run on a smaller clique.
  ASSERT_TRUE(flat || nRanks == kRanks);
  for (int i = 0; i < nRanks; ++i) {
    HIP_CHECK(hipSetDevice(i));
    for (int j = 0; j < nRanks; ++j) {
      if (i != j) {
        hipError_t e = hipDeviceEnablePeerAccess(j, 0);
        ASSERT_TRUE(e == hipSuccess || e == hipErrorPeerAccessAlreadyEnabled) << hipGetErrorString(e);
      }
    }
  }
  const int words = flat ? slice : kMpWords;
  const size_t payloadBytes = static_cast<size_t>(kBlocks) * words * sizeof(uint32_t);
  const size_t flagBytes = static_cast<size_t>(kRanks) * kBlocks * sizeof(uint32_t);

  std::vector<void*> allocations;
  std::vector<uint32_t*> bufs(nRanks);
  std::vector<DeviceMailbox> dataMb(nRanks);
  std::vector<DeviceMailbox> ackMb(nRanks);
  for (int r = 0; r < nRanks; ++r) {
    HIP_CHECK(hipSetDevice(r));
    bufs[r] = static_cast<uint32_t*>(allocZeroed(payloadBytes, alloc));
    // Mailbox flags always use the production DeviceBuffer allocation.
    void* d = allocZeroed(flagBytes, Alloc::kUncached);
    void* a = allocZeroed(flagBytes, Alloc::kUncached);
    allocations.insert(allocations.end(), {bufs[r], d, a});
    dataMb[r] = makeMailbox(static_cast<uint32_t*>(d));
    ackMb[r] = makeMailbox(static_cast<uint32_t*>(a));
  }
  // Production fills only the first nRanks barrier mailboxes. Point each unused lane at its own never-signaled local
  // flags, so a barrier that still walks all eight lanes hangs on them and trips the timeout rather than faulting.
  std::vector<std::vector<DeviceMailbox>> barMb(nRanks, std::vector<DeviceMailbox>(kRanks));
  for (int r = 0; r < nRanks; ++r) {
    HIP_CHECK(hipSetDevice(r));
    for (int p = 0; p < kRanks; ++p) {
      if (p < nRanks) {
        barMb[r][p] = dataMb[p];
      } else {
        void* t = allocZeroed(flagBytes, Alloc::kUncached);
        allocations.push_back(t);
        barMb[r][p] = makeMailbox(static_cast<uint32_t*>(t));
      }
    }
  }

  std::vector<uint32_t**> dBufs(nRanks);
  std::vector<DeviceMailbox*> dData(nRanks);
  std::vector<DeviceMailbox*> dAck(nRanks);
  std::vector<unsigned long long*> dBad(nRanks);
  std::vector<hipStream_t> streams(nRanks);
  for (int r = 0; r < nRanks; ++r) {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipMalloc(&dBufs[r], nRanks * sizeof(uint32_t*)));
    HIP_CHECK(hipMemcpy(dBufs[r], bufs.data(), nRanks * sizeof(uint32_t*), hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&dData[r], nRanks * sizeof(DeviceMailbox)));
    HIP_CHECK(hipMemcpy(dData[r], dataMb.data(), nRanks * sizeof(DeviceMailbox), hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&dAck[r], nRanks * sizeof(DeviceMailbox)));
    HIP_CHECK(hipMemcpy(dAck[r], ackMb.data(), nRanks * sizeof(DeviceMailbox), hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&dBad[r], sizeof(unsigned long long)));
    HIP_CHECK(hipMemset(dBad[r], 0, sizeof(unsigned long long)));
    HIP_CHECK(hipStreamCreateWithFlags(&streams[r], hipStreamNonBlocking));
  }

  // All the kernels spin on each other, so every launch precedes the first synchronize.
  for (int r = 0; r < nRanks; ++r) {
    HIP_CHECK(hipSetDevice(r));
    if (shape == Shape::kFlat) {
      hipLaunchKernelGGL(flatSync<false>, dim3(kBlocks), dim3(kFlatThreads), 0, streams[r],
                         makeBarrier(r, barMb[r], DdaKernelParams{}, nRanks), dBufs[r], r, nRanks, iters, slice,
                         dBad[r]);
    } else if (shape == Shape::kFlatAcquireOnly) {
      hipLaunchKernelGGL(flatSync<true>, dim3(kBlocks), dim3(kFlatThreads), 0, streams[r],
                         makeBarrier(r, barMb[r], DdaKernelParams{}, nRanks), dBufs[r], r, nRanks, iters, slice,
                         dBad[r]);
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
  for (int r = 0; r < nRanks; ++r) {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipStreamSynchronize(streams[r]));
    unsigned long long bad = 0;
    HIP_CHECK(hipMemcpy(&bad, dBad[r], sizeof(bad), hipMemcpyDeviceToHost));
    totalBad += bad;
  }
  EXPECT_EQ(totalBad, 0ull) << "receivers observed payload words older than the flag they acquired";

  for (int r = 0; r < nRanks; ++r) {
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

// All eight mailboxes live on one GPU, so no peer access is needed and the other "ranks" never run.
// With nRanks < kRanks the barrier still holds all eight, as unused-lane decoys a correct barrier never touches.
void runStickyAbortTest(int nRanks = kRanks) {
  constexpr int kSelf = 0;
  HIP_CHECK(hipSetDevice(0));
  const size_t flagBytes = static_cast<size_t>(kRanks) * kBlocks * sizeof(uint32_t);
  std::vector<uint32_t*> flags(kRanks);
  std::vector<DeviceMailbox> mailboxes(kRanks);
  for (int r = 0; r < kRanks; ++r) {
    flags[r] = static_cast<uint32_t*>(allocZeroed(flagBytes, Alloc::kUncached));
    mailboxes[r] = makeMailbox(flags[r]);
  }
  uint32_t* abortFlag = nullptr;
  const uint32_t raised = 1;
  HIP_CHECK(hipMalloc(&abortFlag, sizeof(uint32_t)));
  HIP_CHECK(hipMemcpy(abortFlag, &raised, sizeof(raised), hipMemcpyHostToDevice));
  uint32_t** dFlags = nullptr;
  HIP_CHECK(hipMalloc(&dFlags, kRanks * sizeof(uint32_t*)));
  HIP_CHECK(hipMemcpy(dFlags, flags.data(), kRanks * sizeof(uint32_t*), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(abortedBarrierStaysSilent, dim3(kBlocks), dim3(64), 0, nullptr,
                     makeBarrier(kSelf, mailboxes, DdaKernelParams{abortFlag}, nRanks), dFlags, kSelf, nRanks);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  int signaled = 0;
  int strayed = 0;
  std::vector<uint32_t> host(static_cast<size_t>(kRanks) * kBlocks);
  for (int p = 0; p < kRanks; ++p) {
    HIP_CHECK(hipMemcpy(host.data(), flags[p], flagBytes, hipMemcpyDeviceToHost));
    for (int b = 0; p != kSelf && b < kBlocks; ++b) {
      (p < nRanks ? signaled : strayed) += host[static_cast<size_t>(b) * kRanks + kSelf] != 0;
    }
  }
  EXPECT_EQ(signaled, 0) << "an aborted block signaled a barrier after giving up on an earlier one";
  EXPECT_EQ(strayed, 0) << "a barrier with nRanks = " << nRanks << " signaled a lane at or above nRanks";

  HIP_EXPECT(hipFree(dFlags));
  HIP_EXPECT(hipFree(abortFlag));
  for (uint32_t* f : flags) {
    HIP_EXPECT(hipFree(f));
  }
}

ProcessIsolatedTestRunner::TestConfig makeConfig(const char* name, Shape shape, Alloc alloc, int iters, int slice = 0,
                                                 int nRanks = kRanks) {
  return ProcessIsolatedTestRunner::TestConfig(name, [=]() { runOrderingTest(shape, alloc, iters, slice, nRanks); })
    .withTimeout(std::chrono::seconds(60))
    .withNumGpus(nRanks);
}

constexpr const char* kSkipReason = "needs 8 GPUs with full peer access";
constexpr const char* kPartialSkipReason = "needs 4 GPUs with full peer access";

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

// The barriers carry nRanks = 4: one that still signals or waits on all eight lanes hangs on the tail and times out;
// one that signals too few peers leaves a rank waiting forever, or lets a reader see a slice before it is written.
TEST(DdaIpcGpuBarrierOrdering, FlatSync_FourRanks) {
  if (shouldSkip(kPartialRanks)) {
    GTEST_SKIP() << kPartialSkipReason;
  }
  RUN_ISOLATED_TESTS(makeConfig("DdaIpcGpuBarrierOrdering.FlatSync_FourRanks", Shape::kFlat, Alloc::kUncached,
                                kFlatIters, kFlatSlice, kPartialRanks));
}

TEST(DdaIpcGpuBarrierOrdering, AbortedBarrierStaysSilent) {
  int n = 0;
  if (std::getenv(ProcessIsolatedTestRunner::kReexecMarkerEnvVar) == nullptr &&
      (hipGetDeviceCount(&n) != hipSuccess || n < 1)) {
    GTEST_SKIP() << "needs a GPU";
  }
  RUN_ISOLATED_TESTS(ProcessIsolatedTestRunner::TestConfig("DdaIpcGpuBarrierOrdering.AbortedBarrierStaysSilent",
                                                           []() { runStickyAbortTest(); })
                       .withTimeout(std::chrono::seconds(60))
                       .withNumGpus(1));
}

// nRanks = 4 with all eight mailboxes valid: the barrier must still give up and stay silent, and only its four live
// lanes may ever signal, so a barrier that walks all eight would leave a flag in an unused lane's mailbox.
TEST(DdaIpcGpuBarrierOrdering, AbortedBarrierStaysSilent_FourRanks) {
  int n = 0;
  if (std::getenv(ProcessIsolatedTestRunner::kReexecMarkerEnvVar) == nullptr &&
      (hipGetDeviceCount(&n) != hipSuccess || n < 1)) {
    GTEST_SKIP() << "needs a GPU";
  }
  RUN_ISOLATED_TESTS(
    ProcessIsolatedTestRunner::TestConfig("DdaIpcGpuBarrierOrdering.AbortedBarrierStaysSilent_FourRanks",
                                          []() { runStickyAbortTest(kPartialRanks); })
      .withTimeout(std::chrono::seconds(60))
      .withNumGpus(1));
}

} // namespace RcclUnitTesting
