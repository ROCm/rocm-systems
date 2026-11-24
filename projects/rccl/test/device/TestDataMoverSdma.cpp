/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// SdmaMover against the Anvil device stub (test/device/sdma/anvil_device.hpp):
// which packets are posted, that wait() polls its ticket with isFlushed() rather
// than quiet(), that only team rank 0 posts, that wait() honours an abort flag,
// and that a DataMover chain falls through to the vector path below the
// threshold. Built only with ENABLE_ROCSHMEM_GIN, like the other stub users.

#include "DeviceTestBase.hpp"

#include <cstdint>
#include <vector>

#include "datamover/dataMover.h"

namespace RcclUnitTesting {
namespace {

namespace dm = rccl::dm;
using Queue = sdma_anvil::SdmaQueueDeviceHandle;

void setRecordOnly(bool on) {
  sdma_anvil::SdmaStubLog log{};
  log.recordOnly = on;
  HIP_EXPECT(hipMemcpyToSymbol(HIP_SYMBOL(sdma_anvil::g_sdmaStubLog), &log, sizeof(log)));
}

// Restores copying puts on every exit path, so later tests in the process are unaffected.
class RecordOnlyScope {
public:
  RecordOnlyScope() { setRecordOnly(true); }
  ~RecordOnlyScope() { setRecordOnly(false); }
  RecordOnlyScope(const RecordOnlyScope&) = delete;
  RecordOnlyScope& operator=(const RecordOnlyScope&) = delete;
};

sdma_anvil::SdmaStubLog readLog() {
  sdma_anvil::SdmaStubLog log{};
  HIP_EXPECT(hipMemcpyFromSymbol(&log, HIP_SYMBOL(sdma_anvil::g_sdmaStubLog), sizeof(log)));
  return log;
}

// The stub's isFlushed() reports busy for busyPolls polls, then compares
// writeIndex - readLag against the target; quietTarget() returns writeIndex.
void setQueue(DeviceBuffer<Queue>& q, uint64_t writeIndex, uint64_t busyPolls) {
  Queue h{};
  h.writeIndex = writeIndex;
  h.busyPolls = busyPolls;
  q.upload(h);
}

__global__ void kernelSdmaMove(Queue* q, uint32_t* abortFlag, const char* src, char* dst0, char* dst1,
                               int nDst, size_t bytes, int* aborted) {
  dm::DataMover<dm::SdmaMover> mover(dm::SdmaMover(q, /*minSdmaBytes=*/1, abortFlag));
  const dm::Team team = dm::Team::block();
  if (nDst == 1) mover.move(team, dm::global(src), dm::global(dst0), bytes);
  else mover.move(team, dm::global(src), dm::global(dst0, dst1), bytes);
  if (threadIdx.x == 0) *aborted = mover.head.wasAborted() ? 1 : 0;
}

__global__ void kernelChainMove(Queue* q, size_t minSdmaBytes, const char* src, char* dst, size_t bytes) {
  dm::DataMover<dm::SdmaMover, dm::VectorMover<>> mover(dm::SdmaMover(q, minSdmaBytes), dm::VectorMover<>());
  mover.move(dm::Team::block(), dm::global(src), dm::global(dst), bytes);
}

class DataMoverSdmaTest : public DeviceTestBase {};

// One put per (source, destination), posted by rank 0 only, and wait() polls the
// ticket (isFlushed) instead of quiet().
TEST_F(DataMoverSdmaTest, PostsOnePutFromRankZeroAndPollsTicket) {
  DeviceBuffer<Queue> q(1);
  setQueue(q, /*writeIndex=*/7, /*busyPolls=*/3);
  DeviceBuffer<char> src(64), dst(64);
  DeviceBuffer<int> aborted(1);
  sdma_anvil::SdmaStubLog log{};
  {
    RecordOnlyScope recordOnly;
    kernelSdmaMove<<<1, 128>>>(q.ptr, nullptr, src.ptr, dst.ptr, nullptr, 1, 4096, aborted.ptr);
    syncAndCheck();
    log = readLog();
  }
  ASSERT_EQ(log.count, 1u) << "only team rank 0 may post";
  EXPECT_EQ(log.calls[0].op, sdma_anvil::kSdmaStubPut);
  EXPECT_EQ(log.calls[0].dst, static_cast<void*>(dst.ptr));
  EXPECT_EQ(log.calls[0].src, static_cast<void*>(src.ptr));
  EXPECT_EQ(log.calls[0].size, 4096u);
  EXPECT_EQ(q.download().polls, 5u)
    << "3 busy polls, 1 that reaches the ticket (writeIndex 7), 1 that confirms it was not an abort";
  EXPECT_EQ(aborted.download(), 0);
}

// Transfers above gin_sdma::kGinPutSegBytes split at the cap; a 1->2 broadcast
// posts the segments once per destination.
TEST_F(DataMoverSdmaTest, SegmentsLargeTransfersPerDestination) {
  constexpr size_t kSeg = gin_sdma::kGinPutSegBytes;
  DeviceBuffer<Queue> q(1);
  setQueue(q, /*writeIndex=*/1, /*busyPolls=*/0);
  DeviceBuffer<char> src(8), d0(8), d1(8);  // record-only: the stub never touches the bytes
  DeviceBuffer<int> aborted(1);
  sdma_anvil::SdmaStubLog log{};
  {
    RecordOnlyScope recordOnly;
    kernelSdmaMove<<<1, 64>>>(q.ptr, nullptr, src.ptr, d0.ptr, d1.ptr, 2, kSeg + 1, aborted.ptr);
    syncAndCheck();
    log = readLog();
  }
  ASSERT_EQ(log.count, 4u);
  const char* dsts[2] = {d0.ptr, d1.ptr};
  for (int d = 0; d < 2; ++d) {
    EXPECT_EQ(log.calls[2 * d].dst, static_cast<void*>(const_cast<char*>(dsts[d])));
    EXPECT_EQ(log.calls[2 * d].size, kSeg);
    EXPECT_EQ(log.calls[2 * d + 1].dst, static_cast<void*>(const_cast<char*>(dsts[d]) + kSeg));
    EXPECT_EQ(log.calls[2 * d + 1].src, static_cast<void*>(src.ptr + kSeg));
    EXPECT_EQ(log.calls[2 * d + 1].size, 1u);
  }
}

// A set abort flag ends a wait that would otherwise poll (practically) forever
// (waitFlushed() reads the flag every few thousand polls).
TEST_F(DataMoverSdmaTest, WaitStopsOnAbort) {
  DeviceBuffer<Queue> q(1);
  setQueue(q, /*writeIndex=*/1, /*busyPolls=*/~0ull);
  DeviceBuffer<uint32_t> abortFlag(1);
  abortFlag.upload(1u);
  DeviceBuffer<char> src(64), dst(64);
  DeviceBuffer<int> aborted(1);
  {
    RecordOnlyScope recordOnly;
    kernelSdmaMove<<<1, 64>>>(q.ptr, abortFlag.ptr, src.ptr, dst.ptr, nullptr, 1, 64, aborted.ptr);
    syncAndCheck();
  }
  EXPECT_EQ(aborted.download(), 1);
  EXPECT_GE(q.download().polls, 1u);
}

// wait() polls up to the ticket captured after the put (quietTarget() = the stub's
// writeIndex), not to some other index: with readLag 3 the stub's read pointer
// stops at writeIndex - 3, so the ticket is never reached and only the abort
// flag ends the wait. A ticket of 0 (or anything <= 7) would return at once.
TEST_F(DataMoverSdmaTest, WaitTargetsTheCapturedTicket) {
  DeviceBuffer<Queue> q(1);
  Queue h{};
  h.writeIndex = 10;
  h.readLag = 3;
  q.upload(h);
  DeviceBuffer<uint32_t> abortFlag(1);
  abortFlag.upload(1u);
  DeviceBuffer<char> src(64), dst(64);
  DeviceBuffer<int> aborted(1);
  {
    RecordOnlyScope recordOnly;
    kernelSdmaMove<<<1, 64>>>(q.ptr, abortFlag.ptr, src.ptr, dst.ptr, nullptr, 1, 64, aborted.ptr);
    syncAndCheck();
  }
  EXPECT_EQ(aborted.download(), 1) << "the wait must target writeIndex 10, which a read pointer of 7 never reaches";

  // Same queue with no lag: the ticket is reached and the abort flag is never consulted.
  h.readLag = 0;
  q.upload(h);
  {
    RecordOnlyScope recordOnly;
    kernelSdmaMove<<<1, 64>>>(q.ptr, abortFlag.ptr, src.ptr, dst.ptr, nullptr, 1, 64, aborted.ptr);
    syncAndCheck();
  }
  EXPECT_EQ(aborted.download(), 0);
}

// Below minSdmaBytes the chain falls through to VectorMover: no packet, real copy.
TEST_F(DataMoverSdmaTest, ChainFallsBackToVectorBelowThreshold) {
  constexpr size_t kBytes = 1000;
  DeviceBuffer<Queue> q(1);
  setQueue(q, /*writeIndex=*/1, /*busyPolls=*/0);
  std::vector<char> host(kBytes);
  for (size_t i = 0; i < kBytes; ++i) host[i] = static_cast<char>(i * 7 + 3);
  DeviceBuffer<char> src(kBytes), dst(kBytes);
  src.copyFrom(host);
  dst.zero();
  sdma_anvil::SdmaStubLog log{};
  {
    RecordOnlyScope recordOnly;
    kernelChainMove<<<1, 128>>>(q.ptr, /*minSdmaBytes=*/kBytes + 1, src.ptr, dst.ptr, kBytes);
    syncAndCheck();
    log = readLog();
  }
  EXPECT_EQ(log.count, 0u) << "the SDMA backend must not accept a move below its threshold";
  EXPECT_EQ(dst.copyTo(), host);
}

}  // namespace
}  // namespace RcclUnitTesting
