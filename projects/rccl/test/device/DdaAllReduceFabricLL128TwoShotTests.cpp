/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// DDA LL128 AllReduce two-shot (cooperative and LDS-staged writer) on one GPU; a helper kernel plays the peers.

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "DeviceTestBase.hpp"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll128_twoshot.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting {

namespace {

using dda::common::DdaKernelParams;
using dda::common::kDdaLL128DataBytesPerSlice;
using dda::common::kDdaLL128Warp;
using dda::common::kDdaLL128WireAsync;
using dda::common::kDdaLL128WireCoop;
using dda::common::kDdaLL128WireWordsPerSlice;
using dda::common::kDdaLL128WordsPerThread;

// Two full slices and a partial third.
constexpr size_t kSmallShardBytes = 2 * kDdaLL128DataBytesPerSlice + 112;
// 40 slices and a partial one: on 1 x 512 most warps fill phase 2 twice, so a 16-rank fill waits on 15 pushes.
constexpr size_t kMediumShardBytes = 40 * kDdaLL128DataBytesPerSlice + 112;
// The launcher's staged cutoff plus a partial last slice: every warp of an 8 x 512 grid refills its windows ~9 times.
constexpr size_t kLargeShardBytes = 2 * 1024 * 1024 + 112;
constexpr uint32_t kFlag = 2;  // epoch cells start at 1, so the kernels use flag 2 and bank 0
constexpr uint8_t kUntouched = 0xA5;

enum class Kind { kCoop, kStaged };

struct Case {
  int nRanks;
  int self;
  int ct;  // NRANKS_CT of the instance: 4, 8 or 0
  Kind kind;
  unsigned grid;
  unsigned block;
  size_t shardBytes;
};

size_t slices(size_t shardBytes) {
  return dda::common::ddaLL128Slices(shardBytes);
}

// A bank of exactly two staging areas of nRanks slots, each slot one shard's slices.
size_t bankBytes(int nRanks, size_t shardBytes) {
  return 2 * (size_t)nRanks * slices(shardBytes) * kDdaLL128WireWordsPerSlice * sizeof(uint64_t);
}

size_t slotWords(int nRanks, size_t shardBytes) {
  return dda::common::ddaLL128ArTwoShotSlotWords(bankBytes(nRanks, shardBytes), nRanks);
}

// Writes rows[peers[blockIdx.x]] into area + peer * slotWords as the LL128 lines that peer sends: one warp per
// (peer, slice).
__global__ void PublishLines(uint64_t* area, const int8_t* rows, const int* peers, size_t slotWordsArg,
                             size_t shardBytes) {
  const int peer = peers[blockIdx.x];
  const size_t s = blockIdx.y;
  const int lane = threadIdx.x;
  const bool flagLane = dda::common::ddaLL128IsFlagLane(lane);
  const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
  const size_t rem = shardBytes - sliceByte;
  const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
  uint64_t regs[kDdaLL128WordsPerThread] = {};
  dda::common::ddaLL128LoadRegs<int8_t>(regs, rows + (size_t)peer * shardBytes + sliceByte, elt, lane, flagLane);
  const uint64_t flag = ((uint64_t)kFlag << 32) | kFlag;
  dda::common::ddaLL128StoreWire(area + (size_t)peer * slotWordsArg + s * kDdaLL128WireWordsPerSlice + 2 * lane, regs,
                                 flag, flagLane);
}

__global__ void ReportStaged(int* out) {
  *out = dda::common::ddaLL128Staged<kDdaLL128WireAsync>() ? 1 : 0;
}

template <typename T, int N>
void launch(const Case& c, T** scratch, T* recv, const T* send, uint32_t* epoch, uint32_t* abortFlag) {
  const DdaKernelParams params{abortFlag};
  const dim3 grid(c.grid);
  const dim3 block(c.block);
  const size_t bank = bankBytes(c.nRanks, c.shardBytes);
  const size_t n = slices(c.shardBytes);
  if (c.kind == Kind::kCoop) {
    hipLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL128<T, N, kDdaLL128WireCoop>), grid, block, 0, 0, scratch,
                       recv, send, c.shardBytes, c.self, c.nRanks, epoch, (int)c.grid, n, bank, params);
  } else {
    hipLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL128<T, N, kDdaLL128WireAsync>), grid, block,
                       dda::common::ddaLL128StageBytes(c.block), 0, scratch, recv, send, c.shardBytes, c.self,
                       c.nRanks, epoch, (int)c.grid, n, bank, params);
  }
}

// What one launch leaves behind: recvbuff and every peer's scratch (self's own scratch is the kernel's private state).
struct Snapshot {
  std::vector<uint8_t> recv;
  std::vector<std::vector<uint8_t>> peerScratch;
};

class DdaAllReduceFabricLL128TwoShot : public DeviceTestBase {
 protected:
  void SetUp() override {
    DeviceTestBase::SetUp();
    if (!isGfx1250()) {
      GTEST_SKIP() << "the LL128 AllReduce tiers run on gfx1250";
    }
  }

  // Skips the test on a device pass without async LDS stores, where the staged instance is the cooperative one.
  bool stagedBuilt() {
    DeviceBuffer<int> staged(1);
    hipLaunchKernelGGL(ReportStaged, dim3(1), dim3(1), 0, 0, staged.ptr);
    HIP_EXPECT(hipDeviceSynchronize());
    return staged.download() != 0;
  }

  // send_[r] is rank r's message (nRanks shards); reduced_[j] is the reduced shard j a peer publishes.
  void setUp(int nRanks, int self, size_t shardBytes, const std::vector<std::vector<uint8_t>>& send,
             const std::vector<std::vector<uint8_t>>& reduced) {
    nRanks_ = nRanks;
    self_ = self;
    shardBytes_ = shardBytes;
    send_ = send;
    reduced_ = reduced;
    scratch_.clear();
    std::vector<uint8_t*> ptrs;
    for (int r = 0; r < nRanks; ++r) {
      scratch_.emplace_back(2 * bankBytes(nRanks, shardBytes));
      ptrs.push_back(scratch_.back().ptr);
    }
    scratchPtrs_ = std::make_unique<DeviceBuffer<uint8_t*>>(ptrs.size());
    scratchPtrs_->copyFrom(ptrs);
    sendBuf_ = std::make_unique<DeviceBuffer<uint8_t>>((size_t)nRanks * shardBytes);
    sendBuf_->copyFrom(send_[(size_t)self]);
    recvBuf_ = std::make_unique<DeviceBuffer<uint8_t>>((size_t)nRanks * shardBytes);
  }

  // Zeroed scratch, untouched recvbuff, fresh epoch cells, then the peers' lines; `skip` sends no first-area copy.
  void reset(uint32_t abortFlag, int skip = -1) {
    for (DeviceBuffer<uint8_t>& b : scratch_) {
      b.zero();
    }
    HIP_CHECK(hipMemset(recvBuf_->ptr, kUntouched, recvBuf_->count));
    epoch_.copyFrom(std::vector<uint32_t>(kMaxGrid, kFlag - 1));
    abort_.upload(abortFlag);

    std::vector<int8_t> copies((size_t)nRanks_ * shardBytes_);
    std::vector<int8_t> reduced((size_t)nRanks_ * shardBytes_);
    for (int p = 0; p < nRanks_; ++p) {
      std::memcpy(&copies[(size_t)p * shardBytes_], &send_[(size_t)p][(size_t)self_ * shardBytes_], shardBytes_);
      std::memcpy(&reduced[(size_t)p * shardBytes_], reduced_[(size_t)p].data(), shardBytes_);
    }
    uint64_t* self = reinterpret_cast<uint64_t*>(scratch_[(size_t)self_].ptr);
    publish(self, copies, skip);
    publish(self + (size_t)nRanks_ * slotWords(nRanks_, shardBytes_), reduced, -1);
  }

  void publish(uint64_t* area, const std::vector<int8_t>& rows, int skip) {
    std::vector<int> peers;
    for (int p = 0; p < nRanks_; ++p) {
      if (p != self_ && p != skip) {
        peers.push_back(p);
      }
    }
    DeviceBuffer<int8_t> dRows(rows.size());
    dRows.copyFrom(rows);
    DeviceBuffer<int> dPeers(peers.size());
    dPeers.copyFrom(peers);
    hipLaunchKernelGGL(PublishLines, dim3((unsigned)peers.size(), (unsigned)slices(shardBytes_)), dim3(kDdaLL128Warp),
                       0, 0, area, dRows.ptr, dPeers.ptr, slotWords(nRanks_, shardBytes_), shardBytes_);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  template <typename T>
  void run(const Case& c) {
    T** scratch = reinterpret_cast<T**>(scratchPtrs_->ptr);
    T* recv = reinterpret_cast<T*>(recvBuf_->ptr);
    const T* send = reinterpret_cast<const T*>(sendBuf_->ptr);
    switch (c.ct) {
      case 4:
        launch<T, 4>(c, scratch, recv, send, epoch_.ptr, abort_.ptr);
        break;
      case 8:
        launch<T, 8>(c, scratch, recv, send, epoch_.ptr, abort_.ptr);
        break;
      default:
        launch<T, 0>(c, scratch, recv, send, epoch_.ptr, abort_.ptr);
        break;
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  Snapshot snapshot() {
    Snapshot snap{recvBuf_->copyTo(), {}};
    for (int p = 0; p < nRanks_; ++p) {
      snap.peerScratch.push_back(p == self_ ? std::vector<uint8_t>() : scratch_[(size_t)p].copyTo());
    }
    return snap;
  }

  // The second area slot[self] of every rank, where only a fully reduced shard of mine may land.
  bool reducedPublished() {
    const size_t words = slotWords(nRanks_, shardBytes_);
    const size_t at = (size_t)(nRanks_ + self_) * words * sizeof(uint64_t);
    for (const DeviceBuffer<uint8_t>& b : scratch_) {
      const std::vector<uint8_t> bytes = b.copyTo();
      for (size_t i = at; i < at + words * sizeof(uint64_t); ++i) {
        if (bytes[i] != 0) {
          return true;
        }
      }
    }
    return false;
  }

  // Runs the cooperative two-shot, then c, on the same peer lines and grid: c must leave the same recvbuff and lines.
  template <typename T>
  Snapshot expectMatchesCoop(const Case& c) {
    SCOPED_TRACE(testing::Message() << c.nRanks << "r ct " << c.ct << " shard " << c.shardBytes << " " << c.grid
                                    << "x" << c.block);
    reset(0);
    run<T>({c.nRanks, c.self, c.ct, Kind::kCoop, c.grid, c.block, c.shardBytes});
    const Snapshot want = snapshot();
    reset(0);
    run<T>(c);
    const Snapshot got = snapshot();
    EXPECT_TRUE(got.recv == want.recv) << "recvbuff differs from the cooperative writer's";
    for (int p = 0; p < c.nRanks; ++p) {
      EXPECT_TRUE(got.peerScratch[(size_t)p] == want.peerScratch[(size_t)p]) << "peer " << p << " lines differ";
    }
    expectEpochAdvanced(c.grid);
    return want;
  }

  void expectEpochAdvanced(unsigned grid) {
    const std::vector<uint32_t> epoch = epoch_.copyTo();
    for (unsigned b = 0; b < grid; ++b) {
      EXPECT_EQ(epoch[b], kFlag) << "block " << b;
    }
  }

  static constexpr size_t kMaxGrid = 8;
  int nRanks_ = 0;
  int self_ = 0;
  size_t shardBytes_ = 0;
  std::vector<std::vector<uint8_t>> send_;
  std::vector<std::vector<uint8_t>> reduced_;
  std::vector<DeviceBuffer<uint8_t>> scratch_;
  std::unique_ptr<DeviceBuffer<uint8_t*>> scratchPtrs_;
  std::unique_ptr<DeviceBuffer<uint8_t>> sendBuf_;
  std::unique_ptr<DeviceBuffer<uint8_t>> recvBuf_;
  DeviceBuffer<uint32_t> epoch_{kMaxGrid};
  DeviceBuffer<uint32_t> abort_{1};
};

// Inexact float sums pin the fold order: my shard first, then the peers from me + 1 on.
std::vector<std::vector<uint8_t>> FloatMessages(int nRanks, size_t shardBytes) {
  std::vector<std::vector<uint8_t>> send((size_t)nRanks, std::vector<uint8_t>((size_t)nRanks * shardBytes));
  for (int r = 0; r < nRanks; ++r) {
    float* f = reinterpret_cast<float*>(send[(size_t)r].data());
    for (size_t i = 0; i < (size_t)nRanks * shardBytes / sizeof(float); ++i) {
      f[i] = 1.0f / (float)(3 + r) + (float)((r * 131 + (int)i * 17) % 1000) * 1.0e-3f;
    }
  }
  return send;
}

// Shard j folded by its owner: rank j's copy, then ranks j + 1, j + 2, ...
std::vector<std::vector<uint8_t>> FloatReduced(const std::vector<std::vector<uint8_t>>& send, int nRanks,
                                               size_t shardBytes) {
  std::vector<std::vector<uint8_t>> reduced((size_t)nRanks, std::vector<uint8_t>(shardBytes));
  for (int j = 0; j < nRanks; ++j) {
    float* acc = reinterpret_cast<float*>(reduced[(size_t)j].data());
    for (size_t i = 0; i < shardBytes / sizeof(float); ++i) {
      const size_t at = (size_t)j * shardBytes / sizeof(float) + i;
      acc[i] = reinterpret_cast<const float*>(send[(size_t)j].data())[at];
      for (int r = 1; r < nRanks; ++r) {
        acc[i] = acc[i] + reinterpret_cast<const float*>(send[(size_t)((j + r) % nRanks)].data())[at];
      }
    }
  }
  return reduced;
}

// fp8 e4m3 positive normals up to 1.0 (0x08..0x38): an 8-rank sum stays finite, so the bytes compare exactly.
std::vector<std::vector<uint8_t>> Fp8Rows(int rows, size_t bytes, int seed) {
  std::vector<std::vector<uint8_t>> out((size_t)rows, std::vector<uint8_t>(bytes));
  for (int r = 0; r < rows; ++r) {
    for (size_t i = 0; i < bytes; ++i) {
      out[(size_t)r][i] = (uint8_t)(0x08 + (r * 37 + (int)i * 11 + seed) % 0x31);
    }
  }
  return out;
}

// The staged writer on 4, 8 (unrolled), 6 and 16 (runtime) ranks, below and above the launcher's 2 MiB staged cutoff:
// every wait rung (0 and 1, 3, 7, 15 pushes), one or many window refills per warp, and a partial last slice.
const Case kStagedCases[] = {
  {4, 1, 4, Kind::kStaged, 2, 512, kSmallShardBytes},    {6, 5, 0, Kind::kStaged, 2, 512, kSmallShardBytes},
  {8, 5, 8, Kind::kStaged, 2, 512, kSmallShardBytes},    {16, 9, 0, Kind::kStaged, 1, 512, kMediumShardBytes},
  {4, 1, 4, Kind::kStaged, 8, 512, kLargeShardBytes},    {6, 5, 0, Kind::kStaged, 8, 512, kLargeShardBytes},
  {8, 5, 8, Kind::kStaged, 8, 512, kLargeShardBytes},
};

// The cooperative two-shot gives the host fold bit for bit, and the staged writer leaves identical recvbuff and peer
// lines on index-dependent data.
TEST_F(DdaAllReduceFabricLL128TwoShot, StagedMatchesTheCooperativeWriter) {
  if (!stagedBuilt()) {
    GTEST_SKIP() << "this device pass has no async LDS stores";
  }
  for (const Case& c : kStagedCases) {
    const auto send = FloatMessages(c.nRanks, c.shardBytes);
    setUp(c.nRanks, c.self, c.shardBytes, send, FloatReduced(send, c.nRanks, c.shardBytes));
    const Snapshot want = expectMatchesCoop<float>(c);
    std::vector<uint8_t> expected;
    for (const std::vector<uint8_t>& shard : reduced_) {
      expected.insert(expected.end(), shard.begin(), shard.end());
    }
    ASSERT_TRUE(want.recv == expected) << "the cooperative two-shot left the host fold";
  }
}

// fp8 at 8 ranks is the staged instance with the most register pressure.
TEST_F(DdaAllReduceFabricLL128TwoShot, StagedMatchesTheCooperativeWriterOnFp8) {
  if (!stagedBuilt()) {
    GTEST_SKIP() << "this device pass has no async LDS stores";
  }
  const Case cases[] = {{8, 5, 8, Kind::kStaged, 2, 512, kSmallShardBytes},
                        {8, 5, 8, Kind::kStaged, 8, 512, kLargeShardBytes}};
  for (const Case& c : cases) {
    setUp(c.nRanks, c.self, c.shardBytes, Fp8Rows(c.nRanks, (size_t)c.nRanks * c.shardBytes, 0),
          Fp8Rows(c.nRanks, c.shardBytes, 5));
    expectMatchesCoop<dda::common::fp8e4m3>(c);
  }
}

// A peer that never sends aborts the staged two-shot: my shard is never stored or published, every async store
// drains, and the epoch still moves on.
TEST_F(DdaAllReduceFabricLL128TwoShot, StagedAbortLeavesNoPartialSum) {
  GTEST_SKIP() << "DDA waits no longer honor abortFlag (#11245 reverted)";
  for (const Case& c : {Case{8, 5, 8, Kind::kStaged, 2, 512, kSmallShardBytes},
                        Case{8, 5, 8, Kind::kStaged, 8, 512, kLargeShardBytes}}) {
    SCOPED_TRACE(testing::Message() << "shard " << c.shardBytes);
    const auto send = FloatMessages(c.nRanks, c.shardBytes);
    setUp(c.nRanks, c.self, c.shardBytes, send, FloatReduced(send, c.nRanks, c.shardBytes));
    reset(1, (c.self + 2) % c.nRanks);
    run<float>(c);
    const Snapshot got = snapshot();
    for (size_t i = 0; i < c.shardBytes; ++i) {
      ASSERT_EQ(got.recv[(size_t)c.self * c.shardBytes + i], kUntouched) << "byte " << i;
    }
    EXPECT_FALSE(reducedPublished());
    expectEpochAdvanced(c.grid);
  }
}

}  // namespace

}  // namespace RcclUnitTesting
