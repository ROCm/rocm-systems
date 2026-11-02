/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// DDA LL128 AllReduce two-shots (cooperative, LDS-staged, peer split) on one GPU; a helper kernel plays the peers.

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

constexpr size_t kShardBytes = 2 * kDdaLL128DataBytesPerSlice + 112;  // two full slices and a partial third
constexpr size_t kSlices = 3;
constexpr size_t kSlotSlices = 4;
constexpr uint32_t kFlag = 2;  // epoch cells start at 1, so the kernels use flag 2 and bank 0
constexpr uint8_t kUntouched = 0xA5;

enum class Kind { kCoop, kStaged, kSplit };

struct Case {
  int nRanks;
  int self;
  int ct;  // NRANKS_CT of the instance: 4, 8 or 0
  Kind kind;
  unsigned grid;
  unsigned block;
};

size_t bankBytes(int nRanks) {
  return 2 * (size_t)nRanks * kSlotSlices * kDdaLL128WireWordsPerSlice * sizeof(uint64_t);
}

size_t slotWords(int nRanks) {
  return dda::common::ddaLL128ArTwoShotSlotWords(bankBytes(nRanks), nRanks);
}

// Writes rows[peers[blockIdx.x]] into area + peer * slotWords as the LL128 lines that peer sends: one warp per
// (peer, slice).
__global__ void PublishLines(uint64_t* area, const int8_t* rows, const int* peers, size_t slotWordsArg) {
  const int peer = peers[blockIdx.x];
  const size_t s = blockIdx.y;
  const int lane = threadIdx.x;
  const bool flagLane = dda::common::ddaLL128IsFlagLane(lane);
  const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
  const size_t rem = kShardBytes - sliceByte;
  const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
  uint64_t regs[kDdaLL128WordsPerThread] = {};
  dda::common::ddaLL128LoadRegs<int8_t>(regs, rows + (size_t)peer * kShardBytes + sliceByte, elt, lane, flagLane);
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
  const size_t bank = bankBytes(c.nRanks);
  switch (c.kind) {
    case Kind::kCoop:
      hipLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL128<T, N, kDdaLL128WireCoop>), grid, block, 0, 0, scratch,
                         recv, send, kShardBytes, c.self, c.nRanks, epoch, (int)c.grid, kSlices, bank, params);
      break;
    case Kind::kStaged:
      hipLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL128<T, N, kDdaLL128WireAsync>), grid, block,
                         dda::common::ddaLL128StageBytes(c.block), 0, scratch, recv, send, kShardBytes, c.self,
                         c.nRanks, epoch, (int)c.grid, kSlices, bank, params);
      break;
    case Kind::kSplit:
      hipLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL128Split<T, N>), grid, block,
                         dda::common::ddaLL128ArSplitSharedBytes(c.block, c.nRanks), 0, scratch, recv, send,
                         kShardBytes, c.self, c.nRanks, epoch, (int)c.grid, kSlices, bank, params);
      break;
  }
}

// What one launch leaves behind: recvbuff and every peer's scratch (self's own scratch is the kernels' private state).
struct Snapshot {
  std::vector<uint8_t> recv;
  std::vector<std::vector<uint8_t>> peerScratch;
};

class DdaAllReduceFabricLL128Split : public DeviceTestBase {
 protected:
  void SetUp() override {
    DeviceTestBase::SetUp();
    if (!isGfx1250()) {
      GTEST_SKIP() << "the LL128 AllReduce tiers run on gfx1250";
    }
  }

  // send_[r] is rank r's message (nRanks shards); reduced_[j] is the reduced shard j a peer publishes.
  void setUp(int nRanks, int self, const std::vector<std::vector<uint8_t>>& send,
             const std::vector<std::vector<uint8_t>>& reduced) {
    nRanks_ = nRanks;
    self_ = self;
    send_ = send;
    reduced_ = reduced;
    scratch_.clear();
    std::vector<uint8_t*> ptrs;
    for (int r = 0; r < nRanks; ++r) {
      scratch_.emplace_back(2 * bankBytes(nRanks));
      ptrs.push_back(scratch_.back().ptr);
    }
    scratchPtrs_ = std::make_unique<DeviceBuffer<uint8_t*>>(ptrs.size());
    scratchPtrs_->copyFrom(ptrs);
    sendBuf_ = std::make_unique<DeviceBuffer<uint8_t>>((size_t)nRanks * kShardBytes);
    sendBuf_->copyFrom(send_[(size_t)self]);
    recvBuf_ = std::make_unique<DeviceBuffer<uint8_t>>((size_t)nRanks * kShardBytes);
  }

  // Zeroed scratch, untouched recvbuff, fresh epoch cells, then the peers' lines; `skip` sends no first-area copy.
  void reset(uint32_t abortFlag, int skip = -1) {
    for (DeviceBuffer<uint8_t>& b : scratch_) {
      b.zero();
    }
    HIP_CHECK(hipMemset(recvBuf_->ptr, kUntouched, recvBuf_->count));
    epoch_.copyFrom(std::vector<uint32_t>(kMaxGrid, kFlag - 1));
    abort_.upload(abortFlag);

    std::vector<int8_t> copies((size_t)nRanks_ * kShardBytes);
    std::vector<int8_t> reduced((size_t)nRanks_ * kShardBytes);
    for (int p = 0; p < nRanks_; ++p) {
      std::memcpy(&copies[(size_t)p * kShardBytes], &send_[(size_t)p][(size_t)self_ * kShardBytes], kShardBytes);
      std::memcpy(&reduced[(size_t)p * kShardBytes], reduced_[(size_t)p].data(), kShardBytes);
    }
    uint64_t* self = reinterpret_cast<uint64_t*>(scratch_[(size_t)self_].ptr);
    publish(self, copies, skip);
    publish(self + (size_t)nRanks_ * slotWords(nRanks_), reduced, -1);
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
    hipLaunchKernelGGL(PublishLines, dim3((unsigned)peers.size(), kSlices), dim3(kDdaLL128Warp), 0, 0, area, dRows.ptr,
                       dPeers.ptr, slotWords(nRanks_));
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
    const size_t at = (size_t)(nRanks_ + self_) * slotWords(nRanks_) * sizeof(uint64_t);
    for (const DeviceBuffer<uint8_t>& b : scratch_) {
      const std::vector<uint8_t> bytes = b.copyTo();
      for (size_t i = at; i < at + slotWords(nRanks_) * sizeof(uint64_t); ++i) {
        if (bytes[i] != 0) {
          return true;
        }
      }
    }
    return false;
  }

  // Runs the cooperative two-shot, then c, on the same peer lines: c must leave the same recvbuff and peer lines.
  template <typename T>
  Snapshot expectMatchesCoop(const Case& c) {
    SCOPED_TRACE(testing::Message() << c.nRanks << "r kind " << (int)c.kind << " " << c.grid << "x" << c.block);
    reset(0);
    run<T>({c.nRanks, c.self, c.ct, Kind::kCoop, 2, 512});
    const Snapshot want = snapshot();
    reset(0);
    run<T>(c);
    const Snapshot got = snapshot();
    EXPECT_EQ(got.recv, want.recv);
    for (int p = 0; p < c.nRanks; ++p) {
      EXPECT_EQ(got.peerScratch[(size_t)p], want.peerScratch[(size_t)p]) << "peer " << p;
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
std::vector<std::vector<uint8_t>> FloatMessages(int nRanks) {
  std::vector<std::vector<uint8_t>> send((size_t)nRanks, std::vector<uint8_t>((size_t)nRanks * kShardBytes));
  for (int r = 0; r < nRanks; ++r) {
    float* f = reinterpret_cast<float*>(send[(size_t)r].data());
    for (size_t i = 0; i < (size_t)nRanks * kShardBytes / sizeof(float); ++i) {
      f[i] = 1.0f / (float)(3 + r) + (float)((r * 131 + (int)i * 17) % 1000) * 1.0e-3f;
    }
  }
  return send;
}

// Shard j folded by its owner: rank j's copy, then ranks j + 1, j + 2, ...
std::vector<std::vector<uint8_t>> FloatReduced(const std::vector<std::vector<uint8_t>>& send, int nRanks) {
  std::vector<std::vector<uint8_t>> reduced((size_t)nRanks, std::vector<uint8_t>(kShardBytes));
  for (int j = 0; j < nRanks; ++j) {
    float* acc = reinterpret_cast<float*>(reduced[(size_t)j].data());
    for (size_t i = 0; i < kShardBytes / sizeof(float); ++i) {
      const size_t at = (size_t)j * kShardBytes / sizeof(float) + i;
      acc[i] = reinterpret_cast<const float*>(send[(size_t)j].data())[at];
      for (int r = 1; r < nRanks; ++r) {
        acc[i] = acc[i] + reinterpret_cast<const float*>(send[(size_t)((j + r) % nRanks)].data())[at];
      }
    }
  }
  return reduced;
}

const Case kCases[] = {
  {4, 1, 4, Kind::kStaged, 2, 512}, {4, 1, 4, Kind::kSplit, 1, 64},  {4, 1, 4, Kind::kSplit, 2, 512},
  {6, 5, 0, Kind::kStaged, 2, 512}, {6, 5, 0, Kind::kSplit, 1, 64},  {6, 5, 0, Kind::kSplit, 2, 512},
  {8, 5, 8, Kind::kStaged, 2, 512}, {8, 5, 8, Kind::kSplit, 1, 64},  {8, 5, 8, Kind::kSplit, 2, 512},
};

// The cooperative two-shot gives the host fold bit for bit; the staged writer and the split, on 1 x 64 threads (several
// passes, the look-ahead wrapping past the last unit) and on 2 x 512, leave identical recvbuff and peer lines.
TEST_F(DdaAllReduceFabricLL128Split, StagedAndSplitMatchTheCooperativeWriter) {
  DeviceBuffer<int> staged(1);
  hipLaunchKernelGGL(ReportStaged, dim3(1), dim3(1), 0, 0, staged.ptr);
  HIP_CHECK(hipDeviceSynchronize());
  if (staged.download() == 0) {
    GTEST_SKIP() << "this device pass has no async LDS stores";
  }
  for (const Case& c : kCases) {
    const auto send = FloatMessages(c.nRanks);
    setUp(c.nRanks, c.self, send, FloatReduced(send, c.nRanks));
    const Snapshot want = expectMatchesCoop<float>(c);
    std::vector<uint8_t> expected;
    for (const std::vector<uint8_t>& shard : reduced_) {
      expected.insert(expected.end(), shard.begin(), shard.end());
    }
    ASSERT_EQ(want.recv, expected) << "the cooperative two-shot left the host fold";
  }
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

TEST_F(DdaAllReduceFabricLL128Split, SplitMatchesTheCooperativeWriterOnFp8) {
  const Case cases[] = {{6, 5, 0, Kind::kSplit, 1, 64}, {8, 5, 8, Kind::kSplit, 2, 512}};
  for (const Case& c : cases) {
    setUp(c.nRanks, c.self, Fp8Rows(c.nRanks, (size_t)c.nRanks * kShardBytes, 0), Fp8Rows(c.nRanks, kShardBytes, 5));
    expectMatchesCoop<dda::common::fp8e4m3>(c);
  }
}

// A peer that never sends aborts the split and the staged two-shot: my shard is never stored or published, every async
// store drains, and the epoch still moves on.
TEST_F(DdaAllReduceFabricLL128Split, AbortLeavesNoPartialSum) {
  for (const Case& c : {Case{8, 5, 8, Kind::kSplit, 2, 512}, Case{8, 5, 8, Kind::kStaged, 2, 512}}) {
    const auto send = FloatMessages(c.nRanks);
    setUp(c.nRanks, c.self, send, FloatReduced(send, c.nRanks));
    reset(1, (c.self + 2) % c.nRanks);
    run<float>(c);
    const Snapshot got = snapshot();
    for (size_t i = 0; i < kShardBytes; ++i) {
      ASSERT_EQ(got.recv[(size_t)c.self * kShardBytes + i], kUntouched) << "kind " << (int)c.kind << " byte " << i;
    }
    EXPECT_FALSE(reducedPublished());
    expectEpochAdvanced(c.grid);
  }
}

}  // namespace

}  // namespace RcclUnitTesting
