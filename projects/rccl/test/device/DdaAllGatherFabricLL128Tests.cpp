/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// DDA LL128 AllGather one-shot on one GPU, a helper kernel playing the peers: the default and staged kernels must
// gather the host reference.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "DeviceTestBase.hpp"
#include "algorithms/dda/all_gather/all_gather_dda_fabric_ll128.h"

namespace RcclUnitTesting {

namespace {

using dda::common::DdaKernelParams;
using dda::common::kDdaLL128AGStageMaxPushPeers;
using dda::common::kDdaLL128AGStagePushPeers;
using dda::common::kDdaLL128DataBytesPerSlice;
using dda::common::kDdaLL128FlagElem;
using dda::common::kDdaLL128LineElems;
using dda::common::kDdaLL128Warp;
using dda::common::kDdaLL128WireWordsPerSlice;
using dda::common::kDdaLL128WordsPerThread;

// Two full slices and a partial third.
constexpr size_t kSmallBytes = 2 * kDdaLL128DataBytesPerSlice + 112;
// 40 slices and a partial one: a 2 x 256 column runs several slices per warp.
constexpr size_t kMediumBytes = 40 * kDdaLL128DataBytesPerSlice + 112;
// 300 slices and a partial one: a 3 x 512 column runs ~6 slices per warp.
constexpr size_t kLargeBytes = 300 * kDdaLL128DataBytesPerSlice + 16;
// 2000 slices and a partial one: on 1 x 128 columns a warp runs hundreds of fills.
constexpr size_t kStressBytes = 2000 * kDdaLL128DataBytesPerSlice + 48;
constexpr uint32_t kFlag = 2;  // epoch cells start at 1, so the kernel uses flag 2 and bank 0; flag 3 uses bank 1
constexpr uint8_t kUntouched = 0xA5;
constexpr size_t kTailBytes = 64;  // canary past recvbuff's last slot

struct Case {
  int nRanks;
  int self;
  int ct;  // NRANKS_CT of the default instance: 4, 8 or 0
  unsigned blocksPerPeer;
  unsigned block;
  size_t perRankBytes;
  bool inPlace;
};

size_t slices(size_t perRankBytes) {
  return dda::common::ddaLL128Slices(perRankBytes);
}

// A bank of exactly nRanks slots, each slot one rank's slices.
size_t bankBytes(int nRanks, size_t perRankBytes) {
  return (size_t)nRanks * slices(perRankBytes) * kDdaLL128WireWordsPerSlice * sizeof(uint64_t);
}

size_t slotWords(int nRanks, size_t perRankBytes) {
  return dda::common::ddaLL128AgSlotWords(bankBytes(nRanks, perRankBytes), nRanks);
}

// Writes peers[blockIdx.x]'s LL128 lines into its slot of `area`, a warp a slice, via the per-lane load (not the
// batched one under test).
__global__ void PublishLines(uint64_t* area, const int8_t* rows, const int* peers, size_t slotWordsArg,
                             size_t perRankBytes, uint64_t wireFlag) {
  const int peer = peers[blockIdx.x];
  const size_t s = blockIdx.y;
  const int lane = threadIdx.x;
  const bool flagLane = dda::common::ddaLL128IsFlagLane(lane);
  const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
  const size_t rem = perRankBytes - sliceByte;
  const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
  uint64_t regs[kDdaLL128WordsPerThread] = {};
  dda::common::ddaLL128LoadRegs<int8_t>(regs, rows + (size_t)peer * perRankBytes + sliceByte, elt, lane, flagLane);
  dda::common::ddaLL128StoreWire(area + (size_t)peer * slotWordsArg + s * kDdaLL128WireWordsPerSlice + 2 * lane, regs,
                                 wireFlag, flagLane);
}

__global__ void ReportStaged(int* out) {
  *out = dda::common::ddaLL128Staged<dda::common::kDdaLL128WireAsync>() ? 1 : 0;
}

template <int N>
void launchDefault(const Case& c, int8_t** scratch, int8_t* recv, const int8_t* send, uint32_t* epoch,
                   uint32_t* abortFlag) {
  const DdaKernelParams params{abortFlag};
  const dim3 grid((unsigned)(c.nRanks - 1), c.blocksPerPeer);
  hipLaunchKernelGGL((dda::common::ddaAllGatherFabricLL128<int8_t, N>), grid, dim3(c.block), 0, 0, scratch, recv,
                     send, c.perRankBytes, c.self, c.nRanks, epoch, (int)(grid.x * grid.y), slices(c.perRankBytes),
                     bankBytes(c.nRanks, c.perRankBytes), params);
}

// The staged kernel on the default grid, with the launcher's LDS: two windows per warp.
void launchStaged(const Case& c, int pushPeers, int8_t** scratch, int8_t* recv, const int8_t* send, uint32_t* epoch,
                  uint32_t* abortFlag) {
  const DdaKernelParams params{abortFlag};
  const dim3 grid((unsigned)(c.nRanks - 1), c.blocksPerPeer);
  hipLaunchKernelGGL((dda::common::ddaAllGatherFabricLL128Staged<int8_t>), grid, dim3(c.block),
                     dda::common::ddaLL128StageBytes(c.block), 0, scratch, recv, send, c.perRankBytes, c.self,
                     c.nRanks, epoch, (int)(grid.x * grid.y), slices(c.perRankBytes),
                     bankBytes(c.nRanks, c.perRankBytes), params, pushPeers);
}

// What one launch leaves behind: recvbuff and every peer's scratch (self's own scratch is the kernel's input).
struct Snapshot {
  std::vector<uint8_t> recv;
  std::vector<std::vector<uint8_t>> peerScratch;
};

class DdaAllGatherFabricLL128 : public DeviceTestBase {
 protected:
  void SetUp() override {
    DeviceTestBase::SetUp();
    if (!isGfx1250()) {
      GTEST_SKIP() << "the LL128 AllGather tier runs on gfx1250";
    }
  }

  // Whether the device pass has async LDS stores; without them the staged kernel is the default one.
  bool stagedBuilt() {
    DeviceBuffer<int> staged(1);
    hipLaunchKernelGGL(ReportStaged, dim3(1), dim3(1), 0, 0, staged.ptr);
    HIP_EXPECT(hipDeviceSynchronize());
    return staged.download() != 0;
  }

  // Rank r's payload: index- and rank-dependent bytes, so a misplaced 16B chunk or slot shows.
  void prepare(const Case& c) {
    c_ = c;
    send_.assign((size_t)c.nRanks, std::vector<uint8_t>(c.perRankBytes));
    for (int r = 0; r < c.nRanks; ++r) {
      for (size_t i = 0; i < c.perRankBytes; ++i) {
        send_[(size_t)r][i] = (uint8_t)((r * 131 + (int)(i * 7) + (int)(i >> 9)) & 0xff);
      }
    }
    scratch_.clear();
    std::vector<int8_t*> ptrs;
    for (int r = 0; r < c.nRanks; ++r) {
      scratch_.emplace_back(2 * bankBytes(c.nRanks, c.perRankBytes));
      ptrs.push_back(reinterpret_cast<int8_t*>(scratch_.back().ptr));
    }
    scratchPtrs_ = std::make_unique<DeviceBuffer<int8_t*>>(ptrs.size());
    scratchPtrs_->copyFrom(ptrs);
    recvBuf_ = std::make_unique<DeviceBuffer<uint8_t>>((size_t)c.nRanks * c.perRankBytes + kTailBytes);
    sendBuf_ = std::make_unique<DeviceBuffer<uint8_t>>(c.perRankBytes);
  }

  // Zeroed scratch, untouched recvbuff (my slot preloaded in place), fresh epochs, then the peers' lines; aborted_:
  // no lines and a raised abort flag.
  void reset() {
    for (DeviceBuffer<uint8_t>& b : scratch_) {
      b.zero();
    }
    HIP_CHECK(hipMemset(recvBuf_->ptr, kUntouched, recvBuf_->count));
    sendBuf_->copyFrom(send_[(size_t)c_.self]);
    if (c_.inPlace) {
      HIP_CHECK(hipMemcpy(recvBuf_->ptr + (size_t)c_.self * c_.perRankBytes, send_[(size_t)c_.self].data(),
                          c_.perRankBytes, hipMemcpyHostToDevice));
    }
    epoch_.copyFrom(std::vector<uint32_t>(kMaxGrid, flag_ - 1));
    abort_.upload(aborted_ ? 1 : 0);
    if (aborted_) {
      return;
    }

    std::vector<int8_t> rows((size_t)c_.nRanks * c_.perRankBytes);
    std::vector<int> peers;
    for (int p = 0; p < c_.nRanks; ++p) {
      std::memcpy(&rows[(size_t)p * c_.perRankBytes], send_[(size_t)p].data(), c_.perRankBytes);
      if (p != c_.self) {
        peers.push_back(p);
      }
    }
    DeviceBuffer<int8_t> dRows(rows.size());
    dRows.copyFrom(rows);
    DeviceBuffer<int> dPeers(peers.size());
    dPeers.copyFrom(peers);
    hipLaunchKernelGGL(PublishLines, dim3((unsigned)peers.size(), (unsigned)slices(c_.perRankBytes)),
                       dim3(kDdaLL128Warp), 0, 0, bankBase(scratch_[(size_t)c_.self].ptr), dRows.ptr, dPeers.ptr,
                       slotWords(c_.nRanks, c_.perRankBytes), c_.perRankBytes, wireFlag());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  // pushPeers 0: the default kernel; else the staged one with that many peers a fill.
  Snapshot run(int pushPeers = 0) {
    reset();
    int8_t** scratch = scratchPtrs_->ptr;
    int8_t* recv = reinterpret_cast<int8_t*>(recvBuf_->ptr);
    const int8_t* send =
      c_.inPlace ? recv + (size_t)c_.self * c_.perRankBytes : reinterpret_cast<const int8_t*>(sendBuf_->ptr);
    if (pushPeers > 0) {
      launchStaged(c_, pushPeers, scratch, recv, send, epoch_.ptr, abort_.ptr);
    } else {
      switch (c_.ct) {
        case 4:
          launchDefault<4>(c_, scratch, recv, send, epoch_.ptr, abort_.ptr);
          break;
        case 8:
          launchDefault<8>(c_, scratch, recv, send, epoch_.ptr, abort_.ptr);
          break;
        default:
          launchDefault<0>(c_, scratch, recv, send, epoch_.ptr, abort_.ptr);
          break;
      }
    }
    HIP_EXPECT(hipGetLastError());
    HIP_EXPECT(hipDeviceSynchronize());
    expectEpochAdvanced();
    Snapshot snap{recvBuf_->copyTo(), {}};
    for (int p = 0; p < c_.nRanks; ++p) {
      snap.peerScratch.push_back(p == c_.self ? std::vector<uint8_t>() : scratch_[(size_t)p].copyTo());
    }
    return snap;
  }

  // recvbuff holds every rank's payload in rank order and nothing past it; aborted, only mine.
  void expectGathered(const Snapshot& snap) {
    std::vector<uint8_t> want;
    for (int r = 0; r < c_.nRanks; ++r) {
      const std::vector<uint8_t>& row = send_[(size_t)r];
      if (aborted_ && r != c_.self) {
        want.insert(want.end(), row.size(), kUntouched);
      } else {
        want.insert(want.end(), row.begin(), row.end());
      }
    }
    want.insert(want.end(), kTailBytes, kUntouched);
    ASSERT_EQ(snap.recv.size(), want.size());
    const size_t at = (size_t)(std::mismatch(want.begin(), want.end(), snap.recv.begin()).first - want.begin());
    EXPECT_EQ(at, want.size()) << "first wrong recvbuff byte " << at << " (rank " << at / c_.perRankBytes << ")";
  }

  // The lines this rank sends, from PublishLines' per-lane loads: an oracle apart from the batched loads under test.
  std::vector<uint64_t> myLines() {
    DeviceBuffer<uint64_t> area(slices(c_.perRankBytes) * kDdaLL128WireWordsPerSlice);
    DeviceBuffer<uint8_t> row(c_.perRankBytes);
    row.copyFrom(send_[(size_t)c_.self]);
    DeviceBuffer<int> first(1);
    first.upload(0);
    hipLaunchKernelGGL(PublishLines, dim3(1, (unsigned)slices(c_.perRankBytes)), dim3(kDdaLL128Warp), 0, 0, area.ptr,
                       reinterpret_cast<const int8_t*>(row.ptr), first.ptr, size_t{0}, c_.perRankBytes, wireFlag());
    HIP_EXPECT(hipGetLastError());
    HIP_EXPECT(hipDeviceSynchronize());
    return area.copyTo();
  }

  // My slot in every peer's scratch: my lines in whole slices; only flags in the partial one, whose padding the
  // batched load fills with in-bounds data.
  void expectPushed(const Snapshot& snap) {
    const std::vector<uint64_t> want = myLines();
    const size_t whole = c_.perRankBytes / kDdaLL128DataBytesPerSlice * kDdaLL128WireWordsPerSlice;
    const size_t words = slotWords(c_.nRanks, c_.perRankBytes);
    for (int p = 0; p < c_.nRanks; ++p) {
      if (p == c_.self) {
        continue;
      }
      const uint64_t* slot = bankBase(snap.peerScratch[(size_t)p].data()) + (size_t)c_.self * words;
      const size_t at = (size_t)(std::mismatch(want.begin(), want.begin() + whole, slot).first - want.begin());
      EXPECT_EQ(at, whole) << "peer " << p << " first wrong line " << at / kDdaLL128LineElems;
      for (size_t w = whole + kDdaLL128FlagElem; w < want.size(); w += kDdaLL128LineElems) {
        EXPECT_EQ(slot[w], wireFlag()) << "peer " << p << " line " << w / kDdaLL128LineElems;
      }
    }
  }

  uint64_t wireFlag() const { return ((uint64_t)flag_ << 32) | flag_; }

  // Start of the bank the flag's parity selects in a scratch buffer.
  size_t bankWords() const { return (flag_ & 1u) * bankBytes(c_.nRanks, c_.perRankBytes) / sizeof(uint64_t); }
  uint64_t* bankBase(uint8_t* scratch) const { return reinterpret_cast<uint64_t*>(scratch) + bankWords(); }
  const uint64_t* bankBase(const uint8_t* scratch) const {
    return reinterpret_cast<const uint64_t*>(scratch) + bankWords();
  }

  void expectEpochAdvanced() {
    const std::vector<uint32_t> epoch = epoch_.copyTo();
    const unsigned total = (unsigned)(c_.nRanks - 1) * c_.blocksPerPeer;
    for (unsigned b = 0; b < total; ++b) {
      EXPECT_EQ(epoch[b], flag_) << "block " << b;
    }
  }

  static constexpr size_t kMaxGrid = 16;
  Case c_{};
  uint32_t flag_ = kFlag;
  bool aborted_ = false;
  std::vector<std::vector<uint8_t>> send_;
  std::vector<DeviceBuffer<uint8_t>> scratch_;
  std::unique_ptr<DeviceBuffer<int8_t*>> scratchPtrs_;
  std::unique_ptr<DeviceBuffer<uint8_t>> sendBuf_;
  std::unique_ptr<DeviceBuffer<uint8_t>> recvBuf_;
  DeviceBuffer<uint32_t> epoch_{kMaxGrid};
  DeviceBuffer<uint32_t> abort_{1};
};

// 4 and 8 ranks (unrolled), 6 and 16 (runtime), 1 to 3 blocks a peer at 128 to 512 threads, a partial last slice,
// out of place and in place.
const Case kCases[] = {
  {4, 1, 4, 1, 128, kSmallBytes, false},  {4, 3, 4, 1, 128, kSmallBytes, true},
  {4, 2, 4, 2, 512, kLargeBytes, false},  {8, 5, 8, 2, 256, kMediumBytes, false},
  {8, 0, 8, 2, 256, kMediumBytes, true},  {6, 5, 0, 3, 512, kLargeBytes, false},
  {6, 2, 0, 3, 512, kLargeBytes, true},   {16, 9, 0, 1, 512, kMediumBytes, false},
};

// Both kernels on both banks match the host reference, staged at 1 to 4 and 15 peers a fill. Without async LDS
// stores it fails after the default checks.
TEST_F(DdaAllGatherFabricLL128, AsyncWireMatchesDefaultAndHostReference) {
  const bool stagedOk = stagedBuilt();
  EXPECT_TRUE(stagedOk) << "the gfx1250 device pass has no async LDS stores";
  for (const Case& c : kCases) {
    for (uint32_t flag : {kFlag, kFlag + 1}) {
      flag_ = flag;
      prepare(c);
      SCOPED_TRACE(testing::Message() << c.nRanks << "r self " << c.self << " ct " << c.ct << " bytes "
                                      << c.perRankBytes << " " << (c.nRanks - 1) << "x" << c.blocksPerPeer << "x"
                                      << c.block << " bank " << (flag & 1u) << (c.inPlace ? " in place" : " out"));
      const Snapshot def = run();
      expectGathered(def);
      expectPushed(def);
      if (!stagedOk) {
        continue;
      }
      for (int pushPeers : {1, 2, 3, kDdaLL128AGStagePushPeers, kDdaLL128AGStageMaxPushPeers}) {
        SCOPED_TRACE(testing::Message() << "async wire, " << pushPeers << " peers a fill, "
                                        << dda::common::ddaLL128AGStageGroups(c.nRanks - 1, pushPeers) << " groups");
        const Snapshot staged = run(pushPeers);
        expectGathered(staged);
        expectPushed(staged);
        for (int p = 0; p < c.nRanks; ++p) {
          EXPECT_TRUE(staged.peerScratch[(size_t)p] == def.peerScratch[(size_t)p]) << "peer " << p << " lines differ";
        }
      }
    }
  }
}

// Hundreds of fills a warp (1 x 128 columns, 2001 slices), four launches on both banks: catches a lost wait or drain.
TEST_F(DdaAllGatherFabricLL128, AsyncWireStress) {
  ASSERT_TRUE(stagedBuilt()) << "the gfx1250 device pass has no async LDS stores";
  prepare(Case{4, 2, 4, 1, 128, kStressBytes, false});
  for (int launch = 0; launch < 4; ++launch) {
    flag_ = kFlag + (uint32_t)(launch & 1);
    for (int pushPeers : {1, kDdaLL128AGStagePushPeers}) {
      SCOPED_TRACE(testing::Message() << "launch " << launch << ", " << pushPeers << " peers a fill");
      const Snapshot staged = run(pushPeers);
      expectGathered(staged);
      expectPushed(staged);
    }
  }
}

// Aborted with no peer lines: the staged kernel pushes my lines, copies my slot, drains and publishes its epoch.
TEST_F(DdaAllGatherFabricLL128, AsyncWireAbortLeavesPeerSlotsUntouched) {
  ASSERT_TRUE(stagedBuilt()) << "the gfx1250 device pass has no async LDS stores";
  aborted_ = true;
  prepare(Case{6, 5, 0, 3, 512, kLargeBytes, false});
  const Snapshot got = run(kDdaLL128AGStagePushPeers);
  expectGathered(got);
  expectPushed(got);
}

}  // namespace

}  // namespace RcclUnitTesting
