/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// DDA LL128 AllReduce peer-parallel one-shot and two-shot kernels on one GPU, against the rank-order host fold. A
// helper kernel plays the peers.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "DeviceTestBase.hpp"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll128_peerpar.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting {

namespace {

using dda::common::DdaKernelParams;
using dda::common::kDdaLL128DataBytesPerSlice;
using dda::common::kDdaLL128Warp;
using dda::common::kDdaLL128WireWordsPerSlice;
using dda::common::kDdaLL128WordsPerThread;

// 37 full slices and one 16-byte chunk, so the last slice is all but empty.
constexpr size_t kPartialBytes = 37 * kDdaLL128DataBytesPerSlice + 16;
constexpr size_t kWholeBytes = 40 * kDdaLL128DataBytesPerSlice;
constexpr uint32_t kFlag = 2;  // epoch cells start at 1, so the kernels use flag 2 and bank 0
constexpr uint8_t kUntouched = 0xA5;
constexpr unsigned kMaxGrid = 8;

enum class Kind { kOneShot, kTwoShot };

struct Case {
  Kind kind;
  int nRanks;
  int self;
  int ct;  // NRANKS_CT of the instance: 4, 8 or 0
  unsigned grid;
  unsigned block;
  size_t bytes;  // what one slot holds: the message (one-shot) or one shard (two-shot)
};

const char* name(Kind kind) {
  return kind == Kind::kOneShot ? "one-shot" : "two-shot";
}

size_t slices(size_t bytes) {
  return dda::common::ddaLL128Slices(bytes);
}

// The whole message: one slot for the one-shot, nRanks shards for the two-shot.
size_t messageBytes(const Case& c) {
  return c.kind == Kind::kOneShot ? c.bytes : (size_t)c.nRanks * c.bytes;
}

// A bank of exactly one (one-shot) or two (two-shot) staging areas of nRanks slots.
size_t bankBytes(Kind kind, int nRanks, size_t bytes) {
  const size_t areas = kind == Kind::kOneShot ? 1 : 2;
  return areas * (size_t)nRanks * slices(bytes) * kDdaLL128WireWordsPerSlice * sizeof(uint64_t);
}

size_t slotWords(Kind kind, int nRanks, size_t bytes) {
  const size_t bank = bankBytes(kind, nRanks, bytes);
  return kind == Kind::kOneShot ? dda::common::ddaLL128ArSlotWords(bank, nRanks)
                                : dda::common::ddaLL128ArTwoShotSlotWords(bank, nRanks);
}

// Writes rows[peers[blockIdx.x]] into area + peer * slotWords as the LL128 lines that peer sends: one warp per
// (peer, slice).
__global__ void PublishLines(uint64_t* area, const int8_t* rows, const int* peers, size_t slotWordsArg, size_t bytes) {
  const int peer = peers[blockIdx.x];
  const size_t s = blockIdx.y;
  const int lane = threadIdx.x;
  const bool flagLane = dda::common::ddaLL128IsFlagLane(lane);
  const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
  const size_t rem = bytes - sliceByte;
  const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
  uint64_t regs[kDdaLL128WordsPerThread] = {};
  dda::common::ddaLL128LoadRegs<int8_t>(regs, rows + (size_t)peer * bytes + sliceByte, elt, lane, flagLane);
  const uint64_t flag = ((uint64_t)kFlag << 32) | kFlag;
  dda::common::ddaLL128StoreWire(area + (size_t)peer * slotWordsArg + s * kDdaLL128WireWordsPerSlice + 2 * lane, regs,
                                 flag, flagLane);
}

// Unpacks the lines at slot into out through the poll and unpack a receiving kernel runs: one warp per slice. Run with
// the abort flag raised, so a line without the flag sets *missing instead of waiting.
__global__ void UnpackSlot(const uint64_t* slot, int8_t* out, size_t bytes, uint32_t* abortFlag, int* missing) {
  const size_t s = blockIdx.x;
  const int lane = threadIdx.x;
  const bool flagLane = dda::common::ddaLL128IsFlagLane(lane);
  const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
  const size_t rem = bytes - sliceByte;
  const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
  const uint64_t flag = ((uint64_t)kFlag << 32) | kFlag;
  uint64_t vr[kDdaLL128WordsPerThread];
  if (!dda::common::ddaLL128PollWire(slot + s * kDdaLL128WireWordsPerSlice + 2 * lane, vr, flag, lane,
                                     DdaKernelParams{abortFlag})) {
    if (lane == 0) {
      *missing = 1;
    }
    return;
  }
  dda::common::ddaLL128StoreRegs<int8_t>(out + sliceByte, vr, elt, lane, flagLane);
}

template <typename T, int N>
void launch(const Case& c, T** scratch, T* recv, const T* send, uint32_t* epoch, uint32_t* abortFlag) {
  const DdaKernelParams params{abortFlag};
  const size_t bank = bankBytes(c.kind, c.nRanks, c.bytes);
  const size_t n = slices(c.bytes);
  const size_t lds = dda::common::ddaLL128ArPeerLdsBytes(c.block);
  if (c.kind == Kind::kOneShot) {
    hipLaunchKernelGGL((dda::common::ddaAllReduceFlatLL128PeerPar<T, N>), dim3(c.grid), dim3(c.block), lds, 0, scratch,
                       recv, send, c.bytes, c.self, c.nRanks, epoch, (int)kMaxGrid, n, bank, params);
  } else {
    hipLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL128PeerPar<T, N>), dim3(c.grid), dim3(c.block), lds, 0,
                       scratch, recv, send, c.bytes, c.self, c.nRanks, epoch, (int)kMaxGrid, n, bank, params);
  }
}

class DdaAllReduceFabricLL128PeerPar : public DeviceTestBase {
 protected:
  void SetUp() override {
    DeviceTestBase::SetUp();
    if (!isGfx1250()) {
      GTEST_SKIP() << "the LL128 AllReduce tiers run on gfx1250";
    }
  }

  // send_[r] is rank r's message; reduced_ is the whole message folded in rank order.
  void setUp(const Case& c, const std::vector<std::vector<uint8_t>>& send, const std::vector<uint8_t>& reduced) {
    c_ = c;
    send_ = send;
    reduced_ = reduced;
    scratch_.clear();
    std::vector<uint8_t*> ptrs;
    for (int r = 0; r < c.nRanks; ++r) {
      scratch_.emplace_back(2 * bankBytes(c.kind, c.nRanks, c.bytes));
      ptrs.push_back(scratch_.back().ptr);
    }
    scratchPtrs_ = std::make_unique<DeviceBuffer<uint8_t*>>(ptrs.size());
    scratchPtrs_->copyFrom(ptrs);
    sendBuf_ = std::make_unique<DeviceBuffer<uint8_t>>(messageBytes(c));
    recvBuf_ = std::make_unique<DeviceBuffer<uint8_t>>(messageBytes(c));
  }

  // Zeroed scratch, my message, untouched recvbuff, fresh epoch cells, then the peers' lines: every peer's copy of
  // what I fold but skip's, and for the two-shot every peer's reduced shard.
  void reset(uint32_t abortFlag, int skip = -1) {
    for (DeviceBuffer<uint8_t>& b : scratch_) {
      b.zero();
    }
    sendBuf_->copyFrom(send_[(size_t)c_.self]);
    HIP_CHECK(hipMemset(recvBuf_->ptr, kUntouched, recvBuf_->count));
    epoch_.copyFrom(std::vector<uint32_t>(kMaxGrid, kFlag - 1));
    abort_.upload(abortFlag);

    // The one-shot folds whole messages, the two-shot my shard of each.
    const size_t from = c_.kind == Kind::kOneShot ? 0 : (size_t)c_.self * c_.bytes;
    std::vector<int8_t> copies((size_t)c_.nRanks * c_.bytes);
    for (int p = 0; p < c_.nRanks; ++p) {
      std::memcpy(&copies[(size_t)p * c_.bytes], &send_[(size_t)p][from], c_.bytes);
    }
    uint64_t* self = reinterpret_cast<uint64_t*>(scratch_[(size_t)c_.self].ptr);
    publish(self, copies, skip);
    if (c_.kind == Kind::kTwoShot) {
      std::vector<int8_t> reduced(reduced_.begin(), reduced_.end());
      publish(self + (size_t)c_.nRanks * slotWords(c_.kind, c_.nRanks, c_.bytes), reduced, -1);
    }
  }

  void publish(uint64_t* area, const std::vector<int8_t>& rows, int skip) {
    std::vector<int> peers;
    for (int p = 0; p < c_.nRanks; ++p) {
      if (p != c_.self && p != skip) {
        peers.push_back(p);
      }
    }
    DeviceBuffer<int8_t> dRows(rows.size());
    dRows.copyFrom(rows);
    DeviceBuffer<int> dPeers(peers.size());
    dPeers.copyFrom(peers);
    hipLaunchKernelGGL(PublishLines, dim3((unsigned)peers.size(), (unsigned)slices(c_.bytes)), dim3(kDdaLL128Warp), 0,
                       0, area, dRows.ptr, dPeers.ptr, slotWords(c_.kind, c_.nRanks, c_.bytes), c_.bytes);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  // The kernel of c_, with recvbuff sendbuff when inPlace.
  template <typename T>
  void run(bool inPlace = false) {
    T** scratch = reinterpret_cast<T**>(scratchPtrs_->ptr);
    T* recv = reinterpret_cast<T*>(inPlace ? sendBuf_->ptr : recvBuf_->ptr);
    const T* send = reinterpret_cast<const T*>(sendBuf_->ptr);
    switch (c_.ct) {
      case 4:
        launch<T, 4>(c_, scratch, recv, send, epoch_.ptr, abort_.ptr);
        break;
      case 8:
        launch<T, 8>(c_, scratch, recv, send, epoch_.ptr, abort_.ptr);
        break;
      default:
        launch<T, 0>(c_, scratch, recv, send, epoch_.ptr, abort_.ptr);
        break;
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  std::vector<uint8_t> result(bool inPlace = false) {
    return inPlace ? sendBuf_->copyTo() : recvBuf_->copyTo();
  }

  // What peer unpacks from my slot of `area` in its scratch.
  std::vector<uint8_t> received(int peer, int area) {
    DeviceBuffer<uint8_t> out(c_.bytes);
    out.zero();
    DeviceBuffer<int> missing(1);
    missing.zero();
    DeviceBuffer<uint32_t> raised(1);
    raised.upload(1);
    const uint64_t* slot = reinterpret_cast<const uint64_t*>(scratch_[(size_t)peer].ptr) +
                           (size_t)(area * c_.nRanks + c_.self) * slotWords(c_.kind, c_.nRanks, c_.bytes);
    hipLaunchKernelGGL(UnpackSlot, dim3((unsigned)slices(c_.bytes)), dim3(kDdaLL128Warp), 0, 0, slot,
                       reinterpret_cast<int8_t*>(out.ptr), c_.bytes, raised.ptr, missing.ptr);
    HIP_EXPECT(hipGetLastError());
    HIP_EXPECT(hipDeviceSynchronize());
    EXPECT_EQ(missing.download(), 0) << "a line to peer " << peer << " has no flag";
    return out.copyTo();
  }

  // Every peer unpacks what I owe it: my message (one-shot), or my copy of its shard and then my reduced shard
  // (two-shot).
  void expectSent() {
    for (int p = 0; p < c_.nRanks; ++p) {
      if (p == c_.self) {
        continue;
      }
      const uint8_t* mine = send_[(size_t)c_.self].data();
      const std::vector<uint8_t> got = received(p, 0);
      const uint8_t* want = c_.kind == Kind::kOneShot ? mine : mine + (size_t)p * c_.bytes;
      EXPECT_TRUE(std::equal(got.begin(), got.end(), want)) << "peer " << p << " unpacks the wrong copy";
      if (c_.kind == Kind::kTwoShot) {
        const std::vector<uint8_t> sum = received(p, 1);
        EXPECT_TRUE(std::equal(sum.begin(), sum.end(), reduced_.begin() + (std::ptrdiff_t)((size_t)c_.self * c_.bytes)))
          << "peer " << p << " unpacks the wrong reduced shard";
      }
    }
  }

  // Runs c_ on fresh peer lines: the result is the rank-order fold, and every peer gets its lines.
  template <typename T>
  void expectRankOrderFold(bool inPlace = false) {
    reset(0);
    run<T>(inPlace);
    EXPECT_TRUE(result(inPlace) == reduced_) << "left the rank-order fold";
    expectSent();
    expectEpochAdvanced();
  }

  // The second area slot[self] of every rank, where only a fully reduced shard of mine may land.
  bool reducedPublished() {
    const size_t words = slotWords(c_.kind, c_.nRanks, c_.bytes);
    const size_t at = (size_t)(c_.nRanks + c_.self) * words * sizeof(uint64_t);
    for (const DeviceBuffer<uint8_t>& b : scratch_) {
      const std::vector<uint8_t> bytes = b.copyTo();
      const auto first = bytes.begin() + (std::ptrdiff_t)at;
      if (std::any_of(first, first + (std::ptrdiff_t)(words * sizeof(uint64_t)), [](uint8_t v) { return v != 0; })) {
        return true;
      }
    }
    return false;
  }

  // Every block, aborted or not, bumps the cells it owns, so all kMaxGrid cells end at kFlag.
  void expectEpochAdvanced() {
    const std::vector<uint32_t> epoch = epoch_.copyTo();
    for (unsigned e = 0; e < kMaxGrid; ++e) {
      EXPECT_EQ(epoch[e], kFlag) << "epoch cell " << e;
    }
  }

  Case c_{};
  std::vector<std::vector<uint8_t>> send_;
  std::vector<uint8_t> reduced_;
  std::vector<DeviceBuffer<uint8_t>> scratch_;
  std::unique_ptr<DeviceBuffer<uint8_t*>> scratchPtrs_;
  std::unique_ptr<DeviceBuffer<uint8_t>> sendBuf_;
  std::unique_ptr<DeviceBuffer<uint8_t>> recvBuf_;
  DeviceBuffer<uint32_t> epoch_{kMaxGrid};
  DeviceBuffer<uint32_t> abort_{1};
};

// Inexact float sums pin the fold order.
std::vector<std::vector<uint8_t>> FloatMessages(int nRanks, size_t bytes) {
  std::vector<std::vector<uint8_t>> send((size_t)nRanks, std::vector<uint8_t>(bytes));
  for (int r = 0; r < nRanks; ++r) {
    float* f = reinterpret_cast<float*>(send[(size_t)r].data());
    for (size_t i = 0; i < bytes / sizeof(float); ++i) {
      f[i] = 1.0f / (float)(3 + r) + (float)((r * 131 + (int)i * 17) % 1000) * 1.0e-3f;
    }
  }
  return send;
}

// The message folded in rank order: rank 0's copy, then ranks 1, 2, ...
std::vector<uint8_t> FloatRankOrderFold(const std::vector<std::vector<uint8_t>>& send) {
  std::vector<uint8_t> reduced(send[0].size());
  float* acc = reinterpret_cast<float*>(reduced.data());
  for (size_t i = 0; i < reduced.size() / sizeof(float); ++i) {
    acc[i] = reinterpret_cast<const float*>(send[0].data())[i];
    for (size_t r = 1; r < send.size(); ++r) {
      acc[i] = acc[i] + reinterpret_cast<const float*>(send[r].data())[i];
    }
  }
  return reduced;
}

// bf16 messages of small integers (0..7): every partial sum of up to 36 ranks stays an integer below 256, which
// bf16 holds exactly, so the result is the integer sum whatever the fold order.
std::vector<std::vector<uint8_t>> Bf16IntMessages(int nRanks, size_t bytes) {
  std::vector<std::vector<uint8_t>> send((size_t)nRanks, std::vector<uint8_t>(bytes));
  for (int r = 0; r < nRanks; ++r) {
    for (size_t i = 0; i < bytes / sizeof(bf16); ++i) {
      const bf16 v = __float2bfloat16((float)((r * 3 + (int)i) % 8));
      std::memcpy(&send[(size_t)r][i * sizeof(bf16)], &v, sizeof(bf16));
    }
  }
  return send;
}

// The integer sum of the bf16 messages, as bf16.
std::vector<uint8_t> Bf16Sum(const std::vector<std::vector<uint8_t>>& send) {
  std::vector<uint8_t> sum(send[0].size());
  for (size_t i = 0; i < sum.size() / sizeof(bf16); ++i) {
    float acc = 0.0f;
    for (const std::vector<uint8_t>& msg : send) {
      bf16 v;
      std::memcpy(&v, &msg[i * sizeof(bf16)], sizeof(bf16));
      acc += __bfloat162float(v);
    }
    const bf16 out = __float2bfloat16(acc);
    std::memcpy(&sum[i * sizeof(bf16)], &out, sizeof(bf16));
  }
  return sum;
}

// The trace line of c.
std::string Describe(const Case& c) {
  return std::string(name(c.kind)) + ", " + std::to_string(c.nRanks) + " ranks, self " +
         std::to_string(c.self) + ", ct " + std::to_string(c.ct) + ", " + std::to_string(c.grid) + "x" +
         std::to_string(c.block) + ", " + std::to_string(c.bytes) + " B";
}

// Group layouts the launcher reaches, unrolled and runtime rank counts, and partial or whole last slices.
const Case kCases[] = {
  {Kind::kOneShot, 4, 1, 4, 2, 512, kPartialBytes},   // four 4-warp groups a block
  {Kind::kOneShot, 3, 2, 0, 2, 512, kPartialBytes},   // five 3-warp groups and a leftover warp
  {Kind::kOneShot, 3, 0, 0, 1, 256, kWholeBytes},     // two 3-warp groups and two leftover warps
  {Kind::kOneShot, 8, 5, 8, 3, 512, kPartialBytes},   // two 8-warp groups a block
  {Kind::kOneShot, 16, 9, 0, 2, 512, kPartialBytes},  // one 16-warp group, a warp per rank
  {Kind::kOneShot, 2, 1, 0, 2, 512, kPartialBytes},   // eight 2-warp groups a block
  {Kind::kOneShot, 24, 17, 0, 2, 512, kPartialBytes}, // 16-warp groups fold in rounds of 16 and 8
  {Kind::kOneShot, 33, 32, 0, 2, 512, kPartialBytes}, // rounds of 16, 16 and 1, self alone in the last
  {Kind::kTwoShot, 4, 1, 4, 2, 512, kPartialBytes},
  {Kind::kTwoShot, 3, 2, 0, 2, 512, kPartialBytes},   // the leftover warp still takes phase 1 and 3 items
  {Kind::kTwoShot, 3, 0, 0, 1, 256, kWholeBytes},
  {Kind::kTwoShot, 8, 5, 8, 3, 512, kPartialBytes},
  {Kind::kTwoShot, 16, 9, 0, 2, 512, kPartialBytes},
  {Kind::kTwoShot, 2, 1, 0, 2, 512, kPartialBytes},
  {Kind::kTwoShot, 24, 17, 0, 2, 512, kPartialBytes},
  {Kind::kTwoShot, 33, 5, 0, 2, 512, kPartialBytes},
};

}  // namespace

// Both kernels give the rank-order host fold bit for bit and send every peer its lines.
TEST_F(DdaAllReduceFabricLL128PeerPar, MatchesTheRankOrderFoldAndSendsEveryPeer) {
  for (const Case& c : kCases) {
    SCOPED_TRACE(Describe(c));
    const auto send = FloatMessages(c.nRanks, messageBytes(c));
    setUp(c, send, FloatRankOrderFold(send));
    expectRankOrderFold<float>();
  }
}

// In place, recvbuff is sendbuff: every slice is sent and folded before the same warps write it back.
TEST_F(DdaAllReduceFabricLL128PeerPar, InPlaceMatchesTheRankOrderFold) {
  for (const Case& c : {Case{Kind::kOneShot, 3, 2, 0, 2, 512, kPartialBytes},
                        Case{Kind::kOneShot, 8, 5, 8, 3, 512, kPartialBytes},
                        Case{Kind::kTwoShot, 3, 2, 0, 2, 512, kPartialBytes},
                        Case{Kind::kTwoShot, 8, 5, 8, 3, 512, kPartialBytes}}) {
    SCOPED_TRACE(Describe(c));
    const auto send = FloatMessages(c.nRanks, messageBytes(c));
    setUp(c, send, FloatRankOrderFold(send));
    expectRankOrderFold<float>(/*inPlace=*/true);
  }
}

// bf16 folds each 4B half of a word as two bf16 values, unrolled (8 ranks) and in rounds (33 ranks).
TEST_F(DdaAllReduceFabricLL128PeerPar, Bf16MatchesTheExactSum) {
  for (const Case& c : {Case{Kind::kOneShot, 8, 5, 8, 3, 512, kPartialBytes},
                        Case{Kind::kOneShot, 33, 32, 0, 2, 512, kPartialBytes},
                        Case{Kind::kTwoShot, 8, 5, 8, 3, 512, kPartialBytes},
                        Case{Kind::kTwoShot, 33, 5, 0, 2, 512, kPartialBytes}}) {
    SCOPED_TRACE(Describe(c));
    const auto send = Bf16IntMessages(c.nRanks, messageBytes(c));
    setUp(c, send, Bf16Sum(send));
    expectRankOrderFold<bf16>();
  }
}

// A peer that never sends aborts the fold: my result is never stored (one-shot: none of recvbuff; two-shot: my shard,
// never published either), and the epoch still moves on.
TEST_F(DdaAllReduceFabricLL128PeerPar, AbortLeavesNoPartialSum) {
  for (const Case& c : {Case{Kind::kOneShot, 3, 1, 0, 2, 512, kPartialBytes},
                        Case{Kind::kOneShot, 8, 5, 8, 3, 512, kPartialBytes},
                        Case{Kind::kTwoShot, 3, 1, 0, 2, 512, kPartialBytes},
                        Case{Kind::kTwoShot, 8, 5, 8, 3, 512, kPartialBytes}}) {
    SCOPED_TRACE(Describe(c));
    const auto send = FloatMessages(c.nRanks, messageBytes(c));
    setUp(c, send, FloatRankOrderFold(send));
    reset(1, (c.self + 2) % c.nRanks);
    run<float>();
    const std::vector<uint8_t> got = result();
    const size_t from = c.kind == Kind::kOneShot ? 0 : (size_t)c.self * c.bytes;
    for (size_t i = from; i < from + c.bytes; ++i) {
      ASSERT_EQ(got[i], kUntouched) << "byte " << i;
    }
    if (c.kind == Kind::kTwoShot) {
      EXPECT_FALSE(reducedPublished());
    }
    expectEpochAdvanced();
  }
}

}  // namespace RcclUnitTesting
