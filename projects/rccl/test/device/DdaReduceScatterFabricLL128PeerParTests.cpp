/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// DDA LL128 ReduceScatter peer-parallel kernel on one GPU, against the batched kernel and the host fold. A helper
// kernel plays the peers.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "DeviceTestBase.hpp"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll128.h"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll128_peerpar.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting {

namespace {

using dda::common::DdaKernelParams;
using dda::common::kDdaLL128DataBytesPerSlice;
using dda::common::kDdaLL128Warp;
using dda::common::kDdaLL128WireWordsPerSlice;
using dda::common::kDdaLL128WordsPerThread;

// 37 full slices and one 16-byte chunk, so the last slice is all but empty.
constexpr size_t kPartialShardBytes = 37 * kDdaLL128DataBytesPerSlice + 16;
constexpr size_t kWholeShardBytes = 40 * kDdaLL128DataBytesPerSlice;
constexpr uint32_t kFlag = 2;  // epoch cells start at 1, so the kernels use flag 2 and bank 0
constexpr uint8_t kUntouched = 0xA5;
constexpr size_t kGuardBytes = 256;  // after recvbuff, so a store past the shard lands here
constexpr size_t kOffsetRoom = 16;   // room for a user buffer at any byte offset up to a 16B chunk
constexpr unsigned kMaxGrid = 8;
constexpr unsigned kBatchedThreads = 256;  // the batched kernel's SPX block: eight slices a block

struct Case {
  int nRanks;
  int self;
  unsigned grid;  // the peer-parallel launch
  unsigned block;
  size_t shardBytes;
};

size_t slices(size_t shardBytes) {
  return dda::common::ddaLL128Slices(shardBytes);
}

// A bank of nRanks slots, each one shard's slices.
size_t bankBytes(int nRanks, size_t shardBytes) {
  return (size_t)nRanks * slices(shardBytes) * kDdaLL128WireWordsPerSlice * sizeof(uint64_t);
}

size_t slotWords(int nRanks, size_t shardBytes) {
  return dda::common::ddaLL128RsSlotWords(bankBytes(nRanks, shardBytes), nRanks);
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

// Unpacks the lines at slot into out through the poll and unpack a receiving kernel runs: one warp per slice. Run with
// the abort flag raised, so a line without the flag sets *missing instead of waiting.
__global__ void UnpackSlot(const uint64_t* slot, int8_t* out, size_t shardBytes, uint32_t* abortFlag, int* missing) {
  const size_t s = blockIdx.x;
  const int lane = threadIdx.x;
  const bool flagLane = dda::common::ddaLL128IsFlagLane(lane);
  const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
  const size_t rem = shardBytes - sliceByte;
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

template <typename T, bool kAligned>
void launch(bool peerPar, const Case& c, unsigned grid, unsigned block, T** scratch, T* recv, const T* send,
            uint32_t* epoch, uint32_t* abortFlag) {
  const DdaKernelParams params{abortFlag};
  const size_t bank = bankBytes(c.nRanks, c.shardBytes);
  const size_t n = slices(c.shardBytes);
  if (peerPar) {
    hipLaunchKernelGGL((dda::common::ddaReduceScatterFabricLL128PeerPar<T, 0, kAligned>), dim3(grid), dim3(block), 0, 0,
                       scratch, recv, send, c.shardBytes, c.self, c.nRanks, epoch, (int)kMaxGrid, n, bank, params);
  } else {
    hipLaunchKernelGGL((dda::common::ddaReduceScatterFabricLL128<T, 0, kAligned>), dim3(grid), dim3(block), 0, 0,
                       scratch, recv, send, c.shardBytes, c.self, c.nRanks, epoch, (int)kMaxGrid, n, bank, params);
  }
}

class DdaReduceScatterFabricLL128PeerPar : public DeviceTestBase {
 protected:
  void SetUp() override {
    DeviceTestBase::SetUp();
    if (!isGfx1250()) {
      GTEST_SKIP() << "the LL128 ReduceScatter tier runs on gfx1250";
    }
  }

  // send_[r] is rank r's message of nRanks shards. My sendbuff and recvbuff start `offset` bytes into their
  // allocations.
  void setUp(int nRanks, int self, size_t shardBytes, const std::vector<std::vector<uint8_t>>& send,
             size_t offset = 0) {
    nRanks_ = nRanks;
    offset_ = offset;
    self_ = self;
    shardBytes_ = shardBytes;
    send_ = send;
    scratch_.clear();
    std::vector<uint8_t*> ptrs;
    for (int r = 0; r < nRanks; ++r) {
      scratch_.emplace_back(2 * bankBytes(nRanks, shardBytes));
      ptrs.push_back(scratch_.back().ptr);
    }
    scratchPtrs_ = std::make_unique<DeviceBuffer<uint8_t*>>(ptrs.size());
    scratchPtrs_->copyFrom(ptrs);
    sendBuf_ = std::make_unique<DeviceBuffer<uint8_t>>((size_t)nRanks * shardBytes + kOffsetRoom);
    recvBuf_ = std::make_unique<DeviceBuffer<uint8_t>>(shardBytes + kGuardBytes + kOffsetRoom);
  }

  // Zeroed scratch, my message, untouched recvbuff, fresh epoch cells, then every peer's copy of my shard but skip's.
  void reset(uint32_t abortFlag, int skip = -1) {
    for (DeviceBuffer<uint8_t>& b : scratch_) {
      b.zero();
    }
    HIP_CHECK(hipMemset(sendBuf_->ptr, kUntouched, sendBuf_->count));
    HIP_CHECK(hipMemcpy(sendBuf_->ptr + offset_, send_[(size_t)self_].data(), send_[(size_t)self_].size(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(recvBuf_->ptr, kUntouched, recvBuf_->count));
    epoch_.copyFrom(std::vector<uint32_t>(kMaxGrid, kFlag - 1));
    abort_.upload(abortFlag);

    std::vector<int8_t> copies((size_t)nRanks_ * shardBytes_);
    std::vector<int> peers;
    for (int p = 0; p < nRanks_; ++p) {
      std::memcpy(&copies[(size_t)p * shardBytes_], &send_[(size_t)p][(size_t)self_ * shardBytes_], shardBytes_);
      if (p != self_ && p != skip) {
        peers.push_back(p);
      }
    }
    DeviceBuffer<int8_t> dRows(copies.size());
    dRows.copyFrom(copies);
    DeviceBuffer<int> dPeers(peers.size());
    dPeers.copyFrom(peers);
    hipLaunchKernelGGL(PublishLines, dim3((unsigned)peers.size(), (unsigned)slices(shardBytes_)), dim3(kDdaLL128Warp),
                       0, 0, reinterpret_cast<uint64_t*>(scratch_[(size_t)self_].ptr), dRows.ptr, dPeers.ptr,
                       slotWords(nRanks_, shardBytes_), shardBytes_);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  // The batched kernel on its launcher's grid, or the peer-parallel one on c's, with recvbuff my shard of sendbuff
  // when inPlace, and the user-buffer access the launcher picks for them.
  template <typename T>
  void run(bool peerPar, const Case& c, bool inPlace = false) {
    constexpr unsigned kBatchedSlices = kBatchedThreads / kDdaLL128Warp;
    const unsigned grid = peerPar ? c.grid : (unsigned)((slices(c.shardBytes) + kBatchedSlices - 1) / kBatchedSlices);
    const unsigned block = peerPar ? c.block : kBatchedThreads;
    uint8_t* send = sendBuf_->ptr + offset_;
    uint8_t* recv = inPlace ? send + (size_t)self_ * shardBytes_ : recvBuf_->ptr + offset_;
    const auto go = [&](auto aligned) {
      launch<T, decltype(aligned)::value>(peerPar, c, grid, block, reinterpret_cast<T**>(scratchPtrs_->ptr),
                                          reinterpret_cast<T*>(recv), reinterpret_cast<const T*>(send), epoch_.ptr,
                                          abort_.ptr);
    };
    if (offset_ % 16 == 0) {
      go(std::true_type{});
    } else {
      go(std::false_type{});
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  // My shard of the result. Also expects nothing written past it: the guard after recvbuff, or in place the other
  // shards of sendbuff.
  std::vector<uint8_t> recvBytes(bool inPlace = false) {
    const size_t at = offset_ + (inPlace ? (size_t)self_ * shardBytes_ : 0);
    const size_t sendEnd = offset_ + send_[(size_t)self_].size();
    const std::vector<uint8_t> buf = inPlace ? sendBuf_->copyTo() : recvBuf_->copyTo();
    for (size_t i = 0; i < buf.size(); ++i) {
      if (i < at || i >= at + shardBytes_) {
        const bool mine = inPlace && i >= offset_ && i < sendEnd;
        const uint8_t want = mine ? send_[(size_t)self_][i - offset_] : kUntouched;
        EXPECT_EQ(buf[i], want) << "a store landed outside my shard at byte " << i;
        if (buf[i] != want) {
          break;
        }
      }
    }
    return std::vector<uint8_t>(buf.begin() + (std::ptrdiff_t)at, buf.begin() + (std::ptrdiff_t)(at + shardBytes_));
  }

  // What peer unpacks from my slot in its scratch.
  std::vector<uint8_t> received(int peer) {
    DeviceBuffer<uint8_t> out(shardBytes_);
    out.zero();
    DeviceBuffer<int> missing(1);
    missing.zero();
    DeviceBuffer<uint32_t> raised(1);
    raised.upload(1);
    const uint64_t* slot =
      reinterpret_cast<const uint64_t*>(scratch_[(size_t)peer].ptr) + (size_t)self_ * slotWords(nRanks_, shardBytes_);
    hipLaunchKernelGGL(UnpackSlot, dim3((unsigned)slices(shardBytes_)), dim3(kDdaLL128Warp), 0, 0, slot,
                       reinterpret_cast<int8_t*>(out.ptr), shardBytes_, raised.ptr, missing.ptr);
    HIP_EXPECT(hipGetLastError());
    HIP_EXPECT(hipDeviceSynchronize());
    EXPECT_EQ(missing.download(), 0) << "a line to peer " << peer << " has no flag";
    return out.copyTo();
  }

  // Every peer unpacks exactly my shard for it.
  void expectSent(const char* kernel) {
    for (int p = 0; p < nRanks_; ++p) {
      if (p != self_) {
        const std::vector<uint8_t> got = received(p);
        const uint8_t* want = send_[(size_t)self_].data() + (size_t)p * shardBytes_;
        EXPECT_TRUE(std::equal(got.begin(), got.end(), want)) << "peer " << p << " unpacks the wrong shard from the "
                                                              << kernel << " kernel";
      }
    }
  }

  // The batched kernel, then the peer-parallel one on the same peer lines: each sends every peer its shard, and at
  // rank 0, where both fold ranks 0, 1, 2, ..., they leave the same recvbuff. Returns the peer-parallel result.
  template <typename T>
  std::vector<uint8_t> expectMatchesBatched(const Case& c, bool inPlace = false) {
    reset(0);
    run<T>(false, c);
    const std::vector<uint8_t> want = recvBytes();
    expectSent("batched");
    reset(0);
    run<T>(true, c, inPlace);
    const std::vector<uint8_t> got = recvBytes(inPlace);
    expectSent("peer-parallel");
    if (c.self == 0) {
      EXPECT_TRUE(got == want) << "recvbuff differs from the batched kernel's";
    }
    expectEpochAdvanced();
    return got;
  }

  // Every block, aborted or not, bumps the cells it owns, so all kMaxGrid cells end at kFlag.
  void expectEpochAdvanced() {
    const std::vector<uint32_t> epoch = epoch_.copyTo();
    for (unsigned e = 0; e < kMaxGrid; ++e) {
      EXPECT_EQ(epoch[e], kFlag) << "epoch cell " << e;
    }
  }

  int nRanks_ = 0;
  int self_ = 0;
  size_t offset_ = 0;
  size_t shardBytes_ = 0;
  std::vector<std::vector<uint8_t>> send_;
  std::vector<DeviceBuffer<uint8_t>> scratch_;
  std::unique_ptr<DeviceBuffer<uint8_t*>> scratchPtrs_;
  std::unique_ptr<DeviceBuffer<uint8_t>> sendBuf_;
  std::unique_ptr<DeviceBuffer<uint8_t>> recvBuf_;
  DeviceBuffer<uint32_t> epoch_{kMaxGrid};
  DeviceBuffer<uint32_t> abort_{1};
};

// Inexact float sums pin the fold order.
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

// Shard self folded in rank order: rank 0's copy, then ranks 1, 2, ...
std::vector<uint8_t> FloatRankOrderFold(const std::vector<std::vector<uint8_t>>& send, int nRanks, int self,
                                        size_t shardBytes) {
  std::vector<uint8_t> reduced(shardBytes);
  float* acc = reinterpret_cast<float*>(reduced.data());
  for (size_t i = 0; i < shardBytes / sizeof(float); ++i) {
    const size_t at = (size_t)self * shardBytes / sizeof(float) + i;
    acc[i] = reinterpret_cast<const float*>(send[0].data())[at];
    for (int r = 1; r < nRanks; ++r) {
      acc[i] = acc[i] + reinterpret_cast<const float*>(send[(size_t)r].data())[at];
    }
  }
  return reduced;
}

// fp8 e4m3 positive normals up to 1.0 (0x08..0x38): an 8-rank sum stays finite, so the bytes compare exactly.
std::vector<std::vector<uint8_t>> Fp8Messages(int nRanks, size_t bytes) {
  std::vector<std::vector<uint8_t>> out((size_t)nRanks, std::vector<uint8_t>(bytes));
  for (int r = 0; r < nRanks; ++r) {
    for (size_t i = 0; i < bytes; ++i) {
      out[(size_t)r][i] = (uint8_t)(0x08 + (r * 37 + (int)i * 11) % 0x31);
    }
  }
  return out;
}

// Group sizes and layouts the launcher reaches, and partial or whole last slices.
const Case kCases[] = {
  {6, 0, 1, 192, kPartialShardBytes},   // one 6-warp group: every slice on one block
  {6, 5, 3, 512, kPartialShardBytes},   // two groups a block and four warps outside them
  {6, 2, 2, 128, kWholeShardBytes},     // 4-warp groups fold 6 ranks in rounds of 4 and 2
  {16, 9, 2, 512, kPartialShardBytes},  // one 16-warp group, a warp per rank
  {20, 3, 2, 512, kPartialShardBytes},  // 16-warp groups fold 20 ranks in rounds of 16 and 4
};

}  // namespace

// The peer-parallel kernel gives the rank-order host fold bit for bit and sends every peer its shard.
TEST_F(DdaReduceScatterFabricLL128PeerPar, MatchesTheRankOrderFoldAndSendsEveryShard) {
  for (const Case& c : kCases) {
    SCOPED_TRACE(testing::Message() << c.nRanks << " ranks, self " << c.self << ", " << c.grid << "x" << c.block
                                    << ", shard " << c.shardBytes);
    const auto send = FloatMessages(c.nRanks, c.shardBytes);
    setUp(c.nRanks, c.self, c.shardBytes, send);
    const std::vector<uint8_t> got = expectMatchesBatched<float>(c);
    EXPECT_TRUE(got == FloatRankOrderFold(send, c.nRanks, c.self, c.shardBytes)) << "left the rank-order fold";
  }
}

// fp8 folds word by word through upconverts: the batched kernel's result at rank 0, and every peer's shard.
TEST_F(DdaReduceScatterFabricLL128PeerPar, MatchesTheBatchedKernelOnFp8) {
  for (const Case& c : {Case{8, 0, 2, 256, kPartialShardBytes}, Case{8, 0, 1, 512, kWholeShardBytes}}) {
    SCOPED_TRACE(testing::Message() << c.grid << "x" << c.block << ", shard " << c.shardBytes);
    setUp(c.nRanks, c.self, c.shardBytes, Fp8Messages(c.nRanks, (size_t)c.nRanks * c.shardBytes));
    expectMatchesBatched<dda::common::fp8e4m3>(c);
  }
}

// In place, recvbuff is my shard of sendbuff: my slice is read before the fold and written after it.
TEST_F(DdaReduceScatterFabricLL128PeerPar, InPlaceMatchesTheRankOrderFold) {
  const Case c{6, 5, 3, 512, kPartialShardBytes};
  const auto send = FloatMessages(c.nRanks, c.shardBytes);
  setUp(c.nRanks, c.self, c.shardBytes, send);
  const std::vector<uint8_t> got = expectMatchesBatched<float>(c, /*inPlace=*/true);
  EXPECT_TRUE(got == FloatRankOrderFold(send, c.nRanks, c.self, c.shardBytes)) << "left the rank-order fold";
}

// Buffers at any byte offset: both kernels' unaligned instances give the same fold and send every peer the same
// shard, so a rank whose buffers are aligned differently from its peers' still runs their algorithm.
TEST_F(DdaReduceScatterFabricLL128PeerPar, MisalignedBuffersMatchTheRankOrderFold) {
  for (size_t offset : {1, 4, 8}) {
    for (const Case& c : {Case{6, 5, 3, 512, kPartialShardBytes}, Case{16, 0, 2, 512, kPartialShardBytes}}) {
      for (bool inPlace : {false, true}) {
        SCOPED_TRACE(testing::Message() << c.nRanks << " ranks, self " << c.self << ", offset " << offset
                                        << (inPlace ? ", in place" : ""));
        const auto send = FloatMessages(c.nRanks, c.shardBytes);
        setUp(c.nRanks, c.self, c.shardBytes, send, offset);
        const std::vector<uint8_t> got = expectMatchesBatched<float>(c, inPlace);
        EXPECT_TRUE(got == FloatRankOrderFold(send, c.nRanks, c.self, c.shardBytes)) << "left the rank-order fold";
      }
    }
  }
}

// A peer that never sends aborts the kernel: recvbuff stays untouched and the epoch still moves on.
TEST_F(DdaReduceScatterFabricLL128PeerPar, AbortLeavesRecvbuffAndAdvancesEpoch) {
  for (const Case& c : {Case{6, 1, 2, 192, kPartialShardBytes}, Case{20, 3, 2, 512, kPartialShardBytes}}) {
    SCOPED_TRACE(testing::Message() << c.nRanks << " ranks");
    setUp(c.nRanks, c.self, c.shardBytes, FloatMessages(c.nRanks, c.shardBytes));
    reset(1, (c.self + 2) % c.nRanks);
    run<float>(true, c);
    const std::vector<uint8_t> got = recvBytes();
    for (size_t i = 0; i < c.shardBytes; ++i) {
      ASSERT_EQ(got[i], kUntouched) << "byte " << i;
    }
    expectEpochAdvanced();
  }
}

}  // namespace RcclUnitTesting
