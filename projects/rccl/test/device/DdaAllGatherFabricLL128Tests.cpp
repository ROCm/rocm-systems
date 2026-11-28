/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// DDA LL128 AllGather kernel on one GPU, aligned and at any byte offset. A helper kernel plays the peers.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "DeviceTestBase.hpp"
#include "algorithms/dda/all_gather/all_gather_dda_fabric_ll128.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting {

namespace {

using dda::common::DdaKernelParams;
using dda::common::kDdaLL128DataBytesPerSlice;
using dda::common::kDdaLL128Warp;
using dda::common::kDdaLL128WireWordsPerSlice;
using dda::common::kDdaLL128WordsPerThread;

// 5 full slices and one 16-byte chunk, so the last slice is all but empty.
constexpr size_t kPartialBytes = 5 * kDdaLL128DataBytesPerSlice + 16;
constexpr uint32_t kFlag = 2;  // epoch cells start at 1, so the kernel uses flag 2 and bank 0
constexpr uint8_t kUntouched = 0xA5;
constexpr unsigned kMaxGrid = 64;
// Room for a user buffer at any byte offset up to a 16B chunk.
constexpr size_t kOffsetRoom = 16;

struct Case {
  int nRanks;
  int self;
  int ct;  // NRANKS_CT of the instance: 4, 8 or 0
  unsigned gridY;
  unsigned block;
  size_t perRankBytes;
};

size_t slices(size_t bytes) {
  return dda::common::ddaLL128Slices(bytes);
}

// A bank of exactly nRanks slots.
size_t bankBytes(int nRanks, size_t bytes) {
  return (size_t)nRanks * slices(bytes) * kDdaLL128WireWordsPerSlice * sizeof(uint64_t);
}

size_t slotWords(int nRanks, size_t bytes) {
  return dda::common::ddaLL128AgSlotWords(bankBytes(nRanks, bytes), nRanks);
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

// Unpacks the lines at slot into out: one warp per slice. Run with the abort flag raised, so a line without the flag
// sets *missing instead of waiting.
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

template <int N, bool kAligned>
void launch(const Case& c, int8_t** scratch, int8_t* recv, const int8_t* send, uint32_t* epoch, uint32_t* abortFlag) {
  hipLaunchKernelGGL((dda::common::ddaAllGatherFabricLL128<int8_t, N, kAligned>),
                     dim3((unsigned)c.nRanks - 1, c.gridY), dim3(c.block), 0, 0, scratch, recv, send, c.perRankBytes,
                     c.self, c.nRanks, epoch, (int)kMaxGrid, slices(c.perRankBytes),
                     bankBytes(c.nRanks, c.perRankBytes), DdaKernelParams{abortFlag});
}

class DdaAllGatherFabricLL128 : public DeviceTestBase {
 protected:
  void SetUp() override {
    DeviceTestBase::SetUp();
    if (!isGfx1250()) {
      GTEST_SKIP() << "the LL128 AllGather tier runs on gfx1250";
    }
  }

  // Rank r sends bytes r * 37 + i; my sendbuff and recvbuff start `offset` bytes into their allocations. In place,
  // sendbuff is my slot of recvbuff.
  void setUp(const Case& c, size_t offset, bool inPlace) {
    c_ = c;
    offset_ = offset;
    inPlace_ = inPlace;
    const size_t total = (size_t)c.nRanks * c.perRankBytes;
    gathered_.resize(total);
    for (size_t i = 0; i < total; ++i) {
      gathered_[i] = (uint8_t)((i / c.perRankBytes) * 37 + i * 11 + 1);
    }
    scratch_.clear();
    std::vector<int8_t*> ptrs;
    for (int r = 0; r < c.nRanks; ++r) {
      scratch_.emplace_back(2 * bankBytes(c.nRanks, c.perRankBytes));
      scratch_.back().zero();
      ptrs.push_back(reinterpret_cast<int8_t*>(scratch_.back().ptr));
    }
    scratchPtrs_ = std::make_unique<DeviceBuffer<int8_t*>>(ptrs.size());
    scratchPtrs_->copyFrom(ptrs);
    sendBuf_ = std::make_unique<DeviceBuffer<uint8_t>>(c.perRankBytes + kOffsetRoom);
    recvBuf_ = std::make_unique<DeviceBuffer<uint8_t>>(total + kOffsetRoom);
    HIP_CHECK(hipMemset(sendBuf_->ptr, kUntouched, sendBuf_->count));
    HIP_CHECK(hipMemset(recvBuf_->ptr, kUntouched, recvBuf_->count));
    const uint8_t* mine = gathered_.data() + (size_t)c.self * c.perRankBytes;
    HIP_CHECK(hipMemcpy(send(), mine, c.perRankBytes, hipMemcpyHostToDevice));
    epoch_.copyFrom(std::vector<uint32_t>(kMaxGrid, kFlag - 1));
    abort_.upload(0);

    // Every peer's lines, already in my scratch at its slot.
    std::vector<int> peers;
    for (int p = 0; p < c.nRanks; ++p) {
      if (p != c.self) {
        peers.push_back(p);
      }
    }
    DeviceBuffer<int8_t> rows(total);
    rows.copyFrom(std::vector<int8_t>(gathered_.begin(), gathered_.end()));
    DeviceBuffer<int> dPeers(peers.size());
    dPeers.copyFrom(peers);
    hipLaunchKernelGGL(PublishLines, dim3((unsigned)peers.size(), (unsigned)slices(c.perRankBytes)),
                       dim3(kDdaLL128Warp), 0, 0, reinterpret_cast<uint64_t*>(scratch_[(size_t)c.self].ptr), rows.ptr,
                       dPeers.ptr, slotWords(c.nRanks, c.perRankBytes), c.perRankBytes);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  uint8_t* recv() { return recvBuf_->ptr + offset_; }
  uint8_t* send() { return inPlace_ ? recv() + (size_t)c_.self * c_.perRankBytes : sendBuf_->ptr + offset_; }

  // The kernel of c_ with the user-buffer access the launcher picks for my buffers.
  void run() {
    if (offset_ % 16 == 0) {
      run<true>();
    } else {
      run<false>();
    }
  }

  template <bool kAligned>
  void run() {
    int8_t** scratch = scratchPtrs_->ptr;
    int8_t* r = reinterpret_cast<int8_t*>(recv());
    const int8_t* s = reinterpret_cast<const int8_t*>(send());
    switch (c_.ct) {
      case 4:
        launch<4, kAligned>(c_, scratch, r, s, epoch_.ptr, abort_.ptr);
        break;
      case 8:
        launch<8, kAligned>(c_, scratch, r, s, epoch_.ptr, abort_.ptr);
        break;
      default:
        launch<0, kAligned>(c_, scratch, r, s, epoch_.ptr, abort_.ptr);
        break;
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  // recvbuff holds every rank's payload in rank order, and no byte around it was written.
  void expectGathered() {
    const std::vector<uint8_t> buf = recvBuf_->copyTo();
    const size_t end = offset_ + gathered_.size();
    for (size_t i = 0; i < buf.size(); ++i) {
      const bool inside = i >= offset_ && i < end;
      const uint8_t want = inside ? gathered_[i - offset_] : kUntouched;
      ASSERT_EQ(buf[i], want) << (inside ? "wrong gathered byte " : "a store landed outside the buffer at byte ")
                              << (inside ? i - offset_ : i);
    }
  }

  // Every peer unpacks my payload from my slot of its scratch.
  void expectSent() {
    const std::vector<uint8_t> mine(gathered_.begin() + (std::ptrdiff_t)((size_t)c_.self * c_.perRankBytes),
                                    gathered_.begin() + (std::ptrdiff_t)((size_t)(c_.self + 1) * c_.perRankBytes));
    for (int p = 0; p < c_.nRanks; ++p) {
      if (p == c_.self) {
        continue;
      }
      DeviceBuffer<uint8_t> out(c_.perRankBytes);
      out.zero();
      DeviceBuffer<int> missing(1);
      missing.zero();
      DeviceBuffer<uint32_t> raised(1);
      raised.upload(1);
      const uint64_t* slot = reinterpret_cast<const uint64_t*>(scratch_[(size_t)p].ptr) +
                             (size_t)c_.self * slotWords(c_.nRanks, c_.perRankBytes);
      hipLaunchKernelGGL(UnpackSlot, dim3((unsigned)slices(c_.perRankBytes)), dim3(kDdaLL128Warp), 0, 0, slot,
                         reinterpret_cast<int8_t*>(out.ptr), c_.perRankBytes, raised.ptr, missing.ptr);
      HIP_EXPECT(hipGetLastError());
      HIP_EXPECT(hipDeviceSynchronize());
      EXPECT_EQ(missing.download(), 0) << "a line to peer " << p << " has no flag";
      EXPECT_TRUE(out.copyTo() == mine) << "peer " << p << " unpacks the wrong payload";
    }
  }

  // Every block bumps the cells it owns, so all kMaxGrid cells end at kFlag.
  void expectEpochAdvanced() {
    const std::vector<uint32_t> epoch = epoch_.copyTo();
    for (unsigned e = 0; e < kMaxGrid; ++e) {
      EXPECT_EQ(epoch[e], kFlag) << "epoch cell " << e;
    }
  }

  Case c_{};
  size_t offset_ = 0;
  bool inPlace_ = false;
  std::vector<uint8_t> gathered_;
  std::vector<DeviceBuffer<uint8_t>> scratch_;
  std::unique_ptr<DeviceBuffer<int8_t*>> scratchPtrs_;
  std::unique_ptr<DeviceBuffer<uint8_t>> sendBuf_;
  std::unique_ptr<DeviceBuffer<uint8_t>> recvBuf_;
  DeviceBuffer<uint32_t> epoch_{kMaxGrid};
  DeviceBuffer<uint32_t> abort_{1};
};

std::string Describe(const Case& c) {
  return std::to_string(c.nRanks) + " ranks, self " + std::to_string(c.self) + ", ct " + std::to_string(c.ct) + ", " +
         std::to_string(c.nRanks - 1) + "x" + std::to_string(c.gridY) + "x" + std::to_string(c.block) + ", " +
         std::to_string(c.perRankBytes) + " B";
}

// Unrolled and runtime rank counts, one or several blocks a peer, and partial or whole last slices.
const Case kCases[] = {
  {4, 1, 4, 1, 256, kPartialBytes},
  {8, 5, 8, 2, 128, kPartialBytes},
  {3, 2, 0, 3, 64, 6 * kDdaLL128DataBytesPerSlice},
  {16, 9, 0, 1, 512, kPartialBytes},
};

}  // namespace

// The aligned and unaligned instances gather the same bytes and send every peer the same lines, so a rank whose
// buffers are aligned differently from its peers' still runs their algorithm.
TEST_F(DdaAllGatherFabricLL128, GathersEveryRankAndSendsEveryPeerAtAnyOffset) {
  for (size_t offset : {0, 1, 4, 8, 12}) {
    for (const Case& c : kCases) {
      for (bool inPlace : {false, true}) {
        SCOPED_TRACE(Describe(c) + ", offset " + std::to_string(offset) + (inPlace ? ", in place" : ""));
        setUp(c, offset, inPlace);
        run();
        expectGathered();
        expectSent();
        expectEpochAdvanced();
      }
    }
  }
}

}  // namespace RcclUnitTesting
