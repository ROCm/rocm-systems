/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// DDA LL128 AllReduce two-shots (cooperative, LDS-staged writer, peer-split) on one GPU; helper kernels play the peers.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
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
constexpr uint32_t kFlag = 2;  // epoch cells start at kFlag - 1, so the kernels use flag 2 and bank 0
constexpr uint8_t kUntouched = 0xA5;

enum class Kind { kCoop, kStaged, kSplit };

struct Case {
  int nRanks;
  int self;
  int ct;  // NRANKS_CT of the instance: 4, 8 or 0
  Kind kind;
  unsigned grid;
  unsigned block;
  size_t shardBytes;
  uint32_t flag = kFlag;  // the epoch the launch takes: kFlag + 1 uses bank 1
  int epochLen = 0;       // epoch cells the launch owns; 0 is one per block
  bool inPlace = false;   // sendbuff == recvbuff
};

int epochLen(const Case& c) {
  return c.epochLen != 0 ? c.epochLen : (int)c.grid;
}

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

uint64_t wireFlag(uint32_t flag) {
  return ((uint64_t)flag << 32) | flag;
}

// Writes slice s of rows[peer] into area + peer * slotWords as the LL128 line that peer sends; one warp.
__device__ void PublishLine(uint64_t* area, const int8_t* rows, int peer, size_t s, size_t slotWordsArg,
                            size_t shardBytes, uint64_t flag) {
  const int lane = threadIdx.x;
  const bool flagLane = dda::common::ddaLL128IsFlagLane(lane);
  const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
  uint64_t regs[kDdaLL128WordsPerThread] = {};
  dda::common::ddaLL128LoadRegs<int8_t>(regs, rows + (size_t)peer * shardBytes + sliceByte,
                                        dda::common::ddaLL128SliceBytes(shardBytes, s), lane, flagLane);
  dda::common::ddaLL128StoreWire(area + (size_t)peer * slotWordsArg + s * kDdaLL128WireWordsPerSlice + 2 * lane, regs,
                                 flag, flagLane);
}

// One warp per (peers[blockIdx.x], slice blockIdx.y).
__global__ void PublishLines(uint64_t* area, const int8_t* rows, const int* peers, size_t slotWordsArg,
                             size_t shardBytes, uint64_t flag) {
  PublishLine(area, rows, peers[blockIdx.x], blockIdx.y, slotWordsArg, shardBytes, flag);
}

// For an in-place call: publishes peer p's reduced slice s only once my copy of slice s reached p (a peer reduces
// after it has every rank's copy), so the gather cannot overwrite a slice before the scatter read it.
__global__ void PublishLinesAfterMyCopy(uint64_t* area, const int8_t* rows, const int* peers, size_t slotWordsArg,
                                        size_t shardBytes, uint64_t flag, uint64_t* const* myCopyAt) {
  const int peer = peers[blockIdx.x];
  const size_t s = blockIdx.y;
  uint64_t seen[kDdaLL128WordsPerThread];
  dda::common::ddaLL128PollWire(myCopyAt[blockIdx.x] + s * kDdaLL128WireWordsPerSlice + 2 * threadIdx.x, seen, flag,
                                threadIdx.x, DdaKernelParams{nullptr});
  PublishLine(area, rows, peer, s, slotWordsArg, shardBytes, flag);
}

__global__ void ReportStaged(int* out) {
  *out = dda::common::ddaLL128Staged<kDdaLL128WireAsync>() ? 1 : 0;
}

// Most blocks of one launch resident at once: each block holds its slot for `ticks` of wall clock.
__global__ void CountResident(int* live, int* most, long long ticks) {
  if (threadIdx.x == 0) {
    atomicMax(most, atomicAdd(live, 1) + 1);
    const long long start = wall_clock64();
    while (wall_clock64() - start < ticks) {
    }
    atomicSub(live, 1);
  }
  __syncthreads();
}

template <typename T, int N>
void launch(const Case& c, T** scratch, T* recv, const T* send, uint32_t* epoch, uint32_t* abortFlag,
            hipStream_t stream = 0) {
  const DdaKernelParams params{abortFlag};
  const dim3 grid(c.grid);
  const dim3 block(c.block);
  const size_t bank = bankBytes(c.nRanks, c.shardBytes);
  const size_t n = slices(c.shardBytes);
  switch (c.kind) {
    case Kind::kCoop:
      hipLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL128<T, N, kDdaLL128WireCoop>), grid, block, 0, 0, scratch,
                         recv, send, c.shardBytes, c.self, c.nRanks, epoch, epochLen(c), n, bank, params);
      break;
    case Kind::kStaged:
      hipLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL128<T, N, kDdaLL128WireAsync>), grid, block,
                         dda::common::ddaLL128StageBytes(c.block), 0, scratch, recv, send, c.shardBytes, c.self,
                         c.nRanks, epoch, epochLen(c), n, bank, params);
      break;
    case Kind::kSplit:
      hipLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL128Split<T, N>), grid, block,
                         dda::common::ddaLL128ArSplitSharedBytes(c.block, c.nRanks), stream, scratch, recv, send,
                         c.shardBytes, c.self, c.nRanks, epoch, epochLen(c), n, bank, params);
      break;
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

  void TearDown() override {
    if (side_ != nullptr) {
      HIP_EXPECT(hipStreamDestroy(side_));
    }
    DeviceTestBase::TearDown();
  }

  // False on a device pass without async LDS stores, where the staged instance is the cooperative one.
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

  // Zeroed scratch, untouched recvbuff (my message in place), fresh epoch cells, then the peers' lines in c's bank.
  // skipCopy sends no first-area copy and skipReduced no reduced shard; in place, run() publishes the reduced shards.
  void reset(const Case& c, uint32_t abortFlag, int skipCopy = -1, int skipReduced = -1) {
    for (DeviceBuffer<uint8_t>& b : scratch_) {
      b.zero();
    }
    if (c.inPlace) {
      recvBuf_->copyFrom(send_[(size_t)self_]);
    } else {
      HIP_CHECK(hipMemset(recvBuf_->ptr, kUntouched, recvBuf_->count));
    }
    epoch_.copyFrom(std::vector<uint32_t>(kEpochCells, c.flag - 1));
    abort_.upload(abortFlag);
    bankOffset_ = (c.flag & 1u) * bankBytes(nRanks_, shardBytes_);

    std::vector<int8_t> copies((size_t)nRanks_ * shardBytes_);
    std::vector<int8_t> reduced((size_t)nRanks_ * shardBytes_);
    for (int p = 0; p < nRanks_; ++p) {
      std::memcpy(&copies[(size_t)p * shardBytes_], &send_[(size_t)p][(size_t)self_ * shardBytes_], shardBytes_);
      std::memcpy(&reduced[(size_t)p * shardBytes_], reduced_[(size_t)p].data(), shardBytes_);
    }
    publish(area(self_, 0), copies, skipCopy, c.flag);
    if (c.inPlace) {
      pendingReduced_ = std::make_unique<DeviceBuffer<int8_t>>(reduced.size());
      pendingReduced_->copyFrom(reduced);
    } else {
      publish(area(self_, nRanks_), reduced, skipReduced, c.flag);
    }
  }

  // Rank r's staging area that starts `slot` slots into c's bank.
  uint64_t* area(int r, int slot) {
    return reinterpret_cast<uint64_t*>(scratch_[(size_t)r].ptr + bankOffset_) +
           (size_t)slot * slotWords(nRanks_, shardBytes_);
  }

  std::vector<int> peersBut(int skip) {
    std::vector<int> peers;
    for (int p = 0; p < nRanks_; ++p) {
      if (p != self_ && p != skip) {
        peers.push_back(p);
      }
    }
    return peers;
  }

  void publish(uint64_t* at, const std::vector<int8_t>& rows, int skip, uint32_t flag) {
    const std::vector<int> peers = peersBut(skip);
    DeviceBuffer<int8_t> dRows(rows.size());
    dRows.copyFrom(rows);
    DeviceBuffer<int> dPeers(peers.size());
    dPeers.copyFrom(peers);
    hipLaunchKernelGGL(PublishLines, dim3((unsigned)peers.size(), (unsigned)slices(shardBytes_)), dim3(kDdaLL128Warp),
                       0, 0, at, dRows.ptr, dPeers.ptr, slotWords(nRanks_, shardBytes_), shardBytes_, wireFlag(flag));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  template <typename T>
  void run(const Case& c) {
    T** scratch = reinterpret_cast<T**>(scratchPtrs_->ptr);
    T* recv = reinterpret_cast<T*>(recvBuf_->ptr);
    const T* send = c.inPlace ? recv : reinterpret_cast<const T*>(sendBuf_->ptr);
    std::unique_ptr<DeviceBuffer<int>> dPeers;
    std::unique_ptr<DeviceBuffer<uint64_t*>> dCopyAt;
    if (c.inPlace) {
      // The peers, on their own stream, publish each reduced slice once my copy of it arrived.
      const std::vector<int> peers = peersBut(-1);
      std::vector<uint64_t*> copyAt;
      for (int p : peers) {
        copyAt.push_back(area(p, self_));
      }
      dPeers = std::make_unique<DeviceBuffer<int>>(peers.size());
      dPeers->copyFrom(peers);
      dCopyAt = std::make_unique<DeviceBuffer<uint64_t*>>(copyAt.size());
      dCopyAt->copyFrom(copyAt);
      if (side_ == nullptr) {
        HIP_CHECK(hipStreamCreateWithFlags(&side_, hipStreamNonBlocking));
      }
      hipLaunchKernelGGL(PublishLinesAfterMyCopy, dim3((unsigned)peers.size(), (unsigned)slices(shardBytes_)),
                         dim3(kDdaLL128Warp), 0, side_, area(self_, nRanks_), pendingReduced_->ptr, dPeers->ptr,
                         slotWords(nRanks_, shardBytes_), shardBytes_, wireFlag(c.flag), dCopyAt->ptr);
      HIP_CHECK(hipGetLastError());
    }
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

  // The second area slot[self] of every rank in the last reset's bank, where only a fully reduced shard of mine lands.
  bool reducedPublished() {
    const size_t words = slotWords(nRanks_, shardBytes_);
    const size_t at = bankOffset_ + (size_t)(nRanks_ + self_) * words * sizeof(uint64_t);
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

  // Runs the cooperative two-shot (and, for a split of 512-thread blocks, the staged one) out of place, then c, on the
  // same peer lines, grid and epoch: c must leave the same recvbuff and peer lines.
  template <typename T>
  Snapshot expectMatchesCoop(const Case& c) {
    SCOPED_TRACE(testing::Message() << c.nRanks << "r kind " << (int)c.kind << " ct " << c.ct << " shard "
                                    << c.shardBytes << " " << c.grid << "x" << c.block << " flag " << c.flag
                                    << " epochLen " << epochLen(c) << " inPlace " << c.inPlace);
    Case ref = c;
    ref.kind = Kind::kCoop;
    ref.inPlace = false;
    reset(ref, 0);
    run<T>(ref);
    const Snapshot want = snapshot();
    std::vector<Case> runs = {c};
    if (c.kind == Kind::kSplit && c.block == 512) {
      ref.kind = Kind::kStaged;
      runs.push_back(ref);
    }
    for (const Case& r : runs) {
      reset(r, 0);
      run<T>(r);
      const Snapshot got = snapshot();
      EXPECT_TRUE(got.recv == want.recv) << "kind " << (int)r.kind << ": recvbuff differs from the cooperative writer's";
      for (int p = 0; p < c.nRanks; ++p) {
        EXPECT_TRUE(got.peerScratch[(size_t)p] == want.peerScratch[(size_t)p])
            << "kind " << (int)r.kind << ": peer " << p << " lines differ";
      }
      expectEpochAdvanced(r);
    }
    return want;
  }

  // Every cell the launch owns moved to its flag; the cells past them stayed.
  void expectEpochAdvanced(const Case& c) {
    const std::vector<uint32_t> epoch = epoch_.copyTo();
    for (int e = 0; e < (int)kEpochCells; ++e) {
      EXPECT_EQ(epoch[(size_t)e], e < epochLen(c) ? c.flag : c.flag - 1) << "cell " << e;
    }
  }

  // My shard of recvbuff: untouched, or the host fold.
  void expectMyShard(bool reduced) {
    const std::vector<uint8_t> recv = recvBuf_->copyTo();
    for (size_t i = 0; i < shardBytes_; ++i) {
      ASSERT_EQ(recv[(size_t)self_ * shardBytes_ + i], reduced ? reduced_[(size_t)self_][i] : kUntouched)
          << "byte " << i;
    }
  }

  static constexpr size_t kEpochCells = 32;
  int nRanks_ = 0;
  int self_ = 0;
  size_t shardBytes_ = 0;
  size_t bankOffset_ = 0;
  std::vector<std::vector<uint8_t>> send_;
  std::vector<std::vector<uint8_t>> reduced_;
  std::vector<DeviceBuffer<uint8_t>> scratch_;
  std::unique_ptr<DeviceBuffer<uint8_t*>> scratchPtrs_;
  std::unique_ptr<DeviceBuffer<uint8_t>> sendBuf_;
  std::unique_ptr<DeviceBuffer<uint8_t>> recvBuf_;
  std::unique_ptr<DeviceBuffer<int8_t>> pendingReduced_;
  DeviceBuffer<uint32_t> epoch_{kEpochCells};
  DeviceBuffer<uint32_t> abort_{1};
  hipStream_t side_ = nullptr;
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

// The split on 3 and 5 ranks (paired polls only) and 4, 6, 8, 16 (an odd last peer), on 1 x 64 (many passes, the
// look-ahead running past the last unit) to 16 x 512, small to large shards, bank 1 with more epoch cells than blocks,
// and in place.
const Case kSplitCases[] = {
  {3, 1, 0, Kind::kSplit, 2, 512, kSmallShardBytes},
  {5, 2, 0, Kind::kSplit, 1, 64, kSmallShardBytes},
  {4, 1, 4, Kind::kSplit, 1, 64, kSmallShardBytes},
  {6, 5, 0, Kind::kSplit, 2, 512, kSmallShardBytes},
  {8, 5, 8, Kind::kSplit, 2, 512, kSmallShardBytes},
  {5, 3, 0, Kind::kSplit, 4, 512, kMediumShardBytes},
  {16, 9, 0, Kind::kSplit, 16, 512, kMediumShardBytes},
  {4, 1, 4, Kind::kSplit, 8, 512, kLargeShardBytes},
  {8, 5, 8, Kind::kSplit, 8, 512, kLargeShardBytes},
  {3, 2, 0, Kind::kSplit, 8, 512, kLargeShardBytes},
  {8, 5, 8, Kind::kSplit, 8, 512, kMediumShardBytes, kFlag + 1, 24},
  {5, 2, 0, Kind::kSplit, 2, 512, kMediumShardBytes, kFlag + 1, 7},
  {8, 5, 8, Kind::kSplit, 2, 512, kMediumShardBytes, kFlag, 0, true},
  {5, 2, 0, Kind::kSplit, 4, 512, kLargeShardBytes, kFlag + 1, 9, true},
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
  if (!stagedBuilt()) {
    GTEST_SKIP() << "this device pass has no async LDS stores";
  }
  for (const Case& c : {Case{8, 5, 8, Kind::kStaged, 2, 512, kSmallShardBytes},
                        Case{8, 5, 8, Kind::kStaged, 8, 512, kLargeShardBytes}}) {
    SCOPED_TRACE(testing::Message() << "shard " << c.shardBytes);
    const auto send = FloatMessages(c.nRanks, c.shardBytes);
    setUp(c.nRanks, c.self, c.shardBytes, send, FloatReduced(send, c.nRanks, c.shardBytes));
    reset(c, 1, (c.self + 2) % c.nRanks);
    run<float>(c);
    expectMyShard(/*reduced=*/false);
    EXPECT_FALSE(reducedPublished());
    expectEpochAdvanced(c);
  }
}

// The split leaves the cooperative and staged two-shots' recvbuff and peer lines bit for bit, the host fold included.
TEST_F(DdaAllReduceFabricLL128TwoShot, SplitMatchesTheCooperativeAndStagedWriters) {
  if (!stagedBuilt()) {
    GTEST_SKIP() << "this device pass has no async LDS stores";
  }
  for (const Case& c : kSplitCases) {
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

TEST_F(DdaAllReduceFabricLL128TwoShot, SplitMatchesTheCooperativeAndStagedWritersOnFp8) {
  if (!stagedBuilt()) {
    GTEST_SKIP() << "this device pass has no async LDS stores";
  }
  const Case cases[] = {{5, 2, 0, Kind::kSplit, 1, 64, kSmallShardBytes},
                        {8, 5, 8, Kind::kSplit, 2, 512, kSmallShardBytes},
                        {8, 5, 8, Kind::kSplit, 8, 512, kLargeShardBytes}};
  for (const Case& c : cases) {
    setUp(c.nRanks, c.self, c.shardBytes, Fp8Rows(c.nRanks, (size_t)c.nRanks * c.shardBytes, 0),
          Fp8Rows(c.nRanks, c.shardBytes, 5));
    expectMatchesCoop<dda::common::fp8e4m3>(c);
  }
}

// A peer that never sends its copy aborts the split's reduce at that peer's wait: first or second of a pair, a later
// pair, or the odd last peer. My shard is never stored or published, and the epoch still moves on.
TEST_F(DdaAllReduceFabricLL128TwoShot, SplitAbortInTheReduceLeavesNoPartialSum) {
  if (!stagedBuilt()) {
    GTEST_SKIP() << "this device pass has no async LDS stores";
  }
  const Case eight{8, 5, 8, Kind::kSplit, 2, 512, kSmallShardBytes};
  const struct {
    Case c;
    int skip;  // r of the peer self + r that never sends
  } cells[] = {{eight, 1}, {eight, 2}, {eight, 6}, {eight, 7},
               {{5, 2, 0, Kind::kSplit, 1, 64, kSmallShardBytes}, 3},
               {{8, 5, 8, Kind::kSplit, 8, 512, kLargeShardBytes, kFlag + 1, 12}, 7}};
  for (const auto& cell : cells) {
    const Case& c = cell.c;
    SCOPED_TRACE(testing::Message() << c.nRanks << "r shard " << c.shardBytes << " missing self + " << cell.skip);
    const auto send = FloatMessages(c.nRanks, c.shardBytes);
    setUp(c.nRanks, c.self, c.shardBytes, send, FloatReduced(send, c.nRanks, c.shardBytes));
    reset(c, 1, (c.self + cell.skip) % c.nRanks);
    run<float>(c);
    expectMyShard(/*reduced=*/false);
    EXPECT_FALSE(reducedPublished());
    expectEpochAdvanced(c);
  }
}

// A peer that never sends its reduced shard aborts the split's gather: my reduced shard is still stored whole, that
// peer's part of recvbuff stays untouched, and the epoch still moves on.
TEST_F(DdaAllReduceFabricLL128TwoShot, SplitAbortInTheGatherKeepsMyReducedShard) {
  if (!stagedBuilt()) {
    GTEST_SKIP() << "this device pass has no async LDS stores";
  }
  for (const Case& c : {Case{8, 5, 8, Kind::kSplit, 2, 512, kSmallShardBytes},
                        Case{5, 2, 0, Kind::kSplit, 4, 512, kMediumShardBytes, kFlag + 1}}) {
    SCOPED_TRACE(testing::Message() << c.nRanks << "r shard " << c.shardBytes);
    const auto send = FloatMessages(c.nRanks, c.shardBytes);
    setUp(c.nRanks, c.self, c.shardBytes, send, FloatReduced(send, c.nRanks, c.shardBytes));
    const int missing = (c.self + 3) % c.nRanks;
    reset(c, 1, -1, missing);
    run<float>(c);
    expectMyShard(/*reduced=*/true);
    const std::vector<uint8_t> recv = recvBuf_->copyTo();
    for (size_t i = 0; i < c.shardBytes; ++i) {
      ASSERT_EQ(recv[(size_t)missing * c.shardBytes + i], kUntouched) << "byte " << i;
    }
    expectEpochAdvanced(c);
  }
}

// Every rank's split at once, each rank's launch on a stream masked to two CUs so far fewer blocks are resident than
// the grid has. Every wait is on the same block index on the peers, so the grid drains; waits across block indices
// would hang until the abort below.
TEST_F(DdaAllReduceFabricLL128TwoShot, SplitDrainsWhenTheGridIsNotCoResident) {
  if (!stagedBuilt()) {
    GTEST_SKIP() << "this device pass has no async LDS stores";
  }
  constexpr int kRanks = 4;
  constexpr int kCusPerRank = 2;
  const Case c{kRanks, 0, 4, Kind::kSplit, 16, 512, kLargeShardBytes};
  int cus = 0;
  int clockKhz = 0;
  HIP_CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, 0));
  HIP_CHECK(hipDeviceGetAttribute(&clockKhz, hipDeviceAttributeWallClockRate, 0));
  if (cus < kRanks * kCusPerRank) {
    GTEST_SKIP() << "needs " << kRanks * kCusPerRank << " CUs";
  }

  const auto send = FloatMessages(kRanks, c.shardBytes);
  std::vector<uint8_t> expected;
  for (const std::vector<uint8_t>& shard : FloatReduced(send, kRanks, c.shardBytes)) {
    expected.insert(expected.end(), shard.begin(), shard.end());
  }
  std::vector<DeviceBuffer<uint8_t>> scratch;
  std::vector<DeviceBuffer<uint8_t>> sendBufs;
  std::vector<DeviceBuffer<uint8_t>> recvBufs;
  std::vector<DeviceBuffer<uint32_t>> epochs;
  std::vector<uint8_t*> ptrs;
  for (int r = 0; r < kRanks; ++r) {
    scratch.emplace_back(2 * bankBytes(kRanks, c.shardBytes));
    scratch.back().zero();
    ptrs.push_back(scratch.back().ptr);
    sendBufs.emplace_back(send[(size_t)r].size());
    sendBufs.back().copyFrom(send[(size_t)r]);
    recvBufs.emplace_back(send[(size_t)r].size());
    epochs.emplace_back(kEpochCells);
    epochs.back().copyFrom(std::vector<uint32_t>(kEpochCells, c.flag - 1));
  }
  DeviceBuffer<uint8_t*> peerTable(ptrs.size());
  peerTable.copyFrom(ptrs);
  // Host mapped, so a hung grid can be told to leave.
  uint32_t* abortFlag = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&abortFlag), sizeof(uint32_t),
                          hipHostMallocMapped | hipHostMallocCoherent));
  *abortFlag = 0;
  uint32_t* abortDev = nullptr;
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&abortDev), abortFlag, 0));

  std::vector<hipStream_t> streams((size_t)kRanks, nullptr);
  for (int r = 0; r < kRanks; ++r) {
    std::vector<uint32_t> mask(((size_t)cus + 31) / 32, 0u);
    for (int cu = r * kCusPerRank; cu < (r + 1) * kCusPerRank; ++cu) {
      mask[(size_t)cu / 32] |= 1u << (cu % 32);
    }
    HIP_CHECK(hipExtStreamCreateWithCUMask(&streams[(size_t)r], (uint32_t)mask.size(), mask.data()));
  }

  // The masks must hold fewer blocks than the grid, or this run proves nothing.
  DeviceBuffer<int> counts(2);
  counts.zero();
  hipLaunchKernelGGL(CountResident, dim3(c.grid), dim3(c.block),
                     dda::common::ddaLL128ArSplitSharedBytes(c.block, kRanks), streams[0], counts.ptr, counts.ptr + 1,
                     (long long)clockKhz);
  HIP_CHECK(hipStreamSynchronize(streams[0]));
  const int mostResident = counts.copyTo()[1];

  if (mostResident < (int)c.grid) {
    float** table = reinterpret_cast<float**>(peerTable.ptr);
    for (int r = 0; r < kRanks; ++r) {
      Case rank = c;
      rank.self = r;
      launch<float, 4>(rank, table, reinterpret_cast<float*>(recvBufs[(size_t)r].ptr),
                       reinterpret_cast<const float*>(sendBufs[(size_t)r].ptr), epochs[(size_t)r].ptr, abortDev,
                       streams[(size_t)r]);
      HIP_CHECK(hipGetLastError());
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    bool drained = false;
    while (!drained && std::chrono::steady_clock::now() < deadline) {
      drained = true;
      for (hipStream_t st : streams) {
        drained = drained && hipStreamQuery(st) == hipSuccess;
      }
      if (!drained) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    if (!drained) {
      *abortFlag = 1;
    }
    for (hipStream_t st : streams) {
      HIP_CHECK(hipStreamSynchronize(st));
    }
    EXPECT_TRUE(drained) << "the split did not drain with at most " << mostResident << " of " << c.grid
                         << " blocks resident per rank";
    for (int r = 0; drained && r < kRanks; ++r) {
      EXPECT_TRUE(recvBufs[(size_t)r].copyTo() == expected) << "rank " << r << " recvbuff is not the host fold";
    }
  }

  for (hipStream_t st : streams) {
    HIP_EXPECT(hipStreamDestroy(st));
  }
  HIP_EXPECT(hipHostFree(abortFlag));
  if (mostResident >= (int)c.grid) {
    GTEST_SKIP() << "the CU masks left all " << c.grid << " blocks resident";
  }
}

}  // namespace

}  // namespace RcclUnitTesting
