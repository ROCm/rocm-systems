/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ddaAllReduceFlatLL and ddaAllReduceSpreadTwoShotLL on one GPU: the host plays the peers by writing LL lines into
// coherent host scratch. Built twice: rccl-UnitTestsFixtures defines DWORDX4_INTRINSICS_FORCE_OFF (dword re-read),
// rccl-UnitTests not (b128).

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <thread>
#include <type_traits>
#include <vector>

#include "DdaLLHostPeers.hpp"
#include "DeviceTestBase.hpp"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll_spread.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting {

namespace {

using dda::common::DdaKernelParams;
using dda::common::LLPacket16;

constexpr int kMaxRanks = 12;  // past 8 ranks the spread folds its LDS rows in two batches
constexpr int kSelf = 1;
constexpr size_t kCount = 3072;  // floats: 1536 packets, 3 blocks of 512
constexpr size_t kOneShotBytes = kCount * sizeof(float);
constexpr size_t kBufBytes = 2 * kOneShotBytes;  // a spread shard of more than one LDS pass on one block
constexpr unsigned kOneShotThreads = 512;
static_assert(dda::common::kDdaLLSpreadThreads == kOneShotThreads, "the spread runs on the one-shot's blocks");
constexpr size_t kBankBytes = (size_t)kMaxRanks * 64 * 1024;
constexpr uint32_t kFlag = 2;  // epoch cells start at 1, so the kernels use flag 2 and bank 0
constexpr unsigned kEpochCells = 16;
constexpr float kUntouched = -1.0f;
constexpr auto kWatchdog = std::chrono::seconds(30);
constexpr auto kSettle = std::chrono::milliseconds(20);  // lets every thread finish its first read of the late lines
constexpr int kCopyArea = 0;  // spread area holding a peer's copy of the owner's shard
constexpr int kReducedArea = 1;  // spread area holding a peer's reduced shard
constexpr int kHoldNothing = -1;
constexpr bool kAbort = true;
constexpr bool kNoAbort = false;

// Rank order folds (-1e8 + 1e8) first, so every later rank's small value survives; any other order loses some of them.
float rankValue(int rank) {
  if (rank == 0) {
    return -1e8f;
  }
  if (rank == kSelf) {
    return 1e8f;
  }
  return (float)(rank - 1);
}

// Exact rank-order sum: 0 + 1 + 2 + ... + (nRanks - 2) from the ranks past kSelf, each a distinct nonzero value.
float rankOrderSum(int nRanks) {
  return (float)((nRanks - 2) * (nRanks - 1) / 2);
}

template <typename T>
const char* typeName() {
  if constexpr (std::is_same_v<T, float>) {
    return "float";
  } else if constexpr (std::is_same_v<T, bf16>) {
    return "bf16";
  } else if constexpr (std::is_same_v<T, half>) {
    return "half";
  } else {
    static_assert(!std::is_same_v<T, T>, "add a name for this type");
  }
}

uint32_t bitsOf(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

class DdaAllReduceFabricLL : public DeviceTestBase {
 protected:
  void SetUp() override {
    DeviceTestBase::SetUp();
    peers_.Init(kMaxRanks, 2 * kBankBytes);
    const std::vector<float> send(kBufBytes / sizeof(float), rankValue(kSelf));
    HIP_CHECK(hipMalloc(&dSend_, kBufBytes));
    HIP_CHECK(hipMemcpy(dSend_, send.data(), kBufBytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&dRecv_, kBufBytes));
    HIP_CHECK(hipMalloc(&dEpoch_, kEpochCells * sizeof(uint32_t)));
    reset();
    HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
  }

  // Zeroed scratch, a clear abort flag, epoch cells one before kFlag and an untouched recvbuff.
  void reset() {
    for (int r = 0; r < kMaxRanks; ++r) {
      std::memset(peers_.Host(r), 0, 2 * kBankBytes);
    }
    *peers_.AbortFlag() = 0;
    const std::vector<uint32_t> epoch(kEpochCells, kFlag - 1);
    HIP_CHECK(hipMemcpy(dEpoch_, epoch.data(), kEpochCells * sizeof(uint32_t), hipMemcpyHostToDevice));
    clearRecv();
  }

  void clearRecv() {
    const std::vector<float> recv(kBufBytes / sizeof(float), kUntouched);
    HIP_CHECK(hipMemcpy(dRecv_, recv.data(), kBufBytes, hipMemcpyHostToDevice));
  }

  void TearDown() override {
    HIP_EXPECT(hipStreamDestroy(stream_));
    HIP_EXPECT(hipFree(dEpoch_));
    HIP_EXPECT(hipFree(dRecv_));
    HIP_EXPECT(hipFree(dSend_));
    peers_.Release();
  }

  // Line `pk` of sender's slot in owner's scratch.
  volatile uint32_t* line(int owner, int sender, size_t pk, size_t slotPkts) {
    LLPacket16* base = static_cast<LLPacket16*>(peers_.Host(owner));
    return reinterpret_cast<volatile uint32_t*>(&base[sender * slotPkts + pk]);
  }

  size_t oneShotSlot() const { return dda::common::ddaLLArSlotPkts(kBankBytes, nRanks_); }
  size_t packets() const { return bytes_ / 8; }
  unsigned oneShotGrid() const { return (unsigned)((packets() + kOneShotThreads - 1) / kOneShotThreads); }

  void publishPeer(int peer) {
    for (size_t pk = 0; pk < packets(); ++pk) {
      DdaLLHostPeers::PublishLine(line(kSelf, peer, pk, oneShotSlot()), bitsOf(rankValue(peer)),
                                  bitsOf(rankValue(peer)), kFlag);
    }
  }

  void publishHalf(int peer, int half) {
    for (size_t pk = 0; pk < packets(); ++pk) {
      DdaLLHostPeers::PublishHalf(line(kSelf, peer, pk, oneShotSlot()), half, bitsOf(rankValue(peer)), kFlag);
    }
  }

  void publishAll() {
    for (int peer = 0; peer < nRanks_; ++peer) {
      if (peer != kSelf) {
        publishPeer(peer);
      }
    }
  }

  // True once every thread has stored its phase-1 line to the next peer, so nearly all of them are already polling.
  bool phase1Done() {
    const int peer = (self_ + 1) % nRanks_;
    for (size_t pk = 0; pk < (spread_ ? shardLen(peer) : packets()); ++pk) {
      volatile uint32_t* w = spread_ ? spreadLine(peer, kCopyArea, self_, pk) : line(peer, self_, pk, oneShotSlot());
      if (w[1] != flag_ || w[3] != flag_) {
        return false;
      }
    }
    return true;
  }

  // Waits for phase 1, then checks the kernel is still polling.
  void expectStillPolling(const char* what) {
    const auto deadline = std::chrono::steady_clock::now() + kWatchdog;
    while (!phase1Done() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    std::this_thread::sleep_for(kSettle);
    EXPECT_EQ(hipStreamQuery(stream_), hipErrorNotReady) << what;
  }

  bool finishes() {
    return peers_.Finishes(stream_, [this] { publishAll(); }, kWatchdog);
  }

  std::vector<float> recv() {
    std::vector<float> out(kCount);
    EXPECT_EQ(hipMemcpy(out.data(), dRecv_, kCount * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
    return out;
  }

  // Every block, idle or aborted, bumps the cells it owns (e % grid), so all kEpochCells cells end at flag.
  void expectEpochAdvanced(uint32_t flag = kFlag) {
    std::vector<uint32_t> epoch(kEpochCells);
    ASSERT_EQ(hipMemcpy(epoch.data(), dEpoch_, kEpochCells * sizeof(uint32_t), hipMemcpyDeviceToHost), hipSuccess);
    for (unsigned e = 0; e < kEpochCells; ++e) {
      EXPECT_EQ(epoch[e], flag) << "epoch cell " << e;
    }
  }

  void expectRankOrderSum() {
    const float want = rankOrderSum(nRanks_);
    for (float v : recv()) {
      ASSERT_EQ(bitsOf(v), bitsOf(want)) << "fold left rank order, every peer once";
    }
  }

  // Rank 0, the first in fold order, publishes only after the kernel has polled every batch at least once.
  template <int N>
  void runLatePeer(int nRanks) {
    nRanks_ = nRanks;
    for (int peer = 2; peer < nRanks_; ++peer) {
      publishPeer(peer);
    }
    ASSERT_NO_FATAL_FAILURE(launchLL<float>(&dda::common::ddaAllReduceFlatLL<float, N>, oneShotGrid(), dSend_));
    expectStillPolling("finished before rank 0 published");
    publishPeer(0);
    ASSERT_TRUE(finishes()) << "kernel never saw the late peer";
    expectRankOrderSum();
    expectEpochAdvanced();
  }

  // Spread two-shot: rank self_ reduces bytes_ of per-rank words in_ (two a packet); ref_ is the one-shot's result.
  size_t shardPkts() const { return dda::common::ddaLLSpreadShardPkts(packets(), nRanks_); }
  size_t shardLen(int rank) const { return dda::common::ddaLLSpreadShardLen(packets(), shardPkts(), rank); }
  size_t twoShotSlot() const { return dda::common::ddaLLArTwoShotSlotPkts(kBankBytes, nRanks_); }
  int heldPeer() const { return self_ == 0 ? 1 : 0; }

  // Line pk of sender's slot in owner's bank flag_ & 1, in kCopyArea (copies of owner's shard) or kReducedArea.
  volatile uint32_t* spreadLine(int owner, int area, int sender, size_t pk) {
    const size_t bank = (size_t)(flag_ & 1u) * (kBankBytes / sizeof(LLPacket16));
    return line(owner, sender, bank + (size_t)area * nRanks_ * twoShotSlot() + pk, twoShotSlot());
  }

  // Random words per rank and element, so a misplaced packet or a reordered fold changes the result bits.
  void randomInputs(bool halves, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> value(-1.0f, 1.0f);
    std::uniform_int_distribution<int> scale(-12, 12);
    auto draw = [&] { return bitsOf(std::ldexp(value(rng), scale(rng))); };
    in_.assign(nRanks_, std::vector<uint32_t>(bytes_ / sizeof(uint32_t)));
    for (std::vector<uint32_t>& words : in_) {
      for (uint32_t& w : words) {
        w = draw();
        if (halves) {
          w = (w >> 16) | (draw() & 0xffff0000u);
        }
      }
    }
  }

  std::vector<uint32_t> recvWords() {
    std::vector<uint32_t> out(bytes_ / sizeof(uint32_t));
    EXPECT_EQ(hipMemcpy(out.data(), dRecv_, bytes_, hipMemcpyDeviceToHost), hipSuccess);
    return out;
  }

  // Launches kernel, a one-shot or spread instance over T, as rank self_ on bytes_ of send: grid x kOneShotThreads.
  template <typename T, typename Kernel>
  void launchLL(Kernel kernel, unsigned grid, const void* send) {
    kernel<<<dim3(grid), dim3(kOneShotThreads), 0, stream_>>>(
        peers_.DevPeers<T>(), reinterpret_cast<T*>(dRecv_), static_cast<const T*>(send), bytes_ / sizeof(T), self_,
        nRanks_, dEpoch_, (int)kEpochCells, kBankBytes, DdaKernelParams{peers_.AbortFlag()});
    ASSERT_EQ(hipGetLastError(), hipSuccess);
  }

  // From reset(): random inputs for nRanks ranks as rank self, then ref_ from the one-shot on its own grid at kFlag.
  template <typename T, int N>
  void spreadReference(int nRanks, int self, size_t bytes, uint32_t seed) {
    nRanks_ = nRanks;
    self_ = self;
    bytes_ = bytes;
    flag_ = kFlag;
    spread_ = false;
    reset();
    randomInputs(sizeof(T) == 2, seed);
    HIP_CHECK(hipMemcpy(dSend_, in_[self_].data(), bytes_, hipMemcpyHostToDevice));
    for (int peer = 0; peer < nRanks_; ++peer) {
      if (peer == self_) {
        continue;
      }
      for (size_t pk = 0; pk < packets(); ++pk) {
        const uint32_t* w = &in_[peer][2 * pk];
        DdaLLHostPeers::PublishLine(line(self_, peer, pk, oneShotSlot()), w[0], w[1], kFlag);
      }
    }
    ASSERT_NO_FATAL_FAILURE(launchLL<T>(&dda::common::ddaAllReduceFlatLL<T, N>, oneShotGrid(), dSend_));
    ASSERT_TRUE(peers_.Finishes(stream_, [] {}, kWatchdog)) << "one-shot reference did not finish";
    ref_ = recvWords();
  }

  // Peer's lines below packet end in my scratch under flag_: its copy of my shard (kCopyArea) or its reduced shard.
  void publishSpread(int peer, int area, size_t end = SIZE_MAX) {
    const int owner = area == kCopyArea ? self_ : peer;
    const std::vector<uint32_t>& words = area == kCopyArea ? in_[peer] : ref_;
    for (size_t pk = 0; pk < shardLen(owner) && pk < end; ++pk) {
      const size_t w = 2 * (owner * shardPkts() + pk);
      DdaLLHostPeers::PublishLine(spreadLine(self_, area, peer, pk), words[w], words[w + 1], flag_);
    }
  }

  // Publishes every peer's lines but heldPeer()'s heldArea from packet heldFrom_ on.
  void publishSpreadAll(int heldArea = kHoldNothing) {
    for (int peer = 0; peer < nRanks_; ++peer) {
      for (int area : {kCopyArea, kReducedArea}) {
        if (peer != self_) {
          publishSpread(peer, area, peer == heldPeer() && area == heldArea ? heldFrom_ : SIZE_MAX);
        }
      }
    }
  }

  // Publishes every line but heldPeer()'s held area; once the kernel waits for it, publishes it or raises abort.
  template <typename T, int N>
  void spreadRun(unsigned grid, int heldArea, bool abort, const void* send) {
    spread_ = true;
    publishSpreadAll(heldArea);
    ASSERT_NO_FATAL_FAILURE(launchLL<T>(&dda::common::ddaAllReduceSpreadTwoShotLL<T, N>, grid, send));
    if (heldArea != kHoldNothing) {
      expectStillPolling("finished before the held lines were published");
      if (abort) {
        *peers_.AbortFlag() = 1;
      } else {
        publishSpread(heldPeer(), heldArea);
      }
    }
    ASSERT_TRUE(peers_.Finishes(stream_, [this] { publishSpreadAll(); }, kWatchdog)) << "spread kernel did not finish";
  }

  // recvbuff is the one-shot's bit for bit; an abort leaves the held shard untouched, every shard for a copy but my
  // packets below heldFrom_.
  void expectSpreadRecv(int heldArea, bool abort) {
    const std::vector<uint32_t> out = recvWords();
    for (size_t i = 0; i < out.size(); ++i) {
      const int owner = (int)(i / 2 / shardPkts());
      if (!abort || (owner == self_ && (heldArea == kReducedArea || i / 2 % shardPkts() < heldFrom_))) {
        ASSERT_EQ(out[i], ref_[i]) << "word " << i << " of shard " << owner;
      } else if (owner == heldPeer() || heldArea == kCopyArea) {
        ASSERT_EQ(out[i], bitsOf(kUntouched)) << "an aborted thread wrote word " << i << " of shard " << owner;
      }
    }
  }

  // Every peer holds my copy of its shard, nothing past its shard's end, and my reduced shard (below heldFrom_ if not
  // reduced).
  void expectSpreadPublished(bool reduced) {
    for (int peer = 0; peer < nRanks_; ++peer) {
      if (peer == self_) {
        continue;
      }
      for (size_t pk = 0; pk < shardPkts(); ++pk) {
        const size_t w = 2 * (peer * shardPkts() + pk);
        volatile uint32_t* c = spreadLine(peer, kCopyArea, self_, pk);
        if (pk < shardLen(peer)) {
          ASSERT_TRUE(c[0] == in_[self_][w] && c[1] == flag_ && c[2] == in_[self_][w + 1] && c[3] == flag_)
              << "copy to " << peer << " pk " << pk;
        } else {
          ASSERT_TRUE(c[0] == 0 && c[1] == 0 && c[2] == 0 && c[3] == 0) << "line past the shard of " << peer;
        }
        if (pk < shardLen(self_)) {
          const size_t m = 2 * (self_ * shardPkts() + pk);
          volatile uint32_t* r = spreadLine(peer, kReducedArea, self_, pk);
          ASSERT_TRUE(reduced || pk < heldFrom_
                          ? r[0] == ref_[m] && r[1] == flag_ && r[2] == ref_[m + 1] && r[3] == flag_
                          : r[1] != flag_ && r[3] != flag_)
              << "reduced shard to " << peer << " pk " << pk;
        }
      }
    }
  }

  // spreadRun, then recvbuff, the published lines and the epoch cells at flag_.
  template <typename T, int N>
  void spreadRunAndCheck(unsigned grid, int heldArea, bool abort, const void* send) {
    ASSERT_NO_FATAL_FAILURE((spreadRun<T, N>(grid, heldArea, abort, send)));
    ASSERT_NO_FATAL_FAILURE(expectSpreadRecv(heldArea, abort));
    ASSERT_NO_FATAL_FAILURE(expectSpreadPublished(!abort || heldArea != kCopyArea));
    expectEpochAdvanced(flag_);
  }

  // One spread run over T on nRanks ranks as rank self from a fresh reset(), checked against the one-shot.
  template <typename T, int N>
  void spreadCase(int nRanks, int self, size_t bytes, unsigned grid, int heldArea = kHoldNothing,
                  bool abort = kNoAbort) {
    SCOPED_TRACE(::testing::Message() << typeName<T>() << ", " << nRanks << " ranks, self " << self << ", " << bytes
                                      << " B, grid " << grid);
    ASSERT_NO_FATAL_FAILURE((spreadReference<T, N>(nRanks, self, bytes, (uint32_t)(nRanks * 7919 + bytes))));
    reset();
    ASSERT_NO_FATAL_FAILURE((spreadRunAndCheck<T, N>(grid, heldArea, abort, dSend_)));
  }

  DdaLLHostPeers peers_;
  int nRanks_ = 6;
  int self_ = kSelf;
  uint32_t flag_ = kFlag;
  bool spread_ = false;
  size_t heldFrom_ = 0;  // packets of the held area published before the hold: the reduce passes that finish
  size_t bytes_ = kOneShotBytes;
  std::vector<std::vector<uint32_t>> in_;
  std::vector<uint32_t> ref_;
  float* dSend_ = nullptr;
  float* dRecv_ = nullptr;
  uint32_t* dEpoch_ = nullptr;
  hipStream_t stream_ = nullptr;
};

}  // namespace

TEST_F(DdaAllReduceFabricLL, OneShotFourRanksLatePeerFoldsInRankOrder) {
  runLatePeer<4>(4);
}

TEST_F(DdaAllReduceFabricLL, OneShotEightRanksLatePeerFoldsInRankOrder) {
  runLatePeer<8>(8);
}

// Two ranks on the runtime-rank kernel, one polled peer line: the kernel waits for the late peer, then folds in order.
TEST_F(DdaAllReduceFabricLL, OneShotTwoRanksLatePeerCompletes) {
  runLatePeer<0>(2);
}

// Six ranks on the runtime-rank kernel: one sweep of 8 lines with 5 polled, so a waiting lane spins on one stale line.
TEST_F(DdaAllReduceFabricLL, OneShotRuntimeRanksLatePeerFoldsInRankOrder) {
  runLatePeer<0>(6);
}

TEST_F(DdaAllReduceFabricLL, OneShotTornLineWaitsForBothHalves) {
  nRanks_ = 6;
  const int torn = 3;
  for (int peer = 0; peer < nRanks_; ++peer) {
    if (peer != kSelf && peer != torn) {
      publishPeer(peer);
    }
  }
  publishHalf(torn, 1);
  ASSERT_NO_FATAL_FAILURE(launchLL<float>(&dda::common::ddaAllReduceFlatLL<float, 0>, oneShotGrid(), dSend_));
  expectStillPolling("finished with one half of a line stale");
  publishHalf(torn, 0);
  ASSERT_TRUE(finishes()) << "kernel never saw the second half land";
  expectRankOrderSum();
}

// An abort raised while the spin-one wait holds a stale line leaves recvbuff untouched and still bumps the epoch.
TEST_F(DdaAllReduceFabricLL, OneShotAbortDuringStaleBatchLeavesRecvbuffAndAdvancesEpoch) {
  nRanks_ = 6;
  publishPeer(0);
  publishPeer(2);
  ASSERT_NO_FATAL_FAILURE(launchLL<float>(&dda::common::ddaAllReduceFlatLL<float, 0>, oneShotGrid(), dSend_));
  expectStillPolling("finished with three peers stale");
  *peers_.AbortFlag() = 1;
  ASSERT_TRUE(finishes()) << "kernel ignored the abort flag";
  for (float v : recv()) {
    ASSERT_EQ(v, kUntouched) << "an aborted thread wrote recvbuff";
  }
  expectEpochAdvanced();
}

// Float, bf16 and half, compile-time and runtime ranks, even (6) and uneven (5) shards: the one-shot's result bits.
TEST_F(DdaAllReduceFabricLL, SpreadMatchesOneShotBitwise) {
  spreadCase<float, 4>(4, kSelf, kOneShotBytes, 3);
  spreadCase<float, 8>(8, kSelf, kOneShotBytes, 3);
  spreadCase<float, 0>(6, kSelf, kOneShotBytes, 3);
  spreadCase<float, 0>(5, kSelf, kOneShotBytes, 3);
  spreadCase<bf16, 0>(6, kSelf, kOneShotBytes, 3);
  spreadCase<bf16, 0>(5, kSelf, kOneShotBytes, 3);
  spreadCase<half, 0>(5, kSelf, kOneShotBytes, 3);
}

// Two LDS passes on one block (12 ranks also fold two batches), idle blocks, and an empty or a one-packet self shard.
TEST_F(DdaAllReduceFabricLL, SpreadPassesIdleBlocksAndShortShards) {
  spreadCase<float, 4>(4, kSelf, kBufBytes, 1);  // 768-packet shards, 512 a pass
  spreadCase<float, 0>(12, kSelf, kBufBytes, 1);  // 256-packet shards, 170 a pass
  spreadCase<float, 0>(12, 11, 400, 8);  // 50 packets in shards of 5: ranks 10 and 11 own none, blocks 5 to 7 idle
  spreadCase<float, 0>(12, 11, 448, 2);  // 56 packets: my shard is one packet, so block 1 reduces nothing
}

// Rank 0's copy of my shard lands only once the kernel is polling (phase 2), then a peer's reduced shard (phase 3).
// On one block the held lines sit in 2- or 3-line masks, so the late reduced shard waits in the batched sweep.
TEST_F(DdaAllReduceFabricLL, SpreadWaitsForLateCopiesAndLateReducedShards) {
  spreadCase<float, 0>(6, kSelf, kOneShotBytes, 3, kCopyArea);
  spreadCase<float, 0>(5, kSelf, kOneShotBytes, 1, kReducedArea);
}

// A phase-2 abort writes and publishes no partial sum; a phase-3 abort leaves the missing shard untouched.
TEST_F(DdaAllReduceFabricLL, SpreadAbortPublishesNoPartialSumAndLeavesMissingShards) {
  spreadCase<float, 0>(6, kSelf, kOneShotBytes, 3, kCopyArea, kAbort);
  spreadCase<float, 0>(6, kSelf, kOneShotBytes, 3, kReducedArea, kAbort);
}

// One block on 4 ranks: pass 0 (packets 0 to 511) finishes, pass 1 waits on rank 0's later copies, then aborts.
TEST_F(DdaAllReduceFabricLL, SpreadAbortInALaterPassKeepsTheFinishedPass) {
  heldFrom_ = dda::common::ddaLLSpreadPassPkts(4);
  spreadCase<float, 4>(4, kSelf, kBufBytes, 1, kCopyArea, kAbort);
}

// In place: the copies of the peers' shards leave before their reduced shards overwrite the same recvbuff range.
TEST_F(DdaAllReduceFabricLL, SpreadInPlaceSendsBeforeOverwriting) {
  ASSERT_NO_FATAL_FAILURE((spreadReference<float, 0>(6, kSelf, kOneShotBytes, 1)));
  reset();
  HIP_CHECK(hipMemcpy(dRecv_, in_[self_].data(), kOneShotBytes, hipMemcpyHostToDevice));
  ASSERT_NO_FATAL_FAILURE((spreadRunAndCheck<float, 0>(3, kHoldNothing, kNoAbort, dRecv_)));
}

// No reset after the one-shot: flag 3 on bank 1 at the one-shot's grid, then flag 4 on bank 0 over its flag-2 lines.
// Flag 4 holds a copy late on one block: the held lines sit in 2- or 3-line masks and meet the sweep's exact flag test.
TEST_F(DdaAllReduceFabricLL, SpreadFollowsTheOneShotAndSkipsStaleLines) {
  ASSERT_NO_FATAL_FAILURE((spreadReference<float, 0>(5, kSelf, kOneShotBytes, 1)));
  for (const uint32_t flag : {kFlag + 1, kFlag + 2}) {
    const unsigned grid = flag == kFlag + 1 ? oneShotGrid() : 1;
    SCOPED_TRACE(::testing::Message() << "flag " << flag << ", grid " << grid);
    flag_ = flag;
    clearRecv();
    const int held = flag == kFlag + 1 ? kHoldNothing : kCopyArea;
    ASSERT_NO_FATAL_FAILURE((spreadRunAndCheck<float, 0>(grid, held, kNoAbort, dSend_)));
  }
}

}  // namespace RcclUnitTesting
