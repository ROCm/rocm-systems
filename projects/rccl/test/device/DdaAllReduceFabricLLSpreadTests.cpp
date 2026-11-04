/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ddaAllReduceSpreadTwoShotLL on one GPU: the host plays the peers by writing LL lines into coherent host scratch, both
// the copies of the self shard (first area) and the peers' reduced shards (second area).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "DdaLLHostPeers.hpp"
#include "DeviceTestBase.hpp"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll_spread.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting {

namespace {

using dda::common::DdaKernelParams;
using dda::common::LLPacket16;

constexpr int kMaxRanks = 8;
constexpr int kSelf = 1;
constexpr size_t kCount = 3072;  // floats: 1536 packets, the one-shot grid of 3 blocks of 512
constexpr unsigned kThreads = dda::common::kDdaLLSpreadThreads;
constexpr size_t kBankBytes = (size_t)kMaxRanks * 64 * 1024;
constexpr uint32_t kFlag = 2;  // epoch cells start at 1, so the kernel uses flag 2 and bank 0
constexpr unsigned kEpochCells = 16;
constexpr float kUntouched = -1.0f;
constexpr auto kWatchdog = std::chrono::seconds(30);
constexpr auto kSettle = std::chrono::milliseconds(20);

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

float rankOrderSum(int nRanks) {
  return (float)((nRanks - 2) * (nRanks - 1) / 2);
}

// What peer p publishes as its reduced shard; distinct per peer so a misplaced shard shows.
float reducedValue(int peer) {
  return 1000.0f + (float)peer;
}

uint32_t bitsOf(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

class DdaAllReduceFabricLLSpread : public DeviceTestBase {
 protected:
  void SetUp() override {
    DeviceTestBase::SetUp();
    peers_.Init(kMaxRanks, 2 * kBankBytes);
    const std::vector<float> send(kCount, rankValue(kSelf));
    HIP_CHECK(hipMalloc(&dSend_, kCount * sizeof(float)));
    HIP_CHECK(hipMemcpy(dSend_, send.data(), kCount * sizeof(float), hipMemcpyHostToDevice));
    const std::vector<float> recv(kCount, kUntouched);
    HIP_CHECK(hipMalloc(&dRecv_, kCount * sizeof(float)));
    HIP_CHECK(hipMemcpy(dRecv_, recv.data(), kCount * sizeof(float), hipMemcpyHostToDevice));
    const std::vector<uint32_t> epoch(kEpochCells, kFlag - 1);
    HIP_CHECK(hipMalloc(&dEpoch_, kEpochCells * sizeof(uint32_t)));
    HIP_CHECK(hipMemcpy(dEpoch_, epoch.data(), kEpochCells * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
  }

  void TearDown() override {
    HIP_EXPECT(hipStreamDestroy(stream_));
    HIP_EXPECT(hipFree(dEpoch_));
    HIP_EXPECT(hipFree(dRecv_));
    HIP_EXPECT(hipFree(dSend_));
    peers_.Release();
  }

  size_t slot() const { return dda::common::ddaLLArTwoShotSlotPkts(kBankBytes, nRanks_); }
  size_t packets() const { return kCount * sizeof(float) / 8; }
  // Shards of ceil(packets / nRanks) packets; the last ones are shorter when nRanks does not divide the message.
  size_t shardPkts() const { return (packets() + nRanks_ - 1) / nRanks_; }
  size_t shardLen(int rank) const {
    const size_t first = (size_t)rank * shardPkts();
    return first < packets() ? std::min(shardPkts(), packets() - first) : 0;
  }
  unsigned blocks() const { return (unsigned)((packets() + kThreads - 1) / kThreads); }

  // Line pk of sender's slot in owner's scratch, staging area 0 (copies of owner's shard) or 1 (reduced shards).
  volatile uint32_t* line(int owner, int area, int sender, size_t pk) {
    LLPacket16* base = static_cast<LLPacket16*>(peers_.Host(owner));
    return reinterpret_cast<volatile uint32_t*>(&base[(size_t)area * nRanks_ * slot() + sender * slot() + pk]);
  }

  void publishCopy(int peer) {
    for (size_t pk = 0; pk < shardLen(kSelf); ++pk) {
      DdaLLHostPeers::PublishLine(line(kSelf, 0, peer, pk), bitsOf(rankValue(peer)), bitsOf(rankValue(peer)), kFlag);
    }
  }

  void publishReduced(int peer) {
    for (size_t pk = 0; pk < shardLen(peer); ++pk) {
      DdaLLHostPeers::PublishLine(line(kSelf, 1, peer, pk), bitsOf(reducedValue(peer)), bitsOf(reducedValue(peer)),
                                  kFlag);
    }
  }

  void publishAll() {
    for (int peer = 0; peer < nRanks_; ++peer) {
      if (peer != kSelf) {
        publishCopy(peer);
        publishReduced(peer);
      }
    }
  }

  template <int N>
  void launch() {
    hipLaunchKernelGGL((dda::common::ddaAllReduceSpreadTwoShotLL<float, N>), dim3(blocks()), dim3(kThreads), 0, stream_,
                       peers_.DevPeers<float>(), dRecv_, dSend_, kCount, kSelf, nRanks_, dEpoch_, (int)kEpochCells,
                       kBankBytes, DdaKernelParams{peers_.AbortFlag()});
    ASSERT_EQ(hipGetLastError(), hipSuccess);
  }

  // True once every line of the self copy bound for the next peer has landed there.
  bool phase1Done() {
    const int peer = (kSelf + 1) % nRanks_;
    for (size_t pk = 0; pk < shardLen(peer); ++pk) {
      volatile uint32_t* w = line(peer, 0, kSelf, pk);
      if (w[1] != kFlag || w[3] != kFlag) {
        return false;
      }
    }
    return true;
  }

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

  // Every peer got the self copy of its shard in area 0 and the rank-order sum in area 1, each under the flag.
  void expectPublished() {
    const uint32_t sum = bitsOf(rankOrderSum(nRanks_));
    const uint32_t mine = bitsOf(rankValue(kSelf));
    for (int peer = 0; peer < nRanks_; ++peer) {
      if (peer == kSelf) {
        continue;
      }
      for (size_t pk = 0; pk < shardLen(peer); ++pk) {
        volatile uint32_t* c = line(peer, 0, kSelf, pk);
        ASSERT_TRUE(c[0] == mine && c[1] == kFlag && c[2] == mine && c[3] == kFlag)
            << "copy to " << peer << " pk " << pk;
      }
      for (size_t pk = 0; pk < shardLen(kSelf); ++pk) {
        volatile uint32_t* r = line(peer, 1, kSelf, pk);
        ASSERT_TRUE(r[0] == sum && r[1] == kFlag && r[2] == sum && r[3] == kFlag) << "sum to " << peer << " pk " << pk;
      }
    }
  }

  // The self shard holds the rank-order sum; every other shard holds that peer's reduced value.
  void expectRecv() {
    const std::vector<float> out = recv();
    for (size_t i = 0; i < kCount; ++i) {
      const int owner = (int)(i / 2 / shardPkts());
      const float want = owner == kSelf ? rankOrderSum(nRanks_) : reducedValue(owner);
      ASSERT_EQ(bitsOf(out[i]), bitsOf(want)) << "element " << i << " of shard " << owner;
    }
  }

  void expectEpochAdvanced() {
    std::vector<uint32_t> epoch(kEpochCells);
    ASSERT_EQ(hipMemcpy(epoch.data(), dEpoch_, kEpochCells * sizeof(uint32_t), hipMemcpyDeviceToHost), hipSuccess);
    for (unsigned b = 0; b < blocks(); ++b) {
      EXPECT_EQ(epoch[b], kFlag) << "block " << b;
    }
  }

  // Rank 0, first in fold order, publishes its copy only once the kernel is polling.
  template <int N>
  void runLatePeer(int nRanks) {
    nRanks_ = nRanks;
    for (int peer = 0; peer < nRanks_; ++peer) {
      if (peer != kSelf) {
        publishReduced(peer);
        if (peer != 0) {
          publishCopy(peer);
        }
      }
    }
    launch<N>();
    expectStillPolling("finished before rank 0 published");
    publishCopy(0);
    ASSERT_TRUE(finishes()) << "kernel never saw the late peer";
    expectRecv();
    expectPublished();
    expectEpochAdvanced();
  }

  DdaLLHostPeers peers_;
  int nRanks_ = 6;
  float* dSend_ = nullptr;
  float* dRecv_ = nullptr;
  uint32_t* dEpoch_ = nullptr;
  hipStream_t stream_ = nullptr;
};

}  // namespace

TEST_F(DdaAllReduceFabricLLSpread, FourRanksLatePeerFoldsInRankOrder) {
  runLatePeer<4>(4);
}

TEST_F(DdaAllReduceFabricLLSpread, EightRanksLatePeerFoldsInRankOrder) {
  runLatePeer<8>(8);
}

// Six ranks on the runtime-rank kernel: 256 packets per shard, so each block owns a partial slice of every shard.
TEST_F(DdaAllReduceFabricLLSpread, RuntimeRanksLatePeerFoldsInRankOrder) {
  runLatePeer<0>(6);
}

// Five ranks do not divide the 1536 packets: four shards of 308 and one of 304, each on its own length.
TEST_F(DdaAllReduceFabricLLSpread, UnevenShardsLatePeerFoldsInRankOrder) {
  runLatePeer<0>(5);
}

// In place: the self copies of the peers' shards leave before the reduced shards overwrite the same recvbuff range.
TEST_F(DdaAllReduceFabricLLSpread, InPlaceSendsBeforeOverwriting) {
  nRanks_ = 6;
  HIP_CHECK(hipMemcpy(dRecv_, dSend_, kCount * sizeof(float), hipMemcpyDeviceToDevice));
  float* const savedSend = dSend_;
  dSend_ = dRecv_;
  publishAll();
  launch<0>();
  ASSERT_TRUE(finishes()) << "in-place kernel did not finish";
  dSend_ = savedSend;
  expectRecv();
  expectPublished();
}

// An abort while rank 0's copy is missing: no partial sum is published and the peer shards of recvbuff stay untouched.
TEST_F(DdaAllReduceFabricLLSpread, AbortBeforeFoldPublishesNoPartialSum) {
  nRanks_ = 6;
  for (int peer = 2; peer < nRanks_; ++peer) {
    publishCopy(peer);
  }
  launch<0>();
  expectStillPolling("finished with rank 0 missing");
  *peers_.AbortFlag() = 1;
  ASSERT_TRUE(finishes()) << "kernel ignored the abort flag";
  for (int peer = 0; peer < nRanks_; ++peer) {
    if (peer == kSelf) {
      continue;
    }
    for (size_t pk = 0; pk < shardLen(kSelf); ++pk) {
      volatile uint32_t* r = line(peer, 1, kSelf, pk);
      ASSERT_NE(r[1], kFlag) << "partial sum published to " << peer;
    }
  }
  const std::vector<float> out = recv();
  for (size_t i = 0; i < kCount; ++i) {
    if ((int)(i / 2 / shardPkts()) != kSelf) {
      ASSERT_EQ(out[i], kUntouched) << "an aborted thread wrote a peer shard";
    }
  }
  expectEpochAdvanced();
}

}  // namespace RcclUnitTesting
