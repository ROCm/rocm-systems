/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ddaAllReduceFlatLL on one GPU: the host plays the peers by writing LL lines into coherent
// host scratch. Built twice: rccl-UnitTestsFixtures defines DWORDX4_INTRINSICS_FORCE_OFF (dword re-read),
// rccl-UnitTests not (b128).

#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "DdaLLHostPeers.hpp"
#include "DeviceTestBase.hpp"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting {

namespace {

using dda::common::DdaKernelParams;
using dda::common::LLPacket16;

constexpr int kMaxRanks = 8;
constexpr int kSelf = 1;
constexpr size_t kCount = 3072;  // floats: 1536 packets, 3 blocks of 512
constexpr unsigned kOneShotThreads = 512;
constexpr size_t kBankBytes = (size_t)kMaxRanks * 64 * 1024;
constexpr uint32_t kFlag = 2;  // epoch cells start at 1, so the kernels use flag 2 and bank 0
constexpr unsigned kEpochCells = 16;
constexpr float kUntouched = -1.0f;
constexpr auto kWatchdog = std::chrono::seconds(30);
constexpr auto kSettle = std::chrono::milliseconds(20);  // lets every thread finish its first read of the late lines

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

  // Line `pk` of sender's slot in owner's scratch.
  volatile uint32_t* line(int owner, int sender, size_t pk, size_t slotPkts) {
    LLPacket16* base = static_cast<LLPacket16*>(peers_.Host(owner));
    return reinterpret_cast<volatile uint32_t*>(&base[sender * slotPkts + pk]);
  }

  size_t oneShotSlot() const { return dda::common::ddaLLArSlotPkts(kBankBytes, nRanks_); }
  size_t packets() const { return kCount * sizeof(float) / 8; }

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

  template <int N>
  void launchOneShot() {
    const unsigned blocks = (packets() + kOneShotThreads - 1) / kOneShotThreads;
    hipLaunchKernelGGL((dda::common::ddaAllReduceFlatLL<float, N>), dim3(blocks), dim3(kOneShotThreads), 0, stream_,
                       peers_.DevPeers<float>(), dRecv_, dSend_, kCount, kSelf, nRanks_, dEpoch_, (int)kEpochCells,
                       kBankBytes, DdaKernelParams{peers_.AbortFlag()});
    ASSERT_EQ(hipGetLastError(), hipSuccess);
  }

  // True once every thread has stored its phase-1 line to the next peer, so nearly all of them are already polling.
  bool phase1Done() {
    const int peer = (kSelf + 1) % nRanks_;
    for (size_t pk = 0; pk < packets(); ++pk) {
      volatile uint32_t* w = line(peer, kSelf, pk, oneShotSlot());
      if (w[1] != kFlag || w[3] != kFlag) {
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

  void expectEpochAdvanced(unsigned blocks) {
    std::vector<uint32_t> epoch(kEpochCells);
    ASSERT_EQ(hipMemcpy(epoch.data(), dEpoch_, kEpochCells * sizeof(uint32_t), hipMemcpyDeviceToHost), hipSuccess);
    for (unsigned b = 0; b < blocks; ++b) {
      EXPECT_EQ(epoch[b], kFlag) << "block " << b;
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
    launchOneShot<N>();
    expectStillPolling("finished before rank 0 published");
    publishPeer(0);
    ASSERT_TRUE(finishes()) << "kernel never saw the late peer";
    expectRankOrderSum();
    expectEpochAdvanced((packets() + kOneShotThreads - 1) / kOneShotThreads);
  }

  DdaLLHostPeers peers_;
  int nRanks_ = 6;
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
  launchOneShot<0>();
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
  launchOneShot<0>();
  expectStillPolling("finished with three peers stale");
  *peers_.AbortFlag() = 1;
  ASSERT_TRUE(finishes()) << "kernel ignored the abort flag";
  for (float v : recv()) {
    ASSERT_EQ(v, kUntouched) << "an aborted thread wrote recvbuff";
  }
  expectEpochAdvanced((packets() + kOneShotThreads - 1) / kOneShotThreads);
}

}  // namespace RcclUnitTesting
