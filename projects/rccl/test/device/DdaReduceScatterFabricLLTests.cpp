/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ddaReduceScatterFabricLL and ddaReduceScatterFabricSpreadLL on one GPU: the host plays the peers by writing LL lines
// into coherent host scratch. Built twice: rccl-UnitTestsFixtures defines DWORDX4_INTRINSICS_FORCE_OFF (dword re-read),
// rccl-UnitTests not (b128).

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "DdaLLHostPeers.hpp"
#include "DeviceTestBase.hpp"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll.h"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll_spread.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting {

namespace {

using dda::common::DdaKernelParams;
using dda::common::LLPacket16;

constexpr int kRanks = 6;  // five peers: one full receive batch, then a batch with one lane in range
constexpr int kSelf = 1;  // peers 2 to 5 fill the first batch and peer 0 wraps into the second
constexpr size_t kRecvCount = 2050;  // floats per shard; 1025 packets, so the last block is partial
constexpr size_t kPackets = kRecvCount * sizeof(float) / 8;
constexpr unsigned kThreads = 256;  // the production launcher's block size
constexpr unsigned kBlocks = (kPackets + kThreads - 1) / kThreads;
constexpr uint32_t kFlag = 2;  // epoch cells start at 1, so the kernel uses flag 2 and bank 0
constexpr float kUntouched = -1.0f;
constexpr auto kWatchdog = std::chrono::seconds(30);
constexpr auto kSettle = std::chrono::milliseconds(20);  // lets every thread finish its first read of the late lines
constexpr auto kPrefetchFaultWait = std::chrono::milliseconds(500);  // a prefetch VM fault is reported asynchronously
#if defined(DWORDX4_INTRINSICS_FORCE_OFF)
constexpr bool kForceOff = true;
#else
constexpr bool kForceOff = false;
#endif

__global__ void noOp() {}

// Writes the device-side RCCL_HAVE_SYS_B128 and whether the toolchain offers either family of b128 builtins.
__global__ void reportReReadPath(int* out) {
  out[0] = RCCL_HAVE_SYS_B128;
#if (__has_builtin(__builtin_amdgcn_av_load_b128) && __has_builtin(__builtin_amdgcn_av_store_b128)) || \
  (__has_builtin(__builtin_amdgcn_global_load_b128) && __has_builtin(__builtin_amdgcn_global_store_b128))
  out[1] = 1;
#else
  out[1] = 0;
#endif
}

float sendValue(int rank, size_t i) {
  return static_cast<float>(rank * 4096 + static_cast<int>(i % 4096));
}

float peerValue(int peer, size_t i) {
  return static_cast<float>((peer + 1) * 3 + static_cast<int>(i % 977));
}

uint32_t bitsOf(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

class DdaReduceScatterFabricLL : public DeviceTestBase {
protected:
  void SetUp() override {
    DeviceTestBase::SetUp();
    peers_.Init(kRanks, dda::common::ddaLLRsBankBytes(kRanks));

    std::vector<float> send(kRecvCount * kRanks);
    for (int r = 0; r < kRanks; ++r) {
      for (size_t i = 0; i < kRecvCount; ++i) {
        send[r * kRecvCount + i] = sendValue(r, i);
      }
    }
    HIP_CHECK(hipMalloc(&dSend_, send.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(dSend_, send.data(), send.size() * sizeof(float), hipMemcpyHostToDevice));
    const std::vector<float> recv(kRecvCount, kUntouched);
    HIP_CHECK(hipMalloc(&dRecv_, kRecvCount * sizeof(float)));
    HIP_CHECK(hipMemcpy(dRecv_, recv.data(), kRecvCount * sizeof(float), hipMemcpyHostToDevice));
    const std::vector<uint32_t> epoch(kBlocks, kFlag - 1);
    HIP_CHECK(hipMalloc(&dEpoch_, kBlocks * sizeof(uint32_t)));
    HIP_CHECK(hipMemcpy(dEpoch_, epoch.data(), kBlocks * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
  }

  void TearDown() override {
    HIP_EXPECT(hipStreamDestroy(stream_));
    HIP_EXPECT(hipFree(dEpoch_));
    HIP_EXPECT(hipFree(dRecv_));
    HIP_EXPECT(hipFree(dSend_));
    peers_.Release();
  }

  volatile uint32_t* line(int owner, int sender, size_t pk) {
    LLPacket16* base = static_cast<LLPacket16*>(peers_.Host(owner));
    return reinterpret_cast<volatile uint32_t*>(&base[sender * dda::common::kDdaLLRsSlotStridePkts + pk]);
  }

  // Writes the payload before the flags, as one 16B store would make them visible together to the poller.
  void publishPeer(int peer) {
    for (size_t pk = 0; pk < kPackets; ++pk) {
      DdaLLHostPeers::PublishLine(line(kSelf, peer, pk), bitsOf(peerValue(peer, 2 * pk)),
                                  bitsOf(peerValue(peer, 2 * pk + 1)), kFlag);
    }
  }

  // Publishes one 8B half of every line of a peer: half 0 is payload word 0 and flag 1, half 1 is word 2 and flag 3.
  void publishHalf(int peer, int half) {
    for (size_t pk = 0; pk < kPackets; ++pk) {
      DdaLLHostPeers::PublishHalf(line(kSelf, peer, pk), half, bitsOf(peerValue(peer, 2 * pk + half)), kFlag);
    }
  }

  // One peer's lines carry only one fresh half, so the kernel must keep waiting until the other half lands.
  void checkTornLineWaits(int freshHalf) {
    const int torn = (kSelf + 2) % kRanks;
    for (int r = 1; r < kRanks; ++r) {
      if ((kSelf + r) % kRanks != torn) {
        publishPeer((kSelf + r) % kRanks);
      }
    }
    publishHalf(torn, freshHalf);
    launch();
    const auto deadline = std::chrono::steady_clock::now() + kWatchdog;
    while (!phase1Done() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    std::this_thread::sleep_for(kSettle);
    ASSERT_EQ(hipStreamQuery(stream_), hipErrorNotReady) << "kernel finished with only half " << freshHalf << " fresh";
    publishHalf(torn, 1 - freshHalf);
    ASSERT_TRUE(finishes()) << "kernel never saw the second half land";
    expectReduced();
  }

  // True once every thread has stored its first phase-1 line, so nearly all of them are already polling.
  bool phase1Done() {
    const int peer = (kSelf + 1) % kRanks;
    for (size_t pk = 0; pk < kPackets; ++pk) {
      volatile uint32_t* w = line(peer, kSelf, pk);
      if (w[1] != kFlag || w[3] != kFlag) {
        return false;
      }
    }
    return true;
  }

  void launch() {
    hipLaunchKernelGGL((dda::common::ddaReduceScatterFabricLL<float, 0>), dim3(kBlocks), dim3(kThreads), 0, stream_,
                       peers_.DevPeers<float>(), dRecv_, dSend_, kRecvCount, kSelf, kRanks, dEpoch_,
                       static_cast<int>(kBlocks), dda::common::ddaLLRsBankBytes(kRanks),
                       DdaKernelParams{peers_.AbortFlag()});
    ASSERT_EQ(hipGetLastError(), hipSuccess);
  }

  // Past the watchdog, raises abort and publishes every peer, so a lost wake-up or an ignored abort fails, not hangs.
  bool finishes() {
    return peers_.Finishes(
      stream_,
      [this] {
        for (int r = 1; r < kRanks; ++r) {
          publishPeer((kSelf + r) % kRanks);
        }
      },
      kWatchdog);
  }

  std::vector<float> recv() {
    std::vector<float> out(kRecvCount);
    EXPECT_EQ(hipMemcpy(out.data(), dRecv_, kRecvCount * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
    return out;
  }

  void expectEpochAdvanced() {
    std::vector<uint32_t> epoch(kBlocks);
    ASSERT_EQ(hipMemcpy(epoch.data(), dEpoch_, kBlocks * sizeof(uint32_t), hipMemcpyDeviceToHost), hipSuccess);
    for (unsigned b = 0; b < kBlocks; ++b) {
      EXPECT_EQ(epoch[b], kFlag) << "block " << b;
    }
  }

  void expectReduced() {
    const std::vector<float> out = recv();
    for (size_t i = 0; i < kRecvCount; ++i) {
      float want = sendValue(kSelf, i);
      for (int r = 1; r < kRanks; ++r) {
        want += peerValue((kSelf + r) % kRanks, i);
      }
      ASSERT_EQ(out[i], want) << "element " << i;
    }
    for (int r = 1; r < kRanks; ++r) {
      const int peer = (kSelf + r) % kRanks;
      for (size_t pk = 0; pk < kPackets; ++pk) {
        volatile uint32_t* w = line(peer, kSelf, pk);
        ASSERT_EQ(w[0], bitsOf(sendValue(peer, 2 * pk))) << "peer " << peer << " packet " << pk;
        ASSERT_EQ(w[2], bitsOf(sendValue(peer, 2 * pk + 1))) << "peer " << peer << " packet " << pk;
      }
    }
    expectEpochAdvanced();
  }

  DdaLLHostPeers peers_;
  float* dSend_ = nullptr;
  float* dRecv_ = nullptr;
  uint32_t* dEpoch_ = nullptr;
  hipStream_t stream_ = nullptr;
};

// The spread kernel's tests run at kRanks too unless they call useRanks(), on more epoch cells than their widest grid.
constexpr unsigned kEpochCells = 32;
constexpr size_t kShardBytes = kRecvCount * sizeof(float);
constexpr size_t kShardWords = kShardBytes / sizeof(uint32_t);
constexpr int kLastRank = kRanks - 1;  // every peer of the last rank wraps past rank 0
constexpr unsigned kOneBlock = 1;  // all 1025 packets in 7 LDS passes of 170
constexpr unsigned kThreeBlocks = 3;  // slices of 344 packets, the last one partial
constexpr unsigned kLaunchGrid = 22;  // the launcher's grid: 25 blocks for 6150 packets, trimmed to 48-packet slices
constexpr unsigned kIdleGrid = 25;  // the same grid untrimmed, so blocks 22 to 24 own no packets
static_assert(dda::common::kDdaLLRsSpreadThreads == kThreads, "the spread kernel runs on the shard kernel's blocks");

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

// ddaReduceScatterFabricSpreadLL against the shard kernel: the same lines and sendbuff must give the same result bits.
class DdaReduceScatterFabricSpreadLL : public DeviceTestBase {
protected:
  void SetUp() override {
    DeviceTestBase::SetUp();
    peers_.Init(nRanks_, dda::common::ddaLLRsBankBytes(nRanks_));
    HIP_CHECK(hipMalloc(&dSend_, nRanks_ * kShardBytes));
    HIP_CHECK(hipMalloc(&dRecv_, kShardBytes));
    HIP_CHECK(hipMalloc(&dEpoch_, kEpochCells * sizeof(uint32_t)));
    HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
  }

  void TearDown() override {
    HIP_EXPECT(hipStreamDestroy(stream_));
    HIP_EXPECT(hipFree(dEpoch_));
    HIP_EXPECT(hipFree(dRecv_));
    HIP_EXPECT(hipFree(dSend_));
    peers_.Release();
  }

  // Rebuilds every rank's scratch and my sendbuff for nRanks ranks.
  void useRanks(int nRanks) {
    peers_.Release();
    HIP_CHECK(hipFree(dSend_));
    nRanks_ = nRanks;
    peers_.Init(nRanks_, dda::common::ddaLLRsBankBytes(nRanks_));
    HIP_CHECK(hipMalloc(&dSend_, nRanks_ * kShardBytes));
  }

  volatile uint32_t* line(int owner, int sender, size_t pk) {
    LLPacket16* base = static_cast<LLPacket16*>(peers_.Host(owner));
    return reinterpret_cast<volatile uint32_t*>(&base[sender * dda::common::kDdaLLRsSlotStridePkts + pk]);
  }

  // Random words, so a misplaced packet or a reordered fold changes the result bits. Halves pack two 16-bit values.
  void randomInputs(bool halves, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> value(-1.0f, 1.0f);
    std::uniform_int_distribution<int> scale(-12, 12);
    auto draw = [&] { return bitsOf(std::ldexp(value(rng), scale(rng))); };
    auto word = [&] {
      const uint32_t lo = draw();
      return halves ? (lo >> 16) | (draw() & 0xffff0000u) : lo;
    };
    send_.resize(nRanks_ * kShardWords);
    for (uint32_t& w : send_) {
      w = word();
    }
    copy_.assign(nRanks_, std::vector<uint32_t>(kShardWords));
    for (std::vector<uint32_t>& words : copy_) {
      for (uint32_t& w : words) {
        w = word();
      }
    }
  }

  // Cleared lines, a clear abort flag, epoch cells one before kFlag, sendbuff from send_ and an untouched recvbuff.
  void reset() {
    for (int owner = 0; owner < nRanks_; ++owner) {
      for (int sender = 0; sender < nRanks_; ++sender) {
        std::memset(const_cast<uint32_t*>(line(owner, sender, 0)), 0, kPackets * sizeof(LLPacket16));
      }
    }
    *peers_.AbortFlag() = 0;
    const std::vector<uint32_t> epoch(kEpochCells, kFlag - 1);
    HIP_CHECK(hipMemcpy(dEpoch_, epoch.data(), kEpochCells * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dSend_, send_.data(), nRanks_ * kShardBytes, hipMemcpyHostToDevice));
    const std::vector<uint32_t> recv(kShardWords, bitsOf(kUntouched));
    HIP_CHECK(hipMemcpy(dRecv_, recv.data(), kShardBytes, hipMemcpyHostToDevice));
  }

  // The peer's copy of my shard, packets below end, into its slot in my scratch.
  void publishPeer(int peer, size_t end = SIZE_MAX) {
    for (size_t pk = 0; pk < kPackets && pk < end; ++pk) {
      DdaLLHostPeers::PublishLine(line(self_, peer, pk), copy_[peer][2 * pk], copy_[peer][2 * pk + 1], kFlag);
    }
  }

  void publishHalf(int peer, int half) {
    for (size_t pk = 0; pk < kPackets; ++pk) {
      DdaLLHostPeers::PublishHalf(line(self_, peer, pk), half, copy_[peer][2 * pk + half], kFlag);
    }
  }

  void publishAll(size_t end = SIZE_MAX) {
    for (int peer = 0; peer < nRanks_; ++peer) {
      if (peer != self_) {
        publishPeer(peer, end);
      }
    }
  }

  // True once my lines to the last peer, the last ones each block sends, are stored, so nearly every thread polls.
  bool phase1Done() {
    const int peer = (self_ + nRanks_ - 1) % nRanks_;
    for (size_t pk = 0; pk < kPackets; ++pk) {
      volatile uint32_t* w = line(peer, self_, pk);
      if (w[1] != kFlag || w[3] != kFlag) {
        return false;
      }
    }
    return true;
  }

  // Waits for phase 1, then expects the kernel still polling.
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

  // Launches kernel over T as rank self_ on grid blocks of kThreads, with recvbuff at recv.
  template <typename T, typename Kernel>
  void launch(Kernel kernel, unsigned grid, uint32_t* recv) {
    kernel<<<dim3(grid), dim3(kThreads), 0, stream_>>>(
      peers_.DevPeers<T>(), reinterpret_cast<T*>(recv), reinterpret_cast<const T*>(dSend_), kShardBytes / sizeof(T),
      self_, nRanks_, dEpoch_, static_cast<int>(kEpochCells), dda::common::ddaLLRsBankBytes(nRanks_),
      DdaKernelParams{peers_.AbortFlag()});
    ASSERT_EQ(hipGetLastError(), hipSuccess);
  }

  void launchSpread(unsigned grid, uint32_t* recv) {
    launch<float>(&dda::common::ddaReduceScatterFabricSpreadLL<float, 0>, grid, recv);
  }

  std::vector<uint32_t> words(const uint32_t* d) {
    std::vector<uint32_t> out(kShardWords);
    EXPECT_EQ(hipMemcpy(out.data(), d, kShardBytes, hipMemcpyDeviceToHost), hipSuccess);
    return out;
  }

  // Fresh inputs as rank self, ref_ from the shard kernel on its own grid, then a reset() for the kernel under test.
  template <typename T>
  void reference(int self, uint32_t seed) {
    self_ = self;
    randomInputs(sizeof(T) == 2, seed);
    reset();
    publishAll();
    ASSERT_NO_FATAL_FAILURE(launch<T>(&dda::common::ddaReduceScatterFabricLL<T, 0>, kBlocks, dRecv_));
    ASSERT_TRUE(peers_.Finishes(stream_, [] {}, kWatchdog)) << "shard kernel reference did not finish";
    ref_ = words(dRecv_);
    reset();
  }

  // The shard kernel's result bits below packet validPkts, untouched words from there.
  void expectRecv(const uint32_t* d, size_t validPkts) {
    const std::vector<uint32_t> out = words(d);
    for (size_t i = 0; i < kShardWords; ++i) {
      if (i / 2 < validPkts) {
        ASSERT_EQ(out[i], ref_[i]) << "word " << i;
      } else {
        ASSERT_EQ(out[i], bitsOf(kUntouched)) << "an aborted block wrote word " << i;
      }
    }
  }

  // Every peer's slot for me holds my sendbuff shard for that peer, flagged, packet for packet.
  void expectSent() {
    for (int peer = 0; peer < nRanks_; ++peer) {
      if (peer == self_) {
        continue;
      }
      for (size_t pk = 0; pk < kPackets; ++pk) {
        volatile uint32_t* w = line(peer, self_, pk);
        const size_t s = static_cast<size_t>(peer) * kShardWords + 2 * pk;
        ASSERT_TRUE(w[0] == send_[s] && w[1] == kFlag && w[2] == send_[s + 1] && w[3] == kFlag)
          << "peer " << peer << " packet " << pk;
      }
    }
  }

  // Every block, idle or aborted, bumps the cells it owns, so all kEpochCells cells end at kFlag.
  void expectEpochAdvanced() {
    std::vector<uint32_t> epoch(kEpochCells);
    ASSERT_EQ(hipMemcpy(epoch.data(), dEpoch_, kEpochCells * sizeof(uint32_t), hipMemcpyDeviceToHost), hipSuccess);
    for (unsigned e = 0; e < kEpochCells; ++e) {
      EXPECT_EQ(epoch[e], kFlag) << "epoch cell " << e;
    }
  }

  // One run over T as rank self on grid blocks with every line fresh: the shard kernel's bits, the sends and the epoch.
  template <typename T>
  void spreadCase(int self, unsigned grid) {
    SCOPED_TRACE(::testing::Message() << typeName<T>() << ", self " << self << ", grid " << grid);
    ASSERT_NO_FATAL_FAILURE(reference<T>(self, static_cast<uint32_t>(self * 7919 + grid)));
    publishAll();
    ASSERT_NO_FATAL_FAILURE(launch<T>(&dda::common::ddaReduceScatterFabricSpreadLL<T, 0>, grid, dRecv_));
    ASSERT_TRUE(finishes()) << "spread kernel did not finish with every line fresh";
    ASSERT_NO_FATAL_FAILURE(expectRecv(dRecv_, kPackets));
    ASSERT_NO_FATAL_FAILURE(expectSent());
    expectEpochAdvanced();
  }

  DdaLLHostPeers peers_;
  int nRanks_ = kRanks;
  int self_ = kSelf;
  std::vector<uint32_t> send_;  // my sendbuff, one shard per rank
  std::vector<std::vector<uint32_t>> copy_;  // copy_[peer] is that peer's copy of my shard
  std::vector<uint32_t> ref_;  // the shard kernel's recvbuff
  uint32_t* dSend_ = nullptr;
  uint32_t* dRecv_ = nullptr;
  uint32_t* dEpoch_ = nullptr;
  hipStream_t stream_ = nullptr;
};

} // namespace

TEST_F(DdaReduceScatterFabricLL, LinesPublishedBeforeLaunchAreReduced) {
  for (int r = 1; r < kRanks; ++r) {
    publishPeer((kSelf + r) % kRanks);
  }
  launch();
  ASSERT_TRUE(finishes()) << "kernel did not finish with every line already fresh";
  expectReduced();
}

TEST_F(DdaReduceScatterFabricLL, LateLinesAreReReadUntilFresh) {
  launch();
  const auto deadline = std::chrono::steady_clock::now() + kWatchdog;
  while (!phase1Done() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  if (!phase1Done()) {
    *peers_.AbortFlag() = 1;
    finishes();
    FAIL() << "phase 1 never stored its lines";
  }
  // Every first read has returned stale lines by now, so only a real re-read can see what is published next.
  std::this_thread::sleep_for(kSettle);
  ASSERT_EQ(hipStreamQuery(stream_), hipErrorNotReady) << "kernel finished on stale lines";
  // The first batch gets three of its four peers, so each pass sees fresh lines next to a stale one.
  for (int r = 1; r < 4; ++r) {
    publishPeer((kSelf + r) % kRanks);
  }
  std::this_thread::sleep_for(kSettle);
  ASSERT_EQ(hipStreamQuery(stream_), hipErrorNotReady) << "kernel finished with the first batch still stale";
  // Completing the first batch moves every thread on to re-read the second batch, whose only peer is still stale.
  publishPeer((kSelf + 4) % kRanks);
  std::this_thread::sleep_for(kSettle);
  ASSERT_EQ(hipStreamQuery(stream_), hipErrorNotReady) << "kernel finished with the second batch still stale";
  publishPeer((kSelf + 5) % kRanks);
  ASSERT_TRUE(finishes()) << "kernel never saw the late lines";
  expectReduced();
}

TEST_F(DdaReduceScatterFabricLL, AbortExitsTheWaitAndAdvancesEpoch) {
  publishPeer((kSelf + 2) % kRanks);
  *peers_.AbortFlag() = 1;
  launch();
  ASSERT_TRUE(finishes()) << "kernel ignored the abort flag";
  for (float v : recv()) {
    ASSERT_EQ(v, kUntouched) << "an aborted thread wrote recvbuff";
  }
  expectEpochAdvanced();
}

TEST_F(DdaReduceScatterFabricLL, TornLineWithOnlyFirstHalfFreshIsReRead) {
  checkTornLineWaits(0);
}

TEST_F(DdaReduceScatterFabricLL, TornLineWithOnlySecondHalfFreshIsReRead) {
  checkTornLineWaits(1);
}

// The gfx1250 send prefetch is non-speculative, so a line past sendbuff in an unmapped page would raise a VM fault.
TEST_F(DdaReduceScatterFabricLL, PrefetchStaysInsideSendbuffEndingAtUnmappedPage) {
  int vmm = 0;
  ASSERT_EQ(hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, 0), hipSuccess);
  if (vmm == 0) {
    GTEST_SKIP() << "virtual memory management unsupported";
  }
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = 0;
  size_t gran = 0;
  ASSERT_EQ(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum), hipSuccess);
  hipMemGenericAllocationHandle_t handle = 0;
  ASSERT_EQ(hipMemCreate(&handle, gran, &prop, 0), hipSuccess);
  // Reserve two granules and map only the first, so the second stays an unmapped guard right after sendbuff.
  hipDeviceptr_t va = 0;
  ASSERT_EQ(hipMemAddressReserve(&va, 2 * gran, gran, 0, 0), hipSuccess);
  ASSERT_EQ(hipMemMap(va, gran, 0, handle, 0), hipSuccess);
  hipMemAccessDesc access = {};
  access.location.type = hipMemLocationTypeDevice;
  access.location.id = 0;
  access.flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipMemSetAccess(va, gran, &access, 1), hipSuccess);

  // 16B is the smallest shard (two packets); 144B is a multiple of 16 but not of the 128B line.
  for (const size_t count : {size_t{4}, size_t{36}}) {
    const size_t total = count * sizeof(float) * kRanks;
    float* send = reinterpret_cast<float*>(reinterpret_cast<char*>(va) + gran - total);
    std::vector<float> host(count * kRanks);
    for (size_t i = 0; i < host.size(); ++i) {
      host[i] = sendValue(static_cast<int>(i / count), i % count);
    }
    ASSERT_EQ(hipMemcpy(send, host.data(), total, hipMemcpyHostToDevice), hipSuccess);
    // Each launch starts from flag 2 again, so it stays in bank 0, the only bank the host scratch holds.
    const uint32_t epoch = kFlag - 1;
    ASSERT_EQ(hipMemcpy(dEpoch_, &epoch, sizeof(epoch), hipMemcpyHostToDevice), hipSuccess);
    // The aborted wait still runs the prefetch and the send phase, then exits without needing any peer.
    *peers_.AbortFlag() = 1;
    hipLaunchKernelGGL((dda::common::ddaReduceScatterFabricLL<float, 0>), dim3(1), dim3(kThreads), 0, stream_,
                       peers_.DevPeers<float>(), dRecv_, send, count, kSelf, kRanks, dEpoch_, 1,
                       dda::common::ddaLLRsBankBytes(kRanks), DdaKernelParams{peers_.AbortFlag()});
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_TRUE(finishes()) << "kernel did not exit on abort, shard bytes " << count * sizeof(float);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess) << "shard bytes " << count * sizeof(float);
  }
  // A prefetch fault reaches the host after the kernel ends and aborts at the next launch, so wait and launch here.
  std::this_thread::sleep_for(kPrefetchFaultWait);
  hipLaunchKernelGGL(noOp, dim3(1), dim3(1), 0, stream_);
  ASSERT_EQ(hipGetLastError(), hipSuccess);
  ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess) << "a prefetch past sendbuff faulted";
  EXPECT_EQ(hipMemUnmap(va, gran), hipSuccess);
  EXPECT_EQ(hipMemRelease(handle), hipSuccess);
  EXPECT_EQ(hipMemAddressFree(va, 2 * gran), hipSuccess);
}

// Names the re-read path this build runs, so the test list shows which binary covers which path.
#if defined(DWORDX4_INTRINSICS_FORCE_OFF)
TEST_F(DdaReduceScatterFabricLL, ReReadPathIsDwordFallback) {
#else
TEST_F(DdaReduceScatterFabricLL, ReReadPathIsB128) {
#endif
  DeviceBuffer<int> out(2);
  out.zero();
  hipLaunchKernelGGL(reportReReadPath, dim3(1), dim3(1), 0, stream_, out.ptr);
  ASSERT_EQ(hipGetLastError(), hipSuccess);
  ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
  const std::vector<int> got = out.copyTo();
  hipDeviceProp_t prop{};
  ASSERT_EQ(hipGetDeviceProperties(&prop, 0), hipSuccess);
  const std::string arch(prop.gcnArchName);
  const bool b128 = got[0] != 0;
  RecordProperty("rereadPath", b128 ? "b128" : "dword");
  std::printf("[ INFO     ] DdaReduceScatterFabricLL re-read path on %s: %s\n", arch.c_str(), b128 ? "b128" : "dword");
  const bool productionArch =
    arch.rfind("gfx942", 0) == 0 || arch.rfind("gfx950", 0) == 0 || arch.rfind("gfx1250", 0) == 0;
  if (kForceOff) {
    EXPECT_FALSE(b128) << "DWORDX4_INTRINSICS_FORCE_OFF build still took the b128 path";
    return;
  }
  if (!productionArch || got[1] == 0) {
    GTEST_SKIP() << arch << " has no b128 builtin in this build, so the dword fallback ran";
  }
  EXPECT_TRUE(b128) << "b128 builtins are available on " << arch << " but the kernel took the dword path";
}

// Float, bf16 and half as rank 1 and as the last rank, on one block (7 LDS passes), three blocks (a partial last
// slice), the launcher's grid and the untrimmed grid with idle blocks.
TEST_F(DdaReduceScatterFabricSpreadLL, MatchesShardKernelBitwise) {
  for (const int self : {kSelf, kLastRank}) {
    for (const unsigned grid : {kOneBlock, kThreeBlocks, kLaunchGrid, kIdleGrid}) {
      ASSERT_NO_FATAL_FAILURE(spreadCase<float>(self, grid));
    }
  }
  ASSERT_NO_FATAL_FAILURE(spreadCase<bf16>(kSelf, kOneBlock));
  ASSERT_NO_FATAL_FAILURE(spreadCase<bf16>(kLastRank, kLaunchGrid));
  ASSERT_NO_FATAL_FAILURE(spreadCase<half>(kSelf, kThreeBlocks));
}

// Rank counts the launcher spreads on, with blocks that take more than one LDS pass: 344-packet slices in passes of 64
// at 16 ranks, and at 32 ranks the 48-packet slices of a 2 MiB launch capped at 192 blocks, in passes of 32.
TEST_F(DdaReduceScatterFabricSpreadLL, MatchesShardKernelBitwiseOnSpreadRanks) {
  static_assert(dda::common::kDdaLLRsSpreadMinRanks <= 16, "both rank counts must take the spread kernel");
  ASSERT_NO_FATAL_FAILURE(useRanks(16));
  ASSERT_NO_FATAL_FAILURE(spreadCase<float>(9, kThreeBlocks));
  ASSERT_NO_FATAL_FAILURE(useRanks(32));
  ASSERT_NO_FATAL_FAILURE(spreadCase<float>(31, kLaunchGrid));
}

// Two peers publish once the kernel polls and the other three later, so every block re-reads until all are fresh.
TEST_F(DdaReduceScatterFabricSpreadLL, WaitsForLateLines) {
  ASSERT_NO_FATAL_FAILURE(reference<float>(kSelf, 1));
  ASSERT_NO_FATAL_FAILURE(launchSpread(kLaunchGrid, dRecv_));
  expectStillPolling("finished on stale lines");
  publishPeer((kSelf + 1) % kRanks);
  publishPeer((kSelf + 3) % kRanks);
  std::this_thread::sleep_for(kSettle);
  EXPECT_EQ(hipStreamQuery(stream_), hipErrorNotReady) << "finished with three peers stale";
  publishAll();
  ASSERT_TRUE(finishes()) << "kernel never saw the late lines";
  ASSERT_NO_FATAL_FAILURE(expectRecv(dRecv_, kPackets));
  ASSERT_NO_FATAL_FAILURE(expectSent());
  expectEpochAdvanced();
}

// One peer's lines carry a single fresh half, so the kernel keeps polling until the other half lands.
TEST_F(DdaReduceScatterFabricSpreadLL, TornLineWaitsForBothHalves) {
  const int torn = (kSelf + 2) % kRanks;
  for (const int fresh : {0, 1}) {
    SCOPED_TRACE(::testing::Message() << "fresh half " << fresh);
    ASSERT_NO_FATAL_FAILURE(reference<float>(kSelf, 2 + fresh));
    for (int peer = 0; peer < kRanks; ++peer) {
      if (peer != kSelf && peer != torn) {
        publishPeer(peer);
      }
    }
    publishHalf(torn, fresh);
    ASSERT_NO_FATAL_FAILURE(launchSpread(kThreeBlocks, dRecv_));
    expectStillPolling("finished with one half of a line stale");
    publishHalf(torn, 1 - fresh);
    ASSERT_TRUE(finishes()) << "kernel never saw the second half land";
    ASSERT_NO_FATAL_FAILURE(expectRecv(dRecv_, kPackets));
  }
}

// An abort with only row 1 published, which the fold threads poll, so only the abort broadcast keeps them off recvbuff.
TEST_F(DdaReduceScatterFabricSpreadLL, AbortLeavesRecvbuffAndAdvancesEpoch) {
  ASSERT_NO_FATAL_FAILURE(reference<float>(kSelf, 4));
  publishPeer((kSelf + 1) % kRanks);
  *peers_.AbortFlag() = 1;
  ASSERT_NO_FATAL_FAILURE(launchSpread(kLaunchGrid, dRecv_));
  ASSERT_TRUE(finishes()) << "kernel ignored the abort flag";
  ASSERT_NO_FATAL_FAILURE(expectRecv(dRecv_, 0));
  ASSERT_NO_FATAL_FAILURE(expectSent());
  expectEpochAdvanced();
}

// One block in LDS passes of 170 packets: pass 0 finishes on its published lines, pass 1 waits and then aborts.
TEST_F(DdaReduceScatterFabricSpreadLL, AbortInALaterPassKeepsTheFinishedPass) {
  const size_t passPkts = dda::common::ddaLLRsSpreadPassPkts(kRanks);
  ASSERT_NO_FATAL_FAILURE(reference<float>(kSelf, 5));
  publishAll(passPkts);
  ASSERT_NO_FATAL_FAILURE(launchSpread(kOneBlock, dRecv_));
  expectStillPolling("finished with pass 1 stale");
  *peers_.AbortFlag() = 1;
  ASSERT_TRUE(finishes()) << "kernel ignored the abort flag";
  ASSERT_NO_FATAL_FAILURE(expectRecv(dRecv_, passPkts));
  expectEpochAdvanced();
}

// In place, recvbuff is my shard of sendbuff: the sends read the other shards and the fold reads mine before writing.
TEST_F(DdaReduceScatterFabricSpreadLL, InPlaceMatchesOutOfPlace) {
  ASSERT_NO_FATAL_FAILURE(reference<float>(kSelf, 6));
  publishAll();
  uint32_t* mine = dSend_ + static_cast<size_t>(kSelf) * kShardWords;
  ASSERT_NO_FATAL_FAILURE(launchSpread(kThreeBlocks, mine));
  ASSERT_TRUE(finishes()) << "spread kernel did not finish in place";
  ASSERT_NO_FATAL_FAILURE(expectRecv(mine, kPackets));
  ASSERT_NO_FATAL_FAILURE(expectSent());
  expectEpochAdvanced();
}

// Epoch cells at 0xffffffff wrap to the 0 sentinel, which ddaNextLLEpoch skips to kFlag, so the run is unchanged.
TEST_F(DdaReduceScatterFabricSpreadLL, EpochWrapSkipsTheZeroSentinel) {
  ASSERT_NO_FATAL_FAILURE(reference<float>(kSelf, 7));
  const std::vector<uint32_t> epoch(kEpochCells, 0xffffffffu);
  ASSERT_EQ(hipMemcpy(dEpoch_, epoch.data(), kEpochCells * sizeof(uint32_t), hipMemcpyHostToDevice), hipSuccess);
  publishAll();
  ASSERT_NO_FATAL_FAILURE(launchSpread(kLaunchGrid, dRecv_));
  ASSERT_TRUE(finishes()) << "spread kernel did not finish across the epoch wrap";
  ASSERT_NO_FATAL_FAILURE(expectRecv(dRecv_, kPackets));
  ASSERT_NO_FATAL_FAILURE(expectSent());
  expectEpochAdvanced();
}

} // namespace RcclUnitTesting
