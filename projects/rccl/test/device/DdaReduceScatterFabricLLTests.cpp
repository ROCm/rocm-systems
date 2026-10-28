/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ddaReduceScatterFabricLL on one GPU: the host plays the peers by writing LL lines into coherent host scratch.
// Built twice: rccl-UnitTestsFixtures defines DWORDX4_INTRINSICS_FORCE_OFF (dword re-read), rccl-UnitTests not (b128).

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "DeviceTestBase.hpp"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll.h"
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

// Writes the device-side RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS and whether the toolchain offers the b128 builtins.
__global__ void reportReReadPath(int* out) {
  out[0] = RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS;
#if __has_builtin(__builtin_amdgcn_global_load_b128) && __has_builtin(__builtin_amdgcn_global_store_b128)
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
    const size_t bankBytes = dda::common::ddaLLRsBankBytes(kRanks);
    for (int r = 0; r < kRanks; ++r) {
      HIP_CHECK(hipHostMalloc(&hostScratch_[r], bankBytes, hipHostMallocMapped | hipHostMallocCoherent));
      std::memset(hostScratch_[r], 0, bankBytes);
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&devScratch_[r]), hostScratch_[r], 0));
    }
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&abortFlag_), sizeof(uint32_t),
                            hipHostMallocMapped | hipHostMallocCoherent));
    *abortFlag_ = 0;
    HIP_CHECK(hipMalloc(&dPeers_, sizeof(devScratch_)));
    HIP_CHECK(hipMemcpy(dPeers_, devScratch_, sizeof(devScratch_), hipMemcpyHostToDevice));

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
    HIP_EXPECT(hipFree(dPeers_));
    HIP_EXPECT(hipHostFree(abortFlag_));
    for (void* p : hostScratch_) {
      HIP_EXPECT(hipHostFree(p));
    }
  }

  volatile uint32_t* line(int owner, int sender, size_t pk) {
    LLPacket16* base = static_cast<LLPacket16*>(hostScratch_[owner]);
    return reinterpret_cast<volatile uint32_t*>(&base[sender * dda::common::kDdaLLRsSlotStridePkts + pk]);
  }

  // Writes the payload before the flags, as one 16B store would make them visible together to the poller.
  void publishPeer(int peer) {
    for (size_t pk = 0; pk < kPackets; ++pk) {
      volatile uint32_t* w = line(kSelf, peer, pk);
      w[0] = bitsOf(peerValue(peer, 2 * pk));
      w[2] = bitsOf(peerValue(peer, 2 * pk + 1));
      std::atomic_thread_fence(std::memory_order_release);
      w[1] = kFlag;
      w[3] = kFlag;
    }
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
                       dPeers_, dRecv_, dSend_, kRecvCount, kSelf, kRanks, dEpoch_, static_cast<int>(kBlocks),
                       dda::common::ddaLLRsBankBytes(kRanks), DdaKernelParams{abortFlag_});
    ASSERT_EQ(hipGetLastError(), hipSuccess);
  }

  // Past the watchdog, raises abort and publishes every peer, so a lost wake-up or an ignored abort fails, not hangs.
  bool finishes() {
    const auto deadline = std::chrono::steady_clock::now() + kWatchdog;
    while (hipStreamQuery(stream_) == hipErrorNotReady) {
      if (std::chrono::steady_clock::now() > deadline) {
        *abortFlag_ = 1;
        for (int r = 1; r < kRanks; ++r) {
          publishPeer((kSelf + r) % kRanks);
        }
        EXPECT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        return false;
      }
      std::this_thread::yield();
    }
    return hipStreamSynchronize(stream_) == hipSuccess;
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

  void* hostScratch_[kRanks] = {};
  float* devScratch_[kRanks] = {};
  uint32_t* abortFlag_ = nullptr;
  float** dPeers_ = nullptr;
  float* dSend_ = nullptr;
  float* dRecv_ = nullptr;
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
    *abortFlag_ = 1;
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
  *abortFlag_ = 1;
  launch();
  ASSERT_TRUE(finishes()) << "kernel ignored the abort flag";
  for (float v : recv()) {
    ASSERT_EQ(v, kUntouched) << "an aborted thread wrote recvbuff";
  }
  expectEpochAdvanced();
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
    *abortFlag_ = 1;
    hipLaunchKernelGGL((dda::common::ddaReduceScatterFabricLL<float, 0>), dim3(1), dim3(kThreads), 0, stream_, dPeers_,
                       dRecv_, send, count, kSelf, kRanks, dEpoch_, 1, dda::common::ddaLLRsBankBytes(kRanks),
                       DdaKernelParams{abortFlag_});
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

} // namespace RcclUnitTesting
