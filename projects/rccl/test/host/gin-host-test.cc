/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <memory>

#include <sched.h>
#include <thread>

#include "fakes/nccl_fakes.h"
#include "fakes/os_fakes.h"

#include "nccl.h"
#include "comm.h"
#include "gin/gin_host.h"

#include "fakes/param_redirect.h"

#include GIN_HOST_CC_PATH

extern "C" ncclTeam_t ncclTeamRail(ncclComm_t) {
  ncclTeam_t team{};
  team.nRanks = 1;
  team.rank = 0;
  team.stride = 1;
  return team;
}
void ncclSetThreadName(std::thread&, const char*, ...) {}

namespace {

class GinHostProxyAffinityMicrotest : public ::testing::Test {
protected:
  ncclGinState ginState_;

  void SetUp() override {
    ResetOsFakes();
    CPU_ZERO(&ginState_.cpuAffinity);
    ginState_.proxyThreadStopSignal.store(true);  // exit at the top of the loop
    ginState_.writePending.store(false);
  }

  void TearDown() override { ResetOsFakes(); }

  // Run ncclGinProgress on a fresh thread (so the affinity apply, if the fake
  // were real, would land on a throwaway thread rather than the test runner)
  // and wait for it to return.
  void RunProgressOnce() {
    std::thread t([this] { ncclGinProgress(&ginState_, /*threadIdx=*/0); });
    t.join();
  }
};

TEST_F(GinHostProxyAffinityMicrotest, NonEmptyAffinity_PinsProxyThreadToThatCpuSet) {
  CPU_SET(3, &ginState_.cpuAffinity);
  g_ncclOsCpuCountValue = 1;

  RunProgressOnce();

  ASSERT_EQ(1u, g_ncclOsSetAffinityMasks.size());
  EXPECT_TRUE(CPU_ISSET(3, &g_ncclOsSetAffinityMasks[0]));
  EXPECT_EQ(1, CPU_COUNT(&g_ncclOsSetAffinityMasks[0]));
}

TEST_F(GinHostProxyAffinityMicrotest, EmptyAffinity_LeavesProxyThreadAffinityUnchanged) {
  g_ncclOsCpuCountValue = 0;

  RunProgressOnce();

  EXPECT_EQ(1, g_ncclOsCpuCountCalls);
  EXPECT_TRUE(g_ncclOsSetAffinityMasks.empty());
}

struct FakeGinBackend {
  ncclNetDeviceHandle_t devHandle{};
  void* ginCtx = reinterpret_cast<void*>(0x1);
  int createContextCalls = 0;
};
FakeGinBackend* g_fakeGinBackend = nullptr;

ncclResult_t FakeCreateContext(void* /*collComm*/, ncclGinConfig_t* /*config*/, void** ginCtx,
                               ncclNetDeviceHandle_t** devHandle) {
  g_fakeGinBackend->createContextCalls++;
  g_fakeGinBackend->devHandle.handle = reinterpret_cast<void*>(0x2);
  g_fakeGinBackend->devHandle.needsProxyProgress = 1;
  *ginCtx = g_fakeGinBackend->ginCtx;
  *devHandle = &g_fakeGinBackend->devHandle;
  return ncclSuccess;
}

ncclResult_t FakeDestroyContext(void* /*ginCtx*/) { return ncclSuccess; }

// The producer: ginDevCommSetupWithBackend copies comm->cpuAffinity into
// ginState->cpuAffinity (gin_host.cc:393) on the branch that first spawns the
// progress threads, so the consumer above has a populated set to read on a real
// comm. Drive that setup path with a scripted backend and assert the copy.
class GinHostProxyAffinitySetupMicrotest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> comm_ = std::make_unique<ncclComm>();
  std::unique_ptr<ncclSharedResources> sr_ = std::make_unique<ncclSharedResources>();
  FakeGinBackend fakeBackend_;
  ncclGin_t vtable_{};
  ncclDevComm devComm_{};

  void SetUp() override {
    ResetOsFakes();
    g_fakeGinBackend = &fakeBackend_;

    comm_->sharedRes = sr_.get();
    struct ncclGinState& ginState = sr_->ginState;
    ginState.ginConnectionType = NCCL_GIN_CONNECTION_FULL;  // connectedStride == 1
    ginState.proxyNthreads = 1;
    ginState.proxyThreadsCreated = false;                  // so needsStart is true
    ginState.proxyThreadStopSignal.store(true);            // spawned threads exit immediately
    ginState.writePending.store(false);

    vtable_.name = "fake-gin";
    vtable_.createContext = &FakeCreateContext;
    vtable_.destroyContext = &FakeDestroyContext;

    struct ncclGinBackendState& backend = ginState.backends[0];
    backend.ginType = NCCL_GIN_TYPE_PROXY;
    backend.ncclGin = &vtable_;
    backend.ginCommCount = 1;
    backend.ginComms[0] = reinterpret_cast<void*>(0x10);
  }

  void TearDown() override {
    struct ncclGinState& ginState = sr_->ginState;
    for (int t = 0; t < ginState.proxyNthreads; t++) {
      if (ginState.thread[t].joinable()) ginState.thread[t].join();
    }

    if (ginState.devComms != nullptr) {
      EXPECT_EQ(ncclSuccess, ncclGinDevCommFree(comm_.get(), &devComm_));
    }
    g_fakeGinBackend = nullptr;
    ResetOsFakes();
  }
};

TEST_F(GinHostProxyAffinitySetupMicrotest, StashesCommAffinityBeforeSpawningProxyThreads) {
  CPU_ZERO(&comm_->cpuAffinity);
  CPU_SET(5, &comm_->cpuAffinity);  // a distinctive mask to spot the copy
  g_ncclOsCpuCountValue = 1;        // so the spawned worker takes the pin branch

  struct ncclGinState& ginState = sr_->ginState;
  ncclDevCommRequirements reqs{};
  reqs.ginContextCount = 1;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_NONE;      // requestedStride stays 1
  reqs.ginTrafficClass = NCCL_CONFIG_UNDEF_INT;

  ncclResult_t ret =
    ginDevCommSetupWithBackend(comm_.get(), &reqs, &devComm_, /*deviceCodeVersion=*/0, &ginState.backends[0]);

  ASSERT_EQ(ncclSuccess, ret);
  EXPECT_EQ(1, fakeBackend_.createContextCalls);         // the setup path really ran
  EXPECT_TRUE(ginState.proxyThreadsCreated);             // it took the spawn branch
  EXPECT_TRUE(CPU_EQUAL(&comm_->cpuAffinity, &ginState.cpuAffinity));  // comm mask stashed verbatim

  ginState.thread[0].join();
  ASSERT_EQ(1u, g_ncclOsSetAffinityMasks.size());
  EXPECT_TRUE(CPU_EQUAL(&comm_->cpuAffinity, &g_ncclOsSetAffinityMasks[0]));
}

}  // namespace
