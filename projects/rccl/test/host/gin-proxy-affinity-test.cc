/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the GIN proxy-thread NUMA affinity pin. Two halves:
//   - the CONSUME half, ncclGinProgress (src/gin/gin_host.cc:60-63), which pins
//     the proxy thread to ginState->cpuAffinity when that set is non-empty; and
//   - the PRODUCE half, ginDevCommSetupWithBackend (src/gin/gin_host.cc:392-393),
//     which stashes comm->cpuAffinity into ginState just before the progress
//     threads are spawned. Covering only the consumer would leave that one
//     assignment free to be deleted with both consumer cases still green.
// The unit under test is pulled in via GIN_HOST_CC_PATH, the standard
// rccl-UnitTestsMicroInit pattern.

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

// init.cc needs the doubles for these two: fakes/nccl_stubs.cc:85 serves init.cc:586 and
// fakes/gin_fakes.cc:18 serves init.cc:4877. Rename gin_host.cc's own definitions at
// inclusion time so both survive in rccl-UnitTestsMicroInit. Nothing else gin_host.cc
// defines is faked in this binary.
#define ncclGinHostFinalize     ncclGinHostFinalizeUut
#define ncclGinQueryLastError   ncclGinQueryLastErrorUut

#include GIN_HOST_CC_PATH

#undef ncclGinHostFinalize
#undef ncclGinQueryLastError

// gin_host.cc references these two symbols only from the setup/spawn path, which
// was dead-code-stripped until the producer test below made it live. ncclTeamRail
// lives in another TU (nccl_device/core.cc) and ncclSetThreadName in debug.cc,
// neither compiled into rccl-UnitTestsMicroInit; stub them here so the one
// setup path this test drives links without dragging in those TUs. A rail team
// of stride 1 keeps ginDevCommSetupWithBackend's stride checks satisfied.
extern "C" ncclTeam_t ncclTeamRail(ncclComm_t) {
  ncclTeam_t team{};
  team.nRanks = 1;
  team.rank = 0;
  team.stride = 1;
  return team;
}
void ncclSetThreadName(std::thread&, const char*, ...) {}

namespace {

// Build a minimal ncclGinState the affinity path reads: cpuAffinity plus the
// stop signal that makes ncclGinProgress return after pinning. Everything else
// is default-constructed and never touched on this path.
class GinProxyAffinityTest : public ::testing::Test {
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

TEST_F(GinProxyAffinityTest, NonEmptyAffinity_PinsProxyThreadToThatCpuSet) {
  CPU_SET(3, &ginState_.cpuAffinity);
  g_ncclOsCpuCountValue = 1;

  RunProgressOnce();

  ASSERT_EQ(1u, g_ncclOsSetAffinityMasks.size());       // applied exactly once
  EXPECT_TRUE(CPU_ISSET(3, &g_ncclOsSetAffinityMasks[0]));   // the stashed mask
  EXPECT_EQ(1, CPU_COUNT(&g_ncclOsSetAffinityMasks[0]));     // and only that cpu
}

// The guard: an empty affinity set (ncclOsCpuCount == 0) means the comm was
// never NUMA-pinned, so the proxy thread must be left with its inherited
// affinity rather than pinned to an empty set (which would be a hard error).
TEST_F(GinProxyAffinityTest, EmptyAffinity_LeavesProxyThreadAffinityUnchanged) {
  g_ncclOsCpuCountValue = 0;  // ncclOsCpuCount(cpuAffinity) reports "empty"

  RunProgressOnce();

  EXPECT_EQ(1, g_ncclOsCpuCountCalls);  // positive anchor: the guard really ran
  EXPECT_TRUE(g_ncclOsSetAffinityMasks.empty());  // ncclOsSetAffinity not called
}

// A scriptable GIN backend vtable. ginDevCommSetupWithBackend calls createContext
// once per connection; we hand back a context and a device handle whose
// needsProxyProgress flag is set, so the setup path decides it must spawn progress
// threads and therefore reaches the affinity stash.
struct FakeGinBackend {
  ncclNetDeviceHandle_t devHandle{};
  void* ginCtx = reinterpret_cast<void*>(0x1);
  int createContextCalls = 0;
};
FakeGinBackend* g_fakeGinBackend = nullptr;

ncclResult_t FakeCreateContext(void* /*collComm*/, ncclGinConfig_t* /*config*/, void** ginCtx,
                               ncclNetDeviceHandle_t** devHandle) {
  g_fakeGinBackend->createContextCalls++;
  g_fakeGinBackend->devHandle.handle = reinterpret_cast<void*>(0x2);  // non-null: setup accepts it
  g_fakeGinBackend->devHandle.needsProxyProgress = 1;                 // forces the spawn path
  *ginCtx = g_fakeGinBackend->ginCtx;
  *devHandle = &g_fakeGinBackend->devHandle;
  return ncclSuccess;
}

ncclResult_t FakeDestroyContext(void* /*ginCtx*/) { return ncclSuccess; }

// The producer: ginDevCommSetupWithBackend copies comm->cpuAffinity into
// ginState->cpuAffinity (gin_host.cc:393) on the branch that first spawns the
// progress threads, so the consumer above has a populated set to read on a real
// comm. Drive that setup path with a scripted backend and assert the copy.
class GinProxyAffinitySetupTest : public ::testing::Test {
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
    // Release the ncclGinStateDevComm the setup path callocs and links into
    // ginState->devComms; nothing else frees it (ncclSharedResources has no
    // destructor), so the opt-in ASAN arm would flag the leak. Done here rather
    // than in the test body so it runs even when an ASSERT above returns early.
    // Also exercises FakeDestroyContext. Guarded on the list actually being
    // populated, so a setup that failed before linking is not freed twice.
    if (ginState.devComms != nullptr) {
      EXPECT_EQ(ncclSuccess, ncclGinDevCommFree(comm_.get(), &devComm_));
    }
    g_fakeGinBackend = nullptr;
    ResetOsFakes();
  }
};

TEST_F(GinProxyAffinitySetupTest, StashesCommAffinityBeforeSpawningProxyThreads) {
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

  // End-to-end: join the sole proxy thread and confirm the real worker pinned
  // itself to the comm's mask, exercising the whole produce -> spawn -> consume
  // handoff rather than just poking the field. The stash-before-spawn ordering
  // is what gives the worker a happens-before view of the mask here; note this
  // is not a deterministic guard against reordering the stash below the spawn
  // loop -- that variant is a data race (UB), which only a thread sanitizer
  // would reliably flag, not a value assertion.
  ginState.thread[0].join();
  ASSERT_EQ(1u, g_ncclOsSetAffinityMasks.size());
  EXPECT_TRUE(CPU_EQUAL(&comm_->cpuAffinity, &g_ncclOsSetAffinityMasks[0]));
}

}  // namespace
