/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the GIN proxy-thread NUMA affinity pin in
// ncclGinProgress (src/gin/gin_host.cc:60-63). The unit under test is pulled in
// via GIN_HOST_CC_PATH, the standard rccl-UnitTestsMicroInit pattern.

#include <gtest/gtest.h>

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

}  // namespace
