/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for the per-NET-peer channel-count knob introduced in the
// NCCL 2.28.3 sync (item 16):
//   - ncclConfig_t::nChannelsPerNetPeer  (programmatic config field)
//   - NCCL_NCHANNELS_PER_NET_PEER        (environment override)
//
// Coverage in this file:
//   * Env -> NCCL_PARAM parsing of the knob (no GPU).
//   * Config-attribute validation bounds (GPU-gated; rejected before init).
//   * End-to-end honoring through a live communicator (GPU-gated): config
//     field applied, env overrides config (precedence), and default left
//     UNDEF for the paths.cc auto-tune fallback.
//
// The end-to-end tests run a full ncclCommInitRankConfig and read the resolved
// value from comm->config via the internal comm.h. That is safe here only
// because the rccl-UnitTestsFixturesDebug target is built with hidden
// visibility (see test/CMakeLists.txt) so the executable does not interpose
// librccl.so's own allocator symbols during init.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>

#include <rccl/rccl.h>

#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

#include "comm.h" // internal: struct ncclComm::config (resolved value read-back)

// NCCL_PARAM(NChannelsPerNetPeer, ...) in src/init.cc generates this
// externally-linkable accessor. Declaring it lets us validate the env->param
// plumbing directly, without a GPU or a full communicator.
int64_t ncclParamNChannelsPerNetPeer();

namespace RcclUnitTesting
{

namespace
{
    bool HasGpu()
    {
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess && count > 0;
    }

    // Single-rank init with the supplied config. On success, writes the
    // resolved comm->config.nChannelsPerNetPeer to *resolved (if non-null) and
    // destroys the communicator. Returns the ncclCommInitRankConfig result.
    ncclResult_t InitSingleRank(ncclConfig_t* config, int* resolved)
    {
        if(hipSetDevice(0) != hipSuccess)
            return ncclSystemError;
        ncclUniqueId id;
        ncclResult_t r = ncclGetUniqueId(&id);
        if(r != ncclSuccess)
            return r;

        ncclComm_t comm = nullptr;
        r               = ncclCommInitRankConfig(&comm, 1, id, 0, config);
        if(r == ncclSuccess && comm != nullptr)
        {
            if(resolved)
                *resolved = comm->config.nChannelsPerNetPeer;
            ncclCommDestroy(comm);
        }
        return r;
    }
} // namespace

// ---------------------------------------------------------------------------
// Env -> NCCL_PARAM plumbing (no GPU required). Process-isolated because
// NCCL_PARAM caches the parsed value in a static for the process lifetime.
// ---------------------------------------------------------------------------

TEST(NChannelsPerNetPeer, EnvParam_Unset_ReturnsUndef)
{
    RUN_ISOLATED_TEST("EnvParam_Unset_ReturnsUndef", []() {
        unsetenv("NCCL_NCHANNELS_PER_NET_PEER");
        EXPECT_EQ(ncclParamNChannelsPerNetPeer(),
                  static_cast<int64_t>(NCCL_CONFIG_UNDEF_INT));
    });
}

TEST(NChannelsPerNetPeer, EnvParam_Set_IsParsed)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "EnvParam_Set_IsParsed",
        []() { EXPECT_EQ(ncclParamNChannelsPerNetPeer(), 4); },
        {{"NCCL_NCHANNELS_PER_NET_PEER", "4"}});
}

// ---------------------------------------------------------------------------
// Config-attribute validation. A value <= 0 or > MAXCHANNELS must be rejected
// with ncclInvalidArgument by parseCommConfig (src/init.cc). 100000 is safely
// above MAXCHANNELS (128 or 512) without needing the internal macro.
// ---------------------------------------------------------------------------

TEST(NChannelsPerNetPeer, Config_AboveMax_Rejected)
{
    RUN_ISOLATED_TEST("Config_AboveMax_Rejected", []() {
        if(!HasGpu())
            GTEST_SKIP() << "No GPU available";
        unsetenv("NCCL_NCHANNELS_PER_NET_PEER");

        ncclConfig_t config        = NCCL_CONFIG_INITIALIZER;
        config.nChannelsPerNetPeer = 100000;
        EXPECT_EQ(InitSingleRank(&config, nullptr), ncclInvalidArgument)
            << "value > MAXCHANNELS should be rejected";
    });
}

TEST(NChannelsPerNetPeer, Config_Zero_Rejected)
{
    RUN_ISOLATED_TEST("Config_Zero_Rejected", []() {
        if(!HasGpu())
            GTEST_SKIP() << "No GPU available";
        unsetenv("NCCL_NCHANNELS_PER_NET_PEER");

        ncclConfig_t config        = NCCL_CONFIG_INITIALIZER;
        config.nChannelsPerNetPeer = 0;
        EXPECT_EQ(InitSingleRank(&config, nullptr), ncclInvalidArgument)
            << "value 0 should be rejected";
    });
}

// ---------------------------------------------------------------------------
// End-to-end honoring through a live communicator (GPU-gated).
// ---------------------------------------------------------------------------

// Config field is applied: comm->config reflects the value set by the user.
TEST(NChannelsPerNetPeer, Config_Field_HonoredEndToEnd)
{
    RUN_ISOLATED_TEST("Config_Field_HonoredEndToEnd", []() {
        if(!HasGpu())
            GTEST_SKIP() << "No GPU available";
        unsetenv("NCCL_NCHANNELS_PER_NET_PEER");

        ncclConfig_t config        = NCCL_CONFIG_INITIALIZER;
        config.nChannelsPerNetPeer = 4;

        int          resolved = -1;
        ncclResult_t r        = InitSingleRank(&config, &resolved);
        ASSERT_EQ(r, ncclSuccess);
        EXPECT_EQ(resolved, 4) << "config field not honored end-to-end";
    });
}

// Documented precedence: a valid env value overrides an explicit config field.
TEST(NChannelsPerNetPeer, Env_OverridesConfig)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "Env_OverridesConfig",
        []() {
            if(!HasGpu())
                GTEST_SKIP() << "No GPU available";

            ncclConfig_t config        = NCCL_CONFIG_INITIALIZER;
            config.nChannelsPerNetPeer = 2; // env (16) must win

            int          resolved = -1;
            ncclResult_t r        = InitSingleRank(&config, &resolved);
            ASSERT_EQ(r, ncclSuccess);
            EXPECT_EQ(resolved, 16) << "env did not override config field";
        },
        {{"NCCL_NCHANNELS_PER_NET_PEER", "16"}});
}

// Default (field UNDEF, env unset) must init cleanly and stay UNDEF so the
// paths.cc auto-tune fallback derives the per-peer channel count.
TEST(NChannelsPerNetPeer, Default_LeavesFieldUndef_AutoTune)
{
    RUN_ISOLATED_TEST("Default_LeavesFieldUndef_AutoTune", []() {
        if(!HasGpu())
            GTEST_SKIP() << "No GPU available";
        unsetenv("NCCL_NCHANNELS_PER_NET_PEER");

        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

        int          resolved = 0;
        ncclResult_t r        = InitSingleRank(&config, &resolved);
        ASSERT_EQ(r, ncclSuccess);
        EXPECT_EQ(resolved, NCCL_CONFIG_UNDEF_INT)
            << "default must stay UNDEF to allow auto-tune fallback";
    });
}

} // namespace RcclUnitTesting
