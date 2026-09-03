/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include <cstdlib>
#include <cstring>
#include <memory>

#include "ce_coll.h"
#include "comm.h"
#include "common/ErrCode.hpp"
#include "common/MockComm.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "debug.h"
#include "enqueue.h"
#include "graph.h"
#include "graph/topo.h"
#include "net.h"
#include "plugin/nccl_tuner.h"
#include "rccl_common.h"
#include "rocmwrap.h"

#include <algorithm>

namespace RcclUnitTesting
{

// Helper function to determine if P2P test should be skipped due to static
// variable state
static bool ShouldSkipP2pTest(const char* requiredEnvValue = nullptr)
{
    const char* envValue = getenv("NCCL_P2P_NET_CHUNKSIZE");

    // If a specific environment value is required, check for it
    if(requiredEnvValue != nullptr)
    {
        if(!envValue || strcmp(envValue, requiredEnvValue) != 0)
        {
            return true; // Skip if env var is not set to required value
        }
        return false; // Don't skip if env var matches required value
    }

    // For architecture logic tests, skip only if environment variable is set
    // (which would override the static variable behavior)
    // Note: We cannot directly check if static variable is RCCL_VALUE_UNSET
    // from test code, so we rely on clean environment for proper testing
    if(envValue != nullptr)
    {
        return true; // Skip if env var is set (prevents testing architecture logic)
    }

    // Environment is clean - proceed with test
    // Warning: Static variable might still be initialized from previous tests
    // For guaranteed clean state, run tests individually or restart binary
    return false; // Don't skip
}

// Helper function to determine if PXN test should be skipped due to static
// variable state
static bool ShouldSkipPxnTest(const char* requiredEnvValue = nullptr)
{
    const char* envValue = getenv("NCCL_PXN_DISABLE");

    // If a specific environment value is required, check for it
    if(requiredEnvValue != nullptr)
    {
        if(!envValue || strcmp(envValue, requiredEnvValue) != 0)
        {
            return true; // Skip if env var is not set to required value
        }
        return false; // Don't skip if env var matches required value
    }

    // For architecture logic tests, skip only if environment variable is set
    // (which would override the static variable behavior)
    if(envValue != nullptr)
    {
        return true; // Skip if env var is set (prevents testing architecture logic)
    }

    // Environment is clean - proceed with test
    return false; // Don't skip
}

// Helper function to test the static expose check
ncclResult_t testStaticExposeCheck()
{
    RCCL_STATIC_EXPOSE_CHECK();
    return ncclSuccess;
}

// CreateMockComm / CleanupMockComm moved to common/MockComm.hpp so other fixtures
// can reuse them.

// Helper function to determine if rcclSetPipelining test should be skipped
static bool ShouldSkipRcclSetPipeliningTests()
{
    const char* disable = getenv("RCCL_DISABLE_REDUCE_COPY_PIPELINING");
    // Skip the test if RCCL_DISABLE_REDUCE_COPY_PIPELINING is set
    if(disable && strcmp(disable, "0") != 0)
    {
        return true;
    }
    return false;
}

// Helper function to validate protocol string against known valid protocols
static bool isProtoStrValid(const char* envStr)
{
    if(!envStr)
        return false;
    for(int i = 0; i < NCCL_NUM_PROTOCOLS; ++i)
    {
        if(strcasecmp(envStr, ncclProtoStr[i]) == 0)
        {
            return true; // Match found
        }
    }
    return false; // No match found
}

// Helper that exercises CUCHECK with a guaranteed-to-fail HIP call
static ncclResult_t triggerCucheckFailure()
{
    CUCHECK(hipPointerGetAttribute(nullptr, HIP_POINTER_ATTRIBUTE_CONTEXT,
                                   (hipDeviceptr_t)0x1));
    return ncclSuccess;
}

// Helper function to validate algorithm string against known valid algorithms
static bool isAlgoStrValid(const char* envStr)
{
    if(!envStr)
        return false;
    for(int i = 0; i < NCCL_NUM_ALGORITHMS; ++i)
    {
        if(strcasecmp(envStr, ncclAlgoStr[i]) == 0)
        {
            return true; // Match found
        }
    }
    return false; // No match found
}

// `new ncclComm()` value-inits ~13MB on the stack (MAXCHANNELS arrays) and
// overflows the 8MB default thread stack. calloc keeps the mock on the heap.
static ncclComm_t AllocBareComm()
{
    auto* comm = static_cast<ncclComm_t>(std::calloc(1, sizeof(ncclComm)));
    comm->topo = static_cast<ncclTopoSystem*>(std::calloc(1, sizeof(ncclTopoSystem)));
    return comm;
}

static void FreeBareComm(ncclComm_t comm)
{
    if(!comm) return;
    std::free(comm->topo);
    std::free(comm);
}

TEST(Rcclwrap, RcclFuncMaxSendRecvCount)
{
    ncclResult_t staticCheckResult = testStaticExposeCheck();
#ifdef RCCL_EXPOSE_STATIC
    EXPECT_EQ(staticCheckResult, ncclSuccess);
#else
    EXPECT_EQ(staticCheckResult, ncclInvalidUsage);
#endif

    size_t       maxCount = 0;
    ncclResult_t result   = rcclFuncMaxSendRecvCount(ncclFuncAllReduce, 4, 1024, maxCount);
    EXPECT_EQ(maxCount, 1024);
    EXPECT_EQ(result, ncclSuccess);
}

TEST(Rcclwrap, RcclUpdateCollectiveProtocol_UsesLL128WhenInRange)
{
    setenv("NCCL_PROTO", "", 1); // Trigger auto selection mode
    unsetenv("NCCL_PROTO");

    ncclComm_t comm = AllocBareComm();
    // Manually populate minimal fields for comm
    comm->nRanks                    = 1;
    comm->nNodes                    = 2; // triggers inter-node logic
    comm->rank                      = 0;
    comm->topo->ll128Enabled        = true;
    comm->topo->nodes[GPU].nodes[0] = {};
    comm->topo->nodes[GPU].count    = 1;
    strncpy(
        comm->topo->nodes[GPU].nodes[0].gpu.gcn,
        "gfx942",
        sizeof(comm->topo->nodes[GPU].nodes[0].gpu.gcn)
    );

    int idx = rcclGetTunableIndex(ncclFuncAllReduce);
    comm->minMaxLLRange[idx][NCCL_PROTO_LL][RCCL_PROTOCOL_MIN_IDX]       = 512;
    comm->minMaxLLRange[idx][NCCL_PROTO_LL][RCCL_PROTOCOL_MAX_IDX]       = 1024;
    comm->minMaxLLRange[idx][NCCL_PROTO_LL128][RCCL_PROTOCOL_MIN_IDX]    = 256;
    comm->minMaxLLRange[idx][NCCL_PROTO_LL128][RCCL_PROTOCOL_MAX_IDX]    = 2048;
    comm->minMaxLLRange[idx][NCCL_PROTO_LL128][RCCL_PROTOCOL_FACTOR_IDX] = 1;

    ncclTaskColl info = {};
    // Manually populate minimal fields for info
    info.func     = ncclFuncAllReduce;
    info.protocol = NCCL_PROTO_UNDEF;

    size_t nBytes = 1024;

    rcclUpdateCollectiveProtocol(comm, nBytes, &info);
    EXPECT_TRUE(info.protocol == NCCL_PROTO_LL128 || info.protocol == NCCL_PROTO_LL);

    FreeBareComm(comm);
}

TEST(Rcclwrap, RcclUpdateCollectiveProtocol_WarnsOnGfx942Arch)
{
    setenv("NCCL_PROTO", "", 1);
    unsetenv("NCCL_PROTO");

    ncclComm_t comm = AllocBareComm();
    // Manually populate minimal fields for comm
    comm->nRanks                    = 1;
    comm->nNodes                    = 2; // triggers inter-node logic
    comm->rank                      = 0;
    comm->topo->ll128Enabled        = true;
    comm->topo->nodes[GPU].nodes[0] = {};
    strncpy(
        comm->topo->nodes[GPU].nodes[0].gpu.gcn,
        "gfx942",
        sizeof(comm->topo->nodes[GPU].nodes[0].gpu.gcn)
    );

    int idx = rcclGetTunableIndex(ncclFuncAllReduce);
    comm->minMaxLLRange[idx][NCCL_PROTO_LL][RCCL_PROTOCOL_MIN_IDX]       = RCCL_LL_LIMITS_UNDEFINED;
    comm->minMaxLLRange[idx][NCCL_PROTO_LL][RCCL_PROTOCOL_MAX_IDX]       = RCCL_LL_LIMITS_UNDEFINED;
    comm->minMaxLLRange[idx][NCCL_PROTO_LL128][RCCL_PROTOCOL_MIN_IDX]    = RCCL_LL_LIMITS_UNDEFINED;
    comm->minMaxLLRange[idx][NCCL_PROTO_LL128][RCCL_PROTOCOL_MAX_IDX]    = RCCL_LL_LIMITS_UNDEFINED;
    comm->minMaxLLRange[idx][NCCL_PROTO_LL128][RCCL_PROTOCOL_FACTOR_IDX] = RCCL_LL_LIMITS_UNDEFINED;

    ncclTaskColl info = {};
    // Manually populate minimal fields for info
    info.func     = ncclFuncAllReduce;
    info.protocol = NCCL_PROTO_UNDEF;
    size_t nBytes = 1024; // 1024 per rank for 4 ranks

    rcclUpdateCollectiveProtocol(comm, nBytes, &info);
    EXPECT_EQ(info.protocol, NCCL_PROTO_UNDEF);

    FreeBareComm(comm);
}

TEST(Rcclwrap, RcclUpdateCollectiveProtocol_HonorsUserProtocolEnv)
{                                 // Why does this pass
                                  // if it does not
                                  // enter the else if
                                  // block
    setenv("NCCL_PROTO", "1", 1); // Simulate manual override

    ncclComm_t comm = AllocBareComm();
    // Manually populate minimal fields for comm
    comm->nRanks = 1;
    comm->nNodes = 2; // triggers inter-node logic
    comm->rank   = 0;
    comm->topo->ll128Enabled        = true;
    comm->topo->nodes[GPU].nodes[0] = {};
    strncpy(
        comm->topo->nodes[GPU].nodes[0].gpu.gcn,
        "gfx942",
        sizeof(comm->topo->nodes[GPU].nodes[0].gpu.gcn)
    );

    ncclTaskColl info = {};
    // Manually populate minimal fields for info
    info.func     = ncclFuncAllReduce;
    info.protocol = NCCL_PROTO_UNDEF;
    size_t nBytes = 1024; // 1024 per rank for 4 ranks

    rcclUpdateCollectiveProtocol(comm, nBytes, &info);
    EXPECT_EQ(info.protocol, NCCL_PROTO_UNDEF);

    FreeBareComm(comm);
}

TEST(Rcclwrap, RcclUpdateCollectiveProtocol_SimpleFallbackWhenNoRanges)
{
    setenv("NCCL_PROTO", "", 1); // Trigger auto selection mode
    unsetenv("NCCL_PROTO");

    ncclComm_t comm = AllocBareComm();
    // Manually populate minimal fields for comm
    comm->nRanks = 1;
    comm->nNodes = 2; // triggers inter-node logic
    comm->rank   = 0;
    comm->topo->ll128Enabled        = true;
    comm->topo->nodes[GPU].nodes[0] = {};
    comm->topo->nodes[GPU].count    = 1;
    strncpy(
        comm->topo->nodes[GPU].nodes[0].gpu.gcn,
        "gfx942",
        sizeof(comm->topo->nodes[GPU].nodes[0].gpu.gcn)
    );

    int idx = rcclGetTunableIndex(ncclFuncAllReduce);
    comm->minMaxLLRange[idx][NCCL_PROTO_LL][RCCL_PROTOCOL_MIN_IDX] = 512;
    comm->minMaxLLRange[idx][NCCL_PROTO_LL][RCCL_PROTOCOL_MAX_IDX] = 1024;

    // Manually populate minimal fields for info
    ncclTaskColl info = {};
    info.func         = ncclFuncAllReduce;
    info.protocol     = NCCL_PROTO_UNDEF;
    size_t nBytes     = 2048; // 1024 per rank for 4 ranks

    rcclUpdateCollectiveProtocol(comm, nBytes, &info);
    EXPECT_EQ(info.protocol, NCCL_PROTO_SIMPLE);

    FreeBareComm(comm);
}

TEST(Rcclwrap, validHsaScratchEnvSettingTest)
{
    // When HSA_NO_SCRATCH_RECLAIM is set, it is always valid
    EXPECT_TRUE(validHsaScratchEnvSetting("1", 0, 0, "gfx950"));

    EXPECT_TRUE(validHsaScratchEnvSetting("1", 0, 0, "gfx942"));

    // When HSA_NO_SCRATCH_RECLAIM is not set, looking at hip version and firmware
    // version
    EXPECT_TRUE(validHsaScratchEnvSetting(nullptr, 60443484, 24, "gfx950"));

    EXPECT_FALSE(validHsaScratchEnvSetting(nullptr, 60443483, 24, "gfx950"));

    EXPECT_FALSE(validHsaScratchEnvSetting(nullptr, 60443484, 23, "gfx950"));

    EXPECT_TRUE(validHsaScratchEnvSetting(nullptr, 60443484, 177, "gfx942"));

    EXPECT_FALSE(validHsaScratchEnvSetting(nullptr, 60443484, 176, "gfx942"));

    EXPECT_FALSE(validHsaScratchEnvSetting(nullptr, 60443483, 177, "gfx942"));

    EXPECT_TRUE(validHsaScratchEnvSetting(nullptr, 60443483, 0, "gfx000"));

    EXPECT_TRUE(validHsaScratchEnvSetting(nullptr, 60300000, 0, "gfx000"));
}

TEST(Rcclwrap, RcclUpdateThreadThreshold_UserEnvSet)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "RcclUpdateThreadThreshold_UserEnvSet",
        []()
        {
            const char* value = getenv("NCCL_THREAD_THRESHOLDS");

            if(!value)
            {
                TEST_INFO(
                    "[Rcclwrap] Test skipped. Set environment variable "
                    "NCCL_THREAD_THRESHOLD"
                );
                GTEST_SKIP() << "[Rcclwrap] Test skipped. Set environment variable "
                                "NCCL_THREAD_THRESHOLD\n";
            }
            else
            {
                ncclComm comm;
                comm.nRanks = 8;
                comm.nNodes = 4;
                memset(comm.minMaxLLRange, 0, sizeof(comm.minMaxLLRange));

                ncclTaskColl info;
                info.func     = ncclFuncReduceScatter;
                info.protocol = 0;

                int threadThreshold = 5; // Any number should do, we should make
                                         // sure this number does not change
                rcclUpdateThreadThreshold(&comm, 0, &info, threadThreshold);

                EXPECT_EQ(threadThreshold, 5);
            }
    },
        {{"NCCL_THREAD_THRESHOLDS", "1"}}
    );
}

TEST(Rcclwrap, RcclUpdateThreadThreshold_MinNChannelsSet)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "RcclUpdateThreadThreshold_MinNChannelsSet",
        []()
        {
            const char* value = getenv("NCCL_MIN_NCHANNELS");
            if(!value)
            {
                TEST_INFO(
                    "[Rcclwrap] Test skipped. Set environment "
                    "variable NCCL_MIN_NCHANNELS"
                );
                GTEST_SKIP() << "[Rcclwrap] Test skipped. Set environment variable "
                                "NCCL_MIN_NCHANNELS\n";
            }
            else
            {
                ncclComm     comm{};
                ncclTaskColl info{};
                int          threadThreshold = 5;

                comm.nRanks   = 4;
                comm.nNodes   = 4;
                info.func     = ncclFuncAllGather;
                info.protocol = 0;
                memset(comm.minMaxLLRange, 0, sizeof(comm.minMaxLLRange));

                rcclUpdateThreadThreshold(&comm, 0, &info, threadThreshold);

                EXPECT_EQ(threadThreshold, 5);
            }
    },
        {{"NCCL_MIN_NCHANNELS", "1"}}
    );
}

TEST(Rcclwrap, RcclUpdateThreadThreshold_MaxChannelsSet)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "RcclUpdateThreadThreshold_MaxChannelsSet",
        []()
        {
            const char* value = getenv("NCCL_MAX_NCHANNELS");
            if(!value)
            {
                TEST_INFO(
                    "[Rcclwrap] Test skipped. Set environment "
                    "variable NCCL_MAX_NCHANNELS"
                );
                GTEST_SKIP() << "[Rcclwrap] Test skipped. Set environment variable "
                                "NCCL_MAX_NCHANNELS\n";
            }
            else
            {
                ncclComm     comm{};
                ncclTaskColl info{};
                int          threadThreshold = 5;

                comm.nRanks   = 4;
                comm.nNodes   = 4;
                info.func     = ncclFuncAllGather;
                info.protocol = 0;
                memset(comm.minMaxLLRange, 0, sizeof(comm.minMaxLLRange));

                rcclUpdateThreadThreshold(&comm, 0, &info, threadThreshold);

                EXPECT_EQ(threadThreshold, 5);
            }
    },
        {{"NCCL_MAX_NCHANNELS", "1"}}
    );
}

TEST(Rcclwrap, RcclUpdateThreadThreshold_NoEnv_nNodesLessThan2)
{
    ncclComm     comm{};
    ncclTaskColl info{};
    int          threadThreshold = 5;

    comm.nRanks   = 4;
    comm.nNodes   = 1; // less than 2
    info.func     = ncclFuncReduceScatter;
    info.protocol = 0;
    memset(comm.minMaxLLRange, 0, sizeof(comm.minMaxLLRange));

    rcclUpdateThreadThreshold(&comm, 0, &info, threadThreshold);

    EXPECT_EQ(threadThreshold, 5); // no change
}

TEST(Rcclwrap, RcclUpdateThreadThreshold_NoEnv_FuncUnsupported)
{
    ncclComm     comm{};
    ncclTaskColl info{};
    int          threadThreshold = 5;

    comm.nRanks   = 4;
    comm.nNodes   = 2;
    info.func     = ncclFuncAllReduce; // unsupported func
    info.protocol = 0;
    memset(comm.minMaxLLRange, 0, sizeof(comm.minMaxLLRange));

    rcclUpdateThreadThreshold(&comm, 0, &info, threadThreshold);

    EXPECT_EQ(threadThreshold, 5);
}

TEST(Rcclwrap, RcclUpdateThreadThreshold_NoEnv_UpdateOccurs)
{
    ncclComm     comm{};
    ncclTaskColl info{};
    int          threadThreshold = 5;

    comm.nRanks   = 4;
    comm.nNodes   = 2;
    info.func     = ncclFuncReduceScatter;
    info.protocol = 0;
    memset(comm.minMaxLLRange, 0, sizeof(comm.minMaxLLRange));

    int idx = rcclGetTunableIndex(info.func);
    comm.minMaxLLRange[idx][info.protocol][RCCL_PROTOCOL_THREAD_THRESHOLD_IDX] = 10;

    rcclUpdateThreadThreshold(&comm, 0, &info, threadThreshold);

    EXPECT_EQ(threadThreshold, 40); // 10 * 4
}

TEST(Rcclwrap, RcclUpdateThreadThreshold_NoEnv_ThresholdUndefined)
{
    ncclComm     comm{};
    ncclTaskColl info{};
    int          threadThreshold = 5;

    comm.nRanks   = 4;
    comm.nNodes   = 3;
    info.func     = ncclFuncAllGather;
    info.protocol = 0;
    memset(comm.minMaxLLRange, 0, sizeof(comm.minMaxLLRange));

    int idx = rcclGetTunableIndex(info.func);
    comm.minMaxLLRange[idx][info.protocol][RCCL_PROTOCOL_THREAD_THRESHOLD_IDX]
        = RCCL_LL_LIMITS_UNDEFINED;

    rcclUpdateThreadThreshold(&comm, 0, &info, threadThreshold);

    EXPECT_EQ(threadThreshold, 5);
}

TEST(Rcclwrap, RcclSetPipelining_Invalid_DType)
{
    // Skip the test if pipelining has been disabled
    // (RCCL_DISABLE_REDUCE_COPY_PIPELINING=1)
    if(ShouldSkipRcclSetPipeliningTests())
    {
        GTEST_SKIP() << "Skipping test: RCCL_DISABLE_REDUCE_COPY_PIPELINING environment "
                        "variable is set. Unset this variable to enable pipelining.";
    }

    // Skip the test if pipelining has been enabled for all data types
    // (RCCL_PIPELINE_ALL_DATA_TYPES=1)
    const char* allowAllDTypes = getenv("RCCL_PIPELINE_ALL_DATA_TYPES");
    if(allowAllDTypes && strcmp(allowAllDTypes, "0") != 0)
    {
        GTEST_SKIP() << "Skipping test: RCCL_PIPELINE_ALL_DATA_TYPES environment "
                        "variable is set. Unset this variable to enable pipelining "
                        "only for bf16 data type.";
    }

    // Pipeline should not be set for non-bf16 datatypes, unless
    // rcclParamPipelineAllDTypes() returns true
    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx950", 8);
    comm->nNodes = 2; // Multi node

    ncclTaskColl info = {};
    info.func         = ncclFuncAllReduce;
    info.datatype     = ncclFloat32;

    size_t nBytes = 16 * 1024 * 1024; // 16MB
    rcclSetPipelining(comm, nBytes, &info);

    EXPECT_EQ(info.pipeline, 0) << "Non-bf16 should not set pipeline by default";

    CleanupMockComm(comm);
}

TEST(Rcclwrap, RcclSetPipelining_GFX950_SingleNode_Disable)
{
    // Skip the test if pipelining has been disabled
    // (RCCL_DISABLE_REDUCE_COPY_PIPELINING=1)
    if(ShouldSkipRcclSetPipeliningTests())
    {
        GTEST_SKIP() << "Skipping test: RCCL_DISABLE_REDUCE_COPY_PIPELINING environment "
                        "variable is set. Unset this variable to enable pipelining.";
    }

    // For single-node, pipeline remains 0
    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx950", 8);
    comm->nNodes = 1; // Single node

    ncclTaskColl info = {};
    // In rcclSetPipelining(), ncclFuncAllReduce, ncclFuncReduceScatter, and
    // ncclFuncReduce share the same case body. Testing any one of them is
    // sufficient to validate that code path.
    info.func     = ncclFuncAllReduce;
    info.datatype = ncclBfloat16;

    size_t nBytes = 16 * 1024 * 1024; // 16MB
    rcclSetPipelining(comm, nBytes, &info);

    EXPECT_EQ(info.pipeline, 0) << "gfx950 single-node should not enable pipelining";

    CleanupMockComm(comm);
}

TEST(Rcclwrap, RcclSetPipelining_GFX942_SingleNode_AllReduce_Enable)
{
    // Skip the test if pipelining has been disabled
    // (RCCL_DISABLE_REDUCE_COPY_PIPELINING=1)
    if(ShouldSkipRcclSetPipeliningTests())
    {
        GTEST_SKIP() << "Skipping test: RCCL_DISABLE_REDUCE_COPY_PIPELINING environment "
                        "variable is set. Unset this variable to enable pipelining.";
    }

    // For single-node, pipeline is set to 1 for AllReduce with bf16
    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx942", 8);
    comm->nNodes = 1; // Single node

    ncclTaskColl info = {};
    info.func         = ncclFuncAllReduce;
    info.datatype     = ncclBfloat16;

    size_t nBytes = 16 * 1024 * 1024; // 16MB
    rcclSetPipelining(comm, nBytes, &info);

    EXPECT_EQ(info.pipeline, 1) << "gfx942 single-node AllReduce bf16 should enable pipelining";

    CleanupMockComm(comm);
}

TEST(Rcclwrap, RcclSetPipelining_GFX942_MultiNode_AllReduce_Enable)
{
    // Skip the test if pipelining has been disabled
    // (RCCL_DISABLE_REDUCE_COPY_PIPELINING=1)
    if(ShouldSkipRcclSetPipeliningTests())
    {
        GTEST_SKIP() << "Skipping test: RCCL_DISABLE_REDUCE_COPY_PIPELINING environment "
                        "variable is set. Unset this variable to enable pipelining.";
    }

    // For multi-node AllReduce with bf16, pipelining is enabled if
    // nBytes <= 512MB * 2^(log2(nNodes)-1)
    // Testing with nNodes = 4  => threshold = 512MB * 2^(2-1) = 1GB
    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx942", 8);
    comm->nNodes = 4;

    ncclTaskColl info = {};
    info.func         = ncclFuncAllReduce;
    info.datatype     = ncclBfloat16;

    size_t nBytes = (1ULL << 30); // 1GB, exactly at threshold
    rcclSetPipelining(comm, nBytes, &info);

    EXPECT_EQ(info.pipeline, 1) << "gfx942 4-node AllReduce at threshold should enable pipelining";

    CleanupMockComm(comm);
}

TEST(Rcclwrap, RcclSetPipelining_GFX942_MultiNode_AllReduce_Disable)
{
    // Skip the test if pipelining has been disabled
    // (RCCL_DISABLE_REDUCE_COPY_PIPELINING=1)
    if(ShouldSkipRcclSetPipeliningTests())
    {
        GTEST_SKIP() << "Skipping test: RCCL_DISABLE_REDUCE_COPY_PIPELINING environment "
                        "variable is set. Unset this variable to enable pipelining.";
    }

    // When nBytes is just above the threshold, pipelining should be disabled
    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx942", 8);
    comm->nNodes = 4;

    ncclTaskColl info = {};
    info.func         = ncclFuncAllReduce;
    info.datatype     = ncclBfloat16;

    size_t nBytes = (1ULL << 30) + 1024; // 1GB + 1KB, just above threshold
    rcclSetPipelining(comm, nBytes, &info);

    EXPECT_EQ(info.pipeline, 0)
        << "gfx942 4-node AllReduce above threshold should disable pipelining";

    CleanupMockComm(comm);
}

TEST(Rcclwrap, RcclSetPipelining_GFX942_Enable)
{
    // Skip the test if pipelining has been disabled
    // (RCCL_DISABLE_REDUCE_COPY_PIPELINING=1)
    if(ShouldSkipRcclSetPipeliningTests())
    {
        GTEST_SKIP() << "Skipping test: RCCL_DISABLE_REDUCE_COPY_PIPELINING environment "
                        "variable is set. Unset this variable to enable pipelining.";
    }

    // ReduceScatter & Reduce should enable pipelining regardless of no. of nodes
    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx942", 8);
    comm->nNodes = 8;

    ncclTaskColl info = {};
    // In rcclSetPipelining(), ncclFuncReduceScatter, and
    // ncclFuncReduce share the same case body. Testing any one of them is
    // sufficient to validate that code path.
    info.func     = ncclFuncReduceScatter;
    info.datatype = ncclBfloat16;

    size_t nBytes = 16 * 1024 * 1024; // 16MB
    rcclSetPipelining(comm, nBytes, &info);

    EXPECT_EQ(info.pipeline, 1) << "gfx942 ReduceScatter and Reduce should enable "
                                   "pipelining with single or multi-node";

    CleanupMockComm(comm);
}

TEST(Rcclwrap, RcclOverrideProtocol_NoOverride)
{
  RUN_ISOLATED_TEST_WITH_ENV("RcclOverrideProtocol_NoOverride",
    []() {
      float        table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
      ncclTaskColl info;

      ncclResult_t result = rcclOverrideProtocol(ncclProtoStr, table, &info);

      EXPECT_EQ(result, ncclSuccess)
        << "Expected ncclSuccess when RCCL_OVERRIDE_PROTO is unset, indicating "
           "no override should be applied.";
    },
    {}
  );
}

TEST(Rcclwrap, RcclOverrideProtocol_UnsupportedOverride)
{
  RUN_ISOLATED_TEST_WITH_ENV("RcclOverrideProtocol_UnsupportedOverride",
    []() {
      // Mark all combinations as unsupported for the purpose of this test.
      float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
      for(int a = 0; a < NCCL_NUM_ALGORITHMS; ++a)
        for(int p = 0; p < NCCL_NUM_PROTOCOLS; ++p)
          table[a][p] = NCCL_ALGO_PROTO_IGNORE;

      ncclTaskColl info;
      info.func         = ncclFuncReduceScatter;
      info.datatype     = ncclBfloat16;
      info.algorithm    = NCCL_ALGO_RING;

      ncclResult_t result = rcclOverrideProtocol(ncclProtoStr, table, &info);

      EXPECT_EQ(result, ncclInternalError)
        << "Expected ncclInternalError when the override protocol is valid, but "
           "not enabled for the selected algorithm.";
    },
    {{"RCCL_OVERRIDE_PROTO", "Simple"}}
  );
}

TEST(Rcclwrap, RcclOverrideProtocol_ValidOverride)
{
  RUN_ISOLATED_TEST_WITH_ENV("RcclOverrideProtocol_ValidOverride",
    []() {
      const char* protoOverrideEnv = getenv("RCCL_OVERRIDE_PROTO");
      ASSERT_NE(protoOverrideEnv, nullptr) << "RCCL_OVERRIDE_PROTO should be set";

      // Get the index of the protocol from the string for later comparison
      int          protoIndex = NCCL_PROTO_UNDEF;
      ncclResult_t idxResult
        = rcclGetAlgoProtoIndex(protoOverrideEnv, ncclProtoStr, NCCL_NUM_PROTOCOLS, protoIndex);
      ASSERT_EQ(idxResult, ncclSuccess) << "Failed to get protocol index from string";

      // Mark all combinations as valid for the purpose of this test.
      float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
      for(int a = 0; a < NCCL_NUM_ALGORITHMS; ++a)
        for(int p = 0; p < NCCL_NUM_PROTOCOLS; ++p)
          table[a][p] = 0.0;

      ncclTaskColl info;
      info.func         = ncclFuncAllReduce;
      info.datatype     = ncclBfloat16;
      info.algorithm    = NCCL_ALGO_RING;
      info.protocol     = NCCL_PROTO_UNDEF;

      ncclResult_t result = rcclOverrideProtocol(ncclProtoStr, table, &info);

      EXPECT_EQ(result, ncclSuccess) << "Expected ncclSuccess when override is applied successfully.";
      EXPECT_EQ(info.protocol, protoIndex) << "Protocol index should match the "
                                              "override value from environment.";
    },
    {{"RCCL_OVERRIDE_PROTO", "Simple"}}
  );
}

TEST(Rcclwrap, RcclOverrideProtocol_ValidOverridePersists)
{
  RUN_ISOLATED_TEST_WITH_ENV("RcclOverrideProtocol_ValidOverridePersists",
    []() {
      const char* protoOverrideEnv = getenv("RCCL_OVERRIDE_PROTO");
      ASSERT_NE(protoOverrideEnv, nullptr) << "RCCL_OVERRIDE_PROTO should be set";

      // Get the index of the protocol from the string for later comparison
      int          protoIndex = NCCL_PROTO_UNDEF;
      ncclResult_t idxResult
        = rcclGetAlgoProtoIndex(protoOverrideEnv, ncclProtoStr, NCCL_NUM_PROTOCOLS, protoIndex);
      ASSERT_EQ(idxResult, ncclSuccess) << "Failed to get protocol index from string";

      // Mark all combinations as valid for the purpose of this test.
      float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
      for(int a = 0; a < NCCL_NUM_ALGORITHMS; ++a)
        for(int p = 0; p < NCCL_NUM_PROTOCOLS; ++p)
          table[a][p] = 0.0;

      ncclTaskColl info;
      info.func         = ncclFuncAllReduce;
      info.datatype     = ncclFloat16;
      info.algorithm    = NCCL_ALGO_RING;
      info.protocol     = NCCL_PROTO_UNDEF;

      // First call
      ncclResult_t result1 = rcclOverrideProtocol(ncclProtoStr, table, &info);
      EXPECT_EQ(result1, ncclSuccess)
        << "Expected rcclOverrideProtocol to succeed with valid override";
      EXPECT_EQ(info.protocol, protoIndex) << "Expected protocol to match override after first call";

      // Second call
      ncclResult_t result2 = rcclOverrideProtocol(ncclProtoStr, table, &info);
      EXPECT_EQ(result2, ncclSuccess)
        << "Expected rcclOverrideProtocol to succeed again on second call";
      EXPECT_EQ(info.protocol, protoIndex) << "Expected protocol to match override after second call";
    },
    {{"RCCL_OVERRIDE_PROTO", "Simple"}}
  );
}

TEST(Rcclwrap, RcclOverrideProtocol_InvalidProtocol)
{
  RUN_ISOLATED_TEST_WITH_ENV("RcclOverrideProtocol_InvalidProtocol",
    []() {
      float        table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
      ncclTaskColl info;

      ncclResult_t result = rcclOverrideProtocol(ncclProtoStr, table, &info);

      EXPECT_EQ(result, ncclInvalidUsage) << "Expected ncclInvalidUsage when the "
                                             "override protocol is invalid.";
    },
    {{"RCCL_OVERRIDE_PROTO", "InvalidProtocol"}}
  );
}

TEST(Rcclwrap, RcclOverrideProtocol_InvalidOverridePersists)
{
  RUN_ISOLATED_TEST_WITH_ENV("RcclOverrideProtocol_InvalidOverridePersists",
    []() {
      float        table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
      ncclTaskColl info;

      // First call should fail due to invalid proto string
      ncclResult_t result1 = rcclOverrideProtocol(ncclProtoStr, table, &info);
      EXPECT_EQ(result1, ncclInvalidUsage) << "Expected rcclOverrideProtocol to fail with invalid "
                                              "RCCL_OVERRIDE_PROTO.";

      // Second call should still fail because the static variable disables further
      // overrides
      ncclResult_t result2 = rcclOverrideProtocol(ncclProtoStr, table, &info);
      EXPECT_EQ(result2, ncclInvalidUsage)
        << "Expected rcclOverrideProtocol to continue returning failure after "
           "invalid proto was set.";
    },
    {{"RCCL_OVERRIDE_PROTO", "InvalidProtocol"}}
  );
}

TEST(Rcclwrap, RcclOverrideAlgorithm_NoOverride)
{
  RUN_ISOLATED_TEST_WITH_ENV("RcclOverrideAlgorithm_NoOverride",
    []() {
      float        table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
      ncclTaskColl info;

      ncclResult_t result = rcclOverrideAlgorithm(ncclAlgoStr, table, &info);

      // Since no override is set, it should return success and do nothing
      EXPECT_EQ(result, ncclSuccess)
        << "Expected ncclSuccess when RCCL_OVERRIDE_ALGO is unset, indicating no "
           "override should be applied.";
    },
    {}
  );
}

TEST(Rcclwrap, RcclOverrideAlgorithm_UnsupportedOverride)
{
  RUN_ISOLATED_TEST_WITH_ENV("RcclOverrideAlgorithm_UnsupportedOverride",
    []() {
      float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
      for(int a = 0; a < NCCL_NUM_ALGORITHMS; ++a)
        for(int p = 0; p < NCCL_NUM_PROTOCOLS; ++p)
          table[a][p] = NCCL_ALGO_PROTO_IGNORE;

      ncclTaskColl info;
      info.func         = ncclFuncReduceScatter;
      info.datatype     = ncclBfloat16;
      info.protocol     = NCCL_PROTO_SIMPLE;

      ncclResult_t result = rcclOverrideAlgorithm(ncclAlgoStr, table, &info);

      EXPECT_EQ(result, ncclInternalError)
        << "Expected ncclInternalError when the override algorithm is valid, but "
           "not enabled for the selected protocol.";
    },
    {{"RCCL_OVERRIDE_ALGO", "Ring"}}
  );
}

TEST(Rcclwrap, RcclOverrideAlgorithm_ValidOverride)
{
  RUN_ISOLATED_TEST_WITH_ENV("RcclOverrideAlgorithm_ValidOverride",
    []() {
      const char* algoOverrideEnv = getenv("RCCL_OVERRIDE_ALGO");
      ASSERT_NE(algoOverrideEnv, nullptr) << "RCCL_OVERRIDE_ALGO should be set";

      // Get the index of the algorithm from the string for later comparison
      int          algoIndex = NCCL_ALGO_UNDEF;
      ncclResult_t idxResult
        = rcclGetAlgoProtoIndex(algoOverrideEnv, ncclAlgoStr, NCCL_NUM_ALGORITHMS, algoIndex);
      ASSERT_EQ(idxResult, ncclSuccess) << "Failed to get algorithm index from string";

      float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
      // Mark all combinations as valid for the purpose of this test.
      for(int a = 0; a < NCCL_NUM_ALGORITHMS; ++a)
        for(int p = 0; p < NCCL_NUM_PROTOCOLS; ++p)
          table[a][p] = 0.0;

      ncclTaskColl info;
      info.func         = ncclFuncAllReduce;
      info.datatype     = ncclBfloat16;
      info.protocol     = NCCL_PROTO_SIMPLE;
      info.algorithm    = NCCL_ALGO_UNDEF;

      ncclResult_t result = rcclOverrideAlgorithm(ncclAlgoStr, table, &info);

      EXPECT_EQ(result, ncclSuccess) << "Expected ncclSuccess when override is applied successfully.";
      EXPECT_EQ(info.algorithm, algoIndex)
        << "Algorithm index should match the override value from environment.";
    },
    {{"RCCL_OVERRIDE_ALGO", "Ring"}}
  );
}

TEST(Rcclwrap, RcclOverrideAlgorithm_ValidOverridePersists)
{
  RUN_ISOLATED_TEST_WITH_ENV("RcclOverrideAlgorithm_ValidOverridePersists",
    []() {
      const char* algoOverrideEnv = getenv("RCCL_OVERRIDE_ALGO");
      ASSERT_NE(algoOverrideEnv, nullptr) << "RCCL_OVERRIDE_ALGO should be set";

      // Get the index of the algorithm from the string for later comparison
      int          algoIndex = NCCL_ALGO_UNDEF;
      ncclResult_t idxResult
        = rcclGetAlgoProtoIndex(algoOverrideEnv, ncclAlgoStr, NCCL_NUM_ALGORITHMS, algoIndex);
      ASSERT_EQ(idxResult, ncclSuccess) << "Failed to get algorithm index from string";

      // Mark all combinations as valid for the purpose of this test.
      float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
      for(int a = 0; a < NCCL_NUM_ALGORITHMS; ++a)
        for(int p = 0; p < NCCL_NUM_PROTOCOLS; ++p)
          table[a][p] = 0.0;

      ncclTaskColl info;
      info.func         = ncclFuncAllReduce;
      info.datatype     = ncclFloat16;
      info.protocol     = NCCL_PROTO_SIMPLE;
      info.algorithm    = NCCL_ALGO_UNDEF;

      // First call
      ncclResult_t result1 = rcclOverrideAlgorithm(ncclAlgoStr, table, &info);
      EXPECT_EQ(result1, ncclSuccess)
        << "Expected rcclOverrideAlgorithm to succeed with valid override.";
      EXPECT_EQ(info.algorithm, algoIndex)
        << "Expected algorithm to match override after first call.";

      // Second call
      ncclResult_t result2 = rcclOverrideAlgorithm(ncclAlgoStr, table, &info);
      EXPECT_EQ(result2, ncclSuccess)
        << "Expected rcclOverrideAlgorithm to succeed again on second call.";
      EXPECT_EQ(info.algorithm, algoIndex)
        << "Expected algorithm to match override after second call.";
    },
    {{"RCCL_OVERRIDE_ALGO", "Ring"}}
  );
}

TEST(Rcclwrap, RcclOverrideAlgorithm_InvalidAlgorithm)
{
  RUN_ISOLATED_TEST_WITH_ENV("RcclOverrideAlgorithm_InvalidAlgorithm",
    []() {
      float        table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
      ncclTaskColl info;

      ncclResult_t result = rcclOverrideAlgorithm(ncclAlgoStr, table, &info);

      EXPECT_EQ(result, ncclInvalidUsage)
        << "Expected ncclInvalidUsage when the override algorithm is invalid.";
    },
    {{"RCCL_OVERRIDE_ALGO", "InvalidAlgorithm"}}
  );
}

TEST(Rcclwrap, RcclOverrideAlgorithm_InvalidOverridePersists)
{
  RUN_ISOLATED_TEST_WITH_ENV("RcclOverrideAlgorithm_InvalidOverridePersists",
    []() {
      float        table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
      ncclTaskColl info;

      // First call should fail due to invalid algo string (and set the static flag)
      ncclResult_t result1 = rcclOverrideAlgorithm(ncclAlgoStr, table, &info);
      EXPECT_EQ(result1, ncclInvalidUsage) << "Expected rcclOverrideAlgorithm to fail with invalid "
                                              "RCCL_OVERRIDE_ALGO.";

      // Second call should also fail due to static validInput=false
      ncclResult_t result2 = rcclOverrideAlgorithm(ncclAlgoStr, table, &info);
      EXPECT_EQ(result2, ncclInvalidUsage)
        << "Expected rcclOverrideAlgorithm to continue returning failure after "
           "invalid algo was set.";
    },
    {{"RCCL_OVERRIDE_ALGO", "InvalidAlgorithm"}}
  );
}

TEST(Rcclwrap, AllrcclSetP2pNetChunkSizeTests)
{
    TEST_INFO(
        "=== Starting Process-Isolated rcclSetP2pNetChunkSize "
        "Tests Execution ==="
    );

    // Define test case structure
    struct P2PChunkSizeTestCase
    {
        std::string                                  name;
        std::string                                  arch;
        int                                          ranks;
        int                                          expectedChunkSize;
        std::unordered_map<std::string, std::string> extraEnv;
    };

    // Define all test cases
    std::vector<P2PChunkSizeTestCase> testCases = {
        // GFX942 tests
        {      "GFX942_LargeRanks_Isolated","gfx942",  128,1 << 19,                                                                  {}                                                            },
        {  "GFX942_BoundaryRank64_Isolated", "gfx942",   64,            1 << 19,                                                                  {}},
        {  "GFX942_BoundaryRank63_Isolated", "gfx942",   63,            1 << 17,                                                                  {}},

        // GFX950 tests
        {      "GFX950_SmallRanks_Isolated", "gfx950",    8,            1 << 17,                                                                  {}},
        {     "GFX950_MediumRanks_Isolated", "gfx950",   24,            1 << 18,                                                                  {}},
        {      "GFX950_LargeRanks_Isolated", "gfx950",   64,            1 << 19,                                                                  {}},
        {  "GFX950_BoundaryRank16_Isolated", "gfx950",   16,            1 << 18,                                                                  {}},
        {  "GFX950_BoundaryRank15_Isolated", "gfx950",   15,            1 << 17,                                                                  {}},
        {  "GFX950_BoundaryRank32_Isolated", "gfx950",   32,            1 << 19,                                                                  {}},
        {  "GFX950_BoundaryRank31_Isolated", "gfx950",   31,            1 << 18,                                                                  {}},

        // Unsupported architectures
        { "UnsupportedArch_GFX908_Isolated", "gfx908",   32, RCCL_VALUE_INVALID,                                                                  {}},
        { "UnsupportedArch_GFX90A_Isolated", "gfx90a",   32, RCCL_VALUE_INVALID,                                                                  {}},

        // Edge cases
        {        "EmptyArchString_Isolated",       "",   32, RCCL_VALUE_INVALID,                                                                  {}},
        {       "PartialArchMatch_Isolated",  "gfx94",   32, RCCL_VALUE_INVALID,                                                                  {}},
        {       "ZeroRanks_GFX942_Isolated", "gfx942",    0,            1 << 17,                                                                  {}},
        {       "ZeroRanks_GFX950_Isolated", "gfx950",    0,            1 << 17,                                                                  {}},
        { "LargeRankValues_GFX950_Isolated", "gfx950", 1000,            1 << 19,                                                                  {}},
        {    "CaseInsensitiveArch_Isolated", "GFX942",   32, RCCL_VALUE_INVALID,                                                                  {}},

        // Environment variable test
        {"WithEnvironmentVariable_Isolated",
         "gfx942",   32,
         RCCL_VALUE_UNSET, {{"NCCL_P2P_NET_CHUNKSIZE", "123456"}, {"NCCL_MAX_NCHANNELS", "1"}}                                                      }
    };

    // Base environment for all tests
    std::unordered_map<std::string, std::string> baseEnv = {
        {       "NCCL_DEBUG", "TRACE"},
        {"NCCL_DEBUG_SUBSYS",   "ALL"}
    };

    // Register all tests using a loop
    for(const auto& tc : testCases)
    {
        ProcessIsolatedTestRunner::registerTest(
            ProcessIsolatedTestRunner::TestConfig(
                tc.name,
                [tc]()
                {
                    ncclComm_t            mockComm = nullptr;
                    auto                  mockTopo = std::make_unique<ncclTopoSystem>();
                    struct ncclTopoNode   mockGpuNode;
                    CreateMockComm(mockComm, *mockTopo, mockGpuNode, tc.arch.c_str(), tc.ranks);

                    int chunkSize = RCCL_VALUE_UNSET;
                    rcclSetP2pNetChunkSize(mockComm, chunkSize);

                    // Special handling for environment variable test
                    if(tc.name == "WithEnvironmentVariable_Isolated")
                    {
                        const char* envValue = getenv("NCCL_P2P_NET_CHUNKSIZE");
                        EXPECT_STREQ(envValue, "123456")
                            << "Environment variable should be set to 123456";
                        EXPECT_NE(chunkSize, RCCL_VALUE_UNSET)
                            << "Environment variable should override default logic";
                    }
                    else
                    {
                        EXPECT_EQ(chunkSize, tc.expectedChunkSize)
                            << "Failed for " << tc.arch << " with " << tc.ranks << " ranks";
                    }

                    CleanupMockComm(mockComm);
                }
            )
                .withEnvironment(
                    [&tc, &baseEnv]()
                    {
                        auto env = baseEnv;
                        env.insert(tc.extraEnv.begin(), tc.extraEnv.end());
                        return env;
                    }()
                )
                .withTimeout(std::chrono::seconds(60))
                .withNumGpus(0)
        );
    }

    // Configure execution options
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false; // Continue running all tests
    options.verboseLogging     = true;

    // Execute all tests
    bool allTestsPassed = ProcessIsolatedTestRunner::executeAllTests(options);

    // Verify that all tests passed
    EXPECT_TRUE(allTestsPassed) << "One or more process-isolated GFX tests failed";

    TEST_INFO(
        "=== Process-Isolated rcclSetP2pNetChunkSize Tests "
        "Execution Completed ==="
    );
}

TEST(Rcclwrap, AllPxnTests)
{
    // Define test case structure
    struct PxnTestCase
    {
        std::string                                  name;
        std::string                                  arch;
        int                                          ranks;
        int                                          expectedPxnDisable;
        std::unordered_map<std::string, std::string> extraEnv;
        bool shouldSkipCheck; // For tests with environment variable set
    };

    // Define all test cases
    std::vector<PxnTestCase> testCases = {
        // GFX942 tests
        {      "PXN_GFX942_SmallRanks_Isolated","gfx942",  32,   1,                          {},true                                                                                                                },
        {      "PXN_GFX942_LargeRanks_Isolated", "gfx942", 128,                  0,                          {}, true},
        {  "PXN_GFX942_BoundaryRank64_Isolated", "gfx942",  64,                  0,                          {}, true},
        {  "PXN_GFX942_BoundaryRank63_Isolated", "gfx942",  63,                  1,                          {}, true},

        // GFX950 tests
        {      "PXN_GFX950_SmallRanks_Isolated", "gfx950",   8,                  1,                          {}, true},
        {      "PXN_GFX950_LargeRanks_Isolated", "gfx950",  64,                  0,                          {}, true},
        {  "PXN_GFX950_BoundaryRank32_Isolated", "gfx950",  32,                  0,                          {}, true},
        {  "PXN_GFX950_BoundaryRank31_Isolated", "gfx950",  31,                  1,                          {}, true},

        // Unsupported architecture
        { "PXN_UnsupportedArch_GFX908_Isolated", "gfx908",  32, RCCL_VALUE_INVALID,                          {}, true},

        // Environment variable test (no skip check needed)
        {"PXN_WithEnvironmentVariable_Isolated",
         "gfx942",  32,
         RCCL_VALUE_INVALID, {{"NCCL_PXN_DISABLE", "1"}},
         false                                                                                                       }
    };

    // Base environment for all tests
    std::unordered_map<std::string, std::string> baseEnv = {
        {       "NCCL_DEBUG", "TRACE"},
        {"NCCL_DEBUG_SUBSYS",   "ALL"}
    };

    // Register all tests using a loop
    for(const auto& tc : testCases)
    {
        ProcessIsolatedTestRunner::registerTest(
            ProcessIsolatedTestRunner::TestConfig(
                tc.name,
                [tc]()
                {
                    // Check if we should skip this test due to environment variable being
                    // set
                    if(tc.shouldSkipCheck && ShouldSkipPxnTest())
                    {
                        GTEST_SKIP()
                            << "Skipping " << tc.name << " due to environment variable being set";
                        return;
                    }

                    TEST_INFO(
                        "Testing rcclSetPxn for %s with %d ranks",
                        tc.arch.c_str(),
                        tc.ranks
                    );

                    ncclComm_t            mockComm = nullptr;
                    auto                  mockTopo = std::make_unique<ncclTopoSystem>();
                    struct ncclTopoNode   mockGpuNode;
                    CreateMockComm(mockComm, *mockTopo, mockGpuNode, tc.arch.c_str(), tc.ranks);

                    int pxnDisable = RCCL_VALUE_UNSET;
                    rcclSetPxn(mockComm, pxnDisable);

                    EXPECT_EQ(pxnDisable, tc.expectedPxnDisable)
                        << "Failed for " << tc.arch << " with " << tc.ranks << " ranks";

                    TEST_INFO(
                        "%s test completed - pxnDisable: %d",
                        tc.name.c_str(),
                        pxnDisable
                    );
                    CleanupMockComm(mockComm);
                }
            )
                .withEnvironment(
                    [&tc, &baseEnv]()
                    {
                        auto env = baseEnv;
                        env.insert(tc.extraEnv.begin(), tc.extraEnv.end());
                        return env;
                    }()
                )
                .withNumGpus(0)
        );
    }

    // Configure execution options for sequential execution with stop on first
    // failure
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = true;
    options.verboseLogging     = true;

    // Execute all registered tests
    bool allTestsPassed = ProcessIsolatedTestRunner::executeAllTests(options);

    EXPECT_TRUE(allTestsPassed) << "One or more PXN process-isolated tests failed";
}

TEST(Rcclwrap, CucheckMacro_CheckStickyHipErrorOnFailure)
{
    hipError_t hipErr = hipSetDevice(0);
    if(hipErr != hipSuccess)
    {
        GTEST_SKIP() << "No GPU available";
    }

    // Clear any pre-existing sticky error so we start clean
    (void)hipGetLastError();

    // Force a HIP failure through CUCHECK using an invalid device pointer (0x1)
    ncclResult_t ret = triggerCucheckFailure();

    EXPECT_EQ(ncclUnhandledCudaError, ret)
        << "CUCHECK should return ncclUnhandledCudaError on failure";
    EXPECT_EQ(hipSuccess, hipGetLastError())
        << "CUCHECK must clear sticky HIP error after failure";
}

TEST(Rcclwrap, CucheckgotoMacro_CheckStickyHipErrorOnFailure)
{
    hipError_t hipErr = hipSetDevice(0);
    if(hipErr != hipSuccess)
    {
        GTEST_SKIP() << "No GPU available";
    }

    // Clear any pre-existing sticky error so we start clean
    (void)hipGetLastError();

    // Force a HIP failure through CUCHECKGOTO using an invalid device pointer (0x1)
    ncclResult_t ret = ncclSuccess;
    CUCHECKGOTO(hipPointerGetAttribute(nullptr, HIP_POINTER_ATTRIBUTE_CONTEXT,
                                       (hipDeviceptr_t)0x1),
                ret, check_sticky);

check_sticky:
    EXPECT_EQ(ncclUnhandledCudaError, ret)
        << "CUCHECKGOTO should set result to ncclUnhandledCudaError on failure";
    EXPECT_EQ(hipSuccess, hipGetLastError())
        << "CUCHECKGOTO must clear sticky HIP error after failure";
}

TEST(Rcclwrap, RcclUseAllGatherDirectNodeCountTests)
{
    TEST_INFO("=== Starting Process-Isolated rcclUseAllGatherDirect Node Count Tests ===");

    // Test case structure for AllGather Direct node-count gating tests
    struct AGDirectNodeCountTestCase
    {
        std::string                                  name;
        std::string                                  arch;
        int                                          nRanks;
        int                                          nNodes;
        bool                                         requiresAinic; // Skip if AINIC not present
        std::unordered_map<std::string, std::string> extraEnv;
    };

    std::vector<AGDirectNodeCountTestCase> testCases = {
        // nNodes > 32: must return false for any NIC type (hardware-independent)
        {
            "AGDirect_Disabled_nNodes33",
            "gfx942",
            128, // >= 64 ranks so PXN is enabled for gfx942
            33,  // > 32 nodes
            false,
            {}
        },
        // nNodes > 16 with AINIC: must return false (skipped when AINIC hardware absent)
        {
            "AGDirect_Disabled_AINIC_nNodes17",
            "gfx942",
            128, // >= 64 ranks so PXN is enabled for gfx942
            17,  // > 16 nodes
            true,
            {}
        },
    };

    // Base environment: clean state, no env vars that would short-circuit earlier checks
    std::unordered_map<std::string, std::string> baseEnv = {
        {       "NCCL_DEBUG", "TRACE"},
        {"NCCL_DEBUG_SUBSYS",   "ALL"}
    };

    for(const auto& tc : testCases)
    {
        ProcessIsolatedTestRunner::registerTest(
            ProcessIsolatedTestRunner::TestConfig(
                tc.name,
                [tc]()
                {
                    // Skip AINIC tests when AINIC hardware is not present, since
                    // rcclUseAinic() relies on hardware detection and cannot be
                    // forced in a test environment without actual AINIC NICs.
                    if(tc.requiresAinic && !rcclUseAinic())
                    {
                        GTEST_SKIP() << "Skipping " << tc.name
                                     << ": AINIC hardware not present";
                        return;
                    }

                    ncclComm_t            mockComm = nullptr;
                    auto                  mockTopo = std::make_unique<ncclTopoSystem>();
                    struct ncclTopoNode   mockGpuNode;
                    CreateMockComm(mockComm, *mockTopo, mockGpuNode, tc.arch.c_str(), tc.nRanks);
                    mockComm->nNodes = tc.nNodes;

                    // Use a message size that passes the threshold check so only
                    // the node-count guards are responsible for any false return.
                    size_t msgSize = 8388608; // 8 MiB

                    bool result = rcclUseAllGatherDirect(mockComm, msgSize);

                    EXPECT_FALSE(result)
                        << "Expected rcclUseAllGatherDirect to return false for "
                        << tc.arch << " with nNodes=" << tc.nNodes;

                    CleanupMockComm(mockComm);
                }
            )
                .withEnvironment(
                    [&tc, &baseEnv]()
                    {
                        auto env = baseEnv;
                        env.insert(tc.extraEnv.begin(), tc.extraEnv.end());
                        return env;
                    }()
                )
                .withTimeout(std::chrono::seconds(60))
                .withNumGpus(0)
        );
    }

    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    bool allTestsPassed = ProcessIsolatedTestRunner::executeAllTests(options);

    EXPECT_TRUE(allTestsPassed) << "One or more AllGather Direct node count tests failed";

    TEST_INFO("=== Process-Isolated rcclUseAllGatherDirect Node Count Tests Completed ===");
}

TEST(Rcclwrap, RcclUseHierarchicalAllGatherTests)
{
    TEST_INFO("=== Starting Process-Isolated rcclUseHierarchicalAllGather Tests ===");
    struct HierAGCase
    {
        std::string                                  name;
        int                                          nNodes;
        bool                                         hierCommsInit;
        size_t                                       msgSize;
        bool                                         expected;
        std::unordered_map<std::string, std::string> extraEnv;
    };

    const size_t HALF = HIERARCHICAL_TEMP_BUFFER_SIZE / 2;
    const size_t QUARTER = HIERARCHICAL_TEMP_BUFFER_SIZE / 4;
    const size_t FULL = HIERARCHICAL_TEMP_BUFFER_SIZE;

    std::vector<HierAGCase> testCases = {
        // nNodes < 8 --> disabled
        {"LessThan8Nodes",               4,  true,  1ULL << 20,  false, {}},
        // sub-comms not initialized --> disabled
        {"CommsNotInitialized",          16, false, 1ULL << 20,  false, {}},
        // 8-15 nodes --> limit is 32MB
        {"Enabled_8Nodes_AtQuarter",     8,  true,  QUARTER,     true,  {}},
        {"Disabled_8Nodes_AboveQuarter", 8,  true,  QUARTER + 1, false, {}},
        {"Enabled_15Nodes_AtQuarter",    15, true,  QUARTER,     true,  {}},
        {"Disabled_15Nodes_AtHalf",      15, true,  HALF,        false, {}},
        // 16-31 nodes --> limit is 64MB
        {"Enabled_16Nodes_AtHalf",       16, true,  HALF,        true,  {}},
        {"Disabled_16Nodes_AboveHalf",   16, true,  HALF + 1,    false, {}},
        {"Enabled_31Nodes_AtHalf",       31, true,  HALF,        true,  {}},
        {"Disabled_31Nodes_AtFull",      31, true,  FULL,        false, {}},
        // 32+ nodes --> limit is 128MB
        {"Enabled_32Nodes_AtHalf",       32, true,  HALF,        true,  {}},
        {"Enabled_32Nodes_AtFull",       32, true,  FULL,        true,  {}},
        {"Disabled_32Nodes_AboveFull",   32, true,  FULL + 1,    false, {}},
        {"Enabled_64Nodes_AtFull",       64, true,  FULL,        true,  {}},
        // env var forces off --> disabled
        {"DisabledByEnvVar",             16, true,  1ULL << 20,  false, {{"RCCL_HIERARCHICAL_ALLGATHER", "0"}}},
    };

    // Base environment shared by every case
    std::unordered_map<std::string, std::string> baseEnv = {
        {       "NCCL_DEBUG", "TRACE"},
        {"NCCL_DEBUG_SUBSYS",   "ALL"}
    };

    for(const auto& tc : testCases)
    {
        ProcessIsolatedTestRunner::registerTest(
            ProcessIsolatedTestRunner::TestConfig(
                tc.name,
                [tc]()
                {
                    ncclComm_t            mockComm = nullptr;
                    auto                  mockTopo = std::make_unique<ncclTopoSystem>();
                    struct ncclTopoNode   mockGpu;
                    CreateMockComm(mockComm,
                                   *mockTopo,
                                   mockGpu,
                                   "gfx942",
                                   /*nRanks=*/8 * tc.nNodes);
                    mockComm->nNodes                       = tc.nNodes;
                    mockComm->hierarchicalCommsInitialized = tc.hierCommsInit;

                    EXPECT_EQ(rcclUseHierarchicalAllGather(mockComm, tc.msgSize),
                              tc.expected)
                        << "Case: " << tc.name
                        << " (nNodes=" << tc.nNodes
                        << ", hierCommsInit=" << tc.hierCommsInit
                        << ", msgSize=" << tc.msgSize << ")";

                    CleanupMockComm(mockComm);
                }
            )
                .withEnvironment(
                    [&tc, &baseEnv]()
                    {
                        auto env = baseEnv;
                        env.insert(tc.extraEnv.begin(), tc.extraEnv.end());
                        return env;
                    }()
                )
                .withTimeout(std::chrono::seconds(60))
        );
    }

    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    bool allTestsPassed = ProcessIsolatedTestRunner::executeAllTests(options);

    EXPECT_TRUE(allTestsPassed)
        << "One or more rcclUseHierarchicalAllGather tests failed";

    TEST_INFO("=== Process-Isolated rcclUseHierarchicalAllGather Tests Completed ===");
}

TEST(Rcclwrap, RcclUseHierarchicalReduceScatterTests)
{
    TEST_INFO("=== Starting Process-Isolated rcclUseHierarchicalReduceScatter Tests ===");
    struct HierRSCase
    {
        std::string                                  name;
        int                                          nNodes;
        bool                                         hierCommsInit;
        size_t                                       msgSize;
        bool                                         expected;
        std::unordered_map<std::string, std::string> extraEnv;
    };

    const size_t HALF = HIERARCHICAL_TEMP_BUFFER_SIZE / 2; // 8-node threshold (64MB)
    const size_t FULL = HIERARCHICAL_TEMP_BUFFER_SIZE;     // 16-node threshold (128MB)

    std::vector<HierRSCase> testCases = {
        // nNodes < 8 --> disabled
        {"LessThan8Nodes",            4,  true,  1ULL << 20, false, {{"RCCL_HIERARCHICAL_REDUCE_SCATTER", "1"}}},
        // sub-comms not initialized --> disabled
        {"CommsNotInitialized",       16, false, 1ULL << 20, false, {{"RCCL_HIERARCHICAL_REDUCE_SCATTER", "1"}}},
        // 8 node size > 64MB --> disabled
        {"Disabled_8Nodes_AboveHalf", 8,  true,  HALF + 1,   false, {{"RCCL_HIERARCHICAL_REDUCE_SCATTER", "1"}}},
        // 16 node size > 128MB --> disabled
        {"Disabled_16N_AboveFull",    16, true,  FULL + 1,   false, {{"RCCL_HIERARCHICAL_REDUCE_SCATTER", "1"}}},
        // disabled by default
        {"DisabledByDefault",          16, true,  1ULL << 20, false, {}},
        // env var forces off --> disabled
        {"DisabledByEnvVar",          16, true,  1ULL << 20, false, {{"RCCL_HIERARCHICAL_REDUCE_SCATTER", "0"}}},
        // 8 nodes, initialized, below threshold --> enabled
        {"Enabled_8Nodes_BelowHalf",  8,  true,  1ULL << 20, true,  {{"RCCL_HIERARCHICAL_REDUCE_SCATTER", "1"}}},
        // 8 nodes, exactly at threshold --> enabled
        {"Enabled_8Nodes_AtHalf",     8,  true,  HALF,       true,  {{"RCCL_HIERARCHICAL_REDUCE_SCATTER", "1"}}},
        // 16 nodes, initialized, below threshold --> enabled
        {"Enabled_16Nodes_BelowFull", 16, true,  1ULL << 20, true,  {{"RCCL_HIERARCHICAL_REDUCE_SCATTER", "1"}}},
        // 16 nodes, exactly at threshold --> enabled
        {"Enabled_16Nodes_AtFull",    16, true,  FULL,       true,  {{"RCCL_HIERARCHICAL_REDUCE_SCATTER", "1"}}},
    };

    // Base environment shared by every case
    std::unordered_map<std::string, std::string> baseEnv = {
        {       "NCCL_DEBUG", "TRACE"},
        {"NCCL_DEBUG_SUBSYS",   "ALL"}
    };

    for(const auto& tc : testCases)
    {
        ProcessIsolatedTestRunner::registerTest(
            ProcessIsolatedTestRunner::TestConfig(
                tc.name,
                [tc]()
                {
                    ncclComm_t            mockComm = nullptr;
                    auto                  mockTopo = std::make_unique<ncclTopoSystem>();
                    struct ncclTopoNode   mockGpu;
                    CreateMockComm(mockComm,
                                   *mockTopo,
                                   mockGpu,
                                   "gfx942",
                                   /*nRanks=*/8 * tc.nNodes);
                    mockComm->nNodes                       = tc.nNodes;
                    mockComm->hierarchicalCommsInitialized = tc.hierCommsInit;

                    EXPECT_EQ(rcclUseHierarchicalReduceScatter(mockComm, tc.msgSize),
                              tc.expected)
                        << "Case: " << tc.name
                        << " (nNodes=" << tc.nNodes
                        << ", hierCommsInit=" << tc.hierCommsInit
                        << ", msgSize=" << tc.msgSize << ")";

                    CleanupMockComm(mockComm);
                }
            )
                .withEnvironment(
                    [&tc, &baseEnv]()
                    {
                        auto env = baseEnv;
                        env.insert(tc.extraEnv.begin(), tc.extraEnv.end());
                        return env;
                    }()
                )
                .withTimeout(std::chrono::seconds(60))
        );
    }

    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    bool allTestsPassed = ProcessIsolatedTestRunner::executeAllTests(options);

    EXPECT_TRUE(allTestsPassed)
        << "One or more rcclUseHierarchicalReduceScatter tests failed";

    TEST_INFO("=== Process-Isolated rcclUseHierarchicalReduceScatter Tests Completed ===");
}

// Direct ReduceScatter reduces the gathered peer slices with PreOpSrcs=0 and
// postOp=false, an unscaled sum. ncclAvg is rewritten to PreMulSum, so selecting that
// backend for it returns the sum instead of the mean. The selector must keep avg (and
// user-defined PreMulSum) on the ring kernel even when the arch/node/size window is
// satisfied, and must still choose the direct path for the unscaled ops.
TEST(Rcclwrap, ReduceScatterSelectionKeepsDirectPathOffScaledOps)
{
    ncclComm_t            mockComm = nullptr;
    struct ncclTopoSystem mockTopo;
    struct ncclTopoNode   mockGpu;
    CreateMockComm(mockComm, mockTopo, mockGpu, "gfx950", /*nRanks=*/16);
    SetMockNodes(mockComm, /*nNodes=*/2, /*topoNRanks=*/16);
    // CreateMockComm leaves archName null, which the DDA gate dereferences.
    mockComm->archName = const_cast<char*>("gfx950");
    // The direct path needs PXN; seed the per-comm cache so this does not depend on
    // NCCL_PXN_DISABLE or on the rank-count auto-detect heuristic.
    mockComm->pxnDisable = 0;

    // rcclSelectReduceScatter compares nRanks * recvcount * typeSize against the
    // window, so this lands at 256 KiB, inside the 2-node 128 KiB .. 2 MiB range.
    const size_t recvcount = (256 * 1024) / (16 * sizeof(float));
    // Only inspected to look up symmetric windows, which a mock comm never has, so
    // these are never dereferenced.
    char sendbuff = 0;
    char recvbuff = 0;

    auto selectedAlgo = [&](ncclRedOp_t op) {
        struct rcclCollDecision decision = {};
        EXPECT_EQ(rcclSelectReduceScatter(mockComm, &sendbuff, &recvbuff, recvcount, ncclFloat32, op,
                                          /*query=*/false, &decision),
                  ncclSuccess);
        return decision.algo;
    };

    ASSERT_EQ(selectedAlgo(ncclSum), static_cast<int>(RCCL_DIRECT_REDUCESCATTER))
        << "mock comm must be direct-eligible for the redop expectations below to mean anything";
    EXPECT_EQ(selectedAlgo(ncclMin), static_cast<int>(RCCL_DIRECT_REDUCESCATTER));
    EXPECT_NE(selectedAlgo(ncclAvg), static_cast<int>(RCCL_DIRECT_REDUCESCATTER));
    EXPECT_NE(selectedAlgo(static_cast<ncclRedOp_t>(ncclNumOps)), static_cast<int>(RCCL_DIRECT_REDUCESCATTER));

    CleanupMockComm(mockComm);
}

TEST(Rcclwrap, RcclHierarchicalTempBufferSizeTests)
{
    const size_t QUARTER = HIERARCHICAL_TEMP_BUFFER_SIZE / 4;
    const size_t HALF    = HIERARCHICAL_TEMP_BUFFER_SIZE / 2;
    const size_t FULL    = HIERARCHICAL_TEMP_BUFFER_SIZE;

    // Below 8 nodes neither algorithm is eligible, so nothing needs to be allocated.
    EXPECT_EQ(rcclHierarchicalTempBufferSize(7, true, true), size_t{0});
    EXPECT_EQ(rcclHierarchicalTempBufferSize(64, false, false), size_t{0});

    // Per-collective thresholds
    EXPECT_EQ(rcclHierarchicalTempBufferSize(8, true, false), QUARTER);
    EXPECT_EQ(rcclHierarchicalTempBufferSize(15, true, false), QUARTER);
    EXPECT_EQ(rcclHierarchicalTempBufferSize(16, true, false), HALF);
    EXPECT_EQ(rcclHierarchicalTempBufferSize(31, true, false), HALF);
    EXPECT_EQ(rcclHierarchicalTempBufferSize(32, true, false), FULL);

    EXPECT_EQ(rcclHierarchicalTempBufferSize(8, false, true), HALF);
    EXPECT_EQ(rcclHierarchicalTempBufferSize(15, false, true), HALF);
    EXPECT_EQ(rcclHierarchicalTempBufferSize(16, false, true), FULL);
    EXPECT_EQ(rcclHierarchicalTempBufferSize(32, false, true), FULL);
}

TEST(Rcclwrap, RcclHierarchicalAlgoInfoTests)
{
    TEST_INFO("=== Starting Process-Isolated rcclHierarchicalAlgoInfo Tests ===");

    ProcessIsolatedTestRunner::registerTest(
        ProcessIsolatedTestRunner::TestConfig(
            "HierAlgoInfo_DirectAllGather_ReportsInterComm",
            []()
            {
                if(rcclUseAinic())
                {
                    GTEST_SKIP() << "Direct AllGather is disabled on AINIC";
                }

                ncclComm_t          parent = nullptr, inter = nullptr, intra = nullptr;
                struct ncclTopoNode parentGpu, interGpu, intraGpu;

                auto parentTopo = std::make_unique<ncclTopoSystem>();
                auto interTopo  = std::make_unique<ncclTopoSystem>();
                auto intraTopo  = std::make_unique<ncclTopoSystem>();

                // 8 nodes x 8 local ranks.
                CreateMockComm(parent, *parentTopo, parentGpu, "gfx942", 64);
                CreateMockComm(inter, *interTopo, interGpu, "gfx942", 8);
                CreateMockComm(intra, *intraTopo, intraGpu, "gfx942", 8);

                parent->p2pnChannels          = 32;
                inter->p2pnChannels           = 12;
                intra->p2pnChannels           = 5;
                parent->hierarchicalInterComm = inter;
                parent->hierarchicalIntraComm = intra;

                // 4 KiB per rank keeps both phases well under the Direct threshold.
                const uint64_t count    = 1024;
                size_t         interMsg = count * sizeof(float) * inter->nRanks;
                size_t         intraMsg = count * inter->nRanks * sizeof(float) * intra->nRanks;

                if(!rcclUseAllGatherDirect(inter, interMsg) || !rcclUseAllGatherDirect(intra, intraMsg))
                {
                    CleanupMockComm(parent);
                    CleanupMockComm(inter);
                    CleanupMockComm(intra);
                    GTEST_SKIP() << "Direct AllGather unavailable in this environment; the "
                                    "tuner fallback cannot run against a mock communicator";
                }

                int algo = -1, protocol = -1, maxChannels = -1;
                EXPECT_EQ(rcclHierarchicalAlgoInfo(parent,
                                                   ncclFuncAllGather,
                                                   count,
                                                   ncclFloat32,
                                                   &algo,
                                                   &protocol,
                                                   &maxChannels),
                          ncclSuccess);

                EXPECT_EQ(algo, RCCL_HIERARCHICAL_ALLGATHER);
                EXPECT_EQ(protocol, NCCL_PROTO_SIMPLE);
                EXPECT_EQ(maxChannels, inter->p2pnChannels);

                CleanupMockComm(parent);
                CleanupMockComm(inter);
                CleanupMockComm(intra);
            }
        )
            // Direct AllGather bails out when user buffer registration is enabled.
            .withEnvironment({{"NCCL_LOCAL_REGISTER", "0"}, {"RCCL_DIRECT_ALLGATHER_DISABLE", "0"}})
            .clearVariable("RCCL_DIRECT_ALLGATHER_THRESHOLD")
            .withTimeout(std::chrono::seconds(60))
            .withNumGpus(0)
    );

    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    EXPECT_TRUE(ProcessIsolatedTestRunner::executeAllTests(options))
        << "rcclHierarchicalAlgoInfo test failed";

    TEST_INFO("=== Process-Isolated rcclHierarchicalAlgoInfo Tests Completed ===");
}

// ===========================================================================
// Direct ReduceScatter getAlgoInfo host-side guard (AICOMRCCL-1819).
// When enableDirectReduceScatter is set, getAlgoInfo must force Ring/Simple
// even if the tuner or RCCL_OVERRIDE_* env picks LL.
// ===========================================================================

namespace
{

constexpr uint64_t kDirectRsTestCount = 65536;

ncclResult_t mockRingLlTunerGetCollInfo(void* /*context*/, ncclFunc_t /*collType*/, size_t /*nBytes*/,
                                        int /*numPipeOps*/, float** collCostTable, int numAlgo, int numProto,
                                        int /*regBuff*/, int* nChannels)
{
    float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;
    for(int a = 0; a < numAlgo; ++a)
    {
        for(int p = 0; p < numProto; ++p)
        {
            table[a][p] = NCCL_ALGO_PROTO_IGNORE;
        }
    }
    table[NCCL_ALGO_RING][NCCL_PROTO_LL] = 0.0;
    if(nChannels)
    {
        *nChannels = 0;
    }
    return ncclSuccess;
}

ncclTuner_t g_mockRingLlTuner = {
  .name         = "MockRingLlTuner",
  .init         = nullptr,
  .getCollInfo  = mockRingLlTunerGetCollInfo,
  .finalize     = nullptr,
  .getChunkSize = nullptr,
};

void InitGetAlgoInfoMockComm(ncclComm_t&           comm,
                             struct ncclTopoSystem& topo,
                             const char*            arch,
                             int                    nRanks,
                             int                    nNodes)
{
    struct ncclTopoNode gpuNode{};
    CreateMockComm(comm, topo, gpuNode, arch, nRanks);
    comm->nNodes        = nNodes;
    comm->nChannels     = 4;
    comm->collChannels  = 4;
    comm->nvlsChannels  = 4;
    comm->WarpSize      = 64;
    comm->maxLocalRanks = std::max(1, nRanks / std::max(1, nNodes));
    comm->topo->tuning  = rcclGetTuningIndexForArch(arch);
    comm->topo->type |= RCCL_TOPO_XGMI_ALL;

    for(int a = 0; a < NCCL_NUM_ALGORITHMS; ++a)
    {
        for(int p = 0; p < NCCL_NUM_PROTOCOLS; ++p)
        {
            comm->maxThreads[a][p]                      = 256;
            comm->threadThresholds[a][p]                 = 8192;
            comm->bandwidths[ncclFuncReduceScatter][a][p] = 100.0f;
            comm->latencies[ncclFuncReduceScatter][a][p]  = 1.0f;
        }
    }
}

ncclTaskColl MakeReduceScatterTask(uint64_t count = kDirectRsTestCount, ncclDataType_t dt = ncclFloat32)
{
    ncclTaskColl task{};
    task.func     = ncclFuncReduceScatter;
    task.count    = count;
    task.datatype = dt;
    return task;
}

void ExpectGetAlgoInfoSelection(ncclComm_t comm, ncclTaskColl& task, int algo, int proto)
{
    ASSERT_EQ(ncclGetAlgoInfo(comm, &task, 0, 0, 1, nullptr), ncclSuccess);
    EXPECT_EQ(task.algorithm, algo);
    EXPECT_EQ(task.protocol, proto);
}

} // namespace

// Tuner picks Ring+LL; Direct RS host guard must override to Ring/Simple.
TEST(Rcclwrap, GetAlgoInfo_DirectRsOverridesMockTunerRingLl)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "GetAlgoInfo_DirectRsOverridesMockTunerRingLl",
        []()
        {
            ncclComm_t          comm = nullptr;
            auto                topo = std::make_unique<ncclTopoSystem>();
            InitGetAlgoInfoMockComm(comm, *topo, "gfx950", 8, 2);
            comm->enableDirectReduceScatter = true;
            comm->tuner                     = &g_mockRingLlTuner;

            ncclTaskColl task = MakeReduceScatterTask();
            ExpectGetAlgoInfoSelection(comm, task, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
            CleanupMockComm(comm);
        },
        {}
    );
}

// Same tuner pick without Direct RS: selection must stay Ring+LL.
TEST(Rcclwrap, GetAlgoInfo_MockTunerRingLlPreservedWithoutDirectRs)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "GetAlgoInfo_MockTunerRingLlPreservedWithoutDirectRs",
        []()
        {
            ncclComm_t          comm = nullptr;
            auto                topo = std::make_unique<ncclTopoSystem>();
            InitGetAlgoInfoMockComm(comm, *topo, "gfx950", 8, 2);
            comm->enableDirectReduceScatter = false;
            comm->tuner                     = &g_mockRingLlTuner;

            ncclTaskColl task = MakeReduceScatterTask();
            ExpectGetAlgoInfoSelection(comm, task, NCCL_ALGO_RING, NCCL_PROTO_LL);
            CleanupMockComm(comm);
        },
        {}
    );
}

// RCCL_OVERRIDE_PROTO=LL forces LL; Direct RS host guard must still pick Ring/Simple.
TEST(Rcclwrap, GetAlgoInfo_DirectRsForcesSimpleDespiteLlOverride)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "GetAlgoInfo_DirectRsForcesSimpleDespiteLlOverride",
        []()
        {
            ncclComm_t          comm = nullptr;
            auto                topo = std::make_unique<ncclTopoSystem>();
            InitGetAlgoInfoMockComm(comm, *topo, "gfx950", 8, 2);
            comm->enableDirectReduceScatter = true;

            ncclTaskColl task = MakeReduceScatterTask();
            ExpectGetAlgoInfoSelection(comm, task, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
            CleanupMockComm(comm);
        },
        {{"RCCL_OVERRIDE_PROTO", "LL"}, {"RCCL_OVERRIDE_ALGO", "Ring"}}
    );
}

// Same env override without Direct RS: selection must stay Ring+LL.
TEST(Rcclwrap, GetAlgoInfo_LlOverridePreservedWithoutDirectRs)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "GetAlgoInfo_LlOverridePreservedWithoutDirectRs",
        []()
        {
            ncclComm_t          comm = nullptr;
            auto                topo = std::make_unique<ncclTopoSystem>();
            InitGetAlgoInfoMockComm(comm, *topo, "gfx950", 8, 2);
            comm->enableDirectReduceScatter = false;

            ncclTaskColl task = MakeReduceScatterTask();
            ExpectGetAlgoInfoSelection(comm, task, NCCL_ALGO_RING, NCCL_PROTO_LL);
            CleanupMockComm(comm);
        },
        {{"RCCL_OVERRIDE_PROTO", "LL"}, {"RCCL_OVERRIDE_ALGO", "Ring"}}
    );
}

// ===========================================================================
// CE AllReduce graph latch state machine (rccl_wrap.cc). Regression coverage
// for the capture-vs-eager ordering bug: the latch must stay set for the
// entire lifetime of a captured plan and must never clear mid-capture.
// ===========================================================================

TEST(RcclCeGraphLatch, SetsLatchOnFirstCapture)
{
    ncclComm comm{};
    comm.ceColl.graphModeSeen = false;
    comm.localPersistentRefs  = 0;

    rcclCeAllReduceGraphLatchTick(&comm, /*ceCapturing=*/true);

    EXPECT_TRUE(comm.ceColl.graphModeSeen);
    EXPECT_FALSE(rcclCeAllReduceAllowed(&comm));
}

TEST(RcclCeGraphLatch, StaysSetAcrossRepeatedCaptureTicks)
{
    ncclComm comm{};
    comm.ceColl.graphModeSeen = false;
    comm.localPersistentRefs  = 0;

    for(int i = 0; i < 3; ++i)
    {
        rcclCeAllReduceGraphLatchTick(&comm, /*ceCapturing=*/true);
        EXPECT_TRUE(comm.ceColl.graphModeSeen) << "iteration " << i;
    }
}

// Regression: an unrelated plan reclaiming to zero refs must not re-enable
// CE AR while still capturing -- this previously caused a cross-rank deadlock.
TEST(RcclCeGraphLatch, DoesNotClearMidCaptureEvenWithZeroRefs)
{
    ncclComm comm{};
    comm.ceColl.graphModeSeen = true;
    comm.localPersistentRefs  = 0;

    rcclCeAllReduceGraphLatchTick(&comm, /*ceCapturing=*/true);

    EXPECT_TRUE(comm.ceColl.graphModeSeen);
    EXPECT_FALSE(rcclCeAllReduceAllowed(&comm));
}

TEST(RcclCeGraphLatch, ClearsWhenCaptureEndsAndNoRefsRemain)
{
    ncclComm comm{};
    comm.ceColl.graphModeSeen = true;
    comm.localPersistentRefs  = 0;

    rcclCeAllReduceGraphLatchTick(&comm, /*ceCapturing=*/false);

    EXPECT_FALSE(comm.ceColl.graphModeSeen);
    EXPECT_TRUE(rcclCeAllReduceAllowed(&comm));
}

TEST(RcclCeGraphLatch, StaysSetWhileCapturedPlanStillReferencesComm)
{
    ncclComm comm{};
    comm.ceColl.graphModeSeen = true;
    comm.localPersistentRefs  = 1;

    rcclCeAllReduceGraphLatchTick(&comm, /*ceCapturing=*/false);

    EXPECT_TRUE(comm.ceColl.graphModeSeen)
        << "Latch must stay set until every captured plan is reclaimed";
    EXPECT_FALSE(rcclCeAllReduceAllowed(&comm));

    // Reclaim completes: the next tick clears the latch.
    comm.localPersistentRefs = 0;
    rcclCeAllReduceGraphLatchTick(&comm, /*ceCapturing=*/false);
    EXPECT_FALSE(comm.ceColl.graphModeSeen);
    EXPECT_TRUE(rcclCeAllReduceAllowed(&comm));
}

TEST(RcclCeGraphLatch, AllowedQueryHasNoSideEffects)
{
    ncclComm comm{};
    comm.ceColl.graphModeSeen = true;

    for(int i = 0; i < 5; ++i)
    {
        EXPECT_FALSE(rcclCeAllReduceAllowed(&comm));
    }
    EXPECT_TRUE(comm.ceColl.graphModeSeen) << "Query must be a pure read, not a state transition";
}

TEST(RcclCeGraphLatch, NeverLatchedAllowsCeAllReduceByDefault)
{
    ncclComm comm{};
    comm.ceColl.graphModeSeen = false;

    EXPECT_TRUE(rcclCeAllReduceAllowed(&comm));
}

#ifdef ENABLE_WARP_SPEED

// ---------------------------------------------------------------------------
// WarpSpeed enablement / channel-math helpers (rccl_wrap.cc)
//
// These exercise the pure decision/tuning helpers that gate WarpSpeed:
//   - rcclCanUseWarpSpeedAuto     (arch/node/env eligibility)
//   - rcclGetMaxWarpsPerBlock     (warps-per-block multiplier)
//   - rcclWarpSpeedComputeNChannels (connect.cc channel math)
//   - rcclWarpSpeedAdjustChannels   (enqueue.cc per-collective channel adj.)
// ---------------------------------------------------------------------------

// rcclCanUseWarpSpeedAuto: gfx950 single-node with auto mode on -> eligible.
TEST(Rcclwrap, CanUseWarpSpeedAuto_Gfx950SingleNode_True)
{
    // Auto mode defaults to 1; if the environment forces it off, the eligibility
    // result would legitimately be false, so skip to keep the test deterministic.
    if(rcclParamWarpSpeedAutoMode() == 0)
    {
        GTEST_SKIP() << "RCCL_WARP_SPEED_AUTO=0 in environment";
    }

    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx950", /*nRanks=*/8);
    comm->cuCount = 256; // SPX mode (256 CU on gfx950); auto mode requires cuCount > 128

    EXPECT_TRUE(rcclCanUseWarpSpeedAuto(comm, /*nNodes=*/1));

    comm->cuCount = 128; // Not SPX mode (128 CU on gfx950); auto mode requires cuCount > 128

    EXPECT_FALSE(rcclCanUseWarpSpeedAuto(comm, /*nNodes=*/1));

    CleanupMockComm(comm);
}

// rcclCanUseWarpSpeedAuto: non-gfx950 arch is never eligible.
TEST(Rcclwrap, CanUseWarpSpeedAuto_NonGfx950_False)
{
    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx942", /*nRanks=*/8);

    EXPECT_FALSE(rcclCanUseWarpSpeedAuto(comm, /*nNodes=*/1));

    CleanupMockComm(comm);
}

// rcclCanUseWarpSpeedAuto: multi-node is never eligible (auto mode is single-node).
TEST(Rcclwrap, CanUseWarpSpeedAuto_MultiNode_False)
{
    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx950", /*nRanks=*/16);

    EXPECT_FALSE(rcclCanUseWarpSpeedAuto(comm, /*nNodes=*/2));

    CleanupMockComm(comm);
}

// rcclCanUseWarpSpeedAuto: RCCL_WARP_SPEED_AUTO=0 disables eligibility even on
// gfx950 single-node. Isolated so the cached RCCL_PARAM value doesn't leak.
TEST(Rcclwrap, CanUseWarpSpeedAuto_AutoModeDisabled_False)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "CanUseWarpSpeedAuto_AutoModeDisabled_False",
        []()
        {
            ncclComm_t            comm = nullptr;
            auto                  topo = std::make_unique<ncclTopoSystem>();
            struct ncclTopoNode   gpu;
            CreateMockComm(comm, *topo, gpu, "gfx950", /*nRanks=*/8);
            comm->cuCount = 256; // ensure only RCCL_WARP_SPEED_AUTO=0 makes this ineligible

            EXPECT_FALSE(rcclCanUseWarpSpeedAuto(comm, /*nNodes=*/1));

            CleanupMockComm(comm);
        },
        {{"RCCL_WARP_SPEED_AUTO", "0"}}
    );
}

// rcclGetMaxWarpsPerBlock: single node -> RCCL_SINGLE_NODE_MAX_NTHREADS / WarpSize.
TEST(Rcclwrap, GetMaxWarpsPerBlock_SingleNode)
{
    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx950", /*nRanks=*/8);
    comm->nNodes   = 1;
    comm->WarpSize = 64;

    EXPECT_EQ(rcclGetMaxWarpsPerBlock(comm), RCCL_SINGLE_NODE_MAX_NTHREADS / 64);

    CleanupMockComm(comm);
}

// rcclGetMaxWarpsPerBlock: multi-node gfx950 -> RCCL_GFX950_MAX_NTHREADS / WarpSize.
TEST(Rcclwrap, GetMaxWarpsPerBlock_MultiNodeGfx950)
{
    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx950", /*nRanks=*/16);
    comm->nNodes   = 2;
    comm->WarpSize = 64;

    EXPECT_EQ(rcclGetMaxWarpsPerBlock(comm), RCCL_GFX950_MAX_NTHREADS / 64);

    CleanupMockComm(comm);
}

// rcclGetMaxWarpsPerBlock: multi-node non-gfx950 -> RCCL_DEFAULT_MAX_NTHREADS / WarpSize.
TEST(Rcclwrap, GetMaxWarpsPerBlock_MultiNodeOtherArch)
{
    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx942", /*nRanks=*/16);
    comm->nNodes   = 2;
    comm->WarpSize = 64;

    EXPECT_EQ(rcclGetMaxWarpsPerBlock(comm), RCCL_DEFAULT_MAX_NTHREADS / 64);

    CleanupMockComm(comm);
}

// Documents a latent issue: rcclGetMaxWarpsPerBlock branches on single-node vs
// multi-node (and arch), and the single-node comment claims "half the number of
// threads", but RCCL_SINGLE_NODE_MAX_NTHREADS == RCCL_GFX950_MAX_NTHREADS ==
// RCCL_DEFAULT_MAX_NTHREADS (all 256), so every branch returns the same value and
// the intended single-node halving is currently a no-op. If the constants are ever
// differentiated (as the comment intends), this test should be updated.
TEST(Rcclwrap, GetMaxWarpsPerBlock_BranchesCurrentlyEquivalent)
{
    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx950", /*nRanks=*/8);
    comm->WarpSize = 64;

    comm->nNodes    = 1;
    const int single = rcclGetMaxWarpsPerBlock(comm);
    comm->nNodes    = 2;
    const int multi  = rcclGetMaxWarpsPerBlock(comm);

    EXPECT_EQ(single, multi)
        << "Single-node is documented to use half the threads, but the MAX_NTHREADS "
           "constants are identical so the branch has no effect.";

    CleanupMockComm(comm);
}

// rcclWarpSpeedComputeNChannels: single-node, no user override, gfx950 8-rank ->
// nc*nChannels*mult, then halved for the gfx950 single-node 8-rank special case.
TEST(Rcclwrap, ComputeNChannels_SingleNode_Gfx950_8Ranks_Halved)
{
    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx950", /*nRanks=*/8);
    comm->nNodes    = 1;
    comm->nChannels = 2;

    // maxNchannels = nc(2) * nChannels(2) * mult(4) = 16; single-node keeps 16;
    // gfx950 && single-node && nRanks==8 -> nc /= 2 -> 8.
    const int nc = rcclWarpSpeedComputeNChannels(comm, /*nc=*/2, /*channelMultiplier=*/4,
                                                 /*maxChannels=*/64, /*adjustedMaxNchannels=*/64,
                                                 /*userUpdatedMaxChannels=*/false);
    EXPECT_EQ(nc, 8);

    CleanupMockComm(comm);
}

// rcclWarpSpeedComputeNChannels: single-node, non-8-rank -> no halving.
TEST(Rcclwrap, ComputeNChannels_SingleNode_4Ranks_NoHalving)
{
    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx950", /*nRanks=*/4);
    comm->nNodes    = 1;
    comm->nChannels = 2;

    const int nc = rcclWarpSpeedComputeNChannels(comm, /*nc=*/2, /*channelMultiplier=*/4,
                                                 /*maxChannels=*/64, /*adjustedMaxNchannels=*/64,
                                                 /*userUpdatedMaxChannels=*/false);
    EXPECT_EQ(nc, 16); // 2*2*4, no halving

    CleanupMockComm(comm);
}

// rcclWarpSpeedComputeNChannels: multi-node, no user override -> capped by maxChannels,
// no halving (halving is single-node only).
TEST(Rcclwrap, ComputeNChannels_MultiNode_CappedByMaxChannels)
{
    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx950", /*nRanks=*/8);
    comm->nNodes    = 2;
    comm->nChannels = 2;

    // maxNchannels = 2*2*4 = 16; multi-node -> min(16, maxChannels=8) = 8; no halving.
    const int nc = rcclWarpSpeedComputeNChannels(comm, /*nc=*/2, /*channelMultiplier=*/4,
                                                 /*maxChannels=*/8, /*adjustedMaxNchannels=*/64,
                                                 /*userUpdatedMaxChannels=*/false);
    EXPECT_EQ(nc, 8);

    CleanupMockComm(comm);
}

// rcclWarpSpeedComputeNChannels: user override path uses adjustedMaxNchannels*mult
// (never halved), below the MAXCHANNELS clamp.
TEST(Rcclwrap, ComputeNChannels_UserOverride_NoClamp)
{
    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx950", /*nRanks=*/8);
    comm->nNodes    = 1;
    comm->nChannels = 2;

    // userUpdated -> nc = min(adjusted(10) * mult(4), MAXCHANNELS) = 40; no halving.
    const int nc = rcclWarpSpeedComputeNChannels(comm, /*nc=*/2, /*channelMultiplier=*/4,
                                                 /*maxChannels=*/64, /*adjustedMaxNchannels=*/10,
                                                 /*userUpdatedMaxChannels=*/true);
    EXPECT_EQ(nc, 40);

    CleanupMockComm(comm);
}

// rcclWarpSpeedComputeNChannels: user override clamps to MAXCHANNELS (512 with WS).
TEST(Rcclwrap, ComputeNChannels_UserOverride_ClampedToMaxChannels)
{
    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx950", /*nRanks=*/8);
    comm->nNodes    = 1;
    comm->nChannels = 2;

    // adjusted(200) * mult(4) = 800, clamped to MAXCHANNELS.
    const int nc = rcclWarpSpeedComputeNChannels(comm, /*nc=*/2, /*channelMultiplier=*/4,
                                                 /*maxChannels=*/64, /*adjustedMaxNchannels=*/200,
                                                 /*userUpdatedMaxChannels=*/true);
    EXPECT_EQ(nc, MAXCHANNELS);

    CleanupMockComm(comm);
}

// rcclWarpSpeedAdjustChannels: disabled -> nc unchanged.
TEST(Rcclwrap, AdjustChannels_Disabled_NoChange)
{
    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx950", /*nRanks=*/8);
    comm->topo->warpSpeedEnabled     = false;
    comm->warpSpeedChannelMultiplier = 4;

    ncclTaskColl info = {};
    info.func         = ncclFuncAllReduce;

    EXPECT_EQ(rcclWarpSpeedAdjustChannels(comm, &info, /*nc=*/32), 32);

    CleanupMockComm(comm);
}

// rcclWarpSpeedAdjustChannels: enabled -> nc divided by the channel multiplier.
// Non-gfx950 arch avoids the single-node 8-rank doubling special case.
TEST(Rcclwrap, AdjustChannels_Enabled_DividesByMultiplier)
{
    ncclComm_t            comm = nullptr;
    auto                  topo = std::make_unique<ncclTopoSystem>();
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, *topo, gpu, "gfx942", /*nRanks=*/8);
    comm->topo->warpSpeedEnabled     = true;
    comm->warpSpeedChannelMultiplier = 4;

    ncclTaskColl info = {};
    info.func         = ncclFuncAllReduce;

    EXPECT_EQ(rcclWarpSpeedAdjustChannels(comm, &info, /*nc=*/32), 8); // 32/4

    CleanupMockComm(comm);
}

// rcclWarpSpeedAdjustChannels: gfx950 single-node 8-rank, non-(AR/AG/RS) collective
// -> divided then doubled (the "reduced CU usage" special case).
TEST(Rcclwrap, AdjustChannels_Gfx950SingleNode8Ranks_NonMainColl_Doubles)
{
    // Relies on default RCCL_MAX_NCHANNELS (-2, i.e. < 0). Isolate so the cached
    // ncclParamMaxNchannels() value is deterministic regardless of test ordering.
    RUN_ISOLATED_TEST_WITH_ENV(
        "AdjustChannels_Gfx950SingleNode8Ranks_NonMainColl_Doubles",
        []()
        {
            ncclComm_t            comm = nullptr;
            auto                  topo = std::make_unique<ncclTopoSystem>();
            struct ncclTopoNode   gpu;
            CreateMockComm(comm, *topo, gpu, "gfx950", /*nRanks=*/8);
            comm->nNodes                     = 1;
            comm->topo->warpSpeedEnabled     = true;
            comm->warpSpeedChannelMultiplier = 4;

            ncclTaskColl info = {};
            info.func         = ncclFuncBroadcast; // not AR/AG/RS

            EXPECT_EQ(rcclWarpSpeedAdjustChannels(comm, &info, /*nc=*/32), 16); // 32/4=8, *2=16

            CleanupMockComm(comm);
        },
        {}
    );
}

// rcclWarpSpeedAdjustChannels: gfx950 single-node 8-rank, main collective
// (AllReduce) -> divided only, not doubled.
TEST(Rcclwrap, AdjustChannels_Gfx950SingleNode8Ranks_MainColl_NoDouble)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "AdjustChannels_Gfx950SingleNode8Ranks_MainColl_NoDouble",
        []()
        {
            ncclComm_t            comm = nullptr;
            auto                  topo = std::make_unique<ncclTopoSystem>();
            struct ncclTopoNode   gpu;
            CreateMockComm(comm, *topo, gpu, "gfx950", /*nRanks=*/8);
            comm->nNodes                     = 1;
            comm->topo->warpSpeedEnabled     = true;
            comm->warpSpeedChannelMultiplier = 4;

            ncclTaskColl info = {};
            info.func         = ncclFuncAllReduce; // main collective, excluded from doubling

            EXPECT_EQ(rcclWarpSpeedAdjustChannels(comm, &info, /*nc=*/32), 8); // 32/4, no doubling

            CleanupMockComm(comm);
        },
        {}
    );
}

#endif // ENABLE_WARP_SPEED

// Builds a mock comm that satisfies every CE AllReduce eligibility rule except
// the one under test, so a single field or argument decides the outcome.
static void CreateCeAllReduceEligibleComm(
    ncclComm_t&            mockComm,
    struct ncclTopoSystem& mockTopo,
    struct ncclTopoNode&   mockGpuNode,
    int                    nRanks
)
{
    CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx950", nRanks);
    mockComm->nNodes             = 1;
    mockComm->symmetricSupport   = 1;
    mockComm->config.CTAPolicy   = NCCL_CTA_POLICY_ZERO;
}

// A count of float32 divisible by the 8 mock ranks; well within NCCL_CE_AR_MAX_MSG_BYTES.
static constexpr size_t kCeAllReduceCount = 4096;

// rcclUseCeAllReduce gates the Copy Engine AllReduce path. The CE kernels never
// read the bias buffer, so an eligible-looking ncclAllReduceWithBias call must
// still be refused; taking CE there silently drops the bias from the result.
TEST(Rcclwrap, RcclUseCeAllReduce_BiasBuffer)
{
    RUN_ISOLATED_TESTS(
        ProcessIsolatedTestRunner::TestConfig(
            "RcclUseCeAllReduce_BiasBufferRejected",
            []()
            {
                ncclComm_t            comm = nullptr;
                auto                  topo = std::make_unique<ncclTopoSystem>();
                struct ncclTopoNode   gpu;
                CreateCeAllReduceEligibleComm(comm, *topo, gpu, /*nRanks=*/8);

                int biasBuffer = 0;
                EXPECT_FALSE(rcclUseCeAllReduce(comm,
                                                kCeAllReduceCount,
                                                ncclFloat32,
                                                ncclSum,
                                                /*acc=*/&biasBuffer))
                    << "CE AllReduce must be refused when a bias buffer is present";

                CleanupMockComm(comm);
            })
            .withEnvironment({{"RCCL_CE_ALLREDUCE", "1"}}),

        // Control case: identical arguments with no bias must still select CE,
        // proving the rejection above is caused by the bias and not by an
        // unrelated ineligibility in the mock comm.
        ProcessIsolatedTestRunner::TestConfig(
            "RcclUseCeAllReduce_NoBiasAccepted",
            []()
            {
                ncclComm_t            comm = nullptr;
                auto                  topo = std::make_unique<ncclTopoSystem>();
                struct ncclTopoNode   gpu;
                CreateCeAllReduceEligibleComm(comm, *topo, gpu, /*nRanks=*/8);

                EXPECT_TRUE(rcclUseCeAllReduce(comm,
                                               kCeAllReduceCount,
                                               ncclFloat32,
                                               ncclSum,
                                               /*acc=*/nullptr))
                    << "CE AllReduce should be selected when no bias buffer is present";

                CleanupMockComm(comm);
            })
            .withEnvironment({{"RCCL_CE_ALLREDUCE", "1"}})
    );
}

// ---------------------------------------------------------------------------
// rcclAllReduceShouldTakeDdaPath: the AllReduce DDA-vs-CE dispatch decision.
//
// The helper returns true when ncclAllReduce_impl should take the DDA path, and
// false when it should yield (to CE AllReduce, the symmetric kernel, or the
// generic ring/tree fallback). DDA is taken when the buffers are not
// symmetric-kernel eligible, CE AllReduce will not service the call, and DDA is
// enabled for this arch and size.
//
// `ceAllReduceAllowed` is passed in directly rather than derived inside the
// test: the call site computes it once from rcclUseCeAllReduce() plus whatever
// additional gating CE AllReduce requires (graph latch, ncclGroupDepth,
// force/symReg, op/dtype/size/divisibility support, etc). Driving it directly
// keeps these cases independent of RCCL_CE_ALLREDUCE's default and of CE
// AllReduce's own eligibility rules -- each case just asserts what the DDA
// guard does for a given (symEligible, ceAllReduceAllowed) combination.
//
// These tests drive the real decision (no GPU): RCCL_DDA_ENABLE defaults to 1
// and ncclGroupDepth to 0, so rcclDdaEnabled() runs its real arch/threshold logic.
namespace
{
// Fill a zero-initialized comm with just the fields the decision reads. Filled by
// reference because ncclComm is not copyable.
void InitDdaDecisionComm(ncclComm& comm, const char* arch, int nRanks, int nNodes, bool symmetricSupport)
{
    comm.archName         = const_cast<char*>(arch);
    comm.nRanks           = nRanks;
    comm.nNodes           = nNodes;
    comm.symmetricSupport = symmetricSupport ? 1 : 0;
}

// count for a target byte size and datatype.
size_t CountForBytes(size_t bytes, ncclDataType_t dt)
{
    return bytes / ncclTypeSize(dt);
}
} // namespace

// gfx950 with symmetricSupport off: CE cannot run, so an 8 MiB call (at/above the
// 4 MiB CE minimum) must still take DDA rather than fall to the generic kernel.
TEST(RcclAllReduceDdaDecision, Gfx950_SymOff_LargeMsg_TakesDda)
{
    ncclComm comm{};
    InitDdaDecisionComm(comm, "gfx950", 8, 1, /*symmetricSupport=*/false);
    size_t   count = CountForBytes(8ull * 1024 * 1024, ncclFloat32);
    EXPECT_TRUE(rcclAllReduceShouldTakeDdaPath(&comm, count, ncclFloat32,
                                               /*symEligible=*/false, /*ceAllReduceAllowed=*/false));
}

// gfx950, small message with CE unavailable: squarely in DDA's range, takes DDA.
TEST(RcclAllReduceDdaDecision, Gfx950_SymOff_SmallMsg_TakesDda)
{
    ncclComm comm{};
    InitDdaDecisionComm(comm, "gfx950", 8, 1, /*symmetricSupport=*/false);
    size_t   count = CountForBytes(2ull * 1024 * 1024, ncclFloat32);
    EXPECT_TRUE(rcclAllReduceShouldTakeDdaPath(&comm, count, ncclFloat32,
                                               /*symEligible=*/false, /*ceAllReduceAllowed=*/false));
}

// gfx950 with symmetricSupport on and every CE prerequisite met: CE will service
// the call, so the DDA guard must yield (returns false).
TEST(RcclAllReduceDdaDecision, Gfx950_SymOn_CeEligible_YieldsToCe)
{
    ncclComm comm{};
    InitDdaDecisionComm(comm, "gfx950", 8, 1, /*symmetricSupport=*/true);
    size_t   count = CountForBytes(8ull * 1024 * 1024, ncclFloat32); // divisible by 8 ranks
    EXPECT_FALSE(rcclAllReduceShouldTakeDdaPath(&comm, count, ncclFloat32,
                                                /*symEligible=*/false, /*ceAllReduceAllowed=*/true));
}

// CE eligible by size/op/dtype but disabled by the graph latch (folded into the
// caller's ceAllReduceAllowed=false): CE will not run, so DDA must reclaim the call.
TEST(RcclAllReduceDdaDecision, Gfx950_SymOn_GraphLatched_TakesDda)
{
    ncclComm comm{};
    InitDdaDecisionComm(comm, "gfx950", 8, 1, /*symmetricSupport=*/true);
    size_t   count = CountForBytes(8ull * 1024 * 1024, ncclFloat32);
    EXPECT_TRUE(rcclAllReduceShouldTakeDdaPath(&comm, count, ncclFloat32,
                                               /*symEligible=*/false, /*ceAllReduceAllowed=*/false));
}

// CE declines on an unsupported op (folded into ceAllReduceAllowed=false) even
// with symmetricSupport on, so DDA reclaims the call.
TEST(RcclAllReduceDdaDecision, Gfx950_SymOn_UnsupportedOp_TakesDda)
{
    ncclComm comm{};
    InitDdaDecisionComm(comm, "gfx950", 8, 1, /*symmetricSupport=*/true);
    size_t   count = CountForBytes(8ull * 1024 * 1024, ncclFloat32);
    EXPECT_TRUE(rcclAllReduceShouldTakeDdaPath(&comm, count, ncclFloat32,
                                               /*symEligible=*/false, /*ceAllReduceAllowed=*/false));
}

// gfx942 with symmetricSupport off: a 6 MiB call is within the 8 MiB gfx942 DDA
// cap and, with CE unavailable, takes DDA.
TEST(RcclAllReduceDdaDecision, Gfx942_SymOff_MidMsg_TakesDda)
{
    ncclComm comm{};
    InitDdaDecisionComm(comm, "gfx942", 8, 1, /*symmetricSupport=*/false);
    size_t   count = CountForBytes(6ull * 1024 * 1024, ncclFloat32);
    EXPECT_TRUE(rcclAllReduceShouldTakeDdaPath(&comm, count, ncclFloat32,
                                               /*symEligible=*/false, /*ceAllReduceAllowed=*/false));
}

// gfx942 above its 8 MiB DDA cap: rcclDdaEnabled returns false, so no DDA.
TEST(RcclAllReduceDdaDecision, Gfx942_SymOff_AboveCap_NoDda)
{
    ncclComm comm{};
    InitDdaDecisionComm(comm, "gfx942", 8, 1, /*symmetricSupport=*/false);
    size_t   count = CountForBytes(9ull * 1024 * 1024, ncclFloat32);
    EXPECT_FALSE(rcclAllReduceShouldTakeDdaPath(&comm, count, ncclFloat32,
                                                /*symEligible=*/false, /*ceAllReduceAllowed=*/false));
}

// gfx942 with symmetricSupport on and every CE prerequisite met: CE claims the call
// and the DDA guard yields, even though 6 MiB is within the 8 MiB gfx942 DDA cap.
// Mirror of Gfx942_SymOff_MidMsg_TakesDda: ceAllReduceAllowed flips the decision.
TEST(RcclAllReduceDdaDecision, Gfx942_SymOn_CeEligible_YieldsToCe)
{
    ncclComm comm{};
    InitDdaDecisionComm(comm, "gfx942", 8, 1, /*symmetricSupport=*/true);
    size_t   count = CountForBytes(6ull * 1024 * 1024, ncclFloat32); // divisible by 8 ranks
    EXPECT_FALSE(rcclAllReduceShouldTakeDdaPath(&comm, count, ncclFloat32,
                                                /*symEligible=*/false, /*ceAllReduceAllowed=*/true));
}

// gfx942 with symmetricSupport on but CE declines on an unsupported op (folded
// into ceAllReduceAllowed=false): DDA reclaims the call since 6 MiB is within
// the 8 MiB gfx942 cap.
TEST(RcclAllReduceDdaDecision, Gfx942_SymOn_UnsupportedOp_TakesDda)
{
    ncclComm comm{};
    InitDdaDecisionComm(comm, "gfx942", 8, 1, /*symmetricSupport=*/true);
    size_t   count = CountForBytes(6ull * 1024 * 1024, ncclFloat32);
    EXPECT_TRUE(rcclAllReduceShouldTakeDdaPath(&comm, count, ncclFloat32,
                                               /*symEligible=*/false, /*ceAllReduceAllowed=*/false));
}

// gfx1250 forces the DDA fabric path regardless of CE eligibility: the
// ddaFabricArch1250 short-circuit means CE never claims the call on this arch.
// ceAllReduceAllowed=true (the call is otherwise fully CE-eligible: 64 MiB, sum,
// divisible), so the short-circuit is the only reason DDA is chosen here.
TEST(RcclAllReduceDdaDecision, Gfx1250_CeEligible_StillTakesDda)
{
    ncclComm comm{};
    InitDdaDecisionComm(comm, "gfx1250", 8, 1, /*symmetricSupport=*/true);
    size_t   count = CountForBytes(64ull * 1024 * 1024, ncclFloat32); // CE-eligible size, divisible by 8
    EXPECT_TRUE(rcclAllReduceShouldTakeDdaPath(&comm, count, ncclFloat32,
                                               /*symEligible=*/false, /*ceAllReduceAllowed=*/true));
}

// An arch DDA never runs on: rcclDdaEnabled returns false, so no DDA on any size.
TEST(RcclAllReduceDdaDecision, UnsupportedArch_NoDda)
{
    ncclComm comm{};
    InitDdaDecisionComm(comm, "gfx90a", 8, 1, /*symmetricSupport=*/false);
    size_t   count = CountForBytes(2ull * 1024 * 1024, ncclFloat32);
    EXPECT_FALSE(rcclAllReduceShouldTakeDdaPath(&comm, count, ncclFloat32,
                                                /*symEligible=*/false, /*ceAllReduceAllowed=*/false));
}

// gfx942/gfx950 DDA requires the full 8-GPU node; fewer ranks disables it.
TEST(RcclAllReduceDdaDecision, Gfx950_TooFewRanks_NoDda)
{
    ncclComm comm{};
    InitDdaDecisionComm(comm, "gfx950", 4, 1, /*symmetricSupport=*/false);
    size_t   count = CountForBytes(2ull * 1024 * 1024, ncclFloat32);
    EXPECT_FALSE(rcclAllReduceShouldTakeDdaPath(&comm, count, ncclFloat32,
                                                /*symEligible=*/false, /*ceAllReduceAllowed=*/false));
}

// Symmetric-kernel eligible buffers win outright: the DDA guard yields (returns false)
// regardless of arch/size.
TEST(RcclAllReduceDdaDecision, SymEligible_YieldsToSymmetricKernel)
{
    ncclComm comm{};
    InitDdaDecisionComm(comm, "gfx950", 8, 1, /*symmetricSupport=*/false);
    size_t   count = CountForBytes(2ull * 1024 * 1024, ncclFloat32);
    EXPECT_FALSE(rcclAllReduceShouldTakeDdaPath(&comm, count, ncclFloat32,
                                                /*symEligible=*/true, /*ceAllReduceAllowed=*/true));
}

// ---------------------------------------------------------------------------
// Tests for skipPresetTopoMatching: gfx1250 skips Rome model matching.
// Runs on real GPU, initializes a communicator, and checks internal state.
// ---------------------------------------------------------------------------

TEST(SkipPresetTopoMatching, Gfx1250_SkipsRomeModelMatching)
{
    RUN_ISOLATED_TEST("Gfx1250_SkipsRomeModelMatching", []()
    {
        int numDevices = 0;
        ASSERT_EQ(hipGetDeviceCount(&numDevices), hipSuccess);
        if (numDevices < 1) {
            GTEST_SKIP() << "No devices available.";
        }

        // Check if this is gfx1250
        hipDeviceProp_t prop{};
        ASSERT_EQ(hipGetDeviceProperties(&prop, 0), hipSuccess);
        bool isGfx1250 = (strncmp(prop.gcnArchName, "gfx1250", 7) == 0);
        if (!isGfx1250) {
            GTEST_SKIP() << "Test only applicable on gfx1250 hardware.";
        }

        ncclComm_t commHandle{};
        ncclUniqueId id{};
        ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        ASSERT_EQ(ncclCommInitRank(&commHandle, 1, id, 0), ncclSuccess);

        // Cast to internal struct to access topo
        ncclComm* comm = commHandle;

        // gfx1250 should skip preset topo matching
        EXPECT_TRUE(comm->topo->skipPresetTopoMatching);
        // Rome model index should be NONE (no preset matched)
        EXPECT_EQ(comm->topo->romeTopoModelIdx, RCCL_ROME_TOPO_PRESET_MODEL_IDX_NONE);

        ASSERT_EQ(ncclCommDestroy(commHandle), ncclSuccess);
    });
}

} // namespace RcclUnitTesting
