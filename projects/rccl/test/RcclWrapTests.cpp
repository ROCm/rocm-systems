/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include <cstdlib>
#include <cstring>

#include "comm.h"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "debug.h"
#include "graph/topo.h"

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

// Helper function to create and initialize mock communicator
static void CreateMockComm(
    ncclComm_t&            mockComm,
    struct ncclTopoSystem& mockTopo,
    struct ncclTopoNode&   mockGpuNode,
    const char*            arch,
    int                    nRanks
)
{
    // Allocate memory for the communicator
    mockComm = new ncclComm();
    memset(mockComm, 0, sizeof(ncclComm));

    // Initialize basic communicator fields
    mockComm->nRanks = nRanks;
    mockComm->nNodes = 1; // Default to single node for P2P tests
    mockComm->rank   = 0; // Default rank

    // Initialize topology
    memset(&mockTopo, 0, sizeof(mockTopo));
    mockComm->topo = &mockTopo;

    // Initialize GPU node
    mockTopo.nodes[GPU].count = 1;
    memset(&mockGpuNode, 0, sizeof(mockGpuNode));

    // Set GPU architecture
    strncpy(mockGpuNode.gpu.gcn, arch, sizeof(mockGpuNode.gpu.gcn) - 1);
    mockGpuNode.gpu.gcn[sizeof(mockGpuNode.gpu.gcn) - 1] = '\0';

    // Copy the node into the topology array
    mockTopo.nodes[GPU].nodes[0] = mockGpuNode;

    // Initialize other required fields for tests
    memset(mockComm->minMaxLLRange, 0, sizeof(mockComm->minMaxLLRange));
}

// Helper function to cleanup mock communicator
static void CleanupMockComm(ncclComm_t& mockComm)
{
    if(mockComm)
    {
        delete mockComm;
        mockComm = nullptr;
    }
}

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

    ncclComm_t comm = new ncclComm();
    *comm           = {};
    // Manually populate minimal fields for comm
    comm->nRanks                    = 1;
    comm->nNodes                    = 2; // triggers inter-node logic
    comm->rank                      = 0;
    comm->topo                      = new ncclTopoSystem();
    *comm->topo                     = {};
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

    delete comm->topo;
    delete comm;
}

TEST(Rcclwrap, RcclUpdateCollectiveProtocol_WarnsOnGfx942Arch)
{
    setenv("NCCL_PROTO", "", 1);
    unsetenv("NCCL_PROTO");

    ncclComm_t comm = new ncclComm();
    *comm           = {};
    // Manually populate minimal fields for comm
    comm->nRanks                    = 1;
    comm->nNodes                    = 2; // triggers inter-node logic
    comm->rank                      = 0;
    comm->topo                      = new ncclTopoSystem();
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

    delete comm->topo;
    delete comm;
}

TEST(Rcclwrap, RcclUpdateCollectiveProtocol_HonorsUserProtocolEnv)
{                                 // Why does this pass
                                  // if it does not
                                  // enter the else if
                                  // block
    setenv("NCCL_PROTO", "1", 1); // Simulate manual override

    ncclComm_t comm = new ncclComm();
    *comm           = {};
    // Manually populate minimal fields for comm
    comm->nRanks = 1;
    comm->nNodes = 2; // triggers inter-node logic
    comm->rank   = 0;
    comm->topo   = new ncclTopoSystem(); //(struct ncclTopoSystem*)calloc(1,
                                         // sizeof(struct ncclTopoSystem));
    *comm->topo                     = {};
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

    delete comm->topo;
    delete comm;
}

TEST(Rcclwrap, RcclUpdateCollectiveProtocol_SimpleFallbackWhenNoRanges)
{
    setenv("NCCL_PROTO", "", 1); // Trigger auto selection mode
    unsetenv("NCCL_PROTO");

    ncclComm_t comm = new ncclComm();
    *comm           = {};
    // Manually populate minimal fields for comm
    comm->nRanks = 1;
    comm->nNodes = 2; // triggers inter-node logic
    comm->rank   = 0;
    comm->topo   = new ncclTopoSystem(); //(struct ncclTopoSystem*)calloc(1,
                                         // sizeof(struct ncclTopoSystem));
    *comm->topo                     = {};
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

    delete comm->topo;
    delete comm;
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
                INFO(
                    NCCL_LOG_INFO,
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
                INFO(
                    NCCL_LOG_INFO,
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
                INFO(
                    NCCL_LOG_INFO,
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
    struct ncclTopoSystem topo;
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, topo, gpu, "gfx950", 8);
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
    struct ncclTopoSystem topo;
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, topo, gpu, "gfx950", 8);
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
    struct ncclTopoSystem topo;
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, topo, gpu, "gfx942", 8);
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
    struct ncclTopoSystem topo;
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, topo, gpu, "gfx942", 8);
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
    struct ncclTopoSystem topo;
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, topo, gpu, "gfx942", 8);
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
    struct ncclTopoSystem topo;
    struct ncclTopoNode   gpu;
    CreateMockComm(comm, topo, gpu, "gfx942", 8);
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
    INFO(
        NCCL_LOG_INFO,
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
                    struct ncclTopoSystem mockTopo;
                    struct ncclTopoNode   mockGpuNode;
                    CreateMockComm(mockComm, mockTopo, mockGpuNode, tc.arch.c_str(), tc.ranks);

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

    INFO(
        NCCL_LOG_INFO,
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

                    INFO(
                        NCCL_LOG_INFO,
                        "Testing rcclSetPxn for %s with %d ranks",
                        tc.arch.c_str(),
                        tc.ranks
                    );

                    ncclComm_t            mockComm = nullptr;
                    struct ncclTopoSystem mockTopo;
                    struct ncclTopoNode   mockGpuNode;
                    CreateMockComm(mockComm, mockTopo, mockGpuNode, tc.arch.c_str(), tc.ranks);

                    int pxnDisable = RCCL_VALUE_UNSET;
                    rcclSetPxn(mockComm, pxnDisable);

                    EXPECT_EQ(pxnDisable, tc.expectedPxnDisable)
                        << "Failed for " << tc.arch << " with " << tc.ranks << " ranks";

                    INFO(
                        NCCL_LOG_INFO,
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

} // namespace RcclUnitTesting
