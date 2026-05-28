/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h" // Ensure full definition of struct ncclComm
#include "debug.h"
#include "graph/topo.h"
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <rccl/rccl.h>

namespace RcclUnitTesting {

// Static flag to ensure only one rcclSetP2pNetChunkSize test runs per execution
static bool s_p2pNetChunkSizeTestExecuted = false;

// Helper function to check if P2P test should be skipped due to execution order
static bool ShouldSkipP2pTestDueToExecutionOrder(const std::string &testName) {
  if (s_p2pNetChunkSizeTestExecuted) {
    INFO(NCCL_LOG_INFO,
         "\n=== IMPORTANT NOTE ===\n"
         "Test '%s' is being skipped because another rcclSetP2pNetChunkSize "
         "test\n"
         "has already executed in this run. The rcclSetP2pNetChunkSize "
         "function uses a static\n"
         "variable that gets initialized on first call, which affects "
         "subsequent tests.\n"
         "\nTo run this test properly, execute it individually using:\n"
         "  --gtest_filter=Rcclwrap.%s\n"
         "\nOr run each rcclSetP2pNetChunkSize test in separate executions to "
         "ensure\n"
         "proper static variable initialization.\n"
         "========================\n",
         testName.c_str(), testName.c_str());
    return true;
  }

  // Mark that a P2P test is now executing
  s_p2pNetChunkSizeTestExecuted = true;
  return false;
}

// Helper function to determine if P2P test should be skipped due to static
// variable state
static bool ShouldSkipP2pTest(const char *requiredEnvValue = nullptr) {
  const char *envValue = getenv("NCCL_P2P_NET_CHUNKSIZE");

  // If a specific environment value is required, check for it
  if (requiredEnvValue != nullptr) {
    if (!envValue || strcmp(envValue, requiredEnvValue) != 0) {
      return true; // Skip if env var is not set to required value
    }
    return false; // Don't skip if env var matches required value
  }

  // For architecture logic tests, skip only if environment variable is set
  // (which would override the static variable behavior)
  // Note: We cannot directly check if static variable is RCCL_VALUE_UNSET
  // from test code, so we rely on clean environment for proper testing
  if (envValue != nullptr) {
    return true; // Skip if env var is set (prevents testing architecture logic)
  }

  // Environment is clean - proceed with test
  // Warning: Static variable might still be initialized from previous tests
  // For guaranteed clean state, run tests individually or restart binary
  return false; // Don't skip
}

// Static flag to ensure only one rcclSetPxn test runs per execution
static bool s_pxnTestExecuted = false;

// Helper function to check if PXN test should be skipped due to execution order
static bool ShouldSkipPxnTestDueToExecutionOrder(const std::string &testName) {
  if (s_pxnTestExecuted) {
    INFO(NCCL_LOG_INFO,
         "\n=== IMPORTANT NOTE ===\n"
         "Test '%s' is being skipped because another rcclSetPxn test\n"
         "has already executed in this run. The rcclSetPxn function uses a "
         "static\n"
         "variable that gets initialized on first call, which affects "
         "subsequent tests.\n"
         "\nTo run this test properly, execute it individually using:\n"
         "  --gtest_filter=Rcclwrap.%s\n"
         "\nOr run each rcclSetPxn test in separate executions to ensure\n"
         "proper static variable initialization.\n"
         "========================\n",
         testName.c_str(), testName.c_str());
    return true;
  }

  // Mark that a PXN test is now executing
  s_pxnTestExecuted = true;
  return false;
}

// Helper function to determine if PXN test should be skipped due to static
// variable state
static bool ShouldSkipPxnTest(const char *requiredEnvValue = nullptr) {
  const char *envValue = getenv("NCCL_PXN_DISABLE");

  // If a specific environment value is required, check for it
  if (requiredEnvValue != nullptr) {
    if (!envValue || strcmp(envValue, requiredEnvValue) != 0) {
      return true; // Skip if env var is not set to required value
    }
    return false; // Don't skip if env var matches required value
  }

  // For architecture logic tests, skip only if environment variable is set
  // (which would override the static variable behavior)
  if (envValue != nullptr) {
    return true; // Skip if env var is set (prevents testing architecture logic)
  }

  // Environment is clean - proceed with test
  return false; // Don't skip
}

// Helper function to test the static expose check
ncclResult_t testStaticExposeCheck() {
  RCCL_STATIC_EXPOSE_CHECK();
  return ncclSuccess;
}

// Helper function to create and initialize mock communicator
static void CreateMockComm(ncclComm_t &mockComm,
                           struct ncclTopoSystem &mockTopo,
                           struct ncclTopoNode &mockGpuNode, const char *arch,
                           int nRanks) {
  // Allocate memory for the communicator
  mockComm = new ncclComm();
  memset(mockComm, 0, sizeof(ncclComm));

  // Initialize basic communicator fields
  mockComm->nRanks = nRanks;
  mockComm->nNodes = 1; // Default to single node for P2P tests
  mockComm->rank = 0;   // Default rank

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
static void CleanupMockComm(ncclComm_t &mockComm) {
  if (mockComm) {
    delete mockComm;
    mockComm = nullptr;
  }
}

// Helper function to determine if rcclSetPipelining test should be skipped
static bool ShouldSkipRcclSetPipeliningTests() {
  const char *disable = getenv("RCCL_DISABLE_REDUCE_COPY_PIPELINING");
  // Skip the test if RCCL_DISABLE_REDUCE_COPY_PIPELINING is set
  if (disable && strcmp(disable, "0") != 0) {
    return true;
  }
  return false;
}

// Helper function to validate protocol string against known valid protocols
static bool isProtoStrValid(const char *envStr) {
  if (!envStr)
    return false;
  for (int i = 0; i < NCCL_NUM_PROTOCOLS; ++i) {
    if (strcasecmp(envStr, ncclProtoStr[i]) == 0) {
      return true; // Match found
    }
  }
  return false; // No match found
}

// Helper function to validate algorithm string against known valid algorithms
static bool isAlgoStrValid(const char *envStr) {
  if (!envStr)
    return false;
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; ++i) {
    if (strcasecmp(envStr, ncclAlgoStr[i]) == 0) {
      return true; // Match found
    }
  }
  return false; // No match found
}

TEST(Rcclwrap, RcclFuncMaxSendRecvCount) {
  ncclResult_t staticCheckResult = testStaticExposeCheck();
#ifdef RCCL_EXPOSE_STATIC
  EXPECT_EQ(staticCheckResult, ncclSuccess);
#else
  EXPECT_EQ(staticCheckResult, ncclInvalidUsage);
#endif

  size_t maxCount = 0;
  ncclResult_t result =
      rcclFuncMaxSendRecvCount(ncclFuncAllReduce, 4, 1024, maxCount);
  EXPECT_EQ(maxCount, 1024);
  EXPECT_EQ(result, ncclSuccess);
}

TEST(Rcclwrap, RcclUpdateCollectiveProtocol_UsesLL128WhenInRange) {
  setenv("NCCL_PROTO", "", 1); // Trigger auto selection mode
  unsetenv("NCCL_PROTO");

  ncclComm_t comm = new ncclComm();
  *comm = {};
  // Manually populate minimal fields for comm
  comm->nRanks = 1;
  comm->nNodes = 2; // triggers inter-node logic
  comm->rank = 0;
  comm->topo = new ncclTopoSystem();
  *comm->topo = {};
  comm->topo->ll128Enabled = true;
  comm->topo->nodes[GPU].nodes[0] = {};
  comm->topo->nodes[GPU].count = 1;
  strncpy(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942",
          sizeof(comm->topo->nodes[GPU].nodes[0].gpu.gcn));

  int idx = rcclGetTunableIndex(ncclFuncAllReduce);
  comm->minMaxLLRange[idx][NCCL_PROTO_LL][RCCL_PROTOCOL_MIN_IDX] = 512;
  comm->minMaxLLRange[idx][NCCL_PROTO_LL][RCCL_PROTOCOL_MAX_IDX] = 1024;
  comm->minMaxLLRange[idx][NCCL_PROTO_LL128][RCCL_PROTOCOL_MIN_IDX] = 256;
  comm->minMaxLLRange[idx][NCCL_PROTO_LL128][RCCL_PROTOCOL_MAX_IDX] = 2048;
  comm->minMaxLLRange[idx][NCCL_PROTO_LL128][RCCL_PROTOCOL_FACTOR_IDX] = 1;

  ncclTaskColl info = {};
  // Manually populate minimal fields for info
  info.func = ncclFuncAllReduce;
  info.protocol = NCCL_PROTO_UNDEF;

  size_t nBytes = 1024;

  rcclUpdateCollectiveProtocol(comm, nBytes, &info);
  EXPECT_TRUE(info.protocol == NCCL_PROTO_LL128 ||
              info.protocol == NCCL_PROTO_LL);

  delete comm->topo;
  delete comm;
}

TEST(Rcclwrap, RcclUpdateCollectiveProtocol_WarnsOnGfx942Arch) {
  setenv("NCCL_PROTO", "", 1);
  unsetenv("NCCL_PROTO");

  ncclComm_t comm = new ncclComm();
  *comm = {};
  // Manually populate minimal fields for comm
  comm->nRanks = 1;
  comm->nNodes = 2; // triggers inter-node logic
  comm->rank = 0;
  comm->topo = new ncclTopoSystem();
  comm->topo->ll128Enabled = true;
  comm->topo->nodes[GPU].nodes[0] = {};
  strncpy(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942",
          sizeof(comm->topo->nodes[GPU].nodes[0].gpu.gcn));

  int idx = rcclGetTunableIndex(ncclFuncAllReduce);
  comm->minMaxLLRange[idx][NCCL_PROTO_LL][RCCL_PROTOCOL_MIN_IDX] =
      RCCL_LL_LIMITS_UNDEFINED;
  comm->minMaxLLRange[idx][NCCL_PROTO_LL][RCCL_PROTOCOL_MAX_IDX] =
      RCCL_LL_LIMITS_UNDEFINED;
  comm->minMaxLLRange[idx][NCCL_PROTO_LL128][RCCL_PROTOCOL_MIN_IDX] =
      RCCL_LL_LIMITS_UNDEFINED;
  comm->minMaxLLRange[idx][NCCL_PROTO_LL128][RCCL_PROTOCOL_MAX_IDX] =
      RCCL_LL_LIMITS_UNDEFINED;
  comm->minMaxLLRange[idx][NCCL_PROTO_LL128][RCCL_PROTOCOL_FACTOR_IDX] =
      RCCL_LL_LIMITS_UNDEFINED;

  ncclTaskColl info = {};
  // Manually populate minimal fields for info
  info.func = ncclFuncAllReduce;
  info.protocol = NCCL_PROTO_UNDEF;
  size_t nBytes = 1024; // 1024 per rank for 4 ranks

  rcclUpdateCollectiveProtocol(comm, nBytes, &info);
  EXPECT_EQ(info.protocol, NCCL_PROTO_UNDEF);

  delete comm->topo;
  delete comm;
}

TEST(Rcclwrap,
     RcclUpdateCollectiveProtocol_HonorsUserProtocolEnv) { // Why does this pass
                                                           // if it does not
                                                           // enter the else if
                                                           // block
  setenv("NCCL_PROTO", "1", 1); // Simulate manual override

  ncclComm_t comm = new ncclComm();
  *comm = {};
  // Manually populate minimal fields for comm
  comm->nRanks = 1;
  comm->nNodes = 2; // triggers inter-node logic
  comm->rank = 0;
  comm->topo = new ncclTopoSystem(); //(struct ncclTopoSystem*)calloc(1,
                                     // sizeof(struct ncclTopoSystem));
  *comm->topo = {};
  comm->topo->ll128Enabled = true;
  comm->topo->nodes[GPU].nodes[0] = {};
  strncpy(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942",
          sizeof(comm->topo->nodes[GPU].nodes[0].gpu.gcn));

  ncclTaskColl info = {};
  // Manually populate minimal fields for info
  info.func = ncclFuncAllReduce;
  info.protocol = NCCL_PROTO_UNDEF;
  size_t nBytes = 1024; // 1024 per rank for 4 ranks

  rcclUpdateCollectiveProtocol(comm, nBytes, &info);
  EXPECT_EQ(info.protocol, NCCL_PROTO_UNDEF);

  delete comm->topo;
  delete comm;
}

TEST(Rcclwrap, RcclUpdateCollectiveProtocol_SimpleFallbackWhenNoRanges) {
  setenv("NCCL_PROTO", "", 1); // Trigger auto selection mode
  unsetenv("NCCL_PROTO");

  ncclComm_t comm = new ncclComm();
  *comm = {};
  // Manually populate minimal fields for comm
  comm->nRanks = 1;
  comm->nNodes = 2; // triggers inter-node logic
  comm->rank = 0;
  comm->topo = new ncclTopoSystem(); //(struct ncclTopoSystem*)calloc(1,
                                     // sizeof(struct ncclTopoSystem));
  *comm->topo = {};
  comm->topo->ll128Enabled = true;
  comm->topo->nodes[GPU].nodes[0] = {};
  comm->topo->nodes[GPU].count = 1;
  strncpy(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942",
          sizeof(comm->topo->nodes[GPU].nodes[0].gpu.gcn));

  int idx = rcclGetTunableIndex(ncclFuncAllReduce);
  comm->minMaxLLRange[idx][NCCL_PROTO_LL][RCCL_PROTOCOL_MIN_IDX] = 512;
  comm->minMaxLLRange[idx][NCCL_PROTO_LL][RCCL_PROTOCOL_MAX_IDX] = 1024;

  // Manually populate minimal fields for info
  ncclTaskColl info = {};
  info.func = ncclFuncAllReduce;
  info.protocol = NCCL_PROTO_UNDEF;
  size_t nBytes = 2048; // 1024 per rank for 4 ranks

  rcclUpdateCollectiveProtocol(comm, nBytes, &info);
  EXPECT_EQ(info.protocol, NCCL_PROTO_SIMPLE);

  delete comm->topo;
  delete comm;
}

TEST(Rcclwrap, validHsaScratchEnvSettingTest) {
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

TEST(Rcclwrap, RcclUpdateThreadThreshold_UserEnvSet) {
  const char *value = getenv("NCCL_THREAD_THRESHOLDS");

  if (!value) {
    INFO(NCCL_LOG_INFO, "[Rcclwrap] Test skipped. Set environment variable "
                        "NCCL_THREAD_THRESHOLD");
    GTEST_SKIP() << "[Rcclwrap] Test skipped. Set environment variable "
                    "NCCL_THREAD_THRESHOLD\n";
  } else {
    ncclComm comm = {.nRanks = 8, .nNodes = 4};
    ncclTaskColl info = {.func = ncclFuncReduceScatter, .protocol = 0};
    memset(comm.minMaxLLRange, 0, sizeof(comm.minMaxLLRange));

    int threadThreshold = 5; // Any number should do, we should make sure this
                             // number does not change
    rcclUpdateThreadThreshold(&comm, 0, &info, threadThreshold);

    EXPECT_EQ(threadThreshold, 5);
  }
}

TEST(Rcclwrap, RcclUpdateThreadThreshold_MinNChannelsSet) {
  const char *value = getenv("NCCL_MIN_NCHANNELS");
  if (!value) {
    INFO(
        NCCL_LOG_INFO,
        "[Rcclwrap] Test skipped. Set environment variable NCCL_MIN_NCHANNELS");
    GTEST_SKIP() << "[Rcclwrap] Test skipped. Set environment variable "
                    "NCCL_MIN_NCHANNELS\n";
  } else {
    ncclComm comm{};
    ncclTaskColl info{};
    int threadThreshold = 5;

    comm.nRanks = 4;
    comm.nNodes = 4;
    info.func = ncclFuncAllGather;
    info.protocol = 0;
    memset(comm.minMaxLLRange, 0, sizeof(comm.minMaxLLRange));

    rcclUpdateThreadThreshold(&comm, 0, &info, threadThreshold);

    EXPECT_EQ(threadThreshold, 5);
  }
}

TEST(Rcclwrap, RcclUpdateThreadThreshold_MNChannelsSet) {
  const char *value = getenv("NCCL_MAX_NCHANNELS");
  if (!value) {
    INFO(
        NCCL_LOG_INFO,
        "[Rcclwrap] Test skipped. Set environment variable NCCL_MAX_NCHANNELS");
    GTEST_SKIP() << "[Rcclwrap] Test skipped. Set environment variable "
                    "NCCL_MAX_NCHANNELS\n";
  } else {
    ncclComm comm{};
    ncclTaskColl info{};
    int threadThreshold = 5;

    comm.nRanks = 4;
    comm.nNodes = 4;
    info.func = ncclFuncAllGather;
    info.protocol = 0;
    memset(comm.minMaxLLRange, 0, sizeof(comm.minMaxLLRange));

    rcclUpdateThreadThreshold(&comm, 0, &info, threadThreshold);

    EXPECT_EQ(threadThreshold, 5);
  }
}

TEST(Rcclwrap, RcclUpdateThreadThreshold_NoEnv_nNodesLessThan2) {
  ncclComm comm{};
  ncclTaskColl info{};
  int threadThreshold = 5;

  comm.nRanks = 4;
  comm.nNodes = 1; // less than 2
  info.func = ncclFuncReduceScatter;
  info.protocol = 0;
  memset(comm.minMaxLLRange, 0, sizeof(comm.minMaxLLRange));

  rcclUpdateThreadThreshold(&comm, 0, &info, threadThreshold);

  EXPECT_EQ(threadThreshold, 5); // no change
}

TEST(Rcclwrap, RcclUpdateThreadThreshold_NoEnv_FuncUnsupported) {
  ncclComm comm{};
  ncclTaskColl info{};
  int threadThreshold = 5;

  comm.nRanks = 4;
  comm.nNodes = 2;
  info.func = ncclFuncAllReduce; // unsupported func
  info.protocol = 0;
  memset(comm.minMaxLLRange, 0, sizeof(comm.minMaxLLRange));

  rcclUpdateThreadThreshold(&comm, 0, &info, threadThreshold);

  EXPECT_EQ(threadThreshold, 5);
}

TEST(Rcclwrap, RcclUpdateThreadThreshold_NoEnv_UpdateOccurs) {
  ncclComm comm{};
  ncclTaskColl info{};
  int threadThreshold = 5;

  comm.nRanks = 4;
  comm.nNodes = 2;
  info.func = ncclFuncReduceScatter;
  info.protocol = 0;
  memset(comm.minMaxLLRange, 0, sizeof(comm.minMaxLLRange));

  int idx = rcclGetTunableIndex(info.func);
  comm.minMaxLLRange[idx][info.protocol][RCCL_PROTOCOL_THREAD_THRESHOLD_IDX] =
      10;

  rcclUpdateThreadThreshold(&comm, 0, &info, threadThreshold);

  EXPECT_EQ(threadThreshold, 40); // 10 * 4
}

TEST(Rcclwrap, RcclUpdateThreadThreshold_NoEnv_ThresholdUndefined) {
  ncclComm comm{};
  ncclTaskColl info{};
  int threadThreshold = 5;

  comm.nRanks = 4;
  comm.nNodes = 3;
  info.func = ncclFuncAllGather;
  info.protocol = 0;
  memset(comm.minMaxLLRange, 0, sizeof(comm.minMaxLLRange));

  int idx = rcclGetTunableIndex(info.func);
  comm.minMaxLLRange[idx][info.protocol][RCCL_PROTOCOL_THREAD_THRESHOLD_IDX] =
      RCCL_LL_LIMITS_UNDEFINED;

  rcclUpdateThreadThreshold(&comm, 0, &info, threadThreshold);

  EXPECT_EQ(threadThreshold, 5);
}

TEST(Rcclwrap, GFX942_SmallRanks) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("GFX942_SmallRanks")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipP2pTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize for GFX942 with small ranks");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx942", 32);

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: 1 << 17 = 131072 for ranks < 64
  EXPECT_EQ(chunkSize, 1 << 17)
      << "GFX942 with ranks < 64 should set chunk size to 131072";

  INFO(NCCL_LOG_INFO, "GFX942 small ranks test completed - chunk size: %d",
       chunkSize);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, GFX942_LargeRanks) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("GFX942_LargeRanks")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipP2pTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize for GFX942 with large ranks");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx942", 128);

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: 1 << 19 = 524288 for ranks >= 64
  EXPECT_EQ(chunkSize, 1 << 19)
      << "GFX942 with ranks >= 64 should set chunk size to 524288";

  INFO(NCCL_LOG_INFO, "GFX942 large ranks test completed - chunk size: %d",
       chunkSize);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, GFX942_BoundaryRank64) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("GFX942_BoundaryRank64")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipP2pTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize for GFX942 with boundary rank 64");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx942", 64);

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: 1 << 19 = 524288 for ranks >= 64
  EXPECT_EQ(chunkSize, 1 << 19)
      << "GFX942 with ranks = 64 should set chunk size to 524288";

  INFO(NCCL_LOG_INFO, "GFX942 boundary rank 64 test completed - chunk size: %d",
       chunkSize);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, GFX942_BoundaryRank63) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("GFX942_BoundaryRank63")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipP2pTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize for GFX942 with boundary rank 63");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx942", 63);

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: 1 << 17 = 131072 for ranks < 64
  EXPECT_EQ(chunkSize, 1 << 17)
      << "GFX942 with ranks = 63 should set chunk size to 131072";

  INFO(NCCL_LOG_INFO, "GFX942 boundary rank 63 test completed - chunk size: %d",
       chunkSize);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, GFX950_SmallRanks) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("GFX950_SmallRanks")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipP2pTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize for GFX950 with small ranks");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx950", 8);

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: 1 << 17 = 131072 for ranks < 16
  EXPECT_EQ(chunkSize, 1 << 17)
      << "GFX950 with ranks < 16 should set chunk size to 131072";

  INFO(NCCL_LOG_INFO, "GFX950 small ranks test completed - chunk size: %d",
       chunkSize);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, GFX950_MediumRanks) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("GFX950_MediumRanks")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipP2pTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize for GFX950 with medium ranks");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx950", 24);

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: 1 << 18 = 262144 for 16 <= ranks < 32
  EXPECT_EQ(chunkSize, 1 << 18)
      << "GFX950 with 16 <= ranks < 32 should set chunk size to 262144";

  INFO(NCCL_LOG_INFO, "GFX950 medium ranks test completed - chunk size: %d",
       chunkSize);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, GFX950_LargeRanks) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("GFX950_LargeRanks")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipP2pTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize for GFX950 with large ranks");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx950", 64);

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: 1 << 19 = 524288 for ranks >= 32
  EXPECT_EQ(chunkSize, 1 << 19)
      << "GFX950 with ranks >= 32 should set chunk size to 524288";

  INFO(NCCL_LOG_INFO, "GFX950 large ranks test completed - chunk size: %d",
       chunkSize);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, GFX950_BoundaryRank16) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("GFX950_BoundaryRank16")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipP2pTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize for GFX950 with boundary rank 16");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx950", 16);

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: 1 << 18 = 262144 for ranks >= 16
  EXPECT_EQ(chunkSize, 1 << 18)
      << "GFX950 with ranks = 16 should set chunk size to 262144";

  INFO(NCCL_LOG_INFO, "GFX950 boundary rank 16 test completed - chunk size: %d",
       chunkSize);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, GFX950_BoundaryRank15) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("GFX950_BoundaryRank15")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipP2pTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize for GFX950 with boundary rank 15");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx950", 15);

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: 1 << 17 = 131072 for ranks < 16
  EXPECT_EQ(chunkSize, 1 << 17)
      << "GFX950 with ranks = 15 should set chunk size to 131072";

  INFO(NCCL_LOG_INFO, "GFX950 boundary rank 15 test completed - chunk size: %d",
       chunkSize);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, GFX950_BoundaryRank32) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("GFX950_BoundaryRank32")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipP2pTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize for GFX950 with boundary rank 32");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx950", 32);

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: 1 << 19 = 524288 for ranks >= 32
  EXPECT_EQ(chunkSize, 1 << 19)
      << "GFX950 with ranks = 32 should set chunk size to 524288";

  INFO(NCCL_LOG_INFO, "GFX950 boundary rank 32 test completed - chunk size: %d",
       chunkSize);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, GFX950_BoundaryRank31) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("GFX950_BoundaryRank31")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipP2pTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize for GFX950 with boundary rank 31");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx950", 31);

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: 1 << 18 = 262144 for 16 <= ranks < 32
  EXPECT_EQ(chunkSize, 1 << 18)
      << "GFX950 with ranks = 31 should set chunk size to 262144";

  INFO(NCCL_LOG_INFO, "GFX950 boundary rank 31 test completed - chunk size: %d",
       chunkSize);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, UnsupportedArch_GFX908) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("UnsupportedArch_GFX908")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipP2pTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize for unsupported architecture GFX908");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx908", 32);

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: RCCL_VALUE_INVALID for unsupported architectures
  EXPECT_EQ(chunkSize, RCCL_VALUE_INVALID)
      << "Unsupported architecture GFX908 should set chunk size to "
         "RCCL_VALUE_INVALID";

  INFO(NCCL_LOG_INFO,
       "Unsupported architecture GFX908 test completed - chunk size: %d",
       chunkSize);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, UnsupportedArch_GFX90A) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("UnsupportedArch_GFX90A")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipP2pTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize for unsupported architecture GFX90A");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx90a", 32);

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: RCCL_VALUE_INVALID for unsupported architectures
  EXPECT_EQ(chunkSize, RCCL_VALUE_INVALID)
      << "Unsupported architecture GFX90A should set chunk size to "
         "RCCL_VALUE_INVALID";

  INFO(NCCL_LOG_INFO,
       "Unsupported architecture GFX90A test completed - chunk size: %d",
       chunkSize);

  CleanupMockComm(mockComm);
}

// This test specifically tests the environment variable behavior
TEST(Rcclwrap, WithEnvironmentVariable) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("WithEnvironmentVariable")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // This test requires environment variable to be set to a specific value
  if (ShouldSkipP2pTest("123456")) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is not "
           "set to '123456'. "
        << "Please set: export NCCL_P2P_NET_CHUNKSIZE=123456 to run this test. "
        << "This test verifies that user override via environment variable "
           "works correctly.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize with environment variable set");

  // Environment variable is confirmed to be set to "123456"
  const char *envVar = getenv("NCCL_P2P_NET_CHUNKSIZE");
  INFO(NCCL_LOG_INFO, "Environment variable found: NCCL_P2P_NET_CHUNKSIZE=%s",
       envVar);

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx942", 32);

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: RCCL_VALUE_INVALID when environment variable is set (user
  // override)
  EXPECT_EQ(chunkSize, RCCL_VALUE_INVALID)
      << "When env var is set, should return RCCL_VALUE_INVALID";

  INFO(NCCL_LOG_INFO, "Environment variable test completed - chunk size: %d",
       chunkSize);
  INFO(NCCL_LOG_INFO,
       "User override via NCCL_P2P_NET_CHUNKSIZE=%s was respected", envVar);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, EmptyArchString) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("EmptyArchString")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipP2pTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize with empty architecture string");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "", 32);

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: RCCL_VALUE_INVALID for empty/invalid architecture
  EXPECT_EQ(chunkSize, RCCL_VALUE_INVALID)
      << "Empty architecture should set chunk size to RCCL_VALUE_INVALID";

  INFO(NCCL_LOG_INFO, "Empty architecture test completed - chunk size: %d",
       chunkSize);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, PartialArchMatch) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("PartialArchMatch")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipP2pTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize with partial architecture match");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx94", 32);

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: RCCL_VALUE_INVALID for partial match
  EXPECT_EQ(chunkSize, RCCL_VALUE_INVALID)
      << "Partial architecture match should set chunk size to "
         "RCCL_VALUE_INVALID";

  INFO(NCCL_LOG_INFO,
       "Partial architecture match test completed - chunk size: %d", chunkSize);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, ZeroRanks_GFX942) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("ZeroRanks_GFX942")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipP2pTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize with zero ranks for GFX942");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx942", 0);

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: 1 << 17 = 131072 (since 0 < 64)
  EXPECT_EQ(chunkSize, 1 << 17)
      << "Zero ranks should be treated as < 64, setting chunk size to 131072";

  INFO(NCCL_LOG_INFO, "Zero ranks test completed - chunk size: %d", chunkSize);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, ZeroRanks_GFX950) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("ZeroRanks_GFX950")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipP2pTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize with zero ranks for GFX950");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx950", 0);

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: 1 << 17 = 131072 (since 0 < 16)
  EXPECT_EQ(chunkSize, 1 << 17)
      << "Zero ranks should be treated as < 16, setting chunk size to 131072";

  INFO(NCCL_LOG_INFO, "Zero ranks GFX950 test completed - chunk size: %d",
       chunkSize);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, LargeRankValues_GFX950) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("LargeRankValues_GFX950")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipP2pTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize with very large rank values for GFX950");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx950", 1000000);

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: 1 << 19 = 524288 (since 1000000 >= 32)
  EXPECT_EQ(chunkSize, 1 << 19) << "Very large ranks should be treated as >= "
                                   "32, setting chunk size to 524288";

  INFO(NCCL_LOG_INFO, "Large rank values test completed - chunk size: %d",
       chunkSize);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, CaseInsensitiveArch) {
  // Check execution order first
  if (ShouldSkipP2pTestDueToExecutionOrder("CaseInsensitiveArch")) {
    GTEST_SKIP() << "Skipping due to execution order - another "
                    "rcclSetP2pNetChunkSize test already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipP2pTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_P2P_NET_CHUNKSIZE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO,
       "Testing rcclSetP2pNetChunkSize with case variations in architecture");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "GFX942", 32); // Uppercase

  int chunkSize = RCCL_VALUE_UNSET;
  rcclSetP2pNetChunkSize(mockComm, chunkSize);

  // Expected: RCCL_VALUE_INVALID (case sensitive matching expected)
  EXPECT_EQ(chunkSize, RCCL_VALUE_INVALID)
      << "Uppercase architecture should not match (case sensitive)";

  INFO(NCCL_LOG_INFO,
       "Case insensitive architecture test completed - chunk size: %d",
       chunkSize);

  CleanupMockComm(mockComm);
}

// Add these test cases after the existing rcclSetP2pNetChunkSize tests

TEST(Rcclwrap, PXN_GFX942_SmallRanks) {
  // Check execution order first
  if (ShouldSkipPxnTestDueToExecutionOrder("PXN_GFX942_SmallRanks")) {
    GTEST_SKIP() << "Skipping due to execution order - another rcclSetPxn test "
                    "already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipPxnTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_PXN_DISABLE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO, "Testing rcclSetPxn for GFX942 with small ranks");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx942", 32);

  int pxnDisable = RCCL_VALUE_UNSET;
  rcclSetPxn(mockComm, pxnDisable);

  // Expected: 1 (disabled) for ranks < 64 on GFX942
  EXPECT_EQ(pxnDisable, 1)
      << "GFX942 with ranks < 64 should disable PXN (pxnDisable = 1)";

  INFO(NCCL_LOG_INFO, "GFX942 small ranks PXN test completed - pxnDisable: %d",
       pxnDisable);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, PXN_GFX942_LargeRanks) {
  // Check execution order first
  if (ShouldSkipPxnTestDueToExecutionOrder("PXN_GFX942_LargeRanks")) {
    GTEST_SKIP() << "Skipping due to execution order - another rcclSetPxn test "
                    "already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipPxnTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_PXN_DISABLE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO, "Testing rcclSetPxn for GFX942 with large ranks");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx942", 128);

  int pxnDisable = RCCL_VALUE_UNSET;
  rcclSetPxn(mockComm, pxnDisable);

  // Expected: 0 (enabled) for ranks >= 64 on GFX942
  EXPECT_EQ(pxnDisable, 0)
      << "GFX942 with ranks >= 64 should enable PXN (pxnDisable = 0)";

  INFO(NCCL_LOG_INFO, "GFX942 large ranks PXN test completed - pxnDisable: %d",
       pxnDisable);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, PXN_GFX942_BoundaryRank64) {
  // Check execution order first
  if (ShouldSkipPxnTestDueToExecutionOrder("PXN_GFX942_BoundaryRank64")) {
    GTEST_SKIP() << "Skipping due to execution order - another rcclSetPxn test "
                    "already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipPxnTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_PXN_DISABLE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO, "Testing rcclSetPxn for GFX942 with boundary rank 64");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx942", 64);

  int pxnDisable = RCCL_VALUE_UNSET;
  rcclSetPxn(mockComm, pxnDisable);

  // Expected: 0 (enabled) for ranks >= 64 on GFX942
  EXPECT_EQ(pxnDisable, 0)
      << "GFX942 with ranks = 64 should enable PXN (pxnDisable = 0)";

  INFO(NCCL_LOG_INFO,
       "GFX942 boundary rank 64 PXN test completed - pxnDisable: %d",
       pxnDisable);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, PXN_GFX942_BoundaryRank63) {
  // Check execution order first
  if (ShouldSkipPxnTestDueToExecutionOrder("PXN_GFX942_BoundaryRank63")) {
    GTEST_SKIP() << "Skipping due to execution order - another rcclSetPxn test "
                    "already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipPxnTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_PXN_DISABLE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO, "Testing rcclSetPxn for GFX942 with boundary rank 63");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx942", 63);

  int pxnDisable = RCCL_VALUE_UNSET;
  rcclSetPxn(mockComm, pxnDisable);

  // Expected: 1 (disabled) for ranks < 64 on GFX942
  EXPECT_EQ(pxnDisable, 1)
      << "GFX942 with ranks = 63 should disable PXN (pxnDisable = 1)";

  INFO(NCCL_LOG_INFO,
       "GFX942 boundary rank 63 PXN test completed - pxnDisable: %d",
       pxnDisable);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, PXN_GFX950_SmallRanks) {
  // Check execution order first
  if (ShouldSkipPxnTestDueToExecutionOrder("PXN_GFX950_SmallRanks")) {
    GTEST_SKIP() << "Skipping due to execution order - another rcclSetPxn test "
                    "already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipPxnTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_PXN_DISABLE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO, "Testing rcclSetPxn for GFX950 with small ranks");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx950", 16);

  int pxnDisable = RCCL_VALUE_UNSET;
  rcclSetPxn(mockComm, pxnDisable);

  // Expected: 1 (disabled) for ranks < 32 on GFX950
  EXPECT_EQ(pxnDisable, 1)
      << "GFX950 with ranks < 32 should disable PXN (pxnDisable = 1)";

  INFO(NCCL_LOG_INFO, "GFX950 small ranks PXN test completed - pxnDisable: %d",
       pxnDisable);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, PXN_GFX950_LargeRanks) {
  // Check execution order first
  if (ShouldSkipPxnTestDueToExecutionOrder("PXN_GFX950_LargeRanks")) {
    GTEST_SKIP() << "Skipping due to execution order - another rcclSetPxn test "
                    "already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipPxnTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_PXN_DISABLE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO, "Testing rcclSetPxn for GFX950 with large ranks");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx950", 64);

  int pxnDisable = RCCL_VALUE_UNSET;
  rcclSetPxn(mockComm, pxnDisable);

  // Expected: 0 (enabled) for ranks >= 32 on GFX950
  EXPECT_EQ(pxnDisable, 0)
      << "GFX950 with ranks >= 32 should enable PXN (pxnDisable = 0)";

  INFO(NCCL_LOG_INFO, "GFX950 large ranks PXN test completed - pxnDisable: %d",
       pxnDisable);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, PXN_GFX950_BoundaryRank32) {
  // Check execution order first
  if (ShouldSkipPxnTestDueToExecutionOrder("PXN_GFX950_BoundaryRank32")) {
    GTEST_SKIP() << "Skipping due to execution order - another rcclSetPxn test "
                    "already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipPxnTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_PXN_DISABLE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO, "Testing rcclSetPxn for GFX950 with boundary rank 32");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx950", 32);

  int pxnDisable = RCCL_VALUE_UNSET;
  rcclSetPxn(mockComm, pxnDisable);

  // Expected: 0 (enabled) for ranks >= 32 on GFX950
  EXPECT_EQ(pxnDisable, 0)
      << "GFX950 with ranks = 32 should enable PXN (pxnDisable = 0)";

  INFO(NCCL_LOG_INFO,
       "GFX950 boundary rank 32 PXN test completed - pxnDisable: %d",
       pxnDisable);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, PXN_GFX950_BoundaryRank31) {
  // Check execution order first
  if (ShouldSkipPxnTestDueToExecutionOrder("PXN_GFX950_BoundaryRank31")) {
    GTEST_SKIP() << "Skipping due to execution order - another rcclSetPxn test "
                    "already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipPxnTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_PXN_DISABLE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO, "Testing rcclSetPxn for GFX950 with boundary rank 31");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx950", 31);

  int pxnDisable = RCCL_VALUE_UNSET;
  rcclSetPxn(mockComm, pxnDisable);

  // Expected: 1 (disabled) for ranks < 32 on GFX950
  EXPECT_EQ(pxnDisable, 1)
      << "GFX950 with ranks = 31 should disable PXN (pxnDisable = 1)";

  INFO(NCCL_LOG_INFO,
       "GFX950 boundary rank 31 PXN test completed - pxnDisable: %d",
       pxnDisable);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, PXN_UnsupportedArch_GFX908) {
  // Check execution order first
  if (ShouldSkipPxnTestDueToExecutionOrder("PXN_UnsupportedArch_GFX908")) {
    GTEST_SKIP() << "Skipping due to execution order - another rcclSetPxn test "
                    "already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipPxnTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_PXN_DISABLE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO, "Testing rcclSetPxn for unsupported architecture GFX908");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx908", 32);

  int pxnDisable = RCCL_VALUE_UNSET;
  rcclSetPxn(mockComm, pxnDisable);

  // Expected: RCCL_VALUE_INVALID for unsupported architectures
  EXPECT_EQ(pxnDisable, RCCL_VALUE_INVALID)
      << "Unsupported architecture GFX908 should set pxnDisable to "
         "RCCL_VALUE_INVALID";

  INFO(NCCL_LOG_INFO,
       "Unsupported architecture GFX908 PXN test completed - pxnDisable: %d",
       pxnDisable);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, PXN_UnsupportedArch_GFX90A) {
  // Check execution order first
  if (ShouldSkipPxnTestDueToExecutionOrder("PXN_UnsupportedArch_GFX90A")) {
    GTEST_SKIP() << "Skipping due to execution order - another rcclSetPxn test "
                    "already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipPxnTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_PXN_DISABLE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO, "Testing rcclSetPxn for unsupported architecture GFX90A");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx90a", 32);

  int pxnDisable = RCCL_VALUE_UNSET;
  rcclSetPxn(mockComm, pxnDisable);

  // Expected: RCCL_VALUE_INVALID for unsupported architectures
  EXPECT_EQ(pxnDisable, RCCL_VALUE_INVALID)
      << "Unsupported architecture GFX90A should set pxnDisable to "
         "RCCL_VALUE_INVALID";

  INFO(NCCL_LOG_INFO,
       "Unsupported architecture GFX90A PXN test completed - pxnDisable: %d",
       pxnDisable);

  CleanupMockComm(mockComm);
}

// This test specifically tests the environment variable behavior
TEST(Rcclwrap, PXN_WithEnvironmentVariable) {
  // Check execution order first
  if (ShouldSkipPxnTestDueToExecutionOrder("PXN_WithEnvironmentVariable")) {
    GTEST_SKIP() << "Skipping due to execution order - another rcclSetPxn test "
                    "already ran";
  }

  // This test requires environment variable to be set to a specific value
  if (ShouldSkipPxnTest("1")) {
    GTEST_SKIP() << "Skipping test: NCCL_PXN_DISABLE environment variable is "
                    "not set to '1'. "
                 << "Please set: export NCCL_PXN_DISABLE=1 to run this test. "
                 << "This test verifies that user override via environment "
                    "variable works correctly.";
  }

  INFO(NCCL_LOG_INFO, "Testing rcclSetPxn with environment variable set");

  // Environment variable is confirmed to be set to "1"
  const char *envVar = getenv("NCCL_PXN_DISABLE");
  INFO(NCCL_LOG_INFO, "Environment variable found: NCCL_PXN_DISABLE=%s",
       envVar);

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx942", 128);

  int pxnDisable = RCCL_VALUE_UNSET;
  rcclSetPxn(mockComm, pxnDisable);

  // Expected: RCCL_VALUE_INVALID when environment variable is set (user
  // override)
  EXPECT_EQ(pxnDisable, RCCL_VALUE_INVALID)
      << "When env var is set, should return RCCL_VALUE_INVALID";

  INFO(NCCL_LOG_INFO,
       "Environment variable PXN test completed - pxnDisable: %d", pxnDisable);
  INFO(NCCL_LOG_INFO, "User override via NCCL_PXN_DISABLE=%s was respected",
       envVar);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, PXN_ZeroRanks_GFX942) {
  // Check execution order first
  if (ShouldSkipPxnTestDueToExecutionOrder("PXN_ZeroRanks_GFX942")) {
    GTEST_SKIP() << "Skipping due to execution order - another rcclSetPxn test "
                    "already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipPxnTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_PXN_DISABLE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO, "Testing rcclSetPxn with zero ranks for GFX942");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx942", 0);

  int pxnDisable = RCCL_VALUE_UNSET;
  rcclSetPxn(mockComm, pxnDisable);

  // Expected: 1 (disabled) since 0 < 64
  EXPECT_EQ(pxnDisable, 1)
      << "Zero ranks should be treated as < 64, disabling PXN (pxnDisable = 1)";

  INFO(NCCL_LOG_INFO, "Zero ranks GFX942 PXN test completed - pxnDisable: %d",
       pxnDisable);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, PXN_ZeroRanks_GFX950) {
  // Check execution order first
  if (ShouldSkipPxnTestDueToExecutionOrder("PXN_ZeroRanks_GFX950")) {
    GTEST_SKIP() << "Skipping due to execution order - another rcclSetPxn test "
                    "already ran";
  }

  // Check if we should skip this test due to environment variable being set
  if (ShouldSkipPxnTest()) {
    GTEST_SKIP()
        << "Skipping test: NCCL_PXN_DISABLE environment variable is set, "
        << "which would override the static variable behavior. "
        << "This test requires clean environment to test architecture logic.";
  }

  INFO(NCCL_LOG_INFO, "Testing rcclSetPxn with zero ranks for GFX950");

  ncclComm_t mockComm = nullptr;
  struct ncclTopoSystem mockTopo;
  struct ncclTopoNode mockGpuNode;
  CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx950", 0);

  int pxnDisable = RCCL_VALUE_UNSET;
  rcclSetPxn(mockComm, pxnDisable);

  // Expected: 1 (disabled) since 0 < 32
  EXPECT_EQ(pxnDisable, 1)
      << "Zero ranks should be treated as < 32, disabling PXN (pxnDisable = 1)";

  INFO(NCCL_LOG_INFO, "Zero ranks GFX950 PXN test completed - pxnDisable: %d",
       pxnDisable);

  CleanupMockComm(mockComm);
}

TEST(Rcclwrap, RcclSetPipelining_Invalid_DType) {
  // Skip the test if pipelining has been disabled
  // (RCCL_DISABLE_REDUCE_COPY_PIPELINING=1)
  if (ShouldSkipRcclSetPipeliningTests()) {
    GTEST_SKIP()
        << "Skipping test: RCCL_DISABLE_REDUCE_COPY_PIPELINING environment "
           "variable is set. Unset this variable to enable pipelining.";
  }

  // Skip the test if pipelining has been enabled for all data types
  // (RCCL_PIPELINE_ALL_DATA_TYPES=1)
  const char *allowAllDTypes = getenv("RCCL_PIPELINE_ALL_DATA_TYPES");
  if (allowAllDTypes && strcmp(allowAllDTypes, "0") != 0) {
    GTEST_SKIP() << "Skipping test: RCCL_PIPELINE_ALL_DATA_TYPES environment "
                    "variable is set. Unset this variable to enable pipelining "
                    "only for bf16 data type.";
  }

  // Pipeline should not be set for non-bf16 datatypes, unless
  // rcclParamPipelineAllDTypes() returns true
  ncclComm_t comm = nullptr;
  struct ncclTopoSystem topo;
  struct ncclTopoNode gpu;
  CreateMockComm(comm, topo, gpu, "gfx950", 8);
  comm->nNodes = 2; // Multi node

  ncclTaskColl info = {};
  info.func = ncclFuncAllReduce;
  info.datatype = ncclFloat32;

  size_t nBytes = 16 * 1024 * 1024; // 16MB
  rcclSetPipelining(comm, nBytes, &info);

  EXPECT_EQ(info.pipeline, 0) << "Non-bf16 should not set pipeline by default";

  CleanupMockComm(comm);
}

TEST(Rcclwrap, RcclSetPipelining_GFX950_MultiNode_Enable) {
  // Skip the test if pipelining has been disabled
  // (RCCL_DISABLE_REDUCE_COPY_PIPELINING=1)
  if (ShouldSkipRcclSetPipeliningTests()) {
    GTEST_SKIP()
        << "Skipping test: RCCL_DISABLE_REDUCE_COPY_PIPELINING environment "
           "variable is set. Unset this variable to enable pipelining.";
  }

  // For multi-node, pipeline is set to 1 for AllReduce with bf16
  ncclComm_t comm = nullptr;
  struct ncclTopoSystem topo;
  struct ncclTopoNode gpu;
  CreateMockComm(comm, topo, gpu, "gfx950", 8);
  comm->nNodes = 2; // Multi node

  ncclTaskColl info = {};
  // In rcclSetPipelining(), ncclFuncAllReduce, ncclFuncReduceScatter, and
  // ncclFuncReduce share the same case body. Testing any one of them is
  // sufficient to validate that code path.
  info.func = ncclFuncAllReduce;
  info.datatype = ncclBfloat16;

  size_t nBytes = 16 * 1024 * 1024; // 16MB
  rcclSetPipelining(comm, nBytes, &info);

  EXPECT_EQ(info.pipeline, 1)
      << "gfx950 multi-node AllReduce bf16 should enable pipelining";

  CleanupMockComm(comm);
}

TEST(Rcclwrap, RcclSetPipelining_GFX950_SingleNode_Disable) {
  // Skip the test if pipelining has been disabled
  // (RCCL_DISABLE_REDUCE_COPY_PIPELINING=1)
  if (ShouldSkipRcclSetPipeliningTests()) {
    GTEST_SKIP()
        << "Skipping test: RCCL_DISABLE_REDUCE_COPY_PIPELINING environment "
           "variable is set. Unset this variable to enable pipelining.";
  }

  // For single-node, pipeline remains 0
  ncclComm_t comm = nullptr;
  struct ncclTopoSystem topo;
  struct ncclTopoNode gpu;
  CreateMockComm(comm, topo, gpu, "gfx950", 8);
  comm->nNodes = 1; // Single node

  ncclTaskColl info = {};
  // In rcclSetPipelining(), ncclFuncAllReduce, ncclFuncReduceScatter, and
  // ncclFuncReduce share the same case body. Testing any one of them is
  // sufficient to validate that code path.
  info.func = ncclFuncAllReduce;
  info.datatype = ncclBfloat16;

  size_t nBytes = 16 * 1024 * 1024; // 16MB
  rcclSetPipelining(comm, nBytes, &info);

  EXPECT_EQ(info.pipeline, 0)
      << "gfx950 single-node should not enable pipelining";

  CleanupMockComm(comm);
}

TEST(Rcclwrap, RcclSetPipelining_GFX942_SingleNode_AllReduce_Enable) {
  // Skip the test if pipelining has been disabled
  // (RCCL_DISABLE_REDUCE_COPY_PIPELINING=1)
  if (ShouldSkipRcclSetPipeliningTests()) {
    GTEST_SKIP()
        << "Skipping test: RCCL_DISABLE_REDUCE_COPY_PIPELINING environment "
           "variable is set. Unset this variable to enable pipelining.";
  }

  // For single-node, pipeline is set to 1 for AllReduce with bf16
  ncclComm_t comm = nullptr;
  struct ncclTopoSystem topo;
  struct ncclTopoNode gpu;
  CreateMockComm(comm, topo, gpu, "gfx942", 8);
  comm->nNodes = 1; // Single node

  ncclTaskColl info = {};
  info.func = ncclFuncAllReduce;
  info.datatype = ncclBfloat16;

  size_t nBytes = 16 * 1024 * 1024; // 16MB
  rcclSetPipelining(comm, nBytes, &info);

  EXPECT_EQ(info.pipeline, 1)
      << "gfx942 single-node AllReduce bf16 should enable pipelining";

  CleanupMockComm(comm);
}

TEST(Rcclwrap, RcclSetPipelining_GFX942_MultiNode_AllReduce_Enable) {
  // Skip the test if pipelining has been disabled
  // (RCCL_DISABLE_REDUCE_COPY_PIPELINING=1)
  if (ShouldSkipRcclSetPipeliningTests()) {
    GTEST_SKIP()
        << "Skipping test: RCCL_DISABLE_REDUCE_COPY_PIPELINING environment "
           "variable is set. Unset this variable to enable pipelining.";
  }

  // For multi-node AllReduce with bf16, pipelining is enabled if
  // nBytes <= 512MB * 2^(log2(nNodes)-1)
  // Testing with nNodes = 4  => threshold = 512MB * 2^(2-1) = 1GB
  ncclComm_t comm = nullptr;
  struct ncclTopoSystem topo;
  struct ncclTopoNode gpu;
  CreateMockComm(comm, topo, gpu, "gfx942", 8);
  comm->nNodes = 4;

  ncclTaskColl info = {};
  info.func = ncclFuncAllReduce;
  info.datatype = ncclBfloat16;

  size_t nBytes = (1ULL << 30); // 1GB, exactly at threshold
  rcclSetPipelining(comm, nBytes, &info);

  EXPECT_EQ(info.pipeline, 1)
      << "gfx942 4-node AllReduce at threshold should enable pipelining";

  CleanupMockComm(comm);
}

TEST(Rcclwrap, RcclSetPipelining_GFX942_MultiNode_AllReduce_Disable) {
  // Skip the test if pipelining has been disabled
  // (RCCL_DISABLE_REDUCE_COPY_PIPELINING=1)
  if (ShouldSkipRcclSetPipeliningTests()) {
    GTEST_SKIP()
        << "Skipping test: RCCL_DISABLE_REDUCE_COPY_PIPELINING environment "
           "variable is set. Unset this variable to enable pipelining.";
  }

  // When nBytes is just above the threshold, pipelining should be disabled
  ncclComm_t comm = nullptr;
  struct ncclTopoSystem topo;
  struct ncclTopoNode gpu;
  CreateMockComm(comm, topo, gpu, "gfx942", 8);
  comm->nNodes = 4;

  ncclTaskColl info = {};
  info.func = ncclFuncAllReduce;
  info.datatype = ncclBfloat16;

  size_t nBytes = (1ULL << 30) + 1024; // 1GB + 1KB, just above threshold
  rcclSetPipelining(comm, nBytes, &info);

  EXPECT_EQ(info.pipeline, 0)
      << "gfx942 4-node AllReduce above threshold should disable pipelining";

  CleanupMockComm(comm);
}

TEST(Rcclwrap, RcclSetPipelining_GFX942_Enable) {
  // Skip the test if pipelining has been disabled
  // (RCCL_DISABLE_REDUCE_COPY_PIPELINING=1)
  if (ShouldSkipRcclSetPipeliningTests()) {
    GTEST_SKIP()
        << "Skipping test: RCCL_DISABLE_REDUCE_COPY_PIPELINING environment "
           "variable is set. Unset this variable to enable pipelining.";
  }

  // ReduceScatter & Reduce should enable pipelining regardless of no. of nodes
  ncclComm_t comm = nullptr;
  struct ncclTopoSystem topo;
  struct ncclTopoNode gpu;
  CreateMockComm(comm, topo, gpu, "gfx942", 8);
  comm->nNodes = 8;

  ncclTaskColl info = {};
  // In rcclSetPipelining(), ncclFuncReduceScatter, and
  // ncclFuncReduce share the same case body. Testing any one of them is
  // sufficient to validate that code path.
  info.func = ncclFuncReduceScatter;
  info.datatype = ncclBfloat16;

  size_t nBytes = 16 * 1024 * 1024; // 16MB
  rcclSetPipelining(comm, nBytes, &info);

  EXPECT_EQ(info.pipeline, 1)
      << "gfx942 ReduceScatter and Reduce should enable "
         "pipelining with single or multi-node";

  CleanupMockComm(comm);
}

TEST(Rcclwrap, RcclOverrideProtocol_NoOverride) {
  const char *protoOverrideEnv = getenv("RCCL_OVERRIDE_PROTO");
  // Skip the test if RCCL_OVERRIDE_PROTO is set
  if (protoOverrideEnv) {
    GTEST_SKIP() << "Skipping test: Variable RCCL_OVERRIDE_PROTO is set. Unset "
                    "it to run this test.";
  }

  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  ncclTaskColl info = {};

  ncclResult_t result = rcclOverrideProtocol(ncclProtoStr, table, &info);

  EXPECT_EQ(result, ncclSuccess)
      << "Expected ncclSuccess when RCCL_OVERRIDE_PROTO is unset, indicating "
         "no override should be applied.";
}

TEST(Rcclwrap, RcclOverrideProtocol_UnsupportedOverride) {
  const char *protoOverrideEnv = getenv("RCCL_OVERRIDE_PROTO");
  // Skip the test if RCCL_OVERRIDE_PROTO is not set or if its set to an invalid
  // value
  if (!isProtoStrValid(protoOverrideEnv)) {
    GTEST_SKIP()
        << "Skipping test: Variable RCCL_OVERRIDE_PROTO is not set or "
           "set to an invalid value. Set it to a valid protocol value to "
           "run this test.";
  }

  // Mark all combinations as unsupported for the purpose of this test.
  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; ++a)
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p)
      table[a][p] = NCCL_ALGO_PROTO_IGNORE;

  ncclTaskColl info = {};
  info.func = ncclFuncReduceScatter;
  info.datatype = ncclBfloat16;
  info.algorithm = NCCL_ALGO_RING; // Set any algorithm

  ncclResult_t result = rcclOverrideProtocol(ncclProtoStr, table, &info);

  EXPECT_EQ(result, ncclInternalError)
      << "Expected ncclInternalError when the override protocol is valid, but "
         "not enabled for the selected algorithm.";
}

TEST(Rcclwrap, RcclOverrideProtocol_ValidOverride) {
  const char *protoOverrideEnv = getenv("RCCL_OVERRIDE_PROTO");
  // Skip the test if RCCL_OVERRIDE_PROTO is not set or if its set to an invalid
  // value
  if (!isProtoStrValid(protoOverrideEnv)) {
    GTEST_SKIP() << "Skipping test: RCCL_OVERRIDE_PROTO is not set or set to "
                    "an invalid value. Set it to a valid protocol name (e.g., "
                    "'Simple') to run this test.";
  }

  // Get the index of the protocol from the string for later comparison
  int protoIndex = NCCL_PROTO_UNDEF;
  ncclResult_t idxResult = rcclGetAlgoProtoIndex(
      protoOverrideEnv, ncclProtoStr, NCCL_NUM_PROTOCOLS, protoIndex);
  ASSERT_EQ(idxResult, ncclSuccess)
      << "Failed to get protocol index from string";

  // Mark all combinations as valid for the purpose of this test.
  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; ++a)
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p)
      table[a][p] = 0.0;

  ncclTaskColl info = {};
  info.func = ncclFuncAllReduce;
  info.datatype = ncclBfloat16;
  info.algorithm = NCCL_ALGO_RING; // Set any algorithm
  info.protocol = NCCL_PROTO_UNDEF;

  ncclResult_t result = rcclOverrideProtocol(ncclProtoStr, table, &info);

  EXPECT_EQ(result, ncclSuccess)
      << "Expected ncclSuccess when override is applied successfully.";
  EXPECT_EQ(info.protocol, protoIndex) << "Protocol index should match the "
                                          "override value from environment.";
}

TEST(Rcclwrap, RcclOverrideProtocol_ValidOverridePersists) {
  const char *protoOverrideEnv = getenv("RCCL_OVERRIDE_PROTO");
  // Skip the test if RCCL_OVERRIDE_PROTO is not set or if its set to an invalid
  // value
  if (!isProtoStrValid(protoOverrideEnv)) {
    GTEST_SKIP()
        << "Skipping test: RCCL_OVERRIDE_PROTO is not set or set to an invalid "
           "value. Set it to a valid protocol name (e.g., 'Simple') to run "
           "this test.";
  }

  // Get the index of the protocol from the string for later comparison
  int protoIndex = NCCL_PROTO_UNDEF;
  ncclResult_t idxResult = rcclGetAlgoProtoIndex(
      protoOverrideEnv, ncclProtoStr, NCCL_NUM_PROTOCOLS, protoIndex);
  ASSERT_EQ(idxResult, ncclSuccess)
      << "Failed to get protocol index from string";

  // Mark all combinations as valid for the purpose of this test.
  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; ++a)
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p)
      table[a][p] = 0.0;

  ncclTaskColl info = {};
  info.func = ncclFuncAllReduce;
  info.datatype = ncclFloat16;
  info.algorithm = NCCL_ALGO_RING; // Set any algorithm
  info.protocol = NCCL_PROTO_UNDEF;

  // First call
  ncclResult_t result1 = rcclOverrideProtocol(ncclProtoStr, table, &info);
  EXPECT_EQ(result1, ncclSuccess)
      << "Expected rcclOverrideProtocol to succeed with valid override";
  EXPECT_EQ(info.protocol, protoIndex)
      << "Expected protocol to match override after first call";

  // Second call
  ncclResult_t result2 = rcclOverrideProtocol(ncclProtoStr, table, &info);
  EXPECT_EQ(result2, ncclSuccess)
      << "Expected rcclOverrideProtocol to succeed again on second call";
  EXPECT_EQ(info.protocol, protoIndex)
      << "Expected protocol to match override after second call";
}

TEST(Rcclwrap, RcclOverrideProtocol_InvalidProtocol) {
  const char *protoOverrideEnv = getenv("RCCL_OVERRIDE_PROTO");
  // Skip the test if RCCL_OVERRIDE_PROTO is not set or if its set to a valid
  // value
  if (!protoOverrideEnv || isProtoStrValid(protoOverrideEnv)) {
    GTEST_SKIP()
        << "Skipping test: Variable RCCL_OVERRIDE_PROTO is not set or set to a "
           "valid value. Set it to an invalid protocol value to run this test.";
  }

  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  ncclTaskColl info = {};

  ncclResult_t result = rcclOverrideProtocol(ncclProtoStr, table, &info);

  EXPECT_EQ(result, ncclInvalidUsage) << "Expected ncclInvalidUsage when the "
                                         "override protocol is invalid.";
}

TEST(Rcclwrap, RcclOverrideProtocol_InvalidOverridePersists) {
  const char *protoOverrideEnv = getenv("RCCL_OVERRIDE_PROTO");
  if (!protoOverrideEnv || isProtoStrValid(protoOverrideEnv)) {
    GTEST_SKIP()
        << "Skipping test: Variable RCCL_OVERRIDE_PROTO is not set or set to a "
           "valid value. Set it to an invalid protocol value to run this test.";
  }

  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  ncclTaskColl info = {};

  // First call should fail due to invalid proto string
  ncclResult_t result1 = rcclOverrideProtocol(ncclProtoStr, table, &info);
  EXPECT_EQ(result1, ncclInvalidUsage)
      << "Expected rcclOverrideProtocol to fail with invalid "
         "RCCL_OVERRIDE_PROTO.";

  // Second call should still fail because the static variable disables further
  // overrides
  ncclResult_t result2 = rcclOverrideProtocol(ncclProtoStr, table, &info);
  EXPECT_EQ(result2, ncclInvalidUsage)
      << "Expected rcclOverrideProtocol to continue returning failure after "
         "invalid proto was set.";
}

TEST(Rcclwrap, RcclOverrideAlgorithm_NoOverride) {
  const char *algoOverrideEnv = getenv("RCCL_OVERRIDE_ALGO");
  // Skip the test if RCCL_OVERRIDE_ALGO is set
  if (algoOverrideEnv) {
    GTEST_SKIP() << "Skipping test: Variable RCCL_OVERRIDE_ALGO is set. Unset "
                    "it to run this test.";
  }

  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  ncclTaskColl info = {};

  ncclResult_t result = rcclOverrideAlgorithm(ncclAlgoStr, table, &info);

  // Since no override is set, it should return success and do nothing
  EXPECT_EQ(result, ncclSuccess)
      << "Expected ncclSuccess when RCCL_OVERRIDE_ALGO is unset, indicating no "
         "override should be applied.";
}

TEST(Rcclwrap, RcclOverrideAlgorithm_UnsupportedOverride) {
  const char *algoOverrideEnv = getenv("RCCL_OVERRIDE_ALGO");
  // Skip the test if RCCL_OVERRIDE_ALGO is not set or if its set to an invalid
  // value
  if (!isAlgoStrValid(algoOverrideEnv)) {
    GTEST_SKIP() << "Skipping test: RCCL_OVERRIDE_ALGO is not set or "
                    "set to an invalid value. Set it to a valid algorithm to "
                    "run this test.";
  }

  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; ++a)
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p)
      table[a][p] = NCCL_ALGO_PROTO_IGNORE;

  ncclTaskColl info = {};
  info.func = ncclFuncReduceScatter;
  info.datatype = ncclBfloat16;
  info.protocol = NCCL_PROTO_SIMPLE; // Set any protocol

  ncclResult_t result = rcclOverrideAlgorithm(ncclAlgoStr, table, &info);

  EXPECT_EQ(result, ncclInternalError)
      << "Expected ncclInternalError when the override algorithm is valid, but "
         "not enabled for the selected protocol.";
}

TEST(Rcclwrap, RcclOverrideAlgorithm_ValidOverride) {
  const char *algoOverrideEnv = getenv("RCCL_OVERRIDE_ALGO");
  // Skip the test if RCCL_OVERRIDE_ALGO is not set or if its set to an invalid
  // value
  if (!isAlgoStrValid(algoOverrideEnv)) {
    GTEST_SKIP() << "Skipping test: RCCL_OVERRIDE_ALGO is not set or set to "
                    "an invalid value. Set it to a valid algorithm name (e.g., "
                    "'Ring') to run this test.";
  }

  // Get the index of the algorithm from the string for later comparison
  int algoIndex = NCCL_ALGO_UNDEF;
  ncclResult_t idxResult = rcclGetAlgoProtoIndex(
      algoOverrideEnv, ncclAlgoStr, NCCL_NUM_ALGORITHMS, algoIndex);
  ASSERT_EQ(idxResult, ncclSuccess)
      << "Failed to get algorithm index from string";

  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  // Mark all combinations as valid for the purpose of this test.
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; ++a)
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p)
      table[a][p] = 0.0;

  ncclTaskColl info = {};
  info.func = ncclFuncAllReduce;
  info.datatype = ncclBfloat16;
  info.protocol = NCCL_PROTO_SIMPLE; // Set any protocol
  info.algorithm = NCCL_ALGO_UNDEF;

  ncclResult_t result = rcclOverrideAlgorithm(ncclAlgoStr, table, &info);

  EXPECT_EQ(result, ncclSuccess)
      << "Expected ncclSuccess when override is applied successfully.";
  EXPECT_EQ(info.algorithm, algoIndex)
      << "Algorithm index should match the override value from environment.";
}

TEST(Rcclwrap, RcclOverrideAlgorithm_ValidOverridePersists) {
  const char *algoOverrideEnv = getenv("RCCL_OVERRIDE_ALGO");
  // Skip the test if RCCL_OVERRIDE_ALGO is not set or if its set to an invalid
  // value
  if (!isAlgoStrValid(algoOverrideEnv)) {
    GTEST_SKIP()
        << "Skipping test: RCCL_OVERRIDE_ALGO is not set or set to an invalid "
           "value. Set it to a valid algorithm name (e.g., 'Ring') to run this "
           "test.";
  }

  // Get the index of the algorithm from the string for later comparison
  int algoIndex = NCCL_ALGO_UNDEF;
  ncclResult_t idxResult = rcclGetAlgoProtoIndex(
      algoOverrideEnv, ncclAlgoStr, NCCL_NUM_ALGORITHMS, algoIndex);
  ASSERT_EQ(idxResult, ncclSuccess)
      << "Failed to get algorithm index from string";

  // Mark all combinations as valid for the purpose of this test.
  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; ++a)
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p)
      table[a][p] = 0.0;

  ncclTaskColl info = {};
  info.func = ncclFuncAllReduce;
  info.datatype = ncclFloat16;
  info.protocol = NCCL_PROTO_SIMPLE; // Set any protocol
  info.algorithm = NCCL_ALGO_UNDEF;

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
}

TEST(Rcclwrap, RcclOverrideAlgorithm_InvalidAlgorithm) {
  const char *algoOverrideEnv = getenv("RCCL_OVERRIDE_ALGO");
  // Skip the test if RCCL_OVERRIDE_ALGO is not set or if its set to a valid
  // value
  if (!algoOverrideEnv || isAlgoStrValid(algoOverrideEnv)) {
    GTEST_SKIP() << "Skipping test: RCCL_OVERRIDE_ALGO is not set or set to a "
                    "valid value. Set it to an invalid algorithm value to run "
                    "this test.";
  }

  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  ncclTaskColl info = {};

  ncclResult_t result = rcclOverrideAlgorithm(ncclAlgoStr, table, &info);

  EXPECT_EQ(result, ncclInvalidUsage)
      << "Expected ncclInvalidUsage when the override algorithm is invalid.";
}

TEST(Rcclwrap, RcclOverrideAlgorithm_InvalidOverridePersists) {
  const char *algoOverrideEnv = getenv("RCCL_OVERRIDE_ALGO");
  // Skip the test if RCCL_OVERRIDE_ALGO is not set or if its set to a valid
  // value
  if (!algoOverrideEnv || isAlgoStrValid(algoOverrideEnv)) {
    GTEST_SKIP()
        << "Skipping test: RCCL_OVERRIDE_ALGO is not set or set to a valid "
           "value. Set it to an invalid algorithm name to run this test.";
  }

  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  ncclTaskColl info = {};

  // First call should fail due to invalid algo string (and set the static flag)
  ncclResult_t result1 = rcclOverrideAlgorithm(ncclAlgoStr, table, &info);
  EXPECT_EQ(result1, ncclInvalidUsage)
      << "Expected rcclOverrideAlgorithm to fail with invalid "
         "RCCL_OVERRIDE_ALGO.";

  // Second call should also fail due to static validInput=false
  ncclResult_t result2 = rcclOverrideAlgorithm(ncclAlgoStr, table, &info);
  EXPECT_EQ(result2, ncclInvalidUsage)
      << "Expected rcclOverrideAlgorithm to continue returning failure after "
         "invalid algo was set.";
}

} // namespace RcclUnitTesting
