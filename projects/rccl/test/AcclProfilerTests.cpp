/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for the accl-profiler plugin.
//
// Tests the pure-function helpers (acclDatatypeSize, acclBusBwFactor) directly
// and exercises the plugin lifecycle (init → startEvent → stopEvent → finalize)
// through process-isolated tests to keep global pool state clean.

#include <gtest/gtest.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include "common/ProcessIsolatedTestRunner.hpp"

// Forward declarations — thin wrappers defined in accl_profiler_wrapper.cc
extern "C" {
  int test_acclDatatypeSize(const char* dt);
  double test_acclBusBwFactor(const char* func, int nRanks);
}

// Plugin API (symbol visibility is __hidden but linked within the same binary)
#include "nccl/profiler.h"
extern ncclResult_t acclPluginInit(void**, uint64_t, int*, const char*,
                                   int, int, int, ncclDebugLogger_t);
extern ncclResult_t acclPluginStartEvent(void*, void**,
                                          ncclProfilerEventDescr_v5_t*);
extern ncclResult_t acclPluginStopEvent(void*);
extern ncclResult_t acclPluginRecordEventState(void*,
                                                ncclProfilerEventState_v5_t,
                                                ncclProfilerEventStateArgs_v5_t*);
extern ncclResult_t acclPluginFinalize(void*);

namespace RcclUnitTesting {

// =========================================================================
// acclDatatypeSize — table-driven coverage of all ncclDatatypeToString outputs
// =========================================================================
class AcclDatatypeSizeTest
    : public ::testing::TestWithParam<std::pair<const char*, int>> {};

TEST_P(AcclDatatypeSizeTest, MatchesExpected) {
    auto [input, expected] = GetParam();
    EXPECT_EQ(test_acclDatatypeSize(input), expected);
}

INSTANTIATE_TEST_SUITE_P(AllDatatypes, AcclDatatypeSizeTest,
    ::testing::Values(
        // 1-byte types (exact match)
        std::make_pair("ncclInt8", 1),
        std::make_pair("ncclUint8", 1),
        std::make_pair("ncclFloat8e4m3", 1),
        std::make_pair("ncclFloat8e5m2", 1),
        // 2-byte types
        std::make_pair("ncclFloat16", 2),
        std::make_pair("ncclBfloat16", 2),
        // 4-byte types
        std::make_pair("ncclInt32", 4),
        std::make_pair("ncclUint32", 4),
        std::make_pair("ncclFloat32", 4),
        // 8-byte types
        std::make_pair("ncclInt64", 8),
        std::make_pair("ncclUint64", 8),
        std::make_pair("ncclFloat64", 8),
        // Edge cases
        std::make_pair("Unknown", 4),
        std::make_pair(static_cast<const char*>(nullptr), 4),
        std::make_pair("", 4)
    )
);

// Substring fallback paths
TEST(AcclDatatypeSize, SubstringFallback_Int8) {
    EXPECT_EQ(test_acclDatatypeSize("someInt8type"), 1);
}

TEST(AcclDatatypeSize, SubstringFallback_Float8) {
    EXPECT_EQ(test_acclDatatypeSize("custom8e4m3"), 1);
    EXPECT_EQ(test_acclDatatypeSize("custom8e5m2"), 1);
}

TEST(AcclDatatypeSize, SubstringFallback_16bit) {
    EXPECT_EQ(test_acclDatatypeSize("customFloat16"), 2);
}

TEST(AcclDatatypeSize, SubstringFallback_64bit) {
    EXPECT_EQ(test_acclDatatypeSize("customInt64"), 8);
}

// =========================================================================
// acclBusBwFactor
// =========================================================================
TEST(AcclBusBwFactor, AllReduce) {
    EXPECT_DOUBLE_EQ(test_acclBusBwFactor("AllReduce", 8), 2.0 * 7.0 / 8.0);
    EXPECT_DOUBLE_EQ(test_acclBusBwFactor("AllReduce", 2), 1.0);
}

TEST(AcclBusBwFactor, ReduceScatter) {
    EXPECT_DOUBLE_EQ(test_acclBusBwFactor("ReduceScatter", 8), 7.0 / 8.0);
}

TEST(AcclBusBwFactor, AllGather) {
    EXPECT_DOUBLE_EQ(test_acclBusBwFactor("AllGather", 8), 7.0 / 8.0);
}

TEST(AcclBusBwFactor, AllToAll) {
    EXPECT_DOUBLE_EQ(test_acclBusBwFactor("AllToAll", 8), 7.0 / 8.0);
}

TEST(AcclBusBwFactor, BroadcastAndReduce) {
    EXPECT_DOUBLE_EQ(test_acclBusBwFactor("Broadcast", 8), 1.0);
    EXPECT_DOUBLE_EQ(test_acclBusBwFactor("Reduce", 8), 1.0);
}

TEST(AcclBusBwFactor, EdgeCases) {
    EXPECT_DOUBLE_EQ(test_acclBusBwFactor(nullptr, 8), 1.0);
    EXPECT_DOUBLE_EQ(test_acclBusBwFactor("AllReduce", 1), 1.0);
    EXPECT_DOUBLE_EQ(test_acclBusBwFactor("AllReduce", 0), 1.0);
    EXPECT_DOUBLE_EQ(test_acclBusBwFactor("UnknownColl", 8), 1.0);
}

// =========================================================================
// Plugin init/finalize lifecycle (process-isolated due to global pools)
// =========================================================================
TEST(AcclProfilerInit, InitAndFinalize) {
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerInit.InitAndFinalize",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ncclResult_t rc = acclPluginInit(
                &ctx, 0x1234, &mask, "test_comm", 1, 8, 0, nullptr);
            ASSERT_EQ(rc, 0);
            ASSERT_NE(ctx, nullptr);
            EXPECT_TRUE(mask & (1 << 1));  // ncclProfileColl
            EXPECT_TRUE(mask & (1 << 6));  // ncclProfileKernelCh
            EXPECT_TRUE(mask & (1 << 3));  // ncclProfileProxyOp
            EXPECT_TRUE(mask & (1 << 4));  // ncclProfileProxyStep
            EXPECT_EQ(acclPluginFinalize(ctx), 0);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", "/tmp"}}
    );
}

// =========================================================================
// Full lifecycle: Coll → KernelCh → stop → finalize → check output JSONL
// =========================================================================
TEST(AcclProfilerLifecycle, CollWithKernelChProducesOutput) {
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.CollWithKernelChProducesOutput",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xBEEF, &mask, "lifecycle_test",
                                     1, 2, 0, nullptr), 0);
            ASSERT_NE(ctx, nullptr);

            // Start a Coll event
            ncclProfilerEventDescr_v5_t collDescr;
            memset(&collDescr, 0, sizeof(collDescr));
            collDescr.type = (1 << 1);  // ncclProfileColl
            collDescr.coll.func = "AllReduce";
            collDescr.coll.algo = "Ring";
            collDescr.coll.proto = "Simple";
            collDescr.coll.datatype = "ncclFloat32";
            collDescr.coll.count = 1024;
            collDescr.coll.seqNumber = 10;
            collDescr.coll.nChannels = 1;

            void* collHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &collHandle, &collDescr), 0);
            ASSERT_NE(collHandle, nullptr);

            // Start a KernelCh event
            ncclProfilerEventDescr_v5_t kchDescr;
            memset(&kchDescr, 0, sizeof(kchDescr));
            kchDescr.type = (1 << 6);  // ncclProfileKernelCh
            kchDescr.parentObj = collHandle;
            kchDescr.kernelCh.channelId = 0;
            kchDescr.kernelCh.pTimer = 1000000;

            void* kchHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kchHandle, &kchDescr), 0);
            ASSERT_NE(kchHandle, nullptr);

            // RecordEventState for KernelCh stop (GPU timer)
            ncclProfilerEventStateArgs_v5_t stateArgs;
            memset(&stateArgs, 0, sizeof(stateArgs));
            stateArgs.kernelCh.pTimer = 1005000;  // 50 us at 100 MHz
            ASSERT_EQ(acclPluginRecordEventState(
                kchHandle,
                static_cast<ncclProfilerEventState_v5_t>(22),  // kernelChStop
                &stateArgs), 0);

            // Stop KernelCh then Coll
            ASSERT_EQ(acclPluginStopEvent(kchHandle), 0);
            ASSERT_EQ(acclPluginStopEvent(collHandle), 0);

            // Finalize and check output file exists with content
            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            // Verify output file contains valid JSONL
            char hostname[256] = {0};
            gethostname(hostname, sizeof(hostname) - 1);
            char path[1024];
            snprintf(path, sizeof(path),
                "/tmp/accl_test_lifecycle/accl_profiler_rank0_%s_pid%d_0xbeef.jsonl",
                hostname, (int)getpid());
            std::ifstream ifs(path);
            ASSERT_TRUE(ifs.good()) << "Output file not found: " << path;
            std::string line;
            ASSERT_TRUE(std::getline(ifs, line));
            EXPECT_FALSE(line.empty());
            EXPECT_EQ(line.front(), '{');
            EXPECT_EQ(line.back(), '}');
            EXPECT_NE(line.find("\"AllReduce\""), std::string::npos);
            EXPECT_NE(line.find("\"coll_msg_size_bytes\":4096"),
                       std::string::npos);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", "/tmp/accl_test_lifecycle"}}
    );
}

// =========================================================================
// ProxyOp refcount: verify proxy ops don't cause use-after-free
// =========================================================================
TEST(AcclProfilerLifecycle, ProxyOpAfterKernelStopIsValid) {
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.ProxyOpAfterKernelStopIsValid",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xCAFE, &mask, "proxy_test",
                                     1, 2, 0, nullptr), 0);

            // Start Coll
            ncclProfilerEventDescr_v5_t collDescr;
            memset(&collDescr, 0, sizeof(collDescr));
            collDescr.type = (1 << 1);
            collDescr.coll.func = "AllReduce";
            collDescr.coll.algo = "Ring";
            collDescr.coll.proto = "Simple";
            collDescr.coll.datatype = "ncclFloat32";
            collDescr.coll.count = 256;
            collDescr.coll.seqNumber = 20;
            collDescr.coll.nChannels = 1;

            void* collHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &collHandle, &collDescr), 0);

            // Start KernelCh
            ncclProfilerEventDescr_v5_t kchDescr;
            memset(&kchDescr, 0, sizeof(kchDescr));
            kchDescr.type = (1 << 6);
            kchDescr.parentObj = collHandle;
            kchDescr.kernelCh.channelId = 0;
            kchDescr.kernelCh.pTimer = 0;

            void* kchHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kchHandle, &kchDescr), 0);

            // Start ProxyOp (referencing the same coll)
            ncclProfilerEventDescr_v5_t proxyDescr;
            memset(&proxyDescr, 0, sizeof(proxyDescr));
            proxyDescr.type = (1 << 3);
            proxyDescr.parentObj = collHandle;
            proxyDescr.proxyOp.channelId = 0;
            proxyDescr.proxyOp.peer = 1;
            proxyDescr.proxyOp.nSteps = 1;
            proxyDescr.proxyOp.isSend = 1;

            void* proxyHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &proxyHandle, &proxyDescr), 0);

            // Stop Coll (self-ref drop)
            ASSERT_EQ(acclPluginStopEvent(collHandle), 0);

            // Stop KernelCh (kernel-completion ref + per-channel ref)
            ASSERT_EQ(acclPluginStopEvent(kchHandle), 0);

            // Stop ProxyOp AFTER kernel — this was the use-after-free scenario
            ASSERT_EQ(acclPluginStopEvent(proxyHandle), 0);

            ASSERT_EQ(acclPluginFinalize(ctx), 0);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", "/tmp/accl_test_proxy"}}
    );
}

} // namespace RcclUnitTesting
