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

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

#include <gtest/gtest.h>

#include "accl_shim.h"
#include "common/ProcessIsolatedTestRunner.hpp"

// Forward declarations — thin wrappers defined in accl_profiler_wrapper.cc
extern "C" {
  int test_acclDatatypeSize(const char* dt);
  double test_acclBusBwFactor(const char* func, int nRanks);
}
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
// Shared helpers for the lifecycle tests
// =========================================================================

// Returns the plugin's whole JSONL output for `commHashHex`, or "" if absent.
// Must be called inside the isolated child, since the filename embeds its pid.
static std::string ReadProfilerOutput(const char* dir, const char* commHashHex) {
    char hostname[256] = {0};
    gethostname(hostname, sizeof(hostname) - 1);
    char path[1024];
    snprintf(path, sizeof(path), "%s/accl_profiler_rank0_%s_pid%d_%s.jsonl",
             dir, hostname, (int)getpid(), commHashHex);
    std::ifstream ifs(path);
    if (!ifs.good()) return std::string();
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

// Fills a Coll event descriptor with the fields every lifecycle test needs.
static void MakeCollDescr(ncclProfilerEventDescr_v5_t* d, uint8_t nChannels,
                          uint64_t seqNumber, size_t count) {
    memset(d, 0, sizeof(*d));
    d->type = ncclProfileColl;
    d->coll.func = "AllReduce";
    d->coll.algo = "Ring";
    d->coll.proto = "Simple";
    d->coll.datatype = "ncclFloat32";
    d->coll.count = count;
    d->coll.seqNumber = seqNumber;
    d->coll.nChannels = nChannels;
}

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
        // ncclUint8 has no case in ncclDatatypeToString, arrives as "Unknown"
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
        std::make_pair("Unknown", 1),
        std::make_pair(static_cast<const char*>(nullptr), 4),
        std::make_pair("", 4)
    )
);

// Unrecognized types fall through to default size 4
TEST(AcclDatatypeSize, UnrecognizedFallback) {
    EXPECT_EQ(test_acclDatatypeSize("someCustomType"), 4);
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

TEST(AcclBusBwFactor, AlltoAll) {
    EXPECT_DOUBLE_EQ(test_acclBusBwFactor("AlltoAll", 8), 7.0 / 8.0);
    EXPECT_DOUBLE_EQ(test_acclBusBwFactor("AlltoAllv", 8), 7.0 / 8.0);
}

TEST(AcclBusBwFactor, BroadcastAndReduce) {
    EXPECT_DOUBLE_EQ(test_acclBusBwFactor("Broadcast", 8), 1.0);
    EXPECT_DOUBLE_EQ(test_acclBusBwFactor("Reduce", 8), 1.0);
    EXPECT_DOUBLE_EQ(test_acclBusBwFactor("Gather", 8), 1.0);
    EXPECT_DOUBLE_EQ(test_acclBusBwFactor("Scatter", 8), 1.0);
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
            EXPECT_TRUE(mask & ncclProfileColl);
            EXPECT_TRUE(mask & ncclProfileKernelCh);
            EXPECT_TRUE(mask & ncclProfileProxyOp);
            EXPECT_TRUE(mask & ncclProfileProxyStep);
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
            collDescr.type = ncclProfileColl;
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
            kchDescr.type = ncclProfileKernelCh;
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
                ncclProfilerKernelChStop,  // kernelChStop
                &stateArgs), 0);

            // Stop Coll first, then KernelCh — matches RCCL teardown ordering
            // where the enqueue thread fires Coll stop while the profiler
            // transport is still draining kernel channel events.
            ASSERT_EQ(acclPluginStopEvent(collHandle), 0);
            ASSERT_EQ(acclPluginStopEvent(kchHandle), 0);

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
            EXPECT_NE(line.find("\"coll_exec_time_us\":50.00"),
                       std::string::npos)
                << "Expected GPU-timed exec of 50us (5000 ticks / 100MHz)";
            EXPECT_NE(line.find("\"coll_timing_source\":\"gpu_globaltimer\""),
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
            collDescr.type = ncclProfileColl;
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
            kchDescr.type = ncclProfileKernelCh;
            kchDescr.parentObj = collHandle;
            kchDescr.kernelCh.channelId = 0;
            kchDescr.kernelCh.pTimer = 0;

            void* kchHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kchHandle, &kchDescr), 0);

            // Start ProxyOp (referencing the same coll)
            ncclProfilerEventDescr_v5_t proxyDescr;
            memset(&proxyDescr, 0, sizeof(proxyDescr));
            proxyDescr.type = ncclProfileProxyOp;
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

            // Verify the output JSONL carries proxy op data
            char hostname[256] = {0};
            gethostname(hostname, sizeof(hostname) - 1);
            char path[1024];
            snprintf(path, sizeof(path),
                "/tmp/accl_test_proxy/accl_profiler_rank0_%s_pid%d_0xcafe.jsonl",
                hostname, (int)getpid());
            std::ifstream ifs(path);
            ASSERT_TRUE(ifs.good()) << "Output file not found: " << path;
            std::string line;
            ASSERT_TRUE(std::getline(ifs, line));
            EXPECT_NE(line.find("\"n_proxy_ops\":1"), std::string::npos)
                << "Record must carry the proxy op that stopped after the kernel";
            EXPECT_NE(line.find("\"n_send_ops\":1"), std::string::npos);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", "/tmp/accl_test_proxy"}}
    );
}

// =========================================================================
// Pool exhaustion: full pool returns NULL and drops the collective
// =========================================================================
TEST(AcclProfilerLifecycle, PoolFullReturnsNull) {
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.PoolFullReturnsNull",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xD00D, &mask, "pool_test",
                                     1, 1, 0, nullptr), 0);

            // Fill the pool (256 slots) with collectives that stay pinned:
            // nChannels=1 but no KernelCh events, so acclShouldFinalize
            // never fires even after coll stop.
            void* handles[256];
            for (int i = 0; i < 256; i++) {
                ncclProfilerEventDescr_v5_t d;
                memset(&d, 0, sizeof(d));
                d.type = ncclProfileColl;
                d.coll.func = "AllReduce";
                d.coll.algo = "Ring";
                d.coll.proto = "Simple";
                d.coll.datatype = "ncclFloat32";
                d.coll.count = 1;
                d.coll.seqNumber = i;
                d.coll.nChannels = 1;
                ASSERT_EQ(acclPluginStartEvent(ctx, &handles[i], &d), 0);
                ASSERT_NE(handles[i], nullptr) << "Pool exhausted at i=" << i;
                ASSERT_EQ(acclPluginStopEvent(handles[i]), 0);
            }

            // Pool is full. The 257th allocation must return NULL (dropped).
            ncclProfilerEventDescr_v5_t d;
            memset(&d, 0, sizeof(d));
            d.type = ncclProfileColl;
            d.coll.func = "AllReduce";
            d.coll.algo = "Ring";
            d.coll.proto = "Simple";
            d.coll.datatype = "ncclFloat32";
            d.coll.count = 1;
            d.coll.seqNumber = 999;
            d.coll.nChannels = 1;
            void* h = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &h, &d), 0);
            EXPECT_EQ(h, nullptr)
                << "Full pool must return NULL, not evict live collectives";

            ASSERT_EQ(acclPluginFinalize(ctx), 0);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", "/tmp/accl_test_pool"}}
    );
}

// =========================================================================
// Coll stop before KernelCh: verify no corrupt/duplicate records
// =========================================================================
TEST(AcclProfilerLifecycle, CollStopBeforeAllChannels) {
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.CollStopBeforeAllChannels",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xFACE, &mask, "ordering_test",
                                     1, 2, 0, nullptr), 0);

            // Start coll with 2 channels
            ncclProfilerEventDescr_v5_t collDescr;
            memset(&collDescr, 0, sizeof(collDescr));
            collDescr.type = ncclProfileColl;
            collDescr.coll.func = "AllReduce";
            collDescr.coll.algo = "Ring";
            collDescr.coll.proto = "Simple";
            collDescr.coll.datatype = "ncclFloat32";
            collDescr.coll.count = 512;
            collDescr.coll.seqNumber = 42;
            collDescr.coll.nChannels = 2;

            void* collHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &collHandle, &collDescr), 0);

            // Start both KernelCh events
            void* kch0 = nullptr;
            void* kch1 = nullptr;
            ncclProfilerEventDescr_v5_t kd;
            memset(&kd, 0, sizeof(kd));
            kd.type = ncclProfileKernelCh;
            kd.parentObj = collHandle;

            kd.kernelCh.channelId = 0;
            kd.kernelCh.pTimer = 2000000;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kch0, &kd), 0);

            kd.kernelCh.channelId = 1;
            kd.kernelCh.pTimer = 2000100;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kch1, &kd), 0);

            // Record GPU stop timestamps
            ncclProfilerEventStateArgs_v5_t sa;
            memset(&sa, 0, sizeof(sa));
            sa.kernelCh.pTimer = 2010000;
            ASSERT_EQ(acclPluginRecordEventState(
                kch0, ncclProfilerKernelChStop, &sa), 0);
            sa.kernelCh.pTimer = 2010200;
            ASSERT_EQ(acclPluginRecordEventState(
                kch1, ncclProfilerKernelChStop, &sa), 0);

            // Stop Coll FIRST (before any KernelCh stop)
            ASSERT_EQ(acclPluginStopEvent(collHandle), 0);

            // Stop both channels — the last one should trigger finalization
            ASSERT_EQ(acclPluginStopEvent(kch0), 0);
            ASSERT_EQ(acclPluginStopEvent(kch1), 0);

            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            // Verify exactly one record with both channels
            char hostname[256] = {0};
            gethostname(hostname, sizeof(hostname) - 1);
            char path[1024];
            snprintf(path, sizeof(path),
                "/tmp/accl_test_ordering/accl_profiler_rank0_%s_pid%d_0xface.jsonl",
                hostname, (int)getpid());
            std::ifstream ifs(path);
            ASSERT_TRUE(ifs.good()) << "Output file not found: " << path;

            // Count coll records only. finalize() always appends a
            // {"summary":...} line, so a raw line count is not a record count.
            int lineCount = 0;
            std::string line;
            while (std::getline(ifs, line)) {
                if (line.find("\"coll_perf\"") != std::string::npos) lineCount++;
            }
            EXPECT_EQ(lineCount, 1)
                << "Expected exactly 1 record, got " << lineCount;

            // Re-read the single line to verify content
            ifs.clear();
            ifs.seekg(0);
            ASSERT_TRUE(std::getline(ifs, line));
            EXPECT_NE(line.find("\"coll_sn\":42"), std::string::npos);
            EXPECT_NE(line.find("\"coll_n_channels\":2"), std::string::npos);
            EXPECT_NE(line.find("\"coll_timing_source\":\"gpu_globaltimer\""),
                       std::string::npos);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", "/tmp/accl_test_ordering"}}
    );
}

// =========================================================================
// ProxyStep lifecycle: Coll → KernelCh → ProxyOp → ProxyStep → stop all
// =========================================================================
TEST(AcclProfilerLifecycle, FullProxyStepDecomposition) {
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.FullProxyStepDecomposition",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xA1B2, &mask, "proxystep_test",
                                     1, 2, 0, nullptr), 0);

            // Start Coll
            ncclProfilerEventDescr_v5_t collDescr;
            memset(&collDescr, 0, sizeof(collDescr));
            collDescr.type = ncclProfileColl;
            collDescr.coll.func = "AllReduce";
            collDescr.coll.algo = "Ring";
            collDescr.coll.proto = "Simple";
            collDescr.coll.datatype = "ncclFloat32";
            collDescr.coll.count = 256;
            collDescr.coll.seqNumber = 30;
            collDescr.coll.nChannels = 1;

            void* collHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &collHandle, &collDescr), 0);

            // Start KernelCh
            ncclProfilerEventDescr_v5_t kchDescr;
            memset(&kchDescr, 0, sizeof(kchDescr));
            kchDescr.type = ncclProfileKernelCh;
            kchDescr.parentObj = collHandle;
            kchDescr.kernelCh.channelId = 0;
            kchDescr.kernelCh.pTimer = 5000000;
            void* kchHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kchHandle, &kchDescr), 0);

            // Start ProxyOp
            ncclProfilerEventDescr_v5_t proxyDescr;
            memset(&proxyDescr, 0, sizeof(proxyDescr));
            proxyDescr.type = ncclProfileProxyOp;
            proxyDescr.parentObj = collHandle;
            proxyDescr.proxyOp.channelId = 0;
            proxyDescr.proxyOp.peer = 1;
            proxyDescr.proxyOp.nSteps = 1;
            proxyDescr.proxyOp.isSend = 1;
            void* proxyHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &proxyHandle, &proxyDescr), 0);

            // Start ProxyStep
            ncclProfilerEventDescr_v5_t stepDescr;
            memset(&stepDescr, 0, sizeof(stepDescr));
            stepDescr.type = ncclProfileProxyStep;
            stepDescr.parentObj = proxyHandle;
            stepDescr.proxyStep.step = 0;
            void* stepHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &stepHandle, &stepDescr), 0);

            // Simulate proxy step state transitions
            ASSERT_EQ(acclPluginRecordEventState(
                stepHandle,
                ncclProfilerProxyStepSendGPUWait,
                nullptr), 0);
            ASSERT_EQ(acclPluginRecordEventState(
                stepHandle,
                ncclProfilerProxyStepSendWait,
                nullptr), 0);

            // Stop in order: step -> coll -> kernel -> op (op last, so the
            // ProxyOp-stop path is what triggers finalization)
            ASSERT_EQ(acclPluginStopEvent(stepHandle), 0);
            ASSERT_EQ(acclPluginStopEvent(collHandle), 0);

            // Record GPU stop
            ncclProfilerEventStateArgs_v5_t sa;
            memset(&sa, 0, sizeof(sa));
            sa.kernelCh.pTimer = 5010000;
            ASSERT_EQ(acclPluginRecordEventState(
                kchHandle, ncclProfilerKernelChStop, &sa), 0);

            ASSERT_EQ(acclPluginStopEvent(kchHandle), 0);
            ASSERT_EQ(acclPluginStopEvent(proxyHandle), 0);

            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            char hostname[256] = {0};
            gethostname(hostname, sizeof(hostname) - 1);
            char path[1024];
            snprintf(path, sizeof(path),
                "/tmp/accl_test_proxystep/accl_profiler_rank0_%s_pid%d_0xa1b2.jsonl",
                hostname, (int)getpid());
            std::ifstream ifs(path);
            ASSERT_TRUE(ifs.good()) << "Output file not found: " << path;
            std::string line;
            ASSERT_TRUE(std::getline(ifs, line));
            EXPECT_NE(line.find("\"n_proxy_ops\":1"), std::string::npos);
            EXPECT_NE(line.find("\"n_send_ops\":1"), std::string::npos);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", "/tmp/accl_test_proxystep"}}
    );
}

// =========================================================================
// Channel bounds: absolute channelId above nChannels must be accepted
// (RCCL uses a plan-wide cursor, so the second collective in a grouped
// launch gets channelIds starting where the first left off)
// =========================================================================
TEST(AcclProfilerLifecycle, AbsoluteChannelIdAboveNChannelsAccepted) {
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.AbsoluteChannelIdAboveNChannelsAccepted",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xCB01, &mask, "chbound_test",
                                     1, 2, 0, nullptr), 0);

            ncclProfilerEventDescr_v5_t collDescr;
            memset(&collDescr, 0, sizeof(collDescr));
            collDescr.type = ncclProfileColl;
            collDescr.coll.func = "AllReduce";
            collDescr.coll.algo = "Ring";
            collDescr.coll.proto = "Simple";
            collDescr.coll.datatype = "ncclFloat32";
            collDescr.coll.count = 256;
            collDescr.coll.seqNumber = 50;
            collDescr.coll.nChannels = 2;

            void* collHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &collHandle, &collDescr), 0);
            ASSERT_NE(collHandle, nullptr);

            ncclProfilerEventDescr_v5_t kchDescr;
            memset(&kchDescr, 0, sizeof(kchDescr));
            kchDescr.type = ncclProfileKernelCh;
            kchDescr.parentObj = collHandle;

            // channelId=4 and 5 with nChannels=2: simulates the second
            // collective in a grouped launch where the first used channels 0-3.
            // These MUST be accepted — the guard is ACCL_MAX_CHANNELS, not nChannels.
            kchDescr.kernelCh.channelId = 4;
            kchDescr.kernelCh.pTimer = 1000000;
            void* kch4 = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kch4, &kchDescr), 0);
            ASSERT_NE(kch4, nullptr)
                << "channelId=4 with nChannels=2 must be accepted (absolute id)";

            kchDescr.kernelCh.channelId = 5;
            kchDescr.kernelCh.pTimer = 1000000;
            void* kch5 = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kch5, &kchDescr), 0);
            ASSERT_NE(kch5, nullptr)
                << "channelId=5 with nChannels=2 must be accepted (absolute id)";

            ASSERT_EQ(acclPluginStopEvent(collHandle), 0);

            ncclProfilerEventStateArgs_v5_t sa;
            memset(&sa, 0, sizeof(sa));
            sa.kernelCh.pTimer = 1010000;
            ASSERT_EQ(acclPluginRecordEventState(
                kch4, ncclProfilerKernelChStop, &sa), 0);
            ASSERT_EQ(acclPluginRecordEventState(
                kch5, ncclProfilerKernelChStop, &sa), 0);
            ASSERT_EQ(acclPluginStopEvent(kch4), 0);
            ASSERT_EQ(acclPluginStopEvent(kch5), 0);

            ASSERT_EQ(acclPluginFinalize(ctx), 0);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", "/tmp/accl_test_chbound"}}
    );
}

// =========================================================================
// Kernel timing assertions: verify gpu_kernel_avg/min/max_us in output
// =========================================================================
TEST(AcclProfilerLifecycle, KernelTimingFieldsPresent) {
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.KernelTimingFieldsPresent",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xD1D2, &mask, "ktime_test",
                                     1, 2, 0, nullptr), 0);

            ncclProfilerEventDescr_v5_t collDescr;
            memset(&collDescr, 0, sizeof(collDescr));
            collDescr.type = ncclProfileColl;
            collDescr.coll.func = "AllReduce";
            collDescr.coll.algo = "Ring";
            collDescr.coll.proto = "Simple";
            collDescr.coll.datatype = "ncclFloat32";
            collDescr.coll.count = 1024;
            collDescr.coll.seqNumber = 60;
            collDescr.coll.nChannels = 2;

            void* collHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &collHandle, &collDescr), 0);

            // Channel 0: 100 us (10000 ticks at 100 MHz)
            ncclProfilerEventDescr_v5_t kd;
            memset(&kd, 0, sizeof(kd));
            kd.type = ncclProfileKernelCh;
            kd.parentObj = collHandle;
            kd.kernelCh.channelId = 0;
            kd.kernelCh.pTimer = 1000000;
            void* kch0 = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kch0, &kd), 0);

            // Channel 1: 200 us (20000 ticks)
            kd.kernelCh.channelId = 1;
            kd.kernelCh.pTimer = 1000000;
            void* kch1 = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kch1, &kd), 0);

            ncclProfilerEventStateArgs_v5_t sa;
            memset(&sa, 0, sizeof(sa));
            sa.kernelCh.pTimer = 1010000;  // +10000 = 100 us
            ASSERT_EQ(acclPluginRecordEventState(
                kch0, ncclProfilerKernelChStop, &sa), 0);
            sa.kernelCh.pTimer = 1020000;  // +20000 = 200 us
            ASSERT_EQ(acclPluginRecordEventState(
                kch1, ncclProfilerKernelChStop, &sa), 0);

            ASSERT_EQ(acclPluginStopEvent(collHandle), 0);
            ASSERT_EQ(acclPluginStopEvent(kch0), 0);
            ASSERT_EQ(acclPluginStopEvent(kch1), 0);

            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            char hostname[256] = {0};
            gethostname(hostname, sizeof(hostname) - 1);
            char path[1024];
            snprintf(path, sizeof(path),
                "/tmp/accl_test_ktime/accl_profiler_rank0_%s_pid%d_0xd1d2.jsonl",
                hostname, (int)getpid());
            std::ifstream ifs(path);
            ASSERT_TRUE(ifs.good()) << "Output file not found: " << path;
            std::string line;
            ASSERT_TRUE(std::getline(ifs, line));
            EXPECT_NE(line.find("\"gpu_kernel_avg_us\":150.00"), std::string::npos)
                << "Expected avg of 100+200=150us";
            EXPECT_NE(line.find("\"gpu_kernel_min_us\":100.00"), std::string::npos);
            EXPECT_NE(line.find("\"gpu_kernel_max_us\":200.00"), std::string::npos);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", "/tmp/accl_test_ktime"}}
    );
}

// =========================================================================
// nChannels uint8_t wrap: RCCL's ncclTaskColl::nChannels is uint16_t but the
// v5 descriptor field is uint8_t, so a collective using MAXCHANNELS (256)
// channels is delivered to the plugin as nChannels == 0.
// =========================================================================
TEST(AcclProfilerNChannels, Wrapped256IsProfiledNotDropped) {
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerNChannels.Wrapped256IsProfiledNotDropped",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0x2601, &mask, "nch256_test",
                                     1, 2, 0, nullptr), 0);

            ncclProfilerEventDescr_v5_t d;
            MakeCollDescr(&d, /*nChannels=*/0, /*seqNumber=*/70, /*count=*/1024);
            void* coll = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &coll, &d), 0);
            ASSERT_NE(coll, nullptr)
                << "A collective reporting nChannels=0 must not be dropped — at "
                   "256 channels the uint8_t ABI field wraps to 0";

            // Drive all 256 channels: start every channel, then stop the coll,
            // then stop every channel (RCCL fires coll-stop right after
            // ncclProxyStart, while kernel events are still in flight).
            void* kch[256];
            for (int c = 0; c < 256; c++) {
                ncclProfilerEventDescr_v5_t kd;
                memset(&kd, 0, sizeof(kd));
                kd.type = ncclProfileKernelCh;
                kd.parentObj = coll;
                kd.kernelCh.channelId = (uint8_t)c;
                kd.kernelCh.pTimer = 1000000;
                ASSERT_EQ(acclPluginStartEvent(ctx, &kch[c], &kd), 0);
                ASSERT_NE(kch[c], nullptr) << "channel " << c;
            }
            ASSERT_EQ(acclPluginStopEvent(coll), 0);

            ncclProfilerEventStateArgs_v5_t sa;
            memset(&sa, 0, sizeof(sa));
            sa.kernelCh.pTimer = 1010000;  // +10000 ticks = 100 us at 100 MHz
            for (int c = 0; c < 256; c++) {
                ASSERT_EQ(acclPluginRecordEventState(
                    kch[c], ncclProfilerKernelChStop, &sa), 0);
                ASSERT_EQ(acclPluginStopEvent(kch[c]), 0);
            }
            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            std::string out =
                ReadProfilerOutput("/tmp/accl_test_nch256", "0x2601");
            ASSERT_FALSE(out.empty()) << "No profiler output produced";
            EXPECT_NE(out.find("\"coll_sn\":70"), std::string::npos)
                << "The 256-channel collective must produce a record";
            EXPECT_NE(out.find("\"coll_n_channels\":256"), std::string::npos)
                << "A reported 0 must be promoted to 256 as the completion target";
            EXPECT_NE(out.find("\"coll_n_channels_reported\":0"),
                      std::string::npos)
                << "The raw ABI value must stay visible so 0 is never ambiguous";
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", "/tmp/accl_test_nch256"}}
    );
}

// A wrapped (0 -> 256) collective whose channels do not all report must NOT
// finalize at coll-stop. Finalizing frees the pool slot and destroys its mutex
// while the proxy thread is still delivering KernelCh events into it, which is
// a use-after-free on a recycled slot. Leaking the slot is the safe direction.
TEST(AcclProfilerNChannels, Wrapped256DoesNotFinalizeEarly) {
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerNChannels.Wrapped256DoesNotFinalizeEarly",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0x2602, &mask, "nch_early_test",
                                     1, 2, 0, nullptr), 0);

            ncclProfilerEventDescr_v5_t d;
            MakeCollDescr(&d, /*nChannels=*/0, /*seqNumber=*/71, /*count=*/1024);
            void* coll = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &coll, &d), 0);
            ASSERT_NE(coll, nullptr);

            // Only channel 0 reports — the other 255 are skipped, as they are
            // when RCCL's profiler transport drains during teardown.
            ncclProfilerEventDescr_v5_t kd;
            memset(&kd, 0, sizeof(kd));
            kd.type = ncclProfileKernelCh;
            kd.parentObj = coll;
            kd.kernelCh.channelId = 0;
            kd.kernelCh.pTimer = 1000000;
            void* kch0 = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kch0, &kd), 0);

            ASSERT_EQ(acclPluginStopEvent(coll), 0);

            ncclProfilerEventStateArgs_v5_t sa;
            memset(&sa, 0, sizeof(sa));
            sa.kernelCh.pTimer = 1010000;
            ASSERT_EQ(acclPluginRecordEventState(
                kch0, ncclProfilerKernelChStop, &sa), 0);
            ASSERT_EQ(acclPluginStopEvent(kch0), 0);

            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            std::string out =
                ReadProfilerOutput("/tmp/accl_test_nch_early", "0x2602");
            ASSERT_FALSE(out.empty()) << "Summary line should still be written";
            EXPECT_EQ(out.find("\"coll_perf\""), std::string::npos)
                << "1 of 256 channels reported: the collective must not be "
                   "finalized, because its slot is still live for RCCL";
            EXPECT_NE(out.find("\"leaked_collectives\":1"), std::string::npos)
                << "The unfinalized slot must be counted as leaked, not hidden";
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", "/tmp/accl_test_nch_early"}}
    );
}

// =========================================================================
// End-of-run summary: data loss must be visible in the output itself
// =========================================================================
TEST(AcclProfilerSummary, CleanRunReportsComplete) {
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerSummary.CleanRunReportsComplete",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0x5101, &mask, "sum_clean_test",
                                     1, 2, 0, nullptr), 0);

            ncclProfilerEventDescr_v5_t d;
            MakeCollDescr(&d, /*nChannels=*/1, /*seqNumber=*/80, /*count=*/256);
            void* coll = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &coll, &d), 0);

            ncclProfilerEventDescr_v5_t kd;
            memset(&kd, 0, sizeof(kd));
            kd.type = ncclProfileKernelCh;
            kd.parentObj = coll;
            kd.kernelCh.channelId = 0;
            kd.kernelCh.pTimer = 1000000;
            void* kch0 = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kch0, &kd), 0);

            ASSERT_EQ(acclPluginStopEvent(coll), 0);
            ncclProfilerEventStateArgs_v5_t sa;
            memset(&sa, 0, sizeof(sa));
            sa.kernelCh.pTimer = 1010000;
            ASSERT_EQ(acclPluginRecordEventState(
                kch0, ncclProfilerKernelChStop, &sa), 0);
            ASSERT_EQ(acclPluginStopEvent(kch0), 0);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            std::string out =
                ReadProfilerOutput("/tmp/accl_test_sum_clean", "0x5101");
            ASSERT_FALSE(out.empty());
            // Emitted even when nothing was lost: a missing summary must mean
            // "the run died before finalize", never "the run was clean".
            EXPECT_NE(out.find("\"complete\":true"), std::string::npos);
            EXPECT_NE(out.find("\"dropped_collectives\":0"), std::string::npos);
            EXPECT_NE(out.find("\"leaked_collectives\":0"), std::string::npos);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", "/tmp/accl_test_sum_clean"}}
    );
}

TEST(AcclProfilerSummary, LeakedSlotsAreCounted) {
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerSummary.LeakedSlotsAreCounted",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0x5102, &mask, "sum_leak_test",
                                     1, 1, 0, nullptr), 0);

            // Three collectives that stop but never report a kernel channel,
            // so acclShouldFinalize never fires and each slot stays pinned.
            for (int i = 0; i < 3; i++) {
                ncclProfilerEventDescr_v5_t d;
                MakeCollDescr(&d, /*nChannels=*/1, /*seqNumber=*/(uint64_t)i,
                              /*count=*/64);
                void* coll = nullptr;
                ASSERT_EQ(acclPluginStartEvent(ctx, &coll, &d), 0);
                ASSERT_NE(coll, nullptr);
                ASSERT_EQ(acclPluginStopEvent(coll), 0);
            }
            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            std::string out =
                ReadProfilerOutput("/tmp/accl_test_sum_leak", "0x5102");
            ASSERT_FALSE(out.empty());
            EXPECT_NE(out.find("\"leaked_collectives\":3"), std::string::npos)
                << "Every slot the drain reclaims must be counted";
            EXPECT_NE(out.find("\"complete\":false"), std::string::npos);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", "/tmp/accl_test_sum_leak"}}
    );
}

TEST(AcclProfilerSummary, PoolExhaustionIsCounted) {
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerSummary.PoolExhaustionIsCounted",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0x5103, &mask, "sum_pool_test",
                                     1, 1, 0, nullptr), 0);

            // Pin all 256 slots, then attempt two more allocations.
            for (int i = 0; i < 256; i++) {
                ncclProfilerEventDescr_v5_t d;
                MakeCollDescr(&d, /*nChannels=*/1, /*seqNumber=*/(uint64_t)i,
                              /*count=*/1);
                void* coll = nullptr;
                ASSERT_EQ(acclPluginStartEvent(ctx, &coll, &d), 0);
                ASSERT_NE(coll, nullptr) << "Pool exhausted early at i=" << i;
                ASSERT_EQ(acclPluginStopEvent(coll), 0);
            }
            for (int i = 0; i < 2; i++) {
                ncclProfilerEventDescr_v5_t d;
                MakeCollDescr(&d, /*nChannels=*/1, /*seqNumber=*/900 + i,
                              /*count=*/1);
                void* coll = nullptr;
                ASSERT_EQ(acclPluginStartEvent(ctx, &coll, &d), 0);
                EXPECT_EQ(coll, nullptr) << "Full pool must return NULL";
            }
            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            std::string out =
                ReadProfilerOutput("/tmp/accl_test_sum_pool", "0x5103");
            ASSERT_FALSE(out.empty());
            EXPECT_NE(out.find("\"dropped_collectives\":2"), std::string::npos)
                << "Both rejected allocations must be counted";
            EXPECT_NE(out.find("\"leaked_collectives\":256"), std::string::npos)
                << "The pinned slots the drain reclaims are leaks, not drops";
            EXPECT_NE(out.find("\"complete\":false"), std::string::npos);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", "/tmp/accl_test_sum_pool"}}
    );
}

} // namespace RcclUnitTesting
