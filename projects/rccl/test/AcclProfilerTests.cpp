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

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <unistd.h>

#include <gtest/gtest.h>

#include "accl_profiler.h"
#include "accl_shim.h"
#include "common/ProcessIsolatedTestRunner.hpp"

// Forward declarations — thin wrappers defined in accl_profiler_wrapper.cc
extern "C" {
  int test_acclDatatypeSize(const char* dt);
  double test_acclBusBwFactor(const char* func, int nRanks);
  int  test_acclRefCount(void* ctx);
  void test_acclMarkFinalized(void* coll);
  void test_acclFreeColl(void* ctx, void* coll);
  int  test_acclProxyOpSlot(void* ctx, void* op);
  int  test_acclProxyOpMutexDestroyed(void* ctx, int slot);
  void test_acclWriteDummyRecord(void* ctx);
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

// Per-test profiler output directory, unique to this process.
//
// The plugin creates ACCL_PROFILER_OUTPUT_DIR itself (mkdir 0755) and never
// removes it, so a fixed path such as /tmp/accl_test_lifecycle ends up owned by
// whichever user ran the suite first; every later user's fopen() then fails and
// the tests that read their own output fail for unrelated reasons.  mkdtemp()
// gives each run its own directory, which teardown removes recursively.
//
// RUN_ISOLATED_TEST_WITH_ENV fork+execv's a fresh copy of this binary, so the
// child re-runs the whole TEST() body.  The child must therefore adopt the
// directory the parent passed down in the environment instead of creating a
// second one, or the reader and the plugin would disagree on the path.
class ScopedProfilerDir {
 public:
  explicit ScopedProfilerDir(const char* tag) {
    if (getenv(ProcessIsolatedTestRunner::kReexecMarkerEnvVar) != nullptr) {
      const char* inherited = getenv("ACCL_PROFILER_OUTPUT_DIR");
      if (inherited) path_ = inherited;
      return;
    }
    std::string tmpl = std::string("/tmp/accl_test_") + tag + "_XXXXXX";
    std::vector<char> buf(tmpl.c_str(), tmpl.c_str() + tmpl.size() + 1);
    const char* made = mkdtemp(buf.data());
    EXPECT_NE(made, nullptr)
        << "mkdtemp(" << tmpl << ") failed: " << strerror(errno);
    if (made) {
      path_ = made;
      owner_ = true;
    }
  }

  ScopedProfilerDir(const ScopedProfilerDir&) = delete;
  ScopedProfilerDir& operator=(const ScopedProfilerDir&) = delete;

  // Runs only in the parent, and only after executeAllTests() has reaped the
  // child, so no writer can still be holding a file open in here.
  ~ScopedProfilerDir() {
    if (!owner_) return;
    std::error_code ec;
    // Recursive: the plugin writes one .jsonl per rank into the directory.
    std::filesystem::remove_all(path_, ec);
  }

  const std::string& path() const { return path_; }
  const char* c_str() const { return path_.c_str(); }

 private:
  std::string path_;
  bool owner_ = false;
};

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
    ScopedProfilerDir dir("init");
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
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

// =========================================================================
// Full lifecycle: Coll → KernelCh → stop → finalize → check output JSONL
// =========================================================================
TEST(AcclProfilerLifecycle, CollWithKernelChProducesOutput) {
    ScopedProfilerDir dir("lifecycle");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.CollWithKernelChProducesOutput",
        [&dir]() {
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
                "%s/accl_profiler_rank0_%s_pid%d_0xbeef.jsonl",
                dir.c_str(), hostname, (int)getpid());
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
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

// =========================================================================
// ProxyOp refcount: verify proxy ops don't cause use-after-free
// =========================================================================
TEST(AcclProfilerLifecycle, ProxyOpAfterKernelStopIsValid) {
    ScopedProfilerDir dir("proxy");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.ProxyOpAfterKernelStopIsValid",
        [&dir]() {
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
                "%s/accl_profiler_rank0_%s_pid%d_0xcafe.jsonl",
                dir.c_str(), hostname, (int)getpid());
            std::ifstream ifs(path);
            ASSERT_TRUE(ifs.good()) << "Output file not found: " << path;
            std::string line;
            ASSERT_TRUE(std::getline(ifs, line));
            EXPECT_NE(line.find("\"n_proxy_ops\":1"), std::string::npos)
                << "Record must carry the proxy op that stopped after the kernel";
            EXPECT_NE(line.find("\"n_send_ops\":1"), std::string::npos);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

// =========================================================================
// Pool exhaustion: full pool returns NULL and drops the collective
// =========================================================================
TEST(AcclProfilerLifecycle, PoolFullReturnsNull) {
    ScopedProfilerDir dir("pool");
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
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

// =========================================================================
// Coll stop before KernelCh: verify no corrupt/duplicate records
// =========================================================================
TEST(AcclProfilerLifecycle, CollStopBeforeAllChannels) {
    ScopedProfilerDir dir("ordering");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.CollStopBeforeAllChannels",
        [&dir]() {
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
                "%s/accl_profiler_rank0_%s_pid%d_0xface.jsonl",
                dir.c_str(), hostname, (int)getpid());
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
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

// =========================================================================
// ProxyStep lifecycle: Coll → KernelCh → ProxyOp → ProxyStep → stop all
// =========================================================================
TEST(AcclProfilerLifecycle, FullProxyStepDecomposition) {
    ScopedProfilerDir dir("proxystep");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.FullProxyStepDecomposition",
        [&dir]() {
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
                "%s/accl_profiler_rank0_%s_pid%d_0xa1b2.jsonl",
                dir.c_str(), hostname, (int)getpid());
            std::ifstream ifs(path);
            ASSERT_TRUE(ifs.good()) << "Output file not found: " << path;
            std::string line;
            ASSERT_TRUE(std::getline(ifs, line));
            EXPECT_NE(line.find("\"n_proxy_ops\":1"), std::string::npos);
            EXPECT_NE(line.find("\"n_send_ops\":1"), std::string::npos);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

// =========================================================================
// Channel bounds: absolute channelId above nChannels must be accepted
// (RCCL uses a plan-wide cursor, so the second collective in a grouped
// launch gets channelIds starting where the first left off)
// =========================================================================
TEST(AcclProfilerLifecycle, AbsoluteChannelIdAboveNChannelsAccepted) {
    ScopedProfilerDir dir("chbound");
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
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

// =========================================================================
// Kernel timing assertions: verify gpu_kernel_avg/min/max_us in output
// =========================================================================
TEST(AcclProfilerLifecycle, KernelTimingFieldsPresent) {
    ScopedProfilerDir dir("ktime");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.KernelTimingFieldsPresent",
        [&dir]() {
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
                "%s/accl_profiler_rank0_%s_pid%d_0xd1d2.jsonl",
                dir.c_str(), hostname, (int)getpid());
            std::ifstream ifs(path);
            ASSERT_TRUE(ifs.good()) << "Output file not found: " << path;
            std::string line;
            ASSERT_TRUE(std::getline(ifs, line));
            EXPECT_NE(line.find("\"gpu_kernel_avg_us\":150.00"), std::string::npos)
                << "Expected avg of 100+200=150us";
            EXPECT_NE(line.find("\"gpu_kernel_min_us\":100.00"), std::string::npos);
            EXPECT_NE(line.find("\"gpu_kernel_max_us\":200.00"), std::string::npos);
            // event_trace_ts is part of the documented output contract (README.md):
            // one entry per channel, in channel order, with the raw pTimer stamps.
            EXPECT_NE(line.find(
                "\"event_trace_ts\":{\"kernel_events\":["
                "{\"channel_id\":0,\"kernel_start_ts\":1000000,"
                "\"kernel_stop_ts\":1010000,\"duration_us\":100},"
                "{\"channel_id\":1,\"kernel_start_ts\":1000000,"
                "\"kernel_stop_ts\":1020000,\"duration_us\":200}]}"),
                std::string::npos)
                << "event_trace_ts kernel_events array missing or malformed in: "
                << line;
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

// =========================================================================
// nChannels uint8_t wrap: RCCL's ncclTaskColl::nChannels is uint16_t but the
// v5 descriptor field is uint8_t, so a collective using MAXCHANNELS (256)
// channels is delivered to the plugin as nChannels == 0.
// =========================================================================
TEST(AcclProfilerNChannels, Wrapped256IsProfiledNotDropped) {
    ScopedProfilerDir dir("nch256");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerNChannels.Wrapped256IsProfiledNotDropped",
        [&dir]() {
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
                ReadProfilerOutput(dir.c_str(), "0x2601");
            ASSERT_FALSE(out.empty()) << "No profiler output produced";
            EXPECT_NE(out.find("\"coll_sn\":70"), std::string::npos)
                << "The 256-channel collective must produce a record";
            EXPECT_NE(out.find("\"coll_n_channels\":256"), std::string::npos)
                << "A reported 0 must be promoted to 256 as the completion target";
            EXPECT_NE(out.find("\"coll_n_channels_reported\":0"),
                      std::string::npos)
                << "The raw ABI value must stay visible so 0 is never ambiguous";
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

// A wrapped (0 -> 256) collective whose channels do not all report must NOT
// finalize at coll-stop. Finalizing frees the pool slot and destroys its mutex
// while the proxy thread is still delivering KernelCh events into it, which is
// a use-after-free on a recycled slot. Leaking the slot is the safe direction.
TEST(AcclProfilerNChannels, Wrapped256DoesNotFinalizeEarly) {
    ScopedProfilerDir dir("nch_early");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerNChannels.Wrapped256DoesNotFinalizeEarly",
        [&dir]() {
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
                ReadProfilerOutput(dir.c_str(), "0x2602");
            ASSERT_FALSE(out.empty()) << "Summary line should still be written";
            EXPECT_EQ(out.find("\"coll_perf\""), std::string::npos)
                << "1 of 256 channels reported: the collective must not be "
                   "finalized, because its slot is still live for RCCL";
            EXPECT_NE(out.find("\"leaked_collectives\":1"), std::string::npos)
                << "The unfinalized slot must be counted as leaked, not hidden";
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

// =========================================================================
// End-of-run summary: data loss must be visible in the output itself
// =========================================================================
TEST(AcclProfilerSummary, CleanRunReportsComplete) {
    ScopedProfilerDir dir("sum_clean");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerSummary.CleanRunReportsComplete",
        [&dir]() {
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
                ReadProfilerOutput(dir.c_str(), "0x5101");
            ASSERT_FALSE(out.empty());
            // Emitted even when nothing was lost: a missing summary must mean
            // "the run died before finalize", never "the run was clean".
            EXPECT_NE(out.find("\"complete\":true"), std::string::npos);
            EXPECT_NE(out.find("\"dropped_collectives\":0"), std::string::npos);
            EXPECT_NE(out.find("\"leaked_collectives\":0"), std::string::npos);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

TEST(AcclProfilerSummary, LeakedSlotsAreCounted) {
    ScopedProfilerDir dir("sum_leak");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerSummary.LeakedSlotsAreCounted",
        [&dir]() {
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
                ReadProfilerOutput(dir.c_str(), "0x5102");
            ASSERT_FALSE(out.empty());
            EXPECT_NE(out.find("\"leaked_collectives\":3"), std::string::npos)
                << "Every slot the drain reclaims must be counted";
            EXPECT_NE(out.find("\"complete\":false"), std::string::npos);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

TEST(AcclProfilerSummary, PoolExhaustionIsCounted) {
    ScopedProfilerDir dir("sum_pool");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerSummary.PoolExhaustionIsCounted",
        [&dir]() {
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
                ReadProfilerOutput(dir.c_str(), "0x5103");
            ASSERT_FALSE(out.empty());
            EXPECT_NE(out.find("\"dropped_collectives\":2"), std::string::npos)
                << "Both rejected allocations must be counted";
            EXPECT_NE(out.find("\"leaked_collectives\":256"), std::string::npos)
                << "The pinned slots the drain reclaims are leaks, not drops";
            EXPECT_NE(out.find("\"complete\":false"), std::string::npos);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}


// =========================================================================
// Context lifetime: a ProxyOp/ProxyStep handle must keep the context alive
//
// acclAllocProxyOp and acclAllocProxyStep hand out pointers INTO
// acclCommContext (the pools are embedded by value) but take no reference on
// it, so acclPluginFinalize can free the context while those handles are still
// live. RCCL delivers proxy events from the proxy progress thread, which is
// shared across communicators (comm->sharedRes), so a split-comm teardown can
// finalize one context while events for it are still in flight.
//
// These are crash tests by construction: sizeof(acclCommContext) is ~4.4 MB,
// which is far above glibc's 128 KB mmap threshold, so free() munmaps the
// region and any later dereference of the handle is a hard SIGSEGV rather than
// a silent read of stale bytes. The process-isolated runner reports the dead
// child as a failure, so no sanitizer build is required to catch this.
// =========================================================================
TEST(AcclProfilerLifecycle, ProxyStepOutlivingFinalizeKeepsContextAlive) {
    ScopedProfilerDir dir("ctxref_step");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.ProxyStepOutlivingFinalizeKeepsContextAlive",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xC7F1, &mask, "ctxref_step",
                                     1, 1, 0, nullptr), 0);

            ncclProfilerEventDescr_v5_t cd;
            MakeCollDescr(&cd, /*nChannels=*/1, /*seqNumber=*/1, /*count=*/1024);
            void* collHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &collHandle, &cd), 0);
            ASSERT_NE(collHandle, nullptr);

            ncclProfilerEventDescr_v5_t od;
            memset(&od, 0, sizeof(od));
            od.type = ncclProfileProxyOp;
            od.parentObj = collHandle;
            od.proxyOp.channelId = 0;
            od.proxyOp.peer = 1;
            od.proxyOp.nSteps = 1;
            od.proxyOp.isSend = 1;
            void* opHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &opHandle, &od), 0);
            ASSERT_NE(opHandle, nullptr);

            ncclProfilerEventDescr_v5_t sd;
            memset(&sd, 0, sizeof(sd));
            sd.type = ncclProfileProxyStep;
            sd.parentObj = opHandle;
            sd.proxyStep.step = 0;
            void* stepHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &stepHandle, &sd), 0);
            ASSERT_NE(stepHandle, nullptr);

            // The proxy op never stops, so nProxyOpsCompleted stays below
            // nProxyOpsStarted and the collective cannot finalize here. Its slot
            // is still in use when finalize runs, which is the teardown-orphan
            // shape the drain exists to handle.
            ASSERT_EQ(acclPluginStopEvent(collHandle), 0);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            // stepHandle points into ctx->proxyStepPool. If finalize freed the
            // context, this dereferences unmapped memory and the child dies.
            EXPECT_EQ(acclPluginStopEvent(stepHandle), 0)
                << "ProxyStep stop after finalize must not touch a freed context";
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

TEST(AcclProfilerLifecycle, ProxyOpOutlivingFinalizeKeepsContextAlive) {
    ScopedProfilerDir dir("ctxref_op");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.ProxyOpOutlivingFinalizeKeepsContextAlive",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xC7F2, &mask, "ctxref_op",
                                     1, 1, 0, nullptr), 0);

            ncclProfilerEventDescr_v5_t cd;
            MakeCollDescr(&cd, /*nChannels=*/1, /*seqNumber=*/2, /*count=*/1024);
            void* collHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &collHandle, &cd), 0);
            ASSERT_NE(collHandle, nullptr);

            ncclProfilerEventDescr_v5_t od;
            memset(&od, 0, sizeof(od));
            od.type = ncclProfileProxyOp;
            od.parentObj = collHandle;
            od.proxyOp.channelId = 0;
            od.proxyOp.peer = 1;
            od.proxyOp.nSteps = 1;
            od.proxyOp.isSend = 1;
            void* opHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &opHandle, &od), 0);
            ASSERT_NE(opHandle, nullptr);

            ASSERT_EQ(acclPluginStopEvent(collHandle), 0);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            // opHandle points into ctx->proxyOpPool. Same contract as above.
            EXPECT_EQ(acclPluginStopEvent(opHandle), 0)
                << "ProxyOp stop after finalize must not touch a freed context";
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}


// =========================================================================
// acclPluginFinalize's drain must not release a slot whose owner has already
// claimed it.
//
// A collective whose owner has passed acclShouldFinalize (finalized = 1) but
// has not yet reached acclFreeColl still has collPoolUsed set. The drain then
// destroys its mutex and decrements refCount for a slot the owner is going to
// release itself. The owner's own decrement drives the count below zero, and
// because the drain's extra decrement already took it to zero,
// acclPluginFinalize frees the context out from under the still-running owner.
//
// The check is a crash test by construction: sizeof(acclCommContext) is ~4.4 MB,
// far above glibc's mmap threshold, so free() unmaps the region and the owner's
// next touch is a hard SIGSEGV. The process-isolated runner reports the dead
// child as a failure, so no sanitizer build is required.
// =========================================================================
TEST(AcclProfilerLifecycle, DrainLeavesSlotsAnOwnerAlreadyClaimed) {
    ScopedProfilerDir dir("drain_claim");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.DrainLeavesSlotsAnOwnerAlreadyClaimed",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0x4B01, &mask, "drain_claim",
                                     1, 1, 0, nullptr), 0);
            EXPECT_EQ(test_acclRefCount(ctx), 1) << "init holds one reference";

            ncclProfilerEventDescr_v5_t cd;
            MakeCollDescr(&cd, /*nChannels=*/1, /*seqNumber=*/1, /*count=*/1024);
            void* coll = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &coll, &cd), 0);
            ASSERT_NE(coll, nullptr);
            EXPECT_EQ(test_acclRefCount(ctx), 2) << "the live collective holds one";

            // The owner has passed acclShouldFinalize and is between
            // acclFinalizeCollective and acclFreeColl. Its slot is still in use.
            test_acclMarkFinalized(coll);

            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            // The drain must not have released a slot the owner still owns, so
            // the owner's reference must survive finalize. On the unpatched
            // plugin this reads a freed context.
            EXPECT_EQ(test_acclRefCount(ctx), 1)
                << "drain released a slot whose owner had already claimed it";

            // Now the owner finishes. This is its normal next step, and it must
            // not drive the count negative or touch a freed context.
            test_acclFreeColl(ctx, coll);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

// =========================================================================
// acclPluginFinalize must not close the output file out from under a writer.
//
// It fcloses ctx->outputFile with no outputMutex held, while acclWriteRecord
// checks that pointer under the mutex and then fprintf()s. A writer that
// passed the check writes into a closed FILE*.
//
// There is no deterministic single-threaded shape for this one, so it is a
// stress test: a writer loop against a concurrent finalize. It fails by killing
// the child (glibc faults inside vfprintf on the freed FILE*), which the
// isolated runner reports. Note ASan cannot see this directly because glibc is
// uninstrumented.
// =========================================================================
TEST(AcclProfilerLifecycle, FinalizeDoesNotCloseOutputUnderAWriter) {
    ScopedProfilerDir dir("fclose_race");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.FinalizeDoesNotCloseOutputUnderAWriter",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0x4A01, &mask, "fclose_race",
                                     1, 1, 0, nullptr), 0);

            // Model the only real writer: acclFinalizeAndFree running on the
            // proxy thread. Its collective holds a context reference and it has
            // already claimed the slot, so the context cannot be freed under it.
            // Without that reference the test would be measuring the context
            // free, not the file close.
            ncclProfilerEventDescr_v5_t cd;
            MakeCollDescr(&cd, /*nChannels=*/1, /*seqNumber=*/1, /*count=*/1024);
            void* coll = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &coll, &cd), 0);
            ASSERT_NE(coll, nullptr);
            test_acclMarkFinalized(coll);

            std::atomic<bool> go{false};
            std::atomic<long> writes{0};
            std::thread writer([&]() {
                while (!go.load(std::memory_order_acquire)) { }
                for (int i = 0; i < 200000; i++) {
                    test_acclWriteDummyRecord(ctx);
                    writes.fetch_add(1, std::memory_order_relaxed);
                }
            });
            go.store(true, std::memory_order_release);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            EXPECT_EQ(acclPluginFinalize(ctx), 0);
            writer.join();
            EXPECT_GT(writes.load(), 0) << "writer never ran; test proved nothing";

            // The owner releases last, which is what frees the context.
            test_acclFreeColl(ctx, coll);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}


// -------------------------------------------------------------------------
// Reads the run summary line back out of the emitted JSONL.
// -------------------------------------------------------------------------
static std::string ReadSummaryLine(const char* dir, const char* commHashHex) {
    char host[256] = {0};
    gethostname(host, sizeof(host) - 1);
    char path[1024];
    snprintf(path, sizeof(path), "%s/accl_profiler_rank0_%s_pid%d_%s.jsonl",
             dir, host, (int)getpid(), commHashHex);
    std::ifstream ifs(path);
    std::string line, summary;
    // Deliberately no break: keep overwriting so we end up holding the LAST
    // summary line in the file, which is the one finalize wrote.
    while (std::getline(ifs, line)) {
        if (line.find("\"summary\"") != std::string::npos) {
            summary = line;
        }
    }
    return summary;
}

// =========================================================================
// A run that lost proxy data must not report itself complete.
//
// The proxy-op and proxy-step pools drop silently when full, and a completed
// proxy op is discarded when its collective already holds ACCL_MAX_PROXY_OPS.
// None of the three affects the emitted records in any visible way: the
// decomposition simply understates the proxy side. The only place the loss can
// surface is the end-of-run summary, so each of these drives one loss channel
// and asserts the summary both counts it and clears "complete".
// =========================================================================
TEST(AcclProfilerSummary, ProxyOpPoolExhaustionIsCounted) {
    ScopedProfilerDir dir("op_pool");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerSummary.ProxyOpPoolExhaustionIsCounted",
        [&dir]() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0x7001, &mask, "op_pool",
                                     1, 1, 0, nullptr), 0);

            // Pin every proxy-op slot by starting ops and never stopping them.
            ncclProfilerEventDescr_v5_t od;
            memset(&od, 0, sizeof(od));
            od.type = ncclProfileProxyOp;
            od.proxyOp.nSteps = 1;
            od.proxyOp.isSend = 1;
            for (int i = 0; i < ACCL_PROXY_OP_POOL_SIZE; i++) {
                void* h = nullptr;
                ASSERT_EQ(acclPluginStartEvent(ctx, &h, &od), 0);
                ASSERT_NE(h, nullptr) << "pool exhausted early at i=" << i;
            }
            // The next three have nowhere to go and must be counted.
            for (int i = 0; i < 3; i++) {
                void* h = nullptr;
                ASSERT_EQ(acclPluginStartEvent(ctx, &h, &od), 0);
                EXPECT_EQ(h, nullptr) << "a full proxy-op pool must return NULL";
            }
            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            std::string s = ReadSummaryLine(dir.c_str(), "0x7001");
            ASSERT_FALSE(s.empty()) << "no summary line was written";
            EXPECT_NE(s.find("\"dropped_proxy_ops\":3"), std::string::npos) << s;
            EXPECT_NE(s.find("\"complete\":false"), std::string::npos)
                << "a run that dropped proxy ops reported itself complete: " << s;
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

TEST(AcclProfilerSummary, ProxyStepPoolExhaustionIsCounted) {
    ScopedProfilerDir dir("step_pool");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerSummary.ProxyStepPoolExhaustionIsCounted",
        [&dir]() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0x7002, &mask, "step_pool",
                                     1, 1, 0, nullptr), 0);

            ncclProfilerEventDescr_v5_t od;
            memset(&od, 0, sizeof(od));
            od.type = ncclProfileProxyOp;
            od.proxyOp.nSteps = 1;
            od.proxyOp.isSend = 1;
            void* op = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &op, &od), 0);
            ASSERT_NE(op, nullptr);

            ncclProfilerEventDescr_v5_t sd;
            memset(&sd, 0, sizeof(sd));
            sd.type = ncclProfileProxyStep;
            sd.parentObj = op;
            for (int i = 0; i < ACCL_PROXY_STEP_POOL_SIZE; i++) {
                void* h = nullptr;
                ASSERT_EQ(acclPluginStartEvent(ctx, &h, &sd), 0);
                ASSERT_NE(h, nullptr) << "pool exhausted early at i=" << i;
            }
            for (int i = 0; i < 2; i++) {
                void* h = nullptr;
                ASSERT_EQ(acclPluginStartEvent(ctx, &h, &sd), 0);
                EXPECT_EQ(h, nullptr) << "a full proxy-step pool must return NULL";
            }
            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            std::string s = ReadSummaryLine(dir.c_str(), "0x7002");
            ASSERT_FALSE(s.empty()) << "no summary line was written";
            EXPECT_NE(s.find("\"dropped_proxy_steps\":2"), std::string::npos) << s;
            EXPECT_NE(s.find("\"complete\":false"), std::string::npos)
                << "a run that dropped proxy steps reported itself complete: " << s;
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

TEST(AcclProfilerSummary, ProxyOpOverflowPerCollectiveIsCounted) {
    ScopedProfilerDir dir("op_overflow");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerSummary.ProxyOpOverflowPerCollectiveIsCounted",
        [&dir]() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0x7003, &mask, "op_overflow",
                                     1, 1, 0, nullptr), 0);

            ncclProfilerEventDescr_v5_t cd;
            MakeCollDescr(&cd, /*nChannels=*/1, /*seqNumber=*/1, /*count=*/1024);
            void* coll = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &coll, &cd), 0);
            ASSERT_NE(coll, nullptr);

            // Two more ops than the collective can record. They complete
            // normally; the plugin has nowhere to put the last two.
            ncclProfilerEventDescr_v5_t od;
            memset(&od, 0, sizeof(od));
            od.type = ncclProfileProxyOp;
            od.parentObj = coll;
            od.proxyOp.nSteps = 1;
            od.proxyOp.isSend = 1;
            for (int i = 0; i < ACCL_MAX_PROXY_OPS + 2; i++) {
                void* h = nullptr;
                ASSERT_EQ(acclPluginStartEvent(ctx, &h, &od), 0);
                ASSERT_NE(h, nullptr);
                ASSERT_EQ(acclPluginStopEvent(h), 0);
            }
            ASSERT_EQ(acclPluginStopEvent(coll), 0);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            std::string s = ReadSummaryLine(dir.c_str(), "0x7003");
            ASSERT_FALSE(s.empty()) << "no summary line was written";
            EXPECT_NE(s.find("\"overflow_proxy_ops\":2"), std::string::npos) << s;
            EXPECT_NE(s.find("\"complete\":false"), std::string::npos)
                << "a run that discarded proxy ops reported itself complete: " << s;
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

TEST(AcclProfilerSummary, CleanRunStillReportsComplete) {
    ScopedProfilerDir dir("clean_run");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerSummary.CleanRunStillReportsComplete",
        [&dir]() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0x7004, &mask, "clean_run",
                                     1, 1, 0, nullptr), 0);

            ncclProfilerEventDescr_v5_t cd;
            MakeCollDescr(&cd, /*nChannels=*/1, /*seqNumber=*/1, /*count=*/1024);
            void* coll = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &coll, &cd), 0);
            ASSERT_NE(coll, nullptr);

            ncclProfilerEventDescr_v5_t kd;
            memset(&kd, 0, sizeof(kd));
            kd.type = ncclProfileKernelCh;
            kd.parentObj = coll;
            kd.kernelCh.channelId = 0;
            kd.kernelCh.pTimer = 1000000;
            void* kch = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kch, &kd), 0);
            ASSERT_EQ(acclPluginStopEvent(coll), 0);
            ncclProfilerEventStateArgs_v5_t sa;
            memset(&sa, 0, sizeof(sa));
            sa.kernelCh.pTimer = 1010000;
            ASSERT_EQ(acclPluginRecordEventState(kch, ncclProfilerKernelChStop, &sa), 0);
            ASSERT_EQ(acclPluginStopEvent(kch), 0);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            std::string s = ReadSummaryLine(dir.c_str(), "0x7004");
            ASSERT_FALSE(s.empty()) << "no summary line was written";
            EXPECT_NE(s.find("\"complete\":true"), std::string::npos)
                << "a clean run must still report complete: " << s;
            EXPECT_NE(s.find("\"dropped_proxy_ops\":0"), std::string::npos) << s;
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

// =========================================================================
// A proxy op's mutex must outlive every tenancy of its pool slot.
//
// The ProxyStep stop path locks step->parentObj->mutex, and a step handle can
// still be live after its parent op's slot has been released — the plugin frees
// ops from three places that a step knows nothing about. Destroying the mutex
// per free therefore leaves that lock aimed at a destroyed mutex as soon as the
// slot is reissued, and the reissued op's accumulators take the stray step's
// timings.
//
// The invariant is checked directly: glibc marks a destroyed mutex with
// __kind == -1, so a freed slot whose mutex reads -1 has been destroyed. Both
// halves matter — after the free and after the slot has been handed out again.
// =========================================================================
TEST(AcclProfilerLifecycle, ProxyOpMutexSurvivesSlotRelease) {
    ScopedProfilerDir dir("opmutex");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.ProxyOpMutexSurvivesSlotRelease",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0x4B02, &mask, "opmutex",
                                     1, 1, 0, nullptr), 0);

            // A proxy op with no parent collective is freed by its own stop
            // event, which is the shortest path through acclFreeProxyOp.
            ncclProfilerEventDescr_v5_t od;
            memset(&od, 0, sizeof(od));
            od.type = ncclProfileProxyOp;
            od.proxyOp.channelId = 0;
            od.proxyOp.nSteps = 2;
            od.proxyOp.isSend = 1;
            void* op = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &op, &od), 0);
            ASSERT_NE(op, nullptr);
            int slot = test_acclProxyOpSlot(ctx, op);
            ASSERT_GE(slot, 0);
            EXPECT_EQ(test_acclProxyOpMutexDestroyed(ctx, slot), 0)
                << "a live proxy op must have a live mutex";

            // A step that is still outstanding when the op is released. This is
            // the handle that later reaches pthread_mutex_lock(&op->mutex).
            ncclProfilerEventDescr_v5_t sd;
            memset(&sd, 0, sizeof(sd));
            sd.type = ncclProfileProxyStep;
            sd.parentObj = op;
            sd.proxyStep.step = 0;
            void* step = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &step, &sd), 0);
            ASSERT_NE(step, nullptr);

            ASSERT_EQ(acclPluginStopEvent(op), 0);   // releases the slot
            EXPECT_EQ(test_acclProxyOpMutexDestroyed(ctx, slot), 0)
                << "acclFreeProxyOp destroyed the slot mutex; the outstanding "
                   "ProxyStep stop now locks a destroyed mutex";

            // Reissue the slot, then let the stale step stop. The mutex must
            // still be live for the new tenant too.
            void* op2 = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &op2, &od), 0);
            ASSERT_EQ(test_acclProxyOpSlot(ctx, op2), slot)
                << "expected the freed slot to be handed out again";
            EXPECT_EQ(test_acclProxyOpMutexDestroyed(ctx, slot), 0)
                << "reissued proxy op slot has a destroyed mutex";

            ASSERT_EQ(acclPluginStopEvent(step), 0);
            ASSERT_EQ(acclPluginStopEvent(op2), 0);
            EXPECT_EQ(test_acclProxyOpMutexDestroyed(ctx, slot), 0);

            ASSERT_EQ(acclPluginFinalize(ctx), 0);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

} // namespace RcclUnitTesting
