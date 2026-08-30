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
  int  test_acclCollSlot(void* ctx, void* coll);
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

// Tests that point ACCL_PROFILER_OUTPUT_DIR *below* the mkdtemp root cannot
// recover that root from ScopedProfilerDir in the re-exec'd child, because there
// it adopts ACCL_PROFILER_OUTPUT_DIR — the derived path, not the root. They pass
// the root down in this variable instead.
static const char kProfilerRootEnvVar[] = "ACCL_TEST_PROFILER_ROOT";

// The mkdtemp root for the current process, parent or re-exec'd child.
static std::string ProfilerDirRoot(const ScopedProfilerDir& dir) {
    const char* inherited = getenv(kProfilerRootEnvVar);
    return inherited ? std::string(inherited) : dir.path();
}

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

// Returns the numeric value of "key":<number> in `json`, or -1 if absent.
static double JsonNumber(const std::string& json, const char* key) {
    std::string needle = std::string("\"") + key + "\":";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return -1;
    return strtod(json.c_str() + pos + needle.size(), nullptr);
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

// Returns the first collective record line from the plugin's JSONL, or "".
//
// Scans for "coll_perf" rather than taking line 1: acclPluginFinalize emits the
// {"summary":...} line unconditionally, so "the first line" is only a record
// while at least one collective finalized before it. Reading line 1 blind turns
// "this test produced no record" into a confusing mismatch on summary fields,
// and would silently pass a record assertion that happened to match the summary.
// Returning the line rather than the whole buffer keeps per-line assertions
// (front/back, and any substring) confined to the record they are about.
static std::string ReadCollRecord(const char* dir, const char* commHashHex) {
    std::stringstream ss(ReadProfilerOutput(dir, commHashHex));
    std::string line;
    while (std::getline(ss, line)) {
        if (line.find("\"coll_perf\"") != std::string::npos) return line;
    }
    return std::string();
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

// Fills a KernelCh event descriptor for one channel of `parent`.
static void MakeKernelChDescr(ncclProfilerEventDescr_v5_t* d, void* parent,
                              uint8_t channelId, uint64_t pTimer) {
    memset(d, 0, sizeof(*d));
    d->type = ncclProfileKernelCh;
    d->parentObj = parent;
    d->kernelCh.channelId = channelId;
    d->kernelCh.pTimer = pTimer;
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

// A multi-level ACCL_PROFILER_OUTPUT_DIR is what README.md documents, and a
// plain mkdir() only ever creates the last component: with two levels missing
// it fails ENOENT, fopen() then fails, and the whole run writes nothing while
// the activation mask still drives the full event stream into the plugin.
TEST(AcclProfilerInit, NestedOutputDirIsCreated) {
    ScopedProfilerDir dir("nesteddir");
    // The re-exec'd child's ScopedProfilerDir adopts ACCL_PROFILER_OUTPUT_DIR,
    // which by then is already the nested path, so deriving `nested` from it a
    // second time would nest twice. Carry the root in its own variable instead.
    const std::string root = ProfilerDirRoot(dir);
    // Two levels below the mkdtemp root: one mkdir() cannot reach this.
    const std::string nested = root + "/a/b";
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerInit.NestedOutputDirIsCreated",
        [&nested]() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xD112, &mask, "nested_dir_test",
                                     1, 1, 0, nullptr), 0);
            ASSERT_NE(ctx, nullptr);
            test_acclWriteDummyRecord(ctx);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            const std::string out = ReadProfilerOutput(nested.c_str(), "0xd112");
            ASSERT_FALSE(out.empty())
                << "no output file under " << nested
                << ": the plugin did not create the nested directory";
            EXPECT_NE(out.find("\"summary\""), std::string::npos);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", nested}, {kProfilerRootEnvVar, root}}
    );
}

// An ACCL_PROFILER_OUTPUT_DIR long enough to truncate the assembled path must
// be refused, not written to. Truncation cuts the rank/pid/hash suffix — and,
// once the directory alone exceeds the buffer, the trailing path component too
// — so the plugin used to silently create one mangled non-.jsonl file that
// accl_report.py's *.jsonl glob can never find.
TEST(AcclProfilerInit, OverlongOutputDirWritesNothing) {
    ScopedProfilerDir dir("longdir");
    const std::string root = ProfilerDirRoot(dir);
    // 1024 is sizeof(acclCommContext::outputPath); build past it out of
    // components each well under NAME_MAX so the directory itself is legal.
    std::string deep = root;
    for (int i = 0; deep.size() < 1100; i++) {
        deep += "/" + std::string(200, static_cast<char>('a' + i));
    }
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerInit.OverlongOutputDirWritesNothing",
        [&root, &deep]() {
            std::error_code ec;
            // Create it here, not via the plugin: the point under test is the
            // truncation, so the directory must already exist either way.
            std::filesystem::create_directories(deep, ec);
            ASSERT_FALSE(ec) << "could not create " << deep.size()
                             << "-char directory: " << ec.message();

            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xD1E7, &mask, "long_dir_test",
                                     1, 1, 0, nullptr), 0);
            ASSERT_NE(ctx, nullptr);
            test_acclWriteDummyRecord(ctx);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            // Nothing anywhere under the temp root: a truncated path can land
            // in any ancestor directory, not just the one that was requested.
            std::vector<std::string> created;
            for (const auto& e :
                 std::filesystem::recursive_directory_iterator(root, ec)) {
                if (e.is_regular_file()) {
                    created.push_back(e.path().filename().string());
                }
            }
            EXPECT_TRUE(created.empty())
                << "an overlong output dir still produced " << created.size()
                << " file(s), first: " << (created.empty() ? "" : created[0]);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", deep}, {kProfilerRootEnvVar, root}}
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
            MakeCollDescr(&collDescr, /*nChannels=*/1, /*seqNumber=*/10,
                          /*count=*/1024);

            void* collHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &collHandle, &collDescr), 0);
            ASSERT_NE(collHandle, nullptr);

            // Start a KernelCh event
            ncclProfilerEventDescr_v5_t kchDescr;
            MakeKernelChDescr(&kchDescr, collHandle, /*channelId=*/0,
                              /*pTimer=*/1000000);

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
            std::string line = ReadCollRecord(dir.c_str(), "0xbeef");
            ASSERT_FALSE(line.empty()) << "No coll record was written";
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
// ACCL_PROFILER_MIN_SIZE_BYTES: collectives below the threshold are dropped at
// start, and one exactly at the threshold is kept.
//
// Every other test leaves the variable unset, so gMinMsgSize is 0 and both the
// comparison and its inverse hold for every size those tests use — deleting the
// filter outright does not move them.  The equal case is a separate collective
// because the comparison is a strict `<`: a `<=` typo would still drop 4096 B
// and still keep 16384 B, so only the 8192 B collective can see it.
//
// gMinMsgSize is a process-global read once in acclPluginInit(); the isolated
// runner fork+execv's /proc/self/exe, so the child starts from the static
// initializer and re-reads the environment.
// =========================================================================
TEST(AcclProfilerMinSize, DropsBelowThresholdAndKeepsEqual) {
    ScopedProfilerDir dir("minsize");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerMinSize.DropsBelowThresholdAndKeepsEqual",
        [&dir]() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0x5127, &mask, "minsize_test",
                                     1, 2, 0, nullptr), 0);
            ASSERT_NE(ctx, nullptr);

            // Drives one single-channel collective to completion and returns the
            // coll handle, or nullptr if the size filter rejected it at start.
            auto runColl = [&](uint64_t seqNumber, size_t count) -> void* {
                ncclProfilerEventDescr_v5_t cd;
                MakeCollDescr(&cd, /*nChannels=*/1, seqNumber, count);
                void* coll = nullptr;
                EXPECT_EQ(acclPluginStartEvent(ctx, &coll, &cd), 0);
                if (!coll) return nullptr;

                ncclProfilerEventDescr_v5_t kd;
                MakeKernelChDescr(&kd, coll, /*channelId=*/0,
                                  /*pTimer=*/1000000);
                void* kch = nullptr;
                EXPECT_EQ(acclPluginStartEvent(ctx, &kch, &kd), 0);
                EXPECT_EQ(acclPluginStopEvent(coll), 0);
                ncclProfilerEventStateArgs_v5_t sa;
                memset(&sa, 0, sizeof(sa));
                sa.kernelCh.pTimer = 1010000;
                EXPECT_EQ(acclPluginRecordEventState(
                    kch, ncclProfilerKernelChStop, &sa), 0);
                EXPECT_EQ(acclPluginStopEvent(kch), 0);
                return coll;
            };

            // MakeCollDescr uses ncclFloat32, which acclDatatypeSize reports as
            // 4 bytes, so count scales by 4: 1024 -> 4096 B, 2048 -> 8192 B,
            // 4096 -> 16384 B against a threshold of 8192.
            EXPECT_EQ(runColl(/*seqNumber=*/80, /*count=*/1024), nullptr)
                << "4096 B is below ACCL_PROFILER_MIN_SIZE_BYTES=8192 and must "
                   "be rejected with a NULL handle";
            EXPECT_NE(runColl(/*seqNumber=*/81, /*count=*/2048), nullptr)
                << "8192 B is exactly the threshold and the comparison is a "
                   "strict `<`, so this collective must be profiled";
            EXPECT_NE(runColl(/*seqNumber=*/82, /*count=*/4096), nullptr)
                << "16384 B is above the threshold and must be profiled";

            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            std::string out = ReadProfilerOutput(dir.c_str(), "0x5127");
            ASSERT_FALSE(out.empty()) << "No profiler output produced";

            // Count coll records only: finalize() always appends a summary line.
            int records = 0;
            std::istringstream lines(out);
            std::string line;
            while (std::getline(lines, line)) {
                if (line.find("\"coll_perf\"") != std::string::npos) records++;
            }
            EXPECT_EQ(records, 2)
                << "Expected the two admitted collectives only, got " << records
                << ":\n" << out;

            EXPECT_EQ(out.find("\"coll_sn\":80"), std::string::npos)
                << "The filtered collective must not reach the JSONL: " << out;
            EXPECT_EQ(out.find("\"coll_msg_size_bytes\":4096"),
                      std::string::npos) << out;
            EXPECT_NE(out.find("\"coll_msg_size_bytes\":8192"),
                      std::string::npos)
                << "The at-threshold collective is missing: " << out;
            EXPECT_NE(out.find("\"coll_msg_size_bytes\":16384"),
                      std::string::npos)
                << "The above-threshold collective is missing: " << out;

            // A size-filtered collective is never allocated, so it must not be
            // counted as lost — the summary still describes a clean run.
            std::string s = ReadSummaryLine(dir.c_str(), "0x5127");
            ASSERT_FALSE(s.empty()) << "no summary line was written";
            EXPECT_NE(s.find("\"complete\":true"), std::string::npos)
                << "a size-filtered collective was miscounted as lost: " << s;
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()},
         {"ACCL_PROFILER_MIN_SIZE_BYTES", "8192"}}
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
            MakeCollDescr(&collDescr, /*nChannels=*/1, /*seqNumber=*/20,
                          /*count=*/256);

            void* collHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &collHandle, &collDescr), 0);

            // Start KernelCh
            ncclProfilerEventDescr_v5_t kchDescr;
            MakeKernelChDescr(&kchDescr, collHandle, /*channelId=*/0,
                              /*pTimer=*/0);

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
            std::string line = ReadCollRecord(dir.c_str(), "0xcafe");
            ASSERT_FALSE(line.empty()) << "No coll record was written";
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
                MakeCollDescr(&d, /*nChannels=*/1, /*seqNumber=*/(uint64_t)i,
                              /*count=*/1);
                ASSERT_EQ(acclPluginStartEvent(ctx, &handles[i], &d), 0);
                ASSERT_NE(handles[i], nullptr) << "Pool exhausted at i=" << i;
                ASSERT_EQ(acclPluginStopEvent(handles[i]), 0);
            }

            // Pool is full. The 257th allocation must return NULL (dropped).
            ncclProfilerEventDescr_v5_t d;
            MakeCollDescr(&d, /*nChannels=*/1, /*seqNumber=*/999, /*count=*/1);
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
            MakeCollDescr(&collDescr, /*nChannels=*/2, /*seqNumber=*/42,
                          /*count=*/512);

            void* collHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &collHandle, &collDescr), 0);

            // Start both KernelCh events
            void* kch0 = nullptr;
            void* kch1 = nullptr;
            ncclProfilerEventDescr_v5_t kd;
            MakeKernelChDescr(&kd, collHandle, /*channelId=*/0,
                              /*pTimer=*/2000000);
            ASSERT_EQ(acclPluginStartEvent(ctx, &kch0, &kd), 0);

            MakeKernelChDescr(&kd, collHandle, /*channelId=*/1,
                              /*pTimer=*/2000100);
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

            // Verify exactly one record with both channels.
            // Count coll records only. finalize() always appends a
            // {"summary":...} line, so a raw line count is not a record count.
            std::stringstream out(ReadProfilerOutput(dir.c_str(), "0xface"));
            int lineCount = 0;
            std::string line;
            while (std::getline(out, line)) {
                if (line.find("\"coll_perf\"") != std::string::npos) lineCount++;
            }
            EXPECT_EQ(lineCount, 1)
                << "Expected exactly 1 record, got " << lineCount;

            // Verify the content of that single record
            line = ReadCollRecord(dir.c_str(), "0xface");
            ASSERT_FALSE(line.empty()) << "No coll record was written";
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
            MakeCollDescr(&collDescr, /*nChannels=*/1, /*seqNumber=*/30,
                          /*count=*/256);

            void* collHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &collHandle, &collDescr), 0);

            // Start KernelCh
            ncclProfilerEventDescr_v5_t kchDescr;
            MakeKernelChDescr(&kchDescr, collHandle, /*channelId=*/0,
                              /*pTimer=*/5000000);
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

            std::string line = ReadCollRecord(dir.c_str(), "0xa1b2");
            ASSERT_FALSE(line.empty()) << "No coll record was written";
            EXPECT_NE(line.find("\"n_proxy_ops\":1"), std::string::npos);
            EXPECT_NE(line.find("\"n_send_ops\":1"), std::string::npos);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

// =========================================================================
// Proxy-step state attribution: RCCL announces a state on ENTRY, so the
// interval between two announcements belongs to the FIRST of the two.
// Charging it to the state being entered shifts every bucket by one, hiding
// the real GPU wait under proxy_peer_wait_us and dropping the trailing
// SendWait interval entirely.
// =========================================================================
TEST(AcclProfilerLifecycle, ProxyStepIntervalsChargedToStateBeingLeft) {
    ScopedProfilerDir dir("stateattr");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.ProxyStepIntervalsChargedToStateBeingLeft",
        [&dir]() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xC3D4, &mask, "stateattr_test",
                                     1, 2, 0, nullptr), 0);

            ncclProfilerEventDescr_v5_t collDescr;
            MakeCollDescr(&collDescr, 1, 31, 256);
            void* collHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &collHandle, &collDescr), 0);

            ncclProfilerEventDescr_v5_t kchDescr;
            MakeKernelChDescr(&kchDescr, collHandle, /*channelId=*/0,
                              /*pTimer=*/5000000);
            void* kchHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kchHandle, &kchDescr), 0);

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

            ncclProfilerEventDescr_v5_t stepDescr;
            memset(&stepDescr, 0, sizeof(stepDescr));
            stepDescr.type = ncclProfileProxyStep;
            stepDescr.parentObj = proxyHandle;
            stepDescr.proxyStep.step = 0;
            void* stepHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &stepHandle, &stepDescr), 0);

            // Mirror the send-side ordering in src/transport/net.cc: each state
            // is announced as the step enters it, so the long sleep below is
            // time spent waiting on the GPU, between SendGPUWait and
            // SendPeerWait.
            const int kLongUs = 40000;
            const int kShortUs = 2000;
            ASSERT_EQ(acclPluginRecordEventState(
                stepHandle, ncclProfilerProxyStepSendGPUWait, nullptr), 0);
            usleep(kLongUs);
            ASSERT_EQ(acclPluginRecordEventState(
                stepHandle, ncclProfilerProxyStepSendPeerWait_v4, nullptr), 0);
            usleep(kShortUs);
            ASSERT_EQ(acclPluginRecordEventState(
                stepHandle, ncclProfilerProxyStepSendWait, nullptr), 0);
            usleep(kShortUs);
            ASSERT_EQ(acclPluginStopEvent(stepHandle), 0);

            ASSERT_EQ(acclPluginStopEvent(collHandle), 0);
            ncclProfilerEventStateArgs_v5_t sa;
            memset(&sa, 0, sizeof(sa));
            sa.kernelCh.pTimer = 5010000;
            ASSERT_EQ(acclPluginRecordEventState(
                kchHandle, ncclProfilerKernelChStop, &sa), 0);
            ASSERT_EQ(acclPluginStopEvent(kchHandle), 0);
            ASSERT_EQ(acclPluginStopEvent(proxyHandle), 0);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            std::string out = ReadCollRecord(dir.c_str(), "0xc3d4");
            ASSERT_FALSE(out.empty())
                << "no collective record in " << dir.path();

            // One proxy op, so the per-op divisor is 1 and the JSON values are
            // the raw accumulated microseconds.
            const double half = kLongUs / 2.0;
            EXPECT_GT(JsonNumber(out, "proxy_gpu_wait_us"), half)
                << "the GPU wait must land in proxy_gpu_wait_us, not the next "
                   "bucket: " << out;
            EXPECT_LT(JsonNumber(out, "proxy_peer_wait_us"), half)
                << "proxy_peer_wait_us must not absorb the preceding GPU wait: "
                << out;
            // The interval after the last announced state is only recovered if
            // the stop path closes it.
            EXPECT_GT(JsonNumber(out, "proxy_network_us"), 0.0)
                << "the trailing SendWait interval must not be dropped: " << out;
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

// =========================================================================
// Proxy decomposition denominators
//
// A send op only ever passes through the send-side states and a recv op only
// through the recv-side ones, so averaging a one-sided total over n_proxy_ops
// scales it by that class's share of the op mix.  Drive a 4-send/4-recv
// collective with known per-op state durations and check each component lands
// on its per-class mean; dividing by n_proxy_ops would halve all four.
// =========================================================================

// Mirrors how src/transport/net.cc drives a step: announce the state on ENTRY,
// then spend the time in it. The interval is charged to the state being left,
// so each sleep below lands in the bucket named just above it, and the last
// sleep of each side is closed by the step stop.
//
// Send side: 4 ms SendGPUWait, 3 ms SendPeerWait, 1 ms SendWait.
// Recv side: 0.5 ms RecvWait, 5 ms RecvFlushWait, 2 ms RecvGPUWait.
static void DriveProxyOpWithTimings(void* ctx, void* coll, int isSend,
                                    void** opOut) {
    ncclProfilerEventDescr_v5_t od;
    memset(&od, 0, sizeof(od));
    od.type = ncclProfileProxyOp;
    od.parentObj = coll;
    od.proxyOp.channelId = 0;
    od.proxyOp.peer = 1;
    od.proxyOp.nSteps = 1;
    od.proxyOp.isSend = isSend;
    ASSERT_EQ(acclPluginStartEvent(ctx, opOut, &od), 0);

    ncclProfilerEventDescr_v5_t sd;
    memset(&sd, 0, sizeof(sd));
    sd.type = ncclProfileProxyStep;
    sd.parentObj = *opOut;
    sd.proxyStep.step = 0;
    void* step = nullptr;
    ASSERT_EQ(acclPluginStartEvent(ctx, &step, &sd), 0);

    if (isSend) {
        ASSERT_EQ(acclPluginRecordEventState(
            step, ncclProfilerProxyStepSendGPUWait, nullptr), 0);
        usleep(4000);
        ASSERT_EQ(acclPluginRecordEventState(
            step, ncclProfilerProxyStepSendPeerWait_v4, nullptr), 0);
        usleep(3000);
        ASSERT_EQ(acclPluginRecordEventState(
            step, ncclProfilerProxyStepSendWait, nullptr), 0);
        usleep(1000);
    } else {
        ASSERT_EQ(acclPluginRecordEventState(
            step, ncclProfilerProxyStepRecvWait, nullptr), 0);
        usleep(500);
        ASSERT_EQ(acclPluginRecordEventState(
            step, ncclProfilerProxyStepRecvFlushWait, nullptr), 0);
        usleep(5000);
        ASSERT_EQ(acclPluginRecordEventState(
            step, ncclProfilerProxyStepRecvGPUWait, nullptr), 0);
        usleep(2000);
    }
    ASSERT_EQ(acclPluginStopEvent(step), 0);
}

TEST(AcclProfilerLifecycle, ProxyComponentsAveragedOverTheirOwnOpClass) {
    ScopedProfilerDir dir("proxydenom");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.ProxyComponentsAveragedOverTheirOwnOpClass",
        [&dir]() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xD1D0, &mask, "denom_test",
                                     1, 8, 0, nullptr), 0);

            ncclProfilerEventDescr_v5_t cd;
            MakeCollDescr(&cd, 1, 31, 256);
            void* coll = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &coll, &cd), 0);

            ncclProfilerEventDescr_v5_t kd;
            MakeKernelChDescr(&kd, coll, /*channelId=*/0, /*pTimer=*/1000000);
            void* kch = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kch, &kd), 0);

            void* ops[8] = {nullptr};
            for (int i = 0; i < 8; i++) {
                DriveProxyOpWithTimings(ctx, coll, i < 4 ? 1 : 0, &ops[i]);
                ASSERT_NE(ops[i], nullptr);
            }

            ASSERT_EQ(acclPluginStopEvent(coll), 0);
            ncclProfilerEventStateArgs_v5_t sa;
            memset(&sa, 0, sizeof(sa));
            sa.kernelCh.pTimer = 1010000;
            ASSERT_EQ(acclPluginRecordEventState(
                kch, ncclProfilerKernelChStop, &sa), 0);
            ASSERT_EQ(acclPluginStopEvent(kch), 0);
            for (int i = 0; i < 8; i++) {
                ASSERT_EQ(acclPluginStopEvent(ops[i]), 0);
            }
            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            std::string line = ReadCollRecord(dir.c_str(), "0xd1d0");
            ASSERT_FALSE(line.empty());

            EXPECT_EQ(JsonNumber(line, "n_proxy_ops"), 8);
            EXPECT_EQ(JsonNumber(line, "n_send_ops"), 4);
            EXPECT_EQ(JsonNumber(line, "n_recv_ops"), 4);

            // usleep only guarantees a lower bound, so each window is
            // [nominal, 2x nominal): wide enough for scheduling jitter, tight
            // enough to exclude the halved /n_proxy_ops value.
            struct { const char* key; double lo, hi; } expect[] = {
                {"proxy_gpu_wait_us",      4000, 8000},
                {"proxy_peer_wait_us",     3000, 6000},
                {"proxy_flush_us",         5000, 10000},
                {"proxy_gpu_recv_wait_us", 2000, 4000},
                // One send op's network time plus one recv op's.
                {"proxy_network_us",       1500, 3000},
            };
            for (auto& e : expect) {
                double v = JsonNumber(line, e.key);
                EXPECT_GE(v, e.lo) << e.key
                    << " is below one op class's mean, so it was averaged over"
                       " every proxy op instead of over its own class";
                EXPECT_LT(v, e.hi) << e.key << " far above expected";
            }
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

// A send-only collective has n_recv_ops == 0. The recv-side numerators are 0
// too, so those fields must read 0 rather than NaN or inf.
TEST(AcclProfilerLifecycle, SendOnlyCollectiveReportsZeroRecvComponents) {
    ScopedProfilerDir dir("sendonly");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.SendOnlyCollectiveReportsZeroRecvComponents",
        [&dir]() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xD1D1, &mask, "sendonly_test",
                                     1, 8, 0, nullptr), 0);

            ncclProfilerEventDescr_v5_t cd;
            MakeCollDescr(&cd, 1, 32, 256);
            void* coll = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &coll, &cd), 0);

            ncclProfilerEventDescr_v5_t kd;
            MakeKernelChDescr(&kd, coll, /*channelId=*/0, /*pTimer=*/1000000);
            void* kch = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kch, &kd), 0);

            void* op = nullptr;
            DriveProxyOpWithTimings(ctx, coll, 1, &op);
            ASSERT_NE(op, nullptr);

            ASSERT_EQ(acclPluginStopEvent(coll), 0);
            ncclProfilerEventStateArgs_v5_t sa;
            memset(&sa, 0, sizeof(sa));
            sa.kernelCh.pTimer = 1010000;
            ASSERT_EQ(acclPluginRecordEventState(
                kch, ncclProfilerKernelChStop, &sa), 0);
            ASSERT_EQ(acclPluginStopEvent(kch), 0);
            ASSERT_EQ(acclPluginStopEvent(op), 0);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            std::string line = ReadCollRecord(dir.c_str(), "0xd1d1");
            ASSERT_FALSE(line.empty());

            EXPECT_EQ(JsonNumber(line, "n_recv_ops"), 0);
            EXPECT_EQ(line.find("nan"), std::string::npos)
                << "a zero op count must not reach a division: " << line;
            EXPECT_EQ(line.find("inf"), std::string::npos) << line;
            EXPECT_DOUBLE_EQ(JsonNumber(line, "proxy_flush_us"), 0.0);
            EXPECT_DOUBLE_EQ(JsonNumber(line, "proxy_gpu_recv_wait_us"), 0.0);
            EXPECT_GE(JsonNumber(line, "proxy_gpu_wait_us"), 4000);
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
            MakeCollDescr(&collDescr, /*nChannels=*/2, /*seqNumber=*/50,
                          /*count=*/256);

            void* collHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &collHandle, &collDescr), 0);
            ASSERT_NE(collHandle, nullptr);

            ncclProfilerEventDescr_v5_t kchDescr;

            // channelId=4 and 5 with nChannels=2: simulates the second
            // collective in a grouped launch where the first used channels 0-3.
            // These MUST be accepted — the guard is ACCL_MAX_CHANNELS, not nChannels.
            MakeKernelChDescr(&kchDescr, collHandle, /*channelId=*/4,
                              /*pTimer=*/1000000);
            void* kch4 = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kch4, &kchDescr), 0);
            ASSERT_NE(kch4, nullptr)
                << "channelId=4 with nChannels=2 must be accepted (absolute id)";

            MakeKernelChDescr(&kchDescr, collHandle, /*channelId=*/5,
                              /*pTimer=*/1000000);
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
            MakeCollDescr(&collDescr, /*nChannels=*/2, /*seqNumber=*/60,
                          /*count=*/1024);

            void* collHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &collHandle, &collDescr), 0);

            // Channel 0: 100 us (10000 ticks at 100 MHz)
            ncclProfilerEventDescr_v5_t kd;
            MakeKernelChDescr(&kd, collHandle, /*channelId=*/0,
                              /*pTimer=*/1000000);
            void* kch0 = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kch0, &kd), 0);

            // Channel 1: 200 us (20000 ticks)
            MakeKernelChDescr(&kd, collHandle, /*channelId=*/1,
                              /*pTimer=*/1000000);
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

            std::string line = ReadCollRecord(dir.c_str(), "0xd1d2");
            ASSERT_FALSE(line.empty()) << "No coll record was written";
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
                MakeKernelChDescr(&kd, coll, /*channelId=*/(uint8_t)c,
                                  /*pTimer=*/1000000);
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
            MakeKernelChDescr(&kd, coll, /*channelId=*/0, /*pTimer=*/1000000);
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
            MakeKernelChDescr(&kd, coll, /*channelId=*/0, /*pTimer=*/1000000);
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
            MakeKernelChDescr(&kd, coll, /*channelId=*/0, /*pTimer=*/1000000);
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

// =========================================================================
// Reusing a coll pool slot must not write the slot's mutex.
//
// acclAllocColl clears a reclaimed slot. The mutex lives inside that struct and
// is created once in acclPluginInit, so the clear has to step around it: a stale
// KernelCh start can be inside pthread_mutex_lock(&coll->mutex) at that instant,
// and that path does not take collPoolMutex. Saving the mutex, memsetting the
// struct and writing the saved bytes back restores the same values, but the
// intervening zero-and-restore is still a write racing a live lock — and POSIX
// does not define copying a pthread_mutex_t at all.
//
// The write is not visible as a value: the saved and restored bytes are the
// bytes that were already there, so no before/after comparison can separate
// fixed from unfixed. What is visible is the race itself, so the test runs the
// two paths against each other. glibc's lock fast path asserts on __owner, and a
// lock word zeroed mid-acquire trips it.
//
// The unfixed failure therefore arrives as an abort, or as the hang it can take
// instead — a waiter parked on a lock word that is then zeroed is never woken.
// The body runs in a forked child, so both are reported as a test failure rather
// than taking the whole binary down, and the config carries an explicit timeout
// because the runner otherwise waits forever and a regression would stall the
// suite instead of failing it.
//
// Measured: unfixed dies on every run in under 200 ms; fixed never writes the
// mutex bytes, so there is nothing left to race and the loop is unconditionally
// clean, finishing in about the same time.
// =========================================================================
static constexpr int kCollMutexTimeoutSeconds = 60;

TEST(AcclProfilerLifecycle, CollSlotReuseDoesNotWriteTheSlotMutex) {
    ScopedProfilerDir dir("collmutex");
    RUN_ISOLATED_TESTS(
        ProcessIsolatedTestRunner::TestConfig(
        "AcclProfilerLifecycle.CollSlotReuseDoesNotWriteTheSlotMutex",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0x4B03, &mask, "collmutex",
                                     1, 1, 0, nullptr), 0);

            ncclProfilerEventDescr_v5_t cd;
            MakeCollDescr(&cd, /*nChannels=*/1, /*seqNumber=*/0, /*count=*/1024);

            void* coll = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &coll, &cd), 0);
            ASSERT_NE(coll, nullptr);
            const int slot = test_acclCollSlot(ctx, coll);
            ASSERT_EQ(slot, 0) << "expected the first coll to take slot 0";

            // The stale handle. This is exactly what RCCL still holds when a
            // KernelCh event arrives after the coll's slot has been released:
            // an interior pointer into the pool, with no idea the slot moved on.
            std::atomic<bool> stop{false};
            std::atomic<unsigned long> locks{0};
            std::thread kernelCh([&]() {
                ncclProfilerEventDescr_v5_t kd;
                MakeKernelChDescr(&kd, coll, /*channelId=*/0, /*pTimer=*/0);
                while (!stop.load(std::memory_order_relaxed)) {
                    void* kh = nullptr;
                    // Reaches pthread_mutex_lock(&coll->mutex) whenever the slot
                    // currently reads as a live coll; the type check skips the
                    // instants when the reuse has the header zeroed.
                    if (acclPluginStartEvent(ctx, &kh, &kd) == 0 && kh) {
                        locks.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            });

            // Free and immediately re-take the same slot. Each iteration runs
            // acclAllocColl's clear while the thread above is locking.
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(20);
            for (int i = 0; i < 400000; i++) {
                test_acclFreeColl(ctx, coll);
                void* again = nullptr;
                ASSERT_EQ(acclPluginStartEvent(ctx, &again, &cd), 0);
                ASSERT_EQ(again, coll) << "expected slot " << slot << " back";
                if ((i & 0x3FF) == 0 &&
                    std::chrono::steady_clock::now() > deadline) {
                    break;
                }
            }

            stop.store(true, std::memory_order_relaxed);
            kernelCh.join();
            EXPECT_GT(locks.load(), 0u)
                << "the KernelCh thread never took coll->mutex, so nothing was "
                   "raced against the slot reuse and this test proved nothing";

            ASSERT_EQ(acclPluginFinalize(ctx), 0);
        })
        .withEnvironment({{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}})
        .withTimeout(std::chrono::seconds(kCollMutexTimeoutSeconds))
    );
}

} // namespace RcclUnitTesting
