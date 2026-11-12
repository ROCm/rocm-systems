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
#include <filesystem>
#include <fstream>
#include <map>
#include <vector>
#include <sstream>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <tuple>
#include <fcntl.h>
#include <unistd.h>
#include <linux/ethtool.h>
#include <linux/netlink.h>
#include <linux/sockios.h>
#include <net/if.h>
#include <rdma/rdma_netlink.h>

#include <gtest/gtest.h>

#include "accl_profiler.h"
#include "accl_shim.h"
#include "accl_boundary.h"
#include "net_telemetry.h"
#include "common/ProcessIsolatedTestRunner.hpp"

// Forward declarations — thin wrappers defined in accl_profiler_wrapper.cc
extern "C" {
  int test_acclDatatypeSize(const char* dt);
  double test_acclBusBwFactor(const char* func, int nRanks);
  size_t test_acclParseMinSize(const char* env, int* bad);
  int  test_acclRefCount(void* ctx);
  void test_acclMarkFinalized(void* coll);
  void test_acclFreeColl(void* ctx, void* coll);
  int  test_acclCollSlot(void* ctx, void* coll);
  int  test_acclProxyOpSlot(void* ctx, void* op);
  int  test_acclProxyOpMutexDestroyed(void* ctx, int slot);
  void test_acclWriteDummyRecord(void* ctx);
  int  test_acclOutputFd(void* ctx);
  unsigned long test_acclProxyOpGeneration(void* ctx, int slot);
  void test_acclNetSetHooks(const struct acclNetHooks* hooks);
  int  test_acclRdmaOpenAndRead(const char* name, uint64_t* first, uint64_t* second,
                                char names[][ACCL_NET_NAME_LEN], int* nNames);
  int  test_acclEthReads(const char* ifname, int nReads, uint64_t out[][ACCL_NET_MAX_ETH],
                         char names[][ETH_GSTRING_LEN + 1], int* nNames);
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

// The path the plugin writes for `commHashHex` under `dir`. Must be called
// inside the isolated child, since the filename embeds its pid.
static std::string ProfilerOutputPath(const char* dir, const char* commHashHex) {
    char hostname[256] = {0};
    gethostname(hostname, sizeof(hostname) - 1);
    char path[1024];
    snprintf(path, sizeof(path), "%s/accl_profiler_rank0_%s_pid%d_%s.jsonl",
             dir, hostname, (int)getpid(), commHashHex);
    return std::string(path);
}

// Number of currently-open descriptors pointing at `path`, via /proc/self/fd.
// Used to observe the FILE* itself rather than a proxy for it.
static int OpenFdCountFor(const std::string& path) {
    int n = 0;
    std::error_code ec;
    std::filesystem::directory_iterator it("/proc/self/fd", ec);
    if (ec) return -1;
    for (const auto& entry : it) {
        std::error_code lec;
        const auto target = std::filesystem::read_symlink(entry.path(), lec);
        if (!lec && target.string() == path) n++;
    }
    return n;
}

// Returns the plugin's whole JSONL output for `commHashHex`, or "" if absent.
static std::string ReadProfilerOutput(const char* dir, const char* commHashHex) {
    std::ifstream ifs(ProfilerOutputPath(dir, commHashHex));
    if (!ifs.good()) return std::string();
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

// Returns the first per-collective record in `commHashHex`'s file, or "" if
// the run emitted none. finalize() always appends a summary line, so a
// non-empty file does not by itself mean a collective was profiled; select on
// the "coll_perf" key rather than taking the first line.
static std::string ReadCollRecord(const char* dir, const char* commHashHex) {
    std::istringstream lines(ReadProfilerOutput(dir, commHashHex));
    std::string line;
    while (std::getline(lines, line)) {
        if (line.find("\"coll_perf\"") != std::string::npos) return line;
    }
    return std::string();
}

// Returns the LAST summary line in `commHashHex`'s file -- the one finalize
// wrote -- or "" if absent.
static std::string ReadSummaryLine(const char* dir, const char* commHashHex) {
    std::istringstream lines(ReadProfilerOutput(dir, commHashHex));
    std::string line, summary;
    // Deliberately no break: keep overwriting so we end up holding the last.
    while (std::getline(lines, line)) {
        if (line.find("\"summary\"") != std::string::npos) summary = line;
    }
    return summary;
}

// Returns the numeric value of "key":<number> in `json`, or -1 if absent.
static double JsonNumber(const std::string& json, const char* key) {
    std::string needle = std::string("\"") + key + "\":";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return -1;
    return strtod(json.c_str() + pos + needle.size(), nullptr);
}

// Every loss counter the summary carries, in emission order.
static const char* const kSummaryCounters[] = {
    "dropped_collectives", "leaked_collectives", "inflight_collectives",
    "dropped_proxy_ops", "dropped_proxy_steps", "overflow_proxy_ops",
    "stale_proxy_steps", "outstanding_proxy_ops", "outstanding_proxy_steps"};

// Checks the whole counter vector, not just the one a test is named for:
// "complete":false is cleared by any non-zero counter, so a test that asserts
// only its own would still pass if a different channel had fired instead.
// Counters absent from `expected` must read 0.
static void ExpectSummaryCounters(
    const std::string& summary,
    const std::map<std::string, double>& expected) {
    for (const char* key : kSummaryCounters) {
        auto it = expected.find(key);
        const double want = (it == expected.end()) ? 0.0 : it->second;
        EXPECT_EQ(JsonNumber(summary, key), want)
            << key << " in summary: " << summary;
    }
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

// Fills a KernelCh event descriptor parented to `collHandle`.
static void MakeKernelChDescr(ncclProfilerEventDescr_v5_t* d, void* collHandle,
                              int channelId, uint64_t pTimer) {
    memset(d, 0, sizeof(*d));
    d->type = ncclProfileKernelCh;
    d->parentObj = collHandle;
    d->kernelCh.channelId = channelId;
    d->kernelCh.pTimer = pTimer;
}

// Fills a ProxyOp event descriptor parented to `collHandle`. Hand-built copies
// of this had already drifted apart on peer and channelId.
static void MakeProxyOpDescr(ncclProfilerEventDescr_v5_t* d, void* collHandle,
                             int channelId, int peer, int nSteps, int isSend) {
    memset(d, 0, sizeof(*d));
    d->type = ncclProfileProxyOp;
    d->parentObj = collHandle;
    d->proxyOp.channelId = channelId;
    d->proxyOp.peer = peer;
    d->proxyOp.nSteps = nSteps;
    d->proxyOp.isSend = isSend;
}

// Fills a ProxyStep event descriptor parented to `opHandle`.
static void MakeProxyStepDescr(ncclProfilerEventDescr_v5_t* d, void* opHandle,
                               int step) {
    memset(d, 0, sizeof(*d));
    d->type = ncclProfileProxyStep;
    d->parentObj = opHandle;
    d->proxyStep.step = step;
}

// Drives one single-channel collective to completion and returns its coll
// handle, or nullptr if the size filter rejected it at start.
static void* RunSingleChannelColl(void* ctx, uint64_t seqNumber, size_t count) {
    ncclProfilerEventDescr_v5_t cd;
    MakeCollDescr(&cd, /*nChannels=*/1, seqNumber, count);
    void* coll = nullptr;
    EXPECT_EQ(acclPluginStartEvent(ctx, &coll, &cd), 0);
    if (!coll) return nullptr;

    ncclProfilerEventDescr_v5_t kd;
    MakeKernelChDescr(&kd, coll, /*channelId=*/0, /*pTimer=*/1000000);
    void* kch = nullptr;
    EXPECT_EQ(acclPluginStartEvent(ctx, &kch, &kd), 0);
    EXPECT_EQ(acclPluginStopEvent(coll), 0);
    ncclProfilerEventStateArgs_v5_t sa;
    memset(&sa, 0, sizeof(sa));
    sa.kernelCh.pTimer = 1010000;
    EXPECT_EQ(acclPluginRecordEventState(kch, ncclProfilerKernelChStop, &sa), 0);
    EXPECT_EQ(acclPluginStopEvent(kch), 0);
    return coll;
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

// getenv() returns a non-NULL pointer for a set-but-empty variable, so a plain
// !outDir check left outDir == "" and every rank assembled
// "/accl_profiler_rank*.jsonl" at the filesystem root: EACCES unprivileged, and
// silently successful as root in a container. Both are quiet — ACCL_WARN is
// level WARN and the RCCL default with NCCL_DEBUG unset is ERROR — so an empty
// value has to fall back to the same /tmp the unset case uses.
TEST(AcclProfilerInit, EmptyOutputDirFallsBackToTmp) {
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerInit.EmptyOutputDirFallsBackToTmp",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xD113, &mask, "empty_dir_test",
                                     1, 1, 0, nullptr), 0);
            ASSERT_NE(ctx, nullptr);
            test_acclWriteDummyRecord(ctx);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            // "" as a directory makes ReadProfilerOutput look at "/...", which
            // is exactly where the unfixed plugin wrote.
            EXPECT_TRUE(ReadProfilerOutput("", "0xd113").empty())
                << "plugin wrote to the filesystem root";
            EXPECT_FALSE(ReadProfilerOutput("/tmp", "0xd113").empty())
                << "empty ACCL_PROFILER_OUTPUT_DIR did not fall back to /tmp";

            // No ScopedProfilerDir owns /tmp, so remove our own file. The name
            // carries this pid and the comm hash, so nothing else can match.
            std::error_code ec;
            std::filesystem::remove(ProfilerOutputPath("/tmp", "0xd113"), ec);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", ""}}
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
            const std::string path =
                ProfilerOutputPath(ProfilerDirRoot(dir).c_str(), "0xbeef");
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
// ACCL_PROFILER_MIN_SIZE_BYTES: collectives below the threshold are dropped at
// start, and one exactly at the threshold is kept.
//
// Every other test leaves the variable unset, so gMinMsgSize is 0 and both the
// comparison and its inverse hold for every size those tests use — deleting the
// filter outright does not move them.  The equal case is a separate collective
// because the comparison is a strict `<`: a `<=` typo would still drop 4096 B
// and still keep 16384 B, so only the 8192 B collective can see it.
//
// The threshold is read once per acclPluginInit() into that comm's context; the
// isolated runner fork+execv's /proc/self/exe, so the child re-reads the
// environment from scratch.
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

            auto runColl = [&](uint64_t seqNumber, size_t count) -> void* {
                return RunSingleChannelColl(ctx, seqNumber, count);
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
// The size threshold belongs to the communicator that read it.
//
// ACCL_PROFILER_MIN_SIZE_BYTES used to land in a file static, so the last
// acclPluginInit() in the process decided the filter for every communicator,
// including ones created earlier: a second comm created with the variable unset
// silently inherited the first comm's threshold (the assignment is conditional,
// so an unset variable does not even reset it), and the minSize= each comm had
// already echoed in its own init log no longer described what it applied.
//
// Reaching that needs a setenv between two communicator creations in one
// process, which no RCCL path performs -- the environment is normally fixed for
// the life of the process, every comm reads the same number, and last-init-wins
// is invisible. The test does it explicitly because that is the property the
// per-context field establishes, not because a caller stumbles into it.
// =========================================================================
TEST(AcclProfilerMinSize, ThresholdIsPerCommunicator) {
    ScopedProfilerDir dir("minsize_percomm");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerMinSize.ThresholdIsPerCommunicator",
        [&dir]() {
            // 1024 x ncclFloat32 is 4096 B; reports whether it was admitted.
            auto runSmallColl = [](void* ctx, uint64_t seqNumber) -> bool {
                return RunSingleChannelColl(ctx, seqNumber, /*count=*/1024)
                       != nullptr;
            };

            // Comm A is created with a threshold above the 4096 B collective.
            ASSERT_EQ(setenv("ACCL_PROFILER_MIN_SIZE_BYTES", "8192", 1), 0);
            void* ctxA = nullptr;
            int maskA = 0;
            ASSERT_EQ(acclPluginInit(&ctxA, 0x5128, &maskA, "minsize_comm_a",
                                     1, 2, 0, nullptr), 0);
            ASSERT_NE(ctxA, nullptr);

            // Comm B is created with no threshold at all and must profile
            // everything. Unsetting also covers the conditional assignment: a
            // file static keeps its old value when the getenv returns NULL.
            ASSERT_EQ(unsetenv("ACCL_PROFILER_MIN_SIZE_BYTES"), 0);
            void* ctxB = nullptr;
            int maskB = 0;
            ASSERT_EQ(acclPluginInit(&ctxB, 0x5129, &maskB, "minsize_comm_b",
                                     1, 2, 0, nullptr), 0);
            ASSERT_NE(ctxB, nullptr);

            EXPECT_TRUE(runSmallColl(ctxB, /*seqNumber=*/90))
                << "comm B was created with ACCL_PROFILER_MIN_SIZE_BYTES unset, "
                   "so its threshold is 0 and a 4096 B collective must be "
                   "profiled; it inherited comm A's 8192 B threshold instead";

            // The reverse direction: comm A must still enforce the threshold it
            // read, so comm B's creation cannot have relaxed it either.
            EXPECT_FALSE(runSmallColl(ctxA, /*seqNumber=*/91))
                << "comm A read ACCL_PROFILER_MIN_SIZE_BYTES=8192 and must keep "
                   "dropping 4096 B collectives after comm B was created";

            ASSERT_EQ(acclPluginFinalize(ctxA), 0);
            ASSERT_EQ(acclPluginFinalize(ctxB), 0);

            // Each comm writes its own file, keyed by commHash.
            const std::string recB = ReadCollRecord(dir.c_str(), "0x5129");
            ASSERT_FALSE(recB.empty()) << "comm B emitted no collective record";
            EXPECT_NE(recB.find("\"coll_sn\":90"), std::string::npos) << recB;

            EXPECT_TRUE(ReadCollRecord(dir.c_str(), "0x5128").empty())
                << "comm A emitted a record for a collective below its own "
                   "threshold";
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

// =========================================================================
// acclParseMinSize — ACCL_PROFILER_MIN_SIZE_BYTES validation.
//
// The threshold used to go through atol(), whose failure modes are all silent.
// The dangerous direction is a value that comes out ENORMOUS: every collective
// then fails the `<` filter, every start hands back a NULL handle, nothing is
// ever allocated, and finalize writes a summary reporting zero drops over zero
// records. The output file looks like a clean run of a job that did no
// collectives. "-1" reaches that through the unsigned conversion and an
// over-long digit string reaches it through atol()'s saturation to LONG_MAX.
// =========================================================================
struct MinSizeParseCase {
    const char* input;
    int         expectBad;
    size_t      expectValue;
};

class AcclParseMinSizeTest
    : public ::testing::TestWithParam<MinSizeParseCase> {};

TEST_P(AcclParseMinSizeTest, MatchesExpected) {
    const MinSizeParseCase& c = GetParam();
    int bad = -1;
    size_t got = test_acclParseMinSize(c.input, &bad);
    EXPECT_EQ(bad, c.expectBad) << "input: \"" << c.input << "\"";
    EXPECT_EQ(got, c.expectValue) << "input: \"" << c.input << "\"";
}

INSTANTIATE_TEST_SUITE_P(Values, AcclParseMinSizeTest, ::testing::Values(
    // Accepted.
    MinSizeParseCase{"0", 0, 0},
    MinSizeParseCase{"8192", 0, 8192},
    MinSizeParseCase{"  8192  ", 0, 8192},
    // An all-blank value is the shell idiom for "off", so it is the default
    // rather than an error.
    MinSizeParseCase{"", 0, 0},
    MinSizeParseCase{"   ", 0, 0},
    // Rejected. Every one of these produced a usable-looking number from atol().
    MinSizeParseCase{"-1", 1, 0},        // was SIZE_MAX: filters everything
    MinSizeParseCase{"-8192", 1, 0},
    // Same case reached through the rest of the whitespace class. strtoull
    // skips all of it, so a skip loop that only knows ' ' and '\t' leaves the
    // '-' for strtoull to wrap to ULLONG_MAX -- silently, without ERANGE.
    MinSizeParseCase{"\n-1", 1, 0},
    MinSizeParseCase{"\r-1", 1, 0},
    MinSizeParseCase{"\v-1", 1, 0},
    MinSizeParseCase{"\f-1", 1, 0},
    MinSizeParseCase{"\n8192\n", 0, 8192},
    MinSizeParseCase{"99999999999999999999999", 1, 0},  // was LONG_MAX: ditto
    MinSizeParseCase{"abc", 1, 0},       // was 0, indistinguishable from a real 0
    MinSizeParseCase{"0x2000", 1, 0},    // was 0, not 8192 as the writer meant
    MinSizeParseCase{"8192garbage", 1, 0},  // was 8192, ignoring the tail
    MinSizeParseCase{"8192 4096", 1, 0},
    MinSizeParseCase{"8.5", 1, 0}
));

// End to end: a negative threshold must not silently swallow the whole run.
// Before the validation this wrote a file holding nothing but a summary line
// claiming "complete":true, which is the one thing the loss accounting on this
// plugin is supposed to make impossible.
TEST(AcclProfilerMinSize, NegativeThresholdDoesNotSwallowTheRun) {
    ScopedProfilerDir dir("minsize_negative");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerMinSize.NegativeThresholdDoesNotSwallowTheRun",
        [&dir]() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0x512A, &mask, "minsize_neg_test",
                                     1, 2, 0, nullptr), 0);
            ASSERT_NE(ctx, nullptr);

            ncclProfilerEventDescr_v5_t cd;
            MakeCollDescr(&cd, /*nChannels=*/1, /*seqNumber=*/95,
                          /*count=*/1024);
            void* coll = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &coll, &cd), 0);
            ASSERT_NE(coll, nullptr)
                << "ACCL_PROFILER_MIN_SIZE_BYTES=-1 converted to SIZE_MAX and "
                   "filtered out a 4096 B collective";

            ncclProfilerEventDescr_v5_t kd;
            MakeKernelChDescr(&kd, coll, /*channelId=*/0, /*pTimer=*/1000000);
            void* kch = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kch, &kd), 0);
            EXPECT_EQ(acclPluginStopEvent(coll), 0);
            ncclProfilerEventStateArgs_v5_t sa;
            memset(&sa, 0, sizeof(sa));
            sa.kernelCh.pTimer = 1010000;
            EXPECT_EQ(acclPluginRecordEventState(
                kch, ncclProfilerKernelChStop, &sa), 0);
            EXPECT_EQ(acclPluginStopEvent(kch), 0);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            const std::string rec = ReadCollRecord(dir.c_str(), "0x512a");
            ASSERT_FALSE(rec.empty())
                << "the run emitted no collective record at all, and its "
                   "summary line still reads: "
                << ReadSummaryLine(dir.c_str(), "0x512a");
            EXPECT_NE(rec.find("\"coll_sn\":95"), std::string::npos) << rec;
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()},
         {"ACCL_PROFILER_MIN_SIZE_BYTES", "-1"}}
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
            MakeProxyOpDescr(&proxyDescr, collHandle, /*channelId=*/0, /*peer=*/1,
                             /*nSteps=*/1, /*isSend=*/1);

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
            const std::string path = ProfilerOutputPath(dir.c_str(), "0xcafe");
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
        [&dir]() {
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
            const std::string path = ProfilerOutputPath(dir.c_str(), "0xface");
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
            MakeKernelChDescr(&kchDescr, collHandle, /*channelId=*/0,
                              /*pTimer=*/5000000);
            void* kchHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &kchHandle, &kchDescr), 0);

            // Start ProxyOp
            ncclProfilerEventDescr_v5_t proxyDescr;
            MakeProxyOpDescr(&proxyDescr, collHandle, /*channelId=*/0, /*peer=*/1,
                             /*nSteps=*/1, /*isSend=*/1);
            void* proxyHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &proxyHandle, &proxyDescr), 0);

            // Start ProxyStep
            ncclProfilerEventDescr_v5_t stepDescr;
            MakeProxyStepDescr(&stepDescr, proxyHandle, /*step=*/0);
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

            const std::string path = ProfilerOutputPath(dir.c_str(), "0xa1b2");
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
            MakeProxyOpDescr(&proxyDescr, collHandle, /*channelId=*/0, /*peer=*/1,
                             /*nSteps=*/1, /*isSend=*/1);
            void* proxyHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &proxyHandle, &proxyDescr), 0);

            ncclProfilerEventDescr_v5_t stepDescr;
            MakeProxyStepDescr(&stepDescr, proxyHandle, /*step=*/0);
            void* stepHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &stepHandle, &stepDescr), 0);

            // Mirror the send-side ordering in src/transport/net.cc: each state
            // is announced as the step enters it, so the long sleep below is
            // time spent waiting on the GPU, between SendGPUWait and
            // SendPeerWait.
            // kTrailUs differs from kShortUs on purpose: with both equal, the
            // interval charge-on-entry would put in SendWait and the one
            // charge-on-exit puts there are the same number, so the
            // proxy_network_us assertion below could not tell them apart.
            const int kLongUs = 40000;
            const int kShortUs = 2000;
            const int kTrailUs = 20000;
            ASSERT_EQ(acclPluginRecordEventState(
                stepHandle, ncclProfilerProxyStepSendGPUWait, nullptr), 0);
            usleep(kLongUs);
            ASSERT_EQ(acclPluginRecordEventState(
                stepHandle, ncclProfilerProxyStepSendPeerWait_v4, nullptr), 0);
            usleep(kShortUs);
            ASSERT_EQ(acclPluginRecordEventState(
                stepHandle, ncclProfilerProxyStepSendWait, nullptr), 0);
            usleep(kTrailUs);
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

            std::string out = ReadProfilerOutput(dir.c_str(), "0xc3d4");
            ASSERT_FALSE(out.empty()) << "no profiler output in " << dir.path();

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
            // the stop path closes it. Charge-on-entry would put kShortUs here;
            // charge-on-exit puts kTrailUs, so the midpoint separates them.
            EXPECT_GT(JsonNumber(out, "proxy_network_us"),
                      (kShortUs + kTrailUs) / 2.0)
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
    MakeProxyOpDescr(&od, coll, /*channelId=*/0, /*peer=*/1,
                     /*nSteps=*/1, /*isSend=*/isSend);
    ASSERT_EQ(acclPluginStartEvent(ctx, opOut, &od), 0);

    ncclProfilerEventDescr_v5_t sd;
    MakeProxyStepDescr(&sd, *opOut, /*step=*/0);
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

// A proxy op or step still held at finalize keeps the context alive, and must:
// that handle may still be delivered, since proxy events come straight from the
// proxy progress thread, which is shared via comm->sharedRes and outlives one
// comm's teardown. So the loss cannot be reclaimed -- it has to be REPORTED,
// or a run that silently dropped proxy timing reads as clean. Three kinds never
// reach acclFreeProxyOp/acclFreeProxyStep on their own: an op with no
// collective parent, an op whose stop never arrives (coll->proxyOpIndices is
// filled at op STOP, so the coll drain cannot see it), and any unstopped step.
//
// The output descriptor is a shared process limit rather than a fixed per-comm
// cost, so it is released at finalize even though the context is not.
TEST(AcclProfilerLifecycle, OutstandingProxyEventsAreReportedAtFinalize) {
    ScopedProfilerDir dir("orphanproxy");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLifecycle.OutstandingProxyEventsAreReportedAtFinalize",
        [&dir]() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xD1D5, &mask, "orphan_proxy_test",
                                     1, 8, 0, nullptr), 0);

            ncclProfilerEventDescr_v5_t cd;
            MakeCollDescr(&cd, 1, 41, 1);
            void* coll = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &coll, &cd), 0);
            ASSERT_NE(coll, nullptr);

            ncclProfilerEventDescr_v5_t od;
            MakeProxyOpDescr(&od, coll, /*channelId=*/0, /*peer=*/0,
                             /*nSteps=*/1, /*isSend=*/1);
            void* op = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &op, &od), 0);
            ASSERT_NE(op, nullptr);

            // The same op with no collective parent: in no index list at all.
            od.parentObj = nullptr;
            void* parentlessOp = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &parentlessOp, &od), 0);
            ASSERT_NE(parentlessOp, nullptr);

            ncclProfilerEventDescr_v5_t sd;
            MakeProxyStepDescr(&sd, op, /*step=*/0);
            void* step = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &step, &sd), 0);
            ASSERT_NE(step, nullptr);

            // Deliberately no stop for any of the three.
            const std::string path = ProfilerOutputPath(dir.c_str(), "0xd1d5");
            ASSERT_EQ(OpenFdCountFor(path), 1)
                << "plugin did not open " << path;

            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            EXPECT_EQ(OpenFdCountFor(path), 0)
                << "output file still open after finalize: a context held by an "
                   "outstanding proxy event would hold the descriptor forever";

            const std::string summary = ReadSummaryLine(dir.c_str(), "0xd1d5");
            ASSERT_FALSE(summary.empty());
            // Outstanding, not dropped or stale: nothing here was rejected.
            // The collective is a leak too -- its op never stopped, so it never
            // reached its completion predicate.
            ExpectSummaryCounters(summary, {{"outstanding_proxy_ops", 2},
                                            {"outstanding_proxy_steps", 1},
                                            {"leaked_collectives", 1}});
            // The loss has to reach the consumer; it cannot be reclaimed.
            EXPECT_NE(summary.find("\"complete\":false"), std::string::npos);

            // The other half of the contract: those handles are still live, so
            // their late stops must find a context, not freed memory. Each
            // releases its reference, and the last one tears the context down.
            EXPECT_EQ(acclPluginStopEvent(step), 0);
            EXPECT_EQ(acclPluginStopEvent(op), 0);
            EXPECT_EQ(acclPluginStopEvent(parentlessOp), 0);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
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

            // 6 send / 2 recv, not 4/4: with equal counts the two divisors are
            // interchangeable, so swapping them would still produce the right
            // numbers and this test would pass.
            void* ops[8] = {nullptr};
            for (int i = 0; i < 8; i++) {
                DriveProxyOpWithTimings(ctx, coll, i < 6 ? 1 : 0, &ops[i]);
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

            const std::string line = ReadCollRecord(dir.c_str(), "0xd1d0");
            ASSERT_FALSE(line.empty());

            EXPECT_EQ(JsonNumber(line, "n_proxy_ops"), 8);
            EXPECT_EQ(JsonNumber(line, "n_send_ops"), 6);
            EXPECT_EQ(JsonNumber(line, "n_recv_ops"), 2);

            // usleep only guarantees a lower bound, so each window is
            // [nominal, 2x nominal): wide enough for scheduling jitter, tight
            // enough to exclude both the /n_proxy_ops value and a send/recv
            // divisor swap.
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

            const std::string line = ReadCollRecord(dir.c_str(), "0xd1d1");
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
        [&dir]() {
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

            const std::string path = ProfilerOutputPath(dir.c_str(), "0xd1d2");
            std::ifstream ifs(path);
            ASSERT_TRUE(ifs.good()) << "Output file not found: " << path;
            std::string line;
            ASSERT_TRUE(std::getline(ifs, line));
            EXPECT_NE(line.find("\"gpu_kernel_avg_us\":150.00"), std::string::npos)
                << "Expected avg of 100+200=150us";
            EXPECT_NE(line.find("\"gpu_kernel_min_us\":100.00"), std::string::npos);
            EXPECT_NE(line.find("\"gpu_kernel_max_us\":200.00"), std::string::npos);
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
            // The unfinalized slot must be counted as leaked, not hidden, and
            // no other loss channel may have fired instead.
            ExpectSummaryCounters(ReadSummaryLine(dir.c_str(), "0x2602"),
                                  {{"leaked_collectives", 1}});
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
            // Every counter, so "complete":true is checked against the same
            // vector the loss tests check against.
            const std::string s = ReadSummaryLine(dir.c_str(), "0x5101");
            ExpectSummaryCounters(s, {});
            // The pool sizes are how a consumer tells "0 drops" from "0 drops
            // because the pool was 16x bigger", so the producer must emit them.
            EXPECT_EQ(JsonNumber(s, "coll_pool_size"), ACCL_COLL_POOL_SIZE);
            EXPECT_EQ(JsonNumber(s, "proxy_op_pool_size"),
                      ACCL_PROXY_OP_POOL_SIZE);
            EXPECT_EQ(JsonNumber(s, "proxy_step_pool_size"),
                      ACCL_PROXY_STEP_POOL_SIZE);
            EXPECT_EQ(JsonNumber(s, "max_proxy_ops_per_coll"),
                      ACCL_MAX_PROXY_OPS);
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
            EXPECT_NE(out.find("\"complete\":false"), std::string::npos);
            // Every slot the drain reclaims must be counted, and as a leak
            // rather than as one of the other loss channels.
            ExpectSummaryCounters(ReadSummaryLine(dir.c_str(), "0x5102"),
                                  {{"leaked_collectives", 3}});
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
            EXPECT_NE(out.find("\"complete\":false"), std::string::npos);
            // Both rejected allocations are drops; the 256 pinned slots the
            // drain reclaims are leaks. The two must not be conflated.
            ExpectSummaryCounters(ReadSummaryLine(dir.c_str(), "0x5103"),
                                  {{"dropped_collectives", 2},
                                   {"leaked_collectives", 256}});
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
        [&dir]() {
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
            MakeProxyOpDescr(&od, collHandle, /*channelId=*/0, /*peer=*/1,
                             /*nSteps=*/1, /*isSend=*/1);
            void* opHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &opHandle, &od), 0);
            ASSERT_NE(opHandle, nullptr);

            ncclProfilerEventDescr_v5_t sd;
            MakeProxyStepDescr(&sd, opHandle, /*step=*/0);
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
        [&dir]() {
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
            MakeProxyOpDescr(&od, collHandle, /*channelId=*/0, /*peer=*/1,
                             /*nSteps=*/1, /*isSend=*/1);
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
        [&dir]() {
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

            // This is the state inflight_collectives exists to name: the record
            // may never reach disk, so the run must not read as clean.
            ExpectSummaryCounters(ReadSummaryLine(dir.c_str(), "0x4b01"),
                                  {{"inflight_collectives", 1}});

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
        [&dir]() {
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
            // Sample before finalize: the loop keeps incrementing after
            // outputFile is NULLed, so the post-join count cannot show overlap.
            const long writesBeforeFinalize = writes.load();
            EXPECT_EQ(acclPluginFinalize(ctx), 0);
            writer.join();
            EXPECT_GT(writesBeforeFinalize, 0)
                << "writer never ran before finalize; test proved nothing";

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
            MakeProxyOpDescr(&od, nullptr, /*channelId=*/0, /*peer=*/0,
                             /*nSteps=*/1, /*isSend=*/1);
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
            // The 1024 pinned ops are outstanding too; naming both is what
            // ties "complete":false to the drop rather than to them.
            ExpectSummaryCounters(s, {{"dropped_proxy_ops", 3},
                                      {"outstanding_proxy_ops",
                                       ACCL_PROXY_OP_POOL_SIZE}});
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
            MakeProxyOpDescr(&od, nullptr, /*channelId=*/0, /*peer=*/0,
                             /*nSteps=*/1, /*isSend=*/1);
            void* op = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &op, &od), 0);
            ASSERT_NE(op, nullptr);

            ncclProfilerEventDescr_v5_t sd;
            MakeProxyStepDescr(&sd, op, /*step=*/0);
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
            ExpectSummaryCounters(s, {{"dropped_proxy_steps", 2},
                                      {"outstanding_proxy_ops", 1},
                                      {"outstanding_proxy_steps",
                                       ACCL_PROXY_STEP_POOL_SIZE}});
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
            MakeProxyOpDescr(&od, coll, /*channelId=*/0, /*peer=*/0,
                             /*nSteps=*/1, /*isSend=*/1);
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
            // No KernelCh ever arrives, so acclShouldFinalize never fires and
            // the drain reclaims the collective: that leak is expected here.
            ExpectSummaryCounters(s, {{"overflow_proxy_ops", 2},
                                      {"leaked_collectives", 1}});
            EXPECT_NE(s.find("\"complete\":false"), std::string::npos)
                << "a run that discarded proxy ops reported itself complete: " << s;
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
            MakeProxyOpDescr(&od, nullptr, /*channelId=*/0, /*peer=*/0,
                             /*nSteps=*/2, /*isSend=*/1);
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
            MakeProxyStepDescr(&sd, op, /*step=*/0);
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
            memset(&cd, 0, sizeof(cd));
            cd.type = ncclProfileColl;
            cd.coll.func = "AllReduce";
            cd.coll.algo = "Ring";
            cd.coll.proto = "Simple";
            cd.coll.datatype = "ncclFloat32";
            cd.coll.count = 1024;
            cd.coll.nChannels = 1;

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
                memset(&kd, 0, sizeof(kd));
                kd.type = ncclProfileKernelCh;
                kd.parentObj = coll;
                kd.kernelCh.channelId = 0;
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

// =========================================================================
// The debug-log path itself.
//
// Every other test passes nullptr for logfn, which leaves gLogFn NULL and makes
// ACCL_WARN and ACCL_INFO no-ops: neither the (level, flags) pair each macro
// uses nor the one-shot guards that keep a pool-exhaustion warning from
// repeating per drop can be observed at all. Only the pair and the call count
// are asserted; the message text is not a contract.
// =========================================================================
namespace {
struct LogRecord { int level; unsigned long flags; };
std::vector<LogRecord> gLogRecords;

void RecordingLogger(int level, unsigned long flags, const char*, int,
                     const char*, ...) {
    gLogRecords.push_back({level, flags});
}

int CountLogsAt(int level) {
    int n = 0;
    for (const auto& r : gLogRecords) if (r.level == level) n++;
    return n;
}
}  // namespace

TEST(AcclProfilerLogging, WarningsUseAllSubsysAndFireOncePerPool) {
    ScopedProfilerDir dir("logger");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerLogging.WarningsUseAllSubsysAndFireOncePerPool",
        [&dir]() {
            gLogRecords.clear();
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0x7101, &mask, "logger",
                                     1, 1, 0, RecordingLogger), 0);

            // init logs its configuration at INFO: routine chatter, so it stays
            // on NCCL_PROFILE and must not be broadcast to every subsystem.
            ASSERT_GT(CountLogsAt(4), 0) << "init logged nothing at INFO";
            for (const auto& r : gLogRecords) {
                if (r.level == 4) {
                    EXPECT_EQ(r.flags, 0x4000UL)
                        << "INFO must stay on NCCL_PROFILE";
                }
            }
            EXPECT_EQ(CountLogsAt(3), 0) << "a clean init must not warn";

            // Exhaust the coll pool, then keep going. The guard means the
            // extra drops add no further lines.
            ncclProfilerEventDescr_v5_t cd;
            MakeCollDescr(&cd, /*nChannels=*/1, /*seqNumber=*/0, /*count=*/64);
            for (int i = 0; i < ACCL_COLL_POOL_SIZE + 5; i++) {
                void* h = nullptr;
                ASSERT_EQ(acclPluginStartEvent(ctx, &h, &cd), 0);
            }
            ASSERT_EQ(CountLogsAt(3), 1)
                << "coll pool exhaustion must warn exactly once, not once per "
                   "dropped collective";
            for (const auto& r : gLogRecords) {
                if (r.level == 3) {
                    EXPECT_EQ(r.flags, ~0UL)
                        << "a WARN on NCCL_PROFILE is dropped under the default "
                           "NCCL_DEBUG_SUBSYS mask, which hides every report of "
                           "lost profiling data";
                }
            }

            // The proxy-op pool carries its own guard, so it adds one more.
            ncclProfilerEventDescr_v5_t od;
            MakeProxyOpDescr(&od, nullptr, /*channelId=*/0, /*peer=*/0,
                             /*nSteps=*/1, /*isSend=*/1);
            for (int i = 0; i < ACCL_PROXY_OP_POOL_SIZE + 5; i++) {
                void* h = nullptr;
                ASSERT_EQ(acclPluginStartEvent(ctx, &h, &od), 0);
            }
            EXPECT_EQ(CountLogsAt(3), 2)
                << "the proxy-op pool guard is separate and must fire once";

            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            // finalize warns once more for the INCOMPLETE run.
            EXPECT_GE(CountLogsAt(3), 3);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

// =========================================================================
// A write failure must reach the summary.
//
// ferror() is the only place a discarded fprintf failure (ENOSPC, EIO) is still
// visible: the records on disk are truncated while every loss counter reads 0,
// so without the flag such a run emits "complete":true. Closing the stream's
// descriptor makes the next flush fail and sets the sticky error bit; dup2
// restores a working descriptor so finalize can still write the summary that
// reports it.
// =========================================================================
TEST(AcclProfilerSummary, WriteErrorClearsComplete) {
    ScopedProfilerDir dir("write_error");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerSummary.WriteErrorClearsComplete",
        [&dir]() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0x7105, &mask, "write_error",
                                     1, 1, 0, nullptr), 0);

            const int fd = test_acclOutputFd(ctx);
            ASSERT_GE(fd, 0) << "plugin opened no output file";
            const int saved = dup(fd);
            ASSERT_GE(saved, 0);
            ASSERT_EQ(close(fd), 0);

            // fflush inside acclWriteRecord now fails on a closed descriptor.
            test_acclWriteDummyRecord(ctx);

            // Put a working descriptor back at the same number. The stream's
            // error bit is sticky, so it survives the repair.
            ASSERT_EQ(dup2(saved, fd), fd);
            ASSERT_EQ(close(saved), 0);

            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            const std::string s = ReadSummaryLine(dir.c_str(), "0x7105");
            ASSERT_FALSE(s.empty()) << "no summary line was written";
            EXPECT_NE(s.find("\"write_error\":true"), std::string::npos)
                << "a truncated file must name its cause: " << s;
            EXPECT_NE(s.find("\"complete\":false"), std::string::npos)
                << "a run whose writes failed reported itself complete: " << s;
            // Nothing else fired: write_error is the only reason here.
            ExpectSummaryCounters(s, {});
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

// =========================================================================
// A ProxyStep that outlives its parent op's tenancy must be dropped.
//
// The op's slot is freed from three paths a step knows nothing about, so it can
// be reissued before the step stops. Charging the stale step's timings to the
// new tenant silently corrupts an unrelated collective's decomposition, which
// is worse than losing the step: the generation pinned at start is what
// separates the two.
// =========================================================================
TEST(AcclProfilerSummary, StaleProxyStepIsDroppedAndCounted) {
    ScopedProfilerDir dir("stalestep");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclProfilerSummary.StaleProxyStepIsDroppedAndCounted",
        [&dir]() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0x7106, &mask, "stalestep",
                                     1, 1, 0, nullptr), 0);

            ncclProfilerEventDescr_v5_t od;
            MakeProxyOpDescr(&od, nullptr, /*channelId=*/0, /*peer=*/0,
                             /*nSteps=*/1, /*isSend=*/1);
            void* op = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &op, &od), 0);
            ASSERT_NE(op, nullptr);
            const int slot = test_acclProxyOpSlot(ctx, op);
            ASSERT_GE(slot, 0);
            const unsigned long gen0 = test_acclProxyOpGeneration(ctx, slot);

            ncclProfilerEventDescr_v5_t sd;
            MakeProxyStepDescr(&sd, op, /*step=*/0);
            void* step = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &step, &sd), 0);
            ASSERT_NE(step, nullptr);
            ASSERT_EQ(acclPluginRecordEventState(
                step, ncclProfilerProxyStepSendGPUWait, nullptr), 0);
            usleep(5000);

            ASSERT_EQ(acclPluginStopEvent(op), 0);   // releases the slot

            void* op2 = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &op2, &od), 0);
            ASSERT_EQ(test_acclProxyOpSlot(ctx, op2), slot)
                << "expected the freed slot to be handed out again";
            EXPECT_EQ(test_acclProxyOpGeneration(ctx, slot), gen0 + 1)
                << "every alloc must bump the slot's tenancy";

            // The stale stop lands on the new tenant's address.
            ASSERT_EQ(acclPluginStopEvent(step), 0);
            ASSERT_EQ(acclPluginStopEvent(op2), 0);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);

            const std::string s = ReadSummaryLine(dir.c_str(), "0x7106");
            ASSERT_FALSE(s.empty()) << "no summary line was written";
            ExpectSummaryCounters(s, {{"stale_proxy_steps", 1}});
            EXPECT_NE(s.find("\"complete\":false"), std::string::npos)
                << "a dropped step is still lost data: " << s;
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

// =========================================================================
// Reusing a proxy-op slot must not write the slot's mutex.
//
// ProxyOpMutexSurvivesSlotRelease checks only that the mutex is not destroyed,
// which a memset running straight through it would still satisfy: that clear
// zeroes __kind rather than setting it to -1. The property here is the race
// itself, so this mirrors CollSlotReuseDoesNotWriteTheSlotMutex on the proxy
// side -- a thread driving ProxyStep stops into pthread_mutex_lock(&op->mutex)
// while the slot is freed and re-allocated underneath it. Unfixed, glibc's lock
// fast path aborts on the zeroed owner word, or a parked waiter is never woken;
// the forked child reports both as a failure, and the explicit timeout keeps a
// regression from stalling the suite.
// =========================================================================
static constexpr int kProxyOpMutexTimeoutSeconds = 60;

TEST(AcclProfilerLifecycle, ProxyOpSlotReuseDoesNotWriteTheSlotMutex) {
    ScopedProfilerDir dir("opmutexrace");
    RUN_ISOLATED_TESTS(
        ProcessIsolatedTestRunner::TestConfig(
        "AcclProfilerLifecycle.ProxyOpSlotReuseDoesNotWriteTheSlotMutex",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0x4B04, &mask, "opmutexrace",
                                     1, 1, 0, nullptr), 0);

            ncclProfilerEventDescr_v5_t od;
            MakeProxyOpDescr(&od, nullptr, /*channelId=*/0, /*peer=*/0,
                             /*nSteps=*/1, /*isSend=*/1);
            void* op = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &op, &od), 0);
            ASSERT_NE(op, nullptr);
            const int slot = test_acclProxyOpSlot(ctx, op);
            ASSERT_EQ(slot, 0) << "expected the first proxy op to take slot 0";

            // Stale-handle traffic: each step stop locks op->mutex at the
            // address RCCL still holds, with no idea the slot moved on.
            std::atomic<bool> stop{false};
            std::atomic<unsigned long> locks{0};
            std::thread steps([&]() {
                ncclProfilerEventDescr_v5_t sd;
                MakeProxyStepDescr(&sd, op, /*step=*/0);
                while (!stop.load(std::memory_order_relaxed)) {
                    void* sh = nullptr;
                    if (acclPluginStartEvent(ctx, &sh, &sd) == 0 && sh) {
                        // Start pins the generation under op->mutex and stop
                        // compares it under the same lock: two more chances to
                        // be inside the lock while the clear runs.
                        if (acclPluginStopEvent(sh) == 0) {
                            locks.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
            });

            // Free and immediately re-take the same slot, so every iteration
            // runs acclAllocProxyOp's clear beside those locks.
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(20);
            for (int i = 0; i < 400000; i++) {
                ASSERT_EQ(acclPluginStopEvent(op), 0);
                void* again = nullptr;
                ASSERT_EQ(acclPluginStartEvent(ctx, &again, &od), 0);
                ASSERT_EQ(again, op) << "expected slot " << slot << " back";
                if ((i & 0x3FF) == 0 &&
                    std::chrono::steady_clock::now() > deadline) {
                    break;
                }
            }

            stop.store(true, std::memory_order_relaxed);
            steps.join();
            EXPECT_GT(locks.load(), 0u)
                << "no ProxyStep ever took op->mutex, so nothing was raced "
                   "against the slot reuse and this test proved nothing";

            ASSERT_EQ(acclPluginStopEvent(op), 0);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
        })
        .withEnvironment({{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}})
        .withTimeout(std::chrono::seconds(kProxyOpMutexTimeoutSeconds))
    );
}


// =========================================================================
// Net counters (ACCL_NET_COUNTERS=1): reader parsing and snapshot chaining
// =========================================================================

// Builds netlink messages the way the kernel lays them out.
class NlBuf {
 public:
    void BeginMsg(uint16_t type, uint32_t seq, uint16_t flags = 0) {
        msg_ = b_.size();
        struct nlmsghdr h;
        memset(&h, 0, sizeof(h));
        h.nlmsg_type = type;
        h.nlmsg_seq = seq;
        h.nlmsg_flags = flags;
        Append(&h, sizeof(h));
    }
    void EndMsg() {
        Pad();
        reinterpret_cast<struct nlmsghdr*>(&b_[msg_])->nlmsg_len = (uint32_t)(b_.size() - msg_);
    }
    size_t BeginNest(uint16_t type) {
        size_t off = b_.size();
        struct nlattr a = {0, (uint16_t)(type | NLA_F_NESTED)};
        Append(&a, sizeof(a));
        return off;
    }
    void EndNest(size_t off) {
        reinterpret_cast<struct nlattr*>(&b_[off])->nla_len = (uint16_t)(b_.size() - off);
    }
    void Attr(uint16_t type, const void* p, size_t n) {
        struct nlattr a = {(uint16_t)(NLA_HDRLEN + n), type};
        Append(&a, sizeof(a));
        Append(p, n);
        Pad();
    }
    void U32(uint16_t type, uint32_t v) { Attr(type, &v, sizeof(v)); }
    void U64(uint16_t type, uint64_t v) { Attr(type, &v, sizeof(v)); }
    void Str(uint16_t type, const std::string& s) { Attr(type, s.c_str(), s.size() + 1); }
    void Error(uint32_t seq, int err) {
        BeginMsg(NLMSG_ERROR, seq);
        struct nlmsgerr e;
        memset(&e, 0, sizeof(e));
        e.error = err;
        Append(&e, sizeof(e));
        EndMsg();
    }
    void Done(uint32_t seq) {
        BeginMsg(NLMSG_DONE, seq, NLM_F_MULTI);
        int zero = 0;
        Append(&zero, sizeof(zero));
        EndMsg();
    }
    void Dev(uint32_t seq, uint32_t idx, const std::string& name) {
        BeginMsg(RDMA_NL_GET_TYPE(RDMA_NL_NLDEV, RDMA_NLDEV_CMD_GET), seq, NLM_F_MULTI);
        U32(RDMA_NLDEV_ATTR_DEV_INDEX, idx);
        Str(RDMA_NLDEV_ATTR_DEV_NAME, name);
        EndMsg();
    }
    // A STAT_GET reply; a PAD attribute before each value, as nla_put_u64_64bit may add.
    void Stat(uint32_t seq, const std::vector<std::pair<std::string, uint64_t>>& counters,
              bool pad = true) {
        BeginMsg(RDMA_NL_GET_TYPE(RDMA_NL_NLDEV, RDMA_NLDEV_CMD_STAT_GET), seq);
        U32(RDMA_NLDEV_ATTR_DEV_INDEX, 3);
        Str(RDMA_NLDEV_ATTR_DEV_NAME, "rdma0");
        U32(RDMA_NLDEV_ATTR_PORT_INDEX, 1);
        size_t table = BeginNest(RDMA_NLDEV_ATTR_STAT_HWCOUNTERS);
        for (const auto& c : counters) {
            size_t entry = BeginNest(RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY);
            Str(RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY_NAME, c.first);
            if (pad) Attr(RDMA_NLDEV_ATTR_PAD, nullptr, 0);
            U64(RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY_VALUE, c.second);
            EndNest(entry);
        }
        EndNest(table);
        EndMsg();
    }
    const std::vector<char>& bytes() const { return b_; }

 private:
    void Append(const void* p, size_t n) {
        const char* c = static_cast<const char*>(p);
        b_.insert(b_.end(), c, c + n);
    }
    void Pad() { while (b_.size() % 4) b_.push_back(0); }
    std::vector<char> b_;
    size_t msg_ = 0;
};

using CounterList = std::vector<std::pair<std::string, uint64_t>>;

static void CollectCounter(const char* name, uint64_t value, int pos, void* arg) {
    auto* out = static_cast<std::vector<std::tuple<std::string, uint64_t, int>>*>(arg);
    out->emplace_back(name, value, pos);
}

TEST(AcclNetNetlink, ParsesNestedHwCounters) {
    NlBuf b;
    b.Stat(7, {{"tx_rdma_ucast_bytes", 1ULL << 40}, {"req_rx_pkt_seq_err", 3}});
    std::vector<std::tuple<std::string, uint64_t, int>> got;
    EXPECT_EQ(acclNlParseHwCounters(b.bytes().data(), b.bytes().size(), CollectCounter, &got), 2);
    ASSERT_EQ(got.size(), 2u);
    EXPECT_EQ(got[0], std::make_tuple(std::string("tx_rdma_ucast_bytes"), 1ULL << 40, 0));
    EXPECT_EQ(got[1], std::make_tuple(std::string("req_rx_pkt_seq_err"), 3ULL, 1));
}

TEST(AcclNetNetlink, ErrorReplyFailsAndAckIsSkipped) {
    NlBuf err;
    err.Error(7, -EINVAL);
    std::vector<std::tuple<std::string, uint64_t, int>> got;
    EXPECT_EQ(acclNlParseHwCounters(err.bytes().data(), err.bytes().size(), CollectCounter, &got), -1);

    NlBuf ack;
    ack.Error(7, 0);
    ack.Stat(7, {{"a", 1}});
    EXPECT_EQ(acclNlParseHwCounters(ack.bytes().data(), ack.bytes().size(), CollectCounter, &got), 1);

    // No STAT_GET message at all is not a successful empty read.
    NlBuf none;
    none.Done(7);
    EXPECT_EQ(acclNlParseHwCounters(none.bytes().data(), none.bytes().size(), CollectCounter, &got), -1);
}

TEST(AcclNetNetlink, EntryWithoutValueIsMalformed) {
    NlBuf b;
    b.BeginMsg(RDMA_NL_GET_TYPE(RDMA_NL_NLDEV, RDMA_NLDEV_CMD_STAT_GET), 1);
    size_t table = b.BeginNest(RDMA_NLDEV_ATTR_STAT_HWCOUNTERS);
    size_t entry = b.BeginNest(RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY);
    b.Str(RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY_NAME, "lonely");
    b.EndNest(entry);
    b.EndNest(table);
    b.EndMsg();
    std::vector<std::tuple<std::string, uint64_t, int>> got;
    EXPECT_EQ(acclNlParseHwCounters(b.bytes().data(), b.bytes().size(), CollectCounter, &got), -1);
}

// A short read must never walk past the buffer.
TEST(AcclNetNetlink, TruncatedReplyStopsAtTheBuffer) {
    NlBuf b;
    b.Stat(1, {{"a", 1}, {"b", 2}, {"c", 3}});
    // nlmsg_len no longer fits, so the message is not walked at all.
    for (size_t cut = 0; cut < b.bytes().size(); cut += 4) {
        std::vector<char> part(b.bytes().begin(), b.bytes().begin() + cut);
        std::vector<std::tuple<std::string, uint64_t, int>> got;
        EXPECT_EQ(acclNlParseHwCounters(part.data(), part.size(), CollectCounter, &got), -1)
            << "cut=" << cut;
        EXPECT_TRUE(got.empty());
    }
}

TEST(AcclNetNetlink, FindsDeviceIndexAcrossDumpParts) {
    NlBuf part1;
    part1.Dev(1, 0, "mlx5_0");
    part1.Dev(1, 3, "rdma0");
    int done = 0;
    EXPECT_EQ(acclNlFindDevIndex(part1.bytes().data(), part1.bytes().size(), "rdma0", &done), 3);
    EXPECT_EQ(done, 0);
    NlBuf part2;
    part2.Done(1);
    EXPECT_EQ(acclNlFindDevIndex(part2.bytes().data(), part2.bytes().size(), "rdma0", &done), -1);
    EXPECT_EQ(done, 1);
    NlBuf err;
    err.Error(1, -EPERM);
    EXPECT_EQ(acclNlFindDevIndex(err.bytes().data(), err.bytes().size(), "rdma0", &done), -2);
}

// A scripted RDMA netlink socket: answers each request from the current counters.
struct FakeNic {
    std::mutex mu;
    CounterList counters = {{"tx_rdma_ucast_bytes", 0}, {"tx_rdma_ucast_pkts", 0},
                            {"rx_rdma_ucast_bytes", 0}, {"rx_rdma_ucast_pkts", 0},
                            {"tx_rdma_retx_pkts", 0}};
    std::vector<std::pair<std::string, uint32_t>> devs = {{"mlx5_0", 0}, {"rdma0", 3}, {"rdma1", 4}};
    std::deque<std::vector<char>> replies;
    std::vector<std::vector<char>> requests;
    int statRequests = 0;
    int failStat = -1;    // 1-based STAT_GET to answer with NLMSG_ERROR
    int slowStat = -1;    // 1-based STAT_GET whose reply is delayed
    int slowMs = 0;
    int pendingSleepMs = 0;
    int staleBefore = -1;     // 1-based STAT_GET preceded by a reply to an older seq
    std::vector<CounterList> script;  // per-STAT_GET counters; then `counters`
    std::vector<uint32_t> statDevIndex;

    void Add(const std::string& name, uint64_t v) {
        std::lock_guard<std::mutex> lk(mu);
        for (auto& c : counters) if (c.first == name) c.second += v;
    }
};
static FakeNic* gFakeNic = nullptr;

static int FakeNlOpen() { return open("/dev/null", O_RDONLY | O_CLOEXEC); }

static ssize_t FakeNlSend(int, const void* buf, size_t len) {
    const struct nlmsghdr* nh = static_cast<const struct nlmsghdr*>(buf);
    std::lock_guard<std::mutex> lk(gFakeNic->mu);
    gFakeNic->requests.emplace_back((const char*)buf, (const char*)buf + len);
    NlBuf r;
    if (nh->nlmsg_type == RDMA_NL_GET_TYPE(RDMA_NL_NLDEV, RDMA_NLDEV_CMD_GET)) {
        for (const auto& d : gFakeNic->devs) r.Dev(nh->nlmsg_seq, d.second, d.first);
        gFakeNic->replies.push_back(r.bytes());
        NlBuf done;
        done.Done(nh->nlmsg_seq);
        gFakeNic->replies.push_back(done.bytes());
        return (ssize_t)len;
    }
    int n = ++gFakeNic->statRequests;
    const struct nlattr* a = (const struct nlattr*)NLMSG_DATA(nh);
    uint32_t idx = 0;
    memcpy(&idx, (const char*)a + NLA_HDRLEN, 4);
    gFakeNic->statDevIndex.push_back(idx);
    if (n == gFakeNic->staleBefore) {
        NlBuf stale;
        stale.Stat(nh->nlmsg_seq - 1, {{"a", 999999}});
        gFakeNic->replies.push_back(stale.bytes());
    }
    if (n == gFakeNic->failStat) r.Error(nh->nlmsg_seq, -EIO);
    else if (n <= (int)gFakeNic->script.size()) r.Stat(nh->nlmsg_seq, gFakeNic->script[n - 1]);
    else r.Stat(nh->nlmsg_seq, gFakeNic->counters);
    if (n == gFakeNic->slowStat) gFakeNic->pendingSleepMs = gFakeNic->slowMs;
    gFakeNic->replies.push_back(r.bytes());
    return (ssize_t)len;
}

static ssize_t FakeNlRecv(int, void* buf, size_t len) {
    int sleepMs;
    {
        std::lock_guard<std::mutex> lk(gFakeNic->mu);
        sleepMs = gFakeNic->pendingSleepMs;
        gFakeNic->pendingSleepMs = 0;
    }
    if (sleepMs) std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
    std::lock_guard<std::mutex> lk(gFakeNic->mu);
    if (gFakeNic->replies.empty()) return -1;
    std::vector<char> r = gFakeNic->replies.front();
    gFakeNic->replies.pop_front();
    size_t n = std::min(len, r.size());
    memcpy(buf, r.data(), n);
    return (ssize_t)n;
}

// No ethtool: on an AINIC host the real rdma0 netdev would otherwise be read.
static int NoEthIoctl(int, unsigned long, void*) { return -1; }

static void InstallFakeNic(FakeNic* nic) {
    gFakeNic = nic;
    struct acclNetHooks h;
    memset(&h, 0, sizeof(h));
    h.nlOpenFn = FakeNlOpen;
    h.sendFn = FakeNlSend;
    h.recvFn = FakeNlRecv;
    h.ioctlFn = NoEthIoctl;
    test_acclNetSetHooks(&h);
}

static void ClearNetHooks() {
    struct acclNetHooks h;
    memset(&h, 0, sizeof(h));
    test_acclNetSetHooks(&h);
    gFakeNic = nullptr;
}

TEST(AcclNetNetlink, ReaderResolvesIndexThenMapsMovedCountersByName) {
    FakeNic nic;
    // The second reply is reordered and preceded by a stale one for an old seq.
    nic.script = {{{"a", 1}, {"b", 2}, {"c", 3}}, {{"c", 30}, {"a", 10}, {"b", 20}}};
    nic.staleBefore = 2;
    InstallFakeNic(&nic);
    uint64_t first[ACCL_NET_MAX_RDMA] = {0}, second[ACCL_NET_MAX_RDMA] = {0};
    char names[ACCL_NET_MAX_RDMA][ACCL_NET_NAME_LEN];
    int nNames = 0;
    int rc = test_acclRdmaOpenAndRead("rdma0", first, second, names, &nNames);
    ClearNetHooks();
    ASSERT_EQ(rc, 0);
    ASSERT_EQ(nNames, 3);
    EXPECT_STREQ(names[0], "a");
    EXPECT_STREQ(names[1], "b");
    EXPECT_STREQ(names[2], "c");
    EXPECT_EQ(first[0], 1u);
    EXPECT_EQ(first[2], 3u);
    EXPECT_EQ(second[0], 10u);
    EXPECT_EQ(second[1], 20u);
    EXPECT_EQ(second[2], 30u);
    // GET dump first, then STAT_GET (request only) for DEV_INDEX=3, PORT_INDEX=1.
    ASSERT_EQ(nic.requests.size(), 3u);
    auto* dump = reinterpret_cast<const struct nlmsghdr*>(nic.requests[0].data());
    EXPECT_EQ(dump->nlmsg_type, RDMA_NL_GET_TYPE(RDMA_NL_NLDEV, RDMA_NLDEV_CMD_GET));
    EXPECT_EQ(dump->nlmsg_flags, NLM_F_REQUEST | NLM_F_DUMP);
    auto* stat = reinterpret_cast<const struct nlmsghdr*>(nic.requests[1].data());
    EXPECT_EQ(stat->nlmsg_type, RDMA_NL_GET_TYPE(RDMA_NL_NLDEV, RDMA_NLDEV_CMD_STAT_GET));
    EXPECT_EQ(stat->nlmsg_flags, NLM_F_REQUEST);
    const struct nlattr* a = (const struct nlattr*)NLMSG_DATA(stat);
    EXPECT_EQ(a->nla_type, RDMA_NLDEV_ATTR_DEV_INDEX);
    const struct nlattr* p = (const struct nlattr*)((const char*)a + NLA_ALIGN(a->nla_len));
    EXPECT_EQ(p->nla_type, RDMA_NLDEV_ATTR_PORT_INDEX);
    uint32_t port = 0;
    memcpy(&port, (const char*)p + NLA_HDRLEN, 4);
    EXPECT_EQ(port, 1u);
    EXPECT_EQ(nic.statDevIndex[0], 3u);
}

TEST(AcclNetNetlink, CounterMissingFromALaterReplyFailsTheRead) {
    FakeNic nic;
    nic.script = {{{"a", 1}, {"b", 2}}, {{"a", 5}}};
    InstallFakeNic(&nic);
    uint64_t first[ACCL_NET_MAX_RDMA] = {0}, second[ACCL_NET_MAX_RDMA] = {0};
    char names[ACCL_NET_MAX_RDMA][ACCL_NET_NAME_LEN];
    int nNames = 0;
    EXPECT_EQ(test_acclRdmaOpenAndRead("rdma0", first, second, names, &nNames), 2);
    ClearNetHooks();
}

TEST(AcclNetNetlink, UnknownDeviceFailsOpen) {
    FakeNic nic;
    InstallFakeNic(&nic);
    uint64_t first[ACCL_NET_MAX_RDMA], second[ACCL_NET_MAX_RDMA];
    char names[ACCL_NET_MAX_RDMA][ACCL_NET_NAME_LEN];
    int nNames = 0;
    EXPECT_NE(test_acclRdmaOpenAndRead("rdma9", first, second, names, &nNames), 0);
    EXPECT_EQ(nic.statRequests, 0) << "must not read another device's counters";
    ClearNetHooks();
}

// A scripted ethtool driver: the stat set can grow between reads.
struct FakeEthtool {
    std::vector<std::string> names = {"rx_bytes", "frames_rx_pripause", "tx_pripause_3_1us_count", "foo"};
    std::vector<uint64_t> values = {1, 2, 3, 4};
    int growAtGstats = -1;   // before this 1-based GSTATS, insert a stat at index 0
    int gstats = 0;
    int gdrvinfo = 0;
};
static FakeEthtool* gFakeEth = nullptr;

static int FakeEthIoctl(int, unsigned long req, void* arg) {
    if (req != SIOCETHTOOL) return -1;
    void* data = static_cast<struct ifreq*>(arg)->ifr_data;
    uint32_t cmd;
    memcpy(&cmd, data, sizeof(cmd));
    FakeEthtool* f = gFakeEth;
    if (cmd == ETHTOOL_GDRVINFO) {
        f->gdrvinfo++;
        static_cast<struct ethtool_drvinfo*>(data)->n_stats = (uint32_t)f->names.size();
    } else if (cmd == ETHTOOL_GSTRINGS) {
        auto* gs = static_cast<struct ethtool_gstrings*>(data);
        gs->len = std::min<uint32_t>(gs->len, (uint32_t)f->names.size());
        for (uint32_t i = 0; i < gs->len; i++) {
            memset(gs->data + i * ETH_GSTRING_LEN, 0, ETH_GSTRING_LEN);
            strncpy((char*)gs->data + i * ETH_GSTRING_LEN, f->names[i].c_str(), ETH_GSTRING_LEN);
        }
    } else if (cmd == ETHTOOL_GSTATS) {
        if (++f->gstats == f->growAtGstats) {
            f->names.insert(f->names.begin(), "new_stat");
            f->values.insert(f->values.begin(), 77);
            f->values[2] = 20;  // frames_rx_pripause, now at index 2
        }
        auto* st = static_cast<struct ethtool_stats*>(data);
        // Kernel behaviour (net/ethtool/ioctl.c): a stale count gets n_stats=0.
        if (st->n_stats && st->n_stats != f->names.size()) {
            st->n_stats = 0;
            return 0;
        }
        st->n_stats = (uint32_t)f->names.size();
        for (size_t i = 0; i < f->values.size(); i++) st->data[i] = f->values[i];
    } else {
        return -1;
    }
    return 0;
}

TEST(AcclNetEthtool, KeepsOnlyPauseStats) {
    EXPECT_TRUE(acclEthKeepName("frames_rx_pripause"));
    EXPECT_TRUE(acclEthKeepName("rx_pripause_3_1us_count"));
    EXPECT_TRUE(acclEthKeepName("tx_pause_ctrl_phy"));
    EXPECT_TRUE(acclEthKeepName("rx_pfc_frames"));
    EXPECT_FALSE(acclEthKeepName("rx_bytes"));
    EXPECT_FALSE(acclEthKeepName("tx_rdma_ucast_pkts"));
}

TEST(AcclNetEthtool, RefetchesNamesWhenStatCountChanges) {
    FakeEthtool eth;
    eth.growAtGstats = 2;
    gFakeEth = &eth;
    struct acclNetHooks h;
    memset(&h, 0, sizeof(h));
    h.ioctlFn = FakeEthIoctl;
    test_acclNetSetHooks(&h);
    uint64_t out[2][ACCL_NET_MAX_ETH] = {};
    char names[ACCL_NET_MAX_ETH][ETH_GSTRING_LEN + 1];
    int nNames = 0;
    int rc = test_acclEthReads("eth0", 2, out, names, &nNames);
    ClearNetHooks();
    gFakeEth = nullptr;
    ASSERT_EQ(rc, 0);
    ASSERT_EQ(nNames, 2);
    EXPECT_STREQ(names[0], "frames_rx_pripause");
    EXPECT_STREQ(names[1], "tx_pripause_3_1us_count");
    EXPECT_EQ(out[0][0], 2u);
    EXPECT_EQ(out[0][1], 3u);
    // Second read: GSTATS answered n_stats=0, so names were re-fetched and the
    // kept stats re-mapped by name before the retry.
    EXPECT_EQ(out[1][0], 20u);
    EXPECT_EQ(out[1][1], 3u);
    EXPECT_EQ(eth.gstats, 3);
    EXPECT_EQ(eth.gdrvinfo, 2);
}

// ---- Snapshot chaining through the plugin callbacks (fake NIC) ----

static std::atomic<uint64_t> gFakeSwTx{0};
static std::atomic<int> gFakeSwDevs{1};

static int FakeSwCapture(void* out, int maxDevs) {
    int n = std::min(maxDevs, gFakeSwDevs.load());
    auto* s = static_cast<RcclTelemetrySwSnapshot*>(out);
    for (int i = 0; i < n; i++) {
        memset(&s[i], 0, sizeof(s[i]));
        s[i].device_id = i;
        s[i].tx_bytes = gFakeSwTx.load();
        s[i].rx_bytes = 2 * gFakeSwTx.load();
    }
    return n;
}

static void InstallFakeNicAndSw(FakeNic* nic) {
    InstallFakeNic(nic);
    struct acclNetHooks h;
    memset(&h, 0, sizeof(h));
    h.nlOpenFn = FakeNlOpen;
    h.sendFn = FakeNlSend;
    h.recvFn = FakeNlRecv;
    h.ioctlFn = NoEthIoctl;
    h.swCaptureFn = FakeSwCapture;
    test_acclNetSetHooks(&h);
}

static std::string NetOutputPath(const std::string& dir, int rank, const char* hash) {
    char hostname[256] = {0};
    gethostname(hostname, sizeof(hostname) - 1);
    char path[1024];
    snprintf(path, sizeof(path), "%s/accl_profiler_rank%d_%s_pid%d_%s.jsonl",
             dir.c_str(), rank, hostname, (int)getpid(), hash);
    return path;
}

static std::vector<std::string> ReadLines(const std::string& path) {
    std::ifstream ifs(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(ifs, line)) lines.push_back(line);
    return lines;
}

static std::vector<std::string> CollRecords(const std::vector<std::string>& lines) {
    std::vector<std::string> recs;
    for (const auto& l : lines) {
        if (l.find("\"coll_perf\"") != std::string::npos) recs.push_back(l);
    }
    return recs;
}

static bool Has(const std::string& s, const char* needle) {
    return s.find(needle) != std::string::npos;
}

// Structural JSON check: balanced {} and [] outside strings, one top-level object.
static bool JsonBalanced(const std::string& s) {
    std::vector<char> stack;
    bool inStr = false;
    for (size_t i = 0; i < s.size(); i++) {
        char c = s[i];
        if (inStr) {
            if (c == '\\') i++;
            else if (c == '"') inStr = false;
            continue;
        }
        if (c == '"') inStr = true;
        else if (c == '{' || c == '[') stack.push_back(c);
        else if (c == '}' || c == ']') {
            if (stack.empty() || stack.back() != (c == '}' ? '{' : '[')) return false;
            stack.pop_back();
            if (stack.empty() && i + 1 != s.size()) return false;
        } else if (stack.empty()) {
            return false;
        }
        if (i + 1 < s.size() && (c == ',' || c == ':') && (s[i + 1] == ',' || s[i + 1] == '}')) return false;
    }
    return stack.empty() && !inStr && !s.empty();
}

// Launch-time half of a collective: KernelLaunch, Coll start, Coll stop.
static void* NetLaunchColl(void* ctx, uint64_t sn, size_t count = 1024) {
    ncclProfilerEventDescr_v5_t kl;
    memset(&kl, 0, sizeof(kl));
    kl.type = ncclProfileKernelLaunch;
    void* klHandle = &kl;
    EXPECT_EQ(acclPluginStartEvent(ctx, &klHandle, &kl), 0);
    EXPECT_EQ(klHandle, nullptr);
    ncclProfilerEventDescr_v5_t cd;
    MakeCollDescr(&cd, /*nChannels=*/1, sn, count);
    void* coll = nullptr;
    EXPECT_EQ(acclPluginStartEvent(ctx, &coll, &cd), 0);
    if (coll) {
        ncclProfilerEventDescr_v5_t registration = {};
        registration.type = ncclProfileNetPlugin;
        registration.parentObj = coll;
        registration.netPlugin.id = RCCL_ACCL_PROXY_BEGIN;
        void* unused = nullptr;
        EXPECT_EQ(acclPluginStartEvent(ctx, &unused, &registration), 0);
        EXPECT_EQ(acclPluginStopEvent(coll), 0);
    }
    return coll;
}

// Completion half: the one kernel channel starts and stops.
static void NetCompleteColl(void* ctx, void* coll) {
    ncclProfilerEventDescr_v5_t kd;
    MakeKernelChDescr(&kd, coll, /*channelId=*/0, /*pTimer=*/1000000);
    void* kch = nullptr;
    ASSERT_EQ(acclPluginStartEvent(ctx, &kch, &kd), 0);
    ncclProfilerEventStateArgs_v5_t sa;
    memset(&sa, 0, sizeof(sa));
    sa.kernelCh.pTimer = 1010000;
    ASSERT_EQ(acclPluginRecordEventState(kch, ncclProfilerKernelChStop, &sa), 0);
    ASSERT_EQ(acclPluginStopEvent(kch), 0);
}

TEST(AcclNetCounters, OffByDefault) {
    ScopedProfilerDir dir("net_off");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.OffByDefault",
        [&dir]() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xA1, &mask, "net", 1, 1, 0, nullptr), 0);
            EXPECT_FALSE(mask & ncclProfileKernelLaunch);
            EXPECT_FALSE(mask & ncclProfileP2p);
            void* coll = NetLaunchColl(ctx, 0);
            ASSERT_NE(coll, nullptr);
            NetCompleteColl(ctx, coll);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            auto lines = ReadLines(NetOutputPath(ProfilerDirRoot(dir), 0, "0xa1"));
            ASSERT_EQ(lines.size(), 2u);
            EXPECT_FALSE(Has(lines[0], "net_counters"));
            EXPECT_FALSE(Has(lines[1], "net_counters"));
            // Unchanged record tail and summary tail with the feature off.
            const std::string tail = "\"duration_us\":100}]}}}";
            ASSERT_GE(lines[0].size(), tail.size());
            EXPECT_EQ(lines[0].substr(lines[0].size() - tail.size()), tail);
            EXPECT_TRUE(Has(lines[1], "\"max_proxy_ops_per_coll\":256,\"complete\":true}}"));
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}}
    );
}

TEST(AcclNetCounters, NoDeviceListTurnsItOff) {
    ScopedProfilerDir dir("net_nodev");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.NoDeviceListTurnsItOff",
        []() {
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xA2, &mask, "net", 1, 1, 0, nullptr), 0);
            EXPECT_FALSE(mask & ncclProfileKernelLaunch);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"}}
    );
}

TEST(AcclNetCounters, ChainsSnapshotsAcrossCollectives) {
    ScopedProfilerDir dir("net_chain");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.ChainsSnapshotsAcrossCollectives",
        [&dir]() {
            FakeNic nic;
            InstallFakeNicAndSw(&nic);
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xB1, &mask, "net", 1, 1, 0, nullptr), 0);
            EXPECT_TRUE(mask & ncclProfileKernelLaunch);
            for (int i = 0; i < 3; i++) {
                void* coll = NetLaunchColl(ctx, i);
                ASSERT_NE(coll, nullptr);
                nic.Add("tx_rdma_ucast_bytes", 1000 * (i + 1));
                nic.Add("tx_rdma_ucast_pkts", 1);
                gFakeSwTx += 100 * (i + 1);
                NetCompleteColl(ctx, coll);
            }
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            ClearNetHooks();

            auto lines = ReadLines(NetOutputPath(ProfilerDirRoot(dir), 0, "0xb1"));
            for (const auto& l : lines) EXPECT_TRUE(JsonBalanced(l)) << l;
            auto recs = CollRecords(lines);
            ASSERT_EQ(recs.size(), 3u);
            for (int i = 0; i < 3; i++) {
                SCOPED_TRACE(recs[i]);
                EXPECT_TRUE(Has(recs[i], "\"overlapped\":false,\"unavailable\":false"));
                EXPECT_EQ(JsonNumber(recs[i], "tx_rdma_ucast_bytes"), 1000.0 * (i + 1));
                EXPECT_EQ(JsonNumber(recs[i], "tx_rdma_ucast_pkts"), 1.0);
                EXPECT_EQ(JsonNumber(recs[i], "rx_rdma_ucast_bytes"), 0.0);
                EXPECT_FALSE(Has(recs[i].substr(0, recs[i].find("snapshot_before")), "tx_rdma_retx_pkts")) << "zero symptoms are omitted";
                EXPECT_TRUE(Has(recs[i], "\"name\":\"rdma0\""));
                EXPECT_TRUE(Has(recs[i], "\"ethtool\":null"));
                EXPECT_EQ(JsonNumber(recs[i], "tx_bytes"), 100.0 * (i + 1));
                EXPECT_EQ(JsonNumber(recs[i], "rx_bytes"), 200.0 * (i + 1));
                EXPECT_EQ(recs[i].back(), '}');
            }
            // The "after" of N is the "before" of N+1.
            EXPECT_EQ(JsonNumber(recs[0], "after_ts_us"), JsonNumber(recs[1], "before_ts_us"));
            EXPECT_EQ(JsonNumber(recs[1], "after_ts_us"), JsonNumber(recs[2], "before_ts_us"));
            // Init probe, S0, then one read per collective -- no extra reads.
            EXPECT_EQ(nic.statRequests, 5);
            const std::string& summary = lines.back();
            EXPECT_TRUE(Has(summary, "\"summary\""));
            EXPECT_EQ(JsonNumber(summary, "records"), 3.0);
            EXPECT_EQ(JsonNumber(summary, "reads"), 3.0);
            EXPECT_EQ(JsonNumber(summary, "late_proxy_ops"), 0.0);
            EXPECT_EQ(JsonNumber(summary, "overlapped"), 0.0);
            EXPECT_EQ(JsonNumber(summary, "unavailable"), 0.0);
            EXPECT_EQ(JsonNumber(summary, "tx_rdma_ucast_bytes"), 6000.0);
            EXPECT_TRUE(Has(summary, "\"complete\":true"));
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}}
    );
}

TEST(AcclNetCounters, OverlappingCollectivesAreFlagged) {
    ScopedProfilerDir dir("net_overlap");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.OverlappingCollectivesAreFlagged",
        [&dir]() {
            FakeNic nic;
            InstallFakeNic(&nic);
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xB2, &mask, "net", 1, 1, 0, nullptr), 0);
            void* a = NetLaunchColl(ctx, 0);
            void* b = NetLaunchColl(ctx, 1);
            ASSERT_NE(a, nullptr);
            ASSERT_NE(b, nullptr);
            NetCompleteColl(ctx, a);
            NetCompleteColl(ctx, b);
            // A later collective launched after both completed is clean again.
            void* c = NetLaunchColl(ctx, 2);
            NetCompleteColl(ctx, c);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            ClearNetHooks();

            auto lines = ReadLines(NetOutputPath(ProfilerDirRoot(dir), 0, "0xb2"));
            auto recs = CollRecords(lines);
            ASSERT_EQ(recs.size(), 3u);
            EXPECT_TRUE(Has(recs[0], "\"overlapped\":true"));
            EXPECT_TRUE(Has(recs[1], "\"overlapped\":true"));
            EXPECT_TRUE(Has(recs[2], "\"overlapped\":false"));
            EXPECT_EQ(JsonNumber(lines.back(), "overlapped"), 2.0);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}, {"ACCL_NET_GRACE_US", "0"}}
    );
}

// Traffic no record owns (below the size filter, or P2p) lands in the next window.
TEST(AcclNetCounters, UntrackedWorkFlagsTheNextRecord) {
    ScopedProfilerDir dir("net_untracked");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.UntrackedWorkFlagsTheNextRecord",
        [&dir]() {
            FakeNic nic;
            InstallFakeNic(&nic);
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xB3, &mask, "net", 1, 1, 0, nullptr), 0);
            const size_t big = 4096;  // 16 KiB of float32
            NetCompleteColl(ctx, NetLaunchColl(ctx, 0, big));
            EXPECT_EQ(NetLaunchColl(ctx, 1, 16), nullptr);  // below the filter
            NetCompleteColl(ctx, NetLaunchColl(ctx, 2, big));
            // The P2P plan's own KernelLaunch: waits for coll 2's read, so only
            // the next window can see the P2P.
            ncclProfilerEventDescr_v5_t kl;
            memset(&kl, 0, sizeof(kl));
            kl.type = ncclProfileKernelLaunch;
            void* klHandle = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &klHandle, &kl), 0);
            ncclProfilerEventDescr_v5_t p2p;
            memset(&p2p, 0, sizeof(p2p));
            p2p.type = ncclProfileP2p;
            void* p2pHandle = &p2p;
            ASSERT_EQ(acclPluginStartEvent(ctx, &p2pHandle, &p2p), 0);
            EXPECT_EQ(p2pHandle, nullptr);
            NetCompleteColl(ctx, NetLaunchColl(ctx, 3, big));
            NetCompleteColl(ctx, NetLaunchColl(ctx, 4, big));
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            ClearNetHooks();

            auto recs = CollRecords(ReadLines(NetOutputPath(ProfilerDirRoot(dir), 0, "0xb3")));
            ASSERT_EQ(recs.size(), 4u);
            EXPECT_TRUE(Has(recs[0], "\"overlapped\":false"));
            EXPECT_TRUE(Has(recs[1], "\"overlapped\":true"));
            EXPECT_TRUE(Has(recs[2], "\"overlapped\":true"));
            EXPECT_TRUE(Has(recs[3], "\"overlapped\":false"));
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}, {"ACCL_PROFILER_MIN_SIZE_BYTES", "8192"}}
    );
}

// A failed read invalidates the two windows it bounds, and nothing else.
TEST(AcclNetCounters, ReadErrorMarksOnlyTheWindowsItBounds) {
    ScopedProfilerDir dir("net_readerr");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.ReadErrorMarksOnlyTheWindowsItBounds",
        [&dir]() {
            FakeNic nic;
            nic.failStat = 3;  // init probe, S0, then S1 fails
            InstallFakeNic(&nic);
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xB4, &mask, "net", 1, 1, 0, nullptr), 0);
            for (int i = 0; i < 3; i++) {
                void* coll = NetLaunchColl(ctx, i);
                nic.Add("tx_rdma_ucast_bytes", 500);
                NetCompleteColl(ctx, coll);
            }
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            ClearNetHooks();

            auto lines = ReadLines(NetOutputPath(ProfilerDirRoot(dir), 0, "0xb4"));
            auto recs = CollRecords(lines);
            ASSERT_EQ(recs.size(), 3u) << "a read error must never drop a record";
            EXPECT_TRUE(Has(recs[0], "\"unavailable\":true"));
            EXPECT_TRUE(Has(recs[0], "\"rdma\":null"));
            EXPECT_TRUE(Has(recs[1], "\"unavailable\":true"));
            EXPECT_TRUE(Has(recs[2], "\"unavailable\":false"));
            EXPECT_EQ(JsonNumber(recs[2], "tx_rdma_ucast_bytes"), 500.0);
            EXPECT_EQ(JsonNumber(lines.back(), "records_with_read_error"), 1.0);
            for (const auto& l : lines) EXPECT_TRUE(JsonBalanced(l)) << l;
            EXPECT_EQ(JsonNumber(lines.back(), "unavailable"), 2.0);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}}
    );
}

// A read that outlasts ACCL_NET_WAIT_MS: the next collective goes ahead
// unmeasured, and the late read is marked as overlapping it.
TEST(AcclNetCounters, SlowReadTimesOutWithoutBlocking) {
    ScopedProfilerDir dir("net_slow");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.SlowReadTimesOutWithoutBlocking",
        [&dir]() {
            FakeNic nic;
            nic.slowStat = 3;  // S1
            nic.slowMs = 200;
            InstallFakeNic(&nic);
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xB5, &mask, "net", 1, 1, 0, nullptr), 0);
            NetCompleteColl(ctx, NetLaunchColl(ctx, 0));
            auto t0 = std::chrono::steady_clock::now();
            void* b = NetLaunchColl(ctx, 1);
            auto waited = std::chrono::steady_clock::now() - t0;
            EXPECT_LT(waited, std::chrono::milliseconds(180));
            NetCompleteColl(ctx, b);
            // Once the slow read is over, the chain is clean again.
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            NetCompleteColl(ctx, NetLaunchColl(ctx, 2));
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            ClearNetHooks();

            auto recs = CollRecords(ReadLines(NetOutputPath(ProfilerDirRoot(dir), 0, "0xb5")));
            ASSERT_EQ(recs.size(), 3u);
            EXPECT_TRUE(Has(recs[0], "\"overlapped\":true"));
            EXPECT_TRUE(Has(recs[1], "\"unavailable\":true"));
            // Its "before" is stale, so it holds the previous coll's tail.
            EXPECT_TRUE(Has(recs[1], "\"rdma\":null"));
            EXPECT_TRUE(Has(recs[2], "\"overlapped\":false,\"unavailable\":false"));
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}, {"ACCL_NET_WAIT_MS", "50"}}
    );
}

TEST(AcclNetCounters, FinalizeWritesQueuedRecordsBeforeTheSummary) {
    ScopedProfilerDir dir("net_flush");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.FinalizeWritesQueuedRecordsBeforeTheSummary",
        [&dir]() {
            FakeNic nic;
            nic.slowStat = 3;  // keep the record queued into finalize
            nic.slowMs = 100;
            InstallFakeNic(&nic);
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xB6, &mask, "net", 1, 1, 0, nullptr), 0);
            void* coll = NetLaunchColl(ctx, 0);
            nic.Add("rx_rdma_ucast_bytes", 64);
            NetCompleteColl(ctx, coll);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            ClearNetHooks();

            auto lines = ReadLines(NetOutputPath(ProfilerDirRoot(dir), 0, "0xb6"));
            ASSERT_EQ(lines.size(), 2u);
            EXPECT_TRUE(Has(lines[0], "\"coll_perf\""));
            EXPECT_EQ(JsonNumber(lines[0], "rx_rdma_ucast_bytes"), 64.0);
            EXPECT_TRUE(Has(lines[1], "\"summary\""));
            EXPECT_EQ(JsonNumber(lines[1], "records"), 1.0);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}}
    );
}

TEST(AcclNetCounters, LocalRankSelectsOneDevice) {
    ScopedProfilerDir dir("net_local");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.LocalRankSelectsOneDevice",
        [&dir]() {
            FakeNic nic;
            InstallFakeNic(&nic);
            void* ctx = nullptr;
            int mask = 0;
            // rank 1 of 4 over 2 nodes: local rank 1 -> rdma1 (index 4).
            ASSERT_EQ(acclPluginInit(&ctx, 0xB7, &mask, "net", 2, 4, 1, nullptr), 0);
            NetCompleteColl(ctx, NetLaunchColl(ctx, 0));
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            ClearNetHooks();

            ASSERT_FALSE(nic.statDevIndex.empty());
            for (uint32_t idx : nic.statDevIndex) EXPECT_EQ(idx, 4u);
            auto recs = CollRecords(ReadLines(NetOutputPath(ProfilerDirRoot(dir), 1, "0xb7")));
            ASSERT_EQ(recs.size(), 1u);
            EXPECT_TRUE(Has(recs[0], "\"name\":\"rdma1\""));
            EXPECT_FALSE(Has(recs[0], "\"name\":\"rdma0\""));
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0,rdma1"}, {"ACCL_NET_DEVICE_SELECT", "local_rank"}}
    );
}

// Narrow firmware counters wrap: only the counter that went backwards is dropped.
TEST(AcclNetCounters, CounterThatWentBackwardsIsOmitted) {
    ScopedProfilerDir dir("net_wrap");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.CounterThatWentBackwardsIsOmitted",
        [&dir]() {
            FakeNic nic;
            nic.counters = {{"tx_rdma_ucast_bytes", 100}, {"tx_rdma_ucast_pkts", 0},
                            {"rx_rdma_ucast_bytes", 0}, {"rx_rdma_ucast_pkts", 0},
                            {"resp_rx_outouf_seq", 65530}};
            InstallFakeNic(&nic);
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xB8, &mask, "net", 1, 1, 0, nullptr), 0);
            void* coll = NetLaunchColl(ctx, 0);
            nic.Add("tx_rdma_ucast_bytes", 50);
            {
                std::lock_guard<std::mutex> lk(nic.mu);
                nic.counters[4].second = 4;  // 16-bit wrap
            }
            NetCompleteColl(ctx, coll);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            ClearNetHooks();

            auto recs = CollRecords(ReadLines(NetOutputPath(ProfilerDirRoot(dir), 0, "0xb8")));
            ASSERT_EQ(recs.size(), 1u);
            EXPECT_TRUE(Has(recs[0], "\"unavailable\":true"));
            EXPECT_TRUE(Has(recs[0], "counter_discontinuities"));
            EXPECT_TRUE(Has(recs[0], "\"rdma\":null"));
            EXPECT_TRUE(Has(recs[0], "rdma0/resp_rx_outouf_seq"));
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}}
    );
}

// An app that never syncs pays the launch grace once, not on every launch.
TEST(AcclNetCounters, PipelinedLaunchesStopWaitingForGrace) {
    ScopedProfilerDir dir("net_pipe");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.PipelinedLaunchesStopWaitingForGrace",
        [&dir]() {
            FakeNic nic;
            InstallFakeNic(&nic);
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xB9, &mask, "net", 1, 1, 0, nullptr), 0);
            std::vector<void*> colls;
            colls.push_back(NetLaunchColl(ctx, 0));
            auto t0 = std::chrono::steady_clock::now();
            colls.push_back(NetLaunchColl(ctx, 1));
            auto first = std::chrono::steady_clock::now() - t0;
            t0 = std::chrono::steady_clock::now();
            for (int i = 2; i < 6; i++) colls.push_back(NetLaunchColl(ctx, i));
            auto rest = std::chrono::steady_clock::now() - t0;
            EXPECT_GE(first, std::chrono::milliseconds(150));
            EXPECT_LT(rest, std::chrono::milliseconds(150));
            for (void* c : colls) NetCompleteColl(ctx, c);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            ClearNetHooks();
            auto recs = CollRecords(ReadLines(NetOutputPath(ProfilerDirRoot(dir), 0, "0xb9")));
            ASSERT_EQ(recs.size(), 6u);
            for (const auto& r : recs) EXPECT_TRUE(Has(r, "\"overlapped\":true")) << r;
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}, {"ACCL_NET_GRACE_US", "200000"}}
    );
}

// A record still waiting for its read when its comm finalizes is reported lost.
TEST(AcclNetCounters, RecordNotWrittenByFinalizeClearsComplete) {
    ScopedProfilerDir dir("net_lost");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.RecordNotWrittenByFinalizeClearsComplete",
        [&dir]() {
            FakeNic nic;
            nic.slowStat = 3;  // S1 outlasts A's finalize drain (2*50 ms + 500 ms)
            nic.slowMs = 1500;
            InstallFakeNic(&nic);
            void* a = nullptr;
            void* b = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&a, 0xBA, &mask, "netA", 1, 1, 0, nullptr), 0);
            ASSERT_EQ(acclPluginInit(&b, 0xBB, &mask, "netB", 1, 1, 0, nullptr), 0);
            NetCompleteColl(a, NetLaunchColl(a, 0));
            ASSERT_EQ(acclPluginFinalize(a), 0);
            ASSERT_EQ(acclPluginFinalize(b), 0);
            ClearNetHooks();

            auto lines = ReadLines(NetOutputPath(ProfilerDirRoot(dir), 0, "0xba"));
            ASSERT_FALSE(lines.empty());
            EXPECT_EQ(JsonNumber(lines.back(), "queued_at_finalize"), 1.0);
            EXPECT_TRUE(Has(lines.back(), "\"complete\":false"));
            EXPECT_TRUE(CollRecords(lines).empty());
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}, {"ACCL_NET_WAIT_MS", "50"}}
    );
}

// Overlapping windows are not added to the totals: they would be counted twice.
TEST(AcclNetCounters, OverlappedWindowsStayOutOfTotals) {
    ScopedProfilerDir dir("net_totals");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.OverlappedWindowsStayOutOfTotals",
        [&dir]() {
            FakeNic nic;
            InstallFakeNic(&nic);
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xC1, &mask, "net", 1, 1, 0, nullptr), 0);
            void* a = NetLaunchColl(ctx, 0);
            void* b = NetLaunchColl(ctx, 1);
            nic.Add("tx_rdma_ucast_bytes", 1000);
            NetCompleteColl(ctx, a);
            NetCompleteColl(ctx, b);
            void* c = NetLaunchColl(ctx, 2);
            nic.Add("tx_rdma_ucast_bytes", 7);
            NetCompleteColl(ctx, c);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            ClearNetHooks();
            auto lines = ReadLines(NetOutputPath(ProfilerDirRoot(dir), 0, "0xc1"));
            EXPECT_EQ(JsonNumber(lines.back(), "tx_rdma_ucast_bytes"), 7.0);
            // a and b share one "after" read.
            EXPECT_EQ(JsonNumber(lines.back(), "reads"), 2.0);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}, {"ACCL_NET_GRACE_US", "0"}}
    );
}

// A proxy op that starts after its collective completed sent in a later window.
TEST(AcclNetCounters, LateProxyOpIsCountedAndFlagsTheNextRecord) {
    ScopedProfilerDir dir("net_late");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.LateProxyOpIsCountedAndFlagsTheNextRecord",
        [&dir]() {
            FakeNic nic;
            InstallFakeNic(&nic);
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xC2, &mask, "net", 2, 2, 0, nullptr), 0);
            void* a = NetLaunchColl(ctx, 0);
            NetCompleteColl(ctx, a);  // finalized with no proxy op started
            ncclProfilerEventDescr_v5_t od;
            MakeProxyOpDescr(&od, a, 0, 1, 1, /*isSend=*/1);
            void* op = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &op, &od), 0);
            ASSERT_EQ(acclPluginStopEvent(op), 0);
            NetCompleteColl(ctx, NetLaunchColl(ctx, 1));
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            ClearNetHooks();
            auto lines = ReadLines(NetOutputPath(ProfilerDirRoot(dir), 0, "0xc2"));
            auto recs = CollRecords(lines);
            ASSERT_EQ(recs.size(), 2u);
            EXPECT_TRUE(Has(recs[1], "\"overlapped\":true"));
            EXPECT_EQ(JsonNumber(lines.back(), "late_proxy_ops"), 1.0);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}}
    );
}

// Symmetric plans (proto "") have no proxy ops and their non-head tasks never
// complete: they must not hold the chain open.
TEST(AcclNetCounters, SymmetricCollIsNotTracked) {
    ScopedProfilerDir dir("net_sym");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.SymmetricCollIsNotTracked",
        [&dir]() {
            FakeNic nic;
            InstallFakeNic(&nic);
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xC3, &mask, "net", 1, 1, 0, nullptr), 0);
            ncclProfilerEventDescr_v5_t cd;
            MakeCollDescr(&cd, /*nChannels=*/0, 0, 1024);
            cd.coll.proto = "";
            void* sym = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &sym, &cd), 0);
            ASSERT_NE(sym, nullptr);
            ASSERT_EQ(acclPluginStopEvent(sym), 0);
            NetCompleteColl(ctx, NetLaunchColl(ctx, 1));
            NetCompleteColl(ctx, NetLaunchColl(ctx, 2));
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            ClearNetHooks();
            auto recs = CollRecords(ReadLines(NetOutputPath(ProfilerDirRoot(dir), 0, "0xc3")));
            ASSERT_EQ(recs.size(), 2u);
            EXPECT_TRUE(Has(recs[0], "\"overlapped\":false")) << recs[0];
            EXPECT_TRUE(Has(recs[1], "\"overlapped\":false")) << recs[1];
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}}
    );
}

// A proxy op no record owns (P2P on another stream) flags every window it is
// live in, not just the next one.
TEST(AcclNetCounters, UntrackedProxyOpFlagsEveryWindowWhileLive) {
    ScopedProfilerDir dir("net_live");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.UntrackedProxyOpFlagsEveryWindowWhileLive",
        [&dir]() {
            FakeNic nic;
            InstallFakeNic(&nic);
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xC5, &mask, "net", 1, 1, 0, nullptr), 0);
            ncclProfilerEventDescr_v5_t od;
            MakeProxyOpDescr(&od, nullptr, 0, 1, 4, /*isSend=*/1);
            void* p2pOp = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &p2pOp, &od), 0);
            ASSERT_NE(p2pOp, nullptr);
            NetCompleteColl(ctx, NetLaunchColl(ctx, 0));
            NetCompleteColl(ctx, NetLaunchColl(ctx, 1));
            ASSERT_EQ(acclPluginStopEvent(p2pOp), 0);
            NetCompleteColl(ctx, NetLaunchColl(ctx, 2));
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            ClearNetHooks();
            auto recs = CollRecords(ReadLines(NetOutputPath(ProfilerDirRoot(dir), 0, "0xc5")));
            ASSERT_EQ(recs.size(), 3u);
            EXPECT_TRUE(Has(recs[0], "\"overlapped\":true"));
            EXPECT_TRUE(Has(recs[1], "\"overlapped\":true"));
            EXPECT_TRUE(Has(recs[2], "\"overlapped\":false"));
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}}
    );
}

// Coll start never waits: on the eager path the kernel is already running and
// on the graph path it is a HIP host callback. A pending read makes it unavailable.
TEST(AcclNetCounters, CollStartNeverBlocks) {
    ScopedProfilerDir dir("net_noblock");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.CollStartNeverBlocks",
        [&dir]() {
            FakeNic nic;
            nic.slowStat = 3;  // S1
            nic.slowMs = 300;
            InstallFakeNic(&nic);
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xC4, &mask, "net", 1, 1, 0, nullptr), 0);
            NetCompleteColl(ctx, NetLaunchColl(ctx, 0));
            ncclProfilerEventDescr_v5_t cd;
            MakeCollDescr(&cd, 1, 1, 1024);
            void* b = nullptr;
            auto t0 = std::chrono::steady_clock::now();
            ASSERT_EQ(acclPluginStartEvent(ctx, &b, &cd), 0);
            EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::milliseconds(100));
            ASSERT_EQ(acclPluginStopEvent(b), 0);
            NetCompleteColl(ctx, b);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            ClearNetHooks();
            auto recs = CollRecords(ReadLines(NetOutputPath(ProfilerDirRoot(dir), 0, "0xc4")));
            ASSERT_EQ(recs.size(), 2u);
            EXPECT_TRUE(Has(recs[0], "\"overlapped\":true"));
            EXPECT_TRUE(Has(recs[1], "\"unavailable\":true"));
            EXPECT_EQ(JsonNumber(recs[1], "launch_wait_us"), 0.0);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}}
    );
}

// A collective drained at finalize may still have been sending: the next
// window (another comm) must not be called clean.
TEST(AcclNetCounters, DroppedCollFlagsTheNextWindow) {
    ScopedProfilerDir dir("net_drop");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.DroppedCollFlagsTheNextWindow",
        [&dir]() {
            FakeNic nic;
            InstallFakeNic(&nic);
            void* a = nullptr;
            void* b = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&a, 0xC6, &mask, "netA", 1, 1, 0, nullptr), 0);
            ASSERT_EQ(acclPluginInit(&b, 0xC7, &mask, "netB", 1, 1, 0, nullptr), 0);
            ASSERT_NE(NetLaunchColl(a, 0), nullptr);  // never completes
            ASSERT_EQ(acclPluginFinalize(a), 0);
            NetCompleteColl(b, NetLaunchColl(b, 0));
            NetCompleteColl(b, NetLaunchColl(b, 1));
            ASSERT_EQ(acclPluginFinalize(b), 0);
            ClearNetHooks();
            auto recs = CollRecords(ReadLines(NetOutputPath(ProfilerDirRoot(dir), 0, "0xc7")));
            ASSERT_EQ(recs.size(), 2u);
            EXPECT_TRUE(Has(recs[0], "\"overlapped\":true"));
            EXPECT_TRUE(Has(recs[1], "\"overlapped\":false"));
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}}
    );
}

// Untracked work starting while a tracked coll runs flags that coll.
TEST(AcclNetCounters, UntrackedStartFlagsCollsInFlight) {
    ScopedProfilerDir dir("net_untr_inflight");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.UntrackedStartFlagsCollsInFlight",
        [&dir]() {
            FakeNic nic;
            InstallFakeNic(&nic);
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xD1, &mask, "net", 1, 1, 0, nullptr), 0);
            void* a = NetLaunchColl(ctx, 0, 4096);
            ncclProfilerEventDescr_v5_t cd;
            MakeCollDescr(&cd, 1, 1, 16);  // below the size filter
            void* small = &cd;
            ASSERT_EQ(acclPluginStartEvent(ctx, &small, &cd), 0);
            EXPECT_EQ(small, nullptr);
            NetCompleteColl(ctx, a);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            ClearNetHooks();
            auto recs = CollRecords(ReadLines(NetOutputPath(ProfilerDirRoot(dir), 0, "0xd1")));
            ASSERT_EQ(recs.size(), 1u);
            EXPECT_TRUE(Has(recs[0], "\"overlapped\":true"));
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}, {"ACCL_PROFILER_MIN_SIZE_BYTES", "8192"}}
    );
}

// A read that already timed out is not waited for again by the next launch.
TEST(AcclNetCounters, TimedOutReadIsNotWaitedForTwice) {
    ScopedProfilerDir dir("net_twice");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.TimedOutReadIsNotWaitedForTwice",
        [&dir]() {
            FakeNic nic;
            nic.slowStat = 3;  // S1
            nic.slowMs = 1500;
            InstallFakeNic(&nic);
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xD2, &mask, "net", 1, 1, 0, nullptr), 0);
            NetCompleteColl(ctx, NetLaunchColl(ctx, 0));
            ncclProfilerEventDescr_v5_t kl;
            memset(&kl, 0, sizeof(kl));
            kl.type = ncclProfileKernelLaunch;
            void* h = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &h, &kl), 0);  // waits ACCL_NET_WAIT_MS, gives up
            auto t0 = std::chrono::steady_clock::now();
            ASSERT_EQ(acclPluginStartEvent(ctx, &h, &kl), 0);
            EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::milliseconds(100));
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            ClearNetHooks();
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}, {"ACCL_NET_WAIT_MS", "200"}}
    );
}

// A record that completes while another read runs gets a read of its own,
// without waiting for a later launch or finalize.
TEST(AcclNetCounters, RecordQueuedDuringAReadGetsTheNextRead) {
    ScopedProfilerDir dir("net_reread");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.RecordQueuedDuringAReadGetsTheNextRead",
        [&dir]() {
            FakeNic nic;
            nic.slowStat = 3;  // S1
            nic.slowMs = 300;
            InstallFakeNic(&nic);
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xD3, &mask, "net", 1, 1, 0, nullptr), 0);
            NetCompleteColl(ctx, NetLaunchColl(ctx, 0));   // S1 starts, slow
            ncclProfilerEventDescr_v5_t cd;
            MakeCollDescr(&cd, 1, 1, 1024);
            void* b = nullptr;
            ASSERT_EQ(acclPluginStartEvent(ctx, &b, &cd), 0);  // no KernelLaunch: no wait
            ASSERT_EQ(acclPluginStopEvent(b), 0);
            NetCompleteColl(ctx, b);                          // queued during S1
            const std::string path = NetOutputPath(ProfilerDirRoot(dir), 0, "0xd3");
            size_t n = 0;
            for (int i = 0; i < 300 && n < 2; i++) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                n = CollRecords(ReadLines(path)).size();
            }
            EXPECT_EQ(n, 2u) << "second record waited for a later trigger";
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            ClearNetHooks();
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}}
    );
}

// The grace waits only for the launching comm's own collectives.
TEST(AcclNetCounters, GraceIgnoresOtherCommsCollectives) {
    ScopedProfilerDir dir("net_grace_comm");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.GraceIgnoresOtherCommsCollectives",
        [&dir]() {
            FakeNic nic;
            InstallFakeNic(&nic);
            void* a = nullptr;
            void* b = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&a, 0xD4, &mask, "netA", 1, 1, 0, nullptr), 0);
            ASSERT_EQ(acclPluginInit(&b, 0xD5, &mask, "netB", 1, 1, 0, nullptr), 0);
            void* ca = NetLaunchColl(a, 0);
            auto t0 = std::chrono::steady_clock::now();
            void* cb = NetLaunchColl(b, 0);
            EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::milliseconds(200));
            // A's own next launch does wait for A's posted coll.
            std::thread finisher([a, ca]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                NetCompleteColl(a, ca);
            });
            t0 = std::chrono::steady_clock::now();
            void* ca2 = NetLaunchColl(a, 1);
            EXPECT_GE(std::chrono::steady_clock::now() - t0, std::chrono::milliseconds(80));
            finisher.join();
            NetCompleteColl(b, cb);
            NetCompleteColl(a, ca2);
            ASSERT_EQ(acclPluginFinalize(a), 0);
            ASSERT_EQ(acclPluginFinalize(b), 0);
            ClearNetHooks();
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}, {"ACCL_NET_GRACE_US", "1000000"}}
    );
}

// launch_wait_us is the KernelLaunch wait, here a slow "before" read.
TEST(AcclNetCounters, LaunchWaitIsRecorded) {
    ScopedProfilerDir dir("net_lw");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.LaunchWaitIsRecorded",
        [&dir]() {
            FakeNic nic;
            nic.slowStat = 2;  // S0
            nic.slowMs = 200;
            InstallFakeNic(&nic);
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xD6, &mask, "net", 1, 1, 0, nullptr), 0);
            NetCompleteColl(ctx, NetLaunchColl(ctx, 0));
            NetCompleteColl(ctx, NetLaunchColl(ctx, 1));
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            ClearNetHooks();
            auto recs = CollRecords(ReadLines(NetOutputPath(ProfilerDirRoot(dir), 0, "0xd6")));
            ASSERT_EQ(recs.size(), 2u);
            EXPECT_GE(JsonNumber(recs[0], "launch_wait_us"), 150000.0);
            EXPECT_LT(JsonNumber(recs[1], "launch_wait_us"), 150000.0);
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}}
    );
}

// A change in the number of SW telemetry devices voids the SW deltas only.
TEST(AcclNetCounters, SwDeviceCountChangeDropsSwDeltas) {
    ScopedProfilerDir dir("net_sw");
    RUN_ISOLATED_TEST_WITH_ENV(
        "AcclNetCounters.SwDeviceCountChangeDropsSwDeltas",
        [&dir]() {
            FakeNic nic;
            InstallFakeNicAndSw(&nic);
            void* ctx = nullptr;
            int mask = 0;
            ASSERT_EQ(acclPluginInit(&ctx, 0xD7, &mask, "net", 1, 1, 0, nullptr), 0);
            void* c = NetLaunchColl(ctx, 0);
            gFakeSwDevs = 2;
            NetCompleteColl(ctx, c);
            ASSERT_EQ(acclPluginFinalize(ctx), 0);
            ClearNetHooks();
            auto recs = CollRecords(ReadLines(NetOutputPath(ProfilerDirRoot(dir), 0, "0xd7")));
            ASSERT_EQ(recs.size(), 1u);
            EXPECT_TRUE(Has(recs[0], "\"sw\":null"));
            EXPECT_TRUE(Has(recs[0], "\"unavailable\":false"));
        },
        {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
         {"ACCL_NET_DEVICES", "rdma0"}}
    );
}


static void NetProtocolEvent(void* ctx, int64_t id, void* parent = nullptr, void* data = nullptr) {
    ncclProfilerEventDescr_v5_t e = {};
    e.type = ncclProfileNetPlugin; e.parentObj = parent;
    e.netPlugin.id = id; e.netPlugin.data = data;
    void* unused = nullptr;
    ASSERT_EQ(acclPluginStartEvent(ctx, &unused, &e), 0);
}

TEST(AcclNetAccuracy, RegisteredProxyMustCompleteBeforeAfterRead) {
    ScopedProfilerDir dir("net_registered_tail");
    RUN_ISOLATED_TEST_WITH_ENV("AcclNetAccuracy.RegisteredProxyMustCompleteBeforeAfterRead", [&dir]() {
        FakeNic nic; InstallFakeNic(&nic);
        void* ctx = nullptr; int mask = 0;
        ASSERT_EQ(acclPluginInit(&ctx, 0xE1, &mask, "net", 2, 2, 0, nullptr), 0);
        // Register before Coll stop, exactly as the core does while uploading ops.
        ncclProfilerEventDescr_v5_t kl = {}; kl.type = ncclProfileKernelLaunch;
        void* unused = nullptr; ASSERT_EQ(acclPluginStartEvent(ctx, &unused, &kl), 0);
        ncclProfilerEventDescr_v5_t cd; MakeCollDescr(&cd, 1, 0, 1024);
        void* coll = nullptr; ASSERT_EQ(acclPluginStartEvent(ctx, &coll, &cd), 0);
        NetProtocolEvent(ctx, RCCL_ACCL_PROXY_BEGIN, coll);
        NetProtocolEvent(ctx, RCCL_ACCL_PROXY_ADD, coll);
        ASSERT_EQ(acclPluginStopEvent(coll), 0);
        NetCompleteColl(ctx, coll);
        const auto path = NetOutputPath(ProfilerDirRoot(dir), 0, "0xe1");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        EXPECT_TRUE(CollRecords(ReadLines(path)).empty());
        ncclProfilerEventDescr_v5_t od; MakeProxyOpDescr(&od, coll, 0, 1, 1, 1);
        void* op = nullptr; ASSERT_EQ(acclPluginStartEvent(ctx, &op, &od), 0);
        nic.Add("tx_rdma_ucast_bytes", 4096);
        ASSERT_EQ(acclPluginStopEvent(op), 0);
        ASSERT_EQ(acclPluginFinalize(ctx), 0); ClearNetHooks();
        auto records = CollRecords(ReadLines(path));
        ASSERT_EQ(records.size(), 1u);
        EXPECT_TRUE(Has(records[0], "\"overlapped\":false,\"unavailable\":false"));
        EXPECT_TRUE(Has(records[0], "\"completion_verified\":true"));
        EXPECT_EQ(JsonNumber(records[0], "tx_rdma_ucast_bytes"), 4096.0);
    }, {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"}, {"ACCL_NET_DEVICES", "rdma0"}});
}

static int gBoundaryCalls = 0;
static int FakeBoundary(void*, int valid) { gBoundaryCalls++; return valid ? 0 : 1; }

TEST(AcclNetAccuracy, StrictBoundaryCoordinatesBothSidesAndReusesSnapshot) {
    ScopedProfilerDir dir("net_strict_boundary");
    RUN_ISOLATED_TEST_WITH_ENV("AcclNetAccuracy.StrictBoundaryCoordinatesBothSidesAndReusesSnapshot", [&dir]() {
        FakeNic nic; InstallFakeNic(&nic); gBoundaryCalls = 0;
        void* ctx = nullptr; int mask = 0;
        ASSERT_EQ(acclPluginInit(&ctx, 0xE2, &mask, "net", 1, 1, 0, nullptr), 0);
        rcclAcclBoundaryInfo info = {1, 1, nullptr, FakeBoundary};
        NetProtocolEvent(ctx, RCCL_ACCL_BOUNDARY_INFO, nullptr, &info);
        auto* first = NetLaunchColl(ctx, 0);
        nic.Add("tx_rdma_ucast_bytes", 4096); NetCompleteColl(ctx, first);
        const auto path = NetOutputPath(ProfilerDirRoot(dir), 0, "0xe2");
        EXPECT_TRUE(CollRecords(ReadLines(path)).empty());
        NetProtocolEvent(ctx, RCCL_ACCL_BOUNDARY_INFO, nullptr, &info);
        auto* second = NetLaunchColl(ctx, 1);
        nic.Add("tx_rdma_ucast_bytes", 8192); NetCompleteColl(ctx, second);
        NetProtocolEvent(ctx, RCCL_ACCL_BOUNDARY_FINAL);
        ASSERT_EQ(acclPluginFinalize(ctx), 0); ClearNetHooks();
        EXPECT_EQ(gBoundaryCalls, 6);
        auto records = CollRecords(ReadLines(path)); ASSERT_EQ(records.size(), 2u);
        for (const auto& r : records) {
            EXPECT_TRUE(Has(r, "\"coordinated\":true"));
            EXPECT_TRUE(Has(r, "\"unavailable\":false"));
            EXPECT_TRUE(Has(r, "snapshot_before")); EXPECT_TRUE(Has(r, "snapshot_after"));
        }
        EXPECT_EQ(JsonNumber(records[0], "tx_rdma_ucast_bytes"), 4096.0);
        EXPECT_EQ(JsonNumber(records[1], "tx_rdma_ucast_bytes"), 8192.0);
        EXPECT_EQ(JsonNumber(records[0], "after_ts_us"), JsonNumber(records[1], "before_ts_us"));
    }, {{"ACCL_PROFILER_OUTPUT_DIR", dir.path()}, {"ACCL_NET_COUNTERS", "1"},
        {"ACCL_NET_STRICT", "1"}, {"ACCL_NET_DEVICES", "rdma0"}});
}


TEST(AcclNetAccuracy, StrictSnapshotRetainsLateHardwareCounterUpdates) {
    ScopedProfilerDir dir("net_hardware_settle");
    RUN_ISOLATED_TEST_WITH_ENV("AcclNetAccuracy.StrictSnapshotRetainsLateHardwareCounterUpdates", [&dir]() {
        FakeNic nic;
        CounterList base = nic.counters, first = base, settled = base;
        first[0].second = settled[0].second = 4096;
        settled[4].second = 5;
        // Probe, two stable baseline reads, then an error counter that becomes
        // visible only on the second after-read and settles on the third.
        nic.script = {base, base, base, first, settled, settled};
        nic.counters = settled;
        InstallFakeNic(&nic); void* ctx = nullptr; int mask = 0;
        ASSERT_EQ(acclPluginInit(&ctx, 0xE3, &mask, "net", 1, 1, 0, nullptr), 0);
        rcclAcclBoundaryInfo info = {1, 1, nullptr, FakeBoundary};
        NetProtocolEvent(ctx, RCCL_ACCL_BOUNDARY_INFO, nullptr, &info);
        NetCompleteColl(ctx, NetLaunchColl(ctx, 0));
        NetProtocolEvent(ctx, RCCL_ACCL_BOUNDARY_FINAL);
        ASSERT_EQ(acclPluginFinalize(ctx), 0); ClearNetHooks();
        auto records = CollRecords(ReadLines(NetOutputPath(ProfilerDirRoot(dir),0,"0xe3")));
        ASSERT_EQ(records.size(),1u);
        EXPECT_TRUE(Has(records[0],"\"unavailable\":false"));
        EXPECT_EQ(JsonNumber(records[0],"tx_rdma_retx_pkts"),5.0);
        EXPECT_TRUE(Has(records[0],"\"read_attempts\":3"));
    }, {{"ACCL_PROFILER_OUTPUT_DIR",dir.path()}, {"ACCL_NET_COUNTERS","1"},
        {"ACCL_NET_STRICT","1"}, {"ACCL_NET_DEVICES","rdma0"}});
}

TEST(AcclNetAccuracy, StrictSnapshotRejectsCountersThatNeverSettle) {
    ScopedProfilerDir dir("net_hardware_unstable");
    RUN_ISOLATED_TEST_WITH_ENV("AcclNetAccuracy.StrictSnapshotRejectsCountersThatNeverSettle", [&dir]() {
        FakeNic nic;
        for (int i=0;i<40;i++) { auto values=nic.counters; values[0].second=i; nic.script.push_back(values); }
        InstallFakeNic(&nic); void* ctx=nullptr; int mask=0;
        ASSERT_EQ(acclPluginInit(&ctx,0xE4,&mask,"net",1,1,0,nullptr),0);
        rcclAcclBoundaryInfo info={1,1,nullptr,FakeBoundary};
        NetProtocolEvent(ctx,RCCL_ACCL_BOUNDARY_INFO,nullptr,&info);
        NetCompleteColl(ctx,NetLaunchColl(ctx,0));
        NetProtocolEvent(ctx,RCCL_ACCL_BOUNDARY_FINAL);
        ASSERT_EQ(acclPluginFinalize(ctx),0); ClearNetHooks();
        auto records=CollRecords(ReadLines(NetOutputPath(ProfilerDirRoot(dir),0,"0xe4")));
        ASSERT_EQ(records.size(),1u);
        EXPECT_TRUE(Has(records[0],"\"unavailable\":true"));
        EXPECT_TRUE(Has(records[0],"\"stable\":false"));
    }, {{"ACCL_PROFILER_OUTPUT_DIR",dir.path()}, {"ACCL_NET_COUNTERS","1"},
        {"ACCL_NET_STRICT","1"}, {"ACCL_NET_DEVICES","rdma0"}});
}


TEST(AcclNetAccuracy, StrictTimeoutInvalidatesTheReadBeforeAnyNextCollStart) {
    ScopedProfilerDir dir("net_strict_timeout_parent");
    RUN_ISOLATED_TEST_WITH_ENV("AcclNetAccuracy.StrictTimeoutInvalidatesTheReadBeforeAnyNextCollStart", [&dir]() {
        FakeNic nic; InstallFakeNic(&nic); void* ctx=nullptr; int mask=0;
        ASSERT_EQ(acclPluginInit(&ctx,0xE5,&mask,"net",1,1,0,nullptr),0);
        rcclAcclBoundaryInfo info={1,1,nullptr,FakeBoundary};
        NetProtocolEvent(ctx,RCCL_ACCL_BOUNDARY_INFO,nullptr,&info);
        auto* coll=NetLaunchColl(ctx,0); nic.Add("tx_rdma_ucast_bytes",100);
        NetCompleteColl(ctx,coll);
        { std::lock_guard<std::mutex> lock(nic.mu); nic.slowStat=4; nic.slowMs=150; }
        ncclProfilerEventDescr_v5_t kl={}; kl.type=ncclProfileKernelLaunch;
        void* unused=nullptr; ASSERT_EQ(acclPluginStartEvent(ctx,&unused,&kl),0);
        // Another rank could now start traffic. Do not deliver a local Coll
        // start: that callback must not be needed to invalidate the old job.
        nic.Add("tx_rdma_ucast_bytes",4096);
        std::this_thread::sleep_for(std::chrono::milliseconds(220));
        NetProtocolEvent(ctx,RCCL_ACCL_BOUNDARY_FINAL);
        ASSERT_EQ(acclPluginFinalize(ctx),0); ClearNetHooks();
        auto records=CollRecords(ReadLines(NetOutputPath(ProfilerDirRoot(dir),0,"0xe5")));
        ASSERT_EQ(records.size(),1u);
        EXPECT_TRUE(Has(records[0],"\"unavailable\":true"));
    }, {{"ACCL_PROFILER_OUTPUT_DIR",dir.path()}, {"ACCL_NET_COUNTERS","1"},
        {"ACCL_NET_STRICT","1"}, {"ACCL_NET_WAIT_MS","10"}, {"ACCL_NET_DEVICES","rdma0"}});
}

} // namespace RcclUnitTesting
