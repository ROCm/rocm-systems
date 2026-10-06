/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Tests for communicator-init active diagnostics (NCCL_RUN_DIAGNOSTICS, src/diagnostics.cc,
// src/diagnostics/p2p.cc). The report is written to stdout of the process hosting rank 0 with a
// "<host>:<pid> NCCL DIAG " prefix, so every case captures stdout around communicator creation and
// asserts on the report lines only.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>
#include <rccl/rccl.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "common/ProcessIsolatedTestRunner.hpp"
#include "common/ResourceGuards.hpp"
#include "common/TestChecks.hpp"

namespace RcclUnitTesting
{

// Covers HIP init, ncclCommInitAll on up to 8 GPUs, and the diagnostics exchange itself (about 25 s per case
// on 8x MI355X). The pre-checkin CI entries (ci-precheckin.json) allow 240 s for a TEST_F with one case and
// 300 s for one with two cases, so this must stay at 120 s or below.
static constexpr int kDiagTimeoutSeconds = 120;
static constexpr int kMaxGpus            = 8;
static constexpr size_t kAllReduceElems  = 1 << 20;

static const char* const kDiagPrefix  = "NCCL DIAG ";
static const char* const kDiagHeader  = "NCCL DIAG === NCCL Diagnostics ===";
static const char* const kDiagSummary
    = "NCCL DIAG [OK]   p2p: verified P2P access in both directions between every GPU pair";
static const char* const kDiagPartialSummary = "GPU-to-GPU peer accesses passed verification";

// Report lines that indicate a failed or incomplete check (src/diagnostics.cc, src/diagnostics/p2p.cc).
static const char* const kDiagFailureMarkers[] = {
    "transport detect returned",
    "p2p: check returned",
    "p2p: setup failed",
    "p2p: destination buffer unavailable",
    "p2p: local CUDA setup failed",
    "p2p: peer-memory import failed",
    "p2p: write mismatch",
    "p2p: read mismatch",
    "p2p: topology check failed",
    "p2p: launch/check failed",
    "p2p: resource cleanup failed",
};

struct DiagReport
{
    std::vector<std::string> lines;  // Report lines with the "<host>:<pid> " prefix stripped.

    int count(const std::string& needle) const
    {
        return static_cast<int>(std::count_if(lines.begin(), lines.end(), [&](const std::string& l) {
            return l.find(needle) != std::string::npos;
        }));
    }

    std::vector<std::string> failures() const
    {
        std::vector<std::string> out;
        for(const auto& l : lines)
            for(const char* m : kDiagFailureMarkers)
                if(l.find(m) != std::string::npos)
                {
                    out.push_back(l);
                    break;
                }
        return out;
    }

    std::string dump() const
    {
        std::ostringstream os;
        for(const auto& l : lines)
            os << "  " << l << "\n";
        return os.str();
    }
};

static DiagReport parseDiagReport(const std::string& captured)
{
    DiagReport report;
    std::istringstream is(captured);
    std::string line;
    while(std::getline(is, line))
    {
        const size_t pos = line.find(kDiagPrefix);
        if(pos != std::string::npos)
            report.lines.push_back(line.substr(pos));
    }
    return report;
}

// Directed edge count (one peer access per direction) from the [OK] p2p summary, or -1.
static int okEdgeCount(const std::string& line)
{
    static const std::regex re("p2p: verified P2P access in both directions between every GPU pair "
                               "\\(([0-9]+) peer accesses\\)");
    std::smatch m;
    return std::regex_search(line, m, re) ? std::stoi(m[1].str()) : -1;
}

static int usableGpus()
{
    int devCount = 0;
    if(hipGetDeviceCount(&devCount) != hipSuccess)
        return 0;
    return std::min(devCount, kMaxGpus);
}

static std::vector<int> firstDevices(int n)
{
    std::vector<int> devices(n);
    std::iota(devices.begin(), devices.end(), 0);
    return devices;
}

// True when every pair of the devices has a direct (one-hop) XGMI link. RCCL's topology makes every such pair
// P2P-eligible, so the check must test all of them. On other topologies the eligible pairs depend on the host
// (PCIe layout, P2P level), and a pair may legitimately be left out.
static bool xgmiFullMesh(const std::vector<int>& devices)
{
    for(int a : devices)
        for(int b : devices)
        {
            if(a == b)
                continue;
            uint32_t linkType = 0;
            uint32_t hops     = 0;
            const hipError_t err = hipExtGetLinkTypeAndHopCount(a, b, &linkType, &hops);
            if(err != hipSuccess)
            {
                ADD_FAILURE() << "hipExtGetLinkTypeAndHopCount(" << a << ", " << b << "): " << hipGetErrorString(err);
                return false;
            }
            if(linkType != HSA_AMD_LINK_INFO_TYPE_XGMI || hops != 1)
                return false;
        }
    return true;
}

// Checks the p2p summary lines of a report covering one communicator per device group (groups of equal size).
// On a full XGMI mesh each communicator verifies all of its N * (N - 1) directed pairs. Elsewhere only the tested
// edges are known: each summary must still be [OK] and bounded by the pair count, and a communicator without
// eligible pairs prints no summary. A report with no summary at all tested nothing, so the case is skipped.
static void expectEdgeSummaries(const DiagReport& report, const std::vector<std::vector<int>>& groups)
{
    const int nRanks  = static_cast<int>(groups.front().size());
    const int pairs   = nRanks * (nRanks - 1);
    const int nGroups = static_cast<int>(groups.size());
    const bool full   = std::all_of(groups.begin(), groups.end(), xgmiFullMesh);

    EXPECT_EQ(report.count(kDiagPartialSummary), 0) << "partial p2p summary:\n" << report.dump();
    if(full)
        ASSERT_EQ(report.count(kDiagSummary), nGroups) << report.dump();
    else if(report.count(kDiagSummary) == 0)
        GTEST_SKIP() << "no P2P-eligible pair outside a full XGMI mesh, nothing tested:\n" << report.dump();
    else
        EXPECT_LE(report.count(kDiagSummary), nGroups) << report.dump();
    for(const auto& l : report.lines)
    {
        if(l.find(kDiagSummary) == std::string::npos)
            continue;
        const int edges = okEdgeCount(l);
        if(full)
            EXPECT_EQ(edges, pairs) << l;
        else
        {
            EXPECT_GT(edges, 0) << l;
            EXPECT_LE(edges, pairs) << l;
        }
    }
}

// The parent test environment (e.g. CI categories) may carry these; each case sets what it needs.
// cuMem is pinned off rather than unset: the default auto-enables it on gfx1250, and the check cannot map a
// same-process peer buffer through cuMem on ROCm, so every cross-device edge would report an import failure.
static void clearDiagEnv()
{
    unsetenv("NCCL_RUN_DIAGNOSTICS");
    unsetenv("NCCL_RUN_RAS_DIAGNOSTICS");
    unsetenv("NCCL_P2P_DISABLE");
    unsetenv("NCCL_P2P_LEVEL");
    setenv("NCCL_CUMEM_ENABLE", "0", 1);
}

// Creates nGpus communicators on devices 0..nGpus-1 while capturing stdout. Callers wrap the call in
// ASSERT_NO_FATAL_FAILURE so that a failed init stops the case before the communicators are used.
static void initAllCaptured(std::vector<ncclComm_t>& comms, int nGpus, std::string& captured)
{
    std::vector<int> devices = firstDevices(nGpus);
    comms.assign(nGpus, nullptr);
    testing::internal::CaptureStdout();
    const ncclResult_t res = ncclCommInitAll(comms.data(), nGpus, devices.data());
    captured               = testing::internal::GetCapturedStdout();
    ASSERT_EQ(res, ncclSuccess) << "ncclCommInitAll: " << ncclGetErrorString(res);
}

// Destroys the communicators when the returned guards go out of scope.
static std::vector<RCCLTestGuards::NcclCommAutoGuard> guardComms(const std::vector<ncclComm_t>& comms)
{
    std::vector<RCCLTestGuards::NcclCommAutoGuard> guards;
    guards.reserve(comms.size());
    for(ncclComm_t c : comms)
        guards.push_back(RCCLTestGuards::makeCommAutoGuard(c));
    return guards;
}

// Sum-AllReduce of (rank + 1) on every communicator; checks the full result on every device.
static void checkAllReduce(const std::vector<ncclComm_t>& comms)
{
    const int n = static_cast<int>(comms.size());
    std::vector<float*> send(n, nullptr), recv(n, nullptr);
    std::vector<hipStream_t> streams(n, nullptr);
    std::vector<RCCLTestGuards::DeviceBufferAutoGuard> bufGuards;
    std::vector<RCCLTestGuards::HipStreamAutoGuard> streamGuards;
    bufGuards.reserve(2 * n);
    streamGuards.reserve(n);

    for(int i = 0; i < n; ++i)
    {
        int dev = -1;
        ASSERT_EQ(ncclCommCuDevice(comms[i], &dev), ncclSuccess);
        HIP_CHECK(hipSetDevice(dev));
        HIP_CHECK(hipStreamCreate(&streams[i]));
        streamGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(streams[i]));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&send[i]), kAllReduceElems * sizeof(float)));
        bufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(send[i]));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&recv[i]), kAllReduceElems * sizeof(float)));
        bufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(recv[i]));
        std::vector<float> host(kAllReduceElems, static_cast<float>(i + 1));
        HIP_CHECK(hipMemcpy(send[i], host.data(), kAllReduceElems * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(recv[i], 0, kAllReduceElems * sizeof(float)));
    }

    ASSERT_EQ(ncclGroupStart(), ncclSuccess);
    ncclResult_t collRes = ncclSuccess;
    for(int i = 0; i < n && collRes == ncclSuccess; ++i)
        collRes = ncclAllReduce(send[i], recv[i], kAllReduceElems, ncclFloat, ncclSum, comms[i], streams[i]);
    ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
    ASSERT_EQ(collRes, ncclSuccess) << "ncclAllReduce: " << ncclGetErrorString(collRes);

    const float expected = static_cast<float>(n * (n + 1) / 2);
    std::vector<float> host(kAllReduceElems);
    for(int i = 0; i < n; ++i)
    {
        HIP_CHECK(hipStreamSynchronize(streams[i]));
        HIP_CHECK(hipMemcpy(host.data(), recv[i], kAllReduceElems * sizeof(float), hipMemcpyDeviceToHost));
        const auto bad = std::find_if(host.begin(), host.end(), [&](float v) { return v != expected; });
        ASSERT_EQ(bad, host.end()) << "rank " << i << ": index " << (bad - host.begin()) << " expected "
                                   << expected << " got " << *bad;
    }
}

struct DiagCase
{
    const char* name;
    int minGpus;
    std::function<void()> body;
};

// Runs, each in its own process, the cases that fit the visible GPU count. The GPU count is checked here in the
// parent: RUN_ISOLATED_TESTS-style runs end in EXPECT_TRUE(), so a GTEST_SKIP() inside an isolated case is reported
// as a pass. When no case fits, gtest reports a real skip.
static void runDiagCases(const std::vector<DiagCase>& cases)
{
    const int nGpus = usableGpus();
    std::string notRun;
    int registered = 0;
    for(const DiagCase& c : cases)
    {
        if(nGpus < c.minGpus)
        {
            notRun += std::string(" ") + c.name + " (>= " + std::to_string(c.minGpus) + " GPUs)";
            continue;
        }
        ProcessIsolatedTestRunner::registerTest(ProcessIsolatedTestRunner::TestConfig(c.name, c.body)
                                                    .withNumGpus(kMaxGpus)
                                                    .withTimeout(std::chrono::seconds(kDiagTimeoutSeconds)));
        ++registered;
    }
    if(registered == 0)
        GTEST_SKIP() << nGpus << " usable GPUs, not run:" << notRun;
    if(!notRun.empty())
        std::cout << "[ INFO     ] " << nGpus << " usable GPUs, not run:" << notRun << std::endl;
    EXPECT_TRUE(ProcessIsolatedTestRunner::executeAllTests()) << "One or more isolated tests failed";
}

class Diagnostics : public ::testing::Test
{
    // All work runs under process isolation: NCCL params are cached on first read.
};

// Without NCCL_RUN_DIAGNOSTICS (or with 0) communicator init prints no report.
TEST_F(Diagnostics, DisabledByDefault)
{
    auto body = [](const char* value) {
        return [value]() {
            clearDiagEnv();
            if(value != nullptr)
                setenv("NCCL_RUN_DIAGNOSTICS", value, 1);
            const int nGpus = usableGpus();
            std::vector<ncclComm_t> comms;
            std::string out;
            ASSERT_NO_FATAL_FAILURE(initAllCaptured(comms, nGpus, out));
            const auto commGuards   = guardComms(comms);
            const DiagReport report = parseDiagReport(out);
            EXPECT_TRUE(report.lines.empty()) << "unexpected report lines:\n" << report.dump();
        };
    };
    runDiagCases({{"Unset", 2, body(nullptr)}, {"Zero", 2, body("0")}});
}

// Full single-node communicator: one header, an [OK] summary covering every eligible directed pair
// (all of them on a full XGMI mesh), one completion line naming the rank count, and no failure lines.
TEST_F(Diagnostics, AllDirectedEdgesVerified)
{
    runDiagCases({{"AllDirectedEdgesVerified", 2, []() {
        clearDiagEnv();
        setenv("NCCL_RUN_DIAGNOSTICS", "1", 1);
        const int nGpus = usableGpus();

        std::vector<ncclComm_t> comms;
        std::string out;
        ASSERT_NO_FATAL_FAILURE(initAllCaptured(comms, nGpus, out));
        const auto commGuards   = guardComms(comms);
        const DiagReport report = parseDiagReport(out);

        EXPECT_EQ(report.count(kDiagHeader), 1) << report.dump();
        expectEdgeSummaries(report, {firstDevices(nGpus)});
        const std::string done = "across " + std::to_string(nGpus) + " ranks";
        EXPECT_EQ(report.count("NCCL diagnostics completed in"), 1) << report.dump();
        EXPECT_EQ(report.count(done), 1) << report.dump();
        EXPECT_TRUE(report.failures().empty()) << "failure lines:\n" << report.dump();
    }}});
}

// Edge count follows the communicator size (N * (N - 1) directed edges for N local GPUs on a full XGMI mesh).
TEST_F(Diagnostics, EdgeCountFollowsCommSize)
{
    auto body = [](int nWanted) {
        return [nWanted]() {
            clearDiagEnv();
            setenv("NCCL_RUN_DIAGNOSTICS", "1", 1);
            std::vector<ncclComm_t> comms;
            std::string out;
            ASSERT_NO_FATAL_FAILURE(initAllCaptured(comms, nWanted, out));
            const auto commGuards   = guardComms(comms);
            const DiagReport report = parseDiagReport(out);
            EXPECT_EQ(report.count(kDiagHeader), 1) << report.dump();
            expectEdgeSummaries(report, {firstDevices(nWanted)});
            EXPECT_TRUE(report.failures().empty()) << report.dump();
        };
    };
    runDiagCases({{"TwoGpus", 2, body(2)}, {"FourGpus", 4, body(4)}});
}

// Diagnostics are informational and leave the communicator usable: collectives issued after a
// diagnosed init produce correct results.
TEST_F(Diagnostics, CommUsableAfterDiagnostics)
{
    runDiagCases({{"CommUsableAfterDiagnostics", 2, []() {
        clearDiagEnv();
        setenv("NCCL_RUN_DIAGNOSTICS", "1", 1);
        const int nGpus = usableGpus();
        std::vector<ncclComm_t> comms;
        std::string out;
        ASSERT_NO_FATAL_FAILURE(initAllCaptured(comms, nGpus, out));
        const auto commGuards = guardComms(comms);
        EXPECT_EQ(parseDiagReport(out).count(kDiagHeader), 1);
        checkAllReduce(comms);
    }}});
}

// The report is produced at every communicator initialization: a re-created communicator and the
// children of ncclCommSplit each print their own report.
TEST_F(Diagnostics, RunsAtEveryCommInit)
{
    auto reinit = []() {
        clearDiagEnv();
        setenv("NCCL_RUN_DIAGNOSTICS", "1", 1);
        const int nGpus = usableGpus();
        for(int round = 0; round < 2; ++round)
        {
            SCOPED_TRACE("round " + std::to_string(round));
            std::vector<ncclComm_t> comms;
            std::string out;
            ASSERT_NO_FATAL_FAILURE(initAllCaptured(comms, nGpus, out));
            const auto commGuards   = guardComms(comms);
            const DiagReport report = parseDiagReport(out);
            EXPECT_EQ(report.count(kDiagHeader), 1) << report.dump();
            expectEdgeSummaries(report, {firstDevices(nGpus)});
            EXPECT_TRUE(report.failures().empty()) << report.dump();
        }
    };

    // Splits an even number of GPUs into two equal halves.
    auto split = []() {
        clearDiagEnv();
        setenv("NCCL_RUN_DIAGNOSTICS", "1", 1);
        const int nGpus = usableGpus() & ~1;

        std::vector<ncclComm_t> parents;
        std::string out;
        ASSERT_NO_FATAL_FAILURE(initAllCaptured(parents, nGpus, out));
        const auto parentGuards = guardComms(parents);
        EXPECT_EQ(parseDiagReport(out).count(kDiagHeader), 1);

        std::vector<ncclComm_t> children(nGpus, nullptr);
        testing::internal::CaptureStdout();
        ncclResult_t res = ncclGroupStart();
        for(int i = 0; i < nGpus && res == ncclSuccess; ++i)
            res = ncclCommSplit(parents[i], i % 2, i, &children[i], nullptr);
        const ncclResult_t endRes = ncclGroupEnd();
        out                       = testing::internal::GetCapturedStdout();
        ASSERT_EQ(res, ncclSuccess) << "ncclCommSplit: " << ncclGetErrorString(res);
        ASSERT_EQ(endRes, ncclSuccess) << "ncclGroupEnd: " << ncclGetErrorString(endRes);
        const auto childGuards = guardComms(children);

        const DiagReport report = parseDiagReport(out);
        const int half          = nGpus / 2;
        std::vector<std::vector<int>> colors(2);
        for(int i = 0; i < nGpus; ++i)
            colors[i % 2].push_back(i);
        EXPECT_EQ(report.count(kDiagHeader), 2) << report.dump();
        expectEdgeSummaries(report, colors);
        EXPECT_EQ(report.count("across " + std::to_string(half) + " ranks"), 2) << report.dump();
        EXPECT_TRUE(report.failures().empty()) << report.dump();
    };

    runDiagCases({{"ReinitSameProcess", 2, reinit}, {"CommSplit", 4, split}});
}

// With P2P disabled there are no topology-eligible P2P edges. The run still completes and reports
// no failures; it prints no p2p summary line because nothing was tested.
TEST_F(Diagnostics, P2pDisabledReportsNoEdges)
{
    runDiagCases({{"P2pDisabledReportsNoEdges", 2, []() {
        clearDiagEnv();
        setenv("NCCL_RUN_DIAGNOSTICS", "1", 1);
        setenv("NCCL_P2P_DISABLE", "1", 1);
        const int nGpus = usableGpus();
        std::vector<ncclComm_t> comms;
        std::string out;
        ASSERT_NO_FATAL_FAILURE(initAllCaptured(comms, nGpus, out));
        const auto commGuards   = guardComms(comms);
        const DiagReport report = parseDiagReport(out);
        EXPECT_EQ(report.count(kDiagHeader), 1) << report.dump();
        EXPECT_EQ(report.count("NCCL diagnostics completed in"), 1) << report.dump();
        EXPECT_EQ(report.count(kDiagSummary), 0) << report.dump();
        EXPECT_EQ(report.count(kDiagPartialSummary), 0) << report.dump();
        EXPECT_TRUE(report.failures().empty()) << report.dump();
        checkAllReduce(comms);
    }}});
}

// The report goes to stdout only; stderr carries no "NCCL DIAG" lines.
TEST_F(Diagnostics, ReportOnStdoutOnly)
{
    runDiagCases({{"ReportOnStdoutOnly", 2, []() {
        clearDiagEnv();
        setenv("NCCL_RUN_DIAGNOSTICS", "1", 1);
        const int nGpus = usableGpus();
        std::vector<int> devices = firstDevices(nGpus);
        std::vector<ncclComm_t> comms(nGpus, nullptr);
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        const ncclResult_t res = ncclCommInitAll(comms.data(), nGpus, devices.data());
        const std::string err  = testing::internal::GetCapturedStderr();
        const std::string out  = testing::internal::GetCapturedStdout();
        ASSERT_EQ(res, ncclSuccess) << "ncclCommInitAll: " << ncclGetErrorString(res);
        const auto commGuards = guardComms(comms);
        EXPECT_EQ(parseDiagReport(out).count(kDiagHeader), 1);
        EXPECT_TRUE(parseDiagReport(err).lines.empty()) << parseDiagReport(err).dump();
    }}});
}

// One process driving several GPUs on the legacy (non-cuMem) path: a rank that newly enables context-wide
// peer access to a peer prints one informational notice (src/diagnostics/p2p.cc). The check runs before
// transport setup (src/init.cc), so in a fresh process on a full XGMI mesh every rank enables access and prints
// the notice. The notice is not a failure.
TEST_F(Diagnostics, SingleProcessPeerAccessNotice)
{
    runDiagCases({{"SingleProcessPeerAccessNotice", 2, []() {
        clearDiagEnv();
        setenv("NCCL_RUN_DIAGNOSTICS", "1", 1);
        const int nGpus = usableGpus();
        std::vector<ncclComm_t> comms;
        std::string out;
        ASSERT_NO_FATAL_FAILURE(initAllCaptured(comms, nGpus, out));
        const auto commGuards   = guardComms(comms);
        const DiagReport report = parseDiagReport(out);
        EXPECT_EQ(report.count(kDiagHeader), 1) << report.dump();

        const std::vector<int> devices = firstDevices(nGpus);
        const bool full                = xgmiFullMesh(devices);
        const std::string notice = "NCCL DIAG [INFO] p2p: temporarily enabled context-wide CUDA peer access rank=";
        EXPECT_LE(report.count(notice), nGpus) << report.dump();
        for(int rank = 0; rank < nGpus; ++rank)
        {
            const int n = report.count(notice + std::to_string(rank) + " ");
            if(full)
                EXPECT_EQ(n, 1) << "rank " << rank << "\n" << report.dump();
            else
                EXPECT_LE(n, 1) << "rank " << rank << "\n" << report.dump();
        }
        expectEdgeSummaries(report, {devices});
        EXPECT_TRUE(report.failures().empty()) << report.dump();
    }}});
}

// The report parser and the line constants on a fixed capture, so a parser fault does not hide behind the GPU
// cases skipping on a host without enough GPUs. The lines follow src/diagnostics.cc and src/diagnostics/p2p.cc.
TEST_F(Diagnostics, ReportParserOnFixedCapture)
{
    const std::string captured
        = "application output before init\n"
          "node01:4242 NCCL DIAG === NCCL Diagnostics ===\n"
          "node01:4242 NCCL DIAG [OK] net bw: skipped (single host or no network transport)\n"
          "node01:4242 NCCL DIAG [OK]   p2p: verified P2P access in both directions between every GPU pair "
          "(56 peer accesses)\n"
          "NCCL INFO unrelated log line\n"
          "node01:4242 NCCL DIAG [INFO] p2p: only 3/12 GPU-to-GPU peer accesses passed verification\n"
          "node01:4242 NCCL DIAG [INFO] p2p: write mismatch srcRank=0 dstRank=1\n"
          "node01:4242 NCCL DIAG NCCL diagnostics completed in 37.1 ms across 8 ranks\n";
    const DiagReport report = parseDiagReport(captured);

    ASSERT_EQ(report.lines.size(), 6u) << report.dump();
    EXPECT_EQ(report.lines.front(), kDiagHeader);
    EXPECT_EQ(report.count(kDiagHeader), 1);
    EXPECT_EQ(report.count(kDiagSummary), 1);
    EXPECT_EQ(report.count(kDiagPartialSummary), 1);
    EXPECT_EQ(okEdgeCount(report.lines[2]), 56);
    EXPECT_EQ(okEdgeCount(report.lines[3]), -1);
    ASSERT_EQ(report.failures().size(), 1u) << report.dump();
    EXPECT_NE(report.failures().front().find("p2p: write mismatch"), std::string::npos);
    EXPECT_TRUE(parseDiagReport("no report here\n").lines.empty());

    // One line per failure marker, in the product wording.
    const std::string failing
        = "node01:4242 NCCL DIAG === NCCL Diagnostics ===\n"
          "node01:4242 NCCL DIAG [INFO] transport detect returned 3\n"
          "node01:4242 NCCL DIAG [INFO] p2p: check returned 3\n"
          "node01:4242 NCCL DIAG [INFO] p2p: setup failed on rank 2 result=1\n"
          "node01:4242 NCCL DIAG [INFO] p2p: destination buffer unavailable srcRank=0 dstRank=1 reason=noDescriptor\n"
          "node01:4242 NCCL DIAG [INFO] p2p: local CUDA setup failed srcRank=0 dstRank=1 reason=localCuda\n"
          "node01:4242 NCCL DIAG [INFO] p2p: peer-memory import failed srcRank=0 dstRank=1 reason=import\n"
          "node01:4242 NCCL DIAG [INFO] p2p: write mismatch srcRank=0 dstRank=1 expected=0x1 got=0x0\n"
          "node01:4242 NCCL DIAG [INFO] p2p: read mismatch srcRank=0 dstRank=1 expected=0x1 got=0x0\n"
          "node01:4242 NCCL DIAG [INFO] p2p: topology check failed srcRank=0 dstRank=1 reason=topo\n"
          "node01:4242 NCCL DIAG [INFO] p2p: launch/check failed srcRank=0 dstRank=1 reason=writeLaunch\n"
          "node01:4242 NCCL DIAG [INFO] p2p: resource cleanup failed rank=0 result=1\n";
    const DiagReport failReport             = parseDiagReport(failing);
    const std::vector<std::string> failures = failReport.failures();
    EXPECT_EQ(failures.size(), std::size(kDiagFailureMarkers)) << failReport.dump();
    for(const char* marker : kDiagFailureMarkers)
    {
        EXPECT_EQ(std::count_if(failures.begin(),
                                failures.end(),
                                [&](const std::string& l) { return l.find(marker) != std::string::npos; }),
                  1)
            << marker;
    }
}

} // namespace RcclUnitTesting
