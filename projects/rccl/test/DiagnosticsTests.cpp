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

// Covers HIP init, ncclCommInitAll on up to 8 GPUs, and the diagnostics exchange itself.
static constexpr int kDiagTimeoutSeconds = 180;
static constexpr int kMaxGpus            = 8;
static constexpr size_t kAllReduceElems  = 1 << 20;

static const char* const kDiagPrefix  = "NCCL DIAG ";
static const char* const kDiagHeader  = "NCCL DIAG === NCCL Diagnostics ===";
static const char* const kDiagSummary = "NCCL DIAG [OK]   p2p: all ";

// Report lines that indicate a failed or incomplete check (src/diagnostics.cc, src/diagnostics/p2p.cc).
static const char* const kDiagFailureMarkers[] = {
    "transport detect returned",
    "p2p: active check returned",
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

// Directed edge count from "NCCL DIAG [OK]   p2p: all <N> directed GPU P2P edges verified", or -1.
static int okEdgeCount(const std::string& line)
{
    static const std::regex re("p2p: all ([0-9]+) directed GPU P2P edges verified");
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
            if(hipExtGetLinkTypeAndHopCount(a, b, &linkType, &hops) != hipSuccess
               || linkType != HSA_AMD_LINK_INFO_TYPE_XGMI || hops != 1)
                return false;
        }
    return true;
}

// Checks the p2p summary lines of a report covering one communicator per device group (groups of equal size).
// On a full XGMI mesh each communicator verifies all of its N * (N - 1) directed pairs. Elsewhere only the tested
// edges are known: each summary must still be [OK] and bounded by the pair count, and a communicator without
// eligible pairs prints no summary.
static void expectEdgeSummaries(const DiagReport& report, const std::vector<std::vector<int>>& groups)
{
    const int nRanks  = static_cast<int>(groups.front().size());
    const int pairs   = nRanks * (nRanks - 1);
    const int nGroups = static_cast<int>(groups.size());
    const bool full   = std::all_of(groups.begin(), groups.end(), xgmiFullMesh);

    EXPECT_EQ(report.count("directed GPU P2P edges verified"), report.count(kDiagSummary))
        << "partial p2p summary:\n"
        << report.dump();
    if(full)
        ASSERT_EQ(report.count(kDiagSummary), nGroups) << report.dump();
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
static void clearDiagEnv()
{
    unsetenv("NCCL_RUN_DIAGNOSTICS");
    unsetenv("NCCL_RUN_RAS_DIAGNOSTICS");
    unsetenv("NCCL_P2P_DISABLE");
    unsetenv("NCCL_P2P_LEVEL");
}

// Communicators left by a failed init or split may be partially initialized; abort them instead of destroying.
static void abortAll(std::vector<ncclComm_t>& comms)
{
    for(auto& c : comms)
        if(c != nullptr)
        {
            ncclCommAbort(c);
            c = nullptr;
        }
}

// Creates nGpus communicators on devices 0..nGpus-1 while capturing stdout. On failure the created communicators
// are aborted and a fatal failure is recorded, so callers wrap the call in ASSERT_NO_FATAL_FAILURE.
static void initAllCaptured(std::vector<ncclComm_t>& comms, int nGpus, std::string& captured)
{
    std::vector<int> devices(nGpus);
    std::iota(devices.begin(), devices.end(), 0);
    comms.assign(nGpus, nullptr);
    testing::internal::CaptureStdout();
    const ncclResult_t res = ncclCommInitAll(comms.data(), nGpus, devices.data());
    captured               = testing::internal::GetCapturedStdout();
    if(res != ncclSuccess)
    {
        abortAll(comms);
        FAIL() << "ncclCommInitAll: " << ncclGetErrorString(res);
    }
}

static void destroyAll(std::vector<ncclComm_t>& comms)
{
    for(auto& c : comms)
        if(c != nullptr)
        {
            EXPECT_EQ(ncclCommDestroy(c), ncclSuccess);
            c = nullptr;
        }
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
    for(int i = 0; i < n; ++i)
        ASSERT_EQ(ncclAllReduce(send[i], recv[i], kAllReduceElems, ncclFloat, ncclSum, comms[i], streams[i]),
                  ncclSuccess);
    ASSERT_EQ(ncclGroupEnd(), ncclSuccess);

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

static ProcessIsolatedTestRunner::TestConfig diagCase(const char* name, std::function<void()> body)
{
    return ProcessIsolatedTestRunner::TestConfig(name, std::move(body))
        .withNumGpus(kMaxGpus)
        .withTimeout(std::chrono::seconds(kDiagTimeoutSeconds));
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
            if(nGpus < 2)
                GTEST_SKIP() << "Requires >= 2 GPUs";
            std::vector<ncclComm_t> comms;
            std::string out;
            ASSERT_NO_FATAL_FAILURE(initAllCaptured(comms, nGpus, out));
            const DiagReport report = parseDiagReport(out);
            EXPECT_TRUE(report.lines.empty()) << "unexpected report lines:\n" << report.dump();
            destroyAll(comms);
        };
    };
    RUN_ISOLATED_TESTS(diagCase("Unset", body(nullptr)), diagCase("Zero", body("0")));
}

// Full single-node communicator: one header, an [OK] summary covering every eligible directed pair
// (all of them on a full XGMI mesh), one completion line naming the rank count, and no failure lines.
TEST_F(Diagnostics, AllDirectedEdgesVerified)
{
    RUN_ISOLATED_TESTS(diagCase("AllDirectedEdgesVerified", []() {
        clearDiagEnv();
        setenv("NCCL_RUN_DIAGNOSTICS", "1", 1);
        const int nGpus = usableGpus();
        if(nGpus < 2)
            GTEST_SKIP() << "Requires >= 2 GPUs";

        std::vector<ncclComm_t> comms;
        std::string out;
        ASSERT_NO_FATAL_FAILURE(initAllCaptured(comms, nGpus, out));
        const DiagReport report = parseDiagReport(out);

        EXPECT_EQ(report.count(kDiagHeader), 1) << report.dump();
        expectEdgeSummaries(report, {firstDevices(nGpus)});
        const std::string done = "across " + std::to_string(nGpus) + " ranks";
        EXPECT_EQ(report.count("NCCL diagnostics completed in"), 1) << report.dump();
        EXPECT_EQ(report.count(done), 1) << report.dump();
        EXPECT_TRUE(report.failures().empty()) << "failure lines:\n" << report.dump();
        destroyAll(comms);
    }));
}

// Edge count follows the communicator size (N * (N - 1) directed edges for N local GPUs on a full XGMI mesh).
TEST_F(Diagnostics, EdgeCountFollowsCommSize)
{
    auto body = [](int nWanted) {
        return [nWanted]() {
            clearDiagEnv();
            setenv("NCCL_RUN_DIAGNOSTICS", "1", 1);
            if(usableGpus() < nWanted)
                GTEST_SKIP() << "Requires >= " << nWanted << " GPUs";
            std::vector<ncclComm_t> comms;
            std::string out;
            ASSERT_NO_FATAL_FAILURE(initAllCaptured(comms, nWanted, out));
            const DiagReport report = parseDiagReport(out);
            expectEdgeSummaries(report, {firstDevices(nWanted)});
            EXPECT_TRUE(report.failures().empty()) << report.dump();
            destroyAll(comms);
        };
    };
    RUN_ISOLATED_TESTS(diagCase("TwoGpus", body(2)), diagCase("FourGpus", body(4)));
}

// Diagnostics are informational and leave the communicator usable: collectives issued after a
// diagnosed init produce correct results.
TEST_F(Diagnostics, CommUsableAfterDiagnostics)
{
    RUN_ISOLATED_TESTS(diagCase("CommUsableAfterDiagnostics", []() {
        clearDiagEnv();
        setenv("NCCL_RUN_DIAGNOSTICS", "1", 1);
        const int nGpus = usableGpus();
        if(nGpus < 2)
            GTEST_SKIP() << "Requires >= 2 GPUs";
        std::vector<ncclComm_t> comms;
        std::string out;
        ASSERT_NO_FATAL_FAILURE(initAllCaptured(comms, nGpus, out));
        EXPECT_EQ(parseDiagReport(out).count(kDiagHeader), 1);
        checkAllReduce(comms);
        destroyAll(comms);
    }));
}

// The report is produced at every communicator initialization: a re-created communicator and the
// children of ncclCommSplit each print their own report.
TEST_F(Diagnostics, RunsAtEveryCommInit)
{
    auto reinit = []() {
        clearDiagEnv();
        setenv("NCCL_RUN_DIAGNOSTICS", "1", 1);
        const int nGpus = usableGpus();
        if(nGpus < 2)
            GTEST_SKIP() << "Requires >= 2 GPUs";
        for(int round = 0; round < 2; ++round)
        {
            std::vector<ncclComm_t> comms;
            std::string out;
            ASSERT_NO_FATAL_FAILURE(initAllCaptured(comms, nGpus, out));
            EXPECT_EQ(parseDiagReport(out).count(kDiagHeader), 1) << "round " << round;
            destroyAll(comms);
        }
    };

    auto split = []() {
        clearDiagEnv();
        setenv("NCCL_RUN_DIAGNOSTICS", "1", 1);
        const int nGpus = usableGpus();
        if(nGpus < 4 || nGpus % 2 != 0)
            GTEST_SKIP() << "Requires an even GPU count >= 4";

        std::vector<ncclComm_t> parents;
        std::string out;
        ASSERT_NO_FATAL_FAILURE(initAllCaptured(parents, nGpus, out));
        EXPECT_EQ(parseDiagReport(out).count(kDiagHeader), 1);

        std::vector<ncclComm_t> children(nGpus, nullptr);
        testing::internal::CaptureStdout();
        ncclResult_t res = ncclGroupStart();
        for(int i = 0; i < nGpus && res == ncclSuccess; ++i)
            res = ncclCommSplit(parents[i], i % 2, i, &children[i], nullptr);
        const ncclResult_t endRes = ncclGroupEnd();
        out                       = testing::internal::GetCapturedStdout();
        if(res != ncclSuccess || endRes != ncclSuccess)
        {
            abortAll(children);
            destroyAll(parents);
            FAIL() << "ncclCommSplit: " << ncclGetErrorString(res != ncclSuccess ? res : endRes);
        }

        const DiagReport report = parseDiagReport(out);
        const int half          = nGpus / 2;
        std::vector<std::vector<int>> colors(2);
        for(int i = 0; i < nGpus; ++i)
            colors[i % 2].push_back(i);
        EXPECT_EQ(report.count(kDiagHeader), 2) << report.dump();
        expectEdgeSummaries(report, colors);
        EXPECT_EQ(report.count("across " + std::to_string(half) + " ranks"), 2) << report.dump();
        EXPECT_TRUE(report.failures().empty()) << report.dump();
        destroyAll(children);
        destroyAll(parents);
    };

    RUN_ISOLATED_TESTS(diagCase("ReinitSameProcess", reinit), diagCase("CommSplit", split));
}

// With P2P disabled there are no topology-eligible P2P edges. The run still completes and reports
// no failures; it prints no p2p summary line because nothing was tested.
TEST_F(Diagnostics, P2pDisabledReportsNoEdges)
{
    RUN_ISOLATED_TESTS(diagCase("P2pDisabledReportsNoEdges", []() {
        clearDiagEnv();
        setenv("NCCL_RUN_DIAGNOSTICS", "1", 1);
        setenv("NCCL_P2P_DISABLE", "1", 1);
        const int nGpus = usableGpus();
        if(nGpus < 2)
            GTEST_SKIP() << "Requires >= 2 GPUs";
        std::vector<ncclComm_t> comms;
        std::string out;
        ASSERT_NO_FATAL_FAILURE(initAllCaptured(comms, nGpus, out));
        const DiagReport report = parseDiagReport(out);
        EXPECT_EQ(report.count(kDiagHeader), 1) << report.dump();
        EXPECT_EQ(report.count("NCCL diagnostics completed in"), 1) << report.dump();
        EXPECT_EQ(report.count("directed GPU P2P edges verified"), 0) << report.dump();
        EXPECT_TRUE(report.failures().empty()) << report.dump();
        checkAllReduce(comms);
        destroyAll(comms);
    }));
}

// The report goes to stdout only; stderr carries no "NCCL DIAG" lines.
TEST_F(Diagnostics, ReportOnStdoutOnly)
{
    RUN_ISOLATED_TESTS(diagCase("ReportOnStdoutOnly", []() {
        clearDiagEnv();
        setenv("NCCL_RUN_DIAGNOSTICS", "1", 1);
        const int nGpus = usableGpus();
        if(nGpus < 2)
            GTEST_SKIP() << "Requires >= 2 GPUs";
        std::vector<int> devices(nGpus);
        std::iota(devices.begin(), devices.end(), 0);
        std::vector<ncclComm_t> comms(nGpus, nullptr);
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        const ncclResult_t res = ncclCommInitAll(comms.data(), nGpus, devices.data());
        const std::string err  = testing::internal::GetCapturedStderr();
        const std::string out  = testing::internal::GetCapturedStdout();
        if(res != ncclSuccess)
        {
            abortAll(comms);
            FAIL() << "ncclCommInitAll: " << ncclGetErrorString(res);
        }
        EXPECT_EQ(parseDiagReport(out).count(kDiagHeader), 1);
        EXPECT_TRUE(parseDiagReport(err).lines.empty()) << parseDiagReport(err).dump();
        destroyAll(comms);
    }));
}

// One process driving several GPUs on the legacy (non-cuMem) path: a rank that newly enables context-wide
// peer access to a peer prints one informational notice (src/diagnostics/p2p.cc); with cuMem the access is
// mapping-scoped and no notice is printed. The check runs before transport setup (src/init.cc), so in a fresh
// process on a full XGMI mesh every rank enables access and prints the notice. The notice is not a failure.
TEST_F(Diagnostics, SingleProcessPeerAccessNotice)
{
    RUN_ISOLATED_TESTS(diagCase("SingleProcessPeerAccessNotice", []() {
        clearDiagEnv();
        setenv("NCCL_RUN_DIAGNOSTICS", "1", 1);
        setenv("NCCL_CUMEM_ENABLE", "0", 1);
        const int nGpus = usableGpus();
        if(nGpus < 2)
            GTEST_SKIP() << "Requires >= 2 GPUs";
        std::vector<ncclComm_t> comms;
        std::string out;
        ASSERT_NO_FATAL_FAILURE(initAllCaptured(comms, nGpus, out));
        const DiagReport report = parseDiagReport(out);

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
        destroyAll(comms);
    }));
}

} // namespace RcclUnitTesting
