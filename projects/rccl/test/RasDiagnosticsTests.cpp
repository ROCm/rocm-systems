/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Tests for RAS diagnostics (src/ras/diagnostics*.cc): the init-time report requested with
// NCCL_RUN_RAS_DIAGNOSTICS=1 and the on-demand report requested by a RAS client (rcclras -D). The init-time
// header is printed synchronously by rank 0 during communicator creation; the check lines and the completion
// line follow asynchronously from the RAS thread of the same process. Every line is written to stdout with a
// "<host>:<pid> NCCL DIAG " prefix and flushed, so the cases redirect stdout to a file and poll it until the
// completion line appears.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#if __has_include(<amd_smi/amdsmi.h>)
#include <amd_smi/amdsmi.h>
#endif

#include <arpa/inet.h>
#include <dlfcn.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "common/EnvVars.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "common/ResourceGuards.hpp"
#include "common/TestChecks.hpp"

namespace RcclUnitTesting
{

// Covers HIP init, ncclCommInitAll on up to 8 GPUs and the RAS diagnostics collective. The pre-checkin CI entries
// (ci-precheckin.json) allow 240 s for a TEST_F with one case and 300 s for one with two cases. A case waits for at
// most two reports, so two kReportWait periods plus two inits (about 35 s) have to fit in kRasTimeoutSeconds.
static constexpr int kRasTimeoutSeconds = 120;
static constexpr auto kReportWait       = std::chrono::seconds(30);
static constexpr int kMaxGpus           = 8;
static constexpr size_t kAllReduceElems = 1 << 20;

static const char* const kDiagPrefix = "NCCL DIAG ";
static const char* const kRasHeader  = "NCCL DIAG === RAS Diagnostics ===";
static const char* const kRasSkipped = "NCCL DIAG INFO RAS diagnostics skipped: NCCL_RAS_ENABLE=0";
static const char* const kRasDone    = "NCCL DIAG RAS diagnostics completed in ";
static const char* const kTagOk      = "NCCL DIAG [OK]   ";
static const char* const kTagInfo    = "NCCL DIAG [INFO] ";

// Labels of the checks in src/ras/diagnostics_checks.h that print a line on every system.
static const char* const kGpuInventory = "GPU inventory: ";
static const char* const kDriver       = "HIP driver version: ";
static const char* const kGpuDriver    = "AMD GPU driver version: ";
static const char* const kEcc          = "ECC: ";
static const char* const kEnv          = "NCCL environment: ";
// The link check prints nothing when no device reports an XGMI link.
static const char* const kXgmi = "XGMI: ";

// Report lines that indicate a failed, incomplete or inconsistent check (src/ras/diagnostics_gpu.cc,
// src/ras/diagnostics_env.cc, src/ras/diagnostics_checks_common.cc). test/host/CMakeLists.txt checks at configure
// time that every marker here and in kRasHealthMarkers is still in the report line it marks.
static const char* const kRasFailureMarkers[] = {
    "diagnostics incomplete",
    "mismatch across",
    "differ across ranks",
    "comparison may be",
};

// Report lines about the health of the GPUs themselves. They show that the checks read real data and are not
// test failures: whether a node has memory errors or a down link is outside this test's control.
static const char* const kRasHealthMarkers[] = {
    "uncorrected volatile errors on rank(s)",
    "at or above threshold",
    "inactive link(s)",
};

// Data source and terms of the NVIDIA build, which an AMD report must not contain.
static const char* const kNvidiaTerms[] = {"NVML", "NVLink", "CUDA"};

// Process environment keys of the env-mismatch workers. They must not start with "NCCL_": the environment check
// compares every NCCL_* variable, and these differ between the ranks by design.
static const char* const kEnvWorkerRank = "RCCL_TEST_RAS_DIAG_RANK";
static const char* const kEnvWorkerUid  = "RCCL_TEST_RAS_DIAG_UID";
static const char* const kEnvWorkerOut  = "RCCL_TEST_RAS_DIAG_OUT";
static const char* const kMarkerVar     = "NCCL_RAS_DIAG_TEST_MARKER";

struct RasReport
{
    std::vector<std::string> lines;  // Report lines with the "<host>:<pid> " prefix stripped.

    int count(const std::string& needle) const
    {
        return static_cast<int>(std::count_if(lines.begin(), lines.end(), [&](const std::string& l) {
            return l.find(needle) != std::string::npos;
        }));
    }

    std::vector<std::string> startingWith(const char* prefix) const
    {
        std::vector<std::string> out;
        for(const auto& l : lines)
            if(l.rfind(prefix, 0) == 0)
                out.push_back(l);
        return out;
    }

    // Completion lines that name the given scope, e.g. "8 ranks" or "1 RAS peers".
    int countDone(const std::string& scope) const
    {
        const std::string suffix = " across " + scope;
        int n                    = 0;
        for(const auto& l : startingWith(kRasDone))
            if(l.size() >= suffix.size() && l.compare(l.size() - suffix.size(), suffix.size(), suffix) == 0)
                ++n;
        return n;
    }

    // Tagged lines of one check, e.g. checkLines(kEcc).
    std::vector<std::string> checkLines(const char* label) const
    {
        std::vector<std::string> out;
        for(const auto& l : lines)
            if((l.rfind(kTagOk, 0) == 0 || l.rfind(kTagInfo, 0) == 0) && l.find(label) != std::string::npos)
                out.push_back(l);
        return out;
    }

    std::vector<std::string> failures() const
    {
        std::vector<std::string> out;
        for(const auto& l : lines)
            for(const char* m : kRasFailureMarkers)
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

static RasReport parseRasReport(const std::string& captured)
{
    RasReport report;
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

// Redirects stdout (fd 1) of this process to a temporary file until restore() or destruction. The file can be read
// while the redirection is active, which lets a case wait for the asynchronous part of the report.
class StdoutToFile
{
public:
    StdoutToFile()
    {
        char path[] = "/tmp/rccl-ras-diag-XXXXXX";
        fd_ = mkstemp(path);
        if(fd_ < 0)
            return;
        path_ = path;
        fflush(stdout);
        saved_ = dup(STDOUT_FILENO);
        if(saved_ >= 0 && dup2(fd_, STDOUT_FILENO) < 0)
        {
            close(saved_);
            saved_ = -1;
        }
    }
    ~StdoutToFile()
    {
        restore();
        // gtest prints a failure where it happens, so a failure under the redirection is only in the file.
        if(::testing::Test::HasFailure() && !path_.empty())
            std::cout << "[ captured stdout ]\n" << read() << "[ end of captured stdout ]" << std::endl;
        if(!path_.empty())
            unlink(path_.c_str());
    }
    StdoutToFile(const StdoutToFile&)            = delete;
    StdoutToFile& operator=(const StdoutToFile&) = delete;

    bool active() const { return saved_ >= 0; }

    void restore()
    {
        if(saved_ >= 0)
        {
            fflush(stdout);
            dup2(saved_, STDOUT_FILENO);
            close(saved_);
            saved_ = -1;
        }
        if(fd_ >= 0)
        {
            close(fd_);
            fd_ = -1;
        }
    }

    std::string read() const
    {
        std::ifstream f(path_);
        std::ostringstream os;
        os << f.rdbuf();
        return os.str();
    }

    // Polls the file until it holds nReports completion lines or the wait expires; returns the parsed report.
    RasReport waitForReports(int nReports, std::chrono::milliseconds wait = kReportWait) const
    {
        const auto deadline = std::chrono::steady_clock::now() + wait;
        RasReport report    = parseRasReport(read());
        while(report.count(kRasDone) < nReports && std::chrono::steady_clock::now() < deadline)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            report = parseRasReport(read());
        }
        return report;
    }

private:
    std::string path_;
    int fd_    = -1;
    int saved_ = -1;
};

// Picks a likely-free loopback TCP port (bind to port 0, read the assignment, close). The RAS client listener binds
// with SO_REUSEADDR, so the closed probe does not block it; another process may still take the port in between.
static int pickFreePort()
{
    const int s = ::socket(AF_INET, SOCK_STREAM, 0);
    if(s < 0)
        return 0;
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t len        = sizeof(addr);
    int port             = 0;
    if(::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0
       && ::getsockname(s, reinterpret_cast<sockaddr*>(&addr), &len) == 0)
        port = ntohs(addr.sin_port);
    ::close(s);
    return port;
}

// The parent test environment (e.g. CI categories) may carry these; each case sets what it needs. The RAS client
// listener of every case binds a free loopback port: the default port 28028 may be held by another job on a shared
// host, and a client would then silently talk to that job. Returns the port, 0 on failure.
static int setRasEnv(bool runDiagnostics)
{
    unsetenv("NCCL_RUN_RAS_DIAGNOSTICS");
    unsetenv("NCCL_RUN_DIAGNOSTICS");
    unsetenv("NCCL_RAS_ENABLE");
    unsetenv("NCCL_DIAGNOSTICS_ECC_THRESHOLD");
    unsetenv("NCCL_RAS_ADDR");
    unsetenv(kMarkerVar);
    if(runDiagnostics)
        setenv("NCCL_RUN_RAS_DIAGNOSTICS", "1", 1);
    const int port = pickFreePort();
    if(port > 0)
        setenv("NCCL_RAS_ADDR", ("127.0.0.1:" + std::to_string(port)).c_str(), 1);
    return port;
}

// Sends one command to the RAS client listener on 127.0.0.1:port, optionally after "SET FORMAT <format>", and
// returns everything the server writes until it closes the connection. Returns "" if the connection fails.
static std::string rasRequest(int port, const std::string& command, const std::string& format = "")
{
    int sock = -1;
    for(int attempt = 0; attempt < 50 && sock < 0; ++attempt)
    {
        const int s = ::socket(AF_INET, SOCK_STREAM, 0);
        if(s < 0)
            return {};
        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port        = htons(static_cast<uint16_t>(port));
        if(::connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0)
            sock = s;
        else
        {
            ::close(s);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    if(sock < 0)
        return {};
    timeval tv{static_cast<time_t>(kReportWait.count()), 0};
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    std::string body;
    char buf[4096];
    if(!format.empty())
    {
        const std::string setFormat = "SET FORMAT " + format + "\n";
        ::send(sock, setFormat.c_str(), setFormat.size(), 0);
        const ssize_t n = ::recv(sock, buf, sizeof(buf), 0);
        if(n <= 0 || std::string(buf, n).find("OK") == std::string::npos)
        {
            ::close(sock);
            return {};
        }
    }
    const std::string line = command + "\n";
    ::send(sock, line.c_str(), line.size(), 0);
    ssize_t n;
    while((n = ::recv(sock, buf, sizeof(buf), 0)) > 0)
        body.append(buf, n);
    if(n < 0)
        ADD_FAILURE() << "RAS client request \"" << command << "\": recv failed before the server closed the "
                      << "connection (" << strerror(errno) << ", timeout " << kReportWait.count() << " s)";
    ::close(sock);
    return body;
}

// Called only in a case process. The gtest parent must not initialize HIP: it forks the TestBed children of the
// tests that run after these, and HIP does not survive a fork().
static int usableGpus()
{
    int devCount = 0;
    if(hipGetDeviceCount(&devCount) != hipSuccess)
        return 0;
    return std::min(devCount, kMaxGpus);
}

// The parent takes the count from the EnvVars probe, which queries HIP in a separate process. A case process, which
// re-enters the test to find its case, is exec'd and may query HIP itself.
static int caseGateGpus()
{
    if(std::getenv(ProcessIsolatedTestRunner::kReexecMarkerEnvVar) != nullptr)
        return usableGpus();
    static const int nGpus = std::min(EnvVars().GetNumDetectedGpus(), kMaxGpus);
    return nGpus;
}

// Creates nGpus communicators on devices 0..nGpus-1. Callers wrap the call in ASSERT_NO_FATAL_FAILURE so that a
// failed init stops the case before the communicators are used.
static void initAll(std::vector<ncclComm_t>& comms, int nGpus)
{
    std::vector<int> devices(nGpus);
    std::iota(devices.begin(), devices.end(), 0);
    comms.assign(nGpus, nullptr);
    const ncclResult_t res = ncclCommInitAll(comms.data(), nGpus, devices.data());
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

// AMD SMI reports ECC counters for the CDNA Instinct GPUs (MI100 and later). Other GPUs, including the gfx9 Radeon
// parts (gfx900, gfx906), may have none to report. It queries HIP, so like usableGpus() it runs in case processes only.
static bool eccCountersExpected()
{
    hipDeviceProp_t prop;
    if(hipGetDeviceProperties(&prop, 0) != hipSuccess)
        return false;
    for(const char* arch : {"gfx908", "gfx90a", "gfx94", "gfx950"})
        if(std::strncmp(prop.gcnArchName, arch, std::strlen(arch)) == 0)
            return true;
    return false;
}

// Whether this host has the amdgpu version that AMD SMI reports; an in-tree amdgpu has no
// /sys/module/amdgpu/version.
static bool amdgpuVersionKnown()
{
    std::string version;
    std::ifstream("/sys/module/amdgpu/version") >> version;
    return !version.empty();
}

// Whether the RAS diagnostics can load AMD SMI on this host. Keep in sync with amd_smi_diagLoadImpl
// (src/misc/amdsmi_wrap.cc): no AMD SMI under WSL2, and the library is opened by its versioned SONAME. This target
// may not see the amdsmi header that librccl is built with, so the unversioned name is tried as well.
static bool amdSmiLoadable()
{
    if(access("/dev/dxg", F_OK) == 0)
        return false;
    std::vector<std::string> libs;
#ifdef AMDSMI_LIB_VERSION_MAJOR
    libs.push_back("libamd_smi.so." + std::to_string(AMDSMI_LIB_VERSION_MAJOR));
#endif
    libs.push_back("libamd_smi.so");
    for(const std::string& lib : libs)
    {
        if(void* handle = dlopen(lib.c_str(), RTLD_LAZY | RTLD_LOCAL))
        {
            dlclose(handle);
            return true;
        }
    }
    return false;
}

// nReports reports covering nComms communicators (one per report by default) of nRanks ranks each: one header and
// one completion line per report, one line per check and communicator covering all ranks, no failure line and no
// NVIDIA term. HIP driver version and NCCL environment are [OK]; GPU inventory and AMD GPU driver version are [OK]
// if AMD SMI is readable and unavailable otherwise. A readable AMD SMI must have answered the inventory and ECC
// checks, except on non-Instinct GPUs, and the AMD GPU driver version, except with an in-tree amdgpu; the ECC result,
// like the XGMI line, may report a health finding instead of [OK]. Call sites wrap it in ASSERT_NO_FATAL_FAILURE.
static void expectCompleteReport(const RasReport& report, int nRanks, const std::string& doneScope, int nReports = 1,
                                 int nComms = 0, bool amdSmiReadable = amdSmiLoadable())
{
    if(nComms == 0)
        nComms = nReports;
    const std::string across = "across " + std::to_string(nRanks) + " ranks";
    // ECC counters read on some ranks only: "across <n> of <nRanks> ranks".
    const std::string ofRanks = " of " + std::to_string(nRanks) + " ranks";
    EXPECT_EQ(report.count(kRasHeader), nReports) << report.dump();
    for(const char* label : {kGpuInventory, kDriver, kGpuDriver, kEcc, kEnv})
    {
        const std::vector<std::string> lines = report.checkLines(label);
        ASSERT_EQ(lines.size(), static_cast<size_t>(nComms)) << label << "\n" << report.dump();
        for(const std::string& line : lines)
            EXPECT_TRUE(line.find(across) != std::string::npos
                        || (label == kEcc && line.find(ofRanks) != std::string::npos))
                << line;
    }
    const std::vector<std::string> xgmi = report.checkLines(kXgmi);
    EXPECT_LE(xgmi.size(), static_cast<size_t>(nComms)) << report.dump();
    for(const std::string& line : xgmi)
        EXPECT_NE(line.find(across), std::string::npos) << line;
    // A readable AMD SMI may still leave the GPU model or the GPU count unanswered; the line then says which.
    int nInventoryPartial = 0;
    for(const std::string& line : report.checkLines(kGpuInventory))
        if(line.find("GPU model unavailable") != std::string::npos
           || line.find("GPU count unavailable") != std::string::npos)
            ++nInventoryPartial;
    if(amdSmiReadable)
    {
        EXPECT_EQ(report.count(std::string(kTagOk) + kGpuInventory) + nInventoryPartial, nComms) << report.dump();
        if(nInventoryPartial > 0)
            EXPECT_FALSE(eccCountersExpected()) << "AMD SMI gave no GPU model or count for an Instinct GPU\n"
                                                << report.dump();
    }
    else
    {
        EXPECT_EQ(report.count(std::string(kTagInfo) + kGpuInventory + "unavailable via AMD SMI"), nComms)
            << report.dump();
    }
    int nGpuDriverUnavailable = 0;
    for(const std::string& line : report.checkLines(kGpuDriver))
        if(line.find("unavailable via AMD SMI") != std::string::npos)
            ++nGpuDriverUnavailable;
    if(amdSmiReadable)
    {
        EXPECT_EQ(report.count(std::string(kTagOk) + kGpuDriver) + nGpuDriverUnavailable, nComms) << report.dump();
        if(nGpuDriverUnavailable > 0)
            EXPECT_FALSE(amdgpuVersionKnown()) << "AMD SMI gave no amdgpu version\n" << report.dump();
    }
    else
    {
        EXPECT_EQ(nGpuDriverUnavailable, nComms) << report.dump();
    }
    EXPECT_EQ(report.count(std::string(kTagOk) + kEnv + "NCCL_* env vars consistent " + across), nComms)
        << report.dump();
    EXPECT_EQ(report.count(std::string(kTagOk) + kDriver), nComms) << report.dump();
    int nEccUnavailable = 0;
    for(const std::string& line : report.checkLines(kEcc))
        if(line.find("unavailable") != std::string::npos)
            ++nEccUnavailable;
    if(nEccUnavailable > 0 && amdSmiReadable)
        EXPECT_FALSE(eccCountersExpected()) << "AMD SMI gave no ECC counters for an Instinct GPU\n" << report.dump();
    // Only the lines of these checks: the checks that read PCI devices or the kernel log may be unavailable anywhere.
    int nUnavailable = 0;
    for(const char* label : {kGpuInventory, kDriver, kGpuDriver, kEcc, kEnv, kXgmi})
        for(const std::string& line : report.checkLines(label))
            if(line.find("unavailable") != std::string::npos)
                ++nUnavailable;
    EXPECT_EQ(nUnavailable,
              nEccUnavailable + nInventoryPartial + nGpuDriverUnavailable + (amdSmiReadable ? 0 : nComms))
        << report.dump();
    for(const char* term : kNvidiaTerms)
        EXPECT_EQ(report.count(term), 0) << term << "\n" << report.dump();
    EXPECT_EQ(report.count(kRasDone), nReports) << report.dump();
    EXPECT_EQ(report.countDone(doneScope), nReports) << report.dump();
    EXPECT_TRUE(report.failures().empty()) << "failure lines:\n" << report.dump();
}

// The communicator a check line names ("0x..." after "in comm "); empty if it names none.
static std::string commIdOf(const std::string& line)
{
    static const std::string key = "in comm ";
    const size_t pos = line.find(key);
    if(pos == std::string::npos)
        return {};
    const size_t begin = pos + key.size();
    const size_t end   = line.find_first_not_of("0123456789abcdefx", begin);
    return line.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
}

// Each check has one line for each of nComms distinct communicators, so a report repeated for one communicator
// cannot stand in for the report of another. Use together with expectCompleteReport(..., nComms).
static void expectOneReportPerComm(const RasReport& report, int nComms)
{
    std::set<std::string> comms;
    for(const char* label : {kGpuInventory, kDriver, kGpuDriver, kEcc, kEnv})
    {
        std::set<std::string> ids;
        for(const std::string& line : report.checkLines(label))
            ids.insert(commIdOf(line));
        EXPECT_EQ(ids.count(""), 0u) << label << "\n" << report.dump();
        EXPECT_EQ(ids.size(), static_cast<size_t>(nComms)) << label << "\n" << report.dump();
        if(comms.empty())
            comms = ids;
        EXPECT_EQ(ids, comms) << label << "\n" << report.dump();
    }
}

struct RasCase
{
    const char* name;
    int minGpus;
    std::function<void()> body;
};

// Runs, each in its own process, the cases that fit the visible GPU count. The GPU count is checked here in the
// parent: an isolated run ends in EXPECT_TRUE(), so a GTEST_SKIP() inside an isolated case is reported as a pass.
// When no case fits, gtest reports a real skip. A case process exits 0 on a skip, which the parent records as a pass,
// so a case process that cannot fit the case the parent started it for, for example after a failed HIP init, fails.
static void runRasCases(const std::vector<RasCase>& cases)
{
    const int nGpus = caseGateGpus();
    const char* caseProcessTarget = std::getenv(ProcessIsolatedTestRunner::kReexecMarkerEnvVar);
    std::string notRun;
    int registered = 0;
    for(const RasCase& c : cases)
    {
        if(nGpus < c.minGpus)
        {
            if(caseProcessTarget != nullptr && c.name == std::string(caseProcessTarget))
                FAIL() << "case process sees " << nGpus << " usable GPUs, " << c.name << " needs " << c.minGpus;
            notRun += std::string(" ") + c.name + " (>= " + std::to_string(c.minGpus) + " GPUs)";
            continue;
        }
        ProcessIsolatedTestRunner::registerTest(ProcessIsolatedTestRunner::TestConfig(c.name, c.body)
                                                    .withNumGpus(kMaxGpus)
                                                    .withTimeout(std::chrono::seconds(kRasTimeoutSeconds)));
        ++registered;
    }
    if(registered == 0)
        GTEST_SKIP() << nGpus << " usable GPUs, not run:" << notRun;
    if(!notRun.empty())
        std::cout << "[ INFO     ] " << nGpus << " usable GPUs, not run:" << notRun << std::endl;
    EXPECT_TRUE(ProcessIsolatedTestRunner::executeAllTests()) << "One or more isolated tests failed";
}

static std::string uniqueIdToHex(const ncclUniqueId& id)
{
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&id);
    std::string out;
    char buf[3];
    for(size_t i = 0; i < sizeof(ncclUniqueId); ++i)
    {
        snprintf(buf, sizeof(buf), "%02x", bytes[i]);
        out += buf;
    }
    return out;
}

static bool uniqueIdFromHex(const std::string& hex, ncclUniqueId& id)
{
    if(hex.size() != sizeof(ncclUniqueId) * 2)
        return false;
    unsigned char* bytes = reinterpret_cast<unsigned char*>(&id);
    for(size_t i = 0; i < sizeof(ncclUniqueId); ++i)
    {
        unsigned int v = 0;
        if(sscanf(hex.c_str() + i * 2, "%02x", &v) != 1)
            return false;
        bytes[i] = static_cast<unsigned char>(v);
    }
    return true;
}

// Starts one env-mismatch worker: a fresh image of this binary running RasDiagnosticsWorker.Run as the given rank.
static pid_t spawnWorker(int rank, const std::string& uidHex, const std::string& outPath)
{
    const pid_t pid = fork();
    if(pid != 0)
        return pid;
    setenv(kEnvWorkerRank, std::to_string(rank).c_str(), 1);
    setenv(kEnvWorkerUid, uidHex.c_str(), 1);
    setenv(kEnvWorkerOut, outPath.c_str(), 1);
    setenv(kMarkerVar, rank == 0 ? "r0" : "r1", 1);
    std::string filter = "--gtest_filter=RasDiagnosticsWorker.Run";
    char argv0[]       = "rccl-UnitTests";
    char color[]       = "--gtest_color=no";
    char* argv[]       = {argv0, filter.data(), color, nullptr};
    execv("/proc/self/exe", argv);
    fprintf(stderr, "[ras worker %d] execv failed: %s\n", rank, strerror(errno));
    _exit(127);
}

// Waits for the workers; kills the rest once one fails or the deadline passes. Returns true if all exited with 0.
static bool reapWorkers(std::vector<pid_t> pids)
{
    const auto deadline = std::chrono::steady_clock::now() + kReportWait + std::chrono::seconds(30);
    bool ok             = true;
    while(!pids.empty())
    {
        int status      = 0;
        const pid_t pid = waitpid(-1, &status, WNOHANG);
        if(pid > 0)
        {
            const auto it = std::find(pids.begin(), pids.end(), pid);
            if(it != pids.end())
            {
                pids.erase(it);
                if(!WIFEXITED(status) || WEXITSTATUS(status) != 0)
                {
                    ADD_FAILURE() << "worker pid " << pid << " failed, status 0x" << std::hex << status;
                    ok = false;
                }
            }
        }
        else if(pid < 0 && errno != EINTR)
        {
            ADD_FAILURE() << "waitpid: " << strerror(errno);
            ok = false;
        }
        if(!pids.empty() && (!ok || std::chrono::steady_clock::now() > deadline))
        {
            if(ok)
                ADD_FAILURE() << pids.size() << " worker(s) still running at the deadline";
            for(pid_t p : pids)
                kill(p, SIGKILL);
            for(pid_t p : pids)
                waitpid(p, nullptr, 0);
            return false;
        }
        if(pid == 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return ok;
}

class RasDiagnostics : public ::testing::Test
{
    // All work runs under process isolation: NCCL params are cached on first read.
};

// Without NCCL_RUN_RAS_DIAGNOSTICS (or with 0) communicator init prints no report.
TEST_F(RasDiagnostics, DisabledByDefault)
{
    auto body = [](const char* value) {
        return [value]() {
            ASSERT_GT(setRasEnv(false), 0);
            if(value != nullptr)
                setenv("NCCL_RUN_RAS_DIAGNOSTICS", value, 1);
            StdoutToFile capture;
            ASSERT_TRUE(capture.active());
            {
                std::vector<ncclComm_t> comms;
                ASSERT_NO_FATAL_FAILURE(initAll(comms, usableGpus()));
                const auto commGuards = guardComms(comms);
            }
            capture.restore();
            const RasReport report = parseRasReport(capture.read());
            EXPECT_TRUE(report.lines.empty()) << "unexpected report lines:\n" << report.dump();
        };
    };
    runRasCases({{"Unset", 2, body(nullptr)}, {"Zero", 2, body("0")}});
}

// Full single-node communicator: one header, one line per check covering every rank, the NCCL environment
// consistent, and one completion line naming the communicator's rank count.
TEST_F(RasDiagnostics, InitTimeReportCompletes)
{
    runRasCases({{"InitTimeReportCompletes", 2, []() {
        ASSERT_GT(setRasEnv(true), 0);
        const int nGpus = usableGpus();
        StdoutToFile capture;
        ASSERT_TRUE(capture.active());
        std::vector<ncclComm_t> comms;
        ASSERT_NO_FATAL_FAILURE(initAll(comms, nGpus));
        const auto commGuards  = guardComms(comms);
        const RasReport report = capture.waitForReports(1);
        capture.restore();

        ASSERT_FALSE(report.lines.empty());
        EXPECT_EQ(report.count(kRasHeader), 1) << report.dump();
        EXPECT_EQ(report.lines.front(), kRasHeader) << report.dump();
        EXPECT_EQ(report.lines.back().rfind(kRasDone, 0), 0u) << report.dump();
        ASSERT_NO_FATAL_FAILURE(expectCompleteReport(report, nGpus, std::to_string(nGpus) + " ranks"));
    }}});
}

// Diagnostics are informational and leave the communicator usable: collectives issued after the report produce
// correct results.
TEST_F(RasDiagnostics, CommUsableAfterRasDiagnostics)
{
    runRasCases({{"CommUsableAfterRasDiagnostics", 2, []() {
        ASSERT_GT(setRasEnv(true), 0);
        const int nGpus = usableGpus();
        StdoutToFile capture;
        ASSERT_TRUE(capture.active());
        std::vector<ncclComm_t> comms;
        ASSERT_NO_FATAL_FAILURE(initAll(comms, nGpus));
        const auto commGuards  = guardComms(comms);
        const RasReport report = capture.waitForReports(1);
        capture.restore();
        ASSERT_NO_FATAL_FAILURE(expectCompleteReport(report, nGpus, std::to_string(nGpus) + " ranks"));
        checkAllReduce(comms);
    }}});
}

// The report is produced at every communicator initialization: a re-created communicator and the children of
// ncclCommSplit each get their own report.
TEST_F(RasDiagnostics, RunsAtEveryCommInit)
{
    auto reinit = []() {
        ASSERT_GT(setRasEnv(true), 0);
        const int nGpus = usableGpus();
        for(int round = 0; round < 2; ++round)
        {
            SCOPED_TRACE("round " + std::to_string(round));
            StdoutToFile capture;
            ASSERT_TRUE(capture.active());
            std::vector<ncclComm_t> comms;
            ASSERT_NO_FATAL_FAILURE(initAll(comms, nGpus));
            const auto commGuards  = guardComms(comms);
            const RasReport report = capture.waitForReports(1);
            capture.restore();
            EXPECT_EQ(report.count(kRasHeader), 1) << report.dump();
            ASSERT_NO_FATAL_FAILURE(expectCompleteReport(report, nGpus, std::to_string(nGpus) + " ranks"));
        }
    };

    // Splits an even number of GPUs into two equal halves.
    auto split = []() {
        ASSERT_GT(setRasEnv(true), 0);
        const int nGpus = usableGpus() & ~1;
        const int half  = nGpus / 2;

        StdoutToFile parentCapture;
        ASSERT_TRUE(parentCapture.active());
        std::vector<ncclComm_t> parents;
        ASSERT_NO_FATAL_FAILURE(initAll(parents, nGpus));
        const auto parentGuards = guardComms(parents);
        EXPECT_EQ(parentCapture.waitForReports(1).count(kRasDone), 1);
        parentCapture.restore();

        StdoutToFile capture;
        ASSERT_TRUE(capture.active());
        std::vector<ncclComm_t> children(nGpus, nullptr);
        ncclResult_t res = ncclGroupStart();
        for(int i = 0; i < nGpus && res == ncclSuccess; ++i)
            res = ncclCommSplit(parents[i], i % 2, i, &children[i], nullptr);
        const ncclResult_t endRes = ncclGroupEnd();
        const auto childGuards    = guardComms(children);
        ASSERT_EQ(res, ncclSuccess) << "ncclCommSplit: " << ncclGetErrorString(res);
        ASSERT_EQ(endRes, ncclSuccess) << "ncclGroupEnd: " << ncclGetErrorString(endRes);
        const RasReport report = capture.waitForReports(2);
        capture.restore();
        ASSERT_NO_FATAL_FAILURE(expectCompleteReport(report, half, std::to_string(half) + " ranks", 2));
        expectOneReportPerComm(report, 2);
    };

    runRasCases({{"ReinitSameProcess", 2, reinit}, {"CommSplit", 4, split}});
}

// A RAS client request ("DIAGNOSTICS", what rcclras -D sends) runs the checks on demand without
// NCCL_RUN_RAS_DIAGNOSTICS and returns the report over the client connection. The report covers every
// communicator of the job and names the number of RAS peers (processes) that answered. The request is text-only.
TEST_F(RasDiagnostics, OnDemandClientRequest)
{
    // Two communicators over the same GPUs: one report, with one line per check for each of them.
    auto text = []() {
        const int port = setRasEnv(false);
        ASSERT_GT(port, 0);
        const int nGpus = usableGpus();
        std::vector<ncclComm_t> comms, others;
        ASSERT_NO_FATAL_FAILURE(initAll(comms, nGpus));
        const auto commGuards = guardComms(comms);
        ASSERT_NO_FATAL_FAILURE(initAll(others, nGpus));
        const auto otherGuards = guardComms(others);

        const std::string response = rasRequest(port, "DIAGNOSTICS");
        ASSERT_FALSE(response.empty()) << "no response from the RAS client listener on port " << port;
        const RasReport report = parseRasReport(response);
        EXPECT_EQ(report.count(kRasHeader), 1) << report.dump();
        ASSERT_NO_FATAL_FAILURE(expectCompleteReport(report, nGpus, "1 RAS peers", 1, 2));
        expectOneReportPerComm(report, 2);
    };
    auto json = []() {
        const int port = setRasEnv(false);
        ASSERT_GT(port, 0);
        std::vector<ncclComm_t> comms;
        ASSERT_NO_FATAL_FAILURE(initAll(comms, usableGpus()));
        const auto commGuards = guardComms(comms);

        const std::string response = rasRequest(port, "DIAGNOSTICS", "json");
        ASSERT_FALSE(response.empty()) << "no response from the RAS client listener on port " << port;
        EXPECT_NE(response.find("ERROR: diagnostics only supports text output"), std::string::npos) << response;
        EXPECT_TRUE(parseRasReport(response).lines.empty()) << response;
    };
    runRasCases({{"Text", 2, text}, {"JsonRejected", 2, json}});
}

// The environment check compares every NCCL_* variable across the ranks of a communicator. Two processes, one rank
// each, differ in one variable: the report names the variable, the value of each rank and the number of differing
// variables, and the other checks still pass.
TEST_F(RasDiagnostics, EnvMismatchAcrossProcesses)
{
    runRasCases({{"EnvMismatchAcrossProcesses", 2, []() {
        ASSERT_GT(setRasEnv(true), 0);
        ncclUniqueId id;
        ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
        char outPath[] = "/tmp/rccl-ras-diag-out-XXXXXX";
        const int outFd = mkstemp(outPath);
        ASSERT_GE(outFd, 0);
        close(outFd);
        const auto outGuard = RCCLTestGuards::makeScopeGuard([&]() { unlink(outPath); });

        const std::string uidHex = uniqueIdToHex(id);
        std::vector<pid_t> pids;
        for(int rank = 0; rank < 2; ++rank)
        {
            const pid_t pid = spawnWorker(rank, uidHex, outPath);
            if(pid <= 0)
            {
                ADD_FAILURE() << "fork failed for rank " << rank << ": " << strerror(errno);
                break;
            }
            pids.push_back(pid);
        }
        const bool reaped = reapWorkers(pids);
        ASSERT_EQ(pids.size(), 2u);
        ASSERT_TRUE(reaped) << "see the worker output above";

        std::ifstream f(outPath);
        std::ostringstream os;
        os << f.rdbuf();
        const RasReport report = parseRasReport(os.str());

        const std::string env = std::string(kTagInfo) + kEnv;
        EXPECT_EQ(report.count(kRasHeader), 1) << report.dump();
        EXPECT_EQ(report.count(env + "mismatch across 2 ranks in comm 0x"), 1) << report.dump();
        EXPECT_EQ(report.count(std::string("for ") + kMarkerVar), 1) << report.dump();
        EXPECT_EQ(report.count(env + kMarkerVar + "=r0 on rank(s) {0}"), 1) << report.dump();
        EXPECT_EQ(report.count(env + kMarkerVar + "=r1 on rank(s) {1}"), 1) << report.dump();
        EXPECT_EQ(report.count(env + "1 NCCL_* env var(s) differ across ranks in comm 0x"), 1) << report.dump();
        EXPECT_EQ(report.count("NCCL_* env vars consistent"), 0) << report.dump();
        for(const char* label : {kGpuInventory, kDriver, kGpuDriver, kEcc})
            EXPECT_EQ(report.checkLines(label).size(), 1u) << label << "\n" << report.dump();
        EXPECT_EQ(report.count(kRasDone), 1) << report.dump();
        EXPECT_EQ(report.countDone("2 ranks"), 1) << report.dump();
    }}});
}

// Worker of EnvMismatchAcrossProcesses; skips unless started by it. Rank 0 captures its stdout until the report is
// complete and writes it to the coordinator's file. The AllReduce afterwards keeps rank 1 (and its RAS thread) alive
// until rank 0 has the report, and checks that the communicator still works.
TEST(RasDiagnosticsWorker, Run)
{
    const char* rankEnv = getenv(kEnvWorkerRank);
    if(rankEnv == nullptr)
        GTEST_SKIP() << "worker entry point, driven by RasDiagnostics.EnvMismatchAcrossProcesses";
    const int rank        = atoi(rankEnv);
    const char* uidEnv    = getenv(kEnvWorkerUid);
    const char* outEnv    = getenv(kEnvWorkerOut);
    ASSERT_NE(uidEnv, nullptr);
    ASSERT_NE(outEnv, nullptr);
    ncclUniqueId id;
    ASSERT_TRUE(uniqueIdFromHex(uidEnv, id)) << "malformed ncclUniqueId in the environment";
    HIP_CHECK(hipSetDevice(rank));

    // Only rank 0 prints the report; rank 1 keeps stdout so that its failures reach the coordinator's output.
    std::optional<StdoutToFile> capture;
    if(rank == 0)
    {
        capture.emplace();
        ASSERT_TRUE(capture->active());
    }
    ncclComm_t comm        = nullptr;
    const ncclResult_t res = ncclCommInitRank(&comm, 2, id, rank);
    if(res != ncclSuccess && capture)
        capture->restore();
    ASSERT_EQ(res, ncclSuccess) << "ncclCommInitRank: " << ncclGetErrorString(res);
    const auto commGuard = RCCLTestGuards::makeCommAutoGuard(comm);
    if(rank == 0)
    {
        const RasReport report = capture->waitForReports(1);
        capture->restore();
        EXPECT_EQ(report.count(kRasDone), 1) << report.dump();
        std::ofstream(outEnv) << capture->read();
    }

    float* buf         = nullptr;
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    const auto streamGuard = RCCLTestGuards::makeStreamAutoGuard(stream);
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&buf), sizeof(float)));
    const auto bufGuard = RCCLTestGuards::makeDeviceBufferAutoGuard(buf);
    float value         = static_cast<float>(rank + 1);
    HIP_CHECK(hipMemcpy(buf, &value, sizeof(float), hipMemcpyHostToDevice));
    ASSERT_EQ(ncclAllReduce(buf, buf, 1, ncclFloat, ncclSum, comm, stream), ncclSuccess);
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(&value, buf, sizeof(float), hipMemcpyDeviceToHost));
    EXPECT_EQ(value, 3.0f);
}

// The init-time report needs the RAS subsystem: with NCCL_RAS_ENABLE=0 rank 0 prints only that the diagnostics were
// skipped, with no header, check or completion line, and communicator creation still succeeds.
TEST_F(RasDiagnostics, RasDisabledRunsNoChecks)
{
    runRasCases({{"RasDisabledRunsNoChecks", 2, []() {
        ASSERT_GT(setRasEnv(true), 0);
        setenv("NCCL_RAS_ENABLE", "0", 1);
        StdoutToFile capture;
        ASSERT_TRUE(capture.active());
        std::vector<ncclComm_t> comms;
        ASSERT_NO_FATAL_FAILURE(initAll(comms, usableGpus()));
        const auto commGuards = guardComms(comms);
        // Only the skip line is synchronous; give a RAS thread the time a report takes on a loaded node to show that
        // nothing follows it.
        capture.waitForReports(1, std::chrono::seconds(5));
        capture.restore();
        checkAllReduce(comms);
        const RasReport report = parseRasReport(capture.read());
        EXPECT_EQ(report.count(kRasSkipped), 1) << report.dump();
        EXPECT_EQ(report.count(kRasHeader), 0) << report.dump();
        EXPECT_EQ(report.lines.size(), 1u) << report.dump();
        EXPECT_EQ(report.count(kRasDone), 0) << report.dump();
        for(const char* label : {kGpuInventory, kDriver, kGpuDriver, kEcc, kEnv})
            EXPECT_TRUE(report.checkLines(label).empty()) << label << "\n" << report.dump();
    }}});
}

// The report parser and the line constants on a fixed capture, so a parser fault does not hide behind the GPU cases
// skipping on a host without enough GPUs. The lines follow src/ras/diagnostics*.cc.
TEST_F(RasDiagnostics, ReportParserOnFixedCapture)
{
    const std::string captured
        = "application output before init\n"
          "node01:4242 NCCL DIAG === RAS Diagnostics ===\n"
          "node01:4242 NCCL DIAG [OK]   GPU inventory: 8x AMD Instinct MI355X per node consistent across 8 ranks "
          "in comm 0x1f\n"
          "NCCL INFO unrelated log line\n"
          "node01:4242 NCCL DIAG [OK]   HIP driver version: 71526333 consistent across 8 ranks in comm 0x1f\n"
          "node01:4242 NCCL DIAG [OK]   AMD GPU driver version: 6.19.14.31400000 consistent across 8 ranks in comm "
          "0x1f\n"
          "node01:4242 NCCL DIAG [OK]   ECC: no uncorrected volatile errors across 8 ranks in comm 0x1f\n"
          "node01:4242 NCCL DIAG [OK]   XGMI: 7 links per GPU, all active across 8 ranks in comm 0x1f\n"
          "node01:4242 NCCL DIAG [OK]   NCCL environment: NCCL_* env vars consistent across 8 ranks in comm 0x1f\n"
          "node01:4242 NCCL DIAG RAS diagnostics completed in 41.7 ms across 8 ranks\n";
    const RasReport report = parseRasReport(captured);
    ASSERT_EQ(report.lines.size(), 8u) << report.dump();
    EXPECT_EQ(report.lines.front(), kRasHeader);
    ASSERT_NO_FATAL_FAILURE(expectCompleteReport(report, 8, "8 ranks", 1, 1, true));
    EXPECT_TRUE(parseRasReport("no report here\n").lines.empty());

    // A host where AMD SMI cannot be loaded: the AMD SMI checks are unavailable, the others still [OK].
    const std::string noAmdSmi
        = "node01:4242 NCCL DIAG === RAS Diagnostics ===\n"
          "node01:4242 NCCL DIAG [INFO] GPU inventory: unavailable via AMD SMI across 8 ranks in comm 0x1f\n"
          "node01:4242 NCCL DIAG [OK]   HIP driver version: 71526333 consistent across 8 ranks in comm 0x1f\n"
          "node01:4242 NCCL DIAG [INFO] AMD GPU driver version: unavailable via AMD SMI across 8 ranks in comm 0x1f\n"
          "node01:4242 NCCL DIAG [INFO] ECC: unavailable via AMD SMI across 8 ranks in comm 0x1f\n"
          "node01:4242 NCCL DIAG [OK]   NCCL environment: NCCL_* env vars consistent across 8 ranks in comm 0x1f\n"
          "node01:4242 NCCL DIAG RAS diagnostics completed in 41.7 ms across 8 ranks\n";
    ASSERT_NO_FATAL_FAILURE(expectCompleteReport(parseRasReport(noAmdSmi), 8, "8 ranks", 1, 1, false));

    // ECC counters read on some ranks only (here alongside an unavailable GPU inventory, so that the case does not
    // depend on the GPUs of the host).
    const std::string partialEcc
        = "node01:4242 NCCL DIAG === RAS Diagnostics ===\n"
          "node01:4242 NCCL DIAG [INFO] GPU inventory: unavailable via AMD SMI across 8 ranks in comm 0x1f\n"
          "node01:4242 NCCL DIAG [OK]   HIP driver version: 71526333 consistent across 8 ranks in comm 0x1f\n"
          "node01:4242 NCCL DIAG [INFO] AMD GPU driver version: unavailable via AMD SMI across 8 ranks in comm 0x1f\n"
          "node01:4242 NCCL DIAG [INFO] ECC: no uncorrected volatile errors across 6 of 8 ranks in comm 0x1f "
          "(ECC counters unavailable via AMD SMI on 2 ranks)\n"
          "node01:4242 NCCL DIAG [OK]   NCCL environment: NCCL_* env vars consistent across 8 ranks in comm 0x1f\n"
          "node01:4242 NCCL DIAG RAS diagnostics completed in 41.7 ms across 8 ranks\n";
    ASSERT_NO_FATAL_FAILURE(expectCompleteReport(parseRasReport(partialEcc), 8, "8 ranks", 1, 1, false));

    // A health finding is a complete report, not a failure.
    const std::string unhealthy
        = "node01:4242 NCCL DIAG === RAS Diagnostics ===\n"
          "node01:4242 NCCL DIAG [OK]   GPU inventory: 8x AMD Instinct MI355X per node consistent across 8 ranks "
          "in comm 0x1f\n"
          "node01:4242 NCCL DIAG [OK]   HIP driver version: 71526333 consistent across 8 ranks in comm 0x1f\n"
          "node01:4242 NCCL DIAG [OK]   AMD GPU driver version: 6.19.14.31400000 consistent across 8 ranks in comm "
          "0x1f\n"
          "node01:4242 NCCL DIAG [INFO] ECC: uncorrected volatile errors on rank(s) {2} (worst=1) across 8 ranks "
          "in comm 0x1f\n"
          "node01:4242 NCCL DIAG [INFO] XGMI: inactive link(s) on rank(s) {5} across 8 ranks in comm 0x1f\n"
          "node01:4242 NCCL DIAG [OK]   NCCL environment: NCCL_* env vars consistent across 8 ranks in comm 0x1f\n"
          "node01:4242 NCCL DIAG RAS diagnostics completed in 41.7 ms across 8 ranks\n";
    ASSERT_NO_FATAL_FAILURE(expectCompleteReport(parseRasReport(unhealthy), 8, "8 ranks", 1, 1, true));

    // One line per failure or health marker, trimmed rather than verbatim product wording.
    const std::string failing
        = "node01:4242 NCCL DIAG [INFO] GPU inventory: diagnostics incomplete, gathered 7/8 ranks in comm 0x1f\n"
          "node01:4242 NCCL DIAG [INFO] GPU inventory: model mismatch across 8 ranks in comm 0x1f, rank(s) {3} "
          "differ from rank 0 (X)\n"
          "node01:4242 NCCL DIAG [INFO] NCCL environment: 1 NCCL_* env var(s) differ across ranks in comm 0x1f\n"
          "node01:4242 NCCL DIAG [INFO] NCCL environment: 1 rank(s) had >16384 bytes of NCCL_* env vars; "
          "comparison may be partial\n"
          "node01:4242 NCCL DIAG [INFO] ECC: uncorrected volatile errors on rank(s) {2} (worst=1) across 8 ranks\n"
          "node01:4242 NCCL DIAG [INFO] ECC: corrected volatile errors at or above threshold 10 on rank(s) {2}\n"
          "node01:4242 NCCL DIAG [INFO] XGMI: inactive link(s) on rank(s) {5} across 8 ranks in comm 0x1f\n";
    const RasReport failReport              = parseRasReport(failing);
    const std::vector<std::string> failures = failReport.failures();
    EXPECT_EQ(failures.size(), std::size(kRasFailureMarkers)) << failReport.dump();
    for(const char* marker : kRasFailureMarkers)
        EXPECT_EQ(failReport.count(marker), 1) << marker;
    for(const char* marker : kRasHealthMarkers)
        EXPECT_EQ(failReport.count(marker), 1) << marker;
}

} // namespace RcclUnitTesting
