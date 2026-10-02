/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_REGISTRATION_TEST_SUPPORT_HPP
#define RCCL_TEST_REGISTRATION_TEST_SUPPORT_HPP

#ifdef MPI_TESTS_ENABLED

#include "DeviceBufferHelpers.hpp"
#include "HybridVmmHelpers.hpp"
#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "CeAllReduceTestHelpers.hpp"
#include "ce_coll.h"
#include "comm.h"
#include "rccl_common.h"
#include "register.h"
#include "register_inline.h"
#ifdef ENABLE_FAULT_INJECTION
#include "ce_fault_inject.h"
#endif

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

// NCCL_CTA_POLICY accepts the documented alias ZERO as well as the integer 2.
inline bool envCtaPolicyIsZero()
{
    const char* p = std::getenv("NCCL_CTA_POLICY");
    if (p == nullptr || p[0] == '\0') return false;
    if (std::atoi(p) == 2) return true;
    std::string s(p);
    for (char& c : s)
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s == "ZERO";
}

// CAST jobs set NCCL_NET=IB-CAST (or ib-cast). Classic IB must not take this arm.
inline bool envNetIsIbCast()
{
    const char* net = std::getenv("NCCL_NET");
    if (net == nullptr || net[0] == '\0') return false;
    std::string s(net);
    for (char& c : s)
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s == "IB-CAST";
}

// Env / driver gates are process-wide, so a rank-local skip is safe. Alloc
// failure is not: that skip must run in the TEST body after allRanksTrue.
inline const char* ceRecvOffsetEnvSkipReason()
{
    const char* ceAllReduce = std::getenv("RCCL_CE_ALLREDUCE");
    if (ceAllReduce == nullptr || std::atoi(ceAllReduce) != 1) {
        return "CE receive-offset regression requires RCCL_CE_ALLREDUCE=1";
    }
    if (!envCtaPolicyIsZero()) {
        return "CE receive-offset regression requires NCCL_CTA_POLICY=2 or ZERO";
    }
    if (!RcclUnitTesting::isCeRuntimeDriverSupported()) {
        return "CE receive-offset regression requires a CE-capable HIP driver "
               "(ROCm >= 7.12 or 7.0.2.x backport)";
    }
    return nullptr;
}

// Test Configuration
namespace RegTestConfig {
    constexpr size_t SMALL_COUNT  = 1024;           // 4KB for float
    constexpr size_t MEDIUM_COUNT = 256 * 1024;     // 1MB for float
    constexpr size_t LARGE_COUNT  = 1024 * 1024;    // 4MB for float

    using DefaultType = hip_bfloat16;

    constexpr int MIN_RANKS_DEFAULT    = 2;
    constexpr int MIN_RANKS_ALLTOALL   = 4;
    constexpr int MIN_NODES_MULTINODE  = 2;
}

// REG Log Checker - Pattern checking for registration debug output
class REGLogChecker
{
public:
    explicit REGLogChecker(const std::string& logContent)
        : m_content(logContent) {}

    bool hasIPCRegistration() const
    {
        return hasPattern("IPC register buffer") ||
               hasPattern("IPC registering buffer") ||
               (hasPattern("Proxy rank") && hasPattern("register success"));
    }

    bool hasIPCReuse() const
    {
        return hasPattern("IPC reuse buffer");
    }

    bool hasNETReuse() const
    {
        return hasPattern("NET reuse buffer");
    }

    bool hasNumSegments(int n) const
    {
        const std::regex re("numSegments\\s*:?\\s*" + std::to_string(n) + "\\b");
        return std::regex_search(m_content, re);
    }

    bool hasSymSysmemHandleReuse() const
    {
        return hasPattern("Symmetric window reusing system-memory handle");
    }

    bool hasNETRegistration() const
    {
        return hasPattern("NET register userbuff");
    }

    bool hasAnyRegistrationSuccess() const
    {
        return hasIPCRegistration() || hasIPCReuse() || hasNETRegistration() || hasNETReuse();
    }

    bool hasIPCFailure() const
    {
        return hasPattern("failed to IPC register") ||
               hasPattern("legacy IPC blocked");
    }

    bool hasNETFailure() const
    {
        return hasPattern("failed to NET register");
    }

    // Direct AllGather path selection (requires NCCL_DEBUG_SUBSYS to include TUNING for
    // the "used" marker and INIT for the "disabled" markers).
    bool usedDirectAllGather() const
    {
        return hasPattern("RCCL DIRECT ALLGATHER count");
    }

    bool directAllGatherDisabled() const
    {
        return hasPattern("RCCL DIRECT ALLGATHER has been disabled") ||
               hasPattern("RCCL DIRECT ALLGATHER disabled") ||
               hasPattern("Direct AllGather disabled");
    }

    bool usedSymmetricCollective(const std::string& collective) const
    {
        return hasPattern(collective + " [Symmetric]:");
    }

    bool usedLegacyCollective(const std::string& collective) const
    {
        const std::regex re(collective + ": [^\\n]*-> Algo ");
        return std::regex_search(m_content, re);
    }

    // AlltoAll has no ring/tree kernel, so it never prints "AlltoAll: ... -> Algo".
    // The selector logs this on the way to the direct p2p path. TUNING must be on.
    bool usedDirectAlltoAll() const
    {
        return hasPattern("A2A CE-registered disqualified") ||
               hasPattern("A2A DDA disqualified");
    }

    bool usedNonSymmetricWindowRegistration() const
    {
        return hasPattern("windowRegisterNonSym:");
    }

    std::string getSummary() const
    {
        std::ostringstream ss;
        ss << "REG Log: ";
        if (hasIPCReuse()) ss << "[IPC-REUSE] ";
        if (hasNETReuse()) ss << "[NET-REUSE] ";
        if (hasIPCRegistration()) ss << "[IPC-REG] ";
        if (hasNETRegistration()) ss << "[NET-REG] ";
        if (hasIPCFailure()) ss << "[IPC-FAIL] ";
        if (hasNETFailure()) ss << "[NET-FAIL] ";
        if (usedDirectAllGather()) ss << "[DIRECT-AG] ";
        if (directAllGatherDisabled()) ss << "[DIRECT-AG-OFF] ";
        if (!hasAnyRegistrationSuccess() && !hasIPCFailure() && !hasNETFailure()) ss << "[NO-REG]";
        return ss.str();
    }

    size_t getContentLength() const { return m_content.size(); }

private:
    bool hasPattern(const std::string& pattern) const
    {
        return m_content.find(pattern) != std::string::npos;
    }

    std::string m_content;
};

// Registration Test Base Class
class RegistrationTestBase : public MPITestBase
{
protected:
    void SetUp() override
    {
        MPITestBase::SetUp();
        // Later tests in this process append to the same rank log. Slice from
        // here so an earlier numSegments line cannot satisfy this test.
        logStartOffset_ = MPIHelpers::getFileSizeBytes(
            MPIHelpers::getRankLogFilePath(getTestMpiRank()));
    }

    struct RegInfo {
        void* buffer = nullptr;
        void* handle = nullptr;
        size_t size = 0;
        bool registered = false;
    };

    // Buffer Management
    RegInfo allocateAndRegister(size_t size)
    {
        RegInfo info;
        info.size = size;

        // VMM-aware so registration exercises the cuMem path when cuMem is on.
        if (allocateDeviceBuffer(&info.buffer, size) != ncclSuccess) {
            return info;
        }

        ncclResult_t result = ncclCommRegister(getActiveCommunicator(),
                                                info.buffer, size, &info.handle);
        info.registered = (result == ncclSuccess && info.handle != nullptr);

        return info;
    }

    void cleanupRegInfo(RegInfo& info)
    {
        if (info.handle) {
            ncclCommDeregister(getActiveCommunicator(), info.handle);
            info.handle = nullptr;
        }
        if (info.buffer) {
            (void)freeDeviceBuffer(info.buffer);
            info.buffer = nullptr;
        }
        info.registered = false;
    }

    // Test Setup
    bool setupMultiNode(int minRanks = 2, int minNodes = 2)
    {
        int nodeCount = MPITestConstants::detectNodeCount();
        if (nodeCount < minNodes) {
            return false;
        }
        if (!validateTestPrerequisites(minRanks, kNoProcessLimit,
                                        kNoPowerOfTwoRequired, minNodes, kNoNodeLimit)) {
            return false;
        }
        return (createTestCommunicator() == ncclSuccess);
    }

    // Environment Checks
    bool isUBREnabled()
    {
        const char* localReg = getenv("NCCL_LOCAL_REGISTER");
        return (localReg && std::string(localReg) == "1");
    }

    bool isGraphRegisterEnabled()
    {
        const char* graphReg = getenv("NCCL_GRAPH_REGISTER");
        return (graphReg && std::string(graphReg) == "1");
    }

    void enableGraphRegisterLogging()
    {
        // Registration success is sniffed from NCCL_REG INFO lines. The MPI
        // harness defaults to NCCL_DEBUG=WARN, which hides them.
        setenv("NCCL_DEBUG", "INFO", 1);
        setenv("NCCL_DEBUG_SUBSYS", "REG", 1);
    }

    bool isCuMemEnabled()
    {
        const char* cuMem = getenv("NCCL_CUMEM_ENABLE");
        return (cuMem && std::string(cuMem) == "1");
    }

    bool isMultiSegmentRegisterEnabled()
    {
        const char* mseg = getenv("NCCL_MULTI_SEGMENT_REGISTER");
        return (!mseg || std::string(mseg) != "0");
    }

    bool isWinEnabled()
    {
        const char* win = getenv("NCCL_WIN_ENABLE");
        return (!win || std::string(win) != "0");
    }

    bool isGinEnabled()
    {
        const char* en = getenv("NCCL_GIN_ENABLE");
        if (en && std::string(en) == "0") return false;
        const char* type = getenv("NCCL_GIN_TYPE");
        return (type && std::string(type) == "2");
    }

    bool isElasticBufferRegisterEnabled()
    {
        const char* en = getenv("NCCL_ELASTIC_BUFFER_REGISTER");
        return (!en || std::string(en) != "0");
    }

    bool isSymSysmemHandleReuseEnabled()
    {
        const char* reuse = getenv("NCCL_SYM_REUSE_SYSMEM_HANDLES");
        return (reuse && std::string(reuse) == "1");
    }

    bool isPerRankLoggingEnabled() { return MPIHelpers::isPerRankLoggingEnabled(); }

    std::uintmax_t logStartOffset_ = 0;

    // Log File Access
    std::string readRankLogFile()
    {
        const std::string full = MPIHelpers::readRankLogFile(getTestMpiRank());
        if (logStartOffset_ >= full.size())
            return {};
        return full.substr(static_cast<std::size_t>(logStartOffset_));
    }

    REGLogChecker getLogChecker()
    {
        return REGLogChecker(readRankLogFile());
    }

    // Data Initialization and Verification
    template<typename T>
    void initSendBuffer(void* buffer, size_t count, int rank)
    {
        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<T>(buffer, count,
            [rank](size_t) { return static_cast<T>(static_cast<float>(rank + 1)); }));
    }

    template<typename T>
    bool verifyAllReduceResult(void* buffer, size_t count, int nRanks)
    {
        T expected = static_cast<T>(static_cast<float>(nRanks * (nRanks + 1) / 2));
        return verifyBufferData<T>(buffer, count, [expected](size_t) { return expected; });
    }

    template<typename T>
    bool verifyReduceScatterResult(void* buffer, size_t count, int nRanks)
    {
        T expected = static_cast<T>(static_cast<float>(nRanks * (nRanks + 1) / 2));
        return verifyBufferData<T>(buffer, count, [expected](size_t) { return expected; });
    }

    template<typename T>
    bool verifyAllGatherResult(void* buffer, size_t countPerRank, int nRanks)
    {
        return verifyBufferData<T>(buffer, countPerRank * nRanks,
            [countPerRank](size_t i) {
                int srcRank = i / countPerRank;
                return static_cast<T>(static_cast<float>(srcRank + 1));
            });
    }

    template<typename T>
    bool verifyBroadcastResult(void* buffer, size_t count, T rootValue)
    {
        return verifyBufferData<T>(buffer, count,
            [rootValue](size_t) { return rootValue; });
    }
};

#endif // MPI_TESTS_ENABLED
#endif // RCCL_TEST_REGISTRATION_TEST_SUPPORT_HPP
