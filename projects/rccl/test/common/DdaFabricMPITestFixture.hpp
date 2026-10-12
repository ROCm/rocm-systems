/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaFabricMPITestFixture.hpp
 * @brief Shared MPI fixture for end-to-end DDA fabric tests: COLL log capture, coordinated
 *        gfx1250 and fabric-initialised skips, log needle counting and a bounded stream wait.
 */

#pragma once

#ifdef MPI_TESTS_ENABLED

#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "TestChecks.hpp"
#include "comm.h" // internal: struct ncclComm::ddaFabricBarrierState

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

namespace DdaFabricMPI
{

// Opt-in CI strictness: when set to 1, a non-gfx1250 device or an uninitialized fabric DDA fails instead of skipping.
constexpr const char* kRequireFabricEnvVar = "UT_DDA_FABRIC_REQUIRE_INIT";

// Every rank must call this; a rank-local env difference still yields one FAIL-or-SKIP decision for the job.
inline bool fabricInitRequired()
{
    const char* value = std::getenv(kRequireFabricEnvVar);
    int         flag  = (value != nullptr && std::string(value) == "1") ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &flag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return flag == 1;
}

inline bool isGfx1250Device()
{
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
    {
        return false;
    }
    return std::string(props.gcnArchName).find("gfx1250") != std::string::npos;
}

inline size_t countOccurrences(const std::string& haystack, const char* needle)
{
    const std::string pattern(needle);
    size_t            count = 0;
    for(size_t pos = haystack.find(pattern); pos != std::string::npos;
        pos        = haystack.find(pattern, pos + pattern.size()))
    {
        ++count;
    }
    return count;
}

// Both sources may carry the same lines, so take the larger count instead of the sum.
inline size_t countLogNeedle(const MPIHelpers::TestLogAssertionContext& logCtx, const char* needle)
{
    return std::max(countOccurrences(logCtx.readNcclDebugLog(), needle),
                    countOccurrences(logCtx.readPerRankStderrLog(), needle));
}

inline bool logContainsNeedle(const MPIHelpers::TestLogAssertionContext& logCtx, const char* needle)
{
    return countLogNeedle(logCtx, needle) > 0;
}

// Polls so that a kernel spinning on a peer flag that never arrives is reported instead of hanging.
// Returns hipSuccess, the stream's error, or hipErrorNotReady when the timeout expires.
inline hipError_t waitStreamBounded(hipStream_t stream, std::chrono::milliseconds timeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while(true)
    {
        const hipError_t query = hipStreamQuery(stream);
        if(query != hipErrorNotReady || std::chrono::steady_clock::now() >= deadline)
        {
            return query;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// Every rank must call this. A spinning kernel or a sticky HIP error cannot be recovered, so either aborts the job.
inline void abortJobUnlessAllRanksDrained(hipError_t status, const char* what)
{
    int flag = status == hipSuccess ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &flag, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(flag == 1)
    {
        return;
    }
    if(status == hipSuccess)
    {
        std::fprintf(stderr, "[rank %d] FATAL: %s failed on another rank; aborting the MPI job\n", getTestMpiRank(),
                     what);
    }
    else if(status == hipErrorNotReady)
    {
        std::fprintf(stderr, "[rank %d] FATAL: %s did not complete before the timeout; aborting the MPI job\n",
                     getTestMpiRank(), what);
    }
    else
    {
        std::fprintf(stderr, "[rank %d] FATAL: %s stream error %s; aborting the MPI job\n", getTestMpiRank(), what,
                     hipGetErrorString(status));
    }
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

} // namespace DdaFabricMPI

// Coordinated skip, escalated to FAIL on every rank under UT_DDA_FABRIC_REQUIRE_INIT=1. Expand in the test body
// or in a helper whose call ends the body; every rank must reach it with the same (coordinated) reason.
#define DDA_FABRIC_SKIP_OR_FAIL(reasonExpr)                                                                 \
    do                                                                                                      \
    {                                                                                                       \
        const std::string ddaFabricReason = (reasonExpr);                                                  \
        if(!ddaFabricReason.empty())                                                                        \
        {                                                                                                   \
            if(DdaFabricMPI::fabricInitRequired())                                                          \
            {                                                                                               \
                FAIL() << ddaFabricReason << " (" << DdaFabricMPI::kRequireFabricEnvVar << "=1)";          \
            }                                                                                               \
            GTEST_SKIP() << ddaFabricReason;                                                                \
        }                                                                                                   \
    } while(0)

/**
 * @class DdaFabricMPITest
 * @brief NCCL_DEBUG=INFO with COLL/TUNING capture per test, plus coordinated DDA fabric skips.
 */
class DdaFabricMPITest : public MPITestBase
{
protected:
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugGuard_;
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugSubsysGuard_;
    std::unique_ptr<MPIHelpers::TestLogAssertionContext> logCtx_;

    void SetUp() override
    {
        MPITestBase::SetUp();
        debugGuard_       = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG", "INFO");
        debugSubsysGuard_ = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG_SUBSYS", "NET,INIT,COLL,TUNING");
        logCtx_           = std::make_unique<MPIHelpers::TestLogAssertionContext>(
            MPIHelpers::makeCombinedAssertionLogOptions(getTestMpiRank()));
    }

    void TearDown() override
    {
        MPITestBase::TearDown();
        logCtx_.reset();
        debugSubsysGuard_.reset();
        debugGuard_.reset();
    }

    // Every rank must skip together, or the ones that did not would block in the next collective.
    static std::string archSkipReason(const char* reason)
    {
        return mpiCoordinatedSkipReason(!DdaFabricMPI::isGfx1250Device(), reason);
    }

    // After createTestCommunicator(): without MNNVL, or on a multi-clique comm, fabric DDA never initializes.
    std::string fabricInitSkipReason()
    {
        return mpiCoordinatedSkipReason(getActiveCommunicator()->ddaFabricBarrierState == nullptr,
                                        "DDA fabric path did not initialize");
    }

    size_t countLogNeedle(const char* needle) const
    {
        return DdaFabricMPI::countLogNeedle(*logCtx_, needle);
    }

    bool logContainsNeedle(const char* needle) const
    {
        return DdaFabricMPI::logContainsNeedle(*logCtx_, needle);
    }
};

#endif // MPI_TESTS_ENABLED
