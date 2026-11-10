/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file NcclTimeoutMPITests.cpp
 * @brief Consolidated MPI coverage for ncclTimeout across all layers (AICOMNET-193).
 *
 * Covers three test suites in one file:
 *
 * TimeoutMPITest — host async-error API on a live comm (no device kernel):
 *   - AsyncErrorRoundTrip: set ncclTimeout → GetAsyncError → clear → AllReduce OK
 *   - AsyncErrorWithInFlightOp: inject during AllReduce, verify advisory (no corruption)
 *   - ErrorStringConsistentAcrossRanks: "timeout" string is uniform on all ranks
 *
 * LsaBarrierTimeoutMPITest — real LSA device barrier timeout (requires NCCL_CUMEM_ENABLE=1):
 *   - HealthyBarrierReturnsSuccess: all peers arrive → ncclSuccess
 *   - AbsentPeerProducesTimeout: one peer absent → ncclTimeout
 *   - ZeroBudgetAbsentPeerTimesOut: budget=0 + absent → immediate ncclTimeout
 *   - RepeatedHealthyBarriersSucceed: 64 sequential healthy barriers, no drift
 *   - RecoversAfterTimeout: timeout then fresh healthy barrier succeeds
 *   - BackToBackTimeouts: 4 sequential stuck barriers, each yields ncclTimeout
 *
 * GinBarrierTimeoutMPITest — real GIN device barrier timeout
 *                            (requires NCCL_GIN_TYPE=2, NCCL_CUMEM_ENABLE=1).
 * AICOMRCCL-2017 / NVIDIA/nccl#2279: the GIN_ProxyNthreads_2 and _4 runner
 * arms re-run HealthyBarrierReturnsSuccess with NCCL_GIN_PROXY_NTHREADS=2
 * and 4 (host progress threads only exist on PROXY). The other cases below
 * stay on the default thread count.
 *   - HealthyBarrierReturnsSuccess
 *   - AbsentPeerProducesTimeout
 *   - ZeroBudgetAbsentPeerTimesOut
 *   - RepeatedHealthyBarriersSucceed: 32 sequential healthy barriers
 *   - RecoversAfterTimeout
 *   - BackToBackTimeouts
 *
 * HybridBarrierTimeoutMPITest — 4-arg ncclBarrierSession::sync (LSA then rail GIN),
 *                               2 nodes x 2 ranks, same env as the GIN suite:
 *   - HealthyBarrierReturnsSuccess
 *   - AbsentLsaPeerStillSignalsRail: rail stage still runs after the LSA timeout
 *   - AbsentLsaAndRailPeersShareOneBudget: both stages time out within one budget
 *
 * Timing note: clock64() is arch-dependent. Tests use a generous kHealthy budget
 * so healthy cases never false-trip, and rely on "absent peer guarantees expiry"
 * for timeout cases — never on a precise cycle-to-wallclock mapping.
 */

#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#include "nccl.h"
#include "nccl_device.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <hip/hip_runtime.h>
#include <mpi.h>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;

// ============================================================
// Shared helpers
// ============================================================

namespace {

constexpr uint64_t kHealthyTimeoutCycles = 5000000000ULL;
constexpr uint64_t kShortTimeoutCycles   = 200000000ULL;

// Every case needs a real inter-node rail, so at least one rank on each of two
// nodes.
constexpr int kMinProcessesForGinTimeout = 2;
constexpr int kMinNodesForGinTimeout     = 2;

// The absent-peer cases withhold the last rail rank, so the rail must hold
// exactly one other rank for the barrier to have someone to wait on. The
// healthy cases only need a rail that spans more than one rank.
constexpr int kAbsentPeerRailRanks    = 2;
constexpr int kMinNonTrivialRailRanks = 2;

// Named at the call sites so setUpGinDevComm()'s rail requirement does not read
// as a bare boolean.
constexpr bool kRequireTwoRankRail = true;
constexpr bool kAnyUniformRail     = false;

// --- Collective helpers ---

bool allocFill(float** ptr, int n, float val) {
    if (hipMalloc(ptr, n * sizeof(float)) != hipSuccess) return false;
    float* h = new float[n];
    for (int i = 0; i < n; ++i) h[i] = val;
    bool ok = (hipMemcpy(*ptr, h, n * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    delete[] h;
    return ok;
}

bool checkAllReduce(float* dBuf, int n, float fillPerRank, int worldSize) {
    float* h = new float[n];
    bool ok = (hipMemcpy(h, dBuf, n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
    if (ok) {
        const float expected = fillPerRank * static_cast<float>(worldSize);
        for (int i = 0; i < n; ++i) if (h[i] != expected) { ok = false; break; }
    }
    delete[] h;
    return ok;
}

// --- LSA barrier helpers ---

__global__ void lsaBarrierTimeoutKernel(struct ncclDevComm devComm,
                                        uint64_t timeoutCycles, int* outResult) {
    ncclLsaBarrierSession<ncclCoopCta> bar{
        ncclCoopCta(), devComm, ncclTeamTagLsa(), /*index=*/0u};
    ncclResult_t r = bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, timeoutCycles);
    if (threadIdx.x == 0 && blockIdx.x == 0) *outResult = static_cast<int>(r);
}

__global__ void lsaRepeatedBarrierKernel(struct ncclDevComm devComm,
                                         uint64_t timeoutCycles, int iters, int* outCount) {
    int successes = 0;
    for (int i = 0; i < iters; ++i) {
        ncclLsaBarrierSession<ncclCoopCta> bar{
            ncclCoopCta(), devComm, ncclTeamTagLsa(), /*index=*/0u};
        if (bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, timeoutCycles) == ncclSuccess)
            ++successes;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) *outCount = successes;
}

ncclResult_t createLsaDevComm(ncclComm_t comm, int nBarriers, ncclDevComm* out) {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = nBarriers;
    return ncclDevCommCreate(comm, &reqs, out);
}

int runLsaBarrier(ncclDevComm devComm, hipStream_t stream, uint64_t timeoutCycles) {
    int* dResult = nullptr;
    if (hipMalloc(&dResult, sizeof(int)) != hipSuccess) return -100;
    int sentinel = -1;
    (void)hipMemcpy(dResult, &sentinel, sizeof(int), hipMemcpyHostToDevice);
    lsaBarrierTimeoutKernel<<<1, 256, 0, stream>>>(devComm, timeoutCycles, dResult);
    (void)hipStreamSynchronize(stream);
    int hResult = -1;
    (void)hipMemcpy(&hResult, dResult, sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(dResult);
    return hResult;
}

// --- GIN barrier helpers ---

std::string ginBarrierSkipReason() {
    if (const char* e = std::getenv("NCCL_GIN_ENABLE"); e && std::strcmp(e, "0") == 0)
        return "GIN explicitly disabled (NCCL_GIN_ENABLE=0)";
    const char* ginType = std::getenv("NCCL_GIN_TYPE");
    if (!ginType || std::atoi(ginType) != 2)
        return "NCCL_GIN_TYPE=2 required";
    const char* cumem = std::getenv("NCCL_CUMEM_ENABLE");
    if (!cumem || std::strcmp(cumem, "1") != 0)
        return "NCCL_CUMEM_ENABLE=1 required";
    // Single-node needs intranet mode.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    int nodeSize = 0, worldSize = 0;
    MPI_Comm_size(nodeComm, &nodeSize);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_free(&nodeComm);
    if (nodeSize == worldSize) {
        if (const char* e = std::getenv("RCCL_ENABLE_INTRANET"); !e || std::strcmp(e, "1") != 0)
            return "Single-node: RCCL_ENABLE_INTRANET=1 required";
    }
    return "";
}

__global__ void ginBarrierTimeoutKernel(struct ncclDevComm devComm,
                                        uint64_t timeoutCycles, int* outResult) {
    ncclGin gin{devComm, /*ginContext=*/0};
    ncclGinBarrierSession<ncclCoopCta> bar{
        ncclCoopCta(), gin, ncclTeamTagRail{}, /*index=*/0u};
    ncclResult_t r = bar.sync(ncclCoopCta(), cuda::memory_order_relaxed,
                              ncclGinFenceLevel::Relaxed, timeoutCycles);
    // A timeout can return while signal GFDs are still queued. Drain them
    // before host-side teardown or a follow-up devComm is created.
    gin.flush(ncclCoopCta());
    if (threadIdx.x == 0 && blockIdx.x == 0) *outResult = static_cast<int>(r);
}

__global__ void ginRepeatedBarrierKernel(struct ncclDevComm devComm,
                                         uint64_t timeoutCycles, int iters, int* outCount) {
    ncclGin gin{devComm, /*ginContext=*/0};
    int successes = 0;
    for (int i = 0; i < iters; ++i) {
        ncclGinBarrierSession<ncclCoopCta> bar{
            ncclCoopCta(), gin, ncclTeamTagRail{}, /*index=*/0u};
        if (bar.sync(ncclCoopCta(), cuda::memory_order_relaxed,
                     ncclGinFenceLevel::Relaxed, timeoutCycles) == ncclSuccess)
            ++successes;
    }
    gin.flush(ncclCoopCta());
    if (threadIdx.x == 0 && blockIdx.x == 0) *outCount = successes;
}

ncclResult_t createGinDevComm(ncclComm_t comm, int nBarriers, ncclDevComm* out) {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.railGinBarrierCount = nBarriers;
    reqs.ginConnectionType   = NCCL_GIN_CONNECTION_RAIL;
    return ncclDevCommCreate(comm, &reqs, out);
}

int runGinBarrier(ncclDevComm devComm, hipStream_t stream, uint64_t timeoutCycles) {
    int* dResult = nullptr;
    if (hipMalloc(&dResult, sizeof(int)) != hipSuccess) return -100;
    int sentinel = -1;
    (void)hipMemcpy(dResult, &sentinel, sizeof(int), hipMemcpyHostToDevice);
    ginBarrierTimeoutKernel<<<1, 256, 0, stream>>>(devComm, timeoutCycles, dResult);
    (void)hipStreamSynchronize(stream);
    int hResult = -1;
    (void)hipMemcpy(&hResult, dResult, sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(dResult);
    return hResult;
}

// --- Hybrid (LSA + rail GIN) barrier helpers ---

// Covers a kHealthyTimeoutCycles run on any arch; a stuck stage fails here instead of hanging.
constexpr int kHybridKernelWaitSeconds = 120;

struct HybridBarrierOut {
    int result;
    unsigned long long elapsedCycles;
};

__global__ void hybridBarrierTimeoutKernel(struct ncclDevComm devComm,
                                           uint64_t timeoutCycles, HybridBarrierOut* out) {
    ncclGin gin{devComm, /*ginContext=*/0};
    {
        ncclBarrierSession<ncclCoopCta> bar{
            ncclCoopCta(), ncclTeamTagWorld(), gin, /*index=*/0u};
        const uint64_t start = clock64();
        ncclResult_t r = bar.sync(ncclCoopCta(), cuda::memory_order_relaxed,
                                  ncclGinFenceLevel::None, timeoutCycles);
        const uint64_t elapsed = clock64() - start;
        // Only the thread waiting on a peer sees that peer's timeout.
        if (r != ncclSuccess) atomicCAS(&out->result, static_cast<int>(ncclSuccess), static_cast<int>(r));
        if (threadIdx.x == 0) out->elapsedCycles = elapsed;
    }
    gin.flush(ncclCoopCta());
}

bool waitStreamWithin(hipStream_t stream, int seconds) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    while (hipStreamQuery(stream) == hipErrorNotReady) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return true;
}

struct HybridBarrierRun {
    int result = -1;
    unsigned long long elapsedCycles = 0;
    bool finishedInTime = false;
};

// Present ranks run first with a bounded wait; absent ranks then run a zero-budget
// barrier so any rank still stuck on them is released before teardown.
HybridBarrierRun runHybridBarrier(ncclDevComm devComm, hipStream_t stream,
                                  bool isAbsent, uint64_t timeoutCycles) {
    HybridBarrierRun run;
    HybridBarrierOut* dOut = nullptr;
    if (hipMalloc(&dOut, sizeof(*dOut)) != hipSuccess) return run;
    HybridBarrierOut init{static_cast<int>(ncclSuccess), 0};
    (void)hipMemcpy(dOut, &init, sizeof(init), hipMemcpyHostToDevice);
    MPI_Barrier(MPI_COMM_WORLD);
    if (!isAbsent) {
        hybridBarrierTimeoutKernel<<<1, 256, 0, stream>>>(devComm, timeoutCycles, dOut);
        run.finishedInTime = waitStreamWithin(stream, kHybridKernelWaitSeconds);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (isAbsent) {
        hybridBarrierTimeoutKernel<<<1, 256, 0, stream>>>(devComm, /*timeoutCycles=*/0ULL, dOut);
        run.finishedInTime = true;
    }
    (void)hipStreamSynchronize(stream);
    HybridBarrierOut h{-1, 0};
    (void)hipMemcpy(&h, dOut, sizeof(h), hipMemcpyDeviceToHost);
    (void)hipFree(dOut);
    run.result = h.result;
    run.elapsedCycles = h.elapsedCycles;
    MPI_Barrier(MPI_COMM_WORLD);
    return run;
}

} // namespace

// ============================================================
// TimeoutMPITest — host async-error API (3 tests)
// ============================================================

class TimeoutMPITest : public MPITestBase {};

// Set ncclTimeout, read it back, clear, verify AllReduce still works.
// Also tests advisory property: injecting during AllReduce doesn't corrupt data.
TEST_F(TimeoutMPITest, AsyncErrorRoundTrip)
{
    SKIP_UNLESS_MPI_PREREQS(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit);
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int worldSize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Set + verify + clear.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclTimeout));
    ncclResult_t observed = ncclSuccess;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &observed));
    ASSERT_MPI_EQ(ncclTimeout, observed);
    ASSERT_MPI_TRUE(std::strcmp(ncclGetErrorString(observed), "timeout") == 0);
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclSuccess));
    MPI_Barrier(MPI_COMM_WORLD);

    // Comm must be functional after clearing.
    const int n = 1024;
    float* sendD = nullptr; float* recvD = nullptr;
    ASSERT_TRUE(allocFill(&sendD, n, 1.0f));
    ASSERT_TRUE(allocFill(&recvD, n, 0.0f));
    SCOPE_EXIT(if (sendD)(void)hipFree(sendD); if (recvD)(void)hipFree(recvD));
    ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(sendD, recvD, n, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));
    ASSERT_MPI_TRUE(checkAllReduce(recvD, n, 1.0f, worldSize));

    ncclResult_t after = ncclTimeout;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &after));
    ASSERT_MPI_EQ(ncclSuccess, after);
}

// ncclTimeout injected while AllReduce is in-flight: advisory only,
// does not corrupt result, clearable, comm healthy afterwards.
TEST_F(TimeoutMPITest, AsyncErrorWithInFlightOp)
{
    SKIP_UNLESS_MPI_PREREQS(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit);
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int worldSize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    const int n = 64 * 1024;
    float* sendD = nullptr; float* recvD = nullptr;
    ASSERT_TRUE(allocFill(&sendD, n, 2.0f));
    ASSERT_TRUE(allocFill(&recvD, n, 0.0f));
    SCOPE_EXIT(if (sendD)(void)hipFree(sendD); if (recvD)(void)hipFree(recvD));

    ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(sendD, recvD, n, ncclFloat, ncclSum, comm, stream));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclTimeout));  // injected mid-flight
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    // Data correctness: advisory error must not corrupt the collective result.
    ASSERT_MPI_TRUE(checkAllReduce(recvD, n, 2.0f, worldSize));

    // Error visible, then clearable.
    ncclResult_t state = ncclSuccess;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &state));
    ASSERT_MPI_EQ(ncclTimeout, state);
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclSuccess));
    MPI_Barrier(MPI_COMM_WORLD);
    ncclResult_t clean = ncclTimeout;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &clean));
    ASSERT_MPI_EQ(ncclSuccess, clean);
}

// "timeout" string must be identical on every rank/node.
TEST_F(TimeoutMPITest, ErrorStringConsistentAcrossRanks)
{
    SKIP_UNLESS_MPI_PREREQS(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit);
    const char* s = ncclGetErrorString(ncclTimeout);
    ASSERT_MPI_TRUE(s != nullptr && std::strcmp(s, "timeout") == 0);
    // Verify every rank produced the same string length via AllReduce(min/max).
    // If any rank returns a different string, min != max and the test fails.
    int len = static_cast<int>(std::strlen(s));
    int lenMin = len, lenMax = len;
    MPI_Allreduce(MPI_IN_PLACE, &lenMin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &lenMax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    ASSERT_MPI_EQ(lenMin, lenMax);
}

// ============================================================
// LsaBarrierTimeoutMPITest — real LSA device barrier (6 tests)
// ============================================================

class LsaBarrierTimeoutMPITest : public MPITestBase {
 protected:
    std::string skipReason_;

    bool setUpLsaDevComm(int nBarriers, ncclComm_t* commOut,
                         hipStream_t* streamOut, ncclDevComm* devCommOut) {
        skipReason_.clear();
        if (!validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit)) {
            ADD_FAILURE() << "Test requires at least 2 MPI processes"; return false;
        }
        if (createTestCommunicator() != ncclSuccess) {
            ADD_FAILURE() << "createTestCommunicator failed"; return false;
        }
        ncclComm_t comm = getActiveCommunicator();
        ncclResult_t devRc = createLsaDevComm(comm, nBarriers, devCommOut);
        if (devRc != ncclSuccess) {
            skipReason_ = std::string("LSA devComm requires symmetric memory (cuMem/VMM); "
                                      "ncclDevCommCreate returned ") + ncclGetErrorString(devRc);
            return false;
        }
        *commOut = comm; *streamOut = getActiveStream(); return true;
    }
};

#define LSA_SETUP_OR_BAIL() \
    do { if (!skipReason_.empty()) GTEST_SKIP() << skipReason_; return; } while (0)

TEST_F(LsaBarrierTimeoutMPITest, HealthyBarrierReturnsSuccess)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpLsaDevComm(1, &comm, &stream, &devComm)) LSA_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));
    MPI_Barrier(MPI_COMM_WORLD);
    ASSERT_MPI_EQ(static_cast<int>(ncclSuccess), runLsaBarrier(devComm, stream, kHealthyTimeoutCycles));
}

TEST_F(LsaBarrierTimeoutMPITest, AbsentPeerProducesTimeout)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpLsaDevComm(1, &comm, &stream, &devComm)) LSA_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));
    int rank = -1, nRanks = -1;
    ncclCommUserRank(comm, &rank); ncclCommCount(comm, &nRanks);
    // The single-absent-peer timeout only starves reliably with a 2-rank LSA team;
    // at N>2 the all-to-all inbox + rolling-epoch wait can complete, so skip instead.
    // Decide collectively (max LSA size across the job) so every rank skips or runs
    // together — a per-rank skip would deadlock the MPI_Barrier below.
    ncclTeam_t lsaTeam = ncclTeamLsa(comm);
    int maxLsaNRanks = lsaTeam.nRanks;
    MPI_Allreduce(MPI_IN_PLACE, &maxLsaNRanks, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (maxLsaNRanks != 2) {
        GTEST_SKIP() << "LSA absent-peer timeout requires a 2-rank LSA team; got "
                     << maxLsaNRanks << ". Run with 2 ranks sharing one LSA domain.";
    }
    const bool isAbsent = (lsaTeam.rank == lsaTeam.nRanks - 1);
    int hResult = static_cast<int>(ncclSuccess);
    if (!isAbsent) {
        hResult = runLsaBarrier(devComm, stream, kShortTimeoutCycles);
        EXPECT_EQ(static_cast<int>(ncclTimeout), hResult)
            << "LSA rank " << lsaTeam.rank << " expected ncclTimeout, got "
            << ncclGetErrorString(static_cast<ncclResult_t>(hResult));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    ASSERT_MPI_TRUE((isAbsent || hResult == static_cast<int>(ncclTimeout)) ? 1 : 0);
}

TEST_F(LsaBarrierTimeoutMPITest, ZeroBudgetAbsentPeerTimesOut)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpLsaDevComm(1, &comm, &stream, &devComm)) LSA_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));
    int rank = -1, nRanks = -1;
    ncclCommUserRank(comm, &rank); ncclCommCount(comm, &nRanks);
    ncclTeam_t lsaTeam = ncclTeamLsa(comm);
    // The single-absent-peer timeout only starves reliably with a 2-rank LSA team;
    // at N>2 the all-to-all inbox + rolling-epoch wait can complete, so skip instead.
    if (lsaTeam.nRanks != 2) {
        GTEST_SKIP() << "LSA absent-peer timeout requires a 2-rank LSA team; got "
                     << lsaTeam.nRanks << ". Run with 2 ranks sharing one LSA domain.";
    }
    const bool isAbsent = (lsaTeam.rank == lsaTeam.nRanks - 1);
    int hResult = static_cast<int>(ncclSuccess);
    if (!isAbsent) {
        hResult = runLsaBarrier(devComm, stream, /*timeoutCycles=*/0ULL);
        EXPECT_EQ(static_cast<int>(ncclTimeout), hResult)
            << "LSA rank " << lsaTeam.rank << " expected immediate ncclTimeout, got "
            << ncclGetErrorString(static_cast<ncclResult_t>(hResult));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    ASSERT_MPI_TRUE((isAbsent || hResult == static_cast<int>(ncclTimeout)) ? 1 : 0);
}

TEST_F(LsaBarrierTimeoutMPITest, RepeatedHealthyBarriersSucceed)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpLsaDevComm(1, &comm, &stream, &devComm)) LSA_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));
    constexpr int kIters = 64;
    int* dCount = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&dCount, sizeof(int)));
    SCOPE_EXIT(if (dCount)(void)hipFree(dCount));
    int zero = 0;
    HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(dCount, &zero, sizeof(int), hipMemcpyHostToDevice));
    MPI_Barrier(MPI_COMM_WORLD);
    lsaRepeatedBarrierKernel<<<1, 256, 0, stream>>>(devComm, kHealthyTimeoutCycles, kIters, dCount);
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    int hCount = -1;
    HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(&hCount, dCount, sizeof(int), hipMemcpyDeviceToHost));
    ASSERT_MPI_EQ(kIters, hCount);
}

TEST_F(LsaBarrierTimeoutMPITest, RecoversAfterTimeout)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpLsaDevComm(1, &comm, &stream, &devComm)) LSA_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));
    int rank = -1, nRanks = -1;
    ncclCommUserRank(comm, &rank); ncclCommCount(comm, &nRanks);
    ncclTeam_t lsaTeam = ncclTeamLsa(comm);
    // The single-absent-peer timeout only starves reliably with a 2-rank LSA team;
    // at N>2 the all-to-all inbox + rolling-epoch wait can complete, so skip instead.
    if (lsaTeam.nRanks != 2) {
        GTEST_SKIP() << "LSA absent-peer timeout requires a 2-rank LSA team; got "
                     << lsaTeam.nRanks << ". Run with 2 ranks sharing one LSA domain.";
    }
    const bool isAbsent = (lsaTeam.rank == lsaTeam.nRanks - 1);
    if (!isAbsent) {
        EXPECT_EQ(static_cast<int>(ncclTimeout),
                  runLsaBarrier(devComm, stream, kShortTimeoutCycles));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    ncclDevComm devComm2{};
    ASSERT_MPI_EQ(ncclSuccess, createLsaDevComm(comm, 1, &devComm2));
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm2));
    MPI_Barrier(MPI_COMM_WORLD);
    ASSERT_MPI_EQ(static_cast<int>(ncclSuccess),
                  runLsaBarrier(devComm2, stream, kHealthyTimeoutCycles));
}

TEST_F(LsaBarrierTimeoutMPITest, BackToBackTimeouts)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpLsaDevComm(1, &comm, &stream, &devComm)) LSA_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));
    int rank = -1, nRanks = -1;
    ncclCommUserRank(comm, &rank); ncclCommCount(comm, &nRanks);
    ncclTeam_t lsaTeam = ncclTeamLsa(comm);
    // The single-absent-peer timeout only starves reliably with a 2-rank LSA team;
    // at N>2 the all-to-all inbox + rolling-epoch wait can complete, so skip instead.
    if (lsaTeam.nRanks != 2) {
        GTEST_SKIP() << "LSA absent-peer timeout requires a 2-rank LSA team; got "
                     << lsaTeam.nRanks << ". Run with 2 ranks sharing one LSA domain.";
    }
    const bool isAbsent = (lsaTeam.rank == lsaTeam.nRanks - 1);
    constexpr int kRounds = 4;
    int timeouts = 0;
    for (int i = 0; i < kRounds; ++i) {
        ncclDevComm dc{};
        if (createLsaDevComm(comm, 1, &dc) != ncclSuccess) break;
        if (!isAbsent) {
            int r = runLsaBarrier(dc, stream, kShortTimeoutCycles);
            if (r == static_cast<int>(ncclTimeout)) ++timeouts;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        (void)ncclDevCommDestroy(comm, &dc);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (!isAbsent) EXPECT_EQ(kRounds, timeouts);
    MPI_Barrier(MPI_COMM_WORLD);
    ASSERT_MPI_TRUE((isAbsent || timeouts == kRounds) ? 1 : 0);
}

// ============================================================
// GinBarrierTimeoutMPITest — real GIN device barrier (6 tests)
// ============================================================

class GinBarrierTimeoutMPITest : public MPITestBase {
 protected:
    std::string skipReason_;

    bool setUpGinDevComm(int nBarriers, bool requireTwoRankRail,
                         ncclComm_t* commOut, hipStream_t* streamOut,
                         ncclDevComm* devCommOut) {
        skipReason_.clear();
        if (auto reason = ginBarrierSkipReason(); !reason.empty()) {
            skipReason_ = reason; return false;
        }
        if (!validateTestPrerequisites(kMinProcessesForGinTimeout, kNoProcessLimit,
                                       kNoPowerOfTwoRequired,
                                       kMinNodesForGinTimeout, kNoNodeLimit)) {
            skipReason_ = "GIN timeout tests require at least 2 physical nodes";
            return false;
        }
        if (createTestCommunicator() != ncclSuccess) {
            ADD_FAILURE() << "createTestCommunicator failed"; return false;
        }
        ncclComm_t comm = getActiveCommunicator();
        ncclTeam_t railTeam = ncclTeamRail(comm);
        int minRailSize = railTeam.nRanks;
        int maxRailSize = railTeam.nRanks;
        MPI_Allreduce(MPI_IN_PLACE, &minRailSize, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &maxRailSize, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (requireTwoRankRail && (minRailSize != kAbsentPeerRailRanks ||
                                   maxRailSize != kAbsentPeerRailRanks)) {
            skipReason_ =
                "GIN absent-peer timeout tests require two-rank rail teams "
                "(normally exactly two nodes with uniform ranks per node)";
            return false;
        }
        if (!requireTwoRankRail &&
            (minRailSize < kMinNonTrivialRailRanks || minRailSize != maxRailSize)) {
            skipReason_ =
                "Healthy GIN timeout tests require uniform, non-trivial rail teams";
            return false;
        }
        ncclResult_t devRc = createGinDevComm(comm, nBarriers, devCommOut);
        if (devRc != ncclSuccess) {
            skipReason_ = std::string("GIN devComm unavailable; ncclDevCommCreate returned ")
                        + ncclGetErrorString(devRc);
            return false;
        }
        *commOut = comm; *streamOut = getActiveStream(); return true;
    }
};

#define GIN_SETUP_OR_BAIL() \
    do { if (!skipReason_.empty()) GTEST_SKIP() << skipReason_; return; } while (0)

TEST_F(GinBarrierTimeoutMPITest, HealthyBarrierReturnsSuccess)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpGinDevComm(1, kAnyUniformRail, &comm, &stream, &devComm)) GIN_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));
    MPI_Barrier(MPI_COMM_WORLD);
    ASSERT_MPI_EQ(static_cast<int>(ncclSuccess), runGinBarrier(devComm, stream, kHealthyTimeoutCycles));
}

TEST_F(GinBarrierTimeoutMPITest, AbsentPeerProducesTimeout)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpGinDevComm(1, kRequireTwoRankRail, &comm, &stream, &devComm)) GIN_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));
    ncclTeam_t railTeam = ncclTeamRail(comm);
    const bool isAbsent = (railTeam.rank == railTeam.nRanks - 1);
    int hResult = static_cast<int>(ncclSuccess);
    if (!isAbsent) {
        hResult = runGinBarrier(devComm, stream, kShortTimeoutCycles);
        EXPECT_EQ(static_cast<int>(ncclTimeout), hResult)
            << "Rail rank " << railTeam.rank << " expected ncclTimeout, got "
            << ncclGetErrorString(static_cast<ncclResult_t>(hResult));
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (isAbsent) {
        (void)runGinBarrier(devComm, stream, /*timeoutCycles=*/0ULL);
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    ASSERT_MPI_TRUE((isAbsent || hResult == static_cast<int>(ncclTimeout)) ? 1 : 0);
}

TEST_F(GinBarrierTimeoutMPITest, ZeroBudgetAbsentPeerTimesOut)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpGinDevComm(1, kRequireTwoRankRail, &comm, &stream, &devComm)) GIN_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));
    ncclTeam_t railTeam = ncclTeamRail(comm);
    const bool isAbsent = (railTeam.rank == railTeam.nRanks - 1);
    int hResult = static_cast<int>(ncclSuccess);
    if (!isAbsent) {
        hResult = runGinBarrier(devComm, stream, /*timeoutCycles=*/0ULL);
        EXPECT_EQ(static_cast<int>(ncclTimeout), hResult)
            << "Rail rank " << railTeam.rank << " expected immediate ncclTimeout, got "
            << ncclGetErrorString(static_cast<ncclResult_t>(hResult));
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (isAbsent) {
        (void)runGinBarrier(devComm, stream, 0ULL);
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    ASSERT_MPI_TRUE((isAbsent || hResult == static_cast<int>(ncclTimeout)) ? 1 : 0);
}

TEST_F(GinBarrierTimeoutMPITest, RepeatedHealthyBarriersSucceed)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpGinDevComm(1, kAnyUniformRail, &comm, &stream, &devComm)) GIN_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));
    constexpr int kIters = 32;
    int* dCount = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&dCount, sizeof(int)));
    SCOPE_EXIT(if (dCount)(void)hipFree(dCount));
    int zero = 0;
    HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(dCount, &zero, sizeof(int), hipMemcpyHostToDevice));
    MPI_Barrier(MPI_COMM_WORLD);
    ginRepeatedBarrierKernel<<<1, 256, 0, stream>>>(devComm, kHealthyTimeoutCycles, kIters, dCount);
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    int hCount = -1;
    HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(&hCount, dCount, sizeof(int), hipMemcpyDeviceToHost));
    ASSERT_MPI_EQ(kIters, hCount);
}

TEST_F(GinBarrierTimeoutMPITest, RecoversAfterTimeout)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpGinDevComm(1, kRequireTwoRankRail, &comm, &stream, &devComm)) GIN_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));
    ncclTeam_t railTeam = ncclTeamRail(comm);
    const bool isAbsent = (railTeam.rank == railTeam.nRanks - 1);
    if (!isAbsent) {
        EXPECT_EQ(static_cast<int>(ncclTimeout),
                  runGinBarrier(devComm, stream, kShortTimeoutCycles));
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (isAbsent) {
        (void)runGinBarrier(devComm, stream, 0ULL);
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    ncclDevComm devComm2{};
    ASSERT_MPI_EQ(ncclSuccess, createGinDevComm(comm, 1, &devComm2));
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm2));
    MPI_Barrier(MPI_COMM_WORLD);
    ASSERT_MPI_EQ(static_cast<int>(ncclSuccess),
                  runGinBarrier(devComm2, stream, kHealthyTimeoutCycles));
}

TEST_F(GinBarrierTimeoutMPITest, BackToBackTimeouts)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpGinDevComm(1, kRequireTwoRankRail, &comm, &stream, &devComm)) GIN_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));
    ncclTeam_t railTeam = ncclTeamRail(comm);
    const bool isAbsent = (railTeam.rank == railTeam.nRanks - 1);
    constexpr int kRounds = 4;
    int timeouts = 0;
    for (int i = 0; i < kRounds; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        ncclDevComm dc{};
        if (createGinDevComm(comm, 1, &dc) != ncclSuccess) break;
        if (!isAbsent) {
            int r = runGinBarrier(dc, stream, kShortTimeoutCycles);
            if (r == static_cast<int>(ncclTimeout)) ++timeouts;
            (void)hipStreamSynchronize(stream);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (isAbsent) {
            (void)runGinBarrier(dc, stream, 0ULL);
            (void)hipStreamSynchronize(stream);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        (void)ncclDevCommDestroy(comm, &dc);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (!isAbsent) EXPECT_EQ(kRounds, timeouts);
    MPI_Barrier(MPI_COMM_WORLD);
    ASSERT_MPI_TRUE((isAbsent || timeouts == kRounds) ? 1 : 0);
}

// ============================================================
// HybridBarrierTimeoutMPITest — 4-arg ncclBarrierSession::sync (3 tests)
// ============================================================

// Roles on 2 nodes x 2 ranks: A=(lsa 1, rail 1), B is A's LSA peer, D is A's
// rail peer, C is B's rail peer and D's LSA peer.
struct HybridRoles {
    bool isA = false, isB = false, isC = false, isD = false;
};

class HybridBarrierTimeoutMPITest : public MPITestBase {
 protected:
    std::string skipReason_;

    bool setUpHybridDevComm(ncclComm_t* commOut, hipStream_t* streamOut,
                            ncclDevComm* devCommOut, HybridRoles* rolesOut) {
        skipReason_.clear();
        if (auto reason = ginBarrierSkipReason(); !reason.empty()) {
            skipReason_ = reason; return false;
        }
        if (!validateTestPrerequisites(4, 4, kNoPowerOfTwoRequired, 2, 2)) {
            skipReason_ = "Hybrid barrier timeout tests require exactly 2 nodes x 2 ranks";
            return false;
        }
        if (createTestCommunicator() != ncclSuccess) {
            ADD_FAILURE() << "createTestCommunicator failed"; return false;
        }
        ncclComm_t comm = getActiveCommunicator();
        ncclTeam_t lsaTeam  = ncclTeamLsa(comm);
        ncclTeam_t railTeam = ncclTeamRail(comm);
        int sizes[2]    = {lsaTeam.nRanks, railTeam.nRanks};
        int minSizes[2] = {0, 0}, maxSizes[2] = {0, 0};
        MPI_Allreduce(sizes, minSizes, 2, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(sizes, maxSizes, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (minSizes[0] != 2 || maxSizes[0] != 2 || minSizes[1] != 2 || maxSizes[1] != 2) {
            skipReason_ = "Hybrid barrier timeout tests require 2-rank LSA and rail teams";
            return false;
        }
        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        reqs.barrierCount      = 1;
        reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
        ncclResult_t devRc = ncclDevCommCreate(comm, &reqs, devCommOut);
        int anyFailed = devRc != ncclSuccess ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &anyFailed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (anyFailed) {
            if (devRc == ncclSuccess) (void)ncclDevCommDestroy(comm, devCommOut);
            skipReason_ = std::string("Hybrid devComm unavailable; ncclDevCommCreate returned ")
                        + ncclGetErrorString(devRc);
            return false;
        }
        rolesOut->isA = lsaTeam.rank == 1 && railTeam.rank == 1;
        rolesOut->isB = lsaTeam.rank == 0 && railTeam.rank == 1;
        rolesOut->isC = lsaTeam.rank == 0 && railTeam.rank == 0;
        rolesOut->isD = lsaTeam.rank == 1 && railTeam.rank == 0;
        *commOut = comm; *streamOut = getActiveStream(); return true;
    }
};

#define HYBRID_SETUP_OR_BAIL() \
    do { if (!skipReason_.empty()) GTEST_SKIP() << skipReason_; return; } while (0)

TEST_F(HybridBarrierTimeoutMPITest, HealthyBarrierReturnsSuccess)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{}; HybridRoles roles;
    if (!setUpHybridDevComm(&comm, &stream, &devComm, &roles)) HYBRID_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));
    HybridBarrierRun run = runHybridBarrier(devComm, stream, /*isAbsent=*/false, kHealthyTimeoutCycles);
    ASSERT_MPI_TRUE(run.finishedInTime);
    ASSERT_MPI_EQ(static_cast<int>(ncclSuccess), run.result);
}

// A is absent. B's LSA stage times out, yet B must still signal C on the rail,
// so C (10x+ budget) succeeds; D's rail stage times out on A.
TEST_F(HybridBarrierTimeoutMPITest, AbsentLsaPeerStillSignalsRail)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{}; HybridRoles roles;
    if (!setUpHybridDevComm(&comm, &stream, &devComm, &roles)) HYBRID_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));
    static_assert(kHealthyTimeoutCycles >= 10 * kShortTimeoutCycles, "C needs >= 10x B's budget");
    const uint64_t budget = roles.isC ? kHealthyTimeoutCycles : kShortTimeoutCycles;
    HybridBarrierRun run = runHybridBarrier(devComm, stream, roles.isA, budget);
    const int expected = static_cast<int>(roles.isC ? ncclSuccess : ncclTimeout);
    bool ok = roles.isA || (run.finishedInTime && run.result == expected);
    EXPECT_TRUE(ok) << (roles.isB ? "B" : roles.isC ? "C" : "D") << " expected "
                    << ncclGetErrorString(static_cast<ncclResult_t>(expected)) << ", got "
                    << run.result << (run.finishedInTime ? "" : " (kernel did not finish in time)");
    ASSERT_MPI_TRUE(ok ? 1 : 0);
}

// A and C are absent, so B and D time out in both stages. The rail stage gets
// only what the LSA stage left, so the total stays near one budget, not two.
TEST_F(HybridBarrierTimeoutMPITest, AbsentLsaAndRailPeersShareOneBudget)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{}; HybridRoles roles;
    if (!setUpHybridDevComm(&comm, &stream, &devComm, &roles)) HYBRID_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));
    const bool isAbsent = roles.isA || roles.isC;
    HybridBarrierRun run = runHybridBarrier(devComm, stream, isAbsent, kShortTimeoutCycles);
    const unsigned long long maxCycles = kShortTimeoutCycles + kShortTimeoutCycles / 2;
    bool ok = isAbsent || (run.finishedInTime && run.result == static_cast<int>(ncclTimeout) &&
                           run.elapsedCycles >= kShortTimeoutCycles && run.elapsedCycles < maxCycles);
    EXPECT_TRUE(ok) << (roles.isB ? "B" : "D") << " expected ncclTimeout within ["
                    << kShortTimeoutCycles << ", " << maxCycles << ") cycles, got result "
                    << run.result << " after " << run.elapsedCycles << " cycles"
                    << (run.finishedInTime ? "" : " (kernel did not finish in time)");
    ASSERT_MPI_TRUE(ok ? 1 : 0);
}

#endif // MPI_TESTS_ENABLED
