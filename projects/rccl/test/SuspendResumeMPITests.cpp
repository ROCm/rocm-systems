/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file SuspendResumeMPITests.cpp
 * @brief Multi-process MPI tests for ncclCommSuspend / ncclCommResume /
 *        ncclCommMemStats.
 *
 * Coverage matrix:
 *
 *  Happy paths
 *    - SuspendResumeBasic_BasicCycle              one Suspend->Resume round-trip
 *    - SuspendResumeBasic_MultipleCycles          5 back-to-back cycles
 *    - SuspendResumeBasic_BarrierSync             rank 0 sleeps before
 *                                                 suspend; all other ranks
 *                                                 must block until it joins
 *                                                 the bootstrapBarrier
 *    - MemStats_BeforeAndAfter                    ncclStatGpuMem* reports the
 *                                                 expected values across the
 *                                                 active/suspended/resumed
 *                                                 transitions
 *    - MemStats_Conservation                      total + suspended bytes is
 *                                                 monotonically conserved
 *                                                 across a Suspend
 *    - CollectiveIntegrity_AllReduceAfterResume   real ncclAllReduce round-trip
 *                                                 still produces the right
 *                                                 answer after Suspend/Resume
 *    - CollectiveIntegrity_AllReduceTwoCycles     same after multiple cycles
 *    - Lifecycle_DestroyWhileSuspended            ncclCommDestroy succeeds on
 *                                                 a comm whose memory is
 *                                                 still suspended (the
 *                                                 force-resume-for-destroy
 *                                                 path is exercised)
 *
 *  Argument-validation / corner cases (no bootstrapBarrier reached, so each
 *  rank can fail independently without deadlocking the others):
 *    - ArgValidation_ZeroFlags
 *    - ArgValidation_UnknownFlagsRejected
 *    - ArgValidation_DoubleSuspend
 *    - ArgValidation_ResumeWhenActive
 *    - ArgValidation_DoubleResume
 *    - MemStatsValidation_NullValuePtr
 *    - MemStatsValidation_UnknownStat
 *
 *  Required environment:
 *    - MPI launch with at least 2 processes (1 GPU per rank)
 *    - For maximum signal, run with NCCL_DEBUG=INFO and inspect the
 *      `ncclCommSuspend / ncclCommResume` lines per rank.
 *
 *  Caveats / known issues exercised here only with NCCL_CUMEM_ENABLE=0:
 *    The pre-existing RCCL+ROCm 7.0.x p2p init crash with
 *    NCCL_CUMEM_ENABLE=1 + nRanks>1 is unrelated to suspend/resume but
 *    means these tests must NOT set NCCL_CUMEM_ENABLE=1; in default
 *    (CUMEM=0) configuration the canary roundtrip is the gated piece.
 *
 *  Example invocations:
 *    mpirun -np 2 ./rccl-UnitTestsMPI \
 *           --gtest_filter="SuspendResumeBasic*:SuspendResumeMemStats*"
 *    mpirun -np 4 ./rccl-UnitTestsMPI \
 *           --gtest_filter="SuspendResumeArgValidation*"
 */

#include "DeviceBufferHelpers.hpp"
#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace SuspendResumeTestConfig
{
    constexpr int    kMinRanks         = 2;     // multi-rank tests need >= 2
    constexpr size_t kAllReduceCount   = 1024;  // 4 KiB float buffer
    constexpr float  kEpsilon          = 1e-3f;
    constexpr int    kMultiCycleCount  = 5;
    constexpr int    kBarrierSleepMs   = 200;   // rank-0 stagger time
} // namespace SuspendResumeTestConfig

using namespace SuspendResumeTestConfig;

/**
 * @class SuspendResumeMPITestBase
 * @brief Shared fixture for all suspend/resume MPI tests.
 *
 * Wraps MPITestBase to keep the setup/teardown trivial for individual
 * test cases.  Test cases call `createTestCommunicator()` themselves so
 * that argument-validation tests can also exercise sequences before the
 * comm is built (e.g. NULL-comm checks).
 */
class SuspendResumeMPITestBase : public MPITestBase
{
protected:
    // Convenience accessor: read one ncclCommMemStat as uint64_t. Fails the
    // test on any non-success return so that callers can read the value
    // directly.
    uint64_t readStat(ncclComm_t comm, ncclCommMemStat_t stat)
    {
        uint64_t v = ~uint64_t{0};
        ncclResult_t r = ncclCommMemStats(comm, stat, &v);
        EXPECT_EQ(r, ncclSuccess) << "ncclCommMemStats failed for stat " << (int)stat;
        return v;
    }

    // Run an AllReduce with float-sum and verify each element equals
    // sum(1..nranks).  Returns true on success.
    bool runAllReduceAndVerify(ncclComm_t comm, hipStream_t stream)
    {
        const size_t n     = kAllReduceCount;
        const int    rank  = MPIEnvironment::world_rank;
        const int    nrnks = MPIEnvironment::world_size;

        void* sendbuf = nullptr;
        void* recvbuf = nullptr;
        if (hipMalloc(&sendbuf, n * sizeof(float)) != hipSuccess) return false;
        if (hipMalloc(&recvbuf, n * sizeof(float)) != hipSuccess)
        {
            (void)hipFree(sendbuf);
            return false;
        }
        auto sendGuard = makeDeviceBufferAutoGuard(sendbuf);
        auto recvGuard = makeDeviceBufferAutoGuard(recvbuf);

        // sendbuf := rank+1
        if (initializeBufferWithPattern<float>(
                sendbuf, n,
                [rank](size_t) { return static_cast<float>(rank + 1); }) != hipSuccess)
            return false;
        if (zeroInitializeBuffer<float>(recvbuf, n) != hipSuccess) return false;

        if (ncclAllReduce(sendbuf, recvbuf, n, ncclFloat32, ncclSum, comm, stream)
            != ncclSuccess)
            return false;
        if (hipStreamSynchronize(stream) != hipSuccess) return false;

        const float expected = static_cast<float>(nrnks * (nrnks + 1) / 2);
        return verifyBufferData<float>(
            recvbuf, n,
            [expected](size_t) { return expected; },
            0,
            static_cast<double>(kEpsilon * expected));
    }
};

// ============================================================================
// 1. Basic happy-path tests
// ============================================================================

class SuspendResumeBasic : public SuspendResumeMPITestBase {};

TEST_F(SuspendResumeBasic, BasicCycle)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    EXPECT_EQ(0u, readStat(comm, ncclStatGpuMemSuspended))
        << "Newly initialized comm must report not-suspended";

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    EXPECT_EQ(1u, readStat(comm, ncclStatGpuMemSuspended))
        << "After Suspend, ncclStatGpuMemSuspended must be 1";

    // ncclStatGpuMemSuspend == totalScratch + totalOffload of *tracked*
    // entries (matches NCCL upstream semantics). The canary is not a
    // tracked user allocation and is not counted here. With NCCL_CUMEM_ENABLE=0
    // the comm's internal allocations are not VMM-backed and therefore not
    // tracked, so the count is legitimately zero. With CUMEM=1 it is > 0.
    uint64_t suspBytes = readStat(comm, ncclStatGpuMemSuspend);
    TEST_INFO("Rank %d ncclStatGpuMemSuspend after Suspend = %llu bytes",
              MPIEnvironment::world_rank, (unsigned long long)suspBytes);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));
    EXPECT_EQ(0u, readStat(comm, ncclStatGpuMemSuspended))
        << "After Resume, ncclStatGpuMemSuspended must be 0";
}

TEST_F(SuspendResumeBasic, MultipleCycles)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    for (int i = 0; i < kMultiCycleCount; ++i)
    {
        TEST_INFO("Cycle %d: Suspend", i);
        ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
        EXPECT_EQ(1u, readStat(comm, ncclStatGpuMemSuspended)) << "cycle " << i;

        TEST_INFO("Cycle %d: Resume", i);
        ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));
        EXPECT_EQ(0u, readStat(comm, ncclStatGpuMemSuspended)) << "cycle " << i;
    }
}

TEST_F(SuspendResumeBasic, BarrierSync)
{
    // Rank 0 stages a deliberate delay before calling Suspend. Without the
    // bootstrapBarrier inside ncclCommSuspend, the other ranks would return
    // immediately and the elapsed time on those ranks would be much smaller
    // than rank 0's. With the barrier wired up correctly, every rank must
    // observe at least kBarrierSleepMs of elapsed wall-clock between the
    // (cross-rank) start of the call and its return.
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    // Synchronize start across ranks.
    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::steady_clock::now();

    if (MPIEnvironment::world_rank == 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(kBarrierSleepMs));
    }
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    auto t1 = std::chrono::steady_clock::now();

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         t1 - t0).count();

    // Allow generous slack -- this is a synchronization assertion, not a
    // micro-benchmark: every rank should have waited *at least* roughly
    // half the staged delay.  Without a barrier, the non-zero ranks would
    // typically observe < 10 ms.
    EXPECT_GE(elapsed_ms, kBarrierSleepMs / 2)
        << "Rank " << MPIEnvironment::world_rank
        << ": Suspend returned in " << elapsed_ms
        << " ms (rank 0 staged a " << kBarrierSleepMs
        << " ms delay; expected the barrier to hold every rank)";

    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));
}

// ============================================================================
// 2. Memory-statistics correctness
// ============================================================================

class SuspendResumeMemStats : public SuspendResumeMPITestBase {};

TEST_F(SuspendResumeMemStats, BeforeAndAfter)
{
    // NCCL upstream semantics: ncclCommMemStats reads per-type counters on
    // the memory manager. The counters cover *tracked* allocations only --
    // i.e. entries that ncclMemTrack accepted. With NCCL_CUMEM_ENABLE=0
    // RCCL skips tracking (HIP introspection is unsafe on non-VMM ptrs),
    // so all counters legitimately read zero; the suspended boolean still
    // toggles via the canary round-trip. With CUMEM=1, counters are
    // non-zero and persist across the suspend boundary (Suspend doesn't
    // change totalScratch/totalOffload -- it only flips `released`).
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    uint64_t totalActive = readStat(comm, ncclStatGpuMemTotal);
    uint64_t suspActive  = readStat(comm, ncclStatGpuMemSuspend);
    uint64_t persActive  = readStat(comm, ncclStatGpuMemPersist);
    uint64_t isSusActive = readStat(comm, ncclStatGpuMemSuspended);
    EXPECT_EQ(isSusActive, 0u) << "fresh comm must not be suspended";
    EXPECT_EQ(totalActive, persActive + suspActive)
        << "Total = persist + (scratch+offload) invariant";

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));

    uint64_t totalSusp = readStat(comm, ncclStatGpuMemTotal);
    uint64_t suspSusp  = readStat(comm, ncclStatGpuMemSuspend);
    uint64_t isSusSusp = readStat(comm, ncclStatGpuMemSuspended);
    EXPECT_EQ(isSusSusp, 1u) << "after Suspend, suspended boolean must be 1";
    EXPECT_EQ(totalSusp, totalActive)
        << "Suspend does not change tracked byte counters (mirrors NCCL)";
    EXPECT_EQ(suspSusp, suspActive)
        << "Suspend does not change scratch/offload byte counters";

    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));

    uint64_t totalResumed = readStat(comm, ncclStatGpuMemTotal);
    uint64_t isSusResumed = readStat(comm, ncclStatGpuMemSuspended);
    EXPECT_EQ(isSusResumed, 0u);
    EXPECT_EQ(totalResumed, totalActive)
        << "Resume does not change tracked byte counters either";

    TEST_INFO("rank %d: total=%llu persist=%llu suspendable=%llu "
              "[scratch+offload] (CUMEM gates whether non-zero)",
              MPIEnvironment::world_rank,
              (unsigned long long)totalActive,
              (unsigned long long)persActive,
              (unsigned long long)suspActive);
}

TEST_F(SuspendResumeMemStats, Conservation)
{
    // ncclCommMemStats reports per-type byte counters on the memory
    // manager; Suspend doesn't move bytes between the persist / scratch /
    // offload buckets, it only flips the released boolean. So the
    // identity Total = Persist + (Scratch+Offload) holds in every state,
    // and Total/Persist/Suspend are constant across a Suspend->Resume
    // round-trip. (CPU backup memory used by ncclMemOffload entries is
    // tracked separately on manager->cpuBackupUsage and not part of the
    // public stats.)
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    uint64_t totalA = readStat(comm, ncclStatGpuMemTotal);
    uint64_t persA  = readStat(comm, ncclStatGpuMemPersist);
    uint64_t suspA  = readStat(comm, ncclStatGpuMemSuspend);
    EXPECT_EQ(totalA, persA + suspA);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    uint64_t totalS = readStat(comm, ncclStatGpuMemTotal);
    uint64_t persS  = readStat(comm, ncclStatGpuMemPersist);
    uint64_t suspS  = readStat(comm, ncclStatGpuMemSuspend);
    EXPECT_EQ(totalS, totalA);
    EXPECT_EQ(persS,  persA);
    EXPECT_EQ(suspS,  suspA);
    EXPECT_EQ(totalS, persS + suspS);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));

    uint64_t totalR = readStat(comm, ncclStatGpuMemTotal);
    EXPECT_EQ(totalR, totalA);
}

// ============================================================================
// 3. Collective integrity across a suspend/resume cycle
// ============================================================================

class SuspendResumeCollectiveIntegrity : public SuspendResumeMPITestBase {};

TEST_F(SuspendResumeCollectiveIntegrity, AllReduceAfterResume)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    // Sanity-check the comm before suspend.
    EXPECT_TRUE(runAllReduceAndVerify(comm, stream))
        << "AllReduce baseline failed before any Suspend";

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));

    EXPECT_TRUE(runAllReduceAndVerify(comm, stream))
        << "AllReduce produced wrong values after Suspend/Resume cycle";
}

TEST_F(SuspendResumeCollectiveIntegrity, AllReduceTwoCycles)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    EXPECT_TRUE(runAllReduceAndVerify(comm, stream)) << "baseline AllReduce";

    for (int i = 0; i < 2; ++i)
    {
        ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
        ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));

        EXPECT_TRUE(runAllReduceAndVerify(comm, stream))
            << "AllReduce wrong after cycle " << i;
    }
}

// ============================================================================
// 4. Lifecycle: destroy while suspended (force-resume-for-destroy path)
// ============================================================================

class SuspendResumeLifecycle : public SuspendResumeMPITestBase
{
protected:
    // We tear down our own comm in this test rather than relying on the
    // base class, so that we drive ncclCommDestroy from a suspended state.
    void TearDown() override
    {
        // No-op: each test explicitly destroys its own comm.
        // Skip MPITestBase::TearDown's cleanupTestCommunicator since the
        // test already destroyed it.
        test_comm_   = nullptr;
        test_stream_ = nullptr;
    }
};

TEST_F(SuspendResumeLifecycle, DestroyWhileSuspended)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    EXPECT_EQ(1u, readStat(comm, ncclStatGpuMemSuspended));

    // Tear the comm down without resuming first. The implementation must
    // restore mappings internally (force-resume-for-destroy) so the
    // destructor walk can free each pointer cleanly. A bug here would
    // SIGSEGV during ncclCuMemFree on an unmapped VA.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommDestroy(comm));
    test_comm_ = nullptr;
    if (stream != nullptr)
    {
        (void)hipStreamDestroy(stream);
        test_stream_ = nullptr;
    }
}

// ============================================================================
// 5. Argument validation / corner cases
//
// Each of these returns from inside ncclCommSuspend/Resume/MemStats BEFORE
// the bootstrapBarrier is reached, so individual ranks failing the call do
// not leave their peers blocked on a barrier.
// ============================================================================

class SuspendResumeArgValidation : public SuspendResumeMPITestBase {};

TEST_F(SuspendResumeArgValidation, ZeroFlags)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    // flags == 0 must be rejected on every rank (no barrier reached).
    ASSERT_MPI_EQ(ncclInvalidArgument, ncclCommSuspend(comm, 0));

    // Comm must remain in active state after the rejected call.
    EXPECT_EQ(0u, readStat(comm, ncclStatGpuMemSuspended))
        << "rejected Suspend should not transition the comm";

    // A subsequent valid call must still succeed.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));
}

TEST_F(SuspendResumeArgValidation, UnknownFlagsRejected)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    // Bogus high bits must be rejected (no barrier reached).
    ASSERT_MPI_EQ(ncclInvalidArgument, ncclCommSuspend(comm, 0xffff));
    ASSERT_MPI_EQ(ncclInvalidArgument,
                  ncclCommSuspend(comm, NCCL_SUSPEND_MEM | 0x80));

    // Subsequent valid call still works.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));
}

TEST_F(SuspendResumeArgValidation, DoubleSuspend)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    // First suspend goes through the barrier successfully.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));

    // Second suspend rejects on every rank before reaching the barrier
    // (state check sees comm->memManager->released==1).
    ASSERT_MPI_EQ(ncclInvalidUsage, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));

    // Resume restores active state.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));
    EXPECT_EQ(0u, readStat(comm, ncclStatGpuMemSuspended));
}

TEST_F(SuspendResumeArgValidation, ResumeWhenActive)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    // Resume on an already-active comm must reject before any barrier.
    ASSERT_MPI_EQ(ncclInvalidUsage, ncclCommResume(comm));

    EXPECT_EQ(0u, readStat(comm, ncclStatGpuMemSuspended))
        << "rejected Resume should not transition the comm";
}

TEST_F(SuspendResumeArgValidation, DoubleResume)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));

    // Second resume rejects -- comm is no longer suspended.
    ASSERT_MPI_EQ(ncclInvalidUsage, ncclCommResume(comm));

    EXPECT_EQ(0u, readStat(comm, ncclStatGpuMemSuspended));
}

// ----- ncclCommMemStats validation --------------------------------------

class SuspendResumeMemStatsValidation : public SuspendResumeMPITestBase {};

TEST_F(SuspendResumeMemStatsValidation, NullValuePtr)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    EXPECT_EQ(ncclInvalidArgument,
              ncclCommMemStats(comm, ncclStatGpuMemTotal, nullptr));
    EXPECT_EQ(ncclInvalidArgument,
              ncclCommMemStats(comm, ncclStatGpuMemSuspend, nullptr));
}

TEST_F(SuspendResumeMemStatsValidation, UnknownStat)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    uint64_t v = 0;
    EXPECT_EQ(ncclInvalidArgument,
              ncclCommMemStats(comm, static_cast<ncclCommMemStat_t>(999), &v));
    EXPECT_EQ(ncclInvalidArgument,
              ncclCommMemStats(comm, static_cast<ncclCommMemStat_t>(-1), &v));
}

#endif // MPI_TESTS_ENABLED
