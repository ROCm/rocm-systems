/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file SymmetricAbortCheckModeMPITests.cpp
 * @brief Tests for abort support in symmetric kernels and NCCL_CHECK_MODE
 *
 * Covers the two halves of the NCCL 2.29.7 release-note item
 * "Symmetric memory: Added support for abort in symmetric kernels. Added
 * NCCL_CHECK_MODE=DEBUG to validate symmetric buffer registration."
 *
 * SymCheckMode_* exercises the host-side validation in src/misc/argcheck.cc:
 *   - NCCL_CHECK_MODE=DEBUG_LOCAL  validates the buffer pointers locally.
 *   - NCCL_CHECK_MODE=DEBUG_GLOBAL additionally runs registrationCheck(), which
 *     bootstrap-allgathers the symmetric window identity of every rank and
 *     rejects the collective when the ranks disagree.
 * The validation is opt-in, so a matching Default_* test documents that the same
 * mismatch is accepted when the variable is unset.
 *
 * SymAbort_* exercises the device-side abort: the spin loops that the symmetric
 * kernels rely on must poll ncclDevComm::abortFlag via nccl::utility::testAbort()
 * so that ncclCommAbort() can unblock a kernel whose peer never arrives. The two
 * tests pin NCCL_SYM_KERNEL so that each of the two distinct spin loops is
 * covered:
 *   AllReduce_AGxLL_R     -> ncclLLA2ASession::recvUnrolled  (LL all-to-all)
 *   AllReduce_RSxLD_AGxST -> ncclLsaBarrierSession::wait     (LSA barrier)
 *
 * REQUIRED Environment Variables:
 *   NCCL_CUMEM_ENABLE=1          Enables cuMem API for symmetric support
 *   HSA_NO_SCRATCH_RECLAIM=1     Required for multi-GPU RCCL tests
 *
 * Symmetric memory additionally needs a Linux kernel >= 6.8 (cuMem/VMM gate in
 * src/misc/rocmwrap.cc); every test here skips when it is unavailable.
 *
 * Run examples:
 *   mpirun -np 8 --bind-to none -x NCCL_CUMEM_ENABLE=1 \
 *     ./rccl-UnitTestsMPI --gtest_filter=SymCheckMode_*
 *   mpirun -np 8 --bind-to none -x NCCL_CUMEM_ENABLE=1 \
 *     ./rccl-UnitTestsMPI --gtest_filter=SymAbort_*
 */

#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace {

constexpr int MIN_RANKS = 2;
// Small enough that the LL symmetric kernels stay eligible.
constexpr size_t ABORT_COUNT = 1024;
// Large enough that the window has room for the offset-mismatch scenario.
constexpr size_t CHECK_COUNT = 64 * 1024;

// ncclCommAbort() has to reach the device through host-pinned memory, so give
// the kernel a generous but bounded window to notice the flag.
constexpr int ABORT_DELAY_MS = 3000;
constexpr int ABORT_DEADLINE_MS = 30000;

// NCCL_CHECK_MODE and NCCL_SYM_KERNEL are read during ncclCommInitRank, so the
// tests set them before creating the communicator and restore them afterwards.
class ScopedEnv
{
public:
    ScopedEnv(const char* name, const char* value)
        : name_(name)
    {
        const char* prev = std::getenv(name);
        if (prev) {
            had_ = true;
            prev_ = prev;
        }
        if (value) {
            setenv(name, value, 1);
        } else {
            unsetenv(name);
        }
    }

    ~ScopedEnv()
    {
        if (had_) {
            setenv(name_.c_str(), prev_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    std::string name_;
    std::string prev_;
    bool had_ = false;
};

} // namespace

// ============================================================================
// Shared fixture: symmetric window allocation + availability probing
// ============================================================================

class SymmetricAbortCheckModeBase : public MPITestBase
{
protected:
    struct Alloc {
        void* ptr = nullptr;
    };

    struct Win {
        ncclWindow_t win = nullptr;
        ncclComm_t comm = nullptr;
    };

    std::vector<Alloc> allocs_;
    std::vector<Win> wins_;

    void TearDown() override
    {
        for (auto it = wins_.rbegin(); it != wins_.rend(); ++it) {
            if (it->win && it->comm) ncclCommWindowDeregister(it->comm, it->win);
        }
        wins_.clear();

        for (auto it = allocs_.rbegin(); it != allocs_.rend(); ++it) {
            if (it->ptr) ncclMemFree(it->ptr);
        }
        allocs_.clear();

        MPITestBase::TearDown();
    }

    void* allocSymBuf(size_t size)
    {
        void* ptr = nullptr;
        if (ncclMemAlloc(&ptr, size) != ncclSuccess || ptr == nullptr) return nullptr;
        allocs_.push_back({ptr});
        return ptr;
    }

    ncclWindow_t registerSymWindow(ncclComm_t comm, void* buf, size_t size)
    {
        ncclWindow_t win = nullptr;
        if (ncclCommWindowRegister(comm, buf, size, &win, NCCL_WIN_COLL_SYMMETRIC) != ncclSuccess) {
            return nullptr;
        }
        wins_.push_back({win, comm});
        return win;
    }

    // Mirrors SymmetricWindowMPITests: symmetric memory needs cuMem opted in and
    // ncclMemAlloc working. The communicator is created by the caller so that
    // NCCL_CHECK_MODE / NCCL_SYM_KERNEL can be installed first.
    bool symmetricPrerequisitesMet(int minRanks = MIN_RANKS)
    {
        const char* cuMemEnv = std::getenv("NCCL_CUMEM_ENABLE");
        if (!cuMemEnv || std::string(cuMemEnv) != "1") return false;
        if (!validateTestPrerequisites(minRanks)) return false;

        void* probe = nullptr;
        if (ncclMemAlloc(&probe, 4096) != ncclSuccess || probe == nullptr) return false;
        ncclMemFree(probe);
        return true;
    }

    // A rank that reports "reject" while its peers report "accept" would deadlock
    // the next collective, so every assertion about the outcome is agreed on by
    // all ranks first.
    static bool allRanksAgree(bool local)
    {
        int in = local ? 1 : 0;
        int out = 0;
        MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return out != 0;
    }
};

// ============================================================================
// NCCL_CHECK_MODE: symmetric buffer registration validation
// ============================================================================

class SymCheckMode_Registration : public SymmetricAbortCheckModeBase
{
protected:
    enum class Deviation {
        None,          // every rank registers both buffers at offset 0
        Unregistered,  // the odd rank leaves its send buffer outside any window
        ShiftedOffset, // the odd rank starts further into its send window
    };

    // Runs one AllReduce under the current NCCL_CHECK_MODE and returns the
    // result that every rank observed.
    ncclResult_t runAllReduce(Deviation deviation)
    {
        const size_t count = CHECK_COUNT;
        // Four times the payload leaves room to shift the user offset.
        const size_t winBytes = count * sizeof(float) * 4;

        ncclComm_t comm = getActiveCommunicator();
        hipStream_t stream = getActiveStream();
        int rank = 0, nRanks = 0;
        ncclCommUserRank(comm, &rank);
        ncclCommCount(comm, &nRanks);

        // Rank 1 is the deviating rank: registrationCheck() compares every rank
        // against rank 0, so the deviation must not live on rank 0.
        const int oddRank = 1;
        const bool isOdd = (rank == oddRank);

        void* sendBuf = allocSymBuf(winBytes);
        void* recvBuf = allocSymBuf(winBytes);
        if (!sendBuf || !recvBuf) return ncclInternalError;

        // ncclCommWindowRegister is collective, so every rank must register the
        // same number of windows: a rank that skipped one would leave its peers
        // waiting in the next registration while it has moved on to the AllReduce.
        // Register both buffers everywhere and express the deviation through the
        // pointer handed to the collective instead.
        if (registerSymWindow(comm, sendBuf, winBytes) == nullptr) return ncclInternalError;
        if (registerSymWindow(comm, recvBuf, winBytes) == nullptr) return ncclInternalError;

        float* sendPtr = static_cast<float*>(sendBuf);
        float* recvPtr = static_cast<float*>(recvBuf);

        if (isOdd && deviation == Deviation::Unregistered) {
            // Hand the collective a symmetric buffer that no window covers: every
            // peer still passes a registered buffer, so registrationCheck()
            // disagrees with rank 0 without unbalancing the registration itself.
            void* unregBuf = allocSymBuf(winBytes);
            if (!unregBuf) return ncclInternalError;
            sendPtr = static_cast<float*>(unregBuf);
        } else if (isOdd && deviation == Deviation::ShiftedOffset) {
            // Registered, but starting further into the window than its peers,
            // which the symmetric kernels cannot express.
            sendPtr = reinterpret_cast<float*>(static_cast<char*>(sendBuf) + 8192);
        }

        ncclResult_t res = ncclAllReduce(sendPtr, recvPtr, count, ncclFloat, ncclSum, comm, stream);
        if (res == ncclSuccess && hipStreamSynchronize(stream) != hipSuccess) {
            res = ncclInternalError;
        }

        // The ranks must agree on the outcome; a split result (some ranks reject,
        // others accept) is itself a failure rather than something to hide behind
        // MPI_MAX, which would let a mixed outcome masquerade as the expected one.
        int localRes = static_cast<int>(res);
        int maxRes = 0;
        int minRes = 0;
        MPI_Allreduce(&localRes, &maxRes, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&localRes, &minRes, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (minRes != maxRes) return ncclInternalError;
        return static_cast<ncclResult_t>(maxRes);
    }
};

TEST_F(SymCheckMode_Registration, DebugGlobal_MatchingRegistration_Succeeds)
{
    ScopedEnv checkMode("NCCL_CHECK_MODE", "DEBUG_GLOBAL");
    if (!symmetricPrerequisitesMet()) {
        GTEST_SKIP() << "Requires symmetric support (NCCL_CUMEM_ENABLE=1, kernel >= 6.8) with 2+ ranks";
    }
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    // The global check must be transparent when every rank agrees.
    ASSERT_MPI_EQ(ncclSuccess, runAllReduce(Deviation::None));
}

TEST_F(SymCheckMode_Registration, DebugGlobal_RegistrationMismatch_Rejected)
{
    ScopedEnv checkMode("NCCL_CHECK_MODE", "DEBUG_GLOBAL");
    if (!symmetricPrerequisitesMet()) {
        GTEST_SKIP() << "Requires symmetric support (NCCL_CUMEM_ENABLE=1, kernel >= 6.8) with 2+ ranks";
    }
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    // One rank passes a buffer that is not in any symmetric window while the
    // others pass registered buffers: isSymRegistered disagrees with rank 0.
    ASSERT_MPI_EQ(ncclInvalidArgument, runAllReduce(Deviation::Unregistered));
}

TEST_F(SymCheckMode_Registration, DebugGlobal_UserOffsetMismatch_Rejected)
{
    ScopedEnv checkMode("NCCL_CHECK_MODE", "DEBUG_GLOBAL");
    if (!symmetricPrerequisitesMet()) {
        GTEST_SKIP() << "Requires symmetric support (NCCL_CUMEM_ENABLE=1, kernel >= 6.8) with 2+ ranks";
    }
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    // Every rank registers a window, but one rank's buffer starts at a different
    // offset inside it, which the symmetric kernels cannot express.
    ASSERT_MPI_EQ(ncclInvalidArgument, runAllReduce(Deviation::ShiftedOffset));
}

TEST_F(SymCheckMode_Registration, Default_RegistrationMismatch_Accepted)
{
    ScopedEnv checkMode("NCCL_CHECK_MODE", nullptr);
    if (!symmetricPrerequisitesMet()) {
        GTEST_SKIP() << "Requires symmetric support (NCCL_CUMEM_ENABLE=1, kernel >= 6.8) with 2+ ranks";
    }
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    // Same mismatch as above: without the variable the library must stay on the
    // fast path and fall back instead of diagnosing.
    ASSERT_MPI_EQ(ncclSuccess, runAllReduce(Deviation::Unregistered));
}

// ============================================================================
// NCCL_CHECK_MODE=DEBUG_LOCAL: local pointer validation
// ============================================================================

// These do not need symmetric memory, so they guard the NCCL_CHECK_MODE
// plumbing on any platform: a mode that reaches comm->checkMode must make the
// library reject a host pointer where a device pointer is required.
class SymCheckMode_Local : public SymmetricAbortCheckModeBase
{
protected:
    void expectHostPointerRejected(ncclResult_t expected)
    {
        ncclComm_t comm = getActiveCommunicator();
        hipStream_t stream = getActiveStream();

        std::vector<float> hostBuf(CHECK_COUNT, 1.0f);
        void* devBuf = nullptr;
        ASSERT_EQ(hipSuccess, hipMalloc(&devBuf, CHECK_COUNT * sizeof(float)));

        ncclResult_t res =
            ncclAllReduce(hostBuf.data(), devBuf, CHECK_COUNT, ncclFloat, ncclSum, comm, stream);

        // Agree on the result across ranks before touching the stream: a rank that
        // wrongly accepted the host pointer would otherwise launch a kernel on
        // unmapped memory and block in hipStreamSynchronize while its peers fail.
        // A correct rejection enqueues nothing, so there is nothing to drain.
        ASSERT_MPI_EQ(expected, res);
        ASSERT_EQ(hipSuccess, hipFree(devBuf));
    }
};

TEST_F(SymCheckMode_Local, DebugLocal_HostPointer_Rejected)
{
    ScopedEnv checkMode("NCCL_CHECK_MODE", "DEBUG_LOCAL");
    if (!validateTestPrerequisites(1)) {
        GTEST_SKIP() << "Needs at least 1 process";
    }
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    expectHostPointerRejected(ncclInvalidArgument);
}

TEST_F(SymCheckMode_Local, DebugGlobal_HostPointer_Rejected)
{
    ScopedEnv checkMode("NCCL_CHECK_MODE", "DEBUG_GLOBAL");
    if (!validateTestPrerequisites(1)) {
        GTEST_SKIP() << "Needs at least 1 process";
    }
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    // DEBUG_GLOBAL is a superset of DEBUG_LOCAL, so the local check must still
    // fire. This is the cheapest signal that the mode reached the communicator
    // at all, which regressed once when commAlloc() reset comm->checkMode after
    // the environment had already been parsed.
    expectHostPointerRejected(ncclInvalidArgument);
}

// The deprecated NCCL_CHECK_POINTERS is intentionally not tested here. It is an
// NCCL_PARAM, latched on first read (src/include/param.h) and evaluated on every
// ncclCommInitRank (src/init.cc), so the two tests above already pin it at 0 for
// the lifetime of the process; a later setenv() in this same binary has no
// effect. NCCL_CHECK_MODE works because it is re-read through ncclGetEnv() on
// each communicator, and DEBUG_LOCAL already covers the same rejection path.

// The opt-in nature of the checking is covered by
// SymCheckMode_Registration.Default_RegistrationMismatch_Accepted, which uses a
// mismatch that is safe to actually execute. Enqueuing a host pointer without a
// check mode would let the kernel dereference unmapped memory and take the whole
// process down with it, so that combination is deliberately not tested.

// ============================================================================
// Abort inside symmetric kernels
// ============================================================================

class SymAbort_Kernel : public SymmetricAbortCheckModeBase
{
protected:
    // All ranks but the last enqueue a symmetric AllReduce, so the kernel parks
    // in its device-side spin loop waiting for a peer that never arrives. A
    // watchdog thread then calls ncclCommAbort(); the kernel must observe the
    // abort flag and return, which lets the stream drain.
    //
    // The communicator is created and cleaned up here instead of through the
    // fixture: ncclCommAbort() is only defined for non-blocking communicators
    // (docs/how-to/fault-tolerance.rst), and an aborted communicator frees itself
    // and reclaims its still-registered windows, so nothing here may be handed to
    // the fixture's window/allocation teardown.
    void runAbortScenario(const char* symKernel)
    {
        ScopedEnv kernelEnv("NCCL_SYM_KERNEL", symKernel);
        if (!symmetricPrerequisitesMet()) {
            GTEST_SKIP() << "Requires symmetric support (NCCL_CUMEM_ENABLE=1, kernel >= 6.8) with 2+ ranks";
        }

        int rank = 0, nRanks = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

        ncclUniqueId id;
        if (rank == 0) ASSERT_MPI_EQ(ncclSuccess, ncclGetUniqueId(&id));
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Non-blocking so the watchdog can abort while the main thread is only
        // polling the stream, never parked inside a blocking RCCL call.
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.blocking     = 0;
        ncclComm_t comm     = nullptr;
        ncclResult_t initRes = ncclCommInitRankConfig(&comm, nRanks, id, rank, &config);
        ASSERT_MPI_NE(comm, nullptr);
        ASSERT_MPI_EQ(ncclSuccess, waitComm(comm, initRes));

        hipStream_t stream = nullptr;
        ASSERT_EQ(hipSuccess, hipStreamCreate(&stream));

        const size_t bytes = ABORT_COUNT * sizeof(float);
        void* sendBuf = nullptr;
        void* recvBuf = nullptr;
        ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&sendBuf, bytes));
        ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&recvBuf, bytes));

        // Register the windows directly (not via the fixture): the comm is aborted
        // below, which reclaims these windows, so a fixture deregister would run
        // through freed memory.
        ncclWindow_t sendWin = nullptr;
        ncclWindow_t recvWin = nullptr;
        ncclResult_t regRes =
            waitComm(comm, ncclCommWindowRegister(comm, sendBuf, bytes, &sendWin, NCCL_WIN_COLL_SYMMETRIC));
        if (regRes == ncclSuccess) {
            regRes =
                waitComm(comm, ncclCommWindowRegister(comm, recvBuf, bytes, &recvWin, NCCL_WIN_COLL_SYMMETRIC));
        }
        if (!allRanksAgree(regRes == ncclSuccess && sendWin != nullptr && recvWin != nullptr)) {
            static_cast<void>(ncclCommAbort(comm));
            static_cast<void>(hipStreamDestroy(stream));
            static_cast<void>(ncclMemFree(sendBuf));
            static_cast<void>(ncclMemFree(recvBuf));
            GTEST_SKIP() << "Symmetric window registration unavailable";
        }

        // Warm up with every rank participating so the symmetric path (device
        // comm, teams, LL buffers) is fully established before the stall.
        ncclResult_t warmup =
            waitComm(comm, ncclAllReduce(sendBuf, recvBuf, ABORT_COUNT, ncclFloat, ncclSum, comm, stream));
        ASSERT_MPI_EQ(ncclSuccess, warmup);
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        std::atomic<bool> aborted{false};
        std::thread watchdog([&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(ABORT_DELAY_MS));
            aborted.store(true);
            ncclCommAbort(comm);
        });

        const int deserter = nRanks - 1;
        bool drained = true;

        if (rank != deserter) {
            // The symmetric kernel launches directly on the stream, so poll the
            // stream rather than the async error. A rejected enqueue is
            // acceptable: it means the abort landed before the launch; only a
            // launched kernel has to honour the flag.
            ncclResult_t enq =
                ncclAllReduce(sendBuf, recvBuf, ABORT_COUNT, ncclFloat, ncclSum, comm, stream);
            if (enq == ncclSuccess || enq == ncclInProgress) {
                drained = waitForStream(stream, symKernel);
            }
        }

        watchdog.join();
        EXPECT_TRUE(aborted.load());
        EXPECT_TRUE(drained) << "Symmetric kernel " << symKernel
                             << " did not return after ncclCommAbort";

        // The comm has been aborted (and with it its windows), so only the local
        // stream and device buffers are released here.
        static_cast<void>(hipStreamDestroy(stream));
        static_cast<void>(ncclMemFree(sendBuf));
        static_cast<void>(ncclMemFree(recvBuf));
        MPI_Barrier(MPI_COMM_WORLD);
    }

private:
    // Drives a non-blocking communicator to a settled state and returns the final
    // async error. The triggering call's ncclInProgress is folded in so call
    // sites can wrap the RCCL call directly.
    static ncclResult_t waitComm(ncclComm_t comm, ncclResult_t res = ncclInProgress)
    {
        if (res != ncclInProgress) return res;
        ncclResult_t state = ncclInProgress;
        do {
            if (ncclCommGetAsyncError(comm, &state) != ncclSuccess) return ncclInternalError;
        } while (state == ncclInProgress);
        return state;
    }

    // Polls instead of blocking so that a kernel ignoring the abort flag is
    // reported as a failure rather than hanging the whole suite.
    static bool waitForStream(hipStream_t stream, const char* symKernel)
    {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(ABORT_DEADLINE_MS);
        while (std::chrono::steady_clock::now() < deadline) {
            hipError_t query = hipStreamQuery(stream);
            if (query != hipErrorNotReady) {
                // Either success or a launch failure: the kernel is no longer
                // resident, which is what the abort is supposed to achieve.
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        TEST_INFO("Symmetric kernel %s still resident %d ms after ncclCommAbort", symKernel,
            ABORT_DEADLINE_MS);
        return false;
    }
};

TEST_F(SymAbort_Kernel, LLAllToAll_AbortReleasesKernel)
{
    // Exercises ncclLLA2ASession::recvUnrolled().
    runAbortScenario("AllReduce_AGxLL_R");
}

TEST_F(SymAbort_Kernel, LsaBarrier_AbortReleasesKernel)
{
    // Exercises ncclLsaBarrierSession::wait().
    runAbortScenario("AllReduce_RSxLD_AGxST");
}

#endif // MPI_TESTS_ENABLED
