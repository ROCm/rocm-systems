/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for the RCCL_DDA_NRANKS_RELAX low-rank DDA IPC AllReduce gate.
//
// These exercise ncclAllReduceDdaIpcEligible() and ncclDdaNranksRelaxEnabled()
// with the mock ncclComm (no GPUs required).
//
// RCCL_PARAM values are cached per-process, so the fixture tests below cover the
// default (relax-off) semantics that the eligibility change must preserve:
// exactly kDdaNranks stays eligible, and 2..7 rank comms are rejected unless the
// operator explicitly opts in.
//
// The relax-enabled path is covered by DdaNranksRelaxIsolatedTest, which re-execs
// this binary with RCCL_DDA_NRANKS_RELAX=1 pre-set (the value must be in the
// environment before any param read) and asserts every count in
// [2, kDdaNranks] becomes eligible while counts outside that range do not.
// No test_runner config currently sets RCCL_DDA_NRANKS_RELAX as an environment
// variable, so end-to-end engagement via the rccl-tests AllReduce sweep is not
// yet exercised in CI; the isolated-process test above is what actually covers
// the relaxed dispatch today.

#include "common/DdaIpcTestHelpers.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

#include <rccl/rccl.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include "algorithms/dda/all_reduce/dda_all_reduce.h"
#include "algorithms/dda/dda_init_detail.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting
{

class DdaNranksRelaxTest : public ::testing::Test
{
protected:
    DdaIpcMockComm mockComm_;
    void*          sendbuff_{reinterpret_cast<void*>(0x10)};
    void*          recvbuff_{reinterpret_cast<void*>(0x20)};
    static constexpr size_t kCount{1024};  // 4 KiB fp32: flat path, 16B aligned
};

// With RCCL_DDA_NRANKS_RELAX unset the knob defaults to disabled.
TEST_F(DdaNranksRelaxTest, RelaxDisabledByDefault)
{
    EXPECT_FALSE(ncclDdaNranksRelaxEnabled());
}

// Default behaviour is preserved: a full kDdaNranks (8) clique stays eligible.
TEST_F(DdaNranksRelaxTest, FullCliqueEligibleByDefault)
{
    mockComm_.comm.nRanks = nccl_dda_detail::kDdaNranks;
    EXPECT_TRUE(ncclAllReduceDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32, ncclSum));
}

// Without the relax knob, 4-rank comms are NOT eligible for the DDA IPC path.
TEST_F(DdaNranksRelaxTest, FourRanksRejectedWhenRelaxOff)
{
    mockComm_.comm.nRanks = 4;
    EXPECT_FALSE(ncclAllReduceDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32, ncclSum));
}

// Likewise for 2-rank comms.
TEST_F(DdaNranksRelaxTest, TwoRanksRejectedWhenRelaxOff)
{
    mockComm_.comm.nRanks = 2;
    EXPECT_FALSE(ncclAllReduceDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32, ncclSum));
}

// With relax off (default), any count other than the full kDdaNranks clique is
// rejected -- e.g. a 3-rank comm.
TEST_F(DdaNranksRelaxTest, ThreeRanksRejectedWhenRelaxOff)
{
    mockComm_.comm.nRanks = 3;
    EXPECT_FALSE(ncclAllReduceDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32, ncclSum));
}

// Standard eligibility guards remain intact at full clique size.
TEST_F(DdaNranksRelaxTest, NullCommRejected)
{
    EXPECT_FALSE(ncclAllReduceDdaIpcEligible(
        nullptr, sendbuff_, recvbuff_, kCount, ncclFloat32, ncclSum));
}

TEST_F(DdaNranksRelaxTest, NonSumOpRejected)
{
    mockComm_.comm.nRanks = nccl_dda_detail::kDdaNranks;
    EXPECT_FALSE(ncclAllReduceDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32, ncclMax));
}

TEST_F(DdaNranksRelaxTest, MissingIpcResourcesRejected)
{
    mockComm_.comm.nRanks = nccl_dda_detail::kDdaNranks;
    mockComm_.setIpcResourcesPresent(false);
    EXPECT_FALSE(ncclAllReduceDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32, ncclSum));
}

// The dispatch gate agrees with eligibility when relax is off: the full clique
// reaches the launch path (and fails only the scratch-size check), while a
// low-rank comm is refused by ncclDdaIpcNranksSupported() before it. Distinct
// return codes keep the two apart. No GPU: both return before any kernel launch.
TEST_F(DdaNranksRelaxTest, DispatchRejectsLowRankWhenRelaxOff)
{
    mockComm_.comm.ddaScratchBytes = 0;

    mockComm_.comm.nRanks = nccl_dda_detail::kDdaNranks;
    EXPECT_EQ(ncclAllReduceDdaIpc(sendbuff_, recvbuff_, kCount, ncclFloat32, ncclSum,
                                  mockComm_.get(), nullptr),
              ncclInvalidArgument);

    for (int nRanks : {2, 3, 4, 5, 6, 7})
    {
        mockComm_.comm.nRanks = nRanks;
        EXPECT_EQ(ncclAllReduceDdaIpc(sendbuff_, recvbuff_, kCount, ncclFloat32, ncclSum,
                                      mockComm_.get(), nullptr),
                  ncclInvalidUsage)
            << "nRanks=" << nRanks << " must be refused with relax off";
    }
}

// Relaxed path (RCCL_DDA_NRANKS_RELAX=1). RCCL_PARAM caches per-process and
// NCCL_NO_CACHE is parsed once, so the relaxed value has to be set before any
// param read: run in a fresh re-exec'd process with the env pre-set. This proves
// the eligibility gate opens for every single-node count in [2, kDdaNranks] (and
// still rejects counts outside that range) when the operator opts in. No
// test_runner config sets RCCL_DDA_NRANKS_RELAX as an environment variable yet,
// so this in-process test is what actually covers the relaxed dispatch today.
TEST(DdaNranksRelaxIsolatedTest, RelaxedPathAdmitsTwoThroughEightRanks)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "RelaxedPathAdmitsTwoThroughEightRanks",
        []()
        {
            void*                  sendbuff = reinterpret_cast<void*>(0x10);
            void*                  recvbuff = reinterpret_cast<void*>(0x20);
            constexpr size_t       count    = 1024;  // 4 KiB fp32: flat path, 16B aligned
            DdaIpcMockComm         mockComm;

            EXPECT_TRUE(ncclDdaNranksRelaxEnabled());

            // With relax on, every single-node count in [2, kDdaNranks] is eligible
            // (kDdaNranks uses the specialised kernel, the rest the NRANKS == 0 one).
            for (int nRanks = 2; nRanks <= nccl_dda_detail::kDdaNranks; ++nRanks)
            {
                mockComm.comm.nRanks = nRanks;
                EXPECT_TRUE(ncclAllReduceDdaIpcEligible(
                    mockComm.get(), sendbuff, recvbuff, count, ncclFloat32, ncclSum))
                    << "nRanks=" << nRanks << " should be eligible with relax on";
            }

            // AllGather, ReduceScatter and AllToAll are also relaxed by this
            // branch (DdaNranksRelaxCollectives_test.cpp verifies that
            // directly for all four counts); the 8-rank-only invariant that
            // held for them on the AllReduce-only PR does not apply here.

            // Counts outside [2, kDdaNranks] stay ineligible.
            for (int nRanks : {1, 9, 16})
            {
                mockComm.comm.nRanks = nRanks;
                EXPECT_FALSE(ncclAllReduceDdaIpcEligible(
                    mockComm.get(), sendbuff, recvbuff, count, ncclFloat32, ncclSum))
                    << "nRanks=" << nRanks << " must not be eligible";
            }

            // Eligibility alone does not prove the dispatch routes the count: with
            // only the checks above, deleting a supported count from
            // ncclAllReduceDdaIpcTyped() would still pass. Drive the real entry
            // point and separate the two failure modes by rank count.
            //
            // ddaScratchBytes = 0 makes every supported count fail the scratch-size
            // check inside ncclAllReduceDdaIpcLaunch() and return
            // ncclInvalidArgument, which is reached only after the dispatch has
            // accepted the count -- and before any barrier deref or kernel launch,
            // so this needs no GPU. An unsupported count is rejected earlier by the
            // ncclDdaIpcNranksSupported() gate and returns ncclInvalidUsage.
            mockComm.comm.ddaScratchBytes = 0;

            for (int nRanks = 2; nRanks <= nccl_dda_detail::kDdaNranks; ++nRanks)
            {
                mockComm.comm.nRanks = nRanks;
                EXPECT_EQ(ncclAllReduceDdaIpc(sendbuff, recvbuff, count, ncclFloat32,
                                              ncclSum, mockComm.get(), nullptr),
                          ncclInvalidArgument)
                    << "nRanks=" << nRanks
                    << " should reach the launch path and fail the scratch check";
            }

            for (int nRanks : {1, 9, 16})
            {
                mockComm.comm.nRanks = nRanks;
                EXPECT_EQ(ncclAllReduceDdaIpc(sendbuff, recvbuff, count, ncclFloat32,
                                              ncclSum, mockComm.get(), nullptr),
                          ncclInvalidUsage)
                    << "nRanks=" << nRanks << " must be refused by the dispatch gate";
            }
        },
        {{"RCCL_DDA_NRANKS_RELAX", "1"}});
}

// The isolated test above only ever uses count=1024 (4 KiB), always below
// kDdaFlatTreeThresholdBytes (256 KiB), so the tree branch's
// `count % comm->nRanks` divisibility check is never exercised at a relaxed
// (non-8) rank count -- that branch only runs above the threshold, and 8 was
// the only rank count reachable there before this PR. Pin it directly at
// nRanks=3: one count that divides evenly (eligible) and one immediately
// above it that does not (ineligible), both otherwise identical.
TEST(DdaNranksRelaxIsolatedTest, TreeThresholdDivisibilityAtRelaxedRankCount)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "TreeThresholdDivisibilityAtRelaxedRankCount",
        []()
        {
            void*          sendbuff = reinterpret_cast<void*>(0x10);
            void*          recvbuff = reinterpret_cast<void*>(0x20);
            DdaIpcMockComm mockComm;
            mockComm.comm.nRanks = 3;

            // 196608 floats = 786432 B (> 256 KiB, tree path); 196608 % 3 == 0
            // and the per-rank slice (65536 floats = 262144 B) is 16-byte aligned.
            EXPECT_TRUE(ncclAllReduceDdaIpcEligible(
                mockComm.get(), sendbuff, recvbuff, 196608, ncclFloat32, ncclSum))
                << "196608 % 3 == 0 should be eligible for the tree path";

            // One count higher: same size class, but 196612 % 3 == 1.
            EXPECT_FALSE(ncclAllReduceDdaIpcEligible(
                mockComm.get(), sendbuff, recvbuff, 196612, ncclFloat32, ncclSum))
                << "196612 % 3 == 1 must not be eligible for the tree path";

            // At nRanks=2, 196612 floats = 786448 B: passes the total-bytes 16-byte
            // check (786448 % 16 == 0) and is divisible by nRanks (196612 % 2 == 0),
            // but the per-rank slice (98306 floats = 393224 B) is not itself
            // 16-byte aligned: 393224 % 16 == 8. Total-bytes alignment and
            // divisibility are each individually satisfied here; only the
            // per-rank-slice check (the one specific to the tree path) catches
            // this, so this case actually exercises it -- unlike a count that
            // also fails the total-bytes check, which would be rejected earlier
            // regardless of the tree-specific logic.
            mockComm.comm.nRanks = 2;
            EXPECT_FALSE(ncclAllReduceDdaIpcEligible(
                mockComm.get(), sendbuff, recvbuff, 196612, ncclFloat32, ncclSum))
                << "196612 % 2 == 0 but the per-rank slice is not 16-byte aligned";
        },
        {{"RCCL_DDA_NRANKS_RELAX", "1"}});
}

// ncclAllReduceDdaIpcTreeEligible() directly: the pure predicate both
// ncclAllReduceDdaIpcEligible() and the IPC launch path call, so a mutation
// or drift between them shows up here as well as at the call sites above.
// No GPU, no comm, no relaxed-rank env var needed.
TEST(DdaAllReduceIpcTreeEligibleTest, RejectsNonPositiveRanks)
{
    EXPECT_FALSE(ncclAllReduceDdaIpcTreeEligible(/*count=*/1024, /*nRanks=*/0, /*typeSize=*/4));
    EXPECT_FALSE(ncclAllReduceDdaIpcTreeEligible(/*count=*/1024, /*nRanks=*/-1, /*typeSize=*/4));
}

TEST(DdaAllReduceIpcTreeEligibleTest, RejectsNonDivisibleCount)
{
    // 196612 % 3 == 1.
    EXPECT_FALSE(ncclAllReduceDdaIpcTreeEligible(/*count=*/196612, /*nRanks=*/3, /*typeSize=*/4));
}

TEST(DdaAllReduceIpcTreeEligibleTest, RejectsDivisibleButUnalignedSlice)
{
    // 196611 % 3 == 0, but (196611 / 3) * 4 == 262148, and 262148 % 16 == 4.
    EXPECT_FALSE(ncclAllReduceDdaIpcTreeEligible(/*count=*/196611, /*nRanks=*/3, /*typeSize=*/4));
}

TEST(DdaAllReduceIpcTreeEligibleTest, AcceptsDivisibleAndAlignedSlice)
{
    // 196608 % 3 == 0, and (196608 / 3) * 4 == 262144, which is 16-byte aligned.
    EXPECT_TRUE(ncclAllReduceDdaIpcTreeEligible(/*count=*/196608, /*nRanks=*/3, /*typeSize=*/4));
}

// ---------------------------------------------------------------------------
// Real-GPU comm-init engagement
// ---------------------------------------------------------------------------
//
// Everything above uses DdaIpcMockComm, which fabricates comm->ddaIpcMemHandler
// and comm->ddaScratch directly. None of it reaches the gate in init.cc:
//
//     } else if (comm->nNodes == 1 && ncclDdaIpcNranksSupported(comm->nRanks)) {
//       NCCLCHECKGOTO(ncclDdaIpcCommInit(comm), res, fail);
//
// Reverting that condition to the pre-feature `comm->nRanks == kDdaNranks`
// would leave the low-rank path dead at runtime with every test above still
// passing. The pair below closes that: a real 4-rank ncclCommInitRank, then an
// assertion on whether ncclDdaIpcCommInit() actually stamped the communicator.
//
// The negative control (relax off -> no resources) is what makes the positive
// case meaningful: without it, "the gate opened for 4 ranks" and "comm init
// always allocates these" are indistinguishable.
//
// --- Why fork() one process per rank, and not ncclCommInitAll ---
//
// comm->directMode is set whenever a single process drives more than one local
// rank, and ncclDdaIpcCommInit() bails out on it before allocating anything.
// ncclCommInitAll runs every rank in one process, so directMode is always true
// and the DDA IPC pointers stay null no matter what the gate decides -- the
// assertion would hold identically on a reverted build, proving nothing. One
// process per rank keeps directMode false and the gate observable. This mirrors
// P2pThenCollective_SameBuffer in RegisterTests.cpp, which forks for the same
// reason. The HIP runtime is not fork-safe, so the parent makes no HIP or RCCL
// call: rank 0's child performs the preconditions and publishes the unique ID.

namespace
{

enum class DdaInitState : int
{
    NotReady = 0,
    Ready    = 1,
    Skip     = -1,
};

struct DdaInitShared
{
    ncclUniqueId              id;
    std::atomic<DdaInitState> state;
};

enum DdaInitChildExit
{
    kDdaInitChildOk   = 0,
    kDdaInitChildFail = 1,
    kDdaInitChildSkip = 2,
};

// GTest assertions do not propagate across fork(); children report via exit code.
#define DDA_INIT_CHILD_HC(x)                                                              \
    do                                                                                    \
    {                                                                                     \
        hipError_t _e = (x);                                                              \
        if (_e != hipSuccess)                                                             \
        {                                                                                 \
            printf("[rank %d] HIP error %d (%s) @ %s:%d\n", rank, _e,                     \
                   hipGetErrorString(_e), __FILE__, __LINE__);                            \
            return kDdaInitChildFail;                                                     \
        }                                                                                 \
    } while (0)

#define DDA_INIT_CHILD_NC(x)                                                              \
    do                                                                                    \
    {                                                                                     \
        ncclResult_t _e = (x);                                                            \
        if (_e != ncclSuccess)                                                             \
        {                                                                                 \
            printf("[rank %d] RCCL error %d (%s) @ %s:%d\n", rank, _e,                    \
                   ncclGetErrorString(_e), __FILE__, __LINE__);                           \
            return kDdaInitChildFail;                                                     \
        }                                                                                 \
    } while (0)

// ncclDdaIpcCommInit() only runs on the architectures the DDA IPC kernels
// support; elsewhere it bails before allocating and the test cannot observe the
// gate either way.
bool ddaInitArchSupported(const char* gcnArchName)
{
    return strncmp(gcnArchName, "gfx942", 6) == 0 || strncmp(gcnArchName, "gfx950", 6) == 0;
}

bool allDevicesSupportDdaInit(int numDevices, char* unsupportedArchOut, size_t outLen)
{
    for (int dev = 0; dev < numDevices; dev++)
    {
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess)
        {
            return false;
        }
        if (!ddaInitArchSupported(prop.gcnArchName))
        {
            if (unsupportedArchOut && outLen > 0)
            {
                strncpy(unsupportedArchOut, prop.gcnArchName, outLen - 1);
                unsupportedArchOut[outLen - 1] = '\0';
            }
            return false;
        }
    }
    return true;
}

// Runs entirely inside a forked child: the first HIP/RCCL call happens here.
int ddaInitRunRank(int rank, int nranks, DdaInitShared* shared, bool expectResources)
{
    if (rank == 0)
    {
        int numDevices = 0;
        DDA_INIT_CHILD_HC(hipGetDeviceCount(&numDevices));
        if (numDevices < nranks)
        {
            printf("Requires %d GPUs (detected %d).\n", nranks, numDevices);
            shared->state.store(DdaInitState::Skip, std::memory_order_release);
            return kDdaInitChildSkip;
        }

        char unsupportedArch[256] = {0};
        if (!allDevicesSupportDdaInit(nranks, unsupportedArch, sizeof(unsupportedArch)))
        {
            printf("Unsupported GPU architecture '%s' for DDA IPC (requires gfx942 or gfx950).\n",
                   unsupportedArch);
            shared->state.store(DdaInitState::Skip, std::memory_order_release);
            return kDdaInitChildSkip;
        }

        // ncclDdaIpcCommInit() requires direct GPU-to-GPU P2P across the whole
        // clique; without it the resources stay null regardless of the gate.
        for (int i = 0; i < nranks; i++)
        {
            for (int j = i + 1; j < nranks; j++)
            {
                int canAccess = 0;
                DDA_INIT_CHILD_HC(hipDeviceCanAccessPeer(&canAccess, i, j));
                if (!canAccess)
                {
                    printf("No direct P2P between GPU %d and %d; DDA IPC would not engage.\n", i, j);
                    shared->state.store(DdaInitState::Skip, std::memory_order_release);
                    return kDdaInitChildSkip;
                }
            }
        }

        DDA_INIT_CHILD_NC(ncclGetUniqueId(&shared->id));
        shared->state.store(DdaInitState::Ready, std::memory_order_release);
    }
    else
    {
        DdaInitState st;
        while ((st = shared->state.load(std::memory_order_acquire)) == DdaInitState::NotReady)
        {
            /* spin until rank 0 publishes the id or reports a skip */
        }
        if (st == DdaInitState::Skip)
        {
            return kDdaInitChildSkip;
        }
    }

    ncclUniqueId id = shared->id;

    DDA_INIT_CHILD_HC(hipSetDevice(rank));

    ncclComm_t commHandle{};
    DDA_INIT_CHILD_NC(ncclCommInitRank(&commHandle, nranks, id, rank));

    // Cast to the internal struct to observe what comm init actually did.
    ncclComm*  c       = commHandle;
    const bool haveDda = (c->ddaIpcMemHandler != nullptr) && (c->ddaScratch != nullptr) &&
                         (c->ddaPeerPtrsDev != nullptr);

    int rc = kDdaInitChildOk;
    if (haveDda != expectResources)
    {
        printf("[rank %d] nranks=%d relax=%d: expected DDA IPC resources %s, but "
               "ddaIpcMemHandler=%p ddaScratch=%p ddaPeerPtrsDev=%p\n",
               rank, nranks, expectResources ? 1 : 0, expectResources ? "present" : "absent",
               c->ddaIpcMemHandler, c->ddaScratch, c->ddaPeerPtrsDev);
        rc = kDdaInitChildFail;
    }

    DDA_INIT_CHILD_NC(ncclCommDestroy(commHandle));
    return rc;
}

// Forks nranks children, each initialising one rank of a real communicator, and
// checks whether comm init engaged the DDA IPC path.
void runDdaIpcCommInitEngagementTest(int nranks, bool expectResources)
{
    // Shared region set up before any fork and before any HIP/RCCL call.
    DdaInitShared* shared = static_cast<DdaInitShared*>(
        mmap(nullptr, sizeof(DdaInitShared), PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    ASSERT_NE(shared, MAP_FAILED) << "mmap for shared bootstrap failed";
    new (&shared->state) std::atomic<DdaInitState>(DdaInitState::NotReady);

    std::vector<pid_t> pids(nranks, -1);
    for (int r = 0; r < nranks; r++)
    {
        pids[r] = fork();
        ASSERT_GE(pids[r], 0) << "fork() failed for rank " << r;
        if (pids[r] == 0)
        {
            _exit(ddaInitRunRank(r, nranks, shared, expectResources));
        }
    }

    bool anySkip = false;
    bool anyFail = false;
    for (int r = 0; r < nranks; r++)
    {
        int status = 0;
        waitpid(pids[r], &status, 0);
        if (WIFEXITED(status))
        {
            int code = WEXITSTATUS(status);
            if (code == kDdaInitChildSkip)
            {
                anySkip = true;
            }
            else if (code != kDdaInitChildOk)
            {
                anyFail = true;
                ADD_FAILURE() << "Rank " << r << " child exited with failure code " << code << ".";
            }
        }
        else
        {
            anyFail = true;
            ADD_FAILURE() << "Rank " << r << " child terminated abnormally (status " << status
                          << ").";
        }
    }

    munmap(shared, sizeof(DdaInitShared));

    if (anySkip && !anyFail)
    {
        GTEST_SKIP() << "Requires " << nranks
                     << " P2P-capable gfx942/gfx950 GPUs; preconditions not met.";
    }
}

}  // namespace

// With the knob on, a 4-rank communicator must come out of ncclCommInitRank with
// the DDA IPC resources allocated -- i.e. init.cc's gate admitted a non-8 rank
// count and ncclDdaIpcCommInit() ran to completion. Reverting that gate to
// `comm->nRanks == kDdaNranks` makes this fail.
TEST(DdaNranksRelaxIsolatedTest, FourRankCommInitAllocatesDdaIpcResources)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "FourRankCommInitAllocatesDdaIpcResources",
        []() { runDdaIpcCommInitEngagementTest(/*nranks=*/4, /*expectResources=*/true); },
        {{"RCCL_DDA_NRANKS_RELAX", "1"}});
}

// Negative control for the test above: with the knob off, the same 4-rank
// communicator must NOT get DDA IPC resources. Without this, the positive test
// cannot distinguish "the gate opened" from "comm init always allocates these".
TEST(DdaNranksRelaxIsolatedTest, FourRankCommInitSkipsDdaIpcWhenRelaxOff)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "FourRankCommInitSkipsDdaIpcWhenRelaxOff",
        []() { runDdaIpcCommInitEngagementTest(/*nranks=*/4, /*expectResources=*/false); },
        {{"RCCL_DDA_NRANKS_RELAX", "0"}});
}

}  // namespace RcclUnitTesting
