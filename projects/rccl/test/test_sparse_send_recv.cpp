/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file test_sparse_send_recv.cpp
 * @brief Regression test for AICOMRCCL-1112 / NCCL 2.29.7 fix:
 *        "Fix hang issue in send/receive scheduling of repeated sparse patterns"
 *
 * Root cause: NCCL 2.29.2 scheduleP2pTasksToPlan() incremented p2pEpoch at
 * the TOP of its while-loop before processing any rounds. If the kernel plan
 * budget was exhausted mid-round the function returned early — but the epoch
 * had already been incremented. The next plan call incremented it again before
 * the remaining rounds. In sparse patterns with asymmetric operation counts,
 * budget exhausts at different points per rank, leading to different epochs on
 * paired send/recv ranks. This corrupts p2pOpCount and the proxy
 * resources->step counter, causing a permanent deadlock in waitPeer().
 *
 * NOTE: RCCL never had the NCCL 2.29.3 intermediate bug state — it merged
 * NCCL 2.29.2+2.29.3+2.29.7 in one shot. As a result, no released RCCL
 * version was susceptible to this bug. This test validates that any future
 * regression in scheduleP2pTasksToPlan() is caught.
 *
 * Bug trigger — three conditions must all hold:
 *   1. Budget exhaustion: NCCL_WORK_ARGS_BYTES=512 and NCCL_WORK_FIFO_BYTES=512
 *      shrink the plan budget so ~7 rounds exhaust it. NCCL_MAX_NCHANNELS=4
 *      fixes the channel count for deterministic plan splitting.
 *   2. Sparse + repeated topology: REPS repetitions of the full star pattern
 *      within ONE ncclGroupStart/End. Rank 0 generates REPS*(nRanks-1) rounds;
 *      each leaf generates only REPS rounds. Budget exhaustion splits rank 0
 *      across multiple plans with ascending epochs while leaves stay in one
 *      plan (epoch=1) — the epoch mismatch.
 *   3. Multiple outer iterations: corrupted p2pOpCount accumulates across group
 *      calls; the hang typically surfaces on the 2nd or 3rd iteration.
 *
 * Run:
 *   NCCL_WORK_ARGS_BYTES=512 NCCL_WORK_FIFO_BYTES=512 \
 *   NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4 \
 *     mpirun -np 16 ./rccl-UnitTestsMPI --gtest_filter=SparseSendRecv.*
 */

#include "MPITestBase.hpp"
#include "TestChecks.hpp"

#ifdef MPI_TESTS_ENABLED

#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <vector>

using namespace MPITestConstants;

namespace {
// Absolute minimum for the star pattern; reliable bug reproduction requires 16
// ranks (REPS=4 gives rank 0 60 rounds, enough to exhaust the ~7-round plan budget).
constexpr int    MIN_RANKS           = 16;
// Repetitions of the full star pattern within ONE ncclGroupStart/End call.
// Creates the asymmetric round count: rank 0 gets REPS*(nRanks-1) rounds,
// each leaf gets REPS rounds. This is what forces the epoch mismatch.
constexpr int    REPS                = 4;
// Outer iterations. Corrupted resources->step accumulates across group calls;
// 5 iterations is enough to make the hang definitive rather than flaky.
constexpr int    ITERS               = 5;
constexpr size_t NUM_ELEMS           = 1024;
// Required environment for this test. Set these in the test-runner config or
// mpirun invocation — the test skips itself if any value is not honored:
//   NCCL_WORK_ARGS_BYTES=512   shrinks inArgsBytes so ~7 rounds exhaust budget
//   NCCL_WORK_FIFO_BYTES=512   shrinks outArgsBytes (second budget threshold)
//   NCCL_MIN_NCHANNELS=4       fixes channel count for deterministic plan split
//   NCCL_MAX_NCHANNELS=4       (same)
constexpr char kWorkArgsBytesEnv[]  = "NCCL_WORK_ARGS_BYTES";
constexpr char kWorkFifoBytesEnv[]  = "NCCL_WORK_FIFO_BYTES";
constexpr char kMinNChannelsEnv[]   = "NCCL_MIN_NCHANNELS";
constexpr char kMaxNChannelsEnv[]   = "NCCL_MAX_NCHANNELS";
constexpr char kBudgetValue[]       = "512";
constexpr char kNChannelsValue[]    = "4";

bool envVarEquals(const char* name, const char* expected)
{
    const char* val = std::getenv(name);
    return val != nullptr && std::strcmp(val, expected) == 0;
}

bool sparseSendRecvEnvConfigured()
{
    return envVarEquals(kWorkArgsBytesEnv, kBudgetValue)
        && envVarEquals(kWorkFifoBytesEnv, kBudgetValue)
        && envVarEquals(kMinNChannelsEnv,  kNChannelsValue)
        && envVarEquals(kMaxNChannelsEnv,  kNChannelsValue);
}
} // namespace

class SparseSendRecv : public MPITestBase
{
protected:
    float*                           send_buf = nullptr;
    // Hub (rank 0): REPS recv buffers per peer to avoid aliasing within a group call.
    // Leaf ranks: REPS recv buffers total.
    std::vector<std::vector<float*>> hub_recv_bufs_;  // [peer-1][rep]
    std::vector<float*>              leaf_recv_bufs_;  // [rep]
    const size_t                     buf_bytes = NUM_ELEMS * sizeof(float);

    void TearDown() override
    {
        for (auto& peer_bufs : hub_recv_bufs_) {
            for (float* buf : peer_bufs) {
                if (buf) hipFree(buf);
            }
        }
        hub_recv_bufs_.clear();

        for (float* buf : leaf_recv_bufs_) {
            if (buf) hipFree(buf);
        }
        leaf_recv_bufs_.clear();

        if (send_buf) { hipFree(send_buf); send_buf = nullptr; }

        MPITestBase::TearDown();
    }
};

/**
 * @brief Validate that repeated sparse P2P group calls do not deadlock.
 *
 * Star topology: rank 0 (hub) <-> all other ranks (leaves).
 * Each group call issues REPS repetitions of the full star pattern, producing
 * REPS*(nRanks-1) rounds on rank 0 vs REPS rounds on each leaf. Budget
 * exhaustion (NCCL_WORK_FIFO_BYTES=512) splits rank 0 across multiple kernel
 * plans with ascending p2pEpoch values while leaves fit in one plan — the
 * epoch mismatch that causes the hang on pre-fix NCCL/RCCL.
 */
TEST_F(SparseSendRecv, StarTopology)
{
    if (!validateTestPrerequisites(MIN_RANKS)) {
        GTEST_SKIP() << "StarTopology requires at least "
                     << MIN_RANKS << " MPI processes.";
    }

    if (!sparseSendRecvEnvConfigured()) {
        GTEST_SKIP() << "StarTopology requires "
                     << kWorkArgsBytesEnv << "=" << kBudgetValue << ", "
                     << kWorkFifoBytesEnv << "=" << kBudgetValue << ", "
                     << kMinNChannelsEnv  << "=" << kNChannelsValue << ", "
                     << kMaxNChannelsEnv  << "=" << kNChannelsValue
                     << " — set these in the test-runner config or mpirun -x flags.";
    }

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    int rank   = MPIEnvironment::world_rank;
    int nRanks = MPIEnvironment::world_size;

    const float        rank_val = static_cast<float>(rank);
    std::vector<float> rank_vals(NUM_ELEMS, rank_val);

    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&send_buf, buf_bytes));
    ASSERT_MPI_EQ(hipSuccess,
              hipMemcpy(send_buf, rank_vals.data(), buf_bytes, hipMemcpyHostToDevice));

    // Allocate per-repetition recv buffers to avoid aliasing within a group call.
    // Buffers are zeroed once at allocation; stream ordering guarantees no stale
    // data from a prior iteration reaches the verification step.
    if (rank == 0) {
        hub_recv_bufs_.assign(nRanks - 1, std::vector<float*>(REPS, nullptr));
        for (int peer = 1; peer < nRanks; ++peer) {
            for (int rep = 0; rep < REPS; ++rep) {
                ASSERT_EQ(hipSuccess, hipMalloc(&hub_recv_bufs_[peer - 1][rep], buf_bytes));
                ASSERT_EQ(hipSuccess, hipMemset(hub_recv_bufs_[peer - 1][rep], 0, buf_bytes));
            }
        }
    } else {
        leaf_recv_bufs_.assign(REPS, nullptr);
        for (int rep = 0; rep < REPS; ++rep) {
            ASSERT_EQ(hipSuccess, hipMalloc(&leaf_recv_bufs_[rep], buf_bytes));
            ASSERT_EQ(hipSuccess, hipMemset(leaf_recv_bufs_[rep], 0, buf_bytes));
        }
    }

    // Outer loop: ITERS group calls with the same sparse pattern.
    // Corrupted resources->step state accumulates across calls; the hang
    // typically surfaces on the 2nd or 3rd iteration.
    for (int iter = 0; iter < ITERS; ++iter) {
        ASSERT_MPI_EQ(ncclSuccess, ncclGroupStart());

        if (rank == 0) {
            // Hub: REPS full sweeps over all leaf peers within one group call.
            // Total rounds for rank 0: REPS * (nRanks-1).
            // Budget exhaustion forces multiple kernel plans with ascending epochs.
            for (int rep = 0; rep < REPS; ++rep) {
                for (int peer = 1; peer < nRanks; ++peer) {
                    // Use EXPECT (not ASSERT) inside the group: an ASSERT would
                    // exit the test body without calling ncclGroupEnd(), leaving
                    // all other ranks blocked forever.
                    EXPECT_EQ(ncclSuccess,
                        ncclSend(send_buf, NUM_ELEMS, ncclFloat, peer, comm, stream));
                    EXPECT_EQ(ncclSuccess,
                        ncclRecv(hub_recv_bufs_[peer - 1][rep], NUM_ELEMS,
                                 ncclFloat, peer, comm, stream));
                }
            }
        } else {
            // Leaf: REPS sends+recvs to rank 0 only.
            // Total rounds: REPS. Fits in one plan (epoch=1) on a pre-fix library.
            for (int rep = 0; rep < REPS; ++rep) {
                EXPECT_EQ(ncclSuccess,
                    ncclSend(send_buf, NUM_ELEMS, ncclFloat, 0, comm, stream));
                EXPECT_EQ(ncclSuccess,
                    ncclRecv(leaf_recv_bufs_[rep], NUM_ELEMS, ncclFloat, 0, comm, stream));
            }
        }

        ASSERT_MPI_EQ(ncclSuccess, ncclGroupEnd());
        // Bail out if any ncclSend/ncclRecv failed — the stream state is undefined.
        if (HasFailure()) break;

        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

        // Spot-check first repetition's recv buffers each iteration.
        // Reps 1-(REPS-1) are not verified; the primary signal is whether
        // hipStreamSynchronize returns at all (hang detection).
        if (rank == 0) {
            for (int peer = 1; peer < nRanks; ++peer) {
                std::vector<float> received(NUM_ELEMS, 0.0f);
                ASSERT_EQ(hipSuccess,
                          hipMemcpy(received.data(), hub_recv_bufs_[peer - 1][0],
                                    buf_bytes, hipMemcpyDeviceToHost));
                const float expected = static_cast<float>(peer);
                for (size_t i = 0; i < NUM_ELEMS; ++i) {
                    ASSERT_FLOAT_EQ(received[i], expected)
                        << "[iter " << iter << "] Rank 0 expected " << expected
                        << " from rank " << peer << " at index " << i
                        << ", got " << received[i];
                }
            }
        } else {
            std::vector<float> received(NUM_ELEMS, 0.0f);
            ASSERT_EQ(hipSuccess,
                      hipMemcpy(received.data(), leaf_recv_bufs_[0],
                                buf_bytes, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < NUM_ELEMS; ++i) {
                ASSERT_FLOAT_EQ(received[i], 0.0f)
                    << "[iter " << iter << "] Rank " << rank
                    << " expected 0.0f from rank 0 at index " << i
                    << ", got " << received[i];
            }
        }
    }
}

#endif // MPI_TESTS_ENABLED
