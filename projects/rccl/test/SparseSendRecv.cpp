/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file SparseSendRecv.cpp
 * @brief Regression test for AICOMRCCL-1112 / NCCL 2.29.7 fix:
 *        "Fix hang issue in send/receive scheduling of repeated sparse patterns"
 *
 * Root cause: NCCL 2.29.2 scheduleP2pTasksToPlan() incremented p2pEpoch at the TOP of its
 * while-loop, before processing rounds. If the kernel plan budget was exhausted
 * mid-round, the function returned early — but the epoch had already been
 * incremented. On the next plan call, the epoch was incremented again before the
 * remaining rounds were processed. In sparse patterns where there is asymmetry in number
 * of operations, budget can be exhausted at different points.
 * This can lead to different epochs.
 * epoch enforces batch boundaries, and a mismatch in batch grouping
 * between sender and receiver breaks the rendezvous protocol.
 * In RCCL, epoch doesn't impact batching, however this test will expose any mismatch
 * due to multi-plan schedule.
 *
 * NOTE: RCCL never had the NCCL 2.29.3 intermediate bug state — it merged
 * NCCL 2.29.2+2.29.3+2.29.7 in one shot. As a result, no RCCL version was susceptible to this bug.
 *
 * Trigger conditions:
 *   1. Budget exhaustion (NCCL_WORK_ARGS_BYTES=512 shrinks inArgsBytes to ~464B;
 *      8 ops x 64B = 512B > 464B, exhausting budget mid-group).
 *   2. Sparse topology (not all-to-all): rank 0 has 2*(N-1) ops while others
 *      have 2. With N>=5, rank 0's ops spill to a 2nd plan; others do not.
 *
 * Test topology: rank 0 is the hub — it sends to and receives from every
 * other rank. All other ranks only communicate with rank 0.
 *
 * Run (requires NCCL_WORK_ARGS_BYTES=512 and NCCL_WORK_FIFO_BYTES=512):
 *   NCCL_WORK_ARGS_BYTES=512 NCCL_WORK_FIFO_BYTES=512 \
 *     mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=SparseSendRecv.*
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
constexpr int    MIN_RANKS              = 5;
constexpr size_t NUM_ELEMS              = 1024;
constexpr char   kWorkArgsBytesEnv[]    = "NCCL_WORK_ARGS_BYTES";
constexpr char   kWorkFifoBytesEnv[]    = "NCCL_WORK_FIFO_BYTES";
constexpr char   kRequiredEnvValue[]    = "512";

bool envVarEquals(const char* name, const char* expected)
{
    const char* val = std::getenv(name);
    return val != nullptr && std::strcmp(val, expected) == 0;
}

bool sparseSendRecvEnvConfigured()
{
    return envVarEquals(kWorkArgsBytesEnv, kRequiredEnvValue)
        && envVarEquals(kWorkFifoBytesEnv, kRequiredEnvValue);
}
}

class SparseSendRecv : public MPITestBase
{
protected:
    float*              send_buf = nullptr;
    float*              recv_buf = nullptr;
    std::vector<float*> hub_recv_bufs_;
    const size_t        buf_bytes = NUM_ELEMS * sizeof(float);

    void TearDown() override
    {
        for (float* buf : hub_recv_bufs_) {
            if (buf) {
                hipFree(buf);
            }
        }
        hub_recv_bufs_.clear();

        if (send_buf) {
            hipFree(send_buf);
            send_buf = nullptr;
        }
        if (recv_buf) {
            hipFree(recv_buf);
            recv_buf = nullptr;
        }

        MPITestBase::TearDown();
    }
};

/**
 * @brief Validate that sparse P2P group calls do not deadlock when there is asymmetry
 *        in the number of operations and multiple plans are used.
 *
 * Topology: rank 0 <-> all other ranks (star).
 * Rank 0 uses a distinct recv buffer per spoke; hub and spokes verify payload correctness.
 *
 * Requires NCCL_WORK_ARGS_BYTES=512 and NCCL_WORK_FIFO_BYTES=512 so rank 0's ops spill
 * across multiple kernel plans while spoke ranks stay within budget.
 */
TEST_F(SparseSendRecv, StarTopology)
{
    if (!validateTestPrerequisites(MIN_RANKS)) {
        GTEST_SKIP() << "StarTopology requires at least "
                     << MIN_RANKS << " MPI processes.";
    }

    if (!sparseSendRecvEnvConfigured()) {
        GTEST_SKIP() << "StarTopology requires "
                     << kWorkArgsBytesEnv << "=" << kRequiredEnvValue << " and "
                     << kWorkFifoBytesEnv << "=" << kRequiredEnvValue << ".";
    }

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    int rank   = MPIEnvironment::world_rank;
    int nRanks = MPIEnvironment::world_size;

    const float        rank_val = static_cast<float>(rank);
    std::vector<float> rank_vals(NUM_ELEMS, rank_val);

    ASSERT_EQ(hipSuccess, hipMalloc(&send_buf, buf_bytes));
    ASSERT_EQ(hipSuccess,
              hipMemcpy(send_buf, rank_vals.data(), buf_bytes, hipMemcpyHostToDevice));

    if (rank == 0) {
        hub_recv_bufs_.resize(nRanks - 1);
        for (int peer = 1; peer < nRanks; peer++) {
            float* peer_recv_buf = nullptr;
            ASSERT_EQ(hipSuccess, hipMalloc(&peer_recv_buf, buf_bytes));
            ASSERT_EQ(hipSuccess, hipMemset(peer_recv_buf, 0, buf_bytes));
            hub_recv_bufs_[peer - 1] = peer_recv_buf;
        }
    } else {
        ASSERT_EQ(hipSuccess, hipMalloc(&recv_buf, buf_bytes));
        ASSERT_EQ(hipSuccess, hipMemset(recv_buf, 0, buf_bytes));
    }

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    if (rank == 0) {
        // Hub: send to and receive from every other rank into a per-peer buffer.
        for (int peer = 1; peer < nRanks; peer++) {
            ASSERT_EQ(ncclSuccess,
                          ncclSend(send_buf, NUM_ELEMS, ncclFloat, peer, comm, stream));
            ASSERT_EQ(ncclSuccess,
                          ncclRecv(hub_recv_bufs_[peer - 1], NUM_ELEMS, ncclFloat, peer, comm,
                                   stream));
        }
    } else {
        // Spoke: send to and receive from rank 0 only.
        ASSERT_EQ(ncclSuccess,
                      ncclSend(send_buf, NUM_ELEMS, ncclFloat, 0, comm, stream));
        ASSERT_EQ(ncclSuccess,
                      ncclRecv(recv_buf, NUM_ELEMS, ncclFloat, 0, comm, stream));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    if (rank == 0) {
        for (int peer = 1; peer < nRanks; peer++) {
            std::vector<float> received_vals(NUM_ELEMS, 0.0f);
            const float        expected = static_cast<float>(peer);
            ASSERT_EQ(hipSuccess,
                      hipMemcpy(received_vals.data(), hub_recv_bufs_[peer - 1], buf_bytes,
                                hipMemcpyDeviceToHost));
           for (size_t i = 0; i < NUM_ELEMS; i++) {
                ASSERT_FLOAT_EQ(received_vals[i], expected)
                    << "Rank 0 expected " << expected << " from rank " << peer << " at index "
                    << i << ", got " << received_vals[i];
            }
        }
    } else {
        std::vector<float> received_vals(NUM_ELEMS, 0.0f);
        const float        expected_from_hub = 0.0f;
        ASSERT_EQ(hipSuccess,
                  hipMemcpy(received_vals.data(), recv_buf, buf_bytes, hipMemcpyDeviceToHost));
        for (size_t i = 0; i < NUM_ELEMS; i++) {
            ASSERT_FLOAT_EQ(received_vals[i], expected_from_hub)
                << "Rank " << rank << " expected " << expected_from_hub << " from rank 0 at index "
                << i << ", got " << received_vals[i];
        }
    }
}

#endif // MPI_TESTS_ENABLED
