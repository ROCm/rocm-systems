/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Multi-node (MPI) end-to-end tests for the per-NET-peer channel-count knob
// from the NCCL 2.28.3 sync (item 16):
//   - ncclConfig_t::nChannelsPerNetPeer  (programmatic config field)
//   - NCCL_NCHANNELS_PER_NET_PEER        (environment override)
//
// Why these live in the MPI binary (rccl-UnitTestsMPI): "NET peers" only exist
// when ranks are on different nodes, so honoring the knob end-to-end and
// exercising the network send/recv path requires a real multi-rank /
// multi-node job. These tests:
//   1. Confirm the config field is honored in the live communicator.
//   2. Confirm NCCL_NCHANNELS_PER_NET_PEER (set at launch) is honored and
//      overrides the config field (documented precedence).
//   3. Run an actual cross-node ring ncclSend/ncclRecv, validate the received
//      data, and confirm the run used the network transport (multi-node comm,
//      nNodes > 1).
//
// NCCL_PARAM note: ncclParamNChannelsPerNetPeer() caches the parsed env value
// in a process-lifetime static, and all GTests share one process per rank.
// Mixing different env values across tests in one run is therefore unreliable,
// and a set env value would override every config-field test. To stay
// deterministic, the ENV behavior is validated against the value present in the
// LAUNCH environment (mpirun -x NCCL_NCHANNELS_PER_NET_PEER=N): the env test
// runs only when it is set, and the config-field tests run only when it is not.

#ifdef MPI_TESTS_ENABLED

#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "TestChecks.hpp"
#include "ResourceGuards.hpp"        // pulls transport.h -> comm.h (struct ncclComm)
#include "DeviceBufferHelpers.hpp"

#include <cstdlib>
#include <string>

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace
{
    // Returns the value of NCCL_NCHANNELS_PER_NET_PEER in the launch
    // environment, or NCCL_CONFIG_UNDEF_INT if unset/empty.
    int launchEnvNChannels()
    {
        const char* v = std::getenv("NCCL_NCHANNELS_PER_NET_PEER");
        if(!v || v[0] == '\0')
            return NCCL_CONFIG_UNDEF_INT;
        return std::atoi(v);
    }

    bool envIsOn(const char* name)
    {
        const char* v = std::getenv(name);
        return v && v[0] != '\0' && std::atoi(v) != 0;
    }

    // On a single node, intra-node ranks use P2P/xGMI/SHM, not the NIC. Setting
    // both of these forces the NET transport, so "net peers" exist on one node
    // and nChannelsPerNetPeer takes effect for the send/recv path.
    bool netForcedOnSingleNode()
    {
        return envIsOn("NCCL_P2P_DISABLE") && envIsOn("NCCL_SHM_DISABLE");
    }

    // Coordinated (all-ranks) run/skip decision: returns the global AND of the
    // per-rank vote so every rank skips or runs together, avoiding MPI hangs if
    // an env var is not propagated to every rank.
    bool allRanksAgreeRun(bool localRun)
    {
        int local  = localRun ? 1 : 0;
        int global = 0;
        MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return global != 0;
    }
} // namespace

/**
 * @class NChannelsPerNetPeerMPITest
 * @brief Fixture that injects ncclConfig_t::nChannelsPerNetPeer into init.
 *
 * Mirrors the TrafficClassMPITest pattern (CommMPITests.cpp): overrides
 * createTestCommunicator() to build the communicator with
 * ncclCommInitRankConfig and the per-test configured value.
 */
class NChannelsPerNetPeerMPITest : public MPITestBase
{
protected:
    static constexpr size_t kNumElements = 1u << 20; // 4 MB of floats per buffer

    int configured_value_ = NCCL_CONFIG_UNDEF_INT;

    ncclResult_t createTestCommunicator() override
    {
        int world_rank = MPIEnvironment::world_rank;
        int world_size = MPIEnvironment::world_size;

        if(world_rank == 0)
            TEST_INFO("Creating communicator with nChannelsPerNetPeer=%d", configured_value_);

        if(world_rank == 0)
            RCCL_TEST_CHECK(ncclGetUniqueId(&nccl_id_));
        MPI_Bcast(&nccl_id_, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

        ncclConfig_t config        = NCCL_CONFIG_INITIALIZER;
        config.nChannelsPerNetPeer = configured_value_;

        RCCL_TEST_CHECK(ncclGroupStart());
        auto group_guard = makeScopeGuard([]() { (void)ncclGroupEnd(); });

        RCCL_TEST_CHECK(
            ncclCommInitRankConfig(&test_comm_, world_size, nccl_id_, world_rank, &config));
        auto comm_guard = makeScopeGuard(
            [this]()
            {
                if(test_comm_)
                {
                    (void)ncclCommDestroy(test_comm_);
                    test_comm_ = nullptr;
                }
            });

        RCCL_TEST_CHECK(ncclGroupEnd());
        group_guard.dismiss();

        HIP_TEST_CHECK(hipStreamCreate(&test_stream_));
        auto stream_guard = makeScopeGuard(
            [this]()
            {
                if(test_stream_)
                {
                    (void)hipStreamDestroy(test_stream_);
                    test_stream_ = nullptr;
                }
            });

        MPI_Barrier(MPI_COMM_WORLD);
        comm_guard.dismiss();
        stream_guard.dismiss();
        return ncclSuccess;
    }

    // Cross-node ring exchange: rank r sends to (r+1)%N and receives from
    // (r-1+N)%N. Verifies the received buffer matches the sender's pattern.
    // *dataOk is set to the verification result. Returns the RCCL/HIP status.
    ncclResult_t runRingSendRecv(bool* dataOk)
    {
        const int    rank = MPIEnvironment::world_rank;
        const int    size = MPIEnvironment::world_size;
        const size_t n    = kNumElements;
        const size_t bytes = n * sizeof(float);

        auto pattern = [](int r, size_t i) -> float
        { return static_cast<float>(r) * 1024.0f + static_cast<float>(i % 1024); };

        void* sendBuf = nullptr;
        void* recvBuf = nullptr;
        HIP_TEST_CHECK(hipMalloc(&sendBuf, bytes));
        HIP_TEST_CHECK(hipMalloc(&recvBuf, bytes));
        HIP_TEST_CHECK(initializeBufferWithPattern<float>(
            sendBuf, n, [&](size_t i) { return pattern(rank, i); }));
        HIP_TEST_CHECK(zeroInitializeBuffer<float>(recvBuf, n));
        HIP_TEST_CHECK(hipStreamSynchronize(0));

        const int sendPeer = (rank + 1) % size;
        const int recvPeer = (rank - 1 + size) % size;

        RCCL_TEST_CHECK(ncclGroupStart());
        RCCL_TEST_CHECK(
            ncclRecv(recvBuf, n, ncclFloat, recvPeer, getActiveCommunicator(), getActiveStream()));
        RCCL_TEST_CHECK(
            ncclSend(sendBuf, n, ncclFloat, sendPeer, getActiveCommunicator(), getActiveStream()));
        RCCL_TEST_CHECK(ncclGroupEnd());
        HIP_TEST_CHECK(hipStreamSynchronize(getActiveStream()));

        const bool ok = verifyBufferData<float>(
            recvBuf, n, [&](size_t i) { return pattern(recvPeer, i); }, n, /*tolerance=*/0.5);
        if(dataOk)
            *dataOk = ok;

        (void)hipFree(sendBuf);
        (void)hipFree(recvBuf);
        return ncclSuccess;
    }
};

/**
 * @test ConfigField_HonoredEndToEnd
 * @brief The ncclConfig_t::nChannelsPerNetPeer value is applied to the comm.
 */
TEST_F(NChannelsPerNetPeerMPITest, ConfigField_HonoredEndToEnd)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));
    if(launchEnvNChannels() != NCCL_CONFIG_UNDEF_INT)
        GTEST_SKIP() << "NCCL_NCHANNELS_PER_NET_PEER is set at launch; it would "
                        "override the config field. Run without it for this test.";

    constexpr int kVal  = 4;
    configured_value_   = kVal;

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ASSERT_MPI_EQ(getActiveCommunicator()->config.nChannelsPerNetPeer, kVal);
}

/**
 * @test ConfigField_SendRecvOverNet
 * @brief With the config field set, run a real ring send/recv over the NET
 *        transport, validate the data, and check the channels picked at runtime.
 *
 * Runs in two modes:
 *   - Multi-node (>= 2 nodes): the ring naturally crosses the NIC.
 *   - Single node with NET forced (NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1):
 *     intra-node peers become NET peers, so the knob applies on one node.
 * Skips cleanly otherwise. Must run with NCCL_NCHANNELS_PER_NET_PEER unset
 * (a launch env value would override the config field).
 */
TEST_F(NChannelsPerNetPeerMPITest, ConfigField_SendRecvOverNet)
{
    const bool multiNode = (detectNodeCount() > 1);
    const bool netForced = netForcedOnSingleNode();
    const bool wantRun   = (MPIEnvironment::world_size >= kMinProcessesForMPI)
                         && (launchEnvNChannels() == NCCL_CONFIG_UNDEF_INT)
                         && (multiNode || netForced);
    if(!allRanksAgreeRun(wantRun))
        GTEST_SKIP() << "Needs >= 2 ranks and either >= 2 nodes, or a single node "
                        "with NET forced (NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1), "
                        "and NCCL_NCHANNELS_PER_NET_PEER unset.";

    constexpr int kVal = 4;
    configured_value_  = kVal;

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    // Honored end-to-end.
    ASSERT_MPI_EQ(comm->config.nChannelsPerNetPeer, kVal);

    // Multi-node communicators have NET peers by construction; on a single node
    // the NET path is in use only because P2P/SHM were force-disabled.
    if(multiNode)
        ASSERT_MPI_TRUE(comm->nNodes > 1);

    // Check the channels actually picked at runtime. ncclTopoComputeP2pChannels
    // sets comm->p2pnChannelsPerPeer = pow2Up(min over peers of per-peer
    // channels); for a NET peer that per-peer count is
    // comm->config.nChannelsPerNetPeer. The configured value bounds the per-peer
    // channels selected; allow x2 for the single-node channel doubling on
    // gfx942/gfx950.
    const int picked   = comm->p2pnChannelsPerPeer;
    const int totalP2p = comm->p2pnChannels;
    ASSERT_MPI_TRUE(picked > 0);
    ASSERT_MPI_TRUE((picked & (picked - 1)) == 0); // power of two
    ASSERT_MPI_TRUE(totalP2p >= picked);
    ASSERT_MPI_TRUE(picked <= 2 * kVal);

    if(getTestMpiRank() == 0)
        TEST_INFO("Picked channels: p2pnChannelsPerPeer=%d, p2pnChannels=%d "
                  "(config nChannelsPerNetPeer=%d, nNodes=%d, netForced=%d)",
                  picked,
                  totalP2p,
                  kVal,
                  comm->nNodes,
                  static_cast<int>(netForced));

    bool dataOk = false;
    ASSERT_MPI_EQ(ncclSuccess, runRingSendRecv(&dataOk));
    ASSERT_MPI_TRUE(dataOk); // correct delivery over the NET path

    if(getTestMpiRank() == 0)
        TEST_INFO("Ring send/recv over NET verified (nChannelsPerNetPeer=%d, nNodes=%d, netForced=%d)",
                  kVal,
                  comm->nNodes,
                  static_cast<int>(netForced));
}

/**
 * @test Env_HonoredAndOverridesConfig
 * @brief NCCL_NCHANNELS_PER_NET_PEER (from the launch env) is honored and
 *        overrides an explicit config field. Skips if the env var is unset.
 */
TEST_F(NChannelsPerNetPeerMPITest, Env_HonoredAndOverridesConfig)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    const int envVal = launchEnvNChannels();
    if(envVal == NCCL_CONFIG_UNDEF_INT)
        GTEST_SKIP() << "Set NCCL_NCHANNELS_PER_NET_PEER (e.g. mpirun -x "
                        "NCCL_NCHANNELS_PER_NET_PEER=8) to run this test.";

    // Set a different config field value so a pass proves env precedence.
    configured_value_ = (envVal == 2) ? 1 : 2;

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ASSERT_MPI_EQ(getActiveCommunicator()->config.nChannelsPerNetPeer, envVal);
    ASSERT_MPI_NE(getActiveCommunicator()->config.nChannelsPerNetPeer, configured_value_);
}

#endif // MPI_TESTS_ENABLED
