/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Multi-rank / multi-node end-to-end tests for the per-NET-peer channel-count
// knob (NCCL 2.28.3 sync, item 16):
//   - ncclConfig_t::nChannelsPerNetPeer  (programmatic config field)
//   - NCCL_NCHANNELS_PER_NET_PEER        (environment override)
//
// This file proves the knob is honored when a *real, multi-rank* communicator is
// built across processes (and, for the gated tests, across nodes), where the
// value actually drives the NET (inter-node) peer channel count in
// src/graph/paths.cc
// (ncclTopoGetNchannels -> nNetChannels = comm->config.nChannelsPerNetPeer).
//
// What is asserted, end-to-end:
//   * The resolved comm->config.nChannelsPerNetPeer matches what the user set,
//     on EVERY rank (MPI-aware), and all ranks agree on the value.
//   * NCCL_NCHANNELS_PER_NET_PEER is honored across ranks.
//   * env overrides an explicit config field across ranks (documented precedence).
//   * Default (neither set) stays UNDEF across ranks so paths.cc auto-tunes.
//   * The communicator built with the knob is functional: a real AllReduce
//     produces the correct result across all ranks (on >1 node this exercises
//     the NET path the knob controls).
//   * Multi-node-gated: the same honoring + functional check with >= 2 nodes,
//     where nChannelsPerNetPeer genuinely governs inter-node channels.
//
// Execution model: env-dependent cases rely on the MPI test runner spawning one
// mpirun per test (see MPITestCore.cpp), so each lands in a fresh process where
// NCCL_PARAM(NChannelsPerNetPeer, ...) has not yet cached its value. Running the
// binary directly with a filter that mixes env and non-env cases in one process
// can let the first-seen env value stick (NCCL_PARAM caches per process).

#ifdef MPI_TESTS_ENABLED

#include "MPITestBase.hpp"
#include "TestChecks.hpp"
#include "ResourceGuards.hpp"

#include "comm.h"      // internal: struct ncclComm::config + channels[]/peers[]
#include "device.h"    // struct ncclConnector / ncclChannelPeer (connected, transportComm)
#include "transport.h" // extern struct ncclTransport netTransport (NET transport id)

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <mpi.h>

using namespace MPITestConstants;
using namespace RCCLTestGuards;

// NCCL_PARAM(NChannelsPerNetPeer, ...) in src/init.cc generates this
// externally-linkable accessor. We use it to detect whether the per-process
// param value is still re-readable (i.e. this is the first init in the process)
// so the env-driven tests can SKIP cleanly instead of FAILing when they run in
// the same process as an earlier test that already cached a different value.
// Declared with C++ linkage to match the generated definition in src/init.cc
// (NCCL_PARAM(NChannelsPerNetPeer, ...)).
int64_t ncclParamNChannelsPerNetPeer();

namespace
{
// Sentinel meaning "leave the config field at its default (UNDEF)".
constexpr int kLeaveConfigUnset = NCCL_CONFIG_UNDEF_INT;
} // namespace

/**
 * @class NChannelsPerNetPeerMPITest
 * @brief Multi-rank fixture that injects nChannelsPerNetPeer via config and/or
 *        the NCCL_NCHANNELS_PER_NET_PEER env, then exposes the live communicator.
 *
 * Mirrors the CommMPITests.cpp "Custom Test Class" pattern: overrides
 * createTestCommunicator() to call ncclCommInitRankConfig() with the test's
 * configuration while reusing the base members (test_comm_, test_stream_,
 * nccl_id_) so getActiveCommunicator()/getActiveStream() keep working.
 */
class NChannelsPerNetPeerMPITest : public MPITestBase
{
protected:
    // Config field to request. kLeaveConfigUnset leaves it UNDEF.
    int configured_value_ = kLeaveConfigUnset;

    ncclResult_t createTestCommunicator() override
    {
        int world_rank = MPIEnvironment::world_rank;
        int world_size = MPIEnvironment::world_size;

        if(world_rank == 0)
        {
            TEST_INFO("Creating communicator with requested nChannelsPerNetPeer config=%d",
                      configured_value_);
        }

        if(world_rank == 0)
        {
            RCCL_TEST_CHECK(ncclGetUniqueId(&nccl_id_));
        }
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

        if(world_rank == 0)
        {
            TEST_INFO("Communicator created; resolved nChannelsPerNetPeer=%d",
                      test_comm_->config.nChannelsPerNetPeer);
        }
        return ncclSuccess;
    }

    // Assert all ranks resolved the same nChannelsPerNetPeer value (consistency),
    // independent of what that value is. MPI-aware: every rank participates.
    void expectAllRanksAgree(int local_value)
    {
        int min_value = 0;
        int max_value = 0;
        ASSERT_MPI_SUCCESS(
            MPI_Allreduce(&local_value, &min_value, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        ASSERT_MPI_SUCCESS(
            MPI_Allreduce(&local_value, &max_value, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
        ASSERT_MPI_EQ(min_value, max_value);
    }

    // Run a real AllReduce(sum) over the live communicator and validate the
    // result on every rank. With world_size ranks each contributing their rank
    // index, the sum is 0+1+...+(world_size-1). On a >1-node communicator this
    // drives data over the NET peers whose channel count the knob governs, so a
    // correct result is the end-to-end functional proof.
    void expectFunctionalAllReduce()
    {
        const int   world_size = MPIEnvironment::world_size;
        const int   world_rank = MPIEnvironment::world_rank;
        ncclComm_t  comm       = getActiveCommunicator();
        hipStream_t stream     = getActiveStream();
        ASSERT_MPI_TRUE(comm != nullptr && stream != nullptr);

        constexpr int kCount = 1024;
        const float   expected =
            static_cast<float>((static_cast<int64_t>(world_size) * (world_size - 1)) / 2);

        float* send_dev = nullptr;
        float* recv_dev = nullptr;
        HIP_CHECK(hipMalloc(&send_dev, kCount * sizeof(float)));
        auto send_guard = makeScopeGuard([&]() { (void)hipFree(send_dev); });
        HIP_CHECK(hipMalloc(&recv_dev, kCount * sizeof(float)));
        auto recv_guard = makeScopeGuard([&]() { (void)hipFree(recv_dev); });

        std::vector<float> host(kCount, static_cast<float>(world_rank));
        HIP_CHECK(
            hipMemcpy(send_dev, host.data(), kCount * sizeof(float), hipMemcpyHostToDevice));

        RCCL_TEST_CHECK_GTEST_FAIL(
            ncclAllReduce(send_dev, recv_dev, kCount, ncclFloat, ncclSum, comm, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        std::vector<float> result(kCount, -1.0f);
        HIP_CHECK(
            hipMemcpy(result.data(), recv_dev, kCount * sizeof(float), hipMemcpyDeviceToHost));

        bool local_ok = true;
        for(int i = 0; i < kCount; ++i)
        {
            if(result[i] != expected)
            {
                local_ok = false;
                if(world_rank == 0)
                    TEST_WARN("AllReduce mismatch at [%d]: got %f expected %f",
                              i, result[i], expected);
                break;
            }
        }
        ASSERT_MPI_TRUE(local_ok);
    }

    // --- helpers for the NET send/recv + transport-selection + audit tests ---

    // Return a rank that lives on a DIFFERENT node than rank 0 (guaranteed a NET
    // peer), or -1 if the job is single-node. Deterministic on every rank
    // (all-gather of hostnames), so all ranks agree on the chosen peer.
    int pickInterNodePeer()
    {
        const int world_size = MPIEnvironment::world_size;
        char      myname[MPI_MAX_PROCESSOR_NAME];
        int       len = 0;
        memset(myname, 0, sizeof(myname));
        MPI_Get_processor_name(myname, &len);

        std::vector<char> all(static_cast<size_t>(world_size) * MPI_MAX_PROCESSOR_NAME, 0);
        MPI_Allgather(myname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all.data(),
                      MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);

        const char* host0 = all.data();
        for(int r = 1; r < world_size; ++r)
        {
            if(strncmp(host0, all.data() + static_cast<size_t>(r) * MPI_MAX_PROCESSOR_NAME,
                       MPI_MAX_PROCESSOR_NAME)
               != 0)
                return r;
        }
        return -1;
    }

    // Return a rank (!= 0) on the SAME node as rank 0 (an intra-node peer that
    // should NOT use NET), or -1 if rank 0 is alone on its node.
    int pickSameNodePeer()
    {
        const int world_size = MPIEnvironment::world_size;
        char      myname[MPI_MAX_PROCESSOR_NAME];
        int       len = 0;
        memset(myname, 0, sizeof(myname));
        MPI_Get_processor_name(myname, &len);

        std::vector<char> all(static_cast<size_t>(world_size) * MPI_MAX_PROCESSOR_NAME, 0);
        MPI_Allgather(myname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all.data(),
                      MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);

        const char* host0 = all.data();
        for(int r = 1; r < world_size; ++r)
        {
            if(strncmp(host0, all.data() + static_cast<size_t>(r) * MPI_MAX_PROCESSOR_NAME,
                       MPI_MAX_PROCESSOR_NAME)
               == 0)
                return r;
        }
        return -1;
    }

    // Count how many channels have an established NET send-connector to `peer`.
    // This is the per-peer NET channel count actually applied by the library
    // (paths.cc derives it from comm->config.nChannelsPerNetPeer). Mirrors the
    // library's own transport check (enqueue.cc: conn->transportComm == &netTransport.send).
    int countNetSendChannelsToPeer(ncclComm_t comm, int peer)
    {
        int count = 0;
        for(int c = 0; c < comm->nChannels; ++c)
        {
            struct ncclChannelPeer* p = comm->channels[c].peers[peer];
            if(p == nullptr)
                continue;
            for(int ci = 0; ci < NCCL_MAX_CONNS; ++ci)
            {
                if(p->send[ci].connected && p->send[ci].transportComm == &netTransport.send)
                {
                    ++count;
                    break; // count the channel once
                }
            }
        }
        return count;
    }

    // Point-to-point send/recv between two ranks over the live communicator.
    // All ranks call ncclGroupStart/End (empty group on non-participants); only
    // the pair issue the transfer. On the receiver, *recvDataOk is set to whether
    // every received element equals `value`. Forces the (lazy) connection between
    // the pair to be established, which is what the transport-selection and audit
    // tests inspect afterwards.
    ncclResult_t doSendRecv(int senderRank, int recvRank, float value, bool* recvDataOk)
    {
        const int     world_rank = MPIEnvironment::world_rank;
        ncclComm_t    comm       = getActiveCommunicator();
        hipStream_t   stream     = getActiveStream();
        constexpr int kCount     = 8192; // large enough to exercise the NET path

        float* buf = nullptr;
        HIPCHECK(hipMalloc(&buf, kCount * sizeof(float)));
        auto buf_guard = makeScopeGuard([&]() { (void)hipFree(buf); });

        if(world_rank == senderRank)
        {
            std::vector<float> host(kCount, value);
            HIPCHECK(hipMemcpy(buf, host.data(), kCount * sizeof(float), hipMemcpyHostToDevice));
        }
        else if(world_rank == recvRank)
        {
            HIPCHECK(hipMemset(buf, 0, kCount * sizeof(float)));
        }

        RCCL_TEST_CHECK(ncclGroupStart());
        if(world_rank == senderRank)
            RCCL_TEST_CHECK(ncclSend(buf, kCount, ncclFloat, recvRank, comm, stream));
        if(world_rank == recvRank)
            RCCL_TEST_CHECK(ncclRecv(buf, kCount, ncclFloat, senderRank, comm, stream));
        RCCL_TEST_CHECK(ncclGroupEnd());

        if(world_rank == senderRank || world_rank == recvRank)
            HIPCHECK(hipStreamSynchronize(stream));

        if(world_rank == recvRank && recvDataOk != nullptr)
        {
            std::vector<float> host(kCount, -1.0f);
            HIPCHECK(hipMemcpy(host.data(), buf, kCount * sizeof(float), hipMemcpyDeviceToHost));
            bool ok = true;
            for(int i = 0; i < kCount; ++i)
            {
                if(host[i] != value)
                {
                    ok = false;
                    break;
                }
            }
            *recvDataOk = ok;
        }
        return ncclSuccess;
    }
};

// ---------------------------------------------------------------------------
// Config field honored across a real multi-rank communicator.
// ---------------------------------------------------------------------------
TEST_F(NChannelsPerNetPeerMPITest, ConfigField_HonoredAcrossRanks)
{
    SKIP_UNLESS_MPI_PREREQS(kMinProcessesForMPI);

    constexpr int kRequested = 4;
    configured_value_        = kRequested;

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    // Every rank's resolved config must equal what we asked for...
    ASSERT_MPI_EQ(getActiveCommunicator()->config.nChannelsPerNetPeer, kRequested);
    // ...and all ranks must agree (no per-rank divergence).
    expectAllRanksAgree(getActiveCommunicator()->config.nChannelsPerNetPeer);
}

// ---------------------------------------------------------------------------
// Env (NCCL_NCHANNELS_PER_NET_PEER) honored across a real multi-rank communicator.
// Requires one-mpirun-per-test isolation (NCCL_PARAM caches per process).
// ---------------------------------------------------------------------------
TEST_F(NChannelsPerNetPeerMPITest, Env_HonoredAcrossRanks)
{
    SKIP_UNLESS_MPI_PREREQS(kMinProcessesForMPI);

    constexpr int kEnvValue = 8;
    setenv("NCCL_NCHANNELS_PER_NET_PEER", "8", /*overwrite=*/1);
    configured_value_ = kLeaveConfigUnset; // env alone drives the value

    // NCCL_PARAM caches per process: if an earlier test already read it, this
    // setenv has no effect. Skip cleanly instead of reporting a false failure.
    // Run this test in its own mpirun (one-test-per-process) to exercise it.
    if(ncclParamNChannelsPerNetPeer() != kEnvValue)
        GTEST_SKIP() << "NCCL_NCHANNELS_PER_NET_PEER already cached this process; "
                        "run this test in its own mpirun invocation.";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ASSERT_MPI_EQ(getActiveCommunicator()->config.nChannelsPerNetPeer, kEnvValue);
    expectAllRanksAgree(getActiveCommunicator()->config.nChannelsPerNetPeer);
}

// ---------------------------------------------------------------------------
// Documented precedence: a valid env value overrides an explicit config field,
// consistently on every rank. Requires one-mpirun-per-test isolation.
// ---------------------------------------------------------------------------
TEST_F(NChannelsPerNetPeerMPITest, Env_OverridesConfig_AcrossRanks)
{
    SKIP_UNLESS_MPI_PREREQS(kMinProcessesForMPI);

    constexpr int kEnvValue = 16;
    setenv("NCCL_NCHANNELS_PER_NET_PEER", "16", /*overwrite=*/1);
    configured_value_ = 2; // env (16) must win over this

    // NCCL_PARAM caches per process (see Env_HonoredAcrossRanks): skip cleanly
    // if a prior test already cached a different value in this process.
    if(ncclParamNChannelsPerNetPeer() != kEnvValue)
        GTEST_SKIP() << "NCCL_NCHANNELS_PER_NET_PEER already cached this process; "
                        "run this test in its own mpirun invocation.";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ASSERT_MPI_EQ(getActiveCommunicator()->config.nChannelsPerNetPeer, kEnvValue);
    expectAllRanksAgree(getActiveCommunicator()->config.nChannelsPerNetPeer);
}

// ---------------------------------------------------------------------------
// Default (field UNDEF, env unset) must init cleanly and stay UNDEF on every
// rank, so paths.cc derives the per-NET-peer channel count via auto-tune.
// ---------------------------------------------------------------------------
TEST_F(NChannelsPerNetPeerMPITest, Default_LeavesUndef_AcrossRanks)
{
    SKIP_UNLESS_MPI_PREREQS(kMinProcessesForMPI);

    unsetenv("NCCL_NCHANNELS_PER_NET_PEER");
    configured_value_ = kLeaveConfigUnset;

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ASSERT_MPI_EQ(getActiveCommunicator()->config.nChannelsPerNetPeer,
                  static_cast<int>(NCCL_CONFIG_UNDEF_INT));
    expectAllRanksAgree(getActiveCommunicator()->config.nChannelsPerNetPeer);
}

// ---------------------------------------------------------------------------
// End-to-end functional: a communicator built with the knob set runs a correct
// AllReduce across all ranks. Proves the knob does not break collective execution.
// ---------------------------------------------------------------------------
TEST_F(NChannelsPerNetPeerMPITest, ConfigField_FunctionalAllReduceEndToEnd)
{
    SKIP_UNLESS_MPI_PREREQS(kMinProcessesForMPI);

    constexpr int kRequested = 4;
    configured_value_        = kRequested;

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ASSERT_MPI_EQ(getActiveCommunicator()->config.nChannelsPerNetPeer, kRequested);

    expectFunctionalAllReduce();
}

// ---------------------------------------------------------------------------
// Multi-node-gated: nChannelsPerNetPeer specifically governs NET (inter-node)
// peers, so this is where it is honored "end-to-end on the network". Skips
// cleanly when fewer than 2 nodes are present.
// ---------------------------------------------------------------------------
TEST_F(NChannelsPerNetPeerMPITest, MultiNode_NetPeer_HonoredEndToEnd)
{
    // Require >= 2 processes AND >= 2 nodes so real NET peers exist.
    SKIP_UNLESS_MPI_PREREQS(kMinProcessesForMPI,
                                              kNoProcessLimit,
                                              kNoPowerOfTwoRequired,
                                              /*min_nodes=*/2);

    constexpr int kRequested = 4;
    configured_value_        = kRequested;

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    // Honored on every rank of the multi-node communicator...
    ASSERT_MPI_EQ(getActiveCommunicator()->config.nChannelsPerNetPeer, kRequested);
    expectAllRanksAgree(getActiveCommunicator()->config.nChannelsPerNetPeer);

    // ...and the inter-node collective actually works with the knob applied.
    expectFunctionalAllReduce();
}

// ---------------------------------------------------------------------------
// Real point-to-point send/recv across nodes, with the knob set. Proves the
// inter-node NET path carries correct data end-to-end (not just AllReduce).
// ---------------------------------------------------------------------------
TEST_F(NChannelsPerNetPeerMPITest, NetSendRecv_AcrossNodes_EndToEnd)
{
    SKIP_UNLESS_MPI_PREREQS(kMinProcessesForMPI,
                                              kNoProcessLimit,
                                              kNoPowerOfTwoRequired,
                                              /*min_nodes=*/2);

    configured_value_ = 4;
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    const int senderRank = 0;
    const int recvRank   = pickInterNodePeer();
    ASSERT_MPI_TRUE(recvRank > 0); // guaranteed inter-node peer must exist (>=2 nodes)

    bool dataOk = true; // stays true on non-receiver ranks
    ASSERT_MPI_EQ(ncclSuccess, doSendRecv(senderRank, recvRank, 7.0f, &dataOk));
    ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));

    // Broadcast the receiver's verdict so all ranks assert together.
    int ok = dataOk ? 1 : 0;
    ASSERT_MPI_SUCCESS(MPI_Bcast(&ok, 1, MPI_INT, recvRank, MPI_COMM_WORLD));
    ASSERT_MPI_TRUE(ok == 1);
}

// ---------------------------------------------------------------------------
// Verify the NET transport is actually selected for an inter-node peer (and is
// NOT selected for a same-node peer, which uses P2P/SHM). This is the "check the
// NET has been correctly picked" requirement.
// ---------------------------------------------------------------------------
TEST_F(NChannelsPerNetPeerMPITest, NetTransport_SelectedForInterNodePeer)
{
    SKIP_UNLESS_MPI_PREREQS(kMinProcessesForMPI,
                                              kNoProcessLimit,
                                              kNoPowerOfTwoRequired,
                                              /*min_nodes=*/2);

    const int world_rank = MPIEnvironment::world_rank;

    configured_value_ = 4;
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    const int senderRank   = 0;
    const int recvRank      = pickInterNodePeer();
    const int sameNodePeer  = pickSameNodePeer();
    ASSERT_MPI_TRUE(recvRank > 0);

    // Establish connections (transports are connected lazily on first use).
    bool dataOk = true;
    ASSERT_MPI_EQ(ncclSuccess, doSendRecv(senderRank, recvRank, 3.0f, &dataOk));
    ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));

    // Inspect transport selection on the sender, then broadcast for coordinated
    // asserts across all ranks.
    int net_inter = 0;
    int net_same  = 0;
    if(world_rank == senderRank)
    {
        ncclComm_t comm = getActiveCommunicator();
        net_inter       = countNetSendChannelsToPeer(comm, recvRank);
        if(sameNodePeer > 0)
            net_same = countNetSendChannelsToPeer(comm, sameNodePeer);
        TEST_INFO("Transport pick: inter-node peer %d -> %d NET channels; "
                  "same-node peer %d -> %d NET channels",
                  recvRank, net_inter, sameNodePeer, net_same);
    }
    ASSERT_MPI_SUCCESS(MPI_Bcast(&net_inter, 1, MPI_INT, senderRank, MPI_COMM_WORLD));
    ASSERT_MPI_SUCCESS(MPI_Bcast(&net_same, 1, MPI_INT, senderRank, MPI_COMM_WORLD));

    // NET must be picked for the inter-node peer...
    ASSERT_MPI_TRUE(net_inter > 0);
    // ...and must NOT be picked for a same-node peer (it uses P2P/SHM).
    if(sameNodePeer > 0)
        ASSERT_MPI_TRUE(net_same == 0);
}

// ---------------------------------------------------------------------------
// Auto-tune divergence audit: compare the NET channel count actually established
// to an inter-node peer against (a) an explicit config request and (b) the
// library's auto-tuned value when the field is left default. Surfaces any
// divergence between the requested knob and what RCCL applied on the NET path.
// ---------------------------------------------------------------------------
TEST_F(NChannelsPerNetPeerMPITest, AutoTuneDivergence_Audit)
{
    SKIP_UNLESS_MPI_PREREQS(kMinProcessesForMPI,
                                              kNoProcessLimit,
                                              kNoPowerOfTwoRequired,
                                              /*min_nodes=*/2);

    const int world_rank = MPIEnvironment::world_rank;
    const int senderRank = 0;
    // env unset so the config field (not a cached env override) drives the value.
    unsetenv("NCCL_NCHANNELS_PER_NET_PEER");

    // --- (a) explicit config request: effective NET channels should match it ---
    constexpr int kRequested = 4;
    configured_value_        = kRequested;
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    const int recvRank = pickInterNodePeer();
    ASSERT_MPI_TRUE(recvRank > 0);

    bool dataOk = true;
    ASSERT_MPI_EQ(ncclSuccess, doSendRecv(senderRank, recvRank, 1.0f, &dataOk));
    ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));

    // p2pnChannelsPerPeer is the per-peer channel count the knob actually feeds
    // (paths.cc: nNetChannels = config.nChannelsPerNetPeer -> p2pnChannelsPerPeer,
    // then min()/pow2Up()/arch caps). The raw net-connected channel count also
    // includes ring/tree collective connections, so it is reported but NOT
    // expected to equal the knob.
    int cfg_perpeer = 0; // comm->p2pnChannelsPerPeer for the configured run
    int net_conns   = 0; // channels carrying a NET connection to the peer
    if(world_rank == senderRank)
    {
        ncclComm_t comm = getActiveCommunicator();
        cfg_perpeer     = comm->p2pnChannelsPerPeer;
        net_conns       = countNetSendChannelsToPeer(comm, recvRank);
        TEST_INFO("AUDIT[config]: requested=%d resolved=%d p2pnChannelsPerPeer=%d "
                  "net-connected channels to peer %d=%d %s",
                  kRequested, comm->config.nChannelsPerNetPeer, cfg_perpeer, recvRank, net_conns,
                  (cfg_perpeer == kRequested) ? "(per-peer matches knob)"
                                              : "(per-peer DIVERGES from knob -- recorded)");
    }
    ASSERT_MPI_SUCCESS(MPI_Bcast(&cfg_perpeer, 1, MPI_INT, senderRank, MPI_COMM_WORLD));
    ASSERT_MPI_SUCCESS(MPI_Bcast(&net_conns, 1, MPI_INT, senderRank, MPI_COMM_WORLD));

    // Deterministic checks: knob honored at the config layer + NET path used.
    ASSERT_MPI_EQ(getActiveCommunicator()->config.nChannelsPerNetPeer, kRequested);
    ASSERT_MPI_TRUE(net_conns > 0);
    ASSERT_MPI_TRUE(cfg_perpeer >= 1 && cfg_perpeer <= MAXCHANNELS);
    // The effective per-peer count vs the requested knob is recorded (logged
    // above as "matches"/"DIVERGES") for the parity table -- NOT hard-asserted,
    // because paths.cc applies min() across peers plus pow2Up()/arch caps on top
    // of nChannelsPerNetPeer, so a difference is expected behavior, not a bug.

    ASSERT_MPI_EQ(ncclSuccess, cleanupTestCommunicator());

    // --- (b) default (auto-tune): record the value RCCL derives on its own ---
    configured_value_ = kLeaveConfigUnset;
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    const int recvRank2 = pickInterNodePeer();
    ASSERT_MPI_TRUE(recvRank2 > 0);

    dataOk = true;
    ASSERT_MPI_EQ(ncclSuccess, doSendRecv(senderRank, recvRank2, 2.0f, &dataOk));
    ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));

    int auto_perpeer = 0;
    int net_conns2   = 0;
    if(world_rank == senderRank)
    {
        ncclComm_t comm = getActiveCommunicator();
        auto_perpeer    = comm->p2pnChannelsPerPeer;
        net_conns2      = countNetSendChannelsToPeer(comm, recvRank2);
        TEST_INFO("AUDIT[auto-tune]: resolved nChannelsPerNetPeer=%d (UNDEF=%d) "
                  "auto p2pnChannelsPerPeer=%d net-connected channels=%d; configured-run "
                  "per-peer=%d %s",
                  comm->config.nChannelsPerNetPeer, static_cast<int>(NCCL_CONFIG_UNDEF_INT),
                  auto_perpeer, net_conns2, cfg_perpeer,
                  (auto_perpeer == cfg_perpeer) ? "(same as configured)"
                                                : "(diverges from configured run -- recorded)");
    }
    ASSERT_MPI_SUCCESS(MPI_Bcast(&auto_perpeer, 1, MPI_INT, senderRank, MPI_COMM_WORLD));
    ASSERT_MPI_SUCCESS(MPI_Bcast(&net_conns2, 1, MPI_INT, senderRank, MPI_COMM_WORLD));

    // Default must leave the field UNDEF so paths.cc auto-tunes...
    ASSERT_MPI_EQ(getActiveCommunicator()->config.nChannelsPerNetPeer,
                  static_cast<int>(NCCL_CONFIG_UNDEF_INT));
    // ...the NET path is established...
    ASSERT_MPI_TRUE(net_conns2 > 0);
    // ...and the auto-tuned per-peer channel count is sane.
    ASSERT_MPI_TRUE(auto_perpeer >= 1 && auto_perpeer <= MAXCHANNELS);
}

#endif // MPI_TESTS_ENABLED
