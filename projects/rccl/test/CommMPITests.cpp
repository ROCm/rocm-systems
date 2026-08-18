/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef MPI_TESTS_ENABLED

#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "TestChecks.hpp"
#include "ResourceGuards.hpp"

#include "nccl_device.h"
#include "comm.h"

#include <array>
#include <cstdlib>
#include <regex>
#include <string>
#include <vector>

using namespace MPITestConstants;
using namespace RCCLTestGuards;

/**
 * @class ConfigCommMPITestBase
 * @brief Shared fixture that builds the test communicator via
 *        ncclCommInitRankConfig() with RAII cleanup. Subclasses fill in the
 *        ncclConfig_t fields they want to exercise by overriding applyConfig().
 */
class ConfigCommMPITestBase : public MPITestBase
{
protected:
    virtual void applyConfig(ncclConfig_t& config) = 0;

    // Human-readable description of the config under test, for diagnostic logs.
    virtual std::string configLabel() const = 0;

    ncclResult_t createTestCommunicator() override
    {
        int world_rank = MPIEnvironment::world_rank;
        int world_size = MPIEnvironment::world_size;

        if(world_rank == 0)
        {
            TEST_INFO("Creating test-specific communicator with %s", configLabel().c_str());
        }

        // Rank 0 generates unique ID
        if(world_rank == 0)
        {
            RCCL_TEST_CHECK(ncclGetUniqueId(&nccl_id_));
        }

        // Broadcast ID to all ranks
        MPI_Bcast(&nccl_id_, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Let the subclass populate the ncclConfig_t fields under test.
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        applyConfig(config);

        // Initialize NCCL communicator with automatic cleanup on error
        RCCL_TEST_CHECK(ncclGroupStart());

        // RAII guard: Automatically calls ncclGroupEnd() if subsequent operations fail
        auto group_guard = makeScopeGuard([]() { (void)ncclGroupEnd(); });

        RCCL_TEST_CHECK(ncclCommInitRankConfig(&test_comm_, world_size, nccl_id_, world_rank, &config));

        // RAII guard: Automatically destroys test_comm_ if subsequent operations fail
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
        group_guard.dismiss(); // ncclGroupEnd succeeded, don't call it again

        // Create HIP stream - if this fails, comm_guard automatically cleans up test_comm_
        HIP_TEST_CHECK(hipStreamCreate(&test_stream_));

        // RAII guard: Automatically destroys test_stream_ if subsequent operations fail
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

        // All succeeded - dismiss guards to keep resources
        comm_guard.dismiss();
        stream_guard.dismiss();

        if(world_rank == 0)
        {
            TEST_INFO("Test-specific communicator created successfully");
        }

        return ncclSuccess;
    }
};

class PatLazyInitMPITest : public MPITestBase
{};

TEST_F(PatLazyInitMPITest, DefersConnectionUntilFirstCollective)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(/*min_processes=*/4));

    MPI_Comm local_comm;
    ASSERT_MPI_EQ(MPI_SUCCESS,
                  MPI_Comm_split_type(
                      MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm));
    int local_size = 0;
    ASSERT_MPI_EQ(MPI_SUCCESS, MPI_Comm_size(local_comm, &local_size));
    ASSERT_MPI_EQ(MPI_SUCCESS, MPI_Comm_free(&local_comm));
    if(local_size != 1)
    {
        GTEST_SKIP() << "PAT requires exactly one MPI rank per node";
    }

    MPIHelpers::MpiEnvGuard debug("NCCL_DEBUG", "INFO");
    MPIHelpers::MpiEnvGuard debug_subsys("NCCL_DEBUG_SUBSYS", "INIT");
    MPIHelpers::MpiEnvGuard pat_enable("NCCL_PAT_ENABLE", "1");
    MPIHelpers::MpiEnvGuard pat_lazy("NCCL_PAT_LAZY_INIT", "1");
    MPIHelpers::MpiEnvGuard algorithm("NCCL_ALGO", "PAT");
    MPIHelpers::MpiEnvGuard protocol("NCCL_PROTO", "SIMPLE");
    MPIHelpers::MpiEnvGuard cumem("NCCL_CUMEM_ENABLE", "0");
    MPIHelpers::TestLogAssertionContext log_ctx(
        MPIHelpers::makeNcclDebugFileAssertionOptions(getTestMpiRank()));

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();
    ASSERT_MPI_TRUE(comm != nullptr);
    ASSERT_MPI_FALSE(comm->initAlgoChannels[NCCL_ALGO_PAT]);

    const std::string init_log = log_ctx.readNcclDebugLog();
    ASSERT_MPI_TRUE(init_log.find("PAT lazy init enabled") != std::string::npos);
    ASSERT_MPI_TRUE(init_log.find("Connected binomial trees") == std::string::npos);

    void* send_buffer = nullptr;
    void* recv_buffer = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&send_buffer, sizeof(uint8_t)));
    auto send_guard = makeDeviceBufferAutoGuard(send_buffer);
    ASSERT_MPI_EQ(hipSuccess,
                  hipMalloc(&recv_buffer, sizeof(uint8_t) * MPIEnvironment::world_size));
    auto recv_guard = makeDeviceBufferAutoGuard(recv_buffer);

    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllGather(send_buffer,
                                recv_buffer,
                                1,
                                ncclUint8,
                                comm,
                                getActiveStream()));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    ASSERT_MPI_TRUE(comm->initAlgoChannels[NCCL_ALGO_PAT]);
    const std::string collective_log = log_ctx.readNcclDebugLog();
    ASSERT_MPI_TRUE(collective_log.find("Connected binomial trees") != std::string::npos);
}

// The environment every PAT test below needs, bundled so each test spends one line on it.
// Hierarchical AllGather has to be off or it intercepts ncclAllGather before PAT sees it, which
// would let a PAT test pass without ever running PAT.
struct PatTestEnv
{
    MPIHelpers::MpiEnvGuard pat_enable{"NCCL_PAT_ENABLE", "1"};
    MPIHelpers::MpiEnvGuard pat_lazy{"NCCL_PAT_LAZY_INIT", "1"};
    MPIHelpers::MpiEnvGuard algorithm{"NCCL_ALGO", "PAT"};
    MPIHelpers::MpiEnvGuard protocol{"NCCL_PROTO", "SIMPLE"};
    MPIHelpers::MpiEnvGuard cumem{"NCCL_CUMEM_ENABLE", "0"};
    MPIHelpers::MpiEnvGuard hag{"RCCL_HIERARCHICAL_ALLGATHER", "0"};
};

class PatSharedConnectionMPITest : public MPITestBase
{
protected:
    // Returns the decision rather than skipping here, because GTEST_SKIP() in a helper does not
    // stop the calling test.
    bool oneRankPerNode()
    {
        MPI_Comm local_comm;
        if(MPI_Comm_split_type(
               MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm) != MPI_SUCCESS)
        {
            return false;
        }
        int local_size = 0;
        MPI_Comm_size(local_comm, &local_size);
        MPI_Comm_free(&local_comm);
        return local_size == 1;
    }

    // Null means the topology can run PAT. Otherwise a skip reason for the caller to GTEST_SKIP().
    const char* patSkipReason()
    {
        if(!validateTestPrerequisites(/*min_processes=*/4))
        {
            return "PAT sharing tests need at least 4 MPI ranks";
        }
        if(!oneRankPerNode())
        {
            return "PAT requires exactly one MPI rank per node";
        }
        return nullptr;
    }
};

/**
 * ReduceScatter and AllGather address the same binomial neighbors, so PAT builds one
 * connection per neighbor and direction instead of a mirrored pair. Lazy init defers that
 * work to the first PAT collective, which lets the test separate the PAT connections from
 * the ring and tree connections already established during ncclCommInitRank.
 *
 * Below four ranks the mask set is closed under mask -> nranks-mask, so both connect passes
 * target the same peers and there is nothing to distinguish.
 *
 * RCCL_PARAM caches into a process-lifetime static, so the first communicator built in this
 * binary fixes RCCL_PAT_SHARED_QPS for every later one. This test therefore skips rather than
 * fails when the kill switch was set before launch, and its counterpart below does the reverse.
 */
TEST_F(PatSharedConnectionMPITest, AllGatherReusesReduceScatterConnections)
{
    if(const char* why = patSkipReason())
    {
        GTEST_SKIP() << why;
    }

    PatTestEnv env;

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();
    ASSERT_MPI_TRUE(comm != nullptr);
    if(!comm->patSharedQps)
    {
        GTEST_SKIP() << "RCCL_PAT_SHARED_QPS is disabled for this process, so PAT builds the "
                        "mirrored connection set; run without it to exercise sharing";
    }
    ASSERT_MPI_FALSE(comm->initAlgoChannels[NCCL_ALGO_PAT]);

    const int rank = comm->rank;
    const int nranks = comm->nRanks;
    const int nchannels = comm->nChannels;

    std::vector<int> send_before(static_cast<size_t>(nchannels) * nranks, 0);
    std::vector<int> recv_before(static_cast<size_t>(nchannels) * nranks, 0);
    for(int channel = 0; channel < nchannels; ++channel)
    {
        for(int peer = 0; peer < nranks; ++peer)
        {
            const size_t index = static_cast<size_t>(channel) * nranks + peer;
            send_before[index] = comm->channels[channel].peers[peer]->send[0].connected;
            recv_before[index] = comm->channels[channel].peers[peer]->recv[0].connected;
        }
    }

    void* send_buffer = nullptr;
    void* recv_buffer = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&send_buffer, sizeof(uint8_t)));
    auto send_guard = makeDeviceBufferAutoGuard(send_buffer);
    ASSERT_MPI_EQ(hipSuccess,
                  hipMalloc(&recv_buffer, sizeof(uint8_t) * MPIEnvironment::world_size));
    auto recv_guard = makeDeviceBufferAutoGuard(recv_buffer);

    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllGather(send_buffer,
                                recv_buffer,
                                1,
                                ncclUint8,
                                comm,
                                getActiveStream()));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    ASSERT_MPI_TRUE(comm->initAlgoChannels[NCCL_ALGO_PAT]);

    // The connections PAT is allowed to add, i.e. ReduceScatter's directions. The two-pass
    // connect also built the mirror of every entry, which is what this test rules out.
    std::vector<char> expect_recv(nranks, 0);
    std::vector<char> expect_send(nranks, 0);
    for(int mask = 1; mask < nranks; mask <<= 1)
    {
        expect_recv[(rank - mask + nranks) % nranks] = 1;
        expect_send[(rank + mask) % nranks] = 1;
    }

    for(int channel = 0; channel < nchannels; ++channel)
    {
        for(int peer = 0; peer < nranks; ++peer)
        {
            const size_t index = static_cast<size_t>(channel) * nranks + peer;
            const int recv_now = comm->channels[channel].peers[peer]->recv[0].connected;
            const int send_now = comm->channels[channel].peers[peer]->send[0].connected;

            if(expect_recv[peer])
            {
                ASSERT_MPI_TRUE(recv_now);
            }
            else
            {
                ASSERT_MPI_EQ(recv_before[index], recv_now);
            }

            if(expect_send[peer])
            {
                ASSERT_MPI_TRUE(send_now);
            }
            else
            {
                ASSERT_MPI_EQ(send_before[index], send_now);
            }
        }
    }
}

/**
 * RCCL_PAT_SHARED_QPS=0 restores the mirrored connection set AllGather used before sharing.
 * Asserting the mirror is present keeps that fallback path from decaying unnoticed, and the
 * AllGather payload check covers the device primitives and PatAGAlgorithm still agreeing with
 * the proxy on peer direction when sharing is off.
 *
 * The MpiEnvGuard below only takes effect when this test builds the first communicator in the
 * process, because RCCL_PARAM caches. Launch with RCCL_PAT_SHARED_QPS=0 already in the
 * environment to run it alongside other communicator tests; otherwise it skips.
 */
TEST_F(PatSharedConnectionMPITest, SeparateConnectionsWhenSharingDisabled)
{
    if(const char* why = patSkipReason())
    {
        GTEST_SKIP() << why;
    }

    PatTestEnv              env;
    MPIHelpers::MpiEnvGuard pat_shared("RCCL_PAT_SHARED_QPS", "0");

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();
    ASSERT_MPI_TRUE(comm != nullptr);
    if(comm->patSharedQps)
    {
        GTEST_SKIP() << "RCCL_PAT_SHARED_QPS was already cached as enabled by an earlier "
                        "communicator in this process; set it to 0 before launch to run this test";
    }

    const int rank = comm->rank;
    const int nranks = comm->nRanks;
    const int nchannels = comm->nChannels;

    const uint8_t expected_byte = static_cast<uint8_t>(rank);
    void* send_buffer = nullptr;
    void* recv_buffer = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&send_buffer, sizeof(uint8_t)));
    auto send_guard = makeDeviceBufferAutoGuard(send_buffer);
    ASSERT_MPI_EQ(hipSuccess,
                  hipMalloc(&recv_buffer, sizeof(uint8_t) * MPIEnvironment::world_size));
    auto recv_guard = makeDeviceBufferAutoGuard(recv_buffer);
    ASSERT_MPI_EQ(hipSuccess,
                  hipMemcpy(send_buffer, &expected_byte, sizeof(uint8_t), hipMemcpyHostToDevice));

    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllGather(send_buffer,
                                recv_buffer,
                                1,
                                ncclUint8,
                                comm,
                                getActiveStream()));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    std::vector<uint8_t> gathered(static_cast<size_t>(nranks), 0);
    ASSERT_MPI_EQ(hipSuccess,
                  hipMemcpy(gathered.data(),
                            recv_buffer,
                            sizeof(uint8_t) * nranks,
                            hipMemcpyDeviceToHost));
    for(int peer = 0; peer < nranks; ++peer)
    {
        ASSERT_MPI_EQ(static_cast<uint8_t>(peer), gathered[static_cast<size_t>(peer)]);
    }

    // Without sharing, every mask connects both its own direction and the mirror.
    for(int mask = 1; mask < nranks; mask <<= 1)
    {
        const int next_peer = (rank - mask + nranks) % nranks;
        const int prev_peer = (rank + mask) % nranks;
        for(int channel = 0; channel < nchannels; ++channel)
        {
            ASSERT_MPI_TRUE(comm->channels[channel].peers[next_peer]->recv[0].connected);
            ASSERT_MPI_TRUE(comm->channels[channel].peers[next_peer]->send[0].connected);
            ASSERT_MPI_TRUE(comm->channels[channel].peers[prev_peer]->recv[0].connected);
            ASSERT_MPI_TRUE(comm->channels[channel].peers[prev_peer]->send[0].connected);
        }
    }
}

/**
 * Sharing puts ReduceScatter and AllGather on the same connections, so they now also share that
 * connection's ring buffer and its persistent step counter (ncclConnInfo::step). A disagreement
 * between the two about how many steps an operation consumes cannot show up while only one of
 * them runs; it shows up on whichever collective runs next, and small drift only after several
 * rounds. Hence one communicator, both collectives, repeatedly.
 *
 * Lazy init makes the ordering meaningful: the first ReduceScatter builds the PAT connections and
 * every AllGather afterwards has to be satisfied by what that single shared pass created.
 *
 * The two are chained, so the pair is an AllReduce: every rank must end each iteration holding
 * nranks times its input. The intermediate ReduceScatter block and final AllGather vector are
 * validated independently. Values are distinct across the whole vector, so a wrong data-block
 * index surfaces as a permutation rather than as garbage, and they change every iteration, so a
 * collective that silently did nothing surfaces as stale values.
 *
 * Runs in both modes. Under RCCL_PAT_SHARED_QPS=0 it validates the fallback path unchanged.
 */
TEST_F(PatSharedConnectionMPITest, AlternatingReduceScatterAndAllGatherOnOneCommunicator)
{
    if(const char* why = patSkipReason())
    {
        GTEST_SKIP() << why;
    }

    PatTestEnv env;

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();
    ASSERT_MPI_TRUE(comm != nullptr);
    ASSERT_MPI_FALSE(comm->initAlgoChannels[NCCL_ALGO_PAT]);

    // Doubles because the check below needs nranks * value to be exact, and the values have to
    // stay distinct across a buffer far larger than float can index exactly.
    constexpr int    kIterations   = 16;
    constexpr size_t kCountPerRank = 65536;
    const size_t     nranks        = static_cast<size_t>(comm->nRanks);
    const size_t     total         = kCountPerRank * nranks;
    const size_t     rank_offset   = static_cast<size_t>(comm->rank) * kCountPerRank;

    void* whole = nullptr;
    void* part  = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&whole, total * sizeof(double)));
    auto whole_guard = makeDeviceBufferAutoGuard(whole);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&part, kCountPerRank * sizeof(double)));
    auto part_guard = makeDeviceBufferAutoGuard(part);

    std::vector<double> input(total);
    std::vector<double> reduce_scatter_result(kCountPerRank);
    std::vector<double> all_gather_result(total);

    // Only prints under NCCL_DEBUG=INFO; it is how a log can confirm the shape actually run.
    TEST_INFO("alternating RS+AG: %d iterations, %zu ranks, %zu KiB/collective",
              kIterations,
              nranks,
              total * sizeof(double) / 1024);

    for(int iter = 0; iter < kIterations; ++iter)
    {
        SCOPED_TRACE("iteration " + std::to_string(iter));

        for(size_t i = 0; i < total; ++i)
        {
            input[i] = static_cast<double>(static_cast<size_t>(iter) * total + i + 1);
        }
        ASSERT_MPI_EQ(
            hipSuccess,
            hipMemcpy(whole, input.data(), total * sizeof(double), hipMemcpyHostToDevice));

        ASSERT_MPI_EQ(ncclSuccess,
                      ncclReduceScatter(whole,
                                        part,
                                        kCountPerRank,
                                        ncclDouble,
                                        ncclSum,
                                        comm,
                                        getActiveStream()));
        if(iter == 0)
        {
            // Enqueue, not execution, does the lazy connect, so this holds without a sync and
            // proves the AllGather below runs on connections ReduceScatter created.
            ASSERT_MPI_TRUE(comm->initAlgoChannels[NCCL_ALGO_PAT]);
        }
        ASSERT_MPI_EQ(
            ncclSuccess,
            ncclAllGather(part, whole, kCountPerRank, ncclDouble, comm, getActiveStream()));
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

        ASSERT_MPI_EQ(
            hipSuccess,
            hipMemcpy(reduce_scatter_result.data(),
                      part,
                      kCountPerRank * sizeof(double),
                      hipMemcpyDeviceToHost));
        ASSERT_MPI_EQ(
            hipSuccess,
            hipMemcpy(all_gather_result.data(),
                      whole,
                      total * sizeof(double),
                      hipMemcpyDeviceToHost));

        // Counted rather than asserted per element: ASSERT_MPI_EQ costs an MPI_Allreduce.
        size_t reduce_scatter_mismatches = 0;
        for(size_t i = 0; i < kCountPerRank; ++i)
        {
            const double expected = input[rank_offset + i] * static_cast<double>(nranks);
            if(reduce_scatter_result[i] != expected)
            {
                ++reduce_scatter_mismatches;
            }
        }
        ASSERT_MPI_EQ(size_t{0}, reduce_scatter_mismatches);

        size_t all_gather_mismatches = 0;
        for(size_t i = 0; i < total; ++i)
        {
            if(all_gather_result[i] != input[i] * static_cast<double>(nranks))
            {
                ++all_gather_mismatches;
            }
        }
        ASSERT_MPI_EQ(size_t{0}, all_gather_mismatches);
    }
}

/**
 * Alternating covers sequential launches. Sharing also lets ReduceScatter and AllGather
 * queue proxy ops onto the same connection and step counter inside one ncclGroupStart/End,
 * before either kernel runs. That is a different interleaving than back-to-back launches.
 *
 * The two collectives use independent buffers so a group launch cannot race a producer-consumer
 * dependency. Values are distinct so a wrong data-block index still shows up as a permutation.
 * Runs in both sharing modes.
 */
TEST_F(PatSharedConnectionMPITest, GroupedReduceScatterAndAllGatherOnOneCommunicator)
{
    if(const char* why = patSkipReason())
    {
        GTEST_SKIP() << why;
    }

    PatTestEnv env;

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();
    ASSERT_MPI_TRUE(comm != nullptr);
    ASSERT_MPI_FALSE(comm->initAlgoChannels[NCCL_ALGO_PAT]);

    constexpr int    kIterations   = 8;
    constexpr size_t kCountPerRank = 1024;
    const size_t     nranks        = static_cast<size_t>(comm->nRanks);
    const size_t     total         = kCountPerRank * nranks;
    const size_t     rank_offset   = static_cast<size_t>(comm->rank) * kCountPerRank;

    void* rs_send = nullptr;
    void* rs_recv = nullptr;
    void* ag_send = nullptr;
    void* ag_recv = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&rs_send, total * sizeof(double)));
    auto rs_send_guard = makeDeviceBufferAutoGuard(rs_send);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&rs_recv, kCountPerRank * sizeof(double)));
    auto rs_recv_guard = makeDeviceBufferAutoGuard(rs_recv);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&ag_send, kCountPerRank * sizeof(double)));
    auto ag_send_guard = makeDeviceBufferAutoGuard(ag_send);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&ag_recv, total * sizeof(double)));
    auto ag_recv_guard = makeDeviceBufferAutoGuard(ag_recv);

    std::vector<double> rs_input(total);
    std::vector<double> rs_result(kCountPerRank);
    std::vector<double> ag_input(kCountPerRank);
    std::vector<double> ag_result(total);

    TEST_INFO("grouped RS+AG: %d iterations, %zu ranks, %zu KiB/collective",
              kIterations,
              nranks,
              total * sizeof(double) / 1024);

    for(int iter = 0; iter < kIterations; ++iter)
    {
        SCOPED_TRACE("iteration " + std::to_string(iter));

        for(size_t i = 0; i < total; ++i)
        {
            rs_input[i] = static_cast<double>(static_cast<size_t>(iter) * total + i + 1);
        }
        for(size_t i = 0; i < kCountPerRank; ++i)
        {
            ag_input[i] = static_cast<double>(
                (static_cast<size_t>(iter) + 1) * total + rank_offset + i + 1);
        }
        ASSERT_MPI_EQ(
            hipSuccess,
            hipMemcpy(rs_send, rs_input.data(), total * sizeof(double), hipMemcpyHostToDevice));
        ASSERT_MPI_EQ(
            hipSuccess,
            hipMemcpy(ag_send, ag_input.data(), kCountPerRank * sizeof(double), hipMemcpyHostToDevice));

        ASSERT_MPI_EQ(ncclSuccess, ncclGroupStart());
        ASSERT_MPI_EQ(ncclSuccess,
                      ncclReduceScatter(rs_send,
                                        rs_recv,
                                        kCountPerRank,
                                        ncclDouble,
                                        ncclSum,
                                        comm,
                                        getActiveStream()));
        ASSERT_MPI_EQ(
            ncclSuccess,
            ncclAllGather(ag_send, ag_recv, kCountPerRank, ncclDouble, comm, getActiveStream()));
        ASSERT_MPI_EQ(ncclSuccess, ncclGroupEnd());
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

        if(iter == 0)
        {
            ASSERT_MPI_TRUE(comm->initAlgoChannels[NCCL_ALGO_PAT]);
        }

        ASSERT_MPI_EQ(
            hipSuccess,
            hipMemcpy(rs_result.data(),
                      rs_recv,
                      kCountPerRank * sizeof(double),
                      hipMemcpyDeviceToHost));
        ASSERT_MPI_EQ(
            hipSuccess,
            hipMemcpy(ag_result.data(), ag_recv, total * sizeof(double), hipMemcpyDeviceToHost));

        size_t rs_mismatches = 0;
        for(size_t i = 0; i < kCountPerRank; ++i)
        {
            const double expected = rs_input[rank_offset + i] * static_cast<double>(nranks);
            if(rs_result[i] != expected)
            {
                ++rs_mismatches;
            }
        }
        ASSERT_MPI_EQ(size_t{0}, rs_mismatches);

        size_t ag_mismatches = 0;
        for(int peer = 0; peer < comm->nRanks; ++peer)
        {
            for(size_t i = 0; i < kCountPerRank; ++i)
            {
                const double expected = static_cast<double>(
                    (static_cast<size_t>(iter) + 1) * total
                    + static_cast<size_t>(peer) * kCountPerRank + i + 1);
                if(ag_result[static_cast<size_t>(peer) * kCountPerRank + i] != expected)
                {
                    ++ag_mismatches;
                }
            }
        }
        ASSERT_MPI_EQ(size_t{0}, ag_mismatches);
    }
}

/**
 * @class TrafficClassMPITest
 * @brief Test fixture for Traffic Class (QoS) configuration via ncclConfig_t.
 */
class TrafficClassMPITest : public ConfigCommMPITestBase
{
protected:
    int configured_traffic_class_ = NCCL_CONFIG_UNDEF_INT;

    void applyConfig(ncclConfig_t& config) override
    {
        config.trafficClass = configured_traffic_class_;
    }

    std::string configLabel() const override
    {
        return "trafficClass=" + std::to_string(configured_traffic_class_);
    }
};

/**
 * @test TrafficClassMPITest.ConfiguredTrafficClass
 * @brief Verify traffic class in communicator and in NCCL debug output
 *
 * Uses MPIHelpers::TestLogAssertionContext with makeCombinedAssertionLogOptions():
 * - Sets NCCL_DEBUG_FILE for this scope (before communicator init) for NCCL-native logs.
 * - Optionally matches the same line in rccl_test_rank_<r>.log when
 *   RCCL_MPI_LOG_ALL_RANKS=1 (stderr/tee). Either sink may contain the substring.
 *
 * Requires NCCL_DEBUG=INFO (or higher) for the log line to exist.
 */
TEST_F(TrafficClassMPITest, ConfiguredTrafficClass)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    constexpr int kTestTrafficClass = 46;
    configured_traffic_class_ = kTestTrafficClass;

    MPIHelpers::TestLogAssertionContext log_ctx(
        MPIHelpers::makeCombinedAssertionLogOptions(getTestMpiRank()));

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    // Verify trafficClass in communicator
    ASSERT_MPI_EQ(getActiveCommunicator()->config.trafficClass, kTestTrafficClass);

    static constexpr const char* kTrafficClassLogNeedle = "Traffic class set to 46";
    const std::string            from_nccl               = log_ctx.readNcclDebugLog();
    const std::string            from_rank_log           = log_ctx.readPerRankStderrLog();
    const bool hit_nccl   = from_nccl.find(kTrafficClassLogNeedle) != std::string::npos;
    const bool hit_stderr = from_rank_log.find(kTrafficClassLogNeedle) != std::string::npos;
    const bool found_line = hit_nccl || hit_stderr;

    if(getTestMpiRank() == 0)
    {
        TEST_INFO("Expected NCCL log line \"%s\": %s",
                  kTrafficClassLogNeedle,
                  found_line ? "passed" : "failed");
    }

    ASSERT_MPI_TRUE(found_line);
}

// Every layer of the traffic-class precedence chain gets a distinct value, so a
// regression reports which layer won instead of aliasing with the value it was
// supposed to override.
constexpr int kHostCommTrafficClass   = 3;   // ncclConfig_t::trafficClass
constexpr int kDeviceCommTrafficClass = 7;   // ncclDevCommRequirements::ginTrafficClass
constexpr int kEnvServiceLevel        = 5;   // NCCL_IB_SL
constexpr int kEnvTrafficClass        = 96;  // NCCL_IB_TC

// ibv_link_layer::IBV_LINK_LAYER_ETHERNET. On native InfiniBand the generic
// traffic-class value programs SL while the GRH traffic-class field stays zero;
// on RoCE both fields are observable.
constexpr int kIbvLinkLayerEthernet      = 2;
constexpr int kInfiniBandGrhTrafficClass = 0;

// getEnvParam() fallback: no valid service level or traffic class is negative.
constexpr int kEnvParamUnset = -1;

// One rank on each of exactly two nodes. The QP oracle reads the leading GIN
// queue pairs of a single cross-node connection, so extra ranks would add
// connections whose attributes it does not expect.
constexpr int kTrafficClassRanks = 2;
constexpr int kTrafficClassNodes = 2;

struct GinQpTrafficClass
{
    int link_layer;
    int service_level;
    int traffic_class;
};

struct GinTrafficClassCapture
{
    ncclResult_t create_result{ncclSuccess};
    ncclResult_t destroy_result{ncclSuccess};
    bool          proxy_handles{true};
    int           connection_count{0};
    bool          saw_create_marker{false};
    std::vector<GinQpTrafficClass> qps;
};

class GinTrafficClassMPITest : public TrafficClassMPITest
{
protected:
    static bool envParamIsUnset(const char* name)
    {
        const char* value = std::getenv(name);
        return value == nullptr || value[0] == '\0';
    }

    static bool proxyPrerequisitesMet()
    {
        const char* gin_type = std::getenv("NCCL_GIN_TYPE");
        const char* cumem    = std::getenv("NCCL_CUMEM_ENABLE");
        return gin_type != nullptr && std::string(gin_type) == "2"
            && cumem != nullptr && std::string(cumem) == "1";
    }

    static std::array<int, 2> collectiveBoolSummary(bool value)
    {
        int minimum = value ? 1 : 0;
        int maximum = minimum;
        MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        return {minimum, maximum};
    }

    GinTrafficClassCapture captureGinQpTrafficClass(
        int requested_traffic_class,
        const MPIHelpers::TestLogAssertionContext& log_ctx)
    {
        GinTrafficClassCapture capture;
        const std::string before = log_ctx.readNcclDebugLog();

        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        reqs.ginConnectionType       = NCCL_GIN_CONNECTION_FULL;
        reqs.ginContextCount         = 1;
        reqs.ginSignalCount          = 1;
        reqs.ginTrafficClass         = requested_traffic_class;

        ncclDevComm dev_comm{};
        capture.create_result = ncclDevCommCreate(getActiveCommunicator(), &reqs, &dev_comm);
        if(capture.create_result != ncclSuccess) return capture;

        capture.connection_count = dev_comm.ginConnectionCount;
        if(capture.connection_count <= 0) capture.proxy_handles = false;
        for(int connection = 0; connection < dev_comm.ginConnectionCount; ++connection)
        {
            capture.proxy_handles =
                capture.proxy_handles
                && dev_comm.ginNetDeviceTypes[connection] == NCCL_NET_DEVICE_GIN_PROXY;
        }

        // The log is sampled immediately before and after ncclDevCommCreate, so
        // the appended text covers only this device communicator. The existing
        // devCommCreate marker narrows it further to the last GIN setup.
        const std::string after = log_ctx.readNcclDebugLog();
        const std::string appended =
            after.size() >= before.size() ? after.substr(before.size()) : after;
        const std::string marker = "devCommCreate: creating";
        const size_t marker_pos = appended.rfind(marker);
        capture.saw_create_marker = marker_pos != std::string::npos;
        const std::string qp_log =
            capture.saw_create_marker ? appended.substr(marker_pos) : appended;

        const std::regex qp_pattern(
            R"(ncclIbQpRtr:.*ll=([0-9]+).*sl: ([0-9]+) tc: ([0-9]+))");
        for(std::sregex_iterator it(qp_log.begin(), qp_log.end(), qp_pattern), end;
            it != end;
            ++it)
        {
            capture.qps.push_back(
                {std::stoi((*it)[1].str()),
                 std::stoi((*it)[2].str()),
                 std::stoi((*it)[3].str())});
        }

        capture.destroy_result =
            ncclDevCommDestroy(getActiveCommunicator(), &dev_comm);
        return capture;
    }

    static bool qpsMatch(
        const std::vector<GinQpTrafficClass>& qps,
        int expected_service_level,
        int expected_roce_traffic_class)
    {
        for(const auto& qp : qps)
        {
            if(qp.service_level != expected_service_level)
            {
                fprintf(stderr,
                        "Unexpected GIN QP traffic class: ll=%d sl=%d tc=%d; "
                        "expected sl=%d\n",
                        qp.link_layer,
                        qp.service_level,
                        qp.traffic_class,
                        expected_service_level);
                return false;
            }
            const int expected_tc = qp.link_layer == kIbvLinkLayerEthernet
                                        ? expected_roce_traffic_class
                                        : kInfiniBandGrhTrafficClass;
            if(qp.traffic_class != expected_tc)
            {
                fprintf(stderr,
                        "Unexpected GIN QP traffic class: ll=%d sl=%d tc=%d; "
                        "expected tc=%d\n",
                        qp.link_layer,
                        qp.service_level,
                        qp.traffic_class,
                        expected_tc);
                return false;
            }
        }
        return !qps.empty();
    }

    static bool containsEthernetQp(const std::vector<GinQpTrafficClass>& qps)
    {
        for(const auto& qp : qps)
            if(qp.link_layer == kIbvLinkLayerEthernet) return true;
        return false;
    }

    // The GIN contexts are created first inside ncclDevCommCreate, so their queue
    // pairs lead the captured window. The ordinary transport connections that
    // follow in the same call always carry the host communicator value, so
    // compare only the leading run sharing one SL/TC pair.
    static std::vector<GinQpTrafficClass> leadingGinQps(
        const std::vector<GinQpTrafficClass>& qps)
    {
        std::vector<GinQpTrafficClass> leading;
        for(const auto& qp : qps)
        {
            if(!leading.empty()
               && (qp.service_level != leading.front().service_level
                   || qp.traffic_class != leading.front().traffic_class))
                break;
            leading.push_back(qp);
        }
        return leading;
    }
};

// Run this case in a fresh process with NCCL_IB_SL/NCCL_IB_TC unset. It
// observes the QP RTR attributes submitted to ibv_modify_qp, proving that a
// device-communicator traffic class overrides the host communicator value and
// that an unset device value falls back to it.
TEST_F(GinTrafficClassMPITest, DeviceHostPrecedence)
{
    const auto proxy_prerequisites = collectiveBoolSummary(proxyPrerequisitesMet());
    if(proxy_prerequisites[0] == 0 && proxy_prerequisites[1] == 0)
        GTEST_SKIP() << "Requires NCCL_GIN_TYPE=2 and NCCL_CUMEM_ENABLE=1";
    ASSERT_MPI_TRUE(proxy_prerequisites[0] == 1 && proxy_prerequisites[1] == 1);

    const bool local_ib_env_unset =
        envParamIsUnset("NCCL_IB_SL") && envParamIsUnset("NCCL_IB_TC");
    const auto ib_env_unset = collectiveBoolSummary(local_ib_env_unset);
    if(ib_env_unset[0] == 0 && ib_env_unset[1] == 0)
        GTEST_SKIP() << "Run with NCCL_IB_SL and NCCL_IB_TC unset";
    ASSERT_MPI_TRUE(ib_env_unset[0] == 1 && ib_env_unset[1] == 1);

    if(!validateTestPrerequisites(/*min_processes=*/kTrafficClassRanks,
                                  /*max_processes=*/kTrafficClassRanks,
                                  /*require_power_of_two=*/false,
                                  /*min_nodes=*/kTrafficClassNodes,
                                  /*max_nodes=*/kTrafficClassNodes))
        GTEST_SKIP() << "Requires exactly " << kTrafficClassRanks << " ranks on "
                     << kTrafficClassNodes << " nodes";

    auto reset_debug =
        makeScopeGuard([]() { MPIHelpers::resetNcclDebugState(); });
    MPIHelpers::MpiEnvGuard debug("NCCL_DEBUG", "TRACE");
    MPIHelpers::MpiEnvGuard debug_subsys("NCCL_DEBUG_SUBSYS", "INIT,NET");
    MPIHelpers::resetNcclDebugState();
    MPIHelpers::TestLogAssertionContext log_ctx(
        MPIHelpers::makeNcclDebugFileAssertionOptions(getTestMpiRank()));

    configured_traffic_class_ = kHostCommTrafficClass;
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    GinTrafficClassCapture device =
        captureGinQpTrafficClass(kDeviceCommTrafficClass, log_ctx);
    ASSERT_MPI_EQ(ncclSuccess, device.create_result);
    ASSERT_MPI_EQ(ncclSuccess, device.destroy_result);
    ASSERT_MPI_GT(device.connection_count, 0);
    ASSERT_MPI_TRUE(device.proxy_handles);
    const bool local_trace_unavailable =
        device.saw_create_marker && device.qps.empty();
    const auto trace_unavailable = collectiveBoolSummary(local_trace_unavailable);
    if(trace_unavailable[0] == 1 && trace_unavailable[1] == 1)
        GTEST_SKIP() << "RCCL was built without TRACE=ON; QP RTR attributes are unavailable";
    const bool local_trace_ready = device.saw_create_marker && !device.qps.empty();
    const auto trace_ready = collectiveBoolSummary(local_trace_ready);
    ASSERT_MPI_TRUE(trace_ready[0] == 1 && trace_ready[1] == 1);
    ASSERT_MPI_TRUE(qpsMatch(leadingGinQps(device.qps),
                             /*expected_service_level=*/kDeviceCommTrafficClass,
                             /*expected_roce_traffic_class=*/kDeviceCommTrafficClass));

    // The IB context is mutable and reused, so observing the host value here
    // also proves the previous device value did not persist.
    GinTrafficClassCapture host =
        captureGinQpTrafficClass(NCCL_CONFIG_UNDEF_INT, log_ctx);
    ASSERT_MPI_EQ(ncclSuccess, host.create_result);
    ASSERT_MPI_EQ(ncclSuccess, host.destroy_result);
    ASSERT_MPI_GT(host.connection_count, 0);
    ASSERT_MPI_TRUE(host.proxy_handles);
    ASSERT_MPI_TRUE(host.saw_create_marker);
    ASSERT_MPI_TRUE(qpsMatch(leadingGinQps(host.qps),
                             /*expected_service_level=*/kHostCommTrafficClass,
                             /*expected_roce_traffic_class=*/kHostCommTrafficClass));
}

// Run in a separate process with NCCL_IB_SL=5 and NCCL_IB_TC=96. Distinct
// values prove the two explicit IB settings independently override both the
// device-communicator value and the host communicator value.
TEST_F(GinTrafficClassMPITest, ExplicitIbEnvironmentOverrides)
{
    const auto proxy_prerequisites = collectiveBoolSummary(proxyPrerequisitesMet());
    if(proxy_prerequisites[0] == 0 && proxy_prerequisites[1] == 0)
        GTEST_SKIP() << "Requires NCCL_GIN_TYPE=2 and NCCL_CUMEM_ENABLE=1";
    ASSERT_MPI_TRUE(proxy_prerequisites[0] == 1 && proxy_prerequisites[1] == 1);

    const bool local_env_matches =
        MPIHelpers::getEnvParam<int>("NCCL_IB_SL", kEnvParamUnset) == kEnvServiceLevel
        && MPIHelpers::getEnvParam<int>("NCCL_IB_TC", kEnvParamUnset) == kEnvTrafficClass;
    const auto env_matches = collectiveBoolSummary(local_env_matches);
    if(env_matches[0] == 0 && env_matches[1] == 0)
        GTEST_SKIP() << "Run in a fresh process with NCCL_IB_SL=" << kEnvServiceLevel
                     << " and NCCL_IB_TC=" << kEnvTrafficClass;
    ASSERT_MPI_TRUE(env_matches[0] == 1 && env_matches[1] == 1);

    if(!validateTestPrerequisites(/*min_processes=*/kTrafficClassRanks,
                                  /*max_processes=*/kTrafficClassRanks,
                                  /*require_power_of_two=*/false,
                                  /*min_nodes=*/kTrafficClassNodes,
                                  /*max_nodes=*/kTrafficClassNodes))
        GTEST_SKIP() << "Requires exactly " << kTrafficClassRanks << " ranks on "
                     << kTrafficClassNodes << " nodes";

    auto reset_debug =
        makeScopeGuard([]() { MPIHelpers::resetNcclDebugState(); });
    MPIHelpers::MpiEnvGuard debug("NCCL_DEBUG", "TRACE");
    MPIHelpers::MpiEnvGuard debug_subsys("NCCL_DEBUG_SUBSYS", "INIT,NET");
    MPIHelpers::resetNcclDebugState();
    MPIHelpers::TestLogAssertionContext log_ctx(
        MPIHelpers::makeNcclDebugFileAssertionOptions(getTestMpiRank()));

    configured_traffic_class_ = kHostCommTrafficClass;
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    GinTrafficClassCapture capture =
        captureGinQpTrafficClass(kDeviceCommTrafficClass, log_ctx);
    ASSERT_MPI_EQ(ncclSuccess, capture.create_result);
    ASSERT_MPI_EQ(ncclSuccess, capture.destroy_result);
    ASSERT_MPI_GT(capture.connection_count, 0);
    ASSERT_MPI_TRUE(capture.proxy_handles);
    const bool local_trace_unavailable =
        capture.saw_create_marker && capture.qps.empty();
    const auto trace_unavailable = collectiveBoolSummary(local_trace_unavailable);
    if(trace_unavailable[0] == 1 && trace_unavailable[1] == 1)
        GTEST_SKIP() << "RCCL was built without TRACE=ON; QP RTR attributes are unavailable";
    const bool local_trace_ready = capture.saw_create_marker && !capture.qps.empty();
    const auto trace_ready = collectiveBoolSummary(local_trace_ready);
    ASSERT_MPI_TRUE(trace_ready[0] == 1 && trace_ready[1] == 1);
    const std::vector<GinQpTrafficClass> gin_qps = leadingGinQps(capture.qps);
    ASSERT_MPI_TRUE(containsEthernetQp(gin_qps));
    ASSERT_MPI_TRUE(qpsMatch(gin_qps,
                             /*expected_service_level=*/kEnvServiceLevel,
                             /*expected_roce_traffic_class=*/kEnvTrafficClass));
}

/**
 * @class CtaConfigMPITest
 * @brief Fixture for the CTA override paths on AMD GPUs. Injects minCTAs/maxCTAs
 *        through ncclCommInitRankConfig() and inspects the resulting comm->config
 *        and comm->nChannels.
 */
class CtaConfigMPITest : public ConfigCommMPITestBase
{
protected:
    int configured_min_ctas_ = NCCL_CONFIG_UNDEF_INT;
    int configured_max_ctas_ = NCCL_CONFIG_UNDEF_INT;

    void applyConfig(ncclConfig_t& config) override
    {
        config.minCTAs = configured_min_ctas_;
        config.maxCTAs = configured_max_ctas_;
    }

    std::string configLabel() const override
    {
        return "minCTAs=" + std::to_string(configured_min_ctas_)
             + " maxCTAs=" + std::to_string(configured_max_ctas_);
    }
};

/**
 * @test CtaConfigMPITest.ConfigOverrideAppliesMinMaxCTAs
 * @brief ncclConfig_t minCTAs/maxCTAs land in comm->config and clamp
 *        comm->nChannels into [minCTAs, maxCTAs].
 */
TEST_F(CtaConfigMPITest, ConfigOverrideAppliesMinMaxCTAs)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    constexpr int kMinCTAs = 2;
    constexpr int kMaxCTAs = 4;
    configured_min_ctas_   = kMinCTAs;
    configured_max_ctas_   = kMaxCTAs;

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();
    ASSERT_MPI_TRUE(comm != nullptr);

    // Config-override path: ncclConfig_t values are accepted into comm->config.
    ASSERT_MPI_EQ(comm->config.minCTAs, kMinCTAs);
    ASSERT_MPI_EQ(comm->config.maxCTAs, kMaxCTAs);

    ASSERT_MPI_TRUE(comm->nChannels >= kMinCTAs);
    ASSERT_MPI_TRUE(comm->nChannels <= kMaxCTAs);

    if(getTestMpiRank() == 0)
    {
        TEST_INFO("minCTAs=%d maxCTAs=%d -> nChannels=%d",
                  comm->config.minCTAs,
                  comm->config.maxCTAs,
                  comm->nChannels);
    }
}

/**
 * @test CtaConfigMPITest.EnvKnobsApplyMinMaxCTAs
 * @brief NCCL_MIN_CTAS / NCCL_MAX_CTAS env knobs override comm->config; skips when unset.
 */
TEST_F(CtaConfigMPITest, EnvKnobsApplyMinMaxCTAs)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    const char* min_env = std::getenv("NCCL_MIN_CTAS");
    const char* max_env = std::getenv("NCCL_MAX_CTAS");
    if(min_env == nullptr && max_env == nullptr)
    {
        GTEST_SKIP() << "NCCL_MIN_CTAS / NCCL_MAX_CTAS not set; run under the CTA env CI tier.";
    }

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();
    ASSERT_MPI_TRUE(comm != nullptr);

    // Env-knob path: NCCL_MIN_CTAS / NCCL_MAX_CTAS override comm->config.
    if(min_env != nullptr)
    {
        const int expected_min = std::atoi(min_env);
        ASSERT_MPI_EQ(comm->config.minCTAs, expected_min);
        ASSERT_MPI_TRUE(comm->nChannels >= expected_min);
    }
    if(max_env != nullptr)
    {
        const int expected_max = std::atoi(max_env);
        ASSERT_MPI_EQ(comm->config.maxCTAs, expected_max);
        ASSERT_MPI_TRUE(comm->nChannels <= expected_max);
    }

    if(getTestMpiRank() == 0)
    {
        TEST_INFO("env NCCL_MIN_CTAS=%s NCCL_MAX_CTAS=%s -> config min=%d max=%d nChannels=%d",
                  min_env ? min_env : "(unset)",
                  max_env ? max_env : "(unset)",
                  comm->config.minCTAs,
                  comm->config.maxCTAs,
                  comm->nChannels);
    }
}

/**
 * @test CtaConfigMPITest.EnvKnobsIgnoreNonPositiveCTAs
 * @brief Negative test: env values <= 0 are rejected and config resolves to its positive default; skips when no knob is non-positive.
 */
TEST_F(CtaConfigMPITest, EnvKnobsIgnoreNonPositiveCTAs)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    const char* min_env     = std::getenv("NCCL_MIN_CTAS");
    const char* max_env     = std::getenv("NCCL_MAX_CTAS");
    const bool  min_nonpos  = (min_env != nullptr && std::atoi(min_env) <= 0);
    const bool  max_nonpos  = (max_env != nullptr && std::atoi(max_env) <= 0);
    if(!min_nonpos && !max_nonpos)
    {
        GTEST_SKIP() << "Negative CTA env test: set NCCL_MIN_CTAS and/or NCCL_MAX_CTAS <= 0.";
    }

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();
    ASSERT_MPI_TRUE(comm != nullptr);

    if(min_nonpos)
    {
        ASSERT_MPI_TRUE(comm->config.minCTAs > 0);
    }
    if(max_nonpos)
    {
        ASSERT_MPI_TRUE(comm->config.maxCTAs > 0);
    }

    // Comm still initialized with a usable channel count (not clamped to 0).
    ASSERT_MPI_TRUE(comm->nChannels >= 1);

    if(getTestMpiRank() == 0)
    {
        TEST_INFO("rejected non-positive CTAs -> config min=%d max=%d nChannels=%d",
                  comm->config.minCTAs,
                  comm->config.maxCTAs,
                  comm->nChannels);
    }
}

/**
 * @test CtaConfigMPITest.DefaultConfigDoesNotClampChannels
 * @brief Guard: with no min/maxCTAs (config or env), the maxCTAs clamp is a no-op and does not reduce nChannels; skips when the env knobs are set.
 */
TEST_F(CtaConfigMPITest, DefaultConfigDoesNotClampChannels)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    if(std::getenv("NCCL_MIN_CTAS") != nullptr || std::getenv("NCCL_MAX_CTAS") != nullptr)
    {
        GTEST_SKIP() << "NCCL_MIN_CTAS / NCCL_MAX_CTAS set; this test validates the default (unset) path.";
    }

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();
    ASSERT_MPI_TRUE(comm != nullptr);

    // Default maxCTAs (MAXCHANNELS) must be >= nChannels, i.e. the cap did not reduce the channel count.
    ASSERT_MPI_TRUE(comm->nChannels >= 1);
    ASSERT_MPI_TRUE(comm->config.maxCTAs >= comm->nChannels);

    if(getTestMpiRank() == 0)
    {
        TEST_INFO("default config -> maxCTAs=%d nChannels=%d (cap is a no-op)",
                  comm->config.maxCTAs,
                  comm->nChannels);
    }
}

namespace
{
/**
 * @brief True if any single line of `log` contains both `needle_a` and `needle_b`.
 *
 * Used to disambiguate log substrings that appear verbatim in more than one
 * INFO call site (e.g. "- Destroy COMPLETE" is emitted by both commFree() and
 * COLLTRACE): a plain std::string::find() on the combined log would be
 * satisfied by either line, so same-line co-occurrence is required instead.
 */
bool logHasLineContainingBoth(const std::string& log, const char* needle_a, const char* needle_b)
{
    std::size_t line_start = 0;
    while(line_start <= log.size())
    {
        const std::size_t line_end = log.find('\n', line_start);
        const std::size_t line_len = (line_end == std::string::npos) ? std::string::npos : (line_end - line_start);
        const std::string line     = log.substr(line_start, line_len);
        if(line.find(needle_a) != std::string::npos && line.find(needle_b) != std::string::npos)
        {
            return true;
        }
        if(line_end == std::string::npos)
        {
            break;
        }
        line_start = line_end + 1;
    }
    return false;
}
} // namespace

/**
 * @class DestroySubsysMPITest
 * @brief Regression coverage for the NCCL 2.30.3 log-volume-reduction cherry-pick
 *        (upstream sync PR #6837 brought in NCCL_DESTROY and retagged the shared
 *        comm-destroy/plugin-unload call sites; this covers the RCCL-only call
 *        sites the sync did not touch: COLLTRACE's destroy-time INFO lines).
 *
 * NCCL_DESTROY is not part of the default NCCL_DEBUG_SUBSYS mask
 * (NCCL_INIT | NCCL_BOOTSTRAP | NCCL_ENV), so destroy/teardown INFO lines
 * disappear from plain `NCCL_DEBUG=INFO` output while remaining reachable via
 * `NCCL_DEBUG_SUBSYS=DESTROY` (or ALL). These tests use the
 * "comm ... - Destroy COMPLETE" line emitted unconditionally from commFree()
 * (src/init.cc) as the marker, since it fires on every plain ncclCommDestroy().
 */
class DestroySubsysMPITest : public MPITestBase {};

/**
 * @test DestroySubsysMPITest.DefaultSubsys_ExcludesDestroyNoise
 * @brief Under the default subsystem mask, destroy-time INFO noise must be absent.
 */
TEST_F(DestroySubsysMPITest, DefaultSubsys_ExcludesDestroyNoise)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    MPIHelpers::MpiEnvGuard debugGuard("NCCL_DEBUG", "INFO");
    // Force NCCL_DEBUG_SUBSYS unset so RCCL falls back to its default mask,
    // regardless of any ambient value left by CI or a previous test.
    MPIHelpers::MpiEnvUnsetGuard subsysGuard("NCCL_DEBUG_SUBSYS");
    // Enable the RCCL-only COLLTRACE latency profiler so ncclCommInitRank*()
    // actually constructs a CollTrace instance and its destroy-time INFO
    // lines get emitted. Without this, collTraceInit() (gated on this env
    // var) never runs and this test would only exercise the shared
    // commFree() marker, not the COLLTRACE-specific retag this PR fixes.
    MPIHelpers::MpiEnvGuard latencyProfilerGuard("RCCL_LATENCY_PROFILER", "1");
    MPIHelpers::resetNcclDebugState();

    MPIHelpers::TestLogAssertionContext log_ctx(
        MPIHelpers::makeCombinedAssertionLogOptions(getTestMpiRank()));

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    // Destroy while log_ctx is still alive so destroy-time output is captured
    // before TearDown() restores NCCL_DEBUG_FILE (and possibly unlinks the log).
    ASSERT_MPI_EQ(ncclSuccess, cleanupTestCommunicator());

    static constexpr const char* kCollTraceDestroyNeedle = "- Destroy START"; // unique to CollTrace::~CollTrace()
    const std::string            from_nccl               = log_ctx.readNcclDebugLog();
    const std::string            from_rank_log            = log_ctx.readPerRankStderrLog();
    const std::string            combined                 = from_nccl + from_rank_log;
    // "comm " (with trailing space) + "- Destroy COMPLETE" on the same line is
    // specific to commFree()'s "comm %p rank ... - %s COMPLETE" line; the
    // substring "- Destroy COMPLETE" alone would also match COLLTRACE's
    // "COLLTRACE: commHash ... - Destroy COMPLETE" line.
    const bool hit_comm_free = logHasLineContainingBoth(combined, "comm ", "- Destroy COMPLETE");
    const bool hit_colltrace = combined.find(kCollTraceDestroyNeedle) != std::string::npos;

    if(getTestMpiRank() == 0)
    {
        TEST_INFO("commFree() destroy marker under default subsys mask: %s",
                   hit_comm_free ? "unexpectedly present" : "correctly absent");
        TEST_INFO("COLLTRACE destroy marker under default subsys mask: %s",
                   hit_colltrace ? "unexpectedly present" : "correctly absent");
    }

    ASSERT_MPI_FALSE(hit_comm_free);
    ASSERT_MPI_FALSE(hit_colltrace);
}

/**
 * @test DestroySubsysMPITest.DestroySubsys_IncludesDestroyNoise
 * @brief With NCCL_DEBUG_SUBSYS=DESTROY, the same destroy-time INFO line must be visible.
 */
TEST_F(DestroySubsysMPITest, DestroySubsys_IncludesDestroyNoise)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    MPIHelpers::MpiEnvGuard debugGuard("NCCL_DEBUG", "INFO");
    MPIHelpers::MpiEnvGuard subsysGuard("NCCL_DEBUG_SUBSYS", "DESTROY");
    // See DefaultSubsys_ExcludesDestroyNoise: without this, CollTrace is never
    // constructed and this test would not exercise the COLLTRACE-specific retag.
    MPIHelpers::MpiEnvGuard latencyProfilerGuard("RCCL_LATENCY_PROFILER", "1");
    MPIHelpers::resetNcclDebugState();

    MPIHelpers::TestLogAssertionContext log_ctx(
        MPIHelpers::makeCombinedAssertionLogOptions(getTestMpiRank()));

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ASSERT_MPI_EQ(ncclSuccess, cleanupTestCommunicator());

    static constexpr const char* kCollTraceDestroyNeedle = "- Destroy START"; // unique to CollTrace::~CollTrace()
    const std::string            from_nccl               = log_ctx.readNcclDebugLog();
    const std::string            from_rank_log            = log_ctx.readPerRankStderrLog();
    const std::string            combined                 = from_nccl + from_rank_log;
    // "comm " (with trailing space) + "- Destroy COMPLETE" on the same line is
    // specific to commFree()'s "comm %p rank ... - %s COMPLETE" line; the
    // substring "- Destroy COMPLETE" alone would also match COLLTRACE's
    // "COLLTRACE: commHash ... - Destroy COMPLETE" line.
    const bool hit_comm_free = logHasLineContainingBoth(combined, "comm ", "- Destroy COMPLETE");
    const bool hit_colltrace = combined.find(kCollTraceDestroyNeedle) != std::string::npos;

    if(getTestMpiRank() == 0)
    {
        TEST_INFO("commFree() destroy marker under NCCL_DEBUG_SUBSYS=DESTROY: %s",
                   hit_comm_free ? "present" : "unexpectedly absent");
        TEST_INFO("COLLTRACE destroy marker under NCCL_DEBUG_SUBSYS=DESTROY: %s",
                   hit_colltrace ? "present" : "unexpectedly absent");
    }

    ASSERT_MPI_TRUE(hit_comm_free);
    ASSERT_MPI_TRUE(hit_colltrace);
}

/**
 * @class GinRmaContextMPITest
 * @brief Inspects the GIN and RMA plugin contexts bound at communicator init
 *        time. No data path is stood up, only the plugin state is read.
 */
class GinRmaContextMPITest : public ConfigCommMPITestBase
{
protected:
    // ncclCommInitChildComm() derives shareResources from the parent config, not
    // from the config handed to ncclCommSplit(), so splitShare has to be set here.
    void applyConfig(ncclConfig_t& config) override
    {
        config.splitShare = 1;
    }

    std::string configLabel() const override
    {
        return "splitShare=1";
    }

    struct ncclComm* ActiveComm()
    {
        return getActiveCommunicator();
    }
};

/**
 * @test GinRmaContextMPITest.RmaAndGinFinalizeWithSplitComm
 * @brief With both plugins initialized, RMA and GIN must hold separate
 *        contexts that a split child inherits and finalize can release.
 */
TEST_F(GinRmaContextMPITest, RmaAndGinFinalizeWithSplitComm)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    struct ncclComm* parent = ActiveComm();
    ASSERT_MPI_TRUE(parent != nullptr);
    ASSERT_MPI_TRUE(parent->sharedRes != nullptr);

    if(parent->sharedRes->ginState.ncclGin == nullptr || parent->rmaState.rmaProxyState.ncclRma == nullptr)
    {
        GTEST_SKIP() << "Requires both the GIN and RMA plugins enabled on this host";
    }

    // One internal backend serves both roles, so the contexts are distinct
    // only when RMA owns a separate field.
    ASSERT_MPI_TRUE(parent->ginContext != nullptr);
    ASSERT_MPI_TRUE(parent->rmaContext != nullptr);
    ASSERT_MPI_TRUE(parent->ginContext != parent->rmaContext);

    // The parent shares its resources, so the child inherits both contexts
    // instead of initializing the plugins again.
    ncclComm_t splitComm = nullptr;
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommSplit(getActiveCommunicator(), 0, getTestMpiRank(), &splitComm, nullptr));

    struct ncclComm* child = splitComm;
    ASSERT_MPI_TRUE(child->sharedRes == parent->sharedRes);
    ASSERT_MPI_TRUE(child->ginContext == parent->ginContext);
    ASSERT_MPI_TRUE(child->rmaContext == parent->rmaContext);

    // Destroy the parent first so the child holds the last reference and
    // finalizes both plugins against the contexts it inherited.
    ASSERT_MPI_EQ(ncclSuccess, cleanupTestCommunicator());
    ASSERT_MPI_EQ(ncclSuccess, ncclCommDestroy(splitComm));
}

/**
 * @class GraphStreamOrderingConfigMPITest
 * @brief Validates NCCL 2.30 graphStreamOrdering config parsing and compatibility
 *        rules (see NCCL_GRAPH_STREAM_ORDERING in env.rst).
 */
class GraphStreamOrderingConfigMPITest : public ConfigCommMPITestBase
{
protected:
    int configured_graph_stream_ordering_ = NCCL_CONFIG_UNDEF_INT;
    int configured_graph_usage_mode_      = NCCL_CONFIG_UNDEF_INT;

    void applyConfig(ncclConfig_t& config) override
    {
        config.graphStreamOrdering = configured_graph_stream_ordering_;
        config.graphUsageMode      = configured_graph_usage_mode_;
    }

    std::string configLabel() const override
    {
        return "graphStreamOrdering=" + std::to_string(configured_graph_stream_ordering_)
             + " graphUsageMode=" + std::to_string(configured_graph_usage_mode_);
    }

    static const char* graphStreamOrderingEnv()
    {
        return std::getenv("NCCL_GRAPH_STREAM_ORDERING");
    }

    static bool isGraphStreamOrderingEnvUnset()
    {
        return graphStreamOrderingEnv() == nullptr;
    }

    static bool isGraphStreamOrderingEnvSetAndValid()
    {
        const char* env = graphStreamOrderingEnv();
        if(env == nullptr)
        {
            return false;
        }
        return std::string(env) == "0" || std::string(env) == "1";
    }

    static bool isGraphStreamOrderingEnvUnsetOrZero()
    {
        const char* env = graphStreamOrderingEnv();
        if(env == nullptr)
        {
            return true;
        }
        return std::string(env) == "0";
    }
};

/**
 * @test GraphStreamOrderingConfigMPITest.ConfigOverrideAppliesGraphStreamOrdering
 * @brief ncclConfig_t graphStreamOrdering is stored on comm->config when env is unset.
 *
 * NCCL 2.30.7 upstream applies NCCL_GRAPH_STREAM_ORDERING after ncclConfig_t in
 * envConfigOverride(), so this test requires the env var to be absent.
 */
TEST_F(GraphStreamOrderingConfigMPITest, ConfigOverrideAppliesGraphStreamOrdering)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    ASSERT_TRUE(isGraphStreamOrderingEnvUnset())
        << "NCCL_GRAPH_STREAM_ORDERING must not be set; upstream NCCL envConfigOverride() overrides "
           "explicit ncclConfig_t graphStreamOrdering.";

    configured_graph_stream_ordering_ = 0;
    configured_graph_usage_mode_      = 1;

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();
    ASSERT_MPI_TRUE(comm != nullptr);
    ASSERT_MPI_EQ(comm->config.graphStreamOrdering, 0);
    ASSERT_MPI_EQ(comm->config.graphUsageMode, 1);
}

/**
 * @test GraphStreamOrderingConfigMPITest.IncompatibleMixingFallsBackToEnabledOrdering
 * @brief graphStreamOrdering=0 with graphUsageMode=2 is unsupported and falls back to 1.
 */
TEST_F(GraphStreamOrderingConfigMPITest, IncompatibleMixingFallsBackToEnabledOrdering)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    configured_graph_stream_ordering_ = 0;
    configured_graph_usage_mode_      = 2;

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();
    ASSERT_MPI_TRUE(comm != nullptr);
    ASSERT_MPI_EQ(comm->config.graphStreamOrdering, 1);
    ASSERT_MPI_EQ(comm->config.graphUsageMode, 2);
}

/**
 * @test GraphStreamOrderingConfigMPITest.EnvOverrideAppliesGraphStreamOrdering
 * @brief NCCL_GRAPH_STREAM_ORDERING overrides ncclConfig_t graphStreamOrdering at init.
 */
TEST_F(GraphStreamOrderingConfigMPITest, EnvOverrideAppliesGraphStreamOrdering)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    ASSERT_TRUE(isGraphStreamOrderingEnvSetAndValid())
        << "NCCL_GRAPH_STREAM_ORDERING must be set to 0 or 1";

    const int expected = std::atoi(graphStreamOrderingEnv());

    // Set config to the opposite value; upstream NCCL envConfigOverride() must win.
    configured_graph_stream_ordering_ = (expected == 0) ? 1 : 0;
    configured_graph_usage_mode_        = 1;

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();
    ASSERT_MPI_TRUE(comm != nullptr);
    ASSERT_MPI_EQ(comm->config.graphStreamOrdering, expected);
}

/**
 * @test GraphStreamOrderingConfigMPITest.DisabledOrderingGraphCaptureSmoke
 * @brief Smoke test: graphStreamOrdering=0 with graphUsageMode=1 can capture and replay AllReduce.
 *
 * Requires effective ordering 0. Skips when NCCL_GRAPH_STREAM_ORDERING is set to 1,
 * because upstream NCCL envConfigOverride() would override config graphStreamOrdering=0.
 */
TEST_F(GraphStreamOrderingConfigMPITest, DisabledOrderingGraphCaptureSmoke)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    ASSERT_TRUE(isGraphStreamOrderingEnvUnsetOrZero())
        << "NCCL_GRAPH_STREAM_ORDERING must be unset or 0; value 1 overrides config "
           "graphStreamOrdering=0 in upstream NCCL.";

    configured_graph_stream_ordering_ = 0;
    configured_graph_usage_mode_      = 1;

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();
    ASSERT_MPI_TRUE(comm != nullptr);
    ASSERT_MPI_EQ(comm->config.graphStreamOrdering, 0);

    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, sizeof(float)));
    auto sendGuard = makeDeviceBufferAutoGuard(sendBuf);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, sizeof(float)));
    auto recvGuard = makeDeviceBufferAutoGuard(recvBuf);

    const float sendVal = static_cast<float>(getTestMpiRank() + 1.0f);
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(sendBuf, &sendVal, sizeof(float), hipMemcpyHostToDevice));

    hipGraph_t graph = nullptr;
    hipGraphExec_t graphExec = nullptr;

    ASSERT_MPI_EQ(hipSuccess, hipStreamBeginCapture(getActiveStream(), hipStreamCaptureModeThreadLocal));
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllReduce(sendBuf, recvBuf, 1, ncclFloat, ncclSum, comm, getActiveStream()));
    ASSERT_MPI_EQ(hipSuccess, hipStreamEndCapture(getActiveStream(), &graph));
    ASSERT_MPI_NE(nullptr, graph);

    ASSERT_MPI_EQ(hipSuccess, hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    auto graphCleanup = makeScopeGuard([&]() {
        if(graphExec) (void)hipGraphExecDestroy(graphExec);
        if(graph) (void)hipGraphDestroy(graph);
    });

    ASSERT_MPI_EQ(hipSuccess, hipGraphLaunch(graphExec, getActiveStream()));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    float result = 0.0f;
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(&result, recvBuf, sizeof(float), hipMemcpyDeviceToHost));

    const int worldSize = MPIEnvironment::world_size;
    const float expected = static_cast<float>(worldSize * (worldSize + 1) / 2);
    ASSERT_NEAR(result, expected, 1e-3f);
}

#endif // MPI_TESTS_ENABLED
