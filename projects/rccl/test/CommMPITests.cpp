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

#include <cstdlib>
#include <string>

using namespace MPITestConstants;
using namespace RCCLTestGuards;

/**
 * @class TrafficClassMPITest
 * @brief Test fixture for Traffic Class (QoS) configuration
 *
 * This test requires ncclCommInitRankConfig() to pass trafficClass,
 * so we override createTestCommunicator() to inject the config while
 * reusing base class members (test_comm_, test_stream_, nccl_id_).
 *
 * Pattern follows MPITestRunner.md "Example 8: Custom Test Class"
 */
class TrafficClassMPITest : public MPITestBase
{
protected:
    int configured_traffic_class_ = NCCL_CONFIG_UNDEF_INT;

    /**
     * @brief Override to use ncclCommInitRankConfig with trafficClass
     *
     * Follows same pattern as base createTestCommunicator() but uses
     * ncclCommInitRankConfig() to pass the trafficClass configuration.
     * Stores results in base class members for getActiveCommunicator()/getActiveStream().
     * Uses RAII guards for proper cleanup on failure.
     */
    ncclResult_t createTestCommunicator() override
    {
        int world_rank = MPIEnvironment::world_rank;
        int world_size = MPIEnvironment::world_size;

        if(world_rank == 0)
        {
            TEST_INFO("Creating test-specific communicator with trafficClass=%d",
                      configured_traffic_class_);
        }

        // Rank 0 generates unique ID
        if(world_rank == 0)
        {
            RCCL_TEST_CHECK(ncclGetUniqueId(&nccl_id_));
        }

        // Broadcast ID to all ranks
        MPI_Bcast(&nccl_id_, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Configure with traffic class
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.trafficClass = configured_traffic_class_;

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

#endif // MPI_TESTS_ENABLED
