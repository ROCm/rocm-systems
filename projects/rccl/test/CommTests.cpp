/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "gtest/gtest.h"
#include "comm.h"

namespace RcclUnitTesting
{
  TEST(CommTests, Sorter)
  {
	// Configuration
	ncclTaskCollSorter* me_ptr = new ncclTaskCollSorter;
	me_ptr->head = nullptr;

	ASSERT_EQ(ncclTaskCollSorterEmpty(me_ptr), true);
	delete me_ptr;
  }
}

#ifdef MPI_TESTS_ENABLED

#include "MPITestBase.hpp"
#include "TestChecks.hpp"
#include "ResourceGuards.hpp"

#include <fstream>
#include <sstream>
#include <unistd.h>

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
 * @brief Verify traffic class configuration is stored in communicator and logged
 *
 * Sets trafficClass=46 via ncclConfig_t and verifies:
 * 1. comm->config.trafficClass stores the value
 * 2. NCCL logs "Traffic class set to 46" (requires NCCL_DEBUG=INFO)
 */
TEST_F(TrafficClassMPITest, ConfiguredTrafficClass)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    constexpr int kTestTrafficClass = 46;
    configured_traffic_class_ = kTestTrafficClass;

    // Save and set NCCL_DEBUG_FILE to capture logs
    std::string logfile = "/tmp/rccl_tc_" + std::to_string(getpid()) + ".log";
    const char* prev_debug_file = getenv("NCCL_DEBUG_FILE");
    std::string saved_debug_file = prev_debug_file ? prev_debug_file : "";
    setenv("NCCL_DEBUG_FILE", logfile.c_str(), 1);

    // RAII guard to restore env var
    auto env_guard = makeScopeGuard([&]() {
        if(saved_debug_file.empty())
            unsetenv("NCCL_DEBUG_FILE");
        else
            setenv("NCCL_DEBUG_FILE", saved_debug_file.c_str(), 1);
    });

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    // Verify trafficClass in communicator
    ASSERT_MPI_EQ(getActiveCommunicator()->config.trafficClass, kTestTrafficClass);

    // Read the NCCL debug log file
    std::ifstream log(logfile);
    std::stringstream buf;
    buf << log.rdbuf();
    log.close();
    unlink(logfile.c_str());

    // Verify trafficClass appears in logs
    ASSERT_MPI_NE(buf.str().find("Traffic class set to 46"), std::string::npos)
        << "Traffic class log not found. Set NCCL_DEBUG=INFO.";
}

#endif // MPI_TESTS_ENABLED
