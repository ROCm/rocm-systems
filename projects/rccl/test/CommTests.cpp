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

#include <fstream>
#include <sstream>
#include <unistd.h>

using namespace MPITestConstants;

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

        // Initialize communicator with config
        RCCL_TEST_CHECK(ncclCommInitRankConfig(&test_comm_, world_size, nccl_id_, world_rank, &config));

        // Create HIP stream
        HIPCHECK(hipStreamCreate(&test_stream_));

        MPI_Barrier(MPI_COMM_WORLD);

        if(world_rank == 0)
        {
            TEST_INFO("Test-specific communicator created successfully");
        }

        return ncclSuccess;
    }
};

/**
 * @test TrafficClassMPITest.ConfiguredTrafficClass
 * @brief Verify traffic class configuration is stored in communicator and logged by NET plugin
 *
 * Sets trafficClass=46 via ncclConfig_t and verifies:
 * 1. comm->config.trafficClass stores the value
 * 2. NET plugin logs "NET: Configured trafficClass=46" (requires NCCL_DEBUG=INFO)
 */
TEST_F(TrafficClassMPITest, ConfiguredTrafficClass)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    constexpr int kTestTrafficClass = 46;
    configured_traffic_class_ = kTestTrafficClass;

    // Capture stderr to verify NET plugin log
    char tmpfile[] = "/tmp/rccl_tc_XXXXXX";
    int fd = mkstemp(tmpfile);
    int saved = dup(STDERR_FILENO);
    dup2(fd, STDERR_FILENO);

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    fflush(stderr);
    dup2(saved, STDERR_FILENO);
    close(saved);
    close(fd);

    // Verify trafficClass in communicator
    ASSERT_EQ(getActiveCommunicator()->config.trafficClass, kTestTrafficClass);

    // Verify NET plugin logged the trafficClass
    std::ifstream log(tmpfile);
    std::stringstream buf;
    buf << log.rdbuf();
    unlink(tmpfile);

    ASSERT_NE(buf.str().find("Traffic class set to 46"), std::string::npos)
        << "Traffic class log not found. Set NCCL_DEBUG=INFO.";
}

#endif // MPI_TESTS_ENABLED




