/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file MPITestBase.hpp
 * @brief Base class infrastructure for MPI-based RCCL testing
 *
 * Provides a common test base class for writing multi-process distributed tests
 * using MPI and RCCL. Handles communicator creation, process validation, and
 * resource cleanup automatically.
 *
 * @see MPITestBase for the main base class
 * @see MPIEnvironment for global MPI setup
 */

#ifndef MPI_TEST_BASE_HPP
#define MPI_TEST_BASE_HPP

#include "MPITestCore.hpp"
#include "gtest/gtest.h"

#ifdef MPI_TESTS_ENABLED
#include "MPIEnvironment.hpp"
#include "TestChecks.hpp"
#include "rccl/rccl.h"
#include "utils.h" // For getHostName() from RCCL
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <string>

/**
 * @class MPITestBase
 * @brief Google Test adapter for MPI tests
 *
 * Integrates MPITestCore with Google Test framework for seamless MPI testing.
 * Inherits from both ::testing::Test (for GTest integration) and MPITestCore
 * (for MPI/RCCL functionality).
 *
 * **Features:**
 * - Process count validation (minimum processes, power-of-two requirements)
 * - Node count validation (single-node vs multi-node)
 * - Test-specific RCCL communicator creation and management
 * - HIP stream management for each test
 * - Automatic resource cleanup via GTest TearDown
 *
 * **Usage Example:**
 * @code
 * class MyMPITest : public MPITestBase {};
 *
 * TEST_F(MyMPITest, BasicAllReduce) {
 *   if (!validateTestPrerequisites(2)) {
 *     GTEST_SKIP() << "Need at least 2 processes";
 *   }
 *   ASSERT_EQ(ncclSuccess, createTestCommunicator());
 *
 *   ncclComm_t comm = getActiveCommunicator();
 *   hipStream_t stream = getActiveStream();
 *
 *   // Your test logic here...
 *   // Cleanup happens automatically in TearDown()
 * }
 * @endcode
 *
 * @note For standalone tests without GTest, use MPIStandaloneTest instead
 * @see MPITestCore for the base framework-agnostic functionality
 * @see MPIEnvironment for global MPI initialization
 */
/**
 * @brief Google Test adapter for MPI tests
 *
 * Integrates MPITestCore with Google Test framework by inheriting from both
 * ::testing::Test and MPITestCore.
 *
 * @note For standalone tests (without GTest), use MPIStandaloneTest instead
 */
class MPITestBase
    : public ::testing::Test
    , public MPITestCore
{
public:
    /**
   * @brief Google Test SetUp hook - initializes test resources
   *
   * Automatically called before each test runs. Calls initializeTest()
   * from MPITestCore for any custom initialization.
   *
   * @note No ambiguity with MPITestCore::initializeTest() - different names
   */
    void SetUp() override
    {
        initializeTest();
    }

    /**
   * @brief Google Test TearDown hook - ensures cleanup of test resources
   *
   * Automatically called after each test completes. Calls cleanupTest()
   * from MPITestCore to ensure proper resource cleanup.
   */
    void TearDown() override
    {
        cleanupTest();
    }
};

#endif // MPI_TESTS_ENABLED

#endif // MPI_TEST_BASE_HPP
