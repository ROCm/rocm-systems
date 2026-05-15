/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file MPITestCore.hpp
 * @brief Framework-agnostic MPI test infrastructure
 *
 * Provides core MPI test functionality independent of any testing framework.
 * Can be used with Google Test, standalone tests, performance benchmarks, etc.
 *
 * @see MPITestBase for GTest integration
 * @see MPIStandaloneTest for standalone usage
 */

#ifndef MPI_TEST_CORE_HPP
#define MPI_TEST_CORE_HPP

#ifdef MPI_TESTS_ENABLED
#include "MPIEnvironment.hpp"
#include "rccl/rccl.h"
#include "utils.h" // For getHostName() from RCCL
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <string>

/**
 * @namespace MPITestConstants
 * @brief Constants and helper functions for MPI test configuration
 */
namespace MPITestConstants
{
/**
     * @brief Minimum number of processes typically required for MPI tests
     */
constexpr int kMinProcessesForMPI = 2;

/**
     * @brief Flag to indicate power-of-two process count is required
     */
constexpr bool kRequirePowerOfTwo = true;

/**
     * @brief Flag to indicate power-of-two process count is not required
     */
constexpr bool kNoPowerOfTwoRequired = false;

/**
     * @brief Value indicating no upper limit on process count
     */
constexpr int kNoProcessLimit = 0;

/**
     * @brief Value indicating single-node only execution required
     */
constexpr int kRequireSingleNode = 1;

/**
     * @brief Value indicating no node limit (multi-node capable)
     */
constexpr int kNoNodeLimit = 0;

/**
     * @brief Check if a number is a power of two
     * @param n The number to check
     * @return true if n is a power of two, false otherwise
     */
inline bool isPowerOfTwo(int n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

/**
     * @brief Detect the number of unique nodes in the MPI configuration
     * @return Number of unique nodes
     */
int detectNodeCount();

} // namespace MPITestConstants

/**
 * @class MPITestCore
 * @brief Framework-agnostic base class for MPI tests
 *
 * Provides core MPI test infrastructure without dependency on any testing framework.
 * Supports both GTest-based tests (via MPITestBase) and standalone tests.
 *
 * **Key Features:**
 * - Framework-agnostic design
 * - Automatic RCCL communicator management
 * - Process and node count validation
 * - HIP stream lifecycle management
 * - Clean resource cleanup
 *
 * **Usage:**
 * - For GTest: Use MPITestBase (inherits from MPITestCore)
 * - For Standalone: Use MPIStandaloneTest (inherits from MPITestCore)
 * - For Custom: Inherit from MPITestCore directly
 *
 * @par Example (Standalone):
 * @code
 * class MyPerfTest : public MPITestCore {
 * public:
 *     int run() {
 *         if (!validateTestPrerequisites(2)) {
 *             return 1; // Skip
 *         }
 *         if (createTestCommunicator() != ncclSuccess) {
 *             return 1; // Error
 *         }
 *         // Test logic...
 *         return 0; // Success
 *     }
 * };
 * @endcode
 */
class MPITestCore
{
protected:
    /**
   * @brief Test-specific NCCL communicator handle
   *
   * Created by createTestCommunicator(), destroyed in cleanup.
   * Access via getActiveCommunicator().
   */
    ncclComm_t test_comm_ = nullptr;

    /**
   * @brief Test-specific HIP stream handle
   *
   * Created with the communicator, destroyed in cleanup.
   * Access via getActiveStream().
   */
    hipStream_t test_stream_ = nullptr;

    /**
   * @brief NCCL unique ID for communicator initialization
   *
   * Generated on rank 0 and broadcast to all ranks.
   */
    ncclUniqueId nccl_id_ = {};

public:
    /**
   * @brief Virtual destructor for proper cleanup
   */
    virtual ~MPITestCore() = default;

    /**
   * @brief Validate test prerequisites (process count, node count)
   *
   * Checks if the current MPI environment meets the test's requirements.
   * Displays what the test requires and whether the environment satisfies those requirements.
   * Returns true if all requirements met, false otherwise.
   *
   * Parameters are organized by category:
   * - Process requirements: min_processes, max_processes, require_power_of_two
   * - Node requirements: min_nodes, max_nodes
   *
   * @param min_processes Minimum number of MPI processes required (default: 1)
   * @param max_processes Maximum number of MPI processes allowed (0 = no limit) (default: 0)
   * @param require_power_of_two If true, world size must be a power of 2 (default: false)
   * @param min_nodes Minimum number of nodes required (default: 1)
   * @param max_nodes Maximum number of nodes allowed (0 = no limit) (default: 0)
   *
   * @return true if all requirements are met, false otherwise
   */
    bool validateTestPrerequisites(int  min_processes        = 1,
                                   int  max_processes        = MPITestConstants::kNoProcessLimit,
                                   bool require_power_of_two = false,
                                   int  min_nodes            = 1,
                                   int  max_nodes            = MPITestConstants::kNoNodeLimit);

    /**
   * @brief Create a test-specific RCCL communicator and HIP stream
   *
   * Creates isolated RCCL communicator and HIP stream for this test.
   * Uses ncclGroupStart/End for proper initialization and MPI barriers
   * for synchronization across all ranks.
   *
   * @return ncclSuccess on success, or NCCL error code on failure
   *
   * @note This function is idempotent - calling it multiple times is safe
   * @note Communicator is automatically destroyed in cleanup
   */
    virtual ncclResult_t createTestCommunicator();

    /**
   * @brief Get the active NCCL communicator for this test
   *
   * Returns the test-specific communicator. Returns nullptr if createTestCommunicator()
   * has not been called first.
   *
   * @return The active NCCL communicator handle, or nullptr if not created
   *
   * @note Always call createTestCommunicator() before this method
   */
    virtual ncclComm_t getActiveCommunicator();

    /**
   * @brief Get the active HIP stream for this test
   *
   * Returns the test-specific HIP stream. Returns nullptr if createTestCommunicator()
   * has not been called first.
   *
   * @return The active HIP stream handle, or nullptr if not created
   *
   * @note Always call createTestCommunicator() before this method
   */
    virtual hipStream_t getActiveStream();

    /**
   * @brief Cleanup test-specific NCCL communicator and HIP stream
   *
   * Destroys the test communicator and stream with proper MPI synchronization.
   * Safe to call multiple times or if resources were never created.
   *
   * @return ncclResult_t - ncclSuccess on success, error code on failure
   *                        Returns ncclUnhandledCudaError if HIP cleanup fails
   *
   * @note For GTest: This is automatically called by cleanupTest()
   * @note For Standalone: Call this explicitly or use RAII wrapper
   * @note Errors are logged but cleanup continues for all resources
   */
    virtual ncclResult_t cleanupTestCommunicator();

    /**
   * @brief Initialize test resources before test execution
   *
   * Override this to perform custom initialization. Default implementation does nothing.
   * This method is framework-agnostic and not tied to GTest's lifecycle.
   * For standalone tests, call this explicitly if needed.
   */
    virtual void initializeTest() {}

    /**
   * @brief Cleanup test resources after test execution
   *
   * Override this to perform custom cleanup. Default implementation calls
   * cleanupTestCommunicator() to destroy NCCL communicator and HIP stream.
   * This method is framework-agnostic and not tied to GTest's lifecycle.
   * For standalone tests, call this explicitly if needed.
   *
   * @note For GTest tests: Errors are logged but don't fail the test (cleanup phase)
   * @note For standalone tests: Check return value and handle errors appropriately
   */
    virtual void cleanupTest()
    {
        (void)cleanupTestCommunicator();
    }
};

#endif // MPI_TESTS_ENABLED

#endif // MPI_TEST_CORE_HPP
