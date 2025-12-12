/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file MPIStandaloneTest.hpp
 * @brief Standalone (non-GTest) adapter for MPI tests
 *
 * Provides infrastructure for writing standalone MPI tests without Google Test.
 * Ideal for performance benchmarks, low-level API tests, and production utilities.
 *
 * @see MPITestCore for the base framework-agnostic functionality
 * @see MPITestBase for GTest integration
 */

#ifndef MPI_STANDALONE_TEST_HPP
#define MPI_STANDALONE_TEST_HPP

#include "MPITestCore.hpp"

#ifdef MPI_TESTS_ENABLED

/**
 * @class MPIStandaloneTest
 * @brief Standalone test adapter for MPI tests (no GTest dependency)
 *
 * Provides a simple base class for standalone MPI tests that don't require
 * Google Test framework. Useful for:
 * - Performance benchmarks (bandwidth, latency)
 * - Low-level API testing
 * - Production utilities
 * - Custom test harnesses
 *
 * **Key Features:**
 * - No GTest dependency
 * - Simple run() interface
 * - Automatic resource cleanup via RAII
 * - Same validation and setup as GTest tests
 * - Return code-based error reporting
 *
 * **Usage Pattern:**
 * @code
 * class MyBandwidthTest : public MPIStandaloneTest {
 * public:
 *     int run() override {
 *         // Validate prerequisites
 *         if (!validateTestPrerequisites(2)) {
 *             if (MPIEnvironment::world_rank == 0) {
 *                 printf("SKIP: Need at least 2 processes\n");
 *             }
 *             return 0; // Skip (not an error)
 *         }
 *
 *         // Setup communicator
 *         if (createTestCommunicator() != ncclSuccess) {
 *             if (MPIEnvironment::world_rank == 0) {
 *                 fprintf(stderr, "ERROR: Failed to create communicator\n");
 *             }
 *             return 1; // Error
 *         }
 *
 *         // Run test logic
 *         ncclComm_t comm = getActiveCommunicator();
 *         hipStream_t stream = getActiveStream();
 *
 *         // Your test code here...
 *
 *         return 0; // Success
 *     }
 * };
 *
 * int main(int argc, char** argv) {
 *     MPI_Init(&argc, &argv);
 *
 *     MyBandwidthTest test;
 *     int result = test.run();
 *     test.cleanup(); // Explicit cleanup
 *
 *     MPI_Finalize();
 *     return result;
 * }
 * @endcode
 *
 * **RAII Wrapper Alternative:**
 * @code
 * int main(int argc, char** argv) {
 *     MPI_Init(&argc, &argv);
 *
 *     int result = 0;
 *     {
 *         MPIStandaloneTestRAII test;
 *         MyBandwidthTest bandwidth_test;
 *         result = bandwidth_test.run();
 *         // Automatic cleanup when test goes out of scope
 *     }
 *
 *     MPI_Finalize();
 *     return result;
 * }
 * @endcode
 *
 * @note For GTest-based tests, use MPITestBase instead
 */
class MPIStandaloneTest : public MPITestCore
{
public:
    /**
   * @brief Virtual destructor for proper cleanup
   */
    virtual ~MPIStandaloneTest() = default;

    /**
   * @brief Main test execution method - override this
   *
   * Override this method to implement your test logic.
   *
   * @return 0 for success/skip, non-zero for error
   *
   * @par Return Codes:
   * - 0: Success or test skipped (validation failed)
   * - 1: Generic error
   * - Other: Custom error codes
   */
    virtual int run() = 0;

    /**
   * @brief Explicit cleanup method
   *
   * Call this after run() completes to ensure proper resource cleanup.
   * Alternatively, use MPIStandaloneTestRAII for automatic cleanup.
   */
    void cleanup()
    {
        cleanupTestCommunicator();
    }

    /**
   * @brief Setup hook (optional)
   *
   * Override this to perform custom setup before run().
   * Default implementation does nothing.
   */
    void setUp() override
    {
        SetUp();
    }

    /**
   * @brief Teardown hook (optional)
   *
   * Override this to perform custom cleanup after run().
   * Default implementation calls cleanupTestCommunicator().
   */
    void tearDown() override
    {
        TearDown();
    }
};

/**
 * @class MPIStandaloneTestRAII
 * @brief RAII wrapper for automatic MPIStandaloneTest cleanup
 *
 * Provides scope-based automatic cleanup for MPIStandaloneTest.
 * Useful for ensuring cleanup even with early returns or exceptions.
 *
 * @par Example:
 * @code
 * int main(int argc, char** argv) {
 *     MPI_Init(&argc, &argv);
 *
 *     int result = 0;
 *     {
 *         MPIStandaloneTestRAII raii_wrapper;
 *         MyTest test;
 *         result = test.run();
 *         // Automatic cleanup when raii_wrapper goes out of scope
 *     }
 *
 *     MPI_Finalize();
 *     return result;
 * }
 * @endcode
 */
class MPIStandaloneTestRAII
{
private:
    MPIStandaloneTest* test_ = nullptr;

public:
    /**
   * @brief Constructor - registers test for cleanup
   * @param test Pointer to test instance (optional)
   */
    explicit MPIStandaloneTestRAII(MPIStandaloneTest* test = nullptr) : test_(test) {}

    /**
   * @brief Destructor - performs automatic cleanup
   */
    ~MPIStandaloneTestRAII()
    {
        if(test_)
        {
            test_->cleanup();
        }
    }

    /**
   * @brief Set test instance to manage
   * @param test Pointer to test instance
   */
    void setTest(MPIStandaloneTest* test)
    {
        test_ = test;
    }

    // Delete copy constructor and assignment operator
    MPIStandaloneTestRAII(const MPIStandaloneTestRAII&)            = delete;
    MPIStandaloneTestRAII& operator=(const MPIStandaloneTestRAII&) = delete;
};

#endif // MPI_TESTS_ENABLED

#endif // MPI_STANDALONE_TEST_HPP
