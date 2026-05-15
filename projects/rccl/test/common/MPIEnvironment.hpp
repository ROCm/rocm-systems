/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file MPIEnvironment.hpp
 * @brief Global MPI environment and error checking macros for RCCL testing
 *
 * Provides a Google Test Environment for managing MPI initialization/finalization
 * and error checking macros for MPI, NCCL, and HIP operations in tests.
 */

#ifndef RCCL_MPI_ENVIRONMENT_HPP
#define RCCL_MPI_ENVIRONMENT_HPP

#include <gtest/gtest.h>

// Conditionally include MPI headers for MPI-based tests
#ifdef MPI_TESTS_ENABLED

#include "rccl/rccl.h"
#include <hip/hip_runtime.h>
#include <mpi.h>

#include "TestChecks.hpp"
#include "ResourceGuards.hpp"

/**
 * @class MPIEnvironment
 * @brief Google Test Environment for global MPI setup and teardown
 *
 * Manages the global MPI state for all MPI-based tests:
 * - One-time MPI initialization (MPI_Init_thread)
 * - GPU device initialization and assignment
 * - MPI finalization and result aggregation across ranks
 *
 * @note MPI_Init can only be called once, so this uses static flags
 * @note Each MPI rank is assigned to a unique GPU
 * @see MPITestBase for test-level functionality
 */
class MPIEnvironment : public ::testing::Environment
{
public:
    /**
     * @brief Current MPI rank in MPI_COMM_WORLD
     *
     * Valid after MPI initialization. Each rank corresponds to one GPU.
     */
    inline static int world_rank{0};

    /**
     * @brief Total number of MPI processes in MPI_COMM_WORLD
     *
     * Valid after MPI initialization. Must not exceed number of available GPUs.
     */
    inline static int world_size{0};

    /**
     * @brief Aggregated return code for test results
     *
     * Set to non-zero on test failure. Aggregated across all ranks during cleanup.
     */
    inline static int retCode{0};

    /**
     * @brief Flag indicating MPI has been initialized
     *
     * Prevents multiple MPI_Init calls (only allowed once per process).
     */
    inline static bool mpi_initialized{false};

    /**
     * @brief Cached result of multi-node detection
     *
     * Computed once during SetUp() using MPI_Comm_split_type().
     * -1 = not computed, 0 = single node, 1 = multi-node
     *
     * @note MUST be initialized before any TEST_* macros are called
     * @note Prevents nested MPI collective operations in isMultiNodeTest()
     */
    inline static int cached_multi_node_result{-1};

    /**
     * @brief Flag indicating GPU devices have been initialized
     *
     * Prevents redundant device setup across multiple test runs.
     */
    inline static bool devices_initialized{false};

    /**
     * @brief Initialize MPI with thread support
     *
     * Calls MPI_Init_thread() with MPI_THREAD_MULTIPLE support and sets
     * world_rank and world_size. Safe to call multiple times (idempotent).
     *
     * @note Should be called before any MPI operations
     * @see mpi_initialized flag
     */
    static void initialize_mpi();

    /**
     * @brief Initialize and assign GPU devices to MPI ranks
     *
     * Performs the following:
     * 1. Queries available GPU count
     * 2. Validates sufficient GPUs for all ranks
     * 3. Assigns one GPU per rank (rank N → GPU N)
     * 4. Resets and sets HIP device context
     * 5. Synchronizes all ranks
     *
     * @note Requires world_size ≤ number of available GPUs
     * @see devices_initialized flag
     */
    static void initialize_devices();

    /**
     * @brief Clean up MPI resources and finalize
     *
     * Performs the following cleanup:
     * 1. Synchronizes all ranks with MPI_Barrier
     * 2. Aggregates test results across ranks with MPI_Allreduce
     * 3. Prints final results from rank 0
     * 4. Calls MPI_Finalize()
     *
     * @note Uses static guard to prevent multiple cleanup attempts
     * @note Safe to call from signal handlers or error paths
     */
    static void cleanup_mpi();

    /**
     * @brief Google Test SetUp hook - called once before all tests
     *
     * Initializes MPI and GPU devices for the entire test suite.
     */
    void SetUp() override;

    /**
     * @brief Google Test TearDown hook - called once after all tests
     *
     * Synchronizes all ranks and calls cleanup_mpi() to finalize MPI.
     */
    void TearDown() override;
};

#endif // MPI_TESTS_ENABLED

#endif // RCCL_MPI_ENVIRONMENT_HPP
