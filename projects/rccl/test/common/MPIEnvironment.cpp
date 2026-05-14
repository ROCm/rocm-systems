/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file MPIEnvironment.cpp
 * @brief Implementation of global MPI environment for RCCL testing
 */

#include "MPIEnvironment.hpp"
#include "MPITestBase.hpp"

#ifdef MPI_TESTS_ENABLED

#include <chrono>
#include <thread>

/**
 * @brief Initialize the global test environment
 *
 * Performs one-time setup for the entire test suite:
 * - Initializes MPI with thread support
 * - Sets up GPU devices for each rank
 *
 * @note Called automatically by Google Test framework before any tests run
 */
void MPIEnvironment::SetUp()
{
    // One-time initialization (MPI_Init can only be called once)
    initialize_mpi();
    initialize_devices();
}

/**
 * @brief Initialize MPI with multi-threading support
 *
 * Calls MPI_Init_thread() with MPI_THREAD_MULTIPLE to support concurrent
 * MPI operations. Sets world_rank and world_size for use by all tests.
 *
 * Idempotent - safe to call multiple times (uses mpi_initialized flag).
 * Typically called from main_mpi.cpp, but provides fallback initialization.
 */
void MPIEnvironment::initialize_mpi()
{
    if(mpi_initialized)
    {
        // Already initialized in main_mpi.cpp
        if(world_rank == 0)
        {
            TEST_INFO("MPI already initialized - skipping re-initialization");
        }
        return;
    }

    // This path should not be reached when using main_mpi.cpp
    // but kept for compatibility with other test mains
    auto provided = int{};
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    mpi_initialized = true;

    if(world_rank == 0)
    {
        TEST_INFO("MPI initialized - World size: %d, Thread support: %d", world_size, provided);
    }
}

/**
 * @brief Initialize GPU devices and assign one GPU per MPI rank
 *
 * Performs comprehensive GPU setup:
 * 1. Queries number of available GPUs
 * 2. Validates sufficient GPUs for world_size
 * 3. Assigns GPU ID = rank (rank-based assignment)
 * 4. Resets HIP context for clean state
 * 5. Sets active device
 * 6. Verifies device assignment
 * 7. Synchronizes all ranks
 *
 * @note Requires at least world_size GPUs
 * @note Sets retCode=1 on error (insufficient GPUs, assignment failure)
 * @note Idempotent - safe to call multiple times (uses devices_initialized flag)
 */
void MPIEnvironment::initialize_devices()
{
    if(devices_initialized)
    {
        return; // Already initialized
    }

    auto numDevices = int{};
    HIP_TEST_CHECK_GTEST_FAIL(hipGetDeviceCount(&numDevices));

    // Calculate local rank (rank within this node) for multi-node support
    // Split MPI_COMM_WORLD by node using MPI_Comm_split_type
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD,
                        MPI_COMM_TYPE_SHARED,
                        world_rank,
                        MPI_INFO_NULL,
                        &node_comm);

    int local_rank, local_size;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &local_size);

    // Cache multi-node detection result ONCE during initialization
    // local_size < world_size means we have multiple nodes
    cached_multi_node_result = (local_size < world_size) ? 1 : 0;

    if(world_rank == 0)
    {
        TEST_INFO("Detected %d GPU(s) for %d MPI rank(s)", numDevices, world_size);
        TEST_INFO("Local configuration: %d ranks per node", local_size);
        TEST_INFO("Multi-node configuration: %s",
                  cached_multi_node_result ? "YES (multiple nodes)" : "NO (single node)");
    }

    // Check if we have enough GPUs for ranks on THIS node
    if(numDevices < local_size)
    {
        TEST_ABORT(
            "ERROR: (local rank %d): Only %d GPUs available on this node for %d local ranks. "
            "RCCL requires unique GPUs per rank on each node. "
            "Please run with fewer ranks per node (e.g., --ntasks-per-node=%d) "
            "or ensure more GPUs are available.",
            local_rank,
            numDevices,
            local_size,
            numDevices);
        retCode             = 1;
        devices_initialized = true;
        MPI_Comm_free(&node_comm);
        return;
    }

    // Use LOCAL rank for device assignment (not global rank)
    // This ensures ranks 0-7 on each node use GPUs 0-7
    const auto assigned_device = local_rank;

    // Validate device assignment
    if(assigned_device < 0 || assigned_device >= numDevices)
    {
        TEST_ABORT(
            "ERROR: (local rank %d): Invalid device assignment! assigned_device=%d, numDevices=%d",
            local_rank,
            assigned_device,
            numDevices);
        retCode             = 1;
        devices_initialized = true;
        MPI_Comm_free(&node_comm);
        return;
    }

    // Complete HIP context reset and isolation
    HIP_TEST_CHECK_GTEST_FAIL(hipDeviceReset());
    HIP_TEST_CHECK_GTEST_FAIL(hipSetDevice(assigned_device));

    // Force HIP context creation and synchronization
    auto prop = hipDeviceProp_t{};
    HIP_TEST_CHECK_GTEST_FAIL(hipGetDeviceProperties(&prop, assigned_device));
    HIP_TEST_CHECK_GTEST_FAIL(hipDeviceSynchronize());

    // Verify device assignment
    auto current_device = int{};
    HIP_TEST_CHECK_GTEST_FAIL(hipGetDevice(&current_device));
    if(current_device != assigned_device)
    {
        TEST_ABORT("ERROR: (local rank %d) device assignment failed! Expected %d, got %d",
                   local_rank,
                   assigned_device,
                   current_device);
        retCode = 1;
        MPI_Comm_free(&node_comm);
        return;
    }

    // Print device info (only from rank 0 to reduce output)
    if(world_rank == 0)
    {
        TEST_INFO("(local rank %d): Device assignment: global rank %d -> GPU %d",
                  local_rank,
                  world_rank,
                  assigned_device);
        TEST_INFO("PCI Bus ID = 0x%x, Device Name = %s", prop.pciBusID, prop.name);
        TEST_INFO("Total GPUs available per node: %d", numDevices);
        TEST_INFO("Multi-node: Each node's local ranks (0-%d) mapped to GPUs (0-%d)",
                  local_size - 1,
                  numDevices - 1);
    }

    // Clean up node communicator
    MPI_Comm_free(&node_comm);

    // Ensure all ranks have set their devices before proceeding
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    devices_initialized = true;

    if(world_rank == 0)
    {
        TEST_INFO("Device initialization completed");
        TEST_INFO("Each test will create its own NCCL communicator for isolation");
    }
}

/**
 * @brief Tear down the global test environment
 *
 * Ensures all ranks have completed their tests before cleanup:
 * 1. Synchronizes all ranks with MPI_Barrier
 * 2. Calls cleanup_mpi() to finalize MPI
 *
 * @note Critical synchronization point - ensures all test cleanup is complete
 * @note Called automatically by Google Test framework after all tests complete
 */
void MPIEnvironment::TearDown()
{
    // CRITICAL: Handle the case where ranks are out of sync due to test failures
    //
    // Problem: If rank 0 fails with ASSERT/FAIL, it immediately goes to TearDown()
    // while rank 1 is still in the test body. This causes deadlock when rank 0
    // tries to do MPI collectives (like Allreduce) while rank 1 is doing different
    // MPI collectives (like Bcast in createTestCommunicator).
    //
    // Use MPI_Ibarrier (non-blocking) with a timeout to detect if ranks
    // are out of sync, then force cleanup with MPI_Abort if necessary.

    // Try a non-blocking barrier to check if all ranks are ready
    MPI_Request barrier_req;
    int         barrier_result = MPI_Ibarrier(MPI_COMM_WORLD, &barrier_req);

    if(barrier_result == MPI_SUCCESS)
    {
        // Wait for barrier with a timeout (1 second)
        int        flag             = 0;
        auto       timeout_start    = std::chrono::steady_clock::now();
        const auto timeout_duration = std::chrono::seconds(1);

        while(!flag)
        {
            MPI_Test(&barrier_req, &flag, MPI_STATUS_IGNORE);

            if(!flag)
            {
                // Check if timeout exceeded
                auto elapsed = std::chrono::steady_clock::now() - timeout_start;
                if(elapsed > timeout_duration)
                {
                    // Timeout - ranks are out of sync!
                    std::fprintf(
                        stderr,
                        "Rank %d: TIMEOUT in TearDown barrier - ranks out of sync, forcing abort\n",
                        world_rank);
                    std::fflush(stderr);

                    // Cancel the barrier request
                    MPI_Cancel(&barrier_req);
                    MPI_Request_free(&barrier_req);

                    // Force abort - can't safely continue
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return;
                }

                // Sleep briefly to avoid busy-waiting
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        // Barrier completed - all ranks are synchronized
        // Now safe to do collective operations

        // Check if ANY rank had a failure
        int local_failed  = (retCode != 0) ? 1 : 0;
        int global_failed = 0;
        MPI_Allreduce(&local_failed, &global_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        // Update retCode to reflect global failure status
        if(global_failed > 0)
        {
            retCode = 1;
        }
    }
    else
    {
        // MPI_Ibarrier failed - something is very wrong
        std::fprintf(stderr,
                     "Rank %d: MPI_Ibarrier failed in TearDown, forcing abort\n",
                     world_rank);
        std::fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    cleanup_mpi();
}

/**
 * @brief Clean up MPI resources and finalize
 *
 * Performs coordinated cleanup across all ranks:
 * 1. Guards against multiple cleanup attempts
 * 2. Synchronizes all ranks
 * 3. Aggregates test results using MPI_Allreduce
 * 4. Prints final results from rank 0
 * 5. Calls MPI_Finalize()
 * 6. Resets initialization flags
 *
 * Uses context-aware error handling:
 * - MPI_Barrier/Allreduce: MPICHECK with rank (aborts on error)
 * - MPI_Finalize: MPICHECK with rank and true flag (exits on error)
 *
 * @note Uses static guard to prevent multiple cleanup attempts
 * @note Safe to call from signal handlers or error paths
 * @note All ranks must call this function for proper finalization
 */
void MPIEnvironment::cleanup_mpi()
{
    // Use static guard to prevent multiple cleanup attempts
    static bool cleanup_in_progress_or_done = false;

    if(cleanup_in_progress_or_done)
    {
        return; // Already cleaned up or currently cleaning up
    }

    if(!mpi_initialized)
    {
        return; // Never initialized
    }

    cleanup_in_progress_or_done = true;

    // Synchronize all ranks before MPI finalization
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD), world_rank);

    MPICHECK(MPI_Finalize(), world_rank, true);

    mpi_initialized     = false;
    devices_initialized = false;
}

/**
 * @brief Accessor function to get cached multi-node detection result
 *
 * This function is defined here to avoid circular dependency between
 * TestChecks.hpp and MPIEnvironment.hpp.
 *
 * @return The cached multi-node result: -1 (not computed), 0 (single node), 1 (multi-node)
 */
int getMPIEnvironmentCachedMultiNodeResult()
{
    return MPIEnvironment::cached_multi_node_result;
}

#endif // MPI_TESTS_ENABLED
