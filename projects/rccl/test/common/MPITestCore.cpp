/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "MPITestCore.hpp"

#ifdef MPI_TESTS_ENABLED
#include "ResourceGuards.hpp"
#include <cstdlib>
#include <string>

// Import commonly used guards into local scope
using RCCLTestGuards::makeScopeGuard;

// Detect the number of unique nodes
int MPITestConstants::detectNodeCount()
{
    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if(world_rank == 0)
    {
        TEST_INFO("=== MPI Process Distribution ===");
        TEST_INFO("Total processes: %d", world_size);
    }

    // Special case: single process is always single node
    if(world_size == 1)
    {
        if(world_rank == 0)
        {
            TEST_INFO("Detected nodes:  1");
            TEST_INFO("================================");
        }
        return 1;
    }

    // Use MPI_COMM_TYPE_SHARED to split by node
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);

    int node_rank = 0;
    int node_size = 0;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);

    // Gather node sizes to rank 0
    std::vector<int> all_node_sizes;
    if(world_rank == 0)
    {
        all_node_sizes.resize(world_size);
    }
    MPI_Gather(&node_size, 1, MPI_INT, all_node_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Rank 0 analyzes distribution
    int num_nodes = 0;
    if(world_rank == 0)
    {
        std::vector<int> node_counts; // ranks per node
        std::vector<int> node_first_rank; // first rank on each node

        for(int r = 0; r < world_size; r++)
        {
            bool found = false;
            for(size_t n = 0; n < node_counts.size(); n++)
            {
                // Same node if same node_size and rank is within that node
                if(all_node_sizes[r] == all_node_sizes[node_first_rank[n]])
                {
                    // Check if this rank belongs to this node group
                    int local_rank = r - node_first_rank[n];
                    if(local_rank >= 0 && local_rank < node_counts[n])
                    {
                        found = true;
                        break;
                    }
                }
            }

            if(!found)
            {
                node_first_rank.push_back(r);
                node_counts.push_back(all_node_sizes[r]);
            }
        }

        num_nodes = static_cast<int>(node_counts.size());

        TEST_INFO("Detected nodes:  %d", num_nodes);
        TEST_INFO("");

        // Get hostnames for display
        char hostname[MPI_MAX_PROCESSOR_NAME];
        int  hostname_len;
        MPI_Get_processor_name(hostname, &hostname_len);

        for(size_t n = 0; n < node_counts.size(); n++)
        {
            int first_rank = node_first_rank[n];
            TEST_INFO("Node %zu: %d rank(s) starting at rank %d", n, node_counts[n], first_rank);

            // Print ranks on this node - build string first for TEST_INFO
            std::string ranks_str = "  Ranks: ";
            for(int r = first_rank; r < first_rank + node_counts[n]; r++)
            {
                ranks_str += std::to_string(r);
                if(r < first_rank + node_counts[n] - 1)
                    ranks_str += ", ";
            }
            TEST_INFO("%s", ranks_str.c_str());
        }
        TEST_INFO("================================");
    }

    // Broadcast node count to all ranks
    MPI_Bcast(&num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Comm_free(&node_comm);

    return num_nodes;
}

// Validate test prerequisites
bool MPITestCore::validateTestPrerequisites(
    int min_processes, int max_processes, bool require_power_of_two, int min_nodes, int max_nodes)
{
    int world_rank = MPIEnvironment::world_rank;
    int world_size = MPIEnvironment::world_size;

    // Always detect nodes and display process distribution
    // This provides valuable information for all tests
    int actual_nodes = MPITestConstants::detectNodeCount();

    bool validation_passed = true;

    if(world_rank == 0)
    {
        TEST_INFO("=== Test Requirements ===");
        if(min_processes == max_processes)
        {
            TEST_INFO("Processes:         exactly %d", min_processes);
        }
        else if(max_processes == MPITestConstants::kNoProcessLimit)
        {
            TEST_INFO("Processes:         at least %d", min_processes);
        }
        else
        {
            TEST_INFO("Processes:         between %d and %d", min_processes, max_processes);
        }

        if(require_power_of_two)
        {
            TEST_INFO("Power-of-two:      required");
        }

        if(min_nodes > 1 || max_nodes > 0)
        {
            if(min_nodes == max_nodes)
            {
                TEST_INFO("Nodes:             exactly %d", min_nodes);
            }
            else if(max_nodes == MPITestConstants::kNoNodeLimit)
            {
                TEST_INFO("Nodes:             at least %d", min_nodes);
            }
            else
            {
                TEST_INFO("Nodes:             between %d and %d", min_nodes, max_nodes);
            }
        }

        TEST_INFO("");
        TEST_INFO("=== Current Environment ===");
        TEST_INFO("Processes:         %d", world_size);
        TEST_INFO("Nodes:             %d", actual_nodes);
        if(require_power_of_two)
        {
            TEST_INFO("Power-of-two:      %s",
                      MPITestConstants::isPowerOfTwo(world_size) ? "yes" : "no");
        }
        TEST_INFO("===========================");
        TEST_INFO("");
    }

    // Validate process count
    if(world_size < min_processes)
    {
        validation_passed = false;
        if(world_rank == 0)
        {
            printf("Error: REQUIREMENT NOT MET: Need at least %d processes, got %d\n",
                   min_processes,
                   world_size);
            printf("   For test details, set: NCCL_DEBUG=INFO\n");
        }
    }

    if(max_processes != MPITestConstants::kNoProcessLimit && world_size > max_processes)
    {
        validation_passed = false;
        if(world_rank == 0)
        {
            printf("Error: REQUIREMENT NOT MET: Need at most %d processes, got %d\n",
                   max_processes,
                   world_size);
            printf("   For test details, set: NCCL_DEBUG=INFO\n");
        }
    }

    if(require_power_of_two && !MPITestConstants::isPowerOfTwo(world_size))
    {
        validation_passed = false;
        if(world_rank == 0)
        {
            printf("Error: REQUIREMENT NOT MET: Need power-of-two processes, got %d (not power of "
                   "2)\n",
                   world_size);
            printf("   For test details, set: NCCL_DEBUG=INFO\n");
        }
    }

    // Validate node count
    if(min_nodes > 1 || max_nodes > 0)
    {
        if(actual_nodes < min_nodes)
        {
            validation_passed = false;
            if(world_rank == 0)
            {
                printf("Error: REQUIREMENT NOT MET: Need at least %d node(s), detected %d nodes\n",
                       min_nodes,
                       actual_nodes);
                printf("   For test details, set: NCCL_DEBUG=INFO\n");
            }
        }

        if(max_nodes != MPITestConstants::kNoNodeLimit && actual_nodes > max_nodes)
        {
            validation_passed = false;
            if(world_rank == 0)
            {
                printf("Error: REQUIREMENT NOT MET: Need at most %d node(s), detected %d nodes\n",
                       max_nodes,
                       actual_nodes);
                printf("   For test details, set: NCCL_DEBUG=INFO\n");
                if(max_nodes == 1)
                {
                    printf("   This test requires single-node execution\n");
                    printf("   To run on single node, allocate all processes on the same host\n");
                }
            }
        }
    }

    if(world_rank == 0)
    {
        if(validation_passed)
        {
            TEST_INFO("All requirements met - test will run");
        }
        else
        {
            TEST_INFO("===========================");
            TEST_INFO("");
        }
    }

    return validation_passed;
}

// Create test communicator
ncclResult_t MPITestCore::createTestCommunicator()
{
    int world_rank = MPIEnvironment::world_rank;
    int world_size = MPIEnvironment::world_size;

    if(world_rank == 0)
    {
        TEST_INFO("Creating test-specific communicator");
    }

    // Rank 0 generates unique ID
    if(world_rank == 0)
    {
        RCCL_TEST_CHECK(ncclGetUniqueId(&nccl_id_));
    }

    // Broadcast ID to all ranks
    MPI_Bcast(&nccl_id_, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Initialize NCCL communicator with automatic cleanup on error
    RCCL_TEST_CHECK(ncclGroupStart());

    // RAII guard: Automatically calls ncclGroupEnd() if subsequent operations fail
    auto group_guard = makeScopeGuard([]() { (void)ncclGroupEnd(); });

    RCCL_TEST_CHECK(ncclCommInitRank(&test_comm_, world_size, nccl_id_, world_rank));

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

    if(world_rank == 0)
    {
        TEST_INFO("Test-specific communicator created successfully");
    }

    // Success! Dismiss guards so resources aren't destroyed
    comm_guard.dismiss();
    stream_guard.dismiss();

    return ncclSuccess;
}

// Get active communicator
ncclComm_t MPITestCore::getActiveCommunicator()
{
    return test_comm_;
}

// Get active stream
hipStream_t MPITestCore::getActiveStream()
{
    return test_stream_;
}

// Cleanup test communicator
ncclResult_t MPITestCore::cleanupTestCommunicator()
{
    if(!test_comm_ && !test_stream_)
    {
        return ncclSuccess; // Already cleaned up or never created
    }

    int world_rank = MPIEnvironment::world_rank;

    MPI_Barrier(MPI_COMM_WORLD);

    // RAII guard: Ensure test_comm_ is destroyed even if stream cleanup fails
    auto comm_guard = makeScopeGuard(
        [this, world_rank]()
        {
            if(test_comm_)
            {
                ncclResult_t result = ncclCommDestroy(test_comm_);
                if(result != ncclSuccess)
                {
                    TEST_WARN("Rank %d: ncclCommDestroy failed during cleanup: %s",
                              world_rank,
                              ncclGetErrorString(result));
                }
                test_comm_ = nullptr;
            }
        });

    // RAII guard: Ensure test_stream_ is destroyed
    auto stream_guard = makeScopeGuard(
        [this, world_rank]()
        {
            if(test_stream_)
            {
                hipError_t hip_result = hipStreamDestroy(test_stream_);
                if(hip_result != hipSuccess)
                {
                    TEST_WARN("Rank %d: hipStreamDestroy failed during cleanup: %s",
                              world_rank,
                              hipGetErrorString(hip_result));
                }
                test_stream_ = nullptr;
            }
        });

    // Guards will automatically clean up when going out of scope
    // Even if an exception were thrown (though we don't use exceptions)

    MPI_Barrier(MPI_COMM_WORLD);

    return ncclSuccess;
}

#endif // MPI_TESTS_ENABLED
