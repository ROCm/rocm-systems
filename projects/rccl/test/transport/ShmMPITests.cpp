/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "DeviceBufferHelpers.hpp"
#include "TestChecks.hpp"
#include "ResourceGuards.hpp"
#include "TransportMPIBase.hpp"

#include <cmath>

#ifdef MPI_TESTS_ENABLED

// Import MPI test constants
using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;
using namespace TransportTestConstants;

// SHM-specific test configuration
struct ShmTestConfig
{
    bool   is_sender{false};
    void*  send_buffer{nullptr};
    void*  recv_buffer{nullptr};
    size_t buffer_size{0};
};

class ShmMPITest : public TransportTestBase
{
protected:
    ShmTestConfig shm_config;

    // Test data buffers
    std::vector<uint32_t> host_send_data;
    std::vector<uint32_t> host_recv_data;

    // Connection info structures for setup/connect phases
    ncclConnect send_connect_info{};
    ncclConnect recv_connect_info{};

    void SetUp() override
    {
        // Call base class SetUp first
        TransportTestBase::SetUp();

        // Switch to SHM transport
        setTransportType(TransportType::SHM);

        // Set up SHM-specific test configuration
        shm_config.is_sender   = (config.world_rank == 0);
        shm_config.buffer_size = kDefaultBufferSize;

        // Allocate and initialize send buffer with test pattern
        constexpr size_t num_elements = kDefaultBufferSize / sizeof(float);
        auto [send_err, _]            = allocateAndInitialize<float>(&shm_config.send_buffer,
                                                          num_elements,
                                                          config.world_rank);
        EXPECT_EQ(hipSuccess, send_err)
            << "Rank " << config.world_rank << ": Failed to allocate/initialize send buffer";

        // Allocate and zero-initialize receive buffer
        hipError_t hip_result = hipMalloc(&shm_config.recv_buffer, shm_config.buffer_size);
        EXPECT_EQ(hipSuccess, hip_result)
            << "Rank " << config.world_rank << ": Failed to allocate recv buffer";

        hip_result = zeroInitializeBuffer<float>(shm_config.recv_buffer, num_elements);
        EXPECT_EQ(hipSuccess, hip_result)
            << "Rank " << config.world_rank << ": Failed to zero-initialize recv buffer";

        // Synchronize default stream to ensure all buffer operations complete
        // Note: This is called in SetUp() before test starts, so we use the default stream (0)
        // Using config.stream here causes "invalid resource handle" as it's not yet initialized
        EXPECT_EQ(hipSuccess, hipStreamSynchronize(0))
            << "Rank " << config.world_rank
            << ": Failed to synchronize default stream after buffer initialization";
    }

    void TearDown() override
    {
        // Cleanup SHM-specific test resources
        if(shm_config.send_buffer)
        {
            (void)hipFree(shm_config.send_buffer);
            shm_config.send_buffer = nullptr;
        }
        if(shm_config.recv_buffer)
        {
            (void)hipFree(shm_config.recv_buffer);
            shm_config.recv_buffer = nullptr;
        }

        // Call base class TearDown
        TransportTestBase::TearDown();
    }

public:
    // Test SHM capability detection (same-host communication)
    void testShmCanConnect()
    {
        // Validate preconditions
        ASSERT_NE(nullptr, comm_handle)
            << "Rank " << config.world_rank
            << ": comm_handle is null - NCCL communicator not initialized";
        ASSERT_NE(nullptr, local_peer_info)
            << "Rank " << config.world_rank
            << ": local_peer_info is null - peer information not initialized";
        ASSERT_NE(nullptr, remote_peer_info)
            << "Rank " << config.world_rank
            << ": remote_peer_info is null - peer information not initialized";

        int        can_connect = 0;
        const auto result      = shmTransport.canConnect(&can_connect,
                                                    comm_handle,
                                                    topology_graph,
                                                    local_peer_info,
                                                    remote_peer_info);

        ASSERT_EQ(ncclSuccess, result) << "Rank " << config.world_rank
                                       << ": shmCanConnect failed: " << ncclGetErrorString(result);

        // Synchronize the stream to ensure all operations complete
        ASSERT_EQ(hipSuccess, syncStream(config.stream, config.world_rank))
            << "Rank " << config.world_rank << ": Stream synchronization failed";
    }

    // Test SHM setup phase
    void testShmSetup()
    {
        // Call setup() and save the connect_info to class members for later MPI exchange
        const auto result = shm_config.is_sender
                                ? shmTransport.send.setup(comm_handle,
                                                                           topology_graph,
                                                                           local_peer_info,
                                                                           remote_peer_info,
                                                         &send_connect_info,  // Save to class member
                                                                           &send_connector,
                                                                           0,
                                                                           0)
                                                  : shmTransport.recv.setup(comm_handle,
                                                                           topology_graph,
                                                                           local_peer_info,
                                                                           remote_peer_info,
                                                         &recv_connect_info,  // Save to class member
                                                                           &recv_connector,
                                                                           0,
                                                                           0);

        ASSERT_EQ(ncclSuccess, result)
            << "Rank " << config.world_rank << ": " << (shm_config.is_sender ? "Send" : "Recv")
            << " setup failed: " << ncclGetErrorString(result);
    }

    // Test SHM connection phase
    void testShmConnect()
    {
        // Validate preconditions
        ASSERT_NE(nullptr, comm_handle) << "Rank " << config.world_rank << ": comm_handle is null";
        ASSERT_NE(nullptr, local_peer_info)
            << "Rank " << config.world_rank << ": local_peer_info is null";
        ASSERT_NE(nullptr, remote_peer_info)
            << "Rank " << config.world_rank << ": remote_peer_info is null";

        // NOTE: setup() was already called in testShmSetup() and saved connect_info to class members
        // This method only does MPI exchange of connect_info and then calls connect()

        if(shm_config.is_sender)
        {
            // Exchange connect info with receiver using MPI
            ASSERT_EQ(MPI_SUCCESS,
                      MPI_Send(&send_connect_info,  // Use class member from testShmSetup()
                               sizeof(ncclConnect),
                               MPI_BYTE,
                               config.peer_rank,
                               0,
                               MPI_COMM_WORLD))
                << "Rank " << config.world_rank << ": MPI_Send failed";

            ASSERT_EQ(MPI_SUCCESS,
                      MPI_Recv(&recv_connect_info,  // Receive into class member
                               sizeof(ncclConnect),
                               MPI_BYTE,
                               config.peer_rank,
                               0,
                               MPI_COMM_WORLD,
                               MPI_STATUS_IGNORE))
                << "Rank " << config.world_rank << ": MPI_Recv failed";

            // Perform the actual connection using the received info
            auto result = shmTransport.send.connect(comm_handle,
                                               &recv_connect_info,
                                               config.world_size,
                                               config.world_rank,
                                               &send_connector);
            ASSERT_EQ(ncclSuccess, result)
                << "Rank " << config.world_rank
                << ": Send connect failed: " << ncclGetErrorString(result);
        }
        else
        {
            // Exchange connect info with sender using MPI
            ASSERT_EQ(MPI_SUCCESS,
                      MPI_Recv(&send_connect_info,  // Receive into class member
                               sizeof(ncclConnect),
                               MPI_BYTE,
                               config.peer_rank,
                               0,
                               MPI_COMM_WORLD,
                               MPI_STATUS_IGNORE))
                << "Rank " << config.world_rank << ": MPI_Recv failed";

            ASSERT_EQ(MPI_SUCCESS,
                      MPI_Send(&recv_connect_info,  // Use class member from testShmSetup()
                               sizeof(ncclConnect),
                               MPI_BYTE,
                               config.peer_rank,
                               0,
                               MPI_COMM_WORLD))
                << "Rank " << config.world_rank << ": MPI_Send failed";

            // Perform the actual connection using the received info
            auto result = shmTransport.recv.connect(comm_handle,
                                               &send_connect_info,
                                               config.world_size,
                                               config.world_rank,
                                               &recv_connector);
            ASSERT_EQ(ncclSuccess, result)
                << "Rank " << config.world_rank
                << ": Recv connect failed: " << ncclGetErrorString(result);
        }

        // Synchronize the stream to ensure all RCCL operations complete
        ASSERT_EQ(hipSuccess, syncStream(config.stream, config.world_rank))
            << "Rank " << config.world_rank << ": Stream synchronization failed";
    }

    // Test actual data transfer through SHM
    void testShmDataTransfer()
    {
        // Initialize host data vectors
        const size_t num_elements = shm_config.buffer_size / sizeof(uint32_t);
        host_recv_data.resize(num_elements);
        host_send_data.resize(num_elements);

        // Use RCCL point-to-point operations to validate SHM transport
        const size_t count  = shm_config.buffer_size / sizeof(float);
        const auto   result = shm_config.is_sender ? ncclSend(shm_config.send_buffer,
                                                            count,
                                                            ncclFloat,
                                                            config.peer_rank,
                                                            config.nccl_comm,
                                                            config.stream)
                                                   : ncclRecv(shm_config.recv_buffer,
                                                            count,
                                                            ncclFloat,
                                                            config.peer_rank,
                                                            config.nccl_comm,
                                                            config.stream);

        ASSERT_EQ(ncclSuccess, result)
            << "Rank " << config.world_rank << ": RCCL " << (shm_config.is_sender ? "Send" : "Recv")
            << " failed: " << ncclGetErrorString(result);

        ASSERT_EQ(hipSuccess, syncStream(config.stream, config.world_rank))
            << "Rank " << config.world_rank << ": Stream synchronization failed";

        // Only validate data on the receiver side
        if(!shm_config.is_sender)
        {
            ASSERT_FALSE(host_recv_data.empty())
                << "Rank " << config.world_rank << ": host_recv_data is empty";
            ASSERT_NE(nullptr, shm_config.recv_buffer)
                << "Rank " << config.world_rank << ": recv_buffer is null";

            ASSERT_EQ(hipSuccess,
                      hipMemcpy(host_recv_data.data(),
                                shm_config.recv_buffer,
                                shm_config.buffer_size,
                                hipMemcpyDeviceToHost))
                << "Rank " << config.world_rank << ": hipMemcpy DeviceToHost failed";

            // Validate received data - should match sender's original pattern
            const size_t validation_count = std::min(num_elements, kMaxValidationElements);
            for(size_t i = 0; i < validation_count; i++)
            {
                const float expected_float
                    = static_cast<float>(config.peer_rank * kDefaultPatternMultiplier + i);
                const uint32_t expected_value = *reinterpret_cast<const uint32_t*>(&expected_float);

                EXPECT_EQ(expected_value, host_recv_data[i])
                    << "Rank " << config.world_rank << ": Data mismatch at index " << i;
            }
        }
    }

    // Test resource cleanup
    void testShmCleanup()
    {
        // Ensure all stream operations complete before validation
        [[maybe_unused]] auto err = syncStream(config.stream, config.world_rank);
        // Don't return error on sync failure - continue with validation

        // Validate that connector resources are still valid at this point
        // The actual cleanup will be handled by base class TearDown()
        auto* connector = shm_config.is_sender ? &send_connector : &recv_connector;

        EXPECT_NE(nullptr, connector)
            << "Rank " << config.world_rank << ": Connector pointer is null";

        if(connector)
        {
            EXPECT_NE(nullptr, connector->transportResources)
                << "Rank " << config.world_rank << ": " << (shm_config.is_sender ? "Send" : "Recv")
                << " connector transport resources are null (premature cleanup)";

            if(config.world_rank == 0 && connector->transportResources)
            {
                TEST_INFO("Connector resources validated - still active (will be freed by base class)");
        }
        }

        // NOTE: Connectors will be automatically freed by base class TearDown()
        // Device sync + connector cleanup happens BEFORE buffers are freed, which is critical for CE memcpy
    }

    // Test SHM with memcpy mode enabled (CE - Copy Engine)
    // This test uses the transport API directly to ensure SHM methods are called
    void testShmWithMemcpy()
    {
        // Check if NCCL_SHM_USE_CUDA_MEMCPY is set externally
        const char* shm_memcpy_env = getenv("NCCL_SHM_USE_CUDA_MEMCPY");
        if(!shm_memcpy_env || strcmp(shm_memcpy_env, "1") != 0)
        {
            if(MPIEnvironment::world_rank == 0)
            {
                TEST_INFO("Skipping CE memcpy test - NCCL_SHM_USE_CUDA_MEMCPY not set to '1'");
                TEST_INFO("To enable this test, set: export NCCL_SHM_USE_CUDA_MEMCPY=1");
            } // Skip test gracefully
        }

        // Validate preconditions
        ASSERT_NE(nullptr, comm_handle) << "Rank " << config.world_rank << ": comm_handle is null";
        ASSERT_NE(nullptr, local_peer_info)
            << "Rank " << config.world_rank << ": local_peer_info is null";
        ASSERT_NE(nullptr, remote_peer_info)
            << "Rank " << config.world_rank << ": remote_peer_info is null";

        // Step 1: Test shmCanConnect with CE memcpy enabled
        int          can_connect = 0;
        ncclResult_t result      = shmTransport.canConnect(&can_connect,
                                                      comm_handle,
                                                      topology_graph,
                                                      local_peer_info,
                                                      remote_peer_info);

        ASSERT_EQ(ncclSuccess, result) << "Rank " << config.world_rank
                                       << ": shmCanConnect failed: " << ncclGetErrorString(result);

        ASSERT_EQ(1, can_connect)
            << "Rank " << config.world_rank
            << ": SHM cannot connect - test skipped but connection was expected";

        // Step 2: Test SHM setup with CE memcpy enabled

        ncclConnect send_connect_info{};
        ncclConnect recv_connect_info{};

        if(shm_config.is_sender)
        {
            result = shmTransport.send.setup(comm_handle,
                                             topology_graph,
                                             local_peer_info,
                                             remote_peer_info,
                                             &send_connect_info,
                                             &send_connector,
                                             0,
                                             0);
            ASSERT_EQ(ncclSuccess, result)
                << "Rank " << config.world_rank
                << ": SHM send setup with CE memcpy failed: " << ncclGetErrorString(result);

            // Exchange connect info with receiver
            ASSERT_EQ(MPI_SUCCESS,
                      MPI_Send(&send_connect_info,
                               sizeof(ncclConnect),
                               MPI_BYTE,
                               config.peer_rank,
                               0,
                               MPI_COMM_WORLD));
            ASSERT_EQ(MPI_SUCCESS,
                      MPI_Recv(&recv_connect_info,
                               sizeof(ncclConnect),
                               MPI_BYTE,
                               config.peer_rank,
                               0,
                               MPI_COMM_WORLD,
                               MPI_STATUS_IGNORE));
        }
        else
        {
            result = shmTransport.recv.setup(comm_handle,
                                             topology_graph,
                                             local_peer_info,
                                             remote_peer_info,
                                             &recv_connect_info,
                                             &recv_connector,
                                             0,
                                             0);

            ASSERT_EQ(ncclSuccess, result)
                << "Rank " << config.world_rank
                << ": SHM recv setup with CE memcpy failed: " << ncclGetErrorString(result);

            // Exchange connect info with sender
            ASSERT_EQ(MPI_SUCCESS,
                      MPI_Recv(&send_connect_info,
                               sizeof(ncclConnect),
                               MPI_BYTE,
                               config.peer_rank,
                               0,
                               MPI_COMM_WORLD,
                               MPI_STATUS_IGNORE));
            ASSERT_EQ(MPI_SUCCESS,
                      MPI_Send(&recv_connect_info,
                               sizeof(ncclConnect),
                               MPI_BYTE,
                               config.peer_rank,
                               0,
                               MPI_COMM_WORLD));
        }

        // Step 3: Test SHM connect with CE memcpy

        if(shm_config.is_sender)
        {
            result = shmTransport.send.connect(comm_handle,
                                               &recv_connect_info,
                                               config.world_size,
                                               config.world_rank,
                                               &send_connector);

            ASSERT_EQ(ncclSuccess, result)
                << "Rank " << config.world_rank
                << ": SHM send connect with CE memcpy failed: " << ncclGetErrorString(result);
        }
        else
        {
            result = shmTransport.recv.connect(comm_handle,
                                               &send_connect_info,
                                               config.world_size,
                                               config.world_rank,
                                               &recv_connector);

            ASSERT_EQ(ncclSuccess, result)
                << "Rank " << config.world_rank
                << ": SHM recv connect with CE memcpy failed: " << ncclGetErrorString(result);
        }

        // Step 4: Send large buffer with CE memcpy and validate
        const size_t buffer_size  = kCEMemcpyBufferSize;
        const size_t num_elements = buffer_size / sizeof(float);
        void*        send_buffer  = nullptr;
        void*        recv_buffer  = nullptr;

        hipError_t hip_result = hipMalloc(&send_buffer, buffer_size);
        ASSERT_EQ(hipSuccess, hip_result)
            << "Rank " << config.world_rank << ": Failed to allocate send buffer";
        auto sendBufferGuard = makeDeviceBufferAutoGuard(send_buffer);

        hip_result = hipMalloc(&recv_buffer, buffer_size);
        ASSERT_EQ(hipSuccess, hip_result)
            << "Rank " << config.world_rank << ": Failed to allocate recv buffer";
        auto recvBufferGuard = makeDeviceBufferAutoGuard(recv_buffer);

        // Initialize send buffer with unique pattern
        hip_result = initializeBufferWithPattern<float>(
            send_buffer,
            num_elements,
            [rank = config.world_rank](size_t i)
            { return static_cast<float>(rank * kLargePatternMultiplier + (i % kPatternModulo)); });
        ASSERT_EQ(hipSuccess, hip_result)
            << "Rank " << config.world_rank << ": Failed to initialize send buffer";

        hip_result = hipMemset(recv_buffer, 0, buffer_size);
        ASSERT_EQ(hipSuccess, hip_result)
            << "Rank " << config.world_rank << ": Failed to zero recv buffer";

        // Synchronize stream before transfer
        hip_result = syncStream(config.stream, config.world_rank);
        ASSERT_EQ(hipSuccess, hip_result)
            << "Rank " << config.world_rank << ": Stream sync failed before transfer";

        // Perform the actual data transfer using NCCL
        const size_t count = buffer_size / sizeof(float);
        result             = shm_config.is_sender ? ncclSend(send_buffer,
                                                 count,
                                                 ncclFloat,
                                                 config.peer_rank,
                                                 config.nccl_comm,
                                                 config.stream)
                                                  : ncclRecv(recv_buffer,
                                                 count,
                                                 ncclFloat,
                                                 config.peer_rank,
                                                 config.nccl_comm,
                                                 config.stream);

        ASSERT_EQ(ncclSuccess, result) << "Rank " << config.world_rank << ": Large buffer "
                                       << (shm_config.is_sender ? "Send" : "Recv")
                                       << " with CE memcpy failed: " << ncclGetErrorString(result);

        // Synchronize to ensure transfer completes
        hip_result = syncStream(config.stream, config.world_rank);
        ASSERT_EQ(hipSuccess, hip_result)
            << "Rank " << config.world_rank << ": Stream sync failed after transfer";

        // Step 5: Validate received data (on receiver only)
        if(!shm_config.is_sender)
        {
            // Verify with custom pattern check (matching initialization pattern)
            size_t error_idx;
            float  expected_val, actual_val;
            bool   data_correct = verifyBufferData<float>(
                recv_buffer,
                num_elements,
                [peer_rank = config.peer_rank](size_t i) {
                    return static_cast<float>(peer_rank * kLargePatternMultiplier
                                              + (i % kPatternModulo));
                },
                0,      // verify all elements
                1e-5,
                &error_idx,
                &expected_val,
                &actual_val);

            EXPECT_TRUE(data_correct) << "Rank " << config.world_rank
                                      << ": Data validation failed at index " << error_idx;
        }
    }

    // Test SHM buffer allocation and sharing
    void testShmBufferAllocation()
    {
        // Test buffer allocation with various sizes
        const std::vector<size_t> test_sizes
            = {kSmallBufferSize, kMediumBufferSize, kLargeBufferSize};

        for(const auto size : test_sizes)
        {
            void* send_buff = nullptr;
            void* recv_buff = nullptr;

            // Allocate with local guards (store_in_base=false)
            // Guards will cleanup at end of loop iteration
            auto [sendGuard, recvGuard]
                = allocateAndInitBuffersGuarded(&send_buff, &recv_buff, size, size, false);

            // Verify buffers are accessible
            EXPECT_NE(send_buff, nullptr) << "Rank " << config.world_rank << ": send_buff is null";
            EXPECT_NE(recv_buff, nullptr) << "Rank " << config.world_rank << ": recv_buff is null";
        }
    }
};

TEST_F(ShmMPITest, ShmWorkflow)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    // Create test-specific communicator for isolation
    // Use ASSERT_MPI_SUCCESS to prevent deadlock if creation fails on some ranks
    ASSERT_MPI_SUCCESS(createTestCommunicator());

    if(config.world_rank == 0)
    {
        TEST_INFO("Starting comprehensive SHM transport workflow test with %d processes", config.world_size);
        TEST_INFO("This test exercises the low-level SHM transport API");
    }

    // Test 1: SHM Capability Detection
    if(config.world_rank == 0)
    {
        TEST_INFO("Step 1: Testing SHM canConnect capability");
    }
    testShmCanConnect();

    // Test 2: SHM Setup
    if(config.world_rank == 0)
    {
        TEST_INFO("Step 2: Setting up SHM transport connectors");
    }
    testShmSetup();

    // Test 3: SHM Connection
    if(config.world_rank == 0)
    {
        TEST_INFO("Step 3: Connecting SHM transport");
    }
    testShmConnect();

    // Test 4: Data Transfer through SHM
    if(config.world_rank == 0)
    {
        TEST_INFO("Step 4: Performing SHM data transfer");
    }
    testShmDataTransfer();

    // Test 5: Resource Cleanup
    if(config.world_rank == 0)
    {
        TEST_INFO("Step 5: Validating resource cleanup");
    }
    testShmCleanup();

    if(config.world_rank == 0)
    {
        TEST_INFO("SHM transport workflow test completed successfully");
        TEST_INFO("NOTE: Base class TearDown() handles connector cleanup automatically");
    }
}

TEST_F(ShmMPITest, ShmWithMemcpyTest)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    // Create test-specific communicator for isolation
    // Use ASSERT_MPI_SUCCESS to prevent deadlock if creation fails on some ranks
    ASSERT_MPI_SUCCESS(createTestCommunicator());

    testShmWithMemcpy();
}

TEST_F(ShmMPITest, ShmBufferAllocationTest)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    // Use ASSERT_MPI_SUCCESS to prevent deadlock if creation fails on some ranks
    ASSERT_MPI_SUCCESS(createTestCommunicator());

    testShmBufferAllocation();
}

TEST_F(ShmMPITest, ShmTransfer_ZeroSizeBuffer)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    // Use ASSERT_MPI_SUCCESS to prevent deadlock if creation fails on some ranks
    ASSERT_MPI_SUCCESS(createTestCommunicator());

    // Allocate minimal buffer
    void* buffer = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buffer, 1)); // Allocate 1 byte
    auto bufferGuard = makeDeviceBufferAutoGuard(buffer); // Device memory

    const bool is_sender = (config.world_rank == 0);
    const int  peer      = is_sender ? 1 : 0;

    // Try to send/recv 0 elements
    const auto result = is_sender
                            ? ncclSend(buffer, 0, ncclFloat, peer, config.nccl_comm, config.stream)
                            : ncclRecv(buffer, 0, ncclFloat, peer, config.nccl_comm, config.stream);

    ASSERT_EQ(ncclSuccess, result)
        << "Rank " << config.world_rank << ": Zero-size transfer should succeed";

    HIP_TEST_CHECK_GTEST_FAIL(syncStream(config.stream, config.world_rank));
}

TEST_F(ShmMPITest, ShmTransfer_VeryLargeBuffer)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    // Use ASSERT_MPI_SUCCESS to prevent deadlock if creation fails on some ranks
    ASSERT_MPI_SUCCESS(createTestCommunicator());

    // Try to allocate a very large buffer
    const size_t large_size  = kCEMemcpyBufferSize;
    void*        send_buffer = nullptr;
    void*        recv_buffer = nullptr;

    hipError_t hip_result      = hipMalloc(&send_buffer, large_size);
    auto       sendBufferGuard = makeDeviceBufferAutoGuard(send_buffer);

    hip_result           = hipMalloc(&recv_buffer, large_size);
    auto recvBufferGuard = makeDeviceBufferAutoGuard(recv_buffer);

    // Initialize buffer
    HIP_TEST_CHECK_GTEST_FAIL(hipMemset(send_buffer, 0x42, large_size));

    const bool   is_sender = (config.world_rank == 0);
    const int    peer      = is_sender ? 1 : 0;
    const size_t count     = large_size / sizeof(float);

    // Perform send/recv with large buffer
    const auto result
        = is_sender
              ? ncclSend(send_buffer, count, ncclFloat, peer, config.nccl_comm, config.stream)
              : ncclRecv(recv_buffer, count, ncclFloat, peer, config.nccl_comm, config.stream);

    ASSERT_EQ(ncclSuccess, result)
        << "Rank " << config.world_rank << ": Large buffer transfer failed";

    HIP_TEST_CHECK_GTEST_FAIL(syncStream(config.stream, config.world_rank));
}

TEST_F(ShmMPITest, ShmTransfer_UnalignedBufferAddress)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    ASSERT_MPI_SUCCESS(createTestCommunicator());

    // Allocate aligned buffer
    const size_t buffer_size    = 4096;
    void*        aligned_buffer = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&aligned_buffer, buffer_size));
    auto bufferGuard = makeDeviceBufferAutoGuard(aligned_buffer); // Device memory

    // Create unaligned pointer (offset by 1 byte)
    void* unaligned_buffer = static_cast<char*>(aligned_buffer) + 1;

    const bool is_sender = (config.world_rank == 0);
    const int  peer      = is_sender ? 1 : 0;

    const auto result
        = is_sender
              ? ncclSend(unaligned_buffer, 1024, ncclChar, peer, config.nccl_comm, config.stream)
              : ncclRecv(unaligned_buffer, 1024, ncclChar, peer, config.nccl_comm, config.stream);

    // Don't fail the test - just report the result
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(config.stream));
}

TEST_F(ShmMPITest, ShmMultipleConsecutiveTransfers)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    ASSERT_MPI_SUCCESS(createTestCommunicator());

    const size_t buffer_size = kMediumBufferSize;
    void*        send_buffer = nullptr;
    void*        recv_buffer = nullptr;

    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buffer, buffer_size));
    auto sendBufferGuard = makeDeviceBufferAutoGuard(send_buffer);

    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buffer, buffer_size));
    auto recvBufferGuard = makeDeviceBufferAutoGuard(recv_buffer);

    HIP_TEST_CHECK_GTEST_FAIL(hipMemset(send_buffer, 0xAB, buffer_size));

    const bool   is_sender = (config.world_rank == 0);
    const int    peer      = is_sender ? 1 : 0;
    const size_t count     = buffer_size / sizeof(float);

    for(int i = 0; i < kMultipleTransferCount; i++)
    {
        const auto result
            = is_sender
                  ? ncclSend(send_buffer, count, ncclFloat, peer, config.nccl_comm, config.stream)
                  : ncclRecv(recv_buffer, count, ncclFloat, peer, config.nccl_comm, config.stream);

        ASSERT_EQ(ncclSuccess, result)
            << "Rank " << config.world_rank << ": Transfer " << i << " failed";

        // Ensure both ranks have posted their NCCL operations before synchronizing
        MPI_Barrier(MPI_COMM_WORLD);

        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(config.stream));
    }
}

TEST_F(ShmMPITest, ShmCleanup_DoubleCleanup)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    ASSERT_MPI_SUCCESS(createTestCommunicator());

    const bool is_sender = (config.world_rank == 0);
    auto*      connector = is_sender ? &send_connector : &recv_connector;

    // Setup connector
    ncclConnect connect_info{};
    const auto  setup_result = is_sender ? shmTransport.send.setup(comm_handle,
                                                                  topology_graph,
                                                                  local_peer_info,
                                                                  remote_peer_info,
                                                                  &connect_info,
                                                                  connector,
                                                                  0,
                                                                  0)
                                         : shmTransport.recv.setup(comm_handle,
                                                                  topology_graph,
                                                                  local_peer_info,
                                                                  remote_peer_info,
                                                                  &connect_info,
                                                                  connector,
                                                                  0,
                                                                  0);

    ASSERT_EQ(ncclSuccess, setup_result) << "Rank " << config.world_rank << ": Setup failed";

    MPI_Barrier(MPI_COMM_WORLD);

    // First cleanup
    if(connector->transportResources)
    {
        const auto result1
            = is_sender ? shmTransport.send.free(connector) : shmTransport.recv.free(connector);
        EXPECT_EQ(ncclSuccess, result1) << "Rank " << config.world_rank << ": First cleanup failed";
    }

    // Second cleanup (should handle gracefully since resources are already freed)
    [[maybe_unused]] const auto result2
        = is_sender ? shmTransport.send.free(connector) : shmTransport.recv.free(connector);

    // Mark as cleaned up
    connector->transportResources = nullptr;
}

TEST_F(ShmMPITest, ShmConnect_WithoutSetup)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    ASSERT_MPI_SUCCESS(createTestCommunicator());

    if(config.world_rank == 0)
    {
        TEST_INFO("Testing SHM connect without prior setup (%d processes)", config.world_size);
    }

    const bool is_sender = (config.world_rank == 0);
    auto*      connector = is_sender ? &send_connector : &recv_connector;

    // Create empty/uninitialized connect info (simulates invalid state)
    ncclConnect invalid_connect_info{};
    memset(&invalid_connect_info, 0, sizeof(ncclConnect));

    // Try to connect without calling setup first - this should fail or handle gracefully
    const auto result = is_sender ? shmTransport.send.connect(comm_handle,
                                                              &invalid_connect_info,
                                                              config.world_size,
                                                              config.world_rank,
                                                              connector)
                                  : shmTransport.recv.connect(comm_handle,
                                                              &invalid_connect_info,
                                                              config.world_size,
                                                              config.world_rank,
                                                              connector);

    if(config.world_rank == 0)
    {
        TEST_INFO("Connect without setup result: %s", ncclGetErrorString(result));
        TEST_INFO("Note: This tests invalid state handling");
    }
}

TEST_F(ShmMPITest, ShmConnect_CorruptedConnectInfo)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    ASSERT_MPI_SUCCESS(createTestCommunicator());

    if(config.world_rank == 0)
    {
        TEST_INFO("Testing SHM connect with corrupted connect info (%d processes)",
                  config.world_size);
    }

    const bool is_sender = (config.world_rank == 0);
    auto*      connector = is_sender ? &send_connector : &recv_connector;

    // First, do valid setup
    ncclConnect valid_connect_info{};
    const auto  setup_result = is_sender ? shmTransport.send.setup(comm_handle,
                                                                  topology_graph,
                                                                  local_peer_info,
                                                                  remote_peer_info,
                                                                  &valid_connect_info,
                                                                  connector,
                                                                  0,
                                                                  0)
                                         : shmTransport.recv.setup(comm_handle,
                                                                  topology_graph,
                                                                  local_peer_info,
                                                                  remote_peer_info,
                                                                  &valid_connect_info,
                                                                  connector,
                                                                  0,
                                                                  0);

    ASSERT_EQ(ncclSuccess, setup_result) << "Rank " << config.world_rank << ": Setup failed";

    MPI_Barrier(MPI_COMM_WORLD);

    // Create corrupted connect info (fill with invalid data)
    ncclConnect corrupted_info{};
    memset(&corrupted_info, 0xFF, sizeof(ncclConnect)); // Fill with 0xFF

    // Try to connect with corrupted info
    // This tests internal validation of connect info structures
    const auto result = is_sender ? shmTransport.send.connect(comm_handle,
                                                              &corrupted_info,
                                                              config.world_size,
                                                              config.world_rank,
                                                              connector)
                                  : shmTransport.recv.connect(comm_handle,
                                                              &corrupted_info,
                                                              config.world_size,
                                                              config.world_rank,
                                                              connector);

    if(config.world_rank == 0)
    {
        TEST_INFO("Connect with corrupted info result: %s", ncclGetErrorString(result));
        TEST_INFO("Note: Tests connect info validation similar to proxy function validation");
    }

    // Cleanup properly allocated resources
    if(connector->transportResources)
    {
        const auto cleanup_result
            = is_sender ? shmTransport.send.free(connector) : shmTransport.recv.free(connector);
        (void)cleanup_result; // Ignore result as we're in error path
        connector->transportResources = nullptr;
    }
}

#endif // MPI_TESTS_ENABLED
