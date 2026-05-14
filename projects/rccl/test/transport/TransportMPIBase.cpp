/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "TransportMPIBase.hpp"

#ifdef MPI_TESTS_ENABLED

namespace
{
// Test pattern generation constants for TransportTestBase
inline constexpr int kDefaultPatternMultiplier = 100; // For transport base patterns
inline constexpr int kByteValueModulo          = 256; // For uint8_t wraparound
} // namespace

// Override createTestCommunicator to also update config and transport components
ncclResult_t TransportTestBase::createTestCommunicator()
{
    // Call base class implementation
    ncclResult_t result = MPITestBase::createTestCommunicator();

    if(result == ncclSuccess)
    {
        // Update config with the new communicator and stream
        config.nccl_comm = getActiveCommunicator();
        config.stream    = getActiveStream();

        // Initialize transport components now that we have a valid communicator
        comm_handle      = config.nccl_comm;
        local_peer_info  = &comm_handle->peerInfo[config.world_rank];
        remote_peer_info = &comm_handle->peerInfo[config.peer_rank];

        if(config.world_rank == 0)
        {
            TEST_INFO("TransportTestBase config and transport components updated with per-test "
                      "communicator");
        }
    }

    return result;
}

// Set transport type and initialize connectors accordingly
void TransportTestBase::setTransportType(TransportType type)
{
    initialized_transport = type;

    switch(type)
    {
        case TransportType::P2P:
            send_connector.transportComm = &p2pTransport.send;
            recv_connector.transportComm = &p2pTransport.recv;
            break;
        case TransportType::Network:
            send_connector.transportComm = &netTransport.send;
            recv_connector.transportComm = &netTransport.recv;
            break;
        case TransportType::SHM:
            send_connector.transportComm = &shmTransport.send;
            recv_connector.transportComm = &shmTransport.recv;
            break;
        case TransportType::None:
            send_connector.transportComm = nullptr;
            recv_connector.transportComm = nullptr;
            break;
    }
}

// SetUp: Initialize common transport test components
void TransportTestBase::SetUp()
{
    // Call GTest's SetUp (which will call MPITestCore::initializeTest())
    MPITestBase::SetUp();

    // Initialize test configuration using aggregate initialization
    // Note: rccl_comm and stream are set to nullptr initially; tests must call createTestCommunicator()
    config = {.world_rank = MPIEnvironment::world_rank,
              .world_size = MPIEnvironment::world_size,
              .peer_rank  = (MPIEnvironment::world_rank == 0) ? 1 : 0,
              .nccl_comm  = nullptr,
              .stream     = nullptr};

    // Require at least 2 MPI processes for testing
    if(config.world_size < 2)
    {
        GTEST_SKIP() << "Transport testing requires at least 2 MPI processes";
    }

    // Check if MPIEnvironment was properly initialized
    if(MPIEnvironment::retCode != 0)
    {
        GTEST_FAIL() << "MPIEnvironment initialization failed";
    }

    // Initialize transport component pointers to nullptr
    // They will be set in createTestCommunicator() after the communicator is created
    comm_handle      = nullptr;
    local_peer_info  = nullptr;
    remote_peer_info = nullptr;

    // Create and initialize topology graph
    topology_graph = static_cast<ncclTopoGraph*>(malloc(sizeof(ncclTopoGraph)));
    if(topology_graph)
    {
        *topology_graph = {.id        = 0,
                           .pattern   = NCCL_TOPO_PATTERN_RING,
                           .nChannels = 1,
                           .bwIntra   = 0.0f,
                           .bwInter   = 0.0f,
                           .typeIntra = PATH_SYS,
                           .typeInter = PATH_NET};
    }

    // Initialize with P2P transport by default
    // Tests can call setTransportType() to switch to SHM or Network
    setTransportType(TransportType::P2P);
}

// TearDown: Cleanup common transport test components
void TransportTestBase::TearDown()
{
    // CRITICAL: Synchronize device before freeing connectors
    // The transport proxy may have its own internal stream for CE memcpy operations
    // that must be idle before we can destroy it
    // Note: We ignore errors here as we're in cleanup path
    (void)hipDeviceSynchronize();

    // Cleanup topology graph
    if(topology_graph)
    {
        free(topology_graph);
        topology_graph = nullptr;
    }

    // Cleanup transport resources based on initialized transport type
    if(send_connector.transportResources)
    {
        if(initialized_transport == TransportType::P2P)
        {
            p2pTransport.send.free(&send_connector);
        }
        else if(initialized_transport == TransportType::SHM)
        {
            shmTransport.send.free(&send_connector);
        }
        else if(initialized_transport == TransportType::Network)
        {
            netTransport.send.free(&send_connector);
        }
        send_connector.transportResources = nullptr;
    }
    if(recv_connector.transportResources)
    {
        if(initialized_transport == TransportType::P2P)
        {
            p2pTransport.recv.free(&recv_connector);
        }
        else if(initialized_transport == TransportType::SHM)
        {
            shmTransport.recv.free(&recv_connector);
        }
        else if(initialized_transport == TransportType::Network)
        {
            netTransport.recv.free(&recv_connector);
        }
        recv_connector.transportResources = nullptr;
    }

    // Reset transport type
    initialized_transport = TransportType::None;

    // Nullify peer info pointers
    local_peer_info  = nullptr;
    remote_peer_info = nullptr;
    comm_handle      = nullptr;

    // Note: Clear RAII guard vectors BEFORE destroying communicator
    // The guards (especially NcclRegHandleGuard) need the communicator to be valid
    // when they call ncclCommDeregister() in their destructors
    reg_handle_guards_.clear();
    buffer_guards_.clear();

    // Call base class TearDown to cleanup test communicator
    // This calls MPITestBase::TearDown() -> MPITestCore::cleanupTest() -> cleanupTestCommunicator()
    MPITestBase::TearDown();
}

// Allocate and initialize test buffers
void TransportTestBase::allocateAndInitBuffers(void** send_buffer,
                                               void** recv_buffer,
                                               size_t send_bytes,
                                               size_t recv_bytes)
{
    // Allocate send buffer
    ASSERT_EQ(hipSuccess, hipMalloc(send_buffer, send_bytes))
        << "Rank " << config.world_rank << ": Failed to allocate send buffer";

    // Allocate recv buffer
    ASSERT_EQ(hipSuccess, hipMalloc(recv_buffer, recv_bytes))
        << "Rank " << config.world_rank << ": Failed to allocate recv buffer";

    std::vector<uint8_t> host_data(send_bytes);
    for(size_t i = 0; i < host_data.size(); i++)
    {
        host_data[i] = static_cast<uint8_t>((config.world_rank * kDefaultPatternMultiplier + i)
                                            % kByteValueModulo);
    }

    ASSERT_EQ(hipSuccess,
              hipMemcpy(*send_buffer, host_data.data(), send_bytes, hipMemcpyHostToDevice))
        << "Rank " << config.world_rank << ": Failed to initialize send buffer";

    if(config.world_rank == 0)
    {
        TEST_INFO("Allocated and initialized buffers (%zu bytes each)", send_bytes);
    }
}

// Pre-register buffers with ncclCommRegister
void TransportTestBase::preRegisterBuffers(void*  send_buffer,
                                           void*  recv_buffer,
                                           size_t send_bytes,
                                           size_t recv_bytes,
                                           void** send_reg_handle,
                                           void** recv_reg_handle)
{
    ncclComm_t comm = getActiveCommunicator();

    // Register send buffer
    ncclResult_t result = ncclCommRegister(comm, send_buffer, send_bytes, send_reg_handle);
    ASSERT_EQ(ncclSuccess, result)
        << "Rank " << config.world_rank
        << ": Failed to pre-register send buffer: " << ncclGetErrorString(result);

    // Register recv buffer
    result = ncclCommRegister(comm, recv_buffer, recv_bytes, recv_reg_handle);
    ASSERT_EQ(ncclSuccess, result)
        << "Rank " << config.world_rank
        << ": Failed to pre-register recv buffer: " << ncclGetErrorString(result);
}

// Buffer allocation with automatic RAII guards
std::pair<DeviceBufferAutoGuard, DeviceBufferAutoGuard>
    TransportTestBase::allocateAndInitBuffersGuarded(void** send_buffer,
                                                     void** recv_buffer,
                                                     size_t send_bytes,
                                                     size_t recv_bytes,
                                                     bool   store_in_base)
{
    // Allocate buffers using existing method
    allocateAndInitBuffers(send_buffer, recv_buffer, send_bytes, recv_bytes);

    // Create guards
    auto sendGuard = makeDeviceBufferAutoGuard(*send_buffer); // Device memory
    auto recvGuard = makeDeviceBufferAutoGuard(*recv_buffer); // Device memory

    if(store_in_base)
    {
        // Store guards in base class for cleanup at test end
        buffer_guards_.push_back(std::move(sendGuard));
        buffer_guards_.push_back(std::move(recvGuard));

        // Return empty guards (resources now managed by base class)
        return {makeDeviceBufferAutoGuard(nullptr), makeDeviceBufferAutoGuard(nullptr)};
    }
    else
    {
        // Return guards for caller to manage (cleanup at caller's scope exit)
        return {std::move(sendGuard), std::move(recvGuard)};
    }
}

// Buffer registration with automatic RAII guards
std::pair<NcclRegHandleGuard, NcclRegHandleGuard>
    TransportTestBase::preRegisterBuffersGuarded(void*  send_buffer,
                                                 void*  recv_buffer,
                                                 size_t send_bytes,
                                                 size_t recv_bytes,
                                                 void** send_reg_handle,
                                                 void** recv_reg_handle,
                                                 bool   store_in_base)
{
    // Register buffers using existing method
    preRegisterBuffers(send_buffer,
                       recv_buffer,
                       send_bytes,
                       recv_bytes,
                       send_reg_handle,
                       recv_reg_handle);

    // Create guards (handles may be nullptr if registration is not needed)
    NcclRegHandleGuard sendGuard(*send_reg_handle, NcclRegHandleDeleter(getActiveCommunicator()));
    NcclRegHandleGuard recvGuard(*recv_reg_handle, NcclRegHandleDeleter(getActiveCommunicator()));

    if(store_in_base)
    {
        // Store guards in base class for cleanup at test end
        reg_handle_guards_.push_back(std::move(sendGuard));
        reg_handle_guards_.push_back(std::move(recvGuard));

        // Return empty guards (resources now managed by base class)
        return {makeRegHandleGuard(nullptr, nullptr), makeRegHandleGuard(nullptr, nullptr)};
    }
    else
    {
        // Return guards for caller to manage (cleanup at caller's scope exit)
        return {std::move(sendGuard), std::move(recvGuard)};
    }
}

#endif // MPI_TESTS_ENABLED
