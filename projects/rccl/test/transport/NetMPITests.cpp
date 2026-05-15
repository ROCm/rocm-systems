/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "DeviceBufferHelpers.hpp"
#include "TestChecks.hpp"
#include "TransportMPIBase.hpp"

#ifdef MPI_TESTS_ENABLED

// Import MPI test constants
using namespace MPITestConstants;
using namespace RCCLTestHelpers;

// NET-specific RAII deleters
namespace RCCLTestGuards
{

struct NetMHandleDeleter
{
    ncclNet_t* net;
    void*      comm;
    NetMHandleDeleter(ncclNet_t* n = nullptr, void* c = nullptr) : net(n), comm(c) {}
    void operator()(void* mhandle) const
    {
        if(mhandle && comm && net)
        {
            net->deregMr(comm, mhandle);
        }
    }
};

struct NetSendCommDeleter
{
    ncclNet_t* net;
    explicit NetSendCommDeleter(ncclNet_t* n = nullptr) : net(n) {}
    void operator()(void* comm) const
    {
        if(comm && net)
            net->closeSend(comm);
    }
};

struct NetRecvCommDeleter
{
    ncclNet_t* net;
    explicit NetRecvCommDeleter(ncclNet_t* n = nullptr) : net(n) {}
    void operator()(void* comm) const
    {
        if(comm && net)
            net->closeRecv(comm);
    }
};

struct NetListenCommDeleter
{
    ncclNet_t* net;
    explicit NetListenCommDeleter(ncclNet_t* n = nullptr) : net(n) {}
    void operator()(void* comm) const
    {
        if(comm && net)
            net->closeListen(comm);
    }
};

using NetMHandleGuard    = ResourceGuard<void*, NetMHandleDeleter>;
using NetSendCommGuard   = ResourceGuard<void*, NetSendCommDeleter>;
using NetRecvCommGuard   = ResourceGuard<void*, NetRecvCommDeleter>;
using NetListenCommGuard = ResourceGuard<void*, NetListenCommDeleter>;

class NetConnectionGuard
{
private:
    ncclNet_t* net_;
    void*      listen_comm_;
    void*      send_comm_;
    void*      recv_comm_;

public:
    explicit NetConnectionGuard(ncclNet_t* net)
        : net_(net), listen_comm_(nullptr), send_comm_(nullptr), recv_comm_(nullptr)
    {}

    ~NetConnectionGuard()
    {
        if(recv_comm_ && net_)
            net_->closeRecv(recv_comm_);
        if(send_comm_ && net_)
            net_->closeSend(send_comm_);
        if(listen_comm_ && net_)
            net_->closeListen(listen_comm_);
    }

    void setListenComm(void* comm)
    {
        listen_comm_ = comm;
    }
    void setSendComm(void* comm)
    {
        send_comm_ = comm;
    }
    void setRecvComm(void* comm)
    {
        recv_comm_ = comm;
    }

    void* getListenComm() const
    {
        return listen_comm_;
    }
    void* getSendComm() const
    {
        return send_comm_;
    }
    void* getRecvComm() const
    {
        return recv_comm_;
    }

    NetConnectionGuard(const NetConnectionGuard&)            = delete;
    NetConnectionGuard& operator=(const NetConnectionGuard&) = delete;
    NetConnectionGuard(NetConnectionGuard&&)                 = delete;
    NetConnectionGuard& operator=(NetConnectionGuard&&)      = delete;
};

inline NetMHandleGuard makeNetMHandleGuard(void* mhandle, ncclNet_t* net, void* comm)
{
    return NetMHandleGuard(mhandle, NetMHandleDeleter(net, comm));
}

inline NetSendCommGuard makeNetSendCommGuard(void* comm, ncclNet_t* net)
{
    return NetSendCommGuard(comm, NetSendCommDeleter(net));
}

inline NetRecvCommGuard makeNetRecvCommGuard(void* comm, ncclNet_t* net)
{
    return NetRecvCommGuard(comm, NetRecvCommDeleter(net));
}

inline NetListenCommGuard makeNetListenCommGuard(void* comm, ncclNet_t* net)
{
    return NetListenCommGuard(comm, NetListenCommDeleter(net));
}

} // namespace RCCLTestGuards

namespace
{
// Buffer size constants
inline constexpr size_t kTestBufferSize = 16384;

// NET transport test requirements
inline constexpr int kMinNodesForNET   = 2; // NET transport requires at least 2 nodes
inline constexpr int kExactRanksForNET = 2; // NET transport tests use exactly 2 ranks (1 per node)

// Test pattern generation constants
inline constexpr int kDefaultPatternMultiplier = 100; // For NET transport patterns
inline constexpr int kByteValueModulo          = 256; // For uint8_t wraparound

} // namespace

class NetTransportMPITest : public TransportTestBase
{
protected:
    void SetUp() override
    {
        TransportTestBase::SetUp();
        if(config.world_rank == 0)
        {
            TEST_INFO("NetTransport SetUp completed");
        }
    }

    void TearDown() override
    {
        if(config.world_rank == 0)
        {
            TEST_INFO("NetTransport TearDown completed");
        }
        TransportTestBase::TearDown();
    }

public:
    // Test ncclNetGraphRegisterBuffer
    void testNetGraphRegisterBuffer()
    {
        if(config.world_rank == 0)
        {
            TEST_INFO("Testing ncclNetGraphRegisterBuffer...");
        }

        // Verify communicator is ready
        ASSERT_NE(comm_handle, nullptr) << "Rank " << config.world_rank << ": comm_handle is null";

        // Allocate and automatically guard buffers
        void* send_buffer = nullptr;
        void* recv_buffer = nullptr;
        allocateAndInitBuffersGuarded(&send_buffer, &recv_buffer, kTestBufferSize, kTestBufferSize);

        // Register and automatically guard handles
        void* send_reg_handle = nullptr;
        void* recv_reg_handle = nullptr;
        preRegisterBuffersGuarded(send_buffer,
                                  recv_buffer,
                                  kTestBufferSize,
                                  kTestBufferSize,
                                  &send_reg_handle,
                                  &recv_reg_handle);

        // Test ncclNetGraphRegisterBuffer
        int                                                       net_reg_flag{};
        void*                                                     net_handle{};
        ncclIntruQueue<ncclCommCallback, &ncclCommCallback::next> cleanup_queue{};
        int                                                       n_cleanup_elts{};

        ncclConnector* send_conn_array[1] = {&send_connector};

        auto nccl_result
            = ncclNetGraphRegisterBuffer(reinterpret_cast<ncclComm*>(getActiveCommunicator()),
                                         send_buffer,
                                         kTestBufferSize,
                                         send_conn_array,
                                         1,
                                         &net_reg_flag,
                                         &net_handle,
                                         &cleanup_queue,
                                         &n_cleanup_elts);

        EXPECT_EQ(ncclSuccess, nccl_result)
            << "Rank " << config.world_rank
            << ": ncclNetGraphRegisterBuffer failed: " << ncclGetErrorString(nccl_result);

        if(config.world_rank == 0)
        {
            TEST_INFO("    ncclNetGraphRegisterBuffer returned: %s",
                      ncclGetErrorString(nccl_result));
            TEST_INFO("    Registration flag: %d", net_reg_flag);
            TEST_INFO("    Handle: %p", net_handle);
            TEST_INFO("    Cleanup queue elements: %d", n_cleanup_elts);
        }

        if(config.world_rank == 0)
        {
            TEST_INFO("ncclNetGraphRegisterBuffer test completed");
        }
    }

    // Test ncclNetLocalRegisterBuffer
    void testNetLocalRegisterBuffer()
    {
        if(config.world_rank == 0)
        {
            TEST_INFO("Testing ncclNetLocalRegisterBuffer...");
            TEST_INFO("This API internally calls ncclNetLocalRegisterBuffer "
                      "and ncclNetLocalRegisterBuffer");
        }

        // Verify communicator is ready (NCCL has already initialized NET transport)
        ASSERT_NE(comm_handle, nullptr) << "Rank " << config.world_rank << ": comm_handle is null";

        // Allocate and automatically guard buffers
        void* send_buffer = nullptr;
        void* recv_buffer = nullptr;
        allocateAndInitBuffersGuarded(&send_buffer, &recv_buffer, kTestBufferSize, kTestBufferSize);

        // Register and automatically guard handles
        void* send_reg_handle = nullptr;
        void* recv_reg_handle = nullptr;
        preRegisterBuffersGuarded(send_buffer,
                                  recv_buffer,
                                  kTestBufferSize,
                                  kTestBufferSize,
                                  &send_reg_handle,
                                  &recv_reg_handle);

        // Test ncclNetLocalRegisterBuffer
        int   net_reg_flag{};
        void* net_handle{};

        ncclConnector* send_conn_array[1] = {&send_connector};

        auto nccl_result
            = ncclNetLocalRegisterBuffer(reinterpret_cast<ncclComm*>(getActiveCommunicator()),
                                         send_buffer,
                                         kTestBufferSize,
                                         send_conn_array,
                                         1, // nPeers
                                         &net_reg_flag,
                                         &net_handle);

        EXPECT_EQ(ncclSuccess, nccl_result)
            << "Rank " << config.world_rank
            << ": ncclNetLocalRegisterBuffer failed: " << ncclGetErrorString(nccl_result);

        if(config.world_rank == 0)
        {
            TEST_INFO("    ncclNetLocalRegisterBuffer returned: %s",
                      ncclGetErrorString(nccl_result));
            TEST_INFO("    Registration flag: %d", net_reg_flag);
            TEST_INFO("    Handle: %p", net_handle);
        }
    }

    // Test multiple buffer sizes with actual data transfer
    void testMultipleBufferSizes()
    {
        if(config.world_rank == 0)
        {
            TEST_INFO("Testing multiple buffer sizes (aligned and unaligned) with NET "
                      "transport and data transfer...");
        }

        // Verify communicator is ready
        ASSERT_NE(comm_handle, nullptr) << "Rank " << config.world_rank << ": comm_handle is null";

        // Test both aligned and unaligned buffer sizes to validate edge cases
        std::vector<size_t> sizes = {
            // Small sizes (including unaligned)
            1, // Minimum size
            3, // Unaligned (not power of 2)
            7, // Unaligned
            15, // Unaligned
            63, // Unaligned

            // Medium sizes (mix of aligned and unaligned)
            1024, // 1KB (aligned)
            1025, // 1KB + 1 (unaligned)
            1536, // 1.5KB (unaligned)
            4096, // 4KB (aligned)
            4097, // 4KB + 1 (unaligned)
            5000, // Unaligned
            16384, // 16KB (aligned)
            16385, // 16KB + 1 (unaligned)

            // Large sizes (mix of aligned and unaligned)
            65536, // 64KB (aligned)
            65537, // 64KB + 1 (unaligned)
            100000, // ~98KB (unaligned)
            262144, // 256KB (aligned)
            262145, // 256KB + 1 (unaligned)
            500000, // ~488KB (unaligned)
            1048576, // 1MB (aligned)
            1048577, // 1MB + 1 (unaligned)
            4 * 1024 * 1024, // 4MB (aligned)
            4 * 1024 * 1024 + 1 // 4MB + 1 (unaligned)
        };

        int         peer_rank = (config.world_rank == 0) ? 1 : 0;
        hipStream_t stream    = getActiveStream();
        ASSERT_NE(stream, nullptr) << "Rank " << config.world_rank << ": Stream is null";

        for(size_t size : sizes)
        {
            if(config.world_rank == 0)
            {
                TEST_INFO("  Testing size: %zu bytes with data transfer", size);
            }

            // Allocate buffers with local guards (per-iteration cleanup)
            void* send_buffer = nullptr;
            void* recv_buffer = nullptr;
            auto [sendGuard, recvGuard]
                = allocateAndInitBuffersGuarded(&send_buffer, &recv_buffer, size, size, false);

            ASSERT_NE(send_buffer, nullptr) << "Rank " << config.world_rank
                                            << ": Send buffer allocation failed for size " << size;
            ASSERT_NE(recv_buffer, nullptr) << "Rank " << config.world_rank
                                            << ": Recv buffer allocation failed for size " << size;

            // Initialize send buffer with rank and size-specific pattern
            uint8_t* send_data = static_cast<uint8_t*>(send_buffer);
            for(size_t i = 0; i < size; i++)
            {
                send_data[i] = static_cast<uint8_t>(
                    (config.world_rank * kDefaultPatternMultiplier + i) % kByteValueModulo);
            }

            // Initialize recv buffer with invalid pattern
            uint8_t* recv_data = static_cast<uint8_t*>(recv_buffer);
            for(size_t i = 0; i < size; i++)
            {
                recv_data[i] = 0xFF; // Invalid pattern to detect transfer
            }

            // Perform actual data transfer using NCCL Send/Recv
            // Use ASSERT_MPI_SUCCESS to ensure both ranks synchronize on NCCL errors
            ASSERT_MPI_SUCCESS(ncclGroupStart());

            ASSERT_MPI_SUCCESS(
                ncclSend(send_buffer, size, ncclInt8, peer_rank, getActiveCommunicator(), stream));

            ASSERT_MPI_SUCCESS(
                ncclRecv(recv_buffer, size, ncclInt8, peer_rank, getActiveCommunicator(), stream));

            ASSERT_MPI_SUCCESS(ncclGroupEnd());

            // Wait for transfer to complete
            // Use ASSERT_MPI_EQ to ensure both ranks synchronize on HIP errors
            ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

            // Verify received data matches peer's send pattern
            int       errors              = 0;
            const int max_errors_to_print = 5;
            for(size_t i = 0; i < size && errors < max_errors_to_print; i++)
            {
                uint8_t expected = static_cast<uint8_t>((peer_rank * kDefaultPatternMultiplier + i)
                                                        % kByteValueModulo);
                if(recv_data[i] != expected)
                {
                    TEST_WARN("Size %zu - Data mismatch at index %zu: expected %u, got %u",
                              size,
                              i,
                              expected,
                              recv_data[i]);
                    errors++;
                }
            }

            EXPECT_EQ(0, errors) << "Rank " << config.world_rank
                                 << ": Found data mismatches for buffer size " << size;

            if(config.world_rank == 0 && errors == 0)
            {
                TEST_INFO("  Size %zu - Data transfer successful and verified", size);
            }

            // Resource Guards will automatically cleanup at end of loop iteration
        }
    }
};

// Test cases
TEST_F(NetTransportMPITest, NetGraphRegisterBufferTest)
{
    // NET transport tests require exactly 2 ranks on 2 nodes (1 rank per node)
    if(!validateTestPrerequisites(kExactRanksForNET,
                                  kExactRanksForNET,
                                  kNoPowerOfTwoRequired,
                                  kMinNodesForNET,
                                  kMinNodesForNET))
    {
        GTEST_SKIP() << "NET transport test requires exactly " << kExactRanksForNET << " ranks on "
                     << kMinNodesForNET << " nodes (1 rank per node)";
    }

    // Create test-specific communicator
    ASSERT_MPI_SUCCESS(createTestCommunicator());

    if(config.world_rank == 0)
    {
        TEST_INFO("Starting ncclNetGraphRegisterBuffer test (multi-node)");
    }

    testNetGraphRegisterBuffer();

    if(config.world_rank == 0)
    {
        TEST_INFO("ncclNetGraphRegisterBuffer test completed successfully");
    }
}

TEST_F(NetTransportMPITest, NetLocalRegisterBufferTest)
{
    // NET transport tests require exactly 2 ranks on 2 nodes (1 rank per node)
    if(!validateTestPrerequisites(kExactRanksForNET,
                                  kExactRanksForNET,
                                  kNoPowerOfTwoRequired,
                                  kMinNodesForNET,
                                  kMinNodesForNET))
    {
        GTEST_SKIP() << "NET transport test requires exactly " << kExactRanksForNET << " ranks on "
                     << kMinNodesForNET << " nodes (1 rank per node)";
    }

    // Create test-specific communicator
    ASSERT_MPI_SUCCESS(createTestCommunicator());

    if(config.world_rank == 0)
    {
        TEST_INFO("Starting ncclNetLocalRegisterBuffer test (multi-node)");
    }

    testNetLocalRegisterBuffer();

    if(config.world_rank == 0)
    {
        TEST_INFO("ncclNetLocalRegisterBuffer test completed successfully");
    }
}

TEST_F(NetTransportMPITest, MultipleBufferSizesTest)
{
    // NET transport tests require exactly 2 ranks on 2 nodes (1 rank per node)
    if(!validateTestPrerequisites(kExactRanksForNET,
                                  kExactRanksForNET,
                                  kNoPowerOfTwoRequired,
                                  kMinNodesForNET,
                                  kMinNodesForNET))
    {
        GTEST_SKIP() << "NET transport test requires exactly " << kExactRanksForNET << " ranks on "
                     << kMinNodesForNET << " nodes (1 rank per node)";
    }

    ASSERT_MPI_SUCCESS(createTestCommunicator());

    if(config.world_rank == 0)
    {
        TEST_INFO("Starting multiple buffer sizes test (multi-node)");
    }

    testMultipleBufferSizes();

    if(config.world_rank == 0)
    {
        TEST_INFO("Multiple buffer sizes test completed successfully");
    }
}

#endif // MPI_TESTS_ENABLED
