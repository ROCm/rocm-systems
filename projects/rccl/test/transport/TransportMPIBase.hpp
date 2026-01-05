/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef TRANSPORT_MPI_BASE_HPP
#define TRANSPORT_MPI_BASE_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "rccl/rccl.h"
#include "gtest/gtest.h"

#ifdef MPI_TESTS_ENABLED
    #include "MPITestBase.hpp"
    #include "MPIEnvironment.hpp"
    #include "TestChecks.hpp"
    #include "ResourceGuards.hpp"
    #include "comm.h"
    #include "core.h"
    #include "device.h"
    #include "graph.h"
    #include "graph/topo.h"
    #include "nccl_common.h"
    #include "transport.h"

using namespace RCCLTestGuards;

// Transport-specific RAII deleters
namespace RCCLTestGuards
{

struct TransportSendResourceDeleter
{
    ncclTransport* transport;
    explicit TransportSendResourceDeleter(ncclTransport* t = nullptr) : transport(t) {}
    void operator()(ncclConnector* connector) const
    {
        if(connector && transport)
        {
            transport->send.free(connector);
        }
    }
};

struct TransportRecvResourceDeleter
{
    ncclTransport* transport;
    explicit TransportRecvResourceDeleter(ncclTransport* t = nullptr) : transport(t) {}
    void operator()(ncclConnector* connector) const
    {
        if(connector && transport)
        {
            transport->recv.free(connector);
        }
    }
};

using TransportSendResourceGuard = ResourceGuard<ncclConnector*, TransportSendResourceDeleter>;
using TransportRecvResourceGuard = ResourceGuard<ncclConnector*, TransportRecvResourceDeleter>;

class TransportResourceGuard
{
private:
    ncclConnector* send_connector_;
    ncclConnector* recv_connector_;
    ncclTransport* transport_;

public:
    TransportResourceGuard(ncclConnector* send, ncclConnector* recv, ncclTransport* transport)
        : send_connector_(send), recv_connector_(recv), transport_(transport)
    {}

    ~TransportResourceGuard()
    {
        if(recv_connector_ && transport_)
        {
            transport_->recv.free(recv_connector_);
        }
        if(send_connector_ && transport_)
        {
            transport_->send.free(send_connector_);
        }
    }

    TransportResourceGuard(const TransportResourceGuard&)            = delete;
    TransportResourceGuard& operator=(const TransportResourceGuard&) = delete;
    TransportResourceGuard(TransportResourceGuard&&)                 = delete;
    TransportResourceGuard& operator=(TransportResourceGuard&&)      = delete;
};

inline TransportSendResourceGuard makeTransportSendGuard(ncclConnector* connector,
                                                         ncclTransport* transport)
{
    return TransportSendResourceGuard(connector, TransportSendResourceDeleter(transport));
}

inline TransportRecvResourceGuard makeTransportRecvGuard(ncclConnector* connector,
                                                         ncclTransport* transport)
{
    return TransportRecvResourceGuard(connector, TransportRecvResourceDeleter(transport));
}

} // namespace RCCLTestGuards

extern struct ncclTransport p2pTransport;
extern struct ncclTransport netTransport;
extern struct ncclTransport shmTransport;

// ============================================================================
// Transport Test Constants
// ============================================================================

namespace TransportTestConstants
{

// Buffer size constants (common across P2P, SHM, NET tests)
inline constexpr size_t kDefaultBufferSize     = 1024 * sizeof(float);  // 4096 bytes
inline constexpr size_t kSmallBufferSize       = 256;
inline constexpr size_t kMediumBufferSize      = 16384;                 // 16 KB
inline constexpr size_t kLargeBufferSize       = 135168;                // ~132 KB
inline constexpr size_t kVeryLargeBufferSize   = 256 * 1024 * 1024;    // 256 MB
inline constexpr size_t kCEMemcpyBufferSize    = 256 * 1024 * 1024;    // 256 MB (for CE tests)

// Pattern generation constants
inline constexpr int kDefaultPatternMultiplier = 1000;     // Standard rank-based patterns
inline constexpr int kSmallPatternMultiplier   = 100;      // Smaller patterns (memcpy tests)
inline constexpr int kLargePatternMultiplier   = 1000000;  // Large buffer patterns
inline constexpr int kPatternModulo            = 10000;    // Wraparound patterns
inline constexpr int kBytePatternModulo        = 256;      // uint8_t wraparound

// Validation constants
inline constexpr size_t kMaxValidationElements = 100;      // Number of elements to validate
inline constexpr size_t kMinValidationSamples  = 100;      // Minimum samples for validation
inline constexpr size_t kValidationStride      = 1000;     // Stride for sampling validation
inline constexpr int    kMaxErrorsToReport     = 10;       // Max errors to display

// Test iteration constants
inline constexpr int kMultipleTransferCount = 5;           // Number of sequential transfers

} // namespace TransportTestConstants

// Common test configuration
struct TransportTestConfig
{
    int         world_rank{0};
    int         world_size{0};
    int         peer_rank{0};
    ncclComm_t  nccl_comm{nullptr};
    hipStream_t stream{nullptr};
};

// Base class for transport tests with common functionality
// Inherits from MPITestBase to get validation capabilities
class TransportTestBase : public MPITestBase
{
protected:
    TransportTestConfig config;

    // Transport connectors (can be used for P2P or NET)
    ncclConnector send_connector = {};
    ncclConnector recv_connector = {};

    // Track which transport type is initialized
    enum class TransportType
    {
        None,
        P2P,
        SHM,
        Network
    };
    TransportType initialized_transport = TransportType::None;

    // Core NCCL components
    struct ncclComm* comm_handle      = nullptr;
    ncclPeerInfo*    local_peer_info  = nullptr;
    ncclPeerInfo*    remote_peer_info = nullptr;
    ncclTopoGraph*   topology_graph   = nullptr;

    // RAII guards for automatic resource cleanup
    // These are managed by helper methods and cleaned up automatically
    std::vector<DeviceBufferAutoGuard> buffer_guards_;
    std::vector<NcclRegHandleGuard>    reg_handle_guards_;

    // Setup and teardown
    void SetUp() override;
    void TearDown() override;

    // Override createTestCommunicator to also update config
    ncclResult_t createTestCommunicator() override;

    // Set transport type and initialize connectors
    void setTransportType(TransportType type);

    // Buffer allocation (unguarded - for manual management)
    void allocateAndInitBuffers(void** send_buffer,
                                void** recv_buffer,
                                size_t send_bytes,
                                size_t recv_bytes);

    // Buffer allocation with automatic RAII guards
    // store_in_base=true: Guards stored in base class, cleanup at test end
    // store_in_base=false: Guards returned, caller controls cleanup scope
    std::pair<DeviceBufferAutoGuard, DeviceBufferAutoGuard> allocateAndInitBuffersGuarded(void** send_buffer,
                                                                                           void** recv_buffer,
                                                                                           size_t send_bytes,
                                                                                           size_t recv_bytes,
                                                                                           bool   store_in_base = true);

    // Buffer registration (unguarded - for manual management)
    void preRegisterBuffers(void*  send_buffer,
                            void*  recv_buffer,
                            size_t send_bytes,
                            size_t recv_bytes,
                            void** send_reg_handle,
                            void** recv_reg_handle);

    // Buffer registration with automatic RAII guards
    // store_in_base=true: Guards stored in base class, cleanup at test end
    // store_in_base=false: Guards returned, caller controls cleanup scope
    std::pair<NcclRegHandleGuard, NcclRegHandleGuard>
        preRegisterBuffersGuarded(void*  send_buffer,
                                  void*  recv_buffer,
                                  size_t send_bytes,
                                  size_t recv_bytes,
                                  void** send_reg_handle,
                                  void** recv_reg_handle,
                                  bool   store_in_base = true);
};

// ============================================================================
// Generic Stream Synchronization Helpers
// ============================================================================

/**
 * @brief Generic stream synchronization helper
 *
 * Synchronizes a HIP stream and returns the error code. This function is
 * marked [[nodiscard]] to ensure callers check the return value.
 *
 * @param stream HIP stream to synchronize
 * @param rank MPI rank (for error reporting, currently unused but allows
 *             future enhancement with rank-specific error messages)
 * @return hipError_t Result of hipStreamSynchronize
 *
 * Usage examples:
 *   - Manual error checking: hipError_t err = syncStream(stream, rank);
 *   - With HIPCHECK macro: HIPCHECK(syncStream(stream, rank));
 *   - With assertion macro: ASSERT_STREAM_SYNC(stream, rank);
 */
[[nodiscard]] inline hipError_t syncStream(hipStream_t stream, int rank = 0)
{
    return hipStreamSynchronize(stream);
}

    /**
 * @def ASSERT_STREAM_SYNC
 * @brief Macro to assert stream synchronization succeeds
 *
 * Convenience macro that combines syncStream() with ASSERT_EQ to provide
 * clean, consistent stream synchronization checks in tests.
 *
 * @param stream HIP stream to synchronize
 * @param rank MPI rank for error reporting
 *
 * Example: ASSERT_STREAM_SYNC(config.stream, config.world_rank);
 */
    #define ASSERT_STREAM_SYNC(stream, rank)            \
        ASSERT_EQ(hipSuccess, syncStream(stream, rank)) \
            << "Rank " << rank << ": Stream synchronization failed"

    /**
 * @def ASSERT_STREAM_SYNC_MPI
 * @brief MPI-aware stream synchronization assertion
 *
 * Uses ASSERT_MPI_EQ to ensure all ranks synchronize before failing.
 * This prevents deadlocks when one rank fails while others are waiting
 * in collective operations.
 *
 * @param stream HIP stream to synchronize
 * @param rank MPI rank for error reporting
 *
 * Example: ASSERT_STREAM_SYNC_MPI(config.stream, config.world_rank);
 *
 * @note Prefer this version in multi-rank tests to avoid hangs
 */
    #define ASSERT_STREAM_SYNC_MPI(stream, rank) ASSERT_MPI_EQ(hipSuccess, syncStream(stream, rank))

#endif // MPI_TESTS_ENABLED

#endif // TRANSPORT_MPI_BASE_HPP
