/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file NetLl128MPITests.cpp
 * @brief LL128 Ring AllReduce with every ring hop on the NET transport.
 *
 * Regression test for the LL128 post-send fence: if the proxy sees the tail before every 128-byte line of the step
 * is visible, the NIC sends stale lines and the receiver polls them forever. P2P and SHM are disabled so one node is
 * enough. They are cached for the process, so run the test in its own process with both set:
 *   NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=NetLl128MPITest.*
 */

#ifdef MPI_TESTS_ENABLED

#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "TestChecks.hpp"

#include "comm.h"
#include "rccl_common.h"
#include "transport.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

using namespace MPITestConstants;
using namespace RCCLTestGuards;

namespace
{
// 16 MiB of bf16, the size that hung.
constexpr size_t   kCount         = size_t{8} << 20;
constexpr int      kOutOfPlaceOps = 8;
constexpr int      kInPlaceOps    = 4000;
constexpr int      kInFlight      = 32;
constexpr auto     kTimeout       = std::chrono::seconds(120);
constexpr uint16_t kBf16One       = 0x3f80;
constexpr uint16_t kBf16Inf       = 0x7f80;

uint16_t bf16Bits(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

size_t countMismatches(const void* device, uint16_t expected)
{
    std::vector<uint16_t> host(kCount);
    if(hipMemcpy(host.data(), device, kCount * sizeof(uint16_t), hipMemcpyDeviceToHost) != hipSuccess)
    {
        return kCount;
    }
    return static_cast<size_t>(std::count_if(host.begin(), host.end(), [=](uint16_t v) { return v != expected; }));
}

bool selectsRingLl128(ncclComm_t comm, const void* send, void* recv)
{
    int algo = -1, protocol = -1, maxChannels = -1;
    return rcclGetCollImplInfo(comm, ncclFuncAllReduce, kCount, ncclBfloat16, ncclSum, send, recv,
                               /*graphCapturing=*/0, &algo, &protocol, &maxChannels)
               == ncclSuccess
           && algo == NCCL_ALGO_RING && protocol == NCCL_PROTO_LL128;
}

bool ringHopsOnGdrNet(ncclComm_t comm)
{
    for(int c = 0; c < comm->nChannels; ++c)
    {
        const ncclChannel&   channel = comm->channels[c];
        const ncclConnector& send    = channel.peers[channel.ring.next]->send[0];
        const ncclConnector& recv    = channel.peers[channel.ring.prev]->recv[0];
        if(send.transportComm != &netTransport.send || recv.transportComm != &netTransport.recv
           || !(send.conn.flags & NCCL_DIRECT_NIC) || !(recv.conn.flags & NCCL_DIRECT_NIC))
        {
            return false;
        }
    }
    return true;
}
} // namespace

class NetLl128MPITest : public MPITestBase
{
protected:
    // Queues at most kInFlight ops, because launches behind a hung op block the host. A hung communicator cannot be
    // destroyed, so every rank aborts it if any rank timed out.
    bool runAllReduce(const void* send, void* recv, int iterations)
    {
        hipStream_t stream   = getActiveStream();
        const auto  deadline = std::chrono::steady_clock::now() + kTimeout;
        auto        wait     = [&](auto query)
        {
            hipError_t status;
            while((status = query()) == hipErrorNotReady && std::chrono::steady_clock::now() < deadline)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return status == hipSuccess;
        };

        std::vector<hipEvent_t> events(kInFlight, nullptr);
        bool                    ok = true;
        for(hipEvent_t& event : events)
        {
            ok = ok && hipEventCreateWithFlags(&event, hipEventDisableTiming) == hipSuccess;
        }
        for(int i = 0; ok && i < iterations; ++i)
        {
            hipEvent_t event = events[i % kInFlight];
            ok = (i < kInFlight || wait([&] { return hipEventQuery(event); }))
                 && ncclAllReduce(send, recv, kCount, ncclBfloat16, ncclSum, test_comm_, stream) == ncclSuccess
                 && hipEventRecord(event, stream) == hipSuccess;
        }
        ok = ok && wait([&] { return hipStreamQuery(stream); });
        for(hipEvent_t event : events)
        {
            if(event != nullptr)
            {
                (void)hipEventDestroy(event);
            }
        }

        if(mpiAllRanks(ok))
        {
            return true;
        }
        (void)ncclCommAbort(test_comm_);
        test_comm_ = nullptr;
        return false;
    }
};

TEST_F(NetLl128MPITest, RingAllReduceCompletesWithEveryHopOnNet)
{
    SKIP_UNLESS_MPI_PREREQS(kMinProcessesForMPI);

    MPIHelpers::MpiEnvGuard p2p("NCCL_P2P_DISABLE", "1");
    MPIHelpers::MpiEnvGuard shm("NCCL_SHM_DISABLE", "1");
    MPIHelpers::MpiEnvGuard algo("NCCL_ALGO", "Ring");
    MPIHelpers::MpiEnvGuard proto("NCCL_PROTO", "LL128");

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    void* send = nullptr;
    void* recv = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&send, kCount * sizeof(uint16_t)));
    auto send_guard = makeDeviceBufferAutoGuard(send);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recv, kCount * sizeof(uint16_t)));
    auto recv_guard = makeDeviceBufferAutoGuard(recv);

    // NCCL_PROTO=LL128 is a request: the tuner drops LL128 where it is unsupported and falls back to Simple.
    ncclComm_t comm = getActiveCommunicator();
    if(!mpiAllRanks(selectsRingLl128(comm, send, recv) && selectsRingLl128(comm, recv, recv)))
    {
        GTEST_SKIP() << "AllReduce does not select Ring/LL128 on this system";
    }

    std::vector<uint16_t> host(kCount, kBf16One);
    ASSERT_MPI_EQ(hipSuccess,
                  hipMemcpy(send, host.data(), kCount * sizeof(uint16_t), hipMemcpyHostToDevice));

    // Ring connections are made by the first collective.
    ASSERT_MPI_TRUE(runAllReduce(send, recv, 1));
    if(!mpiAllRanks(ringHopsOnGdrNet(getActiveCommunicator())))
    {
        GTEST_SKIP() << "A ring hop is not on NET with GPU Direct RDMA: run this test in its own process with "
                        "NCCL_P2P_DISABLE=1 and NCCL_SHM_DISABLE=1, on hosts with an RDMA NIC";
    }

    ASSERT_MPI_TRUE(runAllReduce(send, recv, kOutOfPlaceOps));
    size_t mismatches = countMismatches(recv, bf16Bits(static_cast<float>(MPIEnvironment::world_size)));
    EXPECT_EQ(size_t{0}, mismatches) << "out of place, rank " << MPIEnvironment::world_rank;
    ASSERT_MPI_TRUE(mismatches == 0);

    // The hang reproduced only with in-place ops, so they get the iteration budget.
    ASSERT_MPI_TRUE(runAllReduce(recv, recv, kInPlaceOps));
    // Each in-place sum multiplies every element by the rank count, so the result saturates at +inf.
    mismatches = countMismatches(recv, kBf16Inf);
    EXPECT_EQ(size_t{0}, mismatches) << "in place, rank " << MPIEnvironment::world_rank;
    ASSERT_MPI_TRUE(mismatches == 0);
}

#endif // MPI_TESTS_ENABLED
