/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file NetLl128MPITests.cpp
 * @brief LL128 Ring AllReduce with every ring hop on the NET transport.
 *
 * NET LL128 hands a step to the proxy by bumping the tail, and the NIC reads the step from GPU memory once the
 * proxy sees it. A 128-byte line that is not yet visible by then reaches the receiver with the previous step's
 * flag, and the receiver polls that line forever. P2P and SHM are disabled so intra-node hops go through the NIC
 * too, which makes one node enough. On gfx942 with the post-send fence that skips the L2 writeback, this hung on
 * most runs within the launch count below.
 *
 * NCCL_P2P_DISABLE and NCCL_SHM_DISABLE are cached for the process, so the test skips unless every ring hop is on
 * NET with GPU Direct RDMA. Run it in its own process:
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=NetLl128MPITest.*
 */

#ifdef MPI_TESTS_ENABLED

#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "TestChecks.hpp"

#include "comm.h"
#include "transport.h"

#include <hip/hip_runtime.h>

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
constexpr size_t   kCount      = size_t{8} << 20;
constexpr int      kIterations = 2000;
constexpr auto     kTimeout    = std::chrono::seconds(120);
constexpr uint16_t kBf16One    = 0x3f80;

uint16_t bf16Bits(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
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

bool allRanks(bool local)
{
    int ok = local ? 1 : 0;
    int all = 0;
    MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return all != 0;
}
} // namespace

class NetLl128MPITest : public MPITestBase
{
protected:
    // Destroying the communicator would wait on the hung kernels, so every rank aborts it if any rank timed out.
    bool drainOrAbort()
    {
        const auto deadline = std::chrono::steady_clock::now() + kTimeout;
        hipError_t status;
        while((status = hipStreamQuery(getActiveStream())) == hipErrorNotReady
              && std::chrono::steady_clock::now() < deadline)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if(allRanks(status == hipSuccess))
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
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    void* send = nullptr;
    void* recv = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&send, kCount * sizeof(uint16_t)));
    auto send_guard = makeDeviceBufferAutoGuard(send);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recv, kCount * sizeof(uint16_t)));
    auto recv_guard = makeDeviceBufferAutoGuard(recv);

    std::vector<uint16_t> host(kCount, kBf16One);
    ASSERT_MPI_EQ(hipSuccess,
                  hipMemcpy(send, host.data(), kCount * sizeof(uint16_t), hipMemcpyHostToDevice));

    // Ring connections are made by the first collective.
    ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(send, recv, kCount, ncclBfloat16, ncclSum, comm, stream));
    ASSERT_MPI_TRUE(drainOrAbort());
    if(!allRanks(ringHopsOnGdrNet(comm)))
    {
        GTEST_SKIP() << "A ring hop is not on NET with GPU Direct RDMA: run this test in its own process, on "
                        "hosts with an RDMA NIC";
    }

    for(int i = 0; i < kIterations; ++i)
    {
        ASSERT_MPI_EQ(ncclSuccess,
                      ncclAllReduce(send, recv, kCount, ncclBfloat16, ncclSum, comm, stream));
    }
    ASSERT_MPI_TRUE(drainOrAbort());

    ASSERT_MPI_EQ(hipSuccess,
                  hipMemcpy(host.data(), recv, kCount * sizeof(uint16_t), hipMemcpyDeviceToHost));
    const uint16_t expected   = bf16Bits(static_cast<float>(MPIEnvironment::world_size));
    size_t         mismatches = 0;
    for(uint16_t v : host)
    {
        mismatches += v != expected;
    }
    EXPECT_EQ(size_t{0}, mismatches) << "rank " << MPIEnvironment::world_rank;
    ASSERT_MPI_TRUE(mismatches == 0);
}

#endif // MPI_TESTS_ENABLED
