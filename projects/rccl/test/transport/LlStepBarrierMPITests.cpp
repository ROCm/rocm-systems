/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file LlStepBarrierMPITests.cpp
 * @brief Ring LL and LL128 collectives over P2P, checked element by element.
 *
 * On gfx1250 these sends run storeFreeStepBarrier(), which does not wait for a step's peer stores before the next
 * step. Every rank and every iteration sends different input, so a line read before its data landed, or from an
 * earlier step, shows up as a wrong element rather than only as a hang. Other architectures run the full-fence
 * barrier through the same test. DDA, CE, direct and symmetric kernels are chosen ahead of Ring and their switches
 * are cached for the process, so run the test in its own process with them off:
 *   RCCL_LL128_FORCE_ENABLE=1 RCCL_DDA_ENABLE=0 RCCL_DIRECT_ALLGATHER_DISABLE=1 RCCL_DIRECT_REDUCE_SCATTER_DISABLE=1
 *   RCCL_SYM_K_MAX_BYTES=0 RCCL_TDM_LL128_ENABLE=0 mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=LlStepBarrierMPITest.*
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

#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

using namespace MPITestConstants;
using namespace RCCLTestGuards;

namespace
{
// Elements per rank. The odd count leaves partial lines and slices; 1M floats spans many steps on every channel.
constexpr size_t kChunkCounts[] = {4093, size_t{1} << 20};
constexpr size_t kMaxChunkCount = size_t{1} << 20;
// Collectives queued back to back, each with its own buffers and input.
constexpr int  kSlots   = 8;
constexpr int  kRounds  = 2;
constexpr auto kTimeout = std::chrono::seconds(120);

struct CollCase
{
    ncclFunc_t  func;
    const char* name;
};
constexpr CollCase kColls[] = {{ncclFuncAllReduce, "AllReduce"},
                               {ncclFuncReduceScatter, "ReduceScatter"},
                               {ncclFuncAllGather, "AllGather"}};

// Differs by rank and by iteration, and stays small enough that sums over ranks are exact in fp32.
float inputValue(int rank, int iteration, size_t i)
{
    return static_cast<float>((i * 7 + static_cast<size_t>(rank) * 131 + static_cast<size_t>(iteration) * 1031)
                              % 4099);
}

size_t sendCount(ncclFunc_t func, size_t chunk, int nranks)
{
    return func == ncclFuncAllGather ? chunk : chunk * nranks;
}

size_t recvCount(ncclFunc_t func, size_t chunk, int nranks)
{
    return func == ncclFuncReduceScatter ? chunk : chunk * nranks;
}

// The count argument of the collective call.
size_t callCount(ncclFunc_t func, size_t chunk, int nranks)
{
    return func == ncclFuncAllReduce ? chunk * nranks : chunk;
}

float expectedValue(ncclFunc_t func, int rank, int nranks, int iteration, size_t chunk, size_t j)
{
    if(func == ncclFuncAllGather)
    {
        return inputValue(static_cast<int>(j / chunk), iteration, j % chunk);
    }
    const size_t i   = func == ncclFuncReduceScatter ? rank * chunk + j : j;
    float        sum = 0;
    for(int r = 0; r < nranks; ++r)
    {
        sum += inputValue(r, iteration, i);
    }
    return sum;
}

ncclResult_t launch(ncclFunc_t func, const void* send, void* recv, size_t count, ncclComm_t comm, hipStream_t stream)
{
    switch(func)
    {
        case ncclFuncAllReduce: return ncclAllReduce(send, recv, count, ncclFloat32, ncclSum, comm, stream);
        case ncclFuncReduceScatter: return ncclReduceScatter(send, recv, count, ncclFloat32, ncclSum, comm, stream);
        default: return ncclAllGather(send, recv, count, ncclFloat32, comm, stream);
    }
}

bool selectsRing(ncclComm_t comm, ncclFunc_t func, size_t count, const void* send, void* recv, int protocol)
{
    int algo = -1, selected = -1, maxChannels = -1;
    return rcclGetCollImplInfo(comm, func, count, ncclFloat32, ncclSum, send, recv, /*graphCapturing=*/0, &algo,
                               &selected, &maxChannels)
               == ncclSuccess
           && algo == NCCL_ALGO_RING && selected == protocol;
}

bool ringHopsOnP2p(ncclComm_t comm)
{
    for(int c = 0; c < comm->nChannels; ++c)
    {
        const ncclChannel& channel = comm->channels[c];
        if(channel.peers[channel.ring.next]->send[0].transportComm != &p2pTransport.send
           || channel.peers[channel.ring.prev]->recv[0].transportComm != &p2pTransport.recv)
        {
            return false;
        }
    }
    return true;
}
} // namespace

class LlStepBarrierMPITest : public MPITestBase
{
protected:
    // A hung communicator cannot be destroyed, so every rank aborts it if any rank failed or timed out.
    bool finish(bool ok)
    {
        hipStream_t stream   = getActiveStream();
        const auto  deadline = std::chrono::steady_clock::now() + kTimeout;
        hipError_t  status   = hipErrorNotReady;
        while(ok && (status = hipStreamQuery(stream)) == hipErrorNotReady
              && std::chrono::steady_clock::now() < deadline)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if(mpiAllRanks(ok && status == hipSuccess))
        {
            return true;
        }
        (void)ncclCommAbort(test_comm_);
        test_comm_ = nullptr;
        return false;
    }

    void runCollectives(const char* protoName, int protocol)
    {
        SKIP_UNLESS_MPI_PREREQS(kMinProcessesForMPI);

        MPIHelpers::MpiEnvGuard algo("NCCL_ALGO", "Ring");
        MPIHelpers::MpiEnvGuard proto("NCCL_PROTO", protoName);

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ncclComm_t  comm   = getActiveCommunicator();
        hipStream_t stream = getActiveStream();
        const int   rank   = MPIEnvironment::world_rank;
        const int   nranks = MPIEnvironment::world_size;

        const size_t slotCount = kMaxChunkCount * nranks;
        void*        sends     = nullptr;
        void*        recvs     = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sends, kSlots * slotCount * sizeof(float)));
        auto sends_guard = makeDeviceBufferAutoGuard(sends);
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvs, kSlots * slotCount * sizeof(float)));
        auto recvs_guard = makeDeviceBufferAutoGuard(recvs);
        auto sendSlot    = [&](int s) { return static_cast<float*>(sends) + s * slotCount; };
        auto recvSlot    = [&](int s) { return static_cast<float*>(recvs) + s * slotCount; };

        // NCCL_PROTO is a request, and the backends above Ring are only off if the process env turned them off.
        bool selected = true;
        for(const CollCase& coll : kColls)
        {
            for(size_t chunk : kChunkCounts)
            {
                selected = selected
                           && selectsRing(comm, coll.func, callCount(coll.func, chunk, nranks), sendSlot(0),
                                          recvSlot(0), protocol);
            }
        }
        if(!mpiAllRanks(selected))
        {
            GTEST_SKIP() << "Not every case selects Ring/" << protoName << ": run this test in its own process with "
                         << "the RCCL overrides in the file comment";
        }

        // Ring connections are made by the first collective.
        ASSERT_MPI_TRUE(finish(launch(ncclFuncAllReduce, sendSlot(0), recvSlot(0), kChunkCounts[0] * nranks, comm,
                                      stream)
                               == ncclSuccess));
        if(!mpiAllRanks(ringHopsOnP2p(getActiveCommunicator())))
        {
            GTEST_SKIP() << "A ring hop is not on P2P";
        }

        std::vector<float> host(slotCount);
        int                iteration = 0;
        for(const CollCase& coll : kColls)
        {
            for(size_t chunk : kChunkCounts)
            {
                const size_t nsend = sendCount(coll.func, chunk, nranks);
                const size_t nrecv = recvCount(coll.func, chunk, nranks);
                for(int round = 0; round < kRounds; ++round, iteration += kSlots)
                {
                    bool ok = true;
                    for(int s = 0; ok && s < kSlots; ++s)
                    {
                        for(size_t i = 0; i < nsend; ++i)
                        {
                            host[i] = inputValue(rank, iteration + s, i);
                        }
                        ok = hipMemcpy(sendSlot(s), host.data(), nsend * sizeof(float), hipMemcpyHostToDevice)
                                 == hipSuccess
                             && hipMemset(recvSlot(s), 0xff, nrecv * sizeof(float)) == hipSuccess;
                    }
                    for(int s = 0; ok && s < kSlots; ++s)
                    {
                        ok = launch(coll.func, sendSlot(s), recvSlot(s), callCount(coll.func, chunk, nranks), comm,
                                    stream)
                             == ncclSuccess;
                    }
                    ASSERT_MPI_TRUE(finish(ok));

                    size_t mismatches = 0;
                    for(int s = 0; s < kSlots; ++s)
                    {
                        if(hipMemcpy(host.data(), recvSlot(s), nrecv * sizeof(float), hipMemcpyDeviceToHost)
                           != hipSuccess)
                        {
                            mismatches += nrecv;
                            continue;
                        }
                        size_t slotMismatches = 0;
                        size_t first          = 0;
                        for(size_t j = 0; j < nrecv; ++j)
                        {
                            if(host[j] != expectedValue(coll.func, rank, nranks, iteration + s, chunk, j)
                               && slotMismatches++ == 0)
                            {
                                first = j;
                            }
                        }
                        EXPECT_EQ(size_t{0}, slotMismatches)
                            << coll.name << " " << chunk << " elements per rank, iteration " << iteration + s
                            << ", rank " << rank << ": first wrong element " << first << " is " << host[first]
                            << ", expected " << expectedValue(coll.func, rank, nranks, iteration + s, chunk, first);
                        mismatches += slotMismatches;
                    }
                    ASSERT_MPI_TRUE(mismatches == 0);
                }
            }
        }
    }
};

TEST_F(LlStepBarrierMPITest, RingLl128OverP2pMatchesPerIterationInput)
{
    runCollectives("LL128", NCCL_PROTO_LL128);
}

TEST_F(LlStepBarrierMPITest, RingLlOverP2pMatchesPerIterationInput)
{
    runCollectives("LL", NCCL_PROTO_LL);
}

#endif // MPI_TESTS_ENABLED
