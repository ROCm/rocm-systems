/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file LlStepBarrierMPITests.cpp
 * @brief Ring and Tree collectives on P2P, SHM and NET, and Simple's local copies, checked element by element.
 *
 * On gfx1250 these run the store-free step barriers (storeFreeStepBarrier() and Simple's stepStartBarrier()) and
 * reduceCopy's 16-byte buffer stores, and catch end-of-step barrier regressions too. Every rank and every iteration
 * sends different input, so data read before it landed, or a slot overwritten while it is still read, shows up as a
 * wrong element rather than only as a hang. Other architectures run the full-fence barriers through the same test.
 * DDA, CE, direct and symmetric kernels are chosen ahead of Ring and their switches are cached for the process, so run
 * the test in its own process with them off:
 *   RCCL_LL128_FORCE_ENABLE=1 RCCL_DDA_ENABLE=0 RCCL_DIRECT_ALLGATHER_DISABLE=1 RCCL_DIRECT_REDUCE_SCATTER_DISABLE=1
 *   RCCL_SYM_K_MAX_BYTES=0 RCCL_TDM_LL128_ENABLE=0 mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=LlStepBarrierMPITest.*
 * The *OverShm* case also needs NCCL_P2P_DISABLE=1, and the *OverNet* case NCCL_P2P_DISABLE=1 and NCCL_SHM_DISABLE=1.
 */

#ifdef MPI_TESTS_ENABLED

#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "TestChecks.hpp"

#include "comm.h"
#include "graph/topo.h"
#include "rccl_common.h"
#include "transport.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
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
// The LL and LL128 cases run the first kNumLlColls. Broadcast and Reduce move their root with the iteration.
constexpr CollCase kRingColls[] = {{ncclFuncAllReduce, "AllReduce"},
                                   {ncclFuncReduceScatter, "ReduceScatter"},
                                   {ncclFuncAllGather, "AllGather"},
                                   {ncclFuncBroadcast, "Broadcast"},
                                   {ncclFuncReduce, "Reduce"}};
constexpr size_t   kNumLlColls   = 3;
constexpr CollCase kAllReduce[]  = {{ncclFuncAllReduce, "AllReduce"}};
// A grouped send and receive to the calling rank, which sendrecv.h copies with reduceCopy.
constexpr CollCase kSelfCopy[] = {{ncclFuncSendRecv, "SendRecv to self"}};

// What a test runs and what it must select and run on.
struct RunSpec
{
    const char*          algoName;  // NCCL_ALGO
    int                  algo;      // NCCL_ALGO_*, or -1 for local copies, which no algorithm carries
    const char*          protoName; // NCCL_PROTO
    int                  protocol;
    const ncclTransport* transport; // every hop of algo must be on it; nullptr checks no hop
    const CollCase*      colls;
    size_t               numColls;
    ncclRedOp_t          op      = ncclSum; // ncclAvg only with oneRank, where it is exact
    bool                 oneRank = false;   // each process runs on a one-rank communicator of its own
};

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
    return func == ncclFuncReduceScatter || func == ncclFuncAllGather ? chunk : chunk * nranks;
}

// root is the root of Broadcast and Reduce, and the calling rank for SendRecv.
float expectedValue(ncclFunc_t func, int rank, int nranks, int root, int iteration, size_t chunk, size_t j)
{
    switch(func)
    {
        case ncclFuncAllGather: return inputValue(static_cast<int>(j / chunk), iteration, j % chunk);
        case ncclFuncBroadcast:
        case ncclFuncSendRecv: return inputValue(root, iteration, j);
        default: break;
    }
    const size_t i   = func == ncclFuncReduceScatter ? rank * chunk + j : j;
    float        sum = 0;
    for(int r = 0; r < nranks; ++r)
    {
        sum += inputValue(r, iteration, i);
    }
    return sum;
}

ncclResult_t launch(ncclFunc_t func, const void* send, void* recv, size_t count, ncclRedOp_t op, int root,
                    ncclComm_t comm, hipStream_t stream)
{
    switch(func)
    {
        case ncclFuncAllReduce: return ncclAllReduce(send, recv, count, ncclFloat32, op, comm, stream);
        case ncclFuncReduceScatter: return ncclReduceScatter(send, recv, count, ncclFloat32, op, comm, stream);
        case ncclFuncAllGather: return ncclAllGather(send, recv, count, ncclFloat32, comm, stream);
        case ncclFuncBroadcast: return ncclBroadcast(send, recv, count, ncclFloat32, root, comm, stream);
        case ncclFuncReduce: return ncclReduce(send, recv, count, ncclFloat32, op, root, comm, stream);
        default: break;
    }
    ncclResult_t result = ncclGroupStart();
    if(result == ncclSuccess)
    {
        result = ncclSend(send, count, ncclFloat32, root, comm, stream);
    }
    if(result == ncclSuccess)
    {
        result = ncclRecv(recv, count, ncclFloat32, root, comm, stream);
    }
    const ncclResult_t end = ncclGroupEnd();
    return result != ncclSuccess ? result : end;
}

bool selects(const RunSpec& spec, ncclComm_t comm, ncclFunc_t func, size_t count, const void* send, void* recv)
{
    int algo = -1, selected = -1, maxChannels = -1;
    return rcclGetCollImplInfo(comm, func, count, ncclFloat32, spec.op, send, recv, /*graphCapturing=*/0, &algo,
                               &selected, &maxChannels)
               == ncclSuccess
           && algo == spec.algo && selected == spec.protocol;
}

// Ring: the send to ring.next and the receive from ring.prev. Tree: both directions to tree.up and every tree.down.
bool hopsOn(ncclComm_t comm, int algo, const ncclTransport* transport)
{
    for(int c = 0; c < comm->nChannels; ++c)
    {
        const ncclChannel& channel = comm->channels[c];
        auto               on      = [&](int peer, bool send, bool recv)
        {
            return peer < 0
                   || ((!send || channel.peers[peer]->send[0].transportComm == &transport->send)
                       && (!recv || channel.peers[peer]->recv[0].transportComm == &transport->recv));
        };
        bool ok = algo == NCCL_ALGO_TREE ? on(channel.tree.up, true, true)
                                         : on(channel.ring.next, true, false) && on(channel.ring.prev, false, true);
        for(int d = 0; algo == NCCL_ALGO_TREE && d < NCCL_MAX_TREE_ARITY; ++d)
        {
            ok = ok && on(channel.tree.down[d], true, true);
        }
        if(!ok)
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

    // Swaps the world communicator for a one-rank one per process, whose AllReduce launches oneRankReduce.
    bool useOneRankCommunicator()
    {
        ncclComm_t one = nullptr;
        const bool ok  = ncclCommSplit(test_comm_, MPIEnvironment::world_rank, 0, &one, nullptr) == ncclSuccess;
        (void)ncclCommDestroy(test_comm_);
        test_comm_ = one;
        return ok;
    }

    void runCollectives(const RunSpec& spec)
    {
        SKIP_UNLESS_MPI_PREREQS(kMinProcessesForMPI);

        MPIHelpers::MpiEnvGuard algo("NCCL_ALGO", spec.algoName);
        MPIHelpers::MpiEnvGuard proto("NCCL_PROTO", spec.protoName);

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        if(spec.oneRank)
        {
            ASSERT_MPI_TRUE(useOneRankCommunicator());
        }
        ncclComm_t  comm   = getActiveCommunicator();
        hipStream_t stream = getActiveStream();
        int         rank   = -1;
        int         nranks = 0;
        ASSERT_MPI_TRUE(ncclCommUserRank(comm, &rank) == ncclSuccess && ncclCommCount(comm, &nranks) == ncclSuccess);
        const CollCase* const collsEnd = spec.colls + spec.numColls;

        const size_t slotCount = kMaxChunkCount * nranks;
        void*        sends     = nullptr;
        void*        recvs     = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sends, kSlots * slotCount * sizeof(float)));
        auto sends_guard = makeDeviceBufferAutoGuard(sends);
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvs, kSlots * slotCount * sizeof(float)));
        auto recvs_guard = makeDeviceBufferAutoGuard(recvs);
        auto sendSlot    = [&](int s) { return static_cast<float*>(sends) + s * slotCount; };
        auto recvSlot    = [&](int s) { return static_cast<float*>(recvs) + s * slotCount; };

        // NCCL_ALGO and NCCL_PROTO are requests, and the backends above them are only off if the process env said so.
        bool selected = true;
        for(const CollCase* coll = spec.colls; spec.algo >= 0 && coll != collsEnd; ++coll)
        {
            for(size_t chunk : kChunkCounts)
            {
                selected = selected
                           && selects(spec, comm, coll->func, callCount(coll->func, chunk, nranks), sendSlot(0),
                                      recvSlot(0));
            }
        }
        if(!mpiAllRanks(selected))
        {
            GTEST_SKIP() << "Not every case selects " << spec.algoName << "/" << spec.protoName << ": run this test in "
                         << "its own process with the RCCL overrides in the file comment";
        }

        // Ring and Tree connections are made by the first collective.
        ASSERT_MPI_TRUE(finish(launch(ncclFuncAllReduce, sendSlot(0), recvSlot(0), kChunkCounts[0] * nranks, spec.op,
                                      0, comm, stream)
                               == ncclSuccess));
        if(spec.transport != nullptr && !mpiAllRanks(hopsOn(getActiveCommunicator(), spec.algo, spec.transport)))
        {
            GTEST_SKIP() << "A " << spec.algoName << " hop is not on " << spec.transport->name;
        }

        std::vector<float> host(slotCount);
        int                iteration = 0;
        for(const CollCase* coll = spec.colls; coll != collsEnd; ++coll)
        {
            for(size_t chunk : kChunkCounts)
            {
                const size_t nsend = sendCount(coll->func, chunk, nranks);
                const size_t nrecv = recvCount(coll->func, chunk, nranks);
                auto         root  = [&](int it) { return coll->func == ncclFuncSendRecv ? rank : it % nranks; };
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
                        ok = launch(coll->func, sendSlot(s), recvSlot(s), callCount(coll->func, chunk, nranks),
                                    spec.op, root(iteration + s), comm, stream)
                             == ncclSuccess;
                    }
                    ASSERT_MPI_TRUE(finish(ok));

                    size_t mismatches = 0;
                    for(int s = 0; s < kSlots; ++s)
                    {
                        const int it = iteration + s;
                        // Only the root receives a Reduce.
                        if(coll->func == ncclFuncReduce && rank != root(it))
                        {
                            continue;
                        }
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
                            if(host[j] != expectedValue(coll->func, rank, nranks, root(it), it, chunk, j)
                               && slotMismatches++ == 0)
                            {
                                first = j;
                            }
                        }
                        EXPECT_EQ(size_t{0}, slotMismatches)
                            << coll->name << " " << chunk << " elements per rank, iteration " << it << ", rank "
                            << rank << ", root " << root(it) << ": first wrong element " << first << " is "
                            << host[first] << ", expected "
                            << expectedValue(coll->func, rank, nranks, root(it), it, chunk, first);
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
    runCollectives({"Ring", NCCL_ALGO_RING, "LL128", NCCL_PROTO_LL128, &p2pTransport, kRingColls, kNumLlColls});
}

TEST_F(LlStepBarrierMPITest, RingLlOverP2pMatchesPerIterationInput)
{
    runCollectives({"Ring", NCCL_ALGO_RING, "LL", NCCL_PROTO_LL, &p2pTransport, kRingColls, kNumLlColls});
}

TEST_F(LlStepBarrierMPITest, RingSimpleOverP2pMatchesPerIterationInput)
{
    runCollectives(
        {"Ring", NCCL_ALGO_RING, "Simple", NCCL_PROTO_SIMPLE, &p2pTransport, kRingColls, std::size(kRingColls)});
}

// Default SHM has no connFifo, so stepStartBarrier() orders LDS only and this checks it over host memory. NET, and SHM
// with NCCL_SHM_USE_CUDA_MEMCPY=1, set a connFifo: the full barrier is kept and the post-send fence drains for the proxy.
TEST_F(LlStepBarrierMPITest, RingSimpleOverShmMatchesPerIterationInput)
{
    runCollectives(
        {"Ring", NCCL_ALGO_RING, "Simple", NCCL_PROTO_SIMPLE, &shmTransport, kRingColls, std::size(kRingColls)});
}

TEST_F(LlStepBarrierMPITest, RingSimpleOverNetMatchesPerIterationInput)
{
    runCollectives(
        {"Ring", NCCL_ALGO_RING, "Simple", NCCL_PROTO_SIMPLE, &netTransport, kRingColls, std::size(kRingColls)});
}

TEST_F(LlStepBarrierMPITest, TreeSimpleAllReduceOverP2pMatchesPerIterationInput)
{
    runCollectives(
        {"Tree", NCCL_ALGO_TREE, "Simple", NCCL_PROTO_SIMPLE, &p2pTransport, kAllReduce, std::size(kAllReduce)});
}

// Simple's buffer stores outside the step loop: oneRankReduce (ncclAvg; a one-rank ncclSum is a memcpy) and the
// sendrecv.h self copy. Neither has a step barrier, so these check the stored bytes.
TEST_F(LlStepBarrierMPITest, OneRankAllReduceMatchesPerIterationInput)
{
    runCollectives({"Ring", -1, "Simple", NCCL_PROTO_SIMPLE, nullptr, kAllReduce, std::size(kAllReduce), ncclAvg,
                    /*oneRank=*/true});
}

TEST_F(LlStepBarrierMPITest, SendRecvToSelfMatchesPerIterationInput)
{
    runCollectives({"Ring", -1, "Simple", NCCL_PROTO_SIMPLE, nullptr, kSelfCopy, std::size(kSelfCopy)});
}

// Its own suite and process: RCCL caches whether NCCL_PROTO is set, and the cases above set it. finish() is reused.
class BroadcastLl128WindowMPITest : public LlStepBarrierMPITest
{};

// One gfx942 node of 3 to 8 GPUs on all-pairs XGMI with LL128 off in the topology model: with no NCCL_PROTO, some Ring
// Broadcast size selects LL128, with right data. There only the tuning model's Broadcast offer enables LL128.
TEST_F(BroadcastLl128WindowMPITest, ModelSelectsRingLl128OnGfx942)
{
    SKIP_UNLESS_MPI_PREREQS(kMinProcessesForMPI);
    if(!mpiAllRanks(std::getenv("NCCL_PROTO") == nullptr && std::getenv("NCCL_ALGO") == nullptr
                    && !MPIHelpers::mpiNcclProtoWasSet()))
    {
        GTEST_SKIP() << "Needs the tuning model's choice: NCCL_PROTO and NCCL_ALGO unset, and not set earlier in this "
                     << "process";
    }
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t           comm = getActiveCommunicator();
    const ncclTopoGraph& ring = comm->graphs[NCCL_ALGO_RING];
    hipDeviceProp_t      prop{};
    // gcnArchName reads like "gfx942:sramecc+:xnack-".
    const bool gfx942 = hipGetDeviceProperties(&prop, comm->cudaDev) == hipSuccess
                        && std::strncmp(prop.gcnArchName, "gfx942", 6) == 0;
    if(!mpiAllRanks(gfx942 && comm->nNodes == 1 && comm->nRanks >= 3 && comm->nRanks <= 8 && !comm->hasMloPart
                    && !comm->topo->ll128Enabled && (comm->topo->type & RCCL_TOPO_XGMI_ALL) != 0
                    && ring.typeInter <= PATH_PXB && ring.typeIntra <= PATH_NVL))
    {
        GTEST_SKIP() << "Needs one gfx942 node of 3 to 8 unpartitioned GPUs on all-pairs XGMI with LL128 off";
    }
    constexpr size_t kMaxCount = size_t{16} << 20;
    void*            buf       = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&buf, kMaxCount * sizeof(float)));
    auto   buf_guard = makeDeviceBufferAutoGuard(buf);
    size_t count     = 0;
    bool   ringRuns  = false;
    for(size_t c = size_t{16} << 10; c <= kMaxCount && count == 0; c *= 2)
    {
        int        algo = -1, protocol = -1, maxChannels = -1;
        const bool ok   = rcclGetCollImplInfo(comm, ncclFuncBroadcast, c, ncclFloat32, ncclSum, buf, buf,
                                            /*graphCapturing=*/0, &algo, &protocol, &maxChannels)
                        == ncclSuccess;
        ringRuns = ringRuns || mpiAllRanks(ok && algo == NCCL_ALGO_RING);
        count    = mpiAllRanks(ok && algo == NCCL_ALGO_RING && protocol == NCCL_PROTO_LL128) ? c : 0;
    }
    if(!ringRuns)
    {
        GTEST_SKIP() << "Another algorithm runs Broadcast here";
    }
    ASSERT_MPI_TRUE(count != 0);

    const int          root = comm->nRanks - 1;
    std::vector<float> host(count);
    for(size_t i = 0; i < count; ++i)
    {
        host[i] = comm->rank == root ? inputValue(root, 0, i) : -1.0f;
    }
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(buf, host.data(), count * sizeof(float), hipMemcpyHostToDevice));
    ASSERT_MPI_TRUE(
        finish(ncclBroadcast(buf, buf, count, ncclFloat32, root, comm, getActiveStream()) == ncclSuccess));
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(host.data(), buf, count * sizeof(float), hipMemcpyDeviceToHost));
    size_t wrong = 0;
    for(size_t i = 0; i < count; ++i)
    {
        if(host[i] != inputValue(root, 0, i))
        {
            ++wrong;
        }
    }
    EXPECT_EQ(size_t{0}, wrong) << count << " floats from root " << root;
    ASSERT_MPI_TRUE(wrong == 0);
}

#endif // MPI_TESTS_ENABLED
