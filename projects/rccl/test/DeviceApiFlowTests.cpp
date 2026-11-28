/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// The Flow processor (nccl_device/flow.h) is compiled only for device-side lambdas.
#ifndef __CUDACC_EXTENDED_LAMBDA__
#define __CUDACC_EXTENDED_LAMBDA__ 1
#endif

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <chrono>
#include <vector>

#include "nccl_device.h"

#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting
{
namespace
{

constexpr int    kRanks       = 2;
constexpr int    kWorkerWarps = 4;
constexpr int    kSlices      = 4;
constexpr size_t kStepBytes   = 4096;
// One part and one connection per direction: the FIFO is ncclFlowFifoSlots steps, then one connection state.
constexpr size_t kChannelBytes = ncclFlowFifoSlots * kStepBytes;
constexpr size_t kWindowBytes  = kChannelBytes + sizeof(ncclFlowConnState);
constexpr int    kChunkElts    = kChannelBytes / kSlices / sizeof(int);
// Several FIFO wraps, so the transfer depends on credits coming back from the peer.
constexpr int kElts = 32 * kChunkElts;

struct FlowCopyOp
{
};

struct FlowCopyOps
{
    template <typename T, typename RedOp, int MaxSrcs, int MaxDsts, typename SrcFn, typename DstFn>
    __device__ void reduceCopy(int thread, int nThreads, int nSrcs, SrcFn const& srcFn, int nDsts,
                               DstFn const& dstFn, RedOp const&, int nElts) const
    {
        for(int i = thread; i < nElts; i += nThreads)
        {
            T value = srcFn(0)[i];
            for(int d = 0; d < nDsts; d++) dstFn(d)[i] = value;
        }
    }
};

// Each rank sends its input to the other rank over one LSA Flow connection and receives the other's into output.
__global__ void lsaFlowExchangeKernel(ncclDevComm_t devComm, ncclWindow_t flowWindow, int* input, int* output)
{
    ncclCoopNamedBarrierInit();
    __shared__ ncclFlowShmem flowShmem;
    const int                     peers[1] = {(devComm.rank + 1) % devComm.nRanks};
    const ncclFlowConfig<0>       config   = {ncclSymPtr<char>(flowWindow, 0),
                                              ncclSymPtr<ncclFlowConnState>(flowWindow, kChannelBytes),
                                              kChannelBytes, kSlices, /*connStatesPerChannel=*/1};
    ncclLsaFlowProcessor<FlowCopyOps> flow(
        FlowCopyOps{}, devComm, config, peers, 1, peers, 1, /*channelId=*/0, kWorkerWarps, flowShmem);
    ncclFlowNullPtrFn<int> nullPtr;
    for(int offset = 0; offset < kElts; offset += kChunkElts)
    {
        int* src = input + offset;
        int* dst = output + offset;
        flow.process(ncclFlowSend{}, ncclFlowNoRecv{}, ncclSymPtr<int>{}, ncclSymPtr<int>{},
                     [=] __device__(int) -> int* { return src; }, 1, nullPtr, 0, FlowCopyOp{}, kChunkElts);
        flow.process(ncclFlowNoSend{}, ncclFlowRecv{}, ncclSymPtr<int>{}, ncclSymPtr<int>{}, nullPtr, 0,
                     [=] __device__(int) -> int* { return dst; }, 1, FlowCopyOp{}, kChunkElts);
    }
    flow.close();
}

struct FlowRank
{
    ncclComm_t    comm   = nullptr;
    hipStream_t   stream = nullptr;
    int*          input  = nullptr;
    int*          output = nullptr;
    void*         flow   = nullptr;
    ncclWindow_t  window = nullptr;
    ncclDevComm_t devComm{};
    bool          devCommCreated = false;
};

static void runLsaFlowExchangeTest()
{
    int gpuCount = 0, canAccess = 0;
    if(hipGetDeviceCount(&gpuCount) != hipSuccess || gpuCount < kRanks)
        GTEST_SKIP() << "This test requires at least 2 visible GPUs.";
    if(hipDeviceCanAccessPeer(&canAccess, 0, 1) != hipSuccess || !canAccess)
        GTEST_SKIP() << "This test requires direct P2P access between the first 2 GPUs.";

    std::vector<FlowRank>   ranks(kRanks);
    std::vector<ncclComm_t> comms(kRanks);
    ASSERT_EQ(ncclCommInitAll(comms.data(), kRanks, nullptr), ncclSuccess);
    std::vector<int> host(kElts);
    for(int r = 0; r < kRanks; r++)
    {
        ranks[r].comm = comms[r];
        ASSERT_EQ(hipSetDevice(r), hipSuccess);
        ASSERT_EQ(hipStreamCreate(&ranks[r].stream), hipSuccess);
        for(int i = 0; i < kElts; i++) host[i] = r * 1000003 + i;
        ASSERT_EQ(hipMalloc(reinterpret_cast<void**>(&ranks[r].input), kElts * sizeof(int)), hipSuccess);
        ASSERT_EQ(hipMemcpy(ranks[r].input, host.data(), kElts * sizeof(int), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMalloc(reinterpret_cast<void**>(&ranks[r].output), kElts * sizeof(int)), hipSuccess);
        ASSERT_EQ(hipMemset(ranks[r].output, 0, kElts * sizeof(int)), hipSuccess);
        // The receive FIFO and the connection state must start zeroed.
        ASSERT_EQ(ncclMemAlloc(&ranks[r].flow, kWindowBytes), ncclSuccess);
        ASSERT_EQ(hipMemset(ranks[r].flow, 0, kWindowBytes), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    }

    std::vector<ncclResult_t> results(kRanks, ncclSuccess);
    ASSERT_EQ(ncclGroupStart(), ncclSuccess);
    for(int r = 0; r < kRanks; r++)
        results[r] = ncclCommWindowRegister(ranks[r].comm, ranks[r].flow, kWindowBytes, &ranks[r].window,
                                            NCCL_WIN_COLL_SYMMETRIC);
    results.push_back(ncclGroupEnd());
    for(ncclResult_t result : results)
    {
        if(result == ncclInvalidUsage)
            GTEST_SKIP() << "Symmetric window registration is unsupported on this configuration.";
        ASSERT_EQ(result, ncclSuccess);
    }

    ncclDevCommRequirements_t requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    results.assign(kRanks, ncclSuccess);
    ASSERT_EQ(ncclGroupStart(), ncclSuccess);
    for(int r = 0; r < kRanks; r++)
        results[r] = ncclDevCommCreate(ranks[r].comm, &requirements, &ranks[r].devComm);
    results.push_back(ncclGroupEnd());
    for(int r = 0; r < kRanks; r++) ranks[r].devCommCreated = results[r] == ncclSuccess;
    for(ncclResult_t result : results)
    {
        if(result == ncclInvalidUsage)
            GTEST_SKIP() << "Symmetric device API is unsupported on this configuration.";
        ASSERT_EQ(result, ncclSuccess);
    }

    for(int r = 0; r < kRanks; r++)
    {
        hipDeviceProp_t props{};
        ASSERT_EQ(hipSetDevice(r), hipSuccess);
        ASSERT_EQ(hipGetDeviceProperties(&props, r), hipSuccess);
        hipLaunchKernelGGL(lsaFlowExchangeKernel, dim3(1),
                           dim3(props.warpSize * (kWorkerWarps + NCCL_FLOW_SIGNAL_WARPS_PER_PART)), 0,
                           ranks[r].stream, ranks[r].devComm, ranks[r].window, ranks[r].input, ranks[r].output);
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }
    for(int r = 0; r < kRanks; r++)
    {
        ASSERT_EQ(hipSetDevice(r), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(ranks[r].stream), hipSuccess);
        ASSERT_EQ(hipMemcpy(host.data(), ranks[r].output, kElts * sizeof(int), hipMemcpyDeviceToHost), hipSuccess);
        const int peer       = (r + 1) % kRanks;
        int       mismatches = 0;
        for(int i = 0; i < kElts; i++) mismatches += host[i] != peer * 1000003 + i;
        EXPECT_EQ(mismatches, 0) << "rank " << r << " did not receive rank " << peer << "'s input over LSA Flow";
    }

    for(int r = 0; r < kRanks; r++)
    {
        (void)hipSetDevice(r);
        if(ranks[r].devCommCreated) (void)ncclDevCommDestroy(ranks[r].comm, &ranks[r].devComm);
        if(ranks[r].window) (void)ncclCommWindowDeregister(ranks[r].comm, ranks[r].window);
        (void)ncclMemFree(ranks[r].flow);
        (void)hipFree(ranks[r].input);
        (void)hipFree(ranks[r].output);
        (void)hipStreamDestroy(ranks[r].stream);
        (void)ncclCommDestroy(ranks[r].comm);
    }
}

} // namespace

// Single-process, 2-GPU test; the bootstrap stays on loopback as in DeviceApiTests.cpp.
TEST(DeviceApiFlow, LsaSimpleExchange)
{
    RUN_ISOLATED_TESTS(ProcessIsolatedTestRunner::TestConfig("DeviceApiFlow.LsaSimpleExchange",
                                                             []() { runLsaFlowExchangeTest(); })
                           .withEnvironment({{"NCCL_CUMEM_ENABLE", "1"},
                                             {"NCCL_WIN_ENABLE", "1"},
                                             {"NCCL_SOCKET_IFNAME", "lo"}})
                           .withTimeout(std::chrono::seconds(60))
                           .withNumGpus(kRanks));
}

} // namespace RcclUnitTesting
