/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <array>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

#include "common/DeviceApiLocal.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting
{

namespace
{

constexpr int    kPositiveRanks    = 2;
constexpr int    kNegativeRanks    = 1;
constexpr int    kBlocksPerRank    = 1;
constexpr int    kThreadsPerBlock  = 64;
constexpr size_t kBufferBytes      = sizeof(int);
constexpr int    kNegativeTestSeed = 7;

// Each rank reads one integer from its peer through a symmetric window.
__global__ void lsaReadPeerValueKernel(
    ncclWindow_t inputWindow, int* outputValue, ncclDevComm_t devComm
)
{
    ncclLsaBarrierSession<ncclCoopCta> barrier(
        ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x
    );
    barrier.sync(ncclCoopCta(), cuda::memory_order_relaxed);

    if(threadIdx.x == 0)
    {
        const int peer = (devComm.rank + 1) % devComm.nRanks;
        int*      peerInput
            = reinterpret_cast<int*>(ncclGetLsaPointer(inputWindow, 0, peer));
        outputValue[0] = peerInput[0];
    }

    barrier.sync(ncclCoopCta(), cuda::memory_order_release);
}

struct DeviceApiRankResources
{
    int            device         = -1;
    ncclComm_t     comm           = nullptr;
    hipStream_t    stream         = nullptr;
    int*           inputBuffer    = nullptr;
    int*           outputBuffer   = nullptr;
    ncclWindow_t   inputWindow    = nullptr;
    ncclDevComm_t  devComm        = {};
    bool           devCommCreated = false;
};

struct DeviceApiResources
{
    explicit DeviceApiResources(int rankCount)
        : ranks(static_cast<size_t>(rankCount))
    {
        for(int rank = 0; rank < rankCount; ++rank)
            ranks[rank].device = rank;
    }

    ~DeviceApiResources()
    {
        for(auto& rank : ranks)
        {
            if(rank.device >= 0)
                (void)hipSetDevice(rank.device);

            if(rank.stream != nullptr)
                (void)hipStreamSynchronize(rank.stream);

            if(rank.devCommCreated && rank.comm != nullptr)
                (void)ncclDevCommDestroy(rank.comm, &rank.devComm);

            if(rank.inputWindow != nullptr && rank.comm != nullptr)
                (void)ncclCommWindowDeregister(rank.comm, rank.inputWindow);

            if(rank.outputBuffer != nullptr)
                (void)hipFree(rank.outputBuffer);

            if(rank.inputBuffer != nullptr)
                (void)ncclMemFree(rank.inputBuffer);

            if(rank.stream != nullptr)
                (void)hipStreamDestroy(rank.stream);

            if(rank.comm != nullptr)
                (void)ncclCommDestroy(rank.comm);
        }
    }

    std::vector<DeviceApiRankResources> ranks;
};

static int getVisibleGpuCount()
{
    int gpuCount = 0;
    return hipGetDeviceCount(&gpuCount) == hipSuccess ? gpuCount : 0;
}

static bool hasFullDirectP2p(int gpuCount)
{
    for(int src = 0; src < gpuCount; ++src)
    {
        for(int dst = 0; dst < gpuCount; ++dst)
        {
            if(src == dst)
                continue;

            int canAccessPeer = 0;
            if(hipDeviceCanAccessPeer(&canAccessPeer, src, dst) != hipSuccess || !canAccessPeer)
                return false;
        }
    }

    return true;
}

static void initializeCommunicators(DeviceApiResources& resources)
{
    std::vector<ncclComm_t> comms(resources.ranks.size(), nullptr);

    ASSERT_EQ(
        ncclCommInitAll(comms.data(), static_cast<int>(comms.size()), nullptr), ncclSuccess
    );

    for(size_t rank = 0; rank < resources.ranks.size(); ++rank)
        resources.ranks[rank].comm = comms[rank];
}

static void allocateInputBuffer(DeviceApiRankResources& rank, int inputValue)
{
    ASSERT_EQ(hipSetDevice(rank.device), hipSuccess);

    void* rawInput = nullptr;
    ASSERT_EQ(ncclMemAlloc(&rawInput, kBufferBytes), ncclSuccess);
    rank.inputBuffer = static_cast<int*>(rawInput);

    ASSERT_EQ(
        hipMemcpy(rank.inputBuffer, &inputValue, kBufferBytes, hipMemcpyHostToDevice),
        hipSuccess
    );
}

static void allocatePositiveBuffers(
    DeviceApiResources& resources, const std::array<int, kPositiveRanks>& inputValues
)
{
    for(size_t rankIdx = 0; rankIdx < resources.ranks.size(); ++rankIdx)
    {
        auto& rank = resources.ranks[rankIdx];
        ASSERT_EQ(hipSetDevice(rank.device), hipSuccess);
        ASSERT_EQ(hipStreamCreate(&rank.stream), hipSuccess);

        allocateInputBuffer(rank, inputValues[rankIdx]);

        ASSERT_EQ(hipMalloc(reinterpret_cast<void**>(&rank.outputBuffer), kBufferBytes), hipSuccess);
        ASSERT_EQ(hipMemset(rank.outputBuffer, 0, kBufferBytes), hipSuccess);
    }
}

static void registerInputWindows(DeviceApiResources& resources)
{
    ASSERT_EQ(ncclGroupStart(), ncclSuccess);

    std::vector<ncclResult_t> results(resources.ranks.size(), ncclSuccess);
    for(size_t rankIdx = 0; rankIdx < resources.ranks.size(); ++rankIdx)
    {
        auto& rank   = resources.ranks[rankIdx];
        results[rankIdx] = ncclCommWindowRegister(
            rank.comm,
            rank.inputBuffer,
            kBufferBytes,
            &rank.inputWindow,
            NCCL_WIN_COLL_SYMMETRIC
        );
    }

    const ncclResult_t groupResult = ncclGroupEnd();

    for(const auto& result : results)
        ASSERT_EQ(result, ncclSuccess);
    ASSERT_EQ(groupResult, ncclSuccess);
}

static void runPositiveLsaRemoteReadTest()
{
    if(getVisibleGpuCount() < kPositiveRanks)
        GTEST_SKIP() << "This test requires at least 2 visible GPUs.";

    if(!hasFullDirectP2p(kPositiveRanks))
        GTEST_SKIP() << "This test requires direct P2P access between the first 2 GPUs.";

    DeviceApiResources resources(kPositiveRanks);
    initializeCommunicators(resources);

    const std::array<int, kPositiveRanks> inputValues = {7, 11};
    allocatePositiveBuffers(resources, inputValues);
    registerInputWindows(resources);

    for(const auto& rank : resources.ranks)
    {
        if(rank.inputWindow == nullptr)
            GTEST_SKIP() << "Symmetric window registration is unavailable on this configuration.";
    }

    ncclDevCommRequirements_t requirements = {};
    requirements.lsaBarrierCount           = kBlocksPerRank;

    ASSERT_EQ(ncclGroupStart(), ncclSuccess);

    std::vector<ncclResult_t> createResults(resources.ranks.size(), ncclSuccess);
    for(size_t rankIdx = 0; rankIdx < resources.ranks.size(); ++rankIdx)
    {
        auto& rank          = resources.ranks[rankIdx];
        createResults[rankIdx] = ncclDevCommCreate(rank.comm, &requirements, &rank.devComm);
    }

    const ncclResult_t groupResult = ncclGroupEnd();

    for(size_t rankIdx = 0; rankIdx < resources.ranks.size(); ++rankIdx)
    {
        if(createResults[rankIdx] == ncclSuccess)
            resources.ranks[rankIdx].devCommCreated = true;
    }

    bool unsupportedConfiguration = (groupResult == ncclInvalidUsage);
    for(const auto& result : createResults)
        unsupportedConfiguration |= (result == ncclInvalidUsage);

    if(unsupportedConfiguration)
        GTEST_SKIP() << "Symmetric device API is unsupported on this configuration.";

    for(const auto& result : createResults)
        ASSERT_EQ(result, ncclSuccess);
    ASSERT_EQ(groupResult, ncclSuccess);

    for(auto& rank : resources.ranks)
    {
        ASSERT_EQ(hipSetDevice(rank.device), hipSuccess);
        hipLaunchKernelGGL(
            lsaReadPeerValueKernel,
            dim3(kBlocksPerRank),
            dim3(kThreadsPerBlock),
            0,
            rank.stream,
            rank.inputWindow,
            rank.outputBuffer,
            rank.devComm
        );
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    for(auto& rank : resources.ranks)
    {
        ASSERT_EQ(hipSetDevice(rank.device), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(rank.stream), hipSuccess);
    }

    const std::array<int, kPositiveRanks> expectedOutputs = {inputValues[1], inputValues[0]};

    for(size_t rankIdx = 0; rankIdx < resources.ranks.size(); ++rankIdx)
    {
        auto& rank = resources.ranks[rankIdx];
        int   hostOutput = 0;

        ASSERT_EQ(hipSetDevice(rank.device), hipSuccess);
        ASSERT_EQ(
            hipMemcpy(&hostOutput, rank.outputBuffer, kBufferBytes, hipMemcpyDeviceToHost),
            hipSuccess
        );
        EXPECT_EQ(hostOutput, expectedOutputs[rankIdx]);
    }
}

static void runDevCommCreateFailureTest()
{
    if(getVisibleGpuCount() < kNegativeRanks)
        GTEST_SKIP() << "This test requires at least 1 visible GPU.";

    DeviceApiResources resources(kNegativeRanks);
    initializeCommunicators(resources);
    allocateInputBuffer(resources.ranks[0], kNegativeTestSeed);
    registerInputWindows(resources);

    ncclDevCommRequirements_t requirements = {};
    requirements.lsaBarrierCount           = kBlocksPerRank;

    const ncclResult_t createResult
        = ncclDevCommCreate(resources.ranks[0].comm, &requirements, &resources.ranks[0].devComm);

    EXPECT_EQ(createResult, ncclInvalidUsage);
    if(createResult == ncclSuccess)
        resources.ranks[0].devCommCreated = true;
}

static ProcessIsolatedTestRunner::TestConfig makeDeviceApiEnabledConfig(
    const std::string& name, std::function<void()> testFn
)
{
    return ProcessIsolatedTestRunner::TestConfig(name, testFn)
        .withEnvironment({{"NCCL_CUMEM_ENABLE", "1"}, {"NCCL_WIN_ENABLE", "1"}})
        .withTimeout(std::chrono::seconds(60));
}

static ProcessIsolatedTestRunner::TestConfig makeCuMemDisabledConfig(
    const std::string& name, std::function<void()> testFn
)
{
    return ProcessIsolatedTestRunner::TestConfig(name, testFn)
        .withEnvironment({{"NCCL_CUMEM_ENABLE", "0"}, {"NCCL_WIN_ENABLE", "1"}})
        .withTimeout(std::chrono::seconds(60));
}

static ProcessIsolatedTestRunner::TestConfig makeWinDisabledConfig(
    const std::string& name, std::function<void()> testFn
)
{
    return ProcessIsolatedTestRunner::TestConfig(name, testFn)
        .withEnvironment({{"NCCL_CUMEM_ENABLE", "1"}, {"NCCL_WIN_ENABLE", "0"}})
        .withTimeout(std::chrono::seconds(60));
}

} // namespace

TEST(DeviceApi, LsaRemoteRead)
{
    RUN_ISOLATED_TESTS(
        makeDeviceApiEnabledConfig(
            "DeviceApi.LsaRemoteRead", []() { runPositiveLsaRemoteReadTest(); }
        )
    );
}

TEST(DeviceApi, CuMemDisabled)
{
    RUN_ISOLATED_TESTS(
        makeCuMemDisabledConfig(
            "DeviceApi.CuMemDisabled", []() { runDevCommCreateFailureTest(); }
        )
    );
}

TEST(DeviceApi, WinDisabled)
{
    RUN_ISOLATED_TESTS(
        makeWinDisabledConfig(
            "DeviceApi.WinDisabled", []() { runDevCommCreateFailureTest(); }
        )
    );
}

} // namespace RcclUnitTesting
