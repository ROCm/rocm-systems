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
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "nccl_device/impl/core__funcs.h"
#include "nccl_device/impl/lsa_barrier__funcs.h"

#include "common/ProcessIsolatedTestRunner.hpp"

// `cuda::memory_order` and `cuda::atomic_ref` are polyfilled on HIP via
// nccl_device/hip_compat.h (transitively included by lsa_barrier__funcs.h),
// so the test kernel can use the same upstream-NCCL signatures on both
// platforms. See [RCCL] PR #6259 (a572f1aabf).

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
__global__ void
    lsaReadPeerValueKernel(ncclWindow_t inputWindow, int* outputValue, ncclDevComm_t devComm)
{
    ncclLsaBarrierSession<ncclCoopCta> barrier(ncclCoopCta(),
                                               devComm,
                                               ncclTeamLsa(devComm),
                                               devComm.lsaBarrier,
                                               blockIdx.x);
    barrier.sync(ncclCoopCta(), cuda::memory_order_relaxed);

    if(threadIdx.x == 0)
    {
        const int peer      = (devComm.rank + 1) % devComm.nRanks;
        int*      peerInput = reinterpret_cast<int*>(ncclGetLsaPointer(inputWindow, 0, peer));
        outputValue[0]      = peerInput[0];
    }

    barrier.sync(ncclCoopCta(), cuda::memory_order_release);
}

struct DeviceApiRankResources
{
    int           device         = -1;
    ncclComm_t    comm           = nullptr;
    hipStream_t   stream         = nullptr;
    int*          inputBuffer    = nullptr;
    int*          outputBuffer   = nullptr;
    ncclWindow_t  inputWindow    = nullptr;
    ncclDevComm_t devComm        = {};
    bool          devCommCreated = false;
};

struct DeviceApiResources
{
    explicit DeviceApiResources(int rankCount) : ranks(static_cast<size_t>(rankCount))
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

    ASSERT_EQ(ncclCommInitAll(comms.data(), static_cast<int>(comms.size()), nullptr), ncclSuccess);

    for(size_t rank = 0; rank < resources.ranks.size(); ++rank)
        resources.ranks[rank].comm = comms[rank];
}

static void allocateInputBuffer(DeviceApiRankResources& rank, int inputValue)
{
    ASSERT_EQ(hipSetDevice(rank.device), hipSuccess);

    void* rawInput = nullptr;
    ASSERT_EQ(ncclMemAlloc(&rawInput, kBufferBytes), ncclSuccess);
    rank.inputBuffer = static_cast<int*>(rawInput);

    ASSERT_EQ(hipMemcpy(rank.inputBuffer, &inputValue, kBufferBytes, hipMemcpyHostToDevice),
              hipSuccess);
}

static void allocatePositiveBuffers(DeviceApiResources&                    resources,
                                    const std::array<int, kPositiveRanks>& inputValues)
{
    for(size_t rankIdx = 0; rankIdx < resources.ranks.size(); ++rankIdx)
    {
        auto& rank = resources.ranks[rankIdx];
        ASSERT_EQ(hipSetDevice(rank.device), hipSuccess);
        ASSERT_EQ(hipStreamCreate(&rank.stream), hipSuccess);

        allocateInputBuffer(rank, inputValues[rankIdx]);

        ASSERT_EQ(hipMalloc(reinterpret_cast<void**>(&rank.outputBuffer), kBufferBytes),
                  hipSuccess);
        ASSERT_EQ(hipMemset(rank.outputBuffer, 0, kBufferBytes), hipSuccess);
    }
}

// Registers a symmetric window per rank and returns the registration result
// instead of asserting success, so callers can distinguish an unsupported
// configuration from an unexpected failure. In NCCL 2.30 an unsupported
// configuration (cuMem / symmetric windows disabled) is rejected here at
// ncclCommWindowRegister with ncclInvalidUsage, whereas older releases accepted
// registration and rejected later at ncclDevCommCreate. Returns the first
// non-success per-rank result, otherwise the ncclGroupEnd() result.
static ncclResult_t tryRegisterInputWindows(DeviceApiResources& resources)
{
    if(ncclGroupStart() != ncclSuccess)
        return ncclInternalError;

    std::vector<ncclResult_t> results(resources.ranks.size(), ncclSuccess);
    for(size_t rankIdx = 0; rankIdx < resources.ranks.size(); ++rankIdx)
    {
        auto& rank       = resources.ranks[rankIdx];
        results[rankIdx] = ncclCommWindowRegister(rank.comm,
                                                  rank.inputBuffer,
                                                  kBufferBytes,
                                                  &rank.inputWindow,
                                                  NCCL_WIN_COLL_SYMMETRIC);
    }

    const ncclResult_t groupResult = ncclGroupEnd();

    for(const auto& result : results)
        if(result != ncclSuccess)
            return result;
    return groupResult;
}

static void clearHipErrorState()
{
    (void)hipGetLastError();
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

    // Symmetric-window registration is a precondition for this test, not the
    // behavior under test. If the runtime reports the configuration is
    // unsupported (ncclInvalidUsage), skip rather than fail; any other error is
    // unexpected and should fail hard.
    const ncclResult_t registerResult = tryRegisterInputWindows(resources);
    if(registerResult == ncclInvalidUsage)
        GTEST_SKIP() << "Symmetric window registration is unsupported on this configuration.";
    ASSERT_EQ(registerResult, ncclSuccess);

    for(const auto& rank : resources.ranks)
    {
        if(rank.inputWindow == nullptr)
            GTEST_SKIP() << "Symmetric window registration is unavailable on this configuration.";
    }

    ncclDevCommRequirements_t requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    requirements.lsaBarrierCount           = kBlocksPerRank;

    ASSERT_EQ(ncclGroupStart(), ncclSuccess);

    std::vector<ncclResult_t> createResults(resources.ranks.size(), ncclSuccess);
    for(size_t rankIdx = 0; rankIdx < resources.ranks.size(); ++rankIdx)
    {
        auto& rank             = resources.ranks[rankIdx];
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
        clearHipErrorState();

        hipLaunchKernelGGL(lsaReadPeerValueKernel,
                           dim3(kBlocksPerRank),
                           dim3(kThreadsPerBlock),
                           0,
                           rank.stream,
                           rank.inputWindow,
                           rank.outputBuffer,
                           rank.devComm);
        const hipError_t launchError = hipGetLastError();
        ASSERT_EQ(launchError, hipSuccess) << "lsaReadPeerValueKernel launch failed on device "
                                           << rank.device << ": " << hipGetErrorString(launchError);
    }

    for(auto& rank : resources.ranks)
    {
        ASSERT_EQ(hipSetDevice(rank.device), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(rank.stream), hipSuccess);
    }

    const std::array<int, kPositiveRanks> expectedOutputs = {inputValues[1], inputValues[0]};

    for(size_t rankIdx = 0; rankIdx < resources.ranks.size(); ++rankIdx)
    {
        auto& rank       = resources.ranks[rankIdx];
        int   hostOutput = 0;

        ASSERT_EQ(hipSetDevice(rank.device), hipSuccess);
        ASSERT_EQ(hipMemcpy(&hostOutput, rank.outputBuffer, kBufferBytes, hipMemcpyDeviceToHost),
                  hipSuccess);
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

    // The device API is gated off in these configs (cuMem / symmetric windows
    // disabled). NCCL 2.30 rejects the unsupported configuration at symmetric-
    // window registration; older releases accepted registration and rejected
    // later at ncclDevCommCreate. Accept the rejection at whichever point the
    // runtime raises it.
    const ncclResult_t registerResult = tryRegisterInputWindows(resources);
    if(registerResult != ncclSuccess)
    {
        EXPECT_EQ(registerResult, ncclInvalidUsage);
        return;
    }

    ncclDevCommRequirements_t requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    requirements.lsaBarrierCount           = kBlocksPerRank;

    const ncclResult_t createResult
        = ncclDevCommCreate(resources.ranks[0].comm, &requirements, &resources.ranks[0].devComm);

    EXPECT_EQ(createResult, ncclInvalidUsage);
    if(createResult == ncclSuccess)
        resources.ranks[0].devCommCreated = true;
}

// Per-test config notes:
//   - These are single-process multi-GPU tests (ncclCommInitAll). All rank-to-
//     rank bootstrap traffic is intra-host by construction, so we pin NCCL's
//     bootstrap socket to loopback. This makes the tests self-contained on
//     any host network configuration (single-NIC, multi-NIC, containers with
//     shared host netns, etc.) without relying on the caller to set
//     NCCL_SOCKET_IFNAME.
//   - withNumGpus(N) declares how many physical GPUs the test occupies so
//     ProcessIsolatedTestRunner can schedule it correctly under
//     maxParallelJobs > 1. The positive test runs 2 ranks (one per GPU); the
//     negative tests run 1 rank but still hipSetDevice / hipMalloc / call
//     ncclCommInitAll on a real GPU, so they declare 1 slot too.
static ProcessIsolatedTestRunner::TestConfig
    makeDeviceApiEnabledConfig(const std::string& name, std::function<void()> testFn)
{
    return ProcessIsolatedTestRunner::TestConfig(name, testFn)
        .withEnvironment({
            { "NCCL_CUMEM_ENABLE",  "1"},
            {   "NCCL_WIN_ENABLE",  "1"},
            {"NCCL_SOCKET_IFNAME", "lo"}
    })
        .withTimeout(std::chrono::seconds(60))
        .withNumGpus(kPositiveRanks);
}

// The negative configs additionally pin NCCL_IB_DISABLE=1: these are single-
// node 2-GPU tests that don't need IB/RDMA, and on the unsupported-config
// rejection path NCCL 2.30 otherwise enters the IB transport and can surface
// an environment-dependent ncclSystemError (ibv_create_qp) that masks the
// clean ncclInvalidUsage gating signal the test is asserting.
static ProcessIsolatedTestRunner::TestConfig makeCuMemDisabledConfig(const std::string&    name,
                                                                     std::function<void()> testFn)
{
    return ProcessIsolatedTestRunner::TestConfig(name, testFn)
        .withEnvironment({
            { "NCCL_CUMEM_ENABLE",  "0"},
            {   "NCCL_WIN_ENABLE",  "1"},
            {"NCCL_SOCKET_IFNAME", "lo"},
            {   "NCCL_IB_DISABLE",  "1"}
    })
        .withTimeout(std::chrono::seconds(60))
        .withNumGpus(kNegativeRanks);
}

static ProcessIsolatedTestRunner::TestConfig makeWinDisabledConfig(const std::string&    name,
                                                                   std::function<void()> testFn)
{
    return ProcessIsolatedTestRunner::TestConfig(name, testFn)
        .withEnvironment({
            { "NCCL_CUMEM_ENABLE",  "1"},
            {   "NCCL_WIN_ENABLE",  "0"},
            {"NCCL_SOCKET_IFNAME", "lo"},
            {   "NCCL_IB_DISABLE",  "1"}
    })
        .withTimeout(std::chrono::seconds(60))
        .withNumGpus(kNegativeRanks);
}

} // namespace

TEST(DeviceApi, LsaRemoteRead)
{
    RUN_ISOLATED_TESTS(makeDeviceApiEnabledConfig("DeviceApi.LsaRemoteRead",
                                                  []() { runPositiveLsaRemoteReadTest(); }));
}

TEST(DeviceApi, CuMemDisabled)
{
    RUN_ISOLATED_TESTS(makeCuMemDisabledConfig("DeviceApi.CuMemDisabled",
                                               []() { runDevCommCreateFailureTest(); }));
}

TEST(DeviceApi, WinDisabled)
{
    RUN_ISOLATED_TESTS(
        makeWinDisabledConfig("DeviceApi.WinDisabled", []() { runDevCommCreateFailureTest(); }));
}

} // namespace RcclUnitTesting
