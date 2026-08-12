/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Tests for ncclLaunchPrepare/ncclLaunchKernel doneEvent stream-ordering.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <vector>

#include "common/DeviceBufferHelpers.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "common/ResourceGuards.hpp"
#include "common/TestChecks.hpp"

namespace RcclUnitTesting
{

// Isolated child pays full HIP init + up to kDoneEventTimeoutSeconds for the
// collective work. A hang (missing doneEvent edge) is caught by SIGKILL here.
static constexpr int kDoneEventTimeoutSeconds = 90;

// 4 MiB per device — large enough for real GPU work; the ordering race is a
// CPU-side scheduler decision independent of transfer volume.
static constexpr size_t kBufferBytes = 4ull * 1024 * 1024;
static constexpr size_t kElemCount   = kBufferBytes / sizeof(float);

// 20 alternating-stream launches suffice: the missing doneEvent edge fires on
// iteration 1 (first stream flip) and produces a detectable hang or wrong sum.
static constexpr int kIterations = 20;

// Cap on communicators/GPUs used by single-node tests.
static constexpr int kMaxGpus = 8;

class DoneEventOrdering : public ::testing::Test
{
    // All work runs under process isolation; no fixture setup/teardown.
};

// Helper: nIterations ungrouped alternating-stream AllReduces, no intermediate sync.
static void runAlternatingStreamAllReduces(int nGpus, int nIterations)
{
    std::vector<int> devices(nGpus);
    std::iota(devices.begin(), devices.end(), 0);

    // --- communicator init ---
    std::vector<ncclComm_t> comms(nGpus, nullptr);
    {
        ncclResult_t res = ncclCommInitAll(comms.data(), nGpus, devices.data());
        ASSERT_EQ(res, ncclSuccess) << "ncclCommInitAll: " << ncclGetErrorString(res);
    }
    // RAII: destroy all communicators on scope exit regardless of assertion failures.
    std::vector<RCCLTestGuards::NcclCommAutoGuard> commGuards;
    commGuards.reserve(nGpus);
    for(int i = 0; i < nGpus; ++i)
        commGuards.push_back(RCCLTestGuards::makeCommAutoGuard(comms[i]));

    // --- per-device streams and buffers ---
    std::vector<RCCLTestGuards::HipStreamAutoGuard>    streamAGuards, streamBGuards;
    std::vector<RCCLTestGuards::DeviceBufferAutoGuard> sendbufGuards, recvbufGuards;
    streamAGuards.reserve(nGpus);
    streamBGuards.reserve(nGpus);
    sendbufGuards.reserve(nGpus);
    recvbufGuards.reserve(nGpus);

    // Keep raw pointers for RCCL calls (guards own the lifetime).
    std::vector<hipStream_t> streamA(nGpus, nullptr);
    std::vector<hipStream_t> streamB(nGpus, nullptr);
    std::vector<float*>      sendbuf(nGpus, nullptr);
    std::vector<float*>      recvbuf(nGpus, nullptr);

    for(int i = 0; i < nGpus; ++i)
    {
        HIP_CHECK(hipSetDevice(devices[i]));

        HIP_CHECK(hipStreamCreate(&streamA[i]));
        streamAGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(streamA[i]));

        HIP_CHECK(hipStreamCreate(&streamB[i]));
        streamBGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(streamB[i]));

        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&sendbuf[i]), kBufferBytes));
        sendbufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(sendbuf[i]));

        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&recvbuf[i]), kBufferBytes));
        recvbufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(recvbuf[i]));

        // Fill sendbuf with (i+1); out-of-place AllReduce never writes sendbuf,
        // so the value is stable across all nIterations launches.
        HIP_CHECK(RCCLTestHelpers::initializeBufferWithPattern<float>(
            sendbuf[i], kElemCount,
            [i](size_t) { return static_cast<float>(i + 1); }));
        HIP_CHECK(RCCLTestHelpers::zeroInitializeBuffer<float>(recvbuf[i], kElemCount));
    }

    // AllReduce-sum of (r+1) over ranks 0..nGpus-1 == nGpus*(nGpus+1)/2.
    // All terms are exactly representable; the sum is too, so strict == is safe.
    const float expected = static_cast<float>(nGpus * (nGpus + 1) / 2);

    // Back-to-back ungrouped launches, alternating stream every iteration, with
    // NO synchronize in between. Ungrouped keeps planner->numStreams == 1
    // (fast path); the flip forces the lastStream != launchStream doneEvent
    // edge on every launch from iteration 1 onward.
    for(int iter = 0; iter < nIterations; ++iter)
    {
        for(int i = 0; i < nGpus; ++i)
        {
            hipStream_t  stream = (iter & 1) ? streamB[i] : streamA[i];
            ncclResult_t res    = ncclAllReduce(
                sendbuf[i], recvbuf[i], kElemCount, ncclFloat, ncclSum, comms[i], stream);
            ASSERT_EQ(res, ncclSuccess)
                << "ncclAllReduce failed iter=" << iter << " rank=" << i << ": "
                << ncclGetErrorString(res);
        }
    }

    // Synchronize once at the very end. If the ordering edge is missing this
    // either deadlocks (caught by withTimeout) or produces a wrong sum (caught below).
    for(int i = 0; i < nGpus; ++i)
    {
        HIP_CHECK(hipSetDevice(devices[i]));
        HIP_CHECK(hipStreamSynchronize(streamA[i]));
        HIP_CHECK(hipStreamSynchronize(streamB[i]));
    }

    // Correctness check via verifyBufferData.
    for(int i = 0; i < nGpus; ++i)
    {
        size_t firstErrIdx = 0;
        float  firstExpVal = 0.0f;
        float  firstActVal = 0.0f;
        bool   ok          = RCCLTestHelpers::verifyBufferData<float>(
            recvbuf[i],
            kElemCount,
            [expected](size_t) { return expected; },
            kElemCount,
            0.0,   // exact match; all values are exactly-representable integers
            &firstErrIdx,
            &firstExpVal,
            &firstActVal);
        EXPECT_TRUE(ok)
            << "rank " << i << ": AllReduce wrong at index " << firstErrIdx
            << " expected " << firstExpVal << " got " << firstActVal;
    }
    // Guards destroy streams, buffers, and comms on scope exit.
}

// Ungrouped alternating-stream AllReduces — tests the lastStream != launchStream doneEvent edge.
TEST_F(DoneEventOrdering, FastPathStreamAlternation)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "FastPathStreamAlternation",
            []()
            {
                int devCount = 0;
                HIP_CHECK(hipGetDeviceCount(&devCount));
                const int nGpus = std::min(devCount, kMaxGpus);
                if(nGpus < 2)
                    GTEST_SKIP() << "Requires >= 2 GPUs; found " << devCount;
                runAlternatingStreamAllReduces(nGpus, kIterations);
            })
            .withNumGpus(kMaxGpus)
            .withEnvironment({{"NCCL_DEBUG", "WARN"}})
            .withTimeout(std::chrono::seconds(kDoneEventTimeoutSeconds)));
}

// Two ncclCommDestroy+ncclCommInitAll reinit cycles, each followed by alternating-stream AllReduces.
TEST_F(DoneEventOrdering, PostReinitStreamAlternation)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "PostReinitStreamAlternation",
            []()
            {
                int devCount = 0;
                HIP_CHECK(hipGetDeviceCount(&devCount));
                const int nGpus = std::min(devCount, kMaxGpus);
                if(nGpus < 2)
                    GTEST_SKIP() << "Requires >= 2 GPUs; found " << devCount;

                std::vector<int> devices(nGpus);
                std::iota(devices.begin(), devices.end(), 0);

                // Per-device streams and buffers (live for the whole test).
                std::vector<RCCLTestGuards::HipStreamAutoGuard>    streamAGuards, streamBGuards;
                std::vector<RCCLTestGuards::DeviceBufferAutoGuard> sendbufGuards, recvbufGuards;
                streamAGuards.reserve(nGpus);
                streamBGuards.reserve(nGpus);
                sendbufGuards.reserve(nGpus);
                recvbufGuards.reserve(nGpus);

                std::vector<hipStream_t> streamA(nGpus, nullptr);
                std::vector<hipStream_t> streamB(nGpus, nullptr);
                std::vector<float*>      sendbuf(nGpus, nullptr);
                std::vector<float*>      recvbuf(nGpus, nullptr);

                for(int i = 0; i < nGpus; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices[i]));
                    HIP_CHECK(hipStreamCreate(&streamA[i]));
                    streamAGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(streamA[i]));
                    HIP_CHECK(hipStreamCreate(&streamB[i]));
                    streamBGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(streamB[i]));

                    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&sendbuf[i]), kBufferBytes));
                    sendbufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(sendbuf[i]));
                    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&recvbuf[i]), kBufferBytes));
                    recvbufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(recvbuf[i]));

                    HIP_CHECK(RCCLTestHelpers::initializeBufferWithPattern<float>(
                        sendbuf[i], kElemCount,
                        [i](size_t) { return static_cast<float>(i + 1); }));
                }

                const float expected = static_cast<float>(nGpus * (nGpus + 1) / 2);

                // Simulate checkpoint/restore: init → work → destroy → reinit → work.
                // Two reinit cycles to exercise that lastStreamValid resets correctly
                // on each ncclCommInitAll and the doneEvent edge fires on first flip.
                for(int cycle = 0; cycle < 2; ++cycle)
                {
                    std::vector<ncclComm_t> comms(nGpus, nullptr);
                    {
                        ncclResult_t res =
                            ncclCommInitAll(comms.data(), nGpus, devices.data());
                        ASSERT_EQ(res, ncclSuccess)
                            << "cycle " << cycle
                            << " ncclCommInitAll: " << ncclGetErrorString(res);
                    }
                    std::vector<RCCLTestGuards::NcclCommAutoGuard> commGuards;
                    commGuards.reserve(nGpus);
                    for(int i = 0; i < nGpus; ++i)
                        commGuards.push_back(RCCLTestGuards::makeCommAutoGuard(comms[i]));

                    // Zero recvbuf before each cycle.
                    for(int i = 0; i < nGpus; ++i)
                        HIP_CHECK(RCCLTestHelpers::zeroInitializeBuffer<float>(
                            recvbuf[i], kElemCount));

                    // Two alternating-stream launches: iter=0 → streamA (sets lastStream),
                    // iter=1 → streamB (fires the doneEvent edge — the critical path).
                    for(int iter = 0; iter < 2; ++iter)
                    {
                        for(int i = 0; i < nGpus; ++i)
                        {
                            hipStream_t  stream = (iter & 1) ? streamB[i] : streamA[i];
                            ncclResult_t res    = ncclAllReduce(
                                sendbuf[i], recvbuf[i], kElemCount, ncclFloat, ncclSum,
                                comms[i], stream);
                            ASSERT_EQ(res, ncclSuccess)
                                << "cycle " << cycle << " iter " << iter << " rank " << i
                                << ": " << ncclGetErrorString(res);
                        }
                    }

                    // Synchronize before verify and before commGuards destroy the comms.
                    for(int i = 0; i < nGpus; ++i)
                    {
                        HIP_CHECK(hipSetDevice(devices[i]));
                        HIP_CHECK(hipStreamSynchronize(streamA[i]));
                        HIP_CHECK(hipStreamSynchronize(streamB[i]));
                    }

                    for(int i = 0; i < nGpus; ++i)
                    {
                        size_t errIdx = 0;
                        float  expVal = 0.0f, actVal = 0.0f;
                        bool   ok = RCCLTestHelpers::verifyBufferData<float>(
                            recvbuf[i], kElemCount,
                            [expected](size_t) { return expected; },
                            kElemCount, 0.0, &errIdx, &expVal, &actVal);
                        EXPECT_TRUE(ok)
                            << "cycle " << cycle << " rank " << i
                            << ": wrong at index " << errIdx
                            << " expected " << expVal << " got " << actVal;
                    }
                    // commGuards destroy all comms at end of loop body (simulating checkpoint).
                }
            })
            .withNumGpus(kMaxGpus)
            .withEnvironment({{"NCCL_DEBUG", "WARN"}})
            .withTimeout(std::chrono::seconds(kDoneEventTimeoutSeconds)));
}

// Negative control: same stream every iteration, no doneEvent edge triggered.
TEST_F(DoneEventOrdering, SameStreamBaseline)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "SameStreamBaseline",
            []()
            {
                int devCount = 0;
                HIP_CHECK(hipGetDeviceCount(&devCount));
                const int nGpus = std::min(devCount, kMaxGpus);
                if(nGpus < 2)
                    GTEST_SKIP() << "Requires >= 2 GPUs; found " << devCount;

                std::vector<int> devices(nGpus);
                std::iota(devices.begin(), devices.end(), 0);

                std::vector<ncclComm_t> comms(nGpus, nullptr);
                {
                    ncclResult_t res = ncclCommInitAll(comms.data(), nGpus, devices.data());
                    ASSERT_EQ(res, ncclSuccess) << ncclGetErrorString(res);
                }
                std::vector<RCCLTestGuards::NcclCommAutoGuard> commGuards;
                commGuards.reserve(nGpus);
                for(int i = 0; i < nGpus; ++i)
                    commGuards.push_back(RCCLTestGuards::makeCommAutoGuard(comms[i]));

                std::vector<RCCLTestGuards::HipStreamAutoGuard>    streamGuards;
                std::vector<RCCLTestGuards::DeviceBufferAutoGuard> sendbufGuards, recvbufGuards;
                streamGuards.reserve(nGpus);
                sendbufGuards.reserve(nGpus);
                recvbufGuards.reserve(nGpus);

                std::vector<hipStream_t> streams(nGpus, nullptr);
                std::vector<float*>      sendbuf(nGpus, nullptr);
                std::vector<float*>      recvbuf(nGpus, nullptr);

                for(int i = 0; i < nGpus; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices[i]));
                    HIP_CHECK(hipStreamCreate(&streams[i]));
                    streamGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(streams[i]));

                    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&sendbuf[i]), kBufferBytes));
                    sendbufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(sendbuf[i]));
                    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&recvbuf[i]), kBufferBytes));
                    recvbufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(recvbuf[i]));

                    HIP_CHECK(RCCLTestHelpers::initializeBufferWithPattern<float>(
                        sendbuf[i], kElemCount,
                        [i](size_t) { return static_cast<float>(i + 1); }));
                    HIP_CHECK(RCCLTestHelpers::zeroInitializeBuffer<float>(recvbuf[i], kElemCount));
                }

                const float expected = static_cast<float>(nGpus * (nGpus + 1) / 2);

                for(int iter = 0; iter < kIterations; ++iter)
                {
                    for(int i = 0; i < nGpus; ++i)
                    {
                        ncclResult_t res = ncclAllReduce(
                            sendbuf[i], recvbuf[i], kElemCount, ncclFloat, ncclSum,
                            comms[i], streams[i]);
                        ASSERT_EQ(res, ncclSuccess)
                            << "iter=" << iter << " rank=" << i << ": "
                            << ncclGetErrorString(res);
                    }
                }

                for(int i = 0; i < nGpus; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices[i]));
                    HIP_CHECK(hipStreamSynchronize(streams[i]));
                }

                for(int i = 0; i < nGpus; ++i)
                {
                    size_t errIdx = 0;
                    float  expVal = 0.0f, actVal = 0.0f;
                    bool   ok = RCCLTestHelpers::verifyBufferData<float>(
                        recvbuf[i], kElemCount,
                        [expected](size_t) { return expected; },
                        kElemCount, 0.0, &errIdx, &expVal, &actVal);
                    EXPECT_TRUE(ok)
                        << "SameStream rank " << i << ": wrong at index " << errIdx
                        << " expected " << expVal << " got " << actVal;
                }
            })
            .withNumGpus(kMaxGpus)
            .withEnvironment({{"NCCL_DEBUG", "WARN"}})
            .withTimeout(std::chrono::seconds(kDoneEventTimeoutSeconds)));
}

// Alternating-stream AllReduces wrapped in ncclGroupStart/End — verifies group calls don't break doneEvent ordering.
TEST_F(DoneEventOrdering, GroupedPathBaseline)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "GroupedPathBaseline",
            []()
            {
                int devCount = 0;
                HIP_CHECK(hipGetDeviceCount(&devCount));
                const int nGpus = std::min(devCount, kMaxGpus);
                if(nGpus < 2)
                    GTEST_SKIP() << "Requires >= 2 GPUs; found " << devCount;

                std::vector<int> devices(nGpus);
                std::iota(devices.begin(), devices.end(), 0);

                std::vector<ncclComm_t> comms(nGpus, nullptr);
                {
                    ncclResult_t res = ncclCommInitAll(comms.data(), nGpus, devices.data());
                    ASSERT_EQ(res, ncclSuccess) << ncclGetErrorString(res);
                }
                std::vector<RCCLTestGuards::NcclCommAutoGuard> commGuards;
                commGuards.reserve(nGpus);
                for(int i = 0; i < nGpus; ++i)
                    commGuards.push_back(RCCLTestGuards::makeCommAutoGuard(comms[i]));

                std::vector<RCCLTestGuards::HipStreamAutoGuard>    streamAGuards, streamBGuards;
                std::vector<RCCLTestGuards::DeviceBufferAutoGuard> sendbufGuards, recvbufGuards;
                streamAGuards.reserve(nGpus);
                streamBGuards.reserve(nGpus);
                sendbufGuards.reserve(nGpus);
                recvbufGuards.reserve(nGpus);

                std::vector<hipStream_t> streamA(nGpus, nullptr);
                std::vector<hipStream_t> streamB(nGpus, nullptr);
                std::vector<float*>      sendbuf(nGpus, nullptr);
                std::vector<float*>      recvbuf(nGpus, nullptr);

                for(int i = 0; i < nGpus; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices[i]));
                    HIP_CHECK(hipStreamCreate(&streamA[i]));
                    streamAGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(streamA[i]));
                    HIP_CHECK(hipStreamCreate(&streamB[i]));
                    streamBGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(streamB[i]));

                    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&sendbuf[i]), kBufferBytes));
                    sendbufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(sendbuf[i]));
                    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&recvbuf[i]), kBufferBytes));
                    recvbufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(recvbuf[i]));

                    HIP_CHECK(RCCLTestHelpers::initializeBufferWithPattern<float>(
                        sendbuf[i], kElemCount,
                        [i](size_t) { return static_cast<float>(i + 1); }));
                    HIP_CHECK(RCCLTestHelpers::zeroInitializeBuffer<float>(recvbuf[i], kElemCount));
                }

                const float expected = static_cast<float>(nGpus * (nGpus + 1) / 2);

                // Each iteration: all ranks enqueued inside one ncclGroupStart/End.
                // Each comm still sees numStreams == 1 (one stream per comm per iteration),
                // so the fast launch path and doneEvent ordering logic still run.
                for(int iter = 0; iter < kIterations; ++iter)
                {
                    ncclGroupStart();
                    for(int i = 0; i < nGpus; ++i)
                    {
                        hipStream_t  stream = (iter & 1) ? streamB[i] : streamA[i];
                        ncclResult_t res    = ncclAllReduce(
                            sendbuf[i], recvbuf[i], kElemCount, ncclFloat, ncclSum,
                            comms[i], stream);
                        ASSERT_EQ(res, ncclSuccess)
                            << "iter=" << iter << " rank=" << i << ": "
                            << ncclGetErrorString(res);
                    }
                    ncclResult_t endRes = ncclGroupEnd();
                    ASSERT_EQ(endRes, ncclSuccess)
                        << "ncclGroupEnd iter=" << iter << ": " << ncclGetErrorString(endRes);
                }

                for(int i = 0; i < nGpus; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices[i]));
                    HIP_CHECK(hipStreamSynchronize(streamA[i]));
                    HIP_CHECK(hipStreamSynchronize(streamB[i]));
                }

                for(int i = 0; i < nGpus; ++i)
                {
                    size_t errIdx = 0;
                    float  expVal = 0.0f, actVal = 0.0f;
                    bool   ok = RCCLTestHelpers::verifyBufferData<float>(
                        recvbuf[i], kElemCount,
                        [expected](size_t) { return expected; },
                        kElemCount, 0.0, &errIdx, &expVal, &actVal);
                    EXPECT_TRUE(ok)
                        << "Grouped rank " << i << ": wrong at index " << errIdx
                        << " expected " << expVal << " got " << actVal;
                }
            })
            .withNumGpus(kMaxGpus)
            .withEnvironment({{"NCCL_DEBUG", "WARN"}})
            .withTimeout(std::chrono::seconds(kDoneEventTimeoutSeconds)));
}

// First AllReduce on hipStreamDefault, second on a named stream — tests lastStreamValid nullptr handling.
TEST_F(DoneEventOrdering, DefaultStreamToNamedStream)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "DefaultStreamToNamedStream",
            []()
            {
                int devCount = 0;
                HIP_CHECK(hipGetDeviceCount(&devCount));
                const int nGpus = std::min(devCount, kMaxGpus);
                if(nGpus < 2)
                    GTEST_SKIP() << "Requires >= 2 GPUs; found " << devCount;

                std::vector<int> devices(nGpus);
                std::iota(devices.begin(), devices.end(), 0);

                std::vector<ncclComm_t> comms(nGpus, nullptr);
                {
                    ncclResult_t res = ncclCommInitAll(comms.data(), nGpus, devices.data());
                    ASSERT_EQ(res, ncclSuccess) << ncclGetErrorString(res);
                }
                std::vector<RCCLTestGuards::NcclCommAutoGuard> commGuards;
                commGuards.reserve(nGpus);
                for(int i = 0; i < nGpus; ++i)
                    commGuards.push_back(RCCLTestGuards::makeCommAutoGuard(comms[i]));

                std::vector<RCCLTestGuards::HipStreamAutoGuard>    namedStreamGuards;
                std::vector<RCCLTestGuards::DeviceBufferAutoGuard> sendbufGuards, recvbufGuards;
                namedStreamGuards.reserve(nGpus);
                sendbufGuards.reserve(nGpus);
                recvbufGuards.reserve(nGpus);

                std::vector<hipStream_t> namedStream(nGpus, nullptr);
                std::vector<float*>      sendbuf(nGpus, nullptr);
                std::vector<float*>      recvbuf(nGpus, nullptr);

                for(int i = 0; i < nGpus; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices[i]));
                    HIP_CHECK(hipStreamCreate(&namedStream[i]));
                    namedStreamGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(namedStream[i]));

                    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&sendbuf[i]), kBufferBytes));
                    sendbufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(sendbuf[i]));
                    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&recvbuf[i]), kBufferBytes));
                    recvbufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(recvbuf[i]));

                    HIP_CHECK(RCCLTestHelpers::initializeBufferWithPattern<float>(
                        sendbuf[i], kElemCount,
                        [i](size_t) { return static_cast<float>(i + 1); }));
                    HIP_CHECK(RCCLTestHelpers::zeroInitializeBuffer<float>(recvbuf[i], kElemCount));
                }

                const float expected = static_cast<float>(nGpus * (nGpus + 1) / 2);

                // Launch 1: default stream (nullptr) — sets lastStream=nullptr, lastStreamValid=true.
                for(int i = 0; i < nGpus; ++i)
                {
                    ncclResult_t res = ncclAllReduce(
                        sendbuf[i], recvbuf[i], kElemCount, ncclFloat, ncclSum,
                        comms[i], nullptr);
                    ASSERT_EQ(res, ncclSuccess)
                        << "default-stream AllReduce rank=" << i << ": "
                        << ncclGetErrorString(res);
                }

                // Launch 2: named stream — ncclLaunchPrepare must detect lastStream(nullptr) != namedStream
                // and fire hipEventRecord(doneEvent, nullptr) + hipStreamWaitEvent(namedStream).
                for(int i = 0; i < nGpus; ++i)
                {
                    ncclResult_t res = ncclAllReduce(
                        sendbuf[i], recvbuf[i], kElemCount, ncclFloat, ncclSum,
                        comms[i], namedStream[i]);
                    ASSERT_EQ(res, ncclSuccess)
                        << "named-stream AllReduce rank=" << i << ": "
                        << ncclGetErrorString(res);
                }

                // Synchronize both streams before verifying.
                for(int i = 0; i < nGpus; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices[i]));
                    HIP_CHECK(hipStreamSynchronize(0));
                    HIP_CHECK(hipStreamSynchronize(namedStream[i]));
                }

                for(int i = 0; i < nGpus; ++i)
                {
                    size_t errIdx = 0;
                    float  expVal = 0.0f, actVal = 0.0f;
                    bool   ok = RCCLTestHelpers::verifyBufferData<float>(
                        recvbuf[i], kElemCount,
                        [expected](size_t) { return expected; },
                        kElemCount, 0.0, &errIdx, &expVal, &actVal);
                    EXPECT_TRUE(ok)
                        << "DefaultToNamed rank " << i << ": wrong at index " << errIdx
                        << " expected " << expVal << " got " << actVal;
                }
            })
            .withNumGpus(kMaxGpus)
            .withEnvironment({{"NCCL_DEBUG", "WARN"}})
            .withTimeout(std::chrono::seconds(kDoneEventTimeoutSeconds)));
}

// kIterations same-stream AllReduces then a single stream switch — tests lazy doneEvent record on stream change.
TEST_F(DoneEventOrdering, SameStreamThenSwitch)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "SameStreamThenSwitch",
            []()
            {
                int devCount = 0;
                HIP_CHECK(hipGetDeviceCount(&devCount));
                const int nGpus = std::min(devCount, kMaxGpus);
                if(nGpus < 2)
                    GTEST_SKIP() << "Requires >= 2 GPUs; found " << devCount;

                std::vector<int> devices(nGpus);
                std::iota(devices.begin(), devices.end(), 0);

                std::vector<ncclComm_t> comms(nGpus, nullptr);
                {
                    ncclResult_t res = ncclCommInitAll(comms.data(), nGpus, devices.data());
                    ASSERT_EQ(res, ncclSuccess) << ncclGetErrorString(res);
                }
                std::vector<RCCLTestGuards::NcclCommAutoGuard> commGuards;
                commGuards.reserve(nGpus);
                for(int i = 0; i < nGpus; ++i)
                    commGuards.push_back(RCCLTestGuards::makeCommAutoGuard(comms[i]));

                std::vector<RCCLTestGuards::HipStreamAutoGuard>    streamAGuards, streamBGuards;
                std::vector<RCCLTestGuards::DeviceBufferAutoGuard> sendbufGuards, recvbufGuards;
                streamAGuards.reserve(nGpus);
                streamBGuards.reserve(nGpus);
                sendbufGuards.reserve(nGpus);
                recvbufGuards.reserve(nGpus);

                std::vector<hipStream_t> streamA(nGpus, nullptr);
                std::vector<hipStream_t> streamB(nGpus, nullptr);
                std::vector<float*>      sendbuf(nGpus, nullptr);
                std::vector<float*>      recvbuf(nGpus, nullptr);

                for(int i = 0; i < nGpus; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices[i]));
                    HIP_CHECK(hipStreamCreate(&streamA[i]));
                    streamAGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(streamA[i]));
                    HIP_CHECK(hipStreamCreate(&streamB[i]));
                    streamBGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(streamB[i]));

                    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&sendbuf[i]), kBufferBytes));
                    sendbufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(sendbuf[i]));
                    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&recvbuf[i]), kBufferBytes));
                    recvbufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(recvbuf[i]));

                    HIP_CHECK(RCCLTestHelpers::initializeBufferWithPattern<float>(
                        sendbuf[i], kElemCount,
                        [i](size_t) { return static_cast<float>(i + 1); }));
                    HIP_CHECK(RCCLTestHelpers::zeroInitializeBuffer<float>(recvbuf[i], kElemCount));
                }

                const float expected = static_cast<float>(nGpus * (nGpus + 1) / 2);

                // Phase 1: kIterations AllReduces all on streamA with NO synchronize.
                // No hipEventRecord is emitted during this phase under the lazy scheme;
                // only lastStream=streamA and lastStreamValid=true are recorded.
                // This mirrors the real training-loop pattern (many collectives on one stream).
                for(int iter = 0; iter < kIterations; ++iter)
                {
                    for(int i = 0; i < nGpus; ++i)
                    {
                        ncclResult_t res = ncclAllReduce(
                            sendbuf[i], recvbuf[i], kElemCount, ncclFloat, ncclSum,
                            comms[i], streamA[i]);
                        ASSERT_EQ(res, ncclSuccess)
                            << "phase1 iter=" << iter << " rank=" << i << ": "
                            << ncclGetErrorString(res);
                    }
                }

                // Phase 2: single AllReduce on streamB per communicator — the stream switch.
                // ncclLaunchPrepare detects lastStream(=streamA) != streamB, fires
                // hipEventRecord(doneEvent, streamA) then hipStreamWaitEvent(streamB, doneEvent).
                // ncclLaunchPrepare detects the stream change, lazily records doneEvent on
                // streamA, then waits on it from streamB. Without this, streamB starts while
                // streamA's kernels are still in flight, producing a wrong sum or a hang.
                for(int i = 0; i < nGpus; ++i)
                {
                    ncclResult_t res = ncclAllReduce(
                        sendbuf[i], recvbuf[i], kElemCount, ncclFloat, ncclSum,
                        comms[i], streamB[i]);
                    ASSERT_EQ(res, ncclSuccess)
                        << "phase2 rank=" << i << ": " << ncclGetErrorString(res);
                }

                // Synchronize both streams before verifying.
                for(int i = 0; i < nGpus; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices[i]));
                    HIP_CHECK(hipStreamSynchronize(streamA[i]));
                    HIP_CHECK(hipStreamSynchronize(streamB[i]));
                }

                for(int i = 0; i < nGpus; ++i)
                {
                    size_t errIdx = 0;
                    float  expVal = 0.0f, actVal = 0.0f;
                    bool   ok = RCCLTestHelpers::verifyBufferData<float>(
                        recvbuf[i], kElemCount,
                        [expected](size_t) { return expected; },
                        kElemCount, 0.0, &errIdx, &expVal, &actVal);
                    EXPECT_TRUE(ok)
                        << "SameStreamThenSwitch rank " << i << ": wrong at index " << errIdx
                        << " expected " << expVal << " got " << actVal;
                }
            })
            .withNumGpus(kMaxGpus)
            .withEnvironment({{"NCCL_DEBUG", "WARN"}})
            .withTimeout(std::chrono::seconds(kDoneEventTimeoutSeconds)));
}

} // namespace RcclUnitTesting
