/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Tests for the ordering of a communicator's collectives across streams. A comm that has run on one
// stream records no doneEvent. Its first stream switch records an event on the previous stream and
// waits on it; it waits on nothing if that stream was destroyed, drains the device if the stream is
// hipStreamPerThread or reuses a destroyed stream's handle, and fails if the stream is capturing. Every
// later switch waits on the doneEvent the previous launch recorded (RCCL_DONE_EVENT_ELISION=0: every
// launch records it).

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
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

// 20 alternating-stream launches suffice: iteration 1 is the comm's first stream switch, iteration 2
// onward waits on the launch-recorded doneEvent; a missing edge produces a detectable hang or wrong sum.
static constexpr int kIterations = 20;

// Cap on communicators/GPUs used by single-node tests.
static constexpr int kMaxGpus = 8;

// Streams created between destroying one stream and creating its replacement, so the replacement
// cannot draw the freed handle and the stream-change branch is reached. One is enough: an idle
// stream's refcount hits zero inside hipStreamDestroy, so clr deletes it there and the decoy takes
// that address. Verified over 100 runs; more decoys only slow the failure path down. The first switch
// then finds the handle on a stream with another HIP id, so it drains the device.
static constexpr int kDecoyStreams = 1;

class DoneEventOrdering : public ::testing::Test
{
    // All work runs under process isolation; no fixture setup/teardown.
};

// Helper: nIterations alternating-stream AllReduces, no intermediate sync. Iteration k reduces its own
// input, (k+1)*(rank+1) everywhere, into the one shared output, so the output must end as the last
// iteration's sum: a collective that ran out of order leaves an earlier iteration's value behind.
// grouped wraps each iteration's per-rank calls in ncclGroupStart/End.
static void runAlternatingStreamAllReduces(int nGpus, int nIterations, bool grouped = false)
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
    std::vector<RCCLTestGuards::DeviceBufferAutoGuard> bufGuards;
    streamAGuards.reserve(nGpus);
    streamBGuards.reserve(nGpus);
    bufGuards.reserve(static_cast<size_t>(nGpus) * (nIterations + 1));

    // Keep raw pointers for RCCL calls (guards own the lifetime).
    std::vector<hipStream_t>         streamA(nGpus, nullptr);
    std::vector<hipStream_t>         streamB(nGpus, nullptr);
    std::vector<std::vector<float*>> sendbuf(nGpus, std::vector<float*>(nIterations, nullptr));
    std::vector<float*>              recvbuf(nGpus, nullptr);

    for(int i = 0; i < nGpus; ++i)
    {
        HIP_CHECK(hipSetDevice(devices[i]));

        HIP_CHECK(hipStreamCreate(&streamA[i]));
        streamAGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(streamA[i]));

        HIP_CHECK(hipStreamCreate(&streamB[i]));
        streamBGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(streamB[i]));

        for(int iter = 0; iter < nIterations; ++iter)
        {
            HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&sendbuf[i][iter]), kBufferBytes));
            bufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(sendbuf[i][iter]));
            HIP_CHECK(RCCLTestHelpers::initializeBufferWithPattern<float>(
                sendbuf[i][iter], kElemCount,
                [i, iter](size_t) { return static_cast<float>((iter + 1) * (i + 1)); }));
        }

        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&recvbuf[i]), kBufferBytes));
        bufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(recvbuf[i]));
        HIP_CHECK(RCCLTestHelpers::zeroInitializeBuffer<float>(recvbuf[i], kElemCount));
    }

    // AllReduce-sum of nIterations*(r+1) over ranks 0..nGpus-1 == nIterations*nGpus*(nGpus+1)/2.
    // All terms are exactly representable; the sum is too, so strict == is safe.
    const float expected = static_cast<float>(nIterations * nGpus * (nGpus + 1) / 2);

    // Back-to-back launches, alternating stream every iteration, with NO synchronize in between.
    // Each comm sees one stream per launch, so planner->numStreams == 1 (fast path); the flip takes
    // the lastStreamTag != ncclStreamTag(launchStream) branch from iteration 1 onward: the comm's
    // first stream switch at iteration 1, a wait on the launch-recorded doneEvent at every later
    // one. Grouped calls skip the DDA addon, so they take the native fast path.
    for(int iter = 0; iter < nIterations; ++iter)
    {
        if(grouped) ASSERT_EQ(ncclGroupStart(), ncclSuccess);
        for(int i = 0; i < nGpus; ++i)
        {
            hipStream_t  stream = (iter & 1) ? streamB[i] : streamA[i];
            ncclResult_t res    = ncclAllReduce(
                sendbuf[i][iter], recvbuf[i], kElemCount, ncclFloat, ncclSum, comms[i], stream);
            ASSERT_EQ(res, ncclSuccess)
                << "ncclAllReduce failed iter=" << iter << " rank=" << i << ": "
                << ncclGetErrorString(res);
        }
        if(grouped)
        {
            ncclResult_t endRes = ncclGroupEnd();
            ASSERT_EQ(endRes, ncclSuccess)
                << "ncclGroupEnd iter=" << iter << ": " << ncclGetErrorString(endRes);
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

// Runs `body` as one process-isolated test on up to kMaxGpus GPUs, skipped below 2.
static void runIsolated(const char* name, std::function<void(int nGpus)> body,
                        const std::unordered_map<std::string, std::string>& env = {{"NCCL_DEBUG", "WARN"}})
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            name,
            [body]()
            {
                int devCount = 0;
                HIP_CHECK(hipGetDeviceCount(&devCount));
                const int nGpus = std::min(devCount, kMaxGpus);
                if(nGpus < 2)
                    GTEST_SKIP() << "Requires >= 2 GPUs; found " << devCount;
                body(nGpus);
            })
            .withNumGpus(kMaxGpus)
            .withEnvironment(env)
            .withTimeout(std::chrono::seconds(kDoneEventTimeoutSeconds)));
}

// Ungrouped alternating-stream AllReduces — tests the edge taken on every lastStreamTag change.
TEST_F(DoneEventOrdering, FastPathStreamAlternation)
{
    runIsolated("FastPathStreamAlternation", [](int nGpus) { runAlternatingStreamAllReduces(nGpus, kIterations); });
}

// Same as FastPathStreamAlternation with the elision off: every launch records doneEvent, so even
// the first switch waits on a launch-recorded event.
TEST_F(DoneEventOrdering, FastPathStreamAlternationElisionOff)
{
    runIsolated("FastPathStreamAlternationElisionOff",
                [](int nGpus) { runAlternatingStreamAllReduces(nGpus, kIterations); },
                {{"NCCL_DEBUG", "WARN"}, {"RCCL_DONE_EVENT_ELISION", "0"}});
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
                // Two reinit cycles to exercise that lastStreamTag and the single-stream state
                // reset on each ncclCommInitAll: every cycle orders after the previous stream at its first switch
                // and waits on the launch-recorded one at its second.
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

                    // Three alternating-stream launches: iter=0 → streamA (sets lastStreamTag),
                    // iter=1 → streamB (first flip: records on streamA), iter=2 → streamA (waits
                    // on the doneEvent iter=1 recorded on streamB).
                    for(int iter = 0; iter < 3; ++iter)
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

// Alternating-stream AllReduces wrapped in ncclGroupStart/End. Grouped calls skip the DDA addon, so this
// covers the native fast path: ncclLaunchPrepare's stream-change branch and ncclLaunchKernel's stopEvent.
TEST_F(DoneEventOrdering, GroupedPathBaseline)
{
    runIsolated("GroupedPathBaseline",
                [](int nGpus) { runAlternatingStreamAllReduces(nGpus, kIterations, /*grouped=*/true); });
}

// Default stream first, named stream second. Tests the default-stream tag (1) vs the no-launch sentinel (0).
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

                // Launch 1: default stream (nullptr) — sets lastStreamTag = ncclStreamTag(nullptr) == 1.
                for(int i = 0; i < nGpus; ++i)
                {
                    ncclResult_t res = ncclAllReduce(
                        sendbuf[i], recvbuf[i], kElemCount, ncclFloat, ncclSum,
                        comms[i], nullptr);
                    ASSERT_EQ(res, ncclSuccess)
                        << "default-stream AllReduce rank=" << i << ": "
                        << ncclGetErrorString(res);
                }

                // Launch 2: named stream — tag 1 != ncclStreamTag(namedStream) is a stream switch, and
                // the comm's first, so it records the comm's first-switch event on the default stream
                // (launch 1 recorded none).
                for(int i = 0; i < nGpus; ++i)
                {
                    ncclResult_t res = ncclAllReduce(
                        sendbuf[i], recvbuf[i], kElemCount, ncclFloat, ncclSum,
                        comms[i], namedStream[i]);
                    ASSERT_EQ(res, ncclSuccess)
                        << "named-stream AllReduce rank=" << i << ": "
                        << ncclGetErrorString(res);
                }

                // Launch 3: back to the default stream, which waits on the doneEvent launch 2 recorded.
                for(int i = 0; i < nGpus; ++i)
                {
                    ncclResult_t res = ncclAllReduce(
                        sendbuf[i], recvbuf[i], kElemCount, ncclFloat, ncclSum,
                        comms[i], nullptr);
                    ASSERT_EQ(res, ncclSuccess)
                        << "default-stream (switch back) AllReduce rank=" << i << ": "
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

// kIterations same-stream AllReduces, a switch to streamB (records on streamA), then back to streamA.
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
                // Mirrors the real training-loop pattern (many collectives on one stream).
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

                // Phase 2: single AllReduce on streamB per communicator — the first stream switch.
                // No phase-1 launch recorded doneEvent, so the switch records the comm's
                // first-switch event on streamA and waits on it. Without that, streamB starts while
                // streamA's kernels are still in flight, producing a wrong sum or a hang.
                for(int i = 0; i < nGpus; ++i)
                {
                    ncclResult_t res = ncclAllReduce(
                        sendbuf[i], recvbuf[i], kElemCount, ncclFloat, ncclSum,
                        comms[i], streamB[i]);
                    ASSERT_EQ(res, ncclSuccess)
                        << "phase2 rank=" << i << ": " << ncclGetErrorString(res);
                }

                // Phase 3: back to streamA. Phase 2 recorded doneEvent on streamB, and this switch
                // waits on it with no host synchronize in between.
                for(int i = 0; i < nGpus; ++i)
                {
                    ncclResult_t res = ncclAllReduce(
                        sendbuf[i], recvbuf[i], kElemCount, ncclFloat, ncclSum,
                        comms[i], streamA[i]);
                    ASSERT_EQ(res, ncclSuccess)
                        << "phase3 rank=" << i << ": " << ncclGetErrorString(res);
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

// Communicator outlives the stream it last launched on. Destroying that stream and then issuing a
// collective on a fresh one must not use the dead handle for anything but a lookup.
TEST_F(DoneEventOrdering, StreamDestroyedThenSwitch)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "StreamDestroyedThenSwitch",
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

                std::vector<RCCLTestGuards::HipStreamAutoGuard>    decoyGuards, streamBGuards;
                std::vector<RCCLTestGuards::DeviceBufferAutoGuard> sendbufGuards, recvbufGuards;
                decoyGuards.reserve(static_cast<size_t>(nGpus) * kDecoyStreams);
                streamBGuards.reserve(nGpus);
                sendbufGuards.reserve(nGpus);
                recvbufGuards.reserve(nGpus);

                // streamA is destroyed mid-test on purpose, so it is not RAII-guarded.
                std::vector<hipStream_t> streamA(nGpus, nullptr);
                std::vector<hipStream_t> streamB(nGpus, nullptr);
                std::vector<float*>      sendbuf(nGpus, nullptr);
                std::vector<float*>      recvbuf(nGpus, nullptr);

                for(int i = 0; i < nGpus; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices[i]));
                    HIP_CHECK(hipStreamCreate(&streamA[i]));

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

                // Phase 1: kIterations collectives on streamA to match the training-loop pattern,
                // then drain so the destroy below is clean.
                for(int iter = 0; iter < kIterations; ++iter)
                {
                    for(int i = 0; i < nGpus; ++i)
                    {
                        ncclResult_t res = ncclAllReduce(sendbuf[i], recvbuf[i], kElemCount,
                                                         ncclFloat, ncclSum, comms[i], streamA[i]);
                        ASSERT_EQ(res, ncclSuccess)
                            << "phase1 iter=" << iter << " rank=" << i << ": "
                            << ncclGetErrorString(res);
                    }
                }
                for(int i = 0; i < nGpus; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices[i]));
                    HIP_CHECK(hipStreamSynchronize(streamA[i]));
                }

                // Each communicator now holds streamA's identity. Destroy it; the comm lives on.
                for(int i = 0; i < nGpus; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices[i]));
                    HIP_CHECK(hipStreamDestroy(streamA[i]));
                }

                for(int i = 0; i < nGpus; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices[i]));
                    for(int k = 0; k < kDecoyStreams; ++k)
                    {
                        hipStream_t decoy = nullptr;
                        HIP_CHECK(hipStreamCreate(&decoy));
                        decoyGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(decoy));
                    }
                    HIP_CHECK(hipStreamCreate(&streamB[i]));
                    streamBGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(streamB[i]));
                }

                // Phase 2: the stream change, the comm's first. streamA's handle now names a decoy with
                // another HIP id, so the switch drains the device rather than record on it. Before the
                // destroyed-stream fix this recorded doneEvent on the destroyed streamA and returned
                // ncclUnhandledCudaError, or segfaulted.
                for(int i = 0; i < nGpus; ++i)
                {
                    ncclResult_t res = ncclAllReduce(sendbuf[i], recvbuf[i], kElemCount,
                                                     ncclFloat, ncclSum, comms[i], streamB[i]);
                    ASSERT_EQ(res, ncclSuccess)
                        << "phase2 rank=" << i << ": " << ncclGetErrorString(res);
                }
                for(int i = 0; i < nGpus; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices[i]));
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
                        << "StreamDestroyedThenSwitch rank " << i << ": wrong at index " << errIdx
                        << " expected " << expVal << " got " << actVal;
                }
            })
            .withNumGpus(kMaxGpus)
            .withEnvironment({{"NCCL_DEBUG", "WARN"}})
            .withTimeout(std::chrono::seconds(kDoneEventTimeoutSeconds)));
}

// The first stream switch happens while this thread has a global-mode capture open on another, blocking
// stream. The previous stream (streamA, or the default stream) still exists and is not capturing, so the
// switch records on it rather than draining the device, which HIP refuses while anything captures (and
// invalidates the capture doing so). The default stream is not probed: HIP reports an implicit-capture
// error for it while any blocking stream captures. grouped: grouped calls, which take the native fast path.
static void runFirstSwitchWhileCapturing(bool fromDefaultStream, bool grouped)
{
#if HIP_VERSION < 70100000
    GTEST_SKIP() << "HIP older than 7.1 records doneEvent on every launch; there is no first switch";
#endif
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
    std::vector<RCCLTestGuards::DeviceBufferAutoGuard> bufGuards;
    streamGuards.reserve(3 * nGpus);
    bufGuards.reserve(3 * nGpus);

    // fromDefaultStream: phase 1 runs on the default stream instead of streamA.
    std::vector<hipStream_t> streamA(nGpus, nullptr);
    std::vector<hipStream_t> streamB(nGpus, nullptr);
    std::vector<hipStream_t> captureStream(nGpus, nullptr);
    std::vector<float*>      sendbuf(nGpus, nullptr);
    std::vector<float*>      recvbuf(nGpus, nullptr);
    std::vector<float*>      capturedBuf(nGpus, nullptr);

    for(int i = 0; i < nGpus; ++i)
    {
        HIP_CHECK(hipSetDevice(devices[i]));
        for(hipStream_t* s : {&streamA[i], &streamB[i], &captureStream[i]})
        {
            HIP_CHECK(hipStreamCreate(s));
            streamGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(*s));
        }
        for(float** b : {&sendbuf[i], &recvbuf[i], &capturedBuf[i]})
        {
            HIP_CHECK(hipMalloc(reinterpret_cast<void**>(b), kBufferBytes));
            bufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(*b));
        }
        HIP_CHECK(RCCLTestHelpers::initializeBufferWithPattern<float>(
            sendbuf[i], kElemCount,
            [i](size_t) { return static_cast<float>(i + 1); }));
        HIP_CHECK(RCCLTestHelpers::zeroInitializeBuffer<float>(recvbuf[i], kElemCount));
    }

    const float expected = static_cast<float>(nGpus * (nGpus + 1) / 2);

    // Phase 1: one AllReduce per comm on the first stream, which records no doneEvent.
    if(grouped) ASSERT_EQ(ncclGroupStart(), ncclSuccess);
    for(int i = 0; i < nGpus; ++i)
    {
        ncclResult_t res = ncclAllReduce(sendbuf[i], recvbuf[i], kElemCount, ncclFloat, ncclSum,
                                         comms[i], fromDefaultStream ? nullptr : streamA[i]);
        ASSERT_EQ(res, ncclSuccess)
            << "phase1 rank=" << i << ": " << ncclGetErrorString(res);
    }
    if(grouped) ASSERT_EQ(ncclGroupEnd(), ncclSuccess);

    // Open a global-mode capture on every device, with one memset in it.
    for(int i = 0; i < nGpus; ++i)
    {
        HIP_CHECK(hipSetDevice(devices[i]));
        HIP_CHECK(hipStreamBeginCapture(captureStream[i], hipStreamCaptureModeGlobal));
        HIP_CHECK(hipMemsetAsync(capturedBuf[i], 0, kBufferBytes, captureStream[i]));
    }

    // Phase 2: eager AllReduce on streamB, each comm's first switch.
    if(grouped) ASSERT_EQ(ncclGroupStart(), ncclSuccess);
    for(int i = 0; i < nGpus; ++i)
    {
        ncclResult_t res = ncclAllReduce(sendbuf[i], recvbuf[i], kElemCount,
                                         ncclFloat, ncclSum, comms[i], streamB[i]);
        ASSERT_EQ(res, ncclSuccess)
            << "phase2 rank=" << i << ": " << ncclGetErrorString(res);
    }
    if(grouped) ASSERT_EQ(ncclGroupEnd(), ncclSuccess);

    // The captures are still intact and end cleanly.
    for(int i = 0; i < nGpus; ++i)
    {
        HIP_CHECK(hipSetDevice(devices[i]));
        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        HIP_CHECK(hipStreamIsCapturing(captureStream[i], &status));
        EXPECT_EQ(status, hipStreamCaptureStatusActive) << "rank " << i;
        hipGraph_t graph = nullptr;
        HIP_CHECK(hipStreamEndCapture(captureStream[i], &graph));
        ASSERT_NE(graph, nullptr) << "rank " << i;
        HIP_CHECK(hipGraphDestroy(graph));
    }

    for(int i = 0; i < nGpus; ++i)
    {
        HIP_CHECK(hipSetDevice(devices[i]));
        HIP_CHECK(hipStreamSynchronize(fromDefaultStream ? nullptr : streamA[i]));
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
            << "FirstSwitchWhileCapturing rank " << i << ": wrong at index "
            << errIdx << " expected " << expVal << " got " << actVal;
    }
}

TEST_F(DoneEventOrdering, FirstSwitchWhileCapturingElsewhere)
{
    runIsolated("FirstSwitchWhileCapturingElsewhere",
                [](int) { runFirstSwitchWhileCapturing(/*fromDefaultStream=*/false, /*grouped=*/false); });
}

TEST_F(DoneEventOrdering, FirstSwitchWhileCapturingElsewhereGrouped)
{
    runIsolated("FirstSwitchWhileCapturingElsewhereGrouped",
                [](int) { runFirstSwitchWhileCapturing(/*fromDefaultStream=*/false, /*grouped=*/true); });
}

TEST_F(DoneEventOrdering, FirstSwitchFromDefaultStreamWhileCapturingElsewhere)
{
    runIsolated("FirstSwitchFromDefaultStreamWhileCapturingElsewhere",
                [](int) { runFirstSwitchWhileCapturing(/*fromDefaultStream=*/true, /*grouped=*/false); });
}

// The comm's first stream switch reads what the first stream wrote. Phase 1: kIterations AllReduces on the
// first stream (streamA, or the default stream), each from its own input into recvA, which keeps that stream
// busy: its last iteration also waits on a host flag that a helper thread sets only later, so the first
// stream provably still has work queued when phase 2 is issued. Phase 2, the first switch: one AllReduce
// on streamB (non-blocking, so the default stream's implicit ordering does not stand in for RCCL's) from
// recvA into recvB. Unless the switch orders streamB after the first stream, it reads recvA early and recvB
// ends up wrong, or the overlapping kernels hang.
// grouped: every iteration's per-rank calls in one group, which takes the native fast path.
// destroyFirst: streamA is destroyed, without a sync, before the switch. hipStreamDestroy then waits for
// the gated work, and the switch finds no stream under the handle (streamB exists from the start, so it
// cannot take it) and waits on nothing.
// captureBetween: a collective captured on another stream (and never launched) comes between phase 1 and
// the switch. A captured launch must not become the stream the switch orders after.
static void runFirstSwitchReadsFirstStream(int nGpus, bool grouped, bool fromDefaultStream, bool destroyFirst,
                                           bool captureBetween = false)
{
#if HIP_VERSION < 70100000
    GTEST_SKIP() << "HIP older than 7.1 records doneEvent on every launch; there is no first switch";
#endif
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
    std::vector<RCCLTestGuards::DeviceBufferAutoGuard> bufGuards;
    streamGuards.reserve(3 * nGpus);
    bufGuards.reserve(static_cast<size_t>(nGpus) * (kIterations + 2));

    // streamA is destroyed mid-test when destroyFirst, so it is guarded only otherwise.
    std::vector<hipStream_t>         streamA(nGpus, nullptr);
    std::vector<hipStream_t>         streamB(nGpus, nullptr);
    std::vector<std::vector<float*>> sendbuf(nGpus, std::vector<float*>(kIterations, nullptr));
    std::vector<hipStream_t>         captureStream(nGpus, nullptr);
    std::vector<float*>              recvA(nGpus, nullptr);
    std::vector<float*>              recvB(nGpus, nullptr);

    for(int i = 0; i < nGpus; ++i)
    {
        HIP_CHECK(hipSetDevice(devices[i]));
        HIP_CHECK(hipStreamCreate(&streamA[i]));
        if(!destroyFirst) streamGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(streamA[i]));
        HIP_CHECK(hipStreamCreateWithFlags(&streamB[i], hipStreamNonBlocking));
        streamGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(streamB[i]));
        HIP_CHECK(hipStreamCreateWithFlags(&captureStream[i], hipStreamNonBlocking));
        streamGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(captureStream[i]));
        for(int iter = 0; iter < kIterations; ++iter)
        {
            HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&sendbuf[i][iter]), kBufferBytes));
            bufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(sendbuf[i][iter]));
            HIP_CHECK(RCCLTestHelpers::initializeBufferWithPattern<float>(
                sendbuf[i][iter], kElemCount,
                [i, iter](size_t) { return static_cast<float>((iter + 1) * (i + 1)); }));
        }
        for(float** b : {&recvA[i], &recvB[i]})
        {
            HIP_CHECK(hipMalloc(reinterpret_cast<void**>(b), kBufferBytes));
            bufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(*b));
            HIP_CHECK(RCCLTestHelpers::zeroInitializeBuffer<float>(*b, kElemCount));
        }
        HIP_CHECK(hipDeviceSynchronize());
    }

    // Phase 1 leaves kIterations * nGpus(nGpus+1)/2 in every recvA; phase 2 sums that over the ranks.
    // At most 8 * 20 * 36 = 5760: exact in float.
    const float expected = static_cast<float>(nGpus * kIterations * nGpus * (nGpus + 1) / 2);

    uint32_t* gate = nullptr;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&gate), sizeof(uint32_t),
                            hipHostMallocMapped | hipHostMallocCoherent | hipHostMallocPortable));
    __atomic_store_n(gate, 0u, __ATOMIC_RELEASE);
    void* gateDev = nullptr;
    HIP_CHECK(hipHostGetDevicePointer(&gateDev, gate, 0));
    // Opens the gate 1 s after start(), and at once on any early return, so a failed assertion cannot
    // leave the first stream blocked (and the guards' stream destroys hanging) or a thread joinable.
    struct GateRelease
    {
        uint32_t*   gate;
        std::thread opener;
        void        start()
        {
            opener = std::thread([g = gate]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                __atomic_store_n(g, 1u, __ATOMIC_RELEASE);
            });
        }
        ~GateRelease()
        {
            __atomic_store_n(gate, 1u, __ATOMIC_RELEASE);
            if(opener.joinable()) opener.join();
            (void)hipHostFree(gate);
        }
    } gateRelease{gate, {}};

    for(int iter = 0; iter < kIterations; ++iter)
    {
        if(iter == kIterations - 1)
        {
            for(int i = 0; i < nGpus; ++i)
            {
                HIP_CHECK(hipSetDevice(devices[i]));
                HIP_CHECK(hipStreamWaitValue32(fromDefaultStream ? nullptr : streamA[i], gateDev, 1,
                                               hipStreamWaitValueGte, 0xFFFFFFFFu));
            }
        }
        if(grouped) ASSERT_EQ(ncclGroupStart(), ncclSuccess);
        for(int i = 0; i < nGpus; ++i)
        {
            ncclResult_t res = ncclAllReduce(sendbuf[i][iter], recvA[i], kElemCount, ncclFloat, ncclSum, comms[i],
                                             fromDefaultStream ? nullptr : streamA[i]);
            ASSERT_EQ(res, ncclSuccess) << "phase1 iter=" << iter << " rank=" << i << ": " << ncclGetErrorString(res);
        }
        if(grouped) ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
    }

    // Opened well after phase 2 is issued (or, with destroyFirst, while hipStreamDestroy waits on it).
    gateRelease.start();

    if(destroyFirst)
    {
        for(int i = 0; i < nGpus; ++i)
        {
            HIP_CHECK(hipSetDevice(devices[i]));
            HIP_CHECK(hipStreamDestroy(streamA[i]));
#if HIP_VERSION >= 70100000
            // The precondition the switch relies on: no stream has the handle any more.
            unsigned long long id = 0;
            EXPECT_NE(hipStreamGetId(streamA[i], &id), hipSuccess) << "rank " << i;
            (void)hipGetLastError();
#endif
        }
    }

    if(captureBetween)
    {
        // Same grouping as phase 1, so the captured launch takes the same (native or addon) path.
        for(int i = 0; i < nGpus; ++i)
        {
            HIP_CHECK(hipSetDevice(devices[i]));
            HIP_CHECK(hipStreamBeginCapture(captureStream[i], hipStreamCaptureModeRelaxed));
        }
        if(grouped) ASSERT_EQ(ncclGroupStart(), ncclSuccess);
        for(int i = 0; i < nGpus; ++i)
        {
            ncclResult_t res = ncclAllReduce(sendbuf[i][0], recvB[i], kElemCount, ncclFloat, ncclSum, comms[i],
                                             captureStream[i]);
            ASSERT_EQ(res, ncclSuccess) << "captured rank=" << i << ": " << ncclGetErrorString(res);
        }
        if(grouped) ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
        for(int i = 0; i < nGpus; ++i)
        {
            HIP_CHECK(hipSetDevice(devices[i]));
            hipGraph_t graph = nullptr;
            HIP_CHECK(hipStreamEndCapture(captureStream[i], &graph));
            HIP_CHECK(hipGraphDestroy(graph));
        }
    }

    // Only the probe of the first switch may leave an error behind; start from a clean slate.
    (void)hipGetLastError();
    if(grouped) ASSERT_EQ(ncclGroupStart(), ncclSuccess);
    for(int i = 0; i < nGpus; ++i)
    {
        ncclResult_t res =
            ncclAllReduce(recvA[i], recvB[i], kElemCount, ncclFloat, ncclSum, comms[i], streamB[i]);
        ASSERT_EQ(res, ncclSuccess) << "phase2 rank=" << i << ": " << ncclGetErrorString(res);
    }
    if(grouped) ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
    // A failed probe of the destroyed handle must not leave an error behind for the app.
    EXPECT_EQ(hipGetLastError(), hipSuccess);
    // Phase 2 went out while the first stream was still held, so it really was ordered by the switch.
    if(!destroyFirst)
    {
        for(int i = 0; i < nGpus; ++i)
        {
            HIP_CHECK(hipSetDevice(devices[i]));
            EXPECT_EQ(hipStreamQuery(fromDefaultStream ? nullptr : streamA[i]), hipErrorNotReady) << "rank " << i;
        }
    }

    for(int i = 0; i < nGpus; ++i)
    {
        HIP_CHECK(hipSetDevice(devices[i]));
        HIP_CHECK(hipDeviceSynchronize());
    }

    for(int i = 0; i < nGpus; ++i)
    {
        size_t errIdx = 0;
        float  expVal = 0.0f, actVal = 0.0f;
        bool   ok = RCCLTestHelpers::verifyBufferData<float>(
            recvB[i], kElemCount, [expected](size_t) { return expected; }, kElemCount, 0.0, &errIdx, &expVal,
            &actVal);
        EXPECT_TRUE(ok) << "first switch rank " << i << ": wrong at index " << errIdx << " expected " << expVal
                        << " got " << actVal;
    }
}

TEST_F(DoneEventOrdering, FirstSwitchReadsFirstStream)
{
    runIsolated("FirstSwitchReadsFirstStream", [](int nGpus) {
        runFirstSwitchReadsFirstStream(nGpus, /*grouped=*/false, /*fromDefaultStream=*/false, /*destroyFirst=*/false);
    });
}

TEST_F(DoneEventOrdering, FirstSwitchReadsFirstStreamGrouped)
{
    runIsolated("FirstSwitchReadsFirstStreamGrouped", [](int nGpus) {
        runFirstSwitchReadsFirstStream(nGpus, /*grouped=*/true, /*fromDefaultStream=*/false, /*destroyFirst=*/false);
    });
}

TEST_F(DoneEventOrdering, FirstSwitchReadsDefaultStream)
{
    runIsolated("FirstSwitchReadsDefaultStream", [](int nGpus) {
        runFirstSwitchReadsFirstStream(nGpus, /*grouped=*/false, /*fromDefaultStream=*/true, /*destroyFirst=*/false);
    });
}

TEST_F(DoneEventOrdering, FirstSwitchReadsFirstStreamAfterCaptureGrouped)
{
    runIsolated("FirstSwitchReadsFirstStreamAfterCaptureGrouped", [](int nGpus) {
        runFirstSwitchReadsFirstStream(nGpus, /*grouped=*/true, /*fromDefaultStream=*/false, /*destroyFirst=*/false,
                                       /*captureBetween=*/true);
    });
}

TEST_F(DoneEventOrdering, FirstSwitchReadsFirstStreamAfterCapture)
{
    runIsolated("FirstSwitchReadsFirstStreamAfterCapture", [](int nGpus) {
        runFirstSwitchReadsFirstStream(nGpus, /*grouped=*/false, /*fromDefaultStream=*/false, /*destroyFirst=*/false,
                                       /*captureBetween=*/true);
    });
}

TEST_F(DoneEventOrdering, FirstSwitchReadsDestroyedStream)
{
    runIsolated("FirstSwitchReadsDestroyedStream", [](int nGpus) {
        runFirstSwitchReadsFirstStream(nGpus, /*grouped=*/false, /*fromDefaultStream=*/false, /*destroyFirst=*/true);
    });
}

} // namespace RcclUnitTesting
