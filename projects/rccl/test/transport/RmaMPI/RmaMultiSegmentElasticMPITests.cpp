/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef MPI_TESTS_ENABLED
#ifdef RCCL_HAS_RMA_IB_PROXY

#include "RmaMultiSegmentFixture.hpp"

namespace RCCLRmaTests
{


// DeepEP Engram: [GPU receive][CPU storage] in one window; IGet from a remote CPU offset into a different
// local GPU offset, so local and remote offsets are tracked independently.
TEST_F(RmaMultiSegmentMPITest, DeepEP_EngramMixedWindowIGet)
{
    if (!SetUpFixture(2, 2)) return;

    constexpr size_t kGpuBytes   = 4 * kMiB;
    constexpr size_t kCpuBytes   = 2 * kMiB;
    constexpr size_t kRemoteOff  = kGpuBytes + 4096;
    constexpr size_t kLocalOff   = 64 * 1024;
    constexpr size_t kPayload    = 128 * 1024;

    MultiSegmentVmmBuffer* window = nullptr;
    void* mh = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareElastic(kGpuBytes, kCpuBytes, &window, &mh));
    Rank0Round(true, window, mh, kRemoteOff, kPayload, window, mh, kLocalOff, /*seed=*/0x4D, /*sentinel=*/0xD7,
               "DeepEP CPU-to-GPU IGet", /*dstChecked=*/kGpuBytes);
}


// Multi-node Engram stress at 2 MiB alignment: changing remote CPU offsets into unrelated GPU offsets,
// with sentinels guarding the GPU receive area.
TEST_F(RmaMultiSegmentMPITest, DeepEP_MultiNodeEngramMixedWindowIGetStress)
{
    if (!SetUpFixture(2, 2)) return;
    if (MPIEnvironment::cached_multi_node_result != 1)
        GTEST_SKIP() << "requires exactly one rank on each of two nodes";

    constexpr size_t kGpuBytes  = 8 * kMiB;
    constexpr size_t kCpuBytes  = 4 * kMiB;
    constexpr int    kIterations = 32;

    MultiSegmentVmmBuffer* window = nullptr;
    void* mh = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareElastic(kGpuBytes, kCpuBytes, &window, &mh));

    for (int i = 0; i < kIterations; ++i)
    {
        const StressStep s = StressOffsets(i, kGpuBytes, kCpuBytes);
        ASSERT_NO_FATAL_FAILURE(Rank0Round(true, window, mh, kGpuBytes + s.cpuOff, s.len, window, mh, s.gpuOff,
                                           static_cast<uint8_t>(0x40 + i), /*sentinel=*/0xD9,
                                           "DeepEP Engram iteration " + std::to_string(i), kGpuBytes));
    }
}

// DeepEP hybrid [GPU][CPU local-rank 0..3]: fetch from the matching CPU segment on the other node into a
// non-zero local GPU offset.
TEST_F(RmaMultiSegmentMPITest, DeepEP_HybridImportedCpuSegmentIGet)
{
    if (!SetUpFixture(/*minProcesses=*/8, /*maxProcesses=*/8, /*minNodes=*/2, /*maxNodes=*/2))
        GTEST_SKIP() << "requires exactly 8 ranks across 2 nodes";
    std::string reason;
    RCCLHybridVmmTests::HybridVmmBuffer* window = nullptr;
    HYBRID_VMM_SKIP_OR_FAIL(AllocFixtureHybrid(kHybridGpuBytes, kHybridCpuBytes, 4, &window, &reason), reason);
    void* mh = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareHybrid(window, &mh));

    constexpr size_t kPayload = 128 * 1024;
    constexpr size_t kLocalOff = 64 * 1024;
    constexpr uint8_t kSentinel = 0xB7;

    const size_t remoteOff = kHybridGpuBytes + static_cast<size_t>(window->localRank) * kHybridCpuBytes + 4096;
    const int peer = MPIHelpers::findRemotePeerForLocalRank(window->localRank);
    ASSERT_TRUE(MPIHelpers::allRanksTrue(peer >= 0)) << "no remote peer for local rank " << window->localRank;

    RCCLTestHelpers::FillSegPattern(static_cast<uint8_t*>(window->ptr) + remoteOff, kPayload,
            static_cast<uint8_t>(0x30 + worldRank_));
    FillSentinel(window->ptr, kHybridGpuBytes, kSentinel);

    Barrier();
    void* req = nullptr;
    bool getOk = rma_->iget(rmaCtx_, 0, remoteOff, mh, kPayload,
                            kLocalOff, mh, peer, ncclRmaOptFlagsDefault, &req) == ncclSuccess;
    if (getOk) getOk = PollUntilDone(req);
    ASSERT_TRUE(MPIHelpers::allRanksTrue(getOk)) << "hybrid IGet failed on at least one rank";
    ExpectPayloadIsolated(window->ptr, kHybridGpuBytes, kLocalOff, kPayload,
                          static_cast<uint8_t>(0x30 + peer), kSentinel,
                          "DeepEP hybrid imported CPU IGet");
    Barrier();
}

// Multi-node hybrid stress: alternate the remote imported CPU segment and vary offsets and sizes, with
// sentinels around the destination.
TEST_F(RmaMultiSegmentMPITest, DeepEP_HybridMultiNodeIGetStress)
{
    if (!SetUpFixture(/*minProcesses=*/8, /*maxProcesses=*/8, /*minNodes=*/2, /*maxNodes=*/2))
        GTEST_SKIP() << "requires exactly 8 ranks across 2 nodes";
    std::string reason;
    RCCLHybridVmmTests::HybridVmmBuffer* window = nullptr;
    HYBRID_VMM_SKIP_OR_FAIL(AllocFixtureHybrid(kHybridGpuBytes, kHybridCpuBytes, 4, &window, &reason), reason);
    void* mh = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareHybrid(window, &mh));

    constexpr int kIterations = 32;
    constexpr uint8_t kSentinel = 0xBC;

    for (int i = 0; i < kIterations; ++i)
    {
        const StressStep s = StressOffsets(i, kHybridGpuBytes, kHybridCpuBytes);
        const size_t ownCpuOff = kHybridGpuBytes +
            static_cast<size_t>(window->localRank) * kHybridCpuBytes + s.cpuOff;
        const uint8_t seed = static_cast<uint8_t>(0x40 + worldRank_ + i);

        RCCLTestHelpers::FillSegPattern(static_cast<uint8_t*>(window->ptr) + ownCpuOff, s.len, seed);
        FillSentinel(window->ptr, kHybridGpuBytes, kSentinel);
        Barrier();

        const int sourceLocalRank = (i + window->localRank) % window->localSize;
        const int peer = MPIHelpers::findRemotePeerForLocalRank(sourceLocalRank);
        ASSERT_TRUE(MPIHelpers::allRanksTrue(peer >= 0))
            << "no remote peer for local rank " << sourceLocalRank << " at iteration " << i;
        const size_t remoteOff = kHybridGpuBytes +
            static_cast<size_t>(sourceLocalRank) * kHybridCpuBytes + s.cpuOff;

        void* req = nullptr;
        bool getOk = rma_->iget(rmaCtx_, 0, remoteOff, mh, s.len,
                                s.gpuOff, mh, peer, ncclRmaOptFlagsDefault, &req) == ncclSuccess;
        if (getOk) getOk = PollUntilDone(req);
        ASSERT_TRUE(MPIHelpers::allRanksTrue(getOk))
            << "hybrid IGet failed on at least one rank at iteration " << i;
        ExpectPayloadIsolated(window->ptr, kHybridGpuBytes, s.gpuOff, s.len,
                              static_cast<uint8_t>(0x40 + peer + i),
                              kSentinel,
                              "hybrid IGet iteration " + std::to_string(i));
        Barrier();
    }
}

// Negative hybrid range guard: an IGet that overruns the final imported CPU
// segment must be rejected before posting and leave the GPU destination intact.
TEST_F(RmaMultiSegmentMPITest, DeepEP_HybridOutOfRangeIGetRejected)
{
    if (!SetUpFixture(/*minProcesses=*/8, /*maxProcesses=*/8, /*minNodes=*/2, /*maxNodes=*/2))
        GTEST_SKIP() << "requires exactly 8 ranks across 2 nodes";
    std::string reason;
    RCCLHybridVmmTests::HybridVmmBuffer* window = nullptr;
    HYBRID_VMM_SKIP_OR_FAIL(AllocFixtureHybrid(kHybridGpuBytes, kHybridCpuBytes, 4, &window, &reason), reason);
    void* mh = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareHybrid(window, &mh, /*requirePath=*/false));

    constexpr uint8_t kSentinel = 0xC7;
    FillSentinel(window->ptr, kHybridGpuBytes, kSentinel);
    Barrier();

    const int peer = MPIHelpers::findRemotePeerForLocalRank(window->localRank);
    ASSERT_TRUE(MPIHelpers::allRanksTrue(peer >= 0)) << "no remote peer for local rank " << window->localRank;
    void* req = nullptr;
    EXPECT_EQ(ncclInvalidArgument,
              rma_->iget(rmaCtx_, 0, window->totalSize - 32, mh, 64,
                         /*localOff=*/0, mh, peer, ncclRmaOptFlagsDefault, &req));
    EXPECT_EQ(req, nullptr);
    EXPECT_TRUE(AllSentinel(window->ptr, kHybridGpuBytes, kSentinel));
    Barrier();
}

} // namespace RCCLRmaTests

#endif // RCCL_HAS_RMA_IB_PROXY
#endif // MPI_TESTS_ENABLED
