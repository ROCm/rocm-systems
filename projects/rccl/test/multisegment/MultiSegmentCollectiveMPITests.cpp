/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef MPI_TESTS_ENABLED

#include "MultiSegmentTestFixture.hpp"


/**
 * @brief Single-node symmetric AllReduce over a GPU multi-segment window.
 *
 * The LSA team maps each cuMem segment and the symmetric kernel reduces into
 * recvBuf, one segment past the window base. CTA policy ZERO and
 * RCCL_FORCE_CE_ALLREDUCE select CE instead, so those jobs skip.
 */
TEST_F(UBR_MultiSegment, Symmetric_Lsa)
{
    if (envCtaPolicyIsZero()) {
        GTEST_SKIP() << "NCCL_CTA_POLICY=ZERO selects CE ahead of the symmetric kernel";
        return;
    }
    const char* forceCe = std::getenv("RCCL_FORCE_CE_ALLREDUCE");
    if (forceCe != nullptr && std::atoi(forceCe) == 1) {
        GTEST_SKIP() << "RCCL_FORCE_CE_ALLREDUCE selects CE ahead of the symmetric kernel";
        return;
    }

    MultiSegmentBuffer buf;
    ncclWindow_t win = nullptr;
    int rank = 0, nRanks = 0;
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    size_t count = 0, totalBytes = 0;
    prepareSymmetricLsaRecvOffset(buf, &win, &rank, &nRanks, &sendBuf, &recvBuf, &count, &totalBytes);
    auto vmmCleanup = makeScopeGuard([&]() { releaseMultiSegmentBuffer(buf); });
    auto winCleanup = makeScopeGuard([&]() {
        if (win) HIP_EXPECT(ncclCommWindowDeregister(getActiveCommunicator(), win));
    });
    if (::testing::Test::IsSkipped() || ::testing::Test::HasFatalFailure()) return;

    {
        const std::string why = mpiCoordinatedSkipReason(
            !symmetricKernelExpected("AllReduce"),
            "single-node symmetric AllReduce kernel is not available");
        if (!why.empty()) GTEST_SKIP() << why;
    }
    if (::testing::Test::IsSkipped()) return;

    ASSERT_TRUE(isPerRankLoggingEnabled()) << "RCCL_MPI_LOG_ALL_RANKS must be set to 1";
    constexpr int kNumSegments = 4;
    SCOPED_TRACE("kNumSegments=" + std::to_string(kNumSegments));
    (void)totalBytes;

    initSendBuffer<T>(sendBuf, count, rank);

    ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(sendBuf, recvBuf, count, getNcclDataType<T>(), ncclSum, getActiveCommunicator(), getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
    ASSERT_TRUE(verifyAllReduceResult<T>(recvBuf, count, nRanks));

    REGLogChecker checker = getLogChecker();
    ASSERT_TRUE(checker.hasNumSegments(kNumSegments))
        << "Expected 'numSegments " << kNumSegments
        << "' in log - the symmetric-window multi-segment LSA registration "
           "(symMemoryMapLsaTeam) did not fire";
    expectSchedulerPath("AllReduce", true);
}

TEST_F(UBR_MultiSegment, Symmetric_Lsa_AllGather)
{
    runLsaGinWindowCollective("AllGather", /*elastic=*/false, ProxyPath::Symmetric, /*singleNode=*/true);
}

TEST_F(UBR_MultiSegment, Symmetric_Lsa_ReduceScatter)
{
    runLsaGinWindowCollective("ReduceScatter", /*elastic=*/false, ProxyPath::Symmetric, /*singleNode=*/true);
}

/**
 * @brief Recv pointer sits inside a 4-segment window but the receive range
 *        overruns into a 5th mapped segment that is not window-registered.
 *
 * Pointer-only containment would take the LSA fast path and write past the
 * window. Range containment must fall back to staging (totalBytes fits).
 */
TEST_F(UBR_MultiSegment, Symmetric_Lsa_RecvRangePastWindowFallsBack)
{
    if (!validateTestPrerequisites(
            /*min_processes=*/2, /*max_processes=*/kNoProcessLimit,
            /*require_power_of_two=*/kNoPowerOfTwoRequired,
            /*min_nodes=*/1, /*max_nodes=*/1)) {
        GTEST_SKIP() << "Requires 2+ ranks on exactly one node";
    }
    if (const char* why = ceRecvOffsetEnvSkipReason()) {
        GTEST_SKIP() << why;
    }
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ASSERT_TRUE(isCuMemEnabled()) << "NCCL_CUMEM_ENABLE must be set to 1";
    ASSERT_TRUE(isWinEnabled()) << "NCCL_WIN_ENABLE must not be set to 0";

    int dev = 0;
    ASSERT_MPI_EQ(hipSuccess, hipGetDevice(&dev));

    constexpr size_t kSegmentSize = 32 * 1024 * 1024;
    constexpr int kAllocSegments = 5;
    constexpr int kWinSegments = 4;

    int rank = 0;
    int nRanks = 0;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    MultiSegmentBuffer buf;
    ASSERT_NO_FATAL_FAILURE(createMultiSegmentBuffer(dev, kSegmentSize, kAllocSegments, buf));
    {
        const std::string why = skipUnlessAllRanksAllocated(buf.totalSize != 0,
            "Raw VMM (hipMemCreate / Reserve / Map) not supported on this runtime");
        if (!why.empty()) GTEST_SKIP() << why;
    }
    auto vmmCleanup = makeScopeGuard([&]() { releaseMultiSegmentBuffer(buf); });

    const size_t windowBytes = buf.segmentSize * static_cast<size_t>(kWinSegments);
    const size_t recvOffset = buf.segmentSize * 3 + buf.segmentSize / 2;
    const size_t totalBytes = buf.segmentSize;
    ASSERT_LT(recvOffset, windowBytes);
    ASSERT_GT(recvOffset + totalBytes, windowBytes);
    ASSERT_LE(recvOffset + totalBytes, buf.totalSize);
    ASSERT_EQ(totalBytes % sizeof(T), 0u);
    ASSERT_LE(totalBytes, ncclCeAllReduceStagingBufBytes(nRanks));

    char* base = reinterpret_cast<char*>(buf.vaBase);
    void* sendBuf = base;
    void* recvBuf = base + recvOffset;
    const size_t count = totalBytes / sizeof(T);
    if (nRanks <= 0 || count % static_cast<size_t>(nRanks) != 0) {
        GTEST_SKIP() << "CE shard path requires count divisible by nRanks";
    }

    ncclWindow_t win = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(
        getActiveCommunicator(), buf.vaBase, windowBytes, &win, NCCL_WIN_COLL_SYMMETRIC));
    auto winCleanup = makeScopeGuard([&]() {
        if (win) HIP_EXPECT(ncclCommWindowDeregister(getActiveCommunicator(), win));
    });
    ASSERT_MPI_NE(win, nullptr);

    {
        const std::string why = mpiCoordinatedSkipReason(
            !ceAllReduceSelected(sendBuf, recvBuf, count),
            "CE AllReduce was not selected (rcclGetCollImplInfo)");
        if (!why.empty()) GTEST_SKIP() << why;
    }

    initSendBuffer<T>(sendBuf, count, rank);
    ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(
        sendBuf, recvBuf, count, getNcclDataType<T>(), ncclSum,
        getActiveCommunicator(), getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
    ASSERT_TRUE(verifyAllReduceResult<T>(recvBuf, count, nRanks));
}

/**
 * @brief Same past-window geometry as RecvRangePastWindowFallsBack, but
 *        totalBytes larger than ceARTmpBuf. Fast path is closed, so the
 *        staging fallback must refuse rather than memcpy past the temp buffer.
 */
TEST_F(UBR_MultiSegment, Symmetric_Lsa_RecvRangePastWindowStagingOverflow)
{
    if (!validateTestPrerequisites(
            /*min_processes=*/2, /*max_processes=*/kNoProcessLimit,
            /*require_power_of_two=*/kNoPowerOfTwoRequired,
            /*min_nodes=*/1, /*max_nodes=*/1)) {
        GTEST_SKIP() << "Requires 2+ ranks on exactly one node";
    }
    if (const char* why = ceRecvOffsetEnvSkipReason()) {
        GTEST_SKIP() << why;
    }
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ASSERT_TRUE(isCuMemEnabled()) << "NCCL_CUMEM_ENABLE must be set to 1";
    ASSERT_TRUE(isWinEnabled()) << "NCCL_WIN_ENABLE must not be set to 0";

    int dev = 0;
    ASSERT_MPI_EQ(hipSuccess, hipGetDevice(&dev));

    constexpr size_t kSegmentSize = 32 * 1024 * 1024;
    constexpr int kAllocSegments = 5;
    constexpr int kWinSegments = 4;

    int rank = 0;
    int nRanks = 0;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    MultiSegmentBuffer buf;
    ASSERT_NO_FATAL_FAILURE(createMultiSegmentBuffer(dev, kSegmentSize, kAllocSegments, buf));
    {
        const std::string why = skipUnlessAllRanksAllocated(buf.totalSize != 0,
            "Raw VMM (hipMemCreate / Reserve / Map) not supported on this runtime");
        if (!why.empty()) GTEST_SKIP() << why;
    }
    auto vmmCleanup = makeScopeGuard([&]() { releaseMultiSegmentBuffer(buf); });

    const size_t windowBytes = buf.segmentSize * static_cast<size_t>(kWinSegments);
    const size_t recvOffset = buf.segmentSize * 3;
    const size_t totalBytes = buf.segmentSize * 2;
    ASSERT_LT(recvOffset, windowBytes);
    ASSERT_GT(recvOffset + totalBytes, windowBytes);
    ASSERT_LE(recvOffset + totalBytes, buf.totalSize);
    ASSERT_EQ(totalBytes % sizeof(T), 0u);
    ASSERT_GT(totalBytes, ncclCeAllReduceStagingBufBytes(nRanks));

    char* base = reinterpret_cast<char*>(buf.vaBase);
    void* sendBuf = base;
    void* recvBuf = base + recvOffset;
    const size_t count = totalBytes / sizeof(T);
    if (nRanks <= 0 || count % static_cast<size_t>(nRanks) != 0) {
        GTEST_SKIP() << "CE shard path requires count divisible by nRanks";
    }

    ncclWindow_t win = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(
        getActiveCommunicator(), buf.vaBase, windowBytes, &win, NCCL_WIN_COLL_SYMMETRIC));
    auto winCleanup = makeScopeGuard([&]() {
        if (win) HIP_EXPECT(ncclCommWindowDeregister(getActiveCommunicator(), win));
    });
    ASSERT_MPI_NE(win, nullptr);

    {
        const std::string why = mpiCoordinatedSkipReason(
            !ceAllReduceSelected(sendBuf, recvBuf, count),
            "CE AllReduce was not selected (rcclGetCollImplInfo)");
        if (!why.empty()) GTEST_SKIP() << why;
    }

    initSendBuffer<T>(sendBuf, count, rank);
    ASSERT_MPI_EQ(ncclInvalidUsage, ncclAllReduce(
        sendBuf, recvBuf, count, getNcclDataType<T>(), ncclSum,
        getActiveCommunicator(), getActiveStream()));
}

/**
 * @brief Multi-segment registration on the combined LSA + GIN symmetric path.
 *
 * Cross-node ReduceScatter over symmetric windows (NCCL_WIN_COLL_SYMMETRIC)
 * whose VA spans several physical cuMem segments.
 *   - LSA: symMemoryMapLsaTeam maps each segment per peer and logs
 *     "... numSegments <N>" (dev_runtime.cc).
 *   - GIN: symMemoryRegisterGin registers the (all-device, contiguous)
 *     multi-segment buffer for the inter-node proxy.
 *
 * Requires >=2 nodes (a GIN inter-node leg) and >=2 ranks per node (an LSA
 * intra-node leg); otherwise the path under test is not exercised and the
 * test SKIPs.
 *
 */
TEST_F(UBR_MultiSegment, Symmetric_LsaGin)
{
    const int nodeCount = MPITestConstants::detectNodeCount();
    if (!validateTestPrerequisites(/*min_processes=*/2)) {
        GTEST_SKIP() << "Requires 2+ ranks";
    }
    if (nodeCount < 2) {
        GTEST_SKIP() << "LSA+GIN ReduceScatter requires >=2 nodes";
    }
    if (!isGinEnabled()) {
        GTEST_SKIP() << "Requires GIN (NCCL_GIN_ENABLE!=0 and NCCL_GIN_TYPE=2)";
    }

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ASSERT_TRUE(isCuMemEnabled()) << "NCCL_CUMEM_ENABLE must be set to 1";
    ASSERT_TRUE(isWinEnabled()) << "NCCL_WIN_ENABLE must not be set to 0";
    ASSERT_TRUE(isPerRankLoggingEnabled()) << "RCCL_MPI_LOG_ALL_RANKS must be set to 1";

    int dev = 0;
    ASSERT_MPI_EQ(hipSuccess, hipGetDevice(&dev));

    int rank   = 0;
    int nRanks = 0;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    if (nRanks / nodeCount < 2) {
        GTEST_SKIP() << "LSA+GIN ReduceScatter requires >=2 ranks per node (LSA intra-node leg)";
    }

    constexpr size_t kSegmentSize = 4 * 1024 * 1024;
    constexpr int    kNumSegments = 4;

    SCOPED_TRACE("kNumSegments=" + std::to_string(kNumSegments) +
                 " nodeCount=" + std::to_string(nodeCount) +
                 " nRanks=" + std::to_string(nRanks));

    MultiSegmentBuffer recvSeg;
    ASSERT_NO_FATAL_FAILURE(createMultiSegmentBuffer(dev, kSegmentSize, kNumSegments, recvSeg));
    {
        const std::string why = skipUnlessAllRanksAllocated(recvSeg.totalSize != 0,
            "Raw VMM (hipMemCreate / Reserve / Map) not supported on this runtime");
        if (!why.empty()) GTEST_SKIP() << why;
    }
    auto recvVmmCleanup = makeScopeGuard([&]() { releaseMultiSegmentBuffer(recvSeg); });

    ASSERT_EQ(recvSeg.totalSize % sizeof(T), 0u);
    const size_t recvCount = recvSeg.totalSize / sizeof(T);

    MultiSegmentBuffer sendSeg;
    ASSERT_NO_FATAL_FAILURE(
        createMultiSegmentBuffer(dev, recvSeg.segmentSize * static_cast<size_t>(nRanks),
                                 kNumSegments, sendSeg));
    {
        const std::string why = skipUnlessAllRanksAllocated(sendSeg.totalSize != 0,
            "Raw VMM (hipMemCreate / Reserve / Map) not supported on this runtime");
        if (!why.empty()) GTEST_SKIP() << why;
    }
    auto sendVmmCleanup = makeScopeGuard([&]() { releaseMultiSegmentBuffer(sendSeg); });

    const size_t sendCount = recvCount * static_cast<size_t>(nRanks);
    ASSERT_GE(sendSeg.totalSize, sendCount * sizeof(T));

    ncclWindow_t sendWin = nullptr;
    ncclWindow_t recvWin = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(getActiveCommunicator(), sendSeg.vaBase, sendSeg.totalSize, &sendWin, NCCL_WIN_COLL_SYMMETRIC));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(getActiveCommunicator(), recvSeg.vaBase, recvSeg.totalSize, &recvWin, NCCL_WIN_COLL_SYMMETRIC));
    auto winCleanup = makeScopeGuard([&]() {
        if (sendWin) HIP_EXPECT(ncclCommWindowDeregister(getActiveCommunicator(), sendWin));
        if (recvWin) HIP_EXPECT(ncclCommWindowDeregister(getActiveCommunicator(), recvWin));
    });
    ASSERT_MPI_NE(sendWin, nullptr);
    ASSERT_MPI_NE(recvWin, nullptr);

    initSendBuffer<T>(sendSeg.vaBase, sendCount, rank);

    ASSERT_MPI_EQ(ncclSuccess, ncclReduceScatter(sendSeg.vaBase, recvSeg.vaBase, recvCount, getNcclDataType<T>(), ncclSum, getActiveCommunicator(), getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
    ASSERT_TRUE(verifyReduceScatterResult<T>(recvSeg.vaBase, recvCount, nRanks));

    REGLogChecker checker = getLogChecker();
    TEST_INFO("LsaGin_MultiSegment_ReduceScatter: %s (log size: %zu bytes)",
              checker.getSummary().c_str(), checker.getContentLength());
    ASSERT_TRUE(checker.hasNumSegments(kNumSegments))
        << "Expected 'numSegments " << kNumSegments
        << "' in log - the symmetric-window multi-segment registration "
           "(symMemoryMapLsaTeam) did not fire for the LSA+GIN ReduceScatter path";
    expectSchedulerPath("ReduceScatter", symmetricKernelExpected("ReduceScatter"));
}

/**
 * @brief LSA-only elastic buffer: symmetric window with a
 *        host-backed segment.
 *
 * Builds a multi-segment symmetric window (NCCL_WIN_COLL_SYMMETRIC) whose
 * trailing segment is CPU memory (hipMemLocationTypeHost). With elastic
 * registration enabled, ncclCommWindowRegister must accept the mixed buffer,
 * the LSA team maps every segment (symMemoryMapLsaTeam logs per segment), and
 * an AllReduce over the window must produce correct results - the symmetric
 * scheduler falls back to legacy kernels for windows with sysmem segments
 * (symmetric_sched.cc).
 *
 */
TEST_F(UBR_MultiSegment, Symmetric_Elastic_Lsa)
{
    if (!validateTestPrerequisites(
            /*min_processes=*/2, /*max_processes=*/kNoProcessLimit,
            /*require_power_of_two=*/kNoPowerOfTwoRequired,
            /*min_nodes=*/1, /*max_nodes=*/1)) {
        GTEST_SKIP() << "Requires 2+ ranks on a single node";
    }

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
 
     ASSERT_TRUE(isCuMemEnabled()) << "NCCL_CUMEM_ENABLE must be set to 1";
     ASSERT_TRUE(isWinEnabled()) << "NCCL_WIN_ENABLE must not be set to 0";
     ASSERT_TRUE(isElasticBufferRegisterEnabled())
         << "NCCL_ELASTIC_BUFFER_REGISTER must not be 0 for the elastic (host-backed) path";
     ASSERT_TRUE(isPerRankLoggingEnabled()) << "RCCL_MPI_LOG_ALL_RANKS must be set to 1";
 
     int dev = 0;
     ASSERT_MPI_EQ(hipSuccess, hipGetDevice(&dev));
 
     int rank   = 0;
     int nRanks = 0;
     ncclCommUserRank(getActiveCommunicator(), &rank);
     ncclCommCount(getActiveCommunicator(), &nRanks);
 
     constexpr size_t kSegmentSize     = 4 * 1024 * 1024;
     constexpr int    kNumSegments     = 4;
     constexpr int    kNumHostSegments = 1; // trailing segment is host-backed
 
     SCOPED_TRACE("kNumSegments=" + std::to_string(kNumSegments) +
                  " kNumHostSegments=" + std::to_string(kNumHostSegments) +
                  " nRanks=" + std::to_string(nRanks));
 
     MultiSegmentBuffer buf;
     ASSERT_NO_FATAL_FAILURE(
         createMixedMultiSegmentBuffer(dev, kSegmentSize, kNumSegments, kNumHostSegments, buf));
     {
         const std::string why = skipUnlessAllRanksAllocated(buf.totalSize != 0,
             "Host VMM (hipMemCreate with hipMemLocationTypeHost) not supported on this runtime");
         if (!why.empty()) GTEST_SKIP() << why;
     }
     auto vmmCleanup = makeScopeGuard([&]() { releaseMultiSegmentBuffer(buf); });
 
     // Split the window into send/recv halves. With kNumHostSegments=1 the recv
     // half straddles the host-backed trailing segment, exercising host access.
     const size_t halfSize = buf.totalSize / 2;
     ASSERT_EQ(halfSize % sizeof(T), 0u);
     char*  base    = reinterpret_cast<char*>(buf.vaBase);
     void*  sendBuf = base;
     void*  recvBuf = base + halfSize;
     size_t count   = halfSize / sizeof(T);
 
     ncclWindow_t win = nullptr;
     ncclResult_t result = ncclCommWindowRegister(
         getActiveCommunicator(), buf.vaBase, buf.totalSize, &win, NCCL_WIN_COLL_SYMMETRIC);
     ASSERT_MPI_EQ(ncclSuccess, result);
     auto winCleanup = makeScopeGuard([&]() {
         if (win) HIP_EXPECT(ncclCommWindowDeregister(getActiveCommunicator(), win));
     });
     ASSERT_MPI_NE(win, nullptr);
 
     initSendBuffer<T>(sendBuf, count, rank);
 
     ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(sendBuf, recvBuf, count, getNcclDataType<T>(), ncclSum, getActiveCommunicator(), getActiveStream()));
     ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
     ASSERT_TRUE(verifyAllReduceResult<T>(recvBuf, count, nRanks));
 
     REGLogChecker checker = getLogChecker();
     TEST_INFO("LsaElastic_MultiSegment_AllReduce: %s (log size: %zu bytes)",
               checker.getSummary().c_str(), checker.getContentLength());
    ASSERT_TRUE(checker.hasNumSegments(kNumSegments))
        << "Expected 'numSegments " << kNumSegments
        << "' in log - the symmetric-window multi-segment LSA registration "
           "(symMemoryMapLsaTeam) did not fire for the elastic (host-backed) buffer";
    expectSchedulerPath("AllReduce", false);
}

TEST_F(UBR_MultiSegment, Symmetric_LsaGin_AllGather)
{
    runLsaGinWindowCollective("AllGather", /*elastic=*/false, ProxyPath::Symmetric);
}

TEST_F(UBR_MultiSegment, Symmetric_LsaGin_AllGather_LegacyFallback)
{
    runLsaGinWindowCollective("AllGather", /*elastic=*/false, ProxyPath::Legacy);
}

TEST_F(UBR_MultiSegment, Symmetric_LsaGin_AllReduce)
{
    runLsaGinWindowCollective("AllReduce", /*elastic=*/false, ProxyPath::Legacy);
}

TEST_F(UBR_MultiSegment, Symmetric_Elastic_Lsa_AllGather)
{
    runLsaGinWindowCollective("AllGather", /*elastic=*/true, ProxyPath::Legacy, /*singleNode=*/true);
}

TEST_F(UBR_MultiSegment, Symmetric_Elastic_Lsa_AlltoAll)
{
    runLsaGinWindowCollective("AlltoAll", /*elastic=*/true, ProxyPath::Legacy, /*singleNode=*/true);
}

TEST_F(UBR_MultiSegment, Symmetric_Elastic_Lsa_ReduceScatter)
{
    runLsaGinWindowCollective("ReduceScatter", /*elastic=*/true, ProxyPath::Legacy, /*singleNode=*/true);
}

TEST_F(UBR_MultiSegment, Symmetric_Elastic_LsaGin_AllGather)
{
    runLsaGinWindowCollective("AllGather", /*elastic=*/true, ProxyPath::Legacy);
}

TEST_F(UBR_MultiSegment, Symmetric_Elastic_LsaGin_AllReduce)
{
    runLsaGinWindowCollective("AllReduce", /*elastic=*/true, ProxyPath::Legacy);
}

TEST_F(UBR_MultiSegment, Symmetric_Elastic_LsaGin_ReduceScatter)
{
    runLsaGinWindowCollective("ReduceScatter", /*elastic=*/true, ProxyPath::Legacy);
}

// Recv is the full hybrid window, including the imported CPU tail. The send
// window is a larger GPU multi-segment buffer so the ReduceScatter count is
// not clipped to the GPU prefix.
TEST_F(UBR_MultiSegment, DeepEP_HybridReduceScatter)
{
    runHybridWindowCollective("ReduceScatter");
}

TEST_F(UBR_MultiSegment, DeepEP_HybridAllGather)
{
    runHybridWindowCollective("AllGather");
}

TEST_F(UBR_MultiSegment, DeepEP_HybridAllReduce)
{
    runHybridWindowCollective("AllReduce");
}

TEST_F(UBR_MultiSegment, DeepEP_HybridSingleNodeReduceScatter)
{
    runHybridWindowCollective("ReduceScatter", /*singleNode=*/true);
}

TEST_F(UBR_MultiSegment, DeepEP_HybridSingleNodeAllGather)
{
    runHybridWindowCollective("AllGather", /*singleNode=*/true);
}

TEST_F(UBR_MultiSegment, DeepEP_HybridSingleNodeAllReduce)
{
    runHybridWindowCollective("AllReduce", /*singleNode=*/true);
}

#endif // MPI_TESTS_ENABLED
