/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// CE AllReduce and CE gating over multi-segment symmetric windows.

#ifdef MPI_TESTS_ENABLED

#include "MultiSegmentTestFixture.hpp"

// recvBuf sits one segment past the window base and CE must still reduce it.
TEST_F(UBR_MultiSegment, CE_RecvOffset)
{
    MultiSegmentBuffer buf;
    ncclWindow_t win = nullptr;
    int rank = 0, nRanks = 0;
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    size_t count = 0, totalBytes = 0;
    prepareSymmetricLsaRecvOffset(buf, &win, &rank, &nRanks, &sendBuf, &recvBuf, &count, &totalBytes,
                                  /*requireCeEnv=*/true);
    auto vmmCleanup = makeScopeGuard([&]() { releaseMultiSegmentBuffer(buf); });
    auto winCleanup = makeScopeGuard([&]() {
        if (win) HIP_EXPECT(ncclCommWindowDeregister(getActiveCommunicator(), win));
    });
    if (::testing::Test::IsSkipped() || ::testing::Test::HasFatalFailure()) return;

    ASSERT_MPI_TRUE(ceAllReduceSelected(sendBuf, recvBuf, count));
    ASSERT_MPI_HELPER(expectCeRecvRangeInWindow(recvBuf, totalBytes, /*contained=*/true));

    initSendBuffer<T>(sendBuf, count, rank);
    ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(
        sendBuf, recvBuf, count, getNcclDataType<T>(), ncclSum,
        getActiveCommunicator(), getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
    ASSERT_MPI_TRUE(verifyAllReduceResult<T>(recvBuf, count, nRanks));
}

#ifdef ENABLE_FAULT_INJECTION
// Phase 3 peer copies use rank * shardBytes without the receive-window offset.
TEST_F(UBR_MultiSegment, CE_RecvOffset_BeforeLegacyCorruptsResult)
{
    MultiSegmentBuffer buf;
    ncclWindow_t win = nullptr;
    int rank = 0, nRanks = 0;
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    size_t count = 0, totalBytes = 0;
    prepareSymmetricLsaRecvOffset(buf, &win, &rank, &nRanks, &sendBuf, &recvBuf, &count, &totalBytes,
                                  /*requireCeEnv=*/true);
    auto vmmCleanup = makeScopeGuard([&]() { releaseMultiSegmentBuffer(buf); });
    auto winCleanup = makeScopeGuard([&]() {
        if (win) HIP_EXPECT(ncclCommWindowDeregister(getActiveCommunicator(), win));
    });
    if (::testing::Test::IsSkipped() || ::testing::Test::HasFatalFailure()) return;

    ASSERT_MPI_TRUE(ceAllReduceSelected(sendBuf, recvBuf, count));

    initSendBuffer<T>(sendBuf, count, rank);
    ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, totalBytes));

    ASSERT_MPI_EQ(ncclSuccess, ncclCeFaultSet(
        getActiveCommunicator(), CE_FAULT_LEGACY_RECV_OFFSET));
    auto faultCleanup = makeScopeGuard([&]() {
        HIP_EXPECT(ncclCeFaultClear(getActiveCommunicator()));
    });

    ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(
        sendBuf, recvBuf, count, getNcclDataType<T>(), ncclSum,
        getActiveCommunicator(), getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    const size_t shardElems = count / static_cast<size_t>(nRanks);
    ASSERT_MPI_TRUE(verifyBufferData<T>(recvBuf, count,
        [shardElems, rank, nRanks](size_t i) {
            return (i / shardElems == static_cast<size_t>(rank)) ? static_cast<T>(reducedValue(nRanks, i))
                                                                  : static_cast<T>(0.0f);
        }));
    EXPECT_FALSE(verifyAllReduceResult<T>(recvBuf, count, nRanks))
        << "BEFORE control did not reproduce the legacy receive-window offset corruption";
}
#endif

// The GPU window is the control; CE must reject the elastic window.
TEST_F(UBR_MultiSegment, CE_Elastic_Gating)
{
    runCeElasticGating(/*hierarchical=*/false);
}

TEST_F(UBR_MultiSegment, CE_Elastic_HierarchicalGating)
{
    runCeElasticGating(/*hierarchical=*/true);
}

#endif // MPI_TESTS_ENABLED
