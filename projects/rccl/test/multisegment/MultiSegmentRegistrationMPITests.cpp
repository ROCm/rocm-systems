/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef MPI_TESTS_ENABLED

#include "MultiSegmentTestFixture.hpp"

/**
 * @brief Out-of-place AllReduce on a buffer that spans multiple VMM segments.
 *
 * Exercises the multi-segment registration branch on Ring AllReduce.
 *
 * Layout (N = kSegmentsPerHalf, total 2 * N physical segments allocated):
 *   Total reserved VA = 2 * N * kSegmentSize
 *   sendbuff = [0,                  N * kSegmentSize)  covers first N segments
 *   recvbuff = [N * kSegmentSize,  2N * kSegmentSize)  covers last  N segments
 *
 * The collective operates on four segments per half, while ncclCommRegister
 * covers the complete eight-segment allocation. One node must complete IPC
 * registration of all eight segments. Two or more nodes take the NET path:
 * under NCCL_NET=IB-CAST the CAST register arm must set NET_REG_COMPLETE and
 * cache eight segments; skip would stay green if IbCastRegMrDmaBufMultiSeg
 * were reverted. Classic IB still skips when no rank completed NET registration.
 */
TEST_F(UBR_MultiSegment, Generic)
{
    if (!validateTestPrerequisites(
            /*min_processes=*/2, /*max_processes=*/kNoProcessLimit,
            /*require_power_of_two=*/kNoPowerOfTwoRequired,
            /*min_nodes=*/1, /*max_nodes=*/kNoNodeLimit)) {
        GTEST_SKIP() << "Requires 2+ ranks";
    }
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ASSERT_TRUE(isUBREnabled()) << "NCCL_LOCAL_REGISTER must be set to 1";
    ASSERT_TRUE(isCuMemEnabled()) << "NCCL_CUMEM_ENABLE must be set to 1";
    ASSERT_TRUE(isMultiSegmentRegisterEnabled()) << "NCCL_MULTI_SEGMENT_REGISTER must be set to 1";

    int dev = 0;
    ASSERT_MPI_EQ(hipSuccess, hipGetDevice(&dev));

    constexpr size_t kSegmentSize     = 32 * 1024 * 1024;
    constexpr int    kNumSegments     = 8;

    int rank   = 0;
    int nRanks = 0;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    SCOPED_TRACE("kNumSegments=" + std::to_string(kNumSegments));

    MultiSegmentBuffer buf;
    ASSERT_NO_FATAL_FAILURE(
        createMultiSegmentBuffer(dev, kSegmentSize, kNumSegments, buf));
    auto vmmCleanup = makeScopeGuard([&]() { releaseMultiSegmentBuffer(buf); });
    {
        const std::string why = skipUnlessAllRanksAllocated(buf.totalSize != 0,
            "Raw VMM (hipMemCreate / Reserve / Map) not supported on this runtime");
        if (!why.empty()) GTEST_SKIP() << why;
    }

    const size_t halfSize = buf.totalSize / 2;
    ASSERT_EQ(halfSize % sizeof(T), 0u);
    char*  base    = reinterpret_cast<char*>(buf.vaBase);
    void*  sendBuf = base;
    void*  recvBuf = base + halfSize;
    size_t count   = halfSize / sizeof(T);

    void* regHandle = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommRegister(getActiveCommunicator(), buf.vaBase, buf.totalSize, &regHandle));
    auto regCleanup = makeScopeGuard([&]() {
        if (regHandle) HIP_EXPECT(ncclCommDeregister(getActiveCommunicator(), regHandle));
    });
    ASSERT_MPI_NE(regHandle, nullptr);

    initSendBuffer<T>(sendBuf, count, rank);

    ncclResult_t result = ncclAllReduce(sendBuf, recvBuf, count, getNcclDataType<T>(), ncclSum, getActiveCommunicator(), getActiveStream());
    ASSERT_MPI_EQ(ncclSuccess, result);
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
    ASSERT_MPI_TRUE(verifyAllReduceResult<T>(recvBuf, count, nRanks));

    struct ncclReg* reg = nullptr;
    ncclRegFind(reinterpret_cast<struct ncclComm*>(getActiveCommunicator()), buf.vaBase, buf.totalSize, &reg);
    ASSERT_MPI_NE(reg, nullptr);
    if (MPITestConstants::detectNodeCount() == 1) {
        ASSERT_NE(reg->state & IPC_REG_COMPLETE, 0u)
            << "Single-node AllReduce did not complete IPC registration";
        ASSERT_NE(reg->ipcInfos, nullptr);
        int ipcPeers = 0;
        for (int i = 0; i < reg->ipcInfosSize; ++i) {
            if (reg->ipcInfos[i] == nullptr) continue;
            ++ipcPeers;
            ASSERT_EQ(reg->ipcInfos[i]->impInfo.numSegments, kNumSegments)
                << "IPC registration walked a prefix of the ncclCommRegister range, not the full 8-segment allocation";
        }
        ASSERT_GT(ipcPeers, 0) << "IPC registration recorded no peer";
    } else {
        const bool netPeersDone = reg->rcclNet.allPeers;
        if (envNetIsIbCast()) {
            const bool netDone = (reg->state & NET_REG_COMPLETE) != 0 && reg->rcclNet.nSegments != 0;
            ASSERT_TRUE(MPIHelpers::allRanksTrue(netDone))
                << "CAST netIbCast register arm did not set NET_REG_COMPLETE";
            ASSERT_EQ(reg->rcclNet.nSegments, kNumSegments)
                << "NET registration walked a prefix of the ncclCommRegister range, not the full 8-segment allocation";
        } else {
            const std::string why = mpiCoordinatedSkipReason(
                !MPIHelpers::anyRankTrue(netPeersDone),
                "NET registration did not finish for every peer on any rank");
            if (!why.empty()) GTEST_SKIP() << why;
            if (netPeersDone) {
                ASSERT_EQ(reg->rcclNet.nSegments, kNumSegments)
                    << "NET registration walked a prefix of the ncclCommRegister range, not the full 8-segment allocation";
            }
        }
    }
}

/**
 * @brief BEFORE control for the registration-reuse regression.
 *
 * The original test repeated an identical AllReduce while the CE fast path was
 * enabled. Forced CE handles the registered buffer directly, so neither call
 * enters the IPC/NET transport registration cache. This control intentionally
 * recreates that setup and passes only when no transport registration or reuse
 * lookup is logged.
 *
 * Run this test in its own process with RCCL_CE_ALLREDUCE=1,
 * RCCL_FORCE_CE_ALLREDUCE=1, and NCCL_CTA_POLICY=2. The corresponding AFTER
 * test is Generic_Reuse, run with RCCL_CE_ALLREDUCE=0.
 */
TEST_F(UBR_MultiSegment, Generic_Reuse_BeforeCePlanBypassesRegistrationCache)
{
    if (!validateTestPrerequisites(/*min_processes=*/2)) {
        GTEST_SKIP() << "Requires 2+ ranks";
    }
    const char* ceAllReduce = std::getenv("RCCL_CE_ALLREDUCE");
    if (ceAllReduce == nullptr || std::atoi(ceAllReduce) != 1) {
        GTEST_SKIP() << "BEFORE control requires RCCL_CE_ALLREDUCE=1";
    }
    const char* forceCeAllReduce = std::getenv("RCCL_FORCE_CE_ALLREDUCE");
    if (forceCeAllReduce == nullptr || std::atoi(forceCeAllReduce) != 1) {
        GTEST_SKIP() << "BEFORE control requires RCCL_FORCE_CE_ALLREDUCE=1";
    }
    if (!envCtaPolicyIsZero()) {
        GTEST_SKIP() << "BEFORE control requires NCCL_CTA_POLICY=2 or ZERO";
    }
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ASSERT_TRUE(isUBREnabled()) << "NCCL_LOCAL_REGISTER must be set to 1";
    ASSERT_TRUE(isCuMemEnabled()) << "NCCL_CUMEM_ENABLE must be set to 1";
    ASSERT_TRUE(isMultiSegmentRegisterEnabled())
        << "NCCL_MULTI_SEGMENT_REGISTER must not be 0";
    ASSERT_TRUE(isPerRankLoggingEnabled()) << "RCCL_MPI_LOG_ALL_RANKS must be set to 1";

    int dev = 0;
    ASSERT_MPI_EQ(hipSuccess, hipGetDevice(&dev));

    constexpr size_t kRequestedSegmentSize = 32 * 1024 * 1024;
    constexpr int kNumSegments = 4;
    MultiSegmentBuffer buf;
    ASSERT_NO_FATAL_FAILURE(createMultiSegmentBuffer(dev, kRequestedSegmentSize, kNumSegments, buf));
    auto vmmCleanup = makeScopeGuard([&]() { releaseMultiSegmentBuffer(buf); });
    {
        const std::string why = skipUnlessAllRanksAllocated(buf.totalSize != 0,
            "Raw VMM (hipMemCreate / Reserve / Map) not supported on this runtime");
        if (!why.empty()) GTEST_SKIP() << why;
    }

    const size_t halfSize = buf.totalSize / 2;
    char* base = reinterpret_cast<char*>(buf.vaBase);
    void* sendBuf = base;
    void* recvBuf = base + halfSize;
    const size_t count = halfSize / sizeof(T);

    void* regHandle = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommRegister(
        getActiveCommunicator(), buf.vaBase, buf.totalSize, &regHandle));
    auto regCleanup = makeScopeGuard([&]() {
        if (regHandle) HIP_EXPECT(ncclCommDeregister(getActiveCommunicator(), regHandle));
    });
    ASSERT_MPI_NE(regHandle, nullptr);

    int rank = 0;
    int nRanks = 0;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);
    initSendBuffer<T>(sendBuf, count, rank);

    for (int iteration = 0; iteration < 2; ++iteration) {
        ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(
            sendBuf, recvBuf, count, getNcclDataType<T>(), ncclSum,
            getActiveCommunicator(), getActiveStream()));
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
        ASSERT_TRUE(verifyAllReduceResult<T>(recvBuf, count, nRanks));
        if (iteration == 0) {
            ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, halfSize));
        }
    }

    REGLogChecker checker = getLogChecker();
    TEST_INFO("RegisterReuse_BEFORE_CePlan: %s (log size: %zu bytes)",
              checker.getSummary().c_str(), checker.getContentLength());
    struct ncclReg* reg = nullptr;
    ncclRegFind(reinterpret_cast<struct ncclComm*>(getActiveCommunicator()), buf.vaBase, buf.totalSize, &reg);
    ASSERT_NE(reg, nullptr) << "ncclCommRegister did not publish a cache entry";
    EXPECT_EQ(reg->state & (NET_REG_COMPLETE | IPC_REG_COMPLETE), 0)
        << "Forced CE unexpectedly completed IPC/NET transport registration";
    EXPECT_FALSE(checker.hasIPCRegistration() || checker.hasNETRegistration())
        << "Forced CE unexpectedly entered the transport registration cache";
    EXPECT_FALSE(checker.hasIPCReuse() || checker.hasNETReuse())
        << "Forced CE unexpectedly revisited the transport registration cache";
}

// Odd hipMalloc sizes page-round past the allocation. The NET segment count must still be 1.
TEST_F(UBR_MultiSegment, NetSegmentCountOddSizedHipMalloc)
{
    if (!validateTestPrerequisites(
            /*min_processes=*/1, /*max_processes=*/kNoProcessLimit,
            /*require_power_of_two=*/kNoPowerOfTwoRequired,
            /*min_nodes=*/1, /*max_nodes=*/kNoNodeLimit)) {
        GTEST_SKIP() << "Requires at least one rank";
    }
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ASSERT_TRUE(isUBREnabled()) << "NCCL_LOCAL_REGISTER must be set to 1";

    auto* comm = reinterpret_cast<struct ncclComm*>(getActiveCommunicator());
    for (const size_t size : {size_t{1000001}, size_t{2 * 1024 * 1024 + 1}}) {
        SCOPED_TRACE("size=" + std::to_string(size));
        void* buf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&buf, size));
        auto bufCleanup = makeScopeGuard([&]() { HIP_EXPECT(hipFree(buf)); });

        void* regHandle = nullptr;
        ASSERT_MPI_EQ(ncclSuccess, ncclCommRegister(getActiveCommunicator(), buf, size, &regHandle));
        auto regCleanup = makeScopeGuard([&]() {
            if (regHandle) HIP_EXPECT(ncclCommDeregister(getActiveCommunicator(), regHandle));
        });

        struct ncclReg* reg = nullptr;
        ncclRegFind(comm, buf, size, &reg);
        ASSERT_MPI_NE(reg, nullptr);
        ASSERT_MPI_TRUE(reg->endAddr > reinterpret_cast<uintptr_t>(buf) + size);

        int numSegments = 0;
        ASSERT_MPI_EQ(ncclSuccess, rcclNetRegSegmentCount(comm, reg, &numSegments));
        ASSERT_MPI_EQ(1, numSegments);
    }
}

/**
 * @brief NET proxy registration clips the final physical segment.
 *
 * Registers two complete VMM mappings plus half of a third, then runs an
 * AllReduce whose receive half ends at the final registered byte. This drives
 * sendProxyRegBuffer and recvProxyRegBuffer through netIbRegMrMultiSeg with a
 * short final MR and verifies the mapped-but-unregistered tail is untouched.
 * Ranks that took the NET path must cache a segment count of 3 on that entry.
 */
TEST_F(UBR_MultiSegment, NetProxyPartialFinalSegment)
{
    if (!validateTestPrerequisites(/*min_processes=*/2)) {
        GTEST_SKIP() << "Requires 2+ ranks";
    }
    if (MPITestConstants::detectNodeCount() != 2) {
        GTEST_SKIP() << "Requires exactly two nodes to exercise NET proxy registration";
    }
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ASSERT_TRUE(isUBREnabled()) << "NCCL_LOCAL_REGISTER must be set to 1";
    ASSERT_TRUE(isCuMemEnabled()) << "NCCL_CUMEM_ENABLE must be set to 1";
    ASSERT_TRUE(isMultiSegmentRegisterEnabled())
        << "NCCL_MULTI_SEGMENT_REGISTER must be set to 1";

    int dev = 0;
    ASSERT_MPI_EQ(hipSuccess, hipGetDevice(&dev));

    constexpr size_t kSegmentSize = 32 * 1024 * 1024;
    constexpr int kMappedSegments = 3;
    constexpr uint8_t kSentinel = 0xA7;
    MultiSegmentBuffer buf;
    ASSERT_NO_FATAL_FAILURE(
        createMultiSegmentBuffer(dev, kSegmentSize, kMappedSegments, buf));
    auto vmmCleanup = makeScopeGuard([&]() { releaseMultiSegmentBuffer(buf); });
    {
        const std::string why = skipUnlessAllRanksAllocated(buf.totalSize != 0,
            "Raw VMM (hipMemCreate / Reserve / Map) not supported on this runtime");
        if (!why.empty()) GTEST_SKIP() << why;
    }

    const size_t registeredBytes = 2 * buf.segmentSize + buf.segmentSize / 2;
    const size_t halfSize = registeredBytes / 2;
    const size_t tailSize = buf.totalSize - registeredBytes;
    ASSERT_EQ(registeredBytes % sizeof(T), 0u);
    ASSERT_EQ(halfSize % sizeof(T), 0u);

    char* base = reinterpret_cast<char*>(buf.vaBase);
    void* sendBuf = base;
    void* recvBuf = base + halfSize;
    ASSERT_MPI_EQ(hipSuccess,
                  hipMemset(base + registeredBytes, kSentinel, tailSize));

    void* regHandle = nullptr;
    ASSERT_MPI_EQ(
        ncclSuccess,
        ncclCommRegister(getActiveCommunicator(), buf.vaBase, registeredBytes,
                         &regHandle));
    auto regCleanup = makeScopeGuard([&]() {
        if (regHandle)
            HIP_EXPECT(ncclCommDeregister(getActiveCommunicator(), regHandle));
    });
    ASSERT_MPI_NE(regHandle, nullptr);

    int rank = 0;
    int nRanks = 0;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);
    const size_t count = halfSize / sizeof(T);
    initSendBuffer<T>(sendBuf, count, rank);

    ASSERT_MPI_EQ(
        ncclSuccess,
        ncclAllReduce(sendBuf, recvBuf, count, getNcclDataType<T>(), ncclSum,
                      getActiveCommunicator(), getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
    ASSERT_TRUE(verifyAllReduceResult<T>(recvBuf, count, nRanks));

    std::vector<uint8_t> tail(tailSize);
    ASSERT_EQ(hipSuccess,
              hipMemcpy(tail.data(), base + registeredBytes, tailSize,
                        hipMemcpyDeviceToHost));
    EXPECT_TRUE(std::all_of(tail.begin(), tail.end(),
                            [](uint8_t byte) { return byte == kSentinel; }))
        << "NET transfer wrote beyond the clipped registration range";

    struct ncclReg* reg = nullptr;
    ncclRegFind(reinterpret_cast<struct ncclComm*>(getActiveCommunicator()), buf.vaBase, registeredBytes, &reg);
    ASSERT_NE(reg, nullptr) << "ncclCommRegister did not publish a cache entry for the clipped range";
    const bool netDone = (reg->state & NET_REG_COMPLETE) != 0;
    {
        const std::string why = mpiCoordinatedSkipReason(
            !MPIHelpers::anyRankTrue(netDone),
            "NET path not taken on any rank (no inter-node NIC MR)");
        if (!why.empty()) GTEST_SKIP() << why;
    }
    if (netDone) {
        ASSERT_NE(reg->netHandleHead, nullptr);
        ASSERT_EQ(reg->rcclNet.nSegments, kMappedSegments)
            << "NET proxy must enumerate three physical segments for a 2.5-segment clip";
    }
}

/**
 * @brief Register once, AllReduce twice - exercises the registration reuse fast path.
 *
 * After the first collective on a registered multi-segment buffer, a subsequent
 * collective on the same buffer must hit the cache entry and skip re-registration.
 *
 * The fast path is transport-dependent: intra-node ranks reuse via the P2P/IPC
 * cache (p2p.cc, "IPC reuse buffer"), while the inter-node leg reuses the NIC
 * MR via the NET cache (net.cc, "NET reuse buffer"). The topology decides which
 * one fires, so this test asserts on "either IPC or NET" rather than assuming a
 * transport:
 *   - the first call performs an initial registration (IPC or NET)
 *   - a subsequent call hits the cached entry          (IPC or NET reuse)
 */
 TEST_F(UBR_MultiSegment, Generic_Reuse)
 {
     if (!validateTestPrerequisites(
             /*min_processes=*/2, /*max_processes=*/kNoProcessLimit,
             /*require_power_of_two=*/kNoPowerOfTwoRequired,
             /*min_nodes=*/1, /*max_nodes=*/1)) {
         GTEST_SKIP() << "Requires 2+ ranks on a single node";
     }
     ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
 
     ASSERT_TRUE(isUBREnabled()) << "NCCL_LOCAL_REGISTER must be set to 1";
     ASSERT_TRUE(isCuMemEnabled())
         << "NCCL_CUMEM_ENABLE must be set to 1 (gates the multi-segment IPC branch in p2p.cc:1071)";
     ASSERT_TRUE(isMultiSegmentRegisterEnabled())
         << "NCCL_MULTI_SEGMENT_REGISTER must not be 0 (gates the multi-segment IPC branch in p2p.cc:1071)";
     ASSERT_TRUE(isPerRankLoggingEnabled()) << "RCCL_MPI_LOG_ALL_RANKS must be set to 1";
 
     int dev = 0;
     ASSERT_MPI_EQ(hipSuccess, hipGetDevice(&dev));
 
     constexpr size_t kRequestedSegmentSize = 128 * 1024 * 1024;
     constexpr int    kNumSegments          = 4;
 
     MultiSegmentBuffer buf;
     ASSERT_NO_FATAL_FAILURE(createMultiSegmentBuffer(dev, kRequestedSegmentSize, kNumSegments, buf));
     auto vmmCleanup = makeScopeGuard([&]() { releaseMultiSegmentBuffer(buf); });
     {
         const std::string why = skipUnlessAllRanksAllocated(buf.totalSize != 0,
             "Raw VMM (hipMemCreate / Reserve / Map) not supported on this runtime");
         if (!why.empty()) GTEST_SKIP() << why;
     }
 
     const size_t halfSize = buf.totalSize / 2;
     ASSERT_EQ(halfSize % sizeof(T), 0u);
     char*  base    = reinterpret_cast<char*>(buf.vaBase);
     void*  sendBuf = base;
     void*  recvBuf = base + halfSize;
     size_t count   = halfSize / sizeof(T);
 
     void* regHandle = nullptr;
     ASSERT_MPI_EQ(ncclSuccess, ncclCommRegister(getActiveCommunicator(), buf.vaBase, buf.totalSize, &regHandle));
     auto regCleanup = makeScopeGuard([&]() {
         if (regHandle) HIP_EXPECT(ncclCommDeregister(getActiveCommunicator(), regHandle));
     });
     ASSERT_MPI_NE(regHandle, nullptr);
 
     int rank   = 0;
     int nRanks = 0;
     ncclCommUserRank(getActiveCommunicator(), &rank);
     ncclCommCount(getActiveCommunicator(), &nRanks);
 
     initSendBuffer<T>(sendBuf, count, rank);
 
     ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(sendBuf, recvBuf, count,getNcclDataType<T>(), ncclSum, getActiveCommunicator(), getActiveStream()));
     ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
     ASSERT_TRUE(verifyAllReduceResult<T>(recvBuf, count, nRanks));
 
     ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, halfSize));
 
     ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(sendBuf, recvBuf, count, getNcclDataType<T>(), ncclSum,getActiveCommunicator(), getActiveStream()));
     ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
     ASSERT_TRUE(verifyAllReduceResult<T>(recvBuf, count, nRanks));
 
     REGLogChecker checker = getLogChecker();
     TEST_INFO("RegisterOnceUseTwice: %s (log size: %zu bytes)",
               checker.getSummary().c_str(), checker.getContentLength());
    ASSERT_TRUE(checker.hasIPCRegistration() || checker.hasNETRegistration())
        << "Expected an initial registration log entry (IPC or NET) from the first AllReduce";
    ASSERT_TRUE(checker.hasIPCReuse() || checker.hasNETReuse())
        << "Expected a reuse log entry (IPC or NET) from the second AllReduce - "
           "the registration cache (p2p.cc / net.cc) was not hit";
 }

/**
 * @brief DeepEP ElasticSymmetricMemory constructor pattern
 *        (DeepEP/csrc/kernels/backend/symmetric.hpp and nccl.cu).
 *
 * DeepEP reserves one 2 MiB-aligned VA range, maps an independently-sized GPU
 * allocation at the front and CPU allocation at the back, registers the full
 * range with NCCL_WIN_STRICT_ORDERING, then resolves its local LSA pointer.
 * Unlike Symmetric_Elastic_Lsa, this intentionally uses two non-uniform
 * segments and the same window flags/API sequence as DeepEP.
 */
TEST_F(UBR_MultiSegment, DeepEP_ElasticWindowRegistration)
{
    if (!validateTestPrerequisites(/*min_processes=*/2)) {
        GTEST_SKIP() << "Requires 2+ ranks";
    }

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ASSERT_TRUE(isCuMemEnabled()) << "NCCL_CUMEM_ENABLE must be set to 1";
    ASSERT_TRUE(isWinEnabled()) << "NCCL_WIN_ENABLE must not be set to 0";
    ASSERT_TRUE(isElasticBufferRegisterEnabled())
        << "NCCL_ELASTIC_BUFFER_REGISTER must not be 0 for DeepEP elastic memory";
    ASSERT_TRUE(isPerRankLoggingEnabled()) << "RCCL_MPI_LOG_ALL_RANKS must be set to 1";

    int dev = 0;
    ASSERT_MPI_EQ(hipSuccess, hipGetDevice(&dev));

    // Mirrors DeepEP's [[[workspace] GPU buffer] CPU storage] geometry. The
    // unequal lengths ensure this is not reduced to the uniform test pattern.
    constexpr size_t kGpuBytes = 6 * 1024 * 1024;
    constexpr size_t kCpuBytes = 2 * 1024 * 1024;

    MultiSegmentBuffer buf;
    ASSERT_NO_FATAL_FAILURE(createDeepEpElasticBuffer(dev, kGpuBytes, kCpuBytes, buf));
    auto vmmCleanup = makeScopeGuard([&]() { releaseMultiSegmentBuffer(buf); });
    {
        const std::string why = skipUnlessAllRanksAllocated(buf.totalSize != 0,
            "DeepEP-style GPU+CPU VMM allocation unavailable on this runtime");
        if (!why.empty()) GTEST_SKIP() << why;
    }

    ncclWindow_t win = nullptr;
    ncclResult_t result = ncclCommWindowRegister(
        getActiveCommunicator(), buf.vaBase, buf.totalSize, &win,
        NCCL_WIN_STRICT_ORDERING);
    ASSERT_MPI_EQ(ncclSuccess, result);
    auto winCleanup = makeScopeGuard([&]() {
        if (win) HIP_EXPECT(ncclCommWindowDeregister(getActiveCommunicator(), win));
    });
    ASSERT_MPI_NE(win, nullptr);

    MPI_Comm localComm = MPI_COMM_NULL;
    ASSERT_MPI_EQ(MPI_SUCCESS,
                  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                      MPI_INFO_NULL, &localComm));
    int localRank = 0;
    ASSERT_MPI_EQ(MPI_SUCCESS, MPI_Comm_rank(localComm, &localRank));
    ASSERT_MPI_EQ(MPI_SUCCESS, MPI_Comm_free(&localComm));

    void* mappedWindow = nullptr;
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclGetLsaDevicePointer(win, 0, localRank, &mappedWindow));
    ASSERT_MPI_NE(mappedWindow, nullptr);

    REGLogChecker checker = getLogChecker();
    TEST_INFO("DeepEP_ElasticWindowRegistration: %s (log size: %zu bytes)",
              checker.getSummary().c_str(), checker.getContentLength());
    ASSERT_TRUE(checker.hasNumSegments(2))
        << "Expected two segments for DeepEP [GPU][CPU] elastic window";
}

/**
 * @brief DeepEP HybridElasticSymmetricMemory positive registration path.
 *
 * Reproduces [GPU][CPU local-rank 0]...[CPU local-rank 3] on each rank. Every
 * local process imports the same ordered CPU handles before registering the
 * complete range with NCCL_WIN_STRICT_ORDERING. Requires
 * NCCL_ELASTIC_BUFFER_REGISTER=1 and NCCL_SYM_REUSE_SYSMEM_HANDLES=1 so LSA
 * reuses local host handles and GIN registers each physical segment independently.
 */
TEST_F(UBR_MultiSegment, DeepEP_HybridWindowRegistrationAndHandleReuse)
{
    runHybridWindowRegistration(/*singleNode=*/false, /*reject=*/false);
}

TEST_F(UBR_MultiSegment, DeepEP_HybridSingleNodeRegistrationAndHandleReuse)
{
    runHybridWindowRegistration(/*singleNode=*/true, /*reject=*/false);
}

/**
 * @brief Negative hybrid elastic-registration gate.
 *
 * The same imported-handle layout must be rejected cleanly when elastic buffer
 * registration is disabled. Run in a separate process with
 * NCCL_ELASTIC_BUFFER_REGISTER=0.
 */
TEST_F(UBR_MultiSegment, DeepEP_HybridElasticRegistrationDisabled)
{
    runHybridWindowRegistration(/*singleNode=*/false, /*reject=*/true);
}

TEST_F(UBR_MultiSegment, DeepEP_HybridSingleNodeElasticRegistrationDisabled)
{
    runHybridWindowRegistration(/*singleNode=*/true, /*reject=*/true);
}

 /**
  * @brief Elastic buffer registration gating: a host-backed symmetric window must be rejected when
  *        NCCL_ELASTIC_BUFFER_REGISTER=0.
  *
  * Only meaningful when elastic registration is disabled; otherwise SKIPs. The
  * rejection happens in ncclCommWindowRegister before any collective bootstrap,
  * so all ranks fail symmetrically.
  */
TEST_F(UBR_MultiSegment, Symmetric_Elastic_Gating)
{
    if (!validateTestPrerequisites(
            /*min_processes=*/2, /*max_processes=*/kNoProcessLimit,
            /*require_power_of_two=*/kNoPowerOfTwoRequired,
            /*min_nodes=*/1, /*max_nodes=*/1)) {
        GTEST_SKIP() << "Requires 2+ ranks on a single node";
    }
    if (isElasticBufferRegisterEnabled()) {
         GTEST_SKIP() << "Run with NCCL_ELASTIC_BUFFER_REGISTER=0 to exercise the rejection path";
     }
 
     ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
 
     ASSERT_TRUE(isCuMemEnabled()) << "NCCL_CUMEM_ENABLE must be set to 1";
     ASSERT_TRUE(isWinEnabled()) << "NCCL_WIN_ENABLE must not be set to 0";
 
     int dev = 0;
     ASSERT_MPI_EQ(hipSuccess, hipGetDevice(&dev));
 
     constexpr size_t kSegmentSize     = 4 * 1024 * 1024;
     constexpr int    kNumSegments     = 2;
     constexpr int    kNumHostSegments = 1;
 
     MultiSegmentBuffer buf;
     ASSERT_NO_FATAL_FAILURE(
         createMixedMultiSegmentBuffer(dev, kSegmentSize, kNumSegments, kNumHostSegments, buf));
     auto vmmCleanup = makeScopeGuard([&]() { releaseMultiSegmentBuffer(buf); });
     {
         const std::string why = skipUnlessAllRanksAllocated(buf.totalSize != 0,
             "Host VMM (hipMemCreate with hipMemLocationTypeHost) not supported on this runtime");
         if (!why.empty()) GTEST_SKIP() << why;
     }
 
     SCOPED_TRACE("Host-backed symmetric window registration must be rejected "
                  "when NCCL_ELASTIC_BUFFER_REGISTER=0");
    ncclWindow_t win = nullptr;
    ncclResult_t rc  = ncclCommWindowRegister(getActiveCommunicator(), buf.vaBase, buf.totalSize, &win, NCCL_WIN_COLL_SYMMETRIC);
    if (win) HIP_EXPECT(ncclCommWindowDeregister(getActiveCommunicator(), win));
    ASSERT_MPI_NE(rc, ncclSuccess);
}

/**
 * @brief Graph-captured AllReduce whose graph registration reuses a wider record.
 *
 * An 8-segment graph record exists before capture. Each half spans four
 * segments, so ncclRegister reuses that record and NET registers all eight.
 * The NET segment count must cover the reused record, not the collective's half.
 */
TEST_F(UBR_MultiSegment, GenericGraph)
{
    if (!validateTestPrerequisites(
            /*min_processes=*/2, /*max_processes=*/kNoProcessLimit,
            /*require_power_of_two=*/kNoPowerOfTwoRequired,
            /*min_nodes=*/2, /*max_nodes=*/kNoNodeLimit)) {
        GTEST_SKIP() << "Requires 2+ ranks across at least 2 nodes";
    }
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ASSERT_TRUE(isGraphRegisterEnabled()) << "NCCL_GRAPH_REGISTER must be set to 1";
    ASSERT_TRUE(isCuMemEnabled()) << "NCCL_CUMEM_ENABLE must be set to 1";
    ASSERT_TRUE(isMultiSegmentRegisterEnabled()) << "NCCL_MULTI_SEGMENT_REGISTER must be set to 1";

    int dev = 0;
    ASSERT_MPI_EQ(hipSuccess, hipGetDevice(&dev));

    constexpr size_t kSegmentSize = 32 * 1024 * 1024;
    constexpr int    kNumSegments = 8;

    int rank   = 0;
    int nRanks = 0;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    MultiSegmentBuffer buf;
    ASSERT_NO_FATAL_FAILURE(createMultiSegmentBuffer(dev, kSegmentSize, kNumSegments, buf));
    auto vmmCleanup = makeScopeGuard([&]() { releaseMultiSegmentBuffer(buf); });
    {
        const std::string why = skipUnlessAllRanksAllocated(buf.totalSize != 0,
            "Raw VMM (hipMemCreate / Reserve / Map) not supported on this runtime");
        if (!why.empty()) GTEST_SKIP() << why;
    }

    const size_t halfSize = buf.totalSize / 2;
    ASSERT_EQ(halfSize % sizeof(T), 0u);
    char*  base    = reinterpret_cast<char*>(buf.vaBase);
    void*  sendBuf = base;
    void*  recvBuf = base + halfSize;
    size_t count   = halfSize / sizeof(T);

    auto* comm = reinterpret_cast<struct ncclComm*>(getActiveCommunicator());
    struct ncclReg* wide = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGraphRegister(comm, buf.vaBase, buf.totalSize, (void**)&wide));
    auto wideCleanup = makeScopeGuard([&]() {
        if (wide) EXPECT_EQ(ncclSuccess, ncclCommGraphDeregister(comm, wide));
    });
    ASSERT_MPI_NE(wide, nullptr);

    initSendBuffer<T>(sendBuf, count, rank);

    hipGraph_t     graph     = nullptr;
    hipGraphExec_t graphExec = nullptr;
    auto graphCleanup = makeScopeGuard([&]() {
        if (graphExec) (void)hipGraphExecDestroy(graphExec);
        if (graph) (void)hipGraphDestroy(graph);
    });
    ASSERT_MPI_EQ(hipSuccess, hipStreamBeginCapture(getActiveStream(), hipStreamCaptureModeThreadLocal));
    ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(sendBuf, recvBuf, count, getNcclDataType<T>(), ncclSum,
                                             getActiveCommunicator(), getActiveStream()));
    ASSERT_MPI_EQ(hipSuccess, hipStreamEndCapture(getActiveStream(), &graph));
    ASSERT_MPI_EQ(hipSuccess, hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    ASSERT_MPI_EQ(hipSuccess, hipGraphLaunch(graphExec, getActiveStream()));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    ASSERT_MPI_TRUE(verifyAllReduceResult<T>(recvBuf, count, nRanks));

    // Gate on NET_REG_COMPLETE, not allPeers: a graph path that never commits must fail here.
    const bool netRegistered = (wide->state & NET_REG_COMPLETE) != 0;
    {
        const std::string why = mpiCoordinatedSkipReason(
            !MPIHelpers::anyRankTrue(netRegistered),
            "Graph NET registration did not happen on any rank");
        if (!why.empty()) GTEST_SKIP() << why;
    }
    if (netRegistered) {
        ASSERT_TRUE(wide->rcclNet.allPeers) << "Graph NET registration did not record all-peer completion";
        ASSERT_EQ(wide->rcclNet.nSegments, kNumSegments)
            << "Graph NET registration counted the collective's half, not the reused 8-segment record";
    }
}

#endif // MPI_TESTS_ENABLED
