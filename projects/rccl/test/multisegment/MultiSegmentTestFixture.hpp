/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef RCCL_TEST_MULTISEGMENT_TEST_FIXTURE_HPP
#define RCCL_TEST_MULTISEGMENT_TEST_FIXTURE_HPP

#ifdef MPI_TESTS_ENABLED

#include "RegistrationTestSupport.hpp"
#include "../ce/CeTestHelpers.hpp"

#include <cstring>

// ============================================================================
// Multi-Segment Registration Tests
// ============================================================================

/**
 * @brief Tests for buffer registration when the user buffer spans multiple
 *        underlying physical allocations ("multi-segment" buffers).
 *
 * A multi-segment buffer is built by mapping N separate physical handles
 * (hipMemCreate) contiguously into a single reserved virtual address range
 * (hipMemAddressReserve + hipMemMap). cuMemGetAddressRange() on the head of
 * such a buffer returns only the first segment, so the registration path
 * detects the cross-boundary case and walks every segment when both 
 * ncclCuMemEnable() and NCCL_MULTI_SEGMENT_REGISTER (default 1) are true.
 *
 */
class UBR_MultiSegment : public RegistrationTestBase
{
protected:
    using T = RegTestConfig::DefaultType;

    // Rank-local GTEST_SKIP after alloc failure hangs peers. Callers must
    // GTEST_SKIP from the TEST body with the returned reason.
    std::string skipUnlessAllRanksAllocated(bool allocated, const char* msg)
    {
        return mpiCoordinatedSkipReason(!allocated, msg);
    }

    bool ceAllReduceSelected(void* sendBuf, void* recvBuf, size_t count)
    {
        int algo = 0, proto = 0, nCh = 0;
        ncclResult_t res = rcclGetCollImplInfo(
            getActiveCommunicator(), ncclFuncAllReduce, count, getNcclDataType<T>(),
            ncclSum, sendBuf, recvBuf, /*graphCapturing=*/0, &algo, &proto, &nCh);
        return res == ncclSuccess && (algo == RCCL_CE_REGISTERED || algo == RCCL_CE_2SHOT);
    }

    struct MultiSegmentBuffer
    {
        hipDeviceptr_t                                vaBase      = 0;
        size_t                                        segmentSize = 0;
        size_t                                        totalSize   = 0;
        std::vector<hipMemGenericAllocationHandle_t>  handles;
    };

    // HIP_TEST_CHECK, not HIP_CHECK: HIP_CHECK is ASSERT_EQ and aborts the TEST,
    // so a failing rank never reaches mpiCoordinatedSkipReason. HIP_TEST_CHECK
    // returns ncclUnhandledCudaError from this helper and leaves totalSize == 0.
    ncclResult_t createMultiSegmentBuffer(int dev,
                                          size_t requestedSegmentSize,
                                          int numSegments,
                                          MultiSegmentBuffer& buf)
    {
        buf = MultiSegmentBuffer{};
        if (numSegments <= 0) return ncclInvalidArgument;

        hipMemAllocationProp prop = {};
        prop.type                = hipMemAllocationTypePinned;
        prop.location.type       = hipMemLocationTypeDevice;
        prop.location.id         = dev;
        prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;

        size_t granularity = 0;
        HIP_TEST_CHECK(hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));

        const size_t segSize   = ((requestedSegmentSize + granularity - 1) / granularity) * granularity;
        const size_t totalSize = segSize * static_cast<size_t>(numSegments);

        hipDeviceptr_t vaBase = 0;
        HIP_TEST_CHECK(hipMemAddressReserve(&vaBase, totalSize, granularity, 0, 0));

        std::vector<hipMemGenericAllocationHandle_t> handles(static_cast<size_t>(numSegments), 0);
        char* vaBaseBytes = static_cast<char*>(vaBase);
        int mapped = 0;
        auto releasePartial = makeScopeGuard([&]() {
            if (buf.totalSize != 0) return;
            for (int i = 0; i < mapped; i++) {
                HIP_EXPECT(hipMemUnmap(vaBaseBytes + static_cast<size_t>(i) * segSize, segSize));
            }
            for (auto h : handles) {
                if (h != 0) HIP_EXPECT(hipMemRelease(h));
            }
            if (vaBase != 0) HIP_EXPECT(hipMemAddressFree(vaBase, totalSize));
        });

        for (int i = 0; i < numSegments; i++) {
            HIP_TEST_CHECK(hipMemCreate(&handles[i], segSize, &prop, 0));
            HIP_TEST_CHECK(hipMemMap(vaBaseBytes + static_cast<size_t>(i) * segSize, segSize, 0, handles[i], 0));
            mapped++;
        }

        hipMemAccessDesc accessDesc = {};
        accessDesc.location.type    = hipMemLocationTypeDevice;
        accessDesc.location.id      = dev;
        accessDesc.flags            = hipMemAccessFlagsProtReadWrite;
        HIP_TEST_CHECK(hipMemSetAccess(vaBase, totalSize, &accessDesc, 1));

        buf.vaBase      = vaBase;
        buf.segmentSize = segSize;
        buf.totalSize   = totalSize;
        buf.handles     = std::move(handles);
        return ncclSuccess;
    }

    void releaseMultiSegmentBuffer(MultiSegmentBuffer& buf)
    {
        if (buf.totalSize == 0) return;
        HIP_EXPECT(hipMemUnmap(buf.vaBase, buf.totalSize));
        for (auto h : buf.handles) {
            if (h != 0) HIP_EXPECT(hipMemRelease(h));
        }
        HIP_EXPECT(hipMemAddressFree(buf.vaBase, buf.totalSize));
        buf.vaBase      = 0;
        buf.segmentSize = 0;
        buf.totalSize   = 0;
        buf.handles.clear();
    }

    // 4-segment window, recv at +1 segment. Caller returns if skipped.
    // requireCeEnv is the CE receive-offset BEFORE control. Symmetric_Lsa leaves
    // it false: CTA policy ZERO would select CE ahead of the symmetric kernel.
    void prepareSymmetricLsaRecvOffset(MultiSegmentBuffer& buf, ncclWindow_t* win, int* rank, int* nRanks,
                                       void** sendBuf, void** recvBuf, size_t* count, size_t* totalBytes,
                                       bool requireCeEnv = false)
    {
        if (!validateTestPrerequisites(
                /*min_processes=*/2, /*max_processes=*/kNoProcessLimit,
                /*require_power_of_two=*/kNoPowerOfTwoRequired,
                /*min_nodes=*/1, /*max_nodes=*/1)) {
            GTEST_SKIP() << "Requires 2+ ranks on exactly one node";
            return;
        }
        if (requireCeEnv) {
            if (const char* why = ceRecvOffsetEnvSkipReason()) {
                GTEST_SKIP() << why;
                return;
            }
        }
        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ASSERT_TRUE(isCuMemEnabled()) << "NCCL_CUMEM_ENABLE must be set to 1";
        ASSERT_TRUE(isWinEnabled()) << "NCCL_WIN_ENABLE must not be set to 0";

        int dev = 0;
        ASSERT_MPI_EQ(hipSuccess, hipGetDevice(&dev));

        constexpr size_t kSegmentSize = 32 * 1024 * 1024;
        constexpr int kNumSegments = 4;
        ASSERT_NO_FATAL_FAILURE(createMultiSegmentBuffer(dev, kSegmentSize, kNumSegments, buf));
        {
            const std::string why = skipUnlessAllRanksAllocated(buf.totalSize != 0,
                "Raw VMM (hipMemCreate / Reserve / Map) not supported on this runtime");
            if (!why.empty()) {
                GTEST_SKIP() << why;
                return;
            }
        }

        ncclCommUserRank(getActiveCommunicator(), rank);
        ncclCommCount(getActiveCommunicator(), nRanks);

        const size_t recvOffset = buf.segmentSize;
        *totalBytes = buf.segmentSize;
        ASSERT_EQ(*totalBytes % sizeof(T), 0u);
        ASSERT_NE(recvOffset, buf.totalSize / 2);
        ASSERT_NE(recvOffset, buf.totalSize - *totalBytes);

        char* base = reinterpret_cast<char*>(buf.vaBase);
        *sendBuf = base;
        *recvBuf = base + recvOffset;
        *count = *totalBytes / sizeof(T);
        if (*nRanks <= 0 || *count % static_cast<size_t>(*nRanks) != 0) {
            GTEST_SKIP() << "count must be divisible by nRanks";
            return;
        }

        *win = nullptr;
        ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(
            getActiveCommunicator(), buf.vaBase, buf.totalSize, win, NCCL_WIN_COLL_SYMMETRIC));
        ASSERT_MPI_NE(*win, nullptr);
    }

    /**
     * @brief Like createMultiSegmentBuffer, but backs the trailing
     *        `numHostSegments` segments with host memory
     *        (hipMemLocationTypeHost) rather than device memory, producing a
     *        mixed device/host "elastic" buffer.
     *
     * HIP/CLR has no host-NUMA VMM type and only accepts a plain Host location
     * (id must be 0). Host VMM follows NCCL_CUMEM_HOST_VERSION_SUPPORTED
     * (native 7.12 or the 7.0.2.x backport); if it is unsupported,
     * buf.totalSize is left 0 so the caller can GTEST_SKIP() instead of failing.
     */
    void createMixedMultiSegmentBuffer(int dev,
                                       size_t requestedSegmentSize,
                                       int numSegments,
                                       int numHostSegments,
                                       MultiSegmentBuffer& buf)
    {
        buf = MultiSegmentBuffer{};
#if NCCL_CUMEM_HOST_VERSION_SUPPORTED(HIP_VERSION)
        ASSERT_GE(numSegments, 1);
        ASSERT_GE(numHostSegments, 0);
        ASSERT_LE(numHostSegments, numSegments);

        hipMemAllocationProp devProp = {};
        devProp.type                = hipMemAllocationTypePinned;
        devProp.location.type       = hipMemLocationTypeDevice;
        devProp.location.id         = dev;
        devProp.requestedHandleType = hipMemHandleTypePosixFileDescriptor;

        hipMemAllocationProp hostProp = {};
        hostProp.type                = hipMemAllocationTypePinned;
        hostProp.location.type       = hipMemLocationTypeHost;
        hostProp.location.id         = 0;
        hostProp.requestedHandleType = hipMemHandleTypePosixFileDescriptor;

        size_t devGran = 0;
        size_t hostGran = 0;
        HIP_CHECK(hipMemGetAllocationGranularity(&devGran, &devProp, hipMemAllocationGranularityMinimum));
        if (numHostSegments > 0 &&
            hipMemGetAllocationGranularity(&hostGran, &hostProp, hipMemAllocationGranularityMinimum) != hipSuccess) {
            return;
        }
        size_t gran = devGran;
        if (hostGran > gran) gran = hostGran;

        const size_t segSize   = ((requestedSegmentSize + gran - 1) / gran) * gran;
        const size_t totalSize = segSize * numSegments;
        const int    numDevSegments = numSegments - numHostSegments;

        hipDeviceptr_t vaBase = 0;
        HIP_CHECK(hipMemAddressReserve(&vaBase, totalSize, gran, 0, 0));
        char* vaBaseBytes = static_cast<char*>(vaBase);

        std::vector<hipMemGenericAllocationHandle_t> handles(numSegments, 0);
        bool hostUnsupported = false;
        int  mapped          = 0;
        for (int i = 0; i < numSegments; i++) {
            const bool isHost = (i >= numDevSegments);
            hipMemAllocationProp& prop = isHost ? hostProp : devProp;
            hipError_t err = hipMemCreate(&handles[i], segSize, &prop, 0);
            if (isHost && err != hipSuccess) {
                hostUnsupported = true;
                break;
            }
            HIP_CHECK(err);
            HIP_CHECK(hipMemMap(vaBaseBytes + i * segSize, segSize, 0, handles[i], 0));
            mapped++;
        }

        if (hostUnsupported) {
            for (int i = 0; i < mapped; i++) {
                HIP_EXPECT(hipMemUnmap(vaBaseBytes + i * segSize, segSize));
            }
            for (auto h : handles) {
                if (h != 0) HIP_EXPECT(hipMemRelease(h));
            }
            HIP_EXPECT(hipMemAddressFree(vaBase, totalSize));
            return;
        }

        hipMemAccessDesc accessDesc = {};
        accessDesc.location.type    = hipMemLocationTypeDevice;
        accessDesc.location.id      = dev;
        accessDesc.flags            = hipMemAccessFlagsProtReadWrite;
        HIP_CHECK(hipMemSetAccess(vaBase, totalSize, &accessDesc, 1));

        buf.vaBase      = vaBase;
        buf.segmentSize = segSize;
        buf.totalSize   = totalSize;
        buf.handles     = std::move(handles);
#else
        // Host VMM is compiled only inside NCCL_CUMEM_HOST_VERSION_SUPPORTED.
        // Leave buf.totalSize == 0 so callers GTEST_SKIP().
        (void)dev;
        (void)requestedSegmentSize;
        (void)numSegments;
        (void)numHostSegments;
#endif
    }

    /**
     * @brief Reproduce DeepEP ElasticSymmetricMemory exactly: a 2 MiB-aligned
     *        contiguous VA range containing one GPU segment followed by one
     *        independently-sized CPU segment.
     */
    void createDeepEpElasticBuffer(int dev,
                                   size_t numGpuBytes,
                                   size_t numCpuBytes,
                                   MultiSegmentBuffer& buf)
    {
        constexpr size_t kDeepEpAlignment = 2 * 1024 * 1024;
        buf = MultiSegmentBuffer{};
        ASSERT_GT(numGpuBytes, 0u);
        ASSERT_GT(numCpuBytes, 0u);
        ASSERT_EQ(numGpuBytes % kDeepEpAlignment, 0u);
        ASSERT_EQ(numCpuBytes % kDeepEpAlignment, 0u);

        RCCLHybridVmmTests::DeepEpElasticRange range;
        if (!RCCLHybridVmmTests::AllocDeepEpElasticRange(dev, numGpuBytes, numCpuBytes, &range))
            return;
        buf.vaBase      = range.base;
        buf.segmentSize = 0; // segments intentionally have different sizes
        buf.totalSize   = range.totalSize;
        buf.handles     = std::move(range.handles);
    }

    bool createHybridVmmBuffer(size_t gpuBytes, size_t localCpuBytes,
                               RCCLHybridVmmTests::HybridVmmBuffer& buf,
                               std::string& reason,
                               int expectedLocalRanks = 4)
    {
        int dev = 0;
        bool supported = hipGetDevice(&dev) == hipSuccess &&
            RCCLHybridVmmTests::CheckHybridVmmRuntimeSupport(dev, &reason);
        if (!MPIHelpers::allRanksTrue(supported)) {
            if (reason.empty())
                reason = "hybrid VMM runtime support is unavailable on another rank";
            return false;
        }
        bool allocated = RCCLHybridVmmTests::AllocHybridForLocalRanks(
            dev, gpuBytes, localCpuBytes, expectedLocalRanks, &buf, &reason);
        if (!MPIHelpers::allRanksTrue(allocated)) {
            RCCLHybridVmmTests::FreeHybridVmm(buf);
            if (reason.empty())
                reason = "hybrid VMM allocation failed on another rank";
            return false;
        }
        return true;
    }

    enum class ProxyPath { Symmetric, Legacy };

    // Single-node LSA uses the direct kernels. Multi-node AllReduce has no GIN
    // kernel. Multi-node AllGather also needs LSA multimem; ReduceScatter does not.
    bool symmetricKernelExpected(const char* collective)
    {
        ncclComm* c = getActiveCommunicator();
        const bool runtime = c->symmetricSupport && c->isAllDirectNvlink;
        const bool ginKernels =
            MPIHelpers::getEnvParam<int>("NCCL_SYM_GIN_KERNELS_ENABLE", 1) != 0;
        if (c->devrState.lsaSize >= c->nRanks)
            return runtime;
        if (std::strcmp(collective, "AllReduce") == 0)
            return false;
        if (std::strcmp(collective, "AllGather") == 0)
            return runtime && ginKernels && c->symkState.hasLsaMultimem;
        if (std::strcmp(collective, "ReduceScatter") == 0)
            return runtime && ginKernels;
        return false;
    }

    void expectSchedulerPath(const char* collective, bool expectSymmetric)
    {
        if (!isPerRankLoggingEnabled())
            return;
        int rank = 0;
        ncclCommUserRank(getActiveCommunicator(), &rank);
        if (rank != 0)
            return;
        REGLogChecker checker = getLogChecker();
        const bool sawSymmetric = checker.usedSymmetricCollective(collective);
        const bool sawLegacy = checker.usedLegacyCollective(collective);
        ASSERT_TRUE(sawSymmetric || sawLegacy)
            << collective
            << ": missing scheduler marker; NCCL_DEBUG_SUBSYS must include TUNING";
        EXPECT_EQ(sawSymmetric, expectSymmetric) << collective;
        EXPECT_NE(sawLegacy, expectSymmetric) << collective;
    }

    // Sysmem windows are rejected by symmetric_sched.cc, so elastic always
    // expects the legacy path even when a GPU kernel would otherwise be selected.
    // singleNode covers the LSA kernels and elastic collectives. The multi-node
    // callers also require a GIN proxy leg.
    void runLsaGinWindowCollective(const char* collective, bool elastic, ProxyPath path,
                                   bool singleNode = false)
    {
        const int nodeCount = MPITestConstants::detectNodeCount();
        if (singleNode) {
            if (!validateTestPrerequisites(
                    /*min_processes=*/2, /*max_processes=*/kNoProcessLimit,
                    /*require_power_of_two=*/kNoPowerOfTwoRequired,
                    /*min_nodes=*/1, /*max_nodes=*/1)) {
                GTEST_SKIP() << "Requires 2+ ranks on exactly one node";
                return;
            }
            // CE outranks the symmetric kernel for every collective it implements.
            // ReduceScatter is not one of them.
            if (!elastic && envCtaPolicyIsZero() && std::strcmp(collective, "ReduceScatter") != 0) {
                GTEST_SKIP() << "NCCL_CTA_POLICY=ZERO selects CE ahead of the symmetric kernel";
                return;
            }
        } else {
            if (!validateTestPrerequisites(2)) {
                GTEST_SKIP() << "Requires 2+ ranks";
                return;
            }
            if (nodeCount < 2) {
                GTEST_SKIP() << "Requires >=2 nodes";
                return;
            }
            if (!isGinEnabled()) {
                GTEST_SKIP() << "Requires GIN proxy (NCCL_GIN_TYPE=2)";
                return;
            }
        }

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ASSERT_TRUE(isCuMemEnabled()) << "NCCL_CUMEM_ENABLE must be set to 1";
        ASSERT_TRUE(isWinEnabled()) << "NCCL_WIN_ENABLE must not be set to 0";
        ASSERT_TRUE(isPerRankLoggingEnabled()) << "RCCL_MPI_LOG_ALL_RANKS must be set to 1";
        if (elastic) {
            ASSERT_TRUE(isElasticBufferRegisterEnabled())
                << "NCCL_ELASTIC_BUFFER_REGISTER must not be 0";
        }

        int dev = 0;
        ASSERT_MPI_EQ(hipSuccess, hipGetDevice(&dev));
        int rank = 0;
        int nRanks = 0;
        ncclCommUserRank(getActiveCommunicator(), &rank);
        ncclCommCount(getActiveCommunicator(), &nRanks);
        if (!singleNode && nRanks / nodeCount < 2) {
            GTEST_SKIP() << "Requires >=2 ranks per node";
            return;
        }

        const bool expectSym = !elastic && symmetricKernelExpected(collective);
        const bool localSkip =
            (path == ProxyPath::Symmetric && !expectSym) ||
            (path == ProxyPath::Legacy && expectSym);
        const char* skipMsg = (path == ProxyPath::Symmetric)
            ? "symmetric kernel is not available for this collective"
            : "symmetric kernel is available; this case covers the legacy fallback";
        {
            const std::string why = mpiCoordinatedSkipReason(localSkip, skipMsg);
            if (!why.empty()) {
                GTEST_SKIP() << why;
                return;
            }
        }

        constexpr size_t kSegmentSize = 4 * 1024 * 1024;
        constexpr int kNumSegments = 4;
        constexpr size_t kGpuBytes = 6 * 1024 * 1024;
        constexpr size_t kCpuBytes = 2 * 1024 * 1024;
        const bool allGather = std::strcmp(collective, "AllGather") == 0;
        const bool reduceScatter = std::strcmp(collective, "ReduceScatter") == 0;
        const bool alltoAll = std::strcmp(collective, "AlltoAll") == 0;

        MultiSegmentBuffer sendSeg;
        MultiSegmentBuffer recvSeg;
        if (!elastic) {
            if (allGather) {
                ASSERT_NO_FATAL_FAILURE(
                    createMultiSegmentBuffer(dev, kSegmentSize, kNumSegments, sendSeg));
                if (sendSeg.totalSize != 0) {
                    ASSERT_NO_FATAL_FAILURE(createMultiSegmentBuffer(
                        dev, sendSeg.segmentSize * static_cast<size_t>(nRanks),
                        kNumSegments, recvSeg));
                }
            } else if (reduceScatter) {
                ASSERT_NO_FATAL_FAILURE(
                    createMultiSegmentBuffer(dev, kSegmentSize, kNumSegments, recvSeg));
                if (recvSeg.totalSize != 0) {
                    ASSERT_NO_FATAL_FAILURE(createMultiSegmentBuffer(
                        dev, recvSeg.segmentSize * static_cast<size_t>(nRanks),
                        kNumSegments, sendSeg));
                }
            } else {
                ASSERT_NO_FATAL_FAILURE(
                    createMultiSegmentBuffer(dev, kSegmentSize, kNumSegments, sendSeg));
                ASSERT_NO_FATAL_FAILURE(
                    createMultiSegmentBuffer(dev, kSegmentSize, kNumSegments, recvSeg));
            }
        } else if (allGather) {
            ASSERT_NO_FATAL_FAILURE(
                createDeepEpElasticBuffer(dev, kGpuBytes, kCpuBytes, sendSeg));
            if (sendSeg.totalSize != 0) {
                const size_t recvBytes = sendSeg.totalSize * static_cast<size_t>(nRanks);
                ASSERT_NO_FATAL_FAILURE(createDeepEpElasticBuffer(
                    dev, recvBytes - kCpuBytes, kCpuBytes, recvSeg));
            }
        } else if (reduceScatter) {
            ASSERT_NO_FATAL_FAILURE(
                createDeepEpElasticBuffer(dev, kGpuBytes, kCpuBytes, recvSeg));
            if (recvSeg.totalSize != 0) {
                const size_t sendBytes = recvSeg.totalSize * static_cast<size_t>(nRanks);
                ASSERT_NO_FATAL_FAILURE(createDeepEpElasticBuffer(
                    dev, sendBytes - kCpuBytes, kCpuBytes, sendSeg));
            }
        } else {
            ASSERT_NO_FATAL_FAILURE(
                createDeepEpElasticBuffer(dev, kGpuBytes, kCpuBytes, sendSeg));
            ASSERT_NO_FATAL_FAILURE(
                createDeepEpElasticBuffer(dev, kGpuBytes, kCpuBytes, recvSeg));
        }

        auto sendCleanup = makeScopeGuard([&]() { releaseMultiSegmentBuffer(sendSeg); });
        auto recvCleanup = makeScopeGuard([&]() { releaseMultiSegmentBuffer(recvSeg); });
        {
            const std::string why = skipUnlessAllRanksAllocated(
                sendSeg.totalSize != 0 && recvSeg.totalSize != 0,
                elastic ? "DeepEP GPU+CPU VMM allocation unavailable"
                        : "Raw VMM allocation unavailable");
            if (!why.empty()) {
                GTEST_SKIP() << why;
                return;
            }
        }

        size_t sendCount = 0;
        size_t recvCount = 0;
        if (allGather) {
            ASSERT_EQ(sendSeg.totalSize % sizeof(T), 0u);
            sendCount = sendSeg.totalSize / sizeof(T);
            recvCount = sendCount * static_cast<size_t>(nRanks);
            ASSERT_GE(recvSeg.totalSize, recvCount * sizeof(T));
        } else if (reduceScatter) {
            ASSERT_EQ(recvSeg.totalSize % sizeof(T), 0u);
            recvCount = recvSeg.totalSize / sizeof(T);
            sendCount = recvCount * static_cast<size_t>(nRanks);
            ASSERT_GE(sendSeg.totalSize, sendCount * sizeof(T));
        } else if (alltoAll) {
            ASSERT_EQ(sendSeg.totalSize % sizeof(T), 0u);
            ASSERT_EQ(recvSeg.totalSize, sendSeg.totalSize);
            if (nRanks <= 0 || sendSeg.totalSize % (sizeof(T) * static_cast<size_t>(nRanks)) != 0) {
                GTEST_SKIP() << "AlltoAll element count is not divisible by nRanks";
                return;
            }
            sendCount = (sendSeg.totalSize / sizeof(T)) / static_cast<size_t>(nRanks);
            recvCount = sendCount;
        } else {
            ASSERT_EQ(sendSeg.totalSize % sizeof(T), 0u);
            ASSERT_EQ(recvSeg.totalSize, sendSeg.totalSize);
            sendCount = sendSeg.totalSize / sizeof(T);
            recvCount = sendCount;
        }

        ncclWindow_t sendWin = nullptr;
        ncclWindow_t recvWin = nullptr;
        ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(
            getActiveCommunicator(), sendSeg.vaBase, sendSeg.totalSize, &sendWin,
            NCCL_WIN_COLL_SYMMETRIC));
        ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(
            getActiveCommunicator(), recvSeg.vaBase, recvSeg.totalSize, &recvWin,
            NCCL_WIN_COLL_SYMMETRIC));
        auto winCleanup = makeScopeGuard([&]() {
            if (sendWin) HIP_EXPECT(ncclCommWindowDeregister(getActiveCommunicator(), sendWin));
            if (recvWin) HIP_EXPECT(ncclCommWindowDeregister(getActiveCommunicator(), recvWin));
        });

        // AlltoAll sends rank+1 in every element, so each receive block matches
        // the AllGather layout.
        const size_t initCount = alltoAll ? sendCount * static_cast<size_t>(nRanks) : sendCount;
        initSendBuffer<T>(reinterpret_cast<void*>(sendSeg.vaBase), initCount, rank);
        ncclComm* comm = getActiveCommunicator();
        hipStream_t stream = getActiveStream();
        const ncclDataType_t dtype = getNcclDataType<T>();
        if (allGather) {
            ASSERT_MPI_EQ(ncclSuccess, ncclAllGather(
                sendSeg.vaBase, recvSeg.vaBase, sendCount, dtype, comm, stream));
        } else if (reduceScatter) {
            ASSERT_MPI_EQ(ncclSuccess, ncclReduceScatter(
                sendSeg.vaBase, recvSeg.vaBase, recvCount, dtype, ncclSum, comm, stream));
        } else if (alltoAll) {
            ASSERT_MPI_EQ(ncclSuccess, ncclAlltoAll(
                sendSeg.vaBase, recvSeg.vaBase, sendCount, dtype, comm, stream));
        } else {
            ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(
                sendSeg.vaBase, recvSeg.vaBase, sendCount, dtype, ncclSum, comm, stream));
        }
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

        if (allGather || alltoAll)
            ASSERT_TRUE(verifyAllGatherResult<T>(recvSeg.vaBase, sendCount, nRanks));
        else if (reduceScatter)
            ASSERT_TRUE(verifyReduceScatterResult<T>(recvSeg.vaBase, recvCount, nRanks));
        else
            ASSERT_TRUE(verifyAllReduceResult<T>(recvSeg.vaBase, sendCount, nRanks));

        REGLogChecker checker = getLogChecker();
        const int expectedSegments = elastic ? 2 : kNumSegments;
        ASSERT_TRUE(checker.hasNumSegments(expectedSegments))
            << "expected numSegments " << expectedSegments;
        expectSchedulerPath(collective, expectSym);
    }

    // Imported host memory forces the normal algorithm. singleNode is four ranks
    // on one node, so LSA reuses the imported CPU handles and there is no GIN leg.
    // AllGather sends from a GPU multi-segment window into the hybrid receive tail.
    // ReduceScatter sends from a larger GPU multi-segment window so the count is
    // not clipped to the GPU prefix. AllReduce uses hybrid windows for both.
    void runHybridWindowCollective(const char* collective, bool singleNode = false)
    {
        const int ranks = singleNode ? 4 : 8;
        if (!validateTestPrerequisites(
                /*min_processes=*/ranks, /*max_processes=*/ranks,
                /*require_power_of_two=*/kNoPowerOfTwoRequired,
                /*min_nodes=*/singleNode ? 1 : 2, /*max_nodes=*/singleNode ? 1 : 2)) {
            GTEST_SKIP() << (singleNode ? "Requires 4 ranks on a single node"
                                        : "Requires 8 ranks across exactly 2 nodes");
            return;
        }
        if (!isSymSysmemHandleReuseEnabled()) {
            GTEST_SKIP() << "Requires NCCL_SYM_REUSE_SYSMEM_HANDLES=1";
            return;
        }
        if (!singleNode && !isGinEnabled()) {
            GTEST_SKIP() << "Requires GIN proxy (NCCL_GIN_TYPE=2)";
            return;
        }

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ASSERT_TRUE(isCuMemEnabled());
        ASSERT_TRUE(isWinEnabled());
        ASSERT_TRUE(isElasticBufferRegisterEnabled());
        ASSERT_TRUE(isPerRankLoggingEnabled());

        int rank = 0;
        int nRanks = 0;
        ncclCommUserRank(getActiveCommunicator(), &rank);
        ncclCommCount(getActiveCommunicator(), &nRanks);
        ASSERT_EQ(nRanks, ranks);

        constexpr size_t kGpuBytes = 8 * 1024 * 1024;
        constexpr size_t kCpuBytes = 2 * 1024 * 1024;
        const bool allGather = std::strcmp(collective, "AllGather") == 0;
        const bool reduceScatter = std::strcmp(collective, "ReduceScatter") == 0;

        RCCLHybridVmmTests::HybridVmmBuffer recvHybrid;
        std::string reason;
        if (!createHybridVmmBuffer(kGpuBytes, kCpuBytes, recvHybrid, reason)) {
            GTEST_SKIP() << "DeepEP hybrid allocation unavailable: " << reason;
            return;
        }
        auto recvCleanup = makeScopeGuard([&]() {
            RCCLHybridVmmTests::FreeHybridVmm(recvHybrid);
        });
        ASSERT_EQ(recvHybrid.localSize, 4);
        ASSERT_LT(recvHybrid.gpuBytes, recvHybrid.totalSize);
        ASSERT_EQ(recvHybrid.totalSize % sizeof(T), 0u);
        ASSERT_EQ(recvHybrid.totalSize % static_cast<size_t>(nRanks), 0u);

        RCCLHybridVmmTests::HybridVmmBuffer sendHybrid;
        MultiSegmentBuffer sendSeg;
        auto sendHybridCleanup = makeScopeGuard([&]() {
            RCCLHybridVmmTests::FreeHybridVmm(sendHybrid);
        });
        auto sendCleanup = makeScopeGuard([&]() { releaseMultiSegmentBuffer(sendSeg); });

        void* sendPtr = nullptr;
        size_t sendBytes = 0;
        if (allGather || reduceScatter) {
            int dev = 0;
            ASSERT_MPI_EQ(hipSuccess, hipGetDevice(&dev));
            if (allGather) {
                sendBytes = recvHybrid.totalSize / static_cast<size_t>(nRanks);
                ASSERT_EQ(sendBytes % 2, 0u);
                ASSERT_NO_FATAL_FAILURE(
                    createMultiSegmentBuffer(dev, sendBytes / 2, 2, sendSeg));
            } else {
                constexpr int kSendSegments = 4;
                const size_t sendTotal = recvHybrid.totalSize * static_cast<size_t>(nRanks);
                ASSERT_EQ(sendTotal % kSendSegments, 0u);
                ASSERT_NO_FATAL_FAILURE(createMultiSegmentBuffer(
                    dev, sendTotal / kSendSegments, kSendSegments, sendSeg));
                sendBytes = sendSeg.totalSize;
            }
            {
                const std::string why = skipUnlessAllRanksAllocated(sendSeg.totalSize != 0,
                    "Raw VMM allocation unavailable");
                if (!why.empty()) {
                    GTEST_SKIP() << why;
                    return;
                }
            }
            if (allGather)
                ASSERT_GE(sendSeg.totalSize, sendBytes);
            sendPtr = reinterpret_cast<void*>(sendSeg.vaBase);
        } else {
            if (!createHybridVmmBuffer(kGpuBytes, kCpuBytes, sendHybrid, reason)) {
                GTEST_SKIP() << "DeepEP hybrid allocation unavailable: " << reason;
                return;
            }
            ASSERT_EQ(sendHybrid.totalSize, recvHybrid.totalSize);
            ASSERT_LT(sendHybrid.gpuBytes, sendHybrid.totalSize);
            sendPtr = sendHybrid.ptr;
            sendBytes = sendHybrid.totalSize;
        }
        launchHybridCollective(collective, sendPtr, sendBytes, recvHybrid, rank, nRanks);
    }

    void launchHybridCollective(const char* collective, void* sendPtr, size_t sendBytes,
                                RCCLHybridVmmTests::HybridVmmBuffer& recvHybrid,
                                int rank, int nRanks)
    {
        const bool allGather = std::strcmp(collective, "AllGather") == 0;
        const bool reduceScatter = std::strcmp(collective, "ReduceScatter") == 0;
        const size_t count = allGather
            ? (recvHybrid.totalSize / static_cast<size_t>(nRanks)) / sizeof(T)
            : recvHybrid.totalSize / sizeof(T);
        ASSERT_EQ(count * static_cast<size_t>(allGather ? nRanks : 1) * sizeof(T),
                  recvHybrid.totalSize);
        const size_t initCount = reduceScatter ? count * static_cast<size_t>(nRanks) : count;
        if (reduceScatter)
            ASSERT_GE(sendBytes, initCount * sizeof(T));
        else if (allGather)
            ASSERT_EQ(sendBytes, count * sizeof(T));
        else
            ASSERT_EQ(sendBytes, recvHybrid.totalSize);

        ncclWindow_t sendWin = nullptr;
        ncclWindow_t recvWin = nullptr;
        ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(
            getActiveCommunicator(), sendPtr, sendBytes, &sendWin, NCCL_WIN_COLL_SYMMETRIC));
        ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(
            getActiveCommunicator(), recvHybrid.ptr, recvHybrid.totalSize, &recvWin,
            NCCL_WIN_COLL_SYMMETRIC));
        auto winCleanup = makeScopeGuard([&]() {
            if (sendWin) HIP_EXPECT(ncclCommWindowDeregister(getActiveCommunicator(), sendWin));
            if (recvWin) HIP_EXPECT(ncclCommWindowDeregister(getActiveCommunicator(), recvWin));
        });

        initSendBuffer<T>(sendPtr, initCount, rank);
        const ncclDataType_t dtype = getNcclDataType<T>();
        ncclComm* comm = getActiveCommunicator();
        hipStream_t stream = getActiveStream();
        if (allGather) {
            ASSERT_MPI_EQ(ncclSuccess, ncclAllGather(
                sendPtr, recvHybrid.ptr, count, dtype, comm, stream));
        } else if (reduceScatter) {
            ASSERT_MPI_EQ(ncclSuccess, ncclReduceScatter(
                sendPtr, recvHybrid.ptr, count, dtype, ncclSum, comm, stream));
        } else {
            ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(
                sendPtr, recvHybrid.ptr, count, dtype, ncclSum, comm, stream));
        }
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));
        if (allGather)
            ASSERT_TRUE(verifyAllGatherResult<T>(recvHybrid.ptr, count, nRanks));
        else if (reduceScatter)
            ASSERT_TRUE(verifyReduceScatterResult<T>(recvHybrid.ptr, count, nRanks));
        else
            ASSERT_TRUE(verifyAllReduceResult<T>(recvHybrid.ptr, count, nRanks));

        REGLogChecker checker = getLogChecker();
        ASSERT_TRUE(checker.hasNumSegments(recvHybrid.localSize + 1));
        ASSERT_TRUE(checker.hasSymSysmemHandleReuse());
        expectSchedulerPath(collective, false);
    }

    // reject is the NCCL_ELASTIC_BUFFER_REGISTER=0 process. Otherwise the window
    // must register and LSA must reuse the imported host handles.
    void runHybridWindowRegistration(bool singleNode, bool reject)
    {
        const int ranks = singleNode ? 4 : 8;
        if (!validateTestPrerequisites(
                /*min_processes=*/ranks, /*max_processes=*/ranks,
                /*require_power_of_two=*/kNoPowerOfTwoRequired,
                /*min_nodes=*/singleNode ? 1 : 2, /*max_nodes=*/singleNode ? 1 : 2)) {
            GTEST_SKIP() << (singleNode ? "Requires 4 ranks on a single node"
                                        : "Requires 8 ranks across exactly 2 nodes");
            return;
        }
        if (reject) {
            if (isElasticBufferRegisterEnabled()) {
                GTEST_SKIP() << "Requires NCCL_ELASTIC_BUFFER_REGISTER=0";
                return;
            }
        } else if (!isSymSysmemHandleReuseEnabled()) {
            GTEST_SKIP() << "Requires NCCL_SYM_REUSE_SYSMEM_HANDLES=1";
            return;
        }

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        if (!reject) {
            ASSERT_TRUE(isCuMemEnabled()) << "NCCL_CUMEM_ENABLE must be set to 1";
            ASSERT_TRUE(isWinEnabled()) << "NCCL_WIN_ENABLE must not be set to 0";
            ASSERT_TRUE(isElasticBufferRegisterEnabled())
                << "NCCL_ELASTIC_BUFFER_REGISTER must not be 0";
            ASSERT_TRUE(isPerRankLoggingEnabled()) << "RCCL_MPI_LOG_ALL_RANKS must be set to 1";
        }

        RCCLHybridVmmTests::HybridVmmBuffer hybrid;
        std::string reason;
        if (!createHybridVmmBuffer(
                /*gpuBytes=*/8 * 1024 * 1024,
                /*localCpuBytes=*/2 * 1024 * 1024, hybrid, reason)) {
            GTEST_SKIP() << "DeepEP hybrid allocation unavailable: " << reason;
            return;
        }
        auto hybridCleanup = makeScopeGuard([&]() {
            RCCLHybridVmmTests::FreeHybridVmm(hybrid);
        });

        ncclWindow_t win = nullptr;
        ncclResult_t result = ncclCommWindowRegister(
            getActiveCommunicator(), hybrid.ptr, hybrid.totalSize, &win,
            NCCL_WIN_STRICT_ORDERING);
        auto winCleanup = makeScopeGuard([&]() {
            if (win) HIP_EXPECT(ncclCommWindowDeregister(getActiveCommunicator(), win));
        });
        if (reject) {
            EXPECT_EQ(result, ncclInvalidArgument);
            EXPECT_EQ(win, nullptr);
            return;
        }

        ASSERT_MPI_EQ(ncclSuccess, result);
        ASSERT_MPI_NE(win, nullptr);
        ASSERT_EQ(hybrid.localSize, 4)
            << "The hybrid test requires exactly four local ranks per node";

        void* mappedWindow = nullptr;
        ASSERT_MPI_EQ(ncclSuccess, ncclGetLsaDevicePointer(
            win, 0, hybrid.localRank, &mappedWindow));
        ASSERT_MPI_NE(mappedWindow, nullptr);

        REGLogChecker checker = getLogChecker();
        ASSERT_TRUE(checker.hasNumSegments(hybrid.localSize + 1))
            << "Expected [GPU] plus one imported CPU segment per local rank";
        ASSERT_TRUE(checker.hasSymSysmemHandleReuse())
            << "Expected explicit NCCL_SYM_REUSE_SYSMEM_HANDLES reuse marker";
    }

    // GPU window is the control. Hierarchical CE implements AllGather and AlltoAll only.
    void runCeElasticGating(bool hierarchical)
    {
        const int nodeCount = MPITestConstants::detectNodeCount();
        if (hierarchical) {
            if (!validateTestPrerequisites(4)) {
                GTEST_SKIP() << "Hierarchical CE requires 4+ ranks";
                return;
            }
            if (nodeCount < 8) {
                GTEST_SKIP() << "Hierarchical CE requires >=8 nodes";
                return;
            }
        } else if (!validateTestPrerequisites(
                       /*min_processes=*/2, /*max_processes=*/kNoProcessLimit,
                       /*require_power_of_two=*/kNoPowerOfTwoRequired,
                       /*min_nodes=*/1, /*max_nodes=*/1)) {
            GTEST_SKIP() << "Requires 2+ ranks on a single node";
            return;
        }
        if (!isElasticBufferRegisterEnabled()) {
            GTEST_SKIP() << "Requires NCCL_ELASTIC_BUFFER_REGISTER=1";
            return;
        }
        if (!isCeDispatchConfigured()) {
            GTEST_SKIP() << "CE dispatch is not configured";
            return;
        }

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        if (hierarchical) {
            ncclComm* comm = getActiveCommunicator();
            const bool ready = comm->nNodes >= 8 && comm->hierarchicalCommsInitialized &&
                               comm->devrState.lsaSize < comm->nRanks && comm->symmetricSupport &&
                               comm->hostRmaSupport && comm->config.numRmaCtx > 0;
            if (!ready) {
                GTEST_SKIP() << "Hierarchical CE requires initialized sub-comms, symmetric memory, and RMA";
                return;
            }
        }
        ASSERT_TRUE(isCuMemEnabled()) << "NCCL_CUMEM_ENABLE must be set to 1";
        ASSERT_TRUE(isWinEnabled()) << "NCCL_WIN_ENABLE must not be set to 0";

        int dev = 0;
        ASSERT_MPI_EQ(hipSuccess, hipGetDevice(&dev));
        int nRanks = 0;
        ncclCommCount(getActiveCommunicator(), &nRanks);

        constexpr size_t kSegmentSize = 4 * 1024 * 1024;
        MultiSegmentBuffer gpuSend, gpuRecv, elasticSend, elasticRecv;
        ASSERT_NO_FATAL_FAILURE(createMultiSegmentBuffer(dev, kSegmentSize, 2, gpuSend));
        ASSERT_NO_FATAL_FAILURE(createMultiSegmentBuffer(dev, kSegmentSize, 2, gpuRecv));
        ASSERT_NO_FATAL_FAILURE(createMixedMultiSegmentBuffer(dev, kSegmentSize, 2, 1, elasticSend));
        ASSERT_NO_FATAL_FAILURE(createMixedMultiSegmentBuffer(dev, kSegmentSize, 2, 1, elasticRecv));
        {
            const bool allocated = gpuSend.totalSize != 0 && gpuRecv.totalSize != 0 &&
                                   elasticSend.totalSize != 0 && elasticRecv.totalSize != 0;
            const std::string why = skipUnlessAllRanksAllocated(
                allocated, "multi-segment VMM allocation unavailable");
            if (!why.empty()) {
                GTEST_SKIP() << why;
                return;
            }
        }
        auto vmmCleanup = makeScopeGuard([&]() {
            releaseMultiSegmentBuffer(gpuSend);
            releaseMultiSegmentBuffer(gpuRecv);
            releaseMultiSegmentBuffer(elasticSend);
            releaseMultiSegmentBuffer(elasticRecv);
        });

        ASSERT_EQ(gpuSend.totalSize, gpuRecv.totalSize);
        ASSERT_EQ(elasticSend.totalSize, elasticRecv.totalSize);
        ASSERT_EQ(gpuSend.totalSize % sizeof(float), 0u);
        ASSERT_EQ(elasticSend.totalSize % sizeof(float), 0u);
        const size_t gpuCount = gpuSend.totalSize / sizeof(float);
        const size_t elasticCount = elasticSend.totalSize / sizeof(float);
        if (nRanks <= 0 || gpuCount % static_cast<size_t>(nRanks) != 0 ||
            elasticCount % static_cast<size_t>(nRanks) != 0) {
            GTEST_SKIP() << "CE element count is not divisible by nRanks";
            return;
        }

        ncclWindow_t wins[4] = {};
        void* ptrs[4] = {gpuSend.vaBase, gpuRecv.vaBase, elasticSend.vaBase, elasticRecv.vaBase};
        size_t sizes[4] = {gpuSend.totalSize, gpuRecv.totalSize, elasticSend.totalSize, elasticRecv.totalSize};
        for (int i = 0; i < 4; ++i) {
            ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(
                getActiveCommunicator(), ptrs[i], sizes[i], &wins[i], NCCL_WIN_COLL_SYMMETRIC));
        }
        auto winCleanup = makeScopeGuard([&]() {
            for (ncclWindow_t win : wins) {
                if (win) HIP_EXPECT(ncclCommWindowDeregister(getActiveCommunicator(), win));
            }
        });

        auto ceSelected = [&](ncclFunc_t coll, void* send, void* recv, size_t count) {
            int algo = 0, proto = 0, nCh = 0;
            const ncclResult_t res = rcclGetCollImplInfo(
                getActiveCommunicator(), coll, count, ncclFloat32, ncclSum,
                send, recv, /*graphCapturing=*/0, &algo, &proto, &nCh);
            return res == ncclSuccess &&
                   (algo == RCCL_CE_REGISTERED || algo == RCCL_CE_2SHOT || algo == RCCL_CE_SCRATCH);
        };
        auto expectRejected = [&](ncclFunc_t coll, size_t gpuQueryCount, size_t elasticQueryCount,
                                  const char* name) {
            EXPECT_TRUE(ceSelected(coll, gpuSend.vaBase, gpuRecv.vaBase, gpuQueryCount))
                << name << ": GPU multi-segment window did not select CE";
            EXPECT_FALSE(ceSelected(coll, elasticSend.vaBase, elasticRecv.vaBase, elasticQueryCount))
                << name << ": CE selected an elastic window";
        };

        const size_t gpuSlice = gpuCount / static_cast<size_t>(nRanks);
        const size_t elasticSlice = elasticCount / static_cast<size_t>(nRanks);
        expectRejected(ncclFuncAllGather, gpuSlice, elasticSlice, "AllGather");
        if (nRanks < 8 || isCeAlltoAllDispatchConfigured())
            expectRejected(ncclFuncAlltoAll, gpuSlice, elasticSlice, "AlltoAll");
        if (!hierarchical && isCeAllReduceDispatchConfigured())
            expectRejected(ncclFuncAllReduce, gpuCount, elasticCount, "AllReduce");
    }
};

#endif // MPI_TESTS_ENABLED
#endif // RCCL_TEST_MULTISEGMENT_TEST_FIXTURE_HPP
