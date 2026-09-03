/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Whitebox tests for the NET/IB GPU-Direct-RDMA flush (ncclIbIflush).
//
// Background: RCCL's GDR flush fences relaxed-ordering (RO=1) data writes through
// a dedicated RO=0 GPU scratchpad. It posts a loopback RDMA_WRITE of a dummy
// payload into that scratchpad and then an RDMA_READ back. The WRITE needs
// REMOTE_WRITE granted on the flush QP (ncclIbReceiverQpsCreateToRts).
//
// These tests exercise ncclIbIflush directly over a 2-rank loopback connection
// (single-node capable), covering both scratchpad backends, the feature-disabled
// fallback, and a repeated-flush burst.

#include "NetIbMPITestBase.hpp"

#ifdef MPI_TESTS_ENABLED

namespace {

class GdrFlushTest : public NetIbMPITest {
protected:
    // NCCL_CUMEM_ENABLE=1 makes the flush scratchpad dma-buf-backed (the path
    // that used to fault). Absent/0 selects the legacy peermem reg_mr path.
    static bool cuMemEnabledEnv() {
        const char* v = getenv("NCCL_CUMEM_ENABLE");
        return v && atoi(v) != 0;
    }

    // The dedicated RO=0 scratchpad flush (default on). When 0, ncclIbIflush
    // falls back to reading the received buffer directly (upstream-NCCL style).
    static bool scratchpadFlushEnabled() {
        const char* v = getenv("RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING");
        return !v || atoi(v) != 0;
    }

    bool gdrSupported() {
        ncclNetProperties_t props;
        if (GetDeviceProperties(0, &props) != ncclSuccess) return false;
        return (props.ptrSupport & NCCL_PTR_CUDA) != 0;
    }

    // Runs `iterations` of GPU recv + flush across 2 ranks on device 0.
    // rank 0 = receiver + flush, rank 1 = sender. Returns rank 0's last flush
    // completion result via `rank0LastFlush` (may be null).
    //
    // The data buffer is plain hipMalloc so the whitebox regMr(NCCL_PTR_CUDA)
    // succeeds (raw VMM pointers require regMrDmaBuf). The exercised scratchpad is
    // the recv comm's INTERNAL gpuFlush, which becomes dma-buf-backed purely from
    // NCCL_CUMEM_ENABLE=1 - independent of the data buffer's allocator.
    void RunRecvFlushBurst(int iterations, bool verifyData,
                           ncclResult_t* rank0LastFlush) {
        const int rank = MPIEnvironment::world_rank;

        AssertInitAndGetDevices(nullptr);

        ConnectionPair pair;
        NetConnectionGuard connGuard(net_);
        SetupConnectionWithGuard(0, pair, connGuard);

        const size_t bufferSize = kSmallBufferSize;
        void* buffer = nullptr;
        EXPECT_EQ(hipMalloc(&buffer, bufferSize), hipSuccess);
        auto bufGuard = makeDeviceBufferAutoGuard(buffer);

        void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
        void* mhandle = nullptr;
        EXPECT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_CUDA, &mhandle), ncclSuccess);
        NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

        ncclResult_t lastFlush = ncclSuccess;
        for (int it = 0; it < iterations; ++it) {
            const int tag  = 700 + it;
            const int seed = 1234 + it;
            void* request  = nullptr;

            if (rank == 0) {
                EXPECT_EQ(hipMemset(buffer, 0, bufferSize), hipSuccess);
                void*  rb[1] = {buffer};
                size_t rs[1] = {bufferSize};
                int    rt[1] = {tag};
                void*  rh[1] = {mhandle};
                EXPECT_EQ(PostRecv(pair.recvComm, 1, rb, rs, rt, rh, &request), ncclSuccess);
                EXPECT_NE(request, nullptr);
            } else {
                EXPECT_EQ(initializeBufferWithPattern<uint8_t>(buffer, bufferSize, makeBytePattern(seed)),
                          hipSuccess);
                PostSendWithRetry(pair.sendComm, buffer, bufferSize, tag, mhandle, &request);
            }

            MPI_Barrier(MPI_COMM_WORLD);

            int sizes[1] = {0};
            EXPECT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

            if (rank == 0) {
                void* fb[1] = {buffer};
                int   fs[1] = {static_cast<int>(bufferSize)};
                void* fh[1] = {mhandle};
                void* flushReq = nullptr;
                ncclResult_t r = FlushRecv(pair.recvComm, 1, fb, fs, fh, &flushReq);
                if (r == ncclSuccess && flushReq != nullptr) {
                    r = WaitForCompletion(flushReq, nullptr);
                }
                lastFlush = r;
                if (verifyData && r == ncclSuccess) {
                    size_t errIdx = 0; uint8_t exp = 0, act = 0;
                    EXPECT_TRUE(verifyBufferData<uint8_t>(buffer, bufferSize, makeBytePattern(seed),
                                                          0, 0, &errIdx, &exp, &act))
                        << "flushed data mismatch at " << errIdx
                        << " exp=" << (int)exp << " act=" << (int)act;
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (rank == 0 && rank0LastFlush) *rank0LastFlush = lastFlush;
    }
};

// dma-buf (cuMem/UBR) scratchpad: the write+read flush must complete cleanly with
// correct data.
TEST_F(GdrFlushTest, CuMemDmaBuf_GpuRecvFlush_NoAsyncFatal) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                          false, kMinGpusPerNode, kNoNodeLimit));
    if (!cuMemEnabledEnv()) GTEST_SKIP() << "Requires NCCL_CUMEM_ENABLE=1 (dma-buf scratchpad path)";
    AssertInitAndGetDevices(nullptr);
    if (!gdrSupported()) GTEST_SKIP() << "GDR (NCCL_PTR_CUDA) not supported on this device";

    ncclResult_t flush = ncclSuccess;
    RunRecvFlushBurst(/*iterations=*/4, /*verifyData=*/true, &flush);
    if (MPIEnvironment::world_rank == 0)
        EXPECT_EQ(flush, ncclSuccess) << "write+read flush over dma-buf scratchpad must not fault";
}

// Legacy peermem (ibv_reg_mr) scratchpad. Confirms Option 2a keeps the peermem
// RO=0 read path working (never regressed).
TEST_F(GdrFlushTest, Peermem_GpuRecvFlush_NoAsyncFatal) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                          false, kMinGpusPerNode, kNoNodeLimit));
    if (cuMemEnabledEnv()) GTEST_SKIP() << "Requires NCCL_CUMEM_ENABLE=0 (peermem scratchpad path)";
    AssertInitAndGetDevices(nullptr);
    if (!gdrSupported()) GTEST_SKIP() << "GDR (NCCL_PTR_CUDA) not supported on this device";

    ncclResult_t flush = ncclSuccess;
    RunRecvFlushBurst(/*iterations=*/4, /*verifyData=*/true, &flush);
    if (MPIEnvironment::world_rank == 0)
        EXPECT_EQ(flush, ncclSuccess) << "peermem RO=0 scratchpad flush must succeed";
}

// Feature disabled: ncclIbIflush falls back to reading the received buffer
// directly (upstream-NCCL behaviour). Exercises the else branch.
TEST_F(GdrFlushTest, FeatureDisabled_FallbackReadRecvBuffer) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                          false, kMinGpusPerNode, kNoNodeLimit));
    if (scratchpadFlushEnabled())
        GTEST_SKIP() << "Requires RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 (fallback path)";
    AssertInitAndGetDevices(nullptr);
    if (!gdrSupported()) GTEST_SKIP() << "GDR (NCCL_PTR_CUDA) not supported on this device";

    ncclResult_t flush = ncclSuccess;
    RunRecvFlushBurst(/*iterations=*/4, /*verifyData=*/true, &flush);
    if (MPIEnvironment::world_rank == 0)
        EXPECT_EQ(flush, ncclSuccess) << "fallback flush (read recv buffer) must succeed";
}

// Repeated-flush burst. An intermittent async-fatal would surface as a
// non-success flush on some iteration; the whole burst must stay clean.
TEST_F(GdrFlushTest, RepeatedFlush_NoFaultBurst) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                          false, kMinGpusPerNode, kNoNodeLimit));
    AssertInitAndGetDevices(nullptr);
    if (!gdrSupported()) GTEST_SKIP() << "GDR (NCCL_PTR_CUDA) not supported on this device";

    ncclResult_t flush = ncclSuccess;
    RunRecvFlushBurst(/*iterations=*/50, /*verifyData=*/false, &flush);
    if (MPIEnvironment::world_rank == 0)
        EXPECT_EQ(flush, ncclSuccess) << "no flush in the burst may raise a QP async-fatal";
}

}  // namespace

#endif  // MPI_TESTS_ENABLED
