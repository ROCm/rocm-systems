/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Whitebox tests for the NET/IB GPU-Direct-RDMA flush (ncclIbIflush).
//
// Background: RCCL's GDR flush issues a loopback RDMA_READ against a dedicated
// RO=0 GPU scratchpad to fence relaxed-ordering data writes. A prior version
// also issued an RDMA_WRITE into that scratchpad; on a dma-buf-backed (cuMem/UBR)
// scratchpad amdgpu cannot resolve the tiny allocation as a writable RDMA
// target, so mlx5 rejected the WRITE with "invalid request local work queue
// error" and tore down the flush QP. The fix removed the redundant WRITE.
//
// These tests exercise ncclIbIflush directly over a 2-rank loopback connection
// (single-node capable), covering both scratchpad backends, the feature-disabled
// fallback, a repeated-flush burst, and - under ENABLE_FAULT_INJECTION - a
// regression guard that forces the removed WRITE back and asserts it faults.

#include "NetIbMPITestBase.hpp"
#include "NetIbFaultInject.hpp"

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
    // completion result via `rank0LastFlush` (may be null). When `forceWrite` is
    // set (ENABLE_FAULT_INJECTION only) the receiver re-issues the removed
    // scratchpad RDMA_WRITE to reproduce the historical fault.
    //
    // The data buffer is plain hipMalloc so the whitebox regMr(NCCL_PTR_CUDA)
    // succeeds (raw VMM pointers require regMrDmaBuf). The faulting scratchpad is
    // the recv comm's INTERNAL gpuFlush, which becomes dma-buf-backed purely from
    // NCCL_CUMEM_ENABLE=1 - independent of the data buffer's allocator.
    //
    // Returns the setup verdict rather than asserting on it. A fatal assertion
    // inside a void helper returns from the helper alone, so the caller carried on
    // with the comms SetupConnectionWithGuard had already closed unless it
    // remembered ASSERT_NO_FATAL_FAILURE -- a wrapper nothing enforced. The status
    // is [[nodiscard]] and these sources build with -Werror=unused-result, so the
    // call sites cannot drop it. Only the rank-agreed setup failure travels back
    // this way; everything below stays EXPECT_, because a rank-local failure that
    // ended the test on one rank would strand its peer at the next collective.
    [[nodiscard]] ncclResult_t RunRecvFlushBurst(int iterations, bool verifyData,
                                                bool forceWrite, ncclResult_t* rank0LastFlush,
                                                std::string* setupWhy = nullptr) {
        const int rank = MPIEnvironment::world_rank;

        AssertInitAndGetDevices(nullptr);

        ConnectionPair pair;
        NetConnectionGuard connGuard(net_);
        const ncclResult_t setup = SetupConnectionWithGuard(0, pair, connGuard, setupWhy);
        if (setup != ncclSuccess) return setup;

        const size_t bufferSize = kSmallBufferSize;
        void* buffer = nullptr;
        EXPECT_EQ(hipMalloc(&buffer, bufferSize), hipSuccess);
        auto bufGuard = makeDeviceBufferAutoGuard(buffer);

        void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
        void* mhandle = nullptr;
        EXPECT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_CUDA, &mhandle), ncclSuccess);
        NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

#if defined(ENABLE_FAULT_INJECTION)
        if (rank == 0 && forceWrite) {
            EXPECT_EQ(ncclIbFlushFaultForceScratchpadWrite(pair.recvComm, true), ncclSuccess);
        }
#else
        (void)forceWrite;
#endif

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

#if defined(ENABLE_FAULT_INJECTION)
        if (rank == 0 && forceWrite) {
            (void)ncclIbFlushFaultForceScratchpadWrite(pair.recvComm, false);
        }
#endif

        if (rank == 0 && rank0LastFlush) *rank0LastFlush = lastFlush;
        return ncclSuccess;
    }
};

// dma-buf (cuMem/UBR) scratchpad: the exact path that used to fault. The
// read-only flush must complete cleanly with correct data.
TEST_F(GdrFlushTest, CuMemDmaBuf_GpuRecvFlush_NoAsyncFatal) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                          false, kMinGpusPerNode, kNoNodeLimit));
    if (!cuMemEnabledEnv()) GTEST_SKIP() << "Requires NCCL_CUMEM_ENABLE=1 (dma-buf scratchpad path)";
    AssertInitAndGetDevices(nullptr);
    if (!gdrSupported()) GTEST_SKIP() << "GDR (NCCL_PTR_CUDA) not supported on this device";

    ncclResult_t flush = ncclSuccess;
    std::string setupWhy;
    ASSERT_EQ(RunRecvFlushBurst(/*iterations=*/4, /*verifyData=*/true,
                                /*forceWrite=*/false, &flush, &setupWhy), ncclSuccess)
        << "GDR recv+flush burst setup failed on at least one rank (this rank: "
        << setupWhy << ")";
    if (MPIEnvironment::world_rank == 0)
        EXPECT_EQ(flush, ncclSuccess) << "read-only flush over dma-buf scratchpad must not fault";
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
    std::string setupWhy;
    ASSERT_EQ(RunRecvFlushBurst(/*iterations=*/4, /*verifyData=*/true,
                                /*forceWrite=*/false, &flush, &setupWhy), ncclSuccess)
        << "GDR recv+flush burst setup failed on at least one rank (this rank: "
        << setupWhy << ")";
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
    std::string setupWhy;
    ASSERT_EQ(RunRecvFlushBurst(/*iterations=*/4, /*verifyData=*/true,
                                /*forceWrite=*/false, &flush, &setupWhy), ncclSuccess)
        << "GDR recv+flush burst setup failed on at least one rank (this rank: "
        << setupWhy << ")";
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
    std::string setupWhy;
    ASSERT_EQ(RunRecvFlushBurst(/*iterations=*/50, /*verifyData=*/false,
                                /*forceWrite=*/false, &flush, &setupWhy), ncclSuccess)
        << "GDR recv+flush burst setup failed on at least one rank (this rank: "
        << setupWhy << ")";
    if (MPIEnvironment::world_rank == 0)
        EXPECT_EQ(flush, ncclSuccess) << "no flush in the burst may raise a QP async-fatal";
}

#if defined(ENABLE_FAULT_INJECTION)
// Regression guard - force the removed scratchpad RDMA_WRITE back. On a dma-buf
// scratchpad this must reproduce the fault (flush no longer succeeds), proving
// the WRITE is the culprit and that removing it is the fix.
TEST_F(GdrFlushTest, ForcedScratchpadWrite_ReproducesFault) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                          false, kMinGpusPerNode, kNoNodeLimit));
    if (!cuMemEnabledEnv()) GTEST_SKIP() << "Requires NCCL_CUMEM_ENABLE=1 (dma-buf scratchpad target)";
    if (!scratchpadFlushEnabled())
        GTEST_SKIP() << "Requires the scratchpad flush enabled (RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=1)";
    AssertInitAndGetDevices(nullptr);
    if (!gdrSupported()) GTEST_SKIP() << "GDR (NCCL_PTR_CUDA) not supported on this device";

    ncclResult_t forced = ncclSuccess;
    std::string setupWhy;
    ASSERT_EQ(RunRecvFlushBurst(/*iterations=*/1, /*verifyData=*/false,
                                /*forceWrite=*/true, &forced, &setupWhy), ncclSuccess)
        << "GDR recv+flush burst setup failed on at least one rank (this rank: "
        << setupWhy << ")";
    if (MPIEnvironment::world_rank == 0)
        EXPECT_NE(forced, ncclSuccess)
            << "forced scratchpad RDMA_WRITE on a dma-buf buffer should fault the flush QP";
}
#endif  // ENABLE_FAULT_INJECTION

}  // namespace

#endif  // MPI_TESTS_ENABLED
