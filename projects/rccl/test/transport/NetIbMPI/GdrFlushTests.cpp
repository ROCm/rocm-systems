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
#include "NetIbFaultInject.hpp"
#include "NetIbCastInspect.hpp"
#include "net_ib_wqe_lat_inspect.h"
#include "rocmwrap.h"

#include <strings.h>

#include <hsa/hsa_ext_amd.h>
#include <string>

#ifdef MPI_TESTS_ENABLED

namespace {

// Preserve fatal failures raised by helpers inside the burst.
#define ASSERT_RECV_FLUSH_BURST(iterations, verifyData, lastFlush)                   \
    do {                                                                             \
        std::string _burstWhy;                                                       \
        ncclResult_t _burstRet = RunRecvFlushBurst((iterations), (verifyData),       \
                                                  (lastFlush), &_burstWhy);          \
        ASSERT_EQ(_burstRet, ncclSuccess)                                            \
            << "GDR recv+flush burst setup failed on at least one rank (this rank: " \
            << _burstWhy << ")";                                                     \
        if (::testing::Test::HasFatalFailure()) return;                              \
    } while (0)

class GdrFlushTest : public NetIbMPITest {
protected:
    void SetUp() override {
        NetIbMPITest::SetUp();
        // GetPlugin() uses strcmp against the plugin name "IB-CAST", so
        // NCCL_NET=ib-cast would otherwise keep running ncclNetIb.
        const char* env = getenv("NCCL_NET");
        if (env != nullptr && strcasecmp(env, netIbCast.name) == 0) net_ = &netIbCast;
    }

    // NCCL_CUMEM_ENABLE=1 makes the flush scratchpad dma-buf-backed (the path
    // that used to fault). Absent/0 selects the legacy peermem reg_mr path.
    static bool cuMemEnabledEnv() {
        // Honor the effective library gate, not just the env var. Forcing
        // NCCL_CUMEM_ENABLE=1 on a kernel without dma-buf still disables cuMem.
        if (!ncclCuMemEnable()) return false;
        const char* v = getenv("NCCL_CUMEM_ENABLE");
        return v && atoi(v) != 0;
    }

    // The dedicated RO=0 scratchpad flush (default on). When 0, ncclIbIflush
    // falls back to reading the received buffer directly (upstream-NCCL style).
    static bool scratchpadFlushEnabled() {
        const char* v = getenv("RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING");
        return !v || atoi(v) != 0;
    }

    int gdrPtrSupport() {
        ncclNetProperties_t props;
        if (GetDeviceProperties(0, &props) != ncclSuccess) return 0;
        return props.ptrSupport & (NCCL_PTR_CUDA | NCCL_PTR_DMABUF);
    }

    static std::string CollectiveSkipReason(std::string why) {
        int skip = why.empty() ? 0 : 1;
        MPI_Allreduce(MPI_IN_PLACE, &skip, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (skip && why.empty()) why = "peer rank cannot run this configuration";
        return skip ? why : std::string();
    }

    // Registers a GPU buffer for GDR: through regMr() where peermem is available,
    // otherwise through a dma-buf export. The export offset is taken for the
    // page-aligned base, because ncclIbRegMrDmaBuf aligns the address down itself.
    ncclResult_t RegisterGdrMemory(void* comm, void* buffer, size_t size, void** mhandle) {
        if (gdrPtrSupport() & NCCL_PTR_CUDA)
            return RegisterMemory(comm, buffer, size, NCCL_PTR_CUDA, mhandle);

        const uintptr_t pageSize = sysconf(_SC_PAGESIZE);
        const uintptr_t alignedAddr = reinterpret_cast<uintptr_t>(buffer) & ~(pageSize - 1);
        const size_t alignedSize =
            (reinterpret_cast<uintptr_t>(buffer) + size - alignedAddr + pageSize - 1) & ~(pageSize - 1);

        int fd = -1;
        uint64_t exportOffset = 0;
        hsa_status_t exportStatus = hsa_amd_portable_export_dmabuf(reinterpret_cast<const void*>(alignedAddr),
                                                                   alignedSize, &fd, &exportOffset);
        if (exportStatus != HSA_STATUS_SUCCESS || fd < 0) {
            ADD_FAILURE() << "hsa_amd_portable_export_dmabuf failed: hsa_status=" << exportStatus;
            return ncclSystemError;
        }

        ncclResult_t res = RegisterDmaBufMemory(comm, buffer, size, NCCL_PTR_CUDA, exportOffset, fd, mhandle);
        (void)close(fd);

        return res;
    }

    // Returns rank-agreed setup status; rank-local transfer checks remain non-fatal.
    [[nodiscard]] ncclResult_t RunRecvFlushBurst(int iterations, bool verifyData,
                                                ncclResult_t* rank0LastFlush,
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
        EXPECT_EQ(RegisterGdrMemory(comm, buffer, bufferSize, &mhandle), ncclSuccess);
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
        return ncclSuccess;
    }

    // One recv+flush. The host sentinel must come back, and the GPU scratchpad
    // must hold it: a flush that posts only the read leaves the scratchpad zero.
    void ExpectScratchpadSentinelRoundTrip() {
        const int rank = MPIEnvironment::world_rank;

        AssertInitAndGetDevices(nullptr);

        ConnectionPair pair;
        NetConnectionGuard connGuard(net_);
        ASSERT_NO_FATAL_FAILURE(SetupConnectionWithGuard(0, pair, connGuard));

        int skip = 0;
        int* hostMem = nullptr;
        int* gpuMem = nullptr;
        if (rank == 0) {
            ncclResult_t got = (net_ == &netIbCast)
                                   ? ncclIbCastGdrFlushGetScratch(pair.recvComm, 0, &hostMem, &gpuMem)
                                   : ncclIbGdrFlushGetScratch(pair.recvComm, 0, &hostMem, &gpuMem);
            if (got != ncclSuccess || hostMem == nullptr || gpuMem == nullptr) skip = 1;
        }
        MPI_Bcast(&skip, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (skip) {
            if (rank == 0) ADD_FAILURE() << "recv comm has no GDR flush scratchpad";
            return;
        }

        constexpr int kSentinel = 0x5a5a5a5a;
        if (rank == 0) {
            *hostMem = kSentinel;
            ASSERT_EQ(hipMemset(gpuMem, 0, sizeof(*gpuMem)), hipSuccess);
        }

        const size_t bufferSize = kSmallBufferSize;
        void* buffer = nullptr;
        EXPECT_EQ(hipMalloc(&buffer, bufferSize), hipSuccess);
        auto bufGuard = makeDeviceBufferAutoGuard(buffer);

        void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
        void* mhandle = nullptr;
        EXPECT_EQ(RegisterGdrMemory(comm, buffer, bufferSize, &mhandle), ncclSuccess);
        NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

        const int tag = 790;
        const int seed = 1234;
        void* request = nullptr;
        if (rank == 0) {
            EXPECT_EQ(hipMemset(buffer, 0, bufferSize), hipSuccess);
            void* rb[1] = {buffer};
            size_t rs[1] = {bufferSize};
            int rt[1] = {tag};
            void* rh[1] = {mhandle};
            EXPECT_EQ(PostRecv(pair.recvComm, 1, rb, rs, rt, rh, &request), ncclSuccess);
        } else {
            EXPECT_EQ(initializeBufferWithPattern<uint8_t>(buffer, bufferSize, makeBytePattern(seed)), hipSuccess);
            PostSendWithRetry(pair.sendComm, buffer, bufferSize, tag, mhandle, &request);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        int sizes[1] = {0};
        EXPECT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

        if (rank == 0) {
            void* fb[1] = {buffer};
            int fs[1] = {static_cast<int>(bufferSize)};
            void* fh[1] = {mhandle};
            void* flushReq = nullptr;
            ASSERT_EQ(FlushRecv(pair.recvComm, 1, fb, fs, fh, &flushReq), ncclSuccess);
            ASSERT_NE(flushReq, nullptr);
            ASSERT_EQ(WaitForCompletion(flushReq, nullptr), ncclSuccess);
            EXPECT_EQ(*hostMem, kSentinel) << "scratchpad read did not return the sentinel";
            int gpuVal = 0;
            ASSERT_EQ(hipMemcpy(&gpuVal, gpuMem, sizeof(gpuVal), hipMemcpyDeviceToHost), hipSuccess);
            EXPECT_EQ(gpuVal, kSentinel) << "scratchpad write did not land";
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
};

// dma-buf (cuMem/UBR) scratchpad: the write+read flush must complete cleanly with
// correct data.
TEST_F(GdrFlushTest, CuMemDmaBuf_GpuRecvFlush_NoAsyncFatal) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses,
                                          false, kMinGpusPerNode, kNoNodeLimit);
    AssertInitAndGetDevices(nullptr);
    std::string why;
    if (!cuMemEnabledEnv()) why = "Requires NCCL_CUMEM_ENABLE=1 (dma-buf scratchpad path)";
    else if (!gdrPtrSupport()) why = "no GDR backend (neither peermem nor dma-buf) on this device";
    why = CollectiveSkipReason(why);
    if (!why.empty()) GTEST_SKIP() << why;

    ncclResult_t flush = ncclSuccess;
    ASSERT_RECV_FLUSH_BURST(/*iterations=*/4, /*verifyData=*/true, &flush);
    if (MPIEnvironment::world_rank == 0)
        EXPECT_EQ(flush, ncclSuccess) << "write+read flush over dma-buf scratchpad must not fault";
}

// Legacy peermem (ibv_reg_mr) scratchpad. Confirms Option 2a keeps the peermem
// RO=0 read path working (never regressed).
TEST_F(GdrFlushTest, Peermem_GpuRecvFlush_NoAsyncFatal) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses,
                                          false, kMinGpusPerNode, kNoNodeLimit);
    AssertInitAndGetDevices(nullptr);
    std::string why;
    if (cuMemEnabledEnv()) why = "Requires NCCL_CUMEM_ENABLE=0 (peermem scratchpad path)";
    else if (!(gdrPtrSupport() & NCCL_PTR_CUDA))
        why = "peermem (NCCL_PTR_CUDA) not available for the reg_mr scratchpad";
    why = CollectiveSkipReason(why);
    if (!why.empty()) GTEST_SKIP() << why;

    ncclResult_t flush = ncclSuccess;
    ASSERT_RECV_FLUSH_BURST(/*iterations=*/4, /*verifyData=*/true, &flush);
    if (MPIEnvironment::world_rank == 0)
        EXPECT_EQ(flush, ncclSuccess) << "peermem RO=0 scratchpad flush must succeed";
}

// Feature disabled: ncclIbIflush falls back to reading the received buffer
// directly (upstream-NCCL behaviour). Exercises the else branch.
TEST_F(GdrFlushTest, FeatureDisabled_FallbackReadRecvBuffer) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses,
                                          false, kMinGpusPerNode, kNoNodeLimit);
    AssertInitAndGetDevices(nullptr);
    std::string why;
    if (scratchpadFlushEnabled())
        why = "Requires RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 (fallback path)";
    else if (!gdrPtrSupport()) why = "no GDR backend (neither peermem nor dma-buf) on this device";
    why = CollectiveSkipReason(why);
    if (!why.empty()) GTEST_SKIP() << why;

    ncclResult_t flush = ncclSuccess;
    ASSERT_RECV_FLUSH_BURST(/*iterations=*/4, /*verifyData=*/true, &flush);
    if (MPIEnvironment::world_rank == 0)
        EXPECT_EQ(flush, ncclSuccess) << "fallback flush (read recv buffer) must succeed";
}

// The chained flush posts an unsignaled scratchpad write and a signaled read.
// Zero the scratchpad and put a sentinel in the host payload: the sentinel
// comes back only if the write ran before the read.
TEST_F(GdrFlushTest, ScratchpadWrite_RoundTripsSentinel) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses,
                                          false, kMinGpusPerNode, kNoNodeLimit);
    if (!scratchpadFlushEnabled())
        GTEST_SKIP() << "Requires the RO=0 scratchpad flush (RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING!=0)";
    AssertInitAndGetDevices(nullptr);
    if (!gdrPtrSupport()) GTEST_SKIP() << "no GDR backend (neither peermem nor dma-buf) on this device";

    ASSERT_NO_FATAL_FAILURE(ExpectScratchpadSentinelRoundTrip());
}

// Repeated-flush burst. An intermittent async-fatal would surface as a
// non-success flush on some iteration; the whole burst must stay clean.
TEST_F(GdrFlushTest, RepeatedFlush_NoFaultBurst) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses,
                                          false, kMinGpusPerNode, kNoNodeLimit);
    AssertInitAndGetDevices(nullptr);
    std::string why = gdrPtrSupport() ? std::string() : "no GDR backend (neither peermem nor dma-buf) on this device";
    why = CollectiveSkipReason(why);
    if (!why.empty()) GTEST_SKIP() << why;

    ncclResult_t flush = ncclSuccess;
    ASSERT_RECV_FLUSH_BURST(/*iterations=*/50, /*verifyData=*/false, &flush);
    if (MPIEnvironment::world_rank == 0)
        EXPECT_EQ(flush, ncclSuccess) << "no flush in the burst may raise a QP async-fatal";
}

// The NCCL v2.31.2-1 sync added ForcedScratchpadWrite_ReproducesFault: force the
// removed scratchpad RDMA_WRITE on a dma-buf flush QP and expect the flush to
// fault. That case never compiled: gdrSupported() does not exist, and
// RunRecvFlushBurst has no forceWrite parameter. If a production forceWrite
// hook is added, restore the case (skip with gdrPtrSupport(), three-arg burst
// ForcedScratchpadWrite_ReproducesFault removed: forcing the flush fault needs a production hook that does not exist.

}  // namespace

#endif  // MPI_TESTS_ENABLED
