/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <vector>

#include "common/ProcessIsolatedTestRunner.hpp"

/*
 * Differential regression test for the NCCL 2.28.7 symmetric AllGather_LL
 * data-corruption fix.
 *
 *   src/device/symmetric/all_gather.cuh, ncclSymkRun_AllGather_LL_impl():
 *     -  uint32_t lowBits = nElts;        // BUG: per-block share, always
 *     -                                   //      multiple of EltPerCell
 *     -                                   //      (= 1024 bytes) so always
 *     -                                   //      %8 == 0
 *     +  uint32_t lowBits = nAllElts;     // FIX: per-OP byte count
 *
 * Trigger conditions (verified empirically on MI300X / HIP 7.13 / 4 ranks,
 * see defect-validation/SYM_BUG_REPRODUCED.md):
 *   1. AllGather on a NCCL_WIN_COLL_SYMMETRIC-registered window
 *   2. Per-rank byte count NOT divisible by 8 (count=257 floats: 1028 B,
 *      1028 % 8 = 4)
 *   3. Per-rank byte count >= ~1024 bytes so the work spans multiple
 *      NCCL_SYM_KERNEL_CELL_SIZE cells and per-block share rounds to a
 *      multiple of 8 while total does not
 *   4. AllGather_LL kernel selected (forced via NCCL_SYM_KERNEL=AllGather_LL)
 *
 * Without the fix: every collected element is shifted by exactly one
 * position (truncating-divide nAllElts/8 loses one element of stride).
 * With the fix: bit-exact correct.
 *
 * Why ProcessIsolatedTestRunner: the test calls hipSetDevice +
 * ncclCommInitAll which dirty HIP/HSA driver state in the test process.
 * If left in the parent, the next gtest case's getDeviceCount() fork
 * inherits the dirty state and fails with "Memory in use".  Running the
 * body in a forked subprocess means all HIP/HSA state goes away when
 * the subprocess exits.
 */

namespace {

constexpr int kNRanks = 4;
constexpr int kCountThatBreaks8ByteAlignment = 257;

struct ThreadArg {
    int rank;
    ncclUniqueId* uid;
    int* failPtr;
    int* skipPtr;
};

void* runRank(void* p) {
    ThreadArg* a = (ThreadArg*)p;
    if (hipSetDevice(a->rank) != hipSuccess) {
        __atomic_store_n(a->failPtr, 1, __ATOMIC_SEQ_CST);
        return nullptr;
    }

    ncclComm_t comm = nullptr;
    if (ncclCommInitRank(&comm, kNRanks, *a->uid, a->rank) != ncclSuccess) {
        __atomic_store_n(a->failPtr, 1, __ATOMIC_SEQ_CST);
        return nullptr;
    }

    const size_t totalBytes = (size_t)kCountThatBreaks8ByteAlignment * kNRanks * sizeof(float);
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    /* ncclMemAlloc may fail on stacks where CUMEM/VMM is not available
     * (HIP < 7.12.60540 without NCCL_CUMEM_ENABLE workaround).  Treat
     * as a skip rather than a failure - the fix only matters when the
     * symmetric path is reachable. */
    if (ncclMemAlloc(&sendBuf, totalBytes) != ncclSuccess ||
        ncclMemAlloc(&recvBuf, totalBytes) != ncclSuccess) {
        __atomic_store_n(a->skipPtr, 1, __ATOMIC_SEQ_CST);
        ncclCommDestroy(comm);
        return nullptr;
    }

    ncclWindow_t winSend = nullptr, winRecv = nullptr;
    if (ncclCommWindowRegister(comm, sendBuf, totalBytes, &winSend,
                                NCCL_WIN_COLL_SYMMETRIC) != ncclSuccess ||
        ncclCommWindowRegister(comm, recvBuf, totalBytes, &winRecv,
                                NCCL_WIN_COLL_SYMMETRIC) != ncclSuccess) {
        __atomic_store_n(a->skipPtr, 1, __ATOMIC_SEQ_CST);
        ncclMemFree(sendBuf);
        ncclMemFree(recvBuf);
        ncclCommDestroy(comm);
        return nullptr;
    }

    /* Initialize sendbuf with rank-distinct, position-distinct pattern so
     * any mis-stride is bit-detectable.  Pattern: send[i] = rank*1000000+i */
    std::vector<float> hSend(kCountThatBreaks8ByteAlignment);
    for (int i = 0; i < kCountThatBreaks8ByteAlignment; ++i) {
        hSend[i] = (float)(a->rank * 1000000 + i);
    }
    if (hipMemcpy(sendBuf, hSend.data(),
                  kCountThatBreaks8ByteAlignment * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess) {
        __atomic_store_n(a->failPtr, 1, __ATOMIC_SEQ_CST);
        goto cleanup;
    }

    /* Initialize recv to a sentinel so partial overwrites are visible. */
    {
        std::vector<float> hRecvInit((size_t)kCountThatBreaks8ByteAlignment * kNRanks, -42.0f);
        if (hipMemcpy(recvBuf, hRecvInit.data(),
                      hRecvInit.size() * sizeof(float),
                      hipMemcpyHostToDevice) != hipSuccess) {
            __atomic_store_n(a->failPtr, 1, __ATOMIC_SEQ_CST);
            goto cleanup;
        }
    }

    {
        hipStream_t stream;
        if (hipStreamCreate(&stream) != hipSuccess) {
            __atomic_store_n(a->failPtr, 1, __ATOMIC_SEQ_CST);
            goto cleanup;
        }
        ncclResult_t agRes = ncclAllGather(sendBuf, recvBuf,
                                            kCountThatBreaks8ByteAlignment,
                                            ncclFloat32, comm, stream);
        hipError_t syncRes = (agRes == ncclSuccess) ? hipStreamSynchronize(stream)
                                                     : hipSuccess;
        /* Builds without GENERATE_SYM_KERNELS=ON have nullptr in the
         * symmetric kernel table; dispatch yields hipErrorInvalidDeviceFunction
         * once the work hits the device.  Treat as SKIP - the bug we want
         * to catch is unreachable on such a build. */
        if (syncRes == hipErrorInvalidDeviceFunction ||
            agRes == ncclInvalidUsage) {
            __atomic_store_n(a->skipPtr, 1, __ATOMIC_SEQ_CST);
            (void)hipStreamDestroy(stream);
            goto cleanup;
        }
        if (agRes != ncclSuccess || syncRes != hipSuccess) {
            __atomic_store_n(a->failPtr, 1, __ATOMIC_SEQ_CST);
            (void)hipStreamDestroy(stream);
            goto cleanup;
        }

        std::vector<float> hRecv((size_t)kCountThatBreaks8ByteAlignment * kNRanks);
        if (hipMemcpy(hRecv.data(), recvBuf, hRecv.size() * sizeof(float),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            __atomic_store_n(a->failPtr, 1, __ATOMIC_SEQ_CST);
            (void)hipStreamDestroy(stream);
            goto cleanup;
        }

        for (int r = 0; r < kNRanks; ++r) {
            for (int i = 0; i < kCountThatBreaks8ByteAlignment; ++i) {
                float expect = (float)(r * 1000000 + i);
                float got = hRecv[(size_t)r * kCountThatBreaks8ByteAlignment + i];
                if (expect != got) {
                    __atomic_store_n(a->failPtr, 1, __ATOMIC_SEQ_CST);
                    goto streamDone;
                }
            }
        }
    streamDone:
        (void)hipStreamDestroy(stream);
    }

cleanup:
    ncclCommWindowDeregister(comm, winSend);
    ncclCommWindowDeregister(comm, winRecv);
    ncclMemFree(sendBuf);
    ncclMemFree(recvBuf);
    ncclCommDestroy(comm);
    return nullptr;
}

} /* anonymous namespace */

namespace RcclUnitTesting
{
    TEST(SymAllGatherCorruption, NoAlignmentBugAtCount257)
    {
        /* This regression test only matters when the build was compiled
         * with -DGENERATE_SYM_KERNELS=ON AND the HIP runtime supports
         * CUMEM/VMM (>= 7.12.60540).  Detecting these reliably from the
         * test would require RCCL-internal symbols, so we SKIP by default
         * and require RCCL_SYM_TEST_FORCE=1 to opt the test in. */
        if (getenv("RCCL_SYM_TEST_FORCE") == nullptr) {
            GTEST_SKIP() << "RCCL_SYM_TEST_FORCE not set; this regression "
                            "test only applies when RCCL was built with "
                            "GENERATE_SYM_KERNELS=ON and the HIP runtime "
                            "supports CUMEM/VMM (HIP >= 7.12.60540).  "
                            "Set RCCL_SYM_TEST_FORCE=1 to enable.";
        }

        int devCount = 0;
        ASSERT_EQ(hipGetDeviceCount(&devCount), hipSuccess);
        if (devCount < kNRanks) {
            GTEST_SKIP() << "needs >= " << kNRanks
                         << " GPUs (this trigger config requires 4 ranks);"
                            " have " << devCount;
        }

        /* Run the test body in a forked subprocess so the HIP/HSA driver
         * state, ncclComm allocations, and any other side effects do not
         * pollute later gtest cases (which would otherwise cause
         * "Memory in use" HSA errors in subsequent tests' getDeviceCount
         * fork calls). */
        RUN_ISOLATED_TEST_WITH_ENV(
            "SymAllGatherCorruption_NoAlignmentBugAtCount257_Body",
            []() {
                ncclUniqueId uid;
                ASSERT_EQ(ncclGetUniqueId(&uid), ncclSuccess);

                int fail = 0;
                int skip = 0;
                std::vector<pthread_t> threads(kNRanks);
                std::vector<ThreadArg> args(kNRanks);
                for (int r = 0; r < kNRanks; ++r) {
                    args[r] = ThreadArg{r, &uid, &fail, &skip};
                    ASSERT_EQ(pthread_create(&threads[r], nullptr, runRank, &args[r]), 0);
                }
                for (int r = 0; r < kNRanks; ++r) {
                    pthread_join(threads[r], nullptr);
                }

                if (skip) {
                    GTEST_SKIP() << "symmetric memory unreachable on this "
                                    "build/stack (GENERATE_SYM_KERNELS=OFF "
                                    "or HIP runtime lacks CUMEM/VMM "
                                    "support); cannot exercise the bug";
                }
                EXPECT_EQ(fail, 0)
                    << "Pre-fix RCCL produces shifted-by-1 corruption at "
                       "this size.  If this test fails, "
                       "src/device/symmetric/all_gather.cuh's "
                       "ncclSymkRun_AllGather_LL_impl() probably reverted "
                       "to 'lowBits = nElts;' instead of "
                       "'lowBits = nAllElts;'.";
            },
            /* env vars set inside the subprocess only - do not affect
             * other gtest cases in the parent binary.  NCCL_SYM_KERNEL
             * forces the buggy LL kernel so the perf model can't pick
             * AllGather_ST and hide the bug.  NCCL_CUMEM_ENABLE bypasses
             * the strict driver-version check on stacks where the HIP
             * runtime version is below the documented minimum. */
            (std::unordered_map<std::string, std::string>{
                {"NCCL_SYM_KERNEL", "AllGather_LL"},
                {"NCCL_CUMEM_ENABLE", "1"}})
        );
    }
}
