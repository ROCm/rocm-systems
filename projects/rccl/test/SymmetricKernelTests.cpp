/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Functional tests for the hybrid LSA+GIN symmetric kernels added across
// NCCL 2.29.2 (AllGather) and 2.29.7 (ReduceScatter), and for the
// NCCL_SYM_GIN_KERNELS_ENABLE env-var gate.
//
// What these tests do:
//
//   * Register input/output buffers as symmetric windows via
//     ncclCommWindowRegister - this is the entry point that drives the
//     built-in symmetric kernel selection in NCCL/RCCL.
//   * Run AllGather and ReduceScatter and verify correctness.
//   * Re-run with NCCL_SYM_GIN_KERNELS_ENABLE=0 to validate that the
//     fallback (non-symmetric) path still produces correct results.
//
// Skip behavior:
//   The symmetric kernels are only enabled on fully-connected NVL systems
//   (see the 2.29.2 "Enabled built-in symmetric kernels only on fully
//   connected nvlink systems" fix). When the topology is not supported,
//   the implementation transparently falls back; we therefore validate
//   correctness rather than algorithm selection.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "StandaloneUtils.hpp"

namespace RcclUnitTesting
{

static int gpuCount() {
    int n = 0;
    hipError_t err = hipGetDeviceCount(&n);
    return (err == hipSuccess) ? n : 0;
}

#define SKIP_IF_FEWER_THAN(N)                                                  \
    do {                                                                       \
        int avail = gpuCount();                                                \
        if (avail < (N)) {                                                     \
            GTEST_SKIP() << "Need at least " << (N) << " GPUs, have "          \
                         << avail;                                             \
            return;                                                            \
        }                                                                      \
    } while (0)

// Allocate a buffer suitable for symmetric registration. We use
// ncclMemAlloc rather than hipMalloc so that the allocation respects the
// alignment / addressing requirements imposed by the symmetric window
// machinery.
static void* allocSymBuffer(size_t bytes) {
    void* p = nullptr;
    NCCLCHECK(ncclMemAlloc(&p, bytes));
    return p;
}

// ---------------------------------------------------------------------------
// AllGather correctness using symmetric windows (2.29.2).
// ---------------------------------------------------------------------------

static bool runAllGatherSym(int nRanks, size_t countPerRank) {
    std::vector<ncclComm_t> comms(nRanks, nullptr);
    if (ncclCommInitAll(comms.data(), nRanks, nullptr) != ncclSuccess) {
        return false;
    }

    std::vector<float*>        devIn (nRanks, nullptr);
    std::vector<float*>        devOut(nRanks, nullptr);
    std::vector<ncclWindow_t>  winIn (nRanks, nullptr);
    std::vector<ncclWindow_t>  winOut(nRanks, nullptr);
    std::vector<hipStream_t>   stream(nRanks);
    const size_t totalCount = countPerRank * (size_t)nRanks;

    for (int r = 0; r < nRanks; ++r) {
        HIPCALL(hipSetDevice(r));
        devIn [r] = (float*)allocSymBuffer(countPerRank * sizeof(float));
        devOut[r] = (float*)allocSymBuffer(totalCount   * sizeof(float));
        HIPCALL(hipStreamCreate(&stream[r]));

        // Fill input with rank-distinguishable pattern.
        std::vector<float> host(countPerRank, (float)(r + 1));
        HIPCALL(hipMemcpy(devIn[r], host.data(),
                          countPerRank * sizeof(float),
                          hipMemcpyHostToDevice));
        HIPCALL(hipMemset(devOut[r], 0, totalCount * sizeof(float)));

        NCCLCHECK(ncclCommWindowRegister(comms[r], devIn[r],
                                         countPerRank * sizeof(float),
                                         &winIn[r], /*winFlags=*/0));
        NCCLCHECK(ncclCommWindowRegister(comms[r], devOut[r],
                                         totalCount * sizeof(float),
                                         &winOut[r], /*winFlags=*/0));
    }

    NCCLCHECK(ncclGroupStart());
    for (int r = 0; r < nRanks; ++r) {
        HIPCALL(hipSetDevice(r));
        NCCLCHECK(ncclAllGather(devIn[r], devOut[r], countPerRank,
                                ncclFloat32, comms[r], stream[r]));
    }
    NCCLCHECK(ncclGroupEnd());

    bool ok = true;
    for (int r = 0; r < nRanks; ++r) {
        HIPCALL(hipSetDevice(r));
        HIPCALL(hipStreamSynchronize(stream[r]));

        std::vector<float> host(totalCount);
        HIPCALL(hipMemcpy(host.data(), devOut[r],
                          totalCount * sizeof(float),
                          hipMemcpyDeviceToHost));

        for (int src = 0; src < nRanks; ++src) {
            float expected = (float)(src + 1);
            for (size_t i = 0; i < countPerRank; ++i) {
                size_t idx = src * countPerRank + i;
                if (host[idx] != expected) {
                    ok = false;
                }
            }
        }
    }

    for (int r = 0; r < nRanks; ++r) {
        HIPCALL(hipSetDevice(r));
        NCCLCHECK(ncclCommWindowDeregister(comms[r], winIn[r]));
        NCCLCHECK(ncclCommWindowDeregister(comms[r], winOut[r]));
        HIPCALL(hipStreamDestroy(stream[r]));
        NCCLCHECK(ncclMemFree(devIn[r]));
        NCCLCHECK(ncclMemFree(devOut[r]));
        NCCLCHECK(ncclCommDestroy(comms[r]));
    }
    return ok;
}

// ---------------------------------------------------------------------------
// ReduceScatter correctness using symmetric windows (2.29.7).
// ---------------------------------------------------------------------------

static bool runReduceScatterSym(int nRanks, size_t countPerRank) {
    std::vector<ncclComm_t> comms(nRanks, nullptr);
    if (ncclCommInitAll(comms.data(), nRanks, nullptr) != ncclSuccess) {
        return false;
    }

    std::vector<float*>       devIn (nRanks, nullptr);
    std::vector<float*>       devOut(nRanks, nullptr);
    std::vector<ncclWindow_t> winIn (nRanks, nullptr);
    std::vector<ncclWindow_t> winOut(nRanks, nullptr);
    std::vector<hipStream_t>  stream(nRanks);
    const size_t totalCount = countPerRank * (size_t)nRanks;

    for (int r = 0; r < nRanks; ++r) {
        HIPCALL(hipSetDevice(r));
        devIn [r] = (float*)allocSymBuffer(totalCount    * sizeof(float));
        devOut[r] = (float*)allocSymBuffer(countPerRank  * sizeof(float));
        HIPCALL(hipStreamCreate(&stream[r]));

        // Every rank contributes a value of 1.0f at every index; the sum
        // over `nRanks` ranks is therefore `nRanks` at every index.
        std::vector<float> host(totalCount, 1.0f);
        HIPCALL(hipMemcpy(devIn[r], host.data(),
                          totalCount * sizeof(float),
                          hipMemcpyHostToDevice));
        HIPCALL(hipMemset(devOut[r], 0, countPerRank * sizeof(float)));

        NCCLCHECK(ncclCommWindowRegister(comms[r], devIn[r],
                                         totalCount * sizeof(float),
                                         &winIn[r], 0));
        NCCLCHECK(ncclCommWindowRegister(comms[r], devOut[r],
                                         countPerRank * sizeof(float),
                                         &winOut[r], 0));
    }

    NCCLCHECK(ncclGroupStart());
    for (int r = 0; r < nRanks; ++r) {
        HIPCALL(hipSetDevice(r));
        NCCLCHECK(ncclReduceScatter(devIn[r], devOut[r], countPerRank,
                                    ncclFloat32, ncclSum,
                                    comms[r], stream[r]));
    }
    NCCLCHECK(ncclGroupEnd());

    bool ok = true;
    const float expected = (float)nRanks;
    for (int r = 0; r < nRanks; ++r) {
        HIPCALL(hipSetDevice(r));
        HIPCALL(hipStreamSynchronize(stream[r]));

        std::vector<float> host(countPerRank);
        HIPCALL(hipMemcpy(host.data(), devOut[r],
                          countPerRank * sizeof(float),
                          hipMemcpyDeviceToHost));
        for (size_t i = 0; i < countPerRank; ++i) {
            if (host[i] != expected) ok = false;
        }
    }

    for (int r = 0; r < nRanks; ++r) {
        HIPCALL(hipSetDevice(r));
        NCCLCHECK(ncclCommWindowDeregister(comms[r], winIn[r]));
        NCCLCHECK(ncclCommWindowDeregister(comms[r], winOut[r]));
        HIPCALL(hipStreamDestroy(stream[r]));
        NCCLCHECK(ncclMemFree(devIn[r]));
        NCCLCHECK(ncclMemFree(devOut[r]));
        NCCLCHECK(ncclCommDestroy(comms[r]));
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Individual tests
// ---------------------------------------------------------------------------

static void testSymAllGather_FourRanks() {
    SKIP_IF_FEWER_THAN(4);
    EXPECT_TRUE(runAllGatherSym(4, 4096));
}

static void testSymReduceScatter_FourRanks() {
    SKIP_IF_FEWER_THAN(4);
    EXPECT_TRUE(runReduceScatterSym(4, 4096));
}

// Same workload with the symmetric path explicitly disabled: must still
// produce correct results via the legacy path.
static void testSymKernelsDisabled_AllGather_StillCorrect() {
    SKIP_IF_FEWER_THAN(4);
    EXPECT_TRUE(runAllGatherSym(4, 4096));
}

static void testSymKernelsDisabled_ReduceScatter_StillCorrect() {
    SKIP_IF_FEWER_THAN(4);
    EXPECT_TRUE(runReduceScatterSym(4, 4096));
}

TEST(SymmetricKernel, ProcessIsolatedSuite)
{
    using TC = ProcessIsolatedTestRunner::TestConfig;
    RUN_ISOLATED_TESTS(
        TC("SymAllGather_FourRanks_Default",
            testSymAllGather_FourRanks),
        TC("SymReduceScatter_FourRanks_Default",
            testSymReduceScatter_FourRanks),
        TC("SymKernelsDisabled_AllGather_StillCorrect",
            testSymKernelsDisabled_AllGather_StillCorrect)
            .withEnvironment({{"NCCL_SYM_GIN_KERNELS_ENABLE", "0"}}),
        TC("SymKernelsDisabled_ReduceScatter_StillCorrect",
            testSymKernelsDisabled_ReduceScatter_StillCorrect)
            .withEnvironment({{"NCCL_SYM_GIN_KERNELS_ENABLE", "0"}})
    );
}

} // namespace RcclUnitTesting
