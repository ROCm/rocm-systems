/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Functional test for the scalable AllGatherV pattern added in NCCL 2.29.2.
//
// AllGatherV is not exposed as a dedicated NCCL public API; rather, the
// 2.29.2 release added a new internal scheduler path (group of broadcasts)
// and new kernels in src/device/all_gather_v.h that handle the
// unequal-counts-per-rank pattern more efficiently.
//
// We exercise it via the canonical client recipe:
//
//   ncclGroupStart();
//     for each rank r in [0, nRanks):
//       ncclBroadcast(... root=r ...);   // each rank broadcasts its slice
//   ncclGroupEnd();
//
// then verify that every rank received the concatenation of all rank
// contributions in rank order, with each rank contributing a different
// number of elements. The test is a single-process, multi-GPU test that
// uses the same threading pattern as the rest of the test suite.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
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

// Generates per-rank counts so that ranks have distinctly different sizes.
// Sum of counts and the offset table are returned in `outTotal` and
// `outOffsets`.
static std::vector<size_t> makeUnequalCounts(int nRanks,
                                             size_t base,
                                             size_t& outTotal,
                                             std::vector<size_t>& outOffsets) {
    std::vector<size_t> counts(nRanks);
    outOffsets.assign(nRanks, 0);
    outTotal = 0;
    for (int r = 0; r < nRanks; ++r) {
        counts[r] = base + (size_t)r * 137 + ((r & 1) ? 11 : 0);
        outOffsets[r] = outTotal;
        outTotal += counts[r];
    }
    return counts;
}

// Single per-rank worker: build a 1-of-N comm via ncclCommInitRankMulti is
// not directly used here; this test instead uses one comm initialized by
// ncclCommInitAll on the parent process.
struct AllGatherVRunResult {
    bool ok;
    std::string err;
};

static AllGatherVRunResult runAllGatherV(int nRanks, size_t baseCount) {
    AllGatherVRunResult result{false, ""};

    // 1. Create comms - one per GPU, all in this process.
    std::vector<ncclComm_t> comms(nRanks, nullptr);
    if (ncclCommInitAll(comms.data(), nRanks, /*devList=*/nullptr) != ncclSuccess) {
        result.err = "ncclCommInitAll failed";
        return result;
    }

    // 2. Generate unequal per-rank counts.
    size_t totalCount = 0;
    std::vector<size_t> offsets;
    std::vector<size_t> counts = makeUnequalCounts(nRanks, baseCount,
                                                   totalCount, offsets);

    // 3. Per-rank state.
    std::vector<int*>         devOut (nRanks, nullptr);
    std::vector<int*>         devIn  (nRanks, nullptr);
    std::vector<hipStream_t>  streams(nRanks);

    for (int r = 0; r < nRanks; ++r) {
        HIPCALL(hipSetDevice(r));
        HIPCALL(hipMalloc(&devOut[r], totalCount  * sizeof(int)));
        HIPCALL(hipMalloc(&devIn [r], counts[r]   * sizeof(int)));
        HIPCALL(hipStreamCreate(&streams[r]));

        // Each rank fills its input slice with a known pattern, so we can
        // tell who contributed what to the output.
        std::vector<int> host(counts[r]);
        for (size_t i = 0; i < counts[r]; ++i) {
            host[i] = (r + 1) * 10000 + (int)i;
        }
        HIPCALL(hipMemcpy(devIn[r], host.data(),
                          counts[r] * sizeof(int),
                          hipMemcpyHostToDevice));
        HIPCALL(hipMemset(devOut[r], 0, totalCount * sizeof(int)));
    }

    // 4. Group of per-rank broadcasts == the AllGatherV pattern.
    NCCLCHECK(ncclGroupStart());
    for (int root = 0; root < nRanks; ++root) {
        for (int r = 0; r < nRanks; ++r) {
            HIPCALL(hipSetDevice(r));
            void* sendbuf = (r == root) ? (void*)devIn[r] : nullptr;
            void* recvbuf = devOut[r] + offsets[root];
            NCCLCHECK(ncclBroadcast(sendbuf, recvbuf,
                                    counts[root], ncclInt32,
                                    /*root=*/root,
                                    comms[r], streams[r]));
        }
    }
    NCCLCHECK(ncclGroupEnd());

    // 5. Sync and validate every rank's full output buffer.
    bool allOk = true;
    for (int r = 0; r < nRanks; ++r) {
        HIPCALL(hipSetDevice(r));
        HIPCALL(hipStreamSynchronize(streams[r]));

        std::vector<int> host(totalCount);
        HIPCALL(hipMemcpy(host.data(), devOut[r],
                          totalCount * sizeof(int),
                          hipMemcpyDeviceToHost));

        for (int root = 0; root < nRanks; ++root) {
            for (size_t i = 0; i < counts[root]; ++i) {
                int expected = (root + 1) * 10000 + (int)i;
                int got      = host[offsets[root] + i];
                if (got != expected) {
                    if (allOk) {
                        result.err = "rank " + std::to_string(r)
                                   + " saw " + std::to_string(got)
                                   + " at offset "
                                   + std::to_string(offsets[root] + i)
                                   + " from root " + std::to_string(root)
                                   + ", expected " + std::to_string(expected);
                    }
                    allOk = false;
                }
            }
        }
    }

    // 6. Tear down.
    for (int r = 0; r < nRanks; ++r) {
        HIPCALL(hipSetDevice(r));
        HIPCALL(hipStreamDestroy(streams[r]));
        HIPCALL(hipFree(devIn[r]));
        HIPCALL(hipFree(devOut[r]));
        NCCLCHECK(ncclCommDestroy(comms[r]));
    }

    result.ok = allOk;
    return result;
}

static void testAllGatherV_TwoRanks_Correctness() {
    SKIP_IF_FEWER_THAN(2);
    auto r = runAllGatherV(/*nRanks=*/2, /*baseCount=*/64);
    EXPECT_TRUE(r.ok) << r.err;
}

static void testAllGatherV_FourRanks_Correctness() {
    SKIP_IF_FEWER_THAN(4);
    auto r = runAllGatherV(/*nRanks=*/4, /*baseCount=*/128);
    EXPECT_TRUE(r.ok) << r.err;
}

static void testAllGatherV_EightRanks_LargeCounts() {
    SKIP_IF_FEWER_THAN(8);
    // Larger counts exercise the new "scheduler path / new kernels" code
    // path described in the 2.29.2 release notes.
    auto r = runAllGatherV(/*nRanks=*/8, /*baseCount=*/4096);
    EXPECT_TRUE(r.ok) << r.err;
}

TEST(AllGatherV, ProcessIsolatedSuite)
{
    RUN_ISOLATED_TESTS(
        ProcessIsolatedTestRunner::TestConfig("AllGatherV_TwoRanks_Correctness",
            testAllGatherV_TwoRanks_Correctness),
        ProcessIsolatedTestRunner::TestConfig("AllGatherV_FourRanks_Correctness",
            testAllGatherV_FourRanks_Correctness),
        ProcessIsolatedTestRunner::TestConfig("AllGatherV_EightRanks_LargeCounts",
            testAllGatherV_EightRanks_LargeCounts)
    );
}

} // namespace RcclUnitTesting
