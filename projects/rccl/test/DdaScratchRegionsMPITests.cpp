/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaScratchRegionsMPITests.cpp
 * @brief MPI end-to-end test that DDA LL never reads another protocol's data as its own flags
 *
 * The LL receivers accept a 16B line as soon as both of its flag words equal the call's epoch. When LL and LL128
 * shared one scratch, an LL128 call two epochs earlier left its payload in the bank the next-but-one LL call polls,
 * so payload words equal to that LL epoch passed for flags. Each protocol now has its own scratch region.
 *
 * The test makes that deterministic: an LL128 AllReduce whose every 32-bit word equals the epoch of the LL call two
 * calls later, a filler LL call to step the epoch, then the checked LL call with rank 0 held back on the GPU so the
 * other ranks poll rank 0's slot before rank 0 writes it.
 */

#ifdef MPI_TESTS_ENABLED

#include "DdaFabricMPITestFixture.hpp"
#include "DeviceBufferHelpers.hpp"
#include "ResourceGuards.hpp"
#include "rccl_common.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace
{
constexpr size_t kKiB          = 1ull << 10;
constexpr size_t kMiB          = 1ull << 20;
constexpr size_t kMaxLL128     = 4 * kMiB;  // total
constexpr size_t kMinLL128     = 64 * kKiB; // must cover the LL call's slot 0 with LL128 payload
constexpr size_t kMaxLL        = 64 * kKiB;
constexpr size_t kMinLL        = 1 * kKiB;
constexpr int    kIterations   = 8;
constexpr int    kHoldSleeps   = 20000; // s_sleep 127 is 127 * 64 clocks: tens of ms in all

// Holds the stream, so the LL call queued behind it starts late on this rank. A fixed count of sleeps rather than a
// clock read, so the hold always ends.
__global__ void ddaRegionsHoldKernel(int sleeps)
{
    for(int i = 0; i < sleeps; ++i)
        __builtin_amdgcn_s_sleep(127);
}

// AllReduce input: small integers, so the float sum is exact.
float llInput(int rank, size_t i, int iter)
{
    return static_cast<float>(rank + 1 + static_cast<int>((i + iter) % 7));
}
} // namespace

class DdaScratchRegionsMPI : public DdaFabricMPITest
{
protected:
    int rank_{};
    int nRanks_{};

    void SetUp() override
    {
        DdaFabricMPITest::SetUp();
        if(!validateTestPrerequisites(kMinProcessesForMPI))
            GTEST_SKIP() << "Need at least 2 MPI ranks";
        DDA_FABRIC_SKIP_OR_FAIL(archSkipReason("DDA fabric requires gfx1250"));
        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(getActiveCommunicator(), &rank_));
        ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(getActiveCommunicator(), &nRanks_));
        DDA_FABRIC_SKIP_OR_FAIL(fabricInitSkipReason());
    }

    // Largest float size from maxBytes down to minBytes, halving, that selects algo; 0 if none does.
    size_t largestSelected(int algo, size_t maxBytes, size_t minBytes, const void* send, void* recv)
    {
        for(size_t bytes = maxBytes; bytes >= minBytes; bytes /= 2)
            if(selectedAlgo(ncclFuncAllReduce, bytes / sizeof(float), ncclFloat32, send, recv) == algo)
                return bytes;
        return 0;
    }

    // The flag the last LL/LL128 call used. Every launch leaves all epoch cells equal.
    uint32_t lastEpoch()
    {
        uint32_t epoch = 0;
        EXPECT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(&epoch, getActiveCommunicator()->ddaLLEpochDev, sizeof(epoch), hipMemcpyDeviceToHost));
        return epoch;
    }
};

TEST_F(DdaScratchRegionsMPI, AllReduceLLIgnoresLL128PayloadEqualToItsEpoch)
{
    void *ll128Send = nullptr, *ll128Recv = nullptr, *llSend = nullptr, *llRecv = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&ll128Send, kMaxLL128));
    DeviceBufferAutoGuard ll128SendGuard(ll128Send);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&ll128Recv, kMaxLL128));
    DeviceBufferAutoGuard ll128RecvGuard(ll128Recv);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&llSend, kMaxLL));
    DeviceBufferAutoGuard llSendGuard(llSend);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&llRecv, kMaxLL));
    DeviceBufferAutoGuard llRecvGuard(llRecv);

    const size_t ll128Bytes = largestSelected(RCCL_DDA_FABRIC_LL128, kMaxLL128, kMinLL128, ll128Send, ll128Recv);
    // The smallest LL size keeps the call one-shot, whose slot for rank 0 starts at the bank base.
    size_t llBytes = 0;
    for(size_t bytes = kMinLL; bytes <= kMaxLL && llBytes == 0; bytes *= 2)
        if(selectedAlgo(ncclFuncAllReduce, bytes / sizeof(float), ncclFloat32, llSend, llRecv) == RCCL_DDA_FABRIC_LL)
            llBytes = bytes;
    const std::string skip = mpiCoordinatedSkipReason(ll128Bytes == 0 || llBytes == 0,
                                                      "AllReduce LL128 or LL is not selected at any size");
    if(!skip.empty())
        GTEST_SKIP() << skip;

    ncclComm_t  comm     = getActiveCommunicator();
    hipStream_t stream   = getActiveStream();
    const size_t llCount = llBytes / sizeof(float);
    const int    nr      = nRanks_;
    for(int iter = 0; iter < kIterations; ++iter)
    {
        // The LL128 call takes flag E+1, the filler LL call E+2 (the other bank), and the checked call E+3, which
        // polls the bank the LL128 call wrote.
        const uint32_t checkedFlag = lastEpoch() + 3;
        SCOPED_TRACE("iteration " + std::to_string(iter) + ", LL128 " + std::to_string(ll128Bytes) + " B, LL " +
                     std::to_string(llBytes) + " B, flag " + std::to_string(checkedFlag));
        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<uint32_t>(ll128Send, ll128Bytes / sizeof(uint32_t),
                                                                        [checkedFlag](size_t) { return checkedFlag; }));
        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(llSend, llCount, [this, iter](size_t i) {
                          return llInput(rank_, i, iter);
                      }));
        ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(llRecv, 0, llBytes, stream));

        ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(ll128Send, ll128Recv, ll128Bytes / sizeof(float), ncclFloat32,
                                                 ncclSum, comm, stream));
        ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(llSend, llRecv, llCount, ncclFloat32, ncclSum, comm, stream));
        ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(llRecv, 0, llBytes, stream));
        if(rank_ == 0)
        {
            hipLaunchKernelGGL(ddaRegionsHoldKernel, dim3(1), dim3(1), 0, stream, kHoldSleeps);
            // Local check: the ASSERT_MPI_* macros are collective, and only rank 0 runs this branch.
            EXPECT_EQ(hipSuccess, hipGetLastError());
        }
        ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(llSend, llRecv, llCount, ncclFloat32, ncclSum, comm, stream));
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
        EXPECT_EQ(lastEpoch(), checkedFlag) << "the LL calls did not take the expected flags";

        // Collective: the ranks that poll ahead of rank 0 are the ones that see stale lines, and only rank 0 prints.
        const bool exact = verifyBufferData<float>(llRecv, llCount, [nr, iter](size_t i) {
            float sum = 0;
            for(int r = 0; r < nr; ++r)
                sum += llInput(r, i, iter);
            return sum;
        });
        ASSERT_MPI_TRUE(exact); // false: AllReduce LL took stale LL128 payload for its flags
    }
}

#endif // MPI_TESTS_ENABLED
