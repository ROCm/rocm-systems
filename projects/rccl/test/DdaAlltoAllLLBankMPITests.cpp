/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaAlltoAllLLBankMPITests.cpp
 * @brief MPI end-to-end test that gfx1250 AlltoAll DDA LL stays out of the other tiers' bank 0
 *
 * Every DDA LL/LL128 tier double-buffers its scratch: a call writes bank (epoch & 1) of its peers
 * while a slower peer may still be reading the previous call from the other bank. AlltoAll LL used
 * to put bank 1 at a fixed nRanks * 32 MiB, which on the large DPX scratch falls inside bank 0,
 * where the two-shot AllReduce LL128 keeps its second-phase slots. A rank that finished such an
 * AllReduce could then write its AlltoAll packets over lines a peer had not read yet. The test
 * runs the largest AllReduce LL128 call, then AlltoAll LL twice so the AllReduce alternates banks,
 * and checks every element of both.
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
constexpr size_t kMiB         = 1ull << 20;
constexpr size_t kMaxArBytes  = 256 * kMiB; // total
constexpr size_t kMaxA2aBytes = 2 * kMiB;   // per peer
constexpr size_t kMinA2aBytes = 1024;
constexpr int    kIterations  = 20;

// AllReduce input: small integers, so the float sum is exact.
float arInput(int rank, size_t i)
{
    return static_cast<float>(rank + 1 + static_cast<int>(i % 7));
}

// Byte i of the chunk rank s sends to rank d; differs per pair, iteration and index.
uint8_t chunkPattern(int src, int dst, int iter, size_t i)
{
    return static_cast<uint8_t>(src * 31 + dst * 7 + iter * 13 + i);
}
} // namespace

class DdaAlltoAllLLBankMPI : public DdaFabricMPITest
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

    int selectedAlgo(ncclFunc_t func, size_t count, const void* send, void* recv)
    {
        int algo = -1, protocol = -1, maxChannels = -1;
        EXPECT_EQ(ncclSuccess, rcclGetCollImplInfo(getActiveCommunicator(), func, count, ncclFloat32, ncclSum, send,
                                                   recv, /*graphCapturing=*/0, &algo, &protocol, &maxChannels));
        return algo;
    }
};

TEST_F(DdaAlltoAllLLBankMPI, AlltoAllLLAfterAllReduceLL128KeepsData)
{
    const size_t nRanks = static_cast<size_t>(nRanks_);
    void *arSend = nullptr, *arRecv = nullptr, *a2aSend = nullptr, *a2aRecv = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&arSend, kMaxArBytes));
    DeviceBufferAutoGuard arSendGuard(arSend);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&arRecv, kMaxArBytes));
    DeviceBufferAutoGuard arRecvGuard(arRecv);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&a2aSend, kMaxA2aBytes * nRanks));
    DeviceBufferAutoGuard a2aSendGuard(a2aSend);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&a2aRecv, kMaxA2aBytes * nRanks));
    DeviceBufferAutoGuard a2aRecvGuard(a2aRecv);

    // The largest AllReduce LL128 call fills its slots furthest into bank 0, so step down by 1 MiB.
    size_t arBytes = 0;
    for(size_t bytes = kMaxArBytes; bytes >= kMiB && arBytes == 0; bytes -= kMiB)
        if(selectedAlgo(ncclFuncAllReduce, bytes / sizeof(float), arSend, arRecv) == RCCL_DDA_FABRIC_LL128)
            arBytes = bytes;
    size_t a2aBytes = 0;
    for(size_t bytes = kMaxA2aBytes; bytes >= kMinA2aBytes && a2aBytes == 0; bytes /= 2)
        if(selectedAlgo(ncclFuncAlltoAll, bytes / sizeof(float), a2aSend, a2aRecv) == RCCL_DDA_FABRIC_LL)
            a2aBytes = bytes;
    const std::string skip = mpiCoordinatedSkipReason(arBytes == 0 || a2aBytes == 0,
                                                      "AllReduce LL128 or AlltoAll LL is not selected at any size");
    if(!skip.empty())
        GTEST_SKIP() << skip;

    const size_t arCount  = arBytes / sizeof(float);
    const size_t a2aTotal = a2aBytes * nRanks;
    const size_t a2aCount = a2aBytes / sizeof(float);
    const int    nr       = nRanks_;
    ASSERT_MPI_EQ(hipSuccess,
                  initializeBufferWithPattern<float>(arSend, arCount, [this](size_t i) { return arInput(rank_, i); }));

    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    for(int iter = 0; iter < kIterations; ++iter)
    {
        SCOPED_TRACE("iteration " + std::to_string(iter) + ", AllReduce " + std::to_string(arBytes) +
                     " B, AlltoAll " + std::to_string(a2aBytes) + " B per peer");
        ASSERT_MPI_EQ(hipSuccess,
                      initializeBufferWithPattern<uint8_t>(a2aSend, a2aTotal, [this, iter, a2aBytes](size_t i) {
                          return chunkPattern(rank_, static_cast<int>(i / a2aBytes), iter, i % a2aBytes);
                      }));
        ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(arRecv, 0, arBytes, stream));
        ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(arSend, arRecv, arCount, ncclFloat32, ncclSum, comm, stream));
        ASSERT_MPI_EQ(ncclSuccess, ncclAlltoAll(a2aSend, a2aRecv, a2aCount, ncclFloat32, comm, stream));
        ASSERT_MPI_EQ(ncclSuccess, ncclAlltoAll(a2aSend, a2aRecv, a2aCount, ncclFloat32, comm, stream));
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

        EXPECT_TRUE(verifyBufferData<float>(arRecv, arCount, [nr](size_t i) {
            float sum = 0;
            for(int r = 0; r < nr; ++r)
                sum += arInput(r, i);
            return sum;
        })) << "Rank " << rank_ << ": AllReduce data mismatch";
        EXPECT_TRUE(verifyBufferData<uint8_t>(a2aRecv, a2aTotal, [this, iter, a2aBytes](size_t i) {
            return chunkPattern(static_cast<int>(i / a2aBytes), rank_, iter, i % a2aBytes);
        })) << "Rank " << rank_ << ": AlltoAll data mismatch";
    }
}

#endif // MPI_TESTS_ENABLED
