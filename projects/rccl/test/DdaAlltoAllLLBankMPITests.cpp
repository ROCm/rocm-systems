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
 * and checks every element of both. A second test interleaves AlltoAll LL with AllGather LL128 and
 * AllReduce LL, the other tiers that share the banks and the epoch on the same communicator.
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
constexpr size_t kMiB          = 1ull << 20;
constexpr size_t kMaxArBytes   = 256 * kMiB; // total
constexpr size_t kMaxA2aBytes  = 2 * kMiB;   // per peer
constexpr size_t kMinA2aBytes  = 1024;
constexpr size_t kMaxArLLBytes = 64 * kMiB;  // total
constexpr size_t kMaxAgPerRank = 16 * kMiB;  // per rank
constexpr size_t kMaxAgBytes   = 128 * kMiB; // gathered total
constexpr size_t kMinAgPerRank = 1024;
constexpr int    kIterations   = 20;

// AllReduce input: small integers, so the float sum is exact.
float arInput(int rank, size_t i, int iter = 0)
{
    return static_cast<float>(rank + 1 + static_cast<int>((i + iter) % 7));
}

// Byte i of the chunk rank s sends to rank d; differs per pair, iteration and index.
uint8_t chunkPattern(int src, int dst, int iter, size_t i)
{
    return static_cast<uint8_t>(src * 31 + dst * 7 + iter * 13 + i);
}

// Byte i of the block rank r contributes to an AllGather; differs per rank, iteration and index.
uint8_t gatherPattern(int rank, int iter, size_t i)
{
    return static_cast<uint8_t>(rank * 17 + iter * 5 + i * 3);
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

    // Largest float size from maxBytes down to minBytes that selects algo, stepping down by step,
    // or halving when step is 0; 0 if none does. minBytes must be at least step.
    size_t largestSelected(ncclFunc_t func, int algo, size_t maxBytes, size_t minBytes, size_t step,
                           const void* send, void* recv)
    {
        for(size_t bytes = maxBytes; bytes >= minBytes; bytes = step ? bytes - step : bytes / 2)
            if(selectedAlgo(func, bytes / sizeof(float), send, recv) == algo)
                return bytes;
        return 0;
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
    const size_t arBytes =
        largestSelected(ncclFuncAllReduce, RCCL_DDA_FABRIC_LL128, kMaxArBytes, kMiB, kMiB, arSend, arRecv);
    const size_t a2aBytes =
        largestSelected(ncclFuncAlltoAll, RCCL_DDA_FABRIC_LL, kMaxA2aBytes, kMinA2aBytes, 0, a2aSend, a2aRecv);
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

TEST_F(DdaAlltoAllLLBankMPI, InterleavedWithAllGatherLL128AndAllReduceLLKeepsData)
{
    const size_t nRanks     = static_cast<size_t>(nRanks_);
    size_t       maxAgBytes = kMaxAgPerRank;
    while(maxAgBytes * nRanks > kMaxAgBytes)
        maxAgBytes /= 2;
    void *agSend = nullptr, *agRecv = nullptr, *arSend = nullptr, *arRecv = nullptr;
    void *a2aSend = nullptr, *a2aRecv0 = nullptr, *a2aRecv1 = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&agSend, maxAgBytes));
    DeviceBufferAutoGuard agSendGuard(agSend);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&agRecv, maxAgBytes * nRanks));
    DeviceBufferAutoGuard agRecvGuard(agRecv);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&arSend, kMaxArLLBytes));
    DeviceBufferAutoGuard arSendGuard(arSend);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&arRecv, kMaxArLLBytes));
    DeviceBufferAutoGuard arRecvGuard(arRecv);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&a2aSend, kMaxA2aBytes * nRanks));
    DeviceBufferAutoGuard a2aSendGuard(a2aSend);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&a2aRecv0, kMaxA2aBytes * nRanks));
    DeviceBufferAutoGuard a2aRecv0Guard(a2aRecv0);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&a2aRecv1, kMaxA2aBytes * nRanks));
    DeviceBufferAutoGuard a2aRecv1Guard(a2aRecv1);

    // Each tier at its largest selected size, so its slots reach furthest into the banks.
    const size_t agBytes =
        largestSelected(ncclFuncAllGather, RCCL_DDA_FABRIC_LL128, maxAgBytes, kMinAgPerRank, 0, agSend, agRecv);
    size_t arBytes = largestSelected(ncclFuncAllReduce, RCCL_DDA_FABRIC_LL, kMaxArLLBytes, kMiB, kMiB, arSend, arRecv);
    if(arBytes == 0)
        arBytes = largestSelected(ncclFuncAllReduce, RCCL_DDA_FABRIC_LL, kMiB / 2, kMinA2aBytes, 0, arSend, arRecv);
    const size_t a2aBytes =
        largestSelected(ncclFuncAlltoAll, RCCL_DDA_FABRIC_LL, kMaxA2aBytes, kMinA2aBytes, 0, a2aSend, a2aRecv0);
    const std::string skip =
        mpiCoordinatedSkipReason(agBytes == 0 || arBytes == 0 || a2aBytes == 0,
                                 "AllGather LL128, AllReduce LL or AlltoAll LL is not selected at any size");
    if(!skip.empty())
        GTEST_SKIP() << skip;

    const size_t agTotal  = agBytes * nRanks;
    const size_t arCount  = arBytes / sizeof(float);
    const size_t a2aTotal = a2aBytes * nRanks;
    const size_t a2aCount = a2aBytes / sizeof(float);
    const int    nr       = nRanks_;

    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    for(int iter = 0; iter < kIterations; ++iter)
    {
        SCOPED_TRACE("iteration " + std::to_string(iter) + ", AllGather " + std::to_string(agBytes) +
                     " B per rank, AllReduce " + std::to_string(arBytes) + " B, AlltoAll " +
                     std::to_string(a2aBytes) + " B per peer");
        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<uint8_t>(agSend, agBytes, [this, iter](size_t i) {
                          return gatherPattern(rank_, iter, i);
                      }));
        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(arSend, arCount, [this, iter](size_t i) {
                          return arInput(rank_, i, iter);
                      }));
        ASSERT_MPI_EQ(hipSuccess,
                      initializeBufferWithPattern<uint8_t>(a2aSend, a2aTotal, [this, iter, a2aBytes](size_t i) {
                          return chunkPattern(rank_, static_cast<int>(i / a2aBytes), iter, i % a2aBytes);
                      }));
        ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(agRecv, 0, agTotal, stream));
        ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(arRecv, 0, arBytes, stream));
        ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(a2aRecv0, 0, a2aTotal, stream));
        ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(a2aRecv1, 0, a2aTotal, stream));
        ASSERT_MPI_EQ(ncclSuccess, ncclAllGather(agSend, agRecv, agBytes / sizeof(float), ncclFloat32, comm, stream));
        ASSERT_MPI_EQ(ncclSuccess, ncclAlltoAll(a2aSend, a2aRecv0, a2aCount, ncclFloat32, comm, stream));
        ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(arSend, arRecv, arCount, ncclFloat32, ncclSum, comm, stream));
        ASSERT_MPI_EQ(ncclSuccess, ncclAlltoAll(a2aSend, a2aRecv1, a2aCount, ncclFloat32, comm, stream));
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

        EXPECT_TRUE(verifyBufferData<uint8_t>(agRecv, agTotal, [iter, agBytes](size_t i) {
            return gatherPattern(static_cast<int>(i / agBytes), iter, i % agBytes);
        })) << "Rank " << rank_ << ": AllGather data mismatch";
        EXPECT_TRUE(verifyBufferData<float>(arRecv, arCount, [nr, iter](size_t i) {
            float sum = 0;
            for(int r = 0; r < nr; ++r)
                sum += arInput(r, i, iter);
            return sum;
        })) << "Rank " << rank_ << ": AllReduce data mismatch";
        for(void* a2aRecv : {a2aRecv0, a2aRecv1})
            EXPECT_TRUE(verifyBufferData<uint8_t>(a2aRecv, a2aTotal, [this, iter, a2aBytes](size_t i) {
                return chunkPattern(static_cast<int>(i / a2aBytes), rank_, iter, i % a2aBytes);
            })) << "Rank " << rank_ << ": AlltoAll data mismatch";
    }
}

#endif // MPI_TESTS_ENABLED
