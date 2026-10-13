/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaFp8MPITests.cpp
 * @brief MPI end-to-end tests for fp8 (e4m3, e5m2) on every DDA fabric tier
 *
 * Each fixture pins one tier through the DDA threshold parameters and runs every
 * collective that tier serves in both fp8 formats. A case checks that RCCL
 * selected the tier for the fp8 operands, that the data is exact, and for
 * AllReduce that the COLL log names the one-shot / two-shot variant.
 *
 * Reductions use VerifiableFp8 data, whose partial sums are exactly
 * representable, so any reduction order has one correct answer and the check
 * is bit-exact. AllGather and AlltoAll move bytes, so their inputs cover all 256
 * encodings, NaNs included, and the check is a byte compare.
 *
 * RCCL_PARAM caches on first read, per process, so each fixture has to run in
 * its own process:
 *
 *   mpirun -np 4 ./rccl-UnitTestsMPI --gtest_filter='DdaFp8MPI_LL.*'
 *   mpirun -np 4 ./rccl-UnitTestsMPI --gtest_filter='DdaFp8MPI_LLTwoShot.*'
 *   mpirun -np 4 ./rccl-UnitTestsMPI --gtest_filter='DdaFp8MPI_LL128.*'
 *   mpirun -np 4 ./rccl-UnitTestsMPI --gtest_filter='DdaFp8MPI_LL128TwoShot.*'
 *   mpirun -np 4 ./rccl-UnitTestsMPI --gtest_filter='DdaFp8MPI_Vmm.*'
 */

#ifdef MPI_TESTS_ENABLED

#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "VerifiableFp8.hpp"
#include "comm.h" // internal: struct ncclComm::ddaFabricBarrierState
#include "rccl_common.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace
{
constexpr ncclDataType_t kFp8Types[] = {ncclFloat8e4m3, ncclFloat8e5m2};
constexpr uint64_t       kSeed       = 0xdda0f8;

// Every DDA gate admits only whole 16-byte lines. A prime number of lines
// divides evenly among no block, chunk, or shard count, so it runs the kernels'
// partial-tail paths that power-of-two sizes skip.
constexpr size_t kLine       = 16;
constexpr size_t kPrimeLines = 251 * kLine;

const char* fp8Name(ncclDataType_t datatype)
{
    return datatype == ncclFloat8e4m3 ? "fp8_e4m3" : "fp8_e5m2";
}

bool isGfx1250Device()
{
    hipDeviceProp_t props{};
    return hipGetDeviceProperties(&props, 0) == hipSuccess
        && std::string(props.gcnArchName).find("gfx1250") != std::string::npos;
}

// The two-shot tiers split the message into one shard per rank, each a whole
// number of 16-byte lines, so the count has to be a multiple of 16 * nRanks.
size_t alignToShards(size_t count, int nRanks)
{
    const size_t unit = 16 * static_cast<size_t>(nRanks);
    return (count + unit - 1) / unit * unit;
}

// One byte per element; varies with the rank and the index so a misplaced
// chunk does not match, and walks through all 256 encodings.
uint8_t copyPattern(int rank, size_t i)
{
    return static_cast<uint8_t>(rank * 37 + i);
}

std::string pathNeedle(const char* path, int nRanks, size_t count, ncclDataType_t datatype)
{
    char buf[160];
    std::snprintf(buf,
                  sizeof(buf),
                  "taking DDA fabric %s path: nRanks=%d nNodes=1 count=%zu datatype=%d",
                  path,
                  nRanks,
                  count,
                  static_cast<int>(datatype));
    return buf;
}
} // namespace

class DdaFp8MPITest : public MPITestBase
{
protected:
    std::vector<std::unique_ptr<MPIHelpers::MpiEnvGuard>> envGuards_;
    std::unique_ptr<MPIHelpers::TestLogAssertionContext>  logCtx_;
    int                                                    rank_{};
    int                                                    nRanks_{};

    // RCCL_DDA_* settings that leave this fixture's tier as the one claiming the
    // sizes its tests use.
    virtual std::vector<std::pair<const char*, const char*>> tierEnv() const = 0;

    void SetUp() override
    {
        MPITestBase::SetUp();
        envGuards_.push_back(std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG", "INFO"));
        envGuards_.push_back(
            std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG_SUBSYS", "INIT,COLL"));
        for(const auto& [name, value] : tierEnv())
            envGuards_.push_back(std::make_unique<MPIHelpers::MpiEnvGuard>(name, value));
        logCtx_ = std::make_unique<MPIHelpers::TestLogAssertionContext>(
            MPIHelpers::makeCombinedAssertionLogOptions(getTestMpiRank()));

        if(!validateTestPrerequisites(kMinProcessesForMPI))
            GTEST_SKIP() << "Need at least 2 MPI ranks";
        // Every rank must skip together, or the rest would block in the next collective.
        const std::string archSkip
            = mpiCoordinatedSkipReason(!isGfx1250Device(), "DDA fabric requires gfx1250");
        if(!archSkip.empty())
            GTEST_SKIP() << archSkip;

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(getActiveCommunicator(), &rank_));
        ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(getActiveCommunicator(), &nRanks_));

        const std::string fabricSkip
            = mpiCoordinatedSkipReason(getActiveCommunicator()->ddaFabricBarrierState == nullptr,
                                       "DDA fabric path did not initialize");
        if(!fabricSkip.empty())
            GTEST_SKIP() << fabricSkip;
    }

    void TearDown() override
    {
        MPITestBase::TearDown();
        logCtx_.reset();
        while(!envGuards_.empty())
            envGuards_.pop_back();
    }

    void expectAlgo(ncclFunc_t     func,
                    size_t         count,
                    ncclDataType_t datatype,
                    const void*    sendBuf,
                    void*          recvBuf,
                    int            expectedAlgo)
    {
        int algo = -1, protocol = -1, maxChannels = -1;
        EXPECT_EQ(ncclSuccess,
                  rcclGetCollImplInfo(getActiveCommunicator(), func, count, datatype, ncclSum,
                                      sendBuf, recvBuf, /*graphCapturing=*/0, &algo, &protocol,
                                      &maxChannels));
        const char* algoName     = nullptr;
        const char* expectedName = nullptr;
        rcclGetAlgoName(algo, &algoName);
        rcclGetAlgoName(expectedAlgo, &expectedName);
        EXPECT_EQ(algo, expectedAlgo)
            << "Rank " << rank_ << ": " << fp8Name(datatype) << " count=" << count << " selected "
            << (algoName ? algoName : "?") << ", expected " << (expectedName ? expectedName : "?");
    }

    void expectLog(const std::string& needle)
    {
        const std::string merged = logCtx_->readNcclDebugLog() + logCtx_->readPerRankStderrLog();
        EXPECT_NE(merged.find(needle), std::string::npos)
            << "Rank " << rank_ << " did not log: " << needle;
    }

    // Exact-count check of a reduced fp8 buffer; eltIx0 places it in the full vector.
    void expectReduced(const void* results, size_t count, ncclDataType_t datatype, size_t eltIx0)
    {
        int64_t* badEltN = nullptr;
        ASSERT_EQ(hipSuccess, hipHostMalloc(&badEltN, sizeof(*badEltN)));
        *badEltN = -1;
        ASSERT_EQ(hipSuccess,
                  RcclUnitTesting::VerifiableFp8Verify(results, count, datatype, ncclSum, nRanks_,
                                                       kSeed, eltIx0, badEltN, getActiveStream()));
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
        EXPECT_EQ(*badEltN, 0) << "Rank " << rank_ << ": " << fp8Name(datatype) << " has "
                               << *badEltN << " of " << count << " elements wrong";
        EXPECT_EQ(hipSuccess, hipHostFree(badEltN));
    }

    void runAllReduce(ncclDataType_t datatype,
                      size_t         count,
                      int            expectedAlgo,
                      const char*    path = nullptr)
    {
        void* sendBuf = nullptr;
        void* recvBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, count));
        DeviceBufferAutoGuard sendGuard(sendBuf);
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, count));
        DeviceBufferAutoGuard recvGuard(recvBuf);

        expectAlgo(ncclFuncAllReduce, count, datatype, sendBuf, recvBuf, expectedAlgo);

        ASSERT_MPI_EQ(hipSuccess,
                      RcclUnitTesting::VerifiableFp8PrepareInput(sendBuf, count, datatype, ncclSum,
                                                                 nRanks_, rank_, kSeed, 0,
                                                                 getActiveStream()));
        ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(recvBuf, 0, count, getActiveStream()));
        ASSERT_MPI_EQ(ncclSuccess,
                      ncclAllReduce(sendBuf, recvBuf, count, datatype, ncclSum,
                                    getActiveCommunicator(), getActiveStream()));
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

        expectReduced(recvBuf, count, datatype, 0);
        if(path != nullptr)
            expectLog(pathNeedle(path, nRanks_, count, datatype));
    }

    void runReduceScatter(ncclDataType_t datatype, size_t recvcount, int expectedAlgo)
    {
        const size_t totalCount = recvcount * static_cast<size_t>(nRanks_);
        void*        sendBuf    = nullptr;
        void*        recvBuf    = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, totalCount));
        DeviceBufferAutoGuard sendGuard(sendBuf);
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, recvcount));
        DeviceBufferAutoGuard recvGuard(recvBuf);

        expectAlgo(ncclFuncReduceScatter, recvcount, datatype, sendBuf, recvBuf, expectedAlgo);

        ASSERT_MPI_EQ(hipSuccess,
                      RcclUnitTesting::VerifiableFp8PrepareInput(sendBuf, totalCount, datatype,
                                                                 ncclSum, nRanks_, rank_, kSeed, 0,
                                                                 getActiveStream()));
        ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(recvBuf, 0, recvcount, getActiveStream()));
        ASSERT_MPI_EQ(ncclSuccess,
                      ncclReduceScatter(sendBuf, recvBuf, recvcount, datatype, ncclSum,
                                        getActiveCommunicator(), getActiveStream()));
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

        expectReduced(recvBuf, recvcount, datatype, static_cast<size_t>(rank_) * recvcount);
    }

    void runAllGather(ncclDataType_t datatype, size_t sendcount, int expectedAlgo)
    {
        const size_t totalCount = sendcount * static_cast<size_t>(nRanks_);
        void*        sendBuf    = nullptr;
        void*        recvBuf    = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, sendcount));
        DeviceBufferAutoGuard sendGuard(sendBuf);
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, totalCount));
        DeviceBufferAutoGuard recvGuard(recvBuf);

        expectAlgo(ncclFuncAllGather, sendcount, datatype, sendBuf, recvBuf, expectedAlgo);

        ASSERT_MPI_EQ(hipSuccess,
                      initializeBufferWithPattern<uint8_t>(
                          sendBuf, sendcount, [this](size_t i) { return copyPattern(rank_, i); }));
        ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, totalCount));
        ASSERT_MPI_EQ(ncclSuccess,
                      ncclAllGather(sendBuf, recvBuf, sendcount, datatype,
                                    getActiveCommunicator(), getActiveStream()));
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

        EXPECT_TRUE(verifyBufferData<uint8_t>(recvBuf, totalCount, [sendcount](size_t i) {
            return copyPattern(static_cast<int>(i / sendcount), i % sendcount);
        })) << "Rank " << rank_ << ": " << fp8Name(datatype) << " AllGather data mismatch";
    }

    void runAllToAll(ncclDataType_t datatype, size_t countPerPeer, int expectedAlgo)
    {
        const size_t totalCount = countPerPeer * static_cast<size_t>(nRanks_);
        void*        sendBuf    = nullptr;
        void*        recvBuf    = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, totalCount));
        DeviceBufferAutoGuard sendGuard(sendBuf);
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, totalCount));
        DeviceBufferAutoGuard recvGuard(recvBuf);

        expectAlgo(ncclFuncAlltoAll, countPerPeer, datatype, sendBuf, recvBuf, expectedAlgo);

        // Element i of the chunk rank s sends to rank d is copyPattern(s * nRanks + d, i).
        const int nRanks = nRanks_;
        ASSERT_MPI_EQ(hipSuccess,
                      initializeBufferWithPattern<uint8_t>(
                          sendBuf, totalCount, [this, nRanks, countPerPeer](size_t i) {
                              const int dest = static_cast<int>(i / countPerPeer);
                              return copyPattern(rank_ * nRanks + dest, i % countPerPeer);
                          }));
        ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, totalCount));
        ASSERT_MPI_EQ(ncclSuccess,
                      ncclAlltoAll(sendBuf, recvBuf, countPerPeer, datatype,
                                   getActiveCommunicator(), getActiveStream()));
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

        EXPECT_TRUE(verifyBufferData<uint8_t>(
            recvBuf, totalCount, [this, nRanks, countPerPeer](size_t i) {
                const int src = static_cast<int>(i / countPerPeer);
                return copyPattern(src * nRanks + rank_, i % countPerPeer);
            }))
            << "Rank " << rank_ << ": " << fp8Name(datatype) << " AlltoAll data mismatch";
    }

    // Runs `body` for each fp8 format at each count. The ASSERT_MPI_* checks fail
    // or skip every rank together, so stopping at the first of either keeps the
    // ranks in step.
    template<typename Body>
    void forEachFp8AndCount(std::initializer_list<size_t> counts, Body body)
    {
        for(ncclDataType_t datatype : kFp8Types)
        {
            for(size_t count : counts)
            {
                SCOPED_TRACE(std::string(fp8Name(datatype)) + " count=" + std::to_string(count));
                body(datatype, count);
                if(HasFatalFailure() || IsSkipped())
                    return;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// LL: the defaults. Every size here is inside every collective's LL window, and
// AllReduce is inside the 1 MiB one-shot threshold.
// ---------------------------------------------------------------------------

class DdaFp8MPI_LL : public DdaFp8MPITest
{
protected:
    std::vector<std::pair<const char*, const char*>> tierEnv() const override
    {
        return {};
    }
};

TEST_F(DdaFp8MPI_LL, AllReduceOneShot)
{
    forEachFp8AndCount({kLine, kPrimeLines, 64 * 1024}, [this](ncclDataType_t datatype, size_t count) {
        runAllReduce(datatype, count, RCCL_DDA_FABRIC_LL, "LL one-shot");
    });
}

TEST_F(DdaFp8MPI_LL, ReduceScatter)
{
    forEachFp8AndCount({kLine, kPrimeLines, 4 * 1024}, [this](ncclDataType_t datatype, size_t count) {
        runReduceScatter(datatype, count, RCCL_DDA_FABRIC_LL);
    });
}

TEST_F(DdaFp8MPI_LL, AllGather)
{
    forEachFp8AndCount({kLine, kPrimeLines, 4 * 1024}, [this](ncclDataType_t datatype, size_t count) {
        runAllGather(datatype, count, RCCL_DDA_FABRIC_LL);
    });
}

TEST_F(DdaFp8MPI_LL, AllToAll)
{
    forEachFp8AndCount({kLine, kPrimeLines, 1024}, [this](ncclDataType_t datatype, size_t count) {
        runAllToAll(datatype, count, RCCL_DDA_FABRIC_LL);
    });
}

// ---------------------------------------------------------------------------
// LL two-shot: a zero one-shot threshold hands every AllReduce size to two-shot.
// ---------------------------------------------------------------------------

class DdaFp8MPI_LLTwoShot : public DdaFp8MPITest
{
protected:
    std::vector<std::pair<const char*, const char*>> tierEnv() const override
    {
        return {{"RCCL_DDA_LL_ONESHOT_THRESHOLD", "0"}};
    }
};

TEST_F(DdaFp8MPI_LLTwoShot, AllReduce)
{
    const size_t ranks = static_cast<size_t>(nRanks_);
    forEachFp8AndCount({kLine * ranks, kPrimeLines * ranks, alignToShards(64 * 1024, nRanks_)},
                       [this](ncclDataType_t datatype, size_t count) {
                           runAllReduce(datatype, count, RCCL_DDA_FABRIC_LL, "LL two-shot");
                       });
}

// ---------------------------------------------------------------------------
// LL128: a zero LL threshold takes the LL tier out of every collective, leaving
// LL128 the first claimant. AllReduce is inside the 4 MiB one-shot threshold.
// ---------------------------------------------------------------------------

class DdaFp8MPI_LL128 : public DdaFp8MPITest
{
protected:
    std::vector<std::pair<const char*, const char*>> tierEnv() const override
    {
        return {{"RCCL_DDA_LL_THRESHOLD", "0"}};
    }
};

TEST_F(DdaFp8MPI_LL128, AllReduceOneShot)
{
    forEachFp8AndCount({kLine, kPrimeLines, 64 * 1024}, [this](ncclDataType_t datatype, size_t count) {
        runAllReduce(datatype, count, RCCL_DDA_FABRIC_LL128, "LL128 one-shot");
    });
}

TEST_F(DdaFp8MPI_LL128, ReduceScatter)
{
    forEachFp8AndCount({kLine, kPrimeLines, 4 * 1024}, [this](ncclDataType_t datatype, size_t count) {
        runReduceScatter(datatype, count, RCCL_DDA_FABRIC_LL128);
    });
}

TEST_F(DdaFp8MPI_LL128, AllGather)
{
    forEachFp8AndCount({kLine, kPrimeLines, 4 * 1024}, [this](ncclDataType_t datatype, size_t count) {
        runAllGather(datatype, count, RCCL_DDA_FABRIC_LL128);
    });
}

TEST_F(DdaFp8MPI_LL128, AllToAll)
{
    forEachFp8AndCount({kLine, kPrimeLines, 1024}, [this](ncclDataType_t datatype, size_t count) {
        runAllToAll(datatype, count, RCCL_DDA_FABRIC_LL128);
    });
}

// ---------------------------------------------------------------------------
// LL128 two-shot: zeroing the LL128 one-shot threshold as well leaves two-shot.
// ---------------------------------------------------------------------------

class DdaFp8MPI_LL128TwoShot : public DdaFp8MPITest
{
protected:
    std::vector<std::pair<const char*, const char*>> tierEnv() const override
    {
        return {{"RCCL_DDA_LL_THRESHOLD", "0"}, {"RCCL_DDA_LL128_ONESHOT_THRESHOLD", "0"}};
    }
};

TEST_F(DdaFp8MPI_LL128TwoShot, AllReduce)
{
    const size_t ranks = static_cast<size_t>(nRanks_);
    forEachFp8AndCount({kLine * ranks, kPrimeLines * ranks, alignToShards(64 * 1024, nRanks_)},
                       [this](ncclDataType_t datatype, size_t count) {
                           runAllReduce(datatype, count, RCCL_DDA_FABRIC_LL128, "LL128 two-shot");
                       });
}

// ---------------------------------------------------------------------------
// VMM (SIMPLE): RCCL_DDA_LL=0 takes out both packet tiers, and a 64 MiB DDA
// threshold opens the VMM tier for the collectives the gfx1250 table caps at 0.
// AllReduce runs once under the 256 KiB flat/tree split and once over it.
// ---------------------------------------------------------------------------

class DdaFp8MPI_Vmm : public DdaFp8MPITest
{
protected:
    std::vector<std::pair<const char*, const char*>> tierEnv() const override
    {
        return {{"RCCL_DDA_LL", "0"}, {"RCCL_DDA_THRESHOLD", "67108864"}};
    }
};

TEST_F(DdaFp8MPI_Vmm, AllReduceFlat)
{
    forEachFp8AndCount({kLine, kPrimeLines, 64 * 1024}, [this](ncclDataType_t datatype, size_t count) {
        runAllReduce(datatype, count, RCCL_DDA_FABRIC_VMM);
    });
}

TEST_F(DdaFp8MPI_Vmm, AllReduceTree)
{
    // 16411 lines per shard is prime and, for any rank count, over the split.
    const size_t primeShardCount = 16411 * kLine * static_cast<size_t>(nRanks_);
    forEachFp8AndCount({primeShardCount, alignToShards(1024 * 1024, nRanks_)},
                       [this](ncclDataType_t datatype, size_t count) {
                           runAllReduce(datatype, count, RCCL_DDA_FABRIC_VMM);
                       });
}

TEST_F(DdaFp8MPI_Vmm, ReduceScatter)
{
    forEachFp8AndCount({kLine, kPrimeLines, 16 * 1024}, [this](ncclDataType_t datatype, size_t count) {
        runReduceScatter(datatype, count, RCCL_DDA_FABRIC_VMM);
    });
}

TEST_F(DdaFp8MPI_Vmm, AllGather)
{
    forEachFp8AndCount({kLine, kPrimeLines, 16 * 1024}, [this](ncclDataType_t datatype, size_t count) {
        runAllGather(datatype, count, RCCL_DDA_FABRIC_VMM);
    });
}

TEST_F(DdaFp8MPI_Vmm, AllToAll)
{
    forEachFp8AndCount({kLine, kPrimeLines, 4 * 1024}, [this](ncclDataType_t datatype, size_t count) {
        runAllToAll(datatype, count, RCCL_DDA_FABRIC_VMM);
    });
}

#endif // MPI_TESTS_ENABLED
