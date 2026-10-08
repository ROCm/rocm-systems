/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaIpcReduceScatterMPITests.cpp
 * @brief MPI end-to-end tests for the DDA IPC ReduceScatter kernel.
 *
 * Every rank sends index- and rank-dependent data, so each output slice is
 * distinct: reducing the wrong slice, or writing it to the wrong place, fails
 * the exact-value check. A uniform per-rank fill would sum to the same value
 * on every slice and hide such a defect.
 *
 * Run (example):
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=DdaIpcMPI_ReduceScatter.*
 */

#ifdef MPI_TESTS_ENABLED

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include <mpi.h>

#include <cstdlib>
#include <string>

#include "DeviceBufferHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "rccl_common.h"

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace {

// DDA IPC runs only on a single node with exactly 8 ranks.
constexpr int kDdaIpcRanks = 8;

// One block (the single-block grid path) and a multi-block 256 KiB shard.
constexpr size_t kSingleBlockCount = 1024;
constexpr size_t kMultiBlockCount = 64 * 1024;

bool isDdaIpcArch() {
  hipDeviceProp_t props{};
  if (hipGetDeviceProperties(&props, 0) != hipSuccess) {
    return false;
  }
  const std::string arch(props.gcnArchName);
  return arch.find("gfx942") != std::string::npos ||
         arch.find("gfx950") != std::string::npos;
}

// Env overrides that legitimately route ReduceScatter away from DDA IPC.
bool ddaRoutingOverridden() {
  const char* ddaEnable = std::getenv("RCCL_DDA_ENABLE");
  const char* implicit = std::getenv("NCCL_LAUNCH_ORDER_IMPLICIT");
  return (ddaEnable != nullptr && std::string(ddaEnable) == "0") ||
         (implicit != nullptr && std::string(implicit) == "1") ||
         std::getenv("NCCL_ALGO") != nullptr ||
         std::getenv("RCCL_DDA_THRESHOLD") != nullptr;
}

// Send value at global index j on rank r; exact in fp32 for these sizes.
float sendValue(int rank, size_t j) {
  return static_cast<float>(j + static_cast<size_t>(rank));
}

}  // namespace

class DdaIpcMPI_ReduceScatter : public MPITestBase {
 protected:
  int rank_ = 0;
  int nRanks_ = 0;

  void SetUp() override {
    MPITestBase::SetUp();
    if (!validateTestPrerequisites(kDdaIpcRanks, kDdaIpcRanks, false, 1, 1)) {
      GTEST_SKIP() << "DDA IPC needs exactly 8 ranks on one node";
    }
    const bool localSkip = !isDdaIpcArch() || ddaRoutingOverridden();
    const std::string skipReason = mpiCoordinatedSkipReason(
        localSkip, "DDA IPC needs gfx942/gfx950 and no DDA routing env");
    if (!skipReason.empty()) {
      GTEST_SKIP() << skipReason;
    }
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommUserRank(getActiveCommunicator(), &rank_));
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommCount(getActiveCommunicator(), &nRanks_));
  }

  void runReduceScatter(size_t recvCount, bool inPlace) {
    const size_t totalCount = recvCount * static_cast<size_t>(nRanks_);
    const size_t recvBytes = recvCount * sizeof(float);

    void* sendBuf = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, totalCount * sizeof(float)));
    DeviceBufferAutoGuard sendGuard(sendBuf);

    void* recvBuf =
        static_cast<float*>(sendBuf) + static_cast<size_t>(rank_) * recvCount;
    void* separateRecvBuf = nullptr;
    if (!inPlace) {
      ASSERT_MPI_EQ(hipSuccess, hipMalloc(&separateRecvBuf, recvBytes));
      recvBuf = separateRecvBuf;
    }
    DeviceBufferAutoGuard recvGuard(separateRecvBuf);

    const int rank = rank_;
    ASSERT_MPI_EQ(hipSuccess,
                  initializeBufferWithPattern<float>(
                      sendBuf, totalCount,
                      [rank](size_t j) { return sendValue(rank, j); }));
    if (!inPlace) {
      ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, recvBytes));
    }

    // The test is only meaningful if these operands reach the DDA IPC kernel.
    int algo = -1;
    int protocol = -1;
    int maxChannels = -1;
    ASSERT_MPI_EQ(ncclSuccess,
                  rcclGetCollImplInfo(getActiveCommunicator(),
                                      ncclFuncReduceScatter, recvCount,
                                      ncclFloat32, ncclSum, sendBuf, recvBuf,
                                      0, &algo, &protocol, &maxChannels));
    const char* algoName = nullptr;
    rcclGetAlgoName(algo, &algoName);
    EXPECT_EQ(static_cast<int>(RCCL_DDA_IPC), algo)
        << "Rank " << rank_ << " selected " << (algoName ? algoName : "?");
    ASSERT_MPI_EQ(static_cast<int>(RCCL_DDA_IPC), algo);

    ASSERT_MPI_EQ(ncclSuccess,
                  ncclReduceScatter(sendBuf, recvBuf, recvCount, ncclFloat32,
                                    ncclSum, getActiveCommunicator(),
                                    getActiveStream()));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    // recv[i] = sum over r of sendValue(r, rank_ * recvCount + i).
    const int nRanks = nRanks_;
    const size_t base = static_cast<size_t>(rank_) * recvCount;
    size_t badIdx = 0;
    float want = 0.0f;
    float got = 0.0f;
    const bool ok = verifyBufferData<float>(
        recvBuf, recvCount,
        [nRanks, base](size_t i) {
          float sum = 0.0f;
          for (int r = 0; r < nRanks; ++r) {
            sum += sendValue(r, base + i);
          }
          return sum;
        },
        0, 0.0, &badIdx, &want, &got);
    EXPECT_TRUE(ok) << "Rank " << rank_ << " recv[" << badIdx << "] = " << got
                    << ", expected " << want;
    // Fail on every rank: rank 0's own slice survives some offset bugs, and
    // only rank 0's gtest summary is shown by default.
    int badRanks = ok ? 0 : 1;
    MPI_Allreduce(MPI_IN_PLACE, &badRanks, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    EXPECT_EQ(0, badRanks) << badRanks << " rank(s) received a wrong slice";
  }
};

TEST_F(DdaIpcMPI_ReduceScatter, SingleBlockOutOfPlace) {
  runReduceScatter(kSingleBlockCount, false);
}

TEST_F(DdaIpcMPI_ReduceScatter, SingleBlockInPlace) {
  runReduceScatter(kSingleBlockCount, true);
}

TEST_F(DdaIpcMPI_ReduceScatter, MultiBlockOutOfPlace) {
  runReduceScatter(kMultiBlockCount, false);
}

#endif  // MPI_TESTS_ENABLED
