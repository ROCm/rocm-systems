/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaIpcCommLifecycleMPITests.cpp
 * @brief MPI regression test for DDA IPC init on communicators created after another one is destroyed.
 *
 * Every communicator exports its DDA IPC barrier mailbox to its peers. If successive mailboxes share one HIP
 * suballocation chunk, the peers' close of a destroyed communicator's mailbox breaks the open of the next one at the
 * same offset: DDA IPC init fails on every rank, the communicator silently falls back, and the HIP error stays
 * pending for the application's next launch check. With cuMem off, the surrounding allocations differ and the bug
 * does not show, so the fixture enables it. NCCL_CUMEM_ENABLE is cached at the first read, so run the test in its
 * own process, as CI does.
 *
 * Run (example):
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=DdaIpcMPI_CommLifecycle.*
 */

#ifdef MPI_TESTS_ENABLED

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <mpi.h>

#include <cstdlib>
#include <string>

#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "comm.h"
#include "rccl_common.h"

using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace {

// The DDA IPC kernels are compiled for exactly 8 ranks on one node.
constexpr int kDdaIpcRanks = 8;

// The fixture's communicator stays alive throughout; cycle 1 is the first one created after a destroy.
constexpr int kCycles = 2;

// 4 KiB fp32: well below the DDA IPC AllReduce caps on gfx942 and gfx950.
constexpr size_t kCount = 1024;

bool ddaIpcReady(const ncclComm* comm) {
  return comm->ddaIpcMemHandler != nullptr && comm->ddaIpcBarrierState != nullptr;
}

// Settings under which RCCL legitimately skips DDA IPC init or routes away from it.
bool ddaRoutingOverridden(const ncclComm* comm) {
  const char* ddaEnable = std::getenv("RCCL_DDA_ENABLE");
  return comm->config.launchOrderImplicit == 1 || comm->checkMode != ncclCheckModeDefault ||
         (ddaEnable != nullptr && std::string(ddaEnable) == "0") || std::getenv("NCCL_ALGO") != nullptr ||
         std::getenv("RCCL_DDA_THRESHOLD") != nullptr || rcclCeAllReduceEnabled(comm);
}

}  // namespace

class DdaIpcMPI_CommLifecycle : public MPITestBase {
protected:
  MPIHelpers::MpiEnvGuard cuMem_{"NCCL_CUMEM_ENABLE", "1"};
  int rank_ = 0;
  int nRanks_ = 0;

  void SetUp() override {
    MPITestBase::SetUp();
    if (!validateTestPrerequisites(kDdaIpcRanks, kDdaIpcRanks, false, 1, 1)) {
      GTEST_SKIP() << "DDA IPC needs exactly 8 ranks on one node";
    }
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    // ddaIpcMemHandler is set only on gfx942/gfx950 with full P2P; skip together or the others block.
    const ncclComm* comm = getActiveCommunicator();
    const std::string skip = mpiCoordinatedSkipReason(!ddaIpcReady(comm) || ddaRoutingOverridden(comm),
                                                      "DDA IPC not initialised, or env routes away from it");
    if (!skip.empty()) {
      GTEST_SKIP() << skip;
    }
    ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(getActiveCommunicator(), &rank_));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(getActiveCommunicator(), &nRanks_));
  }

  ncclResult_t initComm(ncclComm_t* comm) {
    ncclUniqueId id{};
    if (rank_ == 0) {
      const ncclResult_t res = ncclGetUniqueId(&id);
      if (res != ncclSuccess) {
        return res;
      }
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    return ncclCommInitRank(comm, nRanks_, id, rank_);
  }

  // One DDA IPC AllReduce on comm: rank r sends r + 1, so every element sums to nRanks * (nRanks + 1) / 2.
  bool allReduceOk(ncclComm_t comm, hipStream_t stream, std::string* detail) {
    int algo = -1;
    int protocol = -1;
    int maxChannels = -1;
    void* buf = nullptr;
    if (hipMalloc(&buf, kCount * sizeof(float)) != hipSuccess) {
      *detail = "hipMalloc failed";
      return false;
    }
    DeviceBufferAutoGuard guard(buf);
    if (initializeBufferWithPattern<float>(buf, kCount, [this](size_t) { return float(rank_ + 1); }) != hipSuccess) {
      *detail = "buffer init failed";
      return false;
    }
    if (rcclGetCollImplInfo(comm, ncclFuncAllReduce, kCount, ncclFloat32, ncclSum, buf, buf, 0, &algo, &protocol,
                            &maxChannels) != ncclSuccess ||
        algo != static_cast<int>(RCCL_DDA_IPC)) {
      *detail = "AllReduce did not select DDA IPC (algo " + std::to_string(algo) + ")";
      return false;
    }
    if (ncclAllReduce(buf, buf, kCount, ncclFloat32, ncclSum, comm, stream) != ncclSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
      *detail = "AllReduce failed";
      return false;
    }
    const float want = float(nRanks_ * (nRanks_ + 1) / 2);
    size_t badIdx = 0;
    float expected = 0.0f;
    float got = 0.0f;
    if (!verifyBufferData<float>(
          buf, kCount, [=](size_t) { return want; }, 0, 0.0, &badIdx, &expected, &got)) {
      *detail = "recv[" + std::to_string(badIdx) + "] = " + std::to_string(got) + ", expected " +
                std::to_string(expected);
      return false;
    }
    return true;
  }
};

// Create, check and destroy communicators next to the fixture's one. Each new communicator must bring up DDA IPC,
// leave no HIP error pending, and run a DDA IPC AllReduce, also when it reuses the memory of a destroyed one.
TEST_F(DdaIpcMPI_CommLifecycle, RecreateAfterDestroy) {
  for (int cycle = 0; cycle < kCycles; ++cycle) {
    (void)hipGetLastError();
    ncclComm_t comm = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, initComm(&comm));

    std::string detail;
    const hipError_t pending = hipGetLastError();
    bool ok = false;
    if (!ddaIpcReady(comm)) {
      detail = "DDA IPC init failed";
    } else if (pending != hipSuccess) {
      detail = std::string("HIP error pending after init: ") + hipGetErrorName(pending);
    } else {
      ok = allReduceOk(comm, getActiveStream(), &detail);
    }
    EXPECT_TRUE(ok) << "Rank " << rank_ << ", cycle " << cycle << ": " << detail;
    int badRanks = ok ? 0 : 1;
    MPI_Allreduce(MPI_IN_PLACE, &badRanks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(0, badRanks) << "cycle " << cycle << ": " << badRanks << " rank(s) failed";

    ASSERT_MPI_EQ(ncclSuccess, ncclCommDestroy(comm));
    if (badRanks != 0) {
      return;
    }
  }
}

#endif  // MPI_TESTS_ENABLED
