/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaIpcIndexDataMPITests.cpp
 * @brief MPI end-to-end checks for the DDA IPC AllReduce and ReduceScatter kernels and for DDA IPC init.
 *
 * DdaIpcMPI_IndexData: every rank sends rank- and index-dependent data, so each element and each slice of the result
 * is distinct. A per-rank constant fill sums to the same value everywhere and hides a kernel that reads or writes the
 * wrong offset.
 *
 * DdaIpcMPI_CommLifecycle: every communicator exports its DDA IPC barrier mailbox to its peers. If successive
 * mailboxes share one HIP suballocation chunk, the peers' close of a destroyed communicator's mailbox breaks the open
 * of the next one at the same offset: DDA IPC init fails on every rank, the communicator silently falls back, and the
 * HIP error stays pending for the application's next launch check. With cuMem off, the surrounding allocations differ
 * and the bug does not show, so the fixture enables it. NCCL_CUMEM_ENABLE is cached at the first read, so run this
 * suite in its own process, as CI does.
 *
 * Run (example):
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=DdaIpcMPI_IndexData.*
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=DdaIpcMPI_CommLifecycle.*
 */

#ifdef MPI_TESTS_ENABLED

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <mpi.h>

#include <cstdlib>
#include <string>
#include <vector>

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

// Mirrors kDdaFlatTreeThresholdBytes in dda_all_reduce_ipc.cu: AllReduce runs the tree kernel above it.
constexpr size_t kFlatTreeThresholdBytes = 1ULL << 18;

// The fixture's communicator stays alive throughout; cycle 1 is the first one created after a destroy.
constexpr int kLifecycleCycles = 2;

// 4 KiB fp32: well below the DDA IPC AllReduce caps on gfx942 and gfx950.
constexpr size_t kLifecycleCount = 1024;

struct IndexCase {
  ncclDataType_t dtype;
  size_t count;  // AllReduce count, or ReduceScatter recvcount
};

size_t elemBytes(ncclDataType_t dtype) {
  return static_cast<size_t>(ncclTypeSize(dtype));
}

// fp32 sends rank * 1000 + j % 997 and bf16 rank + j % 29, so every partial sum is an integer the type holds exactly.
float inputValue(ncclDataType_t dtype, int rank, size_t j) {
  if (dtype == ncclFloat32) {
    return static_cast<float>(rank) * 1000.0f + static_cast<float>(j % 997);
  }
  return static_cast<float>(rank) + static_cast<float>(j % 29);
}

float expectedSum(ncclDataType_t dtype, int nRanks, size_t j) {
  float sum = 0.0f;
  for (int r = 0; r < nRanks; ++r) {
    sum += inputValue(dtype, r, j);
  }
  return sum;
}

// Settings under which RCCL legitimately routes away from DDA IPC.
bool ddaRoutingOverridden(const ncclComm* comm) {
  const char* ddaEnable = std::getenv("RCCL_DDA_ENABLE");
  // NCCL_CHECK_MODE and NCCL_CHECK_POINTERS force the enqueue path: native kernel runs, DDA IPC is still reported.
  return comm->config.launchOrderImplicit == 1 || comm->checkMode != ncclCheckModeDefault ||
         (ddaEnable != nullptr && std::string(ddaEnable) == "0") || std::getenv("NCCL_ALGO") != nullptr ||
         std::getenv("RCCL_DDA_THRESHOLD") != nullptr;
}

bool ddaIpcReady(const ncclComm* comm) {
  return comm->ddaIpcMemHandler != nullptr && comm->ddaIpcBarrierState != nullptr;
}

template <typename T>
hipError_t fillInput(void* buf, size_t count, ncclDataType_t dtype, int rank) {
  return initializeBufferWithPattern<T>(buf, count, [=](size_t j) { return T(inputValue(dtype, rank, j)); });
}

// Exact check of recv[i] against the analytic sum at global index base + i.
template <typename T>
bool verifyOutput(const void* buf, size_t count, ncclDataType_t dtype, int nRanks, size_t base, std::string* detail) {
  size_t badIdx = 0;
  T want{};
  T got{};
  const bool ok = verifyBufferData<T>(
    buf, count, [=](size_t i) { return T(expectedSum(dtype, nRanks, base + i)); }, 0, 0.0, &badIdx, &want, &got);
  if (!ok) {
    *detail = "recv[" + std::to_string(badIdx) + "] = " + std::to_string(static_cast<float>(got)) + ", expected " +
              std::to_string(static_cast<float>(want));
  }
  return ok;
}

}  // namespace

class DdaIpcMPI_IndexData : public MPITestBase {
protected:
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
    const std::string skip = mpiCoordinatedSkipReason(comm->ddaIpcMemHandler == nullptr || ddaRoutingOverridden(comm),
                                                      "DDA IPC not initialised, or env routes away from it");
    if (!skip.empty()) {
      GTEST_SKIP() << skip;
    }
    ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(getActiveCommunicator(), &rank_));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(getActiveCommunicator(), &nRanks_));
  }

  // Runs one DDA IPC collective, out of place and in place, and checks every element on every rank.
  void runCase(ncclFunc_t func, const IndexCase& c) {
    for (bool inPlace : {false, true}) {
      runOnce(func, c, inPlace);
      // Any failure, not just fatal: runOnce fails or skips every rank together, so all ranks stop on the same case.
      if (HasFailure() || IsSkipped()) {
        return;
      }
    }
  }

  void runOnce(ncclFunc_t func, const IndexCase& c, bool inPlace) {
    const bool rs = func == ncclFuncReduceScatter;
    const size_t elem = elemBytes(c.dtype);
    const size_t sendCount = rs ? c.count * static_cast<size_t>(nRanks_) : c.count;
    const size_t base = rs ? static_cast<size_t>(rank_) * c.count : 0;

    void* sendBuf = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, sendCount * elem));
    DeviceBufferAutoGuard sendGuard(sendBuf);
    void* recvBuf = static_cast<char*>(sendBuf) + base * elem;
    void* separateRecvBuf = nullptr;
    DeviceBufferAutoGuard recvGuard;
    if (!inPlace) {
      ASSERT_MPI_EQ(hipSuccess, hipMalloc(&separateRecvBuf, c.count * elem));
      recvGuard.set(separateRecvBuf);
      // 0xFF is NaN in both types, so an element the kernel never writes cannot match.
      ASSERT_MPI_EQ(hipSuccess, hipMemset(separateRecvBuf, 0xFF, c.count * elem));
      recvBuf = separateRecvBuf;
    }

    const bool fp32 = c.dtype == ncclFloat32;
    ASSERT_MPI_EQ(hipSuccess, fp32 ? fillInput<float>(sendBuf, sendCount, c.dtype, rank_) :
                                     fillInput<hip_bfloat16>(sendBuf, sendCount, c.dtype, rank_));

    const std::string tag = std::string(rs ? "ReduceScatter" : "AllReduce") + " count=" + std::to_string(c.count) +
                            (fp32 ? " fp32" : " bf16") + (inPlace ? " in-place" : " out-of-place");
    int algo = -1;
    int protocol = -1;
    int maxChannels = -1;
    ASSERT_MPI_EQ(ncclSuccess, rcclGetCollImplInfo(getActiveCommunicator(), func, c.count, c.dtype, ncclSum, sendBuf,
                                                   recvBuf, 0, &algo, &protocol, &maxChannels));
    const char* algoName = nullptr;
    rcclGetAlgoName(algo, &algoName);
    EXPECT_EQ(static_cast<int>(RCCL_DDA_IPC), algo)
      << "Rank " << rank_ << ": " << tag << " selected " << (algoName ? algoName : "?");
    ASSERT_MPI_EQ(static_cast<int>(RCCL_DDA_IPC), algo);

    const ncclResult_t res =
      rs ? ncclReduceScatter(sendBuf, recvBuf, c.count, c.dtype, ncclSum, getActiveCommunicator(), getActiveStream()) :
           ncclAllReduce(sendBuf, recvBuf, c.count, c.dtype, ncclSum, getActiveCommunicator(), getActiveStream());
    ASSERT_MPI_EQ(ncclSuccess, res);
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    std::string detail;
    const bool ok = fp32 ? verifyOutput<float>(recvBuf, c.count, c.dtype, nRanks_, base, &detail) :
                           verifyOutput<hip_bfloat16>(recvBuf, c.count, c.dtype, nRanks_, base, &detail);
    EXPECT_TRUE(ok) << "Rank " << rank_ << ": " << tag << ": " << detail;
    // Fail on every rank: rank 0's slice survives some offset bugs, and only rank 0's summary is shown by default.
    int badRanks = ok ? 0 : 1;
    MPI_Allreduce(MPI_IN_PLACE, &badRanks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(0, badRanks) << tag << ": " << badRanks << " rank(s) received wrong values";
  }

  void runAllReduce(const std::vector<IndexCase>& cases, bool tree) {
    // RCCL_CE_ALLREDUCE=1 lets CE take AllReduce ahead of DDA; ReduceScatter is unaffected.
    const bool ceAllReduce = rcclCeAllReduceEnabled(getActiveCommunicator());
    const std::string skip = mpiCoordinatedSkipReason(ceAllReduce, "CE AllReduce routes away from DDA IPC");
    if (!skip.empty()) {
      GTEST_SKIP() << skip;
    }
    for (const IndexCase& c : cases) {
      const size_t bytes = c.count * elemBytes(c.dtype);
      ASSERT_EQ(tree, bytes > kFlatTreeThresholdBytes) << "count " << c.count << " misses the intended kernel";
      runCase(ncclFuncAllReduce, c);
      if (HasFailure() || IsSkipped()) {
        return;
      }
    }
  }

  void runReduceScatter(const std::vector<IndexCase>& cases) {
    for (const IndexCase& c : cases) {
      runCase(ncclFuncReduceScatter, c);
      if (HasFailure() || IsSkipped()) {
        return;
      }
    }
  }
};

// ncclDdaIpcCommInit allocates the LL epoch cells DDA Broadcast LL runs on.
TEST_F(DdaIpcMPI_IndexData, InitSetsLLEpochCells) {
  const ncclComm* comm = getActiveCommunicator();
  ASSERT_MPI_TRUE(comm->ddaLLEpochDev != nullptr);
  ASSERT_MPI_TRUE(comm->ddaLLEpochLen > 0);
}

// One-shot flat kernel: every rank reduces the whole buffer, so a wrong source offset hides behind constant data.
TEST_F(DdaIpcMPI_IndexData, AllReduceFlat) {
  runAllReduce(
    {{ncclFloat32, 4}, {ncclFloat32, 1024}, {ncclFloat32, 49152}, {ncclFloat32, 65536}, {ncclBfloat16, 4096}},
    /*tree=*/false);
}

// Two-shot tree kernel: each rank reduces one count/8 slice, then gathers the peers' slices. Largest case stays below
// gfx942's 8 MiB DDA IPC AllReduce cap.
// Each dtype sits just over the flat/tree threshold and at least 8x over it, so a threshold retune keeps tree coverage.
TEST_F(DdaIpcMPI_IndexData, AllReduceTree) {
  runAllReduce({{ncclFloat32, 65568}, {ncclFloat32, 1u << 20}, {ncclFloat32, 3u << 19}, {ncclBfloat16, 131136},
                {ncclBfloat16, 1u << 20}},
               /*tree=*/true);
}

// Each rank reduces slice rank * recvcount of every peer: one block, then several blocks per shard.
TEST_F(DdaIpcMPI_IndexData, ReduceScatter) {
  runReduceScatter({{ncclFloat32, 1024}, {ncclFloat32, 65536}, {ncclBfloat16, 8192}});
}

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
    const std::string skip =
      mpiCoordinatedSkipReason(!ddaIpcReady(comm) || ddaRoutingOverridden(comm) || rcclCeAllReduceEnabled(comm),
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
    if (hipMalloc(&buf, kLifecycleCount * sizeof(float)) != hipSuccess) {
      *detail = "hipMalloc failed";
      return false;
    }
    DeviceBufferAutoGuard guard(buf);
    if (initializeBufferWithPattern<float>(buf, kLifecycleCount, [this](size_t) { return float(rank_ + 1); }) !=
        hipSuccess) {
      *detail = "buffer init failed";
      return false;
    }
    if (rcclGetCollImplInfo(comm, ncclFuncAllReduce, kLifecycleCount, ncclFloat32, ncclSum, buf, buf, 0, &algo,
                            &protocol, &maxChannels) != ncclSuccess ||
        algo != static_cast<int>(RCCL_DDA_IPC)) {
      *detail = "AllReduce did not select DDA IPC (algo " + std::to_string(algo) + ")";
      return false;
    }
    if (ncclAllReduce(buf, buf, kLifecycleCount, ncclFloat32, ncclSum, comm, stream) != ncclSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
      *detail = "AllReduce failed";
      return false;
    }
    const float want = float(nRanks_ * (nRanks_ + 1) / 2);
    size_t badIdx = 0;
    float expected = 0.0f;
    float got = 0.0f;
    if (!verifyBufferData<float>(
          buf, kLifecycleCount, [=](size_t) { return want; }, 0, 0.0, &badIdx, &expected, &got)) {
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
  for (int cycle = 0; cycle < kLifecycleCycles; ++cycle) {
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
