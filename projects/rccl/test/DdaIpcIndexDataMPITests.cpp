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
 * wrong offset. Data changes per call, so stale peer scratch never holds the next call's bytes. AllReduceTreeStaging
 * runs back to back across the tree's kernel copy cap.
 *
 * DdaIpcMPI_CommLifecycle: DDA IPC must come up on a communicator created after another one is destroyed. It needs
 * cuMem, and NCCL_CUMEM_ENABLE is cached at the first read, so run this suite in its own process.
 *
 * Run (example):
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=DdaIpcMPI_IndexData.*
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=DdaIpcMPI_CommLifecycle.*
 */

#ifdef MPI_TESTS_ENABLED

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "algorithms/dda/device/CollCommon.h"
#include "comm.h"
#include "rccl_arch_thresholds.h"
#include "rccl_common.h"

using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace {

// The DDA IPC kernels are compiled for exactly 8 ranks on one node.
constexpr int kDdaIpcRanks = 8;

// Mirrors kDdaFlatTreeThresholdBytes in dda_all_reduce_ipc.cu: AllReduce runs the tree kernel above it.
constexpr size_t kFlatTreeThresholdBytes = 1ULL << 18;

// Tree kernel copy cap AllReduceTreeStaging sets: below every DDA IPC AllReduce cap, so both sides stay on DDA.
constexpr size_t kArStagingCapBytes = 2ULL << 20;

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

// fp32 sends rank * 1000 + (j + 131 * call) % 997, bf16 rank + (j + 7 * call) % 29: exact sums, new values every call.
float inputValue(ncclDataType_t dtype, int rank, size_t j, int call) {
  const size_t c = static_cast<size_t>(call);
  if (dtype == ncclFloat32) {
    return static_cast<float>(rank) * 1000.0f + static_cast<float>((j + 131 * c) % 997);
  }
  return static_cast<float>(rank) + static_cast<float>((j + 7 * c) % 29);
}

float expectedSum(ncclDataType_t dtype, int nRanks, size_t j, int call) {
  float sum = 0.0f;
  for (int r = 0; r < nRanks; ++r) {
    sum += inputValue(dtype, r, j, call);
  }
  return sum;
}

// Settings under which RCCL legitimately routes away from DDA IPC.
bool ddaRoutingOverridden(const ncclComm* comm) {
  const char* ddaEnable = std::getenv("RCCL_DDA_ENABLE");
  // NCCL_CHECK_MODE and NCCL_CHECK_POINTERS force the enqueue path: native kernel runs, DDA IPC is still reported.
  return comm->checkMode != ncclCheckModeDefault ||
         (ddaEnable != nullptr && std::string(ddaEnable) == "0") || std::getenv("NCCL_ALGO") != nullptr ||
         std::getenv("RCCL_DDA_THRESHOLD") != nullptr;
}

// Points the comm at a copy of its arch table with ddaArTreeKernelCopyMaxBytes = cap; restores the table on scope exit.
class ScopedTreeKernelCopyCap {
public:
  ScopedTreeKernelCopyCap(ncclComm* comm, size_t cap) : comm_(comm), saved_(comm->archThresholds) {
    if (saved_ != nullptr) {
      table_ = *saved_;
      table_.ddaArTreeKernelCopyMaxBytes = cap;
      comm_->archThresholds = &table_;
    }
  }
  ~ScopedTreeKernelCopyCap() {
    comm_->archThresholds = saved_;
  }
  ScopedTreeKernelCopyCap(const ScopedTreeKernelCopyCap&) = delete;
  ScopedTreeKernelCopyCap& operator=(const ScopedTreeKernelCopyCap&) = delete;

private:
  ncclComm* comm_;
  const rcclArchThresholds* saved_;
  rcclArchThresholds table_{};
};

bool ddaIpcReady(const ncclComm* comm) {
  return comm->ddaIpcMemHandler != nullptr && comm->ddaIpcBarrierState != nullptr;
}

template <typename T>
hipError_t fillInput(void* buf, size_t count, ncclDataType_t dtype, int rank, int call) {
  return initializeBufferWithPattern<T>(buf, count, [=](size_t j) { return T(inputValue(dtype, rank, j, call)); });
}

// Exact check of recv[i] against the analytic sum of call `call` at global index base + i.
template <typename T>
bool verifyOutput(const void* buf, size_t count, ncclDataType_t dtype, int nRanks, size_t base, int call,
                  std::string* detail) {
  size_t badIdx = 0;
  T want{};
  T got{};
  const bool ok = verifyBufferData<T>(
    buf, count, [=](size_t i) { return T(expectedSum(dtype, nRanks, base + i, call)); }, 0, 0.0, &badIdx, &want, &got);
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
  // Salts each call's data. Every rank runs the same calls, so it advances identically everywhere.
  int call_ = 0;

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
    const int call = call_++;
    ASSERT_MPI_EQ(hipSuccess, fp32 ? fillInput<float>(sendBuf, sendCount, c.dtype, rank_, call) :
                                     fillInput<hip_bfloat16>(sendBuf, sendCount, c.dtype, rank_, call));

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
    const bool ok = fp32 ? verifyOutput<float>(recvBuf, c.count, c.dtype, nRanks_, base, call, &detail) :
                           verifyOutput<hip_bfloat16>(recvBuf, c.count, c.dtype, nRanks_, base, call, &detail);
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

  // Launches every case back to back at `cap` with no sync in between, then checks each; counts cases on each side.
  void runTreeStaging(size_t cap, const std::vector<IndexCase>& cases, bool inPlace, int* inKernel, int* hostCopy) {
    ncclComm* comm = getActiveCommunicator();
    ScopedTreeKernelCopyCap capGuard(comm, cap);
    constexpr size_t kAlign = 256;

    std::vector<size_t> offsets;
    size_t arenaBytes = 0;
    for (const IndexCase& c : cases) {
      offsets.push_back(arenaBytes);
      arenaBytes += (c.count * elemBytes(c.dtype) + kAlign - 1) / kAlign * kAlign;
    }
    void* sendArena = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendArena, arenaBytes));
    DeviceBufferAutoGuard sendGuard(sendArena);
    void* recvArena = sendArena;
    DeviceBufferAutoGuard recvGuard;
    if (!inPlace) {
      ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvArena, arenaBytes));
      recvGuard.set(recvArena);
      // 0xFF is NaN in both types, so an element the kernel never writes cannot match.
      ASSERT_MPI_EQ(hipSuccess, hipMemset(recvArena, 0xFF, arenaBytes));
    }

    std::vector<int> calls;
    for (size_t i = 0; i < cases.size(); ++i) {
      const IndexCase& c = cases[i];
      const size_t bytes = c.count * elemBytes(c.dtype);
      ASSERT_MPI_TRUE(bytes > kFlatTreeThresholdBytes);
      char* send = static_cast<char*>(sendArena) + offsets[i];
      char* recv = static_cast<char*>(recvArena) + offsets[i];
      const bool fp32 = c.dtype == ncclFloat32;
      calls.push_back(call_++);
      ASSERT_MPI_EQ(hipSuccess, fp32 ? fillInput<float>(send, c.count, c.dtype, rank_, calls[i]) :
                                       fillInput<hip_bfloat16>(send, c.count, c.dtype, rank_, calls[i]));
      int algo = -1;
      int protocol = -1;
      int maxChannels = -1;
      ASSERT_MPI_EQ(ncclSuccess, rcclGetCollImplInfo(comm, ncclFuncAllReduce, c.count, c.dtype, ncclSum, send, recv, 0,
                                                     &algo, &protocol, &maxChannels));
      ASSERT_MPI_EQ(static_cast<int>(RCCL_DDA_IPC), algo);
      if (dda::common::ddaAllReduceTreeIpcStagesInKernel(bytes, cap)) {
        ++*inKernel;
      } else {
        ++*hostCopy;
      }
    }
    // The fills above are synchronous; every launch below is queued without a sync in between.
    for (size_t i = 0; i < cases.size(); ++i) {
      const IndexCase& c = cases[i];
      void* send = static_cast<char*>(sendArena) + offsets[i];
      void* recv = static_cast<char*>(recvArena) + offsets[i];
      ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(send, recv, c.count, c.dtype, ncclSum, comm, getActiveStream()));
    }
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    bool ok = true;
    std::string detail;
    for (size_t i = 0; i < cases.size() && ok; ++i) {
      const IndexCase& c = cases[i];
      const void* recv = static_cast<char*>(recvArena) + offsets[i];
      ok = c.dtype == ncclFloat32 ? verifyOutput<float>(recv, c.count, c.dtype, nRanks_, 0, calls[i], &detail) :
                                    verifyOutput<hip_bfloat16>(recv, c.count, c.dtype, nRanks_, 0, calls[i], &detail);
      if (!ok) {
        detail = "count=" + std::to_string(c.count) + (c.dtype == ncclFloat32 ? " fp32 " : " bf16 ") + detail;
      }
    }
    const std::string tag = "AllReduce cap=" + std::to_string(cap) + (inPlace ? " in-place" : " out-of-place");
    EXPECT_TRUE(ok) << "Rank " << rank_ << ": " << tag << ": " << detail;
    int badRanks = ok ? 0 : 1;
    MPI_Allreduce(MPI_IN_PLACE, &badRanks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(0, badRanks) << tag << ": " << badRanks << " rank(s) received wrong values";
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

// Cap 0 and 2 MiB, out of place then in place, sizes around the flat/tree split and the cap; rounds rotate the order.
TEST_F(DdaIpcMPI_IndexData, AllReduceTreeStaging) {
  const std::string skip =
    mpiCoordinatedSkipReason(rcclCeAllReduceEnabled(getActiveCommunicator()), "CE AllReduce routes away from DDA IPC");
  if (!skip.empty()) {
    GTEST_SKIP() << skip;
  }
  constexpr int kRounds = 5;
  const size_t capFloats = kArStagingCapBytes / elemBytes(ncclFloat32);
  const size_t capBf16 = kArStagingCapBytes / elemBytes(ncclBfloat16);
  std::vector<IndexCase> cases = {
    {ncclFloat32, 65568},   {ncclFloat32, capFloats + 32}, {ncclFloat32, capFloats},   {ncclFloat32, 3u << 19},
    {ncclBfloat16, 131136}, {ncclBfloat16, capBf16 + 64},  {ncclBfloat16, capBf16},
  };
  int inKernel = 0;
  int hostCopy = 0;
  for (int round = 0; round < kRounds; ++round) {
    for (size_t cap : {size_t{0}, kArStagingCapBytes}) {
      for (bool inPlace : {false, true}) {
        runTreeStaging(cap, cases, inPlace, &inKernel, &hostCopy);
        if (HasFailure() || IsSkipped()) {
          return;
        }
      }
    }
    std::rotate(cases.begin(), cases.begin() + 1, cases.end());
  }
  // RCCL_DDA_AR_TREE_KERNEL_COPY_MAX_BYTES overrides the table cap, so the launch sides are only checked without it.
  if (std::getenv("RCCL_DDA_AR_TREE_KERNEL_COPY_MAX_BYTES") != nullptr) {
    return;
  }
  EXPECT_GT(inKernel, 0) << "no call copied in the kernel";
  EXPECT_GT(hostCopy, 0) << "no call copied before the launch";
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
    // ncclCuMemEnable() is 0 also when the runtime probe fails; the bug does not show without cuMem.
    const bool skipLocal =
      !ddaIpcReady(comm) || !ncclCuMemEnable() || ddaRoutingOverridden(comm) || rcclCeAllReduceEnabled(comm);
    const std::string skip =
      mpiCoordinatedSkipReason(skipLocal, "DDA IPC not initialised, cuMem off, or env routes away from it");
    if (!skip.empty()) {
      GTEST_SKIP() << skip;
    }
    ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(getActiveCommunicator(), &rank_));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(getActiveCommunicator(), &nRanks_));
  }

  ncclResult_t initComm(ncclComm_t* comm) {
    ncclUniqueId id{};
    // Peers learn rank 0's status first, so they fail with it instead of waiting for an id that never comes.
    int res = rank_ == 0 ? ncclGetUniqueId(&id) : ncclSuccess;
    MPI_Bcast(&res, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (res != ncclSuccess) {
      return static_cast<ncclResult_t>(res);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    return ncclCommInitRank(comm, nRanks_, id, rank_);
  }

  // Rank r sends r + 1, so every element sums to nRanks * (nRanks + 1) / 2.
  bool prepareAllReduce(ncclComm_t comm, void* buf, std::string* detail) {
    int algo = -1;
    int protocol = -1;
    int maxChannels = -1;
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
    return true;
  }

  bool allReduceOk(ncclComm_t comm, void* buf, hipStream_t stream, std::string* detail) {
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

// Each communicator created next to the fixture's one, also after a destroy, must bring up DDA IPC with no HIP error
// pending and run a DDA IPC AllReduce.
TEST_F(DdaIpcMPI_CommLifecycle, RecreateAfterDestroy) {
  for (int cycle = 0; cycle < kLifecycleCycles; ++cycle) {
    (void)hipGetLastError();
    ncclComm_t comm = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, initComm(&comm));

    std::string detail;
    const hipError_t pending = hipGetLastError();
    void* buf = nullptr;
    DeviceBufferAutoGuard guard;
    bool ok = false;
    if (!ddaIpcReady(comm)) {
      detail = "DDA IPC init failed";
    } else if (pending != hipSuccess) {
      detail = std::string("HIP error pending after init: ") + hipGetErrorName(pending);
    } else if (hipMalloc(&buf, kLifecycleCount * sizeof(float)) != hipSuccess) {
      detail = "hipMalloc failed";
    } else {
      guard.set(buf);
      ok = prepareAllReduce(comm, buf, &detail);
    }
    // A rank that does not launch leaves its peers blocked in the collective, so all ranks launch or none does.
    if (MPIHelpers::allRanksTrue(ok)) {
      ok = allReduceOk(comm, buf, getActiveStream(), &detail);
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
