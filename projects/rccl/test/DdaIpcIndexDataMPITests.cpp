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
 * wrong offset. On gfx950 and gfx942 it also checks the DDA Broadcast LL one-shot over the IPC scratch.
 *
 * DdaIpcMPI_LL: IPC LL one-shots at 1 to 24 blocks, mixed calls, graph replays; needs RCCL_DDA_IPC_LL_THRESHOLD raised.
 *
 * DdaIpcMPI_CommLifecycle: DDA IPC must come up on a communicator created after another one is destroyed. It needs
 * cuMem, and NCCL_CUMEM_ENABLE is cached at the first read, so run this suite in its own process.
 *
 * Run (example):
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=DdaIpcMPI_IndexData.*
 *   RCCL_DDA_IPC_LL_THRESHOLD=2097152 mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=DdaIpcMPI_LL.*
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=DdaIpcMPI_CommLifecycle.*
 */

#ifdef MPI_TESTS_ENABLED

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <mpi.h>

#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "algorithms/dda/all_reduce/dda_all_reduce.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"
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

// AllReduce fp32 count of the LL one-shot case: 2 KiB, under the AllReduce IPC LL table cap.
constexpr size_t kLifecycleLLCount = 512;

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
  return comm->checkMode != ncclCheckModeDefault ||
         (ddaEnable != nullptr && std::string(ddaEnable) == "0") || std::getenv("NCCL_ALGO") != nullptr ||
         std::getenv("RCCL_DDA_THRESHOLD") != nullptr;
}

bool ddaIpcReady(const ncclComm* comm) {
  return comm->ddaIpcMemHandler != nullptr && comm->ddaIpcBarrierState != nullptr;
}

// The flag the next DDA LL call takes from epoch cell 0, as ddaNextLLEpoch does: +1, skipping the 0 sentinel.
uint32_t nextLLFlag(uint32_t epoch) {
  return epoch + 1 == 0 ? 2 : epoch + 1;
}

// gfx950 or gfx942 with the LL epoch cells, and no env var that moves Broadcast off its table tier.
bool bcDdaIpcExpected(const ncclComm* comm) {
  const char* bc = std::getenv("RCCL_DDA_BC");
  const std::string arch = comm->archName != nullptr ? comm->archName : "";
  return (arch.rfind("gfx950", 0) == 0 || arch.rfind("gfx942", 0) == 0) &&
         comm->ddaLLEpochDev != nullptr && (bc == nullptr || std::string(bc) != "0") &&
         std::getenv("RCCL_DDA_LL") == nullptr && std::getenv("RCCL_DDA_LL_THRESHOLD") == nullptr;
}

// Fail on every rank: rank 0's slice survives some offset bugs, and only rank 0's summary is shown by default.
// Returns the shared verdict.
bool expectAllRanksOk(bool ok, const std::string& tag) {
  int badRanks = ok ? 0 : 1;
  MPI_Allreduce(MPI_IN_PLACE, &badRanks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  EXPECT_EQ(0, badRanks) << tag << ": " << badRanks << " rank(s) failed";
  return badRanks == 0;
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
    // The LL one-shot runs up to the comm's IPC LL cap where production eligibility (LL slot, 16 B rule) holds.
    ncclComm* comm = getActiveCommunicator();
    const bool ll = sendCount * elem <= rcclDdaLLThreshold(comm, func) &&
                    (rs ? ncclReduceScatterDdaFabricLLEligible : ncclAllReduceDdaFabricLLEligible)(
                      comm, sendBuf, recvBuf, c.count, c.dtype, ncclSum);
    expectAlgo(func, c.count, c.dtype, sendBuf, recvBuf, ll ? RCCL_DDA_FABRIC_LL : RCCL_DDA_IPC, tag);
    if (HasFailure() || IsSkipped()) {
      return;
    }

    const ncclResult_t res =
      rs ? ncclReduceScatter(sendBuf, recvBuf, c.count, c.dtype, ncclSum, getActiveCommunicator(), getActiveStream()) :
           ncclAllReduce(sendBuf, recvBuf, c.count, c.dtype, ncclSum, getActiveCommunicator(), getActiveStream());
    ASSERT_MPI_EQ(ncclSuccess, res);
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    std::string detail;
    const bool ok = fp32 ? verifyOutput<float>(recvBuf, c.count, c.dtype, nRanks_, base, &detail) :
                           verifyOutput<hip_bfloat16>(recvBuf, c.count, c.dtype, nRanks_, base, &detail);
    EXPECT_TRUE(ok) << "Rank " << rank_ << ": " << tag << ": " << detail;
    expectAllRanksOk(ok, tag);
  }

  // Checks on every rank that rcclGetCollImplInfo picks `want`, or any native algorithm when want < 0.
  void expectAlgo(ncclFunc_t func, size_t count, ncclDataType_t dtype, const void* sendBuf, void* recvBuf, int want,
                  const std::string& tag, int capturing = 0) {
    int algo = -1;
    int protocol = -1;
    int maxChannels = -1;
    ASSERT_MPI_EQ(ncclSuccess, rcclGetCollImplInfo(getActiveCommunicator(), func, count, dtype, ncclSum, sendBuf,
                                                   recvBuf, capturing, &algo, &protocol, &maxChannels));
    const char* algoName = nullptr;
    rcclGetAlgoName(algo, &algoName);
    const bool ok = want >= 0 ? algo == want : algo < NCCL_NUM_ALGORITHMS;
    EXPECT_TRUE(ok) << "Rank " << rank_ << ": " << tag << " selected " << (algoName ? algoName : "?");
    ASSERT_MPI_TRUE(ok);
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

  // The Broadcast DDA LL cap on this comm (the arch table's, as bcDdaIpcExpected leaves the env alone).
  size_t bcDdaLLMaxBytes() {
    return rcclDdaLLThreshold(getActiveCommunicator(), ncclFuncBroadcast);
  }

  // Broadcasts count fp32 from root, out of place: the root sends inputValue(root, j), every rank checks it all.
  // Epoch cell 0 must take the next LL flag exactly when DDA LL runs. rootLast: the non-roots launch first and the
  // root only after an MPI barrier and its stream drains, so the non-roots poll before any root packet lands.
  void runBroadcast(size_t count, int root, bool rootLast = false) {
    const size_t bytes = count * sizeof(float);
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
    DeviceBufferAutoGuard sendGuard(sendBuf);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, bytes));
    DeviceBufferAutoGuard recvGuard(recvBuf);
    ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0xFF, bytes));
    ASSERT_MPI_EQ(hipSuccess, fillInput<float>(sendBuf, count, ncclFloat32, root));

    const std::string tag = "Broadcast count=" + std::to_string(count) + " root=" + std::to_string(root);
    const bool dda = bytes <= bcDdaLLMaxBytes();
    expectAlgo(ncclFuncBroadcast, count, ncclFloat32, sendBuf, recvBuf, dda ? static_cast<int>(RCCL_DDA_FABRIC_LL) : -1,
               tag);
    if (HasFailure() || IsSkipped()) {
      return;
    }
    const ncclComm* comm = getActiveCommunicator();
    uint32_t epoch = 0;
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(&epoch, comm->ddaLLEpochDev, sizeof(epoch), hipMemcpyDeviceToHost));

    auto enqueue = [&] {
      return ncclBroadcast(sendBuf, recvBuf, count, ncclFloat32, root, getActiveCommunicator(), getActiveStream());
    };
    const bool rootWaits = rootLast && rank_ == root;
    ncclResult_t res = rootWaits ? ncclSuccess : enqueue();
    if (rootLast) {
      const hipError_t drained = rootWaits ? hipStreamSynchronize(getActiveStream()) : hipSuccess;
      MPI_Barrier(MPI_COMM_WORLD);
      res = rootWaits && drained == hipSuccess ? enqueue() : res;
      ASSERT_MPI_EQ(hipSuccess, drained);
    }
    ASSERT_MPI_EQ(ncclSuccess, res);
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    uint32_t after = 0;
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(&after, comm->ddaLLEpochDev, sizeof(after), hipMemcpyDeviceToHost));
    const uint32_t wantEpoch = dda ? nextLLFlag(epoch) : epoch;
    EXPECT_EQ(wantEpoch, after) << "Rank " << rank_ << ": " << tag << ": epoch cell 0";

    size_t badIdx = 0;
    float expected = 0.0f;
    float got = 0.0f;
    const bool ok = verifyBufferData<float>(
      recvBuf, count, [=](size_t j) { return inputValue(ncclFloat32, root, j); }, 0, 0.0, &badIdx, &expected, &got);
    EXPECT_TRUE(ok) << "Rank " << rank_ << ": " << tag << ": recv[" << badIdx << "] = " << got << ", expected "
                    << expected;
    expectAllRanksOk(ok && after == wantEpoch, tag);
  }

  void skipUnlessBcDdaIpc() {
    const ncclComm* comm = getActiveCommunicator();
    const std::string skip =
      mpiCoordinatedSkipReason(!bcDdaIpcExpected(comm), "DDA Broadcast LL over IPC needs gfx950/gfx942, no reroute");
    if (!skip.empty()) {
      GTEST_SKIP() << skip;
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
    {{ncclFloat32, 4}, {ncclFloat32, 1024}, {ncclFloat32, 1028}, {ncclFloat32, 49152}, {ncclFloat32, 65536},
     {ncclBfloat16, 4096}},
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
  runReduceScatter({{ncclFloat32, 1024}, {ncclFloat32, 1028}, {ncclFloat32, 65536}, {ncclBfloat16, 8192}});
}

// DDA Broadcast LL one-shot: a root column per peer with 1, 2, 4 and 8 chunk rows (16 B to 16 KiB), uneven chunks, the
// table's cap; two roots. Just above the cap the ring runs.
TEST_F(DdaIpcMPI_IndexData, BroadcastLL) {
  skipUnlessBcDdaIpc();
  if (IsSkipped()) {
    return;
  }
  const size_t capCount = bcDdaLLMaxBytes() / sizeof(float);
  for (size_t count : {size_t{4}, size_t{1024}, size_t{2048}, size_t{4096}, size_t{4100}, capCount, capCount + 4}) {
    for (int root : {0, 5}) {
      runBroadcast(count, root);
      if (HasFailure() || IsSkipped()) {
        return;
      }
    }
  }
}

// Simple DDA calls stage words equal to the next LL flag in the scratch; LL tiers bank in the region past it, so must
// not take them for packets.
TEST_F(DdaIpcMPI_IndexData, LLIgnoresStaleSimpleScratch) {
  skipUnlessBcDdaIpc();
  if (IsSkipped()) {
    return;
  }
  // RCCL_CE_ALLREDUCE=1 lets CE take the staging AllReduce ahead of DDA.
  // An AllReduce LL cap at its size (RCCL_DDA_IPC_LL_THRESHOLD) runs it as LL.
  const bool llStaging =
    rcclDdaLLThreshold(getActiveCommunicator(), ncclFuncAllReduce) >= kFlatTreeThresholdBytes * 3 / 4;
  const std::string ceSkip = mpiCoordinatedSkipReason(rcclCeAllReduceEnabled(getActiveCommunicator()) || llStaging,
                                                      "CE AllReduce or the LL cap takes the staging call off DDA IPC");
  if (!ceSkip.empty()) {
    GTEST_SKIP() << ceSkip;
  }
  // The flat AllReduce stages its whole send buffer in scratch. Sized off the flat/tree cutoff so a retune
  // of it keeps the flat kernel; it must cover bank 0 of the Broadcast: one 16 B token line per rank, then one 16 B
  // line per 8 B of data.
  constexpr size_t kArCount = kFlatTreeThresholdBytes * 3 / 4 / sizeof(float);
  constexpr size_t kBcCount = 16384;
  constexpr size_t kBank0Bytes = (kDdaIpcRanks + kBcCount * sizeof(float) / 8) * 16;
  static_assert(kArCount * sizeof(float) < kFlatTreeThresholdBytes, "staging AllReduce must run the flat kernel");
  static_assert(kBank0Bytes <= kArCount * sizeof(float), "staged words must cover the Broadcast's bank 0");
  constexpr int kRounds = 6;
  void* arSend = nullptr;
  void* arRecv = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&arSend, kArCount * sizeof(float)));
  DeviceBufferAutoGuard sendGuard(arSend);
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&arRecv, kArCount * sizeof(float)));
  DeviceBufferAutoGuard recvGuard(arRecv);
  const ncclComm* comm = getActiveCommunicator();
  for (int round = 0; round < kRounds; ++round) {
    // The flag is the same on every rank; runBroadcast checks the kernel took it. An even flag selects bank 0.
    uint32_t epoch = 0;
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(&epoch, comm->ddaLLEpochDev, sizeof(epoch), hipMemcpyDeviceToHost));
    // Every rank must take the same branch below, or the ranks issue different collectives and hang.
    uint32_t lo = epoch;
    uint32_t hi = epoch;
    MPI_Allreduce(MPI_IN_PLACE, &lo, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &hi, 1, MPI_UINT32_T, MPI_MAX, MPI_COMM_WORLD);
    ASSERT_MPI_EQ(lo, hi);
    uint32_t flag = nextLLFlag(epoch);
    if ((flag & 1) != 0) {
      runBroadcast(4, 0);
      if (HasFailure() || IsSkipped()) {
        return;
      }
      flag = nextLLFlag(flag);
    }
    const std::vector<uint32_t> stale(kArCount, flag);
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(arSend, stale.data(), stale.size() * sizeof(stale[0]), hipMemcpyHostToDevice));
    const std::string tag = "AllReduce staging round " + std::to_string(round);
    expectAlgo(ncclFuncAllReduce, kArCount, ncclFloat32, arSend, arRecv, static_cast<int>(RCCL_DDA_IPC), tag);
    if (HasFailure() || IsSkipped()) {
      return;
    }
    ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(arSend, arRecv, kArCount, ncclFloat32, ncclSum, getActiveCommunicator(),
                                             getActiveStream()));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
    // The poison must sit where the call staged it, or this test checks nothing: the scratch base or a staging bank.
    const char* const base = static_cast<const char*>(comm->ddaScratch);
    const nccl_dda_detail::DdaIpcBarrierState* state = comm->ddaIpcBarrierState;
    std::vector<const char*> stagingBases{base};
    if (state->stagingBankPeerPtrsDev != nullptr) {
      stagingBases = {base + nccl_dda_detail::ddaIpcStagingBankOffset(state->scratchAllocBytes, 0),
                      base + nccl_dda_detail::ddaIpcStagingBankOffset(state->scratchAllocBytes, 1)};
    }
    std::vector<uint32_t> staged(kArCount);  // every word the call staged
    size_t firstOff = 0;
    for (const char* stagingBase : stagingBases) {
      ASSERT_MPI_EQ(hipSuccess, hipMemcpy(staged.data(), stagingBase, kArCount * sizeof(float), hipMemcpyDeviceToHost));
      firstOff = std::max<size_t>(
        firstOff, std::find_if(staged.begin(), staged.end(), [=](uint32_t w) { return w != flag; }) - staged.begin());
    }
    const bool poisoned = firstOff == staged.size();
    EXPECT_TRUE(poisoned) << "Rank " << rank_ << ": " << tag << ": scratch word " << firstOff << " is not the flag";
    expectAllRanksOk(poisoned, tag + " poison");
    if (HasFailure()) {
      return;
    }
    // Rounds 0-3 broadcast from roots 0-3, root last, so a shared bank is polled stale; then AR and RS LL one-shots.
    if (round == 4) {
      runOnce(ncclFuncAllReduce, {ncclFloat32, 1024}, /*inPlace=*/false);
    } else if (round == 5) {
      runOnce(ncclFuncReduceScatter, {ncclFloat32, 1024}, /*inPlace=*/false);
    } else {
      runBroadcast(kBcCount, round % nRanks_, /*rootLast=*/true);
    }
    if (HasFailure() || IsSkipped()) {
      return;
    }
  }
}

// One DDA IPC LL call with its own buffers. count is the API count: AllReduce and Broadcast elements, ReduceScatter
// recvcount.
struct LLOp {
  LLOp(ncclFunc_t f, ncclDataType_t d, size_t n, bool ip = false, int r = 0, size_t off = 0)
    : func(f), dtype(d), count(n), inPlace(ip), root(r), offsetBytes(off) {}
  ncclFunc_t func;
  ncclDataType_t dtype;
  size_t count;
  bool inPlace;
  int root;               // Broadcast only
  size_t offsetBytes;     // of every user buffer from its 256 B-aligned allocation
  // Filled by prepare().
  void* sendBuf = nullptr;
  void* recvBuf = nullptr;
  DeviceBufferAutoGuard sendGuard;
  DeviceBufferAutoGuard recvGuard;
};

class DdaIpcMPI_LL : public DdaIpcMPI_IndexData {
protected:
  // DdaIpcMPI_IndexData's skips (no DDA IPC), then the env's; past both every op asserts the LL one-shot.
  void SetUp() override {
    DdaIpcMPI_IndexData::SetUp();
    if (IsSkipped() || HasFatalFailure()) {
      return;
    }
    ncclComm* comm = getActiveCommunicator();
    const bool on = !rcclCeAllReduceEnabled(comm) && rcclDdaLLThreshold(comm, ncclFuncAllReduce) >= (256u << 10) &&
                    rcclDdaLLThreshold(comm, ncclFuncReduceScatter) >= (2u << 20);
    const std::string skip = mpiCoordinatedSkipReason(
      !on, "CE AllReduce on, or DDA IPC LL caps below the cases (set RCCL_DDA_IPC_LL_THRESHOLD=2097152)");
    if (!skip.empty()) {
      GTEST_SKIP() << skip;
    }
  }

  size_t sendCount(const LLOp& op) const {
    return op.func == ncclFuncReduceScatter ? op.count * static_cast<size_t>(nRanks_) : op.count;
  }

  // Element offset of this rank's ReduceScatter shard: in place, recvbuff = sendbuff + rank * recvcount.
  size_t inPlaceShardOffset(const LLOp& op) const {
    return op.func == ncclFuncReduceScatter ? static_cast<size_t>(rank_) * op.count : 0;
  }

  // All ranks go on only when none failed; a coordinated ASSERT_MPI skips the passing ranks, which must not run on.
  bool allRanksGoOn(const LLOp& op) {
    return expectAllRanksOk(!HasFailure() && !IsSkipped(), tag(op));
  }

  // Value recv element i must hold on this rank: the root's input, or the sum at this rank's shard.
  float wantValue(const LLOp& op, size_t i) const {
    if (op.func == ncclFuncBroadcast) {
      return inputValue(op.dtype, op.root, i);
    }
    return expectedSum(op.dtype, nRanks_, inPlaceShardOffset(op) + i);
  }

  // Allocates and fills op's buffers; in place they share one allocation, a Broadcast non-root sends its recv buffer.
  void prepare(LLOp& op) {
    const size_t elem = elemBytes(op.dtype);
    const size_t recvBytes = op.count * elem;
    const size_t sendBytes = sendCount(op) * elem;
    const size_t bufBytes = op.inPlace ? std::max(recvBytes, sendBytes) : recvBytes;
    void* alloc = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&alloc, bufBytes + 256));
    op.recvGuard.set(alloc);
    char* base = static_cast<char*>(alloc) + op.offsetBytes;
    op.recvBuf = base;
    // 0xFF is NaN in both types, so an element the kernel never writes cannot match.
    ASSERT_MPI_EQ(hipSuccess, hipMemset(alloc, 0xFF, bufBytes + 256));
    if (op.inPlace) {
      op.sendBuf = base;
      op.recvBuf = base + inPlaceShardOffset(op) * elem;
    } else {
      ASSERT_MPI_EQ(hipSuccess, hipMalloc(&alloc, sendBytes + 256));
      op.sendGuard.set(alloc);
      op.sendBuf = static_cast<char*>(alloc) + op.offsetBytes;
    }
    // Every rank reaches the ASSERT_MPI, also an in-place Broadcast non-root that fills nothing.
    const bool rootOrNotBc = op.func != ncclFuncBroadcast || op.root == rank_ || !op.inPlace;
    hipError_t filled = hipSuccess;
    if (rootOrNotBc) {
      const int r = op.func == ncclFuncBroadcast ? op.root : rank_;
      filled = op.dtype == ncclFloat32 ? fillInput<float>(op.sendBuf, sendCount(op), op.dtype, r) :
                                         fillInput<hip_bfloat16>(op.sendBuf, sendCount(op), op.dtype, r);
    }
    ASSERT_MPI_EQ(hipSuccess, filled);
  }

  std::string tag(const LLOp& op) const {
    const char* name = op.func == ncclFuncAllReduce ? "AllReduce" :
                       op.func == ncclFuncReduceScatter ? "ReduceScatter" : "Broadcast";
    return std::string(name) + " count=" + std::to_string(op.count) + (op.dtype == ncclFloat32 ? " fp32" : " bf16") +
           (op.inPlace ? " in-place" : " out-of-place") + " offset=" + std::to_string(op.offsetBytes) +
           (op.func == ncclFuncBroadcast ? " root=" + std::to_string(op.root) : "");
  }

  ncclResult_t launch(const LLOp& op) {
    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    switch (op.func) {
    case ncclFuncAllReduce:
      return ncclAllReduce(op.sendBuf, op.recvBuf, op.count, op.dtype, ncclSum, comm, stream);
    case ncclFuncReduceScatter:
      return ncclReduceScatter(op.sendBuf, op.recvBuf, op.count, op.dtype, ncclSum, comm, stream);
    default:
      return ncclBroadcast(op.sendBuf, op.recvBuf, op.count, op.dtype, op.root, comm, stream);
    }
  }

  // Checks every recv element on every rank; fails on all ranks together.
  void verify(const LLOp& op) {
    size_t badIdx = 0;
    std::string detail;
    bool ok = false;
    const auto want = [&](size_t i) { return wantValue(op, i); };
    if (op.dtype == ncclFloat32) {
      float w = 0, g = 0;
      ok = verifyBufferData<float>(op.recvBuf, op.count, [&](size_t i) { return float(want(i)); }, 0, 0.0,
                                   &badIdx, &w, &g);
      detail = std::to_string(g) + ", expected " + std::to_string(w);
    } else {
      hip_bfloat16 w{}, g{};
      ok = verifyBufferData<hip_bfloat16>(op.recvBuf, op.count, [&](size_t i) { return hip_bfloat16(want(i)); },
                                          0, 0.0, &badIdx, &w, &g);
      detail = std::to_string(static_cast<float>(g)) + ", expected " + std::to_string(static_cast<float>(w));
    }
    EXPECT_TRUE(ok) << "Rank " << rank_ << ": " << tag(op) << ": recv[" << badIdx << "] = " << detail;
    expectAllRanksOk(ok, tag(op) + " received wrong values");
  }

  // Prepares each op and checks it takes the LL one-shot, eager or (capturing) in a graph. Same on every rank.
  bool prepareAll(std::vector<LLOp>& ops, int capturing = 0) {
    for (LLOp& op : ops) {
      prepare(op);
      if (!allRanksGoOn(op)) {
        return false;
      }
      expectAlgo(op.func, op.count, op.dtype, op.sendBuf, op.recvBuf, static_cast<int>(RCCL_DDA_FABRIC_LL), tag(op),
                 capturing);
      if (!allRanksGoOn(op)) {
        return false;
      }
    }
    return true;
  }

  // prepareAll, then launches all back to back with no sync and checks them. Same on every rank.
  bool runLL(std::vector<LLOp>& ops) {
    if (!prepareAll(ops)) {
      return false;
    }
    launchAll(ops);
    return ops.empty() || allRanksGoOn(ops.back());
  }

  // Launches ops back to back with no sync, then checks each; stops on every rank at the first failure.
  void launchAll(const std::vector<LLOp>& ops) {
    for (const LLOp& op : ops) {
      ASSERT_MPI_EQ(ncclSuccess, launch(op));
    }
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
    for (const LLOp& op : ops) {
      verify(op);
      if (!allRanksGoOn(op)) {
        return;
      }
    }
  }

  // Each op alone, out of place and in place. Returns true on every rank or on none.
  bool runEach(ncclFunc_t func, ncclDataType_t dtype, std::initializer_list<size_t> counts) {
    for (size_t count : counts) {
      for (bool inPlace : {false, true}) {
        std::vector<LLOp> ops;
        ops.emplace_back(func, dtype, count, inPlace);
        if (!runLL(ops)) {
          return false;
        }
      }
    }
    return true;
  }

  // A small LL AllReduce, run and checked, to move the epoch on by one. Returns true on every rank or on none.
  bool stepEpoch() {
    std::vector<LLOp> ops;
    ops.emplace_back(ncclFuncAllReduce, ncclFloat32, 4);
    return runLL(ops);
  }
};

// One-shot grids of 1, 2, 4, 8 and 24 blocks (2 KiB to 256 KiB, the LL slot), both banks through the case order.
TEST_F(DdaIpcMPI_LL, AllReduce) {
  if (!runEach(ncclFuncAllReduce, ncclFloat32, {4, 512, 4096, 8192, 16384, 65536})) {
    return;
  }
  runEach(ncclFuncAllReduce, ncclBfloat16, {8, 1032, 131072});
}

// Shards of 16 B to 256 KiB (the LL slot): one block, then 2, 4 and 24.
TEST_F(DdaIpcMPI_LL, ReduceScatter) {
  if (!runEach(ncclFuncReduceScatter, ncclFloat32, {4, 2048, 4096, 8192, 65536})) {
    return;
  }
  runEach(ncclFuncReduceScatter, ncclBfloat16, {8, 16384});
}

// Back-to-back LL calls with no sync, alternating collectives, sizes, places, roots and 2-byte aligned bf16 buffers.
TEST_F(DdaIpcMPI_LL, InterleavedCalls) {
  const bool bc = bcDdaIpcExpected(getActiveCommunicator());
  std::vector<LLOp> ops;
  for (int round = 0; round < 3; ++round) {
    ops.emplace_back(ncclFuncAllReduce, ncclFloat32, 4096 << round, round == 1);
    ops.emplace_back(ncclFuncReduceScatter, ncclFloat32, 1024 >> round, round != 1);
    if (bc) {
      ops.emplace_back(ncclFuncBroadcast, ncclFloat32, 2048 << round, false, (3 * round + 1) % nRanks_);
    }
    ops.emplace_back(ncclFuncAllReduce, ncclBfloat16, 8, round == 2, 0, /*offsetBytes=*/2);
    ops.emplace_back(ncclFuncReduceScatter, ncclBfloat16, 16384 >> (2 * round), false, 0, /*offsetBytes=*/2);
    ops.emplace_back(ncclFuncAllReduce, ncclFloat32, 65536 >> round, false);
  }
  runLL(ops);
}

// Captured LL calls take their flags from the epoch cells at replay, with eager LL calls flipping the bank between.
TEST_F(DdaIpcMPI_LL, GraphReplays) {
  std::vector<LLOp> ops;
  ops.emplace_back(ncclFuncAllReduce, ncclFloat32, 4096);
  ops.emplace_back(ncclFuncReduceScatter, ncclFloat32, 1024);
  ops.emplace_back(ncclFuncAllReduce, ncclFloat32, 8);
  if (!prepareAll(ops, /*capturing=*/1)) {
    return;
  }
  hipGraph_t graph = nullptr;
  hipGraphExec_t graphExec = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipStreamBeginCapture(getActiveStream(), hipStreamCaptureModeThreadLocal));
  for (const LLOp& op : ops) {
    ASSERT_MPI_EQ(ncclSuccess, launch(op));
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamEndCapture(getActiveStream(), &graph));
  ASSERT_MPI_EQ(hipSuccess, hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  for (int replay = 0; replay < 5; ++replay) {
    ASSERT_MPI_EQ(hipSuccess, hipGraphLaunch(graphExec, getActiveStream()));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
    for (LLOp& op : ops) {
      verify(op);
      // Clear the outputs so the next replay has to write them again.
      ASSERT_MPI_EQ(hipSuccess, hipMemset(op.recvBuf, 0xFF, op.count * elemBytes(op.dtype)));
    }
    if (!allRanksGoOn(ops.back())) {
      break;
    }
    if (replay % 2 == 0 && !stepEpoch()) {
      break;
    }
  }
  EXPECT_EQ(hipSuccess, hipGraphExecDestroy(graphExec));
  EXPECT_EQ(hipSuccess, hipGraphDestroy(graph));
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

  // Rank r sends r + 1 + salt, summing to nRanks(nRanks + 1)/2 + nRanks * salt; per-cycle salts fail stale LL lines.
  bool prepareAllReduce(ncclComm_t comm, void* buf, size_t count, int wantAlgo, int salt, std::string* detail) {
    int algo = -1;
    int protocol = -1;
    int maxChannels = -1;
    if (initializeBufferWithPattern<float>(buf, count, [this, salt](size_t) { return float(rank_ + 1 + salt); }) !=
        hipSuccess) {
      *detail = "buffer init failed";
      return false;
    }
    if (rcclGetCollImplInfo(comm, ncclFuncAllReduce, count, ncclFloat32, ncclSum, buf, buf, 0, &algo, &protocol,
                            &maxChannels) != ncclSuccess ||
        algo != wantAlgo) {
      *detail = "count " + std::to_string(count) + ": AllReduce selected algo " + std::to_string(algo) + ", not " +
                std::to_string(wantAlgo);
      return false;
    }
    return true;
  }

  bool allReduceOk(ncclComm_t comm, void* buf, size_t count, hipStream_t stream, int salt, std::string* detail) {
    if (ncclAllReduce(buf, buf, count, ncclFloat32, ncclSum, comm, stream) != ncclSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
      *detail = "AllReduce failed";
      return false;
    }
    const float want = float(nRanks_ * (nRanks_ + 1) / 2 + nRanks_ * salt);
    size_t badIdx = 0;
    float expected = 0.0f;
    float got = 0.0f;
    if (!verifyBufferData<float>(
          buf, count, [=](size_t) { return want; }, 0, 0.0, &badIdx, &expected, &got)) {
      *detail = "recv[" + std::to_string(badIdx) + "] = " + std::to_string(got) + ", expected " +
                std::to_string(expected);
      return false;
    }
    return true;
  }
};

// Each comm, also one created after a destroy, brings up DDA IPC with no HIP error pending,
// then runs an IPC Simple and an LL one-shot AllReduce.
TEST_F(DdaIpcMPI_CommLifecycle, RecreateAfterDestroy) {
  // Sizes from the comm's AllReduce IPC LL cap: the Simple kernel one line above it, the LL one-shot below it.
  const size_t llCap = rcclDdaLLThreshold(getActiveCommunicator(), ncclFuncAllReduce);
  const size_t simpleCount = llCap / sizeof(float) / 4 * 4 + 4;
  const bool runLL = llCap >= kLifecycleLLCount * sizeof(float);
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
    } else if (hipMalloc(&buf, std::max(simpleCount, kLifecycleLLCount) * sizeof(float)) != hipSuccess) {
      detail = "hipMalloc failed";
    } else {
      guard.set(buf);
      ok = prepareAllReduce(comm, buf, simpleCount, static_cast<int>(RCCL_DDA_IPC), cycle, &detail);
    }
    // A rank that does not launch leaves its peers blocked in the collective, so all ranks launch or none does.
    if (MPIHelpers::allRanksTrue(ok)) {
      ok = allReduceOk(comm, buf, simpleCount, getActiveStream(), cycle, &detail);
    }
    if (runLL && MPIHelpers::allRanksTrue(ok)) {
      ok = prepareAllReduce(comm, buf, kLifecycleLLCount, static_cast<int>(RCCL_DDA_FABRIC_LL), cycle, &detail);
      if (MPIHelpers::allRanksTrue(ok)) {
        ok = allReduceOk(comm, buf, kLifecycleLLCount, getActiveStream(), cycle, &detail);
      }
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
