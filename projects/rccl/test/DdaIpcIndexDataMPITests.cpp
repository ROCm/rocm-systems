/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaIpcIndexDataMPITests.cpp
 * @brief MPI end-to-end checks for the DDA IPC AllReduce, ReduceScatter and AlltoAll kernels and for DDA IPC init.
 *
 * DdaIpcMPI_IndexData: every rank sends rank- and index-dependent data, so each element and each slice of the result
 * is distinct. A per-rank constant fill sums to the same value everywhere and hides a kernel that reads or writes the
 * wrong offset. AlltoAll runs in place and out of place, back to back with a DDA IPC AllReduce after each call; every
 * size that selects DDA IPC is data-checked, and the sizes just over each cap must select another path.
 *
 * DdaIpcMPI_CommLifecycle: DDA IPC must come up on a communicator created after another one is destroyed. It needs
 * cuMem, and NCCL_CUMEM_ENABLE is cached at the first read, so run this suite in its own process.
 *
 * Run (example):
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=DdaIpcMPI_IndexData.*
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=DdaIpcMPI_CommLifecycle.*
 */

#ifdef MPI_TESTS_ENABLED

#include <cstdint>
#include <cstdlib>
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

// AlltoAll bytes per peer: one block, several blocks, and each side of the 4 MiB (gfx942) and 64 MiB (gfx950) caps.
constexpr size_t kA2aPerPeerBytes[] = {
  251ULL * 16,          // single block, prime line count
  (8ULL << 10) + 16,    // two blocks
  512ULL << 10,         // 4 MiB total
  (512ULL << 10) + 16,  // just over 4 MiB total
  2ULL << 20,           // 16 MiB total
  8ULL << 20,           // 64 MiB total
  (8ULL << 20) + 16,    // just over 64 MiB total
};

// The first three sizes are within every arch's DDA IPC AlltoAll cap, so they always run.
constexpr size_t kA2aCasesUpTo4Mb = 3;

// fp32 AllReduce run after each AlltoAll: above the flat/tree threshold, so it also stages through the scratch.
constexpr size_t kA2aInterleavedArCount = 1u << 20;

// Word w of the chunk rank src sends to rank dst. Pair and index both differ, so a chunk at the wrong peer or offset
// fails; 24 index bits outlast the largest chunk (2M + 4 words).
uint32_t a2aPattern(int src, int dst, size_t w) {
  return (static_cast<uint32_t>(src) << 28) | (static_cast<uint32_t>(dst) << 24) | static_cast<uint32_t>(w & 0xFFFFFF);
}

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

  // Runs every AlltoAll size out of place, then in place.
  void runAlltoAll() {
    // RCCL_CE_ALLREDUCE=1 lets CE take the interleaved AllReduce ahead of DDA.
    const bool ceAllReduce = rcclCeAllReduceEnabled(getActiveCommunicator());
    const std::string skip = mpiCoordinatedSkipReason(ceAllReduce, "CE AllReduce routes away from DDA IPC");
    if (!skip.empty()) {
      GTEST_SKIP() << skip;
    }
    for (bool inPlace : {false, true}) {
      runAlltoAllOnce(inPlace);
      if (HasFailure() || IsSkipped()) {
        return;
      }
    }
  }

  // An AlltoAll arena call: where in the arena it ran and its bytes per peer.
  struct A2aCall {
    size_t offset;
    size_t perPeer;
  };

  // Launches every DDA IPC size back to back, each followed by a DDA IPC AllReduce, syncs once, then checks all.
  void runAlltoAllOnce(bool inPlace) {
    ncclComm* comm = getActiveCommunicator();
    const size_t ddaMax = rcclDdaEntryThreshold(comm, ncclFuncAlltoAll);
    size_t arenaBytes = 0;
    for (size_t perPeer : kA2aPerPeerBytes) {
      arenaBytes += perPeer * static_cast<size_t>(nRanks_);
    }
    void* sendArena = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendArena, arenaBytes));
    DeviceBufferAutoGuard sendGuard(sendArena);
    void* recvArena = sendArena;
    DeviceBufferAutoGuard recvGuard;
    if (!inPlace) {
      ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvArena, arenaBytes));
      recvGuard.set(recvArena);
      ASSERT_MPI_EQ(hipSuccess, hipMemset(recvArena, 0xA5, arenaBytes));
    }
    // Rank r sends r + 1 to every AllReduce, so each result element is nRanks * (nRanks + 1) / 2.
    const size_t nSizes = sizeof(kA2aPerPeerBytes) / sizeof(kA2aPerPeerBytes[0]);
    void* arSend = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&arSend, kA2aInterleavedArCount * sizeof(float)));
    DeviceBufferAutoGuard arSendGuard(arSend);
    void* arRecv = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&arRecv, nSizes * kA2aInterleavedArCount * sizeof(float)));
    DeviceBufferAutoGuard arRecvGuard(arRecv);
    const int rank = rank_;
    ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(arSend, kA2aInterleavedArCount,
                                                                 [=](size_t) { return float(rank + 1); }));
    int arAlgo = -1;
    int arProtocol = -1;
    int arChannels = -1;
    ASSERT_MPI_EQ(ncclSuccess, rcclGetCollImplInfo(comm, ncclFuncAllReduce, kA2aInterleavedArCount, ncclFloat32,
                                                   ncclSum, arSend, arRecv, 0, &arAlgo, &arProtocol, &arChannels));
    ASSERT_MPI_EQ(static_cast<int>(RCCL_DDA_IPC), arAlgo);

    std::vector<A2aCall> launched;
    size_t offset = 0;
    for (size_t perPeer : kA2aPerPeerBytes) {
      const size_t total = perPeer * static_cast<size_t>(nRanks_);
      const A2aCall call{offset, perPeer};
      offset += total;
      char* send = static_cast<char*>(sendArena) + call.offset;
      char* recv = static_cast<char*>(recvArena) + call.offset;
      const size_t count = perPeer / sizeof(float);
      int algo = -1;
      int protocol = -1;
      int maxChannels = -1;
      ASSERT_MPI_EQ(ncclSuccess, rcclGetCollImplInfo(comm, ncclFuncAlltoAll, count, ncclFloat32, ncclSum, send, recv,
                                                     0, &algo, &protocol, &maxChannels));
      const char* algoName = nullptr;
      rcclGetAlgoName(algo, &algoName);
      const bool dda = total <= ddaMax;
      EXPECT_EQ(dda, algo == static_cast<int>(RCCL_DDA_IPC))
        << "Rank " << rank_ << ": total " << total << " selected " << (algoName ? algoName : "?");
      ASSERT_MPI_EQ(dda, algo == static_cast<int>(RCCL_DDA_IPC));
      if (!dda) {
        continue;
      }
      const size_t wordsPerPeer = perPeer / sizeof(uint32_t);
      ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<uint32_t>(send, total / sizeof(uint32_t), [=](size_t w) {
                      return a2aPattern(rank, static_cast<int>(w / wordsPerPeer), w % wordsPerPeer);
                    }));
      ASSERT_MPI_EQ(ncclSuccess, ncclAlltoAll(send, recv, count, ncclFloat32, comm, getActiveStream()));
      float* arOut = static_cast<float*>(arRecv) + launched.size() * kA2aInterleavedArCount;
      ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(arSend, arOut, kA2aInterleavedArCount, ncclFloat32, ncclSum, comm,
                                               getActiveStream()));
      launched.push_back(call);
    }
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
    EXPECT_GE(launched.size(), kA2aCasesUpTo4Mb) << "Rank " << rank_ << ": DDA cap " << ddaMax;

    const std::string tag = inPlace ? "AlltoAll in-place" : "AlltoAll out-of-place";
    std::string detail;
    const bool ok =
      verifyAlltoAll(recvArena, launched, &detail) && verifyInterleavedAllReduce(arRecv, launched.size(), &detail);
    EXPECT_TRUE(ok) << "Rank " << rank_ << ": " << tag << ": " << detail;
    int badRanks = ok ? 0 : 1;
    MPI_Allreduce(MPI_IN_PLACE, &badRanks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(0, badRanks) << tag << ": " << badRanks << " rank(s) received wrong values";
  }

  // Word w of chunk p in this rank's receive buffer must be the word w peer p sent it.
  bool verifyAlltoAll(const void* recvArena, const std::vector<A2aCall>& launched, std::string* detail) {
    const int rank = rank_;
    for (const A2aCall& call : launched) {
      const size_t wordsPerPeer = call.perPeer / sizeof(uint32_t);
      size_t badIdx = 0;
      uint32_t want = 0;
      uint32_t got = 0;
      if (!verifyBufferData<uint32_t>(
            static_cast<const char*>(recvArena) + call.offset, wordsPerPeer * static_cast<size_t>(nRanks_),
            [=](size_t w) { return a2aPattern(static_cast<int>(w / wordsPerPeer), rank, w % wordsPerPeer); }, 0, 0.0,
            &badIdx, &want, &got)) {
        *detail = "perPeerBytes=" + std::to_string(call.perPeer) + " word " + std::to_string(badIdx) + " = " +
                  std::to_string(got) + ", expected " + std::to_string(want);
        return false;
      }
    }
    return true;
  }

  bool verifyInterleavedAllReduce(const void* arRecv, size_t nCalls, std::string* detail) {
    const float wantSum = float(nRanks_ * (nRanks_ + 1) / 2);
    size_t badIdx = 0;
    float want = 0.0f;
    float got = 0.0f;
    if (!verifyBufferData<float>(
          arRecv, nCalls * kA2aInterleavedArCount, [=](size_t) { return wantSum; }, 0, 0.0, &badIdx, &want, &got)) {
      *detail = "interleaved AllReduce element " + std::to_string(badIdx) + " = " + std::to_string(got) +
                ", expected " + std::to_string(want);
      return false;
    }
    return true;
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

// Each rank sends chunk p of its buffer to peer p.
TEST_F(DdaIpcMPI_IndexData, AlltoAll) {
  runAlltoAll();
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
