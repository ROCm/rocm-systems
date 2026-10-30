/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaBroadcastMPITests.cpp
 * @brief MPI end-to-end data tests for the DDA fabric Broadcast one-shot kernels (LL, LL128).
 *
 * Every call carries a payload derived from (root, call sequence, element), so a
 * stale bank or a missed handshake cannot pass. Every rank checks its whole
 * receive allocation: the payload and the guard bytes around it. Before each
 * call rcclGetCollImplInfo must report the DDA tier the case targets.
 *
 * Broadcast DDA is opt-in and its caps come from RCCL_PARAM, which caches on
 * first read, so the settings below must be in the environment before the first
 * communicator. The standard_env_dda_bc category in test_categories_mpi.yaml does
 * that; a process whose cached settings differ skips on every rank.
 *
 * Run (example):
 *   RCCL_DDA_BC=1 RCCL_DDA_LL_THRESHOLD=262144 RCCL_DDA_LL128_THRESHOLD=4194304 \
 *     mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter='DdaMPI_Broadcast*'
 */

#ifdef MPI_TESTS_ENABLED

#include "DdaFabricMPITestFixture.hpp"
#include "param.h"
#include "rccl_common.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

RCCL_PARAM_DECLARE(DdaBc);

using namespace MPITestConstants;

namespace {

constexpr size_t kLLCap = 256 * 1024;
constexpr size_t kLL128Cap = 4 * 1024 * 1024;
// Guard bytes on each side of a payload; a multiple of 16 so the offset alone sets alignment.
constexpr size_t kGuard = 64;
constexpr uint8_t kGuardByte = 0xA5;
constexpr auto kStreamTimeout = std::chrono::seconds(60);
// Host delay before a slowed rank enqueues, so ranks reach a launch at different times.
constexpr auto kSlowRankDelay = std::chrono::milliseconds(5);
// Long enough that every non-root polls its whole bank before the root starts.
constexpr auto kLateRootDelay = std::chrono::milliseconds(200);

// 514 packets over 256 threads: three chunks, the last one partial.
constexpr size_t kLLMultiChunk = 4112;

// (send, recv) byte offsets. The LL launcher takes kAligned=false below 4-byte alignment;
// 4 and 8 keep the aligned path on buffers that are not 16-byte aligned.
constexpr std::pair<size_t, size_t> kLLOffsets[] = {{4, 4}, {8, 8}, {4, 8}, {1, 1}, {2, 6}, {3, 0}, {0, 5}};

// gfx1250 LL128 slice payload. First size past the LL cap: 136 slices and a partial one.
constexpr size_t kLL128SliceBytes = 1920;
constexpr size_t kLL128PartialSlice = kLLCap + 16;
constexpr size_t kLL128WholeSlices = 137 * kLL128SliceBytes;
constexpr size_t kLL128Mid = 1024 * 1024 + 16;
static_assert(kLL128PartialSlice % kLL128SliceBytes != 0 && kLL128WholeSlices > kLLCap, "LL128 sizes");

// The LL128 launcher takes kAligned=false below 16-byte alignment.
constexpr std::pair<size_t, size_t> kLL128Offsets[] = {{4, 4}, {8, 8}, {4, 12}, {1, 3}, {0, 8}, {8, 0}};

// Exact in float: at most 24 significant bits.
float payload(int root, uint32_t seq, size_t i) {
  uint32_t h = static_cast<uint32_t>(i) * 2654435761u;
  h ^= seq * 0x9E3779B1u + static_cast<uint32_t>(root) * 0x85EBCA77u;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 13;
  return static_cast<float>(h & 0xFFFFFFu);
}

// Host image of one allocation: guard bytes, then the payload at kGuard + offset.
// A non-zero fill sets every 32-bit payload word to it instead.
std::vector<uint8_t> makeImage(size_t bytes, size_t offset, int src, uint32_t seq, uint32_t fill = 0) {
  std::vector<uint8_t> image(kGuard + offset + bytes + kGuard, kGuardByte);
  for (size_t i = 0; i < bytes / sizeof(float); ++i) {
    const float v = payload(src, seq, i);
    uint8_t* dst = image.data() + kGuard + offset + i * sizeof(float);
    if (fill != 0) {
      std::memcpy(dst, &fill, sizeof(fill));
    } else {
      std::memcpy(dst, &v, sizeof(v));
    }
  }
  return image;
}

// Empty when they match, else the first differing byte.
std::string firstMismatch(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want, size_t payloadBegin) {
  for (size_t b = 0; b < want.size(); ++b) {
    if (got[b] == want[b]) continue;
    std::ostringstream os;
    os << "byte " << b << " (payload offset " << static_cast<long long>(b) - static_cast<long long>(payloadBegin)
       << ") is 0x" << std::hex << static_cast<int>(got[b]) << ", expected 0x" << static_cast<int>(want[b]);
    return os.str();
  }
  return {};
}

}  // namespace

class DdaBroadcastMPITest : public DdaFabricMPITest {
protected:
  std::vector<std::unique_ptr<MPIHelpers::MpiEnvGuard>> envGuards_;
  int rank_{};
  int nRanks_{};
  // Process-wide, so no two calls in a run share a payload.
  static inline uint32_t seq_{};

  // RCCL_DDA_* settings the tests size against; set before the communicator exists.
  virtual std::vector<std::pair<const char*, std::string>> ddaEnv() const {
    return {{"RCCL_DDA_BC", "1"},
            {"RCCL_DDA_LL_THRESHOLD", std::to_string(kLLCap)},
            {"RCCL_DDA_LL128_THRESHOLD", std::to_string(kLL128Cap)}};
  }

  // Why the cached settings are not the ones the sizes assume, or empty.
  virtual std::string configMismatch() {
    ncclComm* comm = getActiveCommunicator();
    if (rcclParamDdaBc() <= 0) return "RCCL_DDA_BC is not on in this process";
    if (rcclDdaLLThreshold(comm, ncclFuncBroadcast) != kLLCap) return "Broadcast LL cap is not RCCL_DDA_LL_THRESHOLD";
    if (rcclDdaLL128Threshold(comm, ncclFuncBroadcast) != kLL128Cap) {
      return "Broadcast LL128 cap is not RCCL_DDA_LL128_THRESHOLD";
    }
    return {};
  }

  void SetUp() override {
    DdaFabricMPITest::SetUp();
    for (const auto& [name, value] : ddaEnv())
      envGuards_.push_back(std::make_unique<MPIHelpers::MpiEnvGuard>(name, value.c_str()));

    if (!validateTestPrerequisites(kMinProcessesForMPI)) GTEST_SKIP() << "Need at least 2 MPI ranks";
    DDA_FABRIC_SKIP_OR_FAIL(archSkipReason("DDA fabric Broadcast requires gfx1250"));

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(getActiveCommunicator(), &rank_));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(getActiveCommunicator(), &nRanks_));

    DDA_FABRIC_SKIP_OR_FAIL(mpiCoordinatedSkipReason(getActiveCommunicator()->ddaFabricBarrierState == nullptr,
                                                     "DDA fabric did not initialize (MNNVL off or ranks span "
                                                     "cliques), so Broadcast cannot take DDA"));
    const std::string mismatch = configMismatch();
    DDA_FABRIC_SKIP_OR_FAIL(
      mpiCoordinatedSkipReason(!mismatch.empty(), (mismatch + "; run this suite in its own process").c_str()));
  }

  void TearDown() override {
    DdaFabricMPITest::TearDown();
    while (!envGuards_.empty()) envGuards_.pop_back();
  }

  void expectAlgo(size_t bytes, const void* sendBuf, void* recvBuf, int expectedAlgo) {
    int algo = -1, protocol = -1, maxChannels = -1;
    EXPECT_EQ(ncclSuccess,
              rcclGetCollImplInfo(getActiveCommunicator(), ncclFuncBroadcast, bytes / sizeof(float), ncclFloat32,
                                  ncclSum, sendBuf, recvBuf, /*graphCapturing=*/0, &algo, &protocol, &maxChannels));
    const char* algoName = nullptr;
    const char* expectedName = nullptr;
    rcclGetAlgoName(algo, &algoName);
    rcclGetAlgoName(expectedAlgo, &expectedName);
    EXPECT_EQ(algo, expectedAlgo) << "Rank " << rank_ << ": bytes=" << bytes << " selected "
                                  << (algoName ? algoName : "?") << ", expected "
                                  << (expectedName ? expectedName : "?");
  }

  // One allocation (in place) or two; payload at kGuard + offset.
  struct Buffers {
    void* sendBase{};
    void* recvBase{};
    size_t sendOff{};
    size_t recvOff{};
    char* send() const {
      return static_cast<char*>(sendBase) + kGuard + sendOff;
    }
    char* recv() const {
      return static_cast<char*>(recvBase) + kGuard + recvOff;
    }
  };

  struct Call {
    size_t bytes;
    int root;
    bool inPlace;
    size_t sendOff;
    size_t recvOff;
    // Root payload word, or 0 for the hashed pattern.
    uint32_t fill = 0;
  };

  // This rank's buffers and expected images for one call; frees on destruction.
  struct Launch {
    Call c{};
    int expectedAlgo{};
    uint32_t seq{};
    Buffers buf{};
    void* separateSend{};
    std::vector<uint8_t> want;
    std::vector<uint8_t> sendInit;

    Launch() = default;
    Launch(const Launch&) = delete;
    Launch& operator=(const Launch&) = delete;
    ~Launch() {
      if (separateSend != nullptr) (void)hipFree(separateSend);
      if (buf.recvBase != nullptr) (void)hipFree(buf.recvBase);
    }
  };

  // Allocates and fills this rank's buffers; non-roots start from their own data.
  void prepare(Launch& l, const Call& c, int expectedAlgo) {
    l.c = c;
    l.expectedAlgo = expectedAlgo;
    l.seq = ++seq_;
    const size_t sendOff = c.inPlace ? c.recvOff : c.sendOff;
    const size_t allocBytes = 2 * kGuard + 16 + c.bytes;
    l.buf.sendOff = sendOff;
    l.buf.recvOff = c.recvOff;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&l.buf.recvBase, allocBytes));
    if (c.inPlace) {
      l.buf.sendBase = l.buf.recvBase;
    } else {
      ASSERT_MPI_EQ(hipSuccess, hipMalloc(&l.separateSend, allocBytes));
      l.buf.sendBase = l.separateSend;
    }
    const int initSrc = rank_ == c.root ? c.root : nRanks_ + rank_;
    // Out of place, recv starts as junk on the root too, so its local copy is checked.
    const int recvSrc = c.inPlace ? initSrc : nRanks_ + rank_;
    l.want = makeImage(c.bytes, c.recvOff, c.root, l.seq, c.fill);
    const std::vector<uint8_t> recvInit =
      makeImage(c.bytes, c.recvOff, recvSrc, l.seq, rank_ == c.root && c.inPlace ? c.fill : 0);
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(l.buf.recvBase, recvInit.data(), recvInit.size(), hipMemcpyHostToDevice));
    if (!c.inPlace) {
      l.sendInit = makeImage(c.bytes, sendOff, initSrc, l.seq, rank_ == c.root ? c.fill : 0);
      ASSERT_MPI_EQ(hipSuccess, hipMemcpy(l.buf.sendBase, l.sendInit.data(), l.sendInit.size(), hipMemcpyHostToDevice));
    }
  }

  ncclResult_t enqueue(const Launch& l) {
    return ncclBroadcast(l.buf.send(), l.buf.recv(), l.c.bytes / sizeof(float), ncclFloat32, l.c.root,
                         getActiveCommunicator(), getActiveStream());
  }

  void drain() {
    DdaFabricMPI::abortJobUnlessAllRanksDrained(DdaFabricMPI::waitStreamBounded(getActiveStream(), kStreamTimeout),
                                                "DDA Broadcast");
  }

  // Empty when the recv allocation (and the root's send allocation) hold what they should.
  std::string verify(const Launch& l) {
    std::string err = checkAllocation(l.buf.recvBase, l.want, kGuard + l.c.recvOff, "recv");
    if (err.empty() && !l.c.inPlace && rank_ == l.c.root) {
      err = checkAllocation(l.buf.sendBase, l.sendInit, kGuard + l.buf.sendOff, "send");
    }
    if (err.empty()) return err;
    std::ostringstream os;
    os << "Rank " << rank_ << " seq=" << l.seq << " bytes=" << l.c.bytes << " root=" << l.c.root
       << " inPlace=" << l.c.inPlace << " sendOff=" << l.buf.sendOff << " recvOff=" << l.c.recvOff << ": " << err;
    return os.str();
  }

  // Every rank: fill, broadcast, wait (bounded), then check the payload and guards.
  void broadcastAndCheck(const Call& c, int expectedAlgo) {
    Launch l;
    prepare(l, c, expectedAlgo);
    if (HasFatalFailure() || IsSkipped()) return;
    expectAlgo(c.bytes, l.buf.send(), l.buf.recv(), expectedAlgo);
    ASSERT_MPI_EQ(ncclSuccess, enqueue(l));
    drain();
    const std::string err = verify(l);
    if (!err.empty()) ADD_FAILURE() << err;
    ASSERT_MPI_TRUE(err.empty());
  }

  // Enqueues every call on one stream with no sync or barrier in between, then checks them all.
  // slowRank(j) sleeps before enqueuing call j; -1 slows nobody.
  void backToBack(const std::vector<std::pair<Call, int>>& calls, const std::function<int(size_t)>& slowRank,
                  std::chrono::milliseconds delay = kSlowRankDelay) {
    std::vector<std::unique_ptr<Launch>> ls;
    for (const auto& [c, algo] : calls) {
      ls.push_back(std::make_unique<Launch>());
      prepare(*ls.back(), c, algo);
      if (HasFatalFailure() || IsSkipped()) return;
    }
    for (const auto& l : ls) expectAlgo(l->c.bytes, l->buf.send(), l->buf.recv(), l->expectedAlgo);
    // Only the fills are fenced; the launches below run unsynchronised.
    MPI_Barrier(MPI_COMM_WORLD);
    ncclResult_t res = ncclSuccess;
    for (size_t j = 0; j < ls.size() && res == ncclSuccess; ++j) {
      if (rank_ == slowRank(j)) std::this_thread::sleep_for(delay);
      res = enqueue(*ls[j]);
    }
    drain();
    ASSERT_MPI_EQ(ncclSuccess, res);
    bool ok = true;
    for (const auto& l : ls) {
      const std::string err = verify(*l);
      if (!err.empty()) ADD_FAILURE() << err;
      ok = ok && err.empty();
    }
    ASSERT_MPI_TRUE(ok);
  }

  // Seven calls, roots 0 0 1 n-1 n-1 2 0 (repeats and changes), odd calls in place; four slowing patterns.
  void backToBackRounds(const size_t (&sizes)[7], int expectedAlgo) {
    std::pair<size_t, int> steps[7];
    for (size_t j = 0; j < 7; ++j) steps[j] = {sizes[j], expectedAlgo};
    backToBackRounds(steps);
  }

  void backToBackRounds(const std::pair<size_t, int> (&steps)[7]) {
    const int roots[7] = {0, 0, 1 % nRanks_, nRanks_ - 1, nRanks_ - 1, 2 % nRanks_, 0};
    std::vector<std::pair<Call, int>> calls;
    for (size_t j = 0; j < 7; ++j) calls.push_back({{steps[j].first, roots[j], j % 2 == 1, 0, 0}, steps[j].second});
    const std::function<int(size_t)> patterns[] = {
      [&](size_t j) { return (roots[j] + 1 + static_cast<int>(j) % (nRanks_ - 1)) % nRanks_; },  // a non-root
      [&](size_t j) { return roots[j]; },                                                        // the root
      [&](size_t) { return nRanks_ - 1; },                                                       // one fixed rank
      [](size_t) { return -1; },                                                                 // nobody
    };
    for (const auto& slow : patterns) {
      backToBack(calls, slow);
      if (HasFatalFailure() || IsSkipped()) return;
    }
  }

  std::string checkAllocation(const void* base, const std::vector<uint8_t>& want, size_t payloadBegin,
                              const char* what) {
    std::vector<uint8_t> got(want.size());
    if (hipMemcpy(got.data(), base, got.size(), hipMemcpyDeviceToHost) != hipSuccess) return "copy-back failed";
    const std::string m = firstMismatch(got, want, payloadBegin);
    return m.empty() ? m : std::string(what) + " " + m;
  }

  void allRoots(std::initializer_list<size_t> sizes, bool inPlace, int expectedAlgo) {
    for (size_t bytes : sizes) {
      for (int root = 0; root < nRanks_; ++root) {
        broadcastAndCheck({bytes, root, inPlace, 0, 0}, expectedAlgo);
        if (HasFatalFailure() || IsSkipped()) return;
      }
    }
  }

  // Back-to-back launches on one comm, root rotating, each checked before the next.
  void repeated(size_t bytes, int iterations, int expectedAlgo) {
    for (int it = 0; it < iterations; ++it) {
      broadcastAndCheck({bytes, it % nRanks_, it % 2 == 1, 0, 0}, expectedAlgo);
      if (HasFatalFailure() || IsSkipped()) return;
    }
  }

  // Value of this rank's epoch cells; the stream must be idle.
  uint32_t currentEpoch() {
    uint32_t e = 0;
    if (hipMemcpy(&e, getActiveCommunicator()->ddaLLEpochDev, sizeof(e), hipMemcpyDeviceToHost) != hipSuccess) return 0;
    return e;
  }

  void unaligned(std::initializer_list<size_t> sizes, const std::pair<size_t, size_t>* offsets, size_t nOffsets,
                 int expectedAlgo) {
    int k = 0;
    for (size_t bytes : sizes) {
      for (size_t o = 0; o < nOffsets; ++o, ++k) {
        broadcastAndCheck({bytes, k % nRanks_, false, offsets[o].first, offsets[o].second}, expectedAlgo);
        if (HasFatalFailure() || IsSkipped()) return;
        broadcastAndCheck({bytes, (k + 1) % nRanks_, true, 0, offsets[o].second}, expectedAlgo);
        if (HasFatalFailure() || IsSkipped()) return;
      }
    }
  }
};

class DdaMPI_BroadcastLL : public DdaBroadcastMPITest {};

// Every root, every LL size up to the cap, separate buffers.
TEST_F(DdaMPI_BroadcastLL, AllRootsOutOfPlace) {
  allRoots({16, 1024, kLLMultiChunk, 64 * 1024, kLLCap}, false, RCCL_DDA_FABRIC_LL);
}

// Every root, every LL size up to the cap, sendbuff == recvbuff.
TEST_F(DdaMPI_BroadcastLL, AllRootsInPlace) {
  allRoots({16, 1024, kLLMultiChunk, 64 * 1024, kLLCap}, true, RCCL_DDA_FABRIC_LL);
}

// Odd then even run of launches: both banks, each start token, a fresh payload every call.
TEST_F(DdaMPI_BroadcastLL, RepeatedLaunches) {
  for (size_t bytes : {size_t{16}, kLLMultiChunk, kLLCap}) {
    repeated(bytes, 5, RCCL_DDA_FABRIC_LL);
    if (HasFatalFailure() || IsSkipped()) return;
    repeated(bytes, 4, RCCL_DDA_FABRIC_LL);
    if (HasFatalFailure() || IsSkipped()) return;
  }
}

// Buffers off 16-byte alignment (aligned path) and off 4-byte alignment (kAligned=false).
TEST_F(DdaMPI_BroadcastLL, UnalignedBuffers) {
  unaligned({16, kLLMultiChunk, 64 * 1024}, kLLOffsets, std::size(kLLOffsets), RCCL_DDA_FABRIC_LL);
}

// Seven launches with no sync between them: ranks run ahead into the reused bank and the start-token wait.
TEST_F(DdaMPI_BroadcastLL, BackToBackNoSync) {
  const size_t sizes[7] = {16, kLLMultiChunk, 64 * 1024, kLLCap, 1024, kLLMultiChunk, 16};
  backToBackRounds(sizes, RCCL_DDA_FABRIC_LL);
}

class DdaMPI_BroadcastLL128 : public DdaBroadcastMPITest {};

// Every root, partial and whole last slices up to the cap, separate buffers.
TEST_F(DdaMPI_BroadcastLL128, AllRootsOutOfPlace) {
  allRoots({kLL128PartialSlice, kLL128WholeSlices, kLL128Mid, kLL128Cap}, false, RCCL_DDA_FABRIC_LL128);
}

// Every root, partial and whole last slices up to the cap, sendbuff == recvbuff.
TEST_F(DdaMPI_BroadcastLL128, AllRootsInPlace) {
  allRoots({kLL128PartialSlice, kLL128WholeSlices, kLL128Mid, kLL128Cap}, true, RCCL_DDA_FABRIC_LL128);
}

// Odd then even run of launches: both banks, each start token, a fresh payload every call.
TEST_F(DdaMPI_BroadcastLL128, RepeatedLaunches) {
  for (size_t bytes : {kLL128PartialSlice, kLL128Mid}) {
    repeated(bytes, 5, RCCL_DDA_FABRIC_LL128);
    if (HasFatalFailure() || IsSkipped()) return;
    repeated(bytes, 4, RCCL_DDA_FABRIC_LL128);
    if (HasFatalFailure() || IsSkipped()) return;
  }
}

// Buffers off 16-byte alignment (kAligned=false).
TEST_F(DdaMPI_BroadcastLL128, UnalignedBuffers) {
  unaligned({kLL128PartialSlice, kLL128Mid}, kLL128Offsets, std::size(kLL128Offsets), RCCL_DDA_FABRIC_LL128);
}

// Seven launches with no sync between them: ranks run ahead into the reused bank and the start-token wait.
TEST_F(DdaMPI_BroadcastLL128, BackToBackNoSync) {
  const size_t sizes[7] = {kLL128PartialSlice, kLL128Mid, kLL128WholeSlices, kLL128Cap,
                           kLL128PartialSlice, kLL128Mid, kLL128PartialSlice};
  backToBackRounds(sizes, RCCL_DDA_FABRIC_LL128);
}

class DdaMPI_BroadcastTransition : public DdaBroadcastMPITest {};

// LL, LL128, LL on one comm, twice over: each protocol reads the scratch the other just wrote.
TEST_F(DdaMPI_BroadcastTransition, LLThenLL128ThenLL) {
  const std::pair<size_t, int> steps[] = {{kLLMultiChunk, RCCL_DDA_FABRIC_LL},
                                          {kLL128PartialSlice, RCCL_DDA_FABRIC_LL128},
                                          {64 * 1024, RCCL_DDA_FABRIC_LL},
                                          {kLL128Mid, RCCL_DDA_FABRIC_LL128},
                                          {kLLCap, RCCL_DDA_FABRIC_LL}};
  int k = 0;
  for (int round = 0; round < 2; ++round) {
    for (const auto& [bytes, algo] : steps) {
      broadcastAndCheck({bytes, k % nRanks_, k % 2 == 1, 0, 0}, algo);
      if (HasFatalFailure() || IsSkipped()) return;
      ++k;
    }
  }
}

// LL and LL128 interleaved with no sync between launches.
TEST_F(DdaMPI_BroadcastTransition, BackToBackNoSync) {
  const std::pair<size_t, int> steps[7] = {{kLLMultiChunk, RCCL_DDA_FABRIC_LL},
                                           {kLL128PartialSlice, RCCL_DDA_FABRIC_LL128},
                                           {64 * 1024, RCCL_DDA_FABRIC_LL},
                                           {kLL128Mid, RCCL_DDA_FABRIC_LL128},
                                           {kLLCap, RCCL_DDA_FABRIC_LL},
                                           {kLL128Cap, RCCL_DDA_FABRIC_LL128},
                                           {16, RCCL_DDA_FABRIC_LL}};
  backToBackRounds(steps);
}

// Stale flags: an LL128 call carries, in every 32-bit word, the flag of the LL call two launches later in the
// same bank, so its leftovers would pass that call's line polls. The LL root enqueues late, so non-roots poll
// the leftovers first; only the LL128 receivers clearing what they consumed keeps the LL data correct.
TEST_F(DdaMPI_BroadcastTransition, LL128LeftoversDoNotPassLLPoll) {
  for (int round = 0; round < 3; ++round) {
    const int root = round % nRanks_;
    const uint32_t epoch = currentEpoch();
    uint32_t lo = epoch, hi = epoch;
    MPI_Allreduce(MPI_IN_PLACE, &lo, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &hi, 1, MPI_UINT32_T, MPI_MAX, MPI_COMM_WORLD);
    ASSERT_MPI_EQ(lo, hi);
    // Flags: LL128 epoch+1, filler epoch+2 (other bank), target LL epoch+3 (the LL128 bank).
    broadcastAndCheck({kLL128PartialSlice, root, false, 0, 0, epoch + 3}, RCCL_DDA_FABRIC_LL128);
    if (HasFatalFailure() || IsSkipped()) return;
    broadcastAndCheck({16, root, false, 0, 0}, RCCL_DDA_FABRIC_LL);
    if (HasFatalFailure() || IsSkipped()) return;
    backToBack({{{kLLCap, root, false, 0, 0}, RCCL_DDA_FABRIC_LL}}, [root](size_t) { return root; }, kLateRootDelay);
    if (HasFatalFailure() || IsSkipped()) return;
    ASSERT_MPI_EQ(currentEpoch(), epoch + 3);
  }
}

#endif  // MPI_TESTS_ENABLED
