/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaBroadcastMPITests.cpp
 * @brief MPI end-to-end data tests for the DDA fabric Broadcast one-shot kernels.
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
 *   RCCL_DDA_BC=1 RCCL_DDA_LL_THRESHOLD=262144 \
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
// Guard bytes on each side of a payload; a multiple of 16 so the offset alone sets alignment.
constexpr size_t kGuard = 64;
constexpr uint8_t kGuardByte = 0xA5;
constexpr auto kStreamTimeout = std::chrono::seconds(60);
// Host delay before a slowed rank enqueues, so ranks reach a launch at different times.
constexpr auto kSlowRankDelay = std::chrono::milliseconds(5);

// 514 packets over 256 threads: three chunks, the last one partial.
constexpr size_t kLLMultiChunk = 4112;

// (send, recv) byte offsets. The LL launcher takes kAligned=false below 4-byte alignment;
// 4 and 8 keep the aligned path on buffers that are not 16-byte aligned.
constexpr std::pair<size_t, size_t> kLLOffsets[] = {{4, 4}, {8, 8}, {4, 8}, {1, 1}, {2, 6}, {3, 0}, {0, 5}};

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
std::vector<uint8_t> makeImage(size_t bytes, size_t offset, int src, uint32_t seq) {
  std::vector<uint8_t> image(kGuard + offset + bytes + kGuard, kGuardByte);
  for (size_t i = 0; i < bytes / sizeof(float); ++i) {
    const float v = payload(src, seq, i);
    std::memcpy(image.data() + kGuard + offset + i * sizeof(float), &v, sizeof(v));
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
    return {{"RCCL_DDA_BC", "1"}, {"RCCL_DDA_LL_THRESHOLD", std::to_string(kLLCap)}};
  }

  // Why the cached settings are not the ones the sizes assume, or empty.
  virtual std::string configMismatch() {
    ncclComm* comm = getActiveCommunicator();
    if (rcclParamDdaBc() <= 0) return "RCCL_DDA_BC is not on in this process";
    if (rcclDdaLLThreshold(comm, ncclFuncBroadcast) != kLLCap) return "Broadcast LL cap is not RCCL_DDA_LL_THRESHOLD";
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
    l.want = makeImage(c.bytes, c.recvOff, c.root, l.seq);
    const std::vector<uint8_t> recvInit = makeImage(c.bytes, c.recvOff, recvSrc, l.seq);
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(l.buf.recvBase, recvInit.data(), recvInit.size(), hipMemcpyHostToDevice));
    if (!c.inPlace) {
      l.sendInit = makeImage(c.bytes, sendOff, initSrc, l.seq);
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
  void backToBack(const std::vector<std::pair<Call, int>>& calls, const std::function<int(size_t)>& slowRank) {
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
      if (rank_ == slowRank(j)) std::this_thread::sleep_for(kSlowRankDelay);
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
    const int roots[7] = {0, 0, 1 % nRanks_, nRanks_ - 1, nRanks_ - 1, 2 % nRanks_, 0};
    std::vector<std::pair<Call, int>> calls;
    for (size_t j = 0; j < 7; ++j) calls.push_back({{sizes[j], roots[j], j % 2 == 1, 0, 0}, expectedAlgo});
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


// ---- DDA fabric Broadcast LL128 two-shot ----
//
// Environment set by standard_env_dda_bc_ll128 in test_categories_mpi.yaml:
//   RCCL_DDA_BC=1                      -- opt-in to DDA Broadcast
//   RCCL_DDA_LL_THRESHOLD=0            -- suppress LL tier so LL128 is chosen
//   RCCL_DDA_LL128_THRESHOLD=<cap>     -- cap LL128 tier (kLL128Cap bytes)
//   RCCL_DDA_LL128_BC_FANOUT=1         -- exercise the two-shot fan-out path

// One LL128 slice carries 1920 bytes of payload (15 × 8-byte words, 32 threads/warp).
// kLL128MultiChunk is 5.5 slices: exercises a partial last chunk in the two-shot tiling.
// kLL128Cap is the fixed env cap; kLLMultiChunk reuses the LL multi-chunk constant.
constexpr size_t kLL128MultiChunk = 1920 * 5 + 960;  // 10560 bytes, partial last chunk
constexpr size_t kLL128Cap = 2 * 1024 * 1024;         // cap baked into the YAML category

class DdaMPI_BroadcastLL128 : public DdaBroadcastMPITest {
 protected:
  std::vector<std::pair<const char*, std::string>> ddaEnv() const override {
    return {
      {"RCCL_DDA_BC", "1"},
      {"RCCL_DDA_LL_THRESHOLD", "0"},
      {"RCCL_DDA_LL128_THRESHOLD", std::to_string(kLL128Cap)},
      {"RCCL_DDA_LL128_BC_FANOUT", "1"},
    };
  }

  std::string configMismatch() override {
    ncclComm* comm = getActiveCommunicator();
    if (rcclParamDdaBc() <= 0) return "RCCL_DDA_BC is not on in this process";
    if (rcclDdaLLThreshold(comm, ncclFuncBroadcast) != 0)
      return "RCCL_DDA_LL_THRESHOLD is not 0; LL tier is not suppressed";
    if (rcclDdaLL128Threshold(comm, ncclFuncBroadcast) != kLL128Cap)
      return "Broadcast LL128 cap is not kLL128Cap; check RCCL_DDA_LL128_THRESHOLD";
    return {};
  }
};

// Nonzero root: rotate through every root so a non-root rank scatters and receives,
// confirming the two-shot fan-out for all sender/receiver pairings.
TEST_F(DdaMPI_BroadcastLL128, AllRootsOutOfPlace) {
  allRoots({16, kLL128MultiChunk, 1024 * 1024, kLL128Cap}, false, RCCL_DDA_FABRIC_LL128);
}

// In-place: sendbuff == recvbuff; root writes into its own recv buffer and all
// non-roots overwrite the same allocation with the broadcast value.
TEST_F(DdaMPI_BroadcastLL128, AllRootsInPlace) {
  allRoots({16, kLL128MultiChunk, 1024 * 1024, kLL128Cap}, true, RCCL_DDA_FABRIC_LL128);
}

// Partial last chunk: kLL128MultiChunk (5.5 slices) exercises the tail-slice handling
// in the two-shot tiling where the last chunk is not a full set of slices.
TEST_F(DdaMPI_BroadcastLL128, PartialLastChunk) {
  for (int root = 0; root < nRanks_; ++root) {
    broadcastAndCheck({kLL128MultiChunk, root, false, 0, 0}, RCCL_DDA_FABRIC_LL128);
    if (HasFatalFailure() || IsSkipped()) return;
    broadcastAndCheck({kLL128MultiChunk, root, true, 0, 0}, RCCL_DDA_FABRIC_LL128);
    if (HasFatalFailure() || IsSkipped()) return;
  }
}

// Back-to-back launches: exercises bank alternation and start-token reuse in the
// two-shot path with RCCL_DDA_LL128_BC_FANOUT=1.
TEST_F(DdaMPI_BroadcastLL128, BackToBackNoSync) {
  const size_t sizes[7] = {16, kLL128MultiChunk, 1024 * 1024, kLL128Cap, kLL128MultiChunk, 16, 1024 * 1024};
  backToBackRounds(sizes, RCCL_DDA_FABRIC_LL128);
}

#endif  // MPI_TESTS_ENABLED
