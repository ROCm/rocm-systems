/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/misc/shmutils.cc over the real os/linux.cc shared memory, with local
// ranks played by threads for ncclShmemAllgather.

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <unistd.h>
#include <vector>

#include "comm.h"
#include "shmutils.h"

#include "linux-test-common.h"

#include SHMUTILS_CC_PATH

namespace {

class ShmutilsMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { g_linuxLog.clear(); }
  void TearDown() override {
    for (int k = 0; k < 2; k++) {
      free(coll_.cnt[k]);
      free(coll_.ptr[k]);
    }
  }
  // Shared counters and slots for nRanks ranks.
  void InitColl(int nRanks, size_t maxTypeSize) {
    coll_.maxTypeSize = maxTypeSize;
    coll_.round = 0;
    for (int k = 0; k < 2; k++) {
      coll_.cnt[k] = (size_t*)calloc(nRanks, CACHE_LINE_SIZE);
      coll_.ptr[k] = calloc(nRanks, maxTypeSize);
    }
  }
  static std::unique_ptr<ncclComm> Comm(int localRank, int localRanks) {
    std::unique_ptr<ncclComm> comm(new ncclComm());
    comm->localRank = localRank;
    comm->localRanks = localRanks;
    return comm;
  }
  ncclShmemCollBuff coll_ = {};
};

TEST_F(ShmutilsMicrotest, OpenAttachUnlinkAndClose) {
  char path[64] = "";
  void* ptr = nullptr;
  ncclShmHandle_t creator = nullptr, peer = nullptr;
  ASSERT_EQ(ncclShmOpen(path, sizeof(path), 128, &ptr, nullptr, 2, &creator), ncclSuccess);
  ASSERT_NE(creator, nullptr);
  EXPECT_EQ(((ncclShmHandleInternal*)creator)->shmPtr, ptr);
  std::strcpy((char*)ptr, "hi");
  void* peerPtr = nullptr;
  ASSERT_EQ(ncclShmOpen(path, sizeof(path), 128, &peerPtr, nullptr, 0, &peer), ncclSuccess);
  EXPECT_STREQ((char*)peerPtr, "hi");
  EXPECT_EQ(access(path, F_OK), 0);  // one peer still to attach
  ASSERT_EQ(ncclShmUnlink(creator), ncclSuccess);
  EXPECT_NE(access(path, F_OK), 0);
  EXPECT_EQ(ncclShmClose(peer), ncclSuccess);
  EXPECT_EQ(ncclShmClose(creator), ncclSuccess);
}

TEST_F(ShmutilsMicrotest, OpenFailureReturnsANullHandle) {
  char path[64] = "/nccl/no/such/dir/seg";
  void* ptr = nullptr;
  ncclShmHandle_t h = (ncclShmHandle_t)0x1;
  EXPECT_EQ(ncclShmOpen(path, sizeof(path), 128, &ptr, nullptr, 1, &h), ncclSystemError);
  EXPECT_EQ(h, nullptr);
}

TEST_F(ShmutilsMicrotest, Allgather_RejectsMissingArgumentsAndAnOversizedType) {
  InitColl(1, 8);
  auto comm = Comm(0, 1);
  char send[16] = {}, recv[16];
  EXPECT_EQ(ncclShmemAllgather(nullptr, &coll_, send, recv, 4), ncclInvalidArgument);
  EXPECT_EQ(ncclShmemAllgather(comm.get(), &coll_, nullptr, recv, 4), ncclInvalidArgument);
  EXPECT_EQ(ncclShmemAllgather(comm.get(), &coll_, send, nullptr, 4), ncclInvalidArgument);
  EXPECT_EQ(ncclShmemAllgather(comm.get(), &coll_, send, recv, 9), ncclInvalidArgument);
  EXPECT_EQ(coll_.round, 0);
  EXPECT_EQ(ncclShmemAllgather(comm.get(), &coll_, send, recv, 8), ncclSuccess);
}

// Observed: shmem is read before the NULL check, so a NULL shmem crashes instead of failing.
TEST_F(ShmutilsMicrotest, Allgather_NullShmemCrashes) {
  auto comm = Comm(0, 1);
  char send[4] = {}, recv[4];
  EXPECT_DEATH(ncclShmemAllgather(comm.get(), nullptr, send, recv, 4), "");
}

TEST_F(ShmutilsMicrotest, Allgather_SingleRankAlternatesBuffersEachRound) {
  InitColl(1, 8);
  auto comm = Comm(0, 1);
  for (int round = 1; round <= 3; round++) {
    int send = 100 + round, recv = 0;
    ASSERT_EQ(ncclShmemAllgather(comm.get(), &coll_, &send, &recv, sizeof(int)), ncclSuccess);
    EXPECT_EQ(recv, send);
    EXPECT_EQ(coll_.round, round);
    int slot = (round - 1) % 2;
    EXPECT_EQ(*(int*)coll_.cnt[slot], round);  // this rank's arrival
    EXPECT_EQ(*(int*)coll_.ptr[slot], send);
  }
}

// Each thread is a local rank with its own round counter over the shared slots.
TEST_F(ShmutilsMicrotest, Allgather_EveryRankGetsEveryRanksValueInRankOrder) {
  constexpr int kRanks = 4, kRounds = 5;
  InitColl(kRanks, 16);  // slots wider than the type: packed into recvbuff
  std::vector<std::vector<int>> got(kRanks);
  std::vector<std::thread> threads;
  for (int r = 0; r < kRanks; r++) {
    threads.emplace_back([&, r] {
      auto comm = Comm(r, kRanks);
      ncclShmemCollBuff mine = coll_;
      for (int round = 0; round < kRounds; round++) {
        int send = 10 * round + r, recv[kRanks];
        ASSERT_EQ(ncclShmemAllgather(comm.get(), &mine, &send, recv, sizeof(int)), ncclSuccess);
        got[r].insert(got[r].end(), recv, recv + kRanks);
      }
      EXPECT_EQ(mine.round, kRounds);
    });
  }
  for (auto& t : threads) t.join();
  std::vector<int> expected;
  for (int round = 0; round < kRounds; round++)
    for (int r = 0; r < kRanks; r++) expected.push_back(10 * round + r);
  for (int r = 0; r < kRanks; r++) EXPECT_EQ(got[r], expected) << "rank " << r;
}

}  // namespace
