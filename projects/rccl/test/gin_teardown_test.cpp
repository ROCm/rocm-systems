/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/dev_runtime.cc.
 *
 * This translation unit #includes the (hipified) dev_runtime.cc source
 * directly so it can reach the file-static symMemory* helpers. It links no
 * librccl.so; every dependency the source references is satisfied by no-op
 * stubs in DevRuntimeTestsStubs.cc (including host-memory fakes for the HIP
 * VMM driver API).
 *************************************************************************/

#include DEV_RUNTIME_CC_PATH

#include <memory>

#include <gtest/gtest.h>

#include "common/ProcessIsolatedTestRunner.hpp"

extern int rcclTestHipMemAddressFreeCount;

// Build the smallest ncclComm/ncclDevrState that symMemoryObtain will accept:
// a single-rank, single-LSA-team comm with GIN and RMA proxy disabled.
class SymMemoryObtainTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  int lsaRank0 = 0;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();

    comm->nRanks = 1;
    comm->rank = 0;
    comm->cudaDev = 0;
    comm->bootstrap = reinterpret_cast<void*>(0x1);  // opaque; bootstrap* are stubbed
    comm->globalRmaProxySupport = false;
    comm->config.numRmaCtx = 0;

    ncclDevrState* devr = &comm->devrState;
    devr->lsaSelf = 0;
    devr->lsaSize = 1;
    devr->nLsaTeams = 1;
    devr->lsaRankList = &lsaRank0;
    devr->granularity = 4096;
    devr->bigSize = 1 << 20;
    devr->ginEnabled = false;
    devr->lsaFlatBase = nullptr;
    devr->memHead = nullptr;
    devr->teamHead = nullptr;
  }
};

// Regression guard for the window memory leak. Obtaining then destroying the
// memory must run the free path, which unlinks mem from devrState.memHead.
TEST_F(SymMemoryObtainTest, DestroyFreesMemory) {
  hipMemGenericAllocationHandle_t memHandle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
  void* userAddr = reinterpret_cast<void*>(0x100000);
  const size_t size = 4096;

  struct ncclDevrMemory* mem = nullptr;
  ASSERT_EQ(symMemoryObtain(comm, &memHandle, /*numSegments=*/1, userAddr, size, /*winFlags=*/0, &mem),
            ncclSuccess);
  ASSERT_NE(mem, nullptr);
  ASSERT_EQ(comm->devrState.memHead, mem);

  symMemoryDestroy(comm, mem);

  EXPECT_EQ(comm->devrState.memHead, nullptr);
}

TEST_F(SymMemoryObtainTest, DestroyIsIdempotent) {
  hipMemGenericAllocationHandle_t memHandle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
  void* userAddr = reinterpret_cast<void*>(0x100000);
  struct ncclDevrMemory* mem = nullptr;
  ASSERT_EQ(symMemoryObtain(comm, &memHandle, /*numSegments=*/1, userAddr, /*size=*/4096, /*winFlags=*/0, &mem),
            ncclSuccess);

  symMemoryDestroy(comm, mem);
  EXPECT_EQ(comm->devrState.memHead, nullptr);

  // Second destroy must not walk off memHead or double-free.
  symMemoryDestroy(comm, mem);
  EXPECT_EQ(comm->devrState.memHead, nullptr);
}

TEST_F(SymMemoryObtainTest, DestroyIsIdempotentWhileAnotherMemoryIsLinked) {
  hipMemGenericAllocationHandle_t memHandle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
  struct ncclDevrMemory* first = nullptr;
  struct ncclDevrMemory* second = nullptr;
  ASSERT_EQ(symMemoryObtain(comm, &memHandle, /*numSegments=*/1, reinterpret_cast<void*>(0x100000),
                            /*size=*/4096, /*winFlags=*/0, &first),
            ncclSuccess);
  ASSERT_EQ(symMemoryObtain(comm, &memHandle, /*numSegments=*/1, reinterpret_cast<void*>(0x200000),
                            /*size=*/4096, /*winFlags=*/0, &second),
            ncclSuccess);
  ASSERT_EQ(comm->devrState.memHead, second);
  ASSERT_EQ(second->next, first);

  // Destroy the older node twice while a later memory is still on memHead.
  symMemoryDestroy(comm, first);
  EXPECT_EQ(comm->devrState.memHead, second);
  EXPECT_EQ(second->next, nullptr);
  symMemoryDestroy(comm, first);
  EXPECT_EQ(comm->devrState.memHead, second);
  EXPECT_EQ(second->next, nullptr);

  symMemoryDestroy(comm, second);
  EXPECT_EQ(comm->devrState.memHead, nullptr);
}

// ---------------------------------------------------------------------------
// AICOMRCCL-835 finalize-drain coverage.
//
// A Device-API consumer can create a symmetric-window resource (leaving an
// ncclDevrMemory on devrState.memHead) without a matching destroy. ncclDevrFinalize
// must drain those leftovers before freeing the LSA flat VA reservation. This
// test drives the *real* init/finalize lifecycle (ncclDevrInitOnce pairs with
// ncclDevrFinalize) so the state is self-consistent, then asserts the drain
// empties memHead.
class DevrFinalizeDrainTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  std::unique_ptr<ncclPeerInfo> peerStorage;
  ncclComm* comm = nullptr;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();
    comm = commStorage.get();

    comm->nRanks = 1;
    comm->rank = 0;
    comm->cudaDev = 0;
    comm->localRanks = 1;
    comm->bootstrap = reinterpret_cast<void*>(0x1);
    comm->symmetricSupport = 1;  // required to reach the AICOMRCCL-835 drain block
    comm->globalRmaProxySupport = false;
    comm->config.numRmaCtx = 0;

    // ncclDevrInitOnce (with WIN_STRIDE unset) sizes bigSize from peerInfo.
    peerStorage = std::make_unique<ncclPeerInfo>();
    peerStorage->totalGlobalMem = 1 << 20;
    comm->peerInfo = peerStorage.get();
  }
};

TEST_F(DevrFinalizeDrainTest, FinalizeDrainsLeftoverMemory) {
  ASSERT_EQ(ncclDevrInitOnce(comm), ncclSuccess);

  // Simulate a resource window whose owning devcomm was never destroyed: obtain
  // symmetric memory and leave it linked on memHead.
  hipMemGenericAllocationHandle_t memHandle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
  void* userAddr = reinterpret_cast<void*>(0x100000);
  struct ncclDevrMemory* mem = nullptr;
  ASSERT_EQ(symMemoryObtain(comm, &memHandle, /*numSegments=*/1, userAddr, /*size=*/4096, /*winFlags=*/0, &mem),
            ncclSuccess);
  ASSERT_EQ(comm->devrState.memHead, mem);

  // Finalize must drain the leftover before freeing the flat VA reservation.
  ASSERT_EQ(ncclDevrFinalize(comm), ncclSuccess);

  EXPECT_EQ(comm->devrState.memHead, nullptr);
}

// rcclSkipCuMemFree / rcclSkipLsaFlatAddressFree are memoized per process.
// Isolated children cover skip-on vs skip-off and gfx950 vs gfx1250 auto-detect.
TEST(SkipCuMemFreePolicy, IsolatedArchAndEnvBranches) {
  using RcclUnitTesting::ProcessIsolatedTestRunner;

  // Return false on precondition failure so ASSERT_* does not return from this
  // helper and skip the caller's AddressFree-count check.
  auto drainAndCountAddressFree = []() -> bool {
    auto commStorage = std::make_unique<ncclComm>();
    auto peerStorage = std::make_unique<ncclPeerInfo>();
    ncclComm* comm = commStorage.get();
    comm->nRanks = 1;
    comm->rank = 0;
    comm->cudaDev = 0;
    comm->localRanks = 1;
    comm->bootstrap = reinterpret_cast<void*>(0x1);
    comm->symmetricSupport = 1;
    comm->globalRmaProxySupport = false;
    comm->config.numRmaCtx = 0;
    peerStorage->totalGlobalMem = 1 << 20;
    comm->peerInfo = peerStorage.get();

    if (ncclDevrInitOnce(comm) != ncclSuccess) return false;
    hipMemGenericAllocationHandle_t memHandle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
    struct ncclDevrMemory* mem = nullptr;
    if (symMemoryObtain(comm, &memHandle, 1, reinterpret_cast<void*>(0x100000), 4096, 0, &mem) !=
        ncclSuccess) {
      return false;
    }
    rcclTestHipMemAddressFreeCount = 0;
    if (ncclDevrFinalize(comm) != ncclSuccess) return false;
    EXPECT_EQ(comm->devrState.memHead, nullptr);
    return true;
  };

  RUN_ISOLATED_TESTS(
      ProcessIsolatedTestRunner::TestConfig(
          "ForceOff_FreesLsaFlat",
          [&]() {
            EXPECT_FALSE(rcclSkipCuMemFree());
            EXPECT_FALSE(rcclSkipLsaFlatAddressFree());
            ASSERT_TRUE(drainAndCountAddressFree());
            EXPECT_EQ(rcclTestHipMemAddressFreeCount, 1);
          })
          .setVariable("NCCL_CUMEM_SKIP_FREE", "0")
          .setVariable("RCCL_TEST_GCN_ARCH", "gfx1250"),
      ProcessIsolatedTestRunner::TestConfig(
          "ForceOn_SkipsLsaFlat",
          [&]() {
            EXPECT_TRUE(rcclSkipCuMemFree());
            EXPECT_TRUE(rcclSkipLsaFlatAddressFree());
            ASSERT_TRUE(drainAndCountAddressFree());
            EXPECT_EQ(rcclTestHipMemAddressFreeCount, 0);
          })
          .setVariable("NCCL_CUMEM_SKIP_FREE", "1")
          .setVariable("RCCL_TEST_GCN_ARCH", "gfx900"),
      ProcessIsolatedTestRunner::TestConfig(
          "Gfx950_PeerSkipKeepsLsaFlatFree",
          [&]() {
            EXPECT_TRUE(rcclSkipCuMemFree());
            EXPECT_FALSE(rcclSkipLsaFlatAddressFree());
            ASSERT_TRUE(drainAndCountAddressFree());
            EXPECT_EQ(rcclTestHipMemAddressFreeCount, 1);
          })
          .setVariable("RCCL_TEST_GCN_ARCH", "gfx950")
          .clearVariable("NCCL_CUMEM_SKIP_FREE"),
      ProcessIsolatedTestRunner::TestConfig(
          "Gfx1250_SkipsLsaFlat",
          [&]() {
            EXPECT_TRUE(rcclSkipCuMemFree());
            EXPECT_TRUE(rcclSkipLsaFlatAddressFree());
            ASSERT_TRUE(drainAndCountAddressFree());
            EXPECT_EQ(rcclTestHipMemAddressFreeCount, 0);
          })
          .setVariable("RCCL_TEST_GCN_ARCH", "gfx1250")
          .clearVariable("NCCL_CUMEM_SKIP_FREE"),
      ProcessIsolatedTestRunner::TestConfig(
          "Gfx900_SkipFreeOff",
          [&]() {
            EXPECT_FALSE(rcclSkipCuMemFree());
            EXPECT_FALSE(rcclSkipLsaFlatAddressFree());
            ASSERT_TRUE(drainAndCountAddressFree());
            EXPECT_EQ(rcclTestHipMemAddressFreeCount, 1);
          })
          .setVariable("RCCL_TEST_GCN_ARCH", "gfx900")
          .clearVariable("NCCL_CUMEM_SKIP_FREE"));
}
