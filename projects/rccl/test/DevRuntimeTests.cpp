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

#include <gtest/gtest.h>

#include <memory>

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

// ---------------------------------------------------------------------------
// Multimem team acquisition on a build without multicast object creation.
//
// The multicast-creation body in symTeamObtain sits behind
// `#if CUDART_VERSION >= 12010`. CUDART_VERSION is never defined in an RCCL
// build, so the `#else` arm is the live path and no team ever receives an
// mcBasePtr. That arm must report a failure: returning ncclSuccess there also
// leaves the caller's out-team pointer unwritten, and every caller of
// symTeamObtain dereferences it immediately.
// ---------------------------------------------------------------------------
class SymTeamMultimemTest : public ::testing::Test {
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
    comm->nvlsSupport = 1;  // required to reach the multicast-creation branch

    ncclDevrState* devr = &comm->devrState;
    devr->lsaSelf = 0;
    devr->lsaSize = 1;
    devr->nLsaTeams = 1;
    devr->lsaRankList = &lsaRank0;
    devr->granularity = 4096;
    devr->bigSize = 1 << 20;
    devr->ginEnabled = false;
    devr->memHead = nullptr;
    devr->teamHead = nullptr;
  }

  static struct ncclTeam singleRankTeam() {
    struct ncclTeam team = {};
    team.nRanks = 1;
    team.rank = 0;
    team.stride = 1;
    return team;
  }
};

// symTeamObtain must not claim success on a path that never writes *outTeam.
TEST_F(SymTeamMultimemTest, ObtainReportsFailureWhenMulticastUnavailable) {
  struct ncclDevrTeam* poison = reinterpret_cast<struct ncclDevrTeam*>(0xDEADBEEF);
  struct ncclDevrTeam* tm = poison;

  const ncclResult_t result = symTeamObtain(comm, singleRankTeam(), /*multimem=*/true, &tm);

  EXPECT_NE(result, ncclSuccess)
      << "multimem team acquisition must fail when multicast object creation is not compiled in";
  EXPECT_EQ(tm, poison) << "a failing symTeamObtain must not be reported as having produced a team";
  EXPECT_EQ(comm->devrState.teamHead, nullptr)
      << "the failed team must not be linked into devrState";
}

// The multimem pointer query must surface that failure instead of dereferencing
// an out-team the callee never wrote.
TEST_F(SymTeamMultimemTest, GetLsaTeamPtrMCFailsWhenMulticastUnavailable) {
  struct ncclDevrWindow winHost = {};
  winHost.bigOffset = 0;

  void* poison = reinterpret_cast<void*>(0xFEEDFACE);
  void* outPtr = poison;

  const ncclResult_t result =
      ncclDevrGetLsaTeamPtrMC(comm, &winHost, /*offset=*/0, singleRankTeam(), &outPtr);

  EXPECT_NE(result, ncclSuccess)
      << "the multimem pointer query must fail when no multicast team can be created";
  EXPECT_EQ(outPtr, poison) << "no pointer may be published when the team could not be obtained";
}
