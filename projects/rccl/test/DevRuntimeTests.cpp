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
 *
 * Suites appear in the same order as the functions they cover in
 * dev_runtime.cc.
 *************************************************************************/

#include DEV_RUNTIME_CC_PATH

#include <gtest/gtest.h>

#include <initializer_list>
#include <memory>
#include <vector>

// ---------------------------------------------------------------------------
// ncclSymIsHostSegment: true for host-NUMA, and on AMD from ROCm 7.12 also for
// plain host. The second arm is #if-gated, so the guard is mirrored below to
// keep the suite passing on either toolchain.

// Branch: the unconditional host-NUMA check.
TEST(SymIsHostSegment, HostNuma_ReturnsTrue) {
  EXPECT_TRUE(ncclSymIsHostSegment(hipMemLocationTypeHostNuma));
}

#if defined(__HIP_PLATFORM_AMD__) && ROCM_VERSION >= 71200
// Branch: AMD allocates host segments as plain host, so they count as sysmem.
TEST(SymIsHostSegment, Host_ReturnsTrue) {
  EXPECT_TRUE(ncclSymIsHostSegment(hipMemLocationTypeHost));
}
#else
// Without the AMD arm compiled in, plain host is not a sysmem segment.
TEST(SymIsHostSegment, Host_ReturnsFalse) {
  EXPECT_FALSE(ncclSymIsHostSegment(hipMemLocationTypeHost));
}
#endif

// Falls through every check.
TEST(SymIsHostSegment, Device_ReturnsFalse) {
  EXPECT_FALSE(ncclSymIsHostSegment(hipMemLocationTypeDevice));
}

// Zero value, as a value-initialised symLsaMessage::type would hold.
TEST(SymIsHostSegment, Invalid_ReturnsFalse) {
  EXPECT_FALSE(ncclSymIsHostSegment(hipMemLocationTypeInvalid));
}


// ---------------------------------------------------------------------------
// computeLsaSize returns the LSA team size by one of three paths:
//   cached    bigSize != 0                             -> devrState.lsaSize
//   clique    p2pCrossClique && nvlDomainSize == nRanks -> nRanks
//   node gcd  otherwise, gcd over runs of equal rankToNode entries
//
// File-static, so callable only because this TU #includes dev_runtime.cc. The
// param stub returns defaults, so ncclParamLsaTeamSize() is 0 and gcd(0, n) == n.

class ComputeLsaSizeTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  std::vector<int> rankToNode;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
  }

  // nodes[r] is the node index of rank r; nRanks follows the list length.
  void SetTopology(std::initializer_list<int> nodes) {
    rankToNode.assign(nodes);
    comm->nRanks = static_cast<int>(rankToNode.size());
    comm->rankToNode = rankToNode.data();
  }
};

// Branch: cached early return. lsaSize differs from nRanks so a recomputing
// path cannot pass; rankToNode is left null so a regression faults.
TEST_F(ComputeLsaSizeTest, Cached_ReturnsStoredLsaSize) {
  comm->nRanks = 8;
  comm->devrState.bigSize = 1 << 20;
  comm->devrState.lsaSize = 4;
  EXPECT_EQ(computeLsaSize(comm), 4);
}

// Branch: both && arms hold. The topology would give 2, so this pins the override.
TEST_F(ComputeLsaSizeTest, CrossCliqueFullNvlDomain_ReturnsNRanks) {
  SetTopology({0, 0, 1, 1});
  comm->p2pCrossClique = true;
  comm->nvlDomainSize = comm->nRanks;
  EXPECT_EQ(computeLsaSize(comm), 4);
}

// Branch: second && arm fails, falling through to the gcd path.
TEST_F(ComputeLsaSizeTest, CrossCliquePartialNvlDomain_FallsBackToNodeGcd) {
  SetTopology({0, 0, 1, 1});
  comm->p2pCrossClique = true;
  comm->nvlDomainSize = 2;
  EXPECT_EQ(computeLsaSize(comm), 2);
}

// Branch: one node, so the run reaches nRanks and gcd(0, 4) == 4.
TEST_F(ComputeLsaSizeTest, SingleNode_ReturnsNRanks) {
  SetTopology({0, 0, 0, 0});
  EXPECT_EQ(computeLsaSize(comm), 4);
}

// Branch: node index changes mid-loop; two runs of 2 give gcd(2, 2) == 2.
TEST_F(ComputeLsaSizeTest, TwoEqualNodes_ReturnsRunLength) {
  SetTopology({0, 0, 1, 1});
  EXPECT_EQ(computeLsaSize(comm), 2);
}

// Uneven runs: gcd(2, 1) == 1.
TEST_F(ComputeLsaSizeTest, UnevenNodes_ReturnsGcdOfRuns) {
  SetTopology({0, 0, 1});
  EXPECT_EQ(computeLsaSize(comm), 1);
}

// Boundary: nRanks == 1 skips the loop; gcd(0, 1) == 1.
TEST_F(ComputeLsaSizeTest, SingleRank_ReturnsOne) {
  SetTopology({0});
  EXPECT_EQ(computeLsaSize(comm), 1);
}


// ---------------------------------------------------------------------------
// ncclDevrIsOneLsaTeam: computeLsaSize(comm) == comm->nRanks. No conditional of
// its own, and computeLsaSize is covered above, so these pin only the compare.
// The fixture uses the cached path to hand it a chosen lsaSize.

class DevrIsOneLsaTeamTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->nRanks = 8;
    comm->devrState.bigSize = 1 << 20;  // take computeLsaSize's cached path
  }
};

// lsaSize == nRanks: a single team spans the comm.
TEST_F(DevrIsOneLsaTeamTest, LsaSizeEqualsNRanks_ReturnsTrue) {
  comm->devrState.lsaSize = 8;
  EXPECT_TRUE(ncclDevrIsOneLsaTeam(comm));
}

// lsaSize < nRanks: the comm holds more than one team.
TEST_F(DevrIsOneLsaTeamTest, LsaSizeBelowNRanks_ReturnsFalse) {
  comm->devrState.lsaSize = 4;
  EXPECT_FALSE(ncclDevrIsOneLsaTeam(comm));
}


// ---------------------------------------------------------------------------
// ncclDevrFinalize, AICOMRCCL-835 finalize-drain coverage.
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
// symBindTeamMemory binds memory into a team's multicast handle, guarded by
// `comm->nvlsSupport && tm->mcBasePtr != nullptr`. The body is behind
// `#if CUDART_VERSION >= 12010`, which no HIP build defines, so on this target
// the guard wraps nothing and the function always returns ncclSuccess. That
// leaves the two && arms as the only reachable branches.

// Shared by the bind and unbind suites, which take the same arguments.
class SymTeamMemoryTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  // ncclDevrTeam ends in a flexible array member, so it cannot be held by
  // value here; back it with a zeroed buffer instead.
  std::vector<unsigned char> teamStorage;
  ncclDevrTeam* team = nullptr;
  struct ncclDevrMemory mem{};

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    teamStorage.assign(sizeof(ncclDevrTeam), 0);
    team = reinterpret_cast<ncclDevrTeam*>(teamStorage.data());
  }
};

// Kept distinct so each suite name still names one function.
class SymBindTeamMemoryTest : public SymTeamMemoryTest {};

// Branch: nvlsSupport == 0 short circuits the mcBasePtr check.
TEST_F(SymBindTeamMemoryTest, NvlsUnsupported_ReturnsSuccess) {
  comm->nvlsSupport = 0;
  team->mcBasePtr = reinterpret_cast<void*>(0x1000);
  EXPECT_EQ(symBindTeamMemory(comm, team, &mem), ncclSuccess);
}

// Branch: NVLS is available but the team has no multicast mapping.
TEST_F(SymBindTeamMemoryTest, NoMulticastBase_ReturnsSuccess) {
  comm->nvlsSupport = 1;
  team->mcBasePtr = nullptr;
  EXPECT_EQ(symBindTeamMemory(comm, team, &mem), ncclSuccess);
}

// Branch: both arms hold, entering the guarded block.
TEST_F(SymBindTeamMemoryTest, NvlsWithMulticastBase_ReturnsSuccess) {
  comm->nvlsSupport = 1;
  team->mcBasePtr = reinterpret_cast<void*>(0x1000);
  EXPECT_EQ(symBindTeamMemory(comm, team, &mem), ncclSuccess);
}


// ---------------------------------------------------------------------------
// symUnbindTeamMemory is the counterpart to symBindTeamMemory, with a third
// guard arm: !mem->globalHasSysmemSegment. Its body is behind the same
// CUDART_VERSION check and so is likewise not compiled on HIP.

class SymUnbindTeamMemoryTest : public SymTeamMemoryTest {};

// Branch: nvlsSupport == 0 short circuits the rest.
TEST_F(SymUnbindTeamMemoryTest, NvlsUnsupported_ReturnsSuccess) {
  comm->nvlsSupport = 0;
  team->mcBasePtr = reinterpret_cast<void*>(0x1000);
  EXPECT_EQ(symUnbindTeamMemory(comm, team, &mem), ncclSuccess);
}

// Branch: NVLS is available but the team has no multicast mapping.
TEST_F(SymUnbindTeamMemoryTest, NoMulticastBase_ReturnsSuccess) {
  comm->nvlsSupport = 1;
  team->mcBasePtr = nullptr;
  EXPECT_EQ(symUnbindTeamMemory(comm, team, &mem), ncclSuccess);
}

// Branch: CPU-backed memory is never multicast-bound, so there is nothing to
// unbind -- the arm bind has no equivalent of.
TEST_F(SymUnbindTeamMemoryTest, SysmemSegment_ReturnsSuccess) {
  comm->nvlsSupport = 1;
  team->mcBasePtr = reinterpret_cast<void*>(0x1000);
  mem.globalHasSysmemSegment = true;
  EXPECT_EQ(symUnbindTeamMemory(comm, team, &mem), ncclSuccess);
}

// Branch: all three arms hold, entering the guarded block.
TEST_F(SymUnbindTeamMemoryTest, NvlsDeviceMemory_ReturnsSuccess) {
  comm->nvlsSupport = 1;
  team->mcBasePtr = reinterpret_cast<void*>(0x1000);
  mem.globalHasSysmemSegment = false;
  EXPECT_EQ(symUnbindTeamMemory(comm, team, &mem), ncclSuccess);
}


// ---------------------------------------------------------------------------
// symMemoryObtain / symMemoryDestroy.
//
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
// ncclDevrWindowIsMultiSegment: win && win->memory && maxGlobalNumSegments > 1.
// Each && arm short-circuits, so each needs its own test.

// Branch: win == nullptr.
TEST(DevrWindowPredicates, IsMultiSegment_NullWindow_ReturnsFalse) {
  struct ncclDevrWindow* win{};
  EXPECT_FALSE(ncclDevrWindowIsMultiSegment(win));
}

// Branch: win->memory == nullptr.
TEST(DevrWindowPredicates, IsMultiSegment_NullMemory_ReturnsFalse) {
  struct ncclDevrWindow win{};
  win.memory = nullptr;
  EXPECT_FALSE(ncclDevrWindowIsMultiSegment(&win));
}

// Branch: maxGlobalNumSegments == 1, the boundary of `> 1`.
TEST(DevrWindowPredicates, IsMultiSegment_SingleSegment_ReturnsFalse) {
  struct ncclDevrWindow win{};
  struct ncclDevrMemory memory{};
  memory.maxGlobalNumSegments = 1;
  win.memory = &memory;
  EXPECT_FALSE(ncclDevrWindowIsMultiSegment(&win));
}

// All conditions pass.
TEST(DevrWindowPredicates, IsMultiSegment_MultipleSegments_ReturnsTrue) {
  struct ncclDevrWindow win{};
  struct ncclDevrMemory memory{};
  memory.maxGlobalNumSegments = 2;
  win.memory = &memory;
  EXPECT_TRUE(ncclDevrWindowIsMultiSegment(&win));
}


// ---------------------------------------------------------------------------
// ncclDevrWindowHasSysmemSegment: same shape, ending in globalHasSysmemSegment.

// Branch: win == nullptr.
TEST(DevrWindowPredicates, HasSysmemSegment_NullWindow_ReturnsFalse) {
  struct ncclDevrWindow* win{};
  EXPECT_FALSE(ncclDevrWindowHasSysmemSegment(win));
}

// Branch: win->memory == nullptr.
TEST(DevrWindowPredicates, HasSysmemSegment_NullMemory_ReturnsFalse) {
  struct ncclDevrWindow win{};
  win.memory = nullptr;
  EXPECT_FALSE(ncclDevrWindowHasSysmemSegment(&win));
}

// Branch: no rank has a sysmem segment.
TEST(DevrWindowPredicates, HasSysmemSegment_NoSysmemSegment_ReturnsFalse) {
  struct ncclDevrWindow win{};
  struct ncclDevrMemory memory{};
  memory.globalHasSysmemSegment = false;
  win.memory = &memory;
  EXPECT_FALSE(ncclDevrWindowHasSysmemSegment(&win));
}

// All conditions pass.
TEST(DevrWindowPredicates, HasSysmemSegment_HasSysmemSegment_ReturnsTrue) {
  struct ncclDevrWindow win{};
  struct ncclDevrMemory memory{};
  memory.globalHasSysmemSegment = true;
  win.memory = &memory;
  EXPECT_TRUE(ncclDevrWindowHasSysmemSegment(&win));
}
