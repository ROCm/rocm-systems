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

#include "DevRuntimeTestsStubs.h"

// param.h's NCCL_PARAM caches its value in a function-local static, so a param
// read once is frozen for the process. Replace the generated body with one that
// calls g_loadParam every time, so tests can vary a param between cases. Must
// precede the unit under test, which is where the bodies are emitted.
#include "param.h"
#undef NCCL_PARAM
#define NCCL_PARAM(name, env, deftVal) \
  int64_t ncclParam##name() { return g_loadParam((env), (deftVal)); }

#include DEV_RUNTIME_CC_PATH

#include <gtest/gtest.h>

#include "host/ScopedHook.h"

#include <cstdlib>
#include <fcntl.h>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
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
// ncclDevrInitOnce sizes the symmetric VA space and builds the LSA rank list.
// After the shared prologue it splits on comm->symmetricSupport: the symmetric
// path queries the allocation granularity and sizes bigSize from WIN_STRIDE or
// the largest peer's memory, the proxy-only path rebuilds the rank list from
// the node-local team instead.
//
// bigSize doubles as the "already initialised" marker, so it must stay 0 for
// the body to run at all.

class DevrInitOnceTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  std::vector<ncclPeerInfo> peers;
  std::vector<int> rankToNode;
  std::vector<int> localRankToRank;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->nRanks = 2;
    comm->rank = 0;
    comm->cudaDev = 0;
    comm->localRanks = 2;
    comm->localRank = 0;

    rankToNode.assign({0, 0});
    comm->rankToNode = rankToNode.data();

    localRankToRank.assign({0, 1});
    comm->localRankToRank = localRankToRank.data();

    peers.assign(comm->nRanks, ncclPeerInfo{});
    peers[0].totalGlobalMem = 1u << 20;
    peers[1].totalGlobalMem = 4u << 20;
    comm->peerInfo = peers.data();
  }

  void TearDown() override {
    free(comm->devrState.lsaRankList);
    ResetDevRuntimeFakes();
  }
};

// Branch: bigSize != 0 means a previous call already ran, so this one is a
// no-op. lsaSize stays as set rather than being recomputed from the topology.
TEST_F(DevrInitOnceTest, AlreadyInitialised_ReturnsWithoutTouchingState) {
  comm->devrState.bigSize = 1 << 20;
  comm->devrState.lsaSize = 99;

  EXPECT_EQ(ncclDevrInitOnce(comm), ncclSuccess);
  EXPECT_EQ(comm->devrState.lsaSize, 99);
  EXPECT_EQ(comm->devrState.lsaRankList, nullptr);
}

// Branch: symmetric path with WIN_STRIDE unset (-1). bigSize comes from the
// largest peer's memory, rounded up to a 4GB multiple.
TEST_F(DevrInitOnceTest, SymmetricDefaultStride_SizesFromLargestPeer) {
  comm->symmetricSupport = 1;

  ASSERT_EQ(ncclDevrInitOnce(comm), ncclSuccess);
  EXPECT_EQ(comm->devrState.granularity, 4096u);
  EXPECT_EQ(comm->devrState.bigSize, size_t(1) << 32);  // 4MB aligned up to 4GB
  EXPECT_EQ(comm->devrState.lsaSize, 2);
  EXPECT_EQ(comm->devrState.nLsaTeams, 1);
}

// Branch: WIN_STRIDE set positive, so -bigSize > 1 and the peer-scan is skipped
// -- the configured stride is used instead of the largest peer's memory.
TEST_F(DevrInitOnceTest, SymmetricExplicitStride_UsesConfiguredValue) {
  comm->symmetricSupport = 1;
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t deftVal) -> int64_t {
    return std::string(env) == "WIN_STRIDE" ? (int64_t(1) << 33) : deftVal;
  });

  ASSERT_EQ(ncclDevrInitOnce(comm), ncclSuccess);
  EXPECT_EQ(comm->devrState.bigSize, size_t(1) << 33);  // already 4GB-aligned
}

// Branch: the granularity query fails, so CUCHECKGOTO unwinds via
// fail_lsaRankList and the error propagates.
TEST_F(DevrInitOnceTest, GranularityQueryFails_ReturnsError) {
  comm->symmetricSupport = 1;
  ScopedHook granularity(g_hipMemGetAllocationGranularity,
                         [](size_t*, const hipMemAllocationProp*, hipMemAllocationGranularity_flags) {
                           return hipErrorInvalidValue;
                         });

  EXPECT_NE(ncclDevrInitOnce(comm), ncclSuccess);
  EXPECT_EQ(granularity.calls, 1);
  comm->devrState.lsaRankList = nullptr;  // freed on the failure path already
}

// Branch: proxy-only. bigSize is just an initialised marker, and the LSA team
// is rebuilt from the node-local ranks rather than the gcd-derived team.
TEST_F(DevrInitOnceTest, ProxyOnly_RebuildsTeamFromLocalRanks) {
  comm->symmetricSupport = 0;
  comm->localRanks = 2;
  comm->localRank = 1;

  ASSERT_EQ(ncclDevrInitOnce(comm), ncclSuccess);
  EXPECT_EQ(comm->devrState.bigSize, 1u);
  EXPECT_EQ(comm->devrState.lsaSize, comm->localRanks);
  EXPECT_EQ(comm->devrState.lsaSelf, comm->localRank);
  ASSERT_NE(comm->devrState.lsaRankList, nullptr);
  EXPECT_EQ(comm->devrState.lsaRankList[0], 0);
  EXPECT_EQ(comm->devrState.lsaRankList[1], 1);
}


// ---------------------------------------------------------------------------
// ncclDevrFinalize tears down everything ncclDevrInitOnce built: the reg-task
// queue, any windows still registered, the team and window tables, leftover
// memory records, and finally the flat VA reservation.
//
// These tests drive the real init/finalize lifecycle so the state stays
// self-consistent, rather than hand-building a half-initialised devrState.
class DevrFinalizeTest : public ::testing::Test {
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

    localRankToRank.assign({0});  // proxy-only init rebuilds the team from this
    comm->localRankToRank = localRankToRank.data();
  }

  std::vector<int> localRankToRank;
};

// Branch: bigSize == 0 means init never ran, so there is nothing to tear down.
TEST_F(DevrFinalizeTest, NotInitialised_ReturnsImmediately) {
  ASSERT_EQ(comm->devrState.bigSize, 0u);
  EXPECT_EQ(ncclDevrFinalize(comm), ncclSuccess);
}

// The clean lifecycle: init then finalize with nothing left registered. Pairs
// with the leftover case below, which is the same path with work to drain.
TEST_F(DevrFinalizeTest, NoLeftovers_CompletesCleanly) {
  ASSERT_EQ(ncclDevrInitOnce(comm), ncclSuccess);
  ASSERT_EQ(comm->devrState.memHead, nullptr);

  EXPECT_EQ(ncclDevrFinalize(comm), ncclSuccess);
}

// Branch: the reg-task drain loop. Tasks are malloc'd because finalize frees
// each one it dequeues.
TEST_F(DevrFinalizeTest, DrainsRegTaskQueue) {
  ASSERT_EQ(ncclDevrInitOnce(comm), ncclSuccess);
  for (int i = 0; i < 2; i++) {
    auto* task = static_cast<ncclDevrRegTask*>(calloc(1, sizeof(ncclDevrRegTask)));
    ASSERT_NE(task, nullptr);
    ncclIntruQueueEnqueue(&comm->devrState.regTaskQueue, task);
  }
  ASSERT_FALSE(ncclIntruQueueEmpty(&comm->devrState.regTaskQueue));

  EXPECT_EQ(ncclDevrFinalize(comm), ncclSuccess);
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->devrState.regTaskQueue));
}

// Branch: proxy-only teardown. Non-sym windows are drained through
// windowDeregisterNonSym rather than symWindowDestroy, and the whole symmetric
// block -- teams, window table, memory drain, VA free -- is skipped.
TEST_F(DevrFinalizeTest, ProxyOnly_SkipsSymmetricTeardown) {
  comm->symmetricSupport = 0;
  ASSERT_EQ(ncclDevrInitOnce(comm), ncclSuccess);
  ASSERT_EQ(comm->devrState.bigSize, 1u);

  EXPECT_EQ(ncclDevrFinalize(comm), ncclSuccess);
}

// Branch: every stream call fails. All are wrapped in CUDACHECKIGNORE or
// CUDASUCCESS, so teardown must still complete and still report success --
// finalize runs on the abort path, where giving up would leak.
TEST_F(DevrFinalizeTest, StreamCallsFail_StillCompletes) {
  ASSERT_EQ(ncclDevrInitOnce(comm), ncclSuccess);
  ScopedHook create(g_hipStreamCreateWithFlags,
                    [](hipStream_t*, unsigned int) { return hipErrorInvalidValue; });
  ScopedHook sync(g_hipStreamSynchronize, [](hipStream_t) { return hipErrorInvalidValue; });
  ScopedHook destroy(g_hipStreamDestroy, [](hipStream_t) { return hipErrorInvalidValue; });
  ScopedHook capture(g_hipThreadExchangeStreamCaptureMode,
                     [](hipStreamCaptureMode*) { return hipErrorInvalidValue; });

  EXPECT_EQ(ncclDevrFinalize(comm), ncclSuccess);
  EXPECT_GT(create.calls, 0);
  EXPECT_EQ(capture.calls, 1);
}

// Branch: stream creation succeeds but synchronize and destroy fail, so the
// CUDASUCCESS guards are entered and their inner ignores taken.
TEST_F(DevrFinalizeTest, StreamTeardownFails_StillCompletes) {
  ASSERT_EQ(ncclDevrInitOnce(comm), ncclSuccess);
  ScopedHook sync(g_hipStreamSynchronize, [](hipStream_t) { return hipErrorInvalidValue; });
  ScopedHook destroy(g_hipStreamDestroy, [](hipStream_t) { return hipErrorInvalidValue; });

  EXPECT_EQ(ncclDevrFinalize(comm), ncclSuccess);
  EXPECT_GT(sync.calls, 0);
  EXPECT_GT(destroy.calls, 0);
}

// AICOMRCCL-835: a Device-API consumer can create a symmetric-window resource
// (leaving an ncclDevrMemory on memHead) without a matching destroy. Finalize
// must drain those before freeing the flat VA reservation, or every per-rank
// cuMemMap slice is still live when cuMemAddressFree runs.
TEST_F(DevrFinalizeTest, DrainsLeftoverMemory) {
  ASSERT_EQ(ncclDevrInitOnce(comm), ncclSuccess);

  hipMemGenericAllocationHandle_t memHandle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
  void* userAddr = reinterpret_cast<void*>(0x100000);
  struct ncclDevrMemory* mem = nullptr;
  ASSERT_EQ(symMemoryObtain(comm, &memHandle, /*numSegments=*/1, userAddr, /*size=*/4096, /*winFlags=*/0, &mem),
            ncclSuccess);
  ASSERT_EQ(comm->devrState.memHead, mem);

  ASSERT_EQ(ncclDevrFinalize(comm), ncclSuccess);

  EXPECT_EQ(comm->devrState.memHead, nullptr);
}


// ---------------------------------------------------------------------------
// symMemorySetAccessForVASegment fills a device access descriptor and hands it
// to cuMemSetAccess. The only branch is the CUCHECK on that call.

class SymMemorySetAccessTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  symLsaMessage msg{};

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->cudaDev = 3;
    msg.segmentSize = 4096;
  }
  void TearDown() override { ResetDevRuntimeFakes(); }
};

// Branch: cuMemSetAccess succeeds. Also pins the descriptor the caller builds,
// which is otherwise invisible from the return value alone.
TEST_F(SymMemorySetAccessTest, Succeeds_RequestsReadWriteForOwnDevice) {
  hipMemAccessDesc seen{};
  size_t seenSize = 0;
  ScopedHook setAccess(g_hipMemSetAccess,
                       [&](void*, size_t size, const hipMemAccessDesc* desc, size_t count) {
                         if (desc && count == 1) seen = *desc;
                         seenSize = size;
                         return hipSuccess;
                       });

  EXPECT_EQ(symMemorySetAccessForVASegment(comm, &msg, reinterpret_cast<hipDeviceptr_t>(0x1000)), ncclSuccess);
  EXPECT_EQ(setAccess.calls, 1);
  EXPECT_EQ(seenSize, msg.segmentSize);
  EXPECT_EQ(seen.location.type, hipMemLocationTypeDevice);
  EXPECT_EQ(seen.location.id, comm->cudaDev);
  EXPECT_EQ(seen.flags, hipMemAccessFlagsProtReadWrite);
}

// Branch: cuMemSetAccess fails, so CUCHECK returns instead of ncclSuccess.
TEST_F(SymMemorySetAccessTest, SetAccessFails_ReturnsError) {
  ScopedHook setAccess(g_hipMemSetAccess,
                       [](void*, size_t, const hipMemAccessDesc*, size_t) { return hipErrorInvalidValue; });

  EXPECT_NE(symMemorySetAccessForVASegment(comm, &msg, reinterpret_cast<hipDeviceptr_t>(0x1000)), ncclSuccess);
  EXPECT_EQ(setAccess.calls, 1);
}


// ---------------------------------------------------------------------------
// symMemoryExportSegmentHandle records a segment's location type and size in
// the outgoing message, then publishes the handle one of two ways: POSIX-FD
// handles are passed through as-is, anything else is exported to a shareable
// handle. Both CUCHECKGOTOs jump to the same trailing label.
//
// ncclCuMemHandleType is a stub global fixed to POSIX-FD; the fixture saves and
// restores it so the export branch can be reached without leaking the change.

class SymMemoryExportSegmentHandleTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  symLsaMessage msg{};
  hipMemAllocationHandleType savedHandleType = ncclCuMemHandleType;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
  }
  void TearDown() override {
    ncclCuMemHandleType = savedHandleType;
    ResetDevRuntimeFakes();
  }
};

// Branch: POSIX-FD handles need no export, so the handle is stored directly.
TEST_F(SymMemoryExportSegmentHandleTest, PosixFd_StoresHandleWithoutExporting) {
  ncclCuMemHandleType = hipMemHandleTypePosixFileDescriptor;
  ScopedHook exportHandle(g_hipMemExportToShareableHandle,
                          [](void*, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType,
                             unsigned long long) { return hipSuccess; });
  auto handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x42);

  EXPECT_EQ(symMemoryExportSegmentHandle(comm, &msg, handle, 8192), ncclSuccess);
  EXPECT_EQ(exportHandle.calls, 0);
  EXPECT_EQ(msg.memHandle, handle);
  EXPECT_EQ(msg.segmentSize, 8192u);
  EXPECT_EQ(msg.type, hipMemLocationTypeDevice);  // from the properties fake
}

// Branch: any other handle type takes the shareable-handle export path.
TEST_F(SymMemoryExportSegmentHandleTest, FabricHandle_ExportsShareableHandle) {
  ncclCuMemHandleType = hipMemHandleTypeFabric;
  ScopedHook exportHandle(g_hipMemExportToShareableHandle,
                          [](void*, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType,
                             unsigned long long) { return hipSuccess; });

  EXPECT_EQ(symMemoryExportSegmentHandle(comm, &msg, {}, 4096), ncclSuccess);
  EXPECT_EQ(exportHandle.calls, 1);
  EXPECT_EQ(msg.segmentSize, 4096u);
}

// Branch: the export fails, so the second CUCHECKGOTO propagates the error.
TEST_F(SymMemoryExportSegmentHandleTest, ExportFails_ReturnsError) {
  ncclCuMemHandleType = hipMemHandleTypeFabric;
  ScopedHook exportHandle(g_hipMemExportToShareableHandle,
                          [](void*, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType,
                             unsigned long long) { return hipErrorInvalidValue; });

  EXPECT_NE(symMemoryExportSegmentHandle(comm, &msg, {}, 4096), ncclSuccess);
  EXPECT_EQ(exportHandle.calls, 1);
}

// Branch: the properties fetch fails, so the first CUCHECKGOTO returns before
// the message is touched.
TEST_F(SymMemoryExportSegmentHandleTest, PropertiesFail_ReturnsErrorWithoutWritingMessage) {
  ScopedHook props(g_hipMemGetAllocationPropertiesFromHandle,
                   [](hipMemAllocationProp*, hipMemGenericAllocationHandle_t) { return hipErrorInvalidValue; });

  EXPECT_NE(symMemoryExportSegmentHandle(comm, &msg, {}, 4096), ncclSuccess);
  EXPECT_EQ(props.calls, 1);
  EXPECT_EQ(msg.segmentSize, 0u);  // untouched
}


// ---------------------------------------------------------------------------
// symMemoryImportAndMapSegmentHandle makes one peer segment addressable. It
// picks a handle three ways -- reuse the local one, import an fd fetched from
// the proxy, or import a fabric handle -- then maps it, grants access, and
// releases the imported handle. Everything after the pick is shared, so the
// reuseLocal flag decides both the first branch and the last.

class SymImportAndMapSegmentTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  symLsaMessage msg{};
  std::vector<int> lsaRankList;
  hipMemAllocationHandleType savedHandleType = ncclCuMemHandleType;

  // reinterpret_cast is not a constant expression, so these are plain members.
  const hipDeviceptr_t kAddr = reinterpret_cast<hipDeviceptr_t>(0x100000);
  const hipMemGenericAllocationHandle_t kLocal = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x77);

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    lsaRankList.assign({0, 1});
    comm->devrState.lsaRankList = lsaRankList.data();
    msg.segmentSize = 4096;
  }
  void TearDown() override {
    comm->devrState.lsaRankList = nullptr;  // borrowed from lsaRankList, not malloc'd
    ncclCuMemHandleType = savedHandleType;
    ResetDevRuntimeFakes();
  }
};

// Branch: reuseLocal skips both the import and the matching release -- the
// handle is the caller's, so releasing it would drop a reference it still owns.
TEST_F(SymImportAndMapSegmentTest, ReuseLocal_MapsWithoutImportingOrReleasing) {
  ScopedHook import(g_hipMemImportFromShareableHandle,
                    [](hipMemGenericAllocationHandle_t*, void*, hipMemAllocationHandleType) { return hipSuccess; });
  ScopedHook release(g_hipMemRelease, [](hipMemGenericAllocationHandle_t) { return hipSuccess; });
  hipMemGenericAllocationHandle_t mapped{};
  ScopedHook map(g_hipMemMap,
                 [&](void*, size_t, size_t, hipMemGenericAllocationHandle_t h, unsigned long long) {
                   mapped = h;
                   return hipSuccess;
                 });

  EXPECT_EQ(symMemoryImportAndMapSegmentHandle(comm, 1, kAddr, &msg, kLocal, /*reuseLocal=*/true), ncclSuccess);
  EXPECT_EQ(import.calls, 0);
  EXPECT_EQ(release.calls, 0);
  EXPECT_EQ(map.calls, 1);
  EXPECT_EQ(mapped, kLocal);
}

// Branch: POSIX-FD import. The fd comes from the proxy and is closed after the
// import, so the descriptor must be real for the SYSCHECK on close() to pass.
TEST_F(SymImportAndMapSegmentTest, PosixFd_ImportsThenReleases) {
  ncclCuMemHandleType = hipMemHandleTypePosixFileDescriptor;
  ScopedHook proxy(g_proxyClientGetFdBlocking, [](ncclComm*, int, void*, int* fd) {
    *fd = open("/dev/null", O_RDONLY);
    return *fd < 0 ? ncclSystemError : ncclSuccess;
  });
  ScopedHook release(g_hipMemRelease, [](hipMemGenericAllocationHandle_t) { return hipSuccess; });

  EXPECT_EQ(symMemoryImportAndMapSegmentHandle(comm, 1, kAddr, &msg, {}, /*reuseLocal=*/false), ncclSuccess);
  EXPECT_EQ(proxy.calls, 1);
  EXPECT_EQ(release.calls, 1);
}

// Branch: any other handle type imports the fabric handle directly, with no
// proxy round trip.
TEST_F(SymImportAndMapSegmentTest, FabricHandle_ImportsWithoutProxy) {
  ncclCuMemHandleType = hipMemHandleTypeFabric;
  ScopedHook proxy(g_proxyClientGetFdBlocking,
                   [](ncclComm*, int, void*, int*) { return ncclSuccess; });
  ScopedHook import(g_hipMemImportFromShareableHandle,
                    [](hipMemGenericAllocationHandle_t* h, void*, hipMemAllocationHandleType) {
                      if (h) *h = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
                      return hipSuccess;
                    });

  EXPECT_EQ(symMemoryImportAndMapSegmentHandle(comm, 1, kAddr, &msg, {}, /*reuseLocal=*/false), ncclSuccess);
  EXPECT_EQ(proxy.calls, 0);
  EXPECT_EQ(import.calls, 1);
}

// Branch: the proxy cannot supply an fd, so the import never runs.
TEST_F(SymImportAndMapSegmentTest, ProxyFdFails_ReturnsErrorWithoutImporting) {
  ncclCuMemHandleType = hipMemHandleTypePosixFileDescriptor;
  ScopedHook proxy(g_proxyClientGetFdBlocking,
                   [](ncclComm*, int, void*, int*) { return ncclSystemError; });
  ScopedHook import(g_hipMemImportFromShareableHandle,
                    [](hipMemGenericAllocationHandle_t*, void*, hipMemAllocationHandleType) { return hipSuccess; });

  EXPECT_NE(symMemoryImportAndMapSegmentHandle(comm, 1, kAddr, &msg, {}, /*reuseLocal=*/false), ncclSuccess);
  EXPECT_EQ(import.calls, 0);
}

// Branch: the import itself fails.
TEST_F(SymImportAndMapSegmentTest, ImportFails_ReturnsErrorWithoutMapping) {
  ncclCuMemHandleType = hipMemHandleTypeFabric;
  ScopedHook import(g_hipMemImportFromShareableHandle,
                    [](hipMemGenericAllocationHandle_t*, void*, hipMemAllocationHandleType) {
                      return hipErrorInvalidValue;
                    });
  ScopedHook map(g_hipMemMap,
                 [](void*, size_t, size_t, hipMemGenericAllocationHandle_t, unsigned long long) { return hipSuccess; });

  EXPECT_NE(symMemoryImportAndMapSegmentHandle(comm, 1, kAddr, &msg, {}, /*reuseLocal=*/false), ncclSuccess);
  EXPECT_EQ(map.calls, 0);
}

// Branch: import failure on the POSIX-FD path, which is a separate CUCHECKGOTO
// from the fabric one above.
TEST_F(SymImportAndMapSegmentTest, PosixFdImportFails_ReturnsError) {
  ncclCuMemHandleType = hipMemHandleTypePosixFileDescriptor;
  ScopedHook import(g_hipMemImportFromShareableHandle,
                    [](hipMemGenericAllocationHandle_t*, void*, hipMemAllocationHandleType) {
                      return hipErrorInvalidValue;
                    });

  EXPECT_NE(symMemoryImportAndMapSegmentHandle(comm, 1, kAddr, &msg, {}, /*reuseLocal=*/false), ncclSuccess);
  EXPECT_EQ(import.calls, 1);
}

// Branch: the SYSCHECK on close(). A stale descriptor from the proxy fails with
// EBADF, which the import path treats as fatal rather than ignoring.
TEST_F(SymImportAndMapSegmentTest, CloseFdFails_ReturnsError) {
  ncclCuMemHandleType = hipMemHandleTypePosixFileDescriptor;
  ScopedHook proxy(g_proxyClientGetFdBlocking, [](ncclComm*, int, void*, int* fd) {
    if (fd) *fd = 999999;  // never a live descriptor in this process
    return ncclSuccess;
  });

  EXPECT_NE(symMemoryImportAndMapSegmentHandle(comm, 1, kAddr, &msg, {}, /*reuseLocal=*/false), ncclSuccess);
  EXPECT_EQ(proxy.calls, 1);
}

// Branch: the mapping fails, so access is never granted.
TEST_F(SymImportAndMapSegmentTest, MapFails_ReturnsErrorWithoutSettingAccess) {
  ScopedHook map(g_hipMemMap, [](void*, size_t, size_t, hipMemGenericAllocationHandle_t, unsigned long long) {
    return hipErrorInvalidValue;
  });
  ScopedHook setAccess(g_hipMemSetAccess,
                       [](void*, size_t, const hipMemAccessDesc*, size_t) { return hipSuccess; });

  EXPECT_NE(symMemoryImportAndMapSegmentHandle(comm, 1, kAddr, &msg, kLocal, /*reuseLocal=*/true), ncclSuccess);
  EXPECT_EQ(setAccess.calls, 0);
}

// Branch: granting access fails, propagated through the NCCLCHECKGOTO.
TEST_F(SymImportAndMapSegmentTest, SetAccessFails_ReturnsError) {
  ScopedHook setAccess(g_hipMemSetAccess,
                       [](void*, size_t, const hipMemAccessDesc*, size_t) { return hipErrorInvalidValue; });

  EXPECT_NE(symMemoryImportAndMapSegmentHandle(comm, 1, kAddr, &msg, kLocal, /*reuseLocal=*/true), ncclSuccess);
  EXPECT_EQ(setAccess.calls, 1);
}

// Branch: the release of an imported handle fails.
TEST_F(SymImportAndMapSegmentTest, ReleaseFails_ReturnsError) {
  ncclCuMemHandleType = hipMemHandleTypeFabric;
  ScopedHook release(g_hipMemRelease, [](hipMemGenericAllocationHandle_t) { return hipErrorInvalidValue; });

  EXPECT_NE(symMemoryImportAndMapSegmentHandle(comm, 1, kAddr, &msg, {}, /*reuseLocal=*/false), ncclSuccess);
  EXPECT_EQ(release.calls, 1);
}


// ---------------------------------------------------------------------------
// symMemoryImportAndMapSegmentsForRank walks one rank's segments, mapping each
// at a running address. Per segment it decides whether to reuse the caller's
// handle: always for the local rank, and for a remote rank only when
// SYM_REUSE_SYSMEM_HANDLES is on and that segment is CPU-backed.

class SymImportAndMapForRankTest : public ::testing::Test {
protected:
  static const int kMaxSegments = 2;

  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  std::vector<int> lsaRankList;
  std::vector<symLsaMessage> messages;
  std::vector<hipMemGenericAllocationHandle_t> memHandles;

  const uintptr_t kBase = 0x100000;
  const size_t kBigSize = 1u << 20;
  const hipMemGenericAllocationHandle_t kCallerHandle =
      reinterpret_cast<hipMemGenericAllocationHandle_t>(0x55);

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();

    ncclDevrState* devr = &comm->devrState;
    devr->lsaSelf = 0;
    devr->bigSize = kBigSize;
    devr->lsaFlatBase = reinterpret_cast<void*>(kBase);
    lsaRankList.assign({0, 1});
    devr->lsaRankList = lsaRankList.data();

    // messages is laid out [rank][segment], stride kMaxSegments.
    messages.assign(2 * kMaxSegments, symLsaMessage{});
    for (auto& m : messages) {
      m.segmentSize = 4096;
      m.type = hipMemLocationTypeDevice;
    }
    memHandles.assign(kMaxSegments, kCallerHandle);
  }

  void TearDown() override {
    comm->devrState.lsaRankList = nullptr;  // borrowed, not malloc'd
    ResetDevRuntimeFakes();
  }

  // Turn SYM_REUSE_SYSMEM_HANDLES on; other params keep their defaults.
  static std::function<int64_t(const char*, int64_t)> ReuseSysmemHandlesOn() {
    return [](const char* env, int64_t deftVal) -> int64_t {
      return std::string(env) == "SYM_REUSE_SYSMEM_HANDLES" ? 1 : deftVal;
    };
  }
};

// Branch: r == lsaSelf, so every segment reuses the caller's handle and nothing
// is imported.
TEST_F(SymImportAndMapForRankTest, LocalRank_ReusesCallerHandles) {
  std::vector<hipMemGenericAllocationHandle_t> mapped;
  ScopedHook map(g_hipMemMap, [&](void*, size_t, size_t, hipMemGenericAllocationHandle_t h, unsigned long long) {
    mapped.push_back(h);
    return hipSuccess;
  });
  ScopedHook import(g_hipMemImportFromShareableHandle,
                    [](hipMemGenericAllocationHandle_t*, void*, hipMemAllocationHandleType) { return hipSuccess; });

  EXPECT_EQ(symMemoryImportAndMapSegmentsForRank(comm, 0, messages.data(), kMaxSegments, 2, memHandles.data(), 0),
            ncclSuccess);
  EXPECT_EQ(import.calls, 0);
  ASSERT_EQ(mapped.size(), 2u);
  EXPECT_EQ(mapped[0], kCallerHandle);
  EXPECT_EQ(mapped[1], kCallerHandle);
}

// Branch: remote rank with the reuse param off, so the segment is imported.
TEST_F(SymImportAndMapForRankTest, RemoteRank_ImportsInsteadOfReusing) {
  ScopedHook import(g_hipMemImportFromShareableHandle,
                    [](hipMemGenericAllocationHandle_t* h, void*, hipMemAllocationHandleType) {
                      if (h) *h = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
                      return hipSuccess;
                    });

  EXPECT_EQ(symMemoryImportAndMapSegmentsForRank(comm, 1, messages.data(), kMaxSegments, 1, memHandles.data(), 0),
            ncclSuccess);
  EXPECT_EQ(import.calls, 1);
}

// Branch: the second clause of reuseLocal -- remote rank, param on, CPU-backed
// segment -- so the caller's handle is reused without an import.
TEST_F(SymImportAndMapForRankTest, RemoteHostSegmentWithReuseParam_ReusesHandles) {
  messages[1 * kMaxSegments].type = hipMemLocationTypeHostNuma;
  ScopedHook loadParam(g_loadParam, ReuseSysmemHandlesOn());
  ScopedHook import(g_hipMemImportFromShareableHandle,
                    [](hipMemGenericAllocationHandle_t*, void*, hipMemAllocationHandleType) { return hipSuccess; });

  EXPECT_EQ(symMemoryImportAndMapSegmentsForRank(comm, 1, messages.data(), kMaxSegments, 1, memHandles.data(), 0),
            ncclSuccess);
  EXPECT_EQ(import.calls, 0);
}

// Branch: param on but the segment is device-backed, so reuse does not apply.
TEST_F(SymImportAndMapForRankTest, RemoteDeviceSegmentWithReuseParam_StillImports) {
  ScopedHook loadParam(g_loadParam, ReuseSysmemHandlesOn());
  ScopedHook import(g_hipMemImportFromShareableHandle,
                    [](hipMemGenericAllocationHandle_t* h, void*, hipMemAllocationHandleType) {
                      if (h) *h = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
                      return hipSuccess;
                    });

  EXPECT_EQ(symMemoryImportAndMapSegmentsForRank(comm, 1, messages.data(), kMaxSegments, 1, memHandles.data(), 0),
            ncclSuccess);
  EXPECT_EQ(import.calls, 1);
}

// The running address: each segment maps directly after the previous one,
// starting at lsaFlatBase + r * bigSize + bigOffset.
TEST_F(SymImportAndMapForRankTest, MultipleSegments_AdvanceAddressBySegmentSize) {
  messages[0].segmentSize = 4096;
  messages[1].segmentSize = 8192;
  std::vector<uintptr_t> addrs;
  ScopedHook map(g_hipMemMap, [&](void* p, size_t, size_t, hipMemGenericAllocationHandle_t, unsigned long long) {
    addrs.push_back(reinterpret_cast<uintptr_t>(p));
    return hipSuccess;
  });

  const size_t bigOffset = 512;
  EXPECT_EQ(
      symMemoryImportAndMapSegmentsForRank(comm, 0, messages.data(), kMaxSegments, 2, memHandles.data(), bigOffset),
      ncclSuccess);
  ASSERT_EQ(addrs.size(), 2u);
  EXPECT_EQ(addrs[0], kBase + bigOffset);
  EXPECT_EQ(addrs[1], kBase + bigOffset + 4096);
}

// Boundary: no segments means the loop body never runs.
TEST_F(SymImportAndMapForRankTest, ZeroSegments_MapsNothing) {
  ScopedHook map(g_hipMemMap,
                 [](void*, size_t, size_t, hipMemGenericAllocationHandle_t, unsigned long long) { return hipSuccess; });

  EXPECT_EQ(symMemoryImportAndMapSegmentsForRank(comm, 0, messages.data(), kMaxSegments, 0, memHandles.data(), 0),
            ncclSuccess);
  EXPECT_EQ(map.calls, 0);
}

// Branch: a segment fails, so the loop stops there rather than mapping the rest.
TEST_F(SymImportAndMapForRankTest, SegmentFails_StopsWithoutMappingTheRest) {
  ScopedHook map(g_hipMemMap,
                 [](void*, size_t, size_t, hipMemGenericAllocationHandle_t, unsigned long long) {
                   return hipErrorInvalidValue;
                 });

  EXPECT_NE(symMemoryImportAndMapSegmentsForRank(comm, 0, messages.data(), kMaxSegments, 2, memHandles.data(), 0),
            ncclSuccess);
  EXPECT_EQ(map.calls, 1);  // stopped after the first, did not attempt the second
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
