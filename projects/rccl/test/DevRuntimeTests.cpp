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

// ncclCalloc's failure arms cannot be reached while it always succeeds, and it
// is a macro rather than a symbol, so route it through a call counter.
// TRAP: the substitution is textual and TU-wide, so the index counts every
// ncclCalloc the *test* reaches, not only the unit under test's.
// TRAP: both statics are TU-local, so a fixture must reset them itself --
// ResetDevRuntimeFakes() cannot see them.
#include "alloc.h"
static int g_callocCallIndex = 0;
static int g_callocFailAt = -1;  // -1 = never fail; otherwise a 0-based call index
template <typename... Args>
static ncclResult_t MicroCalloc(const char* file, int line, const char* fn, Args&&... args) {
  if (g_callocCallIndex++ == g_callocFailAt) return ncclSystemError;
  return ncclCallocDebug(std::forward<Args>(args)..., file, line, fn, true);
}
#undef ncclCalloc
#define ncclCalloc(...) MicroCalloc(__FILE__, __LINE__, __func__, __VA_ARGS__)

#include DEV_RUNTIME_CC_PATH

#include <gtest/gtest.h>

#include "host/ScopedHook.h"

#include <cstdlib>
#include <fcntl.h>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
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
// symMemoryMapLsaTeam publishes this rank's segment handles to its LSA team and
// maps every peer's in return: size the message array from the widest rank,
// export our own segments into it, all-gather, reserve the flat VA on first
// use, map each rank's segments, then barrier so nobody unmaps early.

class SymMemoryMapLsaTeamTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  ncclDevrMemory mem{};
  std::vector<int> lsaRankList;
  std::vector<int> lsaNumSegments;
  std::vector<size_t> segmentSizes;
  std::vector<hipMemGenericAllocationHandle_t> memHandles;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->bootstrap = reinterpret_cast<void*>(0x1);  // opaque; bootstrap* are seams

    ncclDevrState* devr = &comm->devrState;
    devr->lsaSelf = 0;
    devr->lsaSize = 2;
    devr->bigSize = 1u << 20;
    devr->lsaFlatBase = nullptr;  // force the reservation on first use
    lsaRankList.assign({0, 1});
    devr->lsaRankList = lsaRankList.data();

    lsaNumSegments.assign({1, 1});
    segmentSizes.assign({4096});
    memHandles.assign(1, reinterpret_cast<hipMemGenericAllocationHandle_t>(0x55));
    mem.lsaNumSegments = lsaNumSegments.data();
    mem.segmentSizes = segmentSizes.data();
    mem.memHandles = memHandles.data();
    mem.numSegments = 1;
    mem.bigOffset = 0;
  }

  void TearDown() override {
    comm->devrState.lsaRankList = nullptr;  // borrowed, not malloc'd
    g_callocCallIndex = 0;                  // TU-local, not covered by the reset below
    g_callocFailAt = -1;
    ResetDevRuntimeFakes();
  }
};

// Branch: lsaFlatBase is null, so the flat VA range is reserved for the whole
// team -- lsaSize * bigSize, not one rank's worth.
TEST_F(SymMemoryMapLsaTeamTest, FirstUse_ReservesFlatVaForWholeTeam) {
  size_t reserved = 0;
  ScopedHook reserve(g_hipMemAddressReserve,
                     [&](void** ptr, size_t size, size_t, void*, unsigned long long) {
                       reserved = size;
                       *ptr = reinterpret_cast<void*>(0x200000);
                       return hipSuccess;
                     });

  EXPECT_EQ(symMemoryMapLsaTeam(comm, &mem), ncclSuccess);
  EXPECT_EQ(reserve.calls, 1);
  EXPECT_EQ(reserved, comm->devrState.lsaSize * comm->devrState.bigSize);
  EXPECT_EQ(comm->devrState.lsaFlatBase, reinterpret_cast<void*>(0x200000));
}

// Branch: a base is already reserved, so the reservation is skipped and the
// existing one is left alone.
TEST_F(SymMemoryMapLsaTeamTest, AlreadyReserved_SkipsReservation) {
  void* existing = reinterpret_cast<void*>(0x300000);
  comm->devrState.lsaFlatBase = existing;
  ScopedHook reserve(g_hipMemAddressReserve,
                     [](void**, size_t, size_t, void*, unsigned long long) { return hipSuccess; });

  EXPECT_EQ(symMemoryMapLsaTeam(comm, &mem), ncclSuccess);
  EXPECT_EQ(reserve.calls, 0);
  EXPECT_EQ(comm->devrState.lsaFlatBase, existing);
}

// The message array is sized from the widest rank, so a peer with more segments
// than us still has room. Two ranks x 2 segments each = 4 messages.
TEST_F(SymMemoryMapLsaTeamTest, SizesMessagesFromWidestRank) {
  lsaNumSegments.assign({1, 2});  // the peer has more segments than we do
  size_t gatherBytes = 0;
  ScopedHook gather(g_bootstrapIntraNodeAllGather,
                    [&](void*, int*, int, int, void*, int bytes) {
                      gatherBytes = static_cast<size_t>(bytes);
                      return ncclSuccess;
                    });

  EXPECT_EQ(symMemoryMapLsaTeam(comm, &mem), ncclSuccess);
  EXPECT_EQ(gather.calls, 1);
  EXPECT_EQ(gatherBytes, sizeof(symLsaMessage) * 2);  // per-rank stride = maxSegments
}

// Branch: the message allocation fails before anything is exported.
TEST_F(SymMemoryMapLsaTeamTest, MessageAllocFails_ReturnsError) {
  g_callocFailAt = g_callocCallIndex;  // fail the next ncclCalloc
  ScopedHook gather(g_bootstrapIntraNodeAllGather,
                    [](void*, int*, int, int, void*, int) { return ncclSuccess; });

  EXPECT_NE(symMemoryMapLsaTeam(comm, &mem), ncclSuccess);
  EXPECT_EQ(gather.calls, 0);
}

// Branch: exporting our own segment fails, so nothing is gathered.
TEST_F(SymMemoryMapLsaTeamTest, ExportFails_ReturnsErrorWithoutGathering) {
  ScopedHook props(g_hipMemGetAllocationPropertiesFromHandle,
                   [](hipMemAllocationProp*, hipMemGenericAllocationHandle_t) { return hipErrorInvalidValue; });
  ScopedHook gather(g_bootstrapIntraNodeAllGather,
                    [](void*, int*, int, int, void*, int) { return ncclSuccess; });

  EXPECT_NE(symMemoryMapLsaTeam(comm, &mem), ncclSuccess);
  EXPECT_EQ(gather.calls, 0);
}

// Branch: the all-gather fails, so no VA is reserved.
TEST_F(SymMemoryMapLsaTeamTest, AllGatherFails_ReturnsErrorWithoutReserving) {
  ScopedHook gather(g_bootstrapIntraNodeAllGather,
                    [](void*, int*, int, int, void*, int) { return ncclSystemError; });
  ScopedHook reserve(g_hipMemAddressReserve,
                     [](void**, size_t, size_t, void*, unsigned long long) { return hipSuccess; });

  EXPECT_NE(symMemoryMapLsaTeam(comm, &mem), ncclSuccess);
  EXPECT_EQ(reserve.calls, 0);
}

// Branch: the VA reservation fails.
TEST_F(SymMemoryMapLsaTeamTest, ReserveFails_ReturnsError) {
  ScopedHook reserve(g_hipMemAddressReserve,
                     [](void**, size_t, size_t, void*, unsigned long long) { return hipErrorOutOfMemory; });

  EXPECT_NE(symMemoryMapLsaTeam(comm, &mem), ncclSuccess);
  EXPECT_EQ(comm->devrState.lsaFlatBase, nullptr);
}

// Branch: mapping a rank's segments fails, so the closing barrier is skipped --
// a rank that failed to map must not signal that it is ready.
TEST_F(SymMemoryMapLsaTeamTest, MapFails_ReturnsErrorWithoutBarrier) {
  ScopedHook reserve(g_hipMemAddressReserve,
                     [](void** ptr, size_t, size_t, void*, unsigned long long) {
                       *ptr = reinterpret_cast<void*>(0x200000);
                       return hipSuccess;
                     });
  ScopedHook map(g_hipMemMap, [](void*, size_t, size_t, hipMemGenericAllocationHandle_t, unsigned long long) {
    return hipErrorInvalidValue;
  });
  ScopedHook barrier(g_bootstrapIntraNodeBarrier, [](void*, int*, int, int, int) { return ncclSuccess; });

  EXPECT_NE(symMemoryMapLsaTeam(comm, &mem), ncclSuccess);
  EXPECT_EQ(barrier.calls, 0);
}

// Branch: the closing barrier fails.
TEST_F(SymMemoryMapLsaTeamTest, BarrierFails_ReturnsError) {
  ScopedHook reserve(g_hipMemAddressReserve,
                     [](void** ptr, size_t, size_t, void*, unsigned long long) {
                       *ptr = reinterpret_cast<void*>(0x200000);
                       return hipSuccess;
                     });
  ScopedHook barrier(g_bootstrapIntraNodeBarrier, [](void*, int*, int, int, int) { return ncclSystemError; });

  EXPECT_NE(symMemoryMapLsaTeam(comm, &mem), ncclSuccess);
  EXPECT_EQ(barrier.calls, 1);
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
// symTeamObtain looks up a team by (rank, nRanks, stride) and creates one if
// the list has no match, pushing it onto devrState.teamHead.
//
// Its multimem half is behind `#if CUDART_VERSION >= 12010`, which no HIP build
// defines, so only the nvlsSupport check survives there. The supported-but-
// compiled-out case is deliberately untested: it returns ncclSuccess without
// writing *outTeam, and a test would lock that in as expected. See
// ~/rccl-dev-runtime-findings.md.

class SymTeamObtainTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->rank = 4;
    comm->devrState.bigSize = 1u << 20;
  }

  // Teams are malloc'd by the unit under test and linked onto teamHead.
  void TearDown() override {
    ncclDevrTeam* t = comm->devrState.teamHead;
    while (t != nullptr) {
      ncclDevrTeam* next = t->next;
      free(t);
      t = next;
    }
    comm->devrState.teamHead = nullptr;
    ResetDevRuntimeFakes();
  }

  static ncclTeam MakeTeam(int nRanks, int rank, int stride) {
    ncclTeam team{};
    team.nRanks = nRanks;
    team.rank = rank;
    team.stride = stride;
    return team;
  }
};

// Branch: an empty list, so a team is created and linked. worldRankList is the
// team's members in world terms, derived from our own rank and the stride.
TEST_F(SymTeamObtainTest, EmptyList_CreatesAndLinksTeam) {
  ncclTeam team = MakeTeam(/*nRanks=*/3, /*rank=*/1, /*stride=*/2);
  ncclDevrTeam* out = nullptr;

  ASSERT_EQ(symTeamObtain(comm, team, /*multimem=*/false, &out), ncclSuccess);
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(comm->devrState.teamHead, out);
  EXPECT_EQ(out->mcBasePtr, nullptr);
  // comm->rank + (i - team.rank) * stride, for i in 0..2 with rank 4.
  EXPECT_EQ(out->worldRankList[0], 2);
  EXPECT_EQ(out->worldRankList[1], 4);
  EXPECT_EQ(out->worldRankList[2], 6);
}

// Branch: a team with the same shape is already present, so it is returned as
// is and nothing is pushed onto the list.
TEST_F(SymTeamObtainTest, MatchingTeam_ReturnsExistingWithoutCreating) {
  ncclTeam team = MakeTeam(2, 0, 1);
  ncclDevrTeam* first = nullptr;
  ASSERT_EQ(symTeamObtain(comm, team, false, &first), ncclSuccess);

  ncclDevrTeam* second = nullptr;
  ASSERT_EQ(symTeamObtain(comm, team, false, &second), ncclSuccess);
  EXPECT_EQ(second, first);
  EXPECT_EQ(comm->devrState.teamHead, first);
  EXPECT_EQ(first->next, nullptr);  // still one entry
}

// Branch: the list walk. A team differing in any of the three fields is not a
// match, so the walk continues and a new team is pushed in front.
TEST_F(SymTeamObtainTest, NonMatchingTeam_WalksListThenCreates) {
  ncclDevrTeam* existing = nullptr;
  ASSERT_EQ(symTeamObtain(comm, MakeTeam(2, 0, 1), false, &existing), ncclSuccess);

  ncclDevrTeam* created = nullptr;
  ASSERT_EQ(symTeamObtain(comm, MakeTeam(2, 0, 4), false, &created), ncclSuccess);  // different stride
  EXPECT_NE(created, existing);
  EXPECT_EQ(comm->devrState.teamHead, created);
  EXPECT_EQ(created->next, existing);
}

// Branch: multimem is already satisfied on the matched team, so it is returned
// without entering the multicast setup at all.
TEST_F(SymTeamObtainTest, MultimemAlreadyBound_ReturnsExisting) {
  ncclTeam team = MakeTeam(2, 0, 1);
  ncclDevrTeam* first = nullptr;
  ASSERT_EQ(symTeamObtain(comm, team, false, &first), ncclSuccess);
  first->mcBasePtr = reinterpret_cast<void*>(0x1000);  // pretend multicast is bound

  ncclDevrTeam* second = nullptr;
  EXPECT_EQ(symTeamObtain(comm, team, /*multimem=*/true, &second), ncclSuccess);
  EXPECT_EQ(second, first);
}

// Branch: multimem asked for on a system without NVLS is rejected outright.
TEST_F(SymTeamObtainTest, MultimemWithoutNvls_ReturnsInvalidArgument) {
  comm->nvlsSupport = 0;
  ncclDevrTeam* out = nullptr;

  EXPECT_EQ(symTeamObtain(comm, MakeTeam(2, 0, 1), /*multimem=*/true, &out), ncclInvalidArgument);
  EXPECT_EQ(comm->devrState.teamHead, nullptr);  // the new team was freed, not linked
}

// Branch: outTeam is optional -- callers that only want the team created can
// pass null.
TEST_F(SymTeamObtainTest, NullOutTeam_StillCreatesAndLinks) {
  EXPECT_EQ(symTeamObtain(comm, MakeTeam(2, 0, 1), false, nullptr), ncclSuccess);
  EXPECT_NE(comm->devrState.teamHead, nullptr);
}

// Boundary: a single-rank team is just ourselves.
TEST_F(SymTeamObtainTest, SingleRankTeam_ListsSelfOnly) {
  ncclDevrTeam* out = nullptr;
  ASSERT_EQ(symTeamObtain(comm, MakeTeam(1, 0, 1), false, &out), ncclSuccess);
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(out->worldRankList[0], comm->rank);
}


// ---------------------------------------------------------------------------
// symTeamDestroyAll empties devrState.teamHead, tearing down each team's
// multicast mapping first if it has one. It returns void, so the assertions are
// on the resulting state and on which driver calls were made.

class SymTeamDestroyAllTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->devrState.bigSize = 1u << 20;
  }
  void TearDown() override { ResetDevRuntimeFakes(); }

  // Teams are freed by the unit under test, so they must be malloc'd. nRanks is
  // 1 because worldRankList is a flexible array member.
  ncclDevrTeam* PushTeam(void* mcBasePtr) {
    auto* t = static_cast<ncclDevrTeam*>(calloc(1, sizeof(ncclDevrTeam) + sizeof(int)));
    t->team.nRanks = 1;
    t->mcBasePtr = mcBasePtr;
    t->next = comm->devrState.teamHead;
    comm->devrState.teamHead = t;
    return t;
  }
};

// Branch: an empty list, so the loop body never runs.
TEST_F(SymTeamDestroyAllTest, EmptyList_DoesNothing) {
  ScopedHook unmap(g_hipMemUnmap, [](void*, size_t) { return hipSuccess; });

  symTeamDestroyAll(comm);
  EXPECT_EQ(comm->devrState.teamHead, nullptr);
  EXPECT_EQ(unmap.calls, 0);
}

// Branch: teams without a multicast mapping are freed without touching the
// driver.
TEST_F(SymTeamDestroyAllTest, PlainTeams_FreedWithoutDriverCalls) {
  PushTeam(nullptr);
  PushTeam(nullptr);
  ScopedHook unmap(g_hipMemUnmap, [](void*, size_t) { return hipSuccess; });
  ScopedHook addrFree(g_hipMemAddressFree, [](void*, size_t) { return hipSuccess; });
  ScopedHook release(g_hipMemRelease, [](hipMemGenericAllocationHandle_t) { return hipSuccess; });

  symTeamDestroyAll(comm);
  EXPECT_EQ(comm->devrState.teamHead, nullptr);
  EXPECT_EQ(unmap.calls, 0);
  EXPECT_EQ(addrFree.calls, 0);
  EXPECT_EQ(release.calls, 0);
}

// Branch: a team with a multicast mapping is unmapped, its VA freed and its
// handle released -- in that order, each over the team's full bigSize.
TEST_F(SymTeamDestroyAllTest, MulticastTeam_UnmapsFreesAndReleases) {
  void* mcBase = reinterpret_cast<void*>(0x400000);
  PushTeam(mcBase);
  size_t unmapSize = 0, freeSize = 0;
  ScopedHook unmap(g_hipMemUnmap, [&](void* p, size_t size) {
    EXPECT_EQ(p, mcBase);
    unmapSize = size;
    return hipSuccess;
  });
  ScopedHook addrFree(g_hipMemAddressFree, [&](void* p, size_t size) {
    EXPECT_EQ(p, mcBase);
    freeSize = size;
    return hipSuccess;
  });
  ScopedHook release(g_hipMemRelease, [](hipMemGenericAllocationHandle_t) { return hipSuccess; });

  symTeamDestroyAll(comm);
  EXPECT_EQ(comm->devrState.teamHead, nullptr);
  EXPECT_EQ(unmap.calls, 1);
  EXPECT_EQ(addrFree.calls, 1);
  EXPECT_EQ(release.calls, 1);
  EXPECT_EQ(unmapSize, comm->devrState.bigSize);
  EXPECT_EQ(freeSize, comm->devrState.bigSize);
}

// Branch: the per-memory unbind loop. Every live memory is unbound from the
// team before its mapping goes away, but the memories themselves are owned by
// devrState.memHead and must survive.
TEST_F(SymTeamDestroyAllTest, MulticastTeam_UnbindsMemoriesWithoutFreeingThem) {
  ncclDevrMemory second{};
  ncclDevrMemory first{};
  first.next = &second;
  comm->devrState.memHead = &first;
  comm->nvlsSupport = 1;
  PushTeam(reinterpret_cast<void*>(0x400000));
  ScopedHook unmap(g_hipMemUnmap, [](void*, size_t) { return hipSuccess; });

  symTeamDestroyAll(comm);
  EXPECT_EQ(comm->devrState.teamHead, nullptr);
  EXPECT_EQ(comm->devrState.memHead, &first);  // memories are not this function's to free
  EXPECT_EQ(first.next, &second);
}

// A mixed list: only the multicast team reaches the driver, and both are freed.
TEST_F(SymTeamDestroyAllTest, MixedList_TearsDownOnlyMulticastTeams) {
  PushTeam(nullptr);
  PushTeam(reinterpret_cast<void*>(0x400000));
  ScopedHook unmap(g_hipMemUnmap, [](void*, size_t) { return hipSuccess; });

  symTeamDestroyAll(comm);
  EXPECT_EQ(comm->devrState.teamHead, nullptr);
  EXPECT_EQ(unmap.calls, 1);
}

// The driver calls are CUCHECKIGNORE'd, so a failure must not stop the walk --
// this runs during finalize, where leaving teams linked would leak.
TEST_F(SymTeamDestroyAllTest, DriverCallsFail_StillEmptiesList) {
  PushTeam(reinterpret_cast<void*>(0x400000));
  PushTeam(reinterpret_cast<void*>(0x500000));
  ScopedHook unmap(g_hipMemUnmap, [](void*, size_t) { return hipErrorInvalidValue; });
  ScopedHook addrFree(g_hipMemAddressFree, [](void*, size_t) { return hipErrorInvalidValue; });
  ScopedHook release(g_hipMemRelease, [](hipMemGenericAllocationHandle_t) { return hipErrorInvalidValue; });

  symTeamDestroyAll(comm);
  EXPECT_EQ(comm->devrState.teamHead, nullptr);
  EXPECT_EQ(unmap.calls, 2);  // both teams attempted, not abandoned after the first
}


// ---------------------------------------------------------------------------
// symMemoryRegisterGin publishes memory to the GIN layer. It splits on whether
// any rank contributed a CPU-backed segment: without one the whole allocation
// registers as a single device window; with one it takes the elastic path,
// which validates that every rank agrees on the segment layout and registers
// one window per segment.
//
// This suite covers the single-window path; the elastic path follows below.

class SymMemoryRegisterGinTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  ncclDevrMemory mem{};
  std::vector<size_t> segmentSizes;
  std::vector<hipMemGenericAllocationHandle_t> memHandles;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->rank = 0;
    comm->nRanks = 1;
    comm->bootstrap = reinterpret_cast<void*>(0x1);

    mem.primaryAddr = reinterpret_cast<void*>(0x100000);
    mem.size = 8192;
    mem.numSegments = 1;
    mem.maxGlobalNumSegments = 1;
    mem.globalHasSysmemSegment = false;
    segmentSizes.assign({8192});
    memHandles.assign(1, reinterpret_cast<hipMemGenericAllocationHandle_t>(0x55));
    mem.segmentSizes = segmentSizes.data();
    mem.memHandles = memHandles.data();
  }

  void TearDown() override {
    free(mem.ginSegmentInfos);
    mem.ginSegmentInfos = nullptr;
    g_callocCallIndex = 0;  // TU-local, not covered by the reset below
    g_callocFailAt = -1;
    ResetDevRuntimeFakes();
  }
};

// Branch: no sysmem segment anywhere, so one device window covers the whole
// allocation and a single segment info describes it.
TEST_F(SymMemoryRegisterGinTest, NoSysmemSegment_RegistersOneDeviceWindow) {
  size_t registeredSize = 0;
  int registeredType = -1;
  bool registeredMultiSegment = true;
  ScopedHook reg(g_ginRegister, [&](ncclComm*, void* addr, size_t size, void*[], ncclGinWindow_t[], int, bool multi,
                                    int memType) {
    EXPECT_EQ(addr, mem.primaryAddr);
    registeredSize = size;
    registeredType = memType;
    registeredMultiSegment = multi;
    return ncclSuccess;
  });

  ASSERT_EQ(symMemoryRegisterGin(comm, &mem), ncclSuccess);
  EXPECT_EQ(reg.calls, 1);
  EXPECT_EQ(registeredSize, mem.size);  // the whole allocation, not a segment
  EXPECT_EQ(registeredType, NCCL_PTR_CUDA);
  EXPECT_FALSE(registeredMultiSegment);
  ASSERT_EQ(mem.numGinSegments, 1);
  ASSERT_NE(mem.ginSegmentInfos, nullptr);
  EXPECT_EQ(mem.ginSegmentInfos[0].segmentSize, mem.size);
  EXPECT_EQ(mem.ginSegmentInfos[0].memType, hipMemLocationTypeDevice);
}

// The multiSegment flag passed to GIN comes from the communicator-wide segment
// count, not this rank's.
TEST_F(SymMemoryRegisterGinTest, MultipleGlobalSegments_FlagsRegistrationMultiSegment) {
  mem.maxGlobalNumSegments = 2;
  bool registeredMultiSegment = false;
  ScopedHook reg(g_ginRegister,
                 [&](ncclComm*, void*, size_t, void*[], ncclGinWindow_t[], int, bool multi, int) {
                   registeredMultiSegment = multi;
                   return ncclSuccess;
                 });

  ASSERT_EQ(symMemoryRegisterGin(comm, &mem), ncclSuccess);
  EXPECT_TRUE(registeredMultiSegment);
}

// Branch: the GIN registration fails, so no segment info is allocated.
TEST_F(SymMemoryRegisterGinTest, RegisterFails_ReturnsErrorWithoutSegmentInfo) {
  ScopedHook reg(g_ginRegister,
                 [](ncclComm*, void*, size_t, void*[], ncclGinWindow_t[], int, bool, int) {
                   return ncclSystemError;
                 });

  EXPECT_NE(symMemoryRegisterGin(comm, &mem), ncclSuccess);
  EXPECT_EQ(mem.ginSegmentInfos, nullptr);
  EXPECT_EQ(mem.numGinSegments, 0);
}

// Branch: the segment-info allocation fails after a successful registration.
TEST_F(SymMemoryRegisterGinTest, SegmentInfoAllocFails_ReturnsError) {
  ScopedHook reg(g_ginRegister,
                 [](ncclComm*, void*, size_t, void*[], ncclGinWindow_t[], int, bool, int) { return ncclSuccess; });
  g_callocFailAt = g_callocCallIndex;  // fail the next ncclCalloc

  EXPECT_NE(symMemoryRegisterGin(comm, &mem), ncclSuccess);
  EXPECT_EQ(mem.numGinSegments, 0);
}


// ---------------------------------------------------------------------------
// symMemoryRegisterGin, elastic path: taken when some rank contributed a
// CPU-backed segment. Every rank must agree on the segment layout, so it checks
// the count locally, all-gathers the sizes to check those too, then registers
// one window per segment at its own offset. A failure part-way through
// deregisters the windows already made.

class SymMemoryRegisterGinElasticTest : public SymMemoryRegisterGinTest {
protected:
  std::vector<size_t> gathered;  // what the all-gather hook publishes back

  void SetUp() override {
    SymMemoryRegisterGinTest::SetUp();
    comm->nRanks = 2;
    mem.globalHasSysmemSegment = true;
    mem.numSegments = 2;
    mem.maxGlobalNumSegments = 2;
    segmentSizes.assign({4096, 8192});
    memHandles.assign(2, reinterpret_cast<hipMemGenericAllocationHandle_t>(0x55));
    mem.segmentSizes = segmentSizes.data();
    mem.memHandles = memHandles.data();
  }

  // Every rank reports the same layout, which is what the checks require.
  std::function<ncclResult_t(void*, void*, int)> AgreeingAllGather() {
    return [this](void*, void* buf, int) {
      auto* sizes = static_cast<size_t*>(buf);
      for (int r = 0; r < comm->nRanks; r++) {
        for (int s = 0; s < mem.maxGlobalNumSegments; s++) {
          sizes[r * mem.maxGlobalNumSegments + s] = segmentSizes[s];
        }
      }
      return ncclSuccess;
    };
  }
};

// Branch: this rank's segment count disagrees with the communicator's, rejected
// before any all-gather.
TEST_F(SymMemoryRegisterGinElasticTest, SegmentCountMismatch_ReturnsInvalidUsage) {
  mem.numSegments = 1;  // comm-wide max is 2
  ScopedHook gather(g_bootstrapAllGather, [](void*, void*, int) { return ncclSuccess; });

  EXPECT_EQ(symMemoryRegisterGin(comm, &mem), ncclInvalidUsage);
  EXPECT_EQ(gather.calls, 0);
}

// The happy path: one window per segment, each at its own running offset and
// with the pointer type its location implies.
TEST_F(SymMemoryRegisterGinElasticTest, AgreeingRanks_RegistersOneWindowPerSegment) {
  ScopedHook gather(g_bootstrapAllGather, AgreeingAllGather());
  ScopedHook props(g_hipMemGetAllocationPropertiesFromHandle,
                   [](hipMemAllocationProp* prop, hipMemGenericAllocationHandle_t) {
                     if (prop) {
                       *prop = hipMemAllocationProp{};
                       prop->location.type = hipMemLocationTypeHostNuma;  // CPU-backed
                     }
                     return hipSuccess;
                   });
  std::vector<uintptr_t> addrs;
  std::vector<size_t> sizes;
  std::vector<int> types;
  ScopedHook reg(g_ginRegister,
                 [&](ncclComm*, void* addr, size_t size, void*[], ncclGinWindow_t[], int, bool, int memType) {
                   addrs.push_back(reinterpret_cast<uintptr_t>(addr));
                   sizes.push_back(size);
                   types.push_back(memType);
                   return ncclSuccess;
                 });

  ASSERT_EQ(symMemoryRegisterGin(comm, &mem), ncclSuccess);
  ASSERT_EQ(reg.calls, 2);
  const uintptr_t base = reinterpret_cast<uintptr_t>(mem.primaryAddr);
  EXPECT_EQ(addrs[0], base);
  EXPECT_EQ(addrs[1], base + 4096);  // advanced by the first segment's size
  EXPECT_EQ(sizes[0], 4096u);
  EXPECT_EQ(sizes[1], 8192u);
  EXPECT_EQ(types[0], NCCL_PTR_HOST);  // host-NUMA segments register as host
  EXPECT_EQ(types[1], NCCL_PTR_HOST);
  EXPECT_EQ(mem.numGinSegments, 2);
}

// A device-backed segment on the elastic path still registers as device memory,
// so the pointer type follows the segment rather than the path.
TEST_F(SymMemoryRegisterGinElasticTest, DeviceSegment_RegistersAsCudaPointer) {
  ScopedHook gather(g_bootstrapAllGather, AgreeingAllGather());
  std::vector<int> types;
  ScopedHook reg(g_ginRegister,
                 [&](ncclComm*, void*, size_t, void*[], ncclGinWindow_t[], int, bool, int memType) {
                   types.push_back(memType);
                   return ncclSuccess;
                 });

  ASSERT_EQ(symMemoryRegisterGin(comm, &mem), ncclSuccess);  // props fake reports device
  ASSERT_EQ(types.size(), 2u);
  EXPECT_EQ(types[0], NCCL_PTR_CUDA);
}

// Branch: another rank reports a different segment size, so the layout check
// rejects it after the gather.
TEST_F(SymMemoryRegisterGinElasticTest, SizeMismatchAcrossRanks_ReturnsInvalidUsage) {
  ScopedHook gather(g_bootstrapAllGather, [this](void*, void* buf, int) {
    auto* sizes = static_cast<size_t*>(buf);
    for (int r = 0; r < comm->nRanks; r++) {
      for (int s = 0; s < mem.maxGlobalNumSegments; s++) {
        sizes[r * mem.maxGlobalNumSegments + s] = segmentSizes[s];
      }
    }
    sizes[1 * mem.maxGlobalNumSegments + 0] = 999;  // rank 1 disagrees on segment 0
    return ncclSuccess;
  });
  ScopedHook reg(g_ginRegister,
                 [](ncclComm*, void*, size_t, void*[], ncclGinWindow_t[], int, bool, int) { return ncclSuccess; });

  EXPECT_EQ(symMemoryRegisterGin(comm, &mem), ncclInvalidUsage);
  EXPECT_EQ(reg.calls, 0);
}

// Branch: reading a segment's allocation properties fails.
TEST_F(SymMemoryRegisterGinElasticTest, PropertiesFail_ReturnsErrorWithoutGathering) {
  ScopedHook props(g_hipMemGetAllocationPropertiesFromHandle,
                   [](hipMemAllocationProp*, hipMemGenericAllocationHandle_t) { return hipErrorInvalidValue; });
  ScopedHook gather(g_bootstrapAllGather, [](void*, void*, int) { return ncclSuccess; });

  EXPECT_NE(symMemoryRegisterGin(comm, &mem), ncclSuccess);
  EXPECT_EQ(gather.calls, 0);
}

// Branch: the all-gather itself fails.
TEST_F(SymMemoryRegisterGinElasticTest, AllGatherFails_ReturnsError) {
  ScopedHook gather(g_bootstrapAllGather, [](void*, void*, int) { return ncclSystemError; });
  ScopedHook reg(g_ginRegister,
                 [](ncclComm*, void*, size_t, void*[], ncclGinWindow_t[], int, bool, int) { return ncclSuccess; });

  EXPECT_NE(symMemoryRegisterGin(comm, &mem), ncclSuccess);
  EXPECT_EQ(reg.calls, 0);
}

// Branch: the rollback. The second segment fails to register, so the first --
// already registered -- must be deregistered rather than leaked.
TEST_F(SymMemoryRegisterGinElasticTest, SecondSegmentFails_DeregistersTheFirst) {
  ScopedHook gather(g_bootstrapAllGather, AgreeingAllGather());
  int registered = 0;
  ScopedHook reg(g_ginRegister,
                 [&](ncclComm*, void*, size_t, void*[], ncclGinWindow_t[], int, bool, int) {
                   return ++registered == 2 ? ncclSystemError : ncclSuccess;
                 });
  ScopedHook dereg(g_ginDeregister, [](ncclComm*, void*[]) { return ncclSuccess; });

  EXPECT_NE(symMemoryRegisterGin(comm, &mem), ncclSuccess);
  EXPECT_EQ(reg.calls, 2);
  EXPECT_EQ(dereg.calls, 1);  // exactly the one that succeeded
  EXPECT_EQ(mem.ginSegmentInfos, nullptr);
}


// ---------------------------------------------------------------------------
// symMemoryRegisterRma connects the RMA proxy if it is not up yet, then
// registers the memory with it. Both steps are NCCLCHECK'd, so the only
// branches are their two failure arms.

class SymMemoryRegisterRmaTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  ncclDevrMemory mem{};

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    mem.primaryAddr = reinterpret_cast<void*>(0x100000);
    mem.size = 8192;
  }
  void TearDown() override { ResetDevRuntimeFakes(); }
};

// The happy path: connect first, then register the whole allocation into the
// memory's own RMA window array.
TEST_F(SymMemoryRegisterRmaTest, Succeeds_ConnectsThenRegisters) {
  int order = 0, connectOrder = 0, registerOrder = 0;
  void* registeredAddr = nullptr;
  size_t registeredSize = 0;
  void** registeredWins = nullptr;
  ScopedHook connect(g_rmaProxyConnectOnce, [&](ncclComm*) {
    connectOrder = ++order;
    return ncclSuccess;
  });
  ScopedHook reg(g_rmaProxyRegister, [&](ncclComm*, void* addr, size_t size, void* wins[]) {
    registerOrder = ++order;
    registeredAddr = addr;
    registeredSize = size;
    registeredWins = wins;
    return ncclSuccess;
  });

  EXPECT_EQ(symMemoryRegisterRma(comm, &mem), ncclSuccess);
  EXPECT_EQ(connectOrder, 1);
  EXPECT_EQ(registerOrder, 2);
  EXPECT_EQ(registeredAddr, mem.primaryAddr);
  EXPECT_EQ(registeredSize, mem.size);
  EXPECT_EQ(registeredWins, mem.rmaHostWins);
}

// Branch: the proxy is unreachable, so nothing is registered against it.
TEST_F(SymMemoryRegisterRmaTest, ConnectFails_ReturnsErrorWithoutRegistering) {
  ScopedHook connect(g_rmaProxyConnectOnce, [](ncclComm*) { return ncclSystemError; });
  ScopedHook reg(g_rmaProxyRegister,
                 [](ncclComm*, void*, size_t, void*[]) { return ncclSuccess; });

  EXPECT_NE(symMemoryRegisterRma(comm, &mem), ncclSuccess);
  EXPECT_EQ(reg.calls, 0);
}

// Branch: the registration itself fails.
TEST_F(SymMemoryRegisterRmaTest, RegisterFails_ReturnsError) {
  ScopedHook reg(g_rmaProxyRegister,
                 [](ncclComm*, void*, size_t, void*[]) { return ncclSystemError; });

  EXPECT_NE(symMemoryRegisterRma(comm, &mem), ncclSuccess);
  EXPECT_EQ(reg.calls, 1);
}


// ---------------------------------------------------------------------------
// symMemoryObtain allocates an ncclDevrMemory, agrees a layout with the other
// ranks, reserves space, maps the LSA team, binds existing teams, registers
// with GIN/RMA where enabled, and links the result onto devrState.memHead.
//
// This suite covers the setup half: the aggregation of the all-gathered
// per-rank info, and the failure arms before anything is mapped. The
// register-and-bind half and the rollback paths follow below.

class SymMemoryObtainSetupTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  std::vector<int> lsaRankList;
  std::vector<hipMemGenericAllocationHandle_t> memHandles;
  ncclDevrMemory* obtained = nullptr;

  // Mirrors the anonymous struct symMemoryObtain all-gathers. Layout must match
  // for the hook to populate what the function then reads back.
  struct SegmentInfo {
    int numSegments;
    bool hasSysmemSegment;
    size_t totalSize;
  };

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->rank = 0;
    comm->nRanks = 2;
    comm->bootstrap = reinterpret_cast<void*>(0x1);

    ncclDevrState* devr = &comm->devrState;
    devr->lsaSelf = 0;
    devr->lsaSize = 2;
    devr->nLsaTeams = 1;
    devr->bigSize = 1u << 20;
    devr->granularity = 4096;
    devr->lsaFlatBase = reinterpret_cast<void*>(0x200000);
    lsaRankList.assign({0, 1});
    devr->lsaRankList = lsaRankList.data();

    memHandles.assign(1, reinterpret_cast<hipMemGenericAllocationHandle_t>(0x55));
  }

  void TearDown() override {
    if (obtained != nullptr) symMemoryDestroy(comm, obtained);
    comm->devrState.lsaRankList = nullptr;  // borrowed, not malloc'd
    g_callocCallIndex = 0;                  // TU-local, not covered by the reset below
    g_callocFailAt = -1;
    ResetDevRuntimeFakes();
  }

  ncclResult_t Obtain(int numSegments = 1, size_t size = 4096, bool hasSysmem = false) {
    return symMemoryObtain(comm, memHandles.data(), numSegments, reinterpret_cast<void*>(0x100000), size,
                           /*winFlags=*/0, &obtained, hasSysmem);
  }

  // Publish per-rank info back through the all-gather, as peers would.
  std::function<ncclResult_t(void*, void*, int)> GatherReporting(std::vector<SegmentInfo> perRank) {
    return [perRank](void*, void* buf, int) {
      auto* info = static_cast<SegmentInfo*>(buf);
      for (size_t r = 0; r < perRank.size(); r++) info[r] = perRank[r];
      return ncclSuccess;
    };
  }
};

// maxGlobalNumSegments is the max across every rank, not just ours -- a peer
// with more segments raises it.
TEST_F(SymMemoryObtainSetupTest, AggregatesMaxSegmentsAcrossRanks) {
  ScopedHook gather(g_bootstrapAllGather, GatherReporting({{1, false, 4096}, {3, false, 4096}}));

  ASSERT_EQ(Obtain(/*numSegments=*/1), ncclSuccess);
  EXPECT_EQ(obtained->maxGlobalNumSegments, 3);
  EXPECT_EQ(obtained->numSegments, 1);  // our own count is unchanged
}

// globalHasSysmemSegment is an OR across ranks: one peer with CPU-backed memory
// puts everyone on the elastic path.
TEST_F(SymMemoryObtainSetupTest, PeerSysmemSegment_SetsGlobalFlag) {
  ScopedHook gather(g_bootstrapAllGather, GatherReporting({{1, false, 4096}, {1, true, 4096}}));

  ASSERT_EQ(Obtain(/*numSegments=*/1, /*size=*/4096, /*hasSysmem=*/false), ncclSuccess);
  EXPECT_FALSE(obtained->hasSysmemSegment);       // ours
  EXPECT_TRUE(obtained->globalHasSysmemSegment);  // the communicator's
}

// No rank reporting sysmem leaves the flag clear.
TEST_F(SymMemoryObtainSetupTest, NoSysmemAnywhere_LeavesGlobalFlagClear) {
  ScopedHook gather(g_bootstrapAllGather, GatherReporting({{1, false, 4096}, {1, false, 4096}}));

  ASSERT_EQ(Obtain(), ncclSuccess);
  EXPECT_FALSE(obtained->globalHasSysmemSegment);
}

// lsaMin/MaxSize come from the LSA slice of the gathered info, so asymmetric
// peer sizes widen the range the space allocation has to cover.
TEST_F(SymMemoryObtainSetupTest, AsymmetricPeerSizes_TrackMinAndMax) {
  ScopedHook gather(g_bootstrapAllGather, GatherReporting({{1, false, 4096}, {2, false, 16384}}));
  int64_t allocSize = 0;
  ScopedHook alloc(g_spaceAlloc, [&](ncclSpace*, int64_t, int64_t size, int, int64_t* out) {
    allocSize = size;
    *out = 0;
    return ncclSuccess;
  });

  ASSERT_EQ(Obtain(/*numSegments=*/1, /*size=*/4096), ncclSuccess);
  EXPECT_EQ(obtained->lsaMinSize, 4096u);
  EXPECT_EQ(obtained->lsaMaxSize, 16384u);
  EXPECT_EQ(obtained->lsaNumSegments[1], 2);  // the peer's count, from its slice
  EXPECT_EQ(allocSize, 16384);                // reserved for the largest LSA rank
}

// Branch: the space allocation fails, so nothing is mapped or linked.
TEST_F(SymMemoryObtainSetupTest, SpaceAllocFails_ReturnsErrorWithoutLinking) {
  ScopedHook gather(g_bootstrapAllGather, GatherReporting({{1, false, 4096}, {1, false, 4096}}));
  ScopedHook alloc(g_spaceAlloc,
                   [](ncclSpace*, int64_t, int64_t, int, int64_t*) { return ncclSystemError; });

  EXPECT_NE(Obtain(), ncclSuccess);
  EXPECT_EQ(comm->devrState.memHead, nullptr);
}

// Branch: the all-gather fails before any aggregation happens.
TEST_F(SymMemoryObtainSetupTest, AllGatherFails_ReturnsErrorWithoutLinking) {
  ScopedHook gather(g_bootstrapAllGather, [](void*, void*, int) { return ncclSystemError; });

  EXPECT_NE(Obtain(), ncclSuccess);
  EXPECT_EQ(comm->devrState.memHead, nullptr);
}

// Branch: populating our own segment sizes fails.
TEST_F(SymMemoryObtainSetupTest, PopulateSegmentSizesFails_ReturnsError) {
  ScopedHook populate(g_devrPopulateSegmentSizes,
                      [](ncclDevrMemory*, int) { return ncclSystemError; });
  ScopedHook gather(g_bootstrapAllGather, [](void*, void*, int) { return ncclSuccess; });

  EXPECT_NE(Obtain(), ncclSuccess);
  EXPECT_EQ(gather.calls, 0);
  EXPECT_EQ(comm->devrState.memHead, nullptr);
}

// Branch: the very first allocation fails, before any member is written.
TEST_F(SymMemoryObtainSetupTest, MemoryAllocFails_ReturnsErrorWithoutLinking) {
  g_callocFailAt = g_callocCallIndex;  // fail the next ncclCalloc
  ScopedHook gather(g_bootstrapAllGather, [](void*, void*, int) { return ncclSuccess; });

  EXPECT_NE(Obtain(), ncclSuccess);
  EXPECT_EQ(gather.calls, 0);
  EXPECT_EQ(comm->devrState.memHead, nullptr);
}


// ---------------------------------------------------------------------------
// symMemoryObtain, second half: once space is reserved and the LSA team mapped,
// bind every existing team, register with GIN or RMA where enabled, and link
// the memory onto devrState.memHead.

class SymMemoryObtainRegisterTest : public SymMemoryObtainSetupTest {
protected:
  void SetUp() override {
    SymMemoryObtainSetupTest::SetUp();
    // Every rank agrees on one 4096-byte segment unless a test says otherwise.
    agreeing = GatherReporting({{1, false, 4096}, {1, false, 4096}, {1, false, 4096}, {1, false, 4096}});
  }

  std::function<ncclResult_t(void*, void*, int)> agreeing;

  // Satisfy every condition of rmaProxyEnabled except the param, which each
  // test leaves at its default (RMA_DISABLE=0, i.e. enabled).
  void EnableRmaPrerequisites() {
    comm->nRanks = 4;
    comm->devrState.nLsaTeams = 2;
    comm->config.numRmaCtx = 1;
    comm->globalRmaProxySupport = true;
  }
};

// Branch: a caller with no VA of its own gets the LSA mapping instead --
// lsaFlatBase advanced by our slot in the flat space, plus our offset.
TEST_F(SymMemoryObtainRegisterTest, NullPrimaryAddr_DerivesFromLsaMapping) {
  ScopedHook gather(g_bootstrapAllGather, agreeing);
  ScopedHook alloc(g_spaceAlloc, [](ncclSpace*, int64_t, int64_t, int, int64_t* out) {
    *out = 8192;
    return ncclSuccess;
  });

  ASSERT_EQ(symMemoryObtain(comm, memHandles.data(), 1, /*memAddr=*/nullptr, 4096, 0, &obtained, false), ncclSuccess);
  ncclDevrState* devr = &comm->devrState;
  EXPECT_EQ(obtained->primaryAddr,
            static_cast<char*>(devr->lsaFlatBase) + devr->lsaSelf * devr->bigSize + 8192);
}

// Branch: a caller that supplied a VA keeps it.
TEST_F(SymMemoryObtainRegisterTest, CallerPrimaryAddr_IsKept) {
  ScopedHook gather(g_bootstrapAllGather, agreeing);

  ASSERT_EQ(Obtain(), ncclSuccess);
  EXPECT_EQ(obtained->primaryAddr, reinterpret_cast<void*>(0x100000));
}

// Branch: GIN is on, so the memory is registered with it.
TEST_F(SymMemoryObtainRegisterTest, GinEnabled_RegistersWithGin) {
  comm->devrState.ginEnabled = true;
  ScopedHook gather(g_bootstrapAllGather, agreeing);
  ScopedHook reg(g_ginRegister,
                 [](ncclComm*, void*, size_t, void*[], ncclGinWindow_t[], int, bool, int) { return ncclSuccess; });

  ASSERT_EQ(Obtain(), ncclSuccess);
  EXPECT_EQ(reg.calls, 1);
}

// Branch: GIN is off, so registration is skipped and the segment count is
// provisionally 1 -- recomputed later if GIN is activated.
TEST_F(SymMemoryObtainRegisterTest, GinDisabled_DefaultsToOneSegment) {
  ScopedHook gather(g_bootstrapAllGather, agreeing);
  ScopedHook reg(g_ginRegister,
                 [](ncclComm*, void*, size_t, void*[], ncclGinWindow_t[], int, bool, int) { return ncclSuccess; });

  ASSERT_EQ(Obtain(), ncclSuccess);
  EXPECT_EQ(reg.calls, 0);
  EXPECT_EQ(obtained->numGinSegments, 1);
}

// Branch: every rmaProxyEnabled condition holds and the layout is single
// segment, so RMA registration runs.
TEST_F(SymMemoryObtainRegisterTest, RmaPrerequisitesMet_RegistersWithRma) {
  EnableRmaPrerequisites();
  ScopedHook gather(g_bootstrapAllGather, agreeing);
  ScopedHook reg(g_rmaProxyRegister, [](ncclComm*, void*, size_t, void*[]) { return ncclSuccess; });

  ASSERT_EQ(Obtain(), ncclSuccess);
  EXPECT_TRUE(comm->devrState.rmaProxyEnabled);
  EXPECT_EQ(reg.calls, 1);
}

// Branch: a single LSA team means there is no remote peer to reach over RMA.
TEST_F(SymMemoryObtainRegisterTest, SingleLsaTeam_LeavesRmaProxyDisabled) {
  EnableRmaPrerequisites();
  comm->devrState.nLsaTeams = 1;
  ScopedHook gather(g_bootstrapAllGather, agreeing);
  ScopedHook reg(g_rmaProxyRegister, [](ncclComm*, void*, size_t, void*[]) { return ncclSuccess; });

  ASSERT_EQ(Obtain(), ncclSuccess);
  EXPECT_FALSE(comm->devrState.rmaProxyEnabled);
  EXPECT_EQ(reg.calls, 0);
}

// Branch: RMA_DISABLE overrides the other three conditions.
TEST_F(SymMemoryObtainRegisterTest, RmaDisabledByParam_LeavesRmaProxyDisabled) {
  EnableRmaPrerequisites();
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t deftVal) -> int64_t {
    return std::string(env) == "RMA_DISABLE" ? 1 : deftVal;
  });
  ScopedHook gather(g_bootstrapAllGather, agreeing);
  ScopedHook reg(g_rmaProxyRegister, [](ncclComm*, void*, size_t, void*[]) { return ncclSuccess; });

  ASSERT_EQ(Obtain(), ncclSuccess);
  EXPECT_FALSE(comm->devrState.rmaProxyEnabled);
  EXPECT_EQ(reg.calls, 0);
}

// Branch: the proxy is enabled but the layout is multi-segment, which RMA does
// not handle -- so the flag is set while registration is skipped.
TEST_F(SymMemoryObtainRegisterTest, RmaEnabledButMultiSegment_SkipsRegistration) {
  EnableRmaPrerequisites();
  ScopedHook gather(g_bootstrapAllGather,
                    GatherReporting({{1, false, 4096}, {2, false, 4096}, {1, false, 4096}, {1, false, 4096}}));
  ScopedHook reg(g_rmaProxyRegister, [](ncclComm*, void*, size_t, void*[]) { return ncclSuccess; });

  ASSERT_EQ(Obtain(), ncclSuccess);
  EXPECT_TRUE(comm->devrState.rmaProxyEnabled);
  EXPECT_EQ(obtained->maxGlobalNumSegments, 2);
  EXPECT_EQ(reg.calls, 0);
}

// The memory is pushed onto memHead, ahead of anything already there.
TEST_F(SymMemoryObtainRegisterTest, LinksOntoMemHeadAndReportsOut) {
  ncclDevrMemory existing{};
  comm->devrState.memHead = &existing;
  ScopedHook gather(g_bootstrapAllGather, agreeing);

  ASSERT_EQ(Obtain(), ncclSuccess);
  EXPECT_EQ(comm->devrState.memHead, obtained);
  EXPECT_EQ(obtained->next, &existing);
  // `existing` is a stack object, so drop it from the chain before TearDown.
  // memHead must keep pointing at `obtained`: symMemoryDestroy walks the list
  // to unlink and dereferences null if its argument is not on it.
  obtained->next = nullptr;
}


// ---------------------------------------------------------------------------
// symMemoryObtain / symMemoryDestroy: the original lifecycle regression.
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
