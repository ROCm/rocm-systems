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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <unistd.h>
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

// Three runs, so the fold inside the loop has to carry a value forward.
// With only two runs it cannot be observed: lsaSize starts at 0 and
// gcd(0, n) == n, so dropping the mid-loop fold gives the same answer. Here
// runs of 3, 3 and 2 fold to gcd(gcd(3, 3), 2) == 1, where skipping the
// in-loop folds would yield 2.
TEST_F(ComputeLsaSizeTest, ThreeRuns_FoldsEveryRunNotJustTheLast) {
  SetTopology({0, 0, 0, 1, 1, 1, 2, 2});
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

// Branch: windows still registered at finalize are drained through
// symWindowDestroy, which also releases the window table behind them. Deferred
// until symWindowCreate/Destroy had tests of their own, so reaching this loop
// no longer exercises untested code.
TEST_F(DevrFinalizeTest, DrainsRemainingWindows) {
  ASSERT_EQ(ncclDevrInitOnce(comm), ncclSuccess);

  auto handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
  void* addr = reinterpret_cast<void*>(0x100000);
  ncclDevrMemory* mem = nullptr;
  ASSERT_EQ(symMemoryObtain(comm, &handle, 1, addr, 4096, 0, &mem, false), ncclSuccess);
  ncclWindow_vidmem* winDev = nullptr;
  ASSERT_EQ(symWindowCreate(comm, mem, 0, addr, 4096, 0, nullptr, &winDev, nullptr, nullptr), ncclSuccess);
  ASSERT_EQ(comm->devrState.winSortedCount, 1);
  ASSERT_NE(comm->devrState.windowTable, nullptr);

  ASSERT_EQ(ncclDevrFinalize(comm), ncclSuccess);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
  EXPECT_EQ(comm->devrState.memHead, nullptr);
}

// The drain is a loop, so more than one window must come out -- a version that
// destroyed only the head would leave the rest mapped.
TEST_F(DevrFinalizeTest, DrainsEveryRemainingWindow) {
  ASSERT_EQ(ncclDevrInitOnce(comm), ncclSuccess);

  auto handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
  for (int i = 0; i < 3; i++) {
    void* addr = reinterpret_cast<void*>(0x100000 + i * 0x1000);
    ncclDevrMemory* mem = nullptr;
    ASSERT_EQ(symMemoryObtain(comm, &handle, 1, addr, 4096, 0, &mem, false), ncclSuccess);
    ASSERT_EQ(symWindowCreate(comm, mem, 0, addr, 4096, 0, nullptr, nullptr, nullptr, nullptr), ncclSuccess);
  }
  ASSERT_EQ(comm->devrState.winSortedCount, 3);

  ASSERT_EQ(ncclDevrFinalize(comm), ncclSuccess);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
  EXPECT_EQ(comm->devrState.memHead, nullptr);
}

// Branch: the proxy-only drain. Non-sym windows are not backed by the
// symmetric VMM machinery, so finalize unwinds them through
// windowDeregisterNonSym instead of symWindowDestroy. Deferred until that
// function had tests of its own.
TEST_F(DevrFinalizeTest, ProxyOnly_DrainsNonSymWindows) {
  comm->symmetricSupport = 0;
  ASSERT_EQ(ncclDevrInitOnce(comm), ncclSuccess);

  ncclWindow_t out = nullptr;
  ASSERT_EQ(windowRegisterNonSym(comm, reinterpret_cast<void*>(0x100000), 4096, 0, nullptr, &out), ncclSuccess);
  ASSERT_EQ(comm->devrState.winSortedCount, 1);

  ASSERT_EQ(ncclDevrFinalize(comm), ncclSuccess);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
}

// The drain must make progress even when the deregister fails: it only removes
// the entry on success, so finalize drops it regardless to avoid spinning.
TEST_F(DevrFinalizeTest, ProxyOnlyDrainFails_StillEmptiesWindowList) {
  comm->symmetricSupport = 0;
  ASSERT_EQ(ncclDevrInitOnce(comm), ncclSuccess);

  ncclWindow_t out = nullptr;
  ASSERT_EQ(windowRegisterNonSym(comm, reinterpret_cast<void*>(0x100000), 4096, 0, nullptr, &out), ncclSuccess);
  ScopedHook poolFree(g_shadowPoolFree,
                      [](ncclShadowPool*, void*, hipStream_t) { return ncclSystemError; });

  ASSERT_EQ(ncclDevrFinalize(comm), ncclSuccess);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
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
//
// The descriptor is owned by the test rather than by DefaultProxyClientGetFdBlocking
// so the leak below is contained: dev_runtime.cc:359 jumps to fail on import
// failure, skipping the close(fd) at :362, so the fd is still open when this
// returns -- one leaked descriptor per failed import. EXPECT_EQ(fcntl(fd,
// F_GETFD), -1) is the assertion that belongs here and fails today; see
// AICOMRCCL-2180 finding 14. Pinning the current state instead would lock the
// leak in, so the test just cleans up after it.
TEST_F(SymImportAndMapSegmentTest, PosixFdImportFails_ReturnsError) {
  ncclCuMemHandleType = hipMemHandleTypePosixFileDescriptor;
  int handedOut = -1;
  ScopedHook proxy(g_proxyClientGetFdBlocking, [&](ncclComm*, int, void*, int* fd) {
    handedOut = open("/dev/null", O_RDONLY);
    if (fd) *fd = handedOut;
    return handedOut >= 0 ? ncclSuccess : ncclSystemError;
  });
  ScopedHook import(g_hipMemImportFromShareableHandle,
                    [](hipMemGenericAllocationHandle_t*, void*, hipMemAllocationHandleType) {
                      return hipErrorInvalidValue;
                    });

  EXPECT_NE(symMemoryImportAndMapSegmentHandle(comm, 1, kAddr, &msg, {}, /*reuseLocal=*/false), ncclSuccess);
  EXPECT_EQ(import.calls, 1);

  ASSERT_GE(handedOut, 0);
  if (fcntl(handedOut, F_GETFD) != -1) close(handedOut);  // no double close if the leak is ever fixed
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

// One test, not one per guard arm. On this build the guarded body is empty, so
// the function is `return ncclSuccess` regardless of its inputs: inverting
// nvlsSupport, dropping an && arm or deleting the `if` outright changes
// nothing observable. Splitting this across the guard's arms would report
// branch coverage the build cannot actually exercise. Walk the input space in
// one case instead, so the no-op is asserted without overclaiming.
TEST_F(SymBindTeamMemoryTest, IsANoOpOnThisBuild) {
  for (int nvls : {0, 1}) {
    for (void* mc : {static_cast<void*>(nullptr), reinterpret_cast<void*>(0x1000)}) {
      comm->nvlsSupport = nvls;
      team->mcBasePtr = mc;
      EXPECT_EQ(symBindTeamMemory(comm, team, &mem), ncclSuccess) << "nvls=" << nvls << " mc=" << mc;
    }
  }
}


// ---------------------------------------------------------------------------
// symUnbindTeamMemory is the counterpart to symBindTeamMemory, with a third
// guard arm: !mem->globalHasSysmemSegment. Its body is behind the same
// CUDART_VERSION check and so is likewise not compiled on HIP -- see the note
// on SymBindTeamMemoryTest for why this is one test rather than one per arm.

class SymUnbindTeamMemoryTest : public SymTeamMemoryTest {};

TEST_F(SymUnbindTeamMemoryTest, IsANoOpOnThisBuild) {
  for (int nvls : {0, 1}) {
    for (void* mc : {static_cast<void*>(nullptr), reinterpret_cast<void*>(0x1000)}) {
      for (bool sysmem : {false, true}) {
        comm->nvlsSupport = nvls;
        team->mcBasePtr = mc;
        mem.globalHasSysmemSegment = sysmem;
        EXPECT_EQ(symUnbindTeamMemory(comm, team, &mem), ncclSuccess)
          << "nvls=" << nvls << " mc=" << mc << " sysmem=" << sysmem;
      }
    }
  }
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
// writing *outTeam, so a test asserting that would lock the bug in as expected
// behaviour. Written up against AICOMRCCL-2180.

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
  // Required, not optional: without it DefaultMemAddressFree munmap()s the
  // fabricated 0x400000 for real. Harmless only while the binary is PIE and
  // that address sits below the load base -- test/host links -no-pie, where
  // 0x400000 is the ELF text base.
  ScopedHook addrFree(g_hipMemAddressFree, [](void*, size_t) { return hipSuccess; });

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
  ScopedHook addrFree(g_hipMemAddressFree, [](void*, size_t) { return hipSuccess; });

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

  // Publish per-rank info back through the all-gather, as peers would. The
  // buffer is sized by comm->nRanks, so clamp: a table longer than the comm
  // overflows it and corrupts the heap for whatever runs next.
  std::function<ncclResult_t(void*, void*, int)> GatherReporting(std::vector<SegmentInfo> perRank) {
    return [this, perRank](void*, void* buf, int) {
      auto* info = static_cast<SegmentInfo*>(buf);
      const int n = std::min(static_cast<int>(perRank.size()), comm->nRanks);
      for (int r = 0; r < n; r++) info[r] = perRank[r];
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
// symMemoryObtain, rollback: three labels unwind progressively more work --
// fail_mem frees the allocations, fail_mem_space also returns the reserved
// space, and fail_mem_space_teams also unbinds every team already bound. None
// of this is visible in the return code, so the assertions are on the driver
// and allocator calls the rollback makes.

class SymMemoryObtainRollbackTest : public SymMemoryObtainSetupTest {
protected:
  std::vector<unsigned char> teamStorage;

  void SetUp() override {
    SymMemoryObtainSetupTest::SetUp();
    // A peer larger than us, so lsaMaxSize differs from our own size and the
    // space accounting has to use the right one.
    agreeing = GatherReporting({{1, false, 4096}, {1, false, 16384}});
  }

  void TearDown() override {
    comm->devrState.teamHead = nullptr;  // borrowed storage, not malloc'd
    SymMemoryObtainSetupTest::TearDown();
  }

  std::function<ncclResult_t(void*, void*, int)> agreeing;

  // ncclDevrTeam ends in a flexible array member, so back it with a buffer.
  void PushTeam() {
    teamStorage.assign(sizeof(ncclDevrTeam) + sizeof(int), 0);
    auto* t = reinterpret_cast<ncclDevrTeam*>(teamStorage.data());
    t->team.nRanks = 1;
    comm->devrState.teamHead = t;
  }

  // Hand out a recognisable offset so the matching free can be checked.
  std::function<ncclResult_t(ncclSpace*, int64_t, int64_t, int, int64_t*)> AllocAt(int64_t offset) {
    return [offset](ncclSpace*, int64_t, int64_t, int, int64_t* out) {
      *out = offset;
      return ncclSuccess;
    };
  }
};

// Label fail_mem_space: the LSA mapping failed after space was reserved, so the
// reservation is returned -- at the offset it was given, sized on lsaMaxSize
// (the largest LSA rank) rather than our own size.
TEST_F(SymMemoryObtainRollbackTest, MapLsaTeamFails_ReturnsReservedSpace) {
  ScopedHook gather(g_bootstrapAllGather, agreeing);
  ScopedHook alloc(g_spaceAlloc, AllocAt(8192));
  int64_t freedOffset = -1, freedSize = -1;
  ScopedHook spaceFree(g_spaceFree, [&](ncclSpace*, int64_t offset, int64_t size) {
    freedOffset = offset;
    freedSize = size;
    return ncclSuccess;
  });
  // Fail the team barrier rather than the VA reservation: the fixture already
  // has an lsaFlatBase, so the reservation is skipped and never runs.
  ScopedHook barrier(g_bootstrapIntraNodeBarrier, [](void*, int*, int, int, int) { return ncclSystemError; });

  EXPECT_NE(Obtain(/*numSegments=*/1, /*size=*/4096), ncclSuccess);
  EXPECT_EQ(spaceFree.calls, 1);
  EXPECT_EQ(freedOffset, 8192);
  EXPECT_EQ(freedSize, 16384);  // lsaMaxSize, the peer's size -- not our 4096
  EXPECT_EQ(comm->devrState.memHead, nullptr);
}

// Label fail_mem_space_teams: GIN registration failed after teams were bound,
// so the unbind loop runs and the space is still returned.
TEST_F(SymMemoryObtainRollbackTest, GinRegisterFails_UnbindsTeamsAndReturnsSpace) {
  PushTeam();
  comm->devrState.ginEnabled = true;
  ScopedHook gather(g_bootstrapAllGather, agreeing);
  ScopedHook alloc(g_spaceAlloc, AllocAt(4096));
  ScopedHook spaceFree(g_spaceFree, [](ncclSpace*, int64_t, int64_t) { return ncclSuccess; });
  ScopedHook reg(g_ginRegister,
                 [](ncclComm*, void*, size_t, void*[], ncclGinWindow_t[], int, bool, int) { return ncclSystemError; });

  EXPECT_NE(Obtain(), ncclSuccess);
  EXPECT_EQ(reg.calls, 1);
  EXPECT_EQ(spaceFree.calls, 1);
  EXPECT_EQ(comm->devrState.memHead, nullptr);
}

// Same label, reached from the RMA branch instead.
TEST_F(SymMemoryObtainRollbackTest, RmaRegisterFails_ReturnsSpaceWithoutLinking) {
  PushTeam();
  comm->nRanks = 2;
  comm->devrState.nLsaTeams = 2;
  comm->config.numRmaCtx = 1;
  comm->globalRmaProxySupport = true;
  ScopedHook gather(g_bootstrapAllGather, agreeing);
  ScopedHook alloc(g_spaceAlloc, AllocAt(0));
  ScopedHook spaceFree(g_spaceFree, [](ncclSpace*, int64_t, int64_t) { return ncclSuccess; });
  ScopedHook reg(g_rmaProxyRegister, [](ncclComm*, void*, size_t, void*[]) { return ncclSystemError; });

  EXPECT_NE(Obtain(), ncclSuccess);
  EXPECT_EQ(reg.calls, 1);
  EXPECT_EQ(spaceFree.calls, 1);
  EXPECT_EQ(comm->devrState.memHead, nullptr);
}

// Label fail_mem: a failure before the reservation must not free space that was
// never taken.
TEST_F(SymMemoryObtainRollbackTest, EarlyFailure_DoesNotFreeUnreservedSpace) {
  ScopedHook gather(g_bootstrapAllGather, [](void*, void*, int) { return ncclSystemError; });
  ScopedHook spaceFree(g_spaceFree, [](ncclSpace*, int64_t, int64_t) { return ncclSuccess; });

  EXPECT_NE(Obtain(), ncclSuccess);
  EXPECT_EQ(spaceFree.calls, 0);
  EXPECT_EQ(comm->devrState.memHead, nullptr);
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
// symWindowTableInitOnce allocates the device-side window table the first time
// it is needed and caches it on devrState.

class SymWindowTableInitOnceTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
  }
  void TearDown() override {
    free(comm->devrState.windowTable);  // the shadow-pool fake hands out calloc'd memory
    comm->devrState.windowTable = nullptr;
    ResetDevRuntimeFakes();
  }
};

// Branch: no table yet, so one is allocated and cached.
TEST_F(SymWindowTableInitOnceTest, FirstCall_AllocatesAndCaches) {
  ASSERT_EQ(symWindowTableInitOnce(comm, nullptr), ncclSuccess);
  EXPECT_NE(comm->devrState.windowTable, nullptr);
}

// Branch: a cached table is reused rather than reallocated.
TEST_F(SymWindowTableInitOnceTest, SecondCall_ReusesCachedTable) {
  ASSERT_EQ(symWindowTableInitOnce(comm, nullptr), ncclSuccess);
  ncclDevCommWindowTable* first = comm->devrState.windowTable;

  ScopedHook alloc(g_shadowPoolAlloc,
                   [](ncclShadowPool*, size_t, void**, void**, hipStream_t) { return ncclSuccess; });
  ASSERT_EQ(symWindowTableInitOnce(comm, nullptr), ncclSuccess);
  EXPECT_EQ(alloc.calls, 0);
  EXPECT_EQ(comm->devrState.windowTable, first);
}

// Branch: the allocation fails, so nothing is cached and a later call retries.
TEST_F(SymWindowTableInitOnceTest, AllocFails_LeavesTableUnset) {
  ScopedHook alloc(g_shadowPoolAlloc,
                   [](ncclShadowPool*, size_t, void**, void**, hipStream_t) { return ncclSystemError; });

  EXPECT_NE(symWindowTableInitOnce(comm, nullptr), ncclSuccess);
  EXPECT_EQ(comm->devrState.windowTable, nullptr);
}


// ---------------------------------------------------------------------------
// allocAndPopulateSegmentWindows allocates the per-segment window array from
// the shadow pool, and when GIN is on fills the host copy from the memory's
// per-segment info and pushes it to the device.

class AllocAndPopulateSegmentWindowsTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  ncclDevrMemory mem{};
  std::vector<ncclDevrGinSegmentInfo> ginInfos;
  ncclSegmentWindow* dev = nullptr;
  ncclSegmentWindow* host = nullptr;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();

    ginInfos.assign(2, ncclDevrGinSegmentInfo{});
    ginInfos[0].memType = hipMemLocationTypeDevice;
    ginInfos[0].segmentSize = 4096;
    ginInfos[1].memType = hipMemLocationTypeHostNuma;
    ginInfos[1].segmentSize = 8192;
    mem.ginSegmentInfos = ginInfos.data();
    mem.numGinSegments = 2;
  }
  void TearDown() override {
    free(dev);  // the shadow-pool fake hands out calloc'd memory; dev == host
    dev = host = nullptr;
    ResetDevRuntimeFakes();
  }
};

// Branch: GIN off, so the array is allocated but left untouched -- it is filled
// later, once GIN is activated.
TEST_F(AllocAndPopulateSegmentWindowsTest, GinDisabled_AllocatesWithoutPopulating) {
  ASSERT_EQ(allocAndPopulateSegmentWindows(&comm->devrState, &mem, nullptr, &dev, &host), ncclSuccess);
  ASSERT_NE(host, nullptr);
  EXPECT_EQ(host[0].segmentSize, 0u);  // still zeroed
}

// Branch: GIN on, so each segment's type and size are copied across.
TEST_F(AllocAndPopulateSegmentWindowsTest, GinEnabled_PopulatesEachSegment) {
  comm->devrState.ginEnabled = true;

  ASSERT_EQ(allocAndPopulateSegmentWindows(&comm->devrState, &mem, nullptr, &dev, &host), ncclSuccess);
  ASSERT_NE(host, nullptr);
  EXPECT_EQ(host[0].segmentSize, 4096u);
  EXPECT_EQ(host[0].memType, hipMemLocationTypeDevice);
  EXPECT_EQ(host[1].segmentSize, 8192u);
  EXPECT_EQ(host[1].memType, hipMemLocationTypeHostNuma);
}

// Branch: the shadow-pool allocation fails, so no windows are reported back.
TEST_F(AllocAndPopulateSegmentWindowsTest, AllocFails_ReturnsErrorWithoutOutputs) {
  ScopedHook alloc(g_shadowPoolAlloc,
                   [](ncclShadowPool*, size_t, void**, void**, hipStream_t) { return ncclSystemError; });
  ScopedHook poolFree(g_shadowPoolFree, [](ncclShadowPool*, void*, hipStream_t) { return ncclSuccess; });

  EXPECT_NE(allocAndPopulateSegmentWindows(&comm->devrState, &mem, nullptr, &dev, &host), ncclSuccess);
  EXPECT_EQ(dev, nullptr);
  EXPECT_EQ(host, nullptr);
  EXPECT_EQ(poolFree.calls, 0);  // nothing was allocated, so nothing to release
}


// ---------------------------------------------------------------------------
// symWindowCreate builds a window over a slice of an existing memory: fill the
// device-side descriptor, publish it into the 32-entry window table (chaining a
// new table when the current one is full), and insert into the address-sorted
// window list.
//
// The shadow-pool fake returns one buffer for both device and host and makes
// ToHost the identity, so the table walk operates on real memory here.

class SymWindowCreateTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  ncclDevrMemory mem{};

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->rank = 7;

    ncclDevrState* devr = &comm->devrState;
    devr->lsaSelf = 1;
    devr->lsaSize = 2;
    devr->bigSize = size_t(1) << 33;  // 8GB, so stride4G is a non-trivial 2
    devr->lsaFlatBase = reinterpret_cast<void*>(0x40000000);

    mem.bigOffset = 4096;
    mem.numGinSegments = 3;
  }

  void TearDown() override {
    ncclDevrState* devr = &comm->devrState;
    for (int i = 0; i < devr->winSortedCount; i++) free(devr->winSorted[i].win);
    free(devr->winSorted);
    devr->winSorted = nullptr;
    devr->winSortedCount = devr->winSortedCapacity = 0;
    // The table is a chain of shadow-pool buffers, which the fake calloc'd.
    ncclDevCommWindowTable* t = devr->windowTable;
    while (t != nullptr) {
      ncclDevCommWindowTable* next = t->next;
      free(t);
      t = next;
    }
    devr->windowTable = nullptr;
    ResetDevRuntimeFakes();
  }

  ncclResult_t Create(void* userPtr, size_t userSize, size_t memOffset = 0,
                      ncclWindow_vidmem** outWinDev = nullptr, ncclDevrWindow** outWin = nullptr) {
    return symWindowCreate(comm, &mem, memOffset, userPtr, userSize, /*winFlags=*/0, /*localReg=*/nullptr,
                           outWinDev, outWin, nullptr);
  }
};

// The device-side descriptor is the function's real output. Each field is
// derived from a different input, so a swapped assignment would show here and
// nowhere else.
TEST_F(SymWindowCreateTest, PopulatesDeviceDescriptor) {
  ncclWindow_vidmem* winDev = nullptr;
  ncclDevrWindow* win = nullptr;
  ASSERT_EQ(Create(reinterpret_cast<void*>(0x100000), 8192, /*memOffset=*/8192, &winDev, &win), ncclSuccess);
  ASSERT_NE(winDev, nullptr);
  ASSERT_NE(win, nullptr);

  ncclDevrState* devr = &comm->devrState;
  EXPECT_EQ(win->bigOffset, mem.bigOffset + 8192);  // memory's offset plus ours
  EXPECT_EQ(win->size, 8192u);
  EXPECT_EQ(win->memory, &mem);
  // dev == host under the shadow-pool fake, so winDev is readable directly.
  EXPECT_EQ(winDev->lsaFlatBase, static_cast<char*>(devr->lsaFlatBase) + win->bigOffset);
  EXPECT_EQ(winDev->mcOffset4K, win->bigOffset >> 12);
  EXPECT_EQ(winDev->stride4G, devr->bigSize >> 32);
  EXPECT_EQ(winDev->lsaRank, devr->lsaSelf);
  EXPECT_EQ(winDev->worldRank, comm->rank);
  EXPECT_EQ(winDev->winHost, static_cast<void*>(win));
  EXPECT_EQ(winDev->ginOffset4K, 8192u >> 12);
  EXPECT_EQ(winDev->numSegments, mem.numGinSegments);
}

// Branch: a caller with no VA of its own gets the LSA flat mapping.
TEST_F(SymWindowCreateTest, NullUserPtr_DerivesFromLsaMapping) {
  ncclDevrWindow* win = nullptr;
  ASSERT_EQ(Create(nullptr, 4096, 0, nullptr, &win), ncclSuccess);

  ncclDevrState* devr = &comm->devrState;
  EXPECT_EQ(win->userPtr,
            static_cast<char*>(devr->lsaFlatBase) + devr->lsaSelf * devr->bigSize + mem.bigOffset);
}

// Branch: a caller-supplied VA is kept as is.
TEST_F(SymWindowCreateTest, CallerUserPtr_IsKept) {
  ncclDevrWindow* win = nullptr;
  ASSERT_EQ(Create(reinterpret_cast<void*>(0x100000), 4096, 0, nullptr, &win), ncclSuccess);
  EXPECT_EQ(win->userPtr, reinterpret_cast<void*>(0x100000));
}

// The window is published into the first free table slot, keyed by user address.
TEST_F(SymWindowCreateTest, PublishesIntoWindowTable) {
  ncclWindow_vidmem* winDev = nullptr;
  ASSERT_EQ(Create(reinterpret_cast<void*>(0x100000), 8192, 0, &winDev), ncclSuccess);

  ncclDevCommWindowTable* table = comm->devrState.windowTable;
  ASSERT_NE(table, nullptr);
  EXPECT_EQ(table->entries[0].base, 0x100000u);
  EXPECT_EQ(table->entries[0].size, 8192u);
  EXPECT_EQ(table->entries[0].window, winDev);
}

// Branch: the table holds 32 entries, so the 33rd window chains a new one.
TEST_F(SymWindowCreateTest, FullTable_ChainsAnotherTable) {
  for (int i = 0; i < 32; i++) {
    ASSERT_EQ(Create(reinterpret_cast<void*>(0x100000 + i * 0x1000), 4096), ncclSuccess);
  }
  ncclDevCommWindowTable* first = comm->devrState.windowTable;
  ASSERT_NE(first, nullptr);
  ASSERT_EQ(first->next, nullptr);  // still one table

  ncclWindow_vidmem* winDev = nullptr;
  ASSERT_EQ(Create(reinterpret_cast<void*>(0x200000), 4096, 0, &winDev), ncclSuccess);
  ASSERT_NE(first->next, nullptr);
  EXPECT_EQ(first->next->entries[0].window, winDev);
}

// The sorted list is keyed on user address, so windows created out of order are
// still stored in address order.
TEST_F(SymWindowCreateTest, InsertsIntoSortedListByAddress) {
  ASSERT_EQ(Create(reinterpret_cast<void*>(0x300000), 4096), ncclSuccess);
  ASSERT_EQ(Create(reinterpret_cast<void*>(0x100000), 4096), ncclSuccess);
  ASSERT_EQ(Create(reinterpret_cast<void*>(0x200000), 4096), ncclSuccess);

  ncclDevrState* devr = &comm->devrState;
  ASSERT_EQ(devr->winSortedCount, 3);
  EXPECT_EQ(devr->winSorted[0].userAddr, 0x100000u);
  EXPECT_EQ(devr->winSorted[1].userAddr, 0x200000u);
  EXPECT_EQ(devr->winSorted[2].userAddr, 0x300000u);
}

// Branch: the segment-window allocation fails, so nothing is published.
TEST_F(SymWindowCreateTest, SegmentWindowsFail_ReturnsErrorWithoutPublishing) {
  ScopedHook segWins(g_devrAllocAndPopulateSegmentWindows,
                     [](ncclDevrState*, ncclDevrMemory*, hipStream_t, ncclSegmentWindow**) {
                       return ncclSystemError;
                     });

  EXPECT_NE(Create(reinterpret_cast<void*>(0x100000), 4096), ncclSuccess);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
  EXPECT_EQ(comm->devrState.windowTable, nullptr);
}

// Branch: the descriptor allocation fails before any of it is filled in.
TEST_F(SymWindowCreateTest, DescriptorAllocFails_ReturnsError) {
  ScopedHook alloc(g_shadowPoolAlloc,
                   [](ncclShadowPool*, size_t, void**, void**, hipStream_t) { return ncclSystemError; });

  EXPECT_NE(Create(reinterpret_cast<void*>(0x100000), 4096), ncclSuccess);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
}


// ---------------------------------------------------------------------------
// symWindowDestroy undoes symWindowCreate: release the underlying memory, clear
// the window's slot in the table (searching the chain for it), free the shadow
// allocations, drop it from the sorted list and the global window map.
//
// The fixture drives the real lifecycle -- obtain memory, create a window --
// rather than hand-building state, because symMemoryDestroy walks memHead to
// unlink and faults if its argument is not on the list.

class SymWindowDestroyTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  std::vector<int> lsaRankList;
  std::vector<hipMemGenericAllocationHandle_t> memHandles;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->rank = 0;
    comm->nRanks = 1;
    comm->bootstrap = reinterpret_cast<void*>(0x1);

    ncclDevrState* devr = &comm->devrState;
    devr->lsaSelf = 0;
    devr->lsaSize = 1;
    devr->nLsaTeams = 1;
    devr->bigSize = size_t(1) << 32;
    devr->granularity = 4096;
    devr->lsaFlatBase = reinterpret_cast<void*>(0x40000000);
    lsaRankList.assign({0});
    devr->lsaRankList = lsaRankList.data();

    memHandles.assign(1, reinterpret_cast<hipMemGenericAllocationHandle_t>(0x55));
  }

  void TearDown() override {
    ncclDevrState* devr = &comm->devrState;
    free(devr->winSorted);
    devr->winSorted = nullptr;
    ncclDevCommWindowTable* t = devr->windowTable;
    while (t != nullptr) {
      ncclDevCommWindowTable* next = t->next;
      free(t);
      t = next;
    }
    devr->windowTable = nullptr;
    devr->lsaRankList = nullptr;  // borrowed, not malloc'd
    g_callocCallIndex = 0;
    g_callocFailAt = -1;
    ResetDevRuntimeFakes();
  }

  // A window over freshly obtained memory, so the whole teardown chain is valid.
  ncclWindow_vidmem* MakeWindow(void* userPtr) {
    ncclDevrMemory* mem = nullptr;
    EXPECT_EQ(symMemoryObtain(comm, memHandles.data(), 1, userPtr, 4096, 0, &mem, false), ncclSuccess);
    ncclWindow_vidmem* winDev = nullptr;
    EXPECT_EQ(symWindowCreate(comm, mem, 0, userPtr, 4096, 0, nullptr, &winDev, nullptr, nullptr), ncclSuccess);
    return winDev;
  }
};

// The happy path: the table slot is cleared and the sorted list shrinks.
TEST_F(SymWindowDestroyTest, Succeeds_ClearsTableSlotAndSortedEntry) {
  ncclWindow_vidmem* winDev = MakeWindow(reinterpret_cast<void*>(0x100000));
  ASSERT_EQ(comm->devrState.winSortedCount, 1);
  ASSERT_EQ(comm->devrState.windowTable->entries[0].window, winDev);

  EXPECT_EQ(symWindowDestroy(comm, winDev, nullptr), ncclSuccess);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
  EXPECT_EQ(comm->devrState.windowTable->entries[0].window, nullptr);
  EXPECT_EQ(comm->devrState.memHead, nullptr);  // the memory went with it
}

// Branch: the multi-segment window array is released only when one exists.
TEST_F(SymWindowDestroyTest, MultiSegmentWins_AreFreed) {
  ncclSegmentWindow segWins{};
  ScopedHook segAlloc(g_devrAllocAndPopulateSegmentWindows,
                      [&](ncclDevrState*, ncclDevrMemory*, hipStream_t, ncclSegmentWindow** out) {
                        *out = &segWins;
                        return ncclSuccess;
                      });
  ncclWindow_vidmem* winDev = MakeWindow(reinterpret_cast<void*>(0x100000));

  std::vector<void*> freed;
  ScopedHook poolFree(g_shadowPoolFree, [&](ncclShadowPool*, void* obj, hipStream_t) {
    freed.push_back(obj);
    return ncclSuccess;
  });
  EXPECT_EQ(symWindowDestroy(comm, winDev, nullptr), ncclSuccess);
  EXPECT_NE(std::find(freed.begin(), freed.end(), static_cast<void*>(&segWins)), freed.end());
  EXPECT_NE(std::find(freed.begin(), freed.end(), static_cast<void*>(winDev)), freed.end());
}

// Branch: no segment array, so only the descriptor is released.
TEST_F(SymWindowDestroyTest, NoMultiSegmentWins_FreesOnlyDescriptor) {
  ncclWindow_vidmem* winDev = MakeWindow(reinterpret_cast<void*>(0x100000));

  ScopedHook poolFree(g_shadowPoolFree,
                      [](ncclShadowPool*, void*, hipStream_t) { return ncclSuccess; });
  EXPECT_EQ(symWindowDestroy(comm, winDev, nullptr), ncclSuccess);
  EXPECT_EQ(poolFree.calls, 1);
}

// Branch: the table search walks the chain. The 33rd window lives in the second
// table, so finding it means following next rather than giving up at 32.
TEST_F(SymWindowDestroyTest, WindowInChainedTable_IsFound) {
  ncclWindow_vidmem* last = nullptr;
  for (int i = 0; i < 33; i++) {
    last = MakeWindow(reinterpret_cast<void*>(0x100000 + i * 0x1000));
  }
  ncclDevCommWindowTable* second = comm->devrState.windowTable->next;
  ASSERT_NE(second, nullptr);
  ASSERT_EQ(second->entries[0].window, last);

  EXPECT_EQ(symWindowDestroy(comm, last, nullptr), ncclSuccess);
  EXPECT_EQ(second->entries[0].window, nullptr);
  EXPECT_EQ(comm->devrState.winSortedCount, 32);
}

// Branch: the cleanup label reached from a failure. However teardown went, the
// window must still leave the sorted list -- leaving it there would dangle.
//
// The return code is deliberately not asserted. remove_winSorted ends with
// NCCLCHECKGOTO on the map removal, and that macro assigns unconditionally, so
// a successful removal overwrites the earlier error with ncclSuccess. Asserting
// either value would lock that in. Written up against AICOMRCCL-2180.
TEST_F(SymWindowDestroyTest, TeardownFailure_StillRemovesFromSortedList) {
  ncclWindow_vidmem* winDev = MakeWindow(reinterpret_cast<void*>(0x100000));
  ScopedHook poolFree(g_shadowPoolFree,
                      [](ncclShadowPool*, void*, hipStream_t) { return ncclSystemError; });

  symWindowDestroy(comm, winDev, nullptr);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
  EXPECT_EQ(poolFree.calls, 1);
}


// ---------------------------------------------------------------------------
// windowCloseIpcPeers releases the IPC mappings a non-symmetric window opened
// onto its node-local peers. Our own slot was never opened, and a peer that
// failed to map has a null entry, so both are skipped.

class WindowCloseIpcPeersTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  ncclDevrWindow win{};
  std::vector<void*> allocBase;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->devrState.lsaSelf = 1;  // differs from 0 so "skip self" is observable

    allocBase = {reinterpret_cast<void*>(0xA000), reinterpret_cast<void*>(0xB000),
                 reinterpret_cast<void*>(0xC000)};
    win.ipcPeerPtrsAllocBase = allocBase.data();
    win.ipcPeerCount = 3;
  }
  void TearDown() override { ResetDevRuntimeFakes(); }
};

// Branch: no peer table means IPC was never active for this window.
TEST_F(WindowCloseIpcPeersTest, NoPeerTable_ClosesNothing) {
  win.ipcPeerPtrsAllocBase = nullptr;
  ScopedHook close(g_hipIpcCloseMemHandle, [](void*) { return hipSuccess; });

  windowCloseIpcPeers(comm, &win);
  EXPECT_EQ(close.calls, 0);
}

// Branch: every peer is closed except our own slot, which was never opened.
TEST_F(WindowCloseIpcPeersTest, ClosesPeersButNotSelf) {
  std::vector<void*> closed;
  ScopedHook close(g_hipIpcCloseMemHandle, [&](void* p) {
    closed.push_back(p);
    return hipSuccess;
  });

  windowCloseIpcPeers(comm, &win);
  ASSERT_EQ(closed.size(), 2u);
  EXPECT_EQ(closed[0], allocBase[0]);
  EXPECT_EQ(closed[1], allocBase[2]);  // index 1 is lsaSelf
}

// Branch: a peer that never mapped has a null entry and is skipped.
TEST_F(WindowCloseIpcPeersTest, NullPeerEntry_IsSkipped) {
  allocBase[0] = nullptr;
  std::vector<void*> closed;
  ScopedHook close(g_hipIpcCloseMemHandle, [&](void* p) {
    closed.push_back(p);
    return hipSuccess;
  });

  windowCloseIpcPeers(comm, &win);
  ASSERT_EQ(closed.size(), 1u);
  EXPECT_EQ(closed[0], allocBase[2]);
}

// The close is CUDACHECKIGNORE'd, so one failure must not abandon the rest --
// this runs during teardown, where stopping early would leak the remaining
// mappings.
TEST_F(WindowCloseIpcPeersTest, CloseFails_StillClosesRemainingPeers) {
  ScopedHook close(g_hipIpcCloseMemHandle, [](void*) { return hipErrorInvalidValue; });

  windowCloseIpcPeers(comm, &win);
  EXPECT_EQ(close.calls, 2);
}


// ---------------------------------------------------------------------------
// windowRegisterNonSym registers a window that is not backed by the symmetric
// VMM machinery. Three stages, the first two conditional:
//
//   1. intra-node IPC mapping   when lsaSize > 1
//   2. inter-node RMA MR        when hostRmaSupport and the team is not the comm
//   3. the device-side handle, always
//
// This suite covers stage 3 and the local case where neither stage 1 nor 2
// applies; the IPC and RMA stages follow below.

class WindowRegisterNonSymTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  std::vector<int> lsaRankList;
  std::vector<ncclPeerInfo> peers;
  void* const kUserPtr = reinterpret_cast<void*>(0x100000);

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->rank = 0;
    comm->nRanks = 1;
    comm->bootstrap = reinterpret_cast<void*>(0x1);

    ncclDevrState* devr = &comm->devrState;
    devr->lsaSelf = 0;
    devr->lsaSize = 1;  // local: no IPC stage
    lsaRankList.assign({0});
    devr->lsaRankList = lsaRankList.data();

    peers.assign(4, ncclPeerInfo{});
    comm->peerInfo = peers.data();
  }

  void TearDown() override {
    ncclDevrState* devr = &comm->devrState;
    for (int i = 0; i < devr->winSortedCount; i++) {
      ncclDevrWindow* w = devr->winSorted[i].win;
      free(w->ipcPeerPtrs);
      free(w->ipcPeerPtrsAllocBase);
      free(w);
    }
    free(devr->winSorted);
    devr->winSorted = nullptr;
    devr->winSortedCount = devr->winSortedCapacity = 0;
    devr->lsaRankList = nullptr;  // borrowed, not malloc'd
    ResetDevRuntimeFakes();
  }

  ncclResult_t Register(ncclWindow_t* out, size_t size = 4096) {
    return windowRegisterNonSym(comm, kUserPtr, size, /*winFlags=*/0, /*localRegHandle=*/nullptr, out);
  }
};

// The local case: a single-rank team with no host RMA skips both optional
// stages and still produces a usable window.
TEST_F(WindowRegisterNonSymTest, LocalOnly_RegistersWithoutIpcOrRma) {
  ScopedHook ipcGet(g_hipIpcGetMemHandle, [](hipIpcMemHandle_t*, void*) { return hipSuccess; });
  ScopedHook rma(g_rmaProxyRegister, [](ncclComm*, void*, size_t, void*[]) { return ncclSuccess; });

  ncclWindow_t out = nullptr;
  ASSERT_EQ(Register(&out), ncclSuccess);
  EXPECT_NE(out, nullptr);
  EXPECT_EQ(ipcGet.calls, 0);
  EXPECT_EQ(rma.calls, 0);

  ncclDevrState* devr = &comm->devrState;
  ASSERT_EQ(devr->winSortedCount, 1);
  ncclDevrWindow* win = devr->winSorted[0].win;
  EXPECT_EQ(win->userPtr, kUserPtr);
  EXPECT_EQ(win->size, 4096u);
  EXPECT_EQ(win->memory, nullptr);  // non-sym windows have no backing ncclDevrMemory
  EXPECT_EQ(win->ipcPeerCount, 0);
}

// Stage 3 fills the device-side header, which is how the kernel finds its way
// back to the host window.
TEST_F(WindowRegisterNonSymTest, PopulatesDeviceHeader) {
  comm->rank = 5;
  comm->devrState.lsaSelf = 0;

  ncclWindow_t out = nullptr;
  ASSERT_EQ(Register(&out), ncclSuccess);
  // dev == host under the shadow-pool fake, so the header is readable directly.
  auto* header = reinterpret_cast<ncclWindow_vidmem*>(out);
  EXPECT_EQ(header->worldRank, 5);
  EXPECT_EQ(header->lsaRank, 0);
  EXPECT_EQ(header->winHost, static_cast<void*>(comm->devrState.winSorted[0].win));
}

// Windows are kept in address order regardless of registration order.
TEST_F(WindowRegisterNonSymTest, InsertsIntoSortedListByAddress) {
  ncclWindow_t a = nullptr, b = nullptr;
  ASSERT_EQ(windowRegisterNonSym(comm, reinterpret_cast<void*>(0x300000), 4096, 0, nullptr, &a), ncclSuccess);
  ASSERT_EQ(windowRegisterNonSym(comm, reinterpret_cast<void*>(0x100000), 4096, 0, nullptr, &b), ncclSuccess);

  ncclDevrState* devr = &comm->devrState;
  ASSERT_EQ(devr->winSortedCount, 2);
  EXPECT_EQ(devr->winSorted[0].userAddr, 0x100000u);
  EXPECT_EQ(devr->winSorted[1].userAddr, 0x300000u);
}

// Branch: the shadow-pool allocation fails, so nothing is published and the
// caller's out-pointer is cleared rather than left dangling.
TEST_F(WindowRegisterNonSymTest, ShadowAllocFails_ClearsOutputAndPublishesNothing) {
  ScopedHook alloc(g_shadowPoolAlloc,
                   [](ncclShadowPool*, size_t, void**, void**, hipStream_t) { return ncclSystemError; });

  ncclWindow_t out = reinterpret_cast<ncclWindow_t>(0xdead);
  EXPECT_NE(Register(&out), ncclSuccess);
  EXPECT_EQ(out, nullptr);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
}

// Branch: the stream needed for stage 3 cannot be created.
TEST_F(WindowRegisterNonSymTest, StreamCreateFails_ClearsOutput) {
  ScopedHook create(g_hipStreamCreateWithFlags,
                    [](hipStream_t*, unsigned int) { return hipErrorInvalidValue; });

  ncclWindow_t out = reinterpret_cast<ncclWindow_t>(0xdead);
  EXPECT_NE(Register(&out), ncclSuccess);
  EXPECT_EQ(out, nullptr);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
}


// ---------------------------------------------------------------------------
// windowRegisterNonSym stages 1 and 2. Stage 1 exchanges IPC handles across the
// node-local team and maps each peer, except our own slot and any peer sharing
// our address space. Stage 2 registers an inter-node MR when the team is
// smaller than the communicator.

class WindowRegisterNonSymIpcTest : public WindowRegisterNonSymTest {
protected:
  // Mirrors the ExchangeEntry symMemory's caller all-gathers. Layout must match
  // for the hook to publish values the function then reads back.
  struct ExchangeEntry {
    hipIpcMemHandle_t handle;
    uint64_t hostHash;
    uint64_t pidHash;
    size_t userOffset;
    size_t userSize;
  };

  void SetUp() override {
    WindowRegisterNonSymTest::SetUp();
    comm->nRanks = 4;
    comm->devrState.lsaSize = 3;
    comm->devrState.lsaSelf = 0;
    lsaRankList.assign({0, 1, 2});
    comm->devrState.lsaRankList = lsaRankList.data();
    // Distinct host/pid hashes per rank, so no peer looks like our own process.
    for (int r = 0; r < 4; r++) {
      peers[r].hostHash = 100 + r;
      peers[r].pidHash = 200 + r;
    }
  }

  // Publish the team's exchange entries, as the all-gather would. Each peer
  // reports a distinct process unless told to impersonate ours.
  std::function<ncclResult_t(void*, int*, int, int, void*, int)> GatherPeers(int sameProcRank = -1) {
    return [this, sameProcRank](void*, int*, int self, int size, void* buf, int) {
      auto* e = static_cast<ExchangeEntry*>(buf);
      for (int r = 0; r < size; r++) {
        e[r].hostHash = (r == self || r == sameProcRank) ? peers[0].hostHash : 500 + r;
        e[r].pidHash = (r == self || r == sameProcRank) ? peers[0].pidHash : 600 + r;
        e[r].userOffset = 64 * r;
        e[r].userSize = 4096;
      }
      return ncclSuccess;
    };
  }
};

// Stage 1: peers in other processes are mapped; our own slot reuses the local
// pointer rather than opening a handle against ourselves.
TEST_F(WindowRegisterNonSymIpcTest, MapsPeersAndReusesSelf) {
  ScopedHook gather(g_bootstrapIntraNodeAllGather, GatherPeers());
  ScopedHook open(g_hipIpcOpenMemHandle, [](void** ptr, hipIpcMemHandle_t, unsigned int) {
    *ptr = reinterpret_cast<void*>(0x900000);
    return hipSuccess;
  });

  ncclWindow_t out = nullptr;
  ASSERT_EQ(Register(&out), ncclSuccess);
  EXPECT_EQ(open.calls, 2);  // ranks 1 and 2, not ourselves

  ncclDevrWindow* win = comm->devrState.winSorted[0].win;
  ASSERT_EQ(win->ipcPeerCount, 3);
  EXPECT_EQ(win->ipcPeerPtrs[0], kUserPtr);  // self reuses the caller's pointer
  // Peers are the mapped base advanced by the offset each reported.
  EXPECT_EQ(win->ipcPeerPtrs[1], static_cast<char*>(reinterpret_cast<void*>(0x900000)) + 64);
  EXPECT_EQ(win->ipcPeerPtrs[2], static_cast<char*>(reinterpret_cast<void*>(0x900000)) + 128);
}

// Branch: a peer in our own process is left unmapped -- opening an IPC handle
// against the same address space is not supported here.
TEST_F(WindowRegisterNonSymIpcTest, SameProcessPeer_IsLeftUnmapped) {
  ScopedHook gather(g_bootstrapIntraNodeAllGather, GatherPeers(/*sameProcRank=*/1));
  ScopedHook open(g_hipIpcOpenMemHandle, [](void** ptr, hipIpcMemHandle_t, unsigned int) {
    *ptr = reinterpret_cast<void*>(0x900000);
    return hipSuccess;
  });

  ncclWindow_t out = nullptr;
  ASSERT_EQ(Register(&out), ncclSuccess);
  EXPECT_EQ(open.calls, 1);  // only rank 2

  ncclDevrWindow* win = comm->devrState.winSorted[0].win;
  EXPECT_EQ(win->ipcPeerPtrs[1], nullptr);
  EXPECT_EQ(win->ipcPeerPtrsAllocBase[1], nullptr);
}

// Branch: the address-range lookup fails, so the exchange never happens.
TEST_F(WindowRegisterNonSymIpcTest, AddressRangeFails_ReturnsErrorWithoutGathering) {
  ScopedHook range(g_hipMemGetAddressRange,
                   [](hipDeviceptr_t*, size_t*, hipDeviceptr_t) { return hipErrorInvalidValue; });
  ScopedHook gather(g_bootstrapIntraNodeAllGather, GatherPeers());

  ncclWindow_t out = nullptr;
  EXPECT_NE(Register(&out), ncclSuccess);
  EXPECT_EQ(gather.calls, 0);
  EXPECT_EQ(out, nullptr);
}

// Branch: our own handle cannot be exported.
TEST_F(WindowRegisterNonSymIpcTest, IpcGetHandleFails_ReturnsError) {
  ScopedHook ipcGet(g_hipIpcGetMemHandle,
                    [](hipIpcMemHandle_t*, void*) { return hipErrorInvalidValue; });
  ScopedHook gather(g_bootstrapIntraNodeAllGather, GatherPeers());

  ncclWindow_t out = nullptr;
  EXPECT_NE(Register(&out), ncclSuccess);
  EXPECT_EQ(gather.calls, 0);
}

// Branch: mapping a peer fails. The peers already opened must be closed rather
// than leaked, which is the only observable difference between a clean failure
// and a leaking one.
TEST_F(WindowRegisterNonSymIpcTest, PeerOpenFails_ClosesAlreadyOpenedPeers) {
  ScopedHook gather(g_bootstrapIntraNodeAllGather, GatherPeers());
  int opened = 0;
  ScopedHook open(g_hipIpcOpenMemHandle, [&](void** ptr, hipIpcMemHandle_t, unsigned int) {
    if (++opened == 2) return hipErrorInvalidValue;  // rank 1 maps, rank 2 fails
    *ptr = reinterpret_cast<void*>(0x900000);
    return hipSuccess;
  });
  ScopedHook close(g_hipIpcCloseMemHandle, [](void*) { return hipSuccess; });

  ncclWindow_t out = nullptr;
  EXPECT_NE(Register(&out), ncclSuccess);
  EXPECT_EQ(open.calls, 2);
  EXPECT_EQ(close.calls, 1);  // the one that succeeded
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
}

// Branch: the post-mapping barrier fails, which happens after the window is
// already in the sorted list -- so registration must take it back out.
TEST_F(WindowRegisterNonSymIpcTest, BarrierFails_RevertsSortedInsert) {
  ScopedHook gather(g_bootstrapIntraNodeAllGather, GatherPeers());
  ScopedHook open(g_hipIpcOpenMemHandle, [](void** ptr, hipIpcMemHandle_t, unsigned int) {
    *ptr = reinterpret_cast<void*>(0x900000);
    return hipSuccess;
  });
  ScopedHook barrier(g_bootstrapIntraNodeBarrier,
                     [](void*, int*, int, int, int) { return ncclSystemError; });
  ScopedHook close(g_hipIpcCloseMemHandle, [](void*) { return hipSuccess; });

  ncclWindow_t out = nullptr;
  EXPECT_NE(Register(&out), ncclSuccess);
  EXPECT_EQ(barrier.calls, 1);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);  // insert reverted
}

// Stage 2: a team smaller than the communicator means remote ranks exist, so
// the inter-node MR is registered.
TEST_F(WindowRegisterNonSymIpcTest, HostRmaWithRemoteRanks_RegistersMr) {
  comm->hostRmaSupport = true;  // lsaSize 3 < nRanks 4
  ScopedHook gather(g_bootstrapIntraNodeAllGather, GatherPeers());
  ScopedHook open(g_hipIpcOpenMemHandle, [](void** ptr, hipIpcMemHandle_t, unsigned int) {
    *ptr = reinterpret_cast<void*>(0x900000);
    return hipSuccess;
  });
  ScopedHook connect(g_rmaProxyConnectOnce, [](ncclComm*) { return ncclSuccess; });
  ScopedHook reg(g_rmaProxyRegister, [](ncclComm*, void*, size_t, void*[]) { return ncclSuccess; });

  ncclWindow_t out = nullptr;
  ASSERT_EQ(Register(&out), ncclSuccess);
  EXPECT_EQ(connect.calls, 1);
  EXPECT_EQ(reg.calls, 1);
}

// Branch: the team spans the whole communicator, so there is no remote rank to
// reach and the MR is skipped even with host RMA available.
TEST_F(WindowRegisterNonSymIpcTest, HostRmaButTeamSpansComm_SkipsMr) {
  comm->hostRmaSupport = true;
  comm->nRanks = 3;  // equal to lsaSize
  ScopedHook gather(g_bootstrapIntraNodeAllGather, GatherPeers());
  ScopedHook open(g_hipIpcOpenMemHandle, [](void** ptr, hipIpcMemHandle_t, unsigned int) {
    *ptr = reinterpret_cast<void*>(0x900000);
    return hipSuccess;
  });
  ScopedHook reg(g_rmaProxyRegister, [](ncclComm*, void*, size_t, void*[]) { return ncclSuccess; });

  ncclWindow_t out = nullptr;
  ASSERT_EQ(Register(&out), ncclSuccess);
  EXPECT_EQ(reg.calls, 0);
}

// Branch: the MR registration fails.
TEST_F(WindowRegisterNonSymIpcTest, RmaRegisterFails_ReturnsError) {
  comm->hostRmaSupport = true;
  ScopedHook gather(g_bootstrapIntraNodeAllGather, GatherPeers());
  ScopedHook open(g_hipIpcOpenMemHandle, [](void** ptr, hipIpcMemHandle_t, unsigned int) {
    *ptr = reinterpret_cast<void*>(0x900000);
    return hipSuccess;
  });
  ScopedHook reg(g_rmaProxyRegister,
                 [](ncclComm*, void*, size_t, void*[]) { return ncclSystemError; });
  ScopedHook close(g_hipIpcCloseMemHandle, [](void*) { return hipSuccess; });

  ncclWindow_t out = nullptr;
  EXPECT_NE(Register(&out), ncclSuccess);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
}


// ---------------------------------------------------------------------------
// ncclDevrWindowRegisterInGroup is the entry point for window registration. It
// takes a local registration, then either hands off to the non-symmetric helper
// or walks the symmetric path itself.
//
// This suite covers the dispatch and the failures reachable before the
// symmetric walk begins.

class DevrWindowRegisterInGroupTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  std::vector<int> lsaRankList;
  std::vector<ncclPeerInfo> peers;
  void* const kUserPtr = reinterpret_cast<void*>(0x100000);

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->rank = 0;
    comm->nRanks = 1;
    comm->bootstrap = reinterpret_cast<void*>(0x1);
    comm->symmetricSupport = 0;  // dispatch to the non-symmetric helper

    ncclDevrState* devr = &comm->devrState;
    devr->lsaSelf = 0;
    devr->lsaSize = 1;
    lsaRankList.assign({0});
    devr->lsaRankList = lsaRankList.data();

    peers.assign(1, ncclPeerInfo{});
    comm->peerInfo = peers.data();
  }

  void TearDown() override {
    ncclDevrState* devr = &comm->devrState;
    for (int i = 0; i < devr->winSortedCount; i++) {
      ncclDevrWindow* w = devr->winSorted[i].win;
      free(w->ipcPeerPtrs);
      free(w->ipcPeerPtrsAllocBase);
      free(w);
    }
    free(devr->winSorted);
    devr->winSorted = nullptr;
    devr->winSortedCount = devr->winSortedCapacity = 0;
    devr->lsaRankList = nullptr;  // borrowed, not malloc'd
    ResetDevRuntimeFakes();
  }
};

// Branch: without symmetric support the whole symmetric path is bypassed and
// the non-symmetric helper owns the result -- recognisable because the window
// it builds has no backing ncclDevrMemory.
TEST_F(DevrWindowRegisterInGroupTest, NoSymmetricSupport_RoutesToNonSymHelper) {
  ncclWindow_t out = nullptr;
  ASSERT_EQ(ncclDevrWindowRegisterInGroup(comm, kUserPtr, 4096, 0, &out), ncclSuccess);
  ASSERT_NE(out, nullptr);
  ASSERT_EQ(comm->devrState.winSortedCount, 1);
  EXPECT_EQ(comm->devrState.winSorted[0].win->memory, nullptr);
}

// The local registration handle is threaded through to the helper, which stores
// it on the window for its own teardown to release.
TEST_F(DevrWindowRegisterInGroupTest, PassesLocalRegHandleToHelper) {
  void* const kHandle = reinterpret_cast<void*>(0xABCD);
  ScopedHook reg(g_ncclCommRegister, [&](const ncclComm_t, void*, size_t, void** h) {
    *h = kHandle;
    return ncclSuccess;
  });

  ncclWindow_t out = nullptr;
  ASSERT_EQ(ncclDevrWindowRegisterInGroup(comm, kUserPtr, 4096, 0, &out), ncclSuccess);
  EXPECT_EQ(comm->devrState.winSorted[0].win->localRegHandle, kHandle);
}

// Branch: the local registration fails, so nothing downstream runs and there is
// no handle to release.
TEST_F(DevrWindowRegisterInGroupTest, CommRegisterFails_ReturnsErrorWithoutDeregistering) {
  ScopedHook reg(g_ncclCommRegister,
                 [](const ncclComm_t, void*, size_t, void**) { return ncclSystemError; });
  ScopedHook dereg(g_ncclCommDeregister, [](const ncclComm_t, void*) { return ncclSuccess; });

  ncclWindow_t out = reinterpret_cast<ncclWindow_t>(0xdead);
  EXPECT_NE(ncclDevrWindowRegisterInGroup(comm, kUserPtr, 4096, 0, &out), ncclSuccess);
  EXPECT_EQ(dereg.calls, 0);
  EXPECT_EQ(out, nullptr);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
}

// Branch: the helper fails after the local registration succeeded, so that
// registration must be released -- the fail_locReg label exists for exactly
// this, and nothing else reports it.
TEST_F(DevrWindowRegisterInGroupTest, NonSymHelperFails_ReleasesLocalRegistration) {
  ScopedHook alloc(g_shadowPoolAlloc,
                   [](ncclShadowPool*, size_t, void**, void**, hipStream_t) { return ncclSystemError; });
  ScopedHook dereg(g_ncclCommDeregister, [](const ncclComm_t, void*) { return ncclSuccess; });

  ncclWindow_t out = nullptr;
  EXPECT_NE(ncclDevrWindowRegisterInGroup(comm, kUserPtr, 4096, 0, &out), ncclSuccess);
  EXPECT_EQ(dereg.calls, 1);
  EXPECT_EQ(out, nullptr);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
}


// ---------------------------------------------------------------------------
// ncclDevrWindowRegisterInGroup, symmetric path. Resolve the allocation the
// user pointer sits in, validate the layout, retain a handle per physical
// segment, then hand the handles to symMemoryObtain and build the window.
//
// The walk over segments is ncclCuMemGetAddressRange (a static inline in
// alloc.h, so only drivable through the cuMemGetAddressRange seam). It advances
// by the size each query reports, so a zero size -- the seam's default -- spins
// forever. AddressRangeOf() supplies a real one.

class DevrWindowRegisterInGroupSymTest : public DevrWindowRegisterInGroupTest {
protected:
  void SetUp() override {
#if ROCM_VERSION < 70000
    // Below 7.0 alloc.h compiles ncclCuMemGetAddressRange as a WARN plus
    // `return ncclInternalError` (alloc.h:805-810) that never consults the
    // g_hipMemGetAddressRange seam, so every test here would fail on the first
    // call rather than exercising anything. The binary's own floor is 6.4, so
    // skip rather than fail on a 6.4-6.9 build.
    GTEST_SKIP() << "ncclCuMemGetAddressRange is a stub below ROCm 7.0";
#endif
    DevrWindowRegisterInGroupTest::SetUp();
    comm->symmetricSupport = 1;
    comm->devrState.bigSize = size_t(1) << 32;
    comm->devrState.granularity = 4096;
    comm->devrState.lsaFlatBase = reinterpret_cast<void*>(0x40000000);
  }

  // Report the queried pointer as its own allocation base, with segments of
  // `segSize`. The walk terminates after userSize / segSize iterations.
  std::function<hipError_t(hipDeviceptr_t*, size_t*, hipDeviceptr_t)> AddressRangeOf(size_t segSize) {
    return [segSize](hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
      if (pbase) *pbase = dptr;
      if (psize) *psize = segSize;
      return hipSuccess;
    };
  }

  // Report every segment as a given location type.
  std::function<hipError_t(hipMemAllocationProp*, hipMemGenericAllocationHandle_t)> SegmentsOfType(
      hipMemLocationType type) {
    return [type](hipMemAllocationProp* prop, hipMemGenericAllocationHandle_t) {
      if (prop) {
        *prop = hipMemAllocationProp{};
        prop->location.type = type;
      }
      return hipSuccess;
    };
  }
};

// The happy path: one device-backed segment, registered end to end.
TEST_F(DevrWindowRegisterInGroupSymTest, SingleDeviceSegment_RegistersWindow) {
  ScopedHook range(g_hipMemGetAddressRange, AddressRangeOf(4096));

  ncclWindow_t out = nullptr;
  ASSERT_EQ(ncclDevrWindowRegisterInGroup(comm, kUserPtr, 4096, 0, &out), ncclSuccess);
  EXPECT_NE(out, nullptr);
  ASSERT_EQ(comm->devrState.winSortedCount, 1);
  EXPECT_NE(comm->devrState.winSorted[0].win->memory, nullptr);  // symmetric: has backing memory
}

// Branch: the symmetric-collective flag defers kernel init until a window that
// needs it exists, so it is only paid for on request.
TEST_F(DevrWindowRegisterInGroupSymTest, CollSymmetricFlag_InitialisesSymKernels) {
  ScopedHook range(g_hipMemGetAddressRange, AddressRangeOf(4096));

  ncclWindow_t out = nullptr;
  ASSERT_EQ(ncclDevrWindowRegisterInGroup(comm, kUserPtr, 4096, NCCL_WIN_COLL_SYMMETRIC, &out), ncclSuccess);
  EXPECT_EQ(comm->devrState.winSorted[0].win->winFlags, NCCL_WIN_COLL_SYMMETRIC);
}

// Branch: resolving the allocation fails, so nothing is registered.
TEST_F(DevrWindowRegisterInGroupSymTest, AddressRangeFails_ReleasesLocalRegistration) {
  ScopedHook range(g_hipMemGetAddressRange,
                   [](hipDeviceptr_t*, size_t*, hipDeviceptr_t) { return hipErrorInvalidValue; });
  ScopedHook dereg(g_ncclCommDeregister, [](const ncclComm_t, void*) { return ncclSuccess; });

  ncclWindow_t out = nullptr;
  EXPECT_NE(ncclDevrWindowRegisterInGroup(comm, kUserPtr, 4096, 0, &out), ncclSuccess);
  EXPECT_EQ(dereg.calls, 1);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
}

// Branch: a window not aligned to NCCL_WIN_REQUIRED_ALIGNMENT within its
// allocation is rejected. The base is reported one byte below the user pointer,
// which is the smallest offset that cannot satisfy the requirement.
//
// Unlike every other rejection in this function, this one is *not* asserted to
// deregister: dev_runtime.cc:1336 jumps to fail, not fail_locReg, so
// ncclCommDeregister never runs and the local registration taken at :1284 leaks
// (as does the memHandles array allocated at :1304). EXPECT_EQ(dereg.calls, 1)
// belongs here and fails today; see AICOMRCCL-2180 finding 13. Asserting
// the current count instead would pin the leak.
TEST_F(DevrWindowRegisterInGroupSymTest, MisalignedWindow_ReturnsInvalidArgument) {
  ScopedHook range(g_hipMemGetAddressRange, [](hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
    if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(static_cast<char*>(dptr) - 1);
    if (psize) *psize = 8192;
    return hipSuccess;
  });

  ncclWindow_t out = nullptr;
  EXPECT_EQ(ncclDevrWindowRegisterInGroup(comm, kUserPtr, 4096, 0, &out), ncclInvalidArgument);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
}

// Branch: CPU-backed segments need the elastic-buffer param, and are rejected
// with a specific code when it is off rather than failing later.
TEST_F(DevrWindowRegisterInGroupSymTest, SysmemSegmentWithoutElasticParam_ReturnsInvalidArgument) {
  ScopedHook range(g_hipMemGetAddressRange, AddressRangeOf(4096));
  ScopedHook props(g_hipMemGetAllocationPropertiesFromHandle, SegmentsOfType(hipMemLocationTypeHost));
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t deftVal) -> int64_t {
    return std::string(env) == "ELASTIC_BUFFER_REGISTER" ? 0 : deftVal;
  });
  ScopedHook dereg(g_ncclCommDeregister, [](const ncclComm_t, void*) { return ncclSuccess; });

  ncclWindow_t out = nullptr;
  EXPECT_EQ(ncclDevrWindowRegisterInGroup(comm, kUserPtr, 4096, 0, &out), ncclInvalidArgument);
  EXPECT_EQ(dereg.calls, 1);  // this arm does unwind, unlike the misaligned one above
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
}

// Branch: with the param on, the same CPU-backed layout is accepted.
TEST_F(DevrWindowRegisterInGroupSymTest, SysmemSegmentWithElasticParam_Registers) {
  ScopedHook range(g_hipMemGetAddressRange, AddressRangeOf(4096));
  ScopedHook props(g_hipMemGetAllocationPropertiesFromHandle, SegmentsOfType(hipMemLocationTypeHost));

  ncclWindow_t out = nullptr;
  ASSERT_EQ(ncclDevrWindowRegisterInGroup(comm, kUserPtr, 4096, 0, &out), ncclSuccess);
  EXPECT_EQ(comm->devrState.winSortedCount, 1);
}

// Branch: a segment that is neither host nor device is rejected -- symmetric
// memory has no mapping strategy for anything else.
TEST_F(DevrWindowRegisterInGroupSymTest, UnsupportedSegmentType_ReturnsInvalidArgument) {
  ScopedHook range(g_hipMemGetAddressRange, AddressRangeOf(4096));
  ScopedHook props(g_hipMemGetAllocationPropertiesFromHandle, SegmentsOfType(hipMemLocationTypeInvalid));
  ScopedHook dereg(g_ncclCommDeregister, [](const ncclComm_t, void*) { return ncclSuccess; });

  ncclWindow_t out = nullptr;
  EXPECT_EQ(ncclDevrWindowRegisterInGroup(comm, kUserPtr, 4096, 0, &out), ncclInvalidArgument);
  EXPECT_EQ(dereg.calls, 1);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
}

// Branch: retaining a segment handle fails part-way through the validation
// loop, after an earlier segment's handle was already retained into
// memHandles[].
//
// Retain budget: ncclCuMemGetAddressRange's walk (alloc.h) retains and releases
// once per iteration, and userSize=8192 over 4096-byte segments gives it two
// iterations, so retains 1 and 2 are the walk's and are already released by the
// time it returns. numSegments is then 2, so the validation loop
// (dev_runtime.cc:1339) retains again per segment: retain 3 lands in
// memHandles[0], and retain 4 is the one made to fail.
//
// Deliberately NOT asserting that memHandles[0] is released: it is not.
// dev_runtime.cc:1344 jumps to fail_locReg, which is *below*
// fail_locReg_memHandle's release loop and free(memHandles), so both are
// skipped -- the retained handle and the array leak. Same for the two other
// exits in this loop (:1346, :1354). Asserting the release count here would
// pin that leak in place; see AICOMRCCL-2180 finding 12. What is asserted
// is the part that does work: the error propagates and the local registration
// is unwound.
TEST_F(DevrWindowRegisterInGroupSymTest, RetainFails_PropagatesErrorAndUnwindsLocalRegistration) {
  ScopedHook range(g_hipMemGetAddressRange, AddressRangeOf(4096));
  int retained = 0;
  ScopedHook retain(g_hipMemRetainAllocationHandle, [&](hipMemGenericAllocationHandle_t* h, void*) {
    if (++retained > 3) return hipErrorInvalidValue;
    if (h) *h = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x77);
    return hipSuccess;
  });
  ScopedHook release(g_hipMemRelease, [](hipMemGenericAllocationHandle_t) { return hipSuccess; });
  ScopedHook dereg(g_ncclCommDeregister, [](const ncclComm_t, void*) { return ncclSuccess; });

  ncclWindow_t out = nullptr;
  EXPECT_NE(ncclDevrWindowRegisterInGroup(comm, kUserPtr, 8192, 0, &out), ncclSuccess);
  EXPECT_EQ(retained, 4);  // proves the failure landed in the validation loop, not the walk
  EXPECT_EQ(dereg.calls, 1);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
}


// ---------------------------------------------------------------------------
// windowDeregisterNonSym undoes windowRegisterNonSym stage by stage: the RMA
// MR, the IPC peer mappings, the shadow-pool entry, then the local
// registration. The window must still be in winSorted, which is how a repeated
// deregister of an already-freed handle is caught.

class WindowDeregisterNonSymTest : public WindowRegisterNonSymTest {
protected:
  // Register a window through the real path, so the state being torn down is
  // exactly what the register path produces.
  ncclWindow_t RegisterOne(void* userPtr) {
    ncclWindow_t out = nullptr;
    EXPECT_EQ(windowRegisterNonSym(comm, userPtr, 4096, 0, nullptr, &out), ncclSuccess);
    return out;
  }
};

// The happy path: the window leaves the sorted list and its local registration
// is released.
TEST_F(WindowDeregisterNonSymTest, Succeeds_RemovesWindowAndReleasesRegistration) {
  ncclWindow_t winDev = RegisterOne(kUserPtr);
  ASSERT_EQ(comm->devrState.winSortedCount, 1);
  ScopedHook dereg(g_ncclCommDeregister, [](const ncclComm_t, void*) { return ncclSuccess; });

  EXPECT_EQ(windowDeregisterNonSym(comm, winDev), ncclSuccess);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
  EXPECT_EQ(dereg.calls, 1);
}

// Branch: a window that is not in winSorted -- a second deregister of the same
// handle -- is reported rather than tearing down freed state again.
TEST_F(WindowDeregisterNonSymTest, AlreadyDeregistered_ReturnsInvalidArgument) {
  ncclWindow_t winDev = RegisterOne(kUserPtr);
  ASSERT_EQ(windowDeregisterNonSym(comm, winDev), ncclSuccess);

  EXPECT_EQ(windowDeregisterNonSym(comm, winDev), ncclInvalidArgument);
}

// Branch: only the window asked for is removed; its neighbours stay registered.
TEST_F(WindowDeregisterNonSymTest, RemovesOnlyTheNamedWindow) {
  ncclWindow_t first = RegisterOne(reinterpret_cast<void*>(0x100000));
  RegisterOne(reinterpret_cast<void*>(0x200000));
  ASSERT_EQ(comm->devrState.winSortedCount, 2);

  ASSERT_EQ(windowDeregisterNonSym(comm, first), ncclSuccess);
  ASSERT_EQ(comm->devrState.winSortedCount, 1);
  EXPECT_EQ(comm->devrState.winSorted[0].userAddr, 0x200000u);
}

// Branch: an RMA MR was taken, so it is released before the rest. rmaHostWins[0]
// is the witness the function uses, so setting it is what selects this arm.
TEST_F(WindowDeregisterNonSymTest, RmaRegistered_DeregistersMr) {
  ncclWindow_t winDev = RegisterOne(kUserPtr);
  comm->devrState.winSorted[0].win->rmaHostWins[0] = reinterpret_cast<void*>(0x1);
  ScopedHook rmaDereg(g_rmaProxyDeregister, [](ncclComm*, void*[]) { return ncclSuccess; });

  EXPECT_EQ(windowDeregisterNonSym(comm, winDev), ncclSuccess);
  EXPECT_EQ(rmaDereg.calls, 1);
}

// Branch: no MR was taken, so nothing is released.
TEST_F(WindowDeregisterNonSymTest, NoRmaRegistration_SkipsMrDeregister) {
  ncclWindow_t winDev = RegisterOne(kUserPtr);
  ScopedHook rmaDereg(g_rmaProxyDeregister, [](ncclComm*, void*[]) { return ncclSuccess; });

  EXPECT_EQ(windowDeregisterNonSym(comm, winDev), ncclSuccess);
  EXPECT_EQ(rmaDereg.calls, 0);
}

// Branch: the shadow-pool release fails, so the window stays registered rather
// than being dropped from the list with its device entry still live.
TEST_F(WindowDeregisterNonSymTest, ShadowFreeFails_LeavesWindowRegistered) {
  ncclWindow_t winDev = RegisterOne(kUserPtr);
  ScopedHook poolFree(g_shadowPoolFree,
                      [](ncclShadowPool*, void*, hipStream_t) { return ncclSystemError; });

  EXPECT_NE(windowDeregisterNonSym(comm, winDev), ncclSuccess);
  EXPECT_EQ(comm->devrState.winSortedCount, 1);
}


// ---------------------------------------------------------------------------
// ncclDevrFindWindow resolves a user address to the window covering it, using
// the address-sorted list. The least upper bound is one past ours, so the
// candidate is the entry before it -- and only if the address falls inside it.

class DevrFindWindowTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  std::vector<ncclDevrWindowSorted> sorted;
  ncclDevrWindow winA{}, winB{};

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    // Two windows with a gap between them, so an address can fall outside both.
    sorted = {{0x100000, 4096, &winA}, {0x200000, 4096, &winB}};
    comm->devrState.winSorted = sorted.data();
    comm->devrState.winSortedCount = 2;
  }
  void TearDown() override { comm->devrState.winSorted = nullptr; }  // borrowed
};

// An address inside a window resolves to it.
TEST_F(DevrFindWindowTest, AddressInsideWindow_ReturnsIt) {
  ncclDevrWindow* out = nullptr;
  ASSERT_EQ(ncclDevrFindWindow(comm, reinterpret_cast<void*>(0x100010), &out), ncclSuccess);
  EXPECT_EQ(out, &winA);
}

// The first byte is inside.
TEST_F(DevrFindWindowTest, AddressAtWindowBase_ReturnsIt) {
  ncclDevrWindow* out = nullptr;
  ASSERT_EQ(ncclDevrFindWindow(comm, reinterpret_cast<void*>(0x100000), &out), ncclSuccess);
  EXPECT_EQ(out, &winA);
}

// Boundary: one past the end belongs to no window, even though it is the
// nearest one below.
TEST_F(DevrFindWindowTest, AddressJustPastWindowEnd_ReturnsNull) {
  ncclDevrWindow* out = &winB;
  ASSERT_EQ(ncclDevrFindWindow(comm, reinterpret_cast<void*>(0x100000 + 4096), &out), ncclSuccess);
  EXPECT_EQ(out, nullptr);
}

// Branch: an address below every window has no candidate at all.
TEST_F(DevrFindWindowTest, AddressBelowAllWindows_ReturnsNull) {
  ncclDevrWindow* out = &winA;
  ASSERT_EQ(ncclDevrFindWindow(comm, reinterpret_cast<void*>(0x1000), &out), ncclSuccess);
  EXPECT_EQ(out, nullptr);
}

// The second window is found through the same path, so the lookup is not
// hard-wired to the first entry.
TEST_F(DevrFindWindowTest, AddressInSecondWindow_ReturnsIt) {
  ncclDevrWindow* out = nullptr;
  ASSERT_EQ(ncclDevrFindWindow(comm, reinterpret_cast<void*>(0x200100), &out), ncclSuccess);
  EXPECT_EQ(out, &winB);
}


// ---------------------------------------------------------------------------
// ncclDevrGetRmaWin returns a window's RMA handle for one context. Symmetric
// windows keep them on the backing memory, non-symmetric ones on the window
// itself -- memory == nullptr is what tells them apart.

TEST(DevrGetRmaWin, NullWindow_ReturnsNull) {
  EXPECT_EQ(ncclDevrGetRmaWin(nullptr, 0), nullptr);
}

// Boundary: a negative context index.
TEST(DevrGetRmaWin, NegativeContext_ReturnsNull) {
  ncclDevrWindow win{};
  EXPECT_EQ(ncclDevrGetRmaWin(&win, -1), nullptr);
}

// Boundary: one past the last context.
TEST(DevrGetRmaWin, ContextPastEnd_ReturnsNull) {
  ncclDevrWindow win{};
  EXPECT_EQ(ncclDevrGetRmaWin(&win, NCCL_GIN_MAX_CONNECTIONS), nullptr);
}

// Branch: non-symmetric windows hold their own handles.
TEST(DevrGetRmaWin, NonSymmetricWindow_ReadsFromWindow) {
  ncclDevrWindow win{};
  win.memory = nullptr;
  win.rmaDevWins[1] = reinterpret_cast<ncclGinWindow_t>(0x1234);
  EXPECT_EQ(ncclDevrGetRmaWin(&win, 1), reinterpret_cast<void*>(0x1234));
}

// Branch: symmetric windows go through the backing memory. Both slots are set
// to different values so reading the wrong one is visible.
TEST(DevrGetRmaWin, SymmetricWindow_ReadsFromMemory) {
  ncclDevrMemory mem{};
  ncclDevrWindow win{};
  win.memory = &mem;
  win.rmaDevWins[1] = reinterpret_cast<ncclGinWindow_t>(0x1111);
  mem.rmaDevWins[1] = reinterpret_cast<ncclGinWindow_t>(0x2222);
  EXPECT_EQ(ncclDevrGetRmaWin(&win, 1), reinterpret_cast<void*>(0x2222));
}


// ---------------------------------------------------------------------------
// ncclDevrGetLsaSelfAddr maps an address into this rank's slot of the LSA flat
// space: addresses already in that range pass through, others are looked up
// against the registered memories.

class DevrGetLsaSelfAddrTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclDevrState* devr = nullptr;
  ncclDevrMemory mem{};

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    devr = &commStorage->devrState;
    devr->lsaSelf = 1;
    devr->lsaSize = 2;
    devr->bigSize = 0x10000;
    devr->lsaFlatBase = reinterpret_cast<void*>(0x400000);

    mem.primaryAddr = reinterpret_cast<void*>(0x900000);
    mem.size = 4096;
    mem.bigOffset = 0x100;
    devr->memHead = &mem;
  }
  void TearDown() override { devr->memHead = nullptr; }  // stack object
};

// Branch: an address already inside the flat range is returned unchanged.
TEST_F(DevrGetLsaSelfAddrTest, AddressInFlatRange_PassesThrough) {
  void* addr = reinterpret_cast<void*>(0x405000);
  void* out = nullptr;
  ASSERT_EQ(ncclDevrGetLsaSelfAddr(devr, addr, &out), ncclSuccess);
  EXPECT_EQ(out, addr);
}

// Branch: an address inside a registered memory is translated into our own slot
// of the flat space, preserving the offset within that memory.
TEST_F(DevrGetLsaSelfAddrTest, AddressInRegisteredMemory_TranslatesToOwnSlot) {
  void* out = nullptr;
  ASSERT_EQ(ncclDevrGetLsaSelfAddr(devr, reinterpret_cast<void*>(0x900040), &out), ncclSuccess);
  EXPECT_EQ(out, static_cast<char*>(devr->lsaFlatBase) + devr->lsaSelf * devr->bigSize + mem.bigOffset + 0x40);
}

// Branch: an address in neither the flat range nor any memory is "not found",
// reported as success with a null result -- the same convention
// ncclDevrFindWindow uses, so callers must check the pointer, not the code.
TEST_F(DevrGetLsaSelfAddrTest, UnknownAddress_ReturnsNullNotError) {
  void* out = reinterpret_cast<void*>(0xdead);
  EXPECT_EQ(ncclDevrGetLsaSelfAddr(devr, reinterpret_cast<void*>(0x50), &out), ncclSuccess);
  EXPECT_EQ(out, nullptr);
}

// Boundary: one past a memory's end is outside it.
TEST_F(DevrGetLsaSelfAddrTest, AddressJustPastMemory_ReturnsNull) {
  void* out = reinterpret_cast<void*>(0xdead);
  EXPECT_EQ(ncclDevrGetLsaSelfAddr(devr, reinterpret_cast<void*>(0x900000 + 4096), &out), ncclSuccess);
  EXPECT_EQ(out, nullptr);
}


// ---------------------------------------------------------------------------
// ncclDevrWorldToLsaRank maps a world rank to its index within the LSA team.
// Without symmetric support the team is the explicit lsaRankList.

class DevrWorldToLsaRankTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  std::vector<int> lsaRankList;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->symmetricSupport = 0;
    lsaRankList.assign({4, 5, 6});
    comm->devrState.lsaSize = 3;
    comm->devrState.lsaRankList = lsaRankList.data();
  }
  void TearDown() override { comm->devrState.lsaRankList = nullptr; }  // borrowed
};

// A member's index is its position in the list, not its world rank.
TEST_F(DevrWorldToLsaRankTest, MemberRank_ReturnsItsIndex) {
  int lsaRank = -1;
  ASSERT_EQ(ncclDevrWorldToLsaRank(comm, 6, &lsaRank), ncclSuccess);
  EXPECT_EQ(lsaRank, 2);
}

// The first entry, to show the search is not off by one.
TEST_F(DevrWorldToLsaRankTest, FirstMember_ReturnsZero) {
  int lsaRank = -1;
  ASSERT_EQ(ncclDevrWorldToLsaRank(comm, 4, &lsaRank), ncclSuccess);
  EXPECT_EQ(lsaRank, 0);
}

// Branch: a rank outside the team is an error, not a silent miss.
TEST_F(DevrWorldToLsaRankTest, NonMemberRank_ReturnsError) {
  int lsaRank = -1;
  EXPECT_NE(ncclDevrWorldToLsaRank(comm, 9, &lsaRank), ncclSuccess);
}


// ---------------------------------------------------------------------------
// ncclWinGetUserPtr recovers the user pointer a window was registered with,
// decoding the host record from the device handle.

class WinGetUserPtrTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  ncclWindow_vidmem vidmem{};
  ncclDevrWindow win{};

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->symmetricSupport = 1;
    win.userPtr = reinterpret_cast<void*>(0x123000);
    vidmem.winHost = &win;
    // vidmem is a stack object, not a shadow-pool allocation, so resolve it
    // explicitly rather than relying on the pool fake to recognise it.
    g_shadowPoolToHost = [this](ncclShadowPool*, void* dev, void** host) {
      if (dev != &vidmem) return ncclInvalidArgument;
      *host = &vidmem;
      return ncclSuccess;
    };
  }
  void TearDown() override { ResetDevRuntimeFakes(); }
};

TEST_F(WinGetUserPtrTest, Succeeds_ReturnsRegisteredPointer) {
  void* out = nullptr;
  ASSERT_EQ(ncclWinGetUserPtr(comm, &vidmem, &out), ncclSuccess);
  EXPECT_EQ(out, win.userPtr);
}

// Branch: without symmetric support there is no symmetric registration to
// report, which is a null result rather than an error.
TEST_F(WinGetUserPtrTest, NoSymmetricSupport_ReturnsNullNotError) {
  comm->symmetricSupport = 0;
  void* out = reinterpret_cast<void*>(0xdead);
  EXPECT_EQ(ncclWinGetUserPtr(comm, &vidmem, &out), ncclSuccess);
  EXPECT_EQ(out, nullptr);
}

// Branch: a device handle whose host record is missing is an internal error --
// distinct from the unsupported case above, which is benign.
TEST_F(WinGetUserPtrTest, MissingHostRecord_ReturnsInternalError) {
  vidmem.winHost = nullptr;
  void* out = nullptr;
  EXPECT_EQ(ncclWinGetUserPtr(comm, &vidmem, &out), ncclInternalError);
}

// Branch: the shadow-pool decode fails.
TEST_F(WinGetUserPtrTest, ShadowDecodeFails_ReturnsError) {
  ScopedHook toHost(g_shadowPoolToHost,
                    [](ncclShadowPool*, void*, void**) { return ncclSystemError; });
  void* out = nullptr;
  EXPECT_NE(ncclWinGetUserPtr(comm, &vidmem, &out), ncclSuccess);
}


// ---------------------------------------------------------------------------
// ncclDevrGetLsaRankPtr resolves an offset within a window to the address it
// occupies on one LSA rank. Three routes: a self-targeted op on a non-symmetric
// window, an IPC-backed window's per-peer table, and otherwise flat-VA
// arithmetic over the symmetric space.

class DevrGetLsaRankPtrTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  ncclDevrWindow win{};
  std::vector<void*> peerPtrs;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->symmetricSupport = 1;

    ncclDevrState* devr = &comm->devrState;
    devr->lsaSelf = 1;
    devr->lsaSize = 3;
    devr->bigSize = 0x10000;
    devr->lsaFlatBase = reinterpret_cast<void*>(0x400000);

    win.userPtr = reinterpret_cast<void*>(0x900000);
    win.size = 4096;
    win.bigOffset = 0x200;
  }
};

// The symmetric route: flat base, the target rank's slot, the window's offset,
// then the caller's. Every term is distinct so a dropped one shows.
TEST_F(DevrGetLsaRankPtrTest, SymmetricWindow_UsesFlatVaArithmetic) {
  void* out = nullptr;
  ASSERT_EQ(ncclDevrGetLsaRankPtr(comm, &win, /*offset=*/0x40, /*lsaRank=*/2, &out), ncclSuccess);
  ncclDevrState* devr = &comm->devrState;
  EXPECT_EQ(out, reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(devr->lsaFlatBase) + 2 * devr->bigSize +
                                         win.bigOffset + 0x40));
}

// Boundary: an offset at the window's size is past its end.
TEST_F(DevrGetLsaRankPtrTest, OffsetAtWindowSize_ReturnsInvalidArgument) {
  void* out = nullptr;
  EXPECT_EQ(ncclDevrGetLsaRankPtr(comm, &win, win.size, 0, &out), ncclInvalidArgument);
}

// Boundary: a rank outside the team.
TEST_F(DevrGetLsaRankPtrTest, RankPastTeam_ReturnsInvalidArgument) {
  void* out = nullptr;
  EXPECT_EQ(ncclDevrGetLsaRankPtr(comm, &win, 0, comm->devrState.lsaSize, &out), ncclInvalidArgument);
}

// Branch: a non-symmetric window targeting ourselves resolves against the local
// window base, not the flat space -- which does not apply to it.
TEST_F(DevrGetLsaRankPtrTest, NonSymmetricSelfTarget_UsesLocalBase) {
  comm->symmetricSupport = 0;
  void* out = nullptr;
  ASSERT_EQ(ncclDevrGetLsaRankPtr(comm, &win, 0x40, comm->devrState.lsaSelf, &out), ncclSuccess);
  EXPECT_EQ(out, static_cast<char*>(win.userPtr) + 0x40);
}

// Branch: an IPC-backed window uses its per-peer table, because IPC mappings
// land wherever the driver puts them rather than at a computable address.
TEST_F(DevrGetLsaRankPtrTest, IpcWindow_UsesPeerTable) {
  comm->symmetricSupport = 0;
  peerPtrs = {reinterpret_cast<void*>(0xA000), nullptr, reinterpret_cast<void*>(0xC000)};
  win.ipcPeerPtrs = peerPtrs.data();
  win.ipcPeerCount = 3;

  void* out = nullptr;
  ASSERT_EQ(ncclDevrGetLsaRankPtr(comm, &win, 0x40, 2, &out), ncclSuccess);
  EXPECT_EQ(out, static_cast<char*>(peerPtrs[2]) + 0x40);
}

// Branch: a peer that was never mapped -- a same-process cross-thread peer --
// is an internal error rather than a null dereference downstream.
TEST_F(DevrGetLsaRankPtrTest, IpcWindowUnmappedPeer_ReturnsInternalError) {
  comm->symmetricSupport = 0;
  comm->devrState.lsaSelf = 0;  // the unmapped rank must not be us, or the
                                // self-target branch answers first
  peerPtrs = {reinterpret_cast<void*>(0xA000), nullptr, reinterpret_cast<void*>(0xC000)};
  win.ipcPeerPtrs = peerPtrs.data();
  win.ipcPeerCount = 3;

  void* out = nullptr;
  EXPECT_EQ(ncclDevrGetLsaRankPtr(comm, &win, 0, 1, &out), ncclInternalError);
}

// Boundary: a rank outside the IPC table.
TEST_F(DevrGetLsaRankPtrTest, IpcWindowRankPastTable_ReturnsInvalidArgument) {
  comm->symmetricSupport = 0;
  peerPtrs = {reinterpret_cast<void*>(0xA000)};
  win.ipcPeerPtrs = peerPtrs.data();
  win.ipcPeerCount = 1;

  void* out = nullptr;
  EXPECT_EQ(ncclDevrGetLsaRankPtr(comm, &win, 0, 5, &out), ncclInvalidArgument);
}


// ---------------------------------------------------------------------------
// The public device-pointer accessors. All four resolve the device window to
// its host record through the global window map first, so a window the map does
// not know is rejected before anything else is read.

class DevicePointerAccessorTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  ncclDevrWindow win{};
  ncclWindow_vidmem vidmem{};

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->nRanks = 4;
    comm->symmetricSupport = 1;

    ncclDevrState* devr = &comm->devrState;
    devr->lsaSelf = 0;
    devr->lsaSize = 2;
    devr->bigSize = 0x10000;
    devr->lsaFlatBase = reinterpret_cast<void*>(0x400000);

    win.comm = comm;
    win.userPtr = reinterpret_cast<void*>(0x900000);
    win.size = 4096;
    win.bigOffset = 0x200;

    // The map resolves our device handle to this window; anything else misses.
    g_intruAddressMapFind = [this](ncclIntruAddressMap_untyped*, int, int, int, uintptr_t key, void** out) {
      *out = (key == reinterpret_cast<uintptr_t>(&vidmem)) ? &win : nullptr;
      return ncclSuccess;
    };
  }
  void TearDown() override { ResetDevRuntimeFakes(); }

  ncclWindow_t Handle() { return &vidmem; }
};

// A window the map does not know cannot be resolved, whichever accessor asks.
TEST_F(DevicePointerAccessorTest, UnknownWindow_ReturnsInvalidArgument) {
  ncclWindow_vidmem stranger{};
  void* out = nullptr;
  EXPECT_EQ(ncclGetLsaDevicePointer(&stranger, 0, 0, &out), ncclInvalidArgument);
  EXPECT_EQ(ncclGetPeerDevicePointer(&stranger, 0, 0, &out), ncclInvalidArgument);
}

// The LSA accessor delegates to the same arithmetic ncclDevrGetLsaRankPtr uses.
TEST_F(DevicePointerAccessorTest, LsaPointer_ResolvesWithinFlatSpace) {
  void* out = nullptr;
  ASSERT_EQ(ncclGetLsaDevicePointer(Handle(), 0x40, 1, &out), ncclSuccess);
  ncclDevrState* devr = &comm->devrState;
  EXPECT_EQ(out, reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(devr->lsaFlatBase) + 1 * devr->bigSize +
                                         win.bigOffset + 0x40));
}

// Boundary: a rank at lsaSize is outside the team. This check is the accessor's
// own, ahead of the one inside ncclDevrGetLsaRankPtr, so it can name lsaSize in
// the diagnostic.
TEST_F(DevicePointerAccessorTest, LsaPointerRankPastTeam_ReturnsInvalidArgument) {
  void* out = nullptr;
  EXPECT_EQ(ncclGetLsaDevicePointer(Handle(), 0, comm->devrState.lsaSize, &out), ncclInvalidArgument);
}

// Boundary: the peer accessor validates against the communicator, not the team,
// because it takes a world rank.
TEST_F(DevicePointerAccessorTest, PeerPointerRankPastComm_ReturnsInvalidArgument) {
  void* out = nullptr;
  EXPECT_EQ(ncclGetPeerDevicePointer(Handle(), 0, comm->nRanks, &out), ncclInvalidArgument);
}

// Branch: multimem needs a base to offset from.
TEST_F(DevicePointerAccessorTest, MultimemWithoutBase_ReturnsInvalidArgument) {
  ncclMultimemHandle mm{};
  mm.mcBasePtr = nullptr;
  void* out = nullptr;
  EXPECT_EQ(ncclGetMultimemDevicePointer(Handle(), 0, mm, &out), ncclInvalidArgument);
}

// Branch: without NVLS there is no multicast mapping, reported as a null result
// rather than an error.
TEST_F(DevicePointerAccessorTest, MultimemWithoutNvls_ReturnsNull) {
  comm->nvlsSupport = 0;
  ncclMultimemHandle mm{};
  mm.mcBasePtr = reinterpret_cast<void*>(0x700000);
  void* out = reinterpret_cast<void*>(0xdead);
  EXPECT_EQ(ncclGetMultimemDevicePointer(Handle(), 0, mm, &out), ncclSuccess);
  EXPECT_EQ(out, nullptr);
}

// Branch: with NVLS the address is the multicast base plus the window's offset
// plus the caller's -- distinct values so a dropped term shows.
TEST_F(DevicePointerAccessorTest, MultimemWithNvls_OffsetsFromMulticastBase) {
  comm->nvlsSupport = 1;
  ncclMultimemHandle mm{};
  mm.mcBasePtr = reinterpret_cast<void*>(0x700000);
  void* out = nullptr;
  ASSERT_EQ(ncclGetMultimemDevicePointer(Handle(), 0x40, mm, &out), ncclSuccess);
  EXPECT_EQ(out, reinterpret_cast<void*>(0x700000 + win.bigOffset + 0x40));
}


// ---------------------------------------------------------------------------
// ncclGinResourcesRequested answers whether a requirements set asks for any GIN
// resource, either at the top level or in the per-resource list.

class GinResourcesRequestedTest : public ::testing::Test {
protected:
  ncclDevCommRequirements reqs{};
  ncclDevResourceRequirements node{}, node2{};
};

// None requested anywhere.
TEST_F(GinResourcesRequestedTest, NothingRequested_ReturnsFalse) {
  EXPECT_FALSE(ncclGinResourcesRequested(&reqs));
}

// Each top-level counter is ORed in, so each has to be able to answer alone.
TEST_F(GinResourcesRequestedTest, EachTopLevelCounterAloneIsEnough) {
  for (int* field : {&reqs.ginSignalCount, &reqs.ginCounterCount, &reqs.barrierCount, &reqs.railGinBarrierCount,
                     &reqs.worldGinBarrierCount}) {
    reqs = ncclDevCommRequirements{};
    *field = 1;
    EXPECT_TRUE(ncclGinResourcesRequested(&reqs));
  }
}

// Branch: the per-resource list is walked when the top level asks for nothing.
TEST_F(GinResourcesRequestedTest, ResourceListEntry_ReturnsTrue) {
  node.ginSignalCount = 1;
  reqs.resourceRequirementsList = &node;
  EXPECT_TRUE(ncclGinResourcesRequested(&reqs));
}

// The walk continues past entries that ask for nothing, so a request on the
// second node still counts.
TEST_F(GinResourcesRequestedTest, LaterResourceListEntry_ReturnsTrue) {
  node2.ginCounterCount = 1;
  node.next = &node2;
  reqs.resourceRequirementsList = &node;
  EXPECT_TRUE(ncclGinResourcesRequested(&reqs));
}

// A list where nothing is requested is still false.
TEST_F(GinResourcesRequestedTest, EmptyResourceList_ReturnsFalse) {
  node.next = &node2;
  reqs.resourceRequirementsList = &node;
  EXPECT_FALSE(ncclGinResourcesRequested(&reqs));
}


// ---------------------------------------------------------------------------
// ncclDevrGetGinAnvilMemLayout reports the flat-VA base and 4G stride for the
// memory an address belongs to, accepting either the user address or its
// mapping in our own slot.

class DevrGetGinAnvilMemLayoutTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclDevrState* devr = nullptr;
  ncclDevrMemory mem{};
  uintptr_t base = 0;
  uint32_t stride = 0;

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    devr = &commStorage->devrState;
    devr->lsaSelf = 1;
    devr->lsaSize = 2;
    devr->bigSize = size_t(2) << 32;  // stride4G of 2
    devr->lsaFlatBase = reinterpret_cast<void*>(0x400000);
    mem.primaryAddr = reinterpret_cast<void*>(0x900000);
    mem.size = 4096;
    mem.bigOffset = 0x300;
    devr->memHead = &mem;
  }
  void TearDown() override { devr->memHead = nullptr; }  // stack object
};

// Branch: an address in a memory's user range.
TEST_F(DevrGetGinAnvilMemLayoutTest, UserAddress_ReportsLayout) {
  ASSERT_EQ(ncclDevrGetGinAnvilMemLayout(devr, reinterpret_cast<void*>(0x900010), &base, &stride), ncclSuccess);
  EXPECT_EQ(base, reinterpret_cast<uintptr_t>(devr->lsaFlatBase) + mem.bigOffset);
  EXPECT_EQ(stride, 2u);  // bigSize >> 32
}

// Branch: the same memory addressed through our own slot of the flat space.
TEST_F(DevrGetGinAnvilMemLayoutTest, FlatAddress_ReportsSameLayout) {
  uintptr_t local = reinterpret_cast<uintptr_t>(devr->lsaFlatBase) + devr->lsaSelf * devr->bigSize + mem.bigOffset;
  ASSERT_EQ(ncclDevrGetGinAnvilMemLayout(devr, reinterpret_cast<void*>(local + 8), &base, &stride), ncclSuccess);
  EXPECT_EQ(base, reinterpret_cast<uintptr_t>(devr->lsaFlatBase) + mem.bigOffset);
}

// Branch: null arguments are rejected before anything is dereferenced.
TEST_F(DevrGetGinAnvilMemLayoutTest, NullArguments_ReturnInvalidArgument) {
  void* addr = reinterpret_cast<void*>(0x900010);
  EXPECT_EQ(ncclDevrGetGinAnvilMemLayout(nullptr, addr, &base, &stride), ncclInvalidArgument);
  EXPECT_EQ(ncclDevrGetGinAnvilMemLayout(devr, nullptr, &base, &stride), ncclInvalidArgument);
  EXPECT_EQ(ncclDevrGetGinAnvilMemLayout(devr, addr, nullptr, &stride), ncclInvalidArgument);
  EXPECT_EQ(ncclDevrGetGinAnvilMemLayout(devr, addr, &base, nullptr), ncclInvalidArgument);
}

// Branch: a devrState with no flat space cannot answer.
TEST_F(DevrGetGinAnvilMemLayoutTest, NoFlatBase_ReturnsInvalidArgument) {
  devr->lsaFlatBase = nullptr;
  EXPECT_EQ(ncclDevrGetGinAnvilMemLayout(devr, reinterpret_cast<void*>(0x900010), &base, &stride),
            ncclInvalidArgument);
}

// Branch: an address belonging to no memory.
TEST_F(DevrGetGinAnvilMemLayoutTest, UnknownAddress_ReturnsInvalidArgument) {
  EXPECT_NE(ncclDevrGetGinAnvilMemLayout(devr, reinterpret_cast<void*>(0x50), &base, &stride), ncclSuccess);
}


// ---------------------------------------------------------------------------
// ncclCommWindowRegister_impl is the public entry point. It does not register
// anything itself -- it validates, then queues a task for the group machinery
// to run at ncclGroupEnd.

class CommWindowRegisterImplTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  std::vector<int> rankToNode;
  std::vector<ncclPeerInfo> peers;
  ncclWindow_vidmem* out = nullptr;
  void* const kUserPtr = reinterpret_cast<void*>(0x100000);

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->rank = 0;
    comm->nRanks = 1;
    comm->localRanks = 1;
    comm->bootstrap = reinterpret_cast<void*>(0x1);
    comm->symmetricSupport = 1;
    rankToNode.assign({0});
    comm->rankToNode = rankToNode.data();
    // ncclDevrInitOnce's proxy-only path rebuilds the team from this.
    localRankToRank.assign({0});
    comm->localRankToRank = localRankToRank.data();
    peers.assign(1, ncclPeerInfo{});
    peers[0].totalGlobalMem = 1u << 20;
    comm->peerInfo = peers.data();
  }

  std::vector<int> localRankToRank;

  void TearDown() override {
    // Drain whatever the call queued; the group machinery would normally own it.
    while (!ncclIntruQueueEmpty(&comm->devrState.regTaskQueue)) {
      free(ncclIntruQueueDequeue(&comm->devrState.regTaskQueue));
    }
    while (!ncclIntruQueueEmpty(&comm->rmaCeInitTaskQueue)) {
      free(ncclIntruQueueDequeue(&comm->rmaCeInitTaskQueue));
    }
    free(comm->devrState.lsaRankList);
    g_callocCallIndex = 0;
    g_callocFailAt = -1;
    ResetDevRuntimeFakes();
  }
};

// The work is deferred: a task carrying the caller's arguments is queued, and
// the out-pointer it will later fill is recorded rather than written now.
TEST_F(CommWindowRegisterImplTest, Succeeds_QueuesTaskCarryingArguments) {
  ASSERT_EQ(ncclCommWindowRegister_impl(comm, kUserPtr, 8192, &out, NCCL_WIN_COLL_SYMMETRIC), ncclSuccess);
  ASSERT_FALSE(ncclIntruQueueEmpty(&comm->devrState.regTaskQueue));

  ncclDevrRegTask* task = ncclIntruQueueHead(&comm->devrState.regTaskQueue);
  EXPECT_EQ(task->userPtr, kUserPtr);
  EXPECT_EQ(task->userSize, 8192u);
  EXPECT_EQ(task->winFlags, NCCL_WIN_COLL_SYMMETRIC);
  EXPECT_EQ(task->outWinDev, &out);
  EXPECT_EQ(out, nullptr);  // cleared on entry, filled when the group runs
}

// Branch: a null pointer is rejected before anything is queued.
TEST_F(CommWindowRegisterImplTest, NullUserPtr_ReturnsInvalidArgument) {
  EXPECT_EQ(ncclCommWindowRegister_impl(comm, nullptr, 8192, &out, 0), ncclInvalidArgument);
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->devrState.regTaskQueue));
}

// Branch: so is a zero-length window.
TEST_F(CommWindowRegisterImplTest, ZeroSize_ReturnsInvalidArgument) {
  EXPECT_EQ(ncclCommWindowRegister_impl(comm, kUserPtr, 0, &out, 0), ncclInvalidArgument);
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->devrState.regTaskQueue));
}

// Branch: a communicator supporting neither symmetric nor host-RMA windows has
// nothing to register, which is success with no work queued -- not an error.
TEST_F(CommWindowRegisterImplTest, NoWindowSupport_SucceedsWithoutQueueing) {
  comm->symmetricSupport = 0;
  comm->hostRmaSupport = false;

  EXPECT_EQ(ncclCommWindowRegister_impl(comm, kUserPtr, 8192, &out, 0), ncclSuccess);
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->devrState.regTaskQueue));
  EXPECT_EQ(out, nullptr);
}

// Host-RMA alone is enough to accept the window, even without symmetric support.
TEST_F(CommWindowRegisterImplTest, HostRmaOnly_QueuesTask) {
  comm->symmetricSupport = 0;
  comm->hostRmaSupport = true;

  EXPECT_EQ(ncclCommWindowRegister_impl(comm, kUserPtr, 8192, &out, 0), ncclSuccess);
  EXPECT_FALSE(ncclIntruQueueEmpty(&comm->devrState.regTaskQueue));
}

// Branch: the RMA CE init is queued lazily on the first window, since it is
// collective and cannot run from this non-collective entry point.
TEST_F(CommWindowRegisterImplTest, FirstWindow_QueuesRmaCeInit) {
  ASSERT_EQ(ncclCommWindowRegister_impl(comm, kUserPtr, 8192, &out, 0), ncclSuccess);
  EXPECT_FALSE(ncclIntruQueueEmpty(&comm->rmaCeInitTaskQueue));
}

// Branch: already initialised, so no second init is queued.
TEST_F(CommWindowRegisterImplTest, RmaCeAlreadyInitialised_SkipsInitTask) {
  comm->rmaState.rmaCeState.initialized = true;

  ASSERT_EQ(ncclCommWindowRegister_impl(comm, kUserPtr, 8192, &out, 0), ncclSuccess);
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->rmaCeInitTaskQueue));
}

// Branch: the task allocation fails, so nothing is queued.
TEST_F(CommWindowRegisterImplTest, TaskAllocFails_ReturnsErrorWithoutQueueing) {
  g_callocFailAt = g_callocCallIndex;  // the register task is the next ncclCalloc

  EXPECT_NE(ncclCommWindowRegister_impl(comm, kUserPtr, 8192, &out, 0), ncclSuccess);
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->devrState.regTaskQueue));
}


// ---------------------------------------------------------------------------
// ncclCommWindowDeregister_impl routes to whichever teardown matches how the
// window was registered.

class CommWindowDeregisterImplTest : public WindowRegisterNonSymTest {};

// Branch: a null window is a no-op, so double-deregister at the API level is
// benign rather than an error.
TEST_F(CommWindowDeregisterImplTest, NullWindow_ReturnsSuccess) {
  EXPECT_EQ(ncclCommWindowDeregister_impl(comm, nullptr), ncclSuccess);
}

// Branch: without symmetric support the non-symmetric teardown runs.
TEST_F(CommWindowDeregisterImplTest, NonSymmetric_RoutesToNonSymTeardown) {
  comm->symmetricSupport = 0;
  ncclWindow_t winDev = nullptr;
  ASSERT_EQ(windowRegisterNonSym(comm, kUserPtr, 4096, 0, nullptr, &winDev), ncclSuccess);
  ASSERT_EQ(comm->devrState.winSortedCount, 1);

  EXPECT_EQ(ncclCommWindowDeregister_impl(comm, winDev), ncclSuccess);
  EXPECT_EQ(comm->devrState.winSortedCount, 0);
}

// Branch: an unregistered handle is rejected by the teardown it routes to,
// and that error reaches the caller rather than being swallowed here.
TEST_F(CommWindowDeregisterImplTest, NonSymmetricUnknownWindow_PropagatesError) {
  comm->symmetricSupport = 0;
  ncclWindow_vidmem stranger{};

  EXPECT_NE(ncclCommWindowDeregister_impl(comm, &stranger), ncclSuccess);
}


// ---------------------------------------------------------------------------
// deepCopyDevCommRequirements takes a private copy of a caller's requirements,
// including both linked lists, so the caller's structures can go away. Its
// counterpart freeDevCommRequirements releases the whole thing.

class DeepCopyDevCommRequirementsTest : public ::testing::Test {
protected:
  ncclDevCommRequirements src = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  ncclDevCommRequirements* dst = nullptr;
  ncclDevResourceRequirements res1{}, res2{};
  ncclTeamRequirements team1{}, team2{};

  void TearDown() override {
    freeDevCommRequirements(dst);
    dst = nullptr;
    g_callocCallIndex = 0;
    g_callocFailAt = -1;
    ResetDevRuntimeFakes();
  }
};

// The scalar body is copied, and the copy is a distinct object.
TEST_F(DeepCopyDevCommRequirementsTest, CopiesScalarFieldsIntoNewObject) {
  src.ginSignalCount = 7;
  src.barrierCount = 3;

  ASSERT_EQ(deepCopyDevCommRequirements(&src, &dst), ncclSuccess);
  ASSERT_NE(dst, nullptr);
  EXPECT_NE(dst, &src);
  EXPECT_EQ(dst->ginSignalCount, 7);
  EXPECT_EQ(dst->barrierCount, 3);
}

// The resource list is duplicated node by node, not aliased -- the copy must
// survive the caller's nodes being reused or freed.
TEST_F(DeepCopyDevCommRequirementsTest, DuplicatesResourceListNodes) {
  res1.bufferSize = 128;
  res2.bufferSize = 256;
  res1.next = &res2;
  src.resourceRequirementsList = &res1;

  ASSERT_EQ(deepCopyDevCommRequirements(&src, &dst), ncclSuccess);
  ASSERT_NE(dst->resourceRequirementsList, nullptr);
  EXPECT_NE(dst->resourceRequirementsList, &res1);  // a copy, not the caller's node
  EXPECT_EQ(dst->resourceRequirementsList->bufferSize, 128u);
  ASSERT_NE(dst->resourceRequirementsList->next, nullptr);
  EXPECT_EQ(dst->resourceRequirementsList->next->bufferSize, 256u);
  EXPECT_EQ(dst->resourceRequirementsList->next->next, nullptr);  // list terminated
}

// The team list likewise.
TEST_F(DeepCopyDevCommRequirementsTest, DuplicatesTeamListNodes) {
  team1.multimem = true;
  team2.multimem = false;
  team1.next = &team2;
  src.teamRequirementsList = &team1;

  ASSERT_EQ(deepCopyDevCommRequirements(&src, &dst), ncclSuccess);
  ASSERT_NE(dst->teamRequirementsList, nullptr);
  EXPECT_NE(dst->teamRequirementsList, &team1);
  EXPECT_TRUE(dst->teamRequirementsList->multimem);
  ASSERT_NE(dst->teamRequirementsList->next, nullptr);
  EXPECT_FALSE(dst->teamRequirementsList->next->multimem);
}

// Empty lists stay empty rather than being left pointing at the source's.
TEST_F(DeepCopyDevCommRequirementsTest, NoLists_LeavesBothEmpty) {
  ASSERT_EQ(deepCopyDevCommRequirements(&src, &dst), ncclSuccess);
  EXPECT_EQ(dst->resourceRequirementsList, nullptr);
  EXPECT_EQ(dst->teamRequirementsList, nullptr);
}

// Branch: an allocation part-way through the copy releases what was built and
// reports nothing back, rather than handing over a half-copied structure.
TEST_F(DeepCopyDevCommRequirementsTest, NodeAllocFails_ReleasesPartialCopy) {
  res1.next = &res2;
  src.resourceRequirementsList = &res1;
  g_callocFailAt = g_callocCallIndex + 2;  // the top-level object and one node succeed

  EXPECT_NE(deepCopyDevCommRequirements(&src, &dst), ncclSuccess);
  EXPECT_EQ(dst, nullptr);
}

// freeDevCommRequirements tolerates null, so callers can release
// unconditionally on a failure path.
TEST_F(DeepCopyDevCommRequirementsTest, FreeNull_IsSafe) {
  freeDevCommRequirements(nullptr);
}


// ---------------------------------------------------------------------------
// getNcclVersionCompat picks the compatibility record covering the version the
// caller compiled against, refusing a caller newer than the library.
//
// The compat table is three globals defined in DevRuntimeTestsStubs.cc, zeroed
// there, so each test sets the version window it needs.

class NcclVersionCompatTest : public ::testing::Test {
protected:
  void SetUp() override { Reset(); }
  void TearDown() override {
    Reset();
    ResetDevRuntimeFakes();
  }
  static void Reset() {
    ncclDevCommCompat_v22902 = ncclDevCommCompat{};
    ncclDevCommCompat_v22907 = ncclDevCommCompat{};
    ncclDevCommCompat_v23000 = ncclDevCommCompat{};
  }
};

// A version inside a record's window selects it.
TEST_F(NcclVersionCompatTest, VersionInWindow_SelectsThatRecord) {
  ncclDevCommCompat_v22902.minVersion = 1;
  ncclDevCommCompat_v22902.maxVersion = 100;
  ncclDevCommCompat_v22907.minVersion = 101;
  ncclDevCommCompat_v22907.maxVersion = 200;

  ncclDevCommCompat* out = nullptr;
  ASSERT_EQ(getNcclVersionCompat(150, &out), ncclSuccess);
  EXPECT_EQ(out, &ncclDevCommCompat_v22907);
}

// Boundaries are inclusive at both ends.
TEST_F(NcclVersionCompatTest, VersionAtWindowEdges_SelectsRecord) {
  ncclDevCommCompat_v22902.minVersion = 10;
  ncclDevCommCompat_v22902.maxVersion = 20;

  ncclDevCommCompat* out = nullptr;
  ASSERT_EQ(getNcclVersionCompat(10, &out), ncclSuccess);
  EXPECT_EQ(out, &ncclDevCommCompat_v22902);
  ASSERT_EQ(getNcclVersionCompat(20, &out), ncclSuccess);
  EXPECT_EQ(out, &ncclDevCommCompat_v22902);
}

// Branch: a version no record covers means the library is not backwards
// compatible with it.
TEST_F(NcclVersionCompatTest, VersionOutsideEveryWindow_ReturnsError) {
  ncclDevCommCompat_v22902.minVersion = 10;
  ncclDevCommCompat_v22902.maxVersion = 20;

  ncclDevCommCompat* out = &ncclDevCommCompat_v23000;
  EXPECT_NE(getNcclVersionCompat(50, &out), ncclSuccess);
  EXPECT_EQ(out, nullptr);  // cleared on entry
}

// Branch: a caller compiled against a newer NCCL than this library is refused
// outright -- a different failure from "no record covers it".
TEST_F(NcclVersionCompatTest, CallerNewerThanLibrary_ReturnsInvalidUsage) {
  ncclDevCommCompat* out = nullptr;
  EXPECT_EQ(getNcclVersionCompat(NCCL_VERSION_CODE + 1, &out), ncclInvalidUsage);
}

// Branch: the version check can be disabled, after which the same version falls
// through to the ordinary table lookup.
TEST_F(NcclVersionCompatTest, VersionCheckDisabled_FallsThroughToLookup) {
  ncclDevCommCompat_v22902.minVersion = 0;
  ncclDevCommCompat_v22902.maxVersion = NCCL_VERSION_CODE + 10;
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t deftVal) -> int64_t {
    return std::string(env) == "ENABLE_VERSION_CHECK" ? 0 : deftVal;
  });

  ncclDevCommCompat* out = nullptr;
  ASSERT_EQ(getNcclVersionCompat(NCCL_VERSION_CODE + 1, &out), ncclSuccess);
  EXPECT_EQ(out, &ncclDevCommCompat_v22902);
}


// ---------------------------------------------------------------------------
// ncclCommQueryProperties reports what the communicator can do. Fields beyond a
// version threshold are only filled for callers new enough to have them.

class CommQueryPropertiesTest : public NcclVersionCompatTest {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  std::vector<int> rankToNode;
  std::vector<int> localRankToRank;
  std::vector<ncclPeerInfo> peers;
  ncclCommProperties_t props{};

  void SetUp() override {
    NcclVersionCompatTest::SetUp();
    ncclDevCommCompat_v22902.minVersion = 0;
    ncclDevCommCompat_v22902.maxVersion = NCCL_VERSION_CODE;

    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->rank = 2;
    comm->nRanks = 8;
    comm->cudaDev = 3;
    comm->nvmlDev = 4;
    comm->localRanks = 1;
    comm->symmetricSupport = 1;
    comm->bootstrap = reinterpret_cast<void*>(0x1);
    // computeLsaSize walks rankToNode[0..nRanks-1], so it must be that long.
    rankToNode.assign(comm->nRanks, 0);
    comm->rankToNode = rankToNode.data();
    localRankToRank.assign({0});
    comm->localRankToRank = localRankToRank.data();
    peers.assign(8, ncclPeerInfo{});
    peers[0].totalGlobalMem = 1u << 20;
    comm->peerInfo = peers.data();

    props = NCCL_COMM_PROPERTIES_INITIALIZER;
  }

  void TearDown() override {
    free(comm->devrState.lsaRankList);
    NcclVersionCompatTest::TearDown();
  }
};

// The basic fields come straight off the communicator.
TEST_F(CommQueryPropertiesTest, ReportsCommIdentity) {
  ASSERT_EQ(ncclCommQueryProperties(comm, &props), ncclSuccess);
  EXPECT_EQ(props.rank, 2);
  EXPECT_EQ(props.nRanks, 8);
  EXPECT_EQ(props.cudaDev, 3);
  EXPECT_EQ(props.nvmlDev, 4);
  EXPECT_EQ(props.deviceApiSupport, 1);
}

// Branch: multicast is reported unavailable across cliques even when the device
// supports it, because NVLS does not span them.
TEST_F(CommQueryPropertiesTest, CrossClique_ReportsNoMultimemSupport) {
  comm->nvlsSupport = 1;
  comm->p2pCrossClique = true;

  ASSERT_EQ(ncclCommQueryProperties(comm, &props), ncclSuccess);
  EXPECT_EQ(props.multimemSupport, 0);
}

// The same communicator without cliques does report it, so the flag is what
// makes the difference rather than nvlsSupport alone.
TEST_F(CommQueryPropertiesTest, SingleClique_ReportsMultimemSupport) {
  comm->nvlsSupport = 1;
  comm->p2pCrossClique = false;

  ASSERT_EQ(ncclCommQueryProperties(comm, &props), ncclSuccess);
  EXPECT_NE(props.multimemSupport, 0);
}

// Branch: an uninitialised properties struct is rejected, since its version and
// size fields would otherwise be read as garbage.
TEST_F(CommQueryPropertiesTest, UninitialisedProps_ReturnsInvalidUsage) {
  ncclCommProperties_t raw{};
  raw.magic = 0;
  EXPECT_EQ(ncclCommQueryProperties(comm, &raw), ncclInvalidUsage);
}


// ---------------------------------------------------------------------------
// ncclDevrGetLsaTeamPtrMC resolves an offset to its multicast address for a
// team, and ncclGetLsaMultimemDevicePointer is the public wrapper.
//
// Only the arm that finds an already-bound team is exercised. Reaching
// symTeamObtain with multimem on a team that is not bound yet hits a known bug
// (AICOMRCCL-2180): on HIP that call returns ncclSuccess without writing its
// out-parameter, so the caller here dereferences an uninitialised pointer. A
// test driving it would crash rather than assert.

class DevrGetLsaTeamPtrMCTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  ncclDevrWindow win{};
  std::vector<unsigned char> teamStorage;
  void* const kMcBase = reinterpret_cast<void*>(0x800000);

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->nvlsSupport = 1;
    win.bigOffset = 0x300;
  }
  void TearDown() override { comm->devrState.teamHead = nullptr; }  // borrowed storage

  // A team already carrying a multicast mapping, which symTeamObtain returns
  // as-is rather than trying to create one.
  ncclTeam SeedBoundTeam(int nRanks, int rank, int stride) {
    ncclTeam team{};
    team.nRanks = nRanks;
    team.rank = rank;
    team.stride = stride;
    teamStorage.assign(sizeof(ncclDevrTeam) + nRanks * sizeof(int), 0);
    auto* t = reinterpret_cast<ncclDevrTeam*>(teamStorage.data());
    t->team = team;
    t->mcBasePtr = kMcBase;
    comm->devrState.teamHead = t;
    return team;
  }
};

// The multicast address is the team's base plus the window's offset plus the
// caller's -- all distinct, so a dropped term shows.
TEST_F(DevrGetLsaTeamPtrMCTest, BoundTeam_OffsetsFromMulticastBase) {
  ncclTeam team = SeedBoundTeam(2, 0, 1);
  void* out = nullptr;
  ASSERT_EQ(ncclDevrGetLsaTeamPtrMC(comm, &win, 0x40, team, &out), ncclSuccess);
  EXPECT_EQ(out, reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(kMcBase) + win.bigOffset + 0x40));
}

// Branch: no NVLS means no multicast address to give.
TEST_F(DevrGetLsaTeamPtrMCTest, NoNvlsSupport_ReturnsInvalidUsage) {
  ncclTeam team = SeedBoundTeam(2, 0, 1);
  comm->nvlsSupport = 0;
  void* out = nullptr;
  EXPECT_EQ(ncclDevrGetLsaTeamPtrMC(comm, &win, 0, team, &out), ncclInvalidUsage);
}

// Branch: null arguments are caught before anything is read.
TEST_F(DevrGetLsaTeamPtrMCTest, NullArguments_ReturnInternalError) {
  ncclTeam team = SeedBoundTeam(2, 0, 1);
  void* out = nullptr;
  EXPECT_EQ(ncclDevrGetLsaTeamPtrMC(comm, nullptr, 0, team, &out), ncclInternalError);
  EXPECT_EQ(ncclDevrGetLsaTeamPtrMC(comm, &win, 0, team, nullptr), ncclInternalError);
}

// The public wrapper, on the arm that does not reach symTeamObtain: without
// NVLS there is no multicast mapping, reported as null with success.
TEST_F(DevicePointerAccessorTest, LsaMultimemWithoutNvls_ReturnsNull) {
  comm->nvlsSupport = 0;
  void* out = reinterpret_cast<void*>(0xdead);
  EXPECT_EQ(ncclGetLsaMultimemDevicePointer(Handle(), 0, &out), ncclSuccess);
  EXPECT_EQ(out, nullptr);
}

// And a window the map does not know is rejected before that check.
TEST_F(DevicePointerAccessorTest, LsaMultimemUnknownWindow_ReturnsInvalidArgument) {
  ncclWindow_vidmem stranger{};
  void* out = nullptr;
  EXPECT_EQ(ncclGetLsaMultimemDevicePointer(&stranger, 0, &out), ncclInvalidArgument);
}


// ---------------------------------------------------------------------------
// ncclDevCommDestroy releases what ncclDevCommCreate set up: the resource
// window and the GIN devcomm, each only if it exists. It also handles devComms
// from older headers, which carry no magic and are assumed to be the earliest
// compat version.

class DevCommDestroyTest : public NcclVersionCompatTest {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  ncclDevComm devComm{};
  ncclWindow_vidmem window{};

  void SetUp() override {
    NcclVersionCompatTest::SetUp();
    ncclDevCommCompat_v22902.minVersion = 0;
    ncclDevCommCompat_v22902.maxVersion = NCCL_VERSION_CODE;

    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->cudaDev = 2;

    devComm.magic = NCCL_API_MAGIC;
    devComm.version = NCCL_VERSION_CODE;
  }
};

// Nothing allocated means nothing to release.
TEST_F(DevCommDestroyTest, EmptyDevComm_ReleasesNothing) {
  ScopedHook dereg(g_ncclCommDeregister, [](const ncclComm_t, void*) { return ncclSuccess; });

  EXPECT_EQ(ncclDevCommDestroy(comm, &devComm), ncclSuccess);
  EXPECT_EQ(dereg.calls, 0);
}

// Branch: a resource window is deregistered, and the one carried by the
// devcomm is what gets passed along.
TEST_F(DevCommDestroyTest, ResourceWindow_IsDeregistered) {
  devComm.resourceWindow = &window;
  ncclWindow_t seen = nullptr;
  ScopedHook winDereg(g_ncclCommWindowDeregister, [&](ncclComm_t, ncclWindow_t w) {
    seen = w;
    return ncclSuccess;
  });

  ASSERT_EQ(ncclDevCommDestroy(comm, &devComm), ncclSuccess);
  EXPECT_EQ(winDereg.calls, 1);
  EXPECT_EQ(seen, &window);
}

// Branch: that deregistration failing is propagated rather than swallowed.
TEST_F(DevCommDestroyTest, ResourceWindowDeregisterFails_PropagatesError) {
  devComm.resourceWindow = &window;
  ScopedHook winDereg(g_ncclCommWindowDeregister,
                      [](ncclComm_t, ncclWindow_t) { return ncclSystemError; });

  EXPECT_NE(ncclDevCommDestroy(comm, &devComm), ncclSuccess);
}

// Branch: the work is scoped to the comm's device and the previous one restored,
// so a caller's current device survives the call.
TEST_F(DevCommDestroyTest, ScopesToCommDeviceAndRestores) {
  std::vector<int> setTo;
  ScopedHook getDev(g_hipGetDevice, [](int* d) {
    *d = 7;
    return hipSuccess;
  });
  ScopedHook setDev(g_hipSetDevice, [&](int d) {
    setTo.push_back(d);
    return hipSuccess;
  });

  ASSERT_EQ(ncclDevCommDestroy(comm, &devComm), ncclSuccess);
  ASSERT_EQ(setTo.size(), 2u);
  EXPECT_EQ(setTo[0], comm->cudaDev);  // switched to the comm's device
  EXPECT_EQ(setTo[1], 7);              // and back to the caller's
}

// Branch: a devComm from an older header carries no magic, so the earliest
// compat record is assumed rather than the version being looked up.
TEST_F(DevCommDestroyTest, UnversionedDevComm_UsesEarliestCompat) {
  devComm.magic = 0;
  devComm.version = 999999;  // would fail a lookup; must not be consulted

  EXPECT_EQ(ncclDevCommDestroy(comm, &devComm), ncclSuccess);
}

// Branch: a versioned devComm the library cannot serve is refused.
TEST_F(DevCommDestroyTest, UnsupportedVersion_ReturnsError) {
  devComm.version = NCCL_VERSION_CODE + 1;

  EXPECT_NE(ncclDevCommDestroy(comm, &devComm), ncclSuccess);
}


// ---------------------------------------------------------------------------
// The devcomm dump helpers are diagnostics: they print and return nothing, so
// the contract is that they survive whatever they are handed. Output is
// captured to keep it out of the test log.

TEST(DevCommDumpTest, DumpsWithoutReadableHandles) {
  ncclDevComm devComm{};
  devComm.rank = 1;
  devComm.nRanks = 4;
  devComm.ginConnectionCount = 0;

  testing::internal::CaptureStdout();
  ncclDevCommDump(&devComm);
  std::string out = testing::internal::GetCapturedStdout();
  EXPECT_NE(out.find("Dev Comm Dump"), std::string::npos);
}

// Branch: per-connection dumps are dispatched by device type -- GDAKI to
// ncclDevCommGdakiDump, PROXY to ncclDevCommProxyDump (dev_runtime.cc:1524-25).
// Both helpers read their context through cudaMemcpy before printing, so the
// copy seam is what proves the dispatch actually happened and that each handle
// reached the right helper. Asserting only on "Connections 2" would not: that
// line prints unconditionally above the loop, so both dispatch lines could be
// deleted and it would still pass.
TEST(DevCommDumpTest, DispatchesPerConnectionDumps) {
  ncclDevComm devComm{};
  devComm.ginConnectionCount = 2;
  devComm.ginNetDeviceTypes[0] = NCCL_GIN_TYPE_GDAKI;
  devComm.ginNetDeviceTypes[1] = NCCL_GIN_TYPE_PROXY;
  devComm.ginHandles[0] = reinterpret_cast<void*>(0x1000);
  devComm.ginHandles[1] = reinterpret_cast<void*>(0x2000);

  std::vector<const void*> readFrom;
  ScopedHook copy(g_hipMemcpy, [&](void* dst, const void* src, size_t n, hipMemcpyKind) {
    readFrom.push_back(src);
    if (dst) memset(dst, 0, n);  // the helpers print through the context they read
    return hipSuccess;
  });

  testing::internal::CaptureStdout();
  ncclDevCommDump(&devComm);
  std::string out = testing::internal::GetCapturedStdout();

  EXPECT_EQ(readFrom, (std::vector<const void*>{devComm.ginHandles[0], devComm.ginHandles[1]}));
  EXPECT_NE(out.find("GDAKI qp"), std::string::npos);
  EXPECT_NE(out.find("PROXY nranks"), std::string::npos);
}


// ---------------------------------------------------------------------------
// ncclDevCommCreate, like the window register entry point, defers the work: it
// validates, deep-copies the caller's requirements so a background thread can
// use them safely, and queues a task for ncclGroupEnd.

class DevCommCreateTest : public NcclVersionCompatTest {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  std::vector<int> rankToNode;
  std::vector<int> localRankToRank;
  std::vector<ncclPeerInfo> peers;
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  ncclDevComm outDevComm{};

  void SetUp() override {
    NcclVersionCompatTest::SetUp();
    ncclDevCommCompat_v22902.minVersion = 0;
    ncclDevCommCompat_v22902.maxVersion = NCCL_VERSION_CODE;

    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->nRanks = 1;
    comm->localRanks = 1;
    comm->symmetricSupport = 1;
    comm->bootstrap = reinterpret_cast<void*>(0x1);
    rankToNode.assign({0});
    comm->rankToNode = rankToNode.data();
    localRankToRank.assign({0});
    comm->localRankToRank = localRankToRank.data();
    peers.assign(1, ncclPeerInfo{});
    peers[0].totalGlobalMem = 1u << 20;
    comm->peerInfo = peers.data();
  }

  void TearDown() override {
    while (!ncclIntruQueueEmpty(&comm->devrState.commCreateTaskQueue)) {
      ncclDevrCommCreateTask* t = ncclIntruQueueDequeue(&comm->devrState.commCreateTaskQueue);
      freeDevCommRequirements(t->reqs);
      free(t);
    }
    free(comm->devrState.lsaRankList);
    g_callocCallIndex = 0;
    g_callocFailAt = -1;
    NcclVersionCompatTest::TearDown();
  }
};

// The queued task owns a copy of the requirements, not the caller's -- that is
// the whole point of the deep copy, since a background thread reads it after
// this call returns.
TEST_F(DevCommCreateTest, Succeeds_QueuesTaskWithCopiedRequirements) {
  reqs.ginSignalCount = 5;

  ASSERT_EQ(ncclDevCommCreate(comm, &reqs, &outDevComm), ncclSuccess);
  ASSERT_FALSE(ncclIntruQueueEmpty(&comm->devrState.commCreateTaskQueue));

  ncclDevrCommCreateTask* task = ncclIntruQueueHead(&comm->devrState.commCreateTaskQueue);
  ASSERT_NE(task->reqs, nullptr);
  EXPECT_NE(task->reqs, &reqs);  // a copy, not the caller's object
  EXPECT_EQ(task->reqs->ginSignalCount, 5);
  EXPECT_EQ(task->outDevComm, &outDevComm);
  EXPECT_EQ(task->devCompat, &ncclDevCommCompat_v22902);
}

// Branch: an uninitialised requirements struct is rejected before its version
// is read.
TEST_F(DevCommCreateTest, UninitialisedRequirements_ReturnsInvalidUsage) {
  ncclDevCommRequirements raw{};
  raw.magic = 0;

  EXPECT_EQ(ncclDevCommCreate(comm, &raw, &outDevComm), ncclInvalidUsage);
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->devrState.commCreateTaskQueue));
}

// Branch: a device communicator needs symmetric memory, so a communicator
// without it is refused -- unlike window registration, which accepts host-RMA.
TEST_F(DevCommCreateTest, NoSymmetricSupport_ReturnsInvalidUsage) {
  comm->symmetricSupport = 0;

  EXPECT_EQ(ncclDevCommCreate(comm, &reqs, &outDevComm), ncclInvalidUsage);
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->devrState.commCreateTaskQueue));
}

// Branch: a version the library cannot serve is refused before any work starts.
TEST_F(DevCommCreateTest, UnsupportedVersion_ReturnsErrorWithoutQueueing) {
  reqs.version = NCCL_VERSION_CODE + 1;

  EXPECT_NE(ncclDevCommCreate(comm, &reqs, &outDevComm), ncclSuccess);
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->devrState.commCreateTaskQueue));
}

// Branch: the task allocation fails.
TEST_F(DevCommCreateTest, TaskAllocFails_ReturnsErrorWithoutQueueing) {
  g_callocFailAt = g_callocCallIndex;

  EXPECT_NE(ncclDevCommCreate(comm, &reqs, &outDevComm), ncclSuccess);
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->devrState.commCreateTaskQueue));
}

// Branch: the requirements filter a compat record may install runs against the
// copy, and its failure aborts the create.
TEST_F(DevCommCreateTest, RequirementsFilterFails_ReturnsErrorWithoutQueueing) {
  ncclDevCommCompat_v22902.devCommRequirementsFilter = [](ncclComm_t, ncclDevCommRequirements_t*) {
    return ncclInvalidArgument;
  };

  EXPECT_NE(ncclDevCommCreate(comm, &reqs, &outDevComm), ncclSuccess);
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->devrState.commCreateTaskQueue));
}


// ---------------------------------------------------------------------------
// ncclDevrCommCreateInternal is where a queued create is actually carried out.
// This suite covers its GIN request validation, which runs before any resource
// is touched and rejects combinations the communicator cannot serve.
//
// The body past that gate builds the whole devcomm -- GIN activation, resource
// windows, barriers -- and is not covered here.

class DevrCommCreateInternalTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> commStorage;
  ncclComm* comm = nullptr;
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  ncclDevComm outDevComm{};
  ncclDevCommCompat compat{};

  void SetUp() override {
    commStorage = std::make_unique<ncclComm>();  // value-initialised: POD members zeroed
    comm = commStorage.get();
    comm->nRanks = 1;
    comm->devrState.lsaSize = 1;
  }

  ncclResult_t Create() {
    return ncclDevrCommCreateInternal(comm, &reqs, &outDevComm, /*isInternal=*/false, &compat);
  }
};

// Branch: asking for GIN resources without asking for GIN itself is a
// contradiction, caught before anything is allocated.
TEST_F(DevrCommCreateInternalTest, GinResourcesWithoutGinConnection_ReturnsInvalidArgument) {
  reqs.ginSignalCount = 1;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_NONE;

  EXPECT_EQ(Create(), ncclInvalidArgument);
}

// Branch: GIN requested on a communicator where some rank cannot provide it.
TEST_F(DevrCommCreateInternalTest, GinRequestedWithoutGlobalSupport_ReturnsInvalidArgument) {
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  comm->globalGinSupport = NCCL_GIN_CONNECTION_NONE;

  EXPECT_EQ(Create(), ncclInvalidArgument);
}

// Branch: a full mesh requested where only rail connectivity exists. This is
// the narrower of the two support checks -- the communicator does support GIN,
// just not this topology.
TEST_F(DevrCommCreateInternalTest, FullGinRequestedOnRailOnlyComm_ReturnsInvalidArgument) {
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  comm->globalGinSupport = NCCL_GIN_CONNECTION_RAIL;

  EXPECT_EQ(Create(), ncclInvalidArgument);
}

// Branch: the deprecated ginForceEnable is equivalent to asking for a full
// connection, so it fails the same support check rather than being ignored.
//
// The counterpart -- the same requirements with the flag clear, which requests
// nothing and passes the gate -- is not tested: it continues into the devcomm
// build, which this fixture does not set up for.
TEST_F(DevrCommCreateInternalTest, GinForceEnable_BehavesAsFullConnection) {
  reqs.ginForceEnable = true;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_NONE;  // overridden by the flag
  comm->globalGinSupport = NCCL_GIN_CONNECTION_RAIL;

  EXPECT_EQ(Create(), ncclInvalidArgument);
}


// ---------------------------------------------------------------------------
// ncclDevCommCopyLsaData copies the LSA-shared span of a devcomm between two
// ranks' copies: everything from `rank` up to but excluding `railGinBarrier`.

TEST(DevCommCopyLsaData, CopiesTheLsaSpanOnly) {
  ncclDevComm src{}, dst{};
  src.rank = 3;
  src.nRanks = 8;
  src.lsaRank = 1;
  src.lsaSize = 4;
  // Outside the copied span: must survive untouched.
  src.railGinBarrier.signal0 = 111;
  dst.railGinBarrier.signal0 = 222;

  ncclDevCommCopyLsaData(&dst.rank, &src.rank);

  EXPECT_EQ(dst.rank, 3);
  EXPECT_EQ(dst.nRanks, 8);
  EXPECT_EQ(dst.lsaRank, 1);
  EXPECT_EQ(dst.lsaSize, 4);
  EXPECT_EQ(dst.railGinBarrier.signal0, 222);  // past the span, not overwritten
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
