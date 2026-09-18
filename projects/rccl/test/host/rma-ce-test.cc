/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for src/rma/rma_ce.cc (AICOMRCCL-2347).
 *************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "ScopedHook.h"
#include "fakes/ce_fakes.h"
#include "fakes/dev_runtime_micro_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/rma_fakes.h"

#include "nccl.h"
#include "comm.h"
#include "rma/rma_ce.h"

// rma_fakes.cc seams ncclRmaCePutLaunch / ncclRmaCeWaitLaunch for rma-test.cc,
// which needs them in this same binary. Rename the unit's own entry points so
// both can coexist; the static helpers are distinct tokens and are untouched.
#define ncclRmaCePutLaunch ncclRmaCePutLaunchUut
#define ncclRmaCeWaitLaunch ncclRmaCeWaitLaunchUut
#include RMA_CE_CC_PATH
#undef ncclRmaCePutLaunch
#undef ncclRmaCeWaitLaunch

namespace {

// Release a window a test left registered: ncclRmaCeFinalize deregisters one
// but never frees the host object. Non-symmetric windows only, so there is no
// backing ncclDevrMemory to drain.
void ReclaimWindow(ncclDevrWindow* w) {
  if (w == nullptr) return;
  free(w->ipcPeerPtrs);
  free(w->ipcPeerPtrsAllocBase);
  free(w);
}

// Minimal comm for the uninitialised guard: both entry points read
// rmaState.rmaCeState.initialized before anything else. ncclComm is ~3.8 MB, so
// it is heap-allocated rather than held by value.
class RmaCeLaunchTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclKernelPlan> plan_;

  void SetUp() override {
    // Reset on entry as well as in TearDown: the seams are process-wide, the
    // fixtures install lambdas capturing `this`, and a test that dies mid-body
    // never reaches its TearDown. Same reason rma-test.cc gives.
    ResetCeFakes();
    ResetDevRuntimeMicroFakes();
    ResetRmaFakes();
    ResetHipFakes();

    comm_ = std::make_unique<ncclComm>();   // value-initialised, so initialized == false
    plan_ = std::make_unique<ncclKernelPlan>();
  }

  void TearDown() override {
    ResetCeFakes();
    ResetDevRuntimeMicroFakes();
    ResetRmaFakes();
    ResetHipFakes();
  }
};

// Both entry points refuse to touch a communicator whose CE state was never
// brought up, rather than dereferencing it.
TEST_F(RmaCeLaunchTest, PutLaunch_CeNotInitialised_ReturnsInternalError) {
  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclInternalError);
}

TEST_F(RmaCeLaunchTest, WaitLaunch_CeNotInitialised_ReturnsInternalError) {
  EXPECT_EQ(ncclRmaCeWaitLaunchUut(comm_.get(), plan_.get(), nullptr), ncclInternalError);
}

// ---------------------------------------------------------------------------
// ncclRmaCeInit
// ---------------------------------------------------------------------------

// Brings up CE state for a comm. The fixture supplies the two things the unit
// cannot compute for itself -- the registered window, and a stream/event pair --
// and lets everything else run for real so the layout arithmetic is the unit's.
class RmaCeInitTest : public ::testing::Test {
protected:
  // dev_runtime.cc is compiled into this binary, so its functions run for real
  // and the fixture sets the state they read rather than stubbing them.
  //
  // nRanks and lsaSize differ deliberately: equal values hide every confusion
  // of the two.
  static constexpr int kNRanks     = 7;
  static constexpr int kLsaSize    = 5;
  static constexpr int kNumRmaSig  = 2;
  // The signal region is indexed [signalIdx][lsaRank], so it holds
  // lsaSize * numRmaSig slots -- not one per rank.
  static constexpr int kSignalSlots = kLsaSize * kNumRmaSig;
  // Three equal regions (signals, graph signals, graph acks) make up a window.
  static constexpr int kWinSlots   = 3 * kSignalSlots;
  // This rank's own LSA slot. Non-zero, different from comm_->rank, and outside
  // the range of peers the tests use: ncclDevrGetLsaRankPtr short-circuits a
  // self-targeted lookup to the local window, which would mask the peer
  // arithmetic the tests are checking.
  static constexpr int kLsaSelf    = 3;
  static constexpr int kRank       = 0;

  // World ranks of the LSA team, in LSA order -- deliberately not the identity,
  // because rma_ce.cc mixes the two (a peer's world rank names the task, its LSA
  // rank names the slot) and an identity mapping makes a swap undetectable.
  // lsaRankList_[kLsaSelf] is kRank, as production requires.
  static constexpr int kLsaWorld[kLsaSize] = {4, 3, 1, 0, 2};

  std::unique_ptr<ncclComm> comm_;
  std::vector<int> lsaRankList_;
  std::vector<ncclPeerInfo> peerInfo_;
  std::vector<int> rankToNode_;
  std::vector<int> localRankToRank_;
  // One window image per LSA rank, laid end to end. The IPC exchange below
  // publishes rank r's image at offset r * kWinSlots, so the real
  // ncclDevrGetLsaRankPtr resolves a peer lookup to PeerWin(r) + offset.
  std::vector<uint64_t> peerWins_;
  size_t registeredBytes_ = 0;
  void* registeredPtr_ = nullptr;
  // The device-side window headers registration allocated, so TearDown can find
  // the host windows behind them.
  std::vector<ncclWindow_vidmem*> shadowWins_;
  // Whether TearDown should run ncclRmaCeFinalize itself; see TearDown.
  bool autoFinalize_ = true;

  // The one window ncclRmaCeInit registered (per context), for tests that need
  // to reach into it.
  ncclDevrWindow* SignalWin(int ctx = 0) { return Ctx(ctx)->signalsWin; }

  // This rank's slot inside a peer's window, at the given region offset. The
  // write side of SignalSlot: a sender raises its own slot in the peer's copy.
  const char* SelfSlotIn(int peerWorldRank, size_t regionOffset, int sigIdx = 0) {
    return reinterpret_cast<const char*>(PeerWin(peerWorldRank)) + regionOffset +
           (sigIdx * kLsaSize + kLsaSelf) * sizeof(uint64_t);
  }

  // Rank r's window image inside peerWins_.
  // World rank -> LSA rank, the lookup ncclDevrWorldToLsaRank performs.
  static int LsaOf(int worldRank) {
    for (int i = 0; i < kLsaSize; i++) {
      if (kLsaWorld[i] == worldRank) return i;
    }
    return -1;
  }
  // The slot a (signalIdx, peer) pair addresses, peer named by world rank.
  static int SignalSlot(int sigIdx, int peerWorldRank) {
    return sigIdx * kLsaSize + LsaOf(peerWorldRank);
  }

  // Window image of a peer, named by world rank. Laid out in LSA order, which
  // is how the IPC exchange below publishes it.
  uint64_t* PeerWinByLsa(int lsaRank) { return peerWins_.data() + lsaRank * kWinSlots; }
  uint64_t* PeerWin(int peerWorldRank) { return PeerWinByLsa(LsaOf(peerWorldRank)); }

  // Mirrors the entry windowRegisterNonSym all-gathers; the layout has to match
  // for the hook to publish values the function then reads back.
  struct ExchangeEntry {
    hipIpcMemHandle_t handle;
    uint64_t hostHash;
    uint64_t pidHash;
    size_t userOffset;
    size_t userSize;
  };

  void SetUp() override {
    ResetCeFakes();
    ResetHipFakes();
    ResetDevRuntimeMicroFakes();

    comm_ = std::make_unique<ncclComm>();
    comm_->rank = kRank;
    comm_->nRanks = kNRanks;
    comm_->symmetricSupport = 0;      // RCCL non-symmetric path
    comm_->bootstrap = reinterpret_cast<void*>(0x1);
    comm_->config.numRmaCtx = 1;
    comm_->config.numRmaSig = kNumRmaSig;
    // Single-node topology. Unused while bigSize is non-zero, but the failure
    // arm that lets the real ncclDevrInitOnce run reads all of it.
    comm_->cudaDev = 0;
    comm_->localRanks = kNRanks;
    comm_->localRank = kRank;
    rankToNode_.assign(kNRanks, 0);
    comm_->rankToNode = rankToNode_.data();
    localRankToRank_.assign({0, 1, 2, 3, 4, 5, 6});
    comm_->localRankToRank = localRankToRank_.data();

    ncclDevrState* devr = &comm_->devrState;
    devr->bigSize = 1;                // ncclDevrInitOnce: already-initialised
    devr->lsaSize = kLsaSize;
    devr->lsaSelf = kLsaSelf;
    lsaRankList_.assign(std::begin(kLsaWorld), std::end(kLsaWorld));
    devr->lsaRankList = lsaRankList_.data();

    // Distinct hashes per rank, so no peer looks like it shares our process --
    // windowRegisterNonSym leaves same-process peers unmapped.
    peerInfo_.assign(kNRanks, ncclPeerInfo{});
    for (int r = 0; r < kNRanks; r++) {
      peerInfo_[r].hostHash = 100 + r;
      peerInfo_[r].pidHash = 200 + r;
    }
    comm_->peerInfo = peerInfo_.data();

    peerWins_.assign(static_cast<size_t>(kLsaSize) * kWinSlots, 0);

    // Signal buffers come from comm->memManager, so the allocation path is the
    // driver VMM surface rather than a plain hipMalloc.
    InstallHipVmmEmulator();
    // Installed after the emulator on purpose: it provides its own
    // g_hipIpcOpenMemHandle, which would otherwise replace the one below.

    // Register the signal window for real. The three hooks below are the node
    // boundary the registration cannot cross on its own: the handle it exports,
    // the exchange with the rest of the LSA team, and the mapping it opens for
    // each peer. Together they place rank r's window image at PeerWin(r).
    g_hipIpcGetMemHandle = [](hipIpcMemHandle_t*, void*) { return hipSuccess; };
    g_devrBootstrapIntraNodeAllGather =
        [this](void*, int*, int self, int size, void* buf, int) {
          auto* e = static_cast<ExchangeEntry*>(buf);
          for (int r = 0; r < size; r++) {
            e[r].hostHash = (r == self) ? peerInfo_[kRank].hostHash : 500 + r;
            e[r].pidHash = (r == self) ? peerInfo_[kRank].pidHash : 600 + r;
            e[r].userOffset = static_cast<size_t>(r) * kWinSlots * sizeof(uint64_t);
            e[r].userSize = kWinSlots * sizeof(uint64_t);
          }
          return ncclSuccess;
        };
    g_hipIpcOpenMemHandle = [this](void** ptr, hipIpcMemHandle_t, unsigned int) {
      *ptr = peerWins_.data();
      return hipSuccess;
    };
    // Registration's last stage allocates the device-side window header; chain
    // to the default and keep the pointer so TearDown can free the host window.
    auto shadowAlloc = g_devrShadowPoolAlloc;
    g_devrShadowPoolAlloc = [this, shadowAlloc](ncclShadowPool* pool, size_t size, void** dev,
                                                void** host, hipStream_t st) {
      ncclResult_t r = shadowAlloc(pool, size, dev, host, st);
      if (r == ncclSuccess && host != nullptr) {
        shadowWins_.push_back(static_cast<ncclWindow_vidmem*>(*host));
      }
      return r;
    };
    // ...and drop it again if registration unwinds, which frees the window
    // itself -- otherwise TearDown would read a dangling header.
    auto shadowFree = g_devrShadowPoolFree;
    g_devrShadowPoolFree = [this, shadowFree](ncclShadowPool* pool, void* devObj, hipStream_t st) {
      shadowWins_.erase(std::remove_if(shadowWins_.begin(), shadowWins_.end(),
                                       [devObj](ncclWindow_vidmem* h) {
                                         return static_cast<void*>(h) == devObj;
                                       }),
                        shadowWins_.end());
      return shadowFree(pool, devObj, st);
    };
    // Record the size the unit asked to register, which is its own arithmetic.
    g_devrNcclCommRegister = [this](const ncclComm_t, void* ptr, size_t size, void** handle) {
      registeredPtr_ = ptr;
      registeredBytes_ = size;
      if (handle) *handle = nullptr;
      return ncclSuccess;
    };

    // The unit creates its CE stream and event last; both default to failure.
    g_hipStreamCreateResult = hipSuccess;
    g_hipEventCreateResult = hipSuccess;
    // The unit seeds its ack flags and signal constants through ncclCudaMemcpy /
    // ncclCudaCalloc, which go via stream-capture, memcpy-async and a stream
    // wait -- all behind the existing async-ops knob.
    g_hipAsyncOpsResult = hipSuccess;
  }

  void TearDown() override {
    // Balance what a test left built; a no-op success if it built nothing.
    // Suites that drive Finalize themselves clear the flag: re-running it after
    // one that failed part-way double-frees what that attempt released.
    if (autoFinalize_) ncclRmaCeFinalize(comm_.get());
    for (ncclWindow_vidmem* h : shadowWins_) {
      ReclaimWindow(static_cast<ncclDevrWindow*>(h->winHost));
    }
    shadowWins_.clear();
    free(comm_->devrState.winSorted);   // opaque here, but free() needs no type
    comm_->devrState.winSorted = nullptr;
    comm_->devrState.winSortedCount = comm_->devrState.winSortedCapacity = 0;
    comm_->devrState.lsaRankList = nullptr;   // borrowed, not malloc'd
    ResetCeFakes();
    ResetHipFakes();
    ResetDevRuntimeMicroFakes();
  }

  // True once every configured context has finished its allocations, which is
  // where the CE stream and event are created.
  bool ContextsBuilt() {
    auto& st = comm_->rmaState.rmaCeState;
    if (st.rmaCeCtxs == nullptr || st.rmaCeCtxCount == 0) return false;
    for (int i = 0; i < st.rmaCeCtxCount; i++) {
      auto* c = static_cast<ncclRmaCeCtx*>(st.rmaCeCtxs[i]);
      if (c == nullptr || c->signalOpSeqsDev == nullptr) return false;
    }
    return true;
  }

  ncclRmaCeCtx* Ctx(int i) {
    return static_cast<ncclRmaCeCtx*>(comm_->rmaState.rmaCeState.rmaCeCtxs[i]);
  }
};

// The flag other entry points gate on is only set once everything above it
// succeeded, so it doubles as "the whole sequence ran".
TEST_F(RmaCeInitTest, Init_AllDependenciesSucceed_MarksStateInitialised) {
  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);

  EXPECT_TRUE(comm_->rmaState.rmaCeState.initialized);
  EXPECT_NE(comm_->rmaState.rmaCeState.ceStream, nullptr);
  EXPECT_NE(comm_->rmaState.rmaCeState.ceEvent, nullptr);
}

// One context per configured RMA context, and the count is taken from config.
TEST_F(RmaCeInitTest, Init_MultipleConfiguredContexts_CreatesOneEach) {
  comm_->config.numRmaCtx = 3;

  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);

  ASSERT_EQ(comm_->rmaState.rmaCeState.rmaCeCtxCount, 3);
  for (int i = 0; i < 3; i++) {
    EXPECT_NE(Ctx(i), nullptr) << "context " << i;
    EXPECT_EQ(Ctx(i)->comm, comm_.get()) << "context " << i;
  }
}

// The signal region is one buffer carved into three equal slot-indexed areas.
// This is the unit's own arithmetic, so it is pinned in both forms it
// publishes: device pointers for the kernels, and byte offsets for the window.
TEST_F(RmaCeInitTest, Init_Succeeds_CarvesSignalRegionBySlotCount) {
  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);

  EXPECT_EQ(registeredBytes_, 3 * kSignalSlots * sizeof(uint64_t));

  const ncclRmaCeCtx* ctx = Ctx(0);
  ASSERT_NE(ctx, nullptr);
  // Anchored, not merely non-null: the buffer published is the one registered,
  // so the relative checks below have an absolute starting point.
  uint64_t* base = ctx->signalsDev;
  ASSERT_NE(base, nullptr);
  EXPECT_EQ(base, registeredPtr_);
  EXPECT_EQ(ctx->graphSignalsDev, base + kSignalSlots);
  EXPECT_EQ(ctx->graphAckDev, base + 2 * kSignalSlots);

  EXPECT_EQ(ctx->signalOffset, 0u);
  EXPECT_EQ(ctx->graphSignalOffset, kSignalSlots * sizeof(uint64_t));
  EXPECT_EQ(ctx->graphAckOffset, 2 * kSignalSlots * sizeof(uint64_t));

  // The window handle is taken from the shadow's host object, not the device one.
  ASSERT_EQ(shadowWins_.size(), 1u);
  EXPECT_EQ(ctx->signalsWin, static_cast<ncclDevrWindow*>(shadowWins_[0]->winHost));
}

// The two graph-safe constants are carved from one 2-element allocation, zero
// first. Nothing else pins which is which, so a swap would make every graph
// signal and every ack write the wrong value.
TEST_F(RmaCeInitTest, Init_Succeeds_PublishesZeroThenOneAsSignalConstants) {
  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);

  const ncclRmaCeCtx* ctx = Ctx(0);
  ASSERT_NE(ctx->signalConstDev, nullptr);
  EXPECT_EQ(ctx->signalConstZeroDev, ctx->signalConstDev);
  EXPECT_EQ(ctx->signalConstOneDev, ctx->signalConstDev + 1);
  // ...and by value: seeding both to the same number would keep the addresses
  // right while every graph signal and ack wrote it.
  EXPECT_EQ(ctx->signalConstZeroDev[0], 0u);
  EXPECT_EQ(ctx->signalConstOneDev[0], 1u);
  // The ack flags start raised, so a first graph put is not blocked.
  EXPECT_EQ(ctx->graphAckDev[0], 1u);
}

// Zero configured contexts is not an error: the stream and event still come up,
// so a comm that never uses RMA CE is still in a defined state.
TEST_F(RmaCeInitTest, Init_NoConfiguredContexts_StillCreatesStreamAndEvent) {
  comm_->config.numRmaCtx = 0;

  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);

  EXPECT_EQ(comm_->rmaState.rmaCeState.rmaCeCtxCount, 0);
  EXPECT_TRUE(comm_->rmaState.rmaCeState.initialized);
  EXPECT_NE(comm_->rmaState.rmaCeState.ceStream, nullptr);
}

// The symmetric runtime is brought up first, so its failure stops everything.
// bigSize == 0 makes the real ncclDevrInitOnce do its work rather than return
// early, and the granularity query it starts with is then made to fail.
TEST_F(RmaCeInitTest, Init_DevrInitOnceFails_PropagatesAndLeavesUninitialised) {
  comm_->devrState.bigSize = 0;
  comm_->symmetricSupport = 1;
  ScopedHook granularity(g_hipMemGetAllocationGranularity,
                         [](size_t*, const hipMemAllocationProp*, hipMemAllocationGranularity_flags) {
                           return hipErrorInvalidValue;
                         });

  EXPECT_NE(ncclRmaCeInit(comm_.get()), ncclSuccess);
  EXPECT_FALSE(comm_->rmaState.rmaCeState.initialized);
  EXPECT_EQ(granularity.calls, 1);
  comm_->devrState.lsaRankList = nullptr;  // freed on that failure path already
}

// A failed window registration unwinds mid-context rather than leaving the state
// half-built and marked ready.
TEST_F(RmaCeInitTest, Init_WindowRegisterFails_PropagatesAndLeavesUninitialised) {
  // The registration exports an IPC handle for the buffer before anything else
  // it can fail on; refusing that unwinds the whole window.
  ScopedHook ipcGet(g_hipIpcGetMemHandle,
                    [](hipIpcMemHandle_t*, void*) { return hipErrorInvalidValue; });

  EXPECT_NE(ncclRmaCeInit(comm_.get()), ncclSuccess);
  EXPECT_FALSE(comm_->rmaState.rmaCeState.initialized);
  EXPECT_EQ(ipcGet.calls, 1);
}

// Failing on a later context is the case that leaves a partially built array
// behind: context 0 is complete and context 1 has its window registered when the
// error surfaces. The ack seed is the only copy sized to the signal region, so
// failing the second one lands exactly there.
TEST_F(RmaCeInitTest, Init_AckSeedFailsOnLaterContext_PropagatesAndLeavesUninitialised) {
  comm_->config.numRmaCtx = 2;
  int ackCopies = 0;
  auto realCopy = g_hipMemcpyAsync;
  ScopedHook copy(g_hipMemcpyAsync, [&ackCopies, realCopy](void* dst, const void* src, size_t n,
                                                           hipMemcpyKind kind, hipStream_t st) {
    if (n == kSignalSlots * sizeof(uint64_t) && ++ackCopies == 2) return hipErrorInvalidValue;
    return realCopy(dst, src, n, kind, st);
  });

  EXPECT_NE(ncclRmaCeInit(comm_.get()), ncclSuccess);
  EXPECT_FALSE(comm_->rmaState.rmaCeState.initialized);
  EXPECT_EQ(ackCopies, 2);
}

// The stream and event are created after every context is built, so a failure
// there still must not report the state as ready.
TEST_F(RmaCeInitTest, Init_StreamCreateFails_LeavesUninitialised) {
  // The allocator creates a stream of its own per buffer, so failing every call
  // would stop at the first allocation instead of at the CE stream. Fail only
  // once the context loop is done, which signalOpSeqsDev (its last allocation)
  // marks.
  auto realCreate = g_hipStreamCreateWithFlags;
  bool failed = false;
  ScopedHook create(g_hipStreamCreateWithFlags,
                    [this, &failed, realCreate](hipStream_t* st, unsigned flags) {
                      if (ContextsBuilt() && !failed) {
                        failed = true;
                        return hipErrorInvalidValue;
                      }
                      return realCreate(st, flags);
                    });

  EXPECT_EQ(ncclRmaCeInit(comm_.get()), ncclUnhandledCudaError);
  EXPECT_TRUE(failed) << "never reached the CE stream creation";
  EXPECT_FALSE(comm_->rmaState.rmaCeState.initialized);
}

TEST_F(RmaCeInitTest, Init_EventCreateFails_LeavesUninitialised) {
  g_hipEventCreateResult = hipErrorInvalidValue;

  EXPECT_EQ(ncclRmaCeInit(comm_.get()), ncclUnhandledCudaError);
  EXPECT_FALSE(comm_->rmaState.rmaCeState.initialized);
}

// ---------------------------------------------------------------------------
// ncclRmaCeFinalize
// ---------------------------------------------------------------------------

// Tears down exactly what ncclRmaCeInit builds, so the fixture builds it with the
// real thing rather than a hand-assembled imitation -- a teardown test whose
// input was not produced by the matching setup proves little.
class RmaCeFinalizeTest : public RmaCeInitTest {
protected:
  std::vector<void*> freed_;
  std::vector<ncclWindow_t> deregistered_;

  void SetUp() override {
    RmaCeInitTest::SetUp();
    autoFinalize_ = false;   // every test here drives Finalize itself
    // Device buffers are released with ncclCudaFree(ptr, comm->memManager),
    // which lands on cudaFree -- not on the public ncclMemFree.
    auto realFree = g_hipFree;
    g_hipFree = [this, realFree](void* p) {
      freed_.push_back(p);
      return realFree(p);
    };
    g_devrNcclCommWindowDeregister = [this](ncclComm_t, ncclWindow_t win) {
      deregistered_.push_back(win);
      return ncclSuccess;
    };
    // ncclCudaFree looks up an allocation's base and size for its accounting
    // before releasing it. Report each pointer as its own base.
    g_hipMemGetAddressRange = [](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
      if (base) *base = ptr;
      if (size) *size = 0;
      return hipSuccess;
    };
  }

};

// The whole point: after teardown the comm reports no CE state, so a later
// ncclRmaCeInit starts clean and the launch entry points refuse work again.
TEST_F(RmaCeFinalizeTest, Finalize_AfterInit_ResetsStateToUninitialised) {
  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);

  ASSERT_EQ(ncclRmaCeFinalize(comm_.get()), ncclSuccess);

  EXPECT_FALSE(comm_->rmaState.rmaCeState.initialized);
  EXPECT_EQ(comm_->rmaState.rmaCeState.rmaCeCtxCount, 0);
  EXPECT_EQ(comm_->rmaState.rmaCeState.rmaCeCtxs, nullptr);
}

// The stream and event are owned by this state, so they are released and the
// handles cleared -- leaving a dangling handle behind would outlive the comm.
TEST_F(RmaCeFinalizeTest, Finalize_AfterInit_ClearsStreamAndEvent) {
  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);
  ASSERT_NE(comm_->rmaState.rmaCeState.ceStream, nullptr);

  ASSERT_EQ(ncclRmaCeFinalize(comm_.get()), ncclSuccess);

  EXPECT_EQ(comm_->rmaState.rmaCeState.ceStream, nullptr);
  EXPECT_EQ(comm_->rmaState.rmaCeState.ceEvent, nullptr);
}

// Releasing the signal window is pure side effect, so it is pinned by observing
// the call. The handle deregistered must be the one the window was registered
// under, not the host shadow the unit reads its fields from.
TEST_F(RmaCeFinalizeTest, Finalize_AfterInit_DeregistersAndFreesEverySignalWindow) {
  comm_->config.numRmaCtx = 3;
  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);
  std::vector<void*> deviceBufs;
  for (int i = 0; i < 3; i++) {
    deviceBufs.push_back(Ctx(i)->signalsDev);
    deviceBufs.push_back(Ctx(i)->signalOpSeqsDev);
    deviceBufs.push_back(Ctx(i)->signalConstDev);
  }
  std::vector<ncclWindow_t> expected{Ctx(0)->signalsWin->vidmem, Ctx(1)->signalsWin->vidmem,
                                     Ctx(2)->signalsWin->vidmem};

  ASSERT_EQ(ncclRmaCeFinalize(comm_.get()), ncclSuccess);

  // Every context, in order -- not just the first, and not one repeated. The
  // handle deregistered is the one the window was registered under (vidmem),
  // not the host object the unit reads its fields from.
  EXPECT_EQ(deregistered_, expected);
  // Distinct, so hoisting the registration out of the per-context loop fails.
  EXPECT_EQ(std::set<ncclWindow_t>(expected.begin(), expected.end()).size(), 3u);
  // Every device buffer each context allocated is released. Containment rather
  // than equality: the order is production's business.
  for (size_t i = 0; i < deviceBufs.size(); i++) {
    EXPECT_NE(std::find(freed_.begin(), freed_.end(), deviceBufs[i]), freed_.end())
        << "device buffer " << i << " not freed";
  }
}

// Releasing the stream is the first thing teardown does after the task queue, so
// a failure there propagates with the contexts still standing.
TEST_F(RmaCeFinalizeTest, Finalize_StreamDestroyFails_Propagates) {
  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);
  ScopedHook destroy(g_hipStreamDestroy, [](hipStream_t) { return hipErrorInvalidValue; });

  EXPECT_EQ(ncclRmaCeFinalize(comm_.get()), ncclUnhandledCudaError);
  EXPECT_TRUE(comm_->rmaState.rmaCeState.initialized);
}

// A device buffer that will not release stops teardown rather than carrying on
// and reporting a clean finalize.
TEST_F(RmaCeFinalizeTest, Finalize_DeviceFreeFails_Propagates) {
  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);
  g_hipFree = [](void*) { return hipErrorInvalidValue; };

  EXPECT_EQ(ncclRmaCeFinalize(comm_.get()), ncclUnhandledCudaError);
  EXPECT_TRUE(comm_->rmaState.rmaCeState.initialized);
}

// Finalizing a comm that was never initialised is not an error: every field it
// would release is null, so the guards skip and it reports success.
TEST_F(RmaCeFinalizeTest, Finalize_NeverInitialised_IsANoOpSuccess) {
  EXPECT_EQ(ncclRmaCeFinalize(comm_.get()), ncclSuccess);

  EXPECT_FALSE(comm_->rmaState.rmaCeState.initialized);
  EXPECT_TRUE(freed_.empty());
  EXPECT_TRUE(deregistered_.empty());
}

// Deferred init tasks are owned by the comm and outlive nothing, so teardown
// drains the queue rather than leaving entries pointing at freed state.
TEST_F(RmaCeFinalizeTest, Finalize_PendingInitTasks_DrainsTheQueue) {
  ncclIntruQueueConstruct(&comm_->rmaCeInitTaskQueue);
  for (int i = 0; i < 2; i++) {
    auto* task = static_cast<ncclRmaCeInitTask*>(::calloc(1, sizeof(ncclRmaCeInitTask)));
    ncclIntruQueueEnqueue(&comm_->rmaCeInitTaskQueue, task);
  }

  ASSERT_EQ(ncclRmaCeFinalize(comm_.get()), ncclSuccess);

  EXPECT_TRUE(ncclIntruQueueEmpty(&comm_->rmaCeInitTaskQueue));
}

// A failed deregistration stops teardown and surfaces, rather than carrying on
// and reporting the state as cleanly torn down.
TEST_F(RmaCeFinalizeTest, Finalize_DeregisterFails_PropagatesAndLeavesStateMarked) {
  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);
  g_devrNcclCommWindowDeregister = [](ncclComm_t, ncclWindow_t) { return ncclInternalError; };

  EXPECT_EQ(ncclRmaCeFinalize(comm_.get()), ncclInternalError);
  EXPECT_TRUE(comm_->rmaState.rmaCeState.initialized);
  // Not retried here, and the state is left allocated on purpose: the failed
  // attempt already released this context's buffers before the deregister, and
  // Finalize restarts at context 0, so a second call double-frees them.
}

// ---------------------------------------------------------------------------
// ncclRmaCePutLaunch / ncclRmaCeWaitLaunch
// ---------------------------------------------------------------------------

// The dispatcher's own contract is the guard and the persistent split; what each
// path then does is that helper's contract, covered separately. The two are told
// apart by the capacity they size their batch-ops params to -- the persistent
// path builds one op at a time, the non-persistent path one per rank.
// Everything a launch suite needs before it installs its own hooks: CE state
// brought up, a task pool, and an empty plan.
class RmaCeLaunchFixture : public RmaCeInitTest {
protected:
  std::unique_ptr<ncclKernelPlan> plan_;
  ncclRmaArgs args_{};

  void SetUp() override {
    RmaCeInitTest::SetUp();
    ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);

    ncclMemoryStackConstruct(&comm_->memPermanent);
    ncclMemoryPoolConstruct(&comm_->memPool_ncclTaskRma);

    plan_ = std::make_unique<ncclKernelPlan>();
    plan_->rmaArgs = &args_;
    args_.nRmaTasksCe = 0;
  }

  void TearDown() override {
    ncclMemoryStackDestruct(&comm_->memPermanent);
    RmaCeInitTest::TearDown();
  }
};

class RmaCePutLaunchTest : public RmaCeLaunchFixture {
protected:
  std::vector<int> initCapacities_;

  void SetUp() override {
    RmaCeLaunchFixture::SetUp();

    auto initParams = g_ceInitBatchOpsParams;
    g_ceInitBatchOpsParams = [this, initParams](ncclCeBatchOpsParams* params, int capacity) {
      initCapacities_.push_back(capacity);
      return initParams(params, capacity);   // still allocate the op arrays
    };
    // This suite reads the sizing above to tell the two paths apart, not what
    // either path then submits -- but the call still has to be declared rather
    // than left to a default.
    g_ceLaunchBatchOps = [](ncclComm*, ncclCeBatchOpsParams*, hipStream_t, ncclCeCollArgs*) {
      return ncclSuccess;
    };
  }
};

// A graph-captured plan takes the persistent path, which sizes its batches for a
// single op because it emits them per task rather than per round.
TEST_F(RmaCePutLaunchTest, PutLaunch_PersistentPlan_TakesPersistentPath) {
  plan_->persistent = true;

  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  EXPECT_EQ(initCapacities_, (std::vector<int>{1, 1}));
}

// A non-captured plan takes the other path, which batches across peers and so
// sizes for the rank count. With no tasks it returns before doing even that.
TEST_F(RmaCePutLaunchTest, PutLaunch_NonPersistentPlanWithNoTasks_TakesNonPersistentPath) {
  plan_->persistent = false;

  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  EXPECT_TRUE(initCapacities_.empty());
}

// Same path, now with work, so the sizing it uses is visible.
TEST_F(RmaCePutLaunchTest, PutLaunch_NonPersistentPlanWithTasks_SizesBatchesPerRank) {
  plan_->persistent = false;
  args_.nRmaTasksCe = 1;
  // Pool-allocated, because the unit returns it to the pool when it is done.
  auto* task = ncclMemoryPoolAlloc<ncclTaskRma>(&comm_->memPool_ncclTaskRma, &comm_->memPermanent);
  task->peer = 0;
  ncclIntruQueueEnqueue(&plan_->rmaTaskQueueCe, task);

  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  EXPECT_EQ(initCapacities_, (std::vector<int>{kNRanks, kNRanks}));
}

// A failure inside the chosen path is the dispatcher's result; it does not
// swallow it or substitute one of its own.
TEST_F(RmaCePutLaunchTest, PutLaunch_ChosenPathFails_PropagatesUnchanged) {
  plan_->persistent = true;
  g_ceInitBatchOpsParams = [](ncclCeBatchOpsParams*, int) { return ncclSystemError; };

  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSystemError);
}

// ---------------------------------------------------------------------------
// ncclRmaCePutLaunchNonPersist
// ---------------------------------------------------------------------------

// What the unit submits, in order. The batch params are reused between rounds,
// so the ops are snapshotted at launch rather than inspected afterwards.
struct SubmittedOp {
  void* src;
  void* dst;
  size_t size;
};
// One entry of a stream batch-memory-op submission.
struct MemOp {
  unsigned operation;
  const void* address;
  uint64_t value;
  unsigned flags;   // the wait predicate; GEQ vs EQ is not otherwise observable
};
struct Submission {
  enum Kind { kMemOps, kBatch };
  Kind kind;
  std::vector<SubmittedOp> ops;    // kBatch
  std::vector<MemOp> memOps;       // kMemOps
};

// Drives one non-persistent put launch and records everything it enqueued.
// Tasks are grouped by peer and issued a round at a time, one task per peer per
// round, so the shape of this log is the unit's contract.
class RmaCeNonPersistTest : public RmaCeLaunchFixture {
protected:
  std::vector<Submission> log_;
  // The window tasks name for their data. Built by hand -- nothing in this unit
  // creates it -- in the shape a registration produces, so the real
  // ncclDevrGetLsaRankPtr resolves against it.
  ncclDevrWindow dataWin_{};
  std::vector<void*> dataPeerPtrs_;
  std::vector<uint64_t> peerData_;

  // Rank r's image inside the data window.
  uint64_t* PeerDataByLsa(int lsaRank) { return peerData_.data() + lsaRank * kWinSlots; }
  uint64_t* PeerData(int peerWorldRank) { return PeerDataByLsa(LsaOf(peerWorldRank)); }
  std::vector<uint64_t> srcBuf_;
  // Every task PushTask handed to the plan, so a test can check which of them
  // the unit returned to the pool.
  std::vector<ncclTaskRma*> pushed_;

  // The pool's free list, head first. Freed cells reuse the object's storage and
  // are chained through the leading next pointer, so each cell is the task that
  // was returned.
  std::vector<const void*> PoolFreeList() const {
    std::vector<const void*> out;
    for (auto* cell = comm_->memPool_ncclTaskRma.head; cell != nullptr; cell = cell->next) {
      out.push_back(cell);
    }
    return out;
  }

  void SetUp() override {
    RmaCeLaunchFixture::SetUp();
    srcBuf_.assign(8, 0);
    plan_->persistent = false;

    // Signal lookups resolve against the window ncclRmaCeInit registered, whose
    // peer table the fixture already points at PeerWin(r). Data lookups resolve
    // against this one.
    peerData_.assign(static_cast<size_t>(kLsaSize) * kWinSlots, 0);
    dataPeerPtrs_.resize(kLsaSize);
    for (int r = 0; r < kLsaSize; r++) dataPeerPtrs_[r] = PeerDataByLsa(r);
    dataWin_.userPtr = peerData_.data();
    dataWin_.size = static_cast<size_t>(kWinSlots) * sizeof(uint64_t);
    dataWin_.ipcPeerPtrs = dataPeerPtrs_.data();
    dataWin_.ipcPeerCount = kLsaSize;
    g_ceLaunchBatchOps = [this](ncclComm*, ncclCeBatchOpsParams* p, hipStream_t,
                                ncclCeCollArgs*) {
      Submission s{Submission::kBatch, {}, {}};
      for (size_t i = 0; i < p->numOps; i++) s.ops.push_back({p->srcs[i], p->dsts[i], p->sizes[i]});
      log_.push_back(std::move(s));
      return ncclSuccess;
    };
    g_cuStreamBatchMemOp = [this](hipStream_t, unsigned int numOps,
                                  hipStreamBatchMemOpParams* ops) {
      Submission s{Submission::kMemOps, {}, {}};
      for (unsigned int i = 0; i < numOps; i++) {
        // The wait and write forms share a layout, so one read covers both.
        s.memOps.push_back({ops[i].writeValue.operation,
                            reinterpret_cast<const void*>(ops[i].writeValue.address),
                            ops[i].writeValue.value64,
                            ops[i].writeValue.flags});
      }
      log_.push_back(std::move(s));
      return ncclSuccess;
    };
  }

  // Queue one CE task. bytes == 0 means signal-only; signal == false means data-only.
  void PushTask(int peer, size_t bytes, bool signal, size_t winOffset = 0, int signalIdx = 0,
                ncclDataType_t datatype = ncclUint8, int ctx = 0) {
    auto* t = ncclMemoryPoolAlloc<ncclTaskRma>(&comm_->memPool_ncclTaskRma, &comm_->memPermanent);
    pushed_.push_back(t);
    t->peer = peer;
    // count is an element count; with the ncclUint8 default it is the byte count.
    t->count = bytes / ncclTypeSize(datatype);
    t->datatype = datatype;
    t->srcBuff = srcBuf_.data();
    t->peerWinHost = &dataWin_;
    t->peerWinOffset = winOffset;
    t->signalMode = signal ? NCCL_SIGNAL : NCCL_SIGNAL_NONE;
    t->signalIdx = signalIdx;
    t->ctx = ctx;
    ncclIntruQueueEnqueue(&plan_->rmaTaskQueueCe, t);
    args_.nRmaTasksCe++;
  }

  // The batches in submission order, ignoring the staging writes.
  std::vector<std::vector<SubmittedOp>> Batches() const {
    std::vector<std::vector<SubmittedOp>> out;
    for (const auto& s : log_) {
      if (s.kind == Submission::kBatch) out.push_back(s.ops);
    }
    return out;
  }
};

// A data-carrying task becomes one copy from the task's own buffer to the peer
// address resolved for it, sized by count and datatype.
TEST_F(RmaCeNonPersistTest, NonPersist_OneDataTask_CopiesTaskBufferToResolvedPeer) {
  // Non-zero window offset, so the destination is pinned to the task's offset
  // rather than just to the peer.
  PushTask(/*peer=*/2, /*bytes=*/64, /*signal=*/false, /*winOffset=*/8);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto batches = Batches();
  ASSERT_EQ(batches.size(), 2u);           // data batch, then signal batch
  ASSERT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[0][0].src, srcBuf_.data());
  EXPECT_EQ(batches[0][0].dst, reinterpret_cast<char*>(PeerData(2)) + 8);
  EXPECT_EQ(batches[0][0].size, 64u);
  EXPECT_TRUE(batches[1].empty());         // nothing signalled
}

// Tasks for different peers travel together: one round, one op per peer.
TEST_F(RmaCeNonPersistTest, NonPersist_TasksForDifferentPeers_BatchedIntoOneRound) {
  PushTask(/*peer=*/1, 32, /*signal=*/true);
  PushTask(/*peer=*/3, 48, /*signal=*/true);
  ncclRmaCeCtx* ceCtx = Ctx(0);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto batches = Batches();
  ASSERT_EQ(batches.size(), 2u);
  ASSERT_EQ(batches[0].size(), 2u);
  EXPECT_EQ(batches[0][0].dst, PeerData(1));
  EXPECT_EQ(batches[0][1].dst, PeerData(3));

  // Both peers signal in the same round, so each takes its own staging slot --
  // sharing one would make the second overwrite the first. Pinned on the staging
  // writes themselves as well as on the copies that read them back.
  ASSERT_EQ(log_[0].kind, Submission::kMemOps);
  ASSERT_EQ(log_[0].memOps.size(), 2u);
  EXPECT_EQ(log_[0].memOps[0].address, &ceCtx->signalOpSeqsDev[0]);
  EXPECT_EQ(log_[0].memOps[1].address, &ceCtx->signalOpSeqsDev[1]);
  ASSERT_EQ(batches[1].size(), 2u);
  EXPECT_EQ(batches[1][0].src, &ceCtx->signalOpSeqsDev[0]);
  EXPECT_EQ(batches[1][1].src, &ceCtx->signalOpSeqsDev[1]);
  EXPECT_EQ(batches[1][0].dst, SelfSlotIn(1, ceCtx->signalOffset));
  EXPECT_EQ(batches[1][1].dst, SelfSlotIn(3, ceCtx->signalOffset));
}

// Two tasks for the same peer cannot share a batch, because a batched copy does
// not order its own operations. They are issued a round apart instead.
TEST_F(RmaCeNonPersistTest, NonPersist_TasksForSamePeer_IssuedInSeparateRounds) {
  PushTask(/*peer=*/1, 32, false);
  PushTask(/*peer=*/1, 48, false);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto batches = Batches();
  ASSERT_EQ(batches.size(), 4u);           // two rounds of (data, signal)
  ASSERT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[0][0].size, 32u);
  ASSERT_EQ(batches[2].size(), 1u);
  EXPECT_EQ(batches[2][0].size, 48u);
}

// The sequence number is staged to device memory before the batch that copies it
// onward, because the copy reads the staged slot.
TEST_F(RmaCeNonPersistTest, NonPersist_SignallingTask_StagesSequenceBeforeCopyingIt) {
  PushTask(/*peer=*/2, 16, /*signal=*/true);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  ASSERT_EQ(log_.size(), 3u);
  EXPECT_EQ(log_[0].kind, Submission::kMemOps);
  // The staging op is a write to this peer's staging slot -- not a wait, and not
  // some other slot; the signal copy below reads exactly this address.
  ASSERT_EQ(log_[0].memOps.size(), 1u);
  EXPECT_EQ(log_[0].memOps[0].operation, hipStreamMemOpWriteValue64);
  EXPECT_EQ(log_[0].memOps[0].address, &Ctx(0)->signalOpSeqsDev[0]);
  EXPECT_EQ(log_[1].kind, Submission::kBatch);   // data
  EXPECT_EQ(log_[2].kind, Submission::kBatch);   // signal
  ASSERT_EQ(log_[2].ops.size(), 1u);
  EXPECT_EQ(log_[2].ops[0].dst, SelfSlotIn(2, Ctx(0)->signalOffset));
  // Copied from the slot that was just staged, not from a constant.
  EXPECT_EQ(log_[2].ops[0].src, &Ctx(0)->signalOpSeqsDev[0]);
  EXPECT_EQ(log_[2].ops[0].size, sizeof(uint64_t));
}

// The sequence a peer is signalled with advances per round, so a receiver can
// tell a repeated signal from a new one.
TEST_F(RmaCeNonPersistTest, NonPersist_RepeatedSignalsToSamePeer_AdvanceTheSequence) {
  PushTask(/*peer=*/1, 0, true);
  PushTask(/*peer=*/1, 0, true);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  std::vector<uint64_t> staged;
  for (const auto& s : log_) {
    for (const auto& op : s.memOps) staged.push_back(op.value);
  }
  EXPECT_EQ(staged, (std::vector<uint64_t>{1, 2}));
}

// The context a task names is the one whose signal buffer it raises. With one
// configured context every selection collapses to context 0, so this configures
// two and signals through the second.
TEST_F(RmaCeNonPersistTest, NonPersist_TaskNamesSecondContext_SignalsThatContext) {
  ASSERT_EQ(ncclRmaCeFinalize(comm_.get()), ncclSuccess);
  comm_->config.numRmaCtx = 2;
  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);
  PushTask(/*peer=*/1, 0, /*signal=*/true, /*winOffset=*/0, /*signalIdx=*/0,
           ncclUint8, /*ctx=*/1);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  // The staged sequence is bumped on context 1, and context 0 is untouched.
  ASSERT_EQ(log_.size(), 3u);
  ASSERT_EQ(log_[2].ops.size(), 1u);
  EXPECT_EQ(log_[2].ops[0].src, &Ctx(1)->signalOpSeqsDev[0]);
  EXPECT_EQ(Ctx(1)->signalOpSeqs[SignalSlot(0, 1)], 1u);
  EXPECT_EQ(Ctx(0)->signalOpSeqs[SignalSlot(0, 1)], 0u);
}

// count is an element count, not a byte count: the unit multiplies by the
// datatype's size. With the uint8 default the two coincide, so this uses floats.
TEST_F(RmaCeNonPersistTest, NonPersist_WideDatatype_CopiesElementCountTimesElementSize) {
  PushTask(/*peer=*/1, /*bytes=*/64, /*signal=*/false, /*winOffset=*/0, /*signalIdx=*/0,
           ncclFloat32);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto batches = Batches();
  ASSERT_EQ(batches.size(), 2u);
  ASSERT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[0][0].size, 16u * sizeof(float));   // 16 elements, not 16 bytes
}

// A task carrying no bytes is signal-only: nothing is copied for it.
TEST_F(RmaCeNonPersistTest, NonPersist_ZeroByteTask_EnqueuesNoDataCopy) {
  PushTask(/*peer=*/2, /*bytes=*/0, /*signal=*/true);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto batches = Batches();
  ASSERT_EQ(batches.size(), 2u);
  EXPECT_TRUE(batches[0].empty());        // no data op
  EXPECT_EQ(batches[1].size(), 1u);       // signal still sent
}

// Bailing out mid-round leaves other peers' tasks still queued. They are
// pool-allocated, so the cleanup path returns them rather than leaking them.
TEST_F(RmaCeNonPersistTest, NonPersist_FailsMidRound_ReturnsQueuedTasksToThePool) {
  PushTask(/*peer=*/1, 32, false);
  PushTask(/*peer=*/1, 32, false);
  PushTask(/*peer=*/2, 32, false);
  PushTask(/*peer=*/2, 32, false);
  // Fail while resolving the second peer of the first round, so both peers still
  // hold their second task.
  // Peer 2 has no mapping, so resolving it is the second lookup of the round
  // and the one that fails.
  dataPeerPtrs_[LsaOf(2)] = nullptr;

  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclInternalError);

  // All four come back, not just the two the unit had already finished with.
  // The pool starts empty, so the free list is exactly what was returned; the
  // size check is what makes dropping the cleanup loop visible.
  auto freeList = PoolFreeList();
  ASSERT_EQ(freeList.size(), 4u);
  for (size_t i = 0; i < pushed_.size(); i++) {
    EXPECT_NE(std::find(freeList.begin(), freeList.end(), pushed_[i]), freeList.end())
        << "task " << i << " was not returned to the pool";
  }
}

// An unresolvable peer address is rejected rather than copied into.
TEST_F(RmaCeNonPersistTest, NonPersist_PeerAddressUnresolved_ReturnsInvalidArgument) {
  // No peer table, and a flat base of zero with nothing to scale it by: the
  // lookup succeeds and hands back a null address, which is the case
  // production's own null check exists for. bigSize has done its job (keeping
  // ncclDevrInitOnce a no-op) by the time the launch runs.
  dataWin_.ipcPeerPtrs = nullptr;
  comm_->devrState.bigSize = 0;
  PushTask(/*peer=*/1, 32, false);

  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclInvalidArgument);
}

// The signal region is indexed by (signalIdx, rank), not by rank alone, so a
// task on a non-zero signal index targets a slot one whole rank-stride further
// in -- both in the peer address written to and in the sequence counter bumped.
TEST_F(RmaCeNonPersistTest, NonPersist_NonZeroSignalIndex_TargetsTheIndexedSlot) {
  constexpr int kSigIdx = 1;
  constexpr int kPeer = 2;
  PushTask(kPeer, 16, /*signal=*/true, /*winOffset=*/0, kSigIdx);
  ncclRmaCeCtx* ceCtx = Ctx(0);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  ASSERT_EQ(log_.size(), 3u);
  ASSERT_EQ(log_[2].ops.size(), 1u);
  EXPECT_EQ(log_[2].ops[0].dst, SelfSlotIn(kPeer, ceCtx->signalOffset, kSigIdx));
  // And the counter bumped is the one for (kSigIdx, peer), not the one for peer.
  EXPECT_EQ(ceCtx->signalOpSeqs[SignalSlot(kSigIdx, kPeer)], 1u);
  EXPECT_EQ(ceCtx->signalOpSeqs[LsaOf(kPeer)], 0u);   // index 0 is a different slot
}

// A peer outside the LSA team cannot be resolved to a local rank, so the launch
// reports rather than indexing the signal region with a bogus slot.
TEST_F(RmaCeNonPersistTest, NonPersist_PeerOutsideLsaTeam_Propagates) {
  PushTask(/*peer=*/5, 32, /*signal=*/false);   // 5 is in nRanks but not in the LSA team

  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclInternalError);
}

// Sizing the batch params is the first thing the round loop needs; a failure
// there aborts before anything is submitted.
TEST_F(RmaCeNonPersistTest, NonPersist_BatchParamsSizingFails_Propagates) {
  PushTask(/*peer=*/1, 32, /*signal=*/false);
  ScopedHook init(g_ceInitBatchOpsParams,
                  [](ncclCeBatchOpsParams*, int) { return ncclSystemError; });

  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSystemError);
  EXPECT_TRUE(log_.empty());
}

// The staging writes go on the stream before the copies; if that submission
// fails the round stops there.
TEST_F(RmaCeNonPersistTest, NonPersist_StagingSubmitFails_Propagates) {
  PushTask(/*peer=*/1, 32, /*signal=*/true);
  ScopedHook memop(g_cuStreamBatchMemOp,
                   [](hipStream_t, unsigned int, hipStreamBatchMemOpParams*) { return ncclSystemError; });

  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSystemError);
  EXPECT_TRUE(Batches().empty());   // never reached the data batch
}

// ...and a failing data batch stops the round with the staging already on the
// stream, which is the same partial-submission shape the wait path has.
TEST_F(RmaCeNonPersistTest, NonPersist_DataBatchFails_Propagates) {
  PushTask(/*peer=*/1, 32, /*signal=*/false);
  ScopedHook launch(g_ceLaunchBatchOps,
                    [](ncclComm*, ncclCeBatchOpsParams*, hipStream_t, ncclCeCollArgs*) {
                      return ncclSystemError;
                    });

  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSystemError);
}

// ---------------------------------------------------------------------------
// ncclRmaCePutLaunchPersist
// ---------------------------------------------------------------------------

// The graph-captured sibling of the non-persistent path. Same recording fixture;
// only the plan differs. It issues one batch pair per task in queue order rather
// than grouping by peer into rounds, and it signals differently: an ack
// handshake before the copies, and a device-resident constant written to the
// graph signal slot instead of a staged sequence.
class RmaCePersistTest : public RmaCeNonPersistTest {
protected:
  void SetUp() override {
    RmaCeNonPersistTest::SetUp();
    plan_->persistent = true;
  }
};

// Data movement is the same as the non-persistent path: the task's own buffer to
// the address resolved for its peer and window offset.
TEST_F(RmaCePersistTest, Persist_OneDataTask_CopiesTaskBufferToResolvedPeer) {
  PushTask(/*peer=*/2, /*bytes=*/64, /*signal=*/false, /*winOffset=*/8);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto batches = Batches();
  ASSERT_EQ(batches.size(), 2u);
  ASSERT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[0][0].src, srcBuf_.data());
  EXPECT_EQ(batches[0][0].dst, reinterpret_cast<char*>(PeerData(2)) + 8);
  EXPECT_EQ(batches[0][0].size, 64u);
  EXPECT_TRUE(batches[1].empty());         // nothing signalled
  EXPECT_EQ(PoolFreeList().size(), 1u);    // the task is returned, not leaked
}

// Under graph capture the sender waits for the receiver's ack and clears it
// before writing, so a replayed graph cannot outrun the receiver. The handshake
// is submitted before the copies it guards.
TEST_F(RmaCePersistTest, Persist_SignallingTask_WaitsForAckAndClearsItBeforeCopying) {
  PushTask(/*peer=*/3, /*bytes=*/32, /*signal=*/true);
  const void* ackAddr = &Ctx(0)->graphAckDev[SignalSlot(0, 3)];

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  ASSERT_EQ(log_.size(), 3u);   // ack handshake, data batch, signal batch
  ASSERT_EQ(log_[0].kind, Submission::kMemOps);
  ASSERT_EQ(log_[0].memOps.size(), 2u);
  EXPECT_EQ(log_[0].memOps[0].operation, hipStreamMemOpWaitValue64);
  EXPECT_EQ(log_[0].memOps[0].address, ackAddr);
  EXPECT_EQ(log_[0].memOps[0].value, 1u);
  EXPECT_EQ(log_[0].memOps[1].operation, hipStreamMemOpWriteValue64);
  EXPECT_EQ(log_[0].memOps[1].address, ackAddr);
  EXPECT_EQ(log_[0].memOps[1].value, 0u);
  // ...and the copies follow it.
  EXPECT_EQ(log_[1].kind, Submission::kBatch);
}

// The graph signal is a copy of a device-resident constant into the peer's graph
// signal slot -- a fixed value, not the advancing sequence the non-graph path
// uses, because a captured graph replays the same ops every time.
TEST_F(RmaCePersistTest, Persist_SignallingTask_CopiesDeviceConstantToGraphSignalSlot) {
  PushTask(/*peer=*/1, /*bytes=*/0, /*signal=*/true);
  ncclRmaCeCtx* ceCtx = Ctx(0);
  const char* expectedDst = SelfSlotIn(1, ceCtx->graphSignalOffset);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto batches = Batches();
  ASSERT_EQ(batches.size(), 2u);
  ASSERT_EQ(batches[1].size(), 1u);
  EXPECT_EQ(batches[1][0].src, ceCtx->signalConstOneDev);
  EXPECT_EQ(batches[1][0].dst, expectedDst);
  EXPECT_EQ(batches[1][0].size, sizeof(uint64_t));
}

// A task that signals nothing needs no handshake, so none is submitted.
TEST_F(RmaCePersistTest, Persist_NonSignallingTask_SubmitsNoAckHandshake) {
  PushTask(/*peer=*/2, /*bytes=*/32, /*signal=*/false);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  for (const auto& s : log_) EXPECT_NE(s.kind, Submission::kMemOps);
}

// One batch pair per task, in queue order, with no grouping across peers. Two
// different peers is what shows it: the non-persistent path batches both into
// one round, this one issues a pair each. Same-peer input cannot tell them
// apart -- both emit four batches.
TEST_F(RmaCePersistTest, Persist_TasksForDifferentPeers_IssueOneBatchPairEach) {
  PushTask(/*peer=*/1, 32, false);
  PushTask(/*peer=*/3, 48, false);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto batches = Batches();
  ASSERT_EQ(batches.size(), 4u);
  ASSERT_EQ(batches[0].size(), 1u);        // one op, not both peers batched
  EXPECT_EQ(batches[0][0].size, 32u);
  ASSERT_EQ(batches[2].size(), 1u);
  EXPECT_EQ(batches[2][0].size, 48u);
}

// A task carrying no bytes is signal-only here too.
TEST_F(RmaCePersistTest, Persist_ZeroByteTask_EnqueuesNoDataCopy) {
  PushTask(/*peer=*/2, /*bytes=*/0, /*signal=*/true);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto batches = Batches();
  ASSERT_EQ(batches.size(), 2u);
  EXPECT_TRUE(batches[0].empty());
  EXPECT_EQ(batches[1].size(), 1u);
}

// An unresolvable peer address is rejected rather than copied into.
TEST_F(RmaCePersistTest, Persist_PeerAddressUnresolved_ReturnsInvalidArgument) {
  // No peer table, and a flat base of zero with nothing to scale it by: the
  // lookup succeeds and hands back a null address, which is the case
  // production's own null check exists for. bigSize has done its job (keeping
  // ncclDevrInitOnce a no-op) by the time the launch runs.
  dataWin_.ipcPeerPtrs = nullptr;
  comm_->devrState.bigSize = 0;
  PushTask(/*peer=*/1, 32, false);

  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclInvalidArgument);
}

// The persistent arm indexes by (signalIdx, rank) too: the ack slot it waits on
// and the peer signal slot it writes both shift by a whole rank-stride.
TEST_F(RmaCePersistTest, Persist_NonZeroSignalIndex_UsesTheIndexedAckAndSignalSlots) {
  constexpr int kSigIdx = 1;
  constexpr int kPeer = 2;
  PushTask(kPeer, 16, /*signal=*/true, /*winOffset=*/0, kSigIdx);
  ncclRmaCeCtx* ceCtx = Ctx(0);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  // The ack handshake waits on the peer's slot for this signal index.
  ASSERT_EQ(log_.size(), 3u);
  ASSERT_EQ(log_[0].kind, Submission::kMemOps);
  ASSERT_EQ(log_[0].memOps.size(), 2u);
  EXPECT_EQ(log_[0].memOps[0].address, &ceCtx->graphAckDev[SignalSlot(kSigIdx, kPeer)]);

  // The signal write lands at this rank's slot for this index in the peer's window.
  auto batches = Batches();
  ASSERT_EQ(batches.size(), 2u);
  ASSERT_EQ(batches[1].size(), 1u);
  const char* expected = SelfSlotIn(kPeer, ceCtx->graphSignalOffset, kSigIdx);
  EXPECT_EQ(batches[1][0].dst, expected);
}

// Same rejection on the graph path: an unresolvable peer stops the launch.
TEST_F(RmaCePersistTest, Persist_PeerOutsideLsaTeam_Propagates) {
  PushTask(/*peer=*/5, 32, /*signal=*/false);

  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclInternalError);
}

// The ack handshake is submitted before the copies, so a failure there leaves
// the task's data unsent rather than half-sent.
TEST_F(RmaCePersistTest, Persist_AckHandshakeFails_Propagates) {
  PushTask(/*peer=*/1, 32, /*signal=*/true);
  ScopedHook memop(g_cuStreamBatchMemOp,
                   [](hipStream_t, unsigned int, hipStreamBatchMemOpParams*) { return ncclSystemError; });

  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSystemError);
  EXPECT_TRUE(Batches().empty());
}

// A failing data batch propagates rather than continuing to the signal batch.
TEST_F(RmaCePersistTest, Persist_DataBatchFails_Propagates) {
  PushTask(/*peer=*/1, 32, /*signal=*/false);
  ScopedHook launch(g_ceLaunchBatchOps,
                    [](ncclComm*, ncclCeBatchOpsParams*, hipStream_t, ncclCeCollArgs*) {
                      return ncclSystemError;
                    });

  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSystemError);
}

// ---------------------------------------------------------------------------
// ncclRmaCeWaitLaunch
// ---------------------------------------------------------------------------

// Consumes the plan's single WaitSignal task and makes the stream wait until each
// named peer has signalled. The two plan kinds wait very differently, so the
// recording fixture is reused and the submissions are the assertions.
class RmaCeWaitLaunchTest : public RmaCeNonPersistTest {
protected:
  std::vector<int> peers_;
  std::vector<int> nsignals_;
  std::vector<int> signalIdxs_;

  // The plan carries exactly one WaitSignal task naming the peers to wait on,
  // how many signals to expect from each, and which signal index each uses.
  // signalIdxs defaults to index 0 for every peer.
  void PushWaitTask(std::vector<int> peers, std::vector<int> nsignals,
                    std::vector<int> signalIdxs = {}) {
    peers_ = std::move(peers);
    nsignals_ = std::move(nsignals);
    signalIdxs_ = signalIdxs.empty() ? std::vector<int>(peers_.size(), 0)
                                     : std::move(signalIdxs);
    auto* t = ncclMemoryPoolAlloc<ncclTaskRma>(&comm_->memPool_ncclTaskRma, &comm_->memPermanent);
    t->func = ncclFuncWaitSignal;
    t->ctx = 0;
    t->signalMode = NCCL_SIGNAL;
    t->npeers = static_cast<int>(peers_.size());
    t->peers = peers_.data();
    t->nsignals = nsignals_.data();
    t->signalIdxs = signalIdxs_.data();
    ncclIntruQueueEnqueue(&plan_->rmaTaskQueueCe, t);
    args_.nRmaTasksCe = 1;
  }

  // Every stream memory operation submitted, flattened across submissions.
  std::vector<MemOp> AllMemOps() const {
    std::vector<MemOp> out;
    for (const auto& s : log_) out.insert(out.end(), s.memOps.begin(), s.memOps.end());
    return out;
  }
};

// Outside graph capture the wait is a single batch with one threshold per peer,
// so the stream blocks once rather than per signal.
TEST_F(RmaCeWaitLaunchTest, WaitLaunch_NonPersistent_WaitsOncePerPeerInOneBatch) {
  PushWaitTask({1, 3}, {2, 5});
  ncclRmaCeCtx* ceCtx = Ctx(0);

  ASSERT_EQ(ncclRmaCeWaitLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  ASSERT_EQ(log_.size(), 1u);
  ASSERT_EQ(log_[0].memOps.size(), 2u);
  EXPECT_EQ(log_[0].memOps[0].operation, hipStreamMemOpWaitValue64);
  EXPECT_EQ(log_[0].memOps[0].address, &ceCtx->signalsDev[SignalSlot(0, 1)]);
  EXPECT_EQ(log_[0].memOps[0].value, 2u);
  // GEQ, not EQ: the running total can be overshot by design, and an exact
  // match would hang.
  EXPECT_EQ(log_[0].memOps[0].flags, unsigned(hipStreamWaitValueGte));
  EXPECT_EQ(log_[0].memOps[1].address, &ceCtx->signalsDev[SignalSlot(0, 3)]);
  EXPECT_EQ(log_[0].memOps[1].value, 5u);
  EXPECT_EQ(PoolFreeList().size(), 1u);   // the wait task is returned, not leaked
}

// The peer's signal counter only ever rises, so each wait threshold is the running
// total. Waiting for the same absolute value twice would pass immediately the
// second time and drop the wait.
TEST_F(RmaCeWaitLaunchTest, WaitLaunch_NonPersistentRepeated_ThresholdsAccumulate) {
  PushWaitTask({1}, {2});
  ASSERT_EQ(ncclRmaCeWaitLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);
  PushWaitTask({1}, {3});
  ASSERT_EQ(ncclRmaCeWaitLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto ops = AllMemOps();
  ASSERT_EQ(ops.size(), 2u);
  EXPECT_EQ(ops[0].value, 2u);
  EXPECT_EQ(ops[1].value, 5u);   // 2 + 3, not 3
}

// Under capture the graph replays unchanged, so a running total cannot be baked
// in. Each expected signal becomes its own wait-reset-ack cycle on a slot the
// sender re-raises every replay.
TEST_F(RmaCeWaitLaunchTest, WaitLaunch_Persistent_RunsOneWaitResetAckCyclePerSignal) {
  plan_->persistent = true;
  PushWaitTask({2}, {2});
  ncclRmaCeCtx* ceCtx = Ctx(0);

  ASSERT_EQ(ncclRmaCeWaitLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  // Two cycles, each a (wait, reset) memop pair followed by the ack copy.
  ASSERT_EQ(log_.size(), 4u);
  for (int cycle = 0; cycle < 2; cycle++) {
    const Submission& memops = log_[cycle * 2];
    ASSERT_EQ(memops.kind, Submission::kMemOps) << "cycle " << cycle;
    ASSERT_EQ(memops.memOps.size(), 2u) << "cycle " << cycle;
    EXPECT_EQ(memops.memOps[0].operation, hipStreamMemOpWaitValue64);
    EXPECT_EQ(memops.memOps[0].address, &ceCtx->graphSignalsDev[SignalSlot(0, 2)]);
    EXPECT_EQ(memops.memOps[0].value, 1u);
    EXPECT_EQ(memops.memOps[1].operation, hipStreamMemOpWriteValue64);
    // Reset targets the slot just waited on, so the next replay waits again.
    EXPECT_EQ(memops.memOps[1].address, &ceCtx->graphSignalsDev[SignalSlot(0, 2)]);
    EXPECT_EQ(memops.memOps[1].value, 0u);
    EXPECT_EQ(log_[cycle * 2 + 1].kind, Submission::kBatch);
  }
}

// The ack tells the sender it may proceed, and is written into the sender's slot
// of the peer's ack region.
TEST_F(RmaCeWaitLaunchTest, WaitLaunch_Persistent_AcksSenderSlotWithDeviceConstant) {
  plan_->persistent = true;
  PushWaitTask({1}, {1});
  ncclRmaCeCtx* ceCtx = Ctx(0);
  const char* expectedAck = SelfSlotIn(1, ceCtx->graphAckOffset);

  ASSERT_EQ(ncclRmaCeWaitLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto batches = Batches();
  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[0][0].src, ceCtx->signalConstOneDev);
  EXPECT_EQ(batches[0][0].dst, expectedAck);
  EXPECT_EQ(batches[0][0].size, sizeof(uint64_t));
}

// A task that is not in signalling mode has nothing to wait for.
TEST_F(RmaCeWaitLaunchTest, WaitLaunch_SignalModeNone_SubmitsNothing) {
  PushWaitTask({1}, {2});
  ncclIntruQueueHead(&plan_->rmaTaskQueueCe)->signalMode = NCCL_SIGNAL_NONE;

  ASSERT_EQ(ncclRmaCeWaitLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  EXPECT_TRUE(log_.empty());
}

// DEFECT PINNED, not endorsed: the graph arm submits each cycle's wait before
// resolving the ack address, so a failure returns with waits already on the
// stream -- the same hazard rma.cc hits when it skips its closing join.
// Fixing the ordering will fail this test, which is the point: it changes with
// the fix. Both are written up on #11581; neither has a Jira ticket yet.
TEST_F(RmaCeWaitLaunchTest, WaitLaunch_PersistentAckResolutionFails_LeavesWaitsOnTheStream) {
  plan_->persistent = true;
  PushWaitTask({1}, {1});
  // The ack is written to the peer's slot of the signal window; with no mapping
  // for that peer the lookup fails, after the wait pair is already submitted.
  SignalWin()->ipcPeerPtrs[LsaOf(1)] = nullptr;

  EXPECT_EQ(ncclRmaCeWaitLaunchUut(comm_.get(), plan_.get(), nullptr), ncclInternalError);

  // The wait-reset pair was already submitted when the failure surfaced.
  ASSERT_EQ(log_.size(), 1u);
  EXPECT_EQ(log_[0].kind, Submission::kMemOps);
  EXPECT_EQ(log_[0].memOps.size(), 2u);
}
// The wait side reads the same (signalIdx, rank) layout, and each peer carries
// its own index, so two peers on different indices wait on slots a rank-stride
// apart rather than on adjacent ones.
TEST_F(RmaCeWaitLaunchTest, WaitLaunch_PerPeerSignalIndex_WaitsOnTheIndexedSlots) {
  PushWaitTask({1, 3}, {2, 5}, /*signalIdxs=*/{0, 1});
  ncclRmaCeCtx* ceCtx = Ctx(0);

  ASSERT_EQ(ncclRmaCeWaitLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  ASSERT_EQ(log_.size(), 1u);
  ASSERT_EQ(log_[0].memOps.size(), 2u);
  EXPECT_EQ(log_[0].memOps[0].address, &ceCtx->signalsDev[SignalSlot(0, 1)]);
  EXPECT_EQ(log_[0].memOps[1].address, &ceCtx->signalsDev[SignalSlot(1, 3)]);
}

// Same for the persistent wait arm: the graph signal it waits on and the ack it
// writes back are both located by (signalIdx, rank).
// Two peers on different signal indices, so the per-peer subscript varies: with
// one peer every read of signalIdxs[i] is signalIdxs[0] and a wrong index is
// invisible. Peer 1 expects one signal, peer 3 expects two, so the second peer
// also runs two cycles.
TEST_F(RmaCeWaitLaunchTest, WaitLaunch_PersistentPerPeerSignalIndex_UsesTheIndexedSlots) {
  plan_->persistent = true;
  PushWaitTask({1, 3}, {1, 2}, /*signalIdxs=*/{0, 1});
  ncclRmaCeCtx* ceCtx = Ctx(0);

  ASSERT_EQ(ncclRmaCeWaitLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  // Three (wait, reset) + ack cycles: one for peer 1, two for peer 3.
  ASSERT_EQ(log_.size(), 6u);
  struct Expect { int peer; int sigIdx; };
  const Expect expected[3] = {{1, 0}, {3, 1}, {3, 1}};
  for (int c = 0; c < 3; c++) {
    const Submission& memops = log_[c * 2];
    ASSERT_EQ(memops.kind, Submission::kMemOps) << "cycle " << c;
    ASSERT_EQ(memops.memOps.size(), 2u) << "cycle " << c;
    EXPECT_EQ(memops.memOps[0].address,
              &ceCtx->graphSignalsDev[SignalSlot(expected[c].sigIdx, expected[c].peer)])
        << "cycle " << c;
    ASSERT_EQ(log_[c * 2 + 1].ops.size(), 1u) << "cycle " << c;
    EXPECT_EQ(log_[c * 2 + 1].ops[0].dst,
              SelfSlotIn(expected[c].peer, ceCtx->graphAckOffset, expected[c].sigIdx))
        << "cycle " << c;
  }
}

// The wait path accepts exactly one WaitSignal task. A task of any other kind is
// rejected and returned to the pool rather than interpreted as a wait.
TEST_F(RmaCeWaitLaunchTest, WaitLaunch_TaskIsNotAWait_ReportsInvalidTask) {
  PushWaitTask({1}, {1});
  ncclIntruQueueHead(&plan_->rmaTaskQueueCe)->func = ncclFuncPutSignal;

  EXPECT_EQ(ncclRmaCeWaitLaunchUut(comm_.get(), plan_.get(), nullptr), ncclInternalError);
  EXPECT_TRUE(log_.empty());
  // Rejected, not leaked: the one task this pushed comes back to the pool.
  EXPECT_EQ(PoolFreeList().size(), 1u);
}

// ...and exactly one: a plan claiming a different CE task count is rejected too,
// since the path only ever dequeues the head.
TEST_F(RmaCeWaitLaunchTest, WaitLaunch_TaskCountIsNotOne_ReportsInvalidTask) {
  PushWaitTask({1}, {1});
  args_.nRmaTasksCe = 2;

  EXPECT_EQ(ncclRmaCeWaitLaunchUut(comm_.get(), plan_.get(), nullptr), ncclInternalError);
  EXPECT_TRUE(log_.empty());
  // Rejected, not leaked: the one task this pushed comes back to the pool.
  EXPECT_EQ(PoolFreeList().size(), 1u);
}

// An unresolvable peer stops the wait before any threshold is computed.
TEST_F(RmaCeWaitLaunchTest, WaitLaunch_PeerOutsideLsaTeam_Propagates) {
  PushWaitTask({5}, {1});

  EXPECT_EQ(ncclRmaCeWaitLaunchUut(comm_.get(), plan_.get(), nullptr), ncclInternalError);
  EXPECT_TRUE(log_.empty());
}

// The batched arm puts one wait per peer on the stream in a single submission;
// a failure there reports and submits nothing.
TEST_F(RmaCeWaitLaunchTest, WaitLaunch_NonPersistentBatchSubmitFails_Propagates) {
  PushWaitTask({1}, {1});
  ScopedHook memop(g_cuStreamBatchMemOp,
                   [](hipStream_t, unsigned int, hipStreamBatchMemOpParams*) { return ncclSystemError; });

  EXPECT_EQ(ncclRmaCeWaitLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSystemError);
}

// On the graph arm the ack is a copy; a failure there surfaces after the cycle's
// wait pair, the partial-submission shape the defect note above describes.
TEST_F(RmaCeWaitLaunchTest, WaitLaunch_PersistentAckCopyFails_Propagates) {
  plan_->persistent = true;
  PushWaitTask({1}, {1});
  ScopedHook launch(g_ceLaunchBatchOps,
                    [](ncclComm*, ncclCeBatchOpsParams*, hipStream_t, ncclCeCollArgs*) {
                      return ncclSystemError;
                    });

  EXPECT_EQ(ncclRmaCeWaitLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSystemError);
  ASSERT_EQ(log_.size(), 1u);              // the wait pair went out first
  EXPECT_EQ(log_[0].kind, Submission::kMemOps);
}

}  // namespace
