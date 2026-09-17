/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for src/rma/rma_ce.cc (AICOMRCCL-2347).
 *************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
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

// rma_ce.cc defines ncclRmaCePutLaunch / ncclRmaCeWaitLaunch, which rma-test.cc
// needs as controllable seams in this same binary (rma.cc dispatches to them).
// Rename the unit's own entry points so both can coexist: rma.cc keeps binding
// to the seams in rma_fakes.cc, and the tests below call the real ones.
// The static helpers (ncclRmaCePutLaunchPersist/NonPersist) are distinct tokens
// and so are untouched.
#define ncclRmaCePutLaunch ncclRmaCePutLaunchUut
#define ncclRmaCeWaitLaunch ncclRmaCeWaitLaunchUut
#include RMA_CE_CC_PATH
#undef ncclRmaCePutLaunch
#undef ncclRmaCeWaitLaunch

namespace {

// Release the windows a test left registered. ncclRmaCeFinalize deregisters a
// window but never frees the host object, and a test whose teardown path fails
// deliberately leaves it registered as well.
//
// dev-runtime-test.cc has a fuller ReclaimDevrWindows, but it is static to that
// TU, walks devrState.winSorted (whose element type is private to
// dev_runtime.cc) and drains memHead through the equally private
// symMemoryDestroy. Every window here comes from the non-symmetric path, so it
// has no backing ncclDevrMemory and memHead stays empty; the windows are
// tracked as they are created rather than recovered from winSorted.
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
  // dev_runtime.cc is compiled into this binary, so the CE unit's calls into it
  // run for real. The fixture sets the terms they read rather than stubbing the
  // functions: comm->symmetricSupport == 0 selects RCCL's non-symmetric path,
  // a non-zero bigSize makes ncclDevrInitOnce a no-op, and lsaRankList is what
  // ncclDevrWorldToLsaRank walks.
  // Deliberately different: the unit indexes some things by comm->nRanks and
  // others by devrState.lsaSize, and equal values make every confusion of the
  // two invisible.
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
    // Balance what a test left built: Finalize releases the context array, the
    // signal region and both device buffers each context allocated, and is a
    // no-op success on a comm that was never initialised. Suites that drive
    // Finalize themselves clear the flag -- re-running it after a Finalize that
    // failed part-way would double-free what that attempt already released.
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
  // Overrides the emulator InstallHipVmmEmulator() put in place in SetUp, which
  // succeeds unconditionally; g_hipStreamCreateResult no longer reaches here.
  g_hipStreamCreateWithFlags = [](hipStream_t*, unsigned) { return hipErrorInvalidValue; };

  EXPECT_EQ(ncclRmaCeInit(comm_.get()), ncclUnhandledCudaError);
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
    g_hipFree = [this](void* p) {
      freed_.push_back(p);
      return hipSuccess;
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

  void TearDown() override {
    RmaCeInitTest::TearDown();
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
  std::vector<void*> signalsDev{Ctx(0)->signalsDev, Ctx(1)->signalsDev, Ctx(2)->signalsDev};
  std::vector<ncclWindow_t> expected{Ctx(0)->signalsWin->vidmem, Ctx(1)->signalsWin->vidmem,
                                     Ctx(2)->signalsWin->vidmem};

  ASSERT_EQ(ncclRmaCeFinalize(comm_.get()), ncclSuccess);

  // Every context, in order -- not just the first, and not one repeated. The
  // handle deregistered is the one the window was registered under (vidmem),
  // not the host object the unit reads its fields from.
  EXPECT_EQ(deregistered_, expected);
  // Each context's signal buffer is released. Containment rather than equality:
  // Init allocates two further device buffers per context, which land here too.
  for (size_t i = 0; i < signalsDev.size(); i++) {
    EXPECT_NE(std::find(freed_.begin(), freed_.end(), signalsDev[i]), freed_.end())
        << "context " << i << " signal buffer not freed";
  }
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
}

// ---------------------------------------------------------------------------
// ncclRmaCePutLaunch / ncclRmaCeWaitLaunch
// ---------------------------------------------------------------------------

// The dispatcher's own contract is the guard and the persistent split; what each
// path then does is that helper's contract, covered separately. The two are told
// apart by the capacity they size their batch-ops params to -- the persistent
// path builds one op at a time, the non-persistent path one per rank.
class RmaCePutLaunchTest : public RmaCeInitTest {
protected:
  std::unique_ptr<ncclKernelPlan> plan_;
  ncclRmaArgs args_{};
  std::vector<int> initCapacities_;

  void SetUp() override {
    RmaCeInitTest::SetUp();
    ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);

    ncclMemoryStackConstruct(&comm_->memPermanent);
    ncclMemoryPoolConstruct(&comm_->memPool_ncclTaskRma);

    plan_ = std::make_unique<ncclKernelPlan>();
    plan_->rmaArgs = &args_;
    args_.nRmaTasksCe = 0;  // no tasks: the split is the only thing under test

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

  void TearDown() override {
    ncclMemoryStackDestruct(&comm_->memPermanent);
    RmaCeInitTest::TearDown();
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

  ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr);

  ASSERT_GE(initCapacities_.size(), 2u);
  EXPECT_EQ(initCapacities_[0], kNRanks);
  EXPECT_EQ(initCapacities_[1], kNRanks);
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
class RmaCeNonPersistTest : public RmaCeInitTest {
protected:
  std::unique_ptr<ncclKernelPlan> plan_;
  ncclRmaArgs args_{};
  std::vector<Submission> log_;
  // Peer addresses handed back per (win, offset) lookup, so a test can tell the
  // data destination from the signal destination.
  // The window tasks name for their data. Built by hand rather than registered:
  // production takes it from the task, so nothing in this unit creates it. Same
  // shape a registration produces -- one image per LSA rank, keyed by
  // ipcPeerPtrs -- so the real ncclDevrGetLsaRankPtr resolves against it.
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
    RmaCeInitTest::SetUp();
    ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);

    ncclMemoryStackConstruct(&comm_->memPermanent);
    ncclMemoryPoolConstruct(&comm_->memPool_ncclTaskRma);
    srcBuf_.assign(8, 0);

    plan_ = std::make_unique<ncclKernelPlan>();
    plan_->rmaArgs = &args_;
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
                            ops[i].writeValue.value64});
      }
      log_.push_back(std::move(s));
      return ncclSuccess;
    };
  }

  void TearDown() override {
    ncclMemoryStackDestruct(&comm_->memPermanent);
    RmaCeInitTest::TearDown();
  }

  // Queue one CE task. bytes == 0 means signal-only; signal == false means data-only.
  void PushTask(int peer, size_t bytes, bool signal, size_t winOffset = 0, int signalIdx = 0) {
    auto* t = ncclMemoryPoolAlloc<ncclTaskRma>(&comm_->memPool_ncclTaskRma, &comm_->memPermanent);
    pushed_.push_back(t);
    t->peer = peer;
    t->count = bytes;
    t->datatype = ncclUint8;  // one byte per element, so count is the byte count
    t->srcBuff = srcBuf_.data();
    t->peerWinHost = &dataWin_;
    t->peerWinOffset = winOffset;
    t->signalMode = signal ? NCCL_SIGNAL : NCCL_SIGNAL_NONE;
    t->signalIdx = signalIdx;
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
  PushTask(/*peer=*/1, 32, false);
  PushTask(/*peer=*/3, 48, false);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto batches = Batches();
  ASSERT_EQ(batches.size(), 2u);
  ASSERT_EQ(batches[0].size(), 2u);
  EXPECT_EQ(batches[0][0].dst, PeerData(1));
  EXPECT_EQ(batches[0][1].dst, PeerData(3));
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
  EXPECT_EQ(log_[1].kind, Submission::kBatch);   // data
  EXPECT_EQ(log_[2].kind, Submission::kBatch);   // signal
  ASSERT_EQ(log_[2].ops.size(), 1u);
  EXPECT_EQ(log_[2].ops[0].dst, PeerWin(2) + kLsaSelf);
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

  // All four tasks come back, not just the two the unit had already finished
  // with. The pool starts empty, so the free list is exactly what was returned:
  // asserting its size as well as its contents is what makes dropping the
  // cleanup loop visible -- the two tasks freed on the way in would otherwise
  // leave the list non-empty and any weaker check would still pass.
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

  // lsaSelf is 0, so the offset into the peer's window is the signal index times
  // one rank-stride: kSigIdx * kLsaSize slots past the peer's own base.
  ASSERT_EQ(log_.size(), 3u);
  ASSERT_EQ(log_[2].ops.size(), 1u);
  EXPECT_EQ(log_[2].ops[0].dst, PeerWin(kPeer) + kSigIdx * kLsaSize + kLsaSelf);
  // And the counter bumped is the one for (kSigIdx, peer), not the one for peer.
  EXPECT_EQ(ceCtx->signalOpSeqs[SignalSlot(kSigIdx, kPeer)], 1u);
  EXPECT_EQ(ceCtx->signalOpSeqs[LsaOf(kPeer)], 0u);   // index 0 is a different slot
}

}  // namespace
