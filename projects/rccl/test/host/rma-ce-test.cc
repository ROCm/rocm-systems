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

}  // namespace
