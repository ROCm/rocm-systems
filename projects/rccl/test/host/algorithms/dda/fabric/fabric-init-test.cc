/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/algorithms/dda/fabric/fabric_init.cu.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "FailWhen.h"
#include "HostAllocLedger.h"
#include "ScopedHook.h"
#include "fakes/bootstrap_stubs.h"
#include "fakes/env_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "fakes/rccl_wrap_fakes.h"

#include "algorithms/dda/fabric/FabricGpuBarrierState.h"
#include "algorithms/dda/fabric/FabricTestFixture.h"
#include "mem_manager.h"

// Included before the shims below so they reach only fabric_init.cu.
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "algorithms/dda/fabric/fabric_init.h"
#include "algorithms/dda/fabric/fabric_mem_handler.h"
#include "alloc.h"
#include "archinfo.h"
#include "bootstrap.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "param.h"
#include "rccl_common.h"

#undef ncclCalloc
#define ncclCalloc(...) g_hostAllocs.Calloc(__VA_ARGS__, __FILE__, __LINE__, __func__)
#define free(ptr) g_hostAllocs.Free(ptr)

#include FABRIC_INIT_CC_PATH

#undef ncclCalloc
#undef free

namespace {

using dda_fabric_test::FabricLedgerTest;
using dda_fabric_test::kBootstrap;
using dda_fabric_test::kHandleType;
using dda_fabric_test::kNRanks;
using dda_fabric_test::kPage;
using dda_fabric_test::kRank;
using dda_fabric_test::MemManagerSpy;
using dda_fabric_test::ParamName;
using dda_fabric_test::PeerRanks;

// Above kNRanks * 8, so the block count sizes the epoch cells.
constexpr int kCuCount = 40;
constexpr int64_t kScratchBytes = 64 * 1024;
char kGfx1250[] = "gfx1250";

// ncclCuMemAlloc zeros a new VMM buffer with hipMemsetAsync before it is tracked.
// Fabric init zeros that scratch again; this fails the second fill.
void FailFabricScratchZeroing() {
  int seen = 0;
  FailWhen(
      g_hipMemsetAsync,
      [seen](void*, int, size_t bytes, hipStream_t) mutable {
        return bytes == static_cast<size_t>(kScratchBytes) && ++seen == 2;
      },
      hipErrorInvalidValue);
}

TEST(HostAllocLedgerMicrotest, UntrackedCopyAndFree_AreRefusedAndCounted) {
  HostAllocLedger ledger;
  int src = 1;
  int dst = 0;
  int copyCalls = 0;
  auto copy = ledger.GuardHostCopies([&](void*, const void*, size_t, hipMemcpyKind) {
    ++copyCalls;
    return hipSuccess;
  });

  EXPECT_EQ(copy(&dst, &src, sizeof(src), hipMemcpyHostToHost), hipErrorInvalidValue);
  EXPECT_EQ(ledger.refusedCopies, 1);
  EXPECT_EQ(copyCalls, 0);

  ledger.Free(&dst);
  EXPECT_EQ(ledger.badFrees, 1);
}

void ExpectNoFabricResources(const ncclComm& comm) {
  EXPECT_EQ(comm.ddaFabricMemHandler, nullptr);
  EXPECT_EQ(comm.ddaScratch, nullptr);
  EXPECT_EQ(comm.ddaScratchBytes, 0u);
  EXPECT_FALSE(comm.ddaScratchIsVmm);
  EXPECT_EQ(comm.ddaPeerPtrsDev, nullptr);
  EXPECT_EQ(comm.ddaPeerPtrsHost, nullptr);
  EXPECT_EQ(comm.ddaFabricBarrierState, nullptr);
  EXPECT_EQ(comm.ddaLLEpochDev, nullptr);
  EXPECT_EQ(comm.ddaLLEpochLen, 0);
}

// Fini does not reset ddaFabricMaxBlocks, so Fini tests skip it.
void ExpectNoFabricState(const ncclComm& comm) {
  ExpectNoFabricResources(comm);
  EXPECT_EQ(comm.ddaFabricMaxBlocks, 0);
}

// ---------------------------------------------------------------------------
// ncclDdaUseFabricPath
// ---------------------------------------------------------------------------

struct ArchCase {
  const char* name;
  int mnnvl;
  const char* arch;
  bool expected;
};

class FabricInitUseFabricPathMicrotest : public ::testing::TestWithParam<ArchCase> {};

TEST_F(FabricInitUseFabricPathMicrotest, UseFabricPath_NullComm_ReturnsFalse) {
  EXPECT_FALSE(ncclDdaUseFabricPath(nullptr));
}

TEST_P(FabricInitUseFabricPathMicrotest, UseFabricPath_ByMnnvlAndArch_SelectsFabricOnlyForMnnvlGfx1250) {
  const ArchCase c = GetParam();
  auto comm = std::make_unique<ncclComm>();
  std::string arch = c.arch ? c.arch : "";
  comm->MNNVL = c.mnnvl;
  comm->archName = c.arch ? &arch[0] : nullptr;

  EXPECT_EQ(ncclDdaUseFabricPath(comm.get()), c.expected);
}

INSTANTIATE_TEST_SUITE_P(FabricInitArch, FabricInitUseFabricPathMicrotest,
                         ::testing::Values(ArchCase{"MnnvlGfx1250", 1, "gfx1250", true},
                                           ArchCase{"MnnvlGfx1250WithFeatures", 1, "gfx1250:sramecc+:xnack-", true},
                                           ArchCase{"NotMnnvl", 0, "gfx1250", false},
                                           ArchCase{"MnnvlOtherArch", 1, "gfx950", false},
                                           ArchCase{"MnnvlGfx1200", 1, "gfx1200", false},
                                           ArchCase{"MnnvlNoArch", 1, nullptr, false}),
                         ParamName<ArchCase>);

// ---------------------------------------------------------------------------
// Fixture for Init / Fini
// ---------------------------------------------------------------------------

class FabricInitMicrotest : public FabricLedgerTest {
 protected:
  void SetUp() override {
    FabricLedgerTest::SetUp();
    if (HasFatalFailure()) return;
    SetMicroEnvAbsent("RCCL_DDA_FABRIC_MAXBLOCKS");
    g_hostAllocs.Reset();
    // Host-to-host copies must land inside a tracked host block.
    g_hipMemcpy = g_hostAllocs.GuardHostCopies(g_hipMemcpy);
    g_loadParam = [this](const char* name, int64_t deftVal) {
      const bool unset = !bufferSize_.has_value() || std::string(name) != "RCCL_DDA_FABRIC_BUFFER_SIZE";
      return unset ? deftVal : *bufferSize_;
    };

    comm_ = std::make_unique<ncclComm>();
    comm_->nRanks = kNRanks;
    comm_->rank = kRank;
    comm_->bootstrap = kBootstrap;
    comm_->clique.size = kNRanks;
    comm_->cuCount = kCuCount;
    comm_->MNNVL = 1;
    comm_->archName = kGfx1250;
    MirrorAllGather(&comm_->nRanks, &comm_->rank);
  }

  void TearDown() override {
    if (comm_) ncclDdaFabricCommFini(comm_.get());
    EXPECT_TRUE(HostClean());
    ResetRcclWrapFakes();
    FabricLedgerTest::TearDown();
  }

  ::testing::AssertionResult HostClean() const {
    if (g_hostAllocs.Clean()) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << g_hostAllocs.Summary();
  }

  ::testing::AssertionResult AllReleased() const {
    ::testing::AssertionResult device = LedgerClean();
    if (!device) return device;
    return HostClean();
  }

  // Peers report these caps; this rank's own slot is ignored.
  void SetPeerBlockCaps(std::vector<int> caps) {
    auto homogeneous = g_bootstrapAllGather;
    g_bootstrapAllGather = [this, caps, homogeneous](void* state, void* allData, int size) {
      if (size != static_cast<int>(sizeof(int))) return homogeneous(state, allData, size);
      auto* slots = static_cast<int*>(allData);
      for (int r : PeerRanks(comm_->nRanks, comm_->rank)) slots[r] = caps[r];
      return ncclSuccess;
    };
  }

  std::optional<int64_t> bufferSize_ = kScratchBytes;  // RCCL_DDA_FABRIC_BUFFER_SIZE
  std::unique_ptr<ncclMemManager> manager_;  // outlives TearDown
  std::unique_ptr<ncclComm> comm_;
};

using FabricInitCommInitMicrotest = FabricInitMicrotest;
using FabricInitCommFiniMicrotest = FabricInitMicrotest;

// ---------------------------------------------------------------------------
// ncclDdaFabricCommInit: skips
// ---------------------------------------------------------------------------

TEST_F(FabricInitCommInitMicrotest, CommInit_NullComm_ReturnsSuccess) {
  EXPECT_EQ(ncclDdaFabricCommInit(nullptr), ncclSuccess);
}

struct SkipCase {
  const char* name;
  std::function<void(ncclComm*, std::optional<int64_t>* bufferSize)> setup;
};

class FabricInitSkipMicrotest : public FabricInitMicrotest, public ::testing::WithParamInterface<SkipCase> {};

TEST_P(FabricInitSkipMicrotest, CommInit_Ineligible_ReturnsSuccessWithoutFabricState) {
  GetParam().setup(comm_.get(), &bufferSize_);
  ScopedHook gather(g_bootstrapAllGather, g_bootstrapAllGather);  // count only
  ScopedHook create(g_hipMemCreate, g_hipMemCreate);              // count only

  EXPECT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  ExpectNoFabricState(*comm_);
  EXPECT_EQ(gather.calls, 0);
  EXPECT_EQ(create.calls, 0);
}

INSTANTIATE_TEST_SUITE_P(
    FabricInitSkip, FabricInitSkipMicrotest,
    ::testing::Values(SkipCase{"SingleRank",
                               [](ncclComm* c, std::optional<int64_t>*) {
                                 c->nRanks = 1;
                                 c->rank = 0;
                                 c->clique.size = 1;
                               }},
                      SkipCase{"RanksAboveCap",
                               [](ncclComm* c, std::optional<int64_t>*) {
                                 c->nRanks = dda::common::kDdaMaxNranks + 1;
                                 c->clique.size = c->nRanks;
                               }},
                      SkipCase{"NoBootstrap", [](ncclComm* c, std::optional<int64_t>*) { c->bootstrap = nullptr; }},
                      SkipCase{"SpansCliques",
                               [](ncclComm* c, std::optional<int64_t>*) { c->clique.size = kNRanks / 2; }},
                      SkipCase{"BufferSizeZero", [](ncclComm*, std::optional<int64_t>* bytes) { *bytes = 0; }},
                      SkipCase{"DdaDisabled",
                               [](ncclComm*, std::optional<int64_t>* bytes) {
                                 bytes->reset();
                                 g_rcclParamDdaEnable = 0;
                               }},
                      SkipCase{"VmmUnavailable",
                               [](ncclComm*, std::optional<int64_t>*) { g_cuMemEnable = [] { return 0; }; }}),
    ParamName<SkipCase>);

// ---------------------------------------------------------------------------
// ncclDdaFabricCommInit: success
// ---------------------------------------------------------------------------

TEST_F(FabricInitCommInitMicrotest, CommInit_AllSucceed_HandsEveryResourceToComm) {
  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  EXPECT_NE(comm_->ddaFabricMemHandler, nullptr);
  ASSERT_NE(comm_->ddaScratch, nullptr);
  EXPECT_EQ(comm_->ddaScratchBytes, static_cast<size_t>(kScratchBytes));
  EXPECT_TRUE(comm_->ddaScratchIsVmm);
  EXPECT_EQ(ledger_.reserved.count(comm_->ddaScratch), 1u);
  EXPECT_NE(comm_->ddaPeerPtrsDev, nullptr);
  EXPECT_NE(comm_->ddaPeerPtrsHost, nullptr);
  ASSERT_NE(comm_->ddaFabricBarrierState, nullptr);
  EXPECT_NE(comm_->ddaFabricBarrierState->resources, nullptr);
  EXPECT_EQ(StateOf(comm_->ddaFabricBarrierState->barrierHost).nRanks, kNRanks);
  EXPECT_EQ(comm_->ddaFabricMaxBlocks, kCuCount);
  EXPECT_NE(comm_->ddaLLEpochDev, nullptr);
  EXPECT_EQ(comm_->ddaLLEpochLen, kCuCount);
}

TEST_F(FabricInitCommInitMicrotest, CommInit_AllSucceed_ExportsTheScratchsOwnHandle) {
  std::vector<hipMemGenericAllocationHandle_t> exported;
  auto ledgerExport = g_hipMemExportToShareableHandle;
  ScopedHook exportHook(g_hipMemExportToShareableHandle,
                        [&exported, ledgerExport](void* shareable, hipMemGenericAllocationHandle_t handle,
                                                  hipMemAllocationHandleType type, unsigned long long flags) {
                          exported.push_back(handle);
                          return ledgerExport(shareable, handle, type, flags);
                        });

  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  ASSERT_FALSE(exported.empty());
  EXPECT_EQ(exported.front(), ledger_.mappedHandle.at(comm_->ddaScratch));
}

TEST_F(FabricInitCommInitMicrotest, CommInit_AllSucceed_GathersOverTheCommsBootstrap) {
  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  ASSERT_FALSE(gatherStates_.empty());
  for (void* state : gatherStates_) EXPECT_EQ(state, kBootstrap);
}

TEST_F(FabricInitCommInitMicrotest, CommInit_BufferSizeUnset_SizesScratchFromDdaPayloadCap) {
  bufferSize_.reset();
  g_rcclParamDdaLL = 0;
  g_rcclParamDdaLL128 = 0;
  // Not a page multiple.
  constexpr size_t kPayloadCap = 3 * kPage + 100;
  ScopedHook cap(g_rcclDdaScratchPayloadCap, [](const ncclComm*) { return kPayloadCap; });

  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  EXPECT_EQ(comm_->ddaScratchBytes, kPayloadCap);
}

TEST_F(FabricInitCommInitMicrotest, CommInit_BufferSizeUnset_SizesScratchForTheAllReduceLL128Threshold) {
  bufferSize_.reset();
  g_rcclParamDdaLL = 0;
  g_rcclParamDdaLL128 = 1;
  constexpr size_t kPayloadCap = kPage;
  constexpr size_t kLL128Threshold = 1024 * 1024;
  ScopedHook cap(g_rcclDdaScratchPayloadCap, [](const ncclComm*) { return kPayloadCap; });
  ScopedHook ll128(g_rcclDdaLL128Threshold, [](const ncclComm*, ncclFunc_t func) {
    EXPECT_EQ(func, ncclFuncAllReduce);
    return kLL128Threshold;
  });

  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  const size_t expected = nccl_dda_detail::ddaFabricScratchSizing(kNRanks, -1, 1, kPayloadCap, 0, 1, kLL128Threshold);
  ASSERT_GT(expected, kPayloadCap);
  EXPECT_EQ(comm_->ddaScratchBytes, expected);
}

TEST_F(FabricInitCommInitMicrotest, CommInit_BufferSizeUnset_SizesScratchForTheLLSlots) {
  comm_->nRanks = 2;  // keeps the LL floor (2 banks * nRanks * 32 MiB) small
  comm_->clique.size = 2;
  bufferSize_.reset();
  g_rcclParamDdaLL = 1;
  g_rcclParamDdaLL128 = 0;
  constexpr size_t kPayloadCap = kPage;
  ScopedHook cap(g_rcclDdaScratchPayloadCap, [](const ncclComm*) { return kPayloadCap; });

  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  const size_t expected = nccl_dda_detail::ddaFabricScratchSizing(2, -1, 1, kPayloadCap, 1, 0, 0);
  ASSERT_GT(expected, kPayloadCap);
  EXPECT_EQ(comm_->ddaScratchBytes, expected);
}

TEST_F(FabricInitCommInitMicrotest, CommInit_AllSucceed_PeersMapTheWholeScratch) {
  constexpr int64_t kUnalignedScratch = 5 * kPage + 100;  // not a page multiple
  bufferSize_ = kUnalignedScratch;
  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);
  auto* const* dev = static_cast<void* const*>(comm_->ddaPeerPtrsDev);  // host stand-in
  const size_t ownSize = ledger_.mappedSize.at(comm_->ddaScratch);

  for (int i : PeerRanks()) {
    ASSERT_EQ(ledger_.mappedSize.count(dev[i]), 1u) << "peer " << i;
    EXPECT_EQ(ledger_.mappedSize.at(dev[i]), ownSize) << "peer " << i;
  }
}

TEST_F(FabricInitCommInitMicrotest, CommInit_AllSucceed_PeerTablesHoldOwnScratchAndEachPeersMapping) {
  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);
  auto* handler = comm_->ddaFabricMemHandler;
  auto* const* dev = static_cast<void* const*>(comm_->ddaPeerPtrsDev);  // host stand-in
  void* const* host = comm_->ddaPeerPtrsHost;

  for (int i = 0; i < kNRanks; ++i) {
    void* expected = comm_->ddaScratch;
    if (i != kRank) ASSERT_EQ(handler->getPeerDeviceMemPtr(i, &expected), ncclSuccess);
    EXPECT_EQ(dev[i], expected) << "rank " << i;
    EXPECT_EQ(host[i], expected) << "rank " << i;
  }
  std::set<void*> scratchMappings;
  for (const auto& entry : ledger_.reserved) {
    if (entry.second == static_cast<size_t>(kScratchBytes) && entry.first != comm_->ddaScratch) {
      scratchMappings.insert(entry.first);
    }
  }
  std::set<void*> peerSlots;
  for (int i : PeerRanks()) peerSlots.insert(dev[i]);
  EXPECT_EQ(peerSlots, scratchMappings);
  EXPECT_EQ(peerSlots.size(), static_cast<size_t>(kNRanks - 1));
}

TEST_F(FabricInitCommInitMicrotest, CommInit_AllSucceed_ZeroesScratchAndEpochCells) {
  // Zero-fill is hipMemsetAsync on a private stream. The ledger only records
  // synchronous hipMemset, which the barrier still uses for its flag buffer.
  struct AsyncMemset {
    void* dst;
    int value;
    size_t bytes;
    hipStream_t stream;
  };
  std::vector<AsyncMemset> asyncMemsets;
  auto emulatedMemset = g_hipMemsetAsync;
  ScopedHook memsetAsync(g_hipMemsetAsync,
                         [&asyncMemsets, emulatedMemset](void* dst, int value, size_t bytes, hipStream_t stream) {
                           asyncMemsets.push_back({dst, value, bytes, stream});
                           return emulatedMemset(dst, value, bytes, stream);
                         });
  std::vector<hipStream_t> synced;
  auto emulatedSync = g_hipStreamSynchronize;
  ScopedHook sync(g_hipStreamSynchronize, [&synced, emulatedSync](hipStream_t stream) {
    synced.push_back(stream);
    return emulatedSync(stream);
  });

  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  const AsyncMemset* scratch = nullptr;
  const AsyncMemset* epoch = nullptr;
  for (const AsyncMemset& m : asyncMemsets) {
    if (m.dst == comm_->ddaScratch) scratch = &m;
    if (m.dst == comm_->ddaLLEpochDev) epoch = &m;
  }
  ASSERT_NE(scratch, nullptr) << "scratch never zeroed";
  EXPECT_EQ(scratch->value, 0);
  EXPECT_EQ(scratch->bytes, static_cast<size_t>(kScratchBytes));
  ASSERT_NE(epoch, nullptr) << "epoch cells never zeroed";
  EXPECT_EQ(epoch->value, 0);
  EXPECT_EQ(epoch->bytes, static_cast<size_t>(comm_->ddaLLEpochLen) * sizeof(uint32_t));
  ASSERT_NE(scratch->stream, nullptr);
  EXPECT_EQ(epoch->stream, scratch->stream);
  EXPECT_NE(std::find(synced.begin(), synced.end(), scratch->stream), synced.end());
}

TEST_F(FabricInitCommInitMicrotest, CommInit_AllSucceed_BarrierUsesCommGeometry) {
  comm_->abortFlagDev = reinterpret_cast<uint32_t*>(0xAB047);
  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);
  const auto* state = comm_->ddaFabricBarrierState;

  const FabricGpuBarrierState barrier = StateOf(state->barrierHost);
  EXPECT_EQ(barrier.nBlocks, comm_->ddaFabricMaxBlocks);
  EXPECT_EQ(barrier.selfRank, kRank);
  EXPECT_EQ(barrier.nRanks, kNRanks);
  EXPECT_EQ(barrier.peerFlags, state->resources->peerFlagsDev->get());
  EXPECT_EQ(barrier.params.flag, comm_->abortFlagDev);
}

TEST_F(FabricInitCommInitMicrotest, CommInit_WithManager_TracksAndUntracksEveryAllocationInIt) {
  manager_ = std::make_unique<ncclMemManager>();
  ncclMemManager* const manager = manager_.get();
  comm_->memManager = manager;
  MemManagerSpy spy;
  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);
  void* const scratch = comm_->ddaScratch;
  void* const flags = comm_->ddaFabricBarrierState->resources->selfFlagBuf->get();
  auto* const* dev = static_cast<void* const*>(comm_->ddaPeerPtrsDev);  // host stand-in
  std::vector<void*> peers;
  for (int i : PeerRanks()) peers.push_back(dev[i]);

  ASSERT_EQ(ncclDdaFabricCommFini(comm_.get()), ncclSuccess);

  EXPECT_TRUE(spy.Tracked(manager, scratch, kHandleType)) << "scratch not tracked in the comm's manager";
  EXPECT_TRUE(spy.Untracked(manager, scratch)) << "scratch not untracked from the comm's manager";
  EXPECT_TRUE(spy.Tracked(manager, flags, kHandleType)) << "barrier flag buffer not tracked in the comm's manager";
  EXPECT_TRUE(spy.Untracked(manager, flags)) << "barrier flag buffer not untracked";
  for (void* peer : peers) EXPECT_TRUE(spy.Untracked(manager, peer)) << "peer mapping " << peer;
  EXPECT_TRUE(AllReleased());
}

TEST_F(FabricInitCommInitMicrotest, CommInit_FailsWithManager_UntracksScratchFromIt) {
  manager_ = std::make_unique<ncclMemManager>();
  comm_->memManager = manager_.get();
  MemManagerSpy spy;
  FailFabricScratchZeroing();

  EXPECT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  ExpectNoFabricState(*comm_);
  ASSERT_FALSE(spy.tracked.empty());
  EXPECT_TRUE(spy.Untracked(manager_.get(), spy.tracked.front().ptr)) << "scratch not untracked on the fail path";
  EXPECT_TRUE(AllReleased());
}

struct CapsCase {
  const char* name;
  std::vector<int> caps;  // this rank's own slot is ignored
};

class FabricInitPeerCapsMicrotest : public FabricInitMicrotest, public ::testing::WithParamInterface<CapsCase> {};

TEST_P(FabricInitPeerCapsMicrotest, CommInit_PeerCapsLower_UsesSmallestCapAcrossRanks) {
  constexpr int kSmallest = 36;  // above kNRanks * 8
  SetPeerBlockCaps(GetParam().caps);

  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  EXPECT_EQ(comm_->ddaFabricMaxBlocks, kSmallest);
  EXPECT_EQ(StateOf(comm_->ddaFabricBarrierState->barrierHost).nBlocks, kSmallest);
  EXPECT_EQ(comm_->ddaLLEpochLen, kSmallest);
}

INSTANTIATE_TEST_SUITE_P(FabricInitCaps, FabricInitPeerCapsMicrotest,
                         ::testing::Values(CapsCase{"SmallestFirst", {36, -1, 38, 39}},
                                           CapsCase{"SmallestMiddle", {kCuCount, -1, 36, 38}},
                                           CapsCase{"SmallestLast", {38, -1, 39, 36}}),
                         ParamName<CapsCase>);

TEST_F(FabricInitCommInitMicrotest, CommInit_PeerCapNotPositive_FallsBackWithoutEpochAllocation) {
  for (int bad : {0, -1}) {
    SCOPED_TRACE(bad);
    SetPeerBlockCaps({kCuCount, -1, bad, kCuCount});
    std::vector<size_t> mallocs;
    auto ledgerMalloc = g_hipMalloc;
    ScopedHook malloc(g_hipMalloc, [&mallocs, ledgerMalloc](void** ptr, size_t size) {
      mallocs.push_back(size);
      return ledgerMalloc(ptr, size);
    });

    EXPECT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

    ExpectNoFabricState(*comm_);
    EXPECT_TRUE(AllReleased());
    for (size_t size : mallocs) EXPECT_EQ(size, kNRanks * sizeof(void*));  // peer table only
  }
}

TEST_F(FabricInitCommInitMicrotest, CommInit_PeerCapsHigher_KeepsLocalCap) {
  SetPeerBlockCaps({256, /*own slot*/ -1, 256, 256});

  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  EXPECT_EQ(comm_->ddaFabricMaxBlocks, kCuCount);
}

TEST_F(FabricInitCommInitMicrotest, CommInit_PublishesLocalCapForThePeers) {
  int published = -1;
  auto homogeneous = g_bootstrapAllGather;
  ScopedHook gather(g_bootstrapAllGather, [&published, homogeneous](void* state, void* allData, int size) {
    // The block cap is the first sizeof(int) exchange. A later one carries each
    // rank's LL-buffer ready flag, which is 0 or 1, not the cap.
    if (published < 0 && size == static_cast<int>(sizeof(int))) published = static_cast<int*>(allData)[kRank];
    return homogeneous(state, allData, size);
  });
  SetMicroEnv("RCCL_DDA_FABRIC_MAXBLOCKS", "7");

  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  EXPECT_EQ(published, 7);
}

TEST_F(FabricInitCommInitMicrotest, CommInit_TwoRanks_SetsUpFabricPath) {
  comm_->nRanks = 2;
  comm_->clique.size = 2;

  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  EXPECT_NE(comm_->ddaFabricMemHandler, nullptr);
}

TEST_F(FabricInitCommInitMicrotest, CommInit_RanksAtCap_SetsUpFabricPath) {
  comm_->nRanks = dda::common::kDdaMaxNranks;
  comm_->clique.size = comm_->nRanks;

  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  EXPECT_NE(comm_->ddaFabricMemHandler, nullptr);
  const size_t epochLen = static_cast<size_t>(dda::common::kDdaMaxNranks) * nccl_dda_detail::kDdaLLAgMaxBlocksPerPeer;
  ASSERT_GT(epochLen, static_cast<size_t>(kCuCount));
  EXPECT_EQ(static_cast<size_t>(comm_->ddaLLEpochLen), epochLen);
}

// ---------------------------------------------------------------------------
// ncclDdaFabricCommInit: failure
// ---------------------------------------------------------------------------

TEST_F(FabricInitCommInitMicrotest, CommInit_BlockCapGatherFails_ReturnsErrorWithoutAllocating) {
  ScopedHook gather(g_bootstrapAllGather, [](void*, void*, int) { return ncclRemoteError; });

  EXPECT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclRemoteError);

  ExpectNoFabricState(*comm_);
  EXPECT_TRUE(AllReleased());
}

// Current behaviour, tracked as AICOMRCCL-2827: a rank that fails here falls back
// alone, skipping exchanges its peers still enter.
struct FailCase {
  const char* name;
  std::function<void()> install;
};

class FabricInitFailMicrotest : public FabricInitMicrotest, public ::testing::WithParamInterface<FailCase> {};

TEST_P(FabricInitFailMicrotest, CommInit_StepFails_FallsBackWithoutFabricStateOrLeaks) {
  GetParam().install();

  EXPECT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  ExpectNoFabricState(*comm_);
  EXPECT_TRUE(AllReleased());
}

constexpr size_t kPeerTableBytes = kNRanks * sizeof(void*);
constexpr size_t kEpochBytes = kCuCount * sizeof(uint32_t);

INSTANTIATE_TEST_SUITE_P(
    FabricInitStep, FabricInitFailMicrotest,
    ::testing::Values(
        FailCase{"ScratchAlloc",
                 [] {
                   FailWhen(
                       g_hipMemCreate,
                       [](hipMemGenericAllocationHandle_t*, size_t size, const hipMemAllocationProp*,
                          unsigned long long) {
                         return size == static_cast<size_t>(kScratchBytes);
                       },
                       hipErrorOutOfMemory);
                 }},
        FailCase{"ScratchExchange",
                 [] {
                   int calls = 0;
                   FailWhen(
                       g_hipMemExportToShareableHandle,
                       [calls](void*, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType,
                               unsigned long long) mutable { return ++calls == 1; },
                       hipErrorInvalidValue);
                 }},
        FailCase{"PeerTableAlloc",
                 [] {
                   FailWhen(
                       g_hipMalloc, [](void**, size_t size) { return size == kPeerTableBytes; }, hipErrorOutOfMemory);
                 }},
        FailCase{"PeerTableCopy",
                 [] {
                   // The barrier's table has the same size; fail only init's.
                   auto table = std::make_shared<void*>(nullptr);
                   auto ledgerMalloc = g_hipMalloc;
                   g_hipMalloc = [table, ledgerMalloc](void** ptr, size_t size) {
                     hipError_t err = ledgerMalloc(ptr, size);
                     if (err == hipSuccess && size == kPeerTableBytes) *table = *ptr;
                     return err;
                   };
                   FailWhen(
                       g_hipMemcpy,
                       [table](void* dst, const void*, size_t, hipMemcpyKind) { return dst == *table; },
                       hipErrorInvalidValue);
                 }},
        FailCase{"PeerTableHostAlloc", [] { g_hostAllocs.failBytes = kPeerTableBytes; }},
        FailCase{"PeerTableHostCopy",
                 [] {
                   FailWhen(
                       g_hipMemcpy,
                       [](void*, const void*, size_t, hipMemcpyKind kind) { return kind == hipMemcpyHostToHost; },
                       hipErrorInvalidValue);
                 }},
        FailCase{"Barrier",
                 [] {
                   FailWhen(
                       g_hipMemCreate,
                       [](hipMemGenericAllocationHandle_t*, size_t size, const hipMemAllocationProp*,
                          unsigned long long) { return size < static_cast<size_t>(kScratchBytes); },
                       hipErrorOutOfMemory);
                 }},
        FailCase{"ScratchZeroing", [] { FailFabricScratchZeroing(); }},
        FailCase{"EpochAlloc",
                 [] {
                   FailWhen(
                       g_hipMalloc, [](void**, size_t size) { return size == kEpochBytes; }, hipErrorOutOfMemory);
                 }},
        FailCase{"EpochZeroing",
                 [] {
                   FailWhen(
                       g_hipMemsetAsync,
                       [](void*, int, size_t bytes, hipStream_t) { return bytes == kEpochBytes; },
                       hipErrorInvalidValue);
                 }}),
    ParamName<FailCase>);

// ---------------------------------------------------------------------------
// ncclDdaFabricCommFini
// ---------------------------------------------------------------------------

TEST_F(FabricInitCommFiniMicrotest, CommFini_NullComm_ReturnsSuccess) {
  EXPECT_EQ(ncclDdaFabricCommFini(nullptr), ncclSuccess);
}

TEST_F(FabricInitCommFiniMicrotest, CommFini_AfterInit_ReleasesEverythingAndClearsFabricState) {
  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);
  ASSERT_NE(comm_->ddaScratch, nullptr);

  EXPECT_EQ(ncclDdaFabricCommFini(comm_.get()), ncclSuccess);

  ExpectNoFabricResources(*comm_);
  EXPECT_TRUE(AllReleased());
}

TEST_F(FabricInitCommFiniMicrotest, CommFini_NonVmmScratch_FreesItAsAPlainBuffer) {
  void* scratch = nullptr;
  ASSERT_EQ(hipMalloc(&scratch, kPage), hipSuccess);
  comm_->ddaScratch = scratch;
  comm_->ddaScratchBytes = kPage;
  comm_->ddaScratchIsVmm = false;

  EXPECT_EQ(ncclDdaFabricCommFini(comm_.get()), ncclSuccess);

  ExpectNoFabricResources(*comm_);
  EXPECT_TRUE(AllReleased());
}

TEST_F(FabricInitCommFiniMicrotest, CommFini_CalledTwice_ReleasesOnce) {
  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);
  ASSERT_EQ(ncclDdaFabricCommFini(comm_.get()), ncclSuccess);

  EXPECT_EQ(ncclDdaFabricCommFini(comm_.get()), ncclSuccess);

  ExpectNoFabricResources(*comm_);
  EXPECT_TRUE(AllReleased());
}

TEST_F(FabricInitCommFiniMicrotest, CommFini_NoFabricState_ReleasesNothing) {
  EXPECT_EQ(ncclDdaFabricCommFini(comm_.get()), ncclSuccess);

  ExpectNoFabricResources(*comm_);
  EXPECT_TRUE(AllReleased());
  EXPECT_TRUE(ledger_.addressFrees.empty());
}

}  // namespace
