/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for src/algorithms/dda/fabric/fabric_init.cu,
 * #include-d via FABRIC_INIT_CC_PATH: the fabric-path gate, per-comm fabric
 * DDA setup, and its teardown. The handler and barrier it builds on are the
 * real ones the sibling tests include, driven at the HIP VMM and bootstrap
 * seams through HipVmmLedger.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "HipVmmLedger.h"
#include "ScopedHook.h"
#include "fakes/bootstrap_stubs.h"
#include "fakes/env_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "fakes/rccl_wrap_fakes.h"

#include "algorithms/dda/fabric/FabricGpuBarrierState.h"
#include "mem_manager.h"

// Everything fabric_init.cu includes, first, so the free/ncclCalloc shims below
// reach only its own body.
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

// Host memory fabric_init.cu allocates with ncclCalloc and releases with free;
// the ledger only sees device memory.
static std::set<void*> g_initHostLive;
template <typename T>
static ncclResult_t InitCalloc(T** ptr, size_t nelem, const char* file, int line, const char* fn) {
  ncclResult_t ret = ncclCallocDebug(ptr, nelem, file, line, fn, true);
  if (ret == ncclSuccess && *ptr != nullptr) g_initHostLive.insert(*ptr);
  return ret;
}
static void InitFree(void* ptr) {
  g_initHostLive.erase(ptr);
  std::free(ptr);
}

#undef ncclCalloc
#define ncclCalloc(...) InitCalloc(__VA_ARGS__, __FILE__, __LINE__, __func__)
#define free(ptr) InitFree(ptr)

#include FABRIC_INIT_CC_PATH

#undef ncclCalloc
#undef free

namespace {

constexpr int kNRanks = 4;
constexpr int kRank = 1;
// The local block cap. Above the AllGather term of ddaLLEpochCount (nRanks * 8 = 32),
// so the agreed block count is what sizes the epoch cells.
constexpr int kCuCount = 40;
constexpr int64_t kScratchBytes = 64 * 1024;
void* const kBootstrap = reinterpret_cast<void*>(0xB007);
char kGfx1250[] = "gfx1250";

// Installs `fail` in front of `slot`'s current behaviour for the calls `when` picks.
template <typename R, typename... Args, typename When>
void FailWhen(std::function<R(Args...)>& slot, When when, R fail) {
  auto pass = slot;
  slot = [pass, when, fail](Args... args) mutable -> R { return when(args...) ? fail : pass(args...); };
}

// The fabric resources ncclDdaFabricCommInit hands to the comm and
// ncclDdaFabricCommFini takes back.
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

// ...plus the block count, which only a successful Init sets. Fini leaves it as
// it was, so Fini tests check ExpectNoFabricResources only.
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

class DdaUseFabricPathTest : public ::testing::TestWithParam<ArchCase> {};

TEST_F(DdaUseFabricPathTest, UseFabricPath_NullComm_ReturnsFalse) { EXPECT_FALSE(ncclDdaUseFabricPath(nullptr)); }

TEST_P(DdaUseFabricPathTest, UseFabricPath_ByMnnvlAndArch_SelectsFabricOnlyForMnnvlGfx1250) {
  const ArchCase c = GetParam();
  auto comm = std::make_unique<ncclComm>();
  std::string arch = c.arch ? c.arch : "";
  comm->MNNVL = c.mnnvl;
  comm->archName = c.arch ? &arch[0] : nullptr;

  EXPECT_EQ(ncclDdaUseFabricPath(comm.get()), c.expected);
}

INSTANTIATE_TEST_SUITE_P(Arch, DdaUseFabricPathTest,
                         ::testing::Values(ArchCase{"MnnvlGfx1250", 1, "gfx1250", true},
                                           ArchCase{"MnnvlGfx1250WithFeatures", 1, "gfx1250:sramecc+:xnack-", true},
                                           ArchCase{"NotMnnvl", 0, "gfx1250", false},
                                           ArchCase{"MnnvlOtherArch", 1, "gfx950", false},
                                           ArchCase{"MnnvlNoArch", 1, nullptr, false}),
                         [](const ::testing::TestParamInfo<ArchCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// Fixture for Init / Fini
// ---------------------------------------------------------------------------

class DdaFabricCommTest : public ::testing::Test {
 protected:
  struct Memset {
    void* dst;
    int value;
    size_t bytes;
  };

  void SetUp() override {
    SetMicroEnvAbsent("NCCL_CUMEM_SKIP_FREE");  // see HipVmmLedger.h
    SetMicroEnvAbsent("RCCL_DDA_FABRIC_MAXBLOCKS");
    ledger_.Install();
    auto ledgerMemset = g_hipMemset;
    g_hipMemset = [this, ledgerMemset](void* dst, int value, size_t bytes) {
      memsets_.push_back({dst, value, bytes});
      return ledgerMemset(dst, value, bytes);
    };
    g_cuMemEnable = [] { return 1; };
    g_loadParam = [this](const char* name, int64_t deftVal) {
      const bool unset = !bufferSize_.has_value() || std::string(name) != "RCCL_DDA_FABRIC_BUFFER_SIZE";
      return unset ? deftVal : *bufferSize_;
    };
    g_initHostLive.clear();
    // A homogeneous clique: every peer publishes what this rank published.
    g_bootstrapAllGather = [this](void* state, void* allData, int size) {
      gatherStates_.push_back(state);
      auto* slots = static_cast<char*>(allData);
      for (int r = 0; r < comm_->nRanks; ++r) {
        if (r != comm_->rank) std::memcpy(slots + r * size, slots + comm_->rank * size, size);
      }
      return ncclSuccess;
    };

    comm_ = std::make_unique<ncclComm>();  // value-initialised: no fabric state
    comm_->nRanks = kNRanks;
    comm_->rank = kRank;
    comm_->bootstrap = kBootstrap;
    comm_->clique.size = kNRanks;
    comm_->cuCount = kCuCount;
    comm_->MNNVL = 1;
    comm_->archName = kGfx1250;
  }

  void TearDown() override {
    // first: Fini frees through the ledger hooks
    if (comm_) ncclDdaFabricCommFini(comm_.get());
    ResetBootstrapStubs();
    ResetHipFakes();
    ResetNcclFakes();
    ResetRcclWrapFakes();
    ResetEnvFakes();
  }

  // Nothing device- or host-side left allocated, and nothing the ledger refused.
  ::testing::AssertionResult AllReleased() const {
    if (ledger_.Clean() && g_initHostLive.empty()) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << ledger_.reserved.size() << " reservations, " << ledger_.liveHandles.size()
                                         << " handles, " << ledger_.liveBuffers.size() << " buffers, "
                                         << g_initHostLive.size() << " host allocations live; "
                                         << ledger_.rejected.size() << " calls refused";
  }

  // Peers report these block caps instead of this rank's (indexed by rank; the
  // entry at this rank's own slot is ignored).
  void SetPeerBlockCaps(std::vector<int> caps) {
    auto homogeneous = g_bootstrapAllGather;
    g_bootstrapAllGather = [this, caps, homogeneous](void* state, void* allData, int size) {
      if (size != static_cast<int>(sizeof(int))) return homogeneous(state, allData, size);
      auto* slots = static_cast<int*>(allData);
      for (int r = 0; r < comm_->nRanks; ++r) {
        if (r != comm_->rank) slots[r] = caps[r];
      }
      return ncclSuccess;
    };
  }

  const Memset* MemsetOf(const void* dst) const {
    for (const Memset& m : memsets_) {
      if (m.dst == dst) return &m;
    }
    return nullptr;
  }

  std::optional<int64_t> bufferSize_ = kScratchBytes;  // RCCL_DDA_FABRIC_BUFFER_SIZE; nullopt = unset
  HipVmmLedger ledger_;
  std::unique_ptr<ncclMemManager> manager_;  // outlives TearDown's Fini
  std::unique_ptr<ncclComm> comm_;
  std::vector<Memset> memsets_;
  std::vector<void*> gatherStates_;
};

using DdaFabricCommInitTest = DdaFabricCommTest;
using DdaFabricCommFiniTest = DdaFabricCommTest;

// ---------------------------------------------------------------------------
// ncclDdaFabricCommInit: skips
// ---------------------------------------------------------------------------

TEST_F(DdaFabricCommInitTest, CommInit_NullComm_ReturnsSuccess) {
  EXPECT_EQ(ncclDdaFabricCommInit(nullptr), ncclSuccess);
}

struct SkipCase {
  const char* name;
  std::function<void(ncclComm*, std::optional<int64_t>* bufferSize)> setup;
};

class DdaFabricCommInitSkipTest : public DdaFabricCommTest, public ::testing::WithParamInterface<SkipCase> {};

TEST_P(DdaFabricCommInitSkipTest, CommInit_Ineligible_ReturnsSuccessWithoutFabricState) {
  GetParam().setup(comm_.get(), &bufferSize_);
  ScopedHook gather(g_bootstrapAllGather, g_bootstrapAllGather);  // count only
  ScopedHook create(g_hipMemCreate, g_hipMemCreate);              // count only

  EXPECT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  ExpectNoFabricState(*comm_);
  EXPECT_EQ(gather.calls, 0);
  EXPECT_EQ(create.calls, 0);
}

INSTANTIATE_TEST_SUITE_P(
    Skip, DdaFabricCommInitSkipTest,
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
                                 bytes->reset();  // unset: sized from DDA, which is off
                                 g_rcclParamDdaEnable = 0;
                               }},
                      SkipCase{"VmmUnavailable",
                               [](ncclComm*, std::optional<int64_t>*) { g_cuMemEnable = [] { return 0; }; }}),
    [](const ::testing::TestParamInfo<SkipCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// ncclDdaFabricCommInit: success
// ---------------------------------------------------------------------------

TEST_F(DdaFabricCommInitTest, CommInit_AllSucceed_HandsEveryResourceToComm) {
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
  EXPECT_EQ(StateOf(comm_->ddaFabricBarrierState->barrierHost).nRanks, kNRanks);  // the barrier was kept
  EXPECT_EQ(comm_->ddaFabricMaxBlocks, kCuCount);
  EXPECT_NE(comm_->ddaLLEpochDev, nullptr);
  EXPECT_EQ(comm_->ddaLLEpochLen, kCuCount);
}

TEST_F(DdaFabricCommInitTest, CommInit_AllSucceed_GathersOverTheCommsBootstrap) {
  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  ASSERT_FALSE(gatherStates_.empty());
  for (void* state : gatherStates_) EXPECT_EQ(state, kBootstrap);
}

TEST_F(DdaFabricCommInitTest, CommInit_BufferSizeUnset_SizesScratchFromDdaPayloadCap) {
  bufferSize_.reset();
  g_rcclParamDdaLL = 0;
  g_rcclParamDdaLL128 = 0;
  // Not a page multiple: ddaScratchBytes is the size asked for, not the rounded reservation.
  constexpr size_t kPayloadCap = 3 * 4096 + 100;
  ScopedHook cap(g_rcclDdaScratchPayloadCap, [](const ncclComm*) { return kPayloadCap; });

  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  EXPECT_EQ(comm_->ddaScratchBytes, kPayloadCap);
}

TEST_F(DdaFabricCommInitTest, CommInit_BufferSizeUnset_SizesScratchForTheAllReduceLL128Threshold) {
  bufferSize_.reset();
  g_rcclParamDdaLL = 0;
  g_rcclParamDdaLL128 = 1;
  constexpr size_t kPayloadCap = 4096;
  constexpr size_t kLL128Threshold = 1024 * 1024;  // its slot arrays outgrow the payload cap
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

TEST_F(DdaFabricCommInitTest, CommInit_AllSucceed_PeersMapTheWholeScratch) {
  constexpr int64_t kUnalignedScratch = 5 * 4096 + 100;  // a wrong size would round differently
  bufferSize_ = kUnalignedScratch;
  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);
  auto* const* dev = static_cast<void* const*>(comm_->ddaPeerPtrsDev);  // host stand-in
  const size_t ownSize = ledger_.reserved.at(comm_->ddaScratch);

  for (int i = 0; i < kNRanks; ++i) {
    if (i == kRank) continue;
    ASSERT_EQ(ledger_.reserved.count(dev[i]), 1u) << "peer " << i;
    EXPECT_EQ(ledger_.reserved.at(dev[i]), ownSize) << "peer " << i;
  }
}

TEST_F(DdaFabricCommInitTest, CommInit_AllSucceed_PeerTablesHoldOwnScratchAndEachPeersMapping) {
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
  // Independently of the handler: the peer slots are exactly the scratch-sized
  // mappings other than this rank's own scratch.
  std::set<void*> scratchMappings;
  for (const auto& entry : ledger_.reserved) {
    if (entry.second == static_cast<size_t>(kScratchBytes) && entry.first != comm_->ddaScratch) {
      scratchMappings.insert(entry.first);
    }
  }
  std::set<void*> peerSlots;
  for (int i = 0; i < kNRanks; ++i) {
    if (i != kRank) peerSlots.insert(dev[i]);
  }
  EXPECT_EQ(peerSlots, scratchMappings);
  EXPECT_EQ(peerSlots.size(), static_cast<size_t>(kNRanks - 1));
}

TEST_F(DdaFabricCommInitTest, CommInit_AllSucceed_ZeroesScratchAndEpochCells) {
  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  const Memset* scratch = MemsetOf(comm_->ddaScratch);
  ASSERT_NE(scratch, nullptr) << "scratch never zeroed";
  EXPECT_EQ(scratch->value, 0);
  EXPECT_EQ(scratch->bytes, static_cast<size_t>(kScratchBytes));
  const Memset* epoch = MemsetOf(comm_->ddaLLEpochDev);
  ASSERT_NE(epoch, nullptr) << "epoch cells never zeroed";
  EXPECT_EQ(epoch->value, 0);
  EXPECT_EQ(epoch->bytes, static_cast<size_t>(comm_->ddaLLEpochLen) * sizeof(uint32_t));
}

TEST_F(DdaFabricCommInitTest, CommInit_AllSucceed_BarrierUsesCommGeometry) {
  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);
  const auto* state = comm_->ddaFabricBarrierState;

  const FabricGpuBarrierState barrier = StateOf(state->barrierHost);
  EXPECT_EQ(barrier.nBlocks, comm_->ddaFabricMaxBlocks);
  EXPECT_EQ(barrier.selfRank, kRank);
  EXPECT_EQ(barrier.nRanks, kNRanks);
  EXPECT_EQ(barrier.peerFlags, state->resources->peerFlagsDev->get());
}

TEST_F(DdaFabricCommInitTest, CommInit_WithManager_TracksScratchAndPeerMappingsInIt) {
  manager_ = std::make_unique<ncclMemManager>();  // value-initialised: nothing released
  ncclMemManager* const manager = manager_.get();
  comm_->memManager = manager;
  std::vector<std::pair<ncclMemManager*, void*>> tracked;
  std::vector<std::pair<ncclMemManager*, void*>> untracked;
  ScopedHook track(g_memTrack, [&tracked](ncclMemManager* m, void* ptr, size_t, hipMemGenericAllocationHandle_t,
                                          hipMemAllocationHandleType, ncclMemType_t) {
    tracked.push_back({m, ptr});
    return ncclSuccess;
  });
  auto untrackLive = g_memUntrackDynamic;
  ScopedHook untrack(g_memUntrackDynamic,
                     [&untracked, untrackLive](ncclMemManager* m, void* ptr, ncclMemUntrackInfo* info) {
                       untracked.push_back({m, ptr});
                       return untrackLive(m, ptr, info);
                     });
  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);
  void* const scratch = comm_->ddaScratch;
  std::vector<void*> peers;
  auto* const* dev = static_cast<void* const*>(comm_->ddaPeerPtrsDev);  // host stand-in
  for (int i = 0; i < kNRanks; ++i) {
    if (i != kRank) peers.push_back(dev[i]);
  }

  ASSERT_EQ(ncclDdaFabricCommFini(comm_.get()), ncclSuccess);

  auto has = [](const std::vector<std::pair<ncclMemManager*, void*>>& v, ncclMemManager* m, void* p) {
    return std::find(v.begin(), v.end(), std::make_pair(m, p)) != v.end();
  };
  EXPECT_TRUE(has(tracked, manager, scratch)) << "scratch not tracked in the comm's manager";
  EXPECT_TRUE(has(untracked, manager, scratch)) << "scratch not untracked from the comm's manager";
  for (void* peer : peers) {
    EXPECT_TRUE(has(untracked, manager, peer)) << "peer mapping " << peer << " not untracked";
  }
}

TEST_F(DdaFabricCommInitTest, CommInit_PeerCapsLower_UsesSmallestCapAcrossRanks) {
  constexpr int kSmallest = 36;  // above the epoch cells' AllGather term (32), so it sizes them
  SetPeerBlockCaps({kCuCount, /*own slot*/ -1, kSmallest, 38});

  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  EXPECT_EQ(comm_->ddaFabricMaxBlocks, kSmallest);
  EXPECT_EQ(StateOf(comm_->ddaFabricBarrierState->barrierHost).nBlocks, kSmallest);
  EXPECT_EQ(comm_->ddaLLEpochLen, kSmallest);
}

TEST_F(DdaFabricCommInitTest, CommInit_PeerCapsHigher_KeepsLocalCap) {
  SetPeerBlockCaps({256, /*own slot*/ -1, 256, 256});

  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  EXPECT_EQ(comm_->ddaFabricMaxBlocks, kCuCount);
}

TEST_F(DdaFabricCommInitTest, CommInit_PublishesLocalCapForThePeers) {
  int published = -1;
  auto homogeneous = g_bootstrapAllGather;
  ScopedHook gather(g_bootstrapAllGather, [&published, homogeneous](void* state, void* allData, int size) {
    if (size == static_cast<int>(sizeof(int))) published = static_cast<int*>(allData)[kRank];
    return homogeneous(state, allData, size);
  });
  SetMicroEnv("RCCL_DDA_FABRIC_MAXBLOCKS", "7");

  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  EXPECT_EQ(published, 7);
}

TEST_F(DdaFabricCommInitTest, CommInit_TwoRanks_SetsUpFabricPath) {
  comm_->nRanks = 2;
  comm_->clique.size = 2;

  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  EXPECT_NE(comm_->ddaFabricMemHandler, nullptr);
}

TEST_F(DdaFabricCommInitTest, CommInit_RanksAtCap_SetsUpFabricPath) {
  comm_->nRanks = dda::common::kDdaMaxNranks;
  comm_->clique.size = comm_->nRanks;

  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  EXPECT_NE(comm_->ddaFabricMemHandler, nullptr);
}

// ---------------------------------------------------------------------------
// ncclDdaFabricCommInit: failure
// ---------------------------------------------------------------------------

TEST_F(DdaFabricCommInitTest, CommInit_BlockCapGatherFails_ReturnsErrorWithoutAllocating) {
  ScopedHook gather(g_bootstrapAllGather, [](void*, void*, int) { return ncclRemoteError; });

  EXPECT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclRemoteError);

  ExpectNoFabricState(*comm_);
  EXPECT_TRUE(AllReleased());
}

// Current behaviour: every failure after the block-cap gather falls back on
// this rank alone -- success, no fabric state on the comm, nothing left
// allocated.
struct FailCase {
  const char* name;
  std::function<void()> install;
};

class DdaFabricCommInitFailTest : public DdaFabricCommTest, public ::testing::WithParamInterface<FailCase> {};

TEST_P(DdaFabricCommInitFailTest, CommInit_StepFails_FallsBackWithoutFabricStateOrLeaks) {
  GetParam().install();

  EXPECT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);

  ExpectNoFabricState(*comm_);
  EXPECT_TRUE(AllReleased());
}

// Sizes that tell the allocations apart: the scratch is kScratchBytes, the
// peer table kNRanks pointers, the epoch cells kCuCount uint32s.
constexpr size_t kPeerTableBytes = kNRanks * sizeof(void*);
constexpr size_t kEpochBytes = kCuCount * sizeof(uint32_t);

INSTANTIATE_TEST_SUITE_P(
    Step, DdaFabricCommInitFailTest,
    ::testing::Values(
        FailCase{"ScratchAlloc",
                 [] {
                   FailWhen(
                       g_hipMemCreate,
                       [](hipMemGenericAllocationHandle_t*, size_t size, const hipMemAllocationProp*,
                          unsigned long long) {
                         return size == static_cast<size_t>(kScratchBytes);  // a page multiple
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
                   // The barrier stages a table of the same size; fail only the one hipMalloc made.
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
        FailCase{"PeerTableHostCopy",
                 [] {
                   FailWhen(
                       g_hipMemcpy,
                       [](void*, const void*, size_t, hipMemcpyKind kind) { return kind == hipMemcpyHostToHost; },
                       hipErrorInvalidValue);
                 }},
        FailCase{"Barrier",
                 [] {
                   // The barrier's flag buffer is the only VMM allocation smaller than the scratch.
                   FailWhen(
                       g_hipMemCreate,
                       [](hipMemGenericAllocationHandle_t*, size_t size, const hipMemAllocationProp*,
                          unsigned long long) { return size < static_cast<size_t>(kScratchBytes); },
                       hipErrorOutOfMemory);
                 }},
        FailCase{"ScratchZeroing",
                 [] {
                   FailWhen(
                       g_hipMemset,
                       [](void*, int, size_t bytes) { return bytes == static_cast<size_t>(kScratchBytes); },
                       hipErrorInvalidValue);
                 }},
        FailCase{"EpochAlloc",
                 [] {
                   FailWhen(
                       g_hipMalloc, [](void**, size_t size) { return size == kEpochBytes; }, hipErrorOutOfMemory);
                 }},
        FailCase{"EpochZeroing",
                 [] {
                   FailWhen(
                       g_hipMemset, [](void*, int, size_t bytes) { return bytes == kEpochBytes; },
                       hipErrorInvalidValue);
                 }}),
    [](const ::testing::TestParamInfo<FailCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// ncclDdaFabricCommFini
// ---------------------------------------------------------------------------

TEST_F(DdaFabricCommFiniTest, CommFini_NullComm_ReturnsSuccess) {
  EXPECT_EQ(ncclDdaFabricCommFini(nullptr), ncclSuccess);
}

TEST_F(DdaFabricCommFiniTest, CommFini_AfterInit_ReleasesEverythingAndClearsFabricState) {
  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);
  ASSERT_NE(comm_->ddaScratch, nullptr);

  EXPECT_EQ(ncclDdaFabricCommFini(comm_.get()), ncclSuccess);

  ExpectNoFabricResources(*comm_);
  EXPECT_TRUE(AllReleased());
}

TEST_F(DdaFabricCommFiniTest, CommFini_NonVmmScratch_FreesItAsAPlainBuffer) {
  void* scratch = nullptr;
  ASSERT_EQ(hipMalloc(&scratch, 4096), hipSuccess);
  comm_->ddaScratch = scratch;
  comm_->ddaScratchBytes = 4096;
  comm_->ddaScratchIsVmm = false;

  EXPECT_EQ(ncclDdaFabricCommFini(comm_.get()), ncclSuccess);

  ExpectNoFabricResources(*comm_);
  EXPECT_TRUE(AllReleased());
}

TEST_F(DdaFabricCommFiniTest, CommFini_CalledTwice_ReleasesOnce) {
  ASSERT_EQ(ncclDdaFabricCommInit(comm_.get()), ncclSuccess);
  ASSERT_EQ(ncclDdaFabricCommFini(comm_.get()), ncclSuccess);

  EXPECT_EQ(ncclDdaFabricCommFini(comm_.get()), ncclSuccess);

  ExpectNoFabricResources(*comm_);
  EXPECT_TRUE(AllReleased());
}

TEST_F(DdaFabricCommFiniTest, CommFini_NoFabricState_ReleasesNothing) {
  EXPECT_EQ(ncclDdaFabricCommFini(comm_.get()), ncclSuccess);

  ExpectNoFabricResources(*comm_);
  EXPECT_TRUE(AllReleased());
  EXPECT_TRUE(ledger_.addressFrees.empty());
}

}  // namespace
