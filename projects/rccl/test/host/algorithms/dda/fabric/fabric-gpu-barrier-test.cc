/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for FabricGpuBarrier::mallocAndInit in src/algorithms/dda/fabric/fabric_gpu_barrier.cu.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include "HipVmmLedger.h"
#include "ScopedHook.h"
#include "fakes/bootstrap_stubs.h"
#include "fakes/env_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"

#include "mem_manager.h"

#include FABRIC_GPU_BARRIER_CC_PATH

namespace {

using dda::common::FabricGpuBarrier;
using dda::common::FabricGpuBarrierResources;
using dda::common::kDdaMaxNranks;
using FlagType = FabricGpuBarrier::FlagType;

constexpr int kNRanks = 4;
constexpr int kRank = 1;
constexpr int kNBlocks = 3;
void* const kBootstrap = reinterpret_cast<void*>(0xB007);

size_t FlagBytes(int nRanks, int nBlocks) { return static_cast<size_t>(nRanks) * nBlocks * sizeof(FlagType); }

// Mirrors FabricGpuBarrier's private layout.
struct BarrierState {
  int nBlocks;
  int selfRank;
  int nRanks;
  FlagType** peerFlags;
};
static_assert(sizeof(BarrierState) == sizeof(FabricGpuBarrier), "FabricGpuBarrier layout changed");
static_assert(std::is_standard_layout<FabricGpuBarrier>::value, "FabricGpuBarrier layout is no longer fixed");
static_assert(std::is_trivially_copyable<FabricGpuBarrier>::value, "FabricGpuBarrier is no longer a plain value");
BarrierState StateOf(const FabricGpuBarrier& barrier) {
  BarrierState state;
  std::memcpy(&state, &barrier, sizeof(state));
  return state;
}

using InitResult = std::pair<std::unique_ptr<FabricGpuBarrierResources>, FabricGpuBarrier>;

class FabricGpuBarrierTest : public ::testing::Test {
 protected:
  void SetUp() override {
    SetMicroEnvAbsent("NCCL_CUMEM_SKIP_FREE");
    ledger_.Install();
    ASSERT_FALSE(rcclSkipCuMemFree());  // alloc.h caches this once per process
    g_cuMemEnable = [] { return 1; };
    auto ledgerMemset = g_hipMemset;
    g_hipMemset = [this, ledgerMemset](void* dst, int value, size_t bytes) {
      events_.push_back("memset");
      memsets_.push_back({dst, value, bytes});
      return ledgerMemset(dst, value, bytes);
    };
    // Every peer publishes what this rank published.
    g_bootstrapAllGather = [this](void* state, void* allData, int size) {
      events_.push_back("allgather");
      gatherStates_.push_back(state);
      auto* slots = static_cast<char*>(allData);
      for (int r = 0; r < nRanks_; ++r) {
        if (r != selfRank_) std::memcpy(slots + r * size, slots + selfRank_ * size, size);
      }
      return ncclSuccess;
    };
  }

  void TearDown() override {
    result_.first.reset();
    EXPECT_TRUE(ledger_.Clean()) << ledger_.reserved.size() << " reservations, " << ledger_.liveHandles.size()
                                 << " handles, " << ledger_.liveBuffers.size() << " buffers live; "
                                 << ledger_.rejected.size() << " calls refused";
    ResetBootstrapStubs();
    ResetHipFakes();
    ResetNcclFakes();
    ResetEnvFakes();
  }

  InitResult& Init(int nRanks = kNRanks, int selfRank = kRank, int nBlocks = kNBlocks,
                   struct ncclMemManager* manager = nullptr) {
    nRanks_ = nRanks;
    selfRank_ = selfRank;
    result_ = FabricGpuBarrier::mallocAndInit(nRanks, nBlocks, selfRank, kBootstrap, manager);
    return result_;
  }

  void ExpectFailsWithoutLeaking(int nRanks = kNRanks, int selfRank = kRank) {
    Init(nRanks, selfRank);
    EXPECT_EQ(result_.first, nullptr);
    EXPECT_TRUE(ledger_.Clean()) << ledger_.reserved.size() << " reservations, " << ledger_.liveHandles.size()
                                 << " handles, " << ledger_.liveBuffers.size() << " buffers live; "
                                 << ledger_.rejected.size() << " calls refused";
  }

  size_t EventIndex(const std::string& event, size_t nth = 0) const {
    for (size_t i = 0; i < events_.size(); ++i) {
      if (events_[i] == event && nth-- == 0) return i;
    }
    return events_.size();
  }

  struct Memset {
    void* dst;
    int value;
    size_t bytes;
  };

  int nRanks_ = kNRanks;
  int selfRank_ = kRank;
  HipVmmLedger ledger_;
  std::vector<std::string> events_;
  std::vector<Memset> memsets_;
  std::vector<void*> gatherStates_;
  InitResult result_{nullptr, FabricGpuBarrier{}};
};

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

struct Geometry {
  const char* name;
  int nRanks;
  int selfRank;
  int nBlocks;
};
std::string GeometryName(const ::testing::TestParamInfo<Geometry>& info) { return info.param.name; }

class FabricGpuBarrierInvalidGeometryTest : public FabricGpuBarrierTest,
                                            public ::testing::WithParamInterface<Geometry> {};

TEST_P(FabricGpuBarrierInvalidGeometryTest, MallocAndInit_InvalidGeometry_ReturnsNullWithoutAllocating) {
  const Geometry g = GetParam();
  ScopedHook create(g_hipMemCreate, g_hipMemCreate);
  ScopedHook extMalloc(g_hipExtMallocWithFlags, g_hipExtMallocWithFlags);

  EXPECT_EQ(Init(g.nRanks, g.selfRank, g.nBlocks).first, nullptr);

  EXPECT_EQ(create.calls, 0);
  EXPECT_EQ(extMalloc.calls, 0);
}

INSTANTIATE_TEST_SUITE_P(Geometry, FabricGpuBarrierInvalidGeometryTest,
                         ::testing::Values(Geometry{"NoRanks", 0, 0, kNBlocks},
                                           Geometry{"NegativeRanks", -1, 0, kNBlocks},
                                           Geometry{"RanksAboveCap", kDdaMaxNranks + 1, 0, kNBlocks},
                                           Geometry{"SelfBelowZero", kNRanks, -1, kNBlocks},
                                           Geometry{"SelfPastLastRank", kNRanks, kNRanks, kNBlocks},
                                           Geometry{"NoBlocks", kNRanks, kRank, 0},
                                           Geometry{"NegativeBlocks", kNRanks, kRank, -1}),
                         GeometryName);

class FabricGpuBarrierValidGeometryTest : public FabricGpuBarrierTest,
                                          public ::testing::WithParamInterface<Geometry> {};

TEST_P(FabricGpuBarrierValidGeometryTest, MallocAndInit_BoundaryGeometry_Succeeds) {
  const Geometry g = GetParam();

  EXPECT_NE(Init(g.nRanks, g.selfRank, g.nBlocks).first, nullptr);
}

INSTANTIATE_TEST_SUITE_P(Geometry, FabricGpuBarrierValidGeometryTest,
                         ::testing::Values(Geometry{"SingleRank", 1, 0, 1},
                                           Geometry{"RanksAtCap", kDdaMaxNranks, kDdaMaxNranks - 1, 1},
                                           Geometry{"SelfFirst", kNRanks, 0, kNBlocks},
                                           Geometry{"SelfLast", kNRanks, kNRanks - 1, kNBlocks}),
                         GeometryName);

// ---------------------------------------------------------------------------
// Success
// ---------------------------------------------------------------------------

TEST_F(FabricGpuBarrierTest, MallocAndInit_AllSucceed_ReturnsVmmFlagBufferAndHandlerAndPeerTable) {
  InitResult& r = Init();

  ASSERT_NE(r.first, nullptr);
  ASSERT_NE(r.first->selfFlagBuf, nullptr);
  EXPECT_NE(r.first->selfFlagBuf->get(), nullptr);
  EXPECT_TRUE(r.first->selfFlagBuf->isVmm());
  EXPECT_NE(r.first->fabricMemHandler, nullptr);
  ASSERT_NE(r.first->peerFlagsDev, nullptr);
  EXPECT_NE(r.first->peerFlagsDev->get(), nullptr);
}

TEST_F(FabricGpuBarrierTest, MallocAndInit_AllSucceed_ZeroesFlagBufferBeforePublishingIt) {
  InitResult& r = Init();
  ASSERT_NE(r.first, nullptr);
  void* const flags = r.first->selfFlagBuf->get();

  auto zeroing = std::find_if(memsets_.begin(), memsets_.end(), [flags](const Memset& m) {
    return m.dst == flags && m.value == 0 && m.bytes == FlagBytes(kNRanks, kNBlocks);
  });
  ASSERT_NE(zeroing, memsets_.end()) << "no zeroing of the whole flag buffer";
  const size_t nth = static_cast<size_t>(zeroing - memsets_.begin());
  EXPECT_LT(EventIndex("memset", nth), EventIndex("allgather"));
}

TEST_F(FabricGpuBarrierTest, MallocAndInit_AllSucceed_SharesTheFlagBuffersHandle) {
  hipMemGenericAllocationHandle_t exported = nullptr;
  ScopedHook exportHook(g_hipMemExportToShareableHandle,
                        [&exported](void*, hipMemGenericAllocationHandle_t handle, hipMemAllocationHandleType,
                                    unsigned long long) {
                          exported = handle;
                          return hipSuccess;
                        });

  InitResult& r = Init();

  ASSERT_NE(r.first, nullptr);
  EXPECT_EQ(exported, r.first->selfFlagBuf->vmmHandle());
}

TEST_F(FabricGpuBarrierTest, MallocAndInit_AllSucceed_ExchangesOverTheGivenBootstrap) {
  ASSERT_NE(Init().first, nullptr);

  ASSERT_FALSE(gatherStates_.empty());
  for (void* state : gatherStates_) EXPECT_EQ(state, kBootstrap);
}

// Not a page multiple, so a wrongly sized peer mapping rounds differently.
TEST_F(FabricGpuBarrierTest, MallocAndInit_AllSucceed_PeersMapTheWholeFlagBuffer) {
  constexpr int kManyBlocks = 300;  // 4800 B of flags
  InitResult& r = Init(kNRanks, kRank, kManyBlocks);
  ASSERT_NE(r.first, nullptr);
  auto* const* table = static_cast<FlagType* const*>(r.first->peerFlagsDev->get());
  const size_t ownSize = ledger_.reserved.at(r.first->selfFlagBuf->get());
  ASSERT_GE(ownSize, FlagBytes(kNRanks, kManyBlocks));

  for (int i = 0; i < kNRanks; ++i) {
    if (i == kRank) continue;
    ASSERT_EQ(ledger_.reserved.count(table[i]), 1u) << "peer " << i;
    EXPECT_EQ(ledger_.reserved.at(table[i]), ownSize) << "peer " << i;
  }
}

TEST_F(FabricGpuBarrierTest, MallocAndInit_AllSucceed_PeerTableHoldsOwnFlagBufferAndEachPeersMapping) {
  std::vector<hipMemcpyKind> kinds;
  auto ledgerCopy = g_hipMemcpy;
  ScopedHook copy(g_hipMemcpy, [&kinds, ledgerCopy](void* dst, const void* src, size_t n, hipMemcpyKind kind) {
    kinds.push_back(kind);
    return ledgerCopy(dst, src, n, kind);
  });
  InitResult& r = Init();
  ASSERT_NE(r.first, nullptr);
  auto* const* table = static_cast<FlagType* const*>(r.first->peerFlagsDev->get());
  void* const flags = r.first->selfFlagBuf->get();

  EXPECT_EQ(table[kRank], flags);
  std::set<void*> peerMappings;
  for (const auto& entry : ledger_.reserved) {
    if (entry.first != flags) peerMappings.insert(entry.first);
  }
  std::set<void*> peerSlots;
  for (int i = 0; i < kNRanks; ++i) {
    if (i == kRank) continue;
    peerSlots.insert(table[i]);
    void* mapped = nullptr;
    ASSERT_EQ(r.first->fabricMemHandler->getPeerDeviceMemPtr(i, &mapped), ncclSuccess);
    EXPECT_EQ(table[i], mapped) << "peer " << i;
  }
  EXPECT_EQ(peerSlots, peerMappings);
  EXPECT_EQ(peerSlots.size(), static_cast<size_t>(kNRanks - 1));
  for (hipMemcpyKind kind : kinds) EXPECT_TRUE(kind == hipMemcpyHostToDevice || kind == hipMemcpyDefault) << kind;
}

TEST_F(FabricGpuBarrierTest, MallocAndInit_AllSucceed_BarrierCarriesGeometryAndDevicePeerTable) {
  InitResult& r = Init();
  ASSERT_NE(r.first, nullptr);

  const BarrierState state = StateOf(r.second);
  EXPECT_EQ(state.nBlocks, kNBlocks);
  EXPECT_EQ(state.selfRank, kRank);
  EXPECT_EQ(state.nRanks, kNRanks);
  EXPECT_EQ(state.peerFlags, r.first->peerFlagsDev->get());
}

TEST_F(FabricGpuBarrierTest, MallocAndInit_WithManager_TracksFlagBufferAndUntracksPeerMappingsInIt) {
  auto manager = std::make_unique<ncclMemManager>();
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
  InitResult& r = Init(kNRanks, kRank, kNBlocks, manager.get());
  ASSERT_NE(r.first, nullptr);
  void* const flags = r.first->selfFlagBuf->get();
  std::set<void*> peers;
  for (int i = 0; i < kNRanks; ++i) {
    void* mapped = nullptr;
    if (i != kRank && r.first->fabricMemHandler->getPeerDeviceMemPtr(i, &mapped) == ncclSuccess) peers.insert(mapped);
  }

  result_.first.reset();

  EXPECT_NE(std::find(tracked.begin(), tracked.end(), std::make_pair(manager.get(), flags)), tracked.end())
      << "flag buffer not tracked in the given manager";
  for (void* peer : peers) {
    EXPECT_NE(std::find(untracked.begin(), untracked.end(), std::make_pair(manager.get(), peer)), untracked.end())
        << "peer mapping " << peer << " not untracked from the given manager";
  }
}

// ---------------------------------------------------------------------------
// Failure: each returns no resources and leaves nothing allocated
// ---------------------------------------------------------------------------

TEST_F(FabricGpuBarrierTest, MallocAndInit_VmmUnavailable_ReturnsNullWithoutLeaking) {
  ScopedHook vmm(g_cuMemEnable, [] { return 0; });

  ExpectFailsWithoutLeaking();
}

TEST_F(FabricGpuBarrierTest, MallocAndInit_FlagBufferAllocationFails_ReturnsNullWithoutLeaking) {
  ScopedHook vmm(g_cuMemEnable, [] { return 0; });
  ScopedHook extMalloc(g_hipExtMallocWithFlags, [](void** ptr, size_t, unsigned) {
    *ptr = nullptr;
    return hipErrorOutOfMemory;
  });

  ExpectFailsWithoutLeaking();

  EXPECT_TRUE(memsets_.empty()) << "zeroed a buffer that was never allocated";
}

TEST_F(FabricGpuBarrierTest, MallocAndInit_FlagBufferZeroingFails_ReturnsNullWithoutLeaking) {
  ScopedHook zeroing(g_hipMemset, [](void*, int, size_t) { return hipErrorInvalidValue; });

  ExpectFailsWithoutLeaking();
}

TEST_F(FabricGpuBarrierTest, MallocAndInit_ExchangeFails_ReturnsNullWithoutLeaking) {
  ScopedHook gather(g_bootstrapAllGather, [](void*, void*, int) { return ncclRemoteError; });

  ExpectFailsWithoutLeaking();
}

TEST_F(FabricGpuBarrierTest, MallocAndInit_SingleRankExchangeFails_ReturnsNullWithoutLeaking) {
  ScopedHook gather(g_bootstrapAllGather, [](void*, void*, int) { return ncclRemoteError; });

  ExpectFailsWithoutLeaking(/*nRanks=*/1, /*selfRank=*/0);
}

TEST_F(FabricGpuBarrierTest, MallocAndInit_ExchangeFailsAfterMappingPeers_ReturnsNullWithoutLeaking) {
  int imports = 0;
  ScopedHook import(g_hipMemImportFromShareableHandle,
                    [&imports](hipMemGenericAllocationHandle_t* handle, void*, hipMemAllocationHandleType) {
                      if (++imports == kNRanks - 1) return hipErrorInvalidValue;
                      *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
                      return hipSuccess;
                    });

  ExpectFailsWithoutLeaking();
}

TEST_F(FabricGpuBarrierTest, MallocAndInit_PeerTableAllocationFails_ReturnsNullWithoutLeaking) {
  constexpr size_t kTableBytes = kNRanks * sizeof(FlagType*);
  auto ledgerMalloc = g_hipExtMallocWithFlags;
  int tableRequests = 0;
  ScopedHook extMalloc(g_hipExtMallocWithFlags,
                       [ledgerMalloc, &tableRequests](void** ptr, size_t size, unsigned flags) {
                         if (size != kTableBytes) return ledgerMalloc(ptr, size, flags);
                         ++tableRequests;
                         *ptr = nullptr;
                         return hipErrorOutOfMemory;
                       });

  ExpectFailsWithoutLeaking();

  EXPECT_EQ(tableRequests, 1);
}

TEST_F(FabricGpuBarrierTest, MallocAndInit_PeerTableCopyFails_ReturnsNullWithoutLeaking) {
  ScopedHook copy(g_hipMemcpy, [](void*, const void*, size_t, hipMemcpyKind) { return hipErrorInvalidValue; });

  ExpectFailsWithoutLeaking();
}

}  // namespace
