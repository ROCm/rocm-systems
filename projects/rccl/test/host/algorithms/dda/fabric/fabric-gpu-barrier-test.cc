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
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "FailWhen.h"
#include "ScopedHook.h"
#include "fakes/bootstrap_stubs.h"
#include "fakes/env_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"

#include "algorithms/dda/fabric/FabricGpuBarrierState.h"
#include "algorithms/dda/fabric/FabricTestFixture.h"

#include FABRIC_GPU_BARRIER_CC_PATH

namespace {

using dda::common::FabricGpuBarrier;
using dda::common::FabricGpuBarrierResources;
using dda::common::kDdaMaxNranks;
using FlagType = FabricGpuBarrier::FlagType;

using dda_fabric_test::FabricLedgerTest;
using dda_fabric_test::kBootstrap;
using dda_fabric_test::kHandleType;
using dda_fabric_test::kNRanks;
using dda_fabric_test::kRank;
using dda_fabric_test::MemManagerSpy;
using dda_fabric_test::ParamName;
using dda_fabric_test::PeerRanks;

constexpr int kNBlocks = 3;
const uint32_t* const kAbortFlag = reinterpret_cast<const uint32_t*>(0xAB047);

// Toggle flags, epoch flags (nRanks * nBlocks each), then one epoch cell per block.
size_t FlagBytes(int nRanks, int nBlocks) {
  return (2 * static_cast<size_t>(nRanks) * nBlocks + static_cast<size_t>(nBlocks)) * sizeof(FlagType);
}

using InitResult = std::pair<std::unique_ptr<FabricGpuBarrierResources>, FabricGpuBarrier>;

class FabricGpuBarrierMicrotest : public FabricLedgerTest {
 protected:
  void SetUp() override {
    FabricLedgerTest::SetUp();
    if (HasFatalFailure()) return;
    MirrorAllGather(&nRanks_, &selfRank_);
  }

  void TearDown() override {
    result_.first.reset();
    FabricLedgerTest::TearDown();
  }

  InitResult& Init(int nRanks = kNRanks, int selfRank = kRank, int nBlocks = kNBlocks,
                   struct ncclMemManager* manager = nullptr) {
    nRanks_ = nRanks;
    selfRank_ = selfRank;
    result_ = FabricGpuBarrier::mallocAndInit(nRanks, nBlocks, selfRank, kBootstrap, manager,
                                              dda::common::DdaKernelParams{kAbortFlag});
    return result_;
  }

  int nRanks_ = kNRanks;
  int selfRank_ = kRank;
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

class FabricGpuBarrierInvalidGeometryMicrotest : public FabricGpuBarrierMicrotest,
                                                 public ::testing::WithParamInterface<Geometry> {};

TEST_P(FabricGpuBarrierInvalidGeometryMicrotest, MallocAndInit_InvalidGeometry_ReturnsNullWithoutAllocating) {
  const Geometry g = GetParam();
  ScopedHook create(g_hipMemCreate, g_hipMemCreate);
  ScopedHook extMalloc(g_hipExtMallocWithFlags, g_hipExtMallocWithFlags);

  EXPECT_EQ(Init(g.nRanks, g.selfRank, g.nBlocks).first, nullptr);

  EXPECT_EQ(create.calls, 0);
  EXPECT_EQ(extMalloc.calls, 0);
}

INSTANTIATE_TEST_SUITE_P(FabricGpuBarrierGeometry, FabricGpuBarrierInvalidGeometryMicrotest,
                         ::testing::Values(Geometry{"NoRanks", 0, 0, kNBlocks},
                                           Geometry{"NegativeRanks", -1, 0, kNBlocks},
                                           Geometry{"RanksAboveCap", kDdaMaxNranks + 1, 0, kNBlocks},
                                           Geometry{"SelfBelowZero", kNRanks, -1, kNBlocks},
                                           Geometry{"SelfPastLastRank", kNRanks, kNRanks, kNBlocks},
                                           Geometry{"NoBlocks", kNRanks, kRank, 0},
                                           Geometry{"NegativeBlocks", kNRanks, kRank, -1}),
                         ParamName<Geometry>);

class FabricGpuBarrierValidGeometryMicrotest : public FabricGpuBarrierMicrotest,
                                               public ::testing::WithParamInterface<Geometry> {};

TEST_P(FabricGpuBarrierValidGeometryMicrotest, MallocAndInit_BoundaryGeometry_Succeeds) {
  const Geometry g = GetParam();

  EXPECT_NE(Init(g.nRanks, g.selfRank, g.nBlocks).first, nullptr);
}

INSTANTIATE_TEST_SUITE_P(FabricGpuBarrierGeometry, FabricGpuBarrierValidGeometryMicrotest,
                         ::testing::Values(Geometry{"SingleRank", 1, 0, 1},
                                           Geometry{"RanksAtCap", kDdaMaxNranks, kDdaMaxNranks - 1, 1},
                                           Geometry{"SelfFirst", kNRanks, 0, kNBlocks},
                                           Geometry{"SelfLast", kNRanks, kNRanks - 1, kNBlocks}),
                         ParamName<Geometry>);

// ---------------------------------------------------------------------------
// Success
// ---------------------------------------------------------------------------

TEST_F(FabricGpuBarrierMicrotest, MallocAndInit_AllSucceed_ReturnsVmmFlagBufferAndHandlerAndPeerTable) {
  InitResult& r = Init();

  ASSERT_NE(r.first, nullptr);
  ASSERT_NE(r.first->selfFlagBuf, nullptr);
  EXPECT_NE(r.first->selfFlagBuf->get(), nullptr);
  EXPECT_TRUE(r.first->selfFlagBuf->isVmm());
  EXPECT_NE(r.first->fabricMemHandler, nullptr);
  ASSERT_NE(r.first->peerFlagsDev, nullptr);
  EXPECT_NE(r.first->peerFlagsDev->get(), nullptr);
}

TEST_F(FabricGpuBarrierMicrotest, MallocAndInit_AllSucceed_ZeroesFlagBufferBeforePublishingIt) {
  InitResult& r = Init();
  ASSERT_NE(r.first, nullptr);
  void* const flags = r.first->selfFlagBuf->get();

  const auto& memsets = ledger_.memsets;
  auto zeroing = std::find_if(memsets.begin(), memsets.end(), [flags](const HipVmmLedger::Memset& m) {
    return m.dst == flags && m.value == 0 && m.bytes == FlagBytes(kNRanks, kNBlocks);
  });
  ASSERT_NE(zeroing, memsets.end()) << "no zeroing of the whole flag buffer";
  EXPECT_LT(static_cast<size_t>(zeroing - memsets.begin()), memsetsBeforeFirstGather_);
}

TEST_F(FabricGpuBarrierMicrotest, MallocAndInit_AllSucceed_SharesTheFlagBuffersHandle) {
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

TEST_F(FabricGpuBarrierMicrotest, MallocAndInit_AllSucceed_ExchangesOverTheGivenBootstrap) {
  ASSERT_NE(Init().first, nullptr);

  ASSERT_FALSE(gatherStates_.empty());
  for (void* state : gatherStates_) EXPECT_EQ(state, kBootstrap);
}

// Not a page multiple, so a wrongly sized peer mapping rounds differently.
TEST_F(FabricGpuBarrierMicrotest, MallocAndInit_AllSucceed_PeersMapTheWholeFlagBuffer) {
  constexpr int kManyBlocks = 300;  // 10800 B of flags
  InitResult& r = Init(kNRanks, kRank, kManyBlocks);
  ASSERT_NE(r.first, nullptr);
  auto* const* table = static_cast<FlagType* const*>(r.first->peerFlagsDev->get());
  const size_t ownSize = ledger_.mappedSize.at(r.first->selfFlagBuf->get());
  ASSERT_GE(ownSize, FlagBytes(kNRanks, kManyBlocks));

  for (int i : PeerRanks()) {
    ASSERT_EQ(ledger_.mappedSize.count(table[i]), 1u) << "peer " << i;
    EXPECT_EQ(ledger_.mappedSize.at(table[i]), ownSize) << "peer " << i;
  }
}

TEST_F(FabricGpuBarrierMicrotest, MallocAndInit_AllSucceed_PeerTableHoldsOwnFlagBufferAndEachPeersMapping) {
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
  for (int i : PeerRanks()) {
    peerSlots.insert(table[i]);
    void* mapped = nullptr;
    ASSERT_EQ(r.first->fabricMemHandler->getPeerDeviceMemPtr(i, &mapped), ncclSuccess);
    EXPECT_EQ(table[i], mapped) << "peer " << i;
  }
  EXPECT_EQ(peerSlots, peerMappings);
  EXPECT_EQ(peerSlots.size(), static_cast<size_t>(kNRanks - 1));
  for (hipMemcpyKind kind : kinds) EXPECT_TRUE(kind == hipMemcpyHostToDevice || kind == hipMemcpyDefault) << kind;
}

TEST_F(FabricGpuBarrierMicrotest, MallocAndInit_AllSucceed_BarrierCarriesGeometryAndDevicePeerTable) {
  InitResult& r = Init();
  ASSERT_NE(r.first, nullptr);

  const FabricGpuBarrierState state = StateOf(r.second);
  EXPECT_EQ(state.nBlocks, kNBlocks);
  EXPECT_EQ(state.selfRank, kRank);
  EXPECT_EQ(state.nRanks, kNRanks);
  EXPECT_EQ(state.peerFlags, r.first->peerFlagsDev->get());
  EXPECT_EQ(state.params.flag, kAbortFlag);
}

TEST_F(FabricGpuBarrierMicrotest, MallocAndInit_WithManager_TracksAndUntracksEveryAllocationInIt) {
  auto manager = std::make_unique<ncclMemManager>();
  MemManagerSpy spy;
  InitResult& r = Init(kNRanks, kRank, kNBlocks, manager.get());
  ASSERT_NE(r.first, nullptr);
  void* const flags = r.first->selfFlagBuf->get();
  std::vector<void*> peers;
  for (int i : PeerRanks()) {
    void* mapped = nullptr;
    ASSERT_EQ(r.first->fabricMemHandler->getPeerDeviceMemPtr(i, &mapped), ncclSuccess);
    peers.push_back(mapped);
  }

  result_.first.reset();

  EXPECT_TRUE(spy.Tracked(manager.get(), flags, kHandleType)) << "flag buffer not tracked in the given manager";
  EXPECT_TRUE(spy.Untracked(manager.get(), flags)) << "flag buffer not untracked from the given manager";
  for (void* peer : peers) EXPECT_TRUE(spy.Untracked(manager.get(), peer)) << "peer mapping " << peer;
}

// ---------------------------------------------------------------------------
// Failure: each returns no resources and leaves nothing allocated
// ---------------------------------------------------------------------------

struct FailCase {
  const char* name;
  // Installs the failure; may return an extra check to run afterwards.
  std::function<std::function<void(const HipVmmLedger&)>()> install;
  int nRanks = kNRanks;
  int selfRank = kRank;
};

class FabricGpuBarrierFailMicrotest : public FabricGpuBarrierMicrotest,
                                      public ::testing::WithParamInterface<FailCase> {};

TEST_P(FabricGpuBarrierFailMicrotest, MallocAndInit_StepFails_ReturnsNullWithoutLeaking) {
  auto check = GetParam().install();

  EXPECT_EQ(Init(GetParam().nRanks, GetParam().selfRank).first, nullptr);

  EXPECT_TRUE(LedgerClean());
  if (check) check(ledger_);
}

constexpr size_t kTableBytes = kNRanks * sizeof(FlagType*);

INSTANTIATE_TEST_SUITE_P(
    FabricGpuBarrierStep, FabricGpuBarrierFailMicrotest,
    ::testing::Values(
        FailCase{"VmmUnavailable",
                 [] {
                   g_cuMemEnable = [] { return 0; };
                   return std::function<void(const HipVmmLedger&)>();
                 }},
        FailCase{"FlagBufferAllocation",
                 [] {
                   g_cuMemEnable = [] { return 0; };
                   FailWhen(
                       g_hipExtMallocWithFlags, [](void**, size_t, unsigned) { return true; }, hipErrorOutOfMemory);
                   // The null-buffer guard returns before the zeroing.
                   return std::function<void(const HipVmmLedger&)>(
                       [](const HipVmmLedger& l) { EXPECT_TRUE(l.memsets.empty()); });
                 }},
        FailCase{"FlagBufferZeroing",
                 [] {
                   FailWhen(g_hipMemset, [](void*, int, size_t) { return true; }, hipErrorInvalidValue);
                   return std::function<void(const HipVmmLedger&)>();
                 }},
        FailCase{"Exchange",
                 [] {
                   FailWhen(g_bootstrapAllGather, [](void*, void*, int) { return true; }, ncclRemoteError);
                   return std::function<void(const HipVmmLedger&)>();
                 }},
        // With no peers to look up afterwards, only the exchange's own result can stop it.
        FailCase{"SingleRankExchange",
                 [] {
                   FailWhen(g_bootstrapAllGather, [](void*, void*, int) { return true; }, ncclRemoteError);
                   return std::function<void(const HipVmmLedger&)>();
                 },
                 1, 0},
        FailCase{"ExchangeAfterMappingPeers",
                 [] {
                   int imports = 0;
                   FailWhen(
                       g_hipMemImportFromShareableHandle,
                       [imports](hipMemGenericAllocationHandle_t*, void*, hipMemAllocationHandleType) mutable {
                         return ++imports == kNRanks - 1;
                       },
                       hipErrorInvalidValue);
                   return std::function<void(const HipVmmLedger&)>();
                 }},
        FailCase{"PeerTableAllocation",
                 [] {
                   auto tableRequests = std::make_shared<int>(0);
                   FailWhen(
                       g_hipExtMallocWithFlags,
                       [tableRequests](void**, size_t size, unsigned) {
                         return size == kTableBytes && ++*tableRequests > 0;
                       },
                       hipErrorOutOfMemory);
                   return std::function<void(const HipVmmLedger&)>(
                       [tableRequests](const HipVmmLedger&) { EXPECT_EQ(*tableRequests, 1); });
                 }},
        FailCase{"PeerTableCopy",
                 [] {
                   FailWhen(g_hipMemcpy, [](void*, const void*, size_t, hipMemcpyKind) { return true; },
                            hipErrorInvalidValue);
                   return std::function<void(const HipVmmLedger&)>();
                 }}),
    ParamName<FailCase>);

TEST(HipVmmLedgerMicrotest, FreeOfUntrackedBuffer_IsRefused) {
  HipVmmLedger ledger;
  ledger.Install();
  int notAllocated = 0;

  EXPECT_NE(hipFree(&notAllocated), hipSuccess);

  EXPECT_EQ(ledger.rejected.size(), 1u);
  ResetHipFakes();
}

}  // namespace
