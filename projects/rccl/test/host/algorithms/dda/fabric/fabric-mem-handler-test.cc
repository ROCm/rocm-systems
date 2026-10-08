/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/algorithms/dda/fabric/fabric_mem_handler.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "ScopedHook.h"
#include "fakes/bootstrap_stubs.h"
#include "fakes/env_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"

#include "alloc.h"
#include "bootstrap.h"
#include "p2p.h"

#include FABRIC_MEM_HANDLER_CC_PATH

namespace {

constexpr int kNRanks = 4;
constexpr int kRank = 1;
constexpr size_t kPage = 4096;  // emulator allocation granularity

void* const kBootstrap = reinterpret_cast<void*>(0xB007);
void* const kSelfPtr = reinterpret_cast<void*>(0x5E1F000);
const hipMemGenericAllocationHandle_t kSelfHandle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x5E1F);
constexpr size_t kSelfSize = 2 * kPage;
constexpr uint64_t kSelfDesc = 0xD5E1F;

// Page multiples, so each advertised size is the size reserved.
uint64_t PeerDesc(int rank) { return 0xD000 + rank; }
size_t PeerSize(int rank) { return static_cast<size_t>(rank + 3) * kPage; }
hipMemGenericAllocationHandle_t HandleForDesc(uint64_t desc) {
  return reinterpret_cast<hipMemGenericAllocationHandle_t>(0x10000 + desc);
}

class FabricMemHandlerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    savedHandleType_ = ncclCuMemHandleType;
    SetMicroEnvAbsent("NCCL_CUMEM_SKIP_FREE");
    InstallHipVmmEmulator();
    ASSERT_FALSE(rcclSkipCuMemFree());  // alloc.h caches this once per process

    // Track reservations so frees can be checked.
    auto emulatedReserve = g_hipMemAddressReserve;
    auto emulatedFree = g_hipMemAddressFree;
    g_hipMemAddressReserve = [this, emulatedReserve](void** ptr, size_t size, size_t align, void* addr,
                                                     unsigned long long flags) {
      hipError_t err = emulatedReserve(ptr, size, align, addr, flags);
      if (err == hipSuccess) reserved_[*ptr] = size;
      return err;
    };
    g_hipMemAddressFree = [this, emulatedFree](void* ptr, size_t size) {
      freed_.push_back(ptr);
      auto it = reserved_.find(ptr);
      if (it == reserved_.end() || it->second != size) return hipErrorInvalidValue;
      reserved_.erase(it);
      return emulatedFree(ptr, size);
    };
    g_hipMemGetAddressRange = [this](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
      auto it = reserved_.find(ptr);
      if (it == reserved_.end()) return hipErrorInvalidValue;
      *base = ptr;
      *size = it->second;
      return hipSuccess;
    };
    g_hipMemMap = [this](void* ptr, size_t, size_t, hipMemGenericAllocationHandle_t handle, unsigned long long) {
      mappedHandle_[ptr] = handle;
      return hipSuccess;
    };

    g_hipMemExportToShareableHandle = [](void* shareable, hipMemGenericAllocationHandle_t,
                                         hipMemAllocationHandleType, unsigned long long) {
      static_cast<ncclCuDesc*>(shareable)->data = kSelfDesc;
      return hipSuccess;
    };
    g_hipMemImportFromShareableHandle = [this](hipMemGenericAllocationHandle_t* handle, void* shareable,
                                               hipMemAllocationHandleType) {
      const uint64_t desc = static_cast<ncclCuDesc*>(shareable)->data;
      importedDescs_.push_back(desc);
      *handle = HandleForDesc(desc);
      return hipSuccess;
    };
    g_hipMemRelease = [this](hipMemGenericAllocationHandle_t handle) {
      released_.push_back(handle);
      return hipSuccess;
    };
    g_bootstrapAllGather = [this](void*, void* allData, int) {
      auto* entries = static_cast<FabricExchEntry*>(allData);
      for (int r = 0; r < nRanks_; ++r) {
        if (r == rank_) continue;
        entries[r].desc.data = PeerDesc(r);
        entries[r].size = PeerSize(r);
      }
      return ncclSuccess;
    };
  }

  void TearDown() override {
    handler_.reset();
    EXPECT_TRUE(reserved_.empty()) << reserved_.size() << " peer mappings never freed";
    ncclCuMemHandleType = savedHandleType_;
    ResetBootstrapStubs();
    ResetHipFakes();
    ResetNcclFakes();
    ResetEnvFakes();
  }

  ncclFabricMemHandler* MakeHandler(int nRanks = kNRanks, int rank = kRank) {
    nRanks_ = nRanks;
    rank_ = rank;
    handler_ = std::make_unique<ncclFabricMemHandler>(kBootstrap, rank, nRanks, nullptr);
    return handler_.get();
  }

  ncclFabricMemHandler* MakeRegisteredHandler(int nRanks = kNRanks, int rank = kRank) {
    ncclFabricMemHandler* h = MakeHandler(nRanks, rank);
    EXPECT_EQ(h->addSelfDeviceMem(kSelfPtr, kSelfHandle, kSelfSize), ncclSuccess);
    return h;
  }

  ncclFabricMemHandler* MakeExchangedHandler(int nRanks = kNRanks, int rank = kRank) {
    ncclFabricMemHandler* h = MakeRegisteredHandler(nRanks, rank);
    EXPECT_EQ(h->exchangeMemPtrs(), ncclSuccess);
    return h;
  }

  auto FailImportOfPeer(int peer) {
    return [this, peer](hipMemGenericAllocationHandle_t* handle, void* shareable, hipMemAllocationHandleType) {
      const uint64_t desc = static_cast<ncclCuDesc*>(shareable)->data;
      importedDescs_.push_back(desc);
      if (desc == PeerDesc(peer)) return hipErrorInvalidValue;
      *handle = HandleForDesc(desc);
      return hipSuccess;
    };
  }

  void* PeerPtr(int peer) {
    void* p = nullptr;
    EXPECT_EQ(handler_->getPeerDeviceMemPtr(peer, &p), ncclSuccess);
    return p;
  }

  int nRanks_ = kNRanks;
  int rank_ = kRank;
  hipMemAllocationHandleType savedHandleType_ = hipMemHandleTypeNone;
  std::unique_ptr<ncclFabricMemHandler> handler_;
  std::map<void*, size_t> reserved_;  // live reservations -> size
  std::vector<void*> freed_;          // every hipMemAddressFree, in order
  std::map<void*, hipMemGenericAllocationHandle_t> mappedHandle_;
  std::vector<uint64_t> importedDescs_;
  std::vector<hipMemGenericAllocationHandle_t> released_;
};

// ---------------------------------------------------------------------------
// getPeerDeviceMemPtr
// ---------------------------------------------------------------------------

TEST_F(FabricMemHandlerTest, GetPeerDeviceMemPtr_BeforeExchange_ReturnsInvalidUsage) {
  const ncclFabricMemHandler* h = MakeRegisteredHandler();
  void* p = nullptr;

  EXPECT_EQ(h->getPeerDeviceMemPtr(kRank, &p), ncclInvalidUsage);
}

TEST_F(FabricMemHandlerTest, GetPeerDeviceMemPtr_RankOutOfRange_ReturnsInvalidArgument) {
  const ncclFabricMemHandler* h = MakeExchangedHandler();
  void* p = nullptr;

  EXPECT_EQ(h->getPeerDeviceMemPtr(-1, &p), ncclInvalidArgument);
  EXPECT_EQ(h->getPeerDeviceMemPtr(kNRanks, &p), ncclInvalidArgument);
}

TEST_F(FabricMemHandlerTest, GetPeerDeviceMemPtr_FirstAndLastRank_Succeed) {
  const ncclFabricMemHandler* h = MakeExchangedHandler();
  void* p = nullptr;

  EXPECT_EQ(h->getPeerDeviceMemPtr(0, &p), ncclSuccess);
  EXPECT_EQ(h->getPeerDeviceMemPtr(kNRanks - 1, &p), ncclSuccess);
}

TEST_F(FabricMemHandlerTest, GetPeerDeviceMemPtr_NullOutput_ReturnsInvalidArgument) {
  const ncclFabricMemHandler* h = MakeExchangedHandler();

  EXPECT_EQ(h->getPeerDeviceMemPtr(0, nullptr), ncclInvalidArgument);
}

// ---------------------------------------------------------------------------
// exchangeMemPtrs: success
// ---------------------------------------------------------------------------

TEST_F(FabricMemHandlerTest, ExchangeMemPtrs_AllSucceed_SelfSlotIsLocalPtrAndPeersAreDistinctMappings) {
  MakeExchangedHandler();

  EXPECT_EQ(PeerPtr(kRank), kSelfPtr);
  std::set<void*> peers;
  for (int r = 0; r < kNRanks; ++r) {
    if (r == kRank) continue;
    void* p = PeerPtr(r);
    EXPECT_NE(p, nullptr) << "peer " << r;
    EXPECT_NE(p, kSelfPtr) << "peer " << r;
    peers.insert(p);
  }
  EXPECT_EQ(peers.size(), static_cast<size_t>(kNRanks - 1));
}

TEST_F(FabricMemHandlerTest, ExchangeMemPtrs_ExportsSelfHandleAndImportsPeersWithConfiguredHandleType) {
  // Not the default, so a hard-coded type fails.
  ncclCuMemHandleType = hipMemHandleTypeWin32Kmt;
  ncclFabricMemHandler* h = MakeRegisteredHandler();
  auto importPeer = g_hipMemImportFromShareableHandle;
  ScopedHook exportHook(g_hipMemExportToShareableHandle,
                        [](void* shareable, hipMemGenericAllocationHandle_t handle,
                           hipMemAllocationHandleType type, unsigned long long) {
                          EXPECT_EQ(handle, kSelfHandle);
                          EXPECT_EQ(type, hipMemHandleTypeWin32Kmt);
                          static_cast<ncclCuDesc*>(shareable)->data = kSelfDesc;
                          return hipSuccess;
                        });
  ScopedHook importHook(g_hipMemImportFromShareableHandle,
                        [importPeer](hipMemGenericAllocationHandle_t* handle, void* shareable,
                                     hipMemAllocationHandleType type) {
                          EXPECT_EQ(type, hipMemHandleTypeWin32Kmt);
                          return importPeer(handle, shareable, type);
                        });

  ASSERT_EQ(h->exchangeMemPtrs(), ncclSuccess);

  EXPECT_EQ(exportHook.calls, 1);
  EXPECT_EQ(importHook.calls, kNRanks - 1);
}

TEST_F(FabricMemHandlerTest, ExchangeMemPtrs_PublishesExportedDescAndSizeAtOwnRankSlot) {
  ncclFabricMemHandler* h = MakeRegisteredHandler();
  ScopedHook gather(g_bootstrapAllGather, [](void* state, void* allData, int size) {
    EXPECT_EQ(state, kBootstrap);
    EXPECT_EQ(size, static_cast<int>(sizeof(FabricExchEntry)));
    const auto* entries = static_cast<const FabricExchEntry*>(allData);
    EXPECT_EQ(entries[kRank].desc.data, kSelfDesc);
    EXPECT_EQ(entries[kRank].size, kSelfSize);
    return ncclSystemError;
  });

  EXPECT_EQ(h->exchangeMemPtrs(), ncclSystemError);

  EXPECT_EQ(gather.calls, 1);
}

TEST_F(FabricMemHandlerTest, ExchangeMemPtrs_ImportsEachPeersDescriptorButNotItsOwn) {
  MakeExchangedHandler();

  EXPECT_EQ(importedDescs_, (std::vector<uint64_t>{PeerDesc(0), PeerDesc(2), PeerDesc(3)}));
}

TEST_F(FabricMemHandlerTest, ExchangeMemPtrs_MapsEachPeerToItsImportedHandleAtItsAdvertisedSize) {
  MakeExchangedHandler();

  for (int r = 0; r < kNRanks; ++r) {
    if (r == kRank) continue;
    void* p = PeerPtr(r);
    ASSERT_EQ(mappedHandle_.count(p), 1u) << "peer " << r;
    ASSERT_EQ(reserved_.count(p), 1u) << "peer " << r;
    EXPECT_EQ(mappedHandle_.at(p), HandleForDesc(PeerDesc(r))) << "peer " << r;
    EXPECT_EQ(reserved_.at(p), PeerSize(r)) << "peer " << r;
  }
}

TEST_F(FabricMemHandlerTest, ExchangeMemPtrs_AllSucceed_ReleasesEachImportedHandle) {
  MakeExchangedHandler();

  EXPECT_EQ(released_, (std::vector<hipMemGenericAllocationHandle_t>{
                           HandleForDesc(PeerDesc(0)), HandleForDesc(PeerDesc(2)), HandleForDesc(PeerDesc(3))}));
}

TEST_F(FabricMemHandlerTest, ExchangeMemPtrs_SingleRank_SucceedsWithoutImporting) {
  MakeRegisteredHandler(/*nRanks=*/1, /*rank=*/0);

  ASSERT_EQ(handler_->exchangeMemPtrs(), ncclSuccess);

  EXPECT_TRUE(importedDescs_.empty());
  EXPECT_EQ(PeerPtr(0), kSelfPtr);
}

TEST_F(FabricMemHandlerTest, ExchangeMemPtrs_AlreadyExchanged_ReturnsSuccessWithoutExchangingAgain) {
  ncclFabricMemHandler* h = MakeExchangedHandler();
  ScopedHook exportHook(g_hipMemExportToShareableHandle, g_hipMemExportToShareableHandle);
  ScopedHook gather(g_bootstrapAllGather, g_bootstrapAllGather);

  EXPECT_EQ(h->exchangeMemPtrs(), ncclSuccess);

  EXPECT_EQ(exportHook.calls, 0);
  EXPECT_EQ(gather.calls, 0);
}

// ---------------------------------------------------------------------------
// exchangeMemPtrs: failure
// ---------------------------------------------------------------------------

TEST_F(FabricMemHandlerTest, ExchangeMemPtrs_NoSelfMem_ReturnsInvalidUsageWithoutExporting) {
  ncclFabricMemHandler* h = MakeHandler();
  ScopedHook exportHook(g_hipMemExportToShareableHandle, g_hipMemExportToShareableHandle);  // count only
  ScopedHook gather(g_bootstrapAllGather, g_bootstrapAllGather);  // count only

  EXPECT_EQ(h->exchangeMemPtrs(), ncclInvalidUsage);

  EXPECT_EQ(exportHook.calls, 0);
  EXPECT_EQ(gather.calls, 0);
}

TEST_F(FabricMemHandlerTest, ExchangeMemPtrs_ExportFails_ReturnsCudaErrorWithoutGathering) {
  ncclFabricMemHandler* h = MakeRegisteredHandler();
  ScopedHook exportHook(g_hipMemExportToShareableHandle,
                        [](void*, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType,
                           unsigned long long) { return hipErrorInvalidValue; });
  ScopedHook gather(g_bootstrapAllGather, g_bootstrapAllGather);  // count only

  EXPECT_EQ(h->exchangeMemPtrs(), ncclUnhandledCudaError);

  EXPECT_EQ(gather.calls, 0);
}

TEST_F(FabricMemHandlerTest, ExchangeMemPtrs_AllGatherFails_PropagatesErrorWithoutImporting) {
  ncclFabricMemHandler* h = MakeRegisteredHandler();
  ScopedHook gather(g_bootstrapAllGather, [](void*, void*, int) { return ncclRemoteError; });

  EXPECT_EQ(h->exchangeMemPtrs(), ncclRemoteError);

  EXPECT_TRUE(importedDescs_.empty());
}

TEST_F(FabricMemHandlerTest, ExchangeMemPtrs_ImportFails_ReturnsCudaErrorAndStopsImporting) {
  ncclFabricMemHandler* h = MakeRegisteredHandler();
  ScopedHook import(g_hipMemImportFromShareableHandle, FailImportOfPeer(2));

  EXPECT_EQ(h->exchangeMemPtrs(), ncclUnhandledCudaError);

  EXPECT_EQ(importedDescs_, (std::vector<uint64_t>{PeerDesc(0), PeerDesc(2)}));
}

TEST_F(FabricMemHandlerTest, ExchangeMemPtrs_PeerMappingFails_ReleasesThatPeersHandleAndStops) {
  ncclFabricMemHandler* h = MakeRegisteredHandler();
  const hipMemGenericAllocationHandle_t failing = HandleForDesc(PeerDesc(2));
  auto recordMap = g_hipMemMap;
  ScopedHook map(g_hipMemMap, [failing, recordMap](void* ptr, size_t size, size_t offset,
                                                   hipMemGenericAllocationHandle_t handle, unsigned long long flags) {
    if (handle == failing) return hipErrorOutOfMemory;
    return recordMap(ptr, size, offset, handle, flags);
  });

  EXPECT_EQ(h->exchangeMemPtrs(), ncclUnhandledCudaError);

  EXPECT_EQ(released_, (std::vector<hipMemGenericAllocationHandle_t>{HandleForDesc(PeerDesc(0)), failing}));
  EXPECT_EQ(importedDescs_, (std::vector<uint64_t>{PeerDesc(0), PeerDesc(2)}));
}

TEST_F(FabricMemHandlerTest, ExchangeMemPtrs_ReleaseFails_ReturnsCudaErrorAndStops) {
  ncclFabricMemHandler* h = MakeRegisteredHandler();
  ScopedHook release(g_hipMemRelease, [](hipMemGenericAllocationHandle_t) { return hipErrorInvalidValue; });

  EXPECT_EQ(h->exchangeMemPtrs(), ncclUnhandledCudaError);

  EXPECT_EQ(release.calls, 1);
}

TEST_F(FabricMemHandlerTest, ExchangeMemPtrs_FailsAfterSomePeersMapped_PeerPointersStayUnavailable) {
  ncclFabricMemHandler* h = MakeRegisteredHandler();
  ScopedHook import(g_hipMemImportFromShareableHandle, FailImportOfPeer(2));
  ASSERT_NE(h->exchangeMemPtrs(), ncclSuccess);
  ASSERT_EQ(reserved_.size(), 1u);  // peer 0
  void* p = nullptr;

  EXPECT_EQ(h->getPeerDeviceMemPtr(0, &p), ncclInvalidUsage);
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

TEST_F(FabricMemHandlerTest, Destructor_AfterExchange_FreesEveryPeerMappingButNotSelf) {
  MakeExchangedHandler();
  std::set<void*> peers;
  for (int r = 0; r < kNRanks; ++r) {
    if (r != kRank) peers.insert(PeerPtr(r));
  }

  handler_.reset();

  EXPECT_EQ(std::set<void*>(freed_.begin(), freed_.end()), peers);
  EXPECT_EQ(freed_.size(), peers.size());
}

TEST_F(FabricMemHandlerTest, Destructor_AfterImportFailsPartway_FreesOnlyTheMappedPeers) {
  ncclFabricMemHandler* h = MakeRegisteredHandler();
  ScopedHook import(g_hipMemImportFromShareableHandle, FailImportOfPeer(3));
  ASSERT_NE(h->exchangeMemPtrs(), ncclSuccess);
  ASSERT_EQ(reserved_.size(), 2u);  // peers 0 and 2

  handler_.reset();

  EXPECT_EQ(freed_.size(), 2u);
}

TEST_F(FabricMemHandlerTest, Destructor_AfterReleaseFails_StillFreesThatPeersMapping) {
  ncclFabricMemHandler* h = MakeRegisteredHandler();
  ScopedHook release(g_hipMemRelease, [](hipMemGenericAllocationHandle_t) { return hipErrorInvalidValue; });
  ASSERT_NE(h->exchangeMemPtrs(), ncclSuccess);
  ASSERT_EQ(reserved_.size(), 1u);  // peer 0
  void* const peer0 = reserved_.begin()->first;

  handler_.reset();

  EXPECT_EQ(freed_, std::vector<void*>{peer0});
}

TEST_F(FabricMemHandlerTest, Destructor_NeverExchanged_FreesNothing) {
  MakeRegisteredHandler();

  handler_.reset();

  EXPECT_TRUE(freed_.empty());
}

}  // namespace
