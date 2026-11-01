/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/algorithms/dda/ipc/ipc_mem_handler.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <hip/hip_runtime.h>

#include "ScopedHook.h"
#include "fakes/bootstrap_stubs.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"

#include IPC_MEM_HANDLER_CC_PATH

namespace {

constexpr int kNRanks = 4;
constexpr int kRank = 1;
void* const kBootstrap = reinterpret_cast<void*>(0xB007);
void* const kSelfPtr = reinterpret_cast<void*>(0x5E1F000);

// Mirrors the slot exchangeMemPtrs() all-gathers: the IPC handle, then an ok byte.
struct ExchangeSlot {
  hipIpcMemHandle_t handle;
  uint8_t ok;
};
static_assert(sizeof(ExchangeSlot) == 65, "IPC handle exchange slot is 64 handle bytes plus one ok byte");

// Every rank but kRank.
std::vector<int> PeerRanks() {
  std::vector<int> peers;
  for (int r = 0; r < kNRanks; ++r) {
    if (r != kRank) peers.push_back(r);
  }
  return peers;
}

// Each rank's handle carries its rank in the first byte, so an open names the peer it maps.
constexpr int kHandleTag = 0x40;
hipIpcMemHandle_t HandleFor(int rank) {
  hipIpcMemHandle_t handle;
  std::memset(&handle, 0, sizeof(handle));
  handle.reserved[0] = static_cast<char>(kHandleTag + rank);
  return handle;
}
int RankOf(const hipIpcMemHandle_t& handle) { return handle.reserved[0] - kHandleTag; }

void* PeerMapping(int rank) { return reinterpret_cast<void*>(0xA0000 + 0x1000 * rank); }

class IpcMemHandlerMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hipIpcGetMemHandle = [](hipIpcMemHandle_t* handle, void* devPtr) {
      EXPECT_EQ(devPtr, kSelfPtr);
      *handle = HandleFor(kRank);
      return hipSuccess;
    };
    // Plays the other ranks: fills each peer slot with that peer's handle and ok byte, and records what
    // this rank put in its own slot.
    g_bootstrapAllGather = [this](void* state, void* allData, int size) {
      ++gathers_;
      gatherSize_ = size;
      EXPECT_EQ(state, kBootstrap);
      if (size != static_cast<int>(sizeof(ExchangeSlot))) {
        ADD_FAILURE() << "unexpected slot size " << size;
        return ncclInternalError;
      }
      auto* slots = static_cast<ExchangeSlot*>(allData);
      selfOkAtGather_ = slots[kRank].ok;
      selfHandleAtGather_ = slots[kRank].handle;
      for (int r : PeerRanks()) {
        slots[r].handle = HandleFor(r);
        slots[r].ok = peerOk_[static_cast<size_t>(r)];
      }
      return ncclSuccess;
    };
    g_hipIpcOpenMemHandle = [this](void** devPtr, hipIpcMemHandle_t handle, unsigned flags) {
      EXPECT_EQ(flags, static_cast<unsigned>(hipIpcMemLazyEnablePeerAccess));
      const int peer = RankOf(handle);
      opened_.push_back(peer);
      if (peer == failOpenOf_) return hipErrorInvalidValue;
      *devPtr = PeerMapping(peer);
      return hipSuccess;
    };
    g_hipIpcCloseMemHandle = [this](void* ptr) {
      closed_.push_back(ptr);
      return hipSuccess;
    };
  }

  void TearDown() override {
    handler_.reset();
    ResetBootstrapStubs();
    ResetHipFakes();
    ResetNcclFakes();
  }

  ncclIpcMemHandler* MakeHandler() {
    handler_ = std::make_unique<ncclIpcMemHandler>(kBootstrap, kRank, kNRanks);
    return handler_.get();
  }

  ncclIpcMemHandler* MakeRegisteredHandler() {
    ncclIpcMemHandler* h = MakeHandler();
    EXPECT_EQ(h->addSelfDeviceMemPtr(kSelfPtr), ncclSuccess);
    return h;
  }

  size_t ClosedCount(void* ptr) const { return static_cast<size_t>(std::count(closed_.begin(), closed_.end(), ptr)); }

  std::unique_ptr<ncclIpcMemHandler> handler_;
  std::vector<uint8_t> peerOk_ = std::vector<uint8_t>(kNRanks, 1);
  int failOpenOf_ = -1;
  int gathers_ = 0;
  int gatherSize_ = 0;
  int selfOkAtGather_ = -1;
  hipIpcMemHandle_t selfHandleAtGather_ = {};
  std::vector<int> opened_;
  std::vector<void*> closed_;
};

// A rank with no local pointer still enters the all-gather; returning before it would leave every peer blocked
// there.
TEST_F(IpcMemHandlerMicrotest, AllRanksEnterTheExchangeWhenTheLocalPointerIsMissing) {
  ncclIpcMemHandler* h = MakeHandler();
  ScopedHook exportHook(g_hipIpcGetMemHandle, g_hipIpcGetMemHandle);  // count only

  EXPECT_NE(h->exchangeMemPtrs(), ncclSuccess);

  EXPECT_EQ(gathers_, 1);
  EXPECT_EQ(gatherSize_, static_cast<int>(sizeof(ExchangeSlot)));
  EXPECT_EQ(selfOkAtGather_, 0);
  EXPECT_EQ(exportHook.calls, 0);
  EXPECT_TRUE(opened_.empty());
}

// A failed handle export is published as ok = 0 in this rank's slot; the rank does not skip the all-gather.
TEST_F(IpcMemHandlerMicrotest, LocalExportFailureIsReportedInTheSlotNotByReturningEarly) {
  ncclIpcMemHandler* h = MakeRegisteredHandler();
  ScopedHook exportHook(g_hipIpcGetMemHandle, [](hipIpcMemHandle_t*, void*) { return hipErrorInvalidValue; });

  EXPECT_NE(h->exchangeMemPtrs(), ncclSuccess);

  EXPECT_EQ(exportHook.calls, 1);
  EXPECT_EQ(gathers_, 1);
  EXPECT_EQ(selfOkAtGather_, 0);
  EXPECT_TRUE(opened_.empty());
}

// One peer could not export (the last rank, so a short scan of the slots misses it): this rank fails too, without
// opening any peer, even though its own export succeeded.
TEST_F(IpcMemHandlerMicrotest, PeerReportsFailureEveryRankFails) {
  ncclIpcMemHandler* h = MakeRegisteredHandler();
  peerOk_[kNRanks - 1] = 0;

  EXPECT_NE(h->exchangeMemPtrs(), ncclSuccess);

  EXPECT_EQ(gathers_, 1);
  EXPECT_EQ(selfOkAtGather_, 1);
  EXPECT_TRUE(opened_.empty());
  void* p = nullptr;
  EXPECT_EQ(h->getPeerDeviceMemPtr(0, &p), ncclInvalidUsage);
}

// Opening the last peer fails after peers 0 and 2 were mapped. The destructor skips a handler that never finished
// exchanging, so exchangeMemPtrs() itself has to close those two mappings -- once each, and never this rank's own
// pointer.
TEST_F(IpcMemHandlerMicrotest, PartialOpenFailureClosesTheAlreadyOpenedPeers) {
  ncclIpcMemHandler* h = MakeRegisteredHandler();
  failOpenOf_ = 3;

  EXPECT_EQ(h->exchangeMemPtrs(), ncclUnhandledCudaError);

  EXPECT_EQ(opened_, (std::vector<int>{0, 2, 3}));
  EXPECT_EQ(ClosedCount(PeerMapping(0)), 1u);
  EXPECT_EQ(ClosedCount(PeerMapping(2)), 1u);
  EXPECT_EQ(ClosedCount(kSelfPtr), 0u);
  EXPECT_EQ(closed_.size(), 2u);
  void* p = nullptr;
  EXPECT_EQ(h->getPeerDeviceMemPtr(0, &p), ncclInvalidUsage);

  handler_.reset();

  EXPECT_EQ(closed_.size(), 2u);
}

// Every peer's handle is opened once, its mapping is what getPeerDeviceMemPtr returns, and the destructor closes
// each peer mapping exactly once and never this rank's own pointer.
TEST_F(IpcMemHandlerMicrotest, HealthyExchangeOpensEveryPeerAndPublishesPointers) {
  ncclIpcMemHandler* h = MakeRegisteredHandler();

  ASSERT_EQ(h->exchangeMemPtrs(), ncclSuccess);

  EXPECT_EQ(gathers_, 1);
  EXPECT_EQ(selfOkAtGather_, 1);
  EXPECT_EQ(RankOf(selfHandleAtGather_), kRank);
  EXPECT_EQ(opened_, PeerRanks());
  void* p = nullptr;
  ASSERT_EQ(h->getPeerDeviceMemPtr(kRank, &p), ncclSuccess);
  EXPECT_EQ(p, kSelfPtr);
  for (int r : PeerRanks()) {
    p = nullptr;
    ASSERT_EQ(h->getPeerDeviceMemPtr(r, &p), ncclSuccess) << "peer " << r;
    EXPECT_EQ(p, PeerMapping(r)) << "peer " << r;
  }
  EXPECT_TRUE(closed_.empty());

  handler_.reset();

  for (int r : PeerRanks()) EXPECT_EQ(ClosedCount(PeerMapping(r)), 1u) << "peer " << r;
  EXPECT_EQ(ClosedCount(kSelfPtr), 0u);
  EXPECT_EQ(closed_.size(), static_cast<size_t>(kNRanks - 1));
}

}  // namespace
