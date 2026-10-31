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

}  // namespace
