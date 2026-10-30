/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only regression tests for ordered DDA IPC teardown.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "ScopedHook.h"
#include "fakes/bootstrap_stubs.h"
#include "fakes/hip_fakes.h"

#include "bootstrap.h"
#include "comm.h"

#include DDA_IPC_MEM_HANDLER_CC_PATH
#include DDA_IPC_GPU_BARRIER_CU_PATH
#include DDA_IPC_INIT_CU_PATH

// Owned by src/init.cc, which this binary does not compile.
struct allocationTracker allocTracker[MAX_ALLOC_TRACK_NGPU] = {};

namespace {

constexpr int kRanks = nccl_dda_detail::kDdaNranks;
constexpr auto kRendezvousTimeout = std::chrono::seconds(3);

int64_t g_ddaEnable = 1;

class SimBootstrap {
public:
  struct Handle {
    SimBootstrap* sim;
    int rank;
  };

  SimBootstrap() {
    for (int rank = 0; rank < kRanks; ++rank) {
      handles_[rank] = Handle{this, rank};
    }
  }

  void* handle(int rank) { return &handles_[rank]; }

  ncclResult_t allGather(int rank, void* allData, int size) {
    std::unique_lock<std::mutex> lock(mu_);
    GatherRound& round = gatherRounds_[nextGather_[rank]++];
    if (round.size < 0) {
      round.size = size;
      round.data.assign(static_cast<size_t>(size) * kRanks, 0);
    } else if (round.size != size) {
      round.mismatch = true;
      ++desyncs;
    }
    if (!round.mismatch) {
      std::memcpy(round.data.data() + static_cast<size_t>(rank) * size,
                  static_cast<char*>(allData) + static_cast<size_t>(rank) * size, size);
    }
    ++round.arrived;
    cv_.notify_all();
    if (!cv_.wait_for(lock, kRendezvousTimeout, [&] { return round.arrived == kRanks; })) {
      ++timeouts;
      return ncclRemoteError;
    }
    if (round.mismatch) {
      return ncclInternalError;
    }
    std::memcpy(allData, round.data.data(), static_cast<size_t>(size) * kRanks);
    return ncclSuccess;
  }

  ncclResult_t barrier(int rank, int nranks, int tag) {
    std::unique_lock<std::mutex> lock(mu_);
    const int roundIndex = nextBarrier_[rank]++;
    BarrierRound& round = barrierRounds_[roundIndex];
    if (round.nranks < 0) {
      round.nranks = nranks;
      round.tag = tag;
    } else if (round.nranks != nranks || round.tag != tag) {
      round.mismatch = true;
      ++desyncs;
    }
    ++round.arrived;
    ++barrierCalls;
    cv_.notify_all();
    if (!cv_.wait_for(lock, kRendezvousTimeout, [&] { return round.arrived == kRanks; })) {
      ++timeouts;
      return ncclRemoteError;
    }
    if (round.mismatch) {
      return ncclInternalError;
    }
    return ncclSuccess;
  }

  int timeouts = 0;
  int desyncs = 0;
  int barrierCalls = 0;

private:
  struct GatherRound {
    int size = -1;
    int arrived = 0;
    bool mismatch = false;
    std::vector<char> data;
  };

  struct BarrierRound {
    int nranks = -1;
    int tag = -1;
    int arrived = 0;
    bool mismatch = false;
  };

  std::mutex mu_;
  std::condition_variable cv_;
  std::map<int, GatherRound> gatherRounds_;
  std::map<int, BarrierRound> barrierRounds_;
  int nextGather_[kRanks] = {};
  int nextBarrier_[kRanks] = {};
  Handle handles_[kRanks];
};

class DdaIpcLifecycleMicrotest : public ::testing::Test {
protected:
  void SetUp() override {
    g_ddaEnable = 1;
    for (int rank = 0; rank < kRanks; ++rank) {
      peerInfo_[rank].rank = rank;
      peerInfo_[rank].cudaDev = rank;
      comms_[rank] = std::make_unique<ncclComm>();
      ncclComm* comm = comms_[rank].get();
      comm->rank = rank;
      comm->nRanks = kRanks;
      comm->nNodes = 1;
      comm->archName = archName_;
      comm->peerInfo = peerInfo_;
      comm->bootstrap = bootstrap_.handle(rank);
      comm->abortFlag = &abortFlags_[rank];
    }
  }

  void TearDown() override {
    bool resourcesRemain = false;
    for (const auto& comm : comms_) {
      resourcesRemain |= comm->ddaIpcMemHandler != nullptr || comm->ddaIpcBarrierState != nullptr ||
                         comm->ddaScratch != nullptr;
    }
    if (resourcesRemain) {
      finalizeDdaIpcSerially();
    }
  }

  void* allocate(size_t requested) {
    const size_t bytes = requested == DDA_IPC_BUFFER_SIZE ? 64 : std::max<size_t>(requested, 64);
    void* ptr = std::malloc(bytes);
    std::lock_guard<std::mutex> lock(memoryMu_);
    liveAllocations_.insert(ptr);
    return ptr;
  }

  void initializeDdaIpcForAllRanks() {
    std::vector<std::thread> threads;
    for (int rank = 0; rank < kRanks; ++rank) {
      threads.emplace_back([this, rank] { initResults_[rank] = ncclDdaIpcCommInit(comms_[rank].get()); });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }

  void closeDdaIpcPeerMappingsForAllRanks() {
    std::vector<std::thread> threads;
    for (int rank = 0; rank < kRanks; ++rank) {
      threads.emplace_back(
        [this, rank] { closeResults_[rank] = ncclDdaIpcCommClosePeerMappings(comms_[rank].get()); });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }

  void finalizeDdaIpcSerially() {
    for (int rank = 0; rank < kRanks; ++rank) {
      finiResults_[rank] = ncclDdaIpcCommFini(comms_[rank].get());
    }
  }

  int openMappingCount() {
    std::lock_guard<std::mutex> lock(memoryMu_);
    int count = 0;
    for (const auto& entry : openMappings_) {
      count += entry.second;
    }
    return count;
  }

  char archName_[8] = "gfx942";
  uint32_t abortFlags_[kRanks] = {};
  ncclPeerInfo peerInfo_[kRanks] = {};
  std::unique_ptr<ncclComm> comms_[kRanks];
  ncclResult_t initResults_[kRanks] = {};
  ncclResult_t closeResults_[kRanks] = {};
  ncclResult_t finiResults_[kRanks] = {};
  SimBootstrap bootstrap_;

  std::mutex memoryMu_;
  std::set<void*> liveAllocations_;
  std::map<void*, int> openMappings_;
  int earlyFrees_ = 0;

  ScopedHook<ncclResult_t(void*, void*, int)> bootstrapAllGatherHook_{
    g_bootstrapAllGather, [](void* state, void* allData, int size) {
    auto* handle = static_cast<SimBootstrap::Handle*>(state);
    return handle->sim->allGather(handle->rank, allData, size);
  }};
  ScopedHook<ncclResult_t(void*, int, int, int)> bootstrapBarrierHook_{
    g_bootstrapBarrier, [](void* state, int rank, int nranks, int tag) {
    auto* handle = static_cast<SimBootstrap::Handle*>(state);
    EXPECT_EQ(handle->rank, rank);
    return handle->sim->barrier(rank, nranks, tag);
  }};
  ScopedHook<hipError_t(int*, int, int)> p2pHook_{g_hipDeviceCanAccessPeer, [](int* canAccess, int, int) {
    *canAccess = 1;
    return hipSuccess;
  }};
  ScopedHook<hipError_t(void**, size_t, unsigned)> extMallocHook_{
    g_hipExtMallocWithFlags, [this](void** ptr, size_t size, unsigned) {
    *ptr = allocate(size);
    return *ptr == nullptr ? hipErrorOutOfMemory : hipSuccess;
  }};
  ScopedHook<hipError_t(void**, size_t)> mallocHook_{g_hipMalloc, [this](void** ptr, size_t size) {
    *ptr = allocate(size);
    return *ptr == nullptr ? hipErrorOutOfMemory : hipSuccess;
  }};
  ScopedHook<hipError_t(void*)> freeHook_{g_hipFree, [this](void* ptr) {
    if (ptr == nullptr) {
      return hipSuccess;
    }
    {
      std::lock_guard<std::mutex> lock(memoryMu_);
      if (openMappings_[ptr] != 0) {
        ++earlyFrees_;
      }
      EXPECT_EQ(1u, liveAllocations_.erase(ptr));
    }
    std::free(ptr);
    return hipSuccess;
  }};
  ScopedHook<hipError_t(void*, int, size_t)> memsetHook_{
    g_hipMemset, [](void*, int, size_t) { return hipSuccess; }};
  ScopedHook<hipError_t(void*, const void*, size_t, hipMemcpyKind)> memcpyHook_{
    g_hipMemcpy, [](void* dst, const void* src, size_t bytes, hipMemcpyKind) {
    std::memcpy(dst, src, bytes);
    return hipSuccess;
  }};
  ScopedHook<hipError_t(hipIpcMemHandle_t*, void*)> exportHook_{
    g_hipIpcGetMemHandle, [](hipIpcMemHandle_t* handle, void* ptr) {
    std::memset(handle, 0, sizeof(*handle));
    std::memcpy(handle, &ptr, sizeof(ptr));
    return hipSuccess;
  }};
  ScopedHook<hipError_t(void**, hipIpcMemHandle_t, unsigned)> openHook_{
    g_hipIpcOpenMemHandle, [this](void** ptr, hipIpcMemHandle_t handle, unsigned) {
    std::memcpy(ptr, &handle, sizeof(*ptr));
    std::lock_guard<std::mutex> lock(memoryMu_);
    ++openMappings_[*ptr];
    return hipSuccess;
  }};
  ScopedHook<hipError_t(void*)> closeHook_{g_hipIpcCloseMemHandle, [this](void* ptr) {
    std::lock_guard<std::mutex> lock(memoryMu_);
    const bool failThisClose = ++closeCalls_ == failCloseCall_;
    auto it = openMappings_.find(ptr);
    EXPECT_NE(it, openMappings_.end());
    if (it != openMappings_.end()) {
      EXPECT_GT(it->second, 0);
      --it->second;
    }
    return failThisClose ? hipErrorInvalidValue : hipSuccess;
  }};

  int closeCalls_ = 0;
  int failCloseCall_ = -1;
};

TEST_F(DdaIpcLifecycleMicrotest, NormalDestroyClosesPeerMappingsBeforeSerialCommFree) {
  initializeDdaIpcForAllRanks();
  for (int rank = 0; rank < kRanks; ++rank) {
    ASSERT_EQ(ncclSuccess, initResults_[rank]) << "rank " << rank;
    ASSERT_NE(nullptr, comms_[rank]->ddaIpcMemHandler) << "rank " << rank;
    ASSERT_NE(nullptr, comms_[rank]->ddaIpcBarrierState) << "rank " << rank;
  }
  EXPECT_EQ(2 * kRanks * (kRanks - 1), openMappingCount());

  // commDestroySync runs concurrently per local rank and closes mappings.
  closeDdaIpcPeerMappingsForAllRanks();
  EXPECT_EQ(0, openMappingCount());
  for (int rank = 0; rank < kRanks; ++rank) {
    EXPECT_EQ(ncclSuccess, closeResults_[rank]) << "rank " << rank;
  }

  // commReclaim subsequently calls commFree serially for every local rank.
  // This must neither block nor free an allocation that a peer still maps.
  finalizeDdaIpcSerially();
  EXPECT_EQ(0, bootstrap_.timeouts);
  EXPECT_EQ(0, bootstrap_.desyncs);
  EXPECT_EQ(0, bootstrap_.barrierCalls);
  EXPECT_EQ(0, earlyFrees_);
  EXPECT_TRUE(liveAllocations_.empty());
  for (int rank = 0; rank < kRanks; ++rank) {
    EXPECT_EQ(ncclSuccess, finiResults_[rank]) << "rank " << rank;
  }
}

TEST_F(DdaIpcLifecycleMicrotest, DisabledDdaKeepsScratchForCeScratchCollectives) {
  // CE-scratch AllGather/AlltoAll select on comm->ddaScratch independently of
  // RCCL_DDA_ENABLE, so disabling DDA dispatch must not skip IPC scratch setup.
  g_ddaEnable = 0;

  initializeDdaIpcForAllRanks();
  for (int rank = 0; rank < kRanks; ++rank) {
    ASSERT_EQ(ncclSuccess, initResults_[rank]) << "rank " << rank;
    EXPECT_NE(nullptr, comms_[rank]->ddaScratch) << "rank " << rank;
    EXPECT_GT(comms_[rank]->ddaScratchBytes, 0u) << "rank " << rank;
    EXPECT_NE(nullptr, comms_[rank]->ddaPeerPtrsHost) << "rank " << rank;
    EXPECT_NE(nullptr, comms_[rank]->ddaIpcMemHandler) << "rank " << rank;
  }
  closeDdaIpcPeerMappingsForAllRanks();
  finalizeDdaIpcSerially();

  EXPECT_EQ(0, bootstrap_.timeouts);
  EXPECT_EQ(0, bootstrap_.desyncs);
  EXPECT_EQ(0, bootstrap_.barrierCalls);
  EXPECT_EQ(0, earlyFrees_);
  EXPECT_EQ(0, openMappingCount());
  EXPECT_TRUE(liveAllocations_.empty());
  for (int rank = 0; rank < kRanks; ++rank) {
    EXPECT_EQ(ncclSuccess, initResults_[rank]) << "rank " << rank;
    EXPECT_EQ(ncclSuccess, finiResults_[rank]) << "rank " << rank;
  }
}

TEST_F(DdaIpcLifecycleMicrotest, FinalizationWithoutIpcInitializationIsNonCollectiveNoOp) {
  finalizeDdaIpcSerially();

  EXPECT_EQ(0, bootstrap_.barrierCalls);
  EXPECT_TRUE(liveAllocations_.empty());
  for (int rank = 0; rank < kRanks; ++rank) {
    EXPECT_EQ(ncclSuccess, finiResults_[rank]) << "rank " << rank;
  }
}

TEST_F(DdaIpcLifecycleMicrotest, RevokeClosesAllLocalPeerMappingsBeforeSerialCommFree) {
  initializeDdaIpcForAllRanks();
  for (int rank = 0; rank < kRanks; ++rank) {
    comms_[rank]->revokedFlag = true;
  }

  // commReclaim closes every revoked local rank's mappings before its serial
  // commCleanup loop, without the normal-destroy barrier.
  closeDdaIpcPeerMappingsForAllRanks();
  EXPECT_EQ(0, openMappingCount());
  finalizeDdaIpcSerially();

  EXPECT_EQ(0, bootstrap_.timeouts);
  EXPECT_EQ(0, bootstrap_.desyncs);
  EXPECT_EQ(0, bootstrap_.barrierCalls);
  EXPECT_EQ(0, earlyFrees_);
  EXPECT_TRUE(liveAllocations_.empty());
  for (int rank = 0; rank < kRanks; ++rank) {
    EXPECT_EQ(ncclSuccess, closeResults_[rank]) << "rank " << rank;
    EXPECT_EQ(ncclSuccess, finiResults_[rank]) << "rank " << rank;
  }
}

TEST_F(DdaIpcLifecycleMicrotest, FiniWithoutPriorCloseStaysLocalAndClosesPeerMappings) {
  initializeDdaIpcForAllRanks();
  abortFlags_[0] = 1;

  // Direct commFree paths (e.g. init failure) have no earlier close stage.
  // Fini must stay local and still release every mapping and allocation.
  finalizeDdaIpcSerially();

  EXPECT_EQ(0, bootstrap_.timeouts);
  EXPECT_EQ(0, bootstrap_.desyncs);
  EXPECT_EQ(0, bootstrap_.barrierCalls);
  EXPECT_EQ(0, openMappingCount());
  EXPECT_TRUE(liveAllocations_.empty());
  for (int rank = 0; rank < kRanks; ++rank) {
    EXPECT_EQ(ncclSuccess, finiResults_[rank]) << "rank " << rank;
  }
}

TEST_F(DdaIpcLifecycleMicrotest, PeerCloseFailureIsBestEffortAndSerialCleanupCompletes) {
  initializeDdaIpcForAllRanks();
  failCloseCall_ = 1;

  closeDdaIpcPeerMappingsForAllRanks();
  finalizeDdaIpcSerially();

  EXPECT_EQ(0, bootstrap_.timeouts);
  EXPECT_EQ(0, bootstrap_.desyncs);
  EXPECT_EQ(0, bootstrap_.barrierCalls);
  EXPECT_EQ(2 * kRanks * (kRanks - 1), closeCalls_);
  EXPECT_EQ(0, openMappingCount());
  EXPECT_TRUE(liveAllocations_.empty());
  for (int rank = 0; rank < kRanks; ++rank) {
    EXPECT_EQ(ncclSuccess, closeResults_[rank]) << "rank " << rank;
    EXPECT_EQ(ncclSuccess, finiResults_[rank]) << "rank " << rank;
  }
}

TEST(DdaIpcMemHandlerMicrotest, PartialOpenFailureClosesMappingsOpenedBeforeTheFailure) {
  constexpr int nranks = 4;
  constexpr int selfRank = 0;
  void* const self = reinterpret_cast<void*>(0x1000);
  int openCalls = 0;
  int closeCalls = 0;

  ScopedHook allGather(g_bootstrapAllGather, [](void*, void* allData, int size) {
    EXPECT_EQ(static_cast<int>(sizeof(hipIpcMemHandle_t)), size);
    auto* handles = static_cast<hipIpcMemHandle_t*>(allData);
    for (int rank = 0; rank < nranks; ++rank) {
      void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x1000 + rank * 0x100));
      std::memset(&handles[rank], 0, sizeof(handles[rank]));
      std::memcpy(&handles[rank], &ptr, sizeof(ptr));
    }
    return ncclSuccess;
  });
  ScopedHook exportHandle(g_hipIpcGetMemHandle, [](hipIpcMemHandle_t* handle, void* ptr) {
    std::memset(handle, 0, sizeof(*handle));
    std::memcpy(handle, &ptr, sizeof(ptr));
    return hipSuccess;
  });
  ScopedHook openHandle(g_hipIpcOpenMemHandle, [&openCalls](void** ptr, hipIpcMemHandle_t handle, unsigned) {
    ++openCalls;
    if (openCalls == 2) {
      *ptr = nullptr;
      return hipErrorInvalidValue;
    }
    std::memcpy(ptr, &handle, sizeof(*ptr));
    return hipSuccess;
  });
  ScopedHook closeHandle(g_hipIpcCloseMemHandle, [&closeCalls](void*) {
    ++closeCalls;
    return hipSuccess;
  });

  ncclIpcMemHandler handler(nullptr, selfRank, nranks);
  ASSERT_EQ(ncclSuccess, handler.addSelfDeviceMemPtr(self));
  EXPECT_EQ(ncclUnhandledCudaError, handler.exchangeMemPtrs());
  EXPECT_EQ(2, openCalls);
  EXPECT_EQ(0, closeCalls);
  EXPECT_EQ(ncclSuccess, handler.closePeerMappings());
  EXPECT_EQ(1, closeCalls);
  EXPECT_EQ(ncclSuccess, handler.closePeerMappings());
  EXPECT_EQ(1, closeCalls);
}

} // namespace

int64_t rcclParamDdaEnable() {
  return g_ddaEnable;
}
