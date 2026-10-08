/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for src/algorithms/dda/ipc/ipc_init.cu and the IPC helpers it drives. Eight threads play the
 * eight ranks; bootstrapAllGather is an in-process rendezvous with a bounded wait, so a stranded rank is a timeout.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

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

// Owned by src/init.cc, which this binary does not compile; alloc.h's inline VMM helpers reference it.
struct allocationTracker allocTracker[MAX_ALLOC_TRACK_NGPU] = {};

namespace {

constexpr int kRanks = nccl_dda_detail::kDdaNranks;
constexpr int kFaultyRank = 5;
// Not the first open, so the mappings opened before the failure must be closed again.
constexpr int kFailingOpen = 2;
constexpr auto kRendezvousTimeout = std::chrono::seconds(3);

// In-process stand-in for the bootstrap ring: the Nth all-gather call of every rank meets the Nth call of the others.
class SimBootstrap {
public:
  struct Handle {
    SimBootstrap* sim;
    int rank;
  };

  SimBootstrap() {
    for (int r = 0; r < kRanks; ++r) {
      handles_[r] = Handle{this, r};
    }
  }

  void* handle(int rank) { return &handles_[rank]; }

  ncclResult_t allGather(int rank, void* allData, int size) {
    std::unique_lock<std::mutex> lk(mu_);
    Round& round = rounds_[nextSeq_[rank]++];
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
    if (!cv_.wait_for(lk, kRendezvousTimeout, [&] { return round.arrived == kRanks; })) {
      ++timeouts;
      return ncclRemoteError;
    }
    if (round.mismatch) {
      return ncclInternalError;
    }
    std::memcpy(allData, round.data.data(), static_cast<size_t>(size) * kRanks);
    return ncclSuccess;
  }

  int timeouts = 0;
  int desyncs = 0;

private:
  struct Round {
    int size = -1;
    int arrived = 0;
    bool mismatch = false;
    std::vector<char> data;
  };

  std::mutex mu_;
  std::condition_variable cv_;
  std::map<int, Round> rounds_;
  int nextSeq_[kRanks] = {};
  Handle handles_[kRanks];
};

enum class Fault {
  kNone,
  kNoP2p,
  kScratchAlloc,
  kScratchExport,
  kScratchOpen,
  kPeerTableAlloc,
  kMailboxAlloc,
  kMailboxExport,
  kMailboxOpen,
};

thread_local int t_rank = -1;
thread_local int t_exports = 0;
thread_local int t_opens = 0;

class DdaIpcInitMicrotest : public ::testing::Test {
protected:
  void SetUp() override {
    for (int r = 0; r < kRanks; ++r) {
      peerInfo_[r].rank = r;
      peerInfo_[r].cudaDev = r;
    }
    for (int r = 0; r < kRanks; ++r) {
      comms_[r] = std::make_unique<ncclComm>();
      ncclComm* c = comms_[r].get();
      c->rank = r;
      c->nRanks = kRanks;
      c->nNodes = 1;
      c->archName = archName_;
      c->peerInfo = peerInfo_;
      c->bootstrap = sim_.handle(r);
    }
  }

  void TearDown() override {
    for (auto& c : comms_) {
      ncclDdaIpcCommFini(c.get());
    }
  }

  bool faultHere(Fault f) const { return fault_ == f && t_rank == kFaultyRank; }

  hipError_t deviceAlloc(void** ptr, size_t size) {
    *ptr = std::malloc(size < 64 ? 64 : size);
    std::lock_guard<std::mutex> lk(mu_);
    live_.insert(*ptr);
    return hipSuccess;
  }

  // Runs ncclDdaIpcCommInit on all ranks concurrently; the fixture hooks fake every HIP and bootstrap seam it reaches.
  void runInit(Fault fault) {
    fault_ = fault;
    std::vector<std::thread> threads;
    for (int r = 0; r < kRanks; ++r) {
      threads.emplace_back([this, r] {
        t_rank = r;
        t_exports = 0;
        t_opens = 0;
        results_[r] = ncclDdaIpcCommInit(comms_[r].get());
      });
    }
    for (auto& t : threads) {
      t.join();
    }
  }

  static bool ddaIpcOn(const ncclComm* c) {
    return c->ddaIpcMemHandler != nullptr && c->ddaIpcBarrierState != nullptr && c->ddaScratch != nullptr &&
           c->ddaPeerPtrsDev != nullptr && c->ddaPeerPtrsHost != nullptr;
  }

  static bool ddaIpcOff(const ncclComm* c) {
    return c->ddaIpcMemHandler == nullptr && c->ddaIpcBarrierState == nullptr && c->ddaScratch == nullptr &&
           c->ddaPeerPtrsDev == nullptr && c->ddaPeerPtrsHost == nullptr;
  }

  // A single-rank fault must neither strand a peer in a bootstrap exchange nor leave any rank with DDA IPC on.
  void expectDisabledEverywhereWithoutHang(Fault fault) {
    runInit(fault);
    EXPECT_EQ(0, sim_.timeouts) << "a rank was left waiting in a bootstrap all-gather";
    EXPECT_EQ(0, sim_.desyncs) << "ranks paired different bootstrap all-gathers";
    for (int r = 0; r < kRanks; ++r) {
      EXPECT_EQ(ncclSuccess, results_[r]) << "rank " << r;
      EXPECT_TRUE(ddaIpcOff(comms_[r].get())) << "rank " << r;
    }
    EXPECT_TRUE(live_.empty()) << live_.size() << " device allocations leaked";
    EXPECT_EQ(0, openMappings_) << "IPC mappings left open";
  }

  char archName_[8] = "gfx942";
  ncclPeerInfo peerInfo_[kRanks] = {};
  std::unique_ptr<ncclComm> comms_[kRanks];
  ncclResult_t results_[kRanks] = {};
  SimBootstrap sim_;
  Fault fault_ = Fault::kNone;
  std::mutex mu_;
  std::set<void*> live_;
  int openMappings_ = 0;
  ScopedHook<ncclResult_t(void*, void*, int)> bootstrapHook_{
    g_bootstrapAllGather, [](void* state, void* allData, int size) {
      auto* h = static_cast<SimBootstrap::Handle*>(state);
      return h->sim->allGather(h->rank, allData, size);
    }};
  ScopedHook<hipError_t(int*, int, int)> p2pHook_{g_hipDeviceCanAccessPeer, [this](int* canAccess, int, int) {
    *canAccess = faultHere(Fault::kNoP2p) ? 0 : 1;
    return hipSuccess;
  }};
  ScopedHook<hipError_t(void**, size_t, unsigned)> extMallocHook_{
    g_hipExtMallocWithFlags, [this](void** ptr, size_t size, unsigned) {
      const bool scratch = size == DDA_IPC_BUFFER_SIZE;
      if ((scratch && faultHere(Fault::kScratchAlloc)) || (!scratch && faultHere(Fault::kMailboxAlloc))) {
        *ptr = nullptr;
        return hipErrorOutOfMemory;
      }
      return deviceAlloc(ptr, scratch ? 64 : size);
    }};
  ScopedHook<hipError_t(void**, size_t)> mallocHook_{g_hipMalloc, [this](void** ptr, size_t size) {
    if (faultHere(Fault::kPeerTableAlloc)) {
      *ptr = nullptr;
      return hipErrorOutOfMemory;
    }
    return deviceAlloc(ptr, size);
  }};
  ScopedHook<hipError_t(void*)> freeHook_{g_hipFree, [this](void* ptr) {
    if (ptr != nullptr) {
      std::lock_guard<std::mutex> lk(mu_);
      EXPECT_EQ(1u, live_.erase(ptr));
      std::free(ptr);
    }
    return hipSuccess;
  }};
  ScopedHook<hipError_t(void*, int, size_t)> memsetHook_{g_hipMemset, [](void*, int, size_t) { return hipSuccess; }};
  ScopedHook<hipError_t(void*, const void*, size_t, hipMemcpyKind)> memcpyHook_{
    g_hipMemcpy, [](void* dst, const void* src, size_t n, hipMemcpyKind) {
      std::memcpy(dst, src, n);
      return hipSuccess;
    }};
  ScopedHook<hipError_t(hipIpcMemHandle_t*, void*)> exportHook_{
    g_hipIpcGetMemHandle, [this](hipIpcMemHandle_t* handle, void* ptr) {
      const int nth = t_exports++;
      if ((nth == 0 && faultHere(Fault::kScratchExport)) || (nth == 1 && faultHere(Fault::kMailboxExport))) {
        return hipErrorInvalidValue;
      }
      std::memset(handle, 0, sizeof(*handle));
      std::memcpy(handle, &ptr, sizeof(ptr));
      return hipSuccess;
    }};
  ScopedHook<hipError_t(void**, hipIpcMemHandle_t, unsigned)> openHook_{
    g_hipIpcOpenMemHandle, [this](void** ptr, hipIpcMemHandle_t handle, unsigned) {
      const int nth = t_opens++;
      if ((nth == kFailingOpen && faultHere(Fault::kScratchOpen)) ||
          (nth == kRanks - 1 + kFailingOpen && faultHere(Fault::kMailboxOpen))) {
        return hipErrorInvalidValue;
      }
      std::memcpy(ptr, &handle, sizeof(*ptr));
      std::lock_guard<std::mutex> lk(mu_);
      ++openMappings_;
      return hipSuccess;
    }};
  ScopedHook<hipError_t(void*)> closeHook_{g_hipIpcCloseMemHandle, [this](void*) {
    std::lock_guard<std::mutex> lk(mu_);
    --openMappings_;
    return hipSuccess;
  }};
};

TEST_F(DdaIpcInitMicrotest, AllRanksHealthy_EnablesDdaIpcEverywhere) {
  runInit(Fault::kNone);
  EXPECT_EQ(0, sim_.timeouts);
  EXPECT_EQ(0, sim_.desyncs);
  for (int r = 0; r < kRanks; ++r) {
    ASSERT_EQ(ncclSuccess, results_[r]) << "rank " << r;
    ASSERT_TRUE(ddaIpcOn(comms_[r].get())) << "rank " << r;
    for (int p = 0; p < kRanks; ++p) {
      EXPECT_EQ(comms_[p]->ddaScratch, comms_[r]->ddaPeerPtrsHost[p]) << "rank " << r << " peer " << p;
    }
  }
  EXPECT_EQ(2 * kRanks * (kRanks - 1), openMappings_);
  for (auto& c : comms_) {
    EXPECT_EQ(ncclSuccess, ncclDdaIpcCommFini(c.get()));
  }
  EXPECT_TRUE(live_.empty()) << live_.size() << " device allocations leaked";
  EXPECT_EQ(0, openMappings_) << "IPC mappings left open";
}

TEST_F(DdaIpcInitMicrotest, OneRankWithoutP2p_DisablesEverywhereWithoutHang) {
  expectDisabledEverywhereWithoutHang(Fault::kNoP2p);
}

TEST_F(DdaIpcInitMicrotest, OneRankScratchAllocFails_DisablesEverywhereWithoutHang) {
  expectDisabledEverywhereWithoutHang(Fault::kScratchAlloc);
}

TEST_F(DdaIpcInitMicrotest, OneRankScratchExportFails_DisablesEverywhereWithoutHang) {
  expectDisabledEverywhereWithoutHang(Fault::kScratchExport);
}

TEST_F(DdaIpcInitMicrotest, OneRankScratchOpenFails_DisablesEverywhereWithoutHang) {
  expectDisabledEverywhereWithoutHang(Fault::kScratchOpen);
}

TEST_F(DdaIpcInitMicrotest, OneRankPeerTableAllocFails_DisablesEverywhereWithoutHang) {
  expectDisabledEverywhereWithoutHang(Fault::kPeerTableAlloc);
}

TEST_F(DdaIpcInitMicrotest, OneRankMailboxAllocFails_DisablesEverywhereWithoutHang) {
  expectDisabledEverywhereWithoutHang(Fault::kMailboxAlloc);
}

TEST_F(DdaIpcInitMicrotest, OneRankMailboxExportFails_DisablesEverywhereWithoutHang) {
  expectDisabledEverywhereWithoutHang(Fault::kMailboxExport);
}

TEST_F(DdaIpcInitMicrotest, OneRankMailboxOpenFails_DisablesEverywhereWithoutHang) {
  expectDisabledEverywhereWithoutHang(Fault::kMailboxOpen);
}

} // namespace
