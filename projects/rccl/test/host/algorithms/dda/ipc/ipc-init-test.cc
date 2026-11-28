/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/algorithms/dda/ipc/ipc_init.cu.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "ScopedHook.h"
#include "fakes/bootstrap_stubs.h"
#include "fakes/hip_fakes.h"

#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/ipc/ipc_gpu_barrier.h"
#include "algorithms/dda/ipc/ipc_init.h"
#include "algorithms/dda/ipc/ipc_mem_handler.h"
#include "comm.h"

#include IPC_INIT_CC_PATH

namespace {
bool g_ipcBarrierFails = false;
}  // namespace

// The real handler and barrier open HIP IPC handles; these hand back this rank's own pointers.
ncclIpcMemHandler::ncclIpcMemHandler(void* bootstrap, int rank, int nranks)
    : bootstrap_(bootstrap), rank_(rank), nranks_(nranks), memPtrs_(nranks, nullptr), exchanged_(false) {}
ncclIpcMemHandler::~ncclIpcMemHandler() = default;
ncclResult_t ncclIpcMemHandler::closePeerMappings() {
  exchanged_ = false;
  return ncclSuccess;
}
ncclResult_t ncclIpcMemHandler::addSelfDeviceMemPtr(void* deviceMemPtr) {
  memPtrs_[rank_] = deviceMemPtr;
  return ncclSuccess;
}
ncclResult_t ncclIpcMemHandler::exchangeMemPtrs() {
  exchanged_ = true;
  return ncclSuccess;
}
ncclResult_t ncclIpcMemHandler::getPeerDeviceMemPtr(int, void** outPeerPtr) const {
  *outPeerPtr = memPtrs_[rank_];
  return ncclSuccess;
}

namespace dda::common {
std::pair<std::unique_ptr<IpcGpuBarrierResources>, IpcGpuBarrier> IpcGpuBarrier::mallocAndInit(int, int, int, void*,
                                                                                              DdaKernelParams) {
  if (g_ipcBarrierFails) return {nullptr, IpcGpuBarrier{}};
  return {std::make_unique<IpcGpuBarrierResources>(), IpcGpuBarrier{}};
}
}  // namespace dda::common

namespace {

constexpr int kNRanks = nccl_dda_detail::kDdaNranks;
constexpr int kRank = 1;
constexpr int kCuCount = 40;
void* const kBootstrap = reinterpret_cast<void*>(0xB007);
char kGfx950[] = "gfx950";
hipStream_t const kZeroStream = reinterpret_cast<hipStream_t>(0x5);

const size_t kEpochBytes =
    nccl_dda_detail::ddaLLEpochCount(kNRanks, nccl_dda_detail::ddaFabricMaxNBlocksForScratch(kCuCount, nullptr)) *
    sizeof(uint32_t);

class IpcInitMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_ipcBarrierFails = false;
    g_hipDeviceCanAccessPeer = [](int* canAccess, int, int) {
      *canAccess = 1;
      return hipSuccess;
    };
    g_hipExtMallocWithFlags = [this](void** ptr, size_t size, unsigned) { return Alloc(ptr, size); };
    g_hipMalloc = [this](void** ptr, size_t size) {
      if (size == kEpochBytes && failEpochAlloc_) {
        *ptr = nullptr;
        return hipErrorOutOfMemory;
      }
      hipError_t err = Alloc(ptr, size);
      if (err == hipSuccess && size == kEpochBytes) epochAllocs_.push_back(*ptr);
      return err;
    };
    g_hipFree = [this](void* ptr) {
      if (ptr != nullptr) {
        EXPECT_EQ(1u, live_.erase(ptr)) << "free of an untracked pointer";
        std::free(ptr);
      }
      return hipSuccess;
    };
    g_hipMemcpy = [](void* dst, const void* src, size_t n, hipMemcpyKind) {
      std::memcpy(dst, src, n);
      return hipSuccess;
    };
    g_hipGetLastError = [] { return hipSuccess; };
    g_hipStreamCreateWithFlags = [this](hipStream_t* stream, unsigned flags) {
      streamFlags_.push_back(flags);
      if (streamCreateResult_ != hipSuccess) {
        return streamCreateResult_;
      }
      *stream = kZeroStream;
      return hipSuccess;
    };
    g_hipStreamDestroy = [this](hipStream_t stream) {
      EXPECT_EQ(kZeroStream, stream);
      ++streamDestroys_;
      return hipSuccess;
    };
    g_hipMemsetAsync = [this](void* dst, int value, size_t n, hipStream_t stream) {
      steps_.push_back({Step::Memset, dst, n, stream, value});
      if ((failEpochFill_ && n == kEpochBytes) || (failScratchFill_ && n == DDA_IPC_BUFFER_SIZE)) {
        return hipErrorInvalidValue;
      }
      std::memset(dst, value, n);
      return hipSuccess;
    };
    g_hipStreamSynchronize = [this](hipStream_t stream) {
      steps_.push_back({Step::Sync, nullptr, 0, stream, 0});
      return syncResult_;
    };
    // Every peer publishes this rank's flag unless peerReady_ overrides one.
    g_bootstrapAllGather = [this](void* state, void* allData, int size) {
      EXPECT_EQ(kBootstrap, state);
      EXPECT_EQ(static_cast<int>(sizeof(int)), size);
      auto* slots = static_cast<int*>(allData);
      published_.push_back(slots[kRank]);
      steps_.push_back({Step::Gather, nullptr, 0, nullptr, 0});
      for (int r = 0; r < kNRanks; ++r) {
        if (r != kRank) slots[r] = slots[kRank];
      }
      if (peerReady_ >= 0) slots[kRank + 1] = peerReady_;
      return gatherResult_;
    };

    peerInfo_.resize(kNRanks);
    for (int r = 0; r < kNRanks; ++r) peerInfo_[r].cudaDev = r;
    comm_ = std::make_unique<ncclComm>();
    comm_->nRanks = kNRanks;
    comm_->rank = kRank;
    comm_->nNodes = 1;
    comm_->bootstrap = kBootstrap;
    comm_->archName = kGfx950;
    comm_->cuCount = kCuCount;
    comm_->peerInfo = peerInfo_.data();
  }

  void TearDown() override {
    ncclDdaIpcCommFini(comm_.get());
    // A stream that init created is destroyed exactly once, on every path.
    const int created = !streamFlags_.empty() && streamCreateResult_ == hipSuccess ? 1 : 0;
    EXPECT_EQ(created, streamDestroys_);
    EXPECT_TRUE(live_.empty()) << live_.size() << " device allocation(s) leaked";
    g_ipcBarrierFails = false;
    ResetBootstrapStubs();
    ResetHipFakes();
  }

  hipError_t Alloc(void** ptr, size_t size) {
    *ptr = std::malloc(size);
    if (*ptr == nullptr) return hipErrorOutOfMemory;
    // Junk, so memory that init does not zero-fill cannot read back as 0.
    std::memset(*ptr, 0xA5, size);
    live_.insert(*ptr);
    return hipSuccess;
  }

  void ExpectDdaWithoutEpochCells() const {
    EXPECT_NE(nullptr, comm_->ddaIpcMemHandler);
    EXPECT_NE(nullptr, comm_->ddaScratch);
    EXPECT_EQ(nullptr, comm_->ddaLLEpochDev);
    EXPECT_EQ(0, comm_->ddaLLEpochLen);
  }

  void ExpectNoDda() const {
    EXPECT_EQ(nullptr, comm_->ddaIpcMemHandler);
    EXPECT_EQ(nullptr, comm_->ddaScratch);
    EXPECT_EQ(nullptr, comm_->ddaPeerPtrsDev);
    EXPECT_EQ(nullptr, comm_->ddaPeerPtrsHost);
    EXPECT_EQ(nullptr, comm_->ddaIpcBarrierState);
    EXPECT_EQ(nullptr, comm_->ddaLLEpochDev);
    EXPECT_EQ(0, comm_->ddaLLEpochLen);
  }

  enum class Step { Memset, Sync, Gather };
  struct InitStep {
    Step step;
    void* dst;
    size_t bytes;
    hipStream_t stream;
    int value;  // fill byte, for Memset
  };

  bool failEpochAlloc_ = false;
  bool failEpochFill_ = false;
  bool failScratchFill_ = false;
  hipError_t streamCreateResult_ = hipSuccess;
  std::vector<unsigned> streamFlags_;
  int streamDestroys_ = 0;
  hipError_t syncResult_ = hipSuccess;
  std::vector<InitStep> steps_;
  int peerReady_ = -1;  // < 0: peers mirror this rank
  ncclResult_t gatherResult_ = ncclSuccess;
  std::vector<int> published_;
  std::vector<void*> epochAllocs_;
  std::set<void*> live_;
  std::vector<ncclPeerInfo> peerInfo_;
  std::unique_ptr<ncclComm> comm_;
};

TEST_F(IpcInitMicrotest, CommInit_AllRanksReady_KeepsEpochCells) {
  EXPECT_EQ(ncclSuccess, ncclDdaIpcCommInit(comm_.get()));

  EXPECT_EQ(std::vector<int>{2}, published_);
  EXPECT_NE(nullptr, comm_->ddaIpcMemHandler);
  ASSERT_EQ(1u, epochAllocs_.size());
  EXPECT_EQ(epochAllocs_[0], comm_->ddaLLEpochDev);
  EXPECT_EQ(kEpochBytes / sizeof(uint32_t), static_cast<size_t>(comm_->ddaLLEpochLen));
  // Zeroed cells make the first LL flag 1.
  for (int i = 0; i < comm_->ddaLLEpochLen; ++i) {
    EXPECT_EQ(0u, comm_->ddaLLEpochDev[i]) << "cell " << i;
  }
}

// The Simple paths see the scratch below the LL region; DDA Broadcast LL banks sit in the region.
TEST_F(IpcInitMicrotest, CommInit_KeepsTheLLRegionOutOfSimpleScratch) {
  EXPECT_EQ(ncclSuccess, ncclDdaIpcCommInit(comm_.get()));

  ASSERT_NE(nullptr, comm_->ddaScratch);
  EXPECT_EQ(static_cast<size_t>(DDA_IPC_BUFFER_SIZE) - dda::common::kDdaIpcLLRegionBytes, comm_->ddaScratchBytes);
}

// Both fills go on init's own stream, and a sync of it lands before the ready gather, so no first kernel races them.
TEST_F(IpcInitMicrotest, CommInit_ZeroFillsOnItsOwnStreamAndSyncsBeforeTheReadyGather) {
  EXPECT_EQ(ncclSuccess, ncclDdaIpcCommInit(comm_.get()));

  ASSERT_EQ(4u, steps_.size());
  EXPECT_EQ(Step::Memset, steps_[0].step);
  EXPECT_EQ(comm_->ddaScratch, steps_[0].dst);
  EXPECT_EQ(static_cast<size_t>(DDA_IPC_BUFFER_SIZE), steps_[0].bytes);
  EXPECT_EQ(Step::Memset, steps_[1].step);
  EXPECT_EQ(static_cast<void*>(comm_->ddaLLEpochDev), steps_[1].dst);
  EXPECT_EQ(kEpochBytes, steps_[1].bytes);
  EXPECT_EQ(Step::Sync, steps_[2].step);
  EXPECT_EQ(Step::Gather, steps_[3].step);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(kZeroStream, steps_[i].stream) << "step " << i;
  }
  EXPECT_EQ(0, steps_[0].value);
  EXPECT_EQ(0, steps_[1].value);
  // Non-blocking, so the fills do not wait on, or sync, the app's work on the null stream.
  EXPECT_EQ(std::vector<unsigned>{hipStreamNonBlocking}, streamFlags_);
}

TEST_F(IpcInitMicrotest, CommInit_ScratchFillFails_PublishesItAndSkipsDdaIpc) {
  failScratchFill_ = true;

  EXPECT_EQ(ncclSuccess, ncclDdaIpcCommInit(comm_.get()));

  EXPECT_EQ(std::vector<int>{0}, published_);
  EXPECT_TRUE(epochAllocs_.empty());
  ExpectNoDda();
}

// No stream, so nothing is filled, synced or destroyed; the rank still joins the gather.
TEST_F(IpcInitMicrotest, CommInit_ZeroStreamCreateFails_PublishesItAndSkipsDdaIpc) {
  streamCreateResult_ = hipErrorOutOfMemory;

  EXPECT_EQ(ncclSuccess, ncclDdaIpcCommInit(comm_.get()));

  EXPECT_EQ(std::vector<int>{0}, published_);
  ASSERT_EQ(1u, steps_.size());
  EXPECT_EQ(Step::Gather, steps_[0].step);
  ExpectNoDda();
}

TEST_F(IpcInitMicrotest, CommInit_ZeroFillSyncFails_PublishesItAndSkipsDdaIpc) {
  syncResult_ = hipErrorInvalidValue;

  EXPECT_EQ(ncclSuccess, ncclDdaIpcCommInit(comm_.get()));

  EXPECT_EQ(std::vector<int>{0}, published_);
  ExpectNoDda();
}

TEST_F(IpcInitMicrotest, CommInit_EpochFillFails_PublishesItAndDropsOnlyTheCells) {
  failEpochFill_ = true;

  EXPECT_EQ(ncclSuccess, ncclDdaIpcCommInit(comm_.get()));

  EXPECT_EQ(std::vector<int>{1}, published_);
  ExpectDdaWithoutEpochCells();
}

TEST_F(IpcInitMicrotest, CommInit_LocalEpochAllocFails_PublishesItAndDropsOnlyTheCells) {
  failEpochAlloc_ = true;

  EXPECT_EQ(ncclSuccess, ncclDdaIpcCommInit(comm_.get()));

  EXPECT_EQ(std::vector<int>{1}, published_);
  ExpectDdaWithoutEpochCells();
}

// The rank-uniform part: this rank's cells were allocated, but a peer's were not.
TEST_F(IpcInitMicrotest, CommInit_PeerEpochAllocFails_FreesLocalCells) {
  peerReady_ = 1;

  EXPECT_EQ(ncclSuccess, ncclDdaIpcCommInit(comm_.get()));

  EXPECT_EQ(std::vector<int>{2}, published_);
  ASSERT_EQ(1u, epochAllocs_.size());
  EXPECT_EQ(0u, live_.count(epochAllocs_[0]));
  ExpectDdaWithoutEpochCells();
}

TEST_F(IpcInitMicrotest, CommInit_PeerBarrierFails_SkipsDdaIpc) {
  peerReady_ = 0;

  EXPECT_EQ(ncclSuccess, ncclDdaIpcCommInit(comm_.get()));

  EXPECT_EQ(std::vector<int>{2}, published_);
  ExpectNoDda();
  EXPECT_TRUE(live_.empty());
}

TEST_F(IpcInitMicrotest, CommInit_LocalBarrierFails_StillJoinsTheReadyGather) {
  g_ipcBarrierFails = true;

  EXPECT_EQ(ncclSuccess, ncclDdaIpcCommInit(comm_.get()));

  EXPECT_EQ(std::vector<int>{0}, published_);
  EXPECT_TRUE(epochAllocs_.empty());
  ExpectNoDda();
}

TEST_F(IpcInitMicrotest, CommInit_ReadyGatherFails_SkipsDdaIpc) {
  gatherResult_ = ncclRemoteError;

  EXPECT_EQ(ncclSuccess, ncclDdaIpcCommInit(comm_.get()));

  ExpectNoDda();
  EXPECT_TRUE(live_.empty());
}

}  // namespace
