/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/mnnvl.cc's ncclMnnvlCheck: the peer fabric-state gates, clique and
// NVL-domain discovery and, when HIP_FABRIC_API is defined, the FABRIC export probe.

#include <gtest/gtest.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "comm.h"
#include "debug.h"
#include "ScopedHook.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"

namespace {
int g_callocFailAt = 0;
int g_callocCalls = 0;

template <typename T>
ncclResult_t MnnvlCalloc(T** ptr, size_t nelem) {
  if (++g_callocCalls == g_callocFailAt) return ncclSystemError;
  return ncclCallocDebug(ptr, nelem, __FILE__, __LINE__, __func__, true);
}
}  // namespace

#undef ncclCalloc
#define ncclCalloc(...) MnnvlCalloc(__VA_ARGS__)
#include MNNVL_CC_PATH
#undef ncclCalloc

struct allocationTracker allocTracker[MAX_ALLOC_TRACK_NGPU] = {};

namespace {

constexpr uint32_t kCliqueA = 0xa;
constexpr uint32_t kCliqueB = 0xb;

class MnnvlMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetHipFakes();
    ResetNcclFakes();
    g_cuMemEnable = [] { return 1; };
    g_callocFailAt = 0;
    g_callocCalls = 0;
    // With HIP_FABRIC_API the check ends in a FABRIC alloc/export/free probe: let it succeed.
    InstallHipVmmEmulator();
    g_hipGetLastError = [this] {
      lastErrorClears_++;
      return hipSuccess;
    };
    g_hipMemCreate = [this](hipMemGenericAllocationHandle_t* h, size_t, const hipMemAllocationProp* prop,
                            unsigned long long) {
      handleTypes_.push_back(prop->requestedHandleType);
      *h = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
      return createResult_;
    };
    g_hipMemAddressReserve = [this](void** p, size_t size, size_t, void*, unsigned long long) {
      *p = buffer_;
      reserved_ = size;
      return hipSuccess;
    };
    g_hipMemAddressFree = [this](void* p, size_t size) {
      EXPECT_EQ(p, (void*)buffer_);
      addressFrees_.push_back(size);
      return hipSuccess;
    };
    g_hipMemGetAddressRange = [this](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
      *base = ptr;
      *size = reserved_;
      return hipSuccess;
    };
    g_hipMemsetAsync = [](void*, int, size_t, hipStream_t) { return hipSuccess; };
    g_hipMemExportToShareableHandle = [this](void*, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType type,
                                             unsigned long long) {
      EXPECT_EQ(type, kFabricHandle);
      exports_++;
      return exportResult_;
    };
    comm_ = std::make_unique<ncclComm>();
  }

  void TearDown() override {
    free(comm_->clique.ranks);
    ResetNcclFakes();
    ResetHipFakes();
  }

  // Every rank fabric-capable, ACTIVE, in cluster 1 and clique A.
  void Build(int rank, int nRanks) {
    free(comm_->clique.ranks);
    comm_->clique = {};
    peers_.assign(nRanks, ncclPeerInfo{});
    for (ncclPeerInfo& p : peers_) {
      p.fabricHandleSupport = 1;
      p.fabricInfo.state = AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE;
      p.fabricInfo.cliqueId = kCliqueA;
      SetCluster(p, 1);
    }
    comm_->rank = rank;
    comm_->nRanks = nRanks;
    comm_->peerInfo = peers_.data();
  }

  static void SetCluster(ncclPeerInfo& p, uint8_t id, int byte = 15) {
    std::memset(p.fabricInfo.clusterUuid, 0, sizeof(p.fabricInfo.clusterUuid));
    p.fabricInfo.clusterUuid[byte] = id;
  }

  std::vector<int> CliqueRanks() const {
    return std::vector<int>(comm_->clique.ranks, comm_->clique.ranks + comm_->clique.size);
  }

  std::unique_ptr<ncclComm> comm_;
  std::vector<ncclPeerInfo> peers_;
  int lastErrorClears_ = 0;
  char buffer_[64];
  std::vector<hipMemAllocationHandleType> handleTypes_;
  size_t reserved_ = 0;
  std::vector<size_t> addressFrees_;
  int exports_ = 0;
  hipError_t createResult_ = hipSuccess;
  hipError_t exportResult_ = hipSuccess;
};

TEST_F(MnnvlMicrotest, Check_NeedsCuMem) {
  Build(0, 2);
  ScopedHook cuMem(g_cuMemEnable, [] { return 0; });
  ASSERT_EQ(ncclMnnvlCheck(comm_.get()), ncclSuccess);
  EXPECT_EQ(comm_->clique.ranks, nullptr);
  EXPECT_EQ(lastErrorClears_, 0);
}

TEST_F(MnnvlMicrotest, Check_NeedsFabricHandlesOnEveryRank) {
  Build(0, 3);
  peers_[2].fabricHandleSupport = 0;
  ASSERT_EQ(ncclMnnvlCheck(comm_.get()), ncclSuccess);
  EXPECT_EQ(comm_->clique.ranks, nullptr);
  EXPECT_EQ(lastErrorClears_, 0);
}

// Only ACTIVE and READY count; a provisioned (CONFIGURED) vPoD cannot exchange handles yet.
TEST_F(MnnvlMicrotest, Check_NeedsEveryPeerActiveOrReady) {
  for (auto state : {AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_UNCONFIGURED, AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_CONFIGURED,
                     AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ERROR, AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_UNKNOWN}) {
    SCOPED_TRACE(state);
    Build(0, 3);
    peers_[1].fabricInfo.state = state;
    lastErrorClears_ = 0;
    ASSERT_EQ(ncclMnnvlCheck(comm_.get()), ncclSuccess);
    EXPECT_EQ(comm_->clique.ranks, nullptr);
    EXPECT_EQ(lastErrorClears_, 1);
  }
  Build(0, 3);
  peers_[1].fabricInfo.state = AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_READY;
  ASSERT_EQ(ncclMnnvlCheck(comm_.get()), ncclSuccess);
  EXPECT_EQ(comm_->clique.size, 3);
}

TEST_F(MnnvlMicrotest, Check_FindsTheCliqueAndTheNvlDomain) {
  Build(2, 5);
  peers_[0].fabricInfo.cliqueId = kCliqueB;
  peers_[3].fabricInfo.cliqueId = kCliqueB;
  SetCluster(peers_[4], 2);
  comm_->nvlDomainSize = 99;
  ASSERT_EQ(ncclMnnvlCheck(comm_.get()), ncclSuccess);
  EXPECT_EQ(comm_->clique.id, (int)kCliqueA);
  EXPECT_EQ(CliqueRanks(), (std::vector<int>{1, 2}));
  EXPECT_EQ(comm_->cliqueRank, 1);
  EXPECT_EQ(comm_->nvlDomainSize, 4);
#ifdef HIP_FABRIC_API
  EXPECT_EQ(comm_->MNNVL, 1);
#else
  EXPECT_EQ(comm_->MNNVL, 0);
  EXPECT_EQ(ncclCuMemHandleType, hipMemHandleTypePosixFileDescriptor);
  EXPECT_TRUE(handleTypes_.empty());
#endif
}

// A rank with an all-zero cluster UUID has no fabric info: discovery stops there, part-way.
TEST_F(MnnvlMicrotest, Check_ZeroClusterUuidStopsDiscovery) {
  Build(0, 4);
  std::memset(peers_[2].fabricInfo.clusterUuid, 0, sizeof(peers_[2].fabricInfo.clusterUuid));
  ASSERT_EQ(ncclMnnvlCheck(comm_.get()), ncclSuccess);
  EXPECT_EQ(CliqueRanks(), (std::vector<int>{0, 1}));
  EXPECT_EQ(comm_->nvlDomainSize, 2);
  EXPECT_EQ(comm_->MNNVL, 0);
}

// Either 8-byte half of the UUID being non-zero is enough.
TEST_F(MnnvlMicrotest, Check_UuidNeedsOnlyOneNonZeroHalf) {
  Build(0, 2);
  for (ncclPeerInfo& p : peers_) SetCluster(p, 7, 0);
  ASSERT_EQ(ncclMnnvlCheck(comm_.get()), ncclSuccess);
  EXPECT_EQ(comm_->clique.size, 2);
  EXPECT_EQ(comm_->nvlDomainSize, 2);
}

TEST_F(MnnvlMicrotest, Check_ALoneRankIsNotMnnvl) {
  Build(0, 1);
  ASSERT_EQ(ncclMnnvlCheck(comm_.get()), ncclSuccess);
  EXPECT_EQ(comm_->clique.size, 1);
  EXPECT_EQ(comm_->MNNVL, 0);
  EXPECT_TRUE(handleTypes_.empty());
}

TEST_F(MnnvlMicrotest, Check_CliqueAllocationFailure) {
  Build(0, 2);
  g_callocFailAt = 1;
  EXPECT_EQ(ncclMnnvlCheck(comm_.get()), ncclSystemError);
  EXPECT_EQ(comm_->clique.ranks, nullptr);
}

#ifdef HIP_FABRIC_API

TEST_F(MnnvlMicrotest, Probe_EnablesMnnvlAndSwitchesToFabricHandles) {
  Build(0, 2);
  ASSERT_EQ(ncclMnnvlCheck(comm_.get()), ncclSuccess);
  EXPECT_EQ(comm_->MNNVL, 1);
  EXPECT_EQ(ncclCuMemHandleType, kFabricHandle);
  EXPECT_EQ(handleTypes_, std::vector<hipMemAllocationHandleType>{kFabricHandle});
  EXPECT_EQ(exports_, 1);
  EXPECT_EQ(addressFrees_, std::vector<size_t>{reserved_});
}

// Two ranks in different cliques of one domain still probe (an ncclCommSplit across cliques).
TEST_F(MnnvlMicrotest, Probe_RunsForASplitWithASingleRankClique) {
  Build(0, 2);
  peers_[1].fabricInfo.cliqueId = kCliqueB;
  ASSERT_EQ(ncclMnnvlCheck(comm_.get()), ncclSuccess);
  EXPECT_EQ(comm_->clique.size, 1);
  EXPECT_EQ(comm_->MNNVL, 1);
}

TEST_F(MnnvlMicrotest, Probe_AllocationFailureIsASystemError) {
  Build(0, 2);
  createResult_ = hipErrorNotSupported;
  EXPECT_EQ(ncclMnnvlCheck(comm_.get()), ncclSystemError);
  EXPECT_EQ(comm_->MNNVL, 0);
  EXPECT_EQ(ncclCuMemHandleType, hipMemHandleTypePosixFileDescriptor);
}

TEST_F(MnnvlMicrotest, Probe_ExportFailureFreesTheBufferAndFails) {
  Build(0, 2);
  exportResult_ = hipErrorNotSupported;
  EXPECT_EQ(ncclMnnvlCheck(comm_.get()), ncclSystemError);
  EXPECT_EQ(addressFrees_, std::vector<size_t>{reserved_});
  EXPECT_EQ(comm_->MNNVL, 0);
}

TEST_F(MnnvlMicrotest, Probe_FreeFailureIsReturned) {
  Build(0, 2);
  ScopedHook retain(g_hipMemRetainAllocationHandle,
                    [](hipMemGenericAllocationHandle_t*, void*) { return hipErrorInvalidValue; });
  EXPECT_EQ(ncclMnnvlCheck(comm_.get()), ncclUnhandledCudaError);
  EXPECT_EQ(comm_->MNNVL, 0);
  exportResult_ = hipErrorNotSupported;
  Build(0, 2);
  EXPECT_EQ(ncclMnnvlCheck(comm_.get()), ncclUnhandledCudaError);
}

#endif  // HIP_FABRIC_API

}  // namespace
