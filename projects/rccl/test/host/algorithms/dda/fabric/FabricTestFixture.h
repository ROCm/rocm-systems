/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Shared fixture and helpers for the src/algorithms/dda/fabric/ tests.

#ifndef RCCL_TEST_HOST_ALGORITHMS_DDA_FABRIC_FABRICTESTFIXTURE_H_
#define RCCL_TEST_HOST_ALGORITHMS_DDA_FABRIC_FABRICTESTFIXTURE_H_

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "HipVmmLedger.h"
#include "ScopedHook.h"
#include "fakes/bootstrap_stubs.h"
#include "fakes/env_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"

#include "alloc.h"
#include "mem_manager.h"

namespace dda_fabric_test {

constexpr int kNRanks = 4;
constexpr int kRank = 1;
constexpr size_t kPage = 4096;  // emulator allocation granularity
inline void* const kBootstrap = reinterpret_cast<void*>(0xB007);
// Not the POSIX-FD default, so a hard-coded handle type fails.
constexpr hipMemAllocationHandleType kHandleType = hipMemHandleTypeWin32Kmt;

// Every rank but `self`.
inline std::vector<int> PeerRanks(int nRanks = kNRanks, int self = kRank) {
  std::vector<int> peers;
  for (int r = 0; r < nRanks; ++r) {
    if (r != self) peers.push_back(r);
  }
  return peers;
}

// gtest name generator for param structs with a `name` field.
template <typename T>
std::string ParamName(const ::testing::TestParamInfo<T>& info) {
  return info.param.name;
}

// Records which manager each allocation is tracked in and untracked from.
class MemManagerSpy {
 public:
  MemManagerSpy()
      : untrackLive_(g_memUntrackDynamic),
        track_(g_memTrack,
               [this](ncclMemManager* m, void* ptr, size_t, hipMemGenericAllocationHandle_t,
                      hipMemAllocationHandleType type, ncclMemType_t) {
                 tracked.push_back({m, ptr, type});
                 return ncclSuccess;
               }),
        untrack_(g_memUntrackDynamic, [this](ncclMemManager* m, void* ptr, ncclMemUntrackInfo* info) {
          untracked.push_back({m, ptr});
          return untrackLive_(m, ptr, info);
        }) {}

  bool Tracked(ncclMemManager* m, void* ptr, hipMemAllocationHandleType type) const {
    return std::any_of(tracked.begin(), tracked.end(),
                       [&](const Track& t) { return t.manager == m && t.ptr == ptr && t.type == type; });
  }
  bool Untracked(ncclMemManager* m, void* ptr) const {
    return std::find(untracked.begin(), untracked.end(), std::make_pair(m, ptr)) != untracked.end();
  }

  struct Track {
    ncclMemManager* manager;
    void* ptr;
    hipMemAllocationHandleType type;
  };
  std::vector<Track> tracked;
  std::vector<std::pair<ncclMemManager*, void*>> untracked;

 private:
  std::function<ncclResult_t(ncclMemManager*, void*, ncclMemUntrackInfo*)> untrackLive_;
  ScopedHook<ncclResult_t(ncclMemManager*, void*, size_t, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType,
                          ncclMemType_t)>
      track_;
  ScopedHook<ncclResult_t(ncclMemManager*, void*, ncclMemUntrackInfo*)> untrack_;
};

class FabricLedgerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    savedHandleType_ = ncclCuMemHandleType;
    ncclCuMemHandleType = kHandleType;
    SetMicroEnvAbsent("NCCL_CUMEM_SKIP_FREE");
    ledger_.Install();
    g_cuMemEnable = [] { return 1; };
    ASSERT_FALSE(rcclSkipCuMemFree());  // alloc.h caches this once per process
  }

  // Call after destroying the unit under test.
  void TearDown() override {
    EXPECT_TRUE(LedgerClean());
    ncclCuMemHandleType = savedHandleType_;
    ResetBootstrapStubs();
    ResetHipFakes();
    ResetNcclFakes();
    ResetEnvFakes();
  }

  ::testing::AssertionResult LedgerClean() const {
    if (ledger_.Clean()) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << ledger_.Summary();
  }

  // Every peer publishes what this rank published. Records each call's
  // bootstrap and how many memsets ran before the first one.
  void MirrorAllGather(const int* nRanks, const int* rank) {
    g_bootstrapAllGather = [this, nRanks, rank](void* state, void* allData, int size) {
      if (gatherStates_.empty()) memsetsBeforeFirstGather_ = ledger_.memsets.size();
      gatherStates_.push_back(state);
      auto* slots = static_cast<char*>(allData);
      for (int r : PeerRanks(*nRanks, *rank)) std::memcpy(slots + r * size, slots + *rank * size, size);
      return ncclSuccess;
    };
  }

  HipVmmLedger ledger_;
  std::vector<void*> gatherStates_;
  size_t memsetsBeforeFirstGather_ = 0;

 private:
  hipMemAllocationHandleType savedHandleType_ = hipMemHandleTypeNone;
};

}  // namespace dda_fabric_test

#endif  // RCCL_TEST_HOST_ALGORITHMS_DDA_FABRIC_FABRICTESTFIXTURE_H_
