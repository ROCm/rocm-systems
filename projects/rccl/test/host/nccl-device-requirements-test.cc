/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the device-resource requirement builders in
// src/nccl_device/{cft_barrier,gin_barrier,gin_scratch,ll_a2a,lsa_barrier}.cc. Goldens are
// ported from test/DeviceResourceRequirementTests.cpp.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <vector>

// ncclDevResourceRequirements / ncclTeam live behind core_tmp.h, transitively included below.
#include "nccl_device/impl/ll_a2a__types.h"
#include "nccl_device/impl/lsa_barrier__types.h"
#include "nccl_device/impl/gin_barrier__types.h"
#include "nccl_device/impl/cft_barrier__types.h"
// gin_scratch__types.h expects the unqualified alignUp/imodFast32 that "core.h" gives production TUs.
#include "bitops.h"
// Outbox/inbox declarations still live at their pre-public-split location, not under nccl_device/impl/.
#include "device/symmetric/gin_scratch.h"

#include CFT_BARRIER_CC_PATH
#include GIN_BARRIER_CC_PATH
#include GIN_SCRATCH_CC_PATH
#include LL_A2A_CC_PATH
#include LSA_BARRIER_CC_PATH

namespace {

// Re-implemented via division, not the production bit-mask trick, so a bug there can't hide here.
constexpr size_t alignUpLocal(size_t x, size_t a) {
  return ((x + a - 1) / a) * a;
}

// Non-zero poison so a missing memset() would leave visible garbage instead of a false-zero read.
ncclDevResourceRequirements_t poisonedRequirement() {
  ncclDevResourceRequirements_t req;
  std::memset(&req, 0xAA, sizeof(req));
  return req;
}

// Same poison for handle out-params: an unpoisoned handle can pass by accident on a zeroed stack.
template <typename T>
T poisonedHandle() {
  T handle;
  std::memset(&handle, 0xAA, sizeof(handle));
  return handle;
}

// GIN-signal fields untouched by the buffer-only builders (LLA2A, outbox, LSA/CFT barrier).
void ExpectNoGinSignalFields(const ncclDevResourceRequirements_t& req) {
  EXPECT_EQ(req.next, nullptr);
  EXPECT_EQ(req.ginSignalCount, 0);
  EXPECT_EQ(req.ginCounterCount, 0);
  EXPECT_EQ(req.outGinSignalStart, nullptr);
  EXPECT_EQ(req.outGinCounterStart, nullptr);
}

// GIN-counter fields untouched by the signal-setting builders.
void ExpectNoGinCounterFields(const ncclDevResourceRequirements_t& req) {
  EXPECT_EQ(req.next, nullptr);
  EXPECT_EQ(req.ginCounterCount, 0);
  EXPECT_EQ(req.outGinCounterStart, nullptr);
}

// Only the forked child is affected: skip its core dump.
void DisableCoreDumpsForExpectedCrash() {
#if defined(__linux__)
  if (prctl(PR_SET_DUMPABLE, 0) != 0) {
    std::perror("prctl(PR_SET_DUMPABLE) failed");
    _exit(1);
  }
#endif
  const struct rlimit noCore = {0, 0};
  if (setrlimit(RLIMIT_CORE, &noCore) != 0) {
    std::perror("setrlimit(RLIMIT_CORE) failed");
    _exit(1);
  }
}

}  // namespace

// ncclLLA2ACalcSlots / ncclLLA2ACreateRequirement (src/nccl_device/ll_a2a.cc)

// divUp rounding: maxEltSize of 1, 7, 8 all round up to one 8-byte slot; 9 and 16 need two.
TEST(NcclDeviceRequirementsMicrotest, LLA2ACalcSlots_DivUpRounding) {
  struct Case {
    const char* name;
    int maxEltSize;
    int expectedSlots;
  };
  const std::vector<Case> cases = {
    {"eltSize_1", 1, 1}, {"eltSize_7", 7, 1}, {"eltSize_8", 8, 1}, {"eltSize_9", 9, 2}, {"eltSize_16", 16, 2},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    EXPECT_EQ(ncclLLA2ACalcSlots(/*maxElts=*/1, c.maxEltSize), c.expectedSlots);
  }
}

// Against literals, not ncclLLA2ACalcSlots's own output.
TEST(NcclDeviceRequirementsMicrotest, LLA2ACalcSlots_LinearInMaxElts) {
  EXPECT_EQ(ncclLLA2ACalcSlots(/*maxElts=*/5, /*maxEltSize=*/8), 5);
  // 2 * divUp(4, 8) == 2; hoisting the divUp, divUp(2 * 4, 8), would give 1.
  EXPECT_EQ(ncclLLA2ACalcSlots(/*maxElts=*/2, /*maxEltSize=*/4), 2);
}

// maxElts = 0 always yields 0 slots, independent of maxEltSize.
TEST(NcclDeviceRequirementsMicrotest, LLA2ACalcSlots_ZeroMaxEltsIsZero) {
  EXPECT_EQ(ncclLLA2ACalcSlots(/*maxElts=*/0, /*maxEltSize=*/8), 0);
}

// bufferSize == nBlocks * (1 + 2*nSlots) * 16; nSlots == 0 is the header-only case.
TEST(NcclDeviceRequirementsMicrotest, LLA2ACreateRequirement_BufferSizeFormula) {
  struct Case {
    const char* name;
    int nBlocks;
    int nSlots;
  };
  const std::vector<Case> cases = {
    {"header_only", 1, 0},
    {"small", 2, 3},
    {"larger", 4, 5},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    ncclLLA2AHandle_t handle = poisonedHandle<ncclLLA2AHandle_t>();
    ncclDevResourceRequirements_t req = poisonedRequirement();
    ASSERT_EQ(ncclLLA2ACreateRequirement(c.nBlocks, c.nSlots, &handle, &req), ncclSuccess);
    EXPECT_EQ(req.bufferSize, static_cast<size_t>(c.nBlocks) * (1 + 2 * c.nSlots) * 16);
    EXPECT_EQ(req.bufferAlign, 16u);
    EXPECT_EQ(req.outBufferHandle, &handle.bufHandle);
    EXPECT_EQ(handle.nSlots, static_cast<uint32_t>(c.nSlots));
    // Unset fields read back zero: the memset ran.
    ExpectNoGinSignalFields(req);
  }
}

// Realistic usage (mirrors src/sym_kernels.cc); ncclLLA2ACalcSlots(4, 16) == 8.
TEST(NcclDeviceRequirementsMicrotest, LLA2ACreateRequirement_RoundTripWithCalcSlots) {
  const int nSlots = ncclLLA2ACalcSlots(/*maxElts=*/4, /*maxEltSize=*/16);
  const int nBlocks = 3;
  ncclLLA2AHandle_t handle = poisonedHandle<ncclLLA2AHandle_t>();
  ncclDevResourceRequirements_t req = poisonedRequirement();
  ASSERT_EQ(ncclLLA2ACreateRequirement(nBlocks, nSlots, &handle, &req), ncclSuccess);
  EXPECT_EQ(req.bufferSize, size_t{816});  // nBlocks=3, nSlots=8: 3 * (1 + 2*8) * 16
  EXPECT_EQ(handle.nSlots, static_cast<uint32_t>(nSlots));
}

// ncclGinOutboxCreateRequirement / ncclGinInboxA2ACreateRequirement (src/nccl_device/gin_scratch.cc)

// The 128-byte floor: size_log2 at or below 7 clamps to 7, and outHandle reads back the clamped value.
TEST(NcclDeviceRequirementsMicrotest, GinOutboxCreateRequirement_FloorClampsSizeLog2AtOrBelow7) {
  const std::vector<int> atOrBelowFloor = {0, 5, 6, 7};
  for (int sizeLog2 : atOrBelowFloor) {
    ncclGinOutboxHandle handle = poisonedHandle<ncclGinOutboxHandle>();
    ncclDevResourceRequirements_t req = poisonedRequirement();
    ASSERT_EQ(ncclGinOutboxCreateRequirement(/*nBlocks=*/1, sizeLog2, &handle, &req), ncclSuccess)
      << "case: size_log2=" << sizeLog2;
    EXPECT_EQ(handle.size_log2, 7u) << "case: size_log2=" << sizeLog2;
  }
}

// Above the floor, size_log2 passes through unclamped.
TEST(NcclDeviceRequirementsMicrotest, GinOutboxCreateRequirement_SizeLog2AboveFloorIsUnclamped) {
  ncclGinOutboxHandle handle = poisonedHandle<ncclGinOutboxHandle>();
  ncclDevResourceRequirements_t req = poisonedRequirement();
  ASSERT_EQ(ncclGinOutboxCreateRequirement(/*nBlocks=*/1, /*size_log2=*/8, &handle, &req), ncclSuccess);
  EXPECT_EQ(handle.size_log2, 8u);
}

// bufferSize == nBlocks * (sizeof(state) + RequestBytes + alignUp(1<<size_log2, alignof(state))).
TEST(NcclDeviceRequirementsMicrotest, GinOutboxCreateRequirement_BufferSizeFormula) {
  struct Case {
    const char* name;
    int nBlocks;
    int sizeLog2;
    size_t goldenBufferSize;
  };
  const std::vector<Case> cases = {
    {"clamped_floor", 1, 0, 8448},   // 1 * (128 + 8192 + 128)
    {"at_floor", 3, 7, 25344},       // 3 * (128 + 8192 + 128)
    {"above_floor", 2, 8, 17152},    // 2 * (128 + 8192 + 256)
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    ncclGinOutboxHandle handle = poisonedHandle<ncclGinOutboxHandle>();
    ncclDevResourceRequirements_t req = poisonedRequirement();
    ASSERT_EQ(ncclGinOutboxCreateRequirement(c.nBlocks, c.sizeLog2, &handle, &req), ncclSuccess);
    const int clampedLog2 = std::max<int>(c.sizeLog2, 7);
    const size_t expected =
      static_cast<size_t>(c.nBlocks) * (sizeof(ncclGinOutboxState) + ncclGinOutboxState::RequestBytes +
                                        alignUpLocal(size_t(1) << clampedLog2, alignof(ncclGinOutboxState)));
    EXPECT_EQ(req.bufferSize, expected);
    EXPECT_EQ(req.bufferSize, c.goldenBufferSize);
    EXPECT_EQ(req.bufferAlign, 128u);
    EXPECT_EQ(req.outBufferHandle, &handle.bufHandle);
    // The outbox has no GIN signals; only the inbox A2A sibling below sets ginSignalCount.
    ExpectNoGinSignalFields(req);
  }
}

// Unlike the outbox, the inbox A2A doesn't floor size_log2 (pinned as observed).
TEST(NcclDeviceRequirementsMicrotest, GinInboxA2ACreateRequirement_NoFloorOnSizeLog2) {
  ncclTeam_t peers{/*nRanks=*/2, /*rank=*/0, /*stride=*/1};
  ncclGinInboxA2AHandle handle = poisonedHandle<ncclGinInboxA2AHandle>();
  ncclDevResourceRequirements_t req = poisonedRequirement();
  ASSERT_EQ(ncclGinInboxA2ACreateRequirement(peers, /*nBlocks=*/1, /*size_log2=*/0, &handle, &req), ncclSuccess);
  EXPECT_EQ(handle.size_log2, 0u) << "asymmetric with the outbox's 128-byte floor; no floor here today";
  // The only size_log2 below 7 here, so the only case where alignUpLocal rounds.
  EXPECT_EQ(req.bufferSize, size_t{256});  // 128-byte state + alignUp(1<<0, 128) rounding 1 up to 128
}

// bufferSize and ginSignalCount over a table of teams and block/size combinations.
TEST(NcclDeviceRequirementsMicrotest, GinInboxA2ACreateRequirement_BufferSizeAndSignalCountFormula) {
  struct Case {
    const char* name;
    int nRanks; // nPeers = nRanks - 1
    int nBlocks;
    int sizeLog2;
    size_t goldenBufferSize;
    int goldenGinSignalCount;
  };
  const std::vector<Case> cases = {
    {"smallest_safe_team", 2, 1, 7, 256, 2052},   // 1*(128+128); 1*4*(1+512)
    {"larger_team", 5, 2, 8, 768, 4128},          // 2*(128+256); 2*4*(4+512)
    // nPeers=3 is the only row whose reciprocal is inexact, so the r >= y correction actually runs.
    {"non_pow2_peers", 4, 1, 7, 256, 2060},       // 1*(128+128); 1*4*(3+512)
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    // Non-identity rank and stride, so a stray `+ team.rank` or `* team.stride` would show.
    ncclTeam_t peers{c.nRanks, /*rank=*/c.nRanks - 1, /*stride=*/4};
    ncclGinInboxA2AHandle handle = poisonedHandle<ncclGinInboxA2AHandle>();
    ncclDevResourceRequirements_t req = poisonedRequirement();
    ASSERT_EQ(ncclGinInboxA2ACreateRequirement(peers, c.nBlocks, c.sizeLog2, &handle, &req), ncclSuccess);
    const int nPeers = c.nRanks - 1;
    const size_t expectedBufferSize =
      static_cast<size_t>(c.nBlocks) *
      (sizeof(ncclGinInboxA2AState) + alignUpLocal(size_t(1) << c.sizeLog2, alignof(ncclGinInboxA2AState)));
    EXPECT_EQ(req.bufferSize, expectedBufferSize);
    EXPECT_EQ(req.bufferSize, c.goldenBufferSize);
    EXPECT_EQ(req.bufferAlign, 128u);
    EXPECT_EQ(req.outBufferHandle, &handle.bufHandle);
    EXPECT_EQ(req.ginSignalCount, c.nBlocks * 4 * (nPeers + (1 << ncclGinScratchMaxBufs_log2)));
    EXPECT_EQ(req.ginSignalCount, c.goldenGinSignalCount);
    EXPECT_EQ(req.outGinSignalStart, &handle.signals);
    ExpectNoGinCounterFields(req);
    // nPeers_rcp32 feeds imodFast32 in device code, so check the reciprocal contract.
    for (uint32_t x : {0u, 1u, 3u, 7u, 100u}) {
      EXPECT_EQ(imodFast32(x, static_cast<uint32_t>(nPeers), handle.nPeers_rcp32), x % static_cast<uint32_t>(nPeers))
        << "x=" << x;
    }
  }
}

// A 1-rank team makes nPeers = 0, so idivRcp32(0) divides by zero. Pins AICOMRCCL-3121; a fix flips this pin.
TEST(NcclDeviceRequirementsMicrotest, GinInboxA2ACreateRequirement_KnownBug_SingleRankTeamDividesByZero) {
#if defined(__x86_64__)  // AArch64 UDIV returns 0 instead of trapping.
  volatile int nRanksVolatile = 1;  // keeps -O2 from folding the divide away
  // Assigned field by field: a braced init's commas would split EXPECT_EXIT's arguments.
  EXPECT_EXIT(
      {
        DisableCoreDumpsForExpectedCrash();
        ncclTeam_t peers;
        peers.nRanks = nRanksVolatile;
        peers.rank = 0;
        peers.stride = 1;
        ncclGinInboxA2AHandle handle = poisonedHandle<ncclGinInboxA2AHandle>();
        ncclDevResourceRequirements_t req = poisonedRequirement();
        // Not reached: SIGFPE fires inside idivRcp32.
        ncclGinInboxA2ACreateRequirement(peers, /*nBlocks=*/1, /*size_log2=*/7, &handle, &req);
        // Read the quotient: otherwise an optimized build drops the unused divide and nothing traps.
        volatile uint32_t rcp = handle.nPeers_rcp32;
        (void)rcp;
      },
      ::testing::KilledBySignal(SIGFPE), "");
#else
  GTEST_SKIP() << "divide-by-zero trap is x86_64-specific; AArch64 UDIV returns 0 without trapping";
#endif
}

// ncclLsaBarrierCreateRequirement (src/nccl_device/lsa_barrier.cc)

// bufferSize == (3*nBarriers + nBarriers*team.nRanks) * sizeof(uint32_t); nBarriers == 0 gives 0.
TEST(NcclDeviceRequirementsMicrotest, LsaBarrierCreateRequirement_BufferSizeFormula) {
  struct Case {
    const char* name;
    int nBarriers;
    int nRanks;
    size_t goldenBufferSize;
  };
  const std::vector<Case> cases = {
    {"zero_barriers", 0, 4, 0},     // (0 + 0) * 4
    {"small", 2, 4, 56},            // (6 + 8) * 4
    {"larger", 5, 3, 120},          // (15 + 15) * 4
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    ncclTeam_t team{c.nRanks, /*rank=*/c.nRanks - 1, /*stride=*/4};
    ncclLsaBarrierHandle_t handle = poisonedHandle<ncclLsaBarrierHandle_t>();
    ncclDevResourceRequirements_t req = poisonedRequirement();
    ASSERT_EQ(ncclLsaBarrierCreateRequirement(team, c.nBarriers, &handle, &req), ncclSuccess);
    const size_t expected = static_cast<size_t>(3 * c.nBarriers + c.nBarriers * c.nRanks) * sizeof(uint32_t);
    EXPECT_EQ(req.bufferSize, expected);
    EXPECT_EQ(req.bufferSize, c.goldenBufferSize);
    EXPECT_EQ(req.bufferAlign, size_t{4});  // alignof(uint32_t) today; a literal, so a change fails here
    EXPECT_EQ(req.outBufferHandle, &handle.bufHandle);
    EXPECT_EQ(handle.nBarriers, c.nBarriers);
    // The buffer-only sibling of the GIN/CFT barriers below: no GIN signals here.
    ExpectNoGinSignalFields(req);
  }
}

// ncclGinBarrierCreateRequirement (src/nccl_device/gin_barrier.cc)

// comm is unused, so nullptr works. Sets ginSignalCount but no buffer, unlike the LSA/CFT barriers.
TEST(NcclDeviceRequirementsMicrotest, GinBarrierCreateRequirement_NullCommAndSignalCountFormula) {
  struct Case {
    const char* name;
    int nBarriers;
    int nRanks;
  };
  const std::vector<Case> cases = {
    {"zero_barriers", 0, 5},
    {"small", 3, 4},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    ncclTeam_t team{c.nRanks, /*rank=*/c.nRanks - 1, /*stride=*/4};
    ncclGinBarrierHandle_t handle = poisonedHandle<ncclGinBarrierHandle_t>();
    ncclDevResourceRequirements_t req = poisonedRequirement();
    ASSERT_EQ(ncclGinBarrierCreateRequirement(/*comm=*/nullptr, team, c.nBarriers, &handle, &req), ncclSuccess);
    EXPECT_EQ(req.ginSignalCount, c.nBarriers * c.nRanks);
    EXPECT_EQ(req.outGinSignalStart, &handle.signal0);
    EXPECT_EQ(req.bufferSize, 0u);
    EXPECT_EQ(req.bufferAlign, 0u);
    EXPECT_EQ(req.outBufferHandle, nullptr);
    ExpectNoGinCounterFields(req);
  }
}

// ncclCftBarrierCreateRequirement (src/nccl_device/cft_barrier.cc)

// bufferSize == (3*nBarriers + nBarriers*team.nRanks) * NCCL_CFT_BARRIER_GRAN (16, vs. the LSA
// barrier's 4), so swapping the two constants fails here.
TEST(NcclDeviceRequirementsMicrotest, CftBarrierCreateRequirement_BufferSizeFormula) {
  struct Case {
    const char* name;
    int nBarriers;
    int nRanks;
    size_t goldenBufferSize;
  };
  const std::vector<Case> cases = {
    {"zero_barriers", 0, 4, 0},      // (0 + 0) * 16
    {"small", 2, 4, 224},            // (6 + 8) * 16
    {"larger", 5, 3, 480},           // (15 + 15) * 16
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    ncclTeam_t team{c.nRanks, /*rank=*/c.nRanks - 1, /*stride=*/4};
    ncclCftBarrierHandle_t handle = poisonedHandle<ncclCftBarrierHandle_t>();
    ncclDevResourceRequirements_t req = poisonedRequirement();
    ASSERT_EQ(ncclCftBarrierCreateRequirement(team, c.nBarriers, &handle, &req), ncclSuccess);
    const size_t expected = static_cast<size_t>(3 * c.nBarriers + c.nBarriers * c.nRanks) * NCCL_CFT_BARRIER_GRAN;
    EXPECT_EQ(req.bufferSize, expected);
    EXPECT_EQ(req.bufferSize, c.goldenBufferSize);
    EXPECT_EQ(req.bufferAlign, size_t{16});  // NCCL_CFT_BARRIER_ALIGN today; a literal, so a change fails here
    EXPECT_EQ(req.outBufferHandle, &handle.bufHandle);
    EXPECT_EQ(handle.nBarriers, c.nBarriers);
    ExpectNoGinSignalFields(req);
  }
}
