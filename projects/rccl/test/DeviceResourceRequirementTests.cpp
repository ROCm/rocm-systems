/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
// Unit tests for the six device-resource requirement builders in src/nccl_device/{gin_scratch,
// ll_a2a,lsa_barrier,gin_barrier}.cc. They compute buffer sizes and GIN signal counts that device
// kernels index into; an off-by-one here under-allocates scratch and surfaces as device corruption.
#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "common/ProcessIsolatedTestRunner.hpp"

// ncclDevResourceRequirements / ncclTeam definitions live behind core_tmp.h, transitively included
// by each of the four headers below.
#include "nccl_device/impl/ll_a2a__types.h"
#include "nccl_device/impl/lsa_barrier__types.h"
#include "nccl_device/impl/gin_barrier__types.h"
// gin_scratch__types.h below expects the unqualified alignUp/imodFast32 that production TUs get
// transitively through "core.h", which we do not include; bitops.h supplies them directly.
#include "bitops.h"
// Outbox/inbox declarations still live at their pre-public-split location, not under nccl_device/impl/.
#include "device/symmetric/gin_scratch.h"

namespace RcclUnitTesting {

namespace {

// Re-implemented via division rather than the production bit-mask trick, so a bug in the real
// alignUp cannot also hide inside the test's own expectation.
constexpr size_t alignUpLocal(size_t x, size_t a) {
  return ((x + a - 1) / a) * a;
}

// Non-zero poison so a missing memset() would leave visible garbage instead of a false-zero read.
ncclDevResourceRequirements_t poisonedRequirement() {
  ncclDevResourceRequirements_t req;
  std::memset(&req, 0xAA, sizeof(req));
  return req;
}

} // namespace

// divUp rounding: maxEltSize of 1, 7, 8 all round up to one 8-byte slot; 9 and 16 need two.
TEST(DeviceResourceRequirementTests, LLA2ACalcSlots_DivUpRounding) {
  struct Case {
    const char* name;
    int maxEltSize;
    int expectedSlots;
  };
  const std::vector<Case> cases = {
    {"eltSize_1", 1, 1}, {"eltSize_7", 7, 1}, {"eltSize_8", 8, 1}, {"eltSize_9", 9, 2}, {"eltSize_16", 16, 2},
  };
  for (const Case& c : cases) {
    EXPECT_EQ(ncclLLA2ACalcSlots(/*maxElts=*/1, c.maxEltSize), c.expectedSlots) << "case: " << c.name;
  }
}

// The result scales linearly with maxElts for a fixed maxEltSize.
TEST(DeviceResourceRequirementTests, LLA2ACalcSlots_LinearInMaxElts) {
  const int base = ncclLLA2ACalcSlots(/*maxElts=*/1, /*maxEltSize=*/8);
  EXPECT_EQ(ncclLLA2ACalcSlots(/*maxElts=*/5, /*maxEltSize=*/8), 5 * base);
}

// maxElts = 0 always yields 0 slots, independent of maxEltSize.
TEST(DeviceResourceRequirementTests, LLA2ACalcSlots_ZeroMaxEltsIsZero) {
  EXPECT_EQ(ncclLLA2ACalcSlots(/*maxElts=*/0, /*maxEltSize=*/8), 0);
}

// bufferSize == nBlocks * (1 + 2*nSlots) * 16; nSlots == 0 is the header-only case. Values stay
// sane on purpose: the plain-int formula can overflow before promoting to size_t, and pinning UB is out of scope.
TEST(DeviceResourceRequirementTests, LLA2ACreateRequirement_BufferSizeFormula) {
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
    ncclLLA2AHandle_t handle;
    std::memset(&handle, 0xAA, sizeof(handle));
    ncclDevResourceRequirements_t req = poisonedRequirement();
    ASSERT_EQ(ncclLLA2ACreateRequirement(c.nBlocks, c.nSlots, &handle, &req), ncclSuccess) << "case: " << c.name;
    EXPECT_EQ(req.bufferSize, static_cast<size_t>(c.nBlocks) * (1 + 2 * c.nSlots) * 16) << "case: " << c.name;
    EXPECT_EQ(req.bufferAlign, 16u) << "case: " << c.name;
    EXPECT_EQ(req.outBufferHandle, &handle.bufHandle) << "case: " << c.name;
    EXPECT_EQ(handle.nSlots, static_cast<uint32_t>(c.nSlots)) << "case: " << c.name;
    // Fields this function never sets must still read back zero: proves the memset prologue ran.
    EXPECT_EQ(req.next, nullptr) << "case: " << c.name;
    EXPECT_EQ(req.ginSignalCount, 0) << "case: " << c.name;
    EXPECT_EQ(req.ginCounterCount, 0) << "case: " << c.name;
    EXPECT_EQ(req.outGinSignalStart, nullptr) << "case: " << c.name;
    EXPECT_EQ(req.outGinCounterStart, nullptr) << "case: " << c.name;
  }
}

// Realistic-usage row (mirrors src/sym_kernels.cc), not a contract: the two functions are independent by signature.
TEST(DeviceResourceRequirementTests, LLA2ACreateRequirement_RoundTripWithCalcSlots) {
  const int nSlots = ncclLLA2ACalcSlots(/*maxElts=*/4, /*maxEltSize=*/16);
  const int nBlocks = 3;
  ncclLLA2AHandle_t handle;
  ncclDevResourceRequirements_t req = poisonedRequirement();
  ASSERT_EQ(ncclLLA2ACreateRequirement(nBlocks, nSlots, &handle, &req), ncclSuccess);
  EXPECT_EQ(req.bufferSize, static_cast<size_t>(nBlocks) * (1 + 2 * nSlots) * 16);
  EXPECT_EQ(handle.nSlots, static_cast<uint32_t>(nSlots));
}

// The 128-byte floor: size_log2 below 7 clamps to 7, and outHandle reads back the clamped value.
TEST(DeviceResourceRequirementTests, GinOutboxCreateRequirement_FloorClampsSizeLog2Below7) {
  const std::vector<int> belowFloor = {0, 5, 6, 7};
  for (int sizeLog2 : belowFloor) {
    ncclGinOutboxHandle handle;
    std::memset(&handle, 0xAA, sizeof(handle));
    ncclDevResourceRequirements_t req = poisonedRequirement();
    ASSERT_EQ(ncclGinOutboxCreateRequirement(/*nBlocks=*/1, sizeLog2, &handle, &req), ncclSuccess)
      << "case: size_log2=" << sizeLog2;
    EXPECT_EQ(handle.size_log2, 7u) << "case: size_log2=" << sizeLog2;
  }
}

// Above the floor, size_log2 passes through unclamped.
TEST(DeviceResourceRequirementTests, GinOutboxCreateRequirement_SizeLog2AboveFloorIsUnclamped) {
  ncclGinOutboxHandle handle;
  ncclDevResourceRequirements_t req = poisonedRequirement();
  ASSERT_EQ(ncclGinOutboxCreateRequirement(/*nBlocks=*/1, /*size_log2=*/8, &handle, &req), ncclSuccess);
  EXPECT_EQ(handle.size_log2, 8u);
}

// bufferSize == nBlocks * (sizeof(state) + RequestBytes + alignUp(1<<size_log2, alignof(state))).
TEST(DeviceResourceRequirementTests, GinOutboxCreateRequirement_BufferSizeFormula) {
  struct Case {
    const char* name;
    int nBlocks;
    int sizeLog2;
  };
  const std::vector<Case> cases = {
    {"clamped_floor", 1, 0},
    {"at_floor", 3, 7},
    {"above_floor", 2, 8},
  };
  for (const Case& c : cases) {
    ncclGinOutboxHandle handle;
    ncclDevResourceRequirements_t req = poisonedRequirement();
    ASSERT_EQ(ncclGinOutboxCreateRequirement(c.nBlocks, c.sizeLog2, &handle, &req), ncclSuccess) << "case: " << c.name;
    const int clampedLog2 = std::max<int>(c.sizeLog2, 7);
    const size_t expected =
      static_cast<size_t>(c.nBlocks) * (sizeof(ncclGinOutboxState) + ncclGinOutboxState::RequestBytes +
                                        alignUpLocal(size_t(1) << clampedLog2, alignof(ncclGinOutboxState)));
    EXPECT_EQ(req.bufferSize, expected) << "case: " << c.name;
    EXPECT_EQ(req.bufferAlign, 128u) << "case: " << c.name;
    EXPECT_EQ(req.outBufferHandle, &handle.bufHandle) << "case: " << c.name;
    // The outbox has no GIN signals; only the inbox A2A sibling below sets ginSignalCount.
    EXPECT_EQ(req.ginSignalCount, 0) << "case: " << c.name;
    EXPECT_EQ(req.next, nullptr) << "case: " << c.name;
    EXPECT_EQ(req.ginCounterCount, 0) << "case: " << c.name;
    EXPECT_EQ(req.outGinSignalStart, nullptr) << "case: " << c.name;
    EXPECT_EQ(req.outGinCounterStart, nullptr) << "case: " << c.name;
  }
}

// Unlike the outbox, the inbox A2A applies no floor to size_log2: pinned as observed, not endorsed.
TEST(DeviceResourceRequirementTests, GinInboxA2ACreateRequirement_NoFloorOnSizeLog2) {
  ncclTeam_t peers{/*nRanks=*/2, /*rank=*/0, /*stride=*/1};
  ncclGinInboxA2AHandle handle;
  ncclDevResourceRequirements_t req = poisonedRequirement();
  ASSERT_EQ(ncclGinInboxA2ACreateRequirement(peers, /*nBlocks=*/1, /*size_log2=*/0, &handle, &req), ncclSuccess);
  EXPECT_EQ(handle.size_log2, 0u) << "asymmetric with the outbox's 128-byte floor; no floor here today";
}

// bufferSize and ginSignalCount formulas over a small table of teams and block/size combinations.
TEST(DeviceResourceRequirementTests, GinInboxA2ACreateRequirement_BufferSizeAndSignalCountFormula) {
  struct Case {
    const char* name;
    int nRanks; // nPeers = nRanks - 1
    int nBlocks;
    int sizeLog2;
  };
  const std::vector<Case> cases = {
    {"smallest_safe_team", 2, 1, 7},
    {"larger_team", 5, 2, 8},
  };
  for (const Case& c : cases) {
    ncclTeam_t peers{c.nRanks, /*rank=*/0, /*stride=*/1};
    ncclGinInboxA2AHandle handle;
    ncclDevResourceRequirements_t req = poisonedRequirement();
    ASSERT_EQ(ncclGinInboxA2ACreateRequirement(peers, c.nBlocks, c.sizeLog2, &handle, &req), ncclSuccess)
      << "case: " << c.name;
    const int nPeers = c.nRanks - 1;
    const size_t expectedBufferSize =
      static_cast<size_t>(c.nBlocks) *
      (sizeof(ncclGinInboxA2AState) + alignUpLocal(size_t(1) << c.sizeLog2, alignof(ncclGinInboxA2AState)));
    EXPECT_EQ(req.bufferSize, expectedBufferSize) << "case: " << c.name;
    EXPECT_EQ(req.bufferAlign, 128u) << "case: " << c.name;
    EXPECT_EQ(req.outBufferHandle, &handle.bufHandle) << "case: " << c.name;
    EXPECT_EQ(req.ginSignalCount, c.nBlocks * 4 * (nPeers + (1 << ncclGinScratchMaxBufs_log2))) << "case: " << c.name;
    EXPECT_EQ(req.outGinSignalStart, &handle.signals) << "case: " << c.name;
    EXPECT_EQ(req.next, nullptr) << "case: " << c.name;
    EXPECT_EQ(req.ginCounterCount, 0) << "case: " << c.name;
    EXPECT_EQ(req.outGinCounterStart, nullptr) << "case: " << c.name;
  }
}

// A 1-rank team makes nPeers = 0, and idivRcp32(0) (bitops.h) divides by zero at runtime: a real
// defect, unreachable from any in-tree caller (src/sym_kernels.cc's only call site guards
// nRanks >= 2 via computeLsaSize's GCD construction). A crash-containment pin, not an assertion:
// it checks the isolated child was killed, since the call itself never returns normally.
TEST(DeviceResourceRequirementTests, GinInboxA2ACreateRequirement_SingleRankTeam_CrashesOnDivideByZero) {
  // Not RUN_ISOLATED_TEST (it EXPECT_TRUE's the run); no lambda assertion either, so a future fix
  // that makes the call return normally flips executeAllTests() true and fails the EXPECT_FALSE below.
  ProcessIsolatedTestRunner::registerTest("GinInboxA2A_SingleRankTeam_DivideByZero", [] {
    ncclTeam_t peers{/*nRanks=*/1, /*rank=*/0, /*stride=*/1};
    ncclGinInboxA2AHandle handle;
    ncclDevResourceRequirements_t req = poisonedRequirement();
    // Never expected to return: SIGFPE fires inside idivRcp32 before this line completes.
    ncclGinInboxA2ACreateRequirement(peers, /*nBlocks=*/1, /*size_log2=*/7, &handle, &req);
  });
  EXPECT_FALSE(ProcessIsolatedTestRunner::executeAllTests())
    << "expected the isolated child to be killed by SIGFPE; if it now passes, the defect was fixed";
}

// bufferSize == (3*nBarriers + nBarriers*team.nRanks) * sizeof(uint32_t); nBarriers == 0 gives 0.
// Same out-of-scope-overflow reasoning as the LLA2A formula above applies to this plain-int formula.
TEST(DeviceResourceRequirementTests, LsaBarrierCreateRequirement_BufferSizeFormula) {
  struct Case {
    const char* name;
    int nBarriers;
    int nRanks;
  };
  const std::vector<Case> cases = {
    {"zero_barriers", 0, 4},
    {"small", 2, 4},
    {"larger", 5, 3},
  };
  for (const Case& c : cases) {
    ncclTeam_t team{c.nRanks, /*rank=*/0, /*stride=*/1};
    ncclLsaBarrierHandle_t handle;
    ncclDevResourceRequirements_t req = poisonedRequirement();
    ASSERT_EQ(ncclLsaBarrierCreateRequirement(team, c.nBarriers, &handle, &req), ncclSuccess) << "case: " << c.name;
    const size_t expected = static_cast<size_t>(3 * c.nBarriers + c.nBarriers * c.nRanks) * sizeof(uint32_t);
    EXPECT_EQ(req.bufferSize, expected) << "case: " << c.name;
    EXPECT_EQ(req.bufferAlign, alignof(uint32_t)) << "case: " << c.name;
    EXPECT_EQ(req.outBufferHandle, &handle.bufHandle) << "case: " << c.name;
    EXPECT_EQ(handle.nBarriers, c.nBarriers) << "case: " << c.name;
    // The buffer-only sibling of the GIN barrier below: no GIN signals here.
    EXPECT_EQ(req.ginSignalCount, 0) << "case: " << c.name;
    EXPECT_EQ(req.next, nullptr) << "case: " << c.name;
    EXPECT_EQ(req.ginCounterCount, 0) << "case: " << c.name;
    EXPECT_EQ(req.outGinSignalStart, nullptr) << "case: " << c.name;
    EXPECT_EQ(req.outGinCounterStart, nullptr) << "case: " << c.name;
  }
}

// comm is unused, so nullptr is the contract that makes this testable without a GPU. Mirrors the
// LSA barrier above: this one sets ginSignalCount but no buffer, that one the reverse.
TEST(DeviceResourceRequirementTests, GinBarrierCreateRequirement_NullCommAndSignalCountFormula) {
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
    ncclTeam_t team{c.nRanks, /*rank=*/0, /*stride=*/1};
    ncclGinBarrierHandle_t handle;
    ncclDevResourceRequirements_t req = poisonedRequirement();
    ASSERT_EQ(ncclGinBarrierCreateRequirement(/*comm=*/nullptr, team, c.nBarriers, &handle, &req), ncclSuccess)
      << "case: " << c.name;
    EXPECT_EQ(req.ginSignalCount, c.nBarriers * c.nRanks) << "case: " << c.name;
    EXPECT_EQ(req.outGinSignalStart, &handle.signal0) << "case: " << c.name;
    EXPECT_EQ(req.bufferSize, 0u) << "case: " << c.name;
    EXPECT_EQ(req.bufferAlign, 0u) << "case: " << c.name;
    EXPECT_EQ(req.outBufferHandle, nullptr) << "case: " << c.name;
    EXPECT_EQ(req.next, nullptr) << "case: " << c.name;
    EXPECT_EQ(req.ginCounterCount, 0) << "case: " << c.name;
    EXPECT_EQ(req.outGinCounterStart, nullptr) << "case: " << c.name;
  }
}

} // namespace RcclUnitTesting
