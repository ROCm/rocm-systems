/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/all_reduce/dda_all_reduce_ipc.cu, #include-d via
// DDA_ALL_REDUCE_IPC_CC_PATH: the eligibility gates, the datatype dispatch and the flat/tree kernel
// selection. Own binary: rccl-UnitTestsMicro links dda_fakes.cc, which defines these same symbols.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../common/DdaIpcTestHelpers.hpp"
#include "fakes/hip_fakes.h"

#include DDA_ALL_REDUCE_IPC_CC_PATH

namespace {

using nccl_dda_detail::DdaIpcBarrierState;
using nccl_dda_detail::kDdaNranks;
using RcclUnitTesting::DdaIpcMockComm;

// Spelled out rather than read from kDdaFlatTreeThresholdBytes, so a moved threshold fails here.
constexpr size_t kFlatTreeThresholdBytes = 256 * 1024;
constexpr size_t kFp32CountAtThreshold = kFlatTreeThresholdBytes / sizeof(float);
constexpr size_t kFp16CountAtThreshold = kFlatTreeThresholdBytes / sizeof(uint16_t);
// 8 MiB fp32 tree payload, which the tree path copies into scratch before it launches.
constexpr size_t kFp32CountCopiedBeforeTree = (8u << 20) / sizeof(float);

void* const kSendbuff = reinterpret_cast<void*>(0x10);
void* const kRecvbuff = reinterpret_cast<void*>(0x20);

// Never destroyed: ~DdaIpcBarrierState would pull in IPC resource destructors this binary does not link.
DdaIpcBarrierState* BarrierState() {
  static DdaIpcBarrierState* const state = new DdaIpcBarrierState();
  return state;
}

// ---------------------------------------------------------------------------
// ncclAllReduceDdaIpcEligible
// ---------------------------------------------------------------------------

class DdaAllReduceIpcEligibleTest : public ::testing::Test {
 protected:
  bool Eligible(size_t count, ncclDataType_t datatype = ncclFloat32, ncclRedOp_t op = ncclSum) {
    return ncclAllReduceDdaIpcEligible(mock_.get(), kSendbuff, kRecvbuff, count, datatype, op);
  }

  DdaIpcMockComm mock_;
};

TEST_F(DdaAllReduceIpcEligibleTest, NullCommIsIneligible) {
  EXPECT_FALSE(ncclAllReduceDdaIpcEligible(nullptr, kSendbuff, kRecvbuff, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaAllReduceIpcEligibleTest, BaselineSmallFp32IsEligible) {
  EXPECT_TRUE(Eligible(4));
}

TEST_F(DdaAllReduceIpcEligibleTest, MissingIpcMemHandlerIsIneligible) {
  mock_.comm.ddaIpcMemHandler = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllReduceIpcEligibleTest, MissingBarrierStateIsIneligible) {
  mock_.comm.ddaIpcBarrierState = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllReduceIpcEligibleTest, MultiNodeIsIneligible) {
  mock_.comm.nNodes = 2;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllReduceIpcEligibleTest, RankCountOtherThanKDdaNranksIsIneligible) {
  mock_.comm.nRanks = kDdaNranks - 1;
  EXPECT_FALSE(Eligible(4));
  mock_.comm.nRanks = kDdaNranks + 1;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllReduceIpcEligibleTest, MissingBootstrapIsIneligible) {
  mock_.comm.bootstrap = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllReduceIpcEligibleTest, MissingScratchIsIneligible) {
  mock_.comm.ddaScratch = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllReduceIpcEligibleTest, MissingPeerPtrsIsIneligible) {
  mock_.comm.ddaPeerPtrsDev = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllReduceIpcEligibleTest, ZeroCountIsIneligible) {
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaAllReduceIpcEligibleTest, OnlySumIsEligible) {
  EXPECT_FALSE(Eligible(4, ncclFloat32, ncclMax));
  EXPECT_FALSE(Eligible(4, ncclFloat32, ncclProd));
  EXPECT_FALSE(Eligible(4, ncclFloat32, ncclMin));
  EXPECT_FALSE(Eligible(4, ncclFloat32, ncclAvg));
}

TEST_F(DdaAllReduceIpcEligibleTest, EachSupportedDatatypeIsEligible) {
  EXPECT_TRUE(Eligible(4, ncclFloat32));
  EXPECT_TRUE(Eligible(8, ncclFloat16));
  EXPECT_TRUE(Eligible(8, ncclBfloat16));
}

TEST_F(DdaAllReduceIpcEligibleTest, UnsupportedDatatypesAreIneligible) {
  EXPECT_FALSE(Eligible(4, ncclInt32));
  EXPECT_FALSE(Eligible(4, ncclUint32));
  EXPECT_FALSE(Eligible(2, ncclFloat64));
  EXPECT_FALSE(Eligible(16, ncclInt8));
}

TEST_F(DdaAllReduceIpcEligibleTest, PayloadUpToScratchIsEligibleAndOneVectorMoreIsNot) {
  mock_.comm.ddaScratchBytes = 64;
  EXPECT_TRUE(Eligible(16));
  EXPECT_FALSE(Eligible(20));
}

TEST_F(DdaAllReduceIpcEligibleTest, PayloadNotMultipleOf16BytesIsIneligible) {
  EXPECT_FALSE(Eligible(3));
  EXPECT_FALSE(Eligible(6, ncclFloat16));
  EXPECT_FALSE(Eligible(4, ncclBfloat16));
}

TEST_F(DdaAllReduceIpcEligibleTest, FlatPathSkipsPerRankSliceCheck) {
  EXPECT_TRUE(Eligible(12));
}

TEST_F(DdaAllReduceIpcEligibleTest, PayloadAtThresholdIsEligible) {
  EXPECT_TRUE(Eligible(kFp32CountAtThreshold));
}

TEST_F(DdaAllReduceIpcEligibleTest, TreePathCountNotDivisibleByRanksIsIneligible) {
  EXPECT_FALSE(Eligible(kFp32CountAtThreshold + 4));
}

TEST_F(DdaAllReduceIpcEligibleTest, TreePathMisalignedPerRankSliceIsIneligible) {
  EXPECT_FALSE(Eligible(kFp32CountAtThreshold + kDdaNranks));
}

TEST_F(DdaAllReduceIpcEligibleTest, TreePathAlignedPerRankSliceIsEligible) {
  EXPECT_TRUE(Eligible(kFp32CountAtThreshold + 4 * kDdaNranks));
  EXPECT_TRUE(Eligible(kFp16CountAtThreshold + 8 * kDdaNranks, ncclBfloat16));
  EXPECT_FALSE(Eligible(kFp16CountAtThreshold + 4 * kDdaNranks, ncclBfloat16));
}

// ---------------------------------------------------------------------------
// ncclAllReduceDdaIpc / ncclAllReduceDdaIpcTyped
// ---------------------------------------------------------------------------

struct Launch {
  const void* kernel;
  unsigned gridX;
  unsigned blockX;
  void* ipcbuffs;
  size_t count;
  const void* sendbuff;
  void* recvbuff;
  int rank;
  hipStream_t stream;
  hipEvent_t stopEvent;
  size_t memcpysBefore;
};

class DdaAllReduceIpcDispatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetHipFakes();
    g_hipMemcpyAsync = [this](void*, const void*, size_t, hipMemcpyKind, hipStream_t stream) {
      memcpyStreams_.push_back(stream);
      return hipSuccess;
    };
    g_hipGetLastError = []() { return hipSuccess; };
    g_hipExtLaunchKernel = [this](const void* kernel, dim3 grid, dim3 block, void** args, size_t, hipStream_t stream,
                                  hipEvent_t, hipEvent_t stopEvent, int) {
      launches_.push_back({kernel, grid.x, block.x, *static_cast<void**>(args[0]), *static_cast<size_t*>(args[2]),
                           *static_cast<const void**>(args[3]), *static_cast<void**>(args[1]),
                           *static_cast<int*>(args[4]), stream, stopEvent, g_hipMemcpyAsyncArgs.size()});
      return hipSuccess;
    };
    mock_.comm.ddaIpcBarrierState = BarrierState();
    mock_.comm.rank = 3;
    mock_.comm.addonStopEvent = kStopEvent;
  }
  void TearDown() override { ResetHipFakes(); }

  ncclResult_t Run(size_t count, ncclDataType_t datatype) {
    return ncclAllReduceDdaIpc(kSendbuff, kRecvbuff, count, datatype, ncclSum, mock_.get(), kStream);
  }

  template <typename T>
  static const void* Flat() {
    return reinterpret_cast<const void*>(&dda::common::ddaAllReduceFlatIpc<T, kDdaNranks, false>);
  }

  template <typename T>
  static const void* Tree() {
    return reinterpret_cast<const void*>(&dda::common::ddaAllReduceTreeIpc<T, kDdaNranks, false>);
  }

  const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);
  const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
  DdaIpcMockComm mock_;
  std::vector<Launch> launches_;
  std::vector<hipStream_t> memcpyStreams_;
};

TEST_F(DdaAllReduceIpcDispatchTest, UnsupportedDatatypeReturnsInvalidArgumentWithoutLaunching) {
  for (ncclDataType_t datatype : {ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclFloat64}) {
    EXPECT_EQ(ncclInvalidArgument, Run(16, datatype)) << "datatype " << datatype;
  }
  EXPECT_TRUE(launches_.empty());
  EXPECT_EQ(0, g_hipMemcpyAsyncCalls);
}

TEST_F(DdaAllReduceIpcDispatchTest, Float32LaunchesFloatKernel) {
  ASSERT_EQ(ncclSuccess, Run(16, ncclFloat32));
  ASSERT_EQ(1u, launches_.size());
  EXPECT_EQ(Flat<float>(), launches_[0].kernel);
}

TEST_F(DdaAllReduceIpcDispatchTest, Float16LaunchesHalfKernel) {
  ASSERT_EQ(ncclSuccess, Run(16, ncclFloat16));
  ASSERT_EQ(1u, launches_.size());
  EXPECT_EQ(Flat<half>(), launches_[0].kernel);
}

TEST_F(DdaAllReduceIpcDispatchTest, Bfloat16LaunchesBf16Kernel) {
  ASSERT_EQ(ncclSuccess, Run(16, ncclBfloat16));
  ASSERT_EQ(1u, launches_.size());
  EXPECT_EQ(Flat<bf16>(), launches_[0].kernel);
}

TEST_F(DdaAllReduceIpcDispatchTest, EachMissingIpcResourceReturnsInvalidUsageWithoutLaunching) {
  ncclComm& comm = mock_.comm;
  for (int missing = 0; missing < 4; ++missing) {
    comm.ddaIpcMemHandler = missing == 0 ? nullptr : reinterpret_cast<ncclIpcMemHandler*>(0x1);
    comm.ddaScratch = missing == 1 ? nullptr : reinterpret_cast<void*>(0x2);
    comm.ddaPeerPtrsDev = missing == 2 ? nullptr : reinterpret_cast<void*>(0x3);
    comm.ddaIpcBarrierState = missing == 3 ? nullptr : BarrierState();
    EXPECT_EQ(ncclInvalidUsage, Run(16, ncclFloat32)) << "missing resource " << missing;
  }
  EXPECT_TRUE(launches_.empty());
}

TEST_F(DdaAllReduceIpcDispatchTest, ScratchCheckUsesTheDispatchedElementSize) {
  mock_.comm.ddaScratchBytes = 64;
  EXPECT_EQ(ncclSuccess, Run(32, ncclFloat16));
  EXPECT_EQ(ncclInvalidArgument, Run(17, ncclFloat32));
  EXPECT_EQ(1u, launches_.size());
}

TEST_F(DdaAllReduceIpcDispatchTest, LaunchCarriesBuffersCountRankAndTakesStopEvent) {
  ASSERT_EQ(ncclSuccess, Run(64, ncclFloat32));
  ASSERT_EQ(1u, launches_.size());
  EXPECT_EQ(mock_.comm.ddaPeerPtrsDev, launches_[0].ipcbuffs);
  EXPECT_EQ(64u, launches_[0].count);
  EXPECT_EQ(kSendbuff, launches_[0].sendbuff);
  EXPECT_EQ(kRecvbuff, launches_[0].recvbuff);
  EXPECT_EQ(3, launches_[0].rank);
  EXPECT_EQ(kStream, launches_[0].stream);
  EXPECT_EQ(kStopEvent, launches_[0].stopEvent);
  EXPECT_EQ(nullptr, mock_.comm.addonStopEvent);
}

TEST_F(DdaAllReduceIpcDispatchTest, PayloadAtThresholdRunsFlatWithoutCopy) {
  ASSERT_EQ(ncclSuccess, Run(kFp32CountAtThreshold, ncclFloat32));
  ASSERT_EQ(1u, launches_.size());
  EXPECT_EQ(Flat<float>(), launches_[0].kernel);
  EXPECT_EQ(0, g_hipMemcpyAsyncCalls);
}

TEST_F(DdaAllReduceIpcDispatchTest, PayloadAboveThresholdRunsTree) {
  const size_t count = kFp32CountAtThreshold + 4 * kDdaNranks;
  ASSERT_EQ(ncclSuccess, Run(count, ncclFloat32));
  ASSERT_EQ(1u, launches_.size());
  EXPECT_EQ(Tree<float>(), launches_[0].kernel);
  EXPECT_EQ(count, launches_[0].count);
}

TEST_F(DdaAllReduceIpcDispatchTest, LargeTreePayloadIsCopiedToScratchBeforeLaunch) {
  const size_t count = kFp32CountCopiedBeforeTree;
  ASSERT_EQ(ncclSuccess, Run(count, ncclFloat32));
  ASSERT_EQ(1u, launches_.size());
  EXPECT_EQ(Tree<float>(), launches_[0].kernel);
  ASSERT_EQ(1u, g_hipMemcpyAsyncArgs.size());
  EXPECT_EQ(1u, launches_[0].memcpysBefore);
  EXPECT_EQ(mock_.comm.ddaScratch, g_hipMemcpyAsyncArgs[0].dst);
  EXPECT_EQ(kSendbuff, g_hipMemcpyAsyncArgs[0].src);
  EXPECT_EQ(count * sizeof(float), g_hipMemcpyAsyncArgs[0].bytes);
  ASSERT_EQ(1u, memcpyStreams_.size());
  EXPECT_EQ(kStream, memcpyStreams_[0]);
  EXPECT_EQ(kStream, launches_[0].stream);
}

TEST_F(DdaAllReduceIpcDispatchTest, ScratchCopyErrorIsReportedWithoutLaunching) {
  g_hipMemcpyAsync = [](void*, const void*, size_t, hipMemcpyKind, hipStream_t) { return hipErrorInvalidValue; };
  EXPECT_EQ(ncclUnhandledCudaError, Run(kFp32CountCopiedBeforeTree, ncclFloat32));
  EXPECT_EQ(1, g_hipMemcpyAsyncCalls);
  EXPECT_TRUE(launches_.empty());
}

TEST_F(DdaAllReduceIpcDispatchTest, AboveThresholdCountNotDivisibleByRanksFallsBackToFlat) {
  ASSERT_EQ(ncclSuccess, Run(kFp32CountAtThreshold + 4, ncclFloat32));
  ASSERT_EQ(1u, launches_.size());
  EXPECT_EQ(Flat<float>(), launches_[0].kernel);
  EXPECT_EQ(0, g_hipMemcpyAsyncCalls);
}

TEST_F(DdaAllReduceIpcDispatchTest, ThresholdIsInBytesNotElements) {
  ASSERT_EQ(ncclSuccess, Run(kFp16CountAtThreshold, ncclBfloat16));
  ASSERT_EQ(ncclSuccess, Run(kFp16CountAtThreshold + 8 * kDdaNranks, ncclBfloat16));
  ASSERT_EQ(2u, launches_.size());
  EXPECT_EQ(Flat<bf16>(), launches_[0].kernel);
  EXPECT_EQ(Tree<bf16>(), launches_[1].kernel);
}

TEST_F(DdaAllReduceIpcDispatchTest, LaunchGeometryFollowsTheDispatchedElementSize) {
  ASSERT_EQ(ncclSuccess, Run(4096, ncclBfloat16));
  ASSERT_EQ(ncclSuccess, Run(4096, ncclFloat32));
  ASSERT_EQ(2u, launches_.size());
  EXPECT_EQ(1u, launches_[0].gridX);
  EXPECT_EQ(512u, launches_[0].blockX);
  EXPECT_EQ(2u, launches_[1].gridX);
  EXPECT_EQ(512u, launches_[1].blockX);
}

TEST_F(DdaAllReduceIpcDispatchTest, LaunchErrorIsReported) {
  g_hipGetLastError = []() { return hipErrorLaunchFailure; };
  EXPECT_EQ(ncclUnhandledCudaError, Run(16, ncclFloat32));
}

// ---------------------------------------------------------------------------
// ncclAllReduceDdaIpcBlocks
// ---------------------------------------------------------------------------

// The 24-block cap is spelled out rather than read from DDA_IPC_MAXBLOCKS, so a changed cap fails here.
TEST(DdaAllReduceIpcBlocksTest, GridFollowsElementSizeAndIsCappedAt24Blocks) {
  EXPECT_EQ(1u, ncclAllReduceDdaIpcBlocks(nullptr, 16, ncclFloat32));
  EXPECT_EQ(2u, ncclAllReduceDdaIpcBlocks(nullptr, 4096, ncclFloat32));
  EXPECT_EQ(1u, ncclAllReduceDdaIpcBlocks(nullptr, 4096, ncclBfloat16));
  EXPECT_EQ(24u, ncclAllReduceDdaIpcBlocks(nullptr, 1u << 20, ncclFloat32));
}

}  // namespace
