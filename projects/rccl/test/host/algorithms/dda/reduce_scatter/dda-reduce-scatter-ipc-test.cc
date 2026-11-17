/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/reduce_scatter/dda_reduce_scatter_ipc.cu: eligibility, the scratch
// check, in-kernel versus host staging, the datatype dispatch, the 24-block grid, and the launch.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../common/DdaIpcTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"

#include DDA_REDUCE_SCATTER_IPC_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {

using dda::common::IpcGpuBarrier;
using nccl_dda_detail::DdaIpcBarrierState;
using RcclUnitTesting::DdaIpcMockComm;

void* const kSendbuff = reinterpret_cast<void*>(0x1000);
void* const kRecvbuff = reinterpret_cast<void*>(0x2000);
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);

// Shards up to 1 MiB are copied in the kernel; larger ones are staged by the host first.
constexpr size_t kInKernelMaxFloats = (1u << 20) / sizeof(float);

template <typename T, bool kCopyInKernel = true>
const void* Kernel() {
  return KernelPtr(&dda::common::ddaReduceScatterIpc<T, 8, false, kCopyInKernel>);
}

class DdaReduceScatterIpcMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hipGetLastError = [] { return hipSuccess; };
    RecordDdaCopies(&copies_);
    comm().ddaIpcBarrierState = barrier_;
    comm().rank = 2;
    comm().addonStopEvent = kStopEvent;
  }

  void TearDown() override {
    ResetDdaCollSeams();
    ResetHipFakes();
  }

  ncclComm& comm() { return mock_.comm; }

  bool Eligible(size_t recvcount, ncclDataType_t datatype = ncclFloat32, ncclRedOp_t op = ncclSum) {
    return ncclReduceScatterDdaIpcEligible(mock_.get(), kSendbuff, kRecvbuff, recvcount, datatype, op);
  }

  ncclResult_t Run(size_t recvcount, ncclDataType_t datatype = ncclFloat32) {
    return ncclReduceScatterDdaIpc(kSendbuff, kRecvbuff, recvcount, datatype, ncclSum, mock_.get(), kStream);
  }

  template <typename T>
  void ExpectLaunchOf(const DdaLaunch& launch, size_t recvcount) {
    EXPECT_EQ(launch.kernel, Kernel<T>());
    ASSERT_EQ(launch.args.size(), 6u);
    EXPECT_EQ(launch.Arg<T**>(0), comm().ddaPeerPtrsDev);
    EXPECT_EQ(launch.Arg<T*>(1), kRecvbuff);
    EXPECT_EQ(launch.Arg<size_t>(2), recvcount);
    EXPECT_EQ(launch.Arg<const T*>(3), kSendbuff);
    EXPECT_EQ(launch.Arg<int>(4), 2);
    EXPECT_TRUE(SameBarrier(launch.Arg<IpcGpuBarrier>(5), barrier_->barrierHost));
  }

  DdaIpcMockComm mock_;
  DdaIpcBarrierState* const barrier_ = SharedBarrierState<DdaIpcBarrierState>(0xa5);
  std::vector<DdaCopy> copies_;
};

TEST_F(DdaReduceScatterIpcMicrotest, Eligible_BaselineFp32Sum) {
  EXPECT_TRUE(Eligible(4));
}

TEST_F(DdaReduceScatterIpcMicrotest, Eligible_NeedsCommAndBootstrap) {
  EXPECT_FALSE(ncclReduceScatterDdaIpcEligible(nullptr, kSendbuff, kRecvbuff, 4, ncclFloat32, ncclSum));
  comm().bootstrap = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaReduceScatterIpcMicrotest, Eligible_EachMissingIpcResourceRejects) {
  for (int missing = 0; missing < 4; ++missing) {
    mock_.setIpcResourcesPresent(true);
    comm().ddaIpcMemHandler = missing == 0 ? nullptr : comm().ddaIpcMemHandler;
    comm().ddaScratch = missing == 1 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev = missing == 2 ? nullptr : comm().ddaPeerPtrsDev;
    comm().ddaIpcBarrierState = missing == 3 ? nullptr : barrier_;
    EXPECT_FALSE(Eligible(4)) << "missing resource " << missing;
  }
}

TEST_F(DdaReduceScatterIpcMicrotest, Eligible_ZeroCountRejects) {
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaReduceScatterIpcMicrotest, Eligible_SingleNodeOfExactlyEightRanks) {
  comm().nNodes = 2;
  EXPECT_FALSE(Eligible(4));
  comm().nNodes = 1;
  comm().nRanks = 7;
  EXPECT_FALSE(Eligible(4));
  comm().nRanks = 9;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaReduceScatterIpcMicrotest, Eligible_OnlySum) {
  for (ncclRedOp_t op : {ncclProd, ncclMax, ncclMin, ncclAvg}) {
    EXPECT_FALSE(Eligible(4, ncclFloat32, op)) << "op " << op;
  }
}

TEST_F(DdaReduceScatterIpcMicrotest, Eligible_OnlyFloat32Float16Bfloat16) {
  EXPECT_TRUE(Eligible(8, ncclFloat16));
  EXPECT_TRUE(Eligible(8, ncclBfloat16));
  for (ncclDataType_t datatype : {ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclFloat64}) {
    EXPECT_FALSE(Eligible(16, datatype)) << "datatype " << datatype;
  }
}

TEST_F(DdaReduceScatterIpcMicrotest, Eligible_PerRankBytesMustBeMultipleOf16) {
  EXPECT_FALSE(Eligible(2));
  EXPECT_FALSE(Eligible(12, ncclFloat16));
  EXPECT_TRUE(Eligible(16, ncclBfloat16));
}

TEST_F(DdaReduceScatterIpcMicrotest, Eligible_BuffersMustBe16ByteAligned) {
  void* const misaligned = reinterpret_cast<void*>(0x1008);
  EXPECT_FALSE(ncclReduceScatterDdaIpcEligible(mock_.get(), misaligned, kRecvbuff, 4, ncclFloat32, ncclSum));
  EXPECT_FALSE(ncclReduceScatterDdaIpcEligible(mock_.get(), kSendbuff, misaligned, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaReduceScatterIpcMicrotest, Eligible_WholeSendBufferMustFitScratch) {
  comm().ddaScratchBytes = 8 * 64;
  EXPECT_TRUE(Eligible(16));
  EXPECT_FALSE(Eligible(20));
  EXPECT_TRUE(Eligible(32, ncclFloat16));
}

TEST_F(DdaReduceScatterIpcMicrotest, Launch_UnsupportedDatatypeReturnsInvalidArgument) {
  for (ncclDataType_t datatype : {ncclInt8, ncclInt32, ncclUint64, ncclFloat64}) {
    EXPECT_EQ(Run(16, datatype), ncclInvalidArgument) << "datatype " << datatype;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
  EXPECT_TRUE(copies_.empty());
}

TEST_F(DdaReduceScatterIpcMicrotest, Launch_EachMissingIpcResourceReturnsInvalidUsage) {
  for (int missing = 0; missing < 4; ++missing) {
    mock_.setIpcResourcesPresent(true);
    comm().ddaIpcMemHandler = missing == 0 ? nullptr : comm().ddaIpcMemHandler;
    comm().ddaScratch = missing == 1 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev = missing == 2 ? nullptr : comm().ddaPeerPtrsDev;
    comm().ddaIpcBarrierState = missing == 3 ? nullptr : barrier_;
    EXPECT_EQ(Run(4), ncclInvalidUsage) << "missing resource " << missing;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
  EXPECT_TRUE(copies_.empty());
}

TEST_F(DdaReduceScatterIpcMicrotest, Launch_ScratchCheckUsesTheDispatchedElementSize) {
  comm().ddaScratchBytes = 8 * 64;
  EXPECT_EQ(Run(32, ncclFloat16), ncclSuccess);
  EXPECT_EQ(Run(17, ncclFloat32), ncclInvalidArgument);
  EXPECT_EQ(Run(33, ncclBfloat16), ncclInvalidArgument);
  EXPECT_EQ(g_ddaLaunches.size(), 1u);
}

TEST_F(DdaReduceScatterIpcMicrotest, Launch_EachDatatypeLaunchesItsOwnKernel) {
  ASSERT_EQ(Run(16, ncclFloat32), ncclSuccess);
  ASSERT_EQ(Run(16, ncclFloat16), ncclSuccess);
  ASSERT_EQ(Run(16, ncclBfloat16), ncclSuccess);
  ASSERT_EQ(g_ddaLaunches.size(), 3u);
  ExpectLaunchOf<float>(g_ddaLaunches[0], 16);
  ExpectLaunchOf<half>(g_ddaLaunches[1], 16);
  ExpectLaunchOf<bf16>(g_ddaLaunches[2], 16);
}

TEST_F(DdaReduceScatterIpcMicrotest, Launch_ShardUpTo1MiBIsCopiedInTheKernel) {
  ASSERT_EQ(Run(kInKernelMaxFloats), ncclSuccess);
  EXPECT_TRUE(copies_.empty());
  ASSERT_EQ(g_ddaLaunches.size(), 1u);
  EXPECT_EQ(g_ddaLaunches[0].kernel, (Kernel<float, true>()));
}

TEST_F(DdaReduceScatterIpcMicrotest, Launch_LargerShardIsStagedIntoScratchBeforeTheKernel) {
  ASSERT_EQ(Run(kInKernelMaxFloats + 4), ncclSuccess);
  ASSERT_EQ(copies_.size(), 1u);
  EXPECT_EQ(copies_[0].dst, comm().ddaScratch);
  EXPECT_EQ(copies_[0].src, kSendbuff);
  EXPECT_EQ(copies_[0].bytes, 8u * (kInKernelMaxFloats + 4) * sizeof(float));
  EXPECT_EQ(copies_[0].kind, hipMemcpyDeviceToDevice);
  EXPECT_EQ(copies_[0].stream, kStream);
  EXPECT_EQ(copies_[0].launchesBefore, 0u);
  ASSERT_EQ(g_ddaLaunches.size(), 1u);
  const DdaLaunch& launch = g_ddaLaunches[0];
  EXPECT_EQ(launch.kernel, (Kernel<float, false>()));
  EXPECT_EQ(launch.stream, kStream);
  EXPECT_EQ(launch.startEvent, nullptr);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  EXPECT_EQ(comm().addonStopEvent, nullptr);
  EXPECT_EQ(launch.sharedMemBytes, 0u);
  EXPECT_EQ(launch.flags, 0u);
}

// The stop event is taken before the staging copy, so a failed copy still consumes it.
TEST_F(DdaReduceScatterIpcMicrotest, Launch_StagingCopyErrorIsReportedWithoutLaunching) {
  g_hipMemcpyAsync = [](void*, const void*, size_t, hipMemcpyKind, hipStream_t) { return hipErrorInvalidValue; };
  EXPECT_EQ(Run(kInKernelMaxFloats + 4), ncclUnhandledCudaError);
  EXPECT_TRUE(g_ddaLaunches.empty());
  EXPECT_EQ(comm().addonStopEvent, nullptr);
}

// Sized on this rank's shard in elements; the 24-block cap is spelled out rather than read from DDA_IPC_MAXBLOCKS.
TEST_F(DdaReduceScatterIpcMicrotest, Launch_GridFollowsShardAndCapsAt24) {
  struct Case {
    size_t count;
    ncclDataType_t datatype;
    unsigned blocks, threads;
  };
  const Case cases[] = {{4, ncclFloat32, 1, 64},         {2048, ncclFloat32, 1, 512},
                        {2052, ncclFloat32, 2, 320},     {4096, ncclFloat16, 1, 512},
                        {1u << 14, ncclFloat32, 8, 512}, {1u << 16, ncclFloat32, 24, 512}};
  for (const Case& c : cases) {
    g_ddaLaunches.clear();
    ASSERT_EQ(Run(c.count, c.datatype), ncclSuccess);
    ASSERT_EQ(g_ddaLaunches.size(), 1u);
    EXPECT_EQ(g_ddaLaunches[0].grid.x, c.blocks) << "count " << c.count;
    EXPECT_EQ(g_ddaLaunches[0].block.x, c.threads) << "count " << c.count;
  }
}

TEST_F(DdaReduceScatterIpcMicrotest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
}

}  // namespace
