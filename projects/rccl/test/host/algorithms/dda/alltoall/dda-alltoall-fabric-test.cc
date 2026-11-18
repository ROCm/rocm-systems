/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/alltoall/dda_alltoall_fabric.cu: eligibility, the scratch check, the
// staging copy ahead of the launch, the grid, and the kernel launched; for the push model also the kernel choice, the
// peer-major geometry, pieces and drain blocks, and the all-peer TDM geometry.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

#include "../common/DdaFabricTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"

#include "fakes/param_redirect.h"

#include DDA_ALLTOALL_FABRIC_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {

using dda::common::FabricGpuBarrier;
using nccl_dda_detail::DdaFabricBarrierState;
using RcclUnitTesting::DdaFabricMockComm;

void* const kSendbuff = reinterpret_cast<void*>(0x1000);
void* const kRecvbuff = reinterpret_cast<void*>(0x2000);
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);

template <int N>
const void* Kernel() {
  return KernelPtr(&dda::common::ddaAllToAllFabric<int8_t, N>);
}

class DdaAlltoallFabricMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    // These tests exercise the pull kernel and its staging copy.
    g_loadParam = ParamOverride({{"RCCL_DDA_A2A_PUSH", 0}});
    g_hipGetLastError = [] { return hipSuccess; };
    RecordDdaCopies(&copies_);
    comm().ddaFabricBarrierState = barrier_;
    comm().rank = 3;
    comm().addonStopEvent = kStopEvent;
  }

  void TearDown() override {
    ResetDdaCollSeams();
    ResetHipFakes();
    ResetNcclFakes();
  }

  ncclComm& comm() { return mock_.comm; }

  bool Eligible(size_t count, ncclDataType_t datatype = ncclFloat32) {
    return ncclAllToAllDdaFabricEligible(mock_.get(), kSendbuff, kRecvbuff, count, datatype);
  }

  ncclResult_t Run(size_t count, ncclDataType_t datatype = ncclFloat32) {
    return ncclAllToAllDdaFabric(kSendbuff, kRecvbuff, count, datatype, mock_.get(), kStream);
  }

  DdaFabricMockComm mock_;
  DdaFabricBarrierState* const barrier_ = SharedBarrierState<DdaFabricBarrierState>(0x5a);
  std::vector<DdaCopy> copies_;
};

TEST_F(DdaAlltoallFabricMicrotest, Eligible_BaselineFp32) {
  EXPECT_TRUE(Eligible(4));
}

TEST_F(DdaAlltoallFabricMicrotest, Eligible_NeedsCommAndBootstrap) {
  EXPECT_FALSE(ncclAllToAllDdaFabricEligible(nullptr, kSendbuff, kRecvbuff, 4, ncclFloat32));
  comm().bootstrap = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAlltoallFabricMicrotest, Eligible_EachMissingFabricResourceRejects) {
  for (int missing = 0; missing < 4; ++missing) {
    mock_.setFabricResourcesPresent(true);
    comm().ddaFabricMemHandler = missing == 0 ? nullptr : comm().ddaFabricMemHandler;
    comm().ddaFabricBarrierState = missing == 1 ? nullptr : barrier_;
    comm().ddaScratch = missing == 2 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev = missing == 3 ? nullptr : comm().ddaPeerPtrsDev;
    EXPECT_FALSE(Eligible(4)) << "missing resource " << missing;
  }
}

TEST_F(DdaAlltoallFabricMicrotest, Eligible_ZeroCountRejects) {
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaAlltoallFabricMicrotest, Eligible_RanksFromTwoTo144) {
  comm().nRanks = 1;
  EXPECT_FALSE(Eligible(4));
  comm().nRanks = 2;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 144;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 145;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAlltoallFabricMicrotest, Eligible_OnlyDatatypesTheDdaGateAdmits) {
  EXPECT_TRUE(Eligible(8, ncclFloat16));
  EXPECT_TRUE(Eligible(8, ncclBfloat16));
  for (ncclDataType_t datatype : {ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclFloat64}) {
    EXPECT_FALSE(Eligible(16, datatype)) << "datatype " << datatype;
  }
  EXPECT_TRUE(Eligible(16, ncclFloat8e4m3));
  EXPECT_TRUE(Eligible(16, ncclFloat8e5m2));
  g_rcclDdaDatatypeSupported = [](const ncclComm*, ncclDataType_t datatype) { return datatype == ncclFloat8e5m2; };
  EXPECT_FALSE(Eligible(8, ncclFloat16));
  EXPECT_FALSE(Eligible(16, ncclFloat8e4m3));
  EXPECT_TRUE(Eligible(16, ncclFloat8e5m2));
}

TEST_F(DdaAlltoallFabricMicrotest, Eligible_PerPeerBytesMustBeMultipleOf16) {
  EXPECT_FALSE(Eligible(2));
  EXPECT_FALSE(Eligible(12, ncclFloat16));
  EXPECT_TRUE(Eligible(16, ncclBfloat16));
}

TEST_F(DdaAlltoallFabricMicrotest, Eligible_WholeSendBufferMustFitScratch) {
  comm().ddaScratchBytes = 8 * 64;
  EXPECT_TRUE(Eligible(16));
  EXPECT_FALSE(Eligible(20));
  EXPECT_TRUE(Eligible(32, ncclFloat16));
}

TEST_F(DdaAlltoallFabricMicrotest, Launch_UnsupportedDatatypeReturnsInvalidArgument) {
  for (ncclDataType_t datatype : {ncclInt8, ncclInt32, ncclUint64, ncclFloat64}) {
    EXPECT_EQ(Run(16, datatype), ncclInvalidArgument) << "datatype " << datatype;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
  EXPECT_TRUE(copies_.empty());
}

TEST_F(DdaAlltoallFabricMicrotest, Launch_EachMissingFabricResourceReturnsInvalidUsage) {
  for (int missing = 0; missing < 4; ++missing) {
    mock_.setFabricResourcesPresent(true);
    comm().ddaFabricMemHandler = missing == 0 ? nullptr : comm().ddaFabricMemHandler;
    comm().ddaScratch = missing == 1 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev = missing == 2 ? nullptr : comm().ddaPeerPtrsDev;
    comm().ddaFabricBarrierState = missing == 3 ? nullptr : barrier_;
    EXPECT_EQ(Run(4), ncclInvalidUsage) << "missing resource " << missing;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
  EXPECT_TRUE(copies_.empty());
}

TEST_F(DdaAlltoallFabricMicrotest, Launch_WholeSendBufferMayFillScratchExactly) {
  comm().ddaScratchBytes = 8 * 64;
  EXPECT_EQ(Run(16), ncclSuccess);
  EXPECT_EQ(Run(20), ncclInvalidArgument);
  EXPECT_EQ(g_ddaLaunches.size(), 1u);
  EXPECT_EQ(copies_.size(), 1u);
}

TEST_F(DdaAlltoallFabricMicrotest, Launch_StagesWholeSendBufferIntoScratchBeforeTheKernel) {
  ASSERT_EQ(Run(8, ncclFloat16), ncclSuccess);
  ASSERT_EQ(copies_.size(), 1u);
  EXPECT_EQ(copies_[0].dst, comm().ddaScratch);
  EXPECT_EQ(copies_[0].src, kSendbuff);
  EXPECT_EQ(copies_[0].bytes, 8u * 16);
  EXPECT_EQ(copies_[0].kind, hipMemcpyDeviceToDevice);
  EXPECT_EQ(copies_[0].stream, kStream);
  EXPECT_EQ(copies_[0].launchesBefore, 0u);
  EXPECT_EQ(g_ddaLaunches.size(), 1u);
}

TEST_F(DdaAlltoallFabricMicrotest, Launch_StagingCopyErrorIsReportedWithoutLaunching) {
  g_hipMemcpyAsync = [](void*, const void*, size_t, hipMemcpyKind, hipStream_t) { return hipErrorInvalidValue; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
  EXPECT_TRUE(g_ddaLaunches.empty());
  EXPECT_EQ(comm().addonStopEvent, kStopEvent);
}

TEST_F(DdaAlltoallFabricMicrotest, Launch_FourAndEightRanksUnrolledOthersRuntime) {
  const std::pair<int, const void*> cases[] = {
      {4, Kernel<4>()}, {8, Kernel<8>()}, {2, Kernel<0>()}, {5, Kernel<0>()}, {144, Kernel<0>()}};
  for (const auto& [nRanks, kernel] : cases) {
    g_ddaLaunches.clear();
    comm().nRanks = nRanks;
    ASSERT_EQ(Run(4), ncclSuccess);
    ASSERT_EQ(g_ddaLaunches.size(), 1u);
    EXPECT_EQ(g_ddaLaunches[0].kernel, kernel) << "nRanks " << nRanks;
    EXPECT_EQ(g_ddaLaunches[0].Arg<int>(4), nRanks);
  }
}

TEST_F(DdaAlltoallFabricMicrotest, Launch_PassesBuffersPerPeerBytesRankAndBarrier) {
  ASSERT_EQ(Run(8, ncclBfloat16), ncclSuccess);
  ASSERT_EQ(g_ddaLaunches.size(), 1u);
  const DdaLaunch& launch = g_ddaLaunches[0];
  EXPECT_EQ(launch.kernel, Kernel<8>());
  EXPECT_EQ(launch.stream, kStream);
  EXPECT_EQ(launch.startEvent, nullptr);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  EXPECT_EQ(comm().addonStopEvent, nullptr);
  EXPECT_EQ(launch.sharedMemBytes, 0u);
  EXPECT_EQ(launch.flags, 0u);
  ASSERT_EQ(launch.args.size(), 6u);
  EXPECT_EQ(launch.Arg<int8_t**>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch.Arg<int8_t*>(1), kRecvbuff);
  EXPECT_EQ(launch.Arg<size_t>(2), 16u);
  EXPECT_EQ(launch.Arg<int>(3), 3);
  EXPECT_TRUE(SameBarrier(launch.Arg<FabricGpuBarrier>(5), barrier_->barrierHost));
}

// Grids spelled out from getGridAndBlockDims over one peer's bytes (16 B per thread, 512 threads, 64-thread warps).
TEST_F(DdaAlltoallFabricMicrotest, Launch_GridFollowsPerPeerBytesAndCapsAtFabricMaxBlocks) {
  struct Case {
    size_t count;
    int maxBlocks;
    unsigned blocks, threads;
  };
  const Case cases[] = {{4, 256, 1, 64}, {512, 256, 1, 128}, {2048, 256, 1, 512}, {2052, 256, 2, 320},
                        {1u << 18, 256, 128, 512}, {1u << 18, 4, 4, 512}};
  for (const Case& c : cases) {
    g_ddaLaunches.clear();
    comm().ddaFabricMaxBlocks = c.maxBlocks;
    ASSERT_EQ(Run(c.count), ncclSuccess);
    ASSERT_EQ(g_ddaLaunches.size(), 1u);
    EXPECT_EQ(g_ddaLaunches[0].grid.x, c.blocks) << "count " << c.count;
    EXPECT_EQ(g_ddaLaunches[0].block.x, c.threads) << "count " << c.count;
    EXPECT_EQ(ncclAllToAllDdaFabricBlocks(mock_.get(), c.count, ncclFloat32), c.blocks);
  }
}

TEST_F(DdaAlltoallFabricMicrotest, Blocks_SizedOnBytesNotElements) {
  EXPECT_EQ(ncclAllToAllDdaFabricBlocks(mock_.get(), 1u << 16, ncclFloat16), 16u);
  EXPECT_EQ(ncclAllToAllDdaFabricBlocks(mock_.get(), 1u << 16, ncclFloat32), 32u);
}

TEST_F(DdaAlltoallFabricMicrotest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
}

// ---------------------------------------------------------------------------
// Push model (RCCL_DDA_A2A_PUSH=1, the default). Kernel choice, in order: peer-major TDM push, all-peer TDM push,
// peer-major vector push, all-peer vector push. The mock comm has 8 ranks on gfx1250 and 256 fabric blocks.
// ---------------------------------------------------------------------------

template <int N>
const void* PushKernel() {
  return KernelPtr(&dda::common::ddaAllToAllFabricPush<int8_t, N>);
}

const void* PeerPushKernel() {
  return KernelPtr(&dda::common::ddaAllToAllFabricPushPeer<int8_t>);
}

const void* TdmPushKernel() {
  return KernelPtr(&dda::common::ddaAllToAllFabricTdmPush<kA2ATdmTileBytes, kA2ATdmUnrollPeers>);
}

const void* TdmPeerPushKernel() {
  return KernelPtr(&dda::common::ddaAllToAllFabricTdmPushPeer<kA2ATdmTileBytes, kA2ATdmUnrollPeers>);
}

constexpr size_t kMiB = size_t{1} << 20;

class DdaAlltoallFabricPushMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    SetParams({});
    g_hipGetLastError = [] { return hipSuccess; };
    // The TDM kernels opt into the full dynamic LDS once; report a device that has it.
    g_hipDeviceGetAttribute = [](int* pi, hipDeviceAttribute_t attr, int) {
      *pi = attr == hipDeviceAttributeSharedMemPerBlockOptin ? dda::common::kDdaA2ATdmMaxSmemBytes : 0;
      return hipSuccess;
    };
    RecordDdaCopies(&copies_);
    comm().ddaFabricBarrierState = barrier_;
    // The 0x5a fill is not a valid bool; start from the default epoch flags.
    barrier_->barrierHost.setEpochFlags(true);
    comm().rank = 3;
    comm().addonStopEvent = kStopEvent;
  }

  void TearDown() override {
    ResetDdaCollSeams();
    ResetHipFakes();
    ResetNcclFakes();
  }

  ncclComm& comm() { return mock_.comm; }

  // Push CVARs on top of their defaults (PUSH=1, PUSH_PEER=1, TDM=1, PUSH_PEER_PIECES=0, PUSH_PEER_DRAIN=-1).
  void SetParams(std::initializer_list<std::pair<const char*, int64_t>> kv) { g_loadParam = ParamOverride(kv); }

  // One alltoall of `bytes` per peer (float32); returns its only launch, or nullptr if it failed or launched more.
  const DdaLaunch* RunBytes(size_t bytes, bool inPlace = false) {
    g_ddaLaunches.clear();
    void* recv = inPlace ? kSendbuff : kRecvbuff;
    if (ncclAllToAllDdaFabric(kSendbuff, recv, bytes / sizeof(float), ncclFloat32, mock_.get(), kStream) !=
            ncclSuccess ||
        g_ddaLaunches.size() != 1) {
      return nullptr;
    }
    return &g_ddaLaunches[0];
  }

  DdaFabricMockComm mock_;
  DdaFabricBarrierState* const barrier_ = SharedBarrierState<DdaFabricBarrierState>(0x5a);
  std::vector<DdaCopy> copies_;
};

#define SKIP_WITHOUT_TDM_TOOLCHAIN() \
  if (!TDM_TOOLCHAIN_AVAILABLE) GTEST_SKIP() << "built without the TDM toolchain"

TEST_F(DdaAlltoallFabricPushMicrotest, Dispatch_DefaultIsPeerMajorTdmWithoutStaging) {
  SKIP_WITHOUT_TDM_TOOLCHAIN();
  const DdaLaunch* launch = RunBytes(kMiB);
  ASSERT_NE(launch, nullptr);
  EXPECT_EQ(launch->kernel, TdmPeerPushKernel());
  EXPECT_TRUE(copies_.empty());
}

TEST_F(DdaAlltoallFabricPushMicrotest, Dispatch_PushPeerOffUsesAllPeerTdm) {
  SKIP_WITHOUT_TDM_TOOLCHAIN();
  SetParams({{"RCCL_DDA_A2A_PUSH_PEER", 0}});
  const DdaLaunch* launch = RunBytes(kMiB);
  ASSERT_NE(launch, nullptr);
  EXPECT_EQ(launch->kernel, TdmPushKernel());
}

TEST_F(DdaAlltoallFabricPushMicrotest, Dispatch_TdmOffOrOtherArchUsesVectorPushes) {
  SetParams({{"RCCL_DDA_A2A_TDM", 0}});
  const DdaLaunch* launch = RunBytes(kMiB);
  ASSERT_NE(launch, nullptr);
  EXPECT_EQ(launch->kernel, PeerPushKernel());

  SetParams({{"RCCL_DDA_A2A_TDM", 0}, {"RCCL_DDA_A2A_PUSH_PEER", 0}});
  launch = RunBytes(kMiB);
  ASSERT_NE(launch, nullptr);
  EXPECT_EQ(launch->kernel, PushKernel<8>());

  SetParams({});
  for (const char* arch : {"gfx942", "gfx950", static_cast<const char*>(nullptr)}) {
    comm().archName = const_cast<char*>(arch);
    launch = RunBytes(kMiB);
    ASSERT_NE(launch, nullptr) << (arch ? arch : "null arch");
    EXPECT_EQ(launch->kernel, PeerPushKernel()) << (arch ? arch : "null arch");
  }
  EXPECT_TRUE(copies_.empty());
}

// In place, peer-major kernels would drain into slots other blocks still read, and the all-peer TDM push can drain a
// tile another block has not pushed yet; the all-peer vector push moves an offset for every peer in one thread.
TEST_F(DdaAlltoallFabricPushMicrotest, Dispatch_InPlaceTakesAllPeerVectorPush) {
  for (int64_t pushPeer : {0, 1}) {
    for (int64_t tdm : {0, 1}) {
      SetParams({{"RCCL_DDA_A2A_PUSH_PEER", pushPeer}, {"RCCL_DDA_A2A_TDM", tdm}});
      const DdaLaunch* launch = RunBytes(kMiB, /*inPlace=*/true);
      ASSERT_NE(launch, nullptr) << "pushPeer " << pushPeer << " tdm " << tdm;
      EXPECT_EQ(launch->kernel, PushKernel<8>()) << "pushPeer " << pushPeer << " tdm " << tdm;
    }
  }
}

TEST_F(DdaAlltoallFabricPushMicrotest, Dispatch_MorePeersThanBlocksFallsBackToAllPeer) {
  comm().ddaFabricMaxBlocks = 4;
  SetParams({{"RCCL_DDA_A2A_TDM", 0}});
  const DdaLaunch* launch = RunBytes(kMiB);
  ASSERT_NE(launch, nullptr);
  EXPECT_EQ(launch->kernel, PushKernel<8>());
  if (!TDM_TOOLCHAIN_AVAILABLE) return;
  SetParams({});
  launch = RunBytes(kMiB);
  ASSERT_NE(launch, nullptr);
  EXPECT_EQ(launch->kernel, TdmPushKernel());
  EXPECT_EQ(launch->grid.x, 4u);
}

// Vector peer-major: one 32-thread wave per 512 B of a peer's chunk, at most 8 waves per block and
// ddaFabricMaxBlocks / nRanks blocks per peer; no drain blocks by default.
TEST_F(DdaAlltoallFabricPushMicrotest, PeerMajorVector_GridFollowsWavesAndCapsAtMaxBlocks) {
  SetParams({{"RCCL_DDA_A2A_TDM", 0}});
  struct Case {
    size_t bytes;
    int maxBlocks;
    int blocksPerPeer;
    unsigned threads;
  };
  const Case cases[] = {{16, 256, 1, 32},   {4096, 256, 1, 256}, {6144, 256, 2, 192},
                        {65536, 256, 16, 256}, {kMiB, 256, 32, 256}, {kMiB, 64, 8, 256}};
  for (const Case& c : cases) {
    comm().ddaFabricMaxBlocks = c.maxBlocks;
    const DdaLaunch* launch = RunBytes(c.bytes);
    ASSERT_NE(launch, nullptr) << "bytes " << c.bytes;
    EXPECT_EQ(launch->kernel, PeerPushKernel());
    EXPECT_EQ(launch->grid.x, static_cast<unsigned>(c.blocksPerPeer * 8)) << "bytes " << c.bytes;
    EXPECT_EQ(launch->block.x, c.threads) << "bytes " << c.bytes;
    EXPECT_EQ(launch->Arg<int>(6), c.blocksPerPeer) << "bytes " << c.bytes;
    EXPECT_EQ(launch->Arg<unsigned>(8), 0u) << "bytes " << c.bytes;
  }
}

// TDM peer-major: one wave per two 8 KiB tiles, up to 16 waves per block. The automatic drain takes half of the 256
// blocks over the 7 other sources (18 each, at most blocksPerPeer), so the push blocks get (256 - 126) / 8 = 16 per
// peer; rounds are balanced so 300 waves take 10 blocks per peer (two rounds) instead of 16.
TEST_F(DdaAlltoallFabricPushMicrotest, PeerMajorTdm_GridBalancesRoundsAndAddsDrainBlocks) {
  SKIP_WITHOUT_TDM_TOOLCHAIN();
  struct Case {
    size_t bytes;
    int blocksPerPeer;
    unsigned threads;
    unsigned drain;
  };
  const Case cases[] = {{16, 1, 32, 1}, {kMiB, 4, 512, 4}, {600 * 8192, 10, 512, 10}, {8 * kMiB, 16, 512, 16}};
  for (const Case& c : cases) {
    const DdaLaunch* launch = RunBytes(c.bytes);
    ASSERT_NE(launch, nullptr) << "bytes " << c.bytes;
    EXPECT_EQ(launch->kernel, TdmPeerPushKernel());
    EXPECT_EQ(launch->grid.x, static_cast<unsigned>(c.blocksPerPeer * 8) + c.drain * 7) << "bytes " << c.bytes;
    EXPECT_EQ(launch->block.x, c.threads) << "bytes " << c.bytes;
    EXPECT_EQ(launch->sharedMemBytes, c.threads / 32 * dda::common::kDdaA2ATdmWindowBytes) << "bytes " << c.bytes;
    EXPECT_EQ(launch->Arg<int>(6), c.blocksPerPeer) << "bytes " << c.bytes;
    EXPECT_EQ(launch->Arg<unsigned>(8), c.drain) << "bytes " << c.bytes;
  }
}

// One piece per MiB of a peer's chunk, at least 1 and at most 16 (vector) or 32 (TDM);
// RCCL_DDA_A2A_PUSH_PEER_PIECES > 0 forces the count, capped at 64.
TEST_F(DdaAlltoallFabricPushMicrotest, PeerMajor_PiecesAutoAndForced) {
  struct Case {
    int64_t tdm, pieces;
    size_t bytes;
    unsigned expected;
  };
  const Case cases[] = {{0, 0, 64 * 1024, 1},  {0, 0, 3 * kMiB, 3},   {0, 0, 64 * kMiB, 16}, {0, -3, 3 * kMiB, 3},
                        {0, 1, 64 * kMiB, 1},  {0, 5, kMiB, 5},       {0, 100, kMiB, 64},    {1, 0, 3 * kMiB, 3},
                        {1, 0, 64 * kMiB, 32}, {1, 100, kMiB, 64}};
  for (const Case& c : cases) {
    if (c.tdm && !TDM_TOOLCHAIN_AVAILABLE) continue;
    SetParams({{"RCCL_DDA_A2A_TDM", c.tdm}, {"RCCL_DDA_A2A_PUSH_PEER_PIECES", c.pieces}});
    const DdaLaunch* launch = RunBytes(c.bytes);
    ASSERT_NE(launch, nullptr) << "tdm " << c.tdm << " pieces " << c.pieces << " bytes " << c.bytes;
    EXPECT_EQ(launch->kernel, c.tdm ? TdmPeerPushKernel() : PeerPushKernel());
    EXPECT_EQ(launch->Arg<unsigned>(7), c.expected)
        << "tdm " << c.tdm << " pieces " << c.pieces << " bytes " << c.bytes;
  }
}

// RCCL_DDA_A2A_PUSH_PEER_DRAIN: D >= 0 asks for D drain blocks per other source (at most 64), taken out of the block
// budget before blocksPerPeer is chosen and then limited to blocksPerPeer; the grid adds drain * (nRanks - 1).
TEST_F(DdaAlltoallFabricPushMicrotest, PeerMajor_DrainAutoAndForced) {
  struct Case {
    int64_t tdm, drainParam;
    int maxBlocks;
    size_t bytes;
    int blocksPerPeer;
    unsigned drain;
  };
  const Case cases[] = {
      {0, -1, 256, kMiB, 32, 0},          // vector default: no drain blocks
      {0, 3, 256, kMiB, 29, 3},           // (256 - 21) / 8 = 29 push blocks per peer
      {1, 0, 256, 8 * kMiB, 32, 0},       // TDM without drain blocks: the whole budget pushes
      {1, 2, 256, 8 * kMiB, 16, 2},       // (256 - 14) / 8 = 30, balanced to 16 for 512 waves
      {1, -1, 256, 16, 1, 1},             // automatic 18, limited to blocksPerPeer
      {1, 100, 1024, 64 * kMiB, 64, 64},  // capped at 64
  };
  auto check = [&](const Case& c) {
    if (c.tdm && !TDM_TOOLCHAIN_AVAILABLE) return;
    SetParams({{"RCCL_DDA_A2A_TDM", c.tdm}, {"RCCL_DDA_A2A_PUSH_PEER_DRAIN", c.drainParam}});
    comm().ddaFabricMaxBlocks = c.maxBlocks;
    const DdaLaunch* launch = RunBytes(c.bytes);
    ASSERT_NE(launch, nullptr) << "tdm " << c.tdm << " drain " << c.drainParam;
    EXPECT_EQ(launch->kernel, c.tdm ? TdmPeerPushKernel() : PeerPushKernel());
    EXPECT_EQ(launch->Arg<int>(6), c.blocksPerPeer) << "tdm " << c.tdm << " drain " << c.drainParam;
    EXPECT_EQ(launch->Arg<unsigned>(8), c.drain) << "tdm " << c.tdm << " drain " << c.drainParam;
    EXPECT_EQ(launch->grid.x, static_cast<unsigned>(c.blocksPerPeer * 8) + c.drain * 7)
        << "tdm " << c.tdm << " drain " << c.drainParam;
  };
  for (const Case& c : cases) check(c);

  // RCCL_DDA_A2A_PUSH_PEER_EPOCH=0 (CAS flags) forces the drain to 0: drain blocks would take the handshake tokens.
  barrier_->barrierHost.setEpochFlags(false);
  check({0, 3, 256, kMiB, 32, 0});
  check({1, 3, 256, 8 * kMiB, 32, 0});
}

// All-peer TDM: one wave per two tiles over all peers, at most 16 waves per block and ddaFabricMaxBlocks blocks;
// 16 KiB of LDS per wave.
TEST_F(DdaAlltoallFabricPushMicrotest, AllPeerTdm_GridBlockAndSmem) {
  SKIP_WITHOUT_TDM_TOOLCHAIN();
  SetParams({{"RCCL_DDA_A2A_PUSH_PEER", 0}});
  struct Case {
    size_t bytes;
    int maxBlocks;
    unsigned blocks, threads;
  };
  const Case cases[] = {{16, 256, 1, 128}, {kMiB, 256, 32, 512}, {64 * kMiB, 256, 256, 512}, {kMiB, 8, 8, 512}};
  for (const Case& c : cases) {
    comm().ddaFabricMaxBlocks = c.maxBlocks;
    const DdaLaunch* launch = RunBytes(c.bytes);
    ASSERT_NE(launch, nullptr) << "bytes " << c.bytes;
    EXPECT_EQ(launch->kernel, TdmPushKernel());
    EXPECT_EQ(launch->grid.x, c.blocks) << "bytes " << c.bytes;
    EXPECT_EQ(launch->block.x, c.threads) << "bytes " << c.bytes;
    EXPECT_EQ(launch->sharedMemBytes, c.threads / 32 * dda::common::kDdaA2ATdmWindowBytes) << "bytes " << c.bytes;
  }
}

TEST_F(DdaAlltoallFabricPushMicrotest, PeerTdmLaunch_PassesBuffersRankPiecesAndBarrier) {
  SKIP_WITHOUT_TDM_TOOLCHAIN();
  const DdaLaunch* launch = RunBytes(3 * kMiB);
  ASSERT_NE(launch, nullptr);
  EXPECT_EQ(launch->stream, kStream);
  EXPECT_EQ(launch->startEvent, nullptr);
  EXPECT_EQ(launch->stopEvent, kStopEvent);
  EXPECT_EQ(comm().addonStopEvent, nullptr);
  EXPECT_EQ(launch->flags, 0u);
  ASSERT_EQ(launch->args.size(), 10u);
  EXPECT_EQ(launch->Arg<uint8_t* const*>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch->Arg<uint8_t*>(1), kRecvbuff);
  EXPECT_EQ(launch->Arg<const uint8_t*>(2), kSendbuff);
  EXPECT_EQ(launch->Arg<size_t>(3), 3 * kMiB);
  EXPECT_EQ(launch->Arg<int>(4), 3);
  EXPECT_EQ(launch->Arg<int>(5), 8);
  EXPECT_EQ(launch->Arg<unsigned>(7), 3u);
  EXPECT_TRUE(SameBarrier(launch->Arg<FabricGpuBarrier>(9), barrier_->barrierHost));
}

TEST_F(DdaAlltoallFabricPushMicrotest, AllPeerTdmLaunch_PassesBuffersRankAndBarrier) {
  SKIP_WITHOUT_TDM_TOOLCHAIN();
  SetParams({{"RCCL_DDA_A2A_PUSH_PEER", 0}});
  const DdaLaunch* launch = RunBytes(kMiB);
  ASSERT_NE(launch, nullptr);
  EXPECT_EQ(launch->stopEvent, kStopEvent);
  EXPECT_EQ(comm().addonStopEvent, nullptr);
  ASSERT_EQ(launch->args.size(), 7u);
  EXPECT_EQ(launch->Arg<uint8_t* const*>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch->Arg<uint8_t*>(1), kRecvbuff);
  EXPECT_EQ(launch->Arg<const uint8_t*>(2), kSendbuff);
  EXPECT_EQ(launch->Arg<size_t>(3), kMiB);
  EXPECT_EQ(launch->Arg<int>(4), 3);
  EXPECT_EQ(launch->Arg<int>(5), 8);
  EXPECT_TRUE(SameBarrier(launch->Arg<FabricGpuBarrier>(6), barrier_->barrierHost));
}

TEST_F(DdaAlltoallFabricPushMicrotest, PeerVectorLaunch_PassesBuffersRankPiecesAndBarrier) {
  SetParams({{"RCCL_DDA_A2A_TDM", 0}});
  const DdaLaunch* launch = RunBytes(3 * kMiB);
  ASSERT_NE(launch, nullptr);
  EXPECT_EQ(launch->stopEvent, kStopEvent);
  EXPECT_EQ(comm().addonStopEvent, nullptr);
  EXPECT_EQ(launch->sharedMemBytes, 0u);
  ASSERT_EQ(launch->args.size(), 10u);
  EXPECT_EQ(launch->Arg<int8_t**>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch->Arg<int8_t*>(1), kRecvbuff);
  EXPECT_EQ(launch->Arg<size_t>(2), 3 * kMiB);
  EXPECT_EQ(launch->Arg<const int8_t*>(3), kSendbuff);
  EXPECT_EQ(launch->Arg<int>(4), 3);
  EXPECT_EQ(launch->Arg<int>(5), 8);
  EXPECT_EQ(launch->Arg<unsigned>(7), 3u);
  EXPECT_TRUE(SameBarrier(launch->Arg<FabricGpuBarrier>(9), barrier_->barrierHost));
}

TEST_F(DdaAlltoallFabricPushMicrotest, AllPeerVectorLaunch_UnrollsFourAndEightRanksAndPassesBuffers) {
  SetParams({{"RCCL_DDA_A2A_TDM", 0}, {"RCCL_DDA_A2A_PUSH_PEER", 0}});
  const std::pair<int, const void*> cases[] = {
      {4, PushKernel<4>()}, {8, PushKernel<8>()}, {2, PushKernel<0>()}, {5, PushKernel<0>()}, {144, PushKernel<0>()}};
  for (const auto& [nRanks, kernel] : cases) {
    comm().nRanks = nRanks;
    comm().addonStopEvent = kStopEvent;
    const DdaLaunch* launch = RunBytes(4096);
    ASSERT_NE(launch, nullptr) << "nRanks " << nRanks;
    EXPECT_EQ(launch->kernel, kernel) << "nRanks " << nRanks;
    EXPECT_EQ(launch->stopEvent, kStopEvent) << "nRanks " << nRanks;
    ASSERT_EQ(launch->args.size(), 7u);
    EXPECT_EQ(launch->Arg<int8_t**>(0), comm().ddaPeerPtrsDev);
    EXPECT_EQ(launch->Arg<int8_t*>(1), kRecvbuff);
    EXPECT_EQ(launch->Arg<size_t>(2), 4096u);
    EXPECT_EQ(launch->Arg<const int8_t*>(3), kSendbuff);
    EXPECT_EQ(launch->Arg<int>(4), 3);
    EXPECT_EQ(launch->Arg<int>(5), nRanks);
    EXPECT_TRUE(SameBarrier(launch->Arg<FabricGpuBarrier>(6), barrier_->barrierHost));
  }
}

TEST_F(DdaAlltoallFabricPushMicrotest, PushLaunch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  for (int64_t pushPeer : {0, 1}) {
    for (int64_t tdm : {0, 1}) {
      if (tdm && !TDM_TOOLCHAIN_AVAILABLE) continue;
      SetParams({{"RCCL_DDA_A2A_PUSH_PEER", pushPeer}, {"RCCL_DDA_A2A_TDM", tdm}});
      g_ddaLaunches.clear();
      EXPECT_EQ(ncclAllToAllDdaFabric(kSendbuff, kRecvbuff, 1024, ncclFloat32, mock_.get(), kStream),
                ncclUnhandledCudaError)
          << "pushPeer " << pushPeer << " tdm " << tdm;
      EXPECT_EQ(g_ddaLaunches.size(), 1u) << "pushPeer " << pushPeer << " tdm " << tdm;
    }
  }
}

#undef SKIP_WITHOUT_TDM_TOOLCHAIN

}  // namespace
