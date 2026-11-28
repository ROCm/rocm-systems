/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric_ll128.cu: eligibility against
// the scratch bank's per-rank slot and the LL128 cap, the datatype dispatch, the peer-parallel and batched split, each
// kernel's slice grid and its caps, and the launch.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../common/DdaFabricTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll.h"

#include "fakes/param_redirect.h"

#include DDA_REDUCE_SCATTER_FABRIC_LL128_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {

using RcclUnitTesting::DdaFabricMockComm;

void* const kSendbuff = reinterpret_cast<void*>(0x1000);
void* const kRecvbuff = reinterpret_cast<void*>(0x2000);
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);
// A slice is 16 lines on the wire and carries 1920 bytes, 480 floats.
constexpr size_t kSliceFloats = 480;
// Scratch holding exactly one slice per rank at 8 ranks: two banks of 8 x 2 KiB.
constexpr size_t kOneSliceScratch8 = 2 * 8 * 2048;

template <typename T, int N>
const void* Kernel() {
  return KernelPtr(&dda::common::ddaReduceScatterFabricLL128<T, N>);
}

template <typename T, int N>
const void* PeerParKernel() {
  return KernelPtr(&dda::common::ddaReduceScatterFabricLL128PeerPar<T, N>);
}

class DdaReduceScatterFabricLL128Microtest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hipGetLastError = [] { return hipSuccess; };
    g_loadParam = [this](const char* env, int64_t deftVal) {
      auto it = params_.find(env);
      return it == params_.end() ? deftVal : it->second;
    };
    // The LL128 tier is off without an arch table; give it a cap well above anything tested.
    g_rcclDdaLL128Threshold = [](const ncclComm*, ncclFunc_t) { return size_t{1} << 30; };
    comm().rank = 7;
    comm().addonStopEvent = kStopEvent;
  }

  void TearDown() override {
    ResetDdaCollSeams();
    ResetHipFakes();
    ResetNcclFakes();
  }

  ncclComm& comm() { return mock_.comm; }

  // The ReduceScatter LL cap the launcher sees.
  void SetLLThreshold(size_t cap) {
    g_rcclDdaLLThreshold = [cap](const ncclComm*, ncclFunc_t func) {
      EXPECT_EQ(func, ncclFuncReduceScatter);
      return cap;
    };
  }

  // The launch-shape entries of the gfx1250 SPX table: 256-thread batched blocks, LL on at its 1 MiB cap.
  void UseShapeTable() {
    table_.rankScaledMinRanks = 8;
    table_.rsDdaLL128NarrowMaxPerRank = size_t{1} << 20;
    comm().archThresholds = &table_;
    g_rcclDdaFabricRsLL128Threads = [](const ncclComm*) { return 256; };
    SetLLThreshold(size_t{1} << 20);
  }

  // The (grid, block) of one launch.
  std::pair<unsigned, unsigned> GridBlock(size_t recvcount, ncclDataType_t datatype = ncclFloat32) {
    const DdaLaunch& launch = RunOne(recvcount, datatype);
    return {launch.grid.x, launch.block.x};
  }

  bool Eligible(size_t recvcount, ncclDataType_t datatype = ncclFloat32, ncclRedOp_t op = ncclSum) {
    return ncclReduceScatterDdaFabricLL128Eligible(mock_.get(), kSendbuff, kRecvbuff, recvcount, datatype, op);
  }

  ncclResult_t Run(size_t recvcount, ncclDataType_t datatype = ncclFloat32) {
    return ncclReduceScatterDdaFabricLL128(kSendbuff, kRecvbuff, recvcount, datatype, ncclSum, mock_.get(), kStream);
  }

  const DdaLaunch& RunOne(size_t recvcount, ncclDataType_t datatype = ncclFloat32) {
    g_ddaLaunches.clear();
    EXPECT_EQ(Run(recvcount, datatype), ncclSuccess);
    EXPECT_EQ(g_ddaLaunches.size(), 1u);
    return g_ddaLaunches.at(0);
  }

  // Messages up to `bytes` take the peer-parallel kernel, as an arch table cap would send them.
  void PeerParallelUpTo(size_t bytes) {
    g_rcclDdaFabricRsLL128PeerMaxBytes = [bytes](const ncclComm*) { return bytes; };
  }

  template <typename T>
  void ExpectLaunchOf(const DdaLaunch& launch, size_t recvcount, const void* kernel = Kernel<T, 8>()) {
    EXPECT_EQ(launch.kernel, kernel);
    ASSERT_EQ(launch.args.size(), 11u);
    EXPECT_EQ(launch.Arg<T**>(0), comm().ddaPeerPtrsDev);
    EXPECT_EQ(launch.Arg<T*>(1), kRecvbuff);
    EXPECT_EQ(launch.Arg<const T*>(2), kSendbuff);
    EXPECT_EQ(launch.Arg<size_t>(3), recvcount * sizeof(T));
    EXPECT_EQ(launch.Arg<int>(4), 7);
    EXPECT_EQ(launch.Arg<int>(5), 8);
    EXPECT_EQ(launch.Arg<uint32_t*>(6), comm().ddaLLEpochDev);
    EXPECT_EQ(launch.Arg<int>(7), comm().ddaLLEpochLen);
    EXPECT_EQ(launch.Arg<size_t>(8), (recvcount * sizeof(T) + 1919) / 1920);
    EXPECT_EQ(launch.Arg<size_t>(9), (comm().ddaScratchBytes / 2) / 128 * 128);
    EXPECT_EQ(launch.Arg<dda::common::DdaKernelParams>(10).flag, comm().abortFlagDev);
  }

  DdaFabricMockComm mock_;
  std::map<std::string, int64_t> params_;
  rcclArchThresholds table_{};
};

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_BaselineFp32Sum) {
  EXPECT_TRUE(Eligible(4));
}

// LL128 is gated by RCCL_DDA_LL128 alone; RCCL_DDA_LL only controls the LL tier.
TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_DdaLLZeroDoesNotGateLL128) {
  params_["RCCL_DDA_LL"] = 0;
  EXPECT_TRUE(Eligible(4));
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_NeedsCommAndBootstrap) {
  EXPECT_FALSE(ncclReduceScatterDdaFabricLL128Eligible(nullptr, kSendbuff, kRecvbuff, 4, ncclFloat32, ncclSum));
  comm().bootstrap = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_NeedsHandlerScratchAndPeersButNotBarrier) {
  comm().ddaFabricBarrierState = nullptr;
  EXPECT_TRUE(Eligible(4));
  for (int missing = 0; missing < 3; ++missing) {
    mock_.setFabricResourcesPresent(true);
    comm().ddaFabricMemHandler = missing == 0 ? nullptr : comm().ddaFabricMemHandler;
    comm().ddaScratch = missing == 1 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev = missing == 2 ? nullptr : comm().ddaPeerPtrsDev;
    EXPECT_FALSE(Eligible(4)) << "missing resource " << missing;
  }
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_NeedsTheEpochArray) {
  comm().ddaLLEpochLen = 0;
  EXPECT_FALSE(Eligible(4));
  comm().ddaLLEpochLen = 1;
  EXPECT_TRUE(Eligible(4));
  comm().ddaLLEpochDev = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_ZeroCountRejects) {
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_RanksFromTwoTo144) {
  comm().nRanks = 1;
  EXPECT_FALSE(Eligible(4));
  comm().nRanks = 2;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 144;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 145;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_OnlySum) {
  for (ncclRedOp_t op : {ncclProd, ncclMax, ncclMin, ncclAvg}) {
    EXPECT_FALSE(Eligible(4, ncclFloat32, op)) << "op " << op;
  }
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_OnlyDatatypesTheDdaGateAdmits) {
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

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_ShardBytesMultipleOf16) {
  EXPECT_TRUE(Eligible(8, ncclFloat16));
  EXPECT_FALSE(Eligible(2));
  EXPECT_FALSE(Eligible(6));
  EXPECT_FALSE(Eligible(12, ncclBfloat16));
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_BuffersMustBe16ByteAligned) {
  void* const misaligned = reinterpret_cast<void*>(0x1008);
  EXPECT_FALSE(ncclReduceScatterDdaFabricLL128Eligible(mock_.get(), misaligned, kRecvbuff, 4, ncclFloat32, ncclSum));
  EXPECT_FALSE(ncclReduceScatterDdaFabricLL128Eligible(mock_.get(), kSendbuff, misaligned, 4, ncclFloat32, ncclSum));
}

// The cap is checked against the whole message, nRanks shards, as the selector does.
TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_WholeMessageWithinTheLL128Cap) {
  size_t cap = 8 * 16;
  g_rcclDdaLL128Threshold = [&](const ncclComm* c, ncclFunc_t func) {
    EXPECT_EQ(c, mock_.get());
    EXPECT_EQ(func, ncclFuncReduceScatter);
    return cap;
  };
  EXPECT_TRUE(Eligible(4));
  cap = 8 * 16 - 1;
  EXPECT_FALSE(Eligible(4));
  cap = 0;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_RankSlotMustHoldEverySlice) {
  comm().ddaScratchBytes = kOneSliceScratch8;
  EXPECT_TRUE(Eligible(kSliceFloats));
  EXPECT_FALSE(Eligible(kSliceFloats + 4));
  comm().ddaScratchBytes = kOneSliceScratch8 - 1;
  EXPECT_FALSE(Eligible(4));
  comm().ddaScratchBytes = 2 * kOneSliceScratch8;
  EXPECT_TRUE(Eligible(kSliceFloats + 4));
  comm().nRanks = 4;
  comm().ddaScratchBytes = kOneSliceScratch8 / 2;
  EXPECT_TRUE(Eligible(4));
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_UnsupportedDatatypeReturnsInvalidArgument) {
  for (ncclDataType_t datatype : {ncclInt8, ncclInt32, ncclUint64, ncclFloat64}) {
    EXPECT_EQ(Run(16, datatype), ncclInvalidArgument) << "datatype " << datatype;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_EachDatatypeLaunchesItsOwnKernel) {
  ExpectLaunchOf<float>(RunOne(16, ncclFloat32), 16);
  ExpectLaunchOf<half>(RunOne(16, ncclFloat16), 16);
  ExpectLaunchOf<bf16>(RunOne(16, ncclBfloat16), 16);
  ExpectLaunchOf<dda::common::fp8e4m3>(RunOne(16, ncclFloat8e4m3), 16);
  ExpectLaunchOf<dda::common::fp8e5m2>(RunOne(16, ncclFloat8e5m2), 16);
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_TakesTheStopEventOnTheStream) {
  const DdaLaunch& launch = RunOne(4);
  EXPECT_EQ(launch.block.x, 512u);
  EXPECT_EQ(launch.stream, kStream);
  EXPECT_EQ(launch.startEvent, nullptr);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  EXPECT_EQ(comm().addonStopEvent, nullptr);
  EXPECT_EQ(launch.sharedMemBytes, 0u);
  EXPECT_EQ(launch.flags, 0u);
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_FourAndEightRanksUnrolledOthersRuntime) {
  const std::pair<int, const void*> cases[] = {{4, Kernel<float, 4>()},
                                               {8, Kernel<float, 8>()},
                                               {2, Kernel<float, 0>()},
                                               {5, Kernel<float, 0>()},
                                               {144, Kernel<float, 0>()}};
  for (const auto& [nRanks, kernel] : cases) {
    comm().nRanks = nRanks;
    const DdaLaunch& launch = RunOne(4);
    EXPECT_EQ(launch.kernel, kernel) << "nRanks " << nRanks;
    EXPECT_EQ(launch.Arg<int>(5), nRanks);
  }
}

// 512 threads are 16 warps, one slice each: a block per 16 slices of the shard.
TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_GridFollowsShardSlicesWithinCaps) {
  EXPECT_EQ(RunOne(8, ncclFloat16).grid.x, 1u);
  EXPECT_EQ(RunOne(16 * kSliceFloats).grid.x, 1u);
  EXPECT_EQ(RunOne(16 * kSliceFloats + 4).grid.x, 2u);
  EXPECT_EQ(RunOne(32 * kSliceFloats, ncclBfloat16).grid.x, 1u);
  EXPECT_EQ(RunOne(16 * 100 * kSliceFloats).grid.x, 100u);
  comm().ddaFabricMaxBlocks = 6;
  EXPECT_EQ(RunOne(16 * 100 * kSliceFloats).grid.x, 6u);
  comm().ddaFabricMaxBlocks = 0;
  EXPECT_EQ(RunOne(16 * 100 * kSliceFloats).grid.x, 1u);
  comm().ddaFabricMaxBlocks = 256;
  comm().ddaLLEpochLen = 9;
  EXPECT_EQ(RunOne(16 * 100 * kSliceFloats).grid.x, 9u);
  comm().ddaLLEpochLen = 0;
  EXPECT_EQ(RunOne(16 * 100 * kSliceFloats).grid.x, 1u);
}

// From the table's rankScaledMinRanks, a batched shard up to rsDdaLL128NarrowMaxPerRank starts at 128-thread blocks
// (4 warps) and widens only once ddaFabricMaxBlocks blocks are full, up to the batched block size. LL settings do not
// change it. Larger shards, smaller comms, the peer-parallel kernel and no table keep that block size.
TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_ShapeTableStartsSmallShardsNarrow) {
  using Shape = std::pair<unsigned, unsigned>;
  constexpr size_t kMiBFloats = (size_t{1} << 20) / sizeof(float);
  UseShapeTable();
  ASSERT_EQ(comm().nRanks, 8);
  ASSERT_EQ(comm().ddaFabricMaxBlocks, 256);
  EXPECT_EQ(GridBlock(8, ncclFloat16), Shape(1u, 128u));
  EXPECT_EQ(GridBlock(4 * kSliceFloats), Shape(1u, 128u));
  EXPECT_EQ(GridBlock(4 * kSliceFloats + 4), Shape(2u, 128u));
  EXPECT_EQ(GridBlock(100 * kSliceFloats), Shape(25u, 128u));
  EXPECT_EQ(GridBlock(kMiBFloats), Shape(137u, 128u)) << "a 1 MiB shard, 547 slices";
  EXPECT_EQ(GridBlock(547 * kSliceFloats), Shape(69u, 256u)) << "past 1 MiB: the batched 256 threads";
  comm().ddaFabricMaxBlocks = 6;
  EXPECT_EQ(GridBlock(24 * kSliceFloats), Shape(6u, 128u));
  EXPECT_EQ(GridBlock(25 * kSliceFloats), Shape(4u, 256u));
  EXPECT_EQ(GridBlock(49 * kSliceFloats), Shape(6u, 256u)) << "widens no further than the batched block size";
  g_rcclDdaFabricRsLL128Threads = [](const ncclComm*) { return 0; };
  EXPECT_EQ(GridBlock(49 * kSliceFloats), Shape(4u, 512u)) << "a 512-thread batched block widens to 512";
  g_rcclDdaFabricRsLL128Threads = [](const ncclComm*) { return 256; };
  comm().ddaFabricMaxBlocks = 256;
  comm().nRanks = 64;
  EXPECT_EQ(GridBlock(kMiBFloats), Shape(137u, 128u));
  comm().nRanks = 8;
  // LL does not run these shards, so its settings leave the shape alone.
  SetLLThreshold(0);
  EXPECT_EQ(GridBlock(kMiBFloats), Shape(137u, 128u)) << "LL off";
  SetLLThreshold(size_t{512} << 10);
  EXPECT_EQ(GridBlock(kMiBFloats), Shape(137u, 128u)) << "LL lowered";
  SetLLThreshold(size_t{1} << 20);
  comm().ddaScratchBytes = 2 * dda::common::ddaLLRsBankBytes(8) - 128;
  EXPECT_EQ(GridBlock(4 * kSliceFloats), Shape(1u, 128u)) << "LL's bank does not fit";
  comm().ddaScratchBytes = std::numeric_limits<size_t>::max();
  table_.rsDdaLL128NarrowMaxPerRank = 0;
  EXPECT_EQ(GridBlock(kMiBFloats), Shape(69u, 256u)) << "a table without the entry";
  table_.rsDdaLL128NarrowMaxPerRank = size_t{1} << 20;
  comm().nRanks = 7;
  EXPECT_EQ(GridBlock(4 * kSliceFloats), Shape(1u, 256u)) << "below rankScaledMinRanks";
  comm().nRanks = 8;
  PeerParallelUpTo(size_t{1} << 30);
  EXPECT_EQ(GridBlock(4 * kSliceFloats), Shape(4u, 256u)) << "the peer-parallel kernel keeps a warp per rank";
  PeerParallelUpTo(0);
  comm().archThresholds = nullptr;
  EXPECT_EQ(GridBlock(4 * kSliceFloats), Shape(1u, 256u)) << "no table";
}

// A valid DDA_LL128_RS_THREADS wins over the narrow start; the grid budget still applies.
TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_ThreadsOverrideWinsOverTheShapeTable) {
  using Shape = std::pair<unsigned, unsigned>;
  UseShapeTable();
  params_["RCCL_DDA_LL128_RS_THREADS"] = 512;
  EXPECT_EQ(GridBlock(4 * kSliceFloats), Shape(1u, 512u));
  EXPECT_EQ(GridBlock(4370 * kSliceFloats), Shape(256u, 512u));
  params_["RCCL_DDA_LL128_RS_THREADS"] = 64;
  EXPECT_EQ(GridBlock(4 * kSliceFloats), Shape(2u, 64u));
  params_["RCCL_DDA_LL128_RS_THREADS"] = 48;
  EXPECT_EQ(GridBlock(4 * kSliceFloats), Shape(1u, 128u)) << "an invalid override is ignored";
}

// The shape table leaves shards past the narrow window on the full grid budget, including Ring Simple's capped band
// (8 MiB, 32 MiB], where capping the grid to Ring's channels was slower.
TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_ShapeTableKeepsTheGridBudgetPastTheNarrowWindow) {
  using Shape = std::pair<unsigned, unsigned>;
  constexpr size_t kMiBFloats = (size_t{1} << 20) / sizeof(float);
  UseShapeTable();
  ASSERT_EQ(comm().ddaFabricMaxBlocks, 256);
  ASSERT_EQ(comm().ddaLLEpochLen, 256);
  EXPECT_EQ(GridBlock(4370 * kSliceFloats), Shape(256u, 256u)) << "just past an 8 MiB shard";
  EXPECT_EQ(GridBlock(32 * kMiBFloats), Shape(256u, 256u)) << "a 32 MiB shard";
  EXPECT_EQ(ncclReduceScatterDdaFabricLL128Blocks(mock_.get(), 4370 * kSliceFloats, ncclFloat32), 256u);
  EXPECT_EQ(ncclReduceScatterDdaFabricLL128Blocks(mock_.get(), 64 * kMiBFloats, ncclFloat8e4m3), 256u);
}

// Past the narrow window a batched block packs 16 slices instead of 8 only where that adds no pass through the slice
// loop: shards up to 2048 slices fit one pass of the 256-block grid and keep it, 4 MiB shards drop to 137 blocks.
TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_ShapeTablePacksWithoutAddingAPass) {
  using Shape = std::pair<unsigned, unsigned>;
  constexpr size_t kMiB = size_t{1} << 20;
  struct Cell {
    int nRanks;
    size_t shardBytes;
    Shape shape;
  };
  const Cell cells[] = {
    {8, kMiB, {137u, 128u}},       {8, 2 * kMiB, {137u, 256u}},  {8, 15 * kMiB / 4, {256u, 256u}},
    {8, 4 * kMiB, {137u, 256u}},   {8, 6 * kMiB, {205u, 256u}},  {8, 8 * kMiB, {256u, 256u}},
    {12, 3 * kMiB / 2, {103u, 256u}}, {12, 2 * kMiB, {137u, 256u}},
    {16, kMiB, {256u, 512u}},      {16, 2 * kMiB, {137u, 256u}}, {16, 4 * kMiB, {137u, 256u}},
    {24, 3 * kMiB, {205u, 256u}},  {32, 2 * kMiB, {137u, 256u}}, {32, 4 * kMiB, {137u, 256u}},
    {64, kMiB, {137u, 128u}},      {64, 2 * kMiB, {137u, 256u}}, {64, 4 * kMiB, {137u, 256u}},
    {64, 8 * kMiB, {256u, 256u}},
  };
  const std::pair<ncclDataType_t, size_t> types[] = {{ncclBfloat16, 2}, {ncclFloat8e4m3, 1}, {ncclFloat32, 4}};
  UseShapeTable();
  for (const Cell& c : cells) {
    comm().nRanks = c.nRanks;
    const size_t n = (size_t)c.nRanks;
    PeerParallelUpTo(std::min(n * std::max(size_t{512} << 10, n * (size_t{64} << 10)), 32 * kMiB));
    for (const auto& [type, size] : types) {
      EXPECT_EQ(GridBlock(c.shardBytes / size, type), c.shape)
        << c.nRanks << " ranks, shard " << c.shardBytes << ", type " << type;
      EXPECT_EQ(ncclReduceScatterDdaFabricLL128Blocks(mock_.get(), c.shardBytes / size, type), c.shape.first);
    }
  }
}

// The packing doubles, up to a 512-thread block's slices, only while it keeps the unpacked grid's pass count, and never
// applies to a threads override, smaller comms or no table.
TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_ShapeTablePackingBoundaries) {
  using Shape = std::pair<unsigned, unsigned>;
  UseShapeTable();
  EXPECT_EQ(GridBlock(2033 * kSliceFloats), Shape(255u, 256u)) << "one pass: packing would add one";
  EXPECT_EQ(GridBlock(2048 * kSliceFloats), Shape(256u, 256u)) << "the last shard 256 blocks take in one pass";
  EXPECT_EQ(GridBlock(2049 * kSliceFloats), Shape(129u, 256u)) << "two passes either way";
  comm().ddaFabricMaxBlocks = 100;
  EXPECT_EQ(GridBlock(800 * kSliceFloats), Shape(100u, 256u)) << "one pass on a 100-block budget";
  EXPECT_EQ(GridBlock(801 * kSliceFloats), Shape(51u, 256u)) << "two passes either way";
  EXPECT_EQ(GridBlock(1104 * kSliceFloats), Shape(69u, 256u)) << "1.38 passes round up to 2, as 69 blocks take exactly";
  comm().ddaFabricMaxBlocks = 256;
  g_rcclDdaFabricRsLL128Threads = [](const ncclComm*) { return 128; };
  EXPECT_EQ(GridBlock(2185 * kSliceFloats), Shape(256u, 128u)) << "packs once: 16 slices would add a pass";
  EXPECT_EQ(GridBlock(3277 * kSliceFloats), Shape(205u, 128u)) << "packs twice: four passes at 4, 8 and 16 slices";
  EXPECT_EQ(GridBlock(1093 * kSliceFloats), Shape(137u, 128u)) << "packs once";
  g_rcclDdaFabricRsLL128Threads = [](const ncclComm*) { return 512; };
  EXPECT_EQ(GridBlock(4370 * kSliceFloats), Shape(256u, 512u)) << "a 512-thread block is already at the limit";
  g_rcclDdaFabricRsLL128Threads = [](const ncclComm*) { return 256; };
  params_["RCCL_DDA_LL128_RS_THREADS"] = 256;
  EXPECT_EQ(GridBlock(2185 * kSliceFloats), Shape(256u, 256u)) << "a threads override keeps a slice a warp";
  params_.erase("RCCL_DDA_LL128_RS_THREADS");
  comm().nRanks = 7;
  EXPECT_EQ(GridBlock(2185 * kSliceFloats), Shape(256u, 256u)) << "below rankScaledMinRanks";
  comm().nRanks = 8;
  comm().archThresholds = nullptr;
  EXPECT_EQ(GridBlock(2185 * kSliceFloats), Shape(256u, 256u)) << "no table";
}

// The reported block count is the launch grid of either kernel, caps included.
TEST_F(DdaReduceScatterFabricLL128Microtest, Blocks_MatchTheLaunchedGrid) {
  struct Case {
    size_t count;
    ncclDataType_t datatype;
  };
  const Case cases[] = {{8, ncclFloat16},
                        {16 * kSliceFloats + 4, ncclFloat32},
                        {32 * kSliceFloats, ncclBfloat16},
                        {16 * 100 * kSliceFloats, ncclFloat32}};
  for (size_t peerMax : {size_t{0}, size_t{1} << 30}) {
    PeerParallelUpTo(peerMax);
    for (int maxBlocks : {256, 6}) {
      comm().ddaFabricMaxBlocks = maxBlocks;
      for (const Case& c : cases) {
        EXPECT_EQ(ncclReduceScatterDdaFabricLL128Blocks(mock_.get(), c.count, c.datatype),
                  RunOne(c.count, c.datatype).grid.x)
          << "count " << c.count << " maxBlocks " << maxBlocks << " peerMax " << peerMax;
      }
    }
  }
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_ThreadsOverrideMustBeAWarpMultipleUpTo512) {
  const std::pair<int64_t, unsigned> cases[] = {{256, 256}, {32, 32},   {512, 512}, {16, 512},
                                                {48, 512},  {544, 512}, {0, 512},   {-32, 512}};
  for (const auto& [env, threads] : cases) {
    params_["RCCL_DDA_LL128_RS_THREADS"] = env;
    EXPECT_EQ(RunOne(8).block.x, threads) << "DDA_LL128_RS_THREADS " << env;
  }
  // 256 threads are 8 warps, so 9 slices take a second block.
  params_["RCCL_DDA_LL128_RS_THREADS"] = 256;
  EXPECT_EQ(RunOne(9 * kSliceFloats).grid.x, 2u);
  EXPECT_EQ(RunOne(8 * kSliceFloats).grid.x, 1u);
  EXPECT_EQ(ncclReduceScatterDdaFabricLL128Blocks(mock_.get(), 9 * kSliceFloats, ncclFloat32), 2u);
}

// The arch table picks the block size, 512 when it sets none or one the launch bound rules out.
// A valid DDA_LL128_RS_THREADS overrides it.
TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_ThreadsFromArchTableUnlessOverridden) {
  g_rcclDdaFabricRsLL128Threads = [](const ncclComm*) { return 256; };
  // 256 threads are 8 warps, so 9 slices take a second block.
  EXPECT_EQ(RunOne(9 * kSliceFloats).block.x, 256u);
  EXPECT_EQ(RunOne(9 * kSliceFloats).grid.x, 2u);
  EXPECT_EQ(ncclReduceScatterDdaFabricLL128Blocks(mock_.get(), 9 * kSliceFloats, ncclFloat32), 2u);
  params_["RCCL_DDA_LL128_RS_THREADS"] = 512;
  EXPECT_EQ(RunOne(9 * kSliceFloats).block.x, 512u);
  params_.erase("RCCL_DDA_LL128_RS_THREADS");
  for (int tuned : {0, 48, 1024}) {
    g_rcclDdaFabricRsLL128Threads = [tuned](const ncclComm*) { return tuned; };
    EXPECT_EQ(RunOne(8).block.x, 512u) << "table " << tuned;
  }
}

// The table's cap is the whole message: up to it the peer-parallel kernel runs, past it the batched one.
TEST_F(DdaReduceScatterFabricLL128Microtest, PeerParallel_UpToTheTableCapBatchedAbove) {
  EXPECT_EQ(RunOne(kSliceFloats).kernel, (Kernel<float, 8>()));
  PeerParallelUpTo(8 * kSliceFloats * sizeof(float));
  EXPECT_EQ(RunOne(kSliceFloats).kernel, (PeerParKernel<float, 8>()));
  EXPECT_EQ(RunOne(kSliceFloats + 4).kernel, (Kernel<float, 8>()));
  EXPECT_EQ(RunOne(2 * kSliceFloats, ncclFloat16).kernel, (PeerParKernel<half, 8>()));
  EXPECT_EQ(RunOne(2 * kSliceFloats + 8, ncclFloat16).kernel, (Kernel<half, 8>()));
}

// DDA_LL128_RS_PEER_MAX_BYTES overrides the table's cap. -1, the default, reads the table, and 0 or another negative
// keeps the batched kernel.
TEST_F(DdaReduceScatterFabricLL128Microtest, PeerParallel_EnvOverridesTheTable) {
  PeerParallelUpTo(size_t{1} << 30);
  for (int64_t env : {int64_t{0}, int64_t{-5}}) {
    params_["RCCL_DDA_LL128_RS_PEER_MAX_BYTES"] = env;
    EXPECT_EQ(RunOne(4).kernel, (Kernel<float, 8>())) << "DDA_LL128_RS_PEER_MAX_BYTES " << env;
  }
  params_["RCCL_DDA_LL128_RS_PEER_MAX_BYTES"] = 8 * 16;
  EXPECT_EQ(RunOne(4).kernel, (PeerParKernel<float, 8>()));
  EXPECT_EQ(RunOne(8).kernel, (Kernel<float, 8>()));
  params_["RCCL_DDA_LL128_RS_PEER_MAX_BYTES"] = -1;
  EXPECT_EQ(RunOne(8).kernel, (PeerParKernel<float, 8>()));
  PeerParallelUpTo(0);
  EXPECT_EQ(RunOne(8).kernel, (Kernel<float, 8>()));
  params_["RCCL_DDA_LL128_RS_PEER_MAX_BYTES"] = 8 * 32;
  EXPECT_EQ(RunOne(8).kernel, (PeerParKernel<float, 8>()));
}

TEST_F(DdaReduceScatterFabricLL128Microtest, PeerParallel_EachDatatypeLaunchesItsOwnKernel) {
  PeerParallelUpTo(size_t{1} << 30);
  ExpectLaunchOf<float>(RunOne(16, ncclFloat32), 16, PeerParKernel<float, 8>());
  ExpectLaunchOf<half>(RunOne(16, ncclFloat16), 16, PeerParKernel<half, 8>());
  ExpectLaunchOf<bf16>(RunOne(16, ncclBfloat16), 16, PeerParKernel<bf16, 8>());
  ExpectLaunchOf<dda::common::fp8e4m3>(RunOne(16, ncclFloat8e4m3), 16, PeerParKernel<dda::common::fp8e4m3, 8>());
  ExpectLaunchOf<dda::common::fp8e5m2>(RunOne(16, ncclFloat8e5m2), 16, PeerParKernel<dda::common::fp8e5m2, 8>());
}

TEST_F(DdaReduceScatterFabricLL128Microtest, PeerParallel_FourAndEightRanksUnrolledOthersRuntime) {
  PeerParallelUpTo(size_t{1} << 30);
  const std::pair<int, const void*> cases[] = {{4, PeerParKernel<float, 4>()},
                                               {8, PeerParKernel<float, 8>()},
                                               {2, PeerParKernel<float, 0>()},
                                               {5, PeerParKernel<float, 0>()},
                                               {144, PeerParKernel<float, 0>()}};
  for (const auto& [nRanks, kernel] : cases) {
    comm().nRanks = nRanks;
    const DdaLaunch& launch = RunOne(4);
    EXPECT_EQ(launch.kernel, kernel) << "nRanks " << nRanks;
    EXPECT_EQ(launch.Arg<int>(5), nRanks);
  }
}

// A block per slice within the block cap, capped at ddaFabricMaxBlocks and at the epoch cells.
TEST_F(DdaReduceScatterFabricLL128Microtest, PeerParallel_GridFollowsShardSlicesWithinCaps) {
  PeerParallelUpTo(size_t{1} << 30);
  EXPECT_EQ(RunOne(8, ncclFloat16).grid.x, 1u);
  EXPECT_EQ(RunOne(kSliceFloats).grid.x, 1u);
  EXPECT_EQ(RunOne(kSliceFloats + 4).grid.x, 2u);
  const DdaLaunch& uncapped = RunOne(100 * kSliceFloats);
  EXPECT_EQ(uncapped.grid.x, 100u);
  EXPECT_EQ(uncapped.block.x, 256u);
  comm().ddaFabricMaxBlocks = 6;
  const DdaLaunch& capped = RunOne(100 * kSliceFloats);
  EXPECT_EQ(capped.grid.x, 6u);
  EXPECT_EQ(capped.block.x, 512u);
  comm().ddaFabricMaxBlocks = 0;
  EXPECT_EQ(RunOne(100 * kSliceFloats).grid.x, 1u);
  comm().ddaFabricMaxBlocks = 256;
  comm().ddaLLEpochLen = 9;
  EXPECT_EQ(RunOne(100 * kSliceFloats).grid.x, 9u);
  comm().ddaLLEpochLen = 0;
  EXPECT_EQ(RunOne(100 * kSliceFloats).grid.x, 1u);
}

// A warp group is a warp per rank, up to the 16 warps of 512 threads.
TEST_F(DdaReduceScatterFabricLL128Microtest, PeerParallel_BlockIsOneWarpPerRank) {
  PeerParallelUpTo(size_t{1} << 30);
  const std::pair<int, unsigned> cases[] = {{2, 64u},   {3, 96u},   {4, 128u},  {8, 256u},
                                            {12, 384u}, {16, 512u}, {32, 512u}, {144, 512u}};
  for (const auto& [nRanks, threads] : cases) {
    comm().nRanks = nRanks;
    const DdaLaunch& launch = RunOne(kSliceFloats);
    EXPECT_EQ(launch.block.x, threads) << "nRanks " << nRanks;
    EXPECT_EQ(launch.grid.x, 1u) << "nRanks " << nRanks;
  }
}

// Past the block cap a block doubles its groups, up to 512 threads: 40 slices at 4 ranks.
TEST_F(DdaReduceScatterFabricLL128Microtest, PeerParallel_BlockTakesMoreGroupsOnlyPastTheCap) {
  PeerParallelUpTo(size_t{1} << 30);
  comm().nRanks = 4;
  const std::pair<int, std::pair<unsigned, unsigned>> cases[] = {{40, {40u, 128u}}, {20, {20u, 256u}}, {6, {6u, 512u}}};
  for (const auto& [maxBlocks, want] : cases) {
    comm().ddaFabricMaxBlocks = maxBlocks;
    const DdaLaunch& launch = RunOne(40 * kSliceFloats);
    EXPECT_EQ(launch.grid.x, want.first) << "maxBlocks " << maxBlocks;
    EXPECT_EQ(launch.block.x, want.second) << "maxBlocks " << maxBlocks;
  }
}

// DDA_LL128_RS_THREADS and the table's block size apply to the batched kernel only.
TEST_F(DdaReduceScatterFabricLL128Microtest, PeerParallel_BlockIgnoresTheBatchedThreadSettings) {
  params_["RCCL_DDA_LL128_RS_THREADS"] = 32;
  g_rcclDdaFabricRsLL128Threads = [](const ncclComm*) { return 128; };
  EXPECT_EQ(RunOne(4).block.x, 32u);
  PeerParallelUpTo(size_t{1} << 30);
  EXPECT_EQ(RunOne(4).block.x, 256u);
  params_.erase("RCCL_DDA_LL128_RS_THREADS");
  EXPECT_EQ(RunOne(4).block.x, 256u);
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
}

// Walks phase 2's poll batches as the kernel does; returns the valid peers in order and counts invalid entries.
template <int kBatch>
std::vector<int> WalkPollBatches(int selfRank, int nRanks, int& invalid, int& invalidNotSelf) {
  std::vector<int> visited;
  int peer = dda::common::ddaLL128RsNextRank(selfRank, nRanks);
  for (int base = 1; base < nRanks; base += kBatch) {
    int peers[kBatch];
    bool valid[kBatch];
    peer = dda::common::ddaLL128RsBatchPeers(peer, base, nRanks, peers, valid);
    for (int k = 0; k < kBatch; ++k) {
      if (valid[k]) {
        visited.push_back(peers[k]);
      } else {
        ++invalid;
        invalidNotSelf += peers[k] != selfRank;
      }
    }
  }
  return visited;
}

template <int kBatch>
void ExpectPollWalkCoversPeersInRankOrder() {
  for (int nRanks = 2; nRanks <= 144; ++nRanks) {
    for (int self = 0; self < nRanks; ++self) {
      std::vector<int> expected;
      for (int r = 1; r < nRanks; ++r) {
        expected.push_back((self + r) % nRanks);
      }
      int invalid = 0;
      int invalidNotSelf = 0;
      ASSERT_EQ(WalkPollBatches<kBatch>(self, nRanks, invalid, invalidNotSelf), expected)
        << "batch " << kBatch << " nRanks " << nRanks << " self " << self;
      ASSERT_EQ(invalid, (kBatch - (nRanks - 1) % kBatch) % kBatch) << "batch " << kBatch << " nRanks " << nRanks;
      ASSERT_EQ(invalidNotSelf, 0) << "batch " << kBatch << " nRanks " << nRanks << " self " << self;
    }
  }
}

// Every peer is polled once in rank order after self; a past-the-end entry names self, whose own slot is never written.
TEST_F(DdaReduceScatterFabricLL128Microtest, PollBatch_WalkVisitsEveryPeerOnceInRankOrder) {
  static_assert(dda::common::kDdaLL128RsPollBatch<float> == 8);
  static_assert(dda::common::kDdaLL128RsPollBatch<half> == 8);
  static_assert(dda::common::kDdaLL128RsPollBatch<bf16> == 8);
  static_assert(dda::common::kDdaLL128RsPollBatch<dda::common::fp8e4m3> == 4);
  static_assert(dda::common::kDdaLL128RsPollBatch<dda::common::fp8e5m2> == 4);
  ExpectPollWalkCoversPeersInRankOrder<dda::common::kDdaLL128RsPollBatch<float>>();
  ExpectPollWalkCoversPeersInRankOrder<dda::common::kDdaLL128RsPollBatch<dda::common::fp8e4m3>>();
  // The 4- and 8-rank instances cap the batch at their peer count.
  ExpectPollWalkCoversPeersInRankOrder<3>();
  ExpectPollWalkCoversPeersInRankOrder<7>();
}

}  // namespace
