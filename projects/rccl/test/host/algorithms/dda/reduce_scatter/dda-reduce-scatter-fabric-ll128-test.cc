/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric_ll128.cu: eligibility against
// the scratch bank's per-rank slot and the LL128 cap, the datatype dispatch, the slice grid and its caps, and the
// launch.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "../common/DdaFabricTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"

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
    // LL at the table's 4 MiB cap, as on gfx1250 SPX, so the shards it ran get narrow blocks.
    SetLLThreshold(size_t{4} << 20);
    // The launch-shape entries of the gfx1250 SPX table, on one node.
    comm().nNodes = 1;
    table_.ddaLLMax[ncclFuncReduceScatter] = size_t{4} << 20;
    table_.rsRingLL128MaxPerRank1B = size_t{32} << 20;
    table_.rankScaledMinRanks = 8;
    table_.rsRingLL128MaxPerRank = size_t{8} << 20;
    table_.rsRingSimpleCappedMaxPerRank = size_t{32} << 20;
    table_.rsRingSimpleCappedChannels = 128;
    table_.rsDdaLL128NarrowMaxPerRank = size_t{1} << 20;
    comm().archThresholds = &table_;
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

  template <typename T>
  void ExpectLaunchOf(const DdaLaunch& launch, size_t recvcount) {
    EXPECT_EQ(launch.kernel, (Kernel<T, 8>()));
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
  EXPECT_EQ(launch.block.x, 128u);
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

// One warp per slice. From the table's rankScaledMinRanks, a shard up to rsDdaLL128NarrowMaxPerRank that LL ran
// runs 128-thread blocks (4 warps) until ddaFabricMaxBlocks blocks are full, then 256 and 512; a larger shard keeps
// the blocks 512 threads take but runs 256, and one Ring Simple runs on capped channels gets at most a quarter more
// of them. Smaller comms, comms without a table and tables without the entry run 512-thread blocks.
TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_GridFollowsShardSlicesWithinCaps) {
  auto shape = [&](size_t count, ncclDataType_t type = ncclFloat32) {
    const DdaLaunch& launch = RunOne(count, type);
    return std::make_pair(launch.grid.x, launch.block.x);
  };
  using Shape = std::pair<unsigned, unsigned>;
  constexpr size_t kMiBFloats = (size_t{1} << 20) / sizeof(float);
  ASSERT_EQ(comm().nRanks, 8);
  ASSERT_EQ(comm().ddaFabricMaxBlocks, 256);
  EXPECT_EQ(shape(8, ncclFloat16), Shape(1u, 128u));
  EXPECT_EQ(shape(4 * kSliceFloats), Shape(1u, 128u));
  EXPECT_EQ(shape(4 * kSliceFloats + 4), Shape(2u, 128u));
  EXPECT_EQ(shape(8 * kSliceFloats, ncclBfloat16), Shape(1u, 128u));
  EXPECT_EQ(shape(100 * kSliceFloats), Shape(25u, 128u));
  EXPECT_EQ(shape(kMiBFloats), Shape(137u, 128u)) << "a 1 MiB shard, 547 slices";
  EXPECT_EQ(shape(547 * kSliceFloats), Shape(35u, 256u)) << "past 1 MiB: the blocks 512 threads take, at 256";
  EXPECT_EQ(shape(600 * kSliceFloats), Shape(38u, 256u));
  comm().ddaFabricMaxBlocks = 6;
  EXPECT_EQ(shape(24 * kSliceFloats), Shape(6u, 128u));
  EXPECT_EQ(shape(25 * kSliceFloats), Shape(4u, 256u));
  EXPECT_EQ(shape(49 * kSliceFloats), Shape(4u, 512u));
  EXPECT_EQ(shape(1600 * kSliceFloats), Shape(6u, 256u));
  EXPECT_EQ(shape(4370 * kSliceFloats), Shape(6u, 256u)) << "the grid budget below the Ring limit still holds";
  comm().ddaFabricMaxBlocks = 0;
  EXPECT_EQ(shape(1600 * kSliceFloats), Shape(1u, 256u));
  comm().ddaFabricMaxBlocks = 256;
  comm().ddaLLEpochLen = 9;
  EXPECT_EQ(shape(1600 * kSliceFloats), Shape(9u, 256u));
  comm().ddaLLEpochLen = 0;
  EXPECT_EQ(shape(1600 * kSliceFloats), Shape(1u, 256u));
  comm().ddaLLEpochLen = 256;
  EXPECT_EQ(shape(8 * kMiBFloats), Shape(256u, 256u)) << "an 8 MiB shard Ring ran on LL128: no Ring limit";
  EXPECT_EQ(shape(4370 * kSliceFloats), Shape(160u, 256u)) << "past 8 MiB: 128 Ring channels and a quarter";
  EXPECT_EQ(shape(32 * kMiBFloats), Shape(160u, 256u)) << "a 32 MiB shard is still within the capped band";
  EXPECT_EQ(shape(17477 * kSliceFloats), Shape(256u, 256u)) << "past 32 MiB Ring Simple is not capped";
  EXPECT_EQ(shape(64 * kMiBFloats, ncclFloat8e4m3), Shape(256u, 256u)) << "1-byte types: Ring LL128 to 32 MiB";
  table_.rsRingLL128MaxPerRank1B = 0;
  EXPECT_EQ(shape(64 * kMiBFloats, ncclFloat8e4m3), Shape(160u, 256u)) << "no 1-byte entry: the 2/4-byte band";
  table_.rsRingLL128MaxPerRank1B = size_t{32} << 20;
  comm().nNodes = 2;
  EXPECT_EQ(shape(4370 * kSliceFloats), Shape(256u, 256u)) << "Ring caps channels on one node only";
  comm().nNodes = 1;
  comm().nRanks = 16;
  EXPECT_EQ(shape(kMiBFloats), Shape(137u, 128u));
  EXPECT_EQ(shape(547 * kSliceFloats), Shape(35u, 256u));
  EXPECT_EQ(shape(4370 * kSliceFloats), Shape(160u, 256u));
  comm().nRanks = 8;
  // Shards LL did not run keep full width: LL off, LL below the table's cap, or LL's bank past the scratch.
  SetLLThreshold(0);
  EXPECT_EQ(shape(kMiBFloats), Shape(35u, 512u)) << "LL off";
  EXPECT_EQ(shape(547 * kSliceFloats), Shape(35u, 256u));
  SetLLThreshold(size_t{1} << 20);
  EXPECT_EQ(shape(kMiBFloats), Shape(35u, 512u)) << "LL lowered to 1 MiB";
  SetLLThreshold(size_t{4} << 20);
  comm().ddaScratchBytes = 2 * dda::common::ddaLLRsBankBytes(8) - 128;
  EXPECT_EQ(shape(4 * kSliceFloats), Shape(1u, 512u)) << "LL's bank does not fit";
  comm().ddaScratchBytes = std::numeric_limits<size_t>::max();
  table_.rsDdaLL128NarrowMaxPerRank = 0;
  EXPECT_EQ(shape(kMiBFloats), Shape(35u, 512u)) << "a table without the entry keeps 512 threads";
  EXPECT_EQ(shape(1600 * kSliceFloats), Shape(100u, 512u));
  table_.rsDdaLL128NarrowMaxPerRank = size_t{1} << 20;
  comm().nRanks = 7;
  EXPECT_EQ(shape(4 * kSliceFloats), Shape(1u, 512u)) << "below rankScaledMinRanks every shard keeps 512 threads";
  EXPECT_EQ(shape(1600 * kSliceFloats), Shape(100u, 512u));
  EXPECT_EQ(shape(4370 * kSliceFloats), Shape(256u, 512u)) << "and no Ring limit";
  comm().nRanks = 8;
  comm().archThresholds = nullptr;
  EXPECT_EQ(shape(100 * kSliceFloats), Shape(7u, 512u)) << "no table: 512 threads";
  EXPECT_EQ(shape(4370 * kSliceFloats), Shape(256u, 512u));
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
  static_assert(dda::common::kDdaLL128RsPollBatch<float> == 3);
  static_assert(dda::common::kDdaLL128RsPollBatch<half> == 3);
  static_assert(dda::common::kDdaLL128RsPollBatch<bf16> == 3);
  static_assert(dda::common::kDdaLL128RsPollBatch<dda::common::fp8e4m3> == 2);
  static_assert(dda::common::kDdaLL128RsPollBatch<dda::common::fp8e5m2> == 2);
  ExpectPollWalkCoversPeersInRankOrder<dda::common::kDdaLL128RsPollBatch<float>>();
  ExpectPollWalkCoversPeersInRankOrder<dda::common::kDdaLL128RsPollBatch<dda::common::fp8e4m3>>();
}

}  // namespace
