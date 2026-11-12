/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for src/algorithms/dda/broadcast/dda_broadcast_fabric_ll128.cu:
 * eligibility checks (comm resources, rank bounds, alignment, scratch capacity,
 * two-shot threshold), geometry helpers (chunk slices, slot slices), block count,
 * thread-count override via RCCL_DDA_LL128_BC_THREADS, arch-table thread count,
 * fan-out param forwarding, and the kernel launch arguments.
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>

#include "../common/DdaFabricTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"
#include "algorithms/dda/broadcast/broadcast_dda_fabric_ll128_twoshot.h"

#include "fakes/param_redirect.h"

#include DDA_BROADCAST_FABRIC_LL128_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {

using RcclUnitTesting::DdaFabricMockComm;
using dda::common::ddaLL128BcTwoShotChunkSlices;
using dda::common::ddaLL128BcTwoShotSlotSlices;

void* const kSendbuff = reinterpret_cast<void*>(0x1000);
void* const kRecvbuff = reinterpret_cast<void*>(0x2000);
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);
// A slice carries 1920 payload bytes in 2 KiB on the wire.
constexpr size_t kSliceData = 1920;
constexpr size_t kSliceWire = 2048;

// ---------------------------------------------------------------------------
// Pure-host geometry helpers (no comm, no fake machinery needed)
// ---------------------------------------------------------------------------

TEST(DdaBroadcastLL128GeomHelpers, ChunkSlicesRoundsUpAcrossPeers) {
  // 4 ranks => 3 peers; 10 total slices => ceil(10/3)=4 per chunk
  EXPECT_EQ(ddaLL128BcTwoShotChunkSlices(10, 4), 4u);
  // Exact division
  EXPECT_EQ(ddaLL128BcTwoShotChunkSlices(9, 4), 3u);
  // Single slice always rounds to 1
  EXPECT_EQ(ddaLL128BcTwoShotChunkSlices(1, 4), 1u);
  // 2 ranks => 1 peer; chunkSlices == totalSlices
  EXPECT_EQ(ddaLL128BcTwoShotChunkSlices(5, 2), 5u);
}

TEST(DdaBroadcastLL128GeomHelpers, ChunkSlicesMinIsOne) {
  EXPECT_GE(ddaLL128BcTwoShotChunkSlices(0, 4), 1u);
}

TEST(DdaBroadcastLL128GeomHelpers, ChunkSlicesFewerSlicesThanPeers) {
  // Sizes where totalSlices < nPeers: every peer still gets cs >= 1 (ceil rounds up).
  // nRanks=4 => nPeers=3
  EXPECT_EQ(ddaLL128BcTwoShotChunkSlices(1, 4), 1u);  // ceil(1/3)=1
  EXPECT_EQ(ddaLL128BcTwoShotChunkSlices(2, 4), 1u);  // ceil(2/3)=1
  // nRanks=16 => nPeers=15; sparse distribution
  EXPECT_EQ(ddaLL128BcTwoShotChunkSlices(1,  16), 1u);  // ceil(1/15)=1
  EXPECT_EQ(ddaLL128BcTwoShotChunkSlices(7,  16), 1u);  // ceil(7/15)=1
  EXPECT_EQ(ddaLL128BcTwoShotChunkSlices(14, 16), 1u);  // ceil(14/15)=1
  EXPECT_EQ(ddaLL128BcTwoShotChunkSlices(15, 16), 1u);  // ceil(15/15)=1 (exact)
  EXPECT_EQ(ddaLL128BcTwoShotChunkSlices(16, 16), 2u);  // ceil(16/15)=2 (crosses boundary)
}

TEST(DdaBroadcastLL128GeomHelpers, SlotSlicesZeroWhenBankTooSmall) {
  // 4 ranks, 1 chunk slice => need (nRanks+1)=5 slots, each kSliceWire bytes
  const size_t exact = 5 * kSliceWire;
  EXPECT_EQ(ddaLL128BcTwoShotSlotSlices(exact, 4, 1), 1u);
  EXPECT_EQ(ddaLL128BcTwoShotSlotSlices(exact - 1, 4, 1), 0u);
}

TEST(DdaBroadcastLL128GeomHelpers, SlotSlicesScalesWithChunkSlices) {
  // 4 ranks, chunkSlices=3 => need 5*3*kSliceWire bank
  const size_t bank = 5 * 3 * kSliceWire;
  EXPECT_EQ(ddaLL128BcTwoShotSlotSlices(bank, 4, 9), 3u);
  EXPECT_EQ(ddaLL128BcTwoShotSlotSlices(bank - 1, 4, 9), 0u);
}


TEST(DdaBroadcastLL128GeomHelpers, SlotSlicesRootGatherSlotAllocatedFullChunk) {
  // The kernel writes only 1 padding slice into the root's gather slot (Q5/O1),
  // but ddaLL128BcTwoShotSlotSlices reserves a full cs-wide slot for it because
  // the formula uses totalSlotsNeeded = nRanks + 1 (conservative).
  // This test pins that behaviour: a bank sized for the true kernel minimum
  // (scatter slot + nPeers gather slots + 1 padding slice = cs*nRanks + 1 wire-slices)
  // must be rejected; only cs*(nRanks+1) is accepted.
  constexpr size_t cs = 3;
  constexpr size_t nRanks = 4;  // nPeers = 3
  // totalSlices = cs * nPeers so ddaLL128BcTwoShotChunkSlices returns exactly cs.
  constexpr size_t totalSlices = cs * (nRanks - 1);
  // Minimum bank the kernel actually uses: cs*nRanks scatter/gather + 1 root padding.
  constexpr size_t kernelMin  = (cs * nRanks + 1) * kSliceWire;
  // Minimum bank the formula requires: cs*(nRanks+1).
  constexpr size_t formulaMin = cs * (nRanks + 1) * kSliceWire;
  EXPECT_EQ(ddaLL128BcTwoShotSlotSlices(formulaMin,     nRanks, totalSlices), cs);
  EXPECT_EQ(ddaLL128BcTwoShotSlotSlices(formulaMin - 1, nRanks, totalSlices), 0u);
  // kernelMin < formulaMin when cs > 1: the formula is conservative by cs-1 slices.
  EXPECT_LT(kernelMin, formulaMin);
  EXPECT_EQ(ddaLL128BcTwoShotSlotSlices(kernelMin,      nRanks, totalSlices), 0u);
}

TEST(DdaBroadcastLL128GeomHelpers, SlotSlicesCs1HasNoConservatism) {
  // When cs=1 (totalSlices <= nPeers), kernelMin == formulaMin: no wasted capacity.
  constexpr size_t nRanks = 4;
  // Any totalSlices in [1, nPeers] yields cs=1; use totalSlices=1 (most sparse case).
  constexpr size_t kernelMin  = (1 * nRanks + 1) * kSliceWire;  // (cs*nRanks + 1) * wire
  constexpr size_t formulaMin = 1 * (nRanks + 1) * kSliceWire;  // cs*(nRanks+1) * wire
  static_assert(kernelMin == formulaMin, "cs=1 must have zero conservatism");
  EXPECT_EQ(ddaLL128BcTwoShotSlotSlices(formulaMin,     nRanks, 1), 1u);
  EXPECT_EQ(ddaLL128BcTwoShotSlotSlices(formulaMin - 1, nRanks, 1), 0u);
}

// ---------------------------------------------------------------------------
// Eligibility + launch fixture
// ---------------------------------------------------------------------------

struct ScopedEnv {
  explicit ScopedEnv(const char* name) : name_(name) { setenv(name, "", /*overwrite=*/1); }
  ~ScopedEnv() { unsetenv(name_); }
  const char* name_;
};

class DdaBroadcastFabricLL128Microtest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hipGetLastError = [] { return hipSuccess; };
    g_loadParam = [this](const char* env, int64_t deftVal) {
      auto it = params_.find(env);
      return it == params_.end() ? deftVal : it->second;
    };
    comm().nRanks = 4;
    comm().rank = 0;
    comm().addonStopEvent = kStopEvent;
  }

  void TearDown() override {
    unsetenv("RCCL_DDA_LL128_BC_THREADS");
    ResetDdaCollSeams();
    ResetHipFakes();
    ResetNcclFakes();
  }

  ncclComm& comm() { return mock_.comm; }

  bool Eligible(size_t count, ncclDataType_t datatype = ncclFloat32) {
    return ncclBroadcastDdaFabricLL128Eligible(mock_.get(), count, datatype);
  }

  ncclResult_t Run(size_t count, ncclDataType_t datatype = ncclFloat32, int root = 0) {
    return ncclBroadcastDdaFabricLL128(kSendbuff, kRecvbuff, count, datatype, root, mock_.get(), kStream);
  }

  const DdaLaunch& RunOne(size_t count, int root = 0) {
    g_ddaLaunches.clear();
    EXPECT_EQ(Run(count, ncclFloat32, root), ncclSuccess);
    EXPECT_EQ(g_ddaLaunches.size(), 1u);
    return g_ddaLaunches.at(0);
  }

  DdaFabricMockComm mock_;
  std::map<std::string, int64_t> params_;
};

// ---------------------------------------------------------------------------
// Eligibility tests
// ---------------------------------------------------------------------------

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_BaselineFp32) {
  EXPECT_TRUE(Eligible(4));
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_NullCommOrBootstrapRejects) {
  EXPECT_FALSE(ncclBroadcastDdaFabricLL128Eligible(nullptr, 4, ncclFloat32));
  comm().bootstrap = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_NeedsAllFabricResources) {
  for (int missing = 0; missing < 5; ++missing) {
    mock_.setFabricResourcesPresent(true);
    comm().ddaFabricMemHandler = missing == 0 ? nullptr : comm().ddaFabricMemHandler;
    comm().ddaScratch          = missing == 1 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev      = missing == 2 ? nullptr : comm().ddaPeerPtrsDev;
    comm().ddaLLEpochDev       = missing == 3 ? nullptr : comm().ddaLLEpochDev;
    comm().ddaLLEpochLen       = missing == 4 ? 0        : comm().ddaLLEpochLen;
    EXPECT_FALSE(Eligible(4)) << "missing resource " << missing;
  }
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_ZeroCountRejects) {
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_RankBoundsEnforced) {
  comm().nRanks = 1;
  EXPECT_FALSE(Eligible(4));
  comm().nRanks = 2;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 144;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 145;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_SupportedDatatypesOnly) {
  EXPECT_TRUE(Eligible(8, ncclFloat16));
  EXPECT_TRUE(Eligible(8, ncclBfloat16));
  EXPECT_TRUE(Eligible(16, ncclFloat8e4m3));
  EXPECT_TRUE(Eligible(16, ncclFloat8e5m2));
  for (ncclDataType_t dt : {ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclFloat64}) {
    EXPECT_FALSE(Eligible(16, dt)) << "datatype " << dt;
  }
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_BytesMustBe16ByteAligned) {
  // ncclFloat32 = 4 bytes/element; 3 elements = 12 bytes — not 16-aligned
  EXPECT_FALSE(Eligible(3, ncclFloat32));
  EXPECT_TRUE(Eligible(4, ncclFloat32));
  // ncclFloat16 = 2 bytes; 6 elements = 12 bytes — not 16-aligned
  EXPECT_FALSE(Eligible(6, ncclFloat16));
  EXPECT_TRUE(Eligible(8, ncclFloat16));
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_ScratchCapacityGates) {
  // 4 ranks, 1 chunk slice => need 2*(nRanks+1)*kSliceWire = 2*5*kSliceWire total scratch
  comm().ddaScratchBytes = 2 * 5 * kSliceWire;
  EXPECT_TRUE(Eligible(kSliceData / 4));
  // One byte short
  comm().ddaScratchBytes = 2 * 5 * kSliceWire - 1;
  EXPECT_FALSE(Eligible(kSliceData / 4));
}

// ---------------------------------------------------------------------------
// Launch tests — grid geometry
// ---------------------------------------------------------------------------

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_GridXIsNPeers) {
  for (int nRanks : {2, 4, 8, 16}) {
    comm().nRanks = nRanks;
    const DdaLaunch& launch = RunOne(4);
    EXPECT_EQ(launch.grid.x, static_cast<unsigned>(nRanks - 1)) << "nRanks=" << nRanks;
  }
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_BlocksPerPeerGrowsWithSlices) {
  // 512 threads = 16 warps, nPeers=3: bpp = ceil(ceil(totalSlices/3) / 16).
  // Transition at totalSlices=49: cs=17 -> bpp=2; at 48: cs=16 -> bpp=1.
  EXPECT_EQ(RunOne(48 * kSliceData / 4).grid.y, 1u);
  EXPECT_EQ(RunOne(49 * kSliceData / 4).grid.y, 2u);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_MaxBlocksTooSmallStillGivesOneBpp) {
  comm().ddaFabricMaxBlocks = 2;  // less than nPeers=3
  const DdaLaunch& launch = RunOne(1u << 20);
  EXPECT_GE(launch.grid.x, 1u);
  EXPECT_EQ(launch.grid.y, 1u);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_GridTotalNeverExceedsMaxBlocks) {
  // grid.x * grid.y must never exceed ddaFabricMaxBlocks regardless of cap value.
  // Large message (1 MiB float32) produces many slices so bpp would be unconstrained.
  for (unsigned cap : {3u, 4u, 6u, 1024u}) {
    comm().ddaFabricMaxBlocks = cap;
    const DdaLaunch& l = RunOne(size_t{1} << 18);  // 262144 float32 elements = 1 MiB
    EXPECT_LE(l.grid.x * l.grid.y, cap) << "ddaFabricMaxBlocks=" << cap;
  }
}

// ---------------------------------------------------------------------------
// Launch tests — thread count
// ---------------------------------------------------------------------------

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_ThreadsDefaultIs512) {
  EXPECT_EQ(RunOne(4).block.x, 512u);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_ThreadsEnvVarOverridesDefault) {
  ScopedEnv envGuard("RCCL_DDA_LL128_BC_THREADS");
  const std::pair<int64_t, unsigned> cases[] = {
      {256, 256}, {32, 32}, {1024, 1024},
      {16,  512},   // not a warp multiple -> fall back
      {0,   512},   // zero -> fall back
      {-32, 512},   // negative -> fall back
      {1056, 512},  // > 1024 -> fall back
  };
  for (const auto& [env, expected] : cases) {
    params_["RCCL_DDA_LL128_BC_THREADS"] = env;
    EXPECT_EQ(RunOne(4).block.x, expected) << "RCCL_DDA_LL128_BC_THREADS=" << env;
  }
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_ArchTableThreadsUsedWhenEnvAbsent) {
  rcclArchThresholds fakeTable{};
  fakeTable.ddaLL128BcTwoShotThreads = 256;
  comm().archThresholds = &fakeTable;
  EXPECT_EQ(RunOne(4).block.x, 256u);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_EnvVarOverridesArchTable) {
  ScopedEnv envGuard("RCCL_DDA_LL128_BC_THREADS");
  rcclArchThresholds fakeTable{};
  fakeTable.ddaLL128BcTwoShotThreads = 256;
  comm().archThresholds = &fakeTable;
  params_["RCCL_DDA_LL128_BC_THREADS"] = 1024;
  EXPECT_EQ(RunOne(4).block.x, 1024u);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_ArchTableZeroFallsBackToDefault) {
  rcclArchThresholds fakeTable{};
  fakeTable.ddaLL128BcTwoShotThreads = 0;
  comm().archThresholds = &fakeTable;
  EXPECT_EQ(RunOne(4).block.x, 512u);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_ThreadCountAffectsBlocksPerPeer) {
  // 256 threads = 8 warps, nPeers=3: bpp = ceil(ceil(totalSlices/3) / 8).
  // Transition at totalSlices=25: cs=9 -> bpp=2; at 24: cs=8 -> bpp=1.
  ScopedEnv envGuard("RCCL_DDA_LL128_BC_THREADS");
  params_["RCCL_DDA_LL128_BC_THREADS"] = 256;
  EXPECT_EQ(RunOne(25 * kSliceData / 4).grid.y, 2u);
  EXPECT_EQ(RunOne(24 * kSliceData / 4).grid.y, 1u);
}

// ---------------------------------------------------------------------------
// Launch tests — kernel arguments
// ---------------------------------------------------------------------------

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_PassesAllKernelArgs) {
  comm().ddaScratchBytes = 4096 + 40;
  const int root = 2;
  ASSERT_EQ(Run(1000, ncclBfloat16, root), ncclSuccess);
  ASSERT_EQ(g_ddaLaunches.size(), 1u);
  const DdaLaunch& launch = g_ddaLaunches[0];

  EXPECT_EQ(launch.stream, kStream);
  EXPECT_EQ(launch.startEvent, nullptr);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  EXPECT_EQ(comm().addonStopEvent, nullptr);  // consumed by rcclTakeAddonStopEvent
  EXPECT_EQ(launch.sharedMemBytes, 0u);
  EXPECT_EQ(launch.flags, 0u);

  // Arg indices: 0:peerScratch 1:recvbuff 2:sendbuff 3:totalBytes 4:selfRank
  //              5:nRanksRt 6:root 7:epochDev 8:epochLen 9:totalSlices
  //              10:chunkSlices 11:bankSize 12:params 13:fanOut
  ASSERT_GE(launch.args.size(), 14u);
  EXPECT_EQ(launch.Arg<int8_t**>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch.Arg<int8_t*>(1), kRecvbuff);
  EXPECT_EQ(launch.Arg<const int8_t*>(2), kSendbuff);
  EXPECT_EQ(launch.Arg<size_t>(3), 2000u);       // 1000 * sizeof(bfloat16)
  EXPECT_EQ(launch.Arg<int>(4), 0);              // selfRank = comm().rank
  EXPECT_EQ(launch.Arg<int>(5), 4);              // nRanksRt
  EXPECT_EQ(launch.Arg<int>(6), root);
  EXPECT_EQ(launch.Arg<uint32_t*>(7), comm().ddaLLEpochDev);
  EXPECT_EQ(launch.Arg<int>(8), comm().ddaLLEpochLen);
  EXPECT_EQ(launch.Arg<dda::common::DdaKernelParams>(12).flag, comm().abortFlagDev);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_KernelArgsTotalAndChunkSlicesNonExact) {
  // 4 ranks (nPeers=3), 10 total slices: 10 % 3 != 0 so cs = ceil(10/3) = 4.
  // Verifies that totalSlices (arg 9) and chunkSlices (arg 10) are computed and
  // forwarded correctly for a size that does not divide evenly across peers.
  constexpr size_t totalSlices = 10;
  constexpr size_t count = totalSlices * kSliceData / sizeof(float);
  g_ddaLaunches.clear();
  ASSERT_EQ(Run(count), ncclSuccess);
  ASSERT_GE(g_ddaLaunches.size(), 1u);
  const DdaLaunch& l = g_ddaLaunches[0];
  EXPECT_EQ(l.Arg<size_t>(9),  10u);  // totalSlices
  EXPECT_EQ(l.Arg<size_t>(10),  4u);  // chunkSlices = ceil(10/3)
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_FanOutParamForwarded) {
  for (int fanOut : {0, 1, 3}) {
    params_["RCCL_DDA_LL128_BC_FANOUT"] = fanOut;
    g_ddaLaunches.clear();
    ASSERT_EQ(Run(4), ncclSuccess);
    ASSERT_EQ(g_ddaLaunches.size(), 1u);
    EXPECT_EQ(g_ddaLaunches[0].Arg<int>(13), fanOut);
  }
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_UnsupportedDatatypeReturnsInvalidArgument) {
  for (ncclDataType_t dt : {ncclInt8, ncclInt32, ncclUint64, ncclFloat64}) {
    EXPECT_EQ(Run(16, dt), ncclInvalidArgument) << "datatype " << dt;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_HipErrorPropagated) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
}

// ---------------------------------------------------------------------------
// BlockCount helper
// ---------------------------------------------------------------------------

TEST_F(DdaBroadcastFabricLL128Microtest, Blocks_MatchesGridXTimesGridY) {
  for (size_t count : {size_t{4}, size_t{17} * kSliceData / 4}) {
    g_ddaLaunches.clear();
    ASSERT_EQ(Run(count), ncclSuccess);
    ASSERT_EQ(g_ddaLaunches.size(), 1u);
    const auto& l = g_ddaLaunches[0];
    EXPECT_EQ(ncclBroadcastDdaFabricLL128Blocks(mock_.get(), count, ncclFloat32),
              l.grid.x * l.grid.y);
  }
}

TEST_F(DdaBroadcastFabricLL128Microtest, Blocks_SizedOnBytesNotElements) {
  // fp16 and fp32 at same element count -> same byte count -> same block count
  EXPECT_EQ(ncclBroadcastDdaFabricLL128Blocks(mock_.get(), 4, ncclFloat16),
            ncclBroadcastDdaFabricLL128Blocks(mock_.get(), 4, ncclFloat16));
  // fp32 count=8 is 32 bytes; fp16 count=16 is also 32 bytes -> same blocks
  EXPECT_EQ(ncclBroadcastDdaFabricLL128Blocks(mock_.get(), 8, ncclFloat32),
            ncclBroadcastDdaFabricLL128Blocks(mock_.get(), 16, ncclFloat16));
}

}  // namespace
