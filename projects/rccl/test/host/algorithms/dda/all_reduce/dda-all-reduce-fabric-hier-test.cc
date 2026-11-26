/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/all_reduce/dda_all_reduce_fabric_hier.cu, the fused host-hierarchical
// AllReduce kernel: eligibility, and the kernel, round size and layout each launch passes.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../common/DdaFabricTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"

#include DDA_ALL_REDUCE_FABRIC_HIER_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {
int64_t g_ddaEnable = 1;
}  // namespace

int64_t rcclParamDdaEnable() { return g_ddaEnable; }

namespace {

using dda::common::FabricGpuBarrier;
using dda::common::HierFusedLayout;
using nccl_dda_detail::DdaFabricBarrierState;
using RcclUnitTesting::DdaFabricMockComm;

void* const kSendbuff = reinterpret_cast<void*>(0x1000);
void* const kRecvbuff = reinterpret_cast<void*>(0x2000);
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);

// One 16-byte vector per local rank: 8 ranks per host take fp32 counts in steps of 32 and 16-bit ones in steps of 64.
constexpr size_t kFp32Granule = 32;
constexpr size_t kHalfGranule = 64;

template <typename T, int P, int N>
const void* Hier() {
  return KernelPtr(&dda::common::ddaAllReduceHierFabric<T, P, N, true>);
}

template <typename T, int P, int N>
const void* HierUnaligned() {
  return KernelPtr(&dda::common::ddaAllReduceHierFabric<T, P, N, false>);
}

class DdaAllReduceFabricHierMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hipGetLastError = [] { return hipSuccess; };
    comm().ddaScratchBytes = size_t{1} << 30;
    comm().ddaFabricBarrierState = barrier_;
    comm().addonStopEvent = kStopEvent;
  }

  void TearDown() override {
    g_ddaEnable = 1;
    ResetDdaCollSeams();
    ResetHipFakes();
  }

  ncclComm& comm() { return mock_.comm; }

  bool Eligible(size_t count, ncclDataType_t datatype = ncclFloat32, int nHosts = 2, int ranksPerHost = 8,
                ncclRedOp_t op = ncclSum) {
    return ncclAllReduceDdaFabricHierEligible(mock_.get(), count, datatype, op, nHosts, ranksPerHost);
  }

  // Launches as the last local rank of the last host, with hosts of consecutive ranks.
  const DdaLaunch& RunOne(size_t count, ncclDataType_t datatype, int nHosts, int ranksPerHost,
                          const void* sendbuff = kSendbuff, void* recvbuff = kRecvbuff) {
    const int hostIndex = nHosts - 1;
    const int localRank = ranksPerHost - 1;
    std::vector<int> intra(ranksPerHost), inter(nHosts);
    for (int l = 0; l < ranksPerHost; l++) intra[l] = hostIndex * ranksPerHost + l;
    for (int h = 0; h < nHosts; h++) inter[h] = h * ranksPerHost + localRank;
    g_ddaLaunches.clear();
    EXPECT_EQ(ncclAllReduceDdaFabricHier(sendbuff, recvbuff, count, datatype, mock_.get(), kStream, nHosts,
                                         ranksPerHost, hostIndex, localRank, intra.data(), inter.data()),
              ncclSuccess);
    EXPECT_EQ(g_ddaLaunches.size(), 1u);
    return g_ddaLaunches.at(0);
  }

  DdaFabricMockComm mock_;
  DdaFabricBarrierState* const barrier_ = SharedBarrierState<DdaFabricBarrierState>(0x5a);
};

TEST_F(DdaAllReduceFabricHierMicrotest, Eligible_SumOnFp32Fp16Bf16) {
  EXPECT_TRUE(Eligible(kFp32Granule));
  EXPECT_TRUE(Eligible(kHalfGranule, ncclFloat16));
  EXPECT_TRUE(Eligible(kHalfGranule, ncclBfloat16));
  for (ncclRedOp_t op : {ncclProd, ncclMax, ncclMin, ncclAvg}) {
    EXPECT_FALSE(Eligible(kFp32Granule, ncclFloat32, 2, 8, op)) << "op " << op;
  }
  for (ncclDataType_t datatype : {ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclFloat64,
                                  ncclFloat8e4m3, ncclFloat8e5m2}) {
    EXPECT_FALSE(Eligible(1024, datatype)) << "datatype " << datatype;
  }
}

TEST_F(DdaAllReduceFabricHierMicrotest, Eligible_NeedsCommAndCount) {
  EXPECT_FALSE(ncclAllReduceDdaFabricHierEligible(nullptr, kFp32Granule, ncclFloat32, ncclSum, 2, 8));
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaAllReduceFabricHierMicrotest, Eligible_DdaGates) {
  g_ddaEnable = 0;
  EXPECT_FALSE(Eligible(kFp32Granule)) << "RCCL_DDA_ENABLE=0";
  g_ddaEnable = 1;
  comm().config.launchOrderImplicit = 1;
  EXPECT_TRUE(Eligible(kFp32Granule)) << "the addon launch keeps implicit launch order";
  comm().config.launchOrderImplicit = 0;
  comm().ddaDisabledAfterAbort = true;
  EXPECT_FALSE(Eligible(kFp32Granule)) << "DDA disabled after an abort";
  comm().ddaDisabledAfterAbort = false;
  EXPECT_TRUE(Eligible(kFp32Granule));
}

TEST_F(DdaAllReduceFabricHierMicrotest, Eligible_EachMissingFabricResourceRejects) {
  for (int missing = 0; missing < 4; ++missing) {
    mock_.setFabricResourcesPresent(true);
    if (missing == 0) comm().ddaFabricMemHandler = nullptr;
    if (missing == 1) comm().ddaFabricBarrierState = nullptr;
    if (missing == 2) comm().ddaScratch = nullptr;
    if (missing == 3) comm().ddaPeerPtrsDev = nullptr;
    EXPECT_FALSE(Eligible(kFp32Granule)) << "missing resource " << missing;
  }
}

TEST_F(DdaAllReduceFabricHierMicrotest, Eligible_HostAndRankCaps) {
  constexpr size_t count = 16 * 16 * 4;
  EXPECT_FALSE(Eligible(count, ncclFloat32, 0, 8));
  EXPECT_TRUE(Eligible(count, ncclFloat32, 1, 8));
  EXPECT_TRUE(Eligible(count, ncclFloat32, 16, 8));
  EXPECT_FALSE(Eligible(count, ncclFloat32, 17, 8));
  EXPECT_FALSE(Eligible(count, ncclFloat32, 2, 1));
  EXPECT_TRUE(Eligible(count, ncclFloat32, 2, 2));
  EXPECT_TRUE(Eligible(count, ncclFloat32, 2, 16));
  EXPECT_FALSE(Eligible(17 * count, ncclFloat32, 2, 17));
}

TEST_F(DdaAllReduceFabricHierMicrotest, Eligible_CountInWholeVectorsPerLocalRank) {
  EXPECT_FALSE(Eligible(kFp32Granule / 2));
  EXPECT_FALSE(Eligible(kFp32Granule + 4));
  EXPECT_TRUE(Eligible(3 * kFp32Granule));
  EXPECT_FALSE(Eligible(kFp32Granule, ncclFloat16));
  EXPECT_FALSE(Eligible(kHalfGranule + 8, ncclBfloat16));
  EXPECT_TRUE(Eligible(kFp32Granule, ncclFloat32, 2, 8));
  EXPECT_TRUE(Eligible(16, ncclFloat32, 2, 4)) << "4 ranks per host take fp32 counts in steps of 16";
}

// Two banks of [8 host | 2 inter-host | 8 gathered] slots must hold at least one granule per round: 2 * 32 * 18 / 8
// fp32 elements, 576 bytes.
TEST_F(DdaAllReduceFabricHierMicrotest, Eligible_ScratchHoldsOneGranulePerRound) {
  comm().ddaScratchBytes = 576;
  EXPECT_TRUE(Eligible(1024));
  comm().ddaScratchBytes = 575;
  EXPECT_FALSE(Eligible(1024));
}

TEST_F(DdaAllReduceFabricHierMicrotest, Launch_UnrolledFor8RanksPerHostOnOneTwoFourAndEightHosts) {
  struct Case {
    int nHosts, ranksPerHost;
    const void* kernel;
  };
  const Case cases[] = {{1, 8, Hier<float, 8, 1>()}, {2, 8, Hier<float, 8, 2>()}, {4, 8, Hier<float, 8, 4>()},
                        {8, 8, Hier<float, 8, 8>()}, {2, 4, Hier<float, 0, 0>()}, {3, 8, Hier<float, 0, 0>()},
                        {2, 16, Hier<float, 0, 0>()}, {16, 8, Hier<float, 0, 0>()}};
  for (const Case& c : cases) {
    EXPECT_EQ(RunOne(size_t{1} << 20, ncclFloat32, c.nHosts, c.ranksPerHost).kernel, c.kernel)
        << c.nHosts << " hosts of " << c.ranksPerHost;
  }
  EXPECT_EQ(RunOne(size_t{1} << 20, ncclFloat16, 2, 8).kernel, (Hier<half, 8, 2>()));
  EXPECT_EQ(RunOne(size_t{1} << 20, ncclBfloat16, 2, 8).kernel, (Hier<bf16, 8, 2>()));
}

TEST_F(DdaAllReduceFabricHierMicrotest, Launch_PassesBuffersRoundLayoutBarrierAndStopEvent) {
  const size_t count = size_t{1} << 20;
  const DdaLaunch& launch = RunOne(count, ncclFloat32, 2, 8);
  ASSERT_EQ(launch.args.size(), 7u);
  EXPECT_EQ(launch.stream, kStream);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  EXPECT_EQ(comm().addonStopEvent, nullptr) << "the launch takes the stop event";
  EXPECT_EQ(launch.Arg<float* const*>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch.Arg<float*>(1), kRecvbuff);
  EXPECT_EQ(launch.Arg<const float*>(2), kSendbuff);
  EXPECT_EQ(launch.Arg<size_t>(3), count);
  EXPECT_EQ(launch.Arg<size_t>(4), count) << "one round when the message fits";
  const HierFusedLayout lay = launch.Arg<HierFusedLayout>(5);
  EXPECT_EQ(lay.nHosts, 2);
  EXPECT_EQ(lay.ranksPerHost, 8);
  EXPECT_EQ(lay.hostIndex, 1);
  EXPECT_EQ(lay.localRank, 7);
  for (int l = 0; l < 8; l++) EXPECT_EQ(lay.intra[l], 8 + l) << "local rank " << l;
  EXPECT_EQ(lay.inter[0], 7);
  EXPECT_EQ(lay.inter[1], 15);
  EXPECT_TRUE(SameBarrier(launch.Arg<FabricGpuBarrier>(6), barrier_->barrierHost));
}

// A round is at most 64 MiB, and two banks of [P | N | P] slices of it must fit in the DDA scratch.
TEST_F(DdaAllReduceFabricHierMicrotest, Launch_RoundCappedAt64MiBAndByScratch) {
  EXPECT_EQ(RunOne(size_t{1} << 25, ncclFloat32, 2, 8).Arg<size_t>(4), size_t{1} << 24) << "64 MiB of fp32";
  EXPECT_EQ(RunOne(size_t{1} << 26, ncclFloat16, 2, 8).Arg<size_t>(4), size_t{1} << 25) << "64 MiB of fp16";
  // 36864 fp32 elements of scratch: 18432 per bank, 8 / 18 of it per round.
  comm().ddaScratchBytes = 36864 * sizeof(float);
  EXPECT_EQ(RunOne(size_t{1} << 20, ncclFloat32, 2, 8).Arg<size_t>(4), 8192u);
}

// Each rank picks its user-buffer access from its own pointers; the scratch layout is the same either way.
TEST_F(DdaAllReduceFabricHierMicrotest, Launch_UnalignedInstanceWhenEitherBufferIsOffSixteenBytes) {
  void* const offSend = reinterpret_cast<void*>(0x1004);
  void* const offRecv = reinterpret_cast<void*>(0x2008);
  EXPECT_EQ(RunOne(size_t{1} << 20, ncclFloat32, 2, 8).kernel, (Hier<float, 8, 2>()));
  EXPECT_EQ(RunOne(size_t{1} << 20, ncclFloat32, 2, 8, offSend, kRecvbuff).kernel, (HierUnaligned<float, 8, 2>()));
  EXPECT_EQ(RunOne(size_t{1} << 20, ncclFloat32, 2, 8, kSendbuff, offRecv).kernel, (HierUnaligned<float, 8, 2>()));
  EXPECT_EQ(RunOne(size_t{1} << 20, ncclFloat32, 2, 8, offSend, offRecv).kernel, (HierUnaligned<float, 8, 2>()));
  EXPECT_EQ(RunOne(size_t{1} << 20, ncclFloat16, 4, 8, offSend, offSend).kernel, (HierUnaligned<half, 8, 4>()));
  EXPECT_EQ(RunOne(size_t{1} << 20, ncclBfloat16, 2, 4, offSend, offRecv).kernel, (HierUnaligned<bf16, 0, 0>()));
  const DdaLaunch& launch = RunOne(size_t{1} << 20, ncclFloat32, 2, 8, offSend, offRecv);
  EXPECT_EQ(launch.Arg<float*>(1), offRecv);
  EXPECT_EQ(launch.Arg<const float*>(2), offSend);
}

}  // namespace
