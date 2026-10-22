/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_ib_cast/qp_sharing.cc
// (NET_IB_CAST_QP_SHARING_CC_PATH): the shared-QP pool, the commId table and
// completion routing, plus the qp_sharing.h helpers.

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <cstring>
#include <string>
#include <vector>

#include "ScopedHook.h"
#include "fakes/ibvwrap_fakes.h"
#include "fakes/nccl_fakes.h"
#include "../common/LogCapture.hpp"

#include "fakes/param_redirect.h"
#include NET_IB_CAST_QP_SHARING_CC_PATH

namespace {

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

union ncclSocketAddress V4(const char* ip, uint16_t port) {
  union ncclSocketAddress a;
  memset(&a, 0, sizeof(a));
  a.sin.sin_family = AF_INET;
  a.sin.sin_port = htons(port);
  inet_pton(AF_INET, ip, &a.sin.sin_addr);
  return a;
}

IbCastSharedQpKey MakeKey(int groupIdx, int qpIdx, bool isSend = true, int ibDevN = 0) {
  IbCastSharedQpKey k;
  memset(&k, 0, sizeof(k));
  k.ibDevN = ibDevN;
  k.peerAddr = V4("10.0.0.7", 0);
  k.peerProcTag = 0xabcdULL;
  k.remIbDevIdx = 3;
  k.isSend = isSend;
  k.groupIdx = groupIdx;
  k.qpIdx = qpIdx;
  return k;
}

class NetIbCastQpSharingMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { ResetPool(); }
  void TearDown() override {
    ResetPool();
    ResetIbvwrapFakes();
    ResetNcclFakes();
  }

  static void ResetPool() {
    memset(g_IbCastSharedQpPool, 0, sizeof(g_IbCastSharedQpPool));
    g_IbCastSharedQpPoolCount = 0;
    g_IbCastSharedQpFreeTop = 0;
    memset(g_IbCastCommTable, 0, sizeof(g_IbCastCommTable));
    g_IbCastNextCommId = 1;
    g_IbCastCommIdFreeTop = 0;
    IbCastQpSharingGlobalEnable = false;
  }

  IbCastSharedQp* Register(const IbCastSharedQpKey& k, ibv_qp* qp = nullptr, ibv_cq* cq = nullptr,
                           int primaryIbDevN = 0, int devIndex = 0, int refcount = 1) {
    return IbCastRegisterSharedQp(&k, qp, cq, primaryIbDevN, devIndex, refcount);
  }
};

TEST_F(NetIbCastQpSharingMicrotest, StripPort_Ipv4_ClearsOnlyThePort) {
  union ncclSocketAddress a = V4("10.1.2.3", 4242);
  IbCastStripPort(&a);
  union ncclSocketAddress want = V4("10.1.2.3", 0);
  EXPECT_EQ(memcmp(&a, &want, sizeof(a)), 0);
}

TEST_F(NetIbCastQpSharingMicrotest, StripPort_Ipv6_ClearsOnlyThePort) {
  union ncclSocketAddress a;
  memset(&a, 0, sizeof(a));
  a.sin6.sin6_family = AF_INET6;
  a.sin6.sin6_port = htons(4242);
  inet_pton(AF_INET6, "fe80::1", &a.sin6.sin6_addr);
  union ncclSocketAddress want = a;
  want.sin6.sin6_port = 0;
  IbCastStripPort(&a);
  EXPECT_EQ(memcmp(&a, &want, sizeof(a)), 0);
}

TEST_F(NetIbCastQpSharingMicrotest, StripPort_OtherFamily_LeavesAddressUntouched) {
  union ncclSocketAddress a;
  memset(&a, 0x5a, sizeof(a));
  a.sa.sa_family = AF_UNIX;
  union ncclSocketAddress want = a;
  IbCastStripPort(&a);
  EXPECT_EQ(memcmp(&a, &want, sizeof(a)), 0);
}

TEST_F(NetIbCastQpSharingMicrotest, LocalProcTag_IsHostAndPidHashCombined) {
  EXPECT_EQ(IbCastLocalProcTag(), hashCombine(getHostHash(), getPidHash()));
  EXPECT_EQ(IbCastLocalProcTag(), IbCastLocalProcTag());
}

TEST_F(NetIbCastQpSharingMicrotest, KeyMatch_EveryFieldDisambiguates) {
  const IbCastSharedQpKey base = MakeKey(1, 2);
  EXPECT_TRUE(IbCastSharedQpKeyMatch(&base, &base));
  std::vector<IbCastSharedQpKey> variants(7, base);
  variants[0].ibDevN++;
  variants[1].isSend = !base.isSend;
  variants[2].groupIdx++;
  variants[3].qpIdx++;
  variants[4].remIbDevIdx++;
  variants[5].peerProcTag++;
  variants[6].peerAddr = V4("10.0.0.8", 0);
  for (size_t i = 0; i < variants.size(); i++) {
    SCOPED_TRACE(i);
    EXPECT_FALSE(IbCastSharedQpKeyMatch(&base, &variants[i]));
  }
}

TEST_F(NetIbCastQpSharingMicrotest, Register_FillsTheNextSlot) {
  ibv_qp qp{};
  ibv_cq cq{};
  Register(MakeKey(0, 0));
  IbCastSharedQp* e = Register(MakeKey(0, 1), &qp, &cq, 5, 2, 3);
  ASSERT_EQ(e, &g_IbCastSharedQpPool[1]);
  EXPECT_EQ(g_IbCastSharedQpPoolCount, 2);
  const IbCastSharedQpKey want = MakeKey(0, 1);
  EXPECT_TRUE(IbCastSharedQpKeyMatch(&e->key, &want));
  EXPECT_EQ(e->qp, &qp);
  EXPECT_EQ(e->primaryCq, &cq);
  EXPECT_EQ(e->primaryIbDevN, 5);
  EXPECT_EQ(e->devIndex, 2);
  EXPECT_EQ(e->refcount, 3);
  EXPECT_EQ(e->cqRefcount, 0);
  EXPECT_TRUE(e->used);
}

TEST_F(NetIbCastQpSharingMicrotest, Register_ReusesFreedSlotsLastFreedFirst) {
  for (int q = 0; q < 3; q++) Register(MakeKey(0, q));
  g_IbCastSharedQpPool[1].cqRefcount = 9;
  IbCastUnregisterSharedQpLocked(&g_IbCastSharedQpPool[0]);
  IbCastUnregisterSharedQpLocked(&g_IbCastSharedQpPool[1]);
  EXPECT_FALSE(g_IbCastSharedQpPool[0].used);
  EXPECT_EQ(g_IbCastSharedQpFreeTop, 2);

  EXPECT_EQ(Register(MakeKey(1, 0)), &g_IbCastSharedQpPool[1]);
  EXPECT_EQ(g_IbCastSharedQpPool[1].cqRefcount, 0);
  EXPECT_EQ(Register(MakeKey(1, 1)), &g_IbCastSharedQpPool[0]);
  EXPECT_EQ(g_IbCastSharedQpPoolCount, 3);
  EXPECT_EQ(Register(MakeKey(1, 2)), &g_IbCastSharedQpPool[3]);
}

TEST_F(NetIbCastQpSharingMicrotest, Register_PoolFull_WarnsAndReturnsNull) {
  g_IbCastSharedQpPoolCount = IBCAST_MAX_SHARED_QPS - 1;
  EXPECT_EQ(Register(MakeKey(0, 0)), &g_IbCastSharedQpPool[IBCAST_MAX_SHARED_QPS - 1]);
  IbCastSharedQp* e = &g_IbCastSharedQpPool[0];
  const std::string log = CaptureLog([&] { e = Register(MakeKey(0, 1)); });
  EXPECT_EQ(e, nullptr);
  EXPECT_EQ(g_IbCastSharedQpPoolCount, IBCAST_MAX_SHARED_QPS);
  EXPECT_TRUE(LogHas(log, ("pool full (" + std::to_string(IBCAST_MAX_SHARED_QPS) + " entries)").c_str())) << log;
}

TEST_F(NetIbCastQpSharingMicrotest, Unregister_Null_IsANoop) {
  IbCastUnregisterSharedQpLocked(nullptr);
  EXPECT_EQ(g_IbCastSharedQpFreeTop, 0);
}

TEST_F(NetIbCastQpSharingMicrotest, Find_ReturnsOnlyALiveMatchingSlot) {
  Register(MakeKey(0, 0));
  Register(MakeKey(0, 1));
  const IbCastSharedQpKey k1 = MakeKey(0, 1);
  EXPECT_EQ(IbCastFindSharedQp(&k1), &g_IbCastSharedQpPool[1]);
  g_IbCastSharedQpPool[1].used = false;
  EXPECT_EQ(IbCastFindSharedQp(&k1), nullptr);
  // A stale entry past the pool count is never scanned.
  g_IbCastSharedQpPool[2].used = true;
  g_IbCastSharedQpPool[2].key = MakeKey(0, 2);
  const IbCastSharedQpKey k2 = MakeKey(0, 2);
  EXPECT_EQ(IbCastFindSharedQp(&k2), nullptr);
}

TEST_F(NetIbCastQpSharingMicrotest, FindByQpn_MatchesQpNumberAndDirection) {
  ibv_qp sendQp{};
  sendQp.qp_num = 77;
  ibv_qp recvQp{};
  recvQp.qp_num = 77;
  Register(MakeKey(0, 0, /*isSend=*/false), nullptr);
  Register(MakeKey(0, 1, /*isSend=*/true), &sendQp);
  Register(MakeKey(0, 2, /*isSend=*/false), &recvQp);
  EXPECT_EQ(IbCastFindSharedQpByQpn(77, true), &g_IbCastSharedQpPool[1]);
  EXPECT_EQ(IbCastFindSharedQpByQpn(77, false), &g_IbCastSharedQpPool[2]);
  EXPECT_EQ(IbCastFindSharedQpByQpn(78, true), nullptr);
  g_IbCastSharedQpPool[2].used = false;
  EXPECT_EQ(IbCastFindSharedQpByQpn(77, false), nullptr);
}

TEST_F(NetIbCastQpSharingMicrotest, CountGroupQpSlots_CountsOnlyThePeersGroupInOneDirection) {
  for (int q = 0; q < 3; q++) Register(MakeKey(1, q, true, /*ibDevN=*/q));
  std::vector<IbCastSharedQpKey> others(5, MakeKey(1, 0));
  others[0].isSend = false;
  others[1].groupIdx = 2;
  others[2].remIbDevIdx = 4;
  others[3].peerProcTag = 1;
  others[4].peerAddr = V4("10.0.0.9", 0);
  for (const auto& k : others) Register(k);
  Register(MakeKey(1, 3))->used = false;

  const IbCastSharedQpKey k = MakeKey(1, 0);
  EXPECT_EQ(IbCastCountGroupQpSlots(&k.peerAddr, k.peerProcTag, k.remIbDevIdx, true, 1), 3);
}

TEST_F(NetIbCastQpSharingMicrotest, CountPeerTotalRefcount_SumsSlotZeroOfEveryGroup) {
  Register(MakeKey(0, 0), nullptr, nullptr, 0, 0, 2);
  Register(MakeKey(0, 1), nullptr, nullptr, 0, 0, 100);  // not slot 0
  Register(MakeKey(1, 0), nullptr, nullptr, 0, 0, 3);
  std::vector<IbCastSharedQpKey> others(4, MakeKey(2, 0));
  others[0].isSend = false;
  others[1].remIbDevIdx = 4;
  others[2].peerProcTag = 1;
  others[3].peerAddr = V4("10.0.0.9", 0);
  for (const auto& k : others) Register(k, nullptr, nullptr, 0, 0, 1000);
  Register(MakeKey(3, 0), nullptr, nullptr, 0, 0, 10000)->used = false;

  const IbCastSharedQpKey k = MakeKey(0, 0);
  EXPECT_EQ(IbCastCountPeerTotalRefcount(0, &k.peerAddr, k.peerProcTag, k.remIbDevIdx, true), 5);
}

TEST_F(NetIbCastQpSharingMicrotest, AllocCommId_IsSequentialThenReusesTheLastFreed) {
  int a = 0, b = 0, c = 0;
  ASSERT_EQ(IbCastAllocCommId(&a, true), 1);
  ASSERT_EQ(IbCastAllocCommId(&b, false), 2);
  ASSERT_EQ(IbCastAllocCommId(&c, true), 3);
  EXPECT_EQ(g_IbCastCommTable[2].comm, &b);
  EXPECT_FALSE(g_IbCastCommTable[2].isSend);
  EXPECT_TRUE(g_IbCastCommTable[3].isSend);
  EXPECT_TRUE(g_IbCastCommTable[3].used);

  IbCastFreeCommId(1);
  IbCastFreeCommId(2);
  EXPECT_FALSE(g_IbCastCommTable[2].used);
  EXPECT_EQ(g_IbCastCommTable[2].comm, nullptr);
  EXPECT_EQ(IbCastAllocCommId(&c, true), 2);
  EXPECT_EQ(IbCastAllocCommId(&c, true), 1);
  EXPECT_EQ(IbCastAllocCommId(&c, true), 4);
}

TEST_F(NetIbCastQpSharingMicrotest, AllocCommId_Exhausted_WarnsAndFallsBackToZero) {
  g_IbCastNextCommId = IBCAST_MAX_COMMS - 1;
  int comm = 0;
  EXPECT_EQ(IbCastAllocCommId(&comm, true), IBCAST_MAX_COMMS - 1);
  uint16_t id = 1;
  const std::string log = CaptureLog([&] { id = IbCastAllocCommId(&comm, true); });
  EXPECT_EQ(id, 0);
  EXPECT_FALSE(g_IbCastCommTable[0].used);
  EXPECT_TRUE(LogHas(log, ("commId pool exhausted (max " + std::to_string(IBCAST_MAX_COMMS) + ")").c_str())) << log;
}

TEST_F(NetIbCastQpSharingMicrotest, FreeCommId_OutOfRange_IsANoop) {
  IbCastFreeCommId(0);
  IbCastFreeCommId(IBCAST_MAX_COMMS);
  IbCastFreeCommIdLocked(0);
  IbCastFreeCommIdLocked(IBCAST_MAX_COMMS);
  EXPECT_EQ(g_IbCastCommIdFreeTop, 0);
  IbCastFreeCommIdLocked(IBCAST_MAX_COMMS - 1);
  EXPECT_EQ(g_IbCastCommIdFreeTop, 1);
}

TEST_F(NetIbCastQpSharingMicrotest, CommIdCodecs_RoundTripAndClearStaleBits) {
  const uint64_t wrId = 0xffff00000000002aULL;
  const uint64_t enc = IbCastEncodeCommId(wrId, 0x1234);
  EXPECT_EQ(enc, 0x123400000000002aULL);
  EXPECT_EQ(IbCastStripCommId(enc), 0x2aULL);
  EXPECT_EQ(IbCastEncodeCommIdImmData(0x1ab, 0x4567), 0x4567abu);
}

TEST_F(NetIbCastQpSharingMicrotest, DepthMultiplier_ScalesOnlyWhenSharingAndNeverBelowOne) {
  int64_t mult = 5;
  ScopedHook param(g_loadParam, [&](const char* env, int64_t deft) {
    return std::string(env) == "RCCL_IB_QP_DEPTH_MULTIPLIER" ? mult : deft;
  });
  EXPECT_EQ(IbCastQpSharingDepthMultiplier(), 1);
  IbCastQpSharingGlobalEnable = true;
  EXPECT_EQ(IbCastQpSharingDepthMultiplier(), 5);
  mult = 0;
  EXPECT_EQ(IbCastQpSharingDepthMultiplier(), 1);
}

TEST_F(NetIbCastQpSharingMicrotest, Params_DefaultsMatchTheDocumentedValues) {
  EXPECT_EQ(rcclParamIbCastQpSharingEnable(), 0);
  EXPECT_EQ(rcclParamIbCastCommNGroups(), 4);
  EXPECT_EQ(rcclParamIbCastQpDepthMultiplier(), 8);
  EXPECT_EQ(rcclParamIbCastQpSharingValidatePool(), 0);
}

TEST_F(NetIbCastQpSharingMicrotest, CommRoles_FollowCommIdAndPrimaryFlag) {
  ncclIbNetCommBase base;
  base.qpSharing.peerProcTag = 0x77;
  base.qpSharing.netIbCommId = 9;
  base.qpSharing.isPrimary = true;
  IbCastCommInitSharingFields(&base);
  EXPECT_EQ(base.qpSharing.netIbCommId, 0);
  EXPECT_FALSE(base.qpSharing.isPrimary);
  EXPECT_EQ(base.qpSharing.groupIdx, -1);
  EXPECT_EQ(base.qpSharing.remIbDevIdx, -1);
  EXPECT_EQ(base.qpSharing.groupNqps, 0);
  EXPECT_EQ(base.qpSharing.peerProcTag, 0x77u);

  IbCastQpSharingGlobalEnable = true;
  EXPECT_FALSE(IbCastCommIsSharing(&base));
  EXPECT_FALSE(IbCastCommIsPrimary(&base));
  EXPECT_FALSE(IbCastCommIsSecondary(&base));
  base.qpSharing.netIbCommId = 3;
  EXPECT_TRUE(IbCastCommIsSharing(&base));
  EXPECT_TRUE(IbCastCommIsSecondary(&base));
  base.qpSharing.isPrimary = true;
  EXPECT_TRUE(IbCastCommIsPrimary(&base));
  EXPECT_FALSE(IbCastCommIsSecondary(&base));
  IbCastQpSharingGlobalEnable = false;
  EXPECT_FALSE(IbCastCommIsSharing(&base));
}

class NetIbCastQpSharingRouteMicrotest : public NetIbCastQpSharingMicrotest {
 protected:
  void SetUp() override {
    NetIbCastQpSharingMicrotest::SetUp();
    send_ = static_cast<ncclIbSendComm*>(calloc(1, sizeof(ncclIbSendComm)));
    recv_ = static_cast<ncclIbRecvComm*>(calloc(1, sizeof(ncclIbRecvComm)));
    sendId_ = IbCastAllocCommId(send_, true);
    recvId_ = IbCastAllocCommId(recv_, false);
    IbCastQpSharingGlobalEnable = true;
    origin_.qpSharing.netIbCommId = sendId_;
  }
  void TearDown() override {
    free(send_);
    free(recv_);
    NetIbCastQpSharingMicrotest::TearDown();
  }
  ncclIbSendComm* send_ = nullptr;
  ncclIbRecvComm* recv_ = nullptr;
  uint16_t sendId_ = 0, recvId_ = 0;
  ncclIbNetCommBase origin_{};
};

TEST_F(NetIbCastQpSharingRouteMicrotest, FromWrId_ResolvesEachCommByItsDirection) {
  EXPECT_EQ(IbCastRouteCommFromWrId(&origin_, IbCastEncodeCommId(5, sendId_)), &send_->base);
  EXPECT_EQ(IbCastRouteCommFromWrId(&origin_, IbCastEncodeCommId(5, recvId_)), &recv_->base);
}

TEST_F(NetIbCastQpSharingRouteMicrotest, FromWrId_RejectsZeroFreedAndOutOfRangeIds) {
  EXPECT_EQ(IbCastRouteCommFromWrId(&origin_, 5), nullptr);
  EXPECT_EQ(IbCastRouteCommFromWrId(&origin_, IbCastEncodeCommId(5, IBCAST_MAX_COMMS)), nullptr);
  IbCastFreeCommId(recvId_);
  EXPECT_EQ(IbCastRouteCommFromWrId(&origin_, IbCastEncodeCommId(5, recvId_)), nullptr);
}

TEST_F(NetIbCastQpSharingRouteMicrotest, FromWrId_NonSharingOriginNeverDecodes) {
  origin_.qpSharing.netIbCommId = 0;
  EXPECT_EQ(IbCastRouteCommFromWrId(&origin_, IbCastEncodeCommId(5, sendId_)), nullptr);
}

TEST_F(NetIbCastQpSharingRouteMicrotest, FromImmData_ResolvesEachCommByItsDirection) {
  EXPECT_EQ(IbCastRouteCommFromImmData(&origin_, IbCastEncodeCommIdImmData(1, sendId_)), &send_->base);
  EXPECT_EQ(IbCastRouteCommFromImmData(&origin_, IbCastEncodeCommIdImmData(1, recvId_)), &recv_->base);
}

TEST_F(NetIbCastQpSharingRouteMicrotest, FromImmData_RejectsZeroFreedAndOutOfRangeIds) {
  EXPECT_EQ(IbCastRouteCommFromImmData(&origin_, 1), nullptr);
  EXPECT_EQ(IbCastRouteCommFromImmData(&origin_, IbCastEncodeCommIdImmData(1, IBCAST_MAX_COMMS)), nullptr);
  IbCastFreeCommId(sendId_);
  EXPECT_EQ(IbCastRouteCommFromImmData(&origin_, IbCastEncodeCommIdImmData(1, sendId_)), nullptr);
}

TEST_F(NetIbCastQpSharingRouteMicrotest, FromImmData_NonSharingOriginNeverDecodes) {
  IbCastQpSharingGlobalEnable = false;
  EXPECT_EQ(IbCastRouteCommFromImmData(&origin_, IbCastEncodeCommIdImmData(1, sendId_)), nullptr);
}

class NetIbCastQpSharingCleanupMicrotest : public NetIbCastQpSharingMicrotest {
 protected:
  void SetUp() override {
    NetIbCastQpSharingMicrotest::SetUp();
    for (int d : {0, 1}) {
      IbCastDevs[d].pdRefs = 2;
      IbCastDevs[d].pd = &pd_[d];
    }
  }
  void TearDown() override {
    for (int d : {0, 1}) {
      IbCastDevs[d].pdRefs = 0;
      IbCastDevs[d].pd = nullptr;
    }
    NetIbCastQpSharingMicrotest::TearDown();
  }
  ibv_pd pd_[2] = {};
  ibv_cq cq_[2] = {};
};

TEST_F(NetIbCastQpSharingCleanupMicrotest, CleanupGroupCqs_DestroysEachGroupCqOnceAndFreesTheGroup) {
  // Group 1 spans two devices with two QPs each, plus a flush slot and two
  // slots of other groups that must survive.
  IbCastSharedQp* slot0 = Register(MakeKey(1, 0, true, 0), nullptr, &cq_[0], 0);
  Register(MakeKey(1, 1, true, 1), nullptr, &cq_[1], 1);
  Register(MakeKey(1, 2, true, 0), nullptr, &cq_[0], 0);
  Register(MakeKey(1, 3, true, 1), nullptr, &cq_[1], 1);
  Register(MakeKey(1, IBCAST_FLUSH_QP_IDX, true, 0), nullptr, &cq_[0], 0);
  Register(MakeKey(2, 0, true, 0), nullptr, &cq_[0], 0);
  Register(MakeKey(1, 0, false, 0), nullptr, &cq_[0], 0);
  std::vector<ibv_cq*> destroyed;
  std::vector<ibv_pd*> dealloced;
  ScopedHook dcq(g_ibvDestroyCq, [&](ibv_cq* cq) {
    destroyed.push_back(cq);
    return ncclSuccess;
  });
  ScopedHook dpd(g_ibvDeallocPd, [&](ibv_pd* pd) {
    dealloced.push_back(pd);
    return ncclSuccess;
  });
  IbCastDevs[1].pdRefs = 1;

  IbCastCleanupGroupCqs(slot0);

  EXPECT_EQ(destroyed, (std::vector<ibv_cq*>{&cq_[0], &cq_[1]}));
  EXPECT_EQ(IbCastDevs[0].pdRefs, 1);
  EXPECT_EQ(IbCastDevs[1].pdRefs, 0);
  EXPECT_EQ(dealloced, (std::vector<ibv_pd*>{&pd_[1]}));
  for (int i = 0; i < 4; i++) EXPECT_FALSE(g_IbCastSharedQpPool[i].used) << i;
  for (int i = 4; i < 7; i++) EXPECT_TRUE(g_IbCastSharedQpPool[i].used) << i;
  EXPECT_EQ(g_IbCastSharedQpFreeTop, 4);
}

TEST_F(NetIbCastQpSharingCleanupMicrotest, CleanupGroupCqs_DestroyFailureWarnsAndStillReleases) {
  IbCastSharedQp* slot0 = Register(MakeKey(0, 0), nullptr, &cq_[0], 0);
  Register(MakeKey(0, 1), nullptr, nullptr, 0);
  ScopedHook dcq(g_ibvDestroyCq, [](ibv_cq*) { return ncclSystemError; });
  const std::string log = CaptureLog([&] { IbCastCleanupGroupCqs(slot0); });
  EXPECT_EQ(dcq.calls, 1);
  EXPECT_TRUE(LogHas(log, "ibv_destroy_cq FAILED")) << log;
  EXPECT_EQ(IbCastDevs[0].pdRefs, 1);
  EXPECT_FALSE(g_IbCastSharedQpPool[0].used);
  EXPECT_FALSE(g_IbCastSharedQpPool[1].used);
}

TEST_F(NetIbCastQpSharingMicrotest, ValidatePool_SharingOff_ChecksNothing) {
  Register(MakeKey(0, 0));
  const std::string log = CaptureLog([] { IbCastValidateSharedQpPool(); });
  EXPECT_EQ(log, "");
}

TEST_F(NetIbCastQpSharingMicrotest, ValidatePool_CleanPool_ReportsCleanShutdown) {
  IbCastQpSharingGlobalEnable = true;
  Register(MakeKey(0, 0));
  IbCastUnregisterSharedQpLocked(&g_IbCastSharedQpPool[0]);
  RcclUnitTesting::ScopedDebugLogging info(NCCL_LOG_INFO, ~0ULL);
  const std::string log = CaptureLog([] { IbCastValidateSharedQpPool(); });
  EXPECT_TRUE(LogHas(log, "clean shutdown")) << log;
  EXPECT_FALSE(LogHas(log, "LEAK")) << log;
}

#ifdef NDEBUG
TEST_F(NetIbCastQpSharingMicrotest, ValidatePool_Leaks_AreReported) {
  IbCastQpSharingGlobalEnable = true;
  Register(MakeKey(0, 0));
  int comm = 0;
  IbCastAllocCommId(&comm, true);
  const std::string log = CaptureLog([] { IbCastValidateSharedQpPool(); });
  EXPECT_TRUE(LogHas(log, "IB CAST POOL LEAK: slot=0")) << log;
  EXPECT_TRUE(LogHas(log, "IB CAST POOL LEAK: commId=1")) << log;
  EXPECT_TRUE(LogHas(log, "1 leaked QP slots, 1 leaked commIds")) << log;
}
#else
TEST_F(NetIbCastQpSharingMicrotest, ValidatePool_LeakedSlot_Asserts) {
  IbCastQpSharingGlobalEnable = true;
  Register(MakeKey(0, 0));
  EXPECT_DEATH(IbCastValidateSharedQpPool(), "IB CAST POOL LEAK: slot=0(.|\n)*leaked QP slots at shutdown");
}

TEST_F(NetIbCastQpSharingMicrotest, ValidatePool_LeakedCommId_Asserts) {
  IbCastQpSharingGlobalEnable = true;
  int comm = 0;
  IbCastAllocCommId(&comm, true);
  EXPECT_DEATH(IbCastValidateSharedQpPool(), "IB CAST POOL LEAK: commId=1(.|\n)*leaked commIds at shutdown");
}
#endif

}  // namespace
