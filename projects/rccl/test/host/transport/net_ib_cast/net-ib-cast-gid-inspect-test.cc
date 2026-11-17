/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_ib_cast/net_ib_cast_gid_inspect.cc
// (NET_IB_CAST_GID_INSPECT_CC_PATH), the fault-injection build's GID inspection hooks.

#include <gtest/gtest.h>

#include <cstring>
#include <memory>

#include "fakes/ibvwrap_fakes.h"
#include "net-ib-cast-test-common.h"

#include NET_IB_CAST_GID_INSPECT_CC_PATH

namespace {

class NetIbCastGidInspectMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    IbCastNDevs = 2;
    send_->base.isSend = true;
    send_->base.vProps.ndevs = 2;
    send_->devs[1].base.ibDevN = 1;
  }
  void TearDown() override {
    NetIbCastResetGlobals();
    ResetIbvwrapFakes();
  }

  static ncclIbGidState State(int linkLayer, int gidIndex, uint8_t fill) {
    ncclIbGidState s{};
    s.linkLayer = linkLayer;
    s.gidIndex = gidIndex;
    memset(s.gid, fill, sizeof(s.gid));
    return s;
  }

  ZeroedPtr<ncclIbSendComm> send_;
};

TEST_F(NetIbCastGidInspectMicrotest, Dev_SetThenGet_RoundTripsAndReportsNoComm) {
  ncclIbGidState in = State(IBV_LINK_LAYER_ETHERNET, 3, 0x5a);
  ASSERT_EQ(ncclIbCastGidSetDev(1, &in), ncclSuccess);
  EXPECT_EQ(IbCastDevs[1].gidInfo.localGidIndex, 3);
  EXPECT_EQ(IbCastDevs[1].gidInfo.link_layer, IBV_LINK_LAYER_ETHERNET);
  ncclIbGidState out = State(0, 0, 0);
  out.ibDev = 7;
  ASSERT_EQ(ncclIbCastGidGetDev(1, &out), ncclSuccess);
  EXPECT_EQ(out.gidIndex, 3);
  EXPECT_EQ(out.linkLayer, IBV_LINK_LAYER_ETHERNET);
  EXPECT_EQ(memcmp(out.gid, in.gid, sizeof(out.gid)), 0);
  EXPECT_EQ(out.ibDev, -1);
}

TEST_F(NetIbCastGidInspectMicrotest, Dev_OutOfRangeOrNull_IsRejected) {
  ncclIbGidState s{};
  EXPECT_EQ(ncclIbCastGidGetDev(-1, &s), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidGetDev(2, &s), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidGetDev(0, nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidSetDev(-1, &s), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidSetDev(2, &s), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidSetDev(0, nullptr), ncclInvalidArgument);
}

TEST_F(NetIbCastGidInspectMicrotest, Comm_SetThenGet_UsesThatDevicesBaseAndReportsItsIbDev) {
  ncclIbGidState in = State(IBV_LINK_LAYER_INFINIBAND, 9, 0x11);
  ASSERT_EQ(ncclIbCastGidSetComm(send_.get(), 1, &in), ncclSuccess);
  EXPECT_EQ(send_->devs[1].base.gidInfo.localGidIndex, 9);
  EXPECT_EQ(send_->devs[0].base.gidInfo.localGidIndex, 0);
  ncclIbGidState out{};
  ASSERT_EQ(ncclIbCastGidGetComm(send_.get(), 1, &out), ncclSuccess);
  EXPECT_EQ(out.gidIndex, 9);
  EXPECT_EQ(out.ibDev, 1);
  EXPECT_EQ(memcmp(out.gid, in.gid, sizeof(out.gid)), 0);
}

TEST_F(NetIbCastGidInspectMicrotest, Comm_RecvCommResolvesItsOwnDeviceLayout) {
  ZeroedPtr<ncclIbRecvComm> recv;
  recv->base.vProps.ndevs = 1;
  recv->devs[0].base.ibDevN = 4;
  recv->devs[0].base.gidInfo.localGidIndex = 6;
  ncclIbGidState out{};
  ASSERT_EQ(ncclIbCastGidGetComm(recv.get(), 0, &out), ncclSuccess);
  EXPECT_EQ(out.gidIndex, 6);
  EXPECT_EQ(out.ibDev, 4);
}

TEST_F(NetIbCastGidInspectMicrotest, Comm_BadCommIndexOrNull_IsRejected) {
  ncclIbGidState s{};
  EXPECT_EQ(ncclIbCastGidGetComm(nullptr, 0, &s), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidGetComm(send_.get(), -1, &s), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidGetComm(send_.get(), 2, &s), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidGetComm(send_.get(), 0, nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidSetComm(nullptr, 0, &s), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidSetComm(send_.get(), 2, &s), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidSetComm(send_.get(), 0, nullptr), ncclInvalidArgument);
}

TEST_F(NetIbCastGidInspectMicrotest, QpState_ReadsBackTheSgidOfEachLiveQp) {
  ibv_qp qps[2] = {};
  send_->base.nqps = 3;
  send_->base.qps[0] = {};
  send_->base.qps[0].qp = &qps[0];
  send_->base.qps[0].devIndex = 0;
  send_->base.qps[0].rtrAttr.localGidIndex = 2;
  send_->base.qps[1].qp = nullptr;  // not created: reported, but not queried
  send_->base.qps[1].devIndex = 1;
  send_->base.qps[1].rtrAttr.localGidIndex = 5;
  send_->base.qps[2].qp = &qps[1];
  send_->base.qps[2].devIndex = 1;
  ScopedHook query(g_ibvQueryQp, [&](struct ibv_qp* qp, struct ibv_qp_attr* attr, int mask, struct ibv_qp_init_attr*) {
    EXPECT_EQ(mask, IBV_QP_AV);
    if (qp == &qps[1]) return ncclSystemError;
    attr->ah_attr.grh.sgid_index = 4;
    return ncclSuccess;
  });
  ncclIbGidQpState out;
  memset(&out, 0xff, sizeof(out));
  ASSERT_EQ(ncclIbCastGidGetQpState(send_.get(), &out), ncclSuccess);
  EXPECT_EQ(query.calls, 2);
  EXPECT_EQ(out.nqps, 3);
  EXPECT_EQ(out.devIndex[1], 1);
  EXPECT_EQ(out.rtrGidIndex[0], 2);
  EXPECT_EQ(out.rtrGidIndex[1], 5);
  EXPECT_TRUE(out.queryOk[0]);
  EXPECT_EQ(out.sgidIndex[0], 4);
  EXPECT_FALSE(out.queryOk[1]);
  EXPECT_FALSE(out.queryOk[2]);
  EXPECT_EQ(out.sgidIndex[3], 0);  // cleared past nqps
}

TEST_F(NetIbCastGidInspectMicrotest, QpState_NqpsIsClampedToTheQpArray) {
  ncclIbGidQpState out;
  send_->base.nqps = -1;
  ASSERT_EQ(ncclIbCastGidGetQpState(send_.get(), &out), ncclSuccess);
  EXPECT_EQ(out.nqps, 0);
  send_->base.nqps = NCCL_IB_MAX_QPS + 5;
  ASSERT_EQ(ncclIbCastGidGetQpState(send_.get(), &out), ncclSuccess);
  EXPECT_EQ(out.nqps, NCCL_IB_MAX_QPS);
  EXPECT_EQ(ncclIbCastGidGetQpState(nullptr, &out), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidGetQpState(send_.get(), nullptr), ncclInvalidArgument);
}

class NetIbCastGidInspectResiliencyMicrotest : public NetIbCastGidInspectMicrotest {
 protected:
  void SetUp() override {
    NetIbCastGidInspectMicrotest::SetUp();
    res_ = std::make_unique<ncclIbResiliency>();
    res_->ndevs = 2;
    send_->base.resiliency = res_.get();
  }
  std::unique_ptr<ncclIbResiliency> res_;
};

TEST_F(NetIbCastGidInspectResiliencyMicrotest, DevState_ReportsTheResiliencyStateOfThatDevice) {
  res_->devs[1].state.store(ncclIbResiliencyDevStateRecoveryInProgress);
  int state = -1;
  ASSERT_EQ(ncclIbCastGidGetDevState(send_.get(), 1, &state), ncclSuccess);
  EXPECT_EQ(state, (int)ncclIbResiliencyDevStateRecoveryInProgress);
  EXPECT_EQ(ncclIbCastGidGetDevState(send_.get(), 2, &state), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidGetDevState(send_.get(), -1, &state), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidGetDevState(send_.get(), 0, nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidGetDevState(nullptr, 0, &state), ncclInvalidArgument);
  send_->base.resiliency = nullptr;
  EXPECT_EQ(ncclIbCastGidGetDevState(send_.get(), 0, &state), ncclInvalidArgument);
}

TEST_F(NetIbCastGidInspectResiliencyMicrotest, RecoveryGidIndex_NeedsAGlobalRecoveryAh) {
  int gidIndex = -1;
  EXPECT_EQ(ncclIbCastGidGetRecoveryGidIndex(send_.get(), 1, &gidIndex), ncclInvalidArgument);  // no AH yet
  ibv_ah ah{};
  res_->portRecoveryAh[1] = &ah;
  EXPECT_EQ(ncclIbCastGidGetRecoveryGidIndex(send_.get(), 1, &gidIndex), ncclInvalidArgument);  // not global
  res_->portRecoveryAhAttr[1].is_global = 1;
  res_->portRecoveryAhAttr[1].grh.sgid_index = 7;
  ASSERT_EQ(ncclIbCastGidGetRecoveryGidIndex(send_.get(), 1, &gidIndex), ncclSuccess);
  EXPECT_EQ(gidIndex, 7);
  EXPECT_EQ(ncclIbCastGidGetRecoveryGidIndex(send_.get(), NCCL_IB_MAX_DEVS_PER_NIC, &gidIndex), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidGetRecoveryGidIndex(send_.get(), -1, &gidIndex), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidGetRecoveryGidIndex(send_.get(), 1, nullptr), ncclInvalidArgument);
  send_->base.resiliency = nullptr;
  EXPECT_EQ(ncclIbCastGidGetRecoveryGidIndex(send_.get(), 1, &gidIndex), ncclInvalidArgument);
}

TEST_F(NetIbCastGidInspectResiliencyMicrotest, ProbingGidIndex_TakesTheFirstLiveProbeQpOnThatDevice) {
  ibv_qp qps[2] = {};
  res_->nProbingQps = 3;
  res_->probingQps[0].qp = &qps[0];
  res_->probingQps[0].devIndex = 0;
  res_->probingQps[1].qp = nullptr;  // not created yet
  res_->probingQps[1].devIndex = 1;
  res_->probingQps[1].rtrAttr.localGidIndex = 9;
  res_->probingQps[2].qp = &qps[1];
  res_->probingQps[2].devIndex = 1;
  res_->probingQps[2].rtrAttr.localGidIndex = 3;
  int gidIndex = -1;
  ASSERT_EQ(ncclIbCastGidGetProbingGidIndex(send_.get(), 1, &gidIndex), ncclSuccess);
  EXPECT_EQ(gidIndex, 3);
  EXPECT_EQ(ncclIbCastGidGetProbingGidIndex(send_.get(), 2, &gidIndex), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidGetProbingGidIndex(send_.get(), 0, nullptr), ncclInvalidArgument);
  send_->base.resiliency = nullptr;
  EXPECT_EQ(ncclIbCastGidGetProbingGidIndex(send_.get(), 1, &gidIndex), ncclInvalidArgument);
}

TEST_F(NetIbCastGidInspectMicrotest, DriveQpToError_ModifiesOnlyTheStateToError) {
  ibv_qp qp{};
  send_->base.nqps = 2;
  send_->base.qps[1].qp = &qp;
  ScopedHook modify(g_ibvModifyQp, [&](struct ibv_qp* q, struct ibv_qp_attr* attr, int mask) {
    EXPECT_EQ(q, &qp);
    EXPECT_EQ(attr->qp_state, IBV_QPS_ERR);
    EXPECT_EQ(mask, IBV_QP_STATE);
    return ncclSuccess;
  });
  ASSERT_EQ(ncclIbCastGidDriveQpToError(send_.get(), 1), ncclSuccess);
  EXPECT_EQ(modify.calls, 1);
  EXPECT_EQ(ncclIbCastGidDriveQpToError(send_.get(), 0), ncclInvalidArgument);  // no QP there
  EXPECT_EQ(ncclIbCastGidDriveQpToError(send_.get(), 2), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidDriveQpToError(send_.get(), -1), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidDriveQpToError(nullptr, 0), ncclInvalidArgument);
}

TEST_F(NetIbCastGidInspectMicrotest, DriveQpToError_ModifyFails_ReturnsTheError) {
  ibv_qp qp{};
  send_->base.nqps = 1;
  send_->base.qps[0].qp = &qp;
  ScopedHook modify(g_ibvModifyQp, [](struct ibv_qp*, struct ibv_qp_attr*, int) { return ncclSystemError; });
  EXPECT_EQ(ncclIbCastGidDriveQpToError(send_.get(), 0), ncclSystemError);
}

}  // namespace
