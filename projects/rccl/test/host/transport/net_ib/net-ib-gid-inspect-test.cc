/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib/net_ib_gid_inspect.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstring>

#include "ScopedHook.h"
#include "fakes/ibvwrap_fakes.h"
#include "fakes/nccl_fakes.h"
#include "transport/net_ib/NetIbGlobals.h"

#include NET_IB_GID_INSPECT_CC_PATH

namespace {

class NetIbGidInspectMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ncclNIbDevs = 1;
    ncclIbDevs[0].gidInfo = {};
  }

  void TearDown() override {
    NetIbResetGlobals();
    ResetIbvwrapFakes();
    ResetNcclFakes();
  }
};

TEST_F(NetIbGidInspectMicrotest, GetAndSetDev_ValidDeviceRoundTripsGidState) {
  ncclIbGidState input{};
  input.linkLayer = IBV_LINK_LAYER_ETHERNET;
  input.gidIndex = 7;
  for (size_t i = 0; i < sizeof(input.gid); ++i) input.gid[i] = static_cast<uint8_t>(i + 1);
  ncclIbGidState output{};

  ASSERT_EQ(ncclIbGidSetDev(0, &input), ncclSuccess);
  ASSERT_EQ(ncclIbGidGetDev(0, &output), ncclSuccess);

  EXPECT_EQ(output.linkLayer, input.linkLayer);
  EXPECT_EQ(output.gidIndex, input.gidIndex);
  EXPECT_EQ(std::memcmp(output.gid, input.gid, sizeof(input.gid)), 0);
  EXPECT_EQ(output.ibDev, -1);
}

TEST_F(NetIbGidInspectMicrotest, GetAndSetDev_InvalidArgumentsAreRejected) {
  ncclIbGidState state{};

  EXPECT_EQ(ncclIbGidGetDev(-1, &state), ncclInvalidArgument);
  EXPECT_EQ(ncclIbGidGetDev(1, &state), ncclInvalidArgument);
  EXPECT_EQ(ncclIbGidGetDev(0, nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclIbGidSetDev(-1, &state), ncclInvalidArgument);
  EXPECT_EQ(ncclIbGidSetDev(0, nullptr), ncclInvalidArgument);
}

TEST_F(NetIbGidInspectMicrotest, GetQpState_ClampsCountAndReportsUnmaterializedQps) {
  ncclIbSendComm comm{};
  comm.base.isSend = true;
  comm.base.nqps = NCCL_IB_MAX_QPS + 1;
  comm.base.qps[0].devIndex = 2;
  comm.base.qps[0].rtrAttr.localGidIndex = 9;
  ncclIbGidQpState state{};

  ASSERT_EQ(ncclIbGidGetQpState(&comm, &state), ncclSuccess);

  EXPECT_EQ(state.nqps, NCCL_IB_MAX_QPS);
  EXPECT_EQ(state.devIndex[0], 2);
  EXPECT_EQ(state.rtrGidIndex[0], 9);
  EXPECT_FALSE(state.queryOk[0]);

  comm.base.nqps = -1;  // not connected yet
  ASSERT_EQ(ncclIbGidGetQpState(&comm, &state), ncclSuccess);
  EXPECT_EQ(state.nqps, 0);
  EXPECT_EQ(state.devIndex[0], 0);
}

TEST_F(NetIbGidInspectMicrotest, GetAndSetComm_ValidDeviceRoundTripsStateAndIbDevice) {
  ncclIbSendComm comm{};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 1;
  comm.devs[0].base.ibDevN = 4;
  ncclIbGidState input{};
  input.linkLayer = IBV_LINK_LAYER_INFINIBAND;
  input.gidIndex = 3;
  input.gid[0] = 0xab;
  ncclIbGidState output{};

  ASSERT_EQ(ncclIbGidSetComm(&comm, 0, &input), ncclSuccess);
  ASSERT_EQ(ncclIbGidGetComm(&comm, 0, &output), ncclSuccess);

  EXPECT_EQ(output.linkLayer, input.linkLayer);
  EXPECT_EQ(output.gidIndex, input.gidIndex);
  EXPECT_EQ(output.gid[0], 0xab);
  EXPECT_EQ(output.ibDev, 4);
}

TEST_F(NetIbGidInspectMicrotest, GetAndSetComm_InvalidArgumentsAreRejected) {
  ncclIbSendComm comm{};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 1;
  ncclIbGidState state{};

  EXPECT_EQ(ncclIbGidGetComm(nullptr, 0, &state), ncclInvalidArgument);
  EXPECT_EQ(ncclIbGidGetComm(&comm, -1, &state), ncclInvalidArgument);
  EXPECT_EQ(ncclIbGidGetComm(&comm, 1, &state), ncclInvalidArgument);
  EXPECT_EQ(ncclIbGidGetComm(&comm, 0, nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclIbGidSetComm(nullptr, 0, &state), ncclInvalidArgument);
  EXPECT_EQ(ncclIbGidSetComm(&comm, 0, nullptr), ncclInvalidArgument);
}

TEST_F(NetIbGidInspectMicrotest, GetQpState_QuerySuccessPublishesSgidIndex) {
  ncclIbSendComm comm{};
  comm.base.isSend = true;
  comm.base.nqps = 1;
  ibv_qp qp{};
  comm.base.qps[0].qp = &qp;
  ScopedHook query(g_ibvQueryQp, [&qp](ibv_qp* value, ibv_qp_attr* attr, int mask, ibv_qp_init_attr*) {
    EXPECT_EQ(value, &qp);
    EXPECT_EQ(mask, IBV_QP_AV);
    attr->ah_attr.grh.sgid_index = 11;
    return ncclSuccess;
  });
  ncclIbGidQpState state{};

  ASSERT_EQ(ncclIbGidGetQpState(&comm, &state), ncclSuccess);

  EXPECT_EQ(state.nqps, 1);
  EXPECT_TRUE(state.queryOk[0]);
  EXPECT_EQ(state.sgidIndex[0], 11);
  EXPECT_EQ(query.calls, 1);
}

TEST_F(NetIbGidInspectMicrotest, GetDevState_ValidResiliencyDeviceReturnsAtomicState) {
  ncclIbSendComm comm{};
  ncclIbResiliency resiliency{};
  comm.base.resiliency = &resiliency;
  resiliency.ndevs = 1;
  resiliency.devs[0].state.store(ncclIbResiliencyDevStateRecovered);
  int state = -1;

  ASSERT_EQ(ncclIbGidGetDevState(&comm, 0, &state), ncclSuccess);

  EXPECT_EQ(state, ncclIbResiliencyDevStateRecovered);
  EXPECT_EQ(ncclIbGidGetDevState(&comm, 1, &state), ncclInvalidArgument);
  EXPECT_EQ(ncclIbGidGetDevState(nullptr, 0, &state), ncclInvalidArgument);
}

TEST_F(NetIbGidInspectMicrotest, RecoveryAndProbingGidLookupsFindMatchingMaterializedQp) {
  ncclIbSendComm comm{};
  ncclIbResiliency resiliency{};
  comm.base.resiliency = &resiliency;
  ibv_qp otherDeviceQp{};
  ibv_qp recoveryQp{};
  ibv_qp probingQp{};
  // Each list puts a QP on another device, then an unmaterialized QP on device 3, ahead of the match.
  resiliency.nPortRecoveryQps = 3;
  resiliency.portRecoveryQps[0].devIndex = 4;
  resiliency.portRecoveryQps[0].qp = &otherDeviceQp;
  resiliency.portRecoveryQps[0].rtrAttr.localGidIndex = 5;
  resiliency.portRecoveryQps[1].devIndex = 3;
  resiliency.portRecoveryQps[1].rtrAttr.localGidIndex = 6;
  resiliency.portRecoveryQps[2].devIndex = 3;
  resiliency.portRecoveryQps[2].qp = &recoveryQp;
  resiliency.portRecoveryQps[2].rtrAttr.localGidIndex = 7;
  resiliency.nProbingQps = 3;
  resiliency.probingQps[0].devIndex = 4;
  resiliency.probingQps[0].qp = &otherDeviceQp;
  resiliency.probingQps[0].rtrAttr.localGidIndex = 5;
  resiliency.probingQps[1].devIndex = 3;
  resiliency.probingQps[1].rtrAttr.localGidIndex = 6;
  resiliency.probingQps[2].devIndex = 3;
  resiliency.probingQps[2].qp = &probingQp;
  resiliency.probingQps[2].rtrAttr.localGidIndex = 8;
  int gid = -1;

  EXPECT_EQ(ncclIbGidGetRecoveryGidIndex(&comm, 3, &gid), ncclSuccess);
  EXPECT_EQ(gid, 7);
  EXPECT_EQ(ncclIbGidGetProbingGidIndex(&comm, 3, &gid), ncclSuccess);
  EXPECT_EQ(gid, 8);
  EXPECT_EQ(ncclIbGidGetRecoveryGidIndex(&comm, 9, &gid), ncclInvalidArgument);
  EXPECT_EQ(ncclIbGidGetProbingGidIndex(&comm, 9, &gid), ncclInvalidArgument);
}

TEST_F(NetIbGidInspectMicrotest, DriveQpToError_ValidIndexRequestsErrorState) {
  ncclIbSendComm comm{};
  comm.base.nqps = 1;
  ibv_qp qp{};
  comm.base.qps[0].qp = &qp;
  ScopedHook modify(g_ibvModifyQp, [&qp](ibv_qp* value, ibv_qp_attr* attr, int mask) {
    EXPECT_EQ(value, &qp);
    EXPECT_EQ(attr->qp_state, IBV_QPS_ERR);
    EXPECT_EQ(mask, IBV_QP_STATE);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclIbGidDriveQpToError(&comm, 0), ncclSuccess);
  EXPECT_EQ(modify.calls, 1);
  ScopedHook fail(g_ibvModifyQp, [](ibv_qp*, ibv_qp_attr*, int) {
    return ncclRemoteError;
  });
  EXPECT_EQ(ncclIbGidDriveQpToError(&comm, 0), ncclRemoteError);
  EXPECT_EQ(ncclIbGidDriveQpToError(&comm, 1), ncclInvalidArgument);
  comm.base.qps[0].qp = nullptr;
  EXPECT_EQ(ncclIbGidDriveQpToError(&comm, 0), ncclInvalidArgument);
}

}  // namespace
