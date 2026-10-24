/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib/common.cc and common.h.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "ScopedHook.h"
#include "fakes/ibvwrap_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "fakes/transport_net_ib_fakes.h"
#include "fakes/transport_net_ib_resiliency_fakes.h"

#include NET_IB_COMMON_CC_PATH

namespace {

class NetIbCommonMicrotest : public ::testing::Test {
 protected:
  void TearDown() override {
    ResetTransportNetIbResiliencyFakes();
    ResetTransportNetIbFakes();
    ResetIbvwrapFakes();
    ResetNcclFakes();
  }
};

TEST_F(NetIbCommonMicrotest, StatsCheckFatalCount_ZeroCountSucceeds) {
  ncclIbStats stats{};

  EXPECT_EQ(ncclIbStatsCheckFatalCount(&stats, "test"), ncclSuccess);
}

TEST_F(NetIbCommonMicrotest, StatsCheckFatalCount_FatalCountReturnsSystemError) {
  ncclIbStats stats{};
  stats.fatalErrorCount = 1;

  EXPECT_EQ(ncclIbStatsCheckFatalCount(&stats, "test"), ncclSystemError);
}

TEST_F(NetIbCommonMicrotest, StatsCheckFatalCount_AsyncEventsDisabledIgnoresCount) {
  ncclIbStats stats{};
  stats.fatalErrorCount = 1;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_RETURN_ASYNC_EVENTS") == 0 ? int64_t{0} : fallback;
  });

  EXPECT_EQ(ncclIbStatsCheckFatalCount(&stats, "test"), ncclSuccess);
}

TEST_F(NetIbCommonMicrotest, GetNetCommDevBase_SelectsSendOrReceiveStorage) {
  ncclIbSendComm send{};
  send.base.isSend = true;
  ncclIbRecvComm recv{};
  recv.base.isSend = false;

  EXPECT_EQ(ncclIbGetNetCommDevBase(&send.base, 1), &send.devs[1].base);
  EXPECT_EQ(ncclIbGetNetCommDevBase(&recv.base, 1), &recv.devs[1].base);
}

TEST_F(NetIbCommonMicrotest, BaseCommInit_InitializesQpStateAndDefaultMatching) {
  ncclIbNetCommBase base{};
  ncclIbResiliency resiliency{};
  ScopedHook init(g_ncclIbResiliencyInit,
                  [&resiliency](ncclIbNetCommBase* owner, ncclIbResiliency** out) {
                    resiliency.baseComm = owner;
                    *out = &resiliency;
                    return ncclSuccess;
                  });
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_SPLIT_DATA_ON_QPS") == 0) return int64_t{1};
    return fallback;
  });

  ASSERT_EQ(ncclIbBaseCommInit(&base, true), ncclSuccess);

  EXPECT_TRUE(base.isSend);
  EXPECT_EQ(base.nqps, -1);
  EXPECT_EQ(base.nDataQps, -1);
  EXPECT_EQ(base.splitDataOnQps, 1);
  EXPECT_EQ(base.ready, 0);
  EXPECT_EQ(base.recvMatchingScheme, BY_INDEX);
  EXPECT_EQ(base.resiliency, &resiliency);
  EXPECT_EQ(base.qps[0].devIndex, -1);
  EXPECT_EQ(base.qps[NCCL_IB_MAX_QPS - 1].remDevIdx, -1);
  EXPECT_EQ(base.activeQps[17], &base.qps[17]);
  EXPECT_EQ(init.calls, 1);
}

TEST_F(NetIbCommonMicrotest, BaseCommInit_OooRqForcesIdMatching) {
  ncclIbNetCommBase base{};
  ncclIbResiliency resiliency{};
  ScopedHook init(g_ncclIbResiliencyInit,
                  [&resiliency](ncclIbNetCommBase*, ncclIbResiliency** out) {
                    *out = &resiliency;
                    return ncclSuccess;
                  });
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_OOO_RQ") == 0) return int64_t{1};
    if (std::strcmp(name, "IB_RECEIVER_SIDE_MATCHING_SCHEME") == 0) return int64_t{BY_INDEX};
    return fallback;
  });

  ASSERT_EQ(ncclIbBaseCommInit(&base, false), ncclSuccess);

  EXPECT_FALSE(base.isSend);
  EXPECT_EQ(base.recvMatchingScheme, BY_ID);
}

TEST_F(NetIbCommonMicrotest, RecvCommInit_ResiliencyOrOooRqForcesPreposting) {
  ncclIbRecvComm recv{};
  ncclIbResiliency resiliency{};
  ScopedHook init(g_ncclIbResiliencyInit,
                  [&resiliency](ncclIbNetCommBase*, ncclIbResiliency** out) {
                    *out = &resiliency;
                    return ncclSuccess;
                  });
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_PREPOST_RECEIVE_WORK_REQUESTS") == 0) return int64_t{0};
    return fallback;
  });

  ASSERT_EQ(ncclIbRecvCommInit(&recv), ncclSuccess);

  EXPECT_TRUE(recv.prepostReceiveWorkRequests);
  EXPECT_EQ(recv.ibRecvWorkRequest.wr_id, NCCL_IB_RECV_WR_ID_DUMMY);
  EXPECT_EQ(recv.ibRecvWorkRequest.num_sge, 0);
}

TEST_F(NetIbCommonMicrotest, RecvCommInit_OooRqForcesPrepostingWithoutResiliency) {
  ncclIbRecvComm recv{};
  ScopedHook init(g_ncclIbResiliencyInit,
                  [](ncclIbNetCommBase*, ncclIbResiliency** out) {
                    *out = nullptr;
                    return ncclSuccess;
                  });
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_PREPOST_RECEIVE_WORK_REQUESTS") == 0) return int64_t{0};
    if (std::strcmp(name, "IB_OOO_RQ") == 0) return int64_t{1};
    return fallback;
  });

  ASSERT_EQ(ncclIbRecvCommInit(&recv), ncclSuccess);

  EXPECT_TRUE(recv.prepostReceiveWorkRequests);
  EXPECT_EQ(recv.base.resiliency, nullptr);
}

TEST_F(NetIbCommonMicrotest, RecvCommInit_DefaultOrExplicitDisabledUsesOnDemandPosting) {
  ScopedHook init(g_ncclIbResiliencyInit, [](ncclIbNetCommBase*, ncclIbResiliency** out) {
    *out = nullptr;
    return ncclSuccess;
  });
  int64_t prepost = -2;
  ScopedHook params(g_loadParam, [&prepost](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_PREPOST_RECEIVE_WORK_REQUESTS") == 0) return prepost;
    return fallback;
  });
  ncclIbRecvComm defaultRecv{};
  ncclIbRecvComm disabledRecv{};

  ASSERT_EQ(ncclIbRecvCommInit(&defaultRecv), ncclSuccess);
  prepost = 0;
  ASSERT_EQ(ncclIbRecvCommInit(&disabledRecv), ncclSuccess);

  EXPECT_FALSE(defaultRecv.prepostReceiveWorkRequests);
  EXPECT_FALSE(disabledRecv.prepostReceiveWorkRequests);
}

TEST_F(NetIbCommonMicrotest, SendCommInit_InitializesSendBase) {
  ncclIbSendComm send{};
  ncclIbResiliency resiliency{};
  ScopedHook init(g_ncclIbResiliencyInit,
                  [&resiliency](ncclIbNetCommBase*, ncclIbResiliency** out) {
                    *out = &resiliency;
                    return ncclSuccess;
                  });

  ASSERT_EQ(ncclIbSendCommInit(&send), ncclSuccess);

  EXPECT_TRUE(send.base.isSend);
  EXPECT_EQ(send.base.resiliency, &resiliency);
}

TEST_F(NetIbCommonMicrotest, ComputeLbWeights_EqualAndSpeedWeightedModesSumToHundred) {
  ncclIbNetCommBase base{};
  base.vProps.ndevs = 3;
  base.totalSpeed = 400;
  base.devSpeeds[0] = 100;
  base.devSpeeds[1] = 100;
  base.devSpeeds[2] = 200;
  ScopedHook equal(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_EVENT_BASED_LB") == 0 ? int64_t{0} : fallback;
  });

  ncclIbComputeLbWeights(&base);

  EXPECT_EQ(base.weights[0], 33);
  EXPECT_EQ(base.weights[1], 33);
  EXPECT_EQ(base.weights[2], 34);

  ScopedHook weighted(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_EVENT_BASED_LB") == 0 ? int64_t{1} : fallback;
  });
  ncclIbComputeLbWeights(&base);
  EXPECT_EQ(base.weights[0], 25);
  EXPECT_EQ(base.weights[1], 25);
  EXPECT_EQ(base.weights[2], 50);
}

TEST_F(NetIbCommonMicrotest, CommBaseGetNqpsPerRequest_ValidatesInitializationAndSplitMode) {
  ncclIbNetCommBase base{};
  int count = -1;
  base.nqps = -1;
  base.nDataQps = -1;
  EXPECT_EQ(ncclIbCommBaseGetNqpsPerRequest(&base, &count), ncclInternalError);
  base.nDataQps = 2;
  EXPECT_EQ(ncclIbCommBaseGetNqpsPerRequest(&base, &count), ncclInternalError);
  base.nqps = 8;
  base.splitDataOnQps = 0;
  EXPECT_EQ(ncclIbCommBaseGetNqpsPerRequest(&base, &count), ncclSuccess);
  EXPECT_EQ(count, 2);
  base.splitDataOnQps = 1;
  EXPECT_EQ(ncclIbCommBaseGetNqpsPerRequest(&base, &count), ncclSuccess);
  EXPECT_EQ(count, 8);
  EXPECT_EQ(ncclIbCommBaseGetNqpsPerRequest(&base, nullptr), ncclInternalError);
}

TEST_F(NetIbCommonMicrotest, CommBaseGetQpForRequest_RotatesAcrossActiveQps) {
  ncclIbNetCommBase base{};
  base.nqps = 4;
  base.nDataQps = 2;
  base.splitDataOnQps = 0;
  for (int i = 0; i < base.nqps; ++i) base.activeQps[i] = &base.qps[i];
  ncclIbQp* qp = nullptr;
  int index = -1;

  ASSERT_EQ(ncclIbCommBaseGetQpForRequest(&base, /*id=*/3, /*qpIndex=*/1, &qp, &index), ncclSuccess);

  EXPECT_EQ(index, 3);
  EXPECT_EQ(qp, &base.qps[3]);
  base.activeQps[3] = nullptr;
  EXPECT_EQ(ncclIbCommBaseGetQpForRequest(&base, 3, 1, &qp, &index), ncclInternalError);
}

TEST_F(NetIbCommonMicrotest, CommBaseGetQpByIndex_ValidatesDeviceAndUsesInterleavedLayout) {
  ncclIbNetCommBase base{};
  base.vProps.ndevs = 2;
  for (int i = 0; i < 4; ++i) base.activeQps[i] = &base.qps[i];
  ncclIbQp* qp = nullptr;

  EXPECT_EQ(ncclIbCommBaseGetQpByIndex(&base, 1, 1, &qp), ncclSuccess);
  EXPECT_EQ(qp, &base.qps[3]);
  EXPECT_EQ(ncclIbCommBaseGetQpByIndex(&base, -1, 0, &qp), ncclInternalError);
  EXPECT_EQ(ncclIbCommBaseGetQpByIndex(&base, 2, 0, &qp), ncclInternalError);
}

TEST_F(NetIbCommonMicrotest, CommBaseGetQpByQpNum_FindsQpWithinRequestedDevice) {
  ncclIbNetCommBase base{};
  base.vProps.ndevs = 2;
  base.nqps = 4;
  ibv_qp verbsQps[4]{};
  for (int i = 0; i < 4; ++i) {
    verbsQps[i].qp_num = 100 + i;
    base.qps[i].qp = &verbsQps[i];
  }
  ncclIbQp* qp = nullptr;
  int index = -1;

  EXPECT_EQ(ncclIbCommBaseGetQpByQpNum(&base, 1, 103, &qp, &index), ncclSuccess);
  EXPECT_EQ(qp, &base.qps[3]);
  EXPECT_EQ(index, 3);
  EXPECT_EQ(ncclIbCommBaseGetQpByQpNum(&base, 0, 103, &qp, &index), ncclInternalError);
  EXPECT_EQ(qp, nullptr);
  EXPECT_EQ(ncclIbCommBaseGetQpByQpNum(&base, 0, 100, nullptr, &index), ncclInternalError);
}

TEST_F(NetIbCommonMicrotest, ComputeDevSpeeds_UsesMinimumOfLocalAndAdvertisedRemote) {
  ncclIbSendComm comm{};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 2;
  comm.devs[0].base.ibDevN = 0;
  comm.devs[1].base.ibDevN = 1;
  comm.base.qps[0].remDevIdx = 1;
  comm.base.qps[1].remDevIdx = 0;
  ncclIbDevs[0].currSpeed = 100;
  ncclIbDevs[1].currSpeed = 200;
  comm.base.remDevs[0].currSpeed = 300;
  comm.base.remDevs[1].currSpeed = 50;

  ncclIbComputeDevSpeeds(&comm.base);

  EXPECT_EQ(comm.base.devSpeeds[0], 50u);
  EXPECT_EQ(comm.base.devSpeeds[1], 200u);
  EXPECT_EQ(comm.base.totalSpeed, 250u);
}

TEST_F(NetIbCommonMicrotest, GidInfoSnapshot_CopiesDeviceStateIntoCommBase) {
  ncclIbDev device{};
  device.gidInfo.link_layer = IBV_LINK_LAYER_ETHERNET;
  device.gidInfo.localGidIndex = 7;
  device.gidInfo.localGid.raw[15] = 0xaa;
  ncclIbNetCommDevBase base{};

  ncclIbGidInfoSnapshot(&base, &device);

  EXPECT_EQ(base.gidInfo.link_layer, IBV_LINK_LAYER_ETHERNET);
  EXPECT_EQ(base.gidInfo.localGidIndex, 7);
  EXPECT_EQ(base.gidInfo.localGid.raw[15], 0xaa);
}

TEST_F(NetIbCommonMicrotest, CheckSpeedChanges_RefreshesRemoteSpeedsWeightsAndCounters) {
  ncclIbSendComm comm{};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 2;
  comm.base.nRemDevs = 2;
  comm.devs[0].base.ibDevN = 0;
  comm.devs[1].base.ibDevN = 1;
  comm.base.qps[0].remDevIdx = 0;
  comm.base.qps[1].remDevIdx = 1;
  ncclIbDevs[0].currSpeed = 1000;
  ncclIbDevs[1].currSpeed = 2000;
  ncclIbSpeedChangeCounter = 4;
  comm.base.speedChangeCounter = 3;
  comm.remoteSpeedCounter = 7;
  comm.remoteSpeedBuf.counter = 8;
  comm.remoteSpeedBuf.speedGbps[0] = 1;
  comm.remoteSpeedBuf.speedGbps[1] = 3;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_EVENT_BASED_LB") == 0 ? int64_t{1} : fallback;
  });

  ncclIbCheckSpeedChanges(&comm, &comm.base);

  EXPECT_EQ(comm.base.speedChangeCounter, 4u);
  EXPECT_EQ(comm.remoteSpeedCounter, 8u);
  EXPECT_EQ(comm.base.remDevs[0].currSpeed, 1000u);
  EXPECT_EQ(comm.base.remDevs[1].currSpeed, 3000u);
  EXPECT_EQ(comm.base.totalSpeed, 3000u);
  EXPECT_EQ(comm.base.weights[0], 33);
  EXPECT_EQ(comm.base.weights[1], 67);
}

TEST_F(NetIbCommonMicrotest, StatsInitAndGidBitExtractorsPublishExpectedValues) {
  ncclIbStats stats{};
  stats.fatalErrorCount = 3;
  EXPECT_EQ(ncclIbStatsInit(&stats), ncclSuccess);
  EXPECT_EQ(stats.fatalErrorCount, 0);

  ibv_gid gid{};
  const uint16_t flid = htons(0x1234);
  std::memcpy(gid.raw + 4, &flid, sizeof(flid));
  EXPECT_EQ(ncclIbExtractFlid(&gid), 0x1234);
  EXPECT_EQ(ncclIbExtractLocalSubnetPrefix(htobe64(0xabcdef0123456789ULL)), 0x6789);
}

TEST_F(NetIbCommonMicrotest, WorkCompletionStatusHints_KnownAndUnknownStatusesDoNotCrash) {
  printIbWcStatusHint(IBV_WC_LOC_PROT_ERR);
  printIbWcStatusHint(IBV_WC_WR_FLUSH_ERR);
  printIbWcStatusHint(IBV_WC_RETRY_EXC_ERR);
  printIbWcStatusHint(IBV_WC_SUCCESS);
  SUCCEED();
}

TEST_F(NetIbCommonMicrotest, RequestHasEvents_ReportsAnyOutstandingDevice) {
  ncclIbRequest request{};
  EXPECT_FALSE(ncclIbRequestHasEvents(&request));
  request.events[NCCL_IB_MAX_DEVS_PER_NIC - 1] = 1;
  EXPECT_TRUE(ncclIbRequestHasEvents(&request));
}

TEST_F(NetIbCommonMicrotest, UpdateDeviceSpeed_ChangedPositiveSpeedPublishesAndIncrementsCounter) {
  ncclIbDev dev{};
  std::strcpy(dev.devName, "mlx5_0");
  dev.portNum = 1;
  dev.currSpeed = 1000;
  ncclIbSpeedChangeCounter = 4;
  ScopedHook speed(g_ibvQueryPortSpeed, [](ibv_context*, uint8_t port, uint64_t* value) {
    EXPECT_EQ(port, 1);
    *value = 20;
    return ncclSuccess;
  });

  ncclIbUpdateDeviceSpeed(&dev);

  EXPECT_EQ(dev.currSpeed, 2000u);
  EXPECT_EQ(ncclIbSpeedChangeCounter, 5u);
  EXPECT_EQ(speed.calls, 1);
}

TEST_F(NetIbCommonMicrotest, UpdateDeviceSpeed_QueryFailureZeroOrSameSpeedLeavesPublishedState) {
  ncclIbDev dev{};
  dev.portNum = 1;
  dev.currSpeed = 1000;
  ncclIbSpeedChangeCounter = 4;
  int mode = 0;
  ScopedHook speed(g_ibvQueryPortSpeed, [&mode](ibv_context*, uint8_t, uint64_t* value) {
    if (mode == 0) return ncclSystemError;
    *value = mode == 1 ? 0 : 10;
    return ncclSuccess;
  });

  ncclIbUpdateDeviceSpeed(&dev);
  mode = 1;
  ncclIbUpdateDeviceSpeed(&dev);
  mode = 2;
  ncclIbUpdateDeviceSpeed(&dev);

  EXPECT_EQ(dev.currSpeed, 1000u);
  EXPECT_EQ(ncclIbSpeedChangeCounter, 4u);
  EXPECT_EQ(speed.calls, 3);
}

TEST_F(NetIbCommonMicrotest, EventGidChange_NonEthernetSkipsRefresh) {
  ncclIbDev dev{};
  dev.gidInfo.link_layer = IBV_LINK_LAYER_INFINIBAND;
  ScopedHook query(g_ibvQueryPort, [](ibv_context*, uint8_t, ibv_port_attr*) {
    return ncclSuccess;
  });

  EXPECT_EQ(ncclIbEventGidChange(&dev), ncclSuccess);
  EXPECT_EQ(query.calls, 0);
}

TEST_F(NetIbCommonMicrotest, EventGidChange_QueryOrGidFailureIsPropagated) {
  ncclIbDev dev{};
  dev.gidInfo.link_layer = IBV_LINK_LAYER_ETHERNET;
  ScopedHook queryFail(g_ibvQueryPort, [](ibv_context*, uint8_t, ibv_port_attr*) {
    return ncclSystemError;
  });
  EXPECT_EQ(ncclIbEventGidChange(&dev), ncclSystemError);

  ScopedHook queryOk(g_ibvQueryPort, [](ibv_context*, uint8_t, ibv_port_attr* attr) {
    *attr = ibv_port_attr{};
    attr->link_layer = IBV_LINK_LAYER_ETHERNET;
    return ncclSuccess;
  });
  ScopedHook gidFail(g_ncclIbGidInfoQuery, [](ibv_context*, uint8_t, ibv_port_attr*, ncclIbGidInfo*) {
    return ncclRemoteError;
  });
  EXPECT_EQ(ncclIbEventGidChange(&dev), ncclRemoteError);
}

TEST_F(NetIbCommonMicrotest, DISABLED_EventGidChange_FailedRefreshDoesNotPublishPartialState) {
  // Production passes published storage directly to the refresh helper, so
  // partial writes survive an error. Enable with AICOMRCCL-3112.
  ncclIbDev dev{};
  dev.gidInfo.link_layer = IBV_LINK_LAYER_ETHERNET;
  dev.gidInfo.localGidIndex = 7;
  dev.gidInfo.localGid.raw[15] = 0xaa;
  ScopedHook query(g_ibvQueryPort, [](ibv_context*, uint8_t, ibv_port_attr* attr) {
    *attr = ibv_port_attr{};
    attr->link_layer = IBV_LINK_LAYER_ETHERNET;
    return ncclSuccess;
  });
  ScopedHook gid(g_ncclIbGidInfoQuery,
                 [](ibv_context*, uint8_t, ibv_port_attr*, ncclIbGidInfo* info) {
                   info->localGidIndex = 99;
                   info->localGid.raw[15] = 0xbb;
                   return ncclRemoteError;
                 });

  EXPECT_EQ(ncclIbEventGidChange(&dev), ncclRemoteError);
  EXPECT_EQ(dev.gidInfo.localGidIndex, 7);
  EXPECT_EQ(dev.gidInfo.localGid.raw[15], 0xaa);
}

TEST_F(NetIbCommonMicrotest, EventGidChange_SuccessPublishesPortAndGid) {
  ncclIbDev dev{};
  dev.portNum = 2;
  dev.gidInfo.link_layer = IBV_LINK_LAYER_ETHERNET;
  dev.gidInfo.localGidIndex = 1;
  ScopedHook query(g_ibvQueryPort, [](ibv_context*, uint8_t, ibv_port_attr* attr) {
    *attr = ibv_port_attr{};
    attr->state = IBV_PORT_ACTIVE;
    attr->link_layer = IBV_LINK_LAYER_ETHERNET;
    return ncclSuccess;
  });
  ScopedHook gid(g_ncclIbGidInfoQuery,
                 [](ibv_context*, uint8_t, ibv_port_attr*, ncclIbGidInfo* info) {
                   info->link_layer = IBV_LINK_LAYER_ETHERNET;
                   info->localGidIndex = 7;
                   info->localGid.raw[15] = 0xaa;
                   return ncclSuccess;
                 });

  ASSERT_EQ(ncclIbEventGidChange(&dev), ncclSuccess);

  EXPECT_EQ(dev.portAttr.state, IBV_PORT_ACTIVE);
  EXPECT_EQ(dev.gidInfo.localGidIndex, 7);
  EXPECT_EQ(dev.gidInfo.localGid.raw[15], 0xaa);
}

TEST_F(NetIbCommonMicrotest, GidChangeEvent_ValidatesDeviceIndexAndDelegates) {
  ncclNIbDevs = 1;
  ncclIbDevs[0].gidInfo.link_layer = IBV_LINK_LAYER_INFINIBAND;

  EXPECT_EQ(ncclIbGidChangeEvent(-1), ncclInvalidArgument);
  EXPECT_EQ(ncclIbGidChangeEvent(1), ncclInvalidArgument);
  EXPECT_EQ(ncclIbGidChangeEvent(0), ncclSuccess);
}

TEST_F(NetIbCommonMicrotest, AsyncThread_HandlesFatalChangeAndInformationalEventsThenAcknowledges) {
  ncclIbDev dev{};
  std::strcpy(dev.devName, "mlx5_0");
  dev.portNum = 1;
  dev.gidInfo.link_layer = IBV_LINK_LAYER_INFINIBAND;
  ibv_cq cq{};
  ncclIbStats cqStats{};
  cq.cq_context = &cqStats;
  ibv_qp qp{};
  ncclIbStats qpStats{};
  qp.qp_context = &qpStats;
  qp.send_cq = &cq;
  qp.recv_cq = &cq;
  ibv_srq srq{};
  std::vector<ibv_async_event> events(8);
  events[0].event_type = IBV_EVENT_DEVICE_FATAL;
  events[1].event_type = IBV_EVENT_CQ_ERR;
  events[1].element.cq = &cq;
  events[2].event_type = IBV_EVENT_QP_FATAL;
  events[2].element.qp = &qp;
  events[3].event_type = IBV_EVENT_SRQ_ERR;
  events[3].element.srq = &srq;
  events[4].event_type = IBV_EVENT_GID_CHANGE;
  events[5].event_type = IBV_EVENT_DEVICE_SPEED_CHANGE;
  events[6].event_type = IBV_EVENT_PORT_ACTIVE;
  events[7].event_type = IBV_EVENT_COMM_EST;
  size_t next = 0;
  ScopedHook getEvent(g_ibvGetAsyncEvent, [&events, &next](ibv_context*, ibv_async_event* out) {
    if (next == events.size()) return ncclSystemError;
    *out = events[next++];
    return ncclSuccess;
  });
  ScopedHook eventText(g_ibvEventTypeStr, [](char** text, ibv_event_type) {
    static char name[] = "event";
    *text = name;
    return ncclSuccess;
  });
  ScopedHook queryQp(g_ibvQueryQp, [](ibv_qp*, ibv_qp_attr* attr, int, ibv_qp_init_attr*) {
    *attr = ibv_qp_attr{};
    return ncclSuccess;
  });
  ScopedHook speed(g_ibvQueryPortSpeed, [](ibv_context*, uint8_t, uint64_t* value) {
    *value = 0;
    return ncclSuccess;
  });
  ScopedHook ack(g_ibvAckAsyncEvent, [](ibv_async_event*) { return ncclSuccess; });

  EXPECT_EQ(ncclIbAsyncThreadMain(&dev), nullptr);

  EXPECT_EQ(dev.stats.fatalErrorCount, 1);
  EXPECT_EQ(cqStats.fatalErrorCount, 1);
  EXPECT_EQ(qpStats.fatalErrorCount, 1);
  EXPECT_EQ(speed.calls, 1);
  EXPECT_EQ(ack.calls, static_cast<int>(events.size()));
}

TEST_F(NetIbCommonMicrotest, AsyncThread_EventStringOrAckFailureStopsProgress) {
  ncclIbDev dev{};
  int getCalls = 0;
  ScopedHook getEvent(g_ibvGetAsyncEvent, [&getCalls](ibv_context*, ibv_async_event* event) {
    ++getCalls;
    *event = ibv_async_event{};
    event->event_type = IBV_EVENT_COMM_EST;
    return ncclSuccess;
  });
  ScopedHook stringFail(g_ibvEventTypeStr, [](char**, ibv_event_type) {
    return ncclSystemError;
  });

  EXPECT_EQ(ncclIbAsyncThreadMain(&dev), nullptr);
  EXPECT_EQ(getCalls, 1);

  getCalls = 0;
  ScopedHook stringOk(g_ibvEventTypeStr, [](char** text, ibv_event_type) {
    static char name[] = "event";
    *text = name;
    return ncclSuccess;
  });
  ScopedHook ackFail(g_ibvAckAsyncEvent, [](ibv_async_event*) {
    return ncclSystemError;
  });
  EXPECT_EQ(ncclIbAsyncThreadMain(&dev), nullptr);
  EXPECT_EQ(getCalls, 1);
  EXPECT_EQ(ackFail.calls, 1);
}

TEST_F(NetIbCommonMicrotest, AsyncThread_GidRefreshFailureMarksDeviceFatal) {
  ncclIbDev dev{};
  dev.gidInfo.link_layer = IBV_LINK_LAYER_ETHERNET;
  int calls = 0;
  ScopedHook getEvent(g_ibvGetAsyncEvent, [&calls](ibv_context*, ibv_async_event* event) {
    if (calls++ != 0) return ncclSystemError;
    *event = ibv_async_event{};
    event->event_type = IBV_EVENT_GID_CHANGE;
    return ncclSuccess;
  });
  ScopedHook eventText(g_ibvEventTypeStr, [](char** text, ibv_event_type) {
    static char name[] = "gid";
    *text = name;
    return ncclSuccess;
  });
  ScopedHook query(g_ibvQueryPort, [](ibv_context*, uint8_t, ibv_port_attr*) {
    return ncclRemoteError;
  });
  ScopedHook ack(g_ibvAckAsyncEvent, [](ibv_async_event*) { return ncclSuccess; });

  EXPECT_EQ(ncclIbAsyncThreadMain(&dev), nullptr);
  EXPECT_EQ(dev.stats.fatalErrorCount, 1);
  EXPECT_EQ(ack.calls, 1);
}

}  // namespace
