/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_ib_cast/common.cc (NET_IB_CAST_COMMON_CC_PATH) and the
// inline helpers of common_cast.h: comm setup, fatal-error accounting and the async-event loop.

#include <gtest/gtest.h>

#include <cstring>
#include <deque>
#include <memory>
#include <string>

#include "../common/LogCapture.hpp"
#include "fakes/net_ib_cast_p2p_resiliency_fakes.h"
#include "fakes/transport_stubs.h"
#include "net-ib-cast-test-common.h"
#include "fakes/param_redirect.h"

#include NET_IB_CAST_COMMON_CC_PATH

namespace {

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

class NetIbCastCommonMicrotest : public ::testing::Test {
 protected:
  void TearDown() override {
    NetIbCastResetGlobals();
    ResetIbvwrapFakes();
    ResetNetIbCastConnectFakes();
    ResetNetIbCastP2pResiliencyFakes();
    ResetNetIbCastSchedulerFakes();
  }

  ParamScript params_;
  // Resiliency is off unless a test turns it on: the comm gets no resiliency context.
  HookOf<decltype(g_ibCastResiliencyInit)> resiliencyInit_{
      g_ibCastResiliencyInit, [this](struct ncclIbNetCommBase*, struct ncclIbResiliency** res) {
        *res = resiliency_;
        return ncclSuccess;
      }};
  struct ncclIbResiliency* resiliency_ = nullptr;
};

TEST_F(NetIbCastCommonMicrotest, StatsCheckFatalCount_AnyFatalErrorStopsTheCaller) {
  struct ncclIbStats stats;
  IbCastStatsInit(&stats);
  EXPECT_EQ(IbCastStatsCheckFatalCount(&stats, "here"), ncclSuccess);
  IbCastStatsFatalError(&stats);
  EXPECT_EQ(IbCastStatsCheckFatalCount(&stats, "here"), ncclSystemError);
  params_.Set("IB_RETURN_ASYNC_EVENTS", 0);
  EXPECT_EQ(IbCastStatsCheckFatalCount(&stats, "here"), ncclSuccess);
}

TEST_F(NetIbCastCommonMicrotest, FatalErrorHelpers_ChargeTheStatsTheObjectCarries) {
  struct ncclIbStats qpStats, cqStats;
  IbCastStatsInit(&qpStats);
  IbCastStatsInit(&cqStats);
  struct ibv_qp qp {};
  qp.qp_context = &qpStats;
  struct ibv_cq cq {};
  cq.cq_context = &cqStats;
  IbCastQpFatalError(&qp);
  IbCastQpFatalError(&qp);
  IbCastCqFatalError(&cq);
  IbCastDevFatalError(&IbCastDevs[0]);
  EXPECT_EQ(qpStats.fatalErrorCount, 2);
  EXPECT_EQ(cqStats.fatalErrorCount, 1);
  EXPECT_EQ(IbCastDevs[0].stats.fatalErrorCount, 1);
}

TEST_F(NetIbCastCommonMicrotest, GetNetCommDevBase_FindsThePerDeviceBaseOfSendAndRecvComms) {
  ZeroedPtr<ncclIbSendComm> send;
  ZeroedPtr<ncclIbRecvComm> recv;
  send->base.isSend = true;
  recv->base.isSend = false;
  EXPECT_EQ(IbCastGetNetCommDevBase(&send->base, 1), &send->devs[1].base);
  EXPECT_EQ(IbCastGetNetCommDevBase(&recv->base, 1), &recv->devs[1].base);
}

TEST_F(NetIbCastCommonMicrotest, SendCommInit_ResetsEveryQpAndTheCommState) {
  ZeroedPtr<ncclIbSendComm> send;
  memset(send->base.qps, 0x5a, sizeof(send->base.qps));
  memset(send->base.activeQps, 0x5a, sizeof(send->base.activeQps));
  send->base.nqps = send->base.nDataQps = 7;
  send->base.optRecvCompletion = true;
  send->base.ready = 1;
  ncclIbResiliency res;
  resiliency_ = &res;
  params_.Set("IB_SPLIT_DATA_ON_QPS", 1);
  ASSERT_EQ(IbCastSendCommInit(send.get()), ncclSuccess);
  EXPECT_TRUE(send->base.isSend);
  EXPECT_EQ(send->base.nqps, -1);
  EXPECT_EQ(send->base.nDataQps, -1);
  EXPECT_EQ(send->base.splitDataOnQps, 1);
  EXPECT_FALSE(send->base.optRecvCompletion);
  EXPECT_EQ(send->base.ready, 0);
  EXPECT_EQ(send->base.resiliency, &res);
  for (int i = 0; i < NCCL_IB_MAX_QPS; i++) {
    const struct ncclIbQp& qp = send->base.qps[i];
    EXPECT_EQ(qp.devIndex, -1) << i;
    EXPECT_EQ(qp.remDevIdx, -1) << i;
    EXPECT_EQ(qp.eceSupported, 0) << i;
    EXPECT_EQ(qp.ece.vendor_id | qp.ece.options | qp.ece.comp_mask, 0u) << i;
    EXPECT_EQ(qp.initAttr.portNum, 0) << i;
    EXPECT_EQ(qp.rtrAttr.remoteQpNum, 0u) << i;
    EXPECT_EQ(qp.rtsAttr.timeout, 0) << i;
    EXPECT_EQ(send->base.activeQps[i], &send->base.qps[i]) << i;
  }
}

TEST_F(NetIbCastCommonMicrotest, BaseCommInit_ResiliencyInitFails_ReturnsTheError) {
  ZeroedPtr<ncclIbSendComm> send;
  ScopedHook fails(g_ibCastResiliencyInit,
                   [](struct ncclIbNetCommBase*, struct ncclIbResiliency**) { return ncclSystemError; });
  EXPECT_EQ(IbCastSendCommInit(send.get()), ncclSystemError);
  ZeroedPtr<ncclIbRecvComm> recv;
  EXPECT_EQ(IbCastRecvCommInit(recv.get()), ncclSystemError);
}

TEST_F(NetIbCastCommonMicrotest, RecvCommInit_PostsOnDemandByDefault) {
  ZeroedPtr<ncclIbRecvComm> recv;
  recv->base.isSend = true;
  ASSERT_EQ(IbCastRecvCommInit(recv.get()), ncclSuccess);
  EXPECT_FALSE(recv->base.isSend);
  EXPECT_FALSE(recv->prepostReceiveWorkRequests);
  EXPECT_EQ(recv->ibRecvWorkRequest.wr_id, NCCL_IB_RECV_WR_ID_DUMMY);
  EXPECT_EQ(recv->ibRecvWorkRequest.num_sge, 0);
  params_.Set("IB_PREPOST_RECEIVE_WORK_REQUESTS", 1);
  ASSERT_EQ(IbCastRecvCommInit(recv.get()), ncclSuccess);
  EXPECT_TRUE(recv->prepostReceiveWorkRequests);
}

TEST_F(NetIbCastCommonMicrotest, RecvCommInit_ResiliencyOrOooRq_ForcePrePosting) {
  ZeroedPtr<ncclIbRecvComm> recv;
  params_.Set("IB_PREPOST_RECEIVE_WORK_REQUESTS", 0);
  auto res = std::make_unique<ncclIbResiliency>();
  resiliency_ = res.get();
  ASSERT_EQ(IbCastRecvCommInit(recv.get()), ncclSuccess);
  EXPECT_TRUE(recv->prepostReceiveWorkRequests);
  resiliency_ = nullptr;
  params_.Set("IB_OOO_RQ", 1);
  ASSERT_EQ(IbCastRecvCommInit(recv.get()), ncclSuccess);
  EXPECT_TRUE(recv->prepostReceiveWorkRequests);
}

struct OptRecv {
  const char* name;
  int64_t control;
  bool ctsOffload;
  int64_t netHint;
  bool resiliency;
  bool scheduler;
  bool expected;
};

class NetIbCastCommonOptRecvMicrotest : public NetIbCastCommonMicrotest,
                                        public ::testing::WithParamInterface<OptRecv> {};

TEST_P(NetIbCastCommonOptRecvMicrotest, InitOptRecvCompletion_OnlyWhenAskedAndNothingNeedsCompletions) {
  const OptRecv c = GetParam();
  params_.Set("RCCL_IB_OPTIONAL_RECV_COMPLETION", c.control);
  params_.Set("NET_OPTIONAL_RECV_COMPLETION", c.netHint);
  castGlobalQpSchedParms.enable = c.scheduler;
  ncclIbResiliency res;
  ZeroedPtr<ncclIbSendComm> send;
  send->base.resiliency = c.resiliency ? &res : nullptr;
  send->base.optRecvCompletion = !c.expected;
  IbCastInitOptRecvCompletion(&send->base, c.ctsOffload);
  EXPECT_EQ(send->base.optRecvCompletion, c.expected);
}

INSTANTIATE_TEST_SUITE_P(NetIbCastCommon, NetIbCastCommonOptRecvMicrotest,
                         ::testing::Values(OptRecv{"Default", 0, false, 1, false, false, false},
                                           OptRecv{"Control", 1, false, 1, false, false, true},
                                           OptRecv{"CtsOffload", 0, true, 1, false, false, true},
                                           OptRecv{"NoNetHint", 1, true, 0, false, false, false},
                                           OptRecv{"Resiliency", 1, false, 1, true, false, false},
                                           OptRecv{"Scheduler", 1, true, 1, false, true, false}),
                         [](const ::testing::TestParamInfo<OptRecv>& info) { return info.param.name; });

TEST_F(NetIbCastCommonMicrotest, CommBaseQpHelpers_IndexStripeAndLookUpQps) {
  ZeroedPtr<ncclIbSendComm> send;
  ASSERT_EQ(IbCastSendCommInit(send.get()), ncclSuccess);
  struct ncclIbNetCommBase* base = &send->base;
  base->vProps.ndevs = 2;
  base->nqps = 4;
  struct ibv_qp qps[4] = {};
  for (int i = 0; i < 4; i++) {
    qps[i].qp_num = 100 + i;
    base->qps[i].qp = &qps[i];
  }
  struct ncclIbQp* qp = nullptr;
  // QP i lives on device i % ndevs, as the i / ndevs-th QP of that device.
  ASSERT_EQ(IbCastCommBaseGetQpByIndex(base, 1, 1, &qp), ncclSuccess);
  EXPECT_EQ(qp, &base->qps[3]);
  int qpIndex = -1;
  ASSERT_EQ(IbCastCommBaseGetQpForRequest(base, 6, 1, &qp, &qpIndex), ncclSuccess);
  EXPECT_EQ(qpIndex, 3);
  EXPECT_EQ(qp, &base->qps[3]);
  ASSERT_EQ(IbCastCommBaseGetQpByQpNum(base, 1, 103, &qp, &qpIndex), ncclSuccess);
  EXPECT_EQ(qpIndex, 3);
  ASSERT_EQ(IbCastCommBaseGetQpByQpNum(base, 0, 100, &qp, nullptr), ncclSuccess);
  EXPECT_EQ(qp, &base->qps[0]);
  EXPECT_EQ(IbCastCommBaseGetQpByQpNum(base, 0, 103, &qp, &qpIndex), ncclInternalError);  // on device 1
  EXPECT_EQ(qp, nullptr);
}

TEST_F(NetIbCastCommonMicrotest, NqpsPerRequest_IsAllQpsOnlyWhenSplittingOrWeighting) {
  ZeroedPtr<ncclIbSendComm> send;
  send->base.nqps = 4;
  EXPECT_EQ(IbCastCommBaseGetNqpsPerRequest(&send->base), 1);
  send->base.schedParms.splitData = true;
  EXPECT_EQ(IbCastCommBaseGetNqpsPerRequest(&send->base), 4);
  send->base.schedParms.splitData = false;
  send->base.schedParms.doWrr = true;
  EXPECT_EQ(IbCastCommBaseGetNqpsPerRequest(&send->base), 4);
}

TEST_F(NetIbCastCommonMicrotest, PostRecvWorkRequest_GoesThroughTheQpsContext) {
  struct ibv_device device {};
  struct ibv_context ctx;
  InitFakeContext(&ctx, &device);
  struct ibv_qp qp {};
  qp.context = &ctx;
  struct ibv_recv_wr wr {};
  int rc = 0;
  ScopedHook post(g_fakePostRecv, [&](struct ibv_qp* q, struct ibv_recv_wr* w, struct ibv_recv_wr**) {
    EXPECT_EQ(q, &qp);
    EXPECT_EQ(w, &wr);
    return rc;
  });
  EXPECT_EQ(IbCastPostRecvWorkRequest(&qp, &wr), ncclSuccess);
  rc = ENOMEM;
  EXPECT_EQ(IbCastPostRecvWorkRequest(&qp, &wr), ncclSystemError);
}

TEST_F(NetIbCastCommonMicrotest, GidInfoSnapshot_CopiesTheDevicesCurrentGid) {
  IbCastDevs[0].gidInfo.localGidIndex = 5;
  IbCastDevs[0].gidInfo.link_layer = IBV_LINK_LAYER_ETHERNET;
  struct ncclIbNetCommDevBase base {};
  IbCastGidInfoSnapshot(&base, &IbCastDevs[0]);
  EXPECT_EQ(base.gidInfo.localGidIndex, 5);
  EXPECT_EQ(base.gidInfo.link_layer, IBV_LINK_LAYER_ETHERNET);
}

TEST_F(NetIbCastCommonMicrotest, NetIbCast_PublishesTheCastEntryPoints) {
  EXPECT_STREQ(netIbCast.name, "IB-CAST");
  EXPECT_EQ(netIbCast.init, IbCastInit);
  EXPECT_EQ(netIbCast.regMr, IbCastRegMr);
  EXPECT_EQ(netIbCast.isend, IbCastIsend);
  EXPECT_EQ(netIbCast.getDeviceMr, nullptr);
  EXPECT_EQ(netIbCast.irecvConsumed, nullptr);
  EXPECT_EQ(netIbCast.makeVDevice, IbCastMakeVDevice);
  EXPECT_EQ(netIbCast.finalize, IbCastFinalize);
  EXPECT_EQ(netIbCast.setNetAttr, IbCastSetNetAttr);
}


class NetIbCastCommonEventMicrotest : public NetIbCastCommonMicrotest {
 protected:
  void SetUp() override {
    InitFakeContext(&ctx_, &device_);
    dev_ = &IbCastDevs[0];
    dev_->context = &ctx_;
    dev_->portNum = 1;
    snprintf(dev_->devName, sizeof(dev_->devName), "ionic_0");
    dev_->gidInfo.link_layer = IBV_LINK_LAYER_ETHERNET;
    dev_->gidInfo.localGidIndex = 1;
    IbCastStatsInit(&dev_->stats);
    IbCastNDevs = 1;
  }
  // Hands the loop `events` in order, then fails the read so the loop ends.
  void Events(std::deque<enum ibv_event_type> types) {
    events_ = std::move(types);
    eventRead_.emplace(g_ibvGetAsyncEvent, [this](struct ibv_context* ctx, struct ibv_async_event* event) {
      EXPECT_EQ(ctx, &ctx_);
      if (events_.empty()) return ncclSystemError;
      memset(event, 0, sizeof(*event));
      event->event_type = events_.front();
      events_.pop_front();
      event->element.cq = &cq_;
      if (event->event_type >= IBV_EVENT_QP_FATAL && event->event_type <= IBV_EVENT_QP_ACCESS_ERR) {
        event->element.qp = &qp_;
      }
      return ncclSuccess;
    });
  }
  struct ibv_device device_ {};
  struct ibv_context ctx_;
  struct ncclIbDev* dev_ = nullptr;
  struct ibv_cq cq_ {};
  struct ibv_qp qp_ {};
  std::deque<enum ibv_event_type> events_;
  std::optional<HookOf<decltype(g_ibvGetAsyncEvent)>> eventRead_;
  HookOf<decltype(g_ibvEventTypeStr)> typeStr_{g_ibvEventTypeStr, [](char** ret, enum ibv_event_type) {
                                                 static char name[] = "fake event";
                                                 *ret = name;
                                                 return ncclSuccess;
                                               }};
  HookOf<decltype(g_ibvAckAsyncEvent)> ack_{g_ibvAckAsyncEvent, [](struct ibv_async_event*) { return ncclSuccess; }};
};

TEST_F(NetIbCastCommonEventMicrotest, GidChange_NotEthernet_IsIgnored) {
  dev_->gidInfo.link_layer = IBV_LINK_LAYER_INFINIBAND;
  ScopedHook query(g_ibvQueryPort, [](struct ibv_context*, uint8_t, struct ibv_port_attr*) { return ncclSuccess; });
  EXPECT_EQ(ncclIbCastGidChangeEvent(0), ncclSuccess);
  EXPECT_EQ(query.calls, 0);
}

TEST_F(NetIbCastCommonEventMicrotest, GidChange_RefreshesTheGidAndPortAttributes) {
  ScopedHook query(g_ibvQueryPort, [](struct ibv_context*, uint8_t port, struct ibv_port_attr* attr) {
    EXPECT_EQ(port, 1);
    memset(attr, 0, sizeof(*attr));
    attr->lid = 42;
    return ncclSuccess;
  });
  ScopedHook gid(g_ibCastGidInfoQuery,
                 [](struct ibv_context*, uint8_t, struct ibv_port_attr* attr, struct ncclIbGidInfo* info) {
                   EXPECT_EQ(attr->lid, 42);
                   info->localGidIndex = 2;
                   memset(info->localGid.raw, 0, sizeof(info->localGid.raw));
                   info->localGid.raw[15] = 9;
                   return ncclSuccess;
                 });
  EXPECT_EQ(ncclIbCastGidChangeEvent(0), ncclSuccess);
  EXPECT_EQ(dev_->gidInfo.localGidIndex, 2);
  EXPECT_EQ(dev_->gidInfo.localGid.raw[15], 9);
  EXPECT_EQ(dev_->portAttr.lid, 42);
}

TEST_F(NetIbCastCommonEventMicrotest, GidChange_QueryFailures_LeaveTheDeviceUnchanged) {
  dev_->portAttr.lid = 7;
  ncclResult_t portResult = ncclSystemError;
  ScopedHook query(g_ibvQueryPort, [&](struct ibv_context*, uint8_t, struct ibv_port_attr* attr) {
    attr->lid = 99;
    return portResult;
  });
  ScopedHook gid(g_ibCastGidInfoQuery, [](struct ibv_context*, uint8_t, struct ibv_port_attr*,
                                          struct ncclIbGidInfo*) { return ncclInternalError; });
  EXPECT_EQ(ncclIbCastGidChangeEvent(0), ncclSystemError);
  portResult = ncclSuccess;
  EXPECT_EQ(ncclIbCastGidChangeEvent(0), ncclInternalError);
  EXPECT_EQ(dev_->portAttr.lid, 7);
}

TEST_F(NetIbCastCommonEventMicrotest, GidChangeEvent_DeviceOutOfRange_IsRejected) {
  EXPECT_EQ(ncclIbCastGidChangeEvent(-1), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGidChangeEvent(1), ncclInvalidArgument);
}

TEST_F(NetIbCastCommonEventMicrotest, AsyncThread_FatalEvents_ChargeTheDeviceCqAndQp) {
  struct ncclIbStats cqStats, qpStats;
  IbCastStatsInit(&cqStats);
  IbCastStatsInit(&qpStats);
  cq_.cq_context = &cqStats;
  qp_.qp_context = &qpStats;
  Events({IBV_EVENT_DEVICE_FATAL, IBV_EVENT_CQ_ERR, IBV_EVENT_QP_FATAL, IBV_EVENT_QP_REQ_ERR, IBV_EVENT_QP_ACCESS_ERR});
  ScopedHook query(g_ibvQueryQp, [](struct ibv_qp*, struct ibv_qp_attr* attr, int mask, struct ibv_qp_init_attr*) {
    EXPECT_EQ(mask, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_RQ_PSN);
    attr->qp_state = IBV_QPS_ERR;
    return ncclSuccess;
  });
  EXPECT_EQ(IbCastAsyncThreadMain(dev_), nullptr);
  EXPECT_EQ(dev_->stats.fatalErrorCount, 1);
  EXPECT_EQ(cqStats.fatalErrorCount, 1);
  EXPECT_EQ(qpStats.fatalErrorCount, 3);
  EXPECT_EQ(query.calls, 3);
  EXPECT_EQ(ack_.calls, 5);
}

TEST_F(NetIbCastCommonEventMicrotest, AsyncThread_QpEventWithCqs_ReportsTheirHandles) {
  struct ncclIbStats qpStats;
  IbCastStatsInit(&qpStats);
  qp_.qp_context = &qpStats;
  struct ibv_cq sendCq {}, recvCq {};
  sendCq.handle = 11;
  recvCq.handle = 12;
  qp_.send_cq = &sendCq;
  qp_.recv_cq = &recvCq;
  Events({IBV_EVENT_QP_FATAL});
  ScopedHook query(g_ibvQueryQp, [](struct ibv_qp*, struct ibv_qp_attr*, int, struct ibv_qp_init_attr*) {
    return ncclSystemError;
  });
  std::string log = CaptureLog([&] { IbCastAsyncThreadMain(dev_); });
  EXPECT_TRUE(LogHas(log, "send_cq=11 recv_cq=12")) << log;
  EXPECT_EQ(qpStats.fatalErrorCount, 1);
}

// DEVICE_SPEED_CHANGE is a real event the loop has no case for.
TEST_F(NetIbCastCommonEventMicrotest, AsyncThread_NonFatalEvents_OnlyWarn) {
  Events({IBV_EVENT_SRQ_ERR, IBV_EVENT_PATH_MIG_ERR, IBV_EVENT_PORT_ERR, IBV_EVENT_PATH_MIG, IBV_EVENT_PORT_ACTIVE,
          IBV_EVENT_SQ_DRAINED, IBV_EVENT_LID_CHANGE, IBV_EVENT_PKEY_CHANGE, IBV_EVENT_SM_CHANGE,
          IBV_EVENT_QP_LAST_WQE_REACHED, IBV_EVENT_CLIENT_REREGISTER, IBV_EVENT_SRQ_LIMIT_REACHED,
          IBV_EVENT_COMM_EST, IBV_EVENT_DEVICE_SPEED_CHANGE});
  std::string log = CaptureLog([&] { IbCastAsyncThreadMain(dev_); });
  EXPECT_EQ(dev_->stats.fatalErrorCount, 0);
  EXPECT_EQ(ack_.calls, 14);
  // These arms only warn.
  EXPECT_TRUE(LogHas(log, "on SRQ")) << log;
  EXPECT_TRUE(LogHas(log, "non-fatal async event")) << log;
  EXPECT_TRUE(LogHas(log, "unknown event type")) << log;
}

TEST_F(NetIbCastCommonEventMicrotest, AsyncThread_GidChangeHandlerFails_MarksTheDeviceFatal) {
  Events({IBV_EVENT_GID_CHANGE, IBV_EVENT_GID_CHANGE});
  int portQueries = 0;
  ScopedHook query(g_ibvQueryPort, [&](struct ibv_context*, uint8_t, struct ibv_port_attr*) {
    return ++portQueries == 1 ? ncclSuccess : ncclSystemError;
  });
  ScopedHook gid(g_ibCastGidInfoQuery, [](struct ibv_context*, uint8_t, struct ibv_port_attr*,
                                          struct ncclIbGidInfo*) { return ncclSuccess; });
  IbCastAsyncThreadMain(dev_);
  EXPECT_EQ(dev_->stats.fatalErrorCount, 1);
}

TEST_F(NetIbCastCommonEventMicrotest, AsyncThread_EventNameOrAckFails_EndsTheLoop) {
  Events({IBV_EVENT_COMM_EST, IBV_EVENT_COMM_EST});
  ScopedHook ack(g_ibvAckAsyncEvent, [](struct ibv_async_event*) { return ncclSystemError; });
  IbCastAsyncThreadMain(dev_);
  EXPECT_EQ(events_.size(), 1u);
  ScopedHook name(g_ibvEventTypeStr, [](char**, enum ibv_event_type) { return ncclSystemError; });
  IbCastAsyncThreadMain(dev_);
  EXPECT_EQ(events_.size(), 0u);
  EXPECT_EQ(ack.calls, 1);
}

}  // namespace
