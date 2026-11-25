/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_ib_cast/capability.cc (NET_IB_CAST_CAPABILITY_CC_PATH):
// the UD and RDMA READ probes the resiliency features depend on, over fake verbs.

#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "../../../common/LogCapture.hpp"
#include "net-ib-cast-test-common.h"

#include NET_IB_CAST_CAPABILITY_CC_PATH

namespace {

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;
using RcclUnitTesting::ScopedDebugLogging;

// The step a probe fails at; None lets it run to the end.
enum class Fail { None, AllocPd, CreateCq, RegMr, CreateQp, NoQp, GidIndex, QueryGid, Init, Rtr, Rts, PostSend };

class NetIbCastCapabilityMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    InitFakeContext(&ctx_, &device_);
    dev_ = &IbCastDevs[0];
    dev_->context = &ctx_;
    dev_->portNum = 2;
    snprintf(dev_->devName, sizeof(dev_->devName), "ionic_0");
    dev_->portAttr.active_mtu = IBV_MTU_4096;
    dev_->portAttr.link_layer = IBV_LINK_LAYER_ETHERNET;
    dev_->portAttr.lid = 9;
    dev_->portAttr.flags = 3;
    dev_->udSupported = -1;
    dev_->rdmaReadSupported = -1;
    IbCastNDevs = 1;
    qp_.qp_num = 77;
    qp_.context = &ctx_;
    cq_.context = &ctx_;
  }
  void TearDown() override {
    NetIbCastResetGlobals();
    ResetIbvwrapFakes();
    ResetNetIbCastConnectFakes();
    ResetNetIbCastP2pResiliencyRecoveryFakes();
  }

  bool Stops(Fail at) const { return fail_ == at; }

  ParamScript params_;
  Fail fail_ = Fail::None;
  struct ibv_device device_ {};
  struct ibv_context ctx_;
  struct ncclIbDev* dev_ = nullptr;
  struct ibv_pd pd_ {};
  struct ibv_cq cq_ {};
  struct ibv_mr mr_ {};
  struct ibv_qp qp_ {};
  struct ncclIbQpCreateAttr created_ {};
  struct ncclIbQp configured_ {};  // the probe QP as last configured
  int destroyedQps_ = 0, destroyedCqs_ = 0, deallocs_ = 0, deregs_ = 0;

  HookOf<decltype(g_ibvAllocPd)> allocPd_{g_ibvAllocPd, [this](struct ibv_pd** ret, struct ibv_context* ctx) {
                                            EXPECT_EQ(ctx, &ctx_);
                                            *ret = Stops(Fail::AllocPd) ? nullptr : &pd_;
                                            return Stops(Fail::AllocPd) ? ncclSystemError : ncclSuccess;
                                          }};
  HookOf<decltype(g_ibvCreateCq)> createCq_{
      g_ibvCreateCq, [this](struct ibv_cq** ret, struct ibv_context*, int, void*, struct ibv_comp_channel*, int) {
        *ret = Stops(Fail::CreateCq) ? nullptr : &cq_;
        return Stops(Fail::CreateCq) ? ncclSystemError : ncclSuccess;
      }};
  HookOf<decltype(g_ibvRegMr)> regMr_{g_ibvRegMr, [this](struct ibv_mr** ret, struct ibv_pd*, void* addr, size_t,
                                                         int access) {
                                        EXPECT_EQ(access, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
                                        mr_.addr = addr;
                                        mr_.lkey = 0x11;
                                        mr_.rkey = 0x22;
                                        *ret = Stops(Fail::RegMr) ? nullptr : &mr_;
                                        return Stops(Fail::RegMr) ? ncclSystemError : ncclSuccess;
                                      }};
  HookOf<decltype(g_ibCastQpCreate)> qpCreate_{g_ibCastQpCreate, [this](struct ncclIbQp* qp,
                                                                        struct ncclIbQpCreateAttr* attr) {
                                                 created_ = *attr;
                                                 qp->qp = Stops(Fail::NoQp) || Stops(Fail::CreateQp) ? nullptr : &qp_;
                                                 return Stops(Fail::CreateQp) ? ncclSystemError : ncclSuccess;
                                               }};
  HookOf<decltype(g_ibCastGetGidIndex)> gidIndex_{
      g_ibCastGetGidIndex, [this](struct ibv_context*, uint8_t port, struct ibv_port_attr* attr, int* index) {
        EXPECT_EQ(port, 2);
        EXPECT_EQ(attr, &dev_->portAttr);
        *index = 3;
        return Stops(Fail::GidIndex) ? ncclSystemError : ncclSuccess;
      }};
  HookOf<decltype(g_ibvQueryGid)> queryGid_{g_ibvQueryGid, [this](struct ibv_context*, uint8_t, int index,
                                                                  union ibv_gid* gid) {
                                              EXPECT_EQ(index, 3);
                                              memset(gid->raw, 0xee, sizeof(gid->raw));
                                              return Stops(Fail::QueryGid) ? ncclSystemError : ncclSuccess;
                                            }};
  HookOf<decltype(g_ibCastQpInit)> qpInit_{g_ibCastQpInit, [this](struct ncclIbQp* qp) {
                                             configured_ = *qp;
                                             return Stops(Fail::Init) ? ncclSystemError : ncclSuccess;
                                           }};
  HookOf<decltype(g_ibCastQpRtr)> qpRtr_{g_ibCastQpRtr, [this](struct ncclIbQp* qp) {
                                           configured_ = *qp;
                                           return Stops(Fail::Rtr) ? ncclSystemError : ncclSuccess;
                                         }};
  HookOf<decltype(g_ibCastQpRts)> qpRts_{g_ibCastQpRts, [this](struct ncclIbQp* qp) {
                                           configured_ = *qp;
                                           return Stops(Fail::Rts) ? ncclSystemError : ncclSuccess;
                                         }};
  HookOf<decltype(g_ibvDestroyQp)> destroyQp_{g_ibvDestroyQp, [this](struct ibv_qp* qp) {
                                                EXPECT_EQ(qp, &qp_);
                                                destroyedQps_++;
                                                return ncclSuccess;
                                              }};
  HookOf<decltype(g_ibvDestroyCq)> destroyCq_{g_ibvDestroyCq, [this](struct ibv_cq*) {
                                                destroyedCqs_++;
                                                return ncclSuccess;
                                              }};
  HookOf<decltype(g_ibvDeallocPd)> dealloc_{g_ibvDeallocPd, [this](struct ibv_pd*) {
                                              deallocs_++;
                                              return ncclSuccess;
                                            }};
  HookOf<decltype(g_ibvDeregMr)> dereg_{g_ibvDeregMr, [this](struct ibv_mr*) {
                                          deregs_++;
                                          return ncclSuccess;
                                        }};
};

TEST_F(NetIbCastCapabilityMicrotest, ProbeDevices_NoResiliencyFeature_ProbesNothing) {
  ASSERT_EQ(IbCastCapProbeDevices(), ncclSuccess);
  EXPECT_EQ(allocPd_.calls, 0);
  EXPECT_EQ(dev_->udSupported, -1);
  EXPECT_EQ(dev_->rdmaReadSupported, -1);
}


class NetIbCastCapabilityUdMicrotest : public NetIbCastCapabilityMicrotest {
 protected:
  void SetUp() override {
    NetIbCastCapabilityMicrotest::SetUp();
    params_.Set("IB_RESILIENCY_PORT_RECOVERY", 1);
  }
  HookOf<decltype(g_ibCastPortRecoveryQpUdToRts)> udToRts_{g_ibCastPortRecoveryQpUdToRts,
                                                           [this](struct ncclIbQp* qp, int port) {
                                                             EXPECT_EQ(qp->qp, &qp_);
                                                             EXPECT_EQ(port, 2);
                                                             return udToRtsResult_;
                                                           }};
  ncclResult_t udToRtsResult_ = ncclSuccess;
};

TEST_F(NetIbCastCapabilityUdMicrotest, ProbeUd_QpReachesRts_IsSupportedAndReleasesEverything) {
  ASSERT_EQ(IbCastCapProbeDevices(), ncclSuccess);
  EXPECT_EQ(dev_->udSupported, 1);
  EXPECT_TRUE(IbCastCapUdSupported(dev_));
  EXPECT_EQ(created_.type, IBV_QPT_UD);
  EXPECT_EQ(created_.cq, &cq_);
  EXPECT_EQ(created_.pd, &pd_);
  EXPECT_EQ(created_.maxSendWorkRequest, (uint32_t)NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX);
  EXPECT_EQ(created_.maxRecvWorkRequest, (uint32_t)NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX);
  EXPECT_EQ(created_.qpContext, &dev_->stats);
  EXPECT_EQ(created_.ctsQpSlot, (int8_t)NCCL_CTS_QP_SLOT_INVALID);
  EXPECT_FALSE(created_.isQpSharingEnabled);
  EXPECT_EQ(createCq_.calls, 1);
  EXPECT_EQ(destroyedQps_, 1);
  EXPECT_EQ(destroyedCqs_, 1);
  EXPECT_EQ(deallocs_, 1);
  // Probed once: a second pass leaves the answer alone.
  ASSERT_EQ(IbCastCapProbeDevices(), ncclSuccess);
  EXPECT_EQ(allocPd_.calls, 1);
}

TEST_F(NetIbCastCapabilityUdMicrotest, ProbeUd_CreateCqUsesThePortRecoveryCqSize) {
  ScopedHook cq(g_ibvCreateCq, [this](struct ibv_cq** ret, struct ibv_context*, int cqe, void*,
                                      struct ibv_comp_channel*, int) {
    EXPECT_EQ(cqe, NCCL_IB_RESILIENCY_PORT_RECOVERY_CQ_SIZE);
    *ret = &cq_;
    return ncclSuccess;
  });
  ASSERT_EQ(IbCastCapProbeDevices(), ncclSuccess);
  EXPECT_EQ(cq.calls, 1);
}

class NetIbCastCapabilityUdFailMicrotest : public NetIbCastCapabilityUdMicrotest,
                                           public ::testing::WithParamInterface<Fail> {};

TEST_P(NetIbCastCapabilityUdFailMicrotest, ProbeUd_AnyStepFails_IsUnsupported) {
  fail_ = GetParam();
  if (fail_ == Fail::Rts) udToRtsResult_ = ncclSystemError;
  ASSERT_EQ(IbCastCapProbeDevices(), ncclSuccess);
  EXPECT_EQ(dev_->udSupported, 0);
  EXPECT_FALSE(IbCastCapUdSupported(dev_));
  // Whatever was acquired is released.
  EXPECT_EQ(deallocs_, fail_ == Fail::AllocPd ? 0 : 1);
  EXPECT_EQ(destroyedCqs_, fail_ == Fail::AllocPd || fail_ == Fail::CreateCq ? 0 : 1);
  EXPECT_EQ(destroyedQps_, fail_ == Fail::Rts ? 1 : 0);
}

INSTANTIATE_TEST_SUITE_P(NetIbCastCapability, NetIbCastCapabilityUdFailMicrotest,
                         ::testing::Values(Fail::AllocPd, Fail::CreateCq, Fail::CreateQp, Fail::NoQp, Fail::Rts));


class NetIbCastCapabilityReadMicrotest : public NetIbCastCapabilityMicrotest {
 protected:
  void SetUp() override {
    NetIbCastCapabilityMicrotest::SetUp();
    params_.Set("IB_RESILIENCY_PORT_FAILOVER", 1);
  }
  // The loopback READ: copies the source word into the destination, as the NIC would.
  HookOf<decltype(g_fakePostSend)> postSend_{g_fakePostSend, [this](struct ibv_qp* qp, struct ibv_send_wr* wr,
                                                                    struct ibv_send_wr**) {
                                               EXPECT_EQ(qp, &qp_);
                                               posted_ = *wr;
                                               postedSge_ = *wr->sg_list;
                                               if (Stops(Fail::PostSend)) return EIO;
                                               if (copy_) {
                                                 memcpy((void*)wr->sg_list->addr, (void*)wr->wr.rdma.remote_addr,
                                                        wr->sg_list->length);
                                               }
                                               return 0;
                                             }};
  HookOf<decltype(g_fakePollCq)> pollCq_{g_fakePollCq, [this](struct ibv_cq* cq, int n, struct ibv_wc* wc) {
                                           EXPECT_EQ(cq, &cq_);
                                           EXPECT_EQ(n, 1);
                                           if (pollResult_ <= 0) return pollResult_;
                                           memset(wc, 0, sizeof(*wc));
                                           wc->status = status_;
                                           wc->vendor_err = 0x55;
                                           return 1;
                                         }};
  bool copy_ = true;
  int pollResult_ = 1;
  enum ibv_wc_status status_ = IBV_WC_SUCCESS;
  struct ibv_send_wr posted_ {};
  struct ibv_sge postedSge_ {};
};

TEST_F(NetIbCastCapabilityReadMicrotest, ProbeRead_LoopbackReadCompletes_IsSupported) {
  params_.Set("IB_PKEY", 4);
  params_.Set("IB_TC", 96);
  params_.Set("IB_SL", 2);
  params_.Set("IB_TIMEOUT", 14);
  params_.Set("IB_RETRY_CNT", 5);
  ASSERT_EQ(IbCastCapProbeDevices(), ncclSuccess);
  EXPECT_EQ(dev_->rdmaReadSupported, 1);
  EXPECT_TRUE(IbCastCapRdmaReadSupported(dev_));
  // An RC QP of depth 1 that connects to itself.
  EXPECT_EQ(created_.type, IBV_QPT_RC);
  EXPECT_EQ(created_.maxSendWorkRequest, 1u);
  EXPECT_EQ(created_.maxRecvWorkRequest, 1u);
  EXPECT_EQ(configured_.initAttr.state, IBV_QPS_INIT);
  EXPECT_EQ(configured_.initAttr.pkeyIndex, 4);
  EXPECT_EQ(configured_.initAttr.portNum, 2);
  EXPECT_EQ(configured_.initAttr.qpAccessFlags, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
  EXPECT_EQ(configured_.rtrAttr.mtu, IBV_MTU_4096);
  EXPECT_EQ(configured_.rtrAttr.linkLayer, IBV_LINK_LAYER_ETHERNET);
  EXPECT_EQ(configured_.rtrAttr.tc, 96);
  EXPECT_EQ(configured_.rtrAttr.sl, 2);
  EXPECT_EQ(configured_.rtrAttr.remoteQpNum, 77u);
  EXPECT_EQ(configured_.rtrAttr.remoteLid, 9u);
  EXPECT_EQ(configured_.rtrAttr.localIbPort, 2);
  EXPECT_EQ(configured_.rtrAttr.localPortFlags, 3);
  EXPECT_EQ(configured_.rtrAttr.localGidIndex, 3);
  EXPECT_EQ(configured_.rtrAttr.remoteGid.raw[0], 0xee);
  EXPECT_EQ(memcmp(&configured_.rtrAttr.localGid, &configured_.rtrAttr.remoteGid, sizeof(union ibv_gid)), 0);
  EXPECT_EQ(configured_.rtsAttr.timeout, 14);
  EXPECT_EQ(configured_.rtsAttr.retryCnt, 5);
  // One signaled READ of the source word into the destination, both in the probe's MR.
  EXPECT_EQ(posted_.opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(posted_.send_flags, (unsigned)IBV_SEND_SIGNALED);
  EXPECT_EQ(posted_.num_sge, 1);
  EXPECT_EQ(posted_.wr.rdma.rkey, 0x22u);
  EXPECT_EQ(postedSge_.lkey, 0x11u);
  EXPECT_EQ(postedSge_.length, sizeof(uint64_t));
  EXPECT_EQ(postedSge_.addr, posted_.wr.rdma.remote_addr + sizeof(uint64_t));
  EXPECT_EQ(posted_.wr.rdma.remote_addr, (uint64_t)(uintptr_t)mr_.addr);
  EXPECT_EQ(destroyedQps_, 1);
  EXPECT_EQ(deregs_, 1);
  EXPECT_EQ(destroyedCqs_, 1);
  EXPECT_EQ(deallocs_, 1);
}

TEST_F(NetIbCastCapabilityReadMicrotest, ProbeRead_TcAndSlUnset_UseZero) {
  ASSERT_EQ(IbCastCapProbeDevices(), ncclSuccess);
  EXPECT_EQ(configured_.rtrAttr.tc, 0);
  EXPECT_EQ(configured_.rtrAttr.sl, 0);
}

TEST_F(NetIbCastCapabilityReadMicrotest, ProbeRead_CompletionError_IsUnsupportedAndReportsTheStatus) {
  status_ = IBV_WC_REM_ACCESS_ERR;
  ScopedDebugLogging info(NCCL_LOG_INFO, NCCL_ALL);
  std::string log = CaptureLog([&] { ASSERT_EQ(IbCastCapProbeDevices(), ncclSuccess); });
  EXPECT_EQ(dev_->rdmaReadSupported, 0);
  EXPECT_TRUE(LogHas(log, "vendor_err 85")) << log;
}

TEST_F(NetIbCastCapabilityReadMicrotest, ProbeRead_SuccessWithoutTheData_IsUnsupported) {
  copy_ = false;
  ASSERT_EQ(IbCastCapProbeDevices(), ncclSuccess);
  EXPECT_EQ(dev_->rdmaReadSupported, 0);
}

TEST_F(NetIbCastCapabilityReadMicrotest, ProbeRead_PollFails_IsUnsupported) {
  pollResult_ = -1;
  ASSERT_EQ(IbCastCapProbeDevices(), ncclSuccess);
  EXPECT_EQ(dev_->rdmaReadSupported, 0);
  EXPECT_EQ(pollCq_.calls, 1);
}

TEST_F(NetIbCastCapabilityReadMicrotest, ProbeRead_NeverCompletes_GivesUpAfterTheTimeout) {
  pollResult_ = 0;
  const uint64_t start = clockNano();
  ASSERT_EQ(IbCastCapProbeDevices(), ncclSuccess);
  EXPECT_GE(clockNano() - start, NCCL_IB_CAP_RDMA_READ_PROBE_TIMEOUT_NS);
  EXPECT_EQ(dev_->rdmaReadSupported, 0);
  EXPECT_GT(pollCq_.calls, 1);
  EXPECT_EQ(destroyedQps_, 1);  // the QP goes before the buffer it may still read into
}

class NetIbCastCapabilityReadFailMicrotest : public NetIbCastCapabilityReadMicrotest,
                                             public ::testing::WithParamInterface<Fail> {};

TEST_P(NetIbCastCapabilityReadFailMicrotest, ProbeRead_AnySetupStepFails_IsUnsupportedAndReleases) {
  fail_ = GetParam();
  ASSERT_EQ(IbCastCapProbeDevices(), ncclSuccess);
  EXPECT_EQ(dev_->rdmaReadSupported, 0);
  EXPECT_EQ(pollCq_.calls, 0);
  EXPECT_EQ(deallocs_, fail_ == Fail::AllocPd ? 0 : 1);
  EXPECT_EQ(destroyedCqs_, fail_ == Fail::AllocPd || fail_ == Fail::CreateCq ? 0 : 1);
  EXPECT_EQ(deregs_, fail_ == Fail::AllocPd || fail_ == Fail::CreateCq || fail_ == Fail::RegMr ? 0 : 1);
  const bool qpMade = fail_ != Fail::AllocPd && fail_ != Fail::CreateCq && fail_ != Fail::RegMr &&
                      fail_ != Fail::CreateQp && fail_ != Fail::NoQp;
  EXPECT_EQ(destroyedQps_, qpMade ? 1 : 0);
}

INSTANTIATE_TEST_SUITE_P(NetIbCastCapability, NetIbCastCapabilityReadFailMicrotest,
                         ::testing::Values(Fail::AllocPd, Fail::CreateCq, Fail::RegMr, Fail::CreateQp, Fail::NoQp,
                                           Fail::GidIndex, Fail::QueryGid, Fail::Init, Fail::Rtr, Fail::Rts,
                                           Fail::PostSend));

TEST_F(NetIbCastCapabilityReadMicrotest, ProbeDevices_AlreadyProbed_IsNotProbedAgain) {
  dev_->rdmaReadSupported = 0;
  dev_->udSupported = 1;
  params_.Set("IB_RESILIENCY_PORT_RECOVERY", 1);
  ASSERT_EQ(IbCastCapProbeDevices(), ncclSuccess);
  EXPECT_EQ(allocPd_.calls, 0);
}


TEST_F(NetIbCastCapabilityMicrotest, GetDeviceCaps_AMergedDeviceHasOnlyWhatEveryMemberHas) {
  IbCastNDevs = 2;
  IbCastNMergedDevs = 2;
  IbCastDevs[0].udSupported = 1;
  IbCastDevs[0].rdmaReadSupported = 1;
  IbCastDevs[1].udSupported = 1;
  IbCastDevs[1].rdmaReadSupported = 0;
  IbCastMergedDevs[0].vProps.ndevs = 1;
  IbCastMergedDevs[0].vProps.devs[0] = 0;
  IbCastMergedDevs[1].vProps.ndevs = 2;
  IbCastMergedDevs[1].vProps.devs[0] = 0;
  IbCastMergedDevs[1].vProps.devs[1] = 1;
  struct ncclIbCastDeviceCaps caps;
  ASSERT_EQ(ncclIbCastGetDeviceCaps(0, &caps), ncclSuccess);
  EXPECT_TRUE(caps.udSupported);
  EXPECT_TRUE(caps.rdmaReadSupported);
  ASSERT_EQ(ncclIbCastGetDeviceCaps(1, &caps), ncclSuccess);
  EXPECT_TRUE(caps.udSupported);
  EXPECT_FALSE(caps.rdmaReadSupported);
  IbCastDevs[0].udSupported = -1;  // not probed counts as missing
  ASSERT_EQ(ncclIbCastGetDeviceCaps(0, &caps), ncclSuccess);
  EXPECT_FALSE(caps.udSupported);
}

TEST_F(NetIbCastCapabilityMicrotest, GetDeviceCaps_BadArguments_AreRejected) {
  IbCastNMergedDevs = 1;
  IbCastMergedDevs[0].vProps.ndevs = 1;
  IbCastMergedDevs[0].vProps.devs[0] = 5;  // past IbCastNDevs
  struct ncclIbCastDeviceCaps caps;
  EXPECT_EQ(ncclIbCastGetDeviceCaps(0, nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGetDeviceCaps(-1, &caps), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGetDeviceCaps(1, &caps), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGetDeviceCaps(0, &caps), ncclInvalidArgument);
  IbCastMergedDevs[0].vProps.devs[0] = -1;
  EXPECT_EQ(ncclIbCastGetDeviceCaps(0, &caps), ncclInvalidArgument);
}

}  // namespace
