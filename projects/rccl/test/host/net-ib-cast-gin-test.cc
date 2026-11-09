/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_ib_cast/gin.cc (NET_IB_CAST_GIN_CC_PATH): the GIN/RMA
// proxy backend, its bootstrap collectives and the verbs work requests it posts. The GDAKI arm is
// compiled out (RCCL_NET_IB_CAST_ENABLE_GDAKI is unset), as in the shipping build.

#include <gtest/gtest.h>

#include <sys/wait.h>

#include <csignal>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "../common/LogCapture.hpp"
#include "../common/ProcessIsolatedTestRunner.hpp"
#include "AsanCheck.h"
#include "fakes/gdr_peermem_fakes.h"
#include "fakes/gdr_probe_fakes.h"
#include "fakes/net_ib_cast_p2p_fakes.h"
#include "fakes/socket_fakes.h"
#include "net-ib-cast-test-common.h"
#include "fakes/param_redirect.h"

#include NET_IB_CAST_GIN_CC_PATH

namespace {

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

// One rank's side of the proxy: a send and a recv comm on device 0, a CQ, a QP and a flush QP,
// all on a fake context; requests come from the comms' own pools the way p2p.cc hands them out.
class NetIbCastGinMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    InstallSucceedingSocketFakes();  // logging reads the peer address
    InitFakeContext(&ctx_, &device_);
    snprintf(device_.name, sizeof(device_.name), "ionic_0");
    pd_.context = &ctx_;
    cq_.context = &ctx_;
    qp_.context = &ctx_;
    flushQp_.context = &ctx_;
    IbCastNDevs = 1;
    IbCastNMergedDevs = 1;
    IbCastDevs[0].context = &ctx_;
    IbCastDevs[0].pd = &pd_;
    IbCastMergedDevs[0].vProps.ndevs = 1;
    for (struct ncclIbNetCommBase* base : {&send_->base, &recv_->base}) {
      base->vProps.ndevs = 1;
      base->nqps = 1;
    }
    send_->base.isSend = true;
    send_->devs[0].base.pd = &pd_;
    send_->devs[0].base.cq = &cq_;
    send_->devs[0].putSignalScratchpadMr = &scratchMr_;
    scratchMr_.lkey = 0x5c;
    send_->base.qps[0].qp = &qp_;
    recv_->devs[0].base.pd = &pd_;
    recv_->devs[0].base.cq = &cq_;
    recv_->devs[0].gpuFlush.qp.qp = &flushQp_;
    cComm_.ctx = &netConfig_;
    cComm_.rank = 0;
    cComm_.nranks = 2;
    cComm_.sendComm = send_.get();
    cComm_.recvComm = recv_.get();
    cComm_.allGather = IbCastGinIbAllGather;
    netConfig_.trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
  }
  void TearDown() override {
    NetIbCastResetGlobals();
    ResetIbvwrapFakes();
    ResetNetIbCastConnectFakes();
    ResetNetIbCastP2pFakes();
    ResetSocketFakes();
    ResetGdrPeermemFakes();
    ResetGdrProbeFakes();
  }

  // An MR per registration, so lkey/rkey tell the registrations apart.
  struct ibv_mr* NewMr() {
    mrs_.emplace_back(new ibv_mr{});
    mrs_.back()->lkey = 0x100 + (uint32_t)mrs_.size();
    mrs_.back()->rkey = 0x200 + (uint32_t)mrs_.size();
    return mrs_.back().get();
  }

  ParamScript params_;
  struct ibv_device device_ {};
  struct ibv_context ctx_;
  struct ibv_pd pd_ {};
  struct ibv_cq cq_ {};
  struct ibv_qp qp_ {}, flushQp_ {};
  struct ibv_mr scratchMr_ {};
  ZeroedPtr<ncclIbSendComm> send_;
  ZeroedPtr<ncclIbRecvComm> recv_;
  struct CastIbGinCollComm cComm_ {};
  ncclNetCommConfig_t netConfig_ {};
  std::vector<std::unique_ptr<ibv_mr>> mrs_;

  // GDR is latched once per process; every test in this binary that reaches it sees peermem.
  ScopedHook<int()> peermem_{g_ncclIbScanDefaultPeerMemClients, [] { return 1; }};
  HookOf<decltype(g_ibvRegMr)> regMr_{g_ibvRegMr, [this](struct ibv_mr** ret, struct ibv_pd*, void*, size_t, int) {
                                        *ret = NewMr();
                                        return ncclSuccess;
                                      }};
  HookOf<decltype(g_ibCastGetRequest)> getRequest_{
      g_ibCastGetRequest, [](struct ncclIbNetCommBase* base, struct ncclIbRequest** req) {
        for (struct ncclIbRequest& r : base->reqs) {
          if (r.type != NCCL_NET_IB_REQ_UNUSED) continue;
          r.base = base;
          memset(r.devBases, 0, sizeof(r.devBases));
          memset(r.events, 0, sizeof(r.events));
          *req = &r;
          return ncclSuccess;
        }
        return ncclInternalError;
      }};
  HookOf<decltype(g_ibCastFreeRequest)> freeRequest_{g_ibCastFreeRequest, [](struct ncclIbRequest* r) {
                                                       r->type = NCCL_NET_IB_REQ_UNUSED;
                                                       return ncclSuccess;
                                                     }};
  HookOf<decltype(g_ibCastAddEvent)> addEvent_{g_ibCastAddEvent, [](struct ncclIbRequest* req, int devIndex) {
                                                 req->events[devIndex]++;
                                                 req->devBases[devIndex] = IbCastGetNetCommDevBase(req->base, devIndex);
                                               }};
};


class NetIbCastGinInitMicrotest : public NetIbCastGinMicrotest {
 protected:
  void SetUp() override {}  // discovery builds the device tables
  void TearDown() override {
    world_.reset();
    NetIbCastGinMicrotest::TearDown();
  }
  std::unique_ptr<FakeIbWorld> world_ = std::make_unique<FakeIbWorld>();
};

TEST_F(NetIbCastGinInitMicrotest, RmaProxyInit_WithADevice_HandsBackAConfigWithNoTrafficClass) {
  world_->AddHca("ionic_0");
  void* ctx = nullptr;
  ASSERT_EQ(IbCastRmaIbProxy.init(&ctx, 0, nullptr), ncclSuccess);
  ASSERT_NE(ctx, nullptr);
  EXPECT_EQ(static_cast<ncclNetCommConfig_t*>(ctx)->trafficClass, NCCL_NET_TRAFFIC_CLASS_UNDEF);
  int ndev = -1;
  ASSERT_EQ(IbCastRmaIbProxy.devices(&ndev), ncclSuccess);
  EXPECT_EQ(ndev, 1);
  ASSERT_EQ(IbCastRmaIbProxy.finalize(ctx), ncclSuccess);
}

TEST_F(NetIbCastGinInitMicrotest, RmaProxyInit_NoDevice_Fails) {
  void* ctx = nullptr;
  EXPECT_EQ(IbCastRmaIbProxyInit(&ctx, 0, nullptr), ncclInternalError);
  EXPECT_EQ(ctx, nullptr);
  EXPECT_EQ(IbCastGinIbFinalize(nullptr), ncclSuccess);  // balances the reference discovery took
}

TEST_F(NetIbCastGinInitMicrotest, RmaProxyInit_DeviceDiscoveryFails_Fails) {
  params_.Set("IB_DISABLE", 1);
  void* ctx = nullptr;
  EXPECT_EQ(IbCastRmaIbProxyInit(&ctx, 0, nullptr), ncclInternalError);
}

TEST_F(NetIbCastGinMicrotest, GinProperties_AdvertiseStrongAndVaSignals) {
  ncclGinProperties_v14_t props{};
  ASSERT_EQ(IbCastGinIbGetGinProperties(&props), ncclSuccess);
  EXPECT_TRUE(props.supportsStrongSignals);
  EXPECT_TRUE(props.supportsVASignals);
}

TEST_F(NetIbCastGinMicrotest, RmaProperties_DoNotClaimSignalsFlushPuts) {
  ncclRmaProperties_t props;
  props.flushesAllPutsOnAnySignal = true;
  ASSERT_EQ(IbCastRmaIbProxy.getRmaProperties(&cComm_, &props), ncclSuccess);
  EXPECT_FALSE(props.flushesAllPutsOnAnySignal);
}

TEST_F(NetIbCastGinMicrotest, RmaProxyGetProperties_IsTheNetDeviceAsAGinProxy) {
  snprintf(IbCastDevs[0].devName, sizeof(IbCastDevs[0].devName), "ionic_0");
  snprintf(IbCastMergedDevs[0].devName, sizeof(IbCastMergedDevs[0].devName), "ionic_0");
  ncclNetProperties_t props;
  ASSERT_EQ(IbCastRmaIbProxy.getProperties(0, &props), ncclSuccess);
  EXPECT_STREQ(props.name, "ionic_0");
  EXPECT_EQ(props.netDeviceType, NCCL_NET_DEVICE_GIN_PROXY);
  EXPECT_EQ(IbCastRmaIbProxyGetProperties(1, &props), ncclInvalidUsage);
}

TEST_F(NetIbCastGinMicrotest, RmaProxy_PublishesTheProxyEntryPointsWithoutProgress) {
  EXPECT_STREQ(IbCastRmaIbProxy.name, "RMA_IB_PROXY");
  EXPECT_EQ(IbCastRmaIbProxy.listen, IbCastListen);
  EXPECT_EQ(IbCastRmaIbProxy.closeColl, IbCastGinIbCloseColl);
  EXPECT_EQ(IbCastRmaIbProxy.closeListen, IbCastCloseListen);
  EXPECT_EQ(IbCastRmaIbProxy.iput, IbCastRmaIbProxyIPut);
  EXPECT_EQ(IbCastRmaIbProxy.test, IbCastRmaIbProxyTest);
  EXPECT_EQ(IbCastRmaIbProxy.rmaProgress, nullptr);
  EXPECT_EQ(IbCastRmaIbProxy.queryLastError, nullptr);
}

class NetIbCastGinMicrotestIsolated : public ::testing::Test {};

TEST_F(NetIbCastGinMicrotestIsolated, GdrSupport_NoPeermemButDmaBuf_IsEnough) {
  RUN_ISOLATED_TEST("NetIbCastGin_GdrViaDmaBuf", []() {
    struct ibv_pd pd {};
    ScopedHook peermem(g_ncclIbScanDefaultPeerMemClients, [] { return 0; });
    ScopedHook probe(g_ncclIbProbeGdrSupport, [](struct ibv_context*, int) { return 0; });
    ScopedHook alloc(g_ibvAllocPd, [&](struct ibv_pd** ret, struct ibv_context*) {
      *ret = &pd;
      return ncclSuccess;
    });
    ScopedHook reg(g_ibvDirectRegDmabufMr, [](struct ibv_pd*, uint64_t, size_t, uint64_t, int, int) {
      errno = EBADF;
      return (struct ibv_mr*)nullptr;
    });
    IbCastNMergedDevs = 1;
    bool gdr = false;
    ASSERT_EQ(IbCastGinIbGdrSupport(&gdr, false), ncclSuccess);
    EXPECT_TRUE(gdr);
    EXPECT_EQ(IbCastGinIbGdrGpuSupport(false), ncclSuccess);
  });
}

TEST_F(NetIbCastGinMicrotestIsolated, GdrSupport_NeitherPeermemNorDmaBuf_RefusesGin) {
  RUN_ISOLATED_TEST("NetIbCastGin_NoGdr", []() {
    ScopedHook peermem(g_ncclIbScanDefaultPeerMemClients, [] { return 0; });
    ScopedHook probe(g_ncclIbProbeGdrSupport, [](struct ibv_context*, int) { return 0; });
    IbCastNMergedDevs = 1;
    bool gdr = true;
    ASSERT_EQ(IbCastGinIbGdrSupport(&gdr, false), ncclSuccess);
    EXPECT_FALSE(gdr);
    EXPECT_EQ(IbCastGinIbGdrGpuSupport(false), ncclInvalidUsage);
    void* collComm = nullptr;
    EXPECT_EQ(IbCastRmaIbProxyConnect(nullptr, nullptr, 2, 0, nullptr, &collComm), ncclInvalidUsage);
  });
}

TEST_F(NetIbCastGinMicrotestIsolated, InitType_NoGdr_FailsAfterDiscovery) {
  RUN_ISOLATED_TEST("NetIbCastGin_InitNoGdr", []() {
    FakeIbWorld world;
    world.AddHca("ionic_0");
    ScopedHook peermem(g_ncclIbScanDefaultPeerMemClients, [] { return 0; });
    ScopedHook probe(g_ncclIbProbeGdrSupport, [](struct ibv_context*, int) { return 0; });
    void* ctx = nullptr;
    EXPECT_EQ(IbCastGinIbInitType(&ctx, 0, nullptr, NCCL_GIN_TYPE_PROXY), ncclInternalError);
    EXPECT_EQ(ctx, nullptr);
  });
}


// A ring that completes every send and receive at once and fills each received chunk with its
// source rank, so the chunk order shows the ring direction.
class NetIbCastGinRingMicrotest : public NetIbCastGinMicrotest {
 protected:
  struct Xfer {
    void* comm;
    size_t offset;
    size_t size;
    int tag;
    void* mhandle;
  };
  void SetUp() override {
    NetIbCastGinMicrotest::SetUp();
    cComm_.rank = 1;
    cComm_.nranks = 3;
    send_->devs[0].base.ibDevN = 1;  // its own MR cache, so both registrations reach verbs
  }
  std::vector<Xfer> sends_, recvs_;
  char* base_ = nullptr;  // the buffer the collective registered
  int pendingTests_ = 0;  // polls each request needs before it completes

  HookOf<decltype(g_ibCastIsend)> isend_{g_ibCastIsend, [this](void* comm, void* data, size_t size, int tag,
                                                               void* mhandle, void*, void** request) {
                                           sends_.push_back({comm, (size_t)((char*)data - base_), size, tag, mhandle});
                                           *request = &sends_;
                                           return ncclSuccess;
                                         }};
  HookOf<decltype(g_ibCastIrecv)> irecv_{g_ibCastIrecv, [this](void* comm, int n, void** data, size_t* sizes,
                                                               int* tags, void** mhandles, void**, void** request) {
                                           EXPECT_EQ(n, 1);
                                           const size_t offset = (char*)data[0] - base_;
                                           recvs_.push_back({comm, offset, sizes[0], tags[0], mhandles[0]});
                                           memset(data[0], 'A' + (int)(offset / sizes[0]), sizes[0]);
                                           *request = &recvs_;
                                           return ncclSuccess;
                                         }};
  HookOf<decltype(g_ibCastTest)> test_{g_ibCastTest, [this](void*, int* done, int* sizes) {
                                         EXPECT_EQ(sizes, nullptr);
                                         *done = pendingTests_ > 0 ? (pendingTests_--, 0) : 1;
                                         return ncclSuccess;
                                       }};
};

TEST_F(NetIbCastGinRingMicrotest, AllGather_PassesChunksAroundTheRingAndDeregisters) {
  char out[3 * 4];
  base_ = out;
  const char src[4] = {'s', 'r', 'c', '1'};
  int deregs = 0;
  ScopedHook dereg(g_ibvDeregMr, [&](struct ibv_mr*) {
    deregs++;
    return ncclSuccess;
  });
  ASSERT_EQ(IbCastGinIbAllGather(&cComm_, (void*)src, out, 4), ncclSuccess);
  EXPECT_EQ(memcmp(out + 4, src, 4), 0);
  EXPECT_EQ(std::string(out, 4), "AAAA");
  EXPECT_EQ(std::string(out + 8, 4), "CCCC");
  // Rank 1 sends its own chunk, then forwards what came from rank 0; it hears from 0, then 2.
  ASSERT_EQ(sends_.size(), 2u);
  EXPECT_EQ(sends_[0].offset, 4u);
  EXPECT_EQ(sends_[1].offset, 0u);
  ASSERT_EQ(recvs_.size(), 2u);
  EXPECT_EQ(recvs_[0].offset, 0u);
  EXPECT_EQ(recvs_[1].offset, 8u);
  EXPECT_EQ(sends_[0].comm, send_.get());
  EXPECT_EQ(recvs_[0].comm, recv_.get());
  EXPECT_EQ(sends_[0].tag, IBCAST_GIN_IB_ALLGATHER_TAG);
  EXPECT_EQ(recvs_[0].tag, IBCAST_GIN_IB_ALLGATHER_TAG);
  EXPECT_NE(sends_[0].mhandle, recvs_[0].mhandle);  // one registration per comm
  EXPECT_EQ(regMr_.calls, 2);
  EXPECT_EQ(deregs, 2);
}

TEST_F(NetIbCastGinRingMicrotest, AllGather_RetriesPostsAndPollsUntilBothSidesComplete) {
  char out[3 * 4];
  base_ = out;
  int sendTries = 0;
  ScopedHook isend(g_ibCastIsend, [&](void*, void*, size_t, int, void*, void*, void** request) {
    *request = ++sendTries % 2 ? nullptr : &sends_;  // every other post is refused for now
    return ncclSuccess;
  });
  pendingTests_ = 3;
  ASSERT_EQ(IbCastGinIbAllGather(&cComm_, out, out, 4), ncclSuccess);
  EXPECT_EQ(sendTries, 4);
  EXPECT_EQ(recvs_.size(), 2u);
}

TEST_F(NetIbCastGinRingMicrotest, AllGather_AnyFailure_ReleasesWhatWasRegistered) {
  char out[3 * 4];
  base_ = out;
  int deregs = 0;
  ScopedHook dereg(g_ibvDeregMr, [&](struct ibv_mr*) {
    deregs++;
    return ncclSuccess;
  });
  {
    int regs = 0;
    ScopedHook reg(g_ibvRegMr, [&](struct ibv_mr** ret, struct ibv_pd*, void*, size_t, int) {
      *ret = NewMr();
      return ++regs == 2 ? ncclSystemError : ncclSuccess;  // the send-side registration fails
    });
    EXPECT_EQ(IbCastGinIbAllGather(&cComm_, out, out, 4), ncclSystemError);
    EXPECT_EQ(deregs, 1);
  }
  deregs = 0;
  {
    ScopedHook isend(g_ibCastIsend, [](void*, void*, size_t, int, void*, void*, void**) { return ncclSystemError; });
    EXPECT_EQ(IbCastGinIbAllGather(&cComm_, out, out, 4), ncclSystemError);
    EXPECT_EQ(deregs, 2);
  }
  deregs = 0;
  {
    ScopedHook irecv(g_ibCastIrecv, [](void*, int, void**, size_t*, int*, void**, void**, void**) {
      return ncclSystemError;
    });
    EXPECT_EQ(IbCastGinIbAllGather(&cComm_, out, out, 4), ncclSystemError);
    EXPECT_EQ(deregs, 2);
  }
  {
    int tests = 0;
    ScopedHook test(g_ibCastTest, [&](void*, int* done, int*) {
      *done = 0;
      return ++tests == 2 ? ncclSystemError : ncclSuccess;  // the send-side poll fails
    });
    EXPECT_EQ(IbCastGinIbAllGather(&cComm_, out, out, 4), ncclSystemError);
  }
  {
    ScopedHook test(g_ibCastTest, [](void*, int*, int*) { return ncclSystemError; });
    EXPECT_EQ(IbCastGinIbAllGather(&cComm_, out, out, 4), ncclSystemError);
  }
  {
    ScopedHook reg(g_ibvRegMr, [](struct ibv_mr**, struct ibv_pd*, void*, size_t, int) { return ncclSystemError; });
    EXPECT_EQ(IbCastGinIbAllGather(&cComm_, out, out, 4), ncclSystemError);
  }
}

// allGather stand-in for AllToAll: rank r's row j is the chunk r sends to rank j, "rj".
ncclResult_t ScriptedAllGather(struct CastIbGinCollComm* cComm, void* src, void* recv, size_t len) {
  for (int r = 0; r < cComm->nranks; r++) {
    for (int j = 0; j < cComm->nranks; j++) {
      char* chunk = (char*)recv + r * len + j * (len / cComm->nranks);
      chunk[0] = (char)('0' + r);
      chunk[1] = (char)('0' + j);
    }
  }
  return ncclSuccess;
}
ncclResult_t FailingAllGather(struct CastIbGinCollComm*, void*, void*, size_t) { return ncclRemoteError; }

TEST_F(NetIbCastGinMicrotest, AllToAll_KeepsTheChunkEachRankAddressedToThisRank) {
  cComm_.rank = 1;
  cComm_.nranks = 3;
  cComm_.allGather = ScriptedAllGather;
  char src[3 * 2] = {}, out[3 * 2] = {};
  ASSERT_EQ(IbCastGinIbAllToAll(&cComm_, src, out, 2), ncclSuccess);
  EXPECT_EQ(std::string(out, 6), "011121");
  cComm_.allGather = FailingAllGather;
  EXPECT_EQ(IbCastGinIbAllToAll(&cComm_, src, out, 2), ncclRemoteError);
}

TEST_F(NetIbCastGinRingMicrotest, P2PBarrier_IsAnAllGatherOfOneIntPerRank) {
  ASSERT_EQ(IbCastGinIbP2PBarrier(&cComm_), ncclSuccess);
  ASSERT_EQ(sends_.size(), 2u);
  EXPECT_EQ(sends_[0].size, sizeof(int));
  if (kAsan) GTEST_SKIP() << "the failed barrier leaks its buffer";
  ScopedHook isend(g_ibCastIsend, [](void*, void*, size_t, int, void*, void*, void**) { return ncclSystemError; });
  EXPECT_EQ(IbCastGinIbP2PBarrier(&cComm_), ncclSystemError);
}


class NetIbCastGinConnectMicrotest : public NetIbCastGinMicrotest {
 protected:
  struct ncclIbListenComm lComm_ {};
  std::vector<int> connectTcs_;
  int connectRefusals_ = 0;  // connects that hand back no comm yet
  int acceptRefusals_ = 0;
  std::vector<void*> connectHandles_;
  HookOf<decltype(g_ibCastConnectImpl)> connect_{
      g_ibCastConnectImpl, [this](void*, int dev, void* handle, void** sendComm, ncclNetDeviceHandle_t**, int tc) {
        EXPECT_EQ(dev, lComm_.dev);
        connectTcs_.push_back(tc);
        connectHandles_.push_back(handle);
        *sendComm = connectRefusals_ > 0 ? (connectRefusals_--, nullptr) : (void*)send_.get();
        return ncclSuccess;
      }};
  HookOf<decltype(g_ibCastAccept)> accept_{g_ibCastAccept, [this](void* listenComm, void** recvComm,
                                                                  ncclNetDeviceHandle_t**) {
                                             EXPECT_EQ(listenComm, &lComm_);
                                             *recvComm = acceptRefusals_ > 0 ? (acceptRefusals_--, nullptr)
                                                                             : (void*)recv_.get();
                                             return ncclSuccess;
                                           }};
};

TEST_F(NetIbCastGinConnectMicrotest, Connect_RetriesUntilBothSidesAreUpAndFillsTheCollComm) {
  connectRefusals_ = 1;
  acceptRefusals_ = 2;
  int h0, h1;
  void* handles[2] = {&h0, &h1};
  void* collComm = nullptr;
  ASSERT_EQ(IbCastRmaIbProxy.connect(&netConfig_, handles, 2, 1, &lComm_, &collComm), ncclSuccess);
  auto* c = static_cast<struct CastIbGinCollComm*>(collComm);
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(c->rank, 1);
  EXPECT_EQ(c->nranks, 2);
  EXPECT_EQ(c->ctx, &netConfig_);
  EXPECT_EQ(c->sendComm, send_.get());
  EXPECT_EQ(c->recvComm, recv_.get());
  EXPECT_EQ(c->allGather, IbCastGinIbAllGather);
  EXPECT_EQ(c->allToAll, IbCastGinIbAllToAll);
  EXPECT_EQ(c->getGidIndex, IbCastGetGidIndex);
  EXPECT_EQ(c->ib.context, &ctx_);
  EXPECT_EQ(c->ib.pd, &pd_);
  EXPECT_EQ(connect_.calls, 2);
  EXPECT_EQ(accept_.calls, 3);
  EXPECT_EQ(connectTcs_[0], -1);
  EXPECT_EQ(connectHandles_[0], &h0);  // rank 1 of 2 connects to rank 0, the next one around
  ScopedHook closeRecv(g_ibCastCloseRecv, [&](void* comm) {
    EXPECT_EQ(comm, recv_.get());
    return ncclSuccess;
  });
  ScopedHook closeSend(g_ibCastCloseSend, [&](void* comm) {
    EXPECT_EQ(comm, send_.get());
    return ncclSuccess;
  });
  ASSERT_EQ(IbCastGinIbCloseColl(collComm), ncclSuccess);
  EXPECT_EQ(closeRecv.calls, 1);
  EXPECT_EQ(closeSend.calls, 1);
}

TEST_F(NetIbCastGinConnectMicrotest, Connect_GinTrafficClassWinsOverTheIbOne) {
  void* handles[2] = {nullptr, nullptr};
  void* collComm = nullptr;
  params_.Set("IB_TC", 40);
  ASSERT_EQ(IbCastGinIbConnect(&netConfig_, handles, 2, 0, &lComm_, &collComm), ncclSuccess);
  free(collComm);
  params_.Set("GIN_IB_TC", 104);
  ASSERT_EQ(IbCastGinIbConnect(&netConfig_, handles, 2, 0, &lComm_, &collComm), ncclSuccess);
  free(collComm);
  EXPECT_EQ(connectTcs_, (std::vector<int>{40, 104}));
}

TEST_F(NetIbCastGinConnectMicrotest, Connect_ConnectOrAcceptFails_ReturnsTheError) {
  if (kAsan) GTEST_SKIP() << "the failed connect leaks the coll comm";
  void* handles[2] = {nullptr, nullptr};
  void* collComm = reinterpret_cast<void*>(0x1);
  {
    ScopedHook fails(g_ibCastConnectImpl, [](void*, int, void*, void**, ncclNetDeviceHandle_t**, int) {
      return ncclRemoteError;
    });
    EXPECT_EQ(IbCastGinIbConnect(&netConfig_, handles, 2, 0, &lComm_, &collComm), ncclRemoteError);
    EXPECT_EQ(collComm, nullptr);
  }
}

// The coll comm and the send comm already connected are left behind when accept fails.
// Pins AICOMRCCL-3094; a fix flips this pin.
TEST_F(NetIbCastGinConnectMicrotest, Connect_KnownBug_AcceptFailureLeavesTheSendCommOpen) {
  if (kAsan) GTEST_SKIP() << "the failed connect leaks the coll comm";
  void* handles[2] = {nullptr, nullptr};
  void* collComm = nullptr;
  ScopedHook fails(g_ibCastAccept, [](void*, void**, ncclNetDeviceHandle_t**) { return ncclSystemError; });
  ScopedHook closeSend(g_ibCastCloseSend, [](void*) { return ncclSuccess; });
  EXPECT_EQ(IbCastGinIbConnect(&netConfig_, handles, 2, 0, &lComm_, &collComm), ncclSystemError);
  EXPECT_EQ(connect_.calls, 1);
  EXPECT_EQ(closeSend.calls, 0);
}

TEST_F(NetIbCastGinMicrotest, CloseColl_NullIsANoOpAndCloseErrorsArePassedUp) {
  EXPECT_EQ(IbCastGinIbCloseColl(nullptr), ncclSuccess);
  auto* c = static_cast<struct CastIbGinCollComm*>(calloc(1, sizeof(struct CastIbGinCollComm)));
  c->recvComm = recv_.get();
  c->sendComm = send_.get();
  {
    ScopedHook closeRecv(g_ibCastCloseRecv, [](void*) { return ncclSystemError; });
    EXPECT_EQ(IbCastGinIbCloseColl(c), ncclSystemError);
  }
  c->recvComm = nullptr;
  ScopedHook closeSend(g_ibCastCloseSend, [](void*) { return ncclSystemError; });
  EXPECT_EQ(IbCastGinIbCloseColl(c), ncclSystemError);
  free(c);
  // With neither comm there is nothing to close.
  c = static_cast<struct CastIbGinCollComm*>(calloc(1, sizeof(struct CastIbGinCollComm)));
  EXPECT_EQ(IbCastGinIbCloseColl(c), ncclSuccess);
  EXPECT_EQ(closeSend.calls, 1);
}

TEST_F(NetIbCastGinMicrotest, RmaProxyCloseColl_FreesTheCollComm) {
  void* c = calloc(1, sizeof(struct CastIbGinCollComm));
  EXPECT_EQ(IbCastRmaIbProxyCloseColl(c), ncclSuccess);
}

// The context: every rank's handle is exchanged, then one send and one recv comm per peer per
// context, with a barrier after each peer.
class NetIbCastGinContextMicrotest : public NetIbCastGinRingMicrotest {
 protected:
  void SetUp() override {
    NetIbCastGinRingMicrotest::SetUp();
    cComm_.rank = 1;
    cComm_.nranks = 4;
    cComm_.dev = 0;
    for (int i = 0; i < 16; i++) {
      peerSends_.emplace_back();
      peerRecvs_.emplace_back();
    }
  }
  void TearDown() override {
    for (struct IbCastRmaIbProxyCtx* c : contexts_) {
      for (int i = 0; i < c[0].nContexts; i++) {
        free(c[i].fullSendComm);
        free(c[i].fullRecvComm);
      }
      free(c);
    }
    NetIbCastGinRingMicrotest::TearDown();
  }
  ncclResult_t Create(int nContexts, int rankStride, void** ctx) {
    ncclRmaConfig_t config{};
    config.nContexts = nContexts;
    config.trafficClass = 72;
    config.rankStride = rankStride;
    ncclResult_t res = IbCastRmaIbProxy.createContext(&cComm_, &config, ctx);
    if (res == ncclSuccess) contexts_.push_back(static_cast<struct IbCastRmaIbProxyCtx*>(*ctx));
    return res;
  }

  std::vector<struct IbCastRmaIbProxyCtx*> contexts_;
  std::vector<int> connectPeers_, acceptPeers_;
  std::deque<ZeroedPtr<ncclIbSendComm>> peerSends_;
  std::deque<ZeroedPtr<ncclIbRecvComm>> peerRecvs_;
  struct ncclIbListenComm lComm_ {};
  int listenBefore_ = 0;
  std::vector<std::string> gatheredHandles_;

  HookOf<decltype(g_ibCastSetTrafficClass)> tc_{g_ibCastSetTrafficClass, [this](void* ctx, int tc) {
                                                  EXPECT_EQ(ctx, &netConfig_);
                                                  EXPECT_EQ(tc, 72);
                                                }};
  HookOf<decltype(g_ibCastListen)> listen_{g_ibCastListen, [this](void* ctx, int dev, void* handle, void** listenComm) {
                                             EXPECT_EQ(ctx, &netConfig_);
                                             memset(handle, 0, NCCL_NET_HANDLE_MAXSIZE);
                                             ((struct ncclIbHandle*)handle)->magic = 0xfeed;
                                             *listenComm = &lComm_;
                                             return ncclSuccess;
                                           }};
  HookOf<decltype(g_ibCastCloseListen)> closeListen_{g_ibCastCloseListen, [this](void* listenComm) {
                                                       EXPECT_EQ(listenComm, &lComm_);
                                                       return ncclSuccess;
                                                     }};
  // The handle exchange: every peer's handle is a copy of ours with its rank in the magic.
  static ncclResult_t HandleAllGather(struct CastIbGinCollComm* cComm, void* src, void* recv, size_t len) {
    EXPECT_EQ(len, (size_t)NCCL_NET_HANDLE_MAXSIZE);
    for (int r = 0; r < cComm->nranks; r++) {
      memcpy((char*)recv + r * len, src, len);
      ((struct ncclIbHandle*)((char*)recv + r * len))->magic = 0x100 + r;
    }
    return ncclSuccess;
  }
  HookOf<decltype(g_ibCastConnectImpl)> connect_{
      g_ibCastConnectImpl, [this](void* ctx, int, void* handle, void** sendComm, ncclNetDeviceHandle_t**, int tc) {
        const struct ncclIbHandle* h = (const struct ncclIbHandle*)handle;
        EXPECT_TRUE(h->isRMA);
        connectPeers_.push_back((int)(h->magic - 0x100));
        *sendComm = peerSends_[connectPeers_.size() - 1].get();
        return ncclSuccess;
      }};
  HookOf<decltype(g_ibCastAccept)> accept_{g_ibCastAccept, [this](void*, void** recvComm, ncclNetDeviceHandle_t**) {
                                             acceptPeers_.push_back(0);
                                             *recvComm = peerRecvs_[acceptPeers_.size() - 1].get();
                                             return ncclSuccess;
                                           }};
};

TEST_F(NetIbCastGinContextMicrotest, CreateContext_ConnectsToEveryPeerInEachContext) {
  // The barrier after each peer is a ring all-gather over the bootstrap comms; the handle
  // exchange goes through cComm->allGather, replaced here.
  cComm_.allGather = HandleAllGather;
  char ring[4 * sizeof(int)];
  base_ = ring;
  void* ctx = nullptr;
  ASSERT_EQ(Create(2, 1, &ctx), ncclSuccess);
  auto* gc = static_cast<struct IbCastRmaIbProxyCtx*>(ctx);
  EXPECT_EQ(gc[0].nContexts, 2);
  EXPECT_EQ(gc[0].nranks, 4);
  EXPECT_EQ(gc[1].rank, 1);
  // Rank 1 connects to 1, 2, 3, 0 and accepts from 1, 0, 3, 2, in each context.
  EXPECT_EQ(connectPeers_, (std::vector<int>{1, 2, 3, 0, 1, 2, 3, 0}));
  for (int c = 0; c < 2; c++) {
    for (int peer = 0; peer < 4; peer++) {
      EXPECT_NE(gc[c].fullSendComm[peer], nullptr) << c << " " << peer;
      EXPECT_NE(gc[c].fullRecvComm[peer], nullptr) << c << " " << peer;
    }
  }
  EXPECT_EQ(gc[0].fullSendComm[2], peerSends_[1].get());
  EXPECT_EQ(gc[0].fullRecvComm[0], peerRecvs_[1].get());  // accepted second, after itself
  EXPECT_EQ(closeListen_.calls, 1);
  EXPECT_EQ(sends_.size(), 8u * 3);  // 8 barriers of nranks - 1 steps
}

TEST_F(NetIbCastGinContextMicrotest, CreateContext_RankStride_ConnectsOnlyToThatStride) {
  cComm_.allGather = HandleAllGather;
  char ring[4 * sizeof(int)];
  base_ = ring;
  void* ctx = nullptr;
  ASSERT_EQ(Create(1, 2, &ctx), ncclSuccess);
  EXPECT_EQ(connectPeers_, (std::vector<int>{1, 3}));
  auto* gc = static_cast<struct IbCastRmaIbProxyCtx*>(ctx);
  EXPECT_EQ(gc[0].fullSendComm[0], nullptr);
  EXPECT_EQ(gc[0].fullRecvComm[2], nullptr);
}

TEST_F(NetIbCastGinContextMicrotest, CreateContext_NoRankStride_MeansEveryRank) {
  cComm_.allGather = HandleAllGather;
  char ring[4 * sizeof(int)];
  base_ = ring;
  void* ctx = nullptr;
  ASSERT_EQ(Create(1, 0, &ctx), ncclSuccess);
  EXPECT_EQ(connectPeers_, (std::vector<int>{1, 2, 3, 0}));
}

TEST_F(NetIbCastGinContextMicrotest, CreateContext_StrideThatDoesNotDivideTheRanks_IsRejected) {
  void* ctx = reinterpret_cast<void*>(0x1);
  EXPECT_EQ(Create(1, 3, &ctx), ncclInternalError);
  EXPECT_EQ(listen_.calls, 0);
}

TEST_F(NetIbCastGinContextMicrotest, CreateContext_AnyStepFails_ReturnsNoContextAndClosesTheListener) {
  cComm_.allGather = HandleAllGather;
  char ring[4 * sizeof(int)];
  base_ = ring;
  void* ctx = reinterpret_cast<void*>(0x1);
  {
    ScopedHook listen(g_ibCastListen, [](void*, int, void*, void**) { return ncclSystemError; });
    EXPECT_EQ(Create(1, 1, &ctx), ncclSystemError);
    EXPECT_EQ(ctx, nullptr);
    EXPECT_EQ(closeListen_.calls, 0);
  }
  cComm_.allGather = FailingAllGather;
  EXPECT_EQ(Create(1, 1, &ctx), ncclRemoteError);
  EXPECT_EQ(closeListen_.calls, 1);
  if (kAsan) GTEST_SKIP() << "a failure past the peer-comm arrays leaks them";
  cComm_.allGather = HandleAllGather;
  {
    ScopedHook fails(g_ibCastConnectImpl, [](void*, int, void*, void**, ncclNetDeviceHandle_t**, int) {
      return ncclSystemError;
    });
    EXPECT_EQ(Create(1, 1, &ctx), ncclSystemError);
  }
  {
    ScopedHook fails(g_ibCastAccept, [](void*, void**, ncclNetDeviceHandle_t**) { return ncclSystemError; });
    EXPECT_EQ(Create(1, 1, &ctx), ncclSystemError);
  }
  {
    ScopedHook fails(g_ibCastIsend, [](void*, void*, size_t, int, void*, void*, void**) { return ncclSystemError; });
    EXPECT_EQ(Create(1, 1, &ctx), ncclSystemError);  // the barrier
  }
  EXPECT_EQ(ctx, nullptr);
}

// Only the context array is freed on failure; the comms already connected stay open.
// Pins AICOMRCCL-3094; a fix flips this pin.
TEST_F(NetIbCastGinContextMicrotest, CreateContext_KnownBug_FailureLeavesConnectedCommsOpen) {
  if (kAsan) GTEST_SKIP() << "a failure past the peer-comm arrays leaks them";
  cComm_.allGather = HandleAllGather;
  char ring[4 * sizeof(int)];
  base_ = ring;
  int connects = 0;
  ScopedHook secondFails(g_ibCastConnectImpl, [&](void*, int, void*, void** sendComm, ncclNetDeviceHandle_t**, int) {
    if (++connects == 2) return ncclSystemError;
    *sendComm = peerSends_[0].get();
    return ncclSuccess;
  });
  ScopedHook closeSend(g_ibCastCloseSend, [](void*) { return ncclSuccess; });
  ScopedHook closeRecv(g_ibCastCloseRecv, [](void*) { return ncclSuccess; });
  void* ctx = nullptr;
  EXPECT_EQ(Create(1, 1, &ctx), ncclSystemError);
  EXPECT_EQ(accept_.calls, 1);
  EXPECT_EQ(closeSend.calls, 0);
  EXPECT_EQ(closeRecv.calls, 0);
}

TEST_F(NetIbCastGinContextMicrotest, DestroyContext_ClosesEveryPeerCommOfEveryContext) {
  cComm_.allGather = HandleAllGather;
  char ring[4 * sizeof(int)];
  base_ = ring;
  void* ctx = nullptr;
  ASSERT_EQ(Create(2, 1, &ctx), ncclSuccess);
  ScopedHook closeRecv(g_ibCastCloseRecv, [](void*) { return ncclSuccess; });
  ScopedHook closeSend(g_ibCastCloseSend, [](void*) { return ncclSuccess; });
  ASSERT_EQ(IbCastRmaIbProxy.destroyContext(ctx), ncclSuccess);
  EXPECT_EQ(closeRecv.calls, 8);
  EXPECT_EQ(closeSend.calls, 8);
  auto* gc = static_cast<struct IbCastRmaIbProxyCtx*>(ctx);
  EXPECT_EQ(gc[1].fullSendComm, nullptr);
  EXPECT_EQ(gc[1].fullRecvComm, nullptr);
}

TEST_F(NetIbCastGinContextMicrotest, DestroyContext_CloseFails_ReturnsTheError) {
  cComm_.allGather = HandleAllGather;
  char ring[4 * sizeof(int)];
  base_ = ring;
  void* ctx = nullptr;
  ASSERT_EQ(Create(1, 1, &ctx), ncclSuccess);
  {
    ScopedHook closeRecv(g_ibCastCloseRecv, [](void*) { return ncclSystemError; });
    EXPECT_EQ(IbCastRmaIbProxyDestroyContext(ctx), ncclSystemError);
  }
  ScopedHook closeRecv(g_ibCastCloseRecv, [](void*) { return ncclSuccess; });
  ScopedHook closeSend(g_ibCastCloseSend, [](void*) { return ncclSystemError; });
  EXPECT_EQ(IbCastRmaIbProxyDestroyContext(ctx), ncclSystemError);
}

// The registration exchange: every rank's base VA and rkey, rank r's being 0x1000 * (r + 1) and r.
ncclResult_t MrAllGather(struct CastIbGinCollComm* cComm, void* src, void* recv, size_t len) {
  for (int r = 0; r < cComm->nranks; r++) {
    if (r == cComm->rank) {
      memcpy((char*)recv + r * len, src, len);
    } else if (len == sizeof(uintptr_t)) {
      ((uintptr_t*)recv)[r] = 0x1000 * (r + 1);
    } else {
      ((uint32_t*)recv)[r] = (uint32_t)r;
    }
  }
  return ncclSuccess;
}

TEST_F(NetIbCastGinMicrotest, RegMrSym_RegistersOnTheRecvCommAndGathersEveryRanksVaAndRkey) {
  cComm_.allGather = MrAllGather;
  char data[64];
  void* mhandle = nullptr;
  ASSERT_EQ(IbCastRmaIbProxy.regMrSym(&cComm_, data, sizeof(data), NCCL_PTR_HOST, 0, &mhandle), ncclSuccess);
  auto* h = static_cast<struct IbCastRmaProxyMrHandle*>(mhandle);
  EXPECT_EQ(h->mrHandle->mrs[0], mrs_[0].get());
  EXPECT_EQ(IbCastDevs[0].mrCache.population, 1);  // the recv comm's device
  EXPECT_EQ(h->base_vas[0], (uintptr_t)data);
  EXPECT_EQ(h->base_vas[1], 0x2000u);
  EXPECT_EQ(h->rkeys[0], mrs_[0]->rkey);
  EXPECT_EQ(h->rkeys[1], 1u);
  ASSERT_EQ(IbCastRmaIbProxy.deregMrSym(&cComm_, mhandle), ncclSuccess);
  EXPECT_EQ(IbCastDevs[0].mrCache.population, 0);
}

TEST_F(NetIbCastGinMicrotest, RegMrSymDmaBuf_PassesTheFdThrough) {
  cComm_.allGather = MrAllGather;
  char data[64];
  int fd = -2;
  ScopedHook dmabuf(g_ibvRegDmabufMr, [&](struct ibv_mr** ret, struct ibv_pd*, uint64_t, size_t, uint64_t, int f,
                                          int) {
    fd = f;
    *ret = NewMr();
    return ncclSuccess;
  });
  void* mhandle = nullptr;
  ASSERT_EQ(IbCastRmaIbProxy.regMrSymDmaBuf(&cComm_, data, sizeof(data), NCCL_PTR_CUDA, 0, 9, 0, &mhandle),
            ncclSuccess);
  EXPECT_EQ(fd, 9);
  ASSERT_EQ(IbCastRmaIbProxyDeregMrSym(&cComm_, mhandle), ncclSuccess);
}

TEST_F(NetIbCastGinMicrotest, RegMrSym_RegistrationOrExchangeFails_ReturnsTheError) {
  if (kAsan) GTEST_SKIP() << "the failed registration leaks the MR handle";
  char data[64];
  void* mhandle = nullptr;
  {
    ScopedHook reg(g_ibvRegMr, [](struct ibv_mr**, struct ibv_pd*, void*, size_t, int) { return ncclSystemError; });
    EXPECT_EQ(IbCastRmaIbProxyRegMrSym(&cComm_, data, sizeof(data), NCCL_PTR_HOST, 0, &mhandle), ncclSystemError);
  }
}

// A failed exchange returns without deregistering, so the MR stays in the device's cache.
// Pins AICOMRCCL-3094; a fix flips this pin.
TEST_F(NetIbCastGinMicrotest, RegMrSym_KnownBug_FailedExchangeLeavesTheMrRegistered) {
  if (kAsan) GTEST_SKIP() << "the failed exchange leaks the MR handle";
  char data[64];
  void* mhandle = nullptr;
  cComm_.allGather = FailingAllGather;
  EXPECT_EQ(IbCastRmaIbProxyRegMrSym(&cComm_, data, sizeof(data), NCCL_PTR_HOST, 0, &mhandle), ncclRemoteError);
  EXPECT_EQ(mhandle, nullptr);
  EXPECT_EQ(IbCastDevs[0].mrCache.population, 1);
}

TEST_F(NetIbCastGinMicrotest, DeregMrSym_DeregFails_ReturnsTheError) {
  if (kAsan) GTEST_SKIP() << "the failed deregistration leaks the MR handle";
  cComm_.allGather = MrAllGather;
  char data[64];
  void* mhandle = nullptr;
  ASSERT_EQ(IbCastRmaIbProxyRegMrSym(&cComm_, data, sizeof(data), NCCL_PTR_HOST, 0, &mhandle), ncclSuccess);
  ScopedHook dereg(g_ibvDeregMr, [](struct ibv_mr*) { return ncclSystemError; });
  EXPECT_EQ(IbCastRmaIbProxyDeregMrSym(&cComm_, mhandle), ncclSystemError);
}


class NetIbCastGinDataMicrotest : public NetIbCastGinMicrotest {
 protected:
  void SetUp() override {
    NetIbCastGinMicrotest::SetUp();
    proxyCtx_[0].rank = 0;
    proxyCtx_[0].nranks = 2;
    proxyCtx_[0].nContexts = 1;
    proxyCtx_[0].fullSendComm = sendComms_;
    proxyCtx_[0].fullRecvComm = recvComms_;
    sendComms_[1] = send_.get();
    recvComms_[1] = recv_.get();
    src_ = Handle(0x10, 0xa0, 0xb0);
    dst_ = Handle(0x20, 0xa1, 0xb1);
    sig_ = Handle(0x30, 0xa2, 0xb2);
    recv_->devs[0].gpuFlush.sge.lkey = 0xf1;
  }
  // A symmetric registration: rank r's buffer at base + r * 0x10000, with rkey rkey0 + r.
  struct IbCastRmaProxyMrHandle Handle(uintptr_t base, uint32_t lkey, uint32_t rkey0) {
    mrStore_.emplace_back(new ibv_mr{});
    mrStore_.back()->lkey = lkey;
    handleStore_.emplace_back(new ncclIbMrHandle{});
    handleStore_.back()->mrs[0] = mrStore_.back().get();
    vas_.emplace_back(new uintptr_t[2]{base, base + 0x10000});
    rkeys_.emplace_back(new uint32_t[2]{rkey0, rkey0 + 1});
    return {handleStore_.back().get(), vas_.back().get(), rkeys_.back().get()};
  }
  struct ibv_send_wr Posted(size_t i) const { return posted_.at(i); }

  struct IbCastRmaIbProxyCtx proxyCtx_[1] {};
  void* sendComms_[2] = {};
  void* recvComms_[2] = {};
  struct IbCastRmaProxyMrHandle src_, dst_, sig_;
  std::vector<std::unique_ptr<ibv_mr>> mrStore_;
  std::vector<std::unique_ptr<ncclIbMrHandle>> handleStore_;
  std::vector<std::unique_ptr<uintptr_t[]>> vas_;
  std::vector<std::unique_ptr<uint32_t[]>> rkeys_;
  std::vector<struct ibv_send_wr> posted_;
  std::vector<struct ibv_sge> postedSges_;
  int postResult_ = 0;
  HookOf<decltype(g_fakePostSend)> postSend_{g_fakePostSend, [this](struct ibv_qp* qp, struct ibv_send_wr* wr,
                                                                    struct ibv_send_wr**) {
                                               postedQp_ = qp;
                                               for (; wr; wr = wr->next) {
                                                 posted_.push_back(*wr);
                                                 postedSges_.push_back(wr->sg_list ? *wr->sg_list : ibv_sge{});
                                               }
                                               return postResult_;
                                             }};
  struct ibv_qp* postedQp_ = nullptr;
};

TEST_F(NetIbCastGinDataMicrotest, IPut_PostsOneSignaledWriteFromOurBufferToThePeers) {
  void* request = nullptr;
  ASSERT_EQ(IbCastRmaIbProxy.iput(proxyCtx_, 0, 8, &src_, 64, 16, &dst_, 1, 0, &request), ncclSuccess);
  ASSERT_EQ(posted_.size(), 1u);
  EXPECT_EQ(postedQp_, &qp_);
  const struct ibv_send_wr wr = Posted(0);
  EXPECT_EQ(wr.opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(wr.send_flags, (unsigned)IBV_SEND_SIGNALED);
  EXPECT_EQ(wr.wr.rdma.remote_addr, 0x20u + 0x10000 + 16);
  EXPECT_EQ(wr.wr.rdma.rkey, 0xb1u + 1);
  EXPECT_EQ(postedSges_[0].addr, 0x10u + 8);
  EXPECT_EQ(postedSges_[0].length, 64u);
  EXPECT_EQ(postedSges_[0].lkey, 0xa0u);
  auto* req = static_cast<struct ncclIbRequest*>(request);
  EXPECT_EQ(wr.wr_id, (uint64_t)(req - send_->base.reqs));
  EXPECT_EQ(req->type, NCCL_NET_IB_REQ_GIN_IPUT);
  EXPECT_EQ(req->iput.rank, 1);
  EXPECT_EQ(req->ginProxyCtx, &proxyCtx_[0]);
  EXPECT_EQ(req->sock, &send_->base.sock);
  EXPECT_EQ(req->events[0], 1);
  EXPECT_EQ(req->devBases[0], &send_->devs[0].base);
}

TEST_F(NetIbCastGinDataMicrotest, IGet_PostsOneSignaledReadFromThePeersBufferIntoOurs) {
  void* request = nullptr;
  ASSERT_EQ(IbCastRmaIbProxy.iget(proxyCtx_, 0, 4, &dst_, 32, 12, &src_, 1, 0, &request), ncclSuccess);
  ASSERT_EQ(posted_.size(), 1u);
  const struct ibv_send_wr wr = Posted(0);
  EXPECT_EQ(wr.opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(wr.send_flags, (unsigned)IBV_SEND_SIGNALED);
  EXPECT_EQ(wr.wr.rdma.remote_addr, 0x20u + 0x10000 + 4);
  EXPECT_EQ(wr.wr.rdma.rkey, 0xb2u);
  EXPECT_EQ(postedSges_[0].addr, 0x10u + 12);
  EXPECT_EQ(postedSges_[0].lkey, 0xa0u);
  EXPECT_EQ(static_cast<struct ncclIbRequest*>(request)->type, NCCL_NET_IB_REQ_GIN_IGET);
  EXPECT_EQ(static_cast<struct ncclIbRequest*>(request)->iget.rank, 1);
}

TEST_F(NetIbCastGinDataMicrotest, IPutSignal_WithData_ChainsAnUnsignaledWriteBeforeTheAtomicAdd) {
  void* request = nullptr;
  ASSERT_EQ(IbCastRmaIbProxy.iputSignal(proxyCtx_, 0, 0, &src_, 128, 0, &dst_, 1, 8, &sig_, 5, NCCL_NET_SIGNAL_OP_ADD,
                                         true, 0, &request),
            ncclSuccess);
  ASSERT_EQ(posted_.size(), 2u);
  EXPECT_EQ(Posted(0).opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(Posted(0).send_flags, 0u);
  EXPECT_EQ(postedSges_[0].length, 128u);
  const struct ibv_send_wr sig = Posted(1);
  EXPECT_EQ(sig.opcode, IBV_WR_ATOMIC_FETCH_AND_ADD);
  EXPECT_EQ(sig.send_flags, (unsigned)IBV_SEND_SIGNALED);
  EXPECT_EQ(sig.wr.atomic.remote_addr, 0x30u + 0x10000 + 8);
  EXPECT_EQ(sig.wr.atomic.compare_add, 5u);
  EXPECT_EQ(sig.wr.atomic.rkey, 0xb3u);
  EXPECT_EQ(postedSges_[1].addr, (uint64_t)(uintptr_t)&send_->putSignalScratchpad);
  EXPECT_EQ(postedSges_[1].lkey, 0x5cu);
  EXPECT_EQ(Posted(0).wr_id, Posted(1).wr_id);
  EXPECT_EQ(static_cast<struct ncclIbRequest*>(request)->events[0], 1);
}

TEST_F(NetIbCastGinDataMicrotest, IPutSignal_SizeZeroOrNoDestination_PostsOnlyTheSignal) {
  void* request = nullptr;
  ASSERT_EQ(IbCastRmaIbProxyIPutSignal(proxyCtx_, 0, 0, &src_, 0, 0, &dst_, 1, 0, &sig_, 5, NCCL_NET_SIGNAL_OP_INC, false,
                                       0, &request),
            ncclSuccess);
  ASSERT_EQ(posted_.size(), 1u);
  EXPECT_EQ(Posted(0).opcode, IBV_WR_ATOMIC_FETCH_AND_ADD);
  EXPECT_EQ(Posted(0).wr.atomic.compare_add, 1u);  // INC ignores the value
}

// With a size but no destination the PUT slot is left empty yet still posted, and the signal it
// should have chained to is never sent. Pins AICOMRCCL-3096; a fix flips this pin.
TEST_F(NetIbCastGinDataMicrotest, IPutSignal_KnownBug_DataWithoutADestinationPostsAnEmptyWrNotTheSignal) {
  void* request = nullptr;
  ASSERT_EQ(IbCastRmaIbProxyIPutSignal(proxyCtx_, 0, 0, &src_, 64, 0, nullptr, 1, 0, &sig_, 5, NCCL_NET_SIGNAL_OP_ADD,
                                       false, 0, &request),
            ncclSuccess);
  ASSERT_EQ(posted_.size(), 1u);
  EXPECT_EQ(Posted(0).num_sge, 0);
  EXPECT_EQ(Posted(0).sg_list, nullptr);
  EXPECT_NE(Posted(0).opcode, IBV_WR_ATOMIC_FETCH_AND_ADD);
}

TEST_F(NetIbCastGinDataMicrotest, IPutSignal_UnsupportedOp_IsRejectedBeforeAnyPost) {
  void* request = nullptr;
  EXPECT_EQ(IbCastRmaIbProxyIPutSignal(proxyCtx_, 0, 0, &src_, 8, 0, &dst_, 1, 0, &sig_, 1, 0x7, false, 0, &request),
            ncclInvalidArgument);
  EXPECT_TRUE(posted_.empty());
}

TEST_F(NetIbCastGinDataMicrotest, IFlush_ReadsOurOwnRegistrationThroughTheLoopbackFlushQp) {
  void* request = nullptr;
  ASSERT_EQ(IbCastRmaIbProxy.iflush(proxyCtx_, 0, &dst_, 1, &request), ncclSuccess);
  ASSERT_EQ(posted_.size(), 1u);
  EXPECT_EQ(postedQp_, &flushQp_);
  const struct ibv_send_wr wr = Posted(0);
  EXPECT_EQ(wr.opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(wr.wr.rdma.remote_addr, 0x20u);  // rank 0's VA, not the peer's
  EXPECT_EQ(wr.wr.rdma.rkey, 0xb1u);
  EXPECT_EQ(postedSges_[0].lkey, 0xf1u);
  auto* req = static_cast<struct ncclIbRequest*>(request);
  EXPECT_EQ(req->type, NCCL_NET_IB_REQ_FLUSH);
  EXPECT_EQ(req->sock, &recv_->base.sock);
  EXPECT_EQ(req->events[0], 1);
}

TEST_F(NetIbCastGinDataMicrotest, DataPath_NoRequestOrPostFails_ReturnsTheError) {
  void* request = nullptr;
  postResult_ = EIO;
  EXPECT_EQ(IbCastRmaIbProxyIPut(proxyCtx_, 0, 0, &src_, 8, 0, &dst_, 1, 0, &request), ncclSystemError);
  EXPECT_EQ(IbCastRmaIbProxyIGet(proxyCtx_, 0, 0, &dst_, 8, 0, &src_, 1, 0, &request), ncclSystemError);
  EXPECT_EQ(IbCastRmaIbProxyIPutSignal(proxyCtx_, 0, 0, &src_, 8, 0, &dst_, 1, 0, &sig_, 1, NCCL_NET_SIGNAL_OP_INC, false,
                                       0, &request),
            ncclSystemError);
  EXPECT_EQ(IbCastRmaIbProxyIFlush(proxyCtx_, 0, &dst_, 1, &request), ncclSystemError);
  ScopedHook none(g_ibCastGetRequest, [](struct ncclIbNetCommBase*, struct ncclIbRequest**) {
    return ncclInternalError;
  });
  EXPECT_EQ(IbCastRmaIbProxyIPut(proxyCtx_, 0, 0, &src_, 8, 0, &dst_, 1, 0, &request), ncclInternalError);
  EXPECT_EQ(IbCastRmaIbProxyIGet(proxyCtx_, 0, 0, &dst_, 8, 0, &src_, 1, 0, &request), ncclInternalError);
  EXPECT_EQ(IbCastRmaIbProxyIPutSignal(proxyCtx_, 0, 0, &src_, 8, 0, &dst_, 1, 0, &sig_, 1, NCCL_NET_SIGNAL_OP_INC, false,
                                       0, &request),
            ncclInternalError);
  EXPECT_EQ(IbCastRmaIbProxyIFlush(proxyCtx_, 0, &dst_, 1, &request), ncclInternalError);
}

class NetIbCastGinTestMicrotest : public NetIbCastGinDataMicrotest {
 protected:
  // Completions handed out by the next poll, as (request, status) pairs.
  void Completions(std::vector<std::pair<struct ncclIbRequest*, enum ibv_wc_status>> wcs) {
    wcs_ = std::move(wcs);
  }
  std::vector<std::pair<struct ncclIbRequest*, enum ibv_wc_status>> wcs_;
  int pollResult_ = 0;
  HookOf<decltype(g_fakePollCq)> pollCq_{g_fakePollCq, [this](struct ibv_cq* cq, int n, struct ibv_wc* wc) {
                                           EXPECT_EQ(cq, &cq_);
                                           EXPECT_EQ(n, 4);
                                           if (pollResult_ < 0) return pollResult_;
                                           int i = 0;
                                           for (auto& [req, status] : wcs_) {
                                             memset(&wc[i], 0, sizeof(wc[i]));
                                             wc[i].wr_id = req - req->base->reqs;
                                             wc[i].status = status;
                                             wc[i].opcode = IBV_WC_RDMA_WRITE;
                                             wc[i].vendor_err = 0x81;
                                             i++;
                                           }
                                           wcs_.clear();
                                           return i;
                                         }};
};

TEST_F(NetIbCastGinTestMicrotest, Test_CompletesARequestOnceItsLastEventArrives) {
  void* a = nullptr;
  void* b = nullptr;
  ASSERT_EQ(IbCastRmaIbProxyIPut(proxyCtx_, 0, 0, &src_, 8, 0, &dst_, 1, 0, &a), ncclSuccess);
  ASSERT_EQ(IbCastRmaIbProxyIPut(proxyCtx_, 0, 0, &src_, 8, 0, &dst_, 1, 0, &b), ncclSuccess);
  auto* ra = static_cast<struct ncclIbRequest*>(a);
  auto* rb = static_cast<struct ncclIbRequest*>(b);
  int done = -1;
  ASSERT_EQ(IbCastRmaIbProxy.test(&cComm_, a, &done), ncclSuccess);  // nothing yet
  EXPECT_EQ(done, 0);
  // b's completion arrives while polling for a: it is charged to b, a stays pending.
  Completions({{rb, IBV_WC_SUCCESS}});
  ASSERT_EQ(IbCastRmaIbProxyTest(&cComm_, a, &done), ncclSuccess);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(rb->events[0], 0);
  Completions({{ra, IBV_WC_SUCCESS}});
  ASSERT_EQ(IbCastRmaIbProxyTest(&cComm_, a, &done), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(ra->type, NCCL_NET_IB_REQ_UNUSED);
  // b already has every event: done without polling.
  const int polls = pollCq_.calls;
  ASSERT_EQ(IbCastRmaIbProxyTest(&cComm_, b, &done), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(pollCq_.calls, polls);
  EXPECT_EQ(rb->type, NCCL_NET_IB_REQ_UNUSED);
}

TEST_F(NetIbCastGinTestMicrotest, Test_FlushRequest_PollsTheRecvCommsCq) {
  struct ibv_cq recvCq {};
  recvCq.context = &ctx_;
  recv_->devs[0].base.cq = &recvCq;
  void* f = nullptr;
  ASSERT_EQ(IbCastRmaIbProxyIFlush(proxyCtx_, 0, &dst_, 1, &f), ncclSuccess);
  int done = -1;
  ScopedHook poll(g_fakePollCq, [&](struct ibv_cq* cq, int, struct ibv_wc*) {
    EXPECT_EQ(cq, &recvCq);
    return 0;
  });
  ASSERT_EQ(IbCastRmaIbProxyTest(&cComm_, f, &done), ncclSuccess);
  EXPECT_EQ(poll.calls, 1);
}

TEST_F(NetIbCastGinTestMicrotest, Test_PollFails_ReturnsTheError) {
  void* a = nullptr;
  ASSERT_EQ(IbCastRmaIbProxyIPut(proxyCtx_, 0, 0, &src_, 8, 0, &dst_, 1, 0, &a), ncclSuccess);
  pollResult_ = -1;
  int done = -1;
  EXPECT_EQ(IbCastRmaIbProxyTest(&cComm_, a, &done), ncclSystemError);
}

TEST_F(NetIbCastGinTestMicrotest, Test_ErrorCompletion_IsARemoteErrorNamingTheHcaAndGids) {
  send_->devs[0].base.gidInfo.link_layer = IBV_LINK_LAYER_ETHERNET;
  send_->devs[0].base.gidInfo.localGid.raw[15] = 1;
  send_->base.remDevs[0].remoteGid.raw[15] = 2;
  struct ibv_context hcaCtx {};
  struct ibv_device hca {};
  snprintf(hca.name, sizeof(hca.name), "ionic_7");
  hcaCtx.device = &hca;
  struct ibv_pd hcaPd {};
  hcaPd.context = &hcaCtx;
  send_->devs[0].base.pd = &hcaPd;
  void* a = nullptr;
  ASSERT_EQ(IbCastRmaIbProxyIPut(proxyCtx_, 0, 0, &src_, 8, 0, &dst_, 1, 0, &a), ncclSuccess);
  Completions({{static_cast<struct ncclIbRequest*>(a), IBV_WC_REM_ACCESS_ERR}});
  int done = -1;
  std::string log = CaptureLog([&] { EXPECT_EQ(IbCastRmaIbProxyTest(&cComm_, a, &done), ncclRemoteError); });
  EXPECT_TRUE(LogHas(log, "localGid ::1 remoteGids::2")) << log;
  EXPECT_TRUE(LogHas(log, "hca ionic_7")) << log;
  EXPECT_EQ(done, 0);
}

// The error report now indexes devBases[] by the request's own device, not by the completion's
// place in the poll batch, so an error in a later completion slot of a one-device comm is reported
// cleanly as a remote error instead of crashing. Fixes AICOMRCCL-3095 (ported from net_ib under
// AICOMRCCL-2750); this flips the former known-bug pin.
TEST_F(NetIbCastGinTestMicrotest, Test_ErrorInTheSecondCompletionSlot_IsReportedWithoutCrashing) {
  struct ibv_context hcaCtx {};
  struct ibv_device hca {};
  snprintf(hca.name, sizeof(hca.name), "ionic_7");
  hcaCtx.device = &hca;
  struct ibv_pd hcaPd {};
  hcaPd.context = &hcaCtx;
  send_->devs[0].base.pd = &hcaPd;
  void* a = nullptr;
  void* b = nullptr;
  ASSERT_EQ(IbCastRmaIbProxyIPut(proxyCtx_, 0, 0, &src_, 8, 0, &dst_, 1, 0, &a), ncclSuccess);
  ASSERT_EQ(IbCastRmaIbProxyIPut(proxyCtx_, 0, 0, &src_, 8, 0, &dst_, 1, 0, &b), ncclSuccess);
  Completions({{static_cast<struct ncclIbRequest*>(b), IBV_WC_SUCCESS},
               {static_cast<struct ncclIbRequest*>(b), IBV_WC_REM_ACCESS_ERR}});
  int done = 0;
  EXPECT_EQ(IbCastRmaIbProxyTest(&cComm_, a, &done), ncclRemoteError);
}

TEST_F(NetIbCastGinTestMicrotest, Test_ErrorCompletionOnInfiniband_LeavesTheGidsOut) {
  struct ibv_context hcaCtx {};
  struct ibv_device hca {};
  hcaCtx.device = &hca;
  struct ibv_pd hcaPd {};
  hcaPd.context = &hcaCtx;
  send_->devs[0].base.pd = &hcaPd;
  send_->devs[0].base.gidInfo.link_layer = IBV_LINK_LAYER_INFINIBAND;
  void* a = nullptr;
  ASSERT_EQ(IbCastRmaIbProxyIPut(proxyCtx_, 0, 0, &src_, 8, 0, &dst_, 1, 0, &a), ncclSuccess);
  Completions({{static_cast<struct ncclIbRequest*>(a), IBV_WC_RETRY_EXC_ERR}});
  int done = -1;
  std::string log = CaptureLog([&] { EXPECT_EQ(IbCastRmaIbProxyTest(&cComm_, a, &done), ncclRemoteError); });
  EXPECT_FALSE(LogHas(log, "localGid")) << log;
}

}  // namespace
