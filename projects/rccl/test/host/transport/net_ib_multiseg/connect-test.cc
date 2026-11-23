/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib_multiseg/connect.cc: net_ib/connect-test.cc against the fork
 * (derived_transport_tests.cmake), then the fork's multi-segment tests.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cctype>
#include <fstream>
#include <sstream>
#include <string>

#include "IbVerbsWorld.h"
#include "fakes/rccl_mseg_fakes.h"

#undef NET_IB_CONNECT_CC_PATH
#define NET_IB_CONNECT_CC_PATH "transport/net_ib_multiseg/shim/connect.cc"
#define NetIbConnectMicrotest NetIbMsegConnectMicrotest
#define NetIbConnectCuMemFlushMicrotest NetIbMsegConnectCuMemFlushMicrotest
#include "transport/derived_transport_tests_begin.h"
#include "transport/net_ib/connect-test.cc"
#include "transport/derived_transport_tests_end.h"

namespace {

// The develop fixtures do not know the fork's fakes; put them back after every test.
class ResetRcclMsegFakesListener : public ::testing::EmptyTestEventListener {
  void OnTestEnd(const ::testing::TestInfo&) override { ResetRcclMsegFakes(); }
};
const bool g_resetRcclMsegFakes =
    (::testing::UnitTest::GetInstance()->listeners().Append(new ResetRcclMsegFakesListener), true);

// The cached decision needs the switch, and a GIN or resilient comm never takes it.
TEST(NetIbMsegConnectUseMultiSeg, NeedsTheSwitchAndIsNotGin) {
  EXPECT_TRUE(IbMsegUseMultiSeg(/*multiSegOn*/ true, /*forGin*/ false, /*resilient*/ false));
  EXPECT_FALSE(IbMsegUseMultiSeg(/*multiSegOn*/ true, /*forGin*/ true, /*resilient*/ false));
  EXPECT_FALSE(IbMsegUseMultiSeg(/*multiSegOn*/ false, /*forGin*/ false, /*resilient*/ false));
  EXPECT_FALSE(IbMsegUseMultiSeg(/*multiSegOn*/ false, /*forGin*/ true, /*resilient*/ false));
  EXPECT_FALSE(IbMsegUseMultiSeg(/*multiSegOn*/ true, /*forGin*/ false, /*resilient*/ true));
}

// Records the useMultiSeg of each metadata this side sends, then hands the transfer to the scripted peer.
class SentMetaRecorder {
 public:
  SentMetaRecorder()
      : next_(g_ncclSocketProgress),
        hook_(g_ncclSocketProgress, [this](int op, ncclSocket* sock, void* data, int size, int* offset, bool* closed) {
          if (op == NCCL_SOCKET_SEND && size == static_cast<int>(sizeof(IbMsegConnectionMetadata))) {
            useMultiSeg = static_cast<IbMsegConnectionMetadata*>(data)->useMultiSeg;
          }
          return next_(op, sock, data, size, offset, closed);
        }) {}

  int useMultiSeg = -1;

 private:
  std::function<ncclResult_t(int, ncclSocket*, void*, int, int*, bool*)> next_;
  ScopedHook<ncclResult_t(int, ncclSocket*, void*, int, int*, bool*)> hook_;
};

// rcclMultiSegOn true: both ends of a classic comm use the CTS slots; GIN comms keep the legacy FIFO.
class NetIbMsegConnectMultiSegMicrotest : public NetIbMsegConnectMicrotest {
 protected:
  void SetUp() override {
    ConfigureSingleDevice();
    InstallSingleDeviceResourceFakes();
    g_multiSegOn = [this](bool pluginRegistersPerSegment) { return pluginRegistersPerSegment && multiSegOn_; };
  }

  static IbMsegConnectionMetadata Peer(uint64_t ctsAddr, uint32_t qpn, bool useMultiSeg) {
    IbMsegConnectionMetadata meta = PeerMetadata(ctsAddr, qpn, /*lid=*/8, /*rkey=*/0x91, Ipv6Gid("2001:db8::2"));
    meta.useMultiSeg = useMultiSeg ? 1 : 0;
    return meta;
  }

  IbMsegSendComm* Connect(IbMsegHandle* handle, int nQpsPerDev = 1) {
    g_ncclSocketConnect = [](ncclSocket*) { return ncclSuccess; };
    g_ncclSocketReady = [](ncclSocket*, int* ready) {
      *ready = 1;
      return ncclSuccess;
    };
    void* sendComm = nullptr;
    for (int step = 0; step < 10 && sendComm == nullptr; ++step) {
      EXPECT_EQ(IbMsegConnectImpl(nullptr, 0, handle, &sendComm, nullptr, nQpsPerDev, /*envTrafficClass=*/17),
                ncclSuccess);
    }
    return static_cast<IbMsegSendComm*>(sendComm);
  }

  IbMsegRecvComm* Accept(IbMsegListenComm* listenComm) {
    g_ncclSocketAccept = [](ncclSocket*, ncclSocket*, bool) { return ncclSuccess; };
    g_ncclSocketReady = [](ncclSocket*, int* ready) {
      *ready = 1;
      return ncclSuccess;
    };
    void* recvComm = nullptr;
    for (int step = 0; step < 10 && recvComm == nullptr; ++step) {
      EXPECT_EQ(IbMsegAcceptImpl(listenComm, &recvComm, nullptr, /*nQpsPerDev=*/1), ncclSuccess);
    }
    return static_cast<IbMsegRecvComm*>(recvComm);
  }

  const ibv_mr* MrAt(const void* addr) const {
    for (const auto& mr : mrs_) {
      if (mr.addr == addr) return &mr;
    }
    return nullptr;
  }

  bool multiSegOn_ = true;
};

TEST_F(NetIbMsegConnectMultiSegMicrotest, Connect_ClassicCommRegistersCtsSlots) {
  IbMsegHandle handle{};
  handle.magic = NCCL_SOCKET_MAGIC;
  ScriptedHandshakePeer peer([&handle] { return static_cast<int>(handle.stage.state); },
                             Peer(/*ctsAddr=*/0x9000, /*qpn=*/200, /*useMultiSeg=*/true));
  SentMetaRecorder sent;

  IbMsegSendComm* comm = Connect(&handle);
  ASSERT_NE(comm, nullptr);
  EXPECT_EQ(sent.useMultiSeg, 1);
  ASSERT_NE(comm->ctsSlots, nullptr);
  EXPECT_EQ(comm->ctsFifo, nullptr);
  const ibv_mr* ctsMr = MrAt(comm->ctsSlots);
  ASSERT_NE(ctsMr, nullptr);
  EXPECT_EQ(ctsMr->length, NCCL_IB_CTS_SLOTS_BYTES);
  EXPECT_EQ(comm->useMultiSeg, 1);

  EXPECT_EQ(IbMsegCloseSend(comm), ncclSuccess);
}

TEST_F(NetIbMsegConnectMultiSegMicrotest, Connect_GinCommKeepsTheLegacyFifo) {
  IbMsegHandle handle{};
  handle.magic = NCCL_SOCKET_MAGIC;
  handle.isGin = 1;
  ScriptedHandshakePeer peer([&handle] { return static_cast<int>(handle.stage.state); },
                             Peer(/*ctsAddr=*/0x9000, /*qpn=*/200, /*useMultiSeg=*/false));

  IbMsegSendComm* comm = Connect(&handle);
  ASSERT_NE(comm, nullptr);
  EXPECT_EQ(comm->ctsSlots, nullptr);
  EXPECT_EQ(comm->useMultiSeg, 0);
  const ibv_mr* ctsMr = MrAt(comm->ctsFifo);
  ASSERT_NE(ctsMr, nullptr);
  EXPECT_EQ(ctsMr->length, NCCL_IB_CTS_FIFO_BYTES);

  EXPECT_EQ(IbMsegCloseSend(comm), ncclSuccess);
}

// Only GIN/RMA connects pass nQpsPerDev, and those QPs get the deeper RMA send queue.
TEST_F(NetIbMsegConnectMultiSegMicrotest, Connect_NQpsPerDevMarksRmaAndSizesSendQueue) {
  auto baseCreate = g_ibvCreateQp;
  std::vector<uint32_t> depths;
  ScopedHook create(g_ibvCreateQp, [&](ibv_qp** out, ibv_pd* pd, ibv_qp_init_attr* attr) {
    depths.push_back(attr->cap.max_send_wr);
    return baseCreate(out, pd, attr);
  });
  for (int nQpsPerDev : {0, 1}) {
    SCOPED_TRACE(nQpsPerDev);
    depths.clear();
    IbMsegHandle handle{};
    handle.magic = NCCL_SOCKET_MAGIC;
    ScriptedHandshakePeer peer([&handle] { return static_cast<int>(handle.stage.state); },
                               Peer(/*ctsAddr=*/0x9000, /*qpn=*/200, /*useMultiSeg=*/true));

    IbMsegSendComm* comm = Connect(&handle, nQpsPerDev);
    ASSERT_NE(comm, nullptr);
    EXPECT_EQ(comm->base.isRma, nQpsPerDev > 0);
    ASSERT_FALSE(depths.empty());
    const uint32_t expected = nQpsPerDev > 0 ? RCCL_RMA_SEND_QP_WRS : 2 * NET_IB_MAX_REQUESTS;
    for (uint32_t depth : depths) EXPECT_EQ(depth, expected);

    EXPECT_EQ(IbMsegCloseSend(comm), ncclSuccess);
  }
}

// The receiver allocates the slots from its own decision; the sender reaches the same one.
TEST_F(NetIbMsegConnectMultiSegMicrotest, Accept_ClassicListenAllocatesCtsSlotsAndGinListenDoesNot) {
  for (bool isGin : {false, true}) {
    SCOPED_TRACE(isGin);
    IbMsegListenComm listenComm{};
    listenComm.isGin = isGin;
    ScriptedHandshakePeer peer([&listenComm] { return static_cast<int>(listenComm.stage->state); },
                               Peer(/*ctsAddr=*/0xa000, /*qpn=*/300, /*useMultiSeg=*/!isGin));
    SentMetaRecorder sent;

    IbMsegRecvComm* comm = Accept(&listenComm);
    ASSERT_NE(comm, nullptr);
    EXPECT_EQ(comm->useMultiSeg, isGin ? 0 : 1);
    EXPECT_EQ(sent.useMultiSeg, isGin ? 0 : 1);
    EXPECT_EQ(comm->remCtsFifo.addr, 0xa000u);
    if (isGin) {
      EXPECT_EQ(comm->ctsSlots, nullptr);
    } else {
      ASSERT_NE(comm->ctsSlots, nullptr);
      EXPECT_EQ(comm->ctsSlotStride, sizeof(IbMsegCtsSlot));
      ASSERT_NE(comm->devs[0].ctsFifoMr, nullptr);
      EXPECT_EQ(comm->devs[0].ctsFifoMr->addr, static_cast<void*>(comm->ctsSlots));
      EXPECT_EQ(comm->devs[0].ctsFifoMr->length, NCCL_IB_CTS_SLOTS_BYTES);
    }

    EXPECT_EQ(IbMsegCloseRecv(comm), ncclSuccess);
  }
}

TEST_F(NetIbMsegConnectMultiSegMicrotest, ConnectAndAccept_SwitchOffKeepTheLegacyCts) {
  multiSegOn_ = false;
  IbMsegHandle handle{};
  handle.magic = NCCL_SOCKET_MAGIC;
  ScriptedHandshakePeer sendPeer([&handle] { return static_cast<int>(handle.stage.state); },
                                 Peer(/*ctsAddr=*/0x9000, /*qpn=*/200, /*useMultiSeg=*/false));
  IbMsegSendComm* send = Connect(&handle);
  ASSERT_NE(send, nullptr);
  EXPECT_EQ(send->useMultiSeg, 0);
  EXPECT_EQ(send->ctsSlots, nullptr);
  EXPECT_NE(send->ctsFifo, nullptr);
  EXPECT_EQ(IbMsegCloseSend(send), ncclSuccess);

  IbMsegListenComm listenComm{};
  ScriptedHandshakePeer recvPeer([&listenComm] { return static_cast<int>(listenComm.stage->state); },
                                 Peer(/*ctsAddr=*/0xa000, /*qpn=*/300, /*useMultiSeg=*/false));
  IbMsegRecvComm* recv = Accept(&listenComm);
  ASSERT_NE(recv, nullptr);
  EXPECT_EQ(recv->useMultiSeg, 0);
  EXPECT_EQ(recv->ctsSlots, nullptr);
  EXPECT_EQ(IbMsegCloseRecv(recv), ncclSuccess);
}

// Only a multi-segment comm pays for the send cursor and layout cache; it drops the receiver's legacy FIFO shadow.
TEST_F(NetIbMsegConnectMultiSegMicrotest, ConnectAndAccept_MultiSegStateFollowsUseMultiSeg) {
  for (bool on : {true, false}) {
    SCOPED_TRACE(on);
    multiSegOn_ = on;
    IbMsegHandle handle{};
    handle.magic = NCCL_SOCKET_MAGIC;
    ScriptedHandshakePeer sendPeer([&handle] { return static_cast<int>(handle.stage.state); },
                                   Peer(/*ctsAddr=*/0x9000, /*qpn=*/200, /*useMultiSeg=*/on));
    IbMsegSendComm* send = Connect(&handle);
    ASSERT_NE(send, nullptr);
    EXPECT_EQ(send->useMultiSeg, on ? 1 : 0);
    EXPECT_EQ(send->sendCursor != nullptr, on);
    EXPECT_EQ(send->layoutCache != nullptr, on);
    EXPECT_EQ(IbMsegCloseSend(send), ncclSuccess);

    IbMsegListenComm listenComm{};
    ScriptedHandshakePeer recvPeer([&listenComm] { return static_cast<int>(listenComm.stage->state); },
                                   Peer(/*ctsAddr=*/0xa000, /*qpn=*/300, /*useMultiSeg=*/on));
    IbMsegRecvComm* recv = Accept(&listenComm);
    ASSERT_NE(recv, nullptr);
    EXPECT_EQ(recv->useMultiSeg, on ? 1 : 0);
    if (on) {
      EXPECT_EQ(recv->remCtsFifo.elems, nullptr);
    } else {
      ASSERT_NE(recv->remCtsFifo.elems, nullptr);
      ASSERT_NE(recv->devs[0].ctsFifoMr, nullptr);
      EXPECT_EQ(recv->devs[0].ctsFifoMr->addr, static_cast<void*>(recv->remCtsFifo.elems));
      EXPECT_EQ(recv->devs[0].ctsFifoMr->length, NCCL_IB_CTS_FIFO_BYTES);
    }
    EXPECT_EQ(IbMsegCloseRecv(recv), ncclSuccess);
  }
}

// Resilient comms replay whole slots, so connect and accept keep the legacy CTS with the switch on.
class NetIbMsegConnectResilientMicrotest : public NetIbMsegConnectMultiSegMicrotest {
 protected:
  void SetUp() override {
    NetIbMsegConnectMultiSegMicrotest::SetUp();
    g_ncclIbResiliencyInit = [this](IbMsegNetCommBase*, IbMsegResiliency** out) {
      *out = &resiliency_;
      return ncclSuccess;
    };
    g_ncclSocketConnect = [](ncclSocket*) { return ncclSuccess; };
    g_ncclSocketAccept = [](ncclSocket*, ncclSocket*, bool) { return ncclSuccess; };
    // The decision is made before the socket is ready, so the first step shows it.
    g_ncclSocketReady = [](ncclSocket*, int* ready) {
      *ready = 0;
      return ncclSuccess;
    };
  }

  IbMsegResiliency resiliency_{};
};

TEST_F(NetIbMsegConnectResilientMicrotest, Connect_ResilientCommKeepsTheLegacyCts) {
  IbMsegHandle handle{};
  handle.magic = NCCL_SOCKET_MAGIC;
  ScriptedHandshakePeer peer([&handle] { return static_cast<int>(handle.stage.state); },
                             Peer(/*ctsAddr=*/0x9000, /*qpn=*/200, /*useMultiSeg=*/false));
  void* sendComm = nullptr;
  ASSERT_EQ(IbMsegConnectImpl(nullptr, 0, &handle, &sendComm, nullptr, /*nQpsPerDev=*/1, /*envTrafficClass=*/17),
            ncclSuccess);
  EXPECT_EQ(sendComm, nullptr);
  auto* comm = static_cast<IbMsegSendComm*>(handle.stage.comm);
  ASSERT_NE(comm, nullptr);
  EXPECT_EQ(comm->base.resiliency, &resiliency_);
  EXPECT_EQ(comm->useMultiSeg, 0);
  EXPECT_EQ(comm->ctsSlots, nullptr);
  EXPECT_NE(comm->ctsFifo, nullptr);
  EXPECT_EQ(comm->sendCursor, nullptr);
  EXPECT_EQ(comm->layoutCache, nullptr);
  EXPECT_EQ(IbMsegCloseSend(comm), ncclSuccess);
}

TEST_F(NetIbMsegConnectResilientMicrotest, Accept_ResilientCommKeepsTheLegacyCts) {
  IbMsegListenComm listenComm{};
  ScriptedHandshakePeer peer([&listenComm] { return static_cast<int>(listenComm.stage->state); },
                             Peer(/*ctsAddr=*/0xa000, /*qpn=*/300, /*useMultiSeg=*/false));
  void* recvComm = nullptr;
  ASSERT_EQ(IbMsegAcceptImpl(&listenComm, &recvComm, nullptr, /*nQpsPerDev=*/1), ncclSuccess);
  EXPECT_EQ(recvComm, nullptr);
  ASSERT_NE(listenComm.stage, nullptr);
  auto* comm = static_cast<IbMsegRecvComm*>(listenComm.stage->comm);
  ASSERT_NE(comm, nullptr);
  EXPECT_EQ(comm->base.resiliency, &resiliency_);
  EXPECT_EQ(comm->useMultiSeg, 0);
  EXPECT_EQ(IbMsegCloseRecv(comm), ncclSuccess);
  free(listenComm.stage->buffer);
  free(listenComm.stage);
  listenComm.stage = nullptr;
}

TEST_F(NetIbMsegConnectMultiSegMicrotest, Connect_CuMemOffStillUsesCtsSlots) {
  g_cuMemEnable = [] { return 0; };
  IbMsegHandle handle{};
  handle.magic = NCCL_SOCKET_MAGIC;
  ScriptedHandshakePeer peer([&handle] { return static_cast<int>(handle.stage.state); },
                             Peer(/*ctsAddr=*/0x9000, /*qpn=*/200, /*useMultiSeg=*/true));

  IbMsegSendComm* comm = Connect(&handle);
  ASSERT_NE(comm, nullptr);
  EXPECT_EQ(comm->useMultiSeg, 1);
  EXPECT_NE(comm->ctsSlots, nullptr);

  EXPECT_EQ(IbMsegCloseSend(comm), ncclSuccess);
}

// A peer whose multi-segment setting differs fails the connect or accept instead of misreading the CTS.
TEST_F(NetIbMsegConnectMultiSegMicrotest, ConnectAndAccept_PeerWithTheOtherSettingFails) {
  for (bool localOn : {true, false}) {
    SCOPED_TRACE(localOn);
    multiSegOn_ = localOn;
    g_ncclSocketConnect = [](ncclSocket*) { return ncclSuccess; };
    g_ncclSocketAccept = [](ncclSocket*, ncclSocket*, bool) { return ncclSuccess; };
    g_ncclSocketReady = [](ncclSocket*, int* ready) {
      *ready = 1;
      return ncclSuccess;
    };

    IbMsegHandle handle{};
    handle.magic = NCCL_SOCKET_MAGIC;
    ScriptedHandshakePeer sendPeer([&handle] { return static_cast<int>(handle.stage.state); },
                                   Peer(/*ctsAddr=*/0x9000, /*qpn=*/200, /*useMultiSeg=*/!localOn));
    void* sendComm = nullptr;
    ncclResult_t sendRet = ncclSuccess;
    for (int step = 0; step < 10 && sendComm == nullptr && sendRet == ncclSuccess; ++step) {
      sendRet = IbMsegConnectImpl(nullptr, 0, &handle, &sendComm, nullptr, /*nQpsPerDev=*/1, /*envTrafficClass=*/17);
    }
    EXPECT_EQ(sendRet, ncclInvalidUsage);
    EXPECT_EQ(sendComm, nullptr);

    IbMsegListenComm listenComm{};
    ScriptedHandshakePeer recvPeer([&listenComm] { return static_cast<int>(listenComm.stage->state); },
                                   Peer(/*ctsAddr=*/0xa000, /*qpn=*/300, /*useMultiSeg=*/!localOn));
    SentMetaRecorder sent;
    void* recvComm = nullptr;
    ncclResult_t recvRet = ncclSuccess;
    for (int step = 0; step < 10 && recvComm == nullptr && recvRet == ncclSuccess; ++step) {
      recvRet = IbMsegAcceptImpl(&listenComm, &recvComm, nullptr, /*nQpsPerDev=*/1);
    }
    EXPECT_EQ(recvRet, ncclInvalidUsage);
    EXPECT_EQ(recvComm, nullptr);
    EXPECT_EQ(sent.useMultiSeg, -1); // the accept fails before it replies
  }
}

// Branch: with multi-segment on (rcclMultiSegOn) a GIN/RMA comm's sender QP holds a full chain per in-flight
// request; NET comms and multi-segment off keep the base.
TEST_F(NetIbMsegConnectMicrotest, SenderQps_MultiSegmentRegisterSetsSendQueueDepth) {
  ConfigureSingleDevice();
  InstallSingleDeviceResourceFakes();
  auto baseCreate = g_ibvCreateQp;
  std::vector<uint32_t> depths;
  ScopedHook create(g_ibvCreateQp, [&](ibv_qp** out, ibv_pd* pd, ibv_qp_init_attr* attr) {
    depths.push_back(attr->cap.max_send_wr);
    return baseCreate(out, pd, attr);
  });
  for (bool multiSeg : {false, true}) {
    for (bool isRma : {false, true}) {
      g_multiSegOn = [multiSeg](bool pluginRegistersPerSegment) { return pluginRegistersPerSegment && multiSeg; };
      IbMsegSendComm comm{};
      comm.base.vProps.ndevs = 1;
      comm.base.nqps = 1;
      comm.base.isRma = isRma;
      comm.devs[0].base.ibDevN = 0;
      comm.devs[0].base.pd = &pd_;
      comm.devs[0].base.cq = &cq_;
      comm.devs[0].base.gidInfo = IbMsegDevs[0].gidInfo;
      IbMsegConnectionMetadata local{};
      const int expected = multiSeg && isRma ? RCCL_RMA_SEND_QP_WRS : 2 * NET_IB_MAX_REQUESTS;

      ASSERT_EQ(IbMsegSenderQpsCreate(&comm, &local), ncclSuccess);
      ASSERT_FALSE(depths.empty());
      EXPECT_EQ(depths.back(), static_cast<uint32_t>(expected)) << "multiSeg=" << multiSeg << " isRma=" << isRma;
    }
  }
}

// Branch: each loopback flush QP records its own device index. In-flight flush chains fit the base depth either way.
TEST_F(NetIbMsegConnectMicrotest, CreateFlushQp_KeepsBaseDepthAndLoopbackDeviceIndex) {
  ConfigureSingleDevice();
  InstallSingleDeviceResourceFakes();
  auto baseParams = g_loadParam;
  auto baseCreate = g_ibvCreateQp;
  std::vector<uint32_t> depths;
  ScopedHook create(g_ibvCreateQp, [&](ibv_qp** out, ibv_pd* pd, ibv_qp_init_attr* attr) {
    depths.push_back(attr->cap.max_send_wr);
    return baseCreate(out, pd, attr);
  });
  for (int64_t multiSeg : {0, 1}) {
    g_loadParam = [baseParams, multiSeg](const char* name, int64_t fallback) {
      return std::strcmp(name, "MULTI_SEGMENT_REGISTER") == 0 ? multiSeg : baseParams(name, fallback);
    };
    IbMsegRecvComm comm{};
    comm.flushEnabled = 1;
    comm.base.vProps.ndevs = 2;
    for (int d = 0; d < 2; d++) {
      comm.devs[d].base.ibDevN = 0;
      comm.devs[d].base.pd = &pd_;
      comm.devs[d].base.cq = &cq_;
      comm.devs[d].base.gidInfo = IbMsegDevs[0].gidInfo;
      comm.devs[d].gpuFlush.qp.devIndex = -1;
      comm.devs[d].gpuFlush.qp.remDevIdx = -1;
    }
    depths.clear();

    ASSERT_EQ(IbMsegCreateFlushQp(&comm), ncclSuccess);
    ASSERT_EQ(depths.size(), 2u);
    for (int d = 0; d < 2; d++) {
      EXPECT_EQ(comm.devs[d].gpuFlush.qp.devIndex, d);
      EXPECT_EQ(comm.devs[d].gpuFlush.qp.remDevIdx, d);
      EXPECT_EQ(depths[d], static_cast<uint32_t>(2 * NET_IB_MAX_REQUESTS)) << "multiSeg=" << multiSeg;
    }
  }
}

// Branch: a peer device index outside [0, peer ndevs) is rejected before it indexes remDevs.
TEST_F(NetIbMsegConnectMicrotest, SenderQpsToRts_PeerDeviceIndexOutOfRangeIsRejected) {
  ConfigureSingleDevice();
  InstallSingleDeviceResourceFakes();
  IbMsegSendComm comm{};
  comm.base.vProps.ndevs = 1;
  comm.base.nqps = 1;
  comm.devs[0].base.ibDevN = 0;
  ScopedHook modify(g_ibvModifyQp, [](ibv_qp*, ibv_qp_attr*, int) {
    ADD_FAILURE() << "a rejected peer must not transition any QP";
    return ncclSystemError;
  });

  for (int devIndex : {-1, 1, NCCL_IB_MAX_DEVS_PER_NIC}) {
    IbMsegConnectionMetadata remote{};
    remote.ndevs = 1;
    remote.qpInfo[0].devIndex = devIndex;
    EXPECT_EQ(IbMsegSenderQpsToRts(&comm, &remote), ncclInternalError) << "devIndex=" << devIndex;
  }
  EXPECT_EQ(modify.calls, 0);
}

// Branch: the receiver rejects a peer device index outside [0, peer ndevs) before creating its QP.
TEST_F(NetIbMsegConnectMicrotest, ReceiverQpsCreateToRts_PeerDeviceIndexOutOfRangeCreatesNoQp) {
  ConfigureSingleDevice();
  InstallSingleDeviceResourceFakes();
  IbMsegRecvComm comm{};
  comm.base.vProps.ndevs = 1;
  comm.base.nqps = 1;
  comm.devs[0].base.ibDevN = 0;
  comm.devs[0].base.pd = &pd_;
  comm.devs[0].base.cq = &cq_;

  for (int devIndex : {-1, 1, NCCL_IB_MAX_DEVS_PER_NIC}) {
    IbMsegConnectionMetadata remote{};
    IbMsegConnectionMetadata meta{};
    remote.ndevs = 1;
    remote.qpInfo[0].devIndex = devIndex;
    EXPECT_EQ(IbMsegReceiverQpsCreateToRts(&comm, &remote, &meta), ncclInternalError) << "devIndex=" << devIndex;
  }
  EXPECT_TRUE(qps_.empty());
}

// Discovery keeps the device's max_mr; multi-segment registration admits against it.
TEST_F(NetIbMsegConnectMicrotest, InitDevices_RecordsTheDeviceMaxMr) {
  IbVerbsWorld world;
  FakeHca& hca = world.AddHca("mlx5_0");
  hca.ports[0] = IbVerbsWorld::ActivePort(IBV_LINK_LAYER_INFINIBAND);
  hca.attr.max_mr = 4321;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_GID_INDEX") == 0 ? int64_t{0} : fallback;
  });
  ScopedHook gid(g_ibvQueryGid, [](ibv_context*, uint8_t, int, ibv_gid* out) {
    std::memset(out, 0, sizeof(*out));
    out->raw[15] = 0xaa;
    return ncclSuccess;
  });

  ASSERT_EQ(IbMsegInitDevices(nullptr, nullptr), ncclSuccess);

  ASSERT_EQ(IbMsegNDevs, 1);
  EXPECT_STREQ(IbMsegDevs[0].devName, "mlx5_0");
  EXPECT_EQ(IbMsegDevs[0].maxMr, 4321);
  EXPECT_EQ(IbMsegDevs[0].maxQp, 1024);
}

bool ReadWholeFile(const std::string& path, std::string* out) {
  std::ifstream in(path);
  if (!in) return false;
  std::ostringstream text;
  text << in.rdbuf();
  *out = text.str();
  return true;
}

std::string WithoutWhitespace(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (!std::isspace(static_cast<unsigned char>(c))) out += c;
  }
  return out;
}

// Parenthesis depth change over one line, ignoring parentheses inside string literals.
int ParenDelta(const std::string& line) {
  int delta = 0;
  bool inString = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (inString) {
      if (c == '\\') ++i;
      else if (c == '"') inString = false;
    } else if (c == '"') {
      inString = true;
    } else {
      delta += (c == '(') - (c == ')');
    }
  }
  return delta;
}

// The hand-copied net_ib parameters in fakes/net_ib_mseg_params*.cc must not drift: every statement, whitespace
// aside, still appears in the net_ib file its "// src/transport/net_ib/<file>" header names.
TEST(NetIbMsegParamCopies, EveryCopiedStatementMatchesItsNetIbSource) {
  static const char* const kCopies[] = {"net_ib_mseg_params.cc", "net_ib_mseg_params_connect.cc",
                                        "net_ib_mseg_params_p2p.cc"};
  const std::string kHeader = "// src/transport/net_ib/";
  int checked = 0;
  for (const char* copy : kCopies) {
    SCOPED_TRACE(copy);
    std::string text;
    ASSERT_TRUE(ReadWholeFile(std::string(NET_IB_MSEG_PARAM_COPIES_DIR) + "/" + copy, &text));
    std::istringstream lines(text);
    std::string line, source, stmt;
    int depth = 0;
    while (std::getline(lines, line)) {
      if (stmt.empty()) {
        if (line.rfind(kHeader, 0) == 0) {
          const std::string file = line.substr(kHeader.size());
          ASSERT_TRUE(ReadWholeFile(std::string(NET_IB_SOURCE_DIR) + "/" + file, &source)) << file;
          source = WithoutWhitespace(source);
          continue;
        }
        if (line.find_first_not_of(" \t") == std::string::npos || line.rfind("//", 0) == 0 ||
            line.rfind("/*", 0) == 0 || line.rfind(" *", 0) == 0 || line.rfind("#include", 0) == 0) {
          continue;
        }
      }
      stmt += line + "\n";
      depth += ParenDelta(line);
      if (depth > 0) continue;
      EXPECT_FALSE(source.empty()) << "statement before any net_ib header:\n" << stmt;
      EXPECT_NE(source.find(WithoutWhitespace(stmt)), std::string::npos) << "drifted from net_ib:\n" << stmt;
      ++checked;
      stmt.clear();
      depth = 0;
    }
    EXPECT_TRUE(stmt.empty()) << "unterminated statement:\n" << stmt;
  }
  EXPECT_GT(checked, 0);
}

}  // namespace
