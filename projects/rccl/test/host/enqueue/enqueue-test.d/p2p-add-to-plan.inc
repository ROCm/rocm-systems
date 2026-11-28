/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// addP2pToPlan: per-arch p2p protocol selection, CE routing and LL thresholds.
// Included into enqueue-test.cc's translation unit; see README.md.

// ===========================================================================
// addP2pToPlan (enqueue.cc:1334) -- gfx1250 SendRecv protocol selection.
// GPU SendRecvTests skip off gfx1250, so host CI never reached this planner.
// These tests fake cudaArch=1250 / nRanks=4 and inspect the emitted
// ncclDevWorkP2p plus the kernel funcId (LL vs LL128).
// ===========================================================================

#if !defined(ENABLE_LL128)
#error "rccl-UnitTestsMicroEnqueue must compile ENABLE_LL128 so addP2pToPlan's gfx1250 window is in this binary"
#endif

namespace {

struct Gfx1250AddP2pScene {
  BatchPlanComm bp{/*nNodes=*/1, /*cudaArch=*/1250};
  std::unique_ptr<ncclTopoSystem> topo{new ncclTopoSystem{}};
  std::vector<ncclChannelPeer> peers;
  std::vector<ncclChannelPeer*> peerSlots;
  std::array<ncclPeerInfo, 4> peerInfo{};
  char llBuf[8]{};
  char ll128Buf[8]{};
  char nanBuf[8]{};
  char sendMem[8]{};
  char recvMem[8]{};
  ncclTaskP2p recvTask{};
  ncclTaskP2p sendTask{};

  Gfx1250AddP2pScene(int nChannels = 4, int channelsPerPeer = 1) {
    InstallSendRecvDevFuncIds();
    ncclComm* c = bp.c();
    c->rank = 0;
    c->nRanks = 4;
    c->nNodes = 1;
    c->cudaArch = 1250;
    c->isAllDirectP2p = true;
    c->isAllCudaP2p = true;
    c->peerInfo = peerInfo.data();
    for (auto& peer : peerInfo) {
      peer.hostHash = 1;
      peer.hasFineGrain = true;
    }
    c->p2pnChannels = nChannels;
    c->p2pnChannelsPerPeer = channelsPerPeer;
    c->p2pChannelShiftSize = 0;
    c->p2pChunkSize = 128 << 10;
    c->ll128LineElems = 16;
    c->ll128DataElems = 15;
    c->buffSizes[NCCL_PROTO_LL] = 1 << 20;
    c->buffSizes[NCCL_PROTO_LL128] = 1 << 20;
    c->buffSizes[NCCL_PROTO_SIMPLE] = 1 << 20;
    c->p2pLl128BuffSize = c->buffSizes[NCCL_PROTO_LL128];
    SetSingleGpuArch(topo.get(), "gfx1250");
    topo->ll128Enabled = true;
    c->topo = topo.get();

    peers.resize(nChannels * c->nRanks);
    peerSlots.resize(nChannels * c->nRanks);
    for (int ch = 0; ch < nChannels; ++ch) {
      for (int rank = 0; rank < c->nRanks; ++rank) {
        int slot = ch * c->nRanks + rank;
        peerSlots[slot] = &peers[slot];
        for (int ci = 0; ci < NCCL_MAX_CONNS; ++ci) {
          peers[slot].send[ci].conn.buffs[NCCL_PROTO_LL] = llBuf;
          peers[slot].send[ci].conn.buffs[NCCL_PROTO_LL128] = ll128Buf;
          peers[slot].recv[ci].conn.buffs[NCCL_PROTO_LL] = llBuf;
          peers[slot].recv[ci].conn.buffs[NCCL_PROTO_LL128] = ll128Buf;
          peers[slot].send[ci].conn.buffs[NCCL_PROTO_NAN] = nanBuf;
          peers[slot].recv[ci].conn.buffs[NCCL_PROTO_NAN] = nanBuf;
        }
      }
      c->channels[ch].peers = &peerSlots[ch * c->nRanks];
    }

    recvTask.collAPI = ncclFuncRecv;
    recvTask.func = ncclFuncRecv;
    recvTask.root = 1;
    sendTask.collAPI = ncclFuncSend;
    sendTask.func = ncclFuncSend;
    sendTask.root = 1;
  }

  ncclResult_t add(ssize_t sendBytes, ssize_t recvBytes, int peer = 1) {
    struct ncclTaskP2p* tasks[2] = {&recvTask, &sendTask};
    int channelsPerPeer = bp.c()->p2pnChannelsPerPeer;
    return addP2pToPlan(bp.c(), bp.p(), /*nChannelsMin=*/channelsPerPeer, /*nChannelsMax=*/channelsPerPeer,
                        /*p2pRound=*/0,
                        /*sendRank=*/peer, sendMem, sendBytes, /*recvRank=*/peer, recvMem, recvBytes,
                        /*sendOpCount=*/0, /*recvOpCount=*/0, tasks);
  }

  ncclConnector* Conn(int channel, bool send, int connIndex = 1) {
    int slot = channel * bp.c()->nRanks + 1;
    return send ? &peers[slot].send[connIndex] : &peers[slot].recv[connIndex];
  }

  // Both halves of `collAPI` on `datatype`, with a NaN FIFO on every link.
  void asCollective(ncclFunc_t collAPI, ncclDataType_t datatype, size_t bytes) {
    bp.c()->buffSizes[NCCL_PROTO_NAN] = 1 << 20;
    for (ncclTaskP2p* t : {&recvTask, &sendTask}) {
      t->collAPI = collAPI;
      t->datatype = datatype;
      t->bytes = bytes;
    }
  }

  int p2pWorkCount() {
    int n = 0;
    for (auto* node = bp.p()->workQueue.head; node != nullptr; node = node->next) {
      if (node->workType == ncclDevWorkTypeP2p) ++n;
    }
    return n;
  }

  ncclDevWorkP2p* p2pWork() {
    for (auto* node = bp.p()->workQueue.head; node != nullptr; node = node->next) {
      if (node->workType == ncclDevWorkTypeP2p) return reinterpret_cast<ncclDevWorkP2p*>(node + 1);
    }
    return nullptr;
  }
};

struct ScopedP2pCeEnv {
  const char* saved = std::getenv("NCCL_P2P_USE_CUDA_MEMCPY");
  std::string old = saved == nullptr ? "" : saved;
  bool hadOld = saved != nullptr;

  ScopedP2pCeEnv() { setenv("NCCL_P2P_USE_CUDA_MEMCPY", "1", 1); }
  ~ScopedP2pCeEnv() {
    if (hadOld) setenv("NCCL_P2P_USE_CUDA_MEMCPY", old.c_str(), 1);
    else unsetenv("NCCL_P2P_USE_CUDA_MEMCPY");
  }
};

// Single-node all-XGMI gfx942/gfx950 legacy-LL gate. cudaArch is 100*major+10*minor,
// so gfx942 is 940; BatchPlanComm's default 942 does not enter the branch.
// p2pChunkSize is the single-node default (512 KiB), so the SIMPLE estimate is 1
// channel at or below 64 KiB/peer. nChannelsMax is 1 on purpose: the bug is the
// protocol choice, not how many channels the LL recompute would then stripe.
struct Gfx94P2pLlOpts {
  int cudaArch = 940;
  int nNodes = 1;
  int isAllNvlink = 1;
  bool ll128OptIn = false;
  int p2pChunkSize = 1 << 19;
};

struct Gfx94P2pLlScene {
  BatchPlanComm bp;
  std::unique_ptr<ncclTopoSystem> topo{new ncclTopoSystem{}};
  std::vector<ncclChannelPeer> peers;
  std::vector<ncclChannelPeer*> peerSlots;
  char llBuf[8]{};
  char ll128Buf[8]{};
  char sendMem[8]{};
  char recvMem[8]{};

  explicit Gfx94P2pLlScene(Gfx94P2pLlOpts opts) : bp(opts.nNodes, opts.cudaArch) {
    InstallSendRecvDevFuncIds();
    ncclComm* c = bp.c();
    c->rank = 0;
    c->nRanks = 2;
    c->nNodes = opts.nNodes;
    c->maxLocalRanks = 2;  // nNodes > 1 divides the P2P channel base by this
    c->cudaArch = opts.cudaArch;
    c->isAllNvlink = opts.isAllNvlink;
    c->allocP2pNetLLBuffers = opts.ll128OptIn;
    c->ll128LineElems = 8;
    c->ll128DataElems = 7;
    c->p2pnChannels = 1;
    c->p2pnChannelsPerPeer = 1;
    c->p2pChannelShiftSize = 0;
    c->p2pChunkSize = opts.p2pChunkSize;
    c->buffSizes[NCCL_PROTO_LL] = 1 << 20;
    c->buffSizes[NCCL_PROTO_LL128] = 1 << 20;
    c->buffSizes[NCCL_PROTO_SIMPLE] = 1 << 20;
    SetSingleGpuArch(topo.get(), opts.cudaArch == 950 ? "gfx950" : "gfx942");
    topo->ll128Enabled = opts.ll128OptIn;
    c->topo = topo.get();

    peers.resize(2);
    peerSlots.resize(2);
    for (int i = 0; i < 2; ++i) {
      peerSlots[i] = &peers[i];
      for (int ci = 0; ci < NCCL_MAX_CONNS; ++ci) {
        peers[i].send[ci].conn.buffs[NCCL_PROTO_LL] = peers[i].recv[ci].conn.buffs[NCCL_PROTO_LL] = llBuf;
        peers[i].send[ci].conn.buffs[NCCL_PROTO_LL128] = peers[i].recv[ci].conn.buffs[NCCL_PROTO_LL128] = ll128Buf;
      }
    }
    c->channels[0].peers = peerSlots.data();
  }

  ncclResult_t add(ssize_t bytes) {
    struct ncclTaskP2p* tasks[2] = {nullptr, nullptr};
    return addP2pToPlan(bp.c(), bp.p(), /*nChannelsMin=*/1, /*nChannelsMax=*/1, /*p2pRound=*/0,
                        /*sendRank=*/1, sendMem, bytes, /*recvRank=*/1, recvMem, bytes,
                        /*sendOpCount=*/0, /*recvOpCount=*/0, tasks);
  }

  ncclDevWorkP2p* p2pWork() {
    for (auto* node = bp.p()->workQueue.head; node != nullptr; node = node->next) {
      if (node->workType == ncclDevWorkTypeP2p) return reinterpret_cast<ncclDevWorkP2p*>(node + 1);
    }
    return nullptr;
  }
};
}  // namespace

TEST_F(EnqueueMicrotest, AddP2pToPlan_Gfx1250Enable1Mixed2KiB8KiB_OneLl128WorkItem) {
  // Argus hang: 2 KiB + 8 KiB used to be LL+LL128 and split into recv-then-send.
  // ENABLE=1 windows start at 0, so both dirs stay LL128 in one work item.
  SetParam("P2P_LL128_ENABLE", 1);
  Gfx1250AddP2pScene sc;
  ASSERT_EQ(ncclSuccess, sc.add(/*sendBytes=*/8192, /*recvBytes=*/2048));
  EXPECT_EQ(1, sc.p2pWorkCount()) << "mixed round must not split into two work items";
  auto* work = sc.p2pWork();
  ASSERT_NE(nullptr, work);
  EXPECT_EQ(1, work->sendProtoLL);
  EXPECT_EQ(1, work->recvProtoLL);
  auto* batch = sc.bp.tailBatch();
  ASSERT_NE(nullptr, batch);
  EXPECT_EQ(ncclDevFuncId_P2p(true), batch->funcId) << "ENABLE=1 in-window uses the LL128 kernel";
}

TEST_F(EnqueueMicrotest, AddP2pToPlan_ScaledCollectiveLl128Buffer_P2pStepKeepsTheP2pSize) {
  // With the 4x collective LL128 buffer, send/recv LL128 steps must still come from p2pLl128BuffSize: the net
  // transport allocates send/recv LL128 buffers at that size.
  SetParam("P2P_LL128_ENABLE", 1);
  Gfx1250AddP2pScene sc;
  ncclComm* c = sc.bp.c();
  c->p2pLl128BuffSize = 1 << 20;
  c->buffSizes[NCCL_PROTO_LL128] = 4 << 20;
  ASSERT_EQ(ncclSuccess, sc.add(/*sendBytes=*/8192, /*recvBytes=*/8192));
  auto* work = sc.p2pWork();
  ASSERT_NE(nullptr, work);
  ASSERT_EQ(1, work->sendProtoLL);
  ASSERT_EQ(1, work->recvProtoLL);
  auto dataChunk = [&](int buffSize) {
    return u32fp8Decode(u32fp8Encode((uint32_t)(buffSize / NCCL_STEPS / c->ll128LineElems * c->ll128DataElems)));
  };
  ASSERT_NE(dataChunk(1 << 20), dataChunk(4 << 20));
  EXPECT_EQ(dataChunk(1 << 20), u32fp8Decode(work->sendChunkSize_u32fp8));
  EXPECT_EQ(dataChunk(1 << 20), u32fp8Decode(work->recvChunkSize_u32fp8));
}

TEST_F(EnqueueMicrotest, AddP2pToPlan_Gfx1250Default2KiB_UsesLegacyLlKernel) {
  // Default ENABLE=-1: below P2P_LL_THRESHOLD stays legacy LL, not LL128.
  SetParam("P2P_LL128_ENABLE", -1);
  Gfx1250AddP2pScene sc;
  ASSERT_EQ(ncclSuccess, sc.add(/*sendBytes=*/2048, /*recvBytes=*/2048));
  EXPECT_EQ(1, sc.p2pWorkCount());
  auto* work = sc.p2pWork();
  ASSERT_NE(nullptr, work);
  EXPECT_EQ(1, work->sendProtoLL);
  EXPECT_EQ(1, work->recvProtoLL);
  auto* batch = sc.bp.tailBatch();
  ASSERT_NE(nullptr, batch);
  EXPECT_EQ(ncclDevFuncId_P2p(false), batch->funcId) << "default must not select the LL128 kernel";
}

TEST_F(EnqueueMicrotest, AddP2pToPlan_Gfx1250Enable1AboveCap_UsesSimple) {
  // 4-rank cap is 1 MiB. Above it ENABLE=1 is SIMPLE (protoLL=0) and the LL kernel.
  SetParam("P2P_LL128_ENABLE", 1);
  Gfx1250AddP2pScene sc;
  ASSERT_EQ(ncclSuccess, sc.add(/*sendBytes=*/(1 << 20) + 1, /*recvBytes=*/(1 << 20) + 1));
  EXPECT_EQ(1, sc.p2pWorkCount());
  auto* work = sc.p2pWork();
  ASSERT_NE(nullptr, work);
  EXPECT_EQ(0, work->sendProtoLL);
  EXPECT_EQ(0, work->recvProtoLL);
  auto* batch = sc.bp.tailBatch();
  ASSERT_NE(nullptr, batch);
  EXPECT_EQ(ncclDevFuncId_P2p(false), batch->funcId);
}

TEST_F(EnqueueMicrotest, AddP2pToPlan_NanAlltoAll_UsesNanKernelAndMarksF64) {
  for (ncclDataType_t dt : {ncclFloat32, ncclFloat64}) {
    UnderNanProto([dt] {
      Gfx1250AddP2pScene sc;
      sc.asCollective(ncclFuncAlltoAll, dt, 2048);
      ASSERT_EQ(ncclSuccess, sc.add(/*sendBytes=*/2048, /*recvBytes=*/2048));
      auto* work = sc.p2pWork();
      ASSERT_NE(nullptr, work);
      EXPECT_EQ(1, work->sendNan);
      EXPECT_EQ(1, work->recvNan);
      EXPECT_EQ(dt == ncclFloat64, work->sendNanF64);
      EXPECT_EQ(dt == ncclFloat64, work->recvNanF64);
      auto* batch = sc.bp.tailBatch();
      ASSERT_NE(nullptr, batch);
      EXPECT_EQ(ncclDevFuncId_P2p(false, true), batch->funcId);
    });
  }
}

// Network links do not carry the NaN-flag protocol.
TEST_F(EnqueueMicrotest, AddP2pToPlan_NanAlltoAllNetworkPeer_StaysOffNan) {
  UnderNanProto([] {
    Gfx1250AddP2pScene sc;
    sc.asCollective(ncclFuncAlltoAll, ncclFloat32, 2048);
    for (int ci = 0; ci < NCCL_MAX_CONNS; ++ci) {
      sc.peers[1].send[ci].transportComm = &netTransport.send;
      sc.peers[1].recv[ci].transportComm = &netTransport.recv;
    }
    ASSERT_EQ(ncclSuccess, sc.add(/*sendBytes=*/2048, /*recvBytes=*/2048));
    auto* work = sc.p2pWork();
    ASSERT_NE(nullptr, work);
    EXPECT_EQ(0, work->sendNan);
    EXPECT_EQ(0, work->recvNan);
    auto* batch = sc.bp.tailBatch();
    ASSERT_NE(nullptr, batch);
    EXPECT_EQ(ncclDevFuncId_P2p(false), batch->funcId);
  });
}

// A Direct AllGather's self copy escapes like the wire (sendrecv.h runCopyNaN), so the owner's slice matches its
// peers'. AlltoAll's does not: each block of its output has one source.
TEST_F(EnqueueMicrotest, AddP2pToPlan_NanSelfCopy_OnlyAllGatherEscapes) {
  for (ncclFunc_t collAPI : {ncclFuncAllGather, ncclFuncAlltoAll}) {
    UnderNanProto([collAPI] {
      Gfx1250AddP2pScene sc;
      sc.asCollective(collAPI, ncclFloat32, 2048);
      ASSERT_EQ(ncclSuccess, sc.add(/*sendBytes=*/2048, /*recvBytes=*/2048, /*peer=*/0));
      auto* work = sc.p2pWork();
      ASSERT_NE(nullptr, work);
      EXPECT_EQ(collAPI == ncclFuncAllGather, work->sendNan);
    });
  }
}

// Across nodes a Direct AllGather slice reaches some ranks over the network, which does not escape it, so none of
// its links, nor its self copy, take NaN.
TEST_F(EnqueueMicrotest, AddP2pToPlan_NanAllGatherMultiNode_StaysOffNan) {
  for (int peer : {1, 0}) {
    UnderNanProto([peer] {
      ScopedHook pxnDisable(g_pxnDisable, [](struct ncclComm*) { return 1; });
      Gfx1250AddP2pScene sc;
      sc.bp.c()->nNodes = 2;
      sc.bp.c()->localRanks = sc.bp.c()->maxLocalRanks = 2;
      sc.asCollective(ncclFuncAllGather, ncclFloat32, 2048);
      ASSERT_EQ(ncclSuccess, sc.add(/*sendBytes=*/2048, /*recvBytes=*/2048, peer));
      auto* work = sc.p2pWork();
      ASSERT_NE(nullptr, work);
      EXPECT_EQ(0, work->sendNan);
      EXPECT_EQ(0, work->recvNan);
    });
  }
}

TEST_F(EnqueueMicrotest, AddP2pToPlan_DirectCeUsesExplicitConnectorAndMinimumStepAcrossParts) {
  ScopedP2pCeEnv ce;
  SetParam("P2P_LL128_ENABLE", 0);
  Gfx1250AddP2pScene sc(/*nChannels=*/4, /*channelsPerPeer=*/2);
  constexpr int kCeStep = 2 << 20;
  constexpr ssize_t kBytes = ssize_t(64) << 20;
  for (int ch = 0; ch < 2; ++ch) {
    auto* send = sc.Conn(ch, /*send=*/true, NCCL_CONN_IDX_P2P_CE);
    auto* recv = sc.Conn(ch, /*send=*/false, NCCL_CONN_IDX_P2P_CE);
    send->useCe = recv->useCe = true;
    send->conn.stepSize = recv->conn.stepSize = ch == 0 ? 4 << 20 : kCeStep;
    send->proxyConn.proxyProgress =
      reinterpret_cast<decltype(send->proxyConn.proxyProgress)>(0x1);
  }
  g_proxySaveOpJustInquire = true;

  ASSERT_EQ(ncclSuccess, sc.add(kBytes, kBytes));
  auto* work = sc.p2pWork();
  ASSERT_NE(nullptr, work);
  ASSERT_EQ(2, work->nSendChannels);
  ASSERT_EQ(2, work->nRecvChannels);
  EXPECT_EQ(kCeStep, u32fp8Decode(work->sendChunkSize_u32fp8));
  EXPECT_EQ(kCeStep, u32fp8Decode(work->recvChunkSize_u32fp8));

  for (int ch = 0; ch < 2; ++ch) {
    int sendOps = 0, recvOps = 0;
    for (ncclProxyOp* op = ChannelProxyHead(sc.bp, ch); op != nullptr; op = op->enqNext) {
      if (op->pattern == ncclPatternSend) {
        sendOps++;
        EXPECT_EQ(kCeStep, op->chunkSize);
        EXPECT_EQ(16, op->nsteps);
      } else if (op->pattern == ncclPatternRecv) {
        recvOps++;
        EXPECT_EQ(kCeStep, op->chunkSize);
        EXPECT_EQ(16, op->nsteps);
      }
    }
    EXPECT_EQ(1, sendOps) << ch;
    EXPECT_EQ(1, recvOps) << ch;
  }
}

TEST_F(EnqueueMicrotest, AddP2pToPlan_FeatureOnLoweredCollectiveStaysOnKernelConnector) {
  ScopedP2pCeEnv ce;
  SetParam("P2P_LL128_ENABLE", 0);
  Gfx1250AddP2pScene sc;
  sc.sendTask.collAPI = ncclFuncAlltoAll;
  sc.recvTask.collAPI = ncclFuncAlltoAll;
  constexpr int kKernelStep = 128 << 10;
  sc.Conn(0, /*send=*/true)->conn.stepSize = kKernelStep;
  sc.Conn(0, /*send=*/false)->conn.stepSize = kKernelStep;
  sc.Conn(0, /*send=*/true, NCCL_CONN_IDX_P2P_CE)->conn.stepSize = 4 << 20;
  sc.Conn(0, /*send=*/false, NCCL_CONN_IDX_P2P_CE)->conn.stepSize = 4 << 20;

  ASSERT_EQ(NCCL_CONN_IDX_P2P_CE, rcclP2pTaskConnIndex(sc.bp.c(), ncclFuncSend, /*peer=*/1));
  ASSERT_EQ(1, rcclP2pTaskConnIndex(sc.bp.c(), ncclFuncAlltoAll, /*peer=*/1));
  ASSERT_EQ(ncclSuccess, sc.add(/*sendBytes=*/64 << 20, /*recvBytes=*/64 << 20));
  auto* work = sc.p2pWork();
  ASSERT_NE(nullptr, work);
  EXPECT_EQ(kKernelStep, u32fp8Decode(work->sendChunkSize_u32fp8));
  EXPECT_EQ(kKernelStep, u32fp8Decode(work->recvChunkSize_u32fp8));
}

TEST_F(EnqueueMicrotest, AddP2pToPlan_MixedDirectSendCollectiveRecvUsesDirectionLocalConnectors) {
  ScopedP2pCeEnv ce;
  SetParam("P2P_LL128_ENABLE", 0);
  Gfx1250AddP2pScene sc;
  sc.recvTask.collAPI = ncclFuncAlltoAll;
  constexpr int kCeStep = 4 << 20;
  constexpr int kKernelStep = 128 << 10;
  auto* send = sc.Conn(0, /*send=*/true, NCCL_CONN_IDX_P2P_CE);
  send->useCe = true;
  send->conn.stepSize = kCeStep;
  sc.Conn(0, /*send=*/false)->conn.stepSize = kKernelStep;

  ASSERT_EQ(ncclSuccess, sc.add(/*sendBytes=*/64 << 20, /*recvBytes=*/64 << 20));
  auto* work = sc.p2pWork();
  ASSERT_NE(nullptr, work);
  EXPECT_EQ(NCCL_CONN_IDX_P2P_CE, work->sendConnIndex);
  EXPECT_EQ(1, work->recvConnIndex);
  EXPECT_EQ(kCeStep, u32fp8Decode(work->sendChunkSize_u32fp8));
  EXPECT_EQ(kKernelStep, u32fp8Decode(work->recvChunkSize_u32fp8));
}

TEST_F(EnqueueMicrotest, AddP2pToPlan_CopyEngineClampsGlobalChunkToConnectorStep) {
  ScopedP2pCeEnv ce;
  SetParam("P2P_LL128_ENABLE", 0);
  SetParam("CHUNK_SIZE", 32 << 20);
  Gfx1250AddP2pScene sc;
  constexpr int kCeStep = 4 << 20;
  for (bool send : {false, true}) {
    auto* conn = sc.Conn(0, send, NCCL_CONN_IDX_P2P_CE);
    conn->useCe = true;
    conn->conn.stepSize = kCeStep;
  }

  ASSERT_EQ(ncclSuccess, sc.add(/*sendBytes=*/64 << 20, /*recvBytes=*/64 << 20));
  auto* work = sc.p2pWork();
  ASSERT_NE(nullptr, work);
  EXPECT_EQ(kCeStep, u32fp8Decode(work->sendChunkSize_u32fp8));
  EXPECT_EQ(kCeStep, u32fp8Decode(work->recvChunkSize_u32fp8));
}

TEST_F(EnqueueMicrotest, P2pTaskConnIndex_CopyEngineRequiresLocalDirectPeer) {
  ScopedP2pCeEnv ce;
  Gfx1250AddP2pScene sc;
  ncclComm* comm = sc.bp.c();
  EXPECT_EQ(NCCL_CONN_IDX_P2P_CE, rcclP2pTaskConnIndex(comm, ncclFuncSend, /*peer=*/1));

  comm->isAllDirectP2p = false;
  EXPECT_EQ(1, rcclP2pTaskConnIndex(comm, ncclFuncSend, /*peer=*/1));

  comm->isAllDirectP2p = true;
  comm->peerInfo[1].hostHash = 2;
  EXPECT_EQ(1, rcclP2pTaskConnIndex(comm, ncclFuncSend, /*peer=*/1));
  comm->peerInfo[1].hostHash = 1;
  comm->peerInfo[1].hasFineGrain = false;
  EXPECT_EQ(1, rcclP2pTaskConnIndex(comm, ncclFuncSend, /*peer=*/1));
  EXPECT_EQ(1, rcclP2pTaskConnIndex(comm, ncclFuncSend, /*peer=*/comm->rank));
}

TEST_F(EnqueueMicrotest, AddP2pToPlan_FeatureOnP2pNetKeepsTheFullPeerTile) {
  ScopedP2pCeEnv ce;
  SetParam("P2P_LL128_ENABLE", 0);
  Gfx1250AddP2pScene sc(/*nChannels=*/32, /*channelsPerPeer=*/32);
  sc.bp.c()->p2pNet = 1;

  ASSERT_EQ(ncclSuccess, sc.add(/*sendBytes=*/64 << 20, /*recvBytes=*/64 << 20));
  auto* work = sc.p2pWork();
  ASSERT_NE(nullptr, work);
  EXPECT_EQ(32, work->nSendChannels);
  EXPECT_EQ(32, work->nRecvChannels);
}

TEST_F(EnqueueMicrotest, P2pReservationChannels_UsesCeCapButKeepsP2pNetTile) {
  ScopedP2pCeEnv ce;
  Gfx1250AddP2pScene sc(/*nChannels=*/64, /*channelsPerPeer=*/64);
  sc.sendTask.bytes = sc.recvTask.bytes = 64 << 20;
  EXPECT_EQ(16, rcclP2pReservationChannels(sc.bp.c(), &sc.recvTask, &sc.sendTask));

  sc.bp.c()->p2pNet = 1;
  EXPECT_EQ(64, rcclP2pReservationChannels(sc.bp.c(), &sc.recvTask, &sc.sendTask));
}

TEST_F(EnqueueMicrotest, HostStreamPlanCallback_AbortOrRevokeReturns_DestroyStillRuns) {
  auto ok = [](struct ncclKernelPlan*) { return ncclSuccess; };
  ScopedHook startGroup(g_profilerStartGroupEvent, ok);
  ScopedHook stopGroup(g_profilerStopGroupEvent, ok);
  ScopedHook startTasks(g_profilerStartTaskEvents, ok);
  ScopedHook stopTasks(g_profilerStopTaskEvents, ok);
  ScopedHook postWork(g_profilerPostPlanWork, [](struct ncclComm*, struct ncclKernelPlan*) { return ncclSuccess; });
  ncclComm comm{};
  uint32_t abortFlag = 0;
  comm.abortFlag = &abortFlag;
  ncclKernelPlan plan{};
  plan.persistent = 1;
  plan.comm = &comm;

  abortFlag = 1;
  hostStreamPlanCallback(&plan);
  EXPECT_EQ(0, startGroup.calls);

  abortFlag = 0;
  comm.revokedFlag = 1;
  hostStreamPlanCallback(&plan);
  EXPECT_EQ(0, startGroup.calls);

  comm.revokedFlag = 0;
  comm.destroyFlag = 1;
  hostStreamPlanCallback(&plan);
  EXPECT_EQ(1, startGroup.calls);

  ncclKernelPlan eager{};
  eager.comm = &comm;
  abortFlag = 1;
  hostStreamPlanCallback(&eager);
  EXPECT_EQ(2, startGroup.calls);
}

TEST_F(EnqueueMicrotest, PersistentDestructor_LiveRef_ReclaimsAndOrphanDoesNotEnqueueAgain) {
  ncclComm comm{};
  ncclKernelPlan plan{};
  ncclPersistentPlanRef ref{};
  ref.comm = &comm;
  ref.planHead = &plan;
  ref.state = ncclPersistentPlanRefLive;
  ref.refCount = 2;
  comm.persistentPlanRefs = &ref;

  persistentDestructor(&ref);
  EXPECT_EQ(ncclPersistentPlanRefReleased, ref.state);
  EXPECT_EQ(1, ref.refCount);
  ncclCommCallback* cb = ncclIntruQueueMpscDequeueAll(&comm.callbackQueue, false);
  ASSERT_NE(nullptr, cb);
  EXPECT_EQ(&plan.reclaimer, cb);
  EXPECT_EQ(nullptr, cb->next);

  ncclCommOrphanPersistentPlans(&comm);
  EXPECT_EQ(nullptr, ncclIntruQueueMpscDequeueAll(&comm.callbackQueue, false));
}

TEST_F(EnqueueMicrotest, PersistentDestructor_OrphanedRef_StillReclaims) {
  ncclComm comm{};
  ncclKernelPlan plan{};
  ncclPersistentPlanRef ref{};
  ref.comm = &comm;
  ref.planHead = &plan;
  ref.state = ncclPersistentPlanRefLive;
  ref.refCount = 2;
  comm.persistentPlanRefs = &ref;

  ncclCommOrphanPersistentPlans(&comm);
  EXPECT_EQ(ncclPersistentPlanRefOrphaned, ref.state);
  EXPECT_EQ(nullptr, ncclIntruQueueMpscDequeueAll(&comm.callbackQueue, false));

  persistentDestructor(&ref);
  EXPECT_EQ(ncclPersistentPlanRefOrphaned, ref.state);
  EXPECT_EQ(1, ref.refCount);
  ncclCommCallback* cb = ncclIntruQueueMpscDequeueAll(&comm.callbackQueue, false);
  ASSERT_NE(nullptr, cb);
  EXPECT_EQ(&plan.reclaimer, cb);
}

TEST_F(EnqueueMicrotest, AddPersistentPlanRef_DropsNonLiveRefsAndKeepsLiveOnes) {
  ncclComm comm{};
  auto* stale = static_cast<ncclPersistentPlanRef*>(calloc(1, sizeof(ncclPersistentPlanRef)));
  auto* live = static_cast<ncclPersistentPlanRef*>(calloc(1, sizeof(ncclPersistentPlanRef)));
  ASSERT_NE(nullptr, stale);
  ASSERT_NE(nullptr, live);
  stale->state = ncclPersistentPlanRefReleased;
  stale->refCount = 2;
  live->state = ncclPersistentPlanRefLive;
  live->refCount = 2;
  live->next = stale;
  comm.persistentPlanRefs = live;

  ncclPersistentPlanRef* registered = nullptr;
  g_cudaGraphAddDestructor = [&](ncclCudaGraph, hipHostFn_t, void* arg) {
    registered = static_cast<ncclPersistentPlanRef*>(arg);
    return ncclSuccess;
  };

  ncclKernelPlan plan{};
  ASSERT_EQ(ncclSuccess, addPersistentPlanRef(&comm, ncclCudaGraph{}, &plan));
  ASSERT_EQ(registered, comm.persistentPlanRefs);
  EXPECT_EQ(live, registered->next);
  EXPECT_EQ(nullptr, live->next);
  EXPECT_EQ(1, stale->refCount);
  EXPECT_EQ(2, live->refCount);
  EXPECT_EQ(2, registered->refCount);
  EXPECT_EQ(ncclPersistentPlanRefLive, registered->state);

  ncclPersistentPlanRefDrop(stale);
  ncclPersistentPlanRefDrop(live);
  ncclPersistentPlanRefDrop(live);
  ncclPersistentPlanRefDrop(registered);
  ncclPersistentPlanRefDrop(registered);
}

// Pin the numeric cutoff through the redirected ncclParamP2pLLThreshold (g_loadParam,
// no cache). Whether the user set NCCL_P2P_LL_THRESHOLD is a separate ncclGetEnv
// read: the widen applies when that name is absent or empty, so an explicit 4096
// stays on the stock per-channel rule even though the number matches the default.
void PinP2pLlThreshold(bool userSet) {
  if (userSet) SetMicroEnv("NCCL_P2P_LL_THRESHOLD", "4096");
  else SetMicroEnvAbsent("NCCL_P2P_LL_THRESHOLD");
  SetParam("P2P_LL_THRESHOLD", 4096);
}

void ExpectGfx94RecvProtoLL(Gfx94P2pLlOpts opts, ssize_t bytes, int protoLL) {
  Gfx94P2pLlScene sc(opts);
  ASSERT_EQ(ncclSuccess, sc.add(bytes)) << "arch=" << opts.cudaArch << " bytes=" << bytes;
  auto* work = sc.p2pWork();
  ASSERT_NE(nullptr, work) << "arch=" << opts.cudaArch << " bytes=" << bytes;
  EXPECT_EQ(protoLL, int(work->recvProtoLL)) << "arch=" << opts.cudaArch << " bytes=" << bytes;
  EXPECT_EQ(protoLL, int(work->sendProtoLL)) << "arch=" << opts.cudaArch << " bytes=" << bytes;
}

void ExpectGfx94RecvProtoLL(int cudaArch, ssize_t bytes, int protoLL) {
  Gfx94P2pLlOpts opts;
  opts.cudaArch = cudaArch;
  ExpectGfx94RecvProtoLL(opts, bytes, protoLL);
}

TEST_F(EnqueueMicrotest, AddP2pToPlan_Gfx94UnsetLlThreshold_WidensLlThrough64KiB) {
  PinP2pLlThreshold(/*userSet=*/false);
  const struct {
    ssize_t bytes;
    int protoLL;
  } cases[] = {
      {16 << 10, 1}, {32 << 10, 1}, {64 << 10, 1}, {(64 << 10) + 1, 0},
      {128 << 10, 0}, {256 << 10, 0}, {512 << 10, 0},
  };
  for (const auto& c : cases) ExpectGfx94RecvProtoLL(940, c.bytes, c.protoLL);
  // gfx950 shares the gate. 942 is not 100*9+10*4, so the default BatchPlanComm
  // arch must stay SIMPLE at 16 KiB.
  ExpectGfx94RecvProtoLL(950, 16 << 10, 1);
  ExpectGfx94RecvProtoLL(942, 16 << 10, 0);
}

TEST_F(EnqueueMicrotest, AddP2pToPlan_Gfx94UserLlThreshold_KeepsPerChannelRule) {
  PinP2pLlThreshold(/*userSet=*/true);
  ExpectGfx94RecvProtoLL(940, 16 << 10, 0);
  ExpectGfx94RecvProtoLL(950, 16 << 10, 0);
}

// NCCL_P2P_LL_THRESHOLD= is present but empty. ncclLoadParam uses the default, so
// the widen still applies. A nullptr-only check would treat this as user-set.
TEST_F(EnqueueMicrotest, AddP2pToPlan_Gfx94EmptyLlThreshold_Widens) {
  SetMicroEnv("NCCL_P2P_LL_THRESHOLD", "");
  SetParam("P2P_LL_THRESHOLD", 4096);
  ExpectGfx94RecvProtoLL(940, 16 << 10, 1);
}

// 32 KiB is inside the widen (stock cutoff is 4 KiB). Each gate that is off must
// stay SIMPLE; dropping nNodes, isAllNvlink, or !useLL128OptIn would pick LL.
TEST_F(EnqueueMicrotest, AddP2pToPlan_Gfx94WidenGates_StaySimple) {
  PinP2pLlThreshold(/*userSet=*/false);
  // nNodes > 1 takes the channel base through rcclParamPxnOptQpUsage, which the
  // host fake aborts on. PXN disabled is the production default and skips that
  // call. The widen still requires a single node.
  {
    ScopedHook pxnDisable(g_pxnDisable, [](struct ncclComm*) { return 1; });
    Gfx94P2pLlOpts multiNode;
    multiNode.nNodes = 2;
    ExpectGfx94RecvProtoLL(multiNode, 32 << 10, 0);
  }
  Gfx94P2pLlOpts pcie;
  pcie.isAllNvlink = 0;
  ExpectGfx94RecvProtoLL(pcie, 32 << 10, 0);
  Gfx94P2pLlOpts ll128;
  ll128.ll128OptIn = true;
  ExpectGfx94RecvProtoLL(ll128, 32 << 10, 0);
}

// The default 512 KiB chunk makes p2pChunkSize/8 equal the 64 KiB cap, so either
// operand of the min would pass. A smaller chunk stops at chunk/8. A larger one
// stays at the cap.
TEST_F(EnqueueMicrotest, AddP2pToPlan_Gfx94ChunkSize_BoundsTheWiden) {
  PinP2pLlThreshold(/*userSet=*/false);
  Gfx94P2pLlOpts small;
  small.p2pChunkSize = 256 << 10;
  ExpectGfx94RecvProtoLL(small, 32 << 10, 1);
  ExpectGfx94RecvProtoLL(small, (32 << 10) + 1, 0);
  Gfx94P2pLlOpts large;
  large.p2pChunkSize = 1 << 20;
  ExpectGfx94RecvProtoLL(large, 64 << 10, 1);
  ExpectGfx94RecvProtoLL(large, (64 << 10) + 1, 0);
  ExpectGfx94RecvProtoLL(large, 128 << 10, 0);
}
