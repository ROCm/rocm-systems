/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the RMA_IB_PROXY backend in src/transport/net_ib/gin.cc: how it groups work requests
// (WRs) into doorbells and how requests whose own WRs raise no completion still complete.
//
// With ncclRmaOptFlagsAggregateRequests, IPut / IPutSignal / IGet append their WRs to a per-context chain instead of
// posting them. One ibv_post_send rings the chain when the next put+signal would not fit NCCL_RMA_IB_WR_BATCHSIZE
// (default 64, clamped to 2..64), when an op without the flag or for another QP arrives, and before Test, IFlush and
// DestroyContext. Only the last WR of a posted chain is signaled. A request is complete once a CQE for it or for any
// later request on the same connection has been polled, since a QP finishes its WRs in order.
//
// gin.cc reaches the NIC only through the verbs ops table, and its peers only through ncclIbConnectImpl /
// ncclIbAcceptImpl and the ncclNetIb table. The tests drive the real CreateContext / IPut / IPutSignal / IGet /
// IFlush / Test / DestroyContext against a fake NIC that records every doorbell and raises one CQE per signaled WR.

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <vector>

#include "fakes/nccl_fakes.h"

// gin.cc declares params of its own. Route them through g_loadParam so a test can set NCCL_RMA_IB_WR_BATCHSIZE.
// Must precede the unit under test.
#include "fakes/param_redirect.h"

#include NET_IB_GIN_CC_PATH

namespace {

constexpr int kNRanks = 2;
constexpr int kRank = 0;
constexpr int kSelf = kRank;
constexpr int kPeer = 1;
constexpr uint32_t kAggregate = ncclRmaOptFlagsAggregateRequests;
constexpr uint32_t kNoFlags = 0;

constexpr uintptr_t kSrcBase = 0x10000000;
constexpr uintptr_t kDstBase = 0x20000000;
constexpr uintptr_t kSignalBase = 0x30000000;
constexpr uint64_t kSignalOff = 0x40;
constexpr size_t kOpBytes = 64;
constexpr size_t kMaxChainLength = 1024;

// A WR as the NIC read it when the doorbell rang; the proxy reuses the memory behind it afterwards.
struct PostedWr {
  ibv_wr_opcode opcode;
  bool signaled;
  uint64_t remoteAddr;
};

struct Doorbell {
  ibv_qp* qp;
  std::vector<PostedWr> wrs;
};

// What the proxy holds for one rank: the send comm IPut / IPutSignal / IGet post on, and the recv comm whose flush
// QP IFlush posts on. Each completes on its own CQ.
struct FakePeer {
  std::unique_ptr<ncclIbSendComm> send = std::make_unique<ncclIbSendComm>();
  std::unique_ptr<ncclIbRecvComm> recv = std::make_unique<ncclIbRecvComm>();
  ibv_qp sendQp{};
  ibv_qp flushQp{};
  ibv_cq sendCq{};
  ibv_cq recvCq{};
  ibv_pd pd{};
  ibv_mr scratchpadMr{};
};

struct FakeIb {
  ibv_device device{};
  ibv_context context{};
  FakePeer peers[kNRanks];
  std::vector<Doorbell> doorbells;
  std::map<ibv_cq*, std::deque<ibv_wc>> unpolledCqes;
  int cqesRaised = 0;
  bool deliverCqes = true;
  ibv_wc_status cqeStatus = IBV_WC_SUCCESS;
  int acceptCalls = 0;
  std::vector<std::unique_ptr<ibv_mr>> mrs;
  std::vector<std::unique_ptr<ncclIbMrHandle>> mrHandles;
  int listenComm = 0;
  int netRequest = 0;
};
FakeIb* g_ib = nullptr;

int FakePostSend(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** badWr) {
  Doorbell doorbell{qp, {}};
  for (ibv_send_wr* w = wr; w != nullptr; w = w->next) {
    if (doorbell.wrs.size() == kMaxChainLength) {
      ADD_FAILURE() << "the posted WR chain does not end";
      *badWr = w;
      return EINVAL;
    }
    bool signaled = (w->send_flags & IBV_SEND_SIGNALED) != 0;
    bool atomic = w->opcode == IBV_WR_ATOMIC_FETCH_AND_ADD;
    doorbell.wrs.push_back({w->opcode, signaled, atomic ? w->wr.atomic.remote_addr : w->wr.rdma.remote_addr});
    if (signaled) {
      ibv_wc wc{};
      wc.wr_id = w->wr_id;
      wc.status = g_ib->cqeStatus;
      g_ib->unpolledCqes[qp->send_cq].push_back(wc);
      g_ib->cqesRaised++;
    }
  }
  g_ib->doorbells.push_back(std::move(doorbell));
  return 0;
}

int FakePollCq(ibv_cq* cq, int numEntries, ibv_wc* wc) {
  if (!g_ib->deliverCqes) return 0;
  std::deque<ibv_wc>& unpolled = g_ib->unpolledCqes[cq];
  int n = 0;
  for (; n < numEntries && !unpolled.empty(); n++) {
    wc[n] = unpolled.front();
    unpolled.pop_front();
  }
  return n;
}

// Rank r's listen handle names r. Every other exchange (MR base addresses and rkeys) is symmetric.
ncclResult_t FakeAllGather(ncclGinIbCollComm* cComm, void* src, void* dst, size_t len) {
  for (int r = 0; r < cComm->nranks; r++) {
    char* slot = static_cast<char*>(dst) + r * len;
    if (slot != src) std::memcpy(slot, src, len);
    if (len == NCCL_NET_HANDLE_MAXSIZE) std::memcpy(slot, &r, sizeof(r));
  }
  return ncclSuccess;
}

// The ncclNetIb entry points CreateContext and DestroyContext reach: listening for peers, the P2P barrier between
// connect steps (an allgather over the coll comm's own send/recv pair), and closing.
ncclResult_t FakeListen(void*, int, void*, void** listenComm) {
  *listenComm = &g_ib->listenComm;
  return ncclSuccess;
}
ncclResult_t FakeRegMr(void*, void*, size_t, int, void** mhandle) {
  *mhandle = &g_ib->netRequest;
  return ncclSuccess;
}
ncclResult_t FakeDeregMr(void*, void*) { return ncclSuccess; }
ncclResult_t FakeIsend(void*, void*, size_t, int, void*, void*, void** request) {
  *request = &g_ib->netRequest;
  return ncclSuccess;
}
ncclResult_t FakeIrecv(void*, int, void**, size_t*, int*, void**, void**, void** request) {
  *request = &g_ib->netRequest;
  return ncclSuccess;
}
ncclResult_t FakeNetTest(void*, int* done, int*) {
  *done = 1;
  return ncclSuccess;
}
ncclResult_t FakeClose(void*) { return ncclSuccess; }

}  // namespace

// --- Link-satisfying definitions for gin.cc's externals ------------------------------------------------------------
//
// The net_ib TUs that own these are not compiled into this binary. The connection, registration and request-pool
// entries are the ones the proxy data path runs through; the rest belong to init / connect paths no test takes.
ncclNet_t ncclNetIb;
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
int ncclNIbDevs = 0;
const char* ncclIbReqTypeStr[] = {"Unused", "Send", "Recv", "Flush", "IPut", "IGet", "Failed"};

ncclResult_t ncclIbGdrSupport() { return ncclSuccess; }
ncclResult_t ncclIbPeerMemSupport() { return ncclSuccess; }
ncclResult_t ncclIbDmaBufSupport(int) { return ncclSuccess; }
ncclResult_t ncclIbInitDevices(ncclDebugLogger_t, ncclProfilerCallback_t) { return ncclInternalError; }
ncclResult_t ncclIbFinalizeDevices() { return ncclSuccess; }
ncclResult_t ncclIbDevices(int* ndev) {
  *ndev = 0;
  return ncclSuccess;
}
ncclResult_t ncclIbGetProperties(int, ncclNetProperties_t*) { return ncclInternalError; }
ncclResult_t ncclIbGetGidIndex(struct ibv_context*, uint8_t, struct ibv_port_attr*, int*) { return ncclInternalError; }
ncclResult_t ncclIbListen(void*, int, void*, void**) { return ncclInternalError; }
ncclResult_t ncclIbCloseListen(void*) { return ncclSuccess; }
void ncclIbSetTrafficClass(void*, int) {}

ncclResult_t ncclIbConnectImpl(void*, int, void* opaqueHandle, void** sendComm, ncclNetDeviceHandle_t**, int, int) {
  int rank;
  std::memcpy(&rank, opaqueHandle, sizeof(rank));
  *sendComm = g_ib->peers[rank].send.get();
  return ncclSuccess;
}
ncclResult_t ncclIbAcceptImpl(void*, void** recvComm, ncclNetDeviceHandle_t**, int) {
  // CreateContext's connect walk accepts from rank, rank-1, rank-2, ... in turn.
  int rank = (kRank - g_ib->acceptCalls++ + kNRanks) % kNRanks;
  *recvComm = g_ib->peers[rank].recv.get();
  return ncclSuccess;
}
// The fixture wires every recv comm's flush QP up front.
ncclResult_t ncclIbCreateFlushQp(struct ncclIbRecvComm*) { return ncclSuccess; }

ncclResult_t ncclIbRegMrDmaBufInternal(void*, void* data, size_t size, int, uint64_t, int, uint64_t, void** mhandle) {
  auto mr = std::make_unique<ibv_mr>();
  mr->addr = data;
  mr->length = size;
  mr->lkey = 0x1000 + static_cast<uint32_t>(g_ib->mrs.size());
  mr->rkey = 0x2000 + static_cast<uint32_t>(g_ib->mrs.size());
  auto handle = std::make_unique<ncclIbMrHandle>();
  handle->mrs[0] = mr.get();
  *mhandle = handle.get();
  g_ib->mrs.push_back(std::move(mr));
  g_ib->mrHandles.push_back(std::move(handle));
  return ncclSuccess;
}

// Same behaviour as the request pool in p2p.cc: Test reads id, events[] and devBases[] from these slots.
ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req) {
  for (int i = 0; i < NET_IB_MAX_REQUESTS; i++) {
    struct ncclIbRequest* r = base->reqs + i;
    if (r->type == NCCL_NET_IB_REQ_UNUSED) {
      r->base = base;
      r->sock = NULL;
      memset(r->devBases, 0, sizeof(r->devBases));
      memset(r->events, 0, sizeof(r->events));
      *req = r;
      return ncclSuccess;
    }
  }
  *req = NULL;
  return ncclInternalError;
}
ncclResult_t ncclIbFreeRequest(struct ncclIbRequest* r) {
  r->type = NCCL_NET_IB_REQ_UNUSED;
  return ncclSuccess;
}
void ncclIbAddEvent(struct ncclIbRequest* req, int devIndex) {
  req->events[devIndex]++;
  req->devBases[devIndex] = req->base->isSend
                                ? &reinterpret_cast<struct ncclIbSendComm*>(req->base)->devs[devIndex].base
                                : &reinterpret_cast<struct ncclIbRecvComm*>(req->base)->devs[devIndex].base;
}

ncclResult_t ncclSocketGetAddr(struct ncclSocket*, union ncclSocketAddress* addr) {
  memset(addr, 0, sizeof(*addr));
  return ncclSuccess;
}
const char* ncclSocketToString(const union ncclSocketAddress*, char* buf, size_t bufLen, const int) {
  snprintf(buf, bufLen, "fake-peer");
  return buf;
}

namespace {

std::vector<ibv_wr_opcode> Opcodes(const Doorbell& doorbell) {
  std::vector<ibv_wr_opcode> opcodes;
  for (const PostedWr& wr : doorbell.wrs) opcodes.push_back(wr.opcode);
  return opcodes;
}

std::vector<uint64_t> RemoteAddrs(const Doorbell& doorbell) {
  std::vector<uint64_t> addrs;
  for (const PostedWr& wr : doorbell.wrs) addrs.push_back(wr.remoteAddr);
  return addrs;
}

std::vector<bool> Signaled(const Doorbell& doorbell) {
  std::vector<bool> signaled;
  for (const PostedWr& wr : doorbell.wrs) signaled.push_back(wr.signaled);
  return signaled;
}

class RmaIbProxyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetNcclFakes();
    g_ib = &ib_;

    ib_.context.device = &ib_.device;
    ib_.context.ops.post_send = FakePostSend;
    ib_.context.ops.poll_cq = FakePollCq;
    snprintf(ib_.device.name, sizeof(ib_.device.name), "fake_ib0");
    for (FakePeer& peer : ib_.peers) {
      peer.sendQp.context = peer.flushQp.context = &ib_.context;
      peer.sendCq.context = peer.recvCq.context = &ib_.context;
      peer.sendQp.send_cq = &peer.sendCq;
      peer.flushQp.send_cq = &peer.recvCq;
      peer.pd.context = &ib_.context;

      ncclIbSendComm* send = peer.send.get();
      send->base.isSend = true;
      send->base.vProps.ndevs = 1;
      send->base.qps[0].qp = &peer.sendQp;
      send->base.qps[0].devIndex = 0;
      send->devs[0].base.cq = &peer.sendCq;
      send->devs[0].base.pd = &peer.pd;
      send->devs[0].putSignalScratchpadMr = &peer.scratchpadMr;

      ncclIbRecvComm* recv = peer.recv.get();
      recv->base.vProps.ndevs = 1;
      recv->flushEnabled = 1;
      recv->devs[0].base.cq = &peer.recvCq;
      recv->devs[0].base.pd = &peer.pd;
      recv->devs[0].gpuFlush.qp.qp = &peer.flushQp;
      recv->devs[0].gpuFlush.qp.devIndex = 0;
    }

    ncclNetIb = {};
    ncclNetIb.listen = FakeListen;
    ncclNetIb.regMr = FakeRegMr;
    ncclNetIb.deregMr = FakeDeregMr;
    ncclNetIb.isend = FakeIsend;
    ncclNetIb.irecv = FakeIrecv;
    ncclNetIb.test = FakeNetTest;
    ncclNetIb.closeSend = FakeClose;
    ncclNetIb.closeRecv = FakeClose;
    ncclNetIb.closeListen = FakeClose;

    cComm_.nranks = kNRanks;
    cComm_.rank = kRank;
    cComm_.sendComm = &ringSend_;
    cComm_.recvComm = &ringRecv_;
    cComm_.allGather = FakeAllGather;

    srcMr_ = Register(kSrcBase);
    dstMr_ = Register(kDstBase);
    signalMr_ = Register(kSignalBase);
  }

  void TearDown() override {
    DestroyContext();
    for (void* mhandle : {srcMr_, dstMr_, signalMr_}) {
      if (mhandle != nullptr) EXPECT_EQ(ncclSuccess, ncclRmaIbProxyDeregMrSym(&cComm_, mhandle));
    }
    g_ib = nullptr;
    ResetNcclFakes();
  }

  void* Register(uintptr_t base) {
    void* mhandle = nullptr;
    EXPECT_EQ(ncclSuccess,
              ncclRmaIbProxyRegMrSym(&cComm_, reinterpret_cast<void*>(base), 1 << 20, NCCL_PTR_CUDA, 0, &mhandle));
    return mhandle;
  }

  static void SetWrBatchsize(int64_t value) {
    g_loadParam = [value](const char* env, int64_t deftVal) {
      return std::strcmp(env, "RMA_IB_WR_BATCHSIZE") == 0 ? value : deftVal;
    };
  }

  void CreateContext() {
    ncclRmaConfig_t config = {};
    config.nContexts = 1;
    config.trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
    config.rankStride = 1;
    ASSERT_EQ(ncclSuccess, ncclRmaIbProxyCreateContext(&cComm_, &config, &rmaCtx_));
    ASSERT_NE(nullptr, rmaCtx_);
  }

  void DestroyContext() {
    if (rmaCtx_ == nullptr) return;
    EXPECT_EQ(ncclSuccess, ncclRmaIbProxyDestroyContext(rmaCtx_));
    // DestroyContext closes the comms but leaves the context array to its caller.
    free(rmaCtx_);
    rmaCtx_ = nullptr;
  }

  void* Put(int rank, uint64_t dstOff, uint32_t optFlags) {
    void* request = nullptr;
    EXPECT_EQ(ncclSuccess,
              ncclRmaIbProxyIPut(rmaCtx_, 0, 0, srcMr_, kOpBytes, dstOff, dstMr_, rank, optFlags, &request));
    return request;
  }

  void* PutSignal(int rank, uint64_t dstOff, uint32_t optFlags) {
    void* request = nullptr;
    EXPECT_EQ(ncclSuccess, ncclRmaIbProxyIPutSignal(rmaCtx_, 0, 0, srcMr_, kOpBytes, dstOff, dstMr_, rank, kSignalOff,
                                                    signalMr_, 1, NCCL_NET_SIGNAL_OP_INC, false, optFlags, &request));
    return request;
  }

  void* Get(int rank, uint64_t remoteOff, uint32_t optFlags) {
    void* request = nullptr;
    EXPECT_EQ(ncclSuccess,
              ncclRmaIbProxyIGet(rmaCtx_, 0, remoteOff, dstMr_, kOpBytes, 0, srcMr_, rank, optFlags, &request));
    return request;
  }

  void* Flush(int rank) {
    void* request = nullptr;
    EXPECT_EQ(ncclSuccess, ncclRmaIbProxyIFlush(rmaCtx_, 0, dstMr_, rank, &request));
    return request;
  }

  int Done(void* request) {
    int done = -1;
    EXPECT_EQ(ncclSuccess, ncclRmaIbProxyTest(&cComm_, request, &done));
    return done;
  }

  std::vector<size_t> DoorbellSizes() const {
    std::vector<size_t> sizes;
    for (const Doorbell& doorbell : ib_.doorbells) sizes.push_back(doorbell.wrs.size());
    return sizes;
  }

  ibv_qp* SendQp(int rank) { return &ib_.peers[rank].sendQp; }

  FakeIb ib_;
  ncclGinIbCollComm cComm_ = {};
  int ringSend_ = 0;
  int ringRecv_ = 0;
  void* rmaCtx_ = nullptr;
  void* srcMr_ = nullptr;
  void* dstMr_ = nullptr;
  void* signalMr_ = nullptr;
};

using RmaIbProxyDoorbellTest = RmaIbProxyTest;
using RmaIbProxyCompletionTest = RmaIbProxyTest;

TEST_F(RmaIbProxyDoorbellTest, AggregatedOpsShareOneDoorbell) {
  CreateContext();
  void* put = Put(kPeer, 0x100, kAggregate);
  PutSignal(kPeer, 0x200, kAggregate);
  Get(kPeer, 0x300, kAggregate);
  EXPECT_TRUE(ib_.doorbells.empty());

  Done(put);
  ASSERT_EQ(1u, ib_.doorbells.size());
  EXPECT_EQ(SendQp(kPeer), ib_.doorbells[0].qp);
  EXPECT_EQ((std::vector<ibv_wr_opcode>{IBV_WR_RDMA_WRITE, IBV_WR_RDMA_WRITE, IBV_WR_ATOMIC_FETCH_AND_ADD,
                                        IBV_WR_RDMA_READ}),
            Opcodes(ib_.doorbells[0]));
  EXPECT_EQ((std::vector<uint64_t>{kDstBase + 0x100, kDstBase + 0x200, kSignalBase + kSignalOff, kDstBase + 0x300}),
            RemoteAddrs(ib_.doorbells[0]));
}

TEST_F(RmaIbProxyDoorbellTest, OpsWithoutTheAggregateFlagRingAtOnce) {
  CreateContext();
  Put(kPeer, 0x100, kNoFlags);
  EXPECT_EQ((std::vector<size_t>{1}), DoorbellSizes());
  PutSignal(kPeer, 0x200, kNoFlags);
  EXPECT_EQ((std::vector<size_t>{1, 2}), DoorbellSizes());
  Get(kPeer, 0x300, kNoFlags);
  EXPECT_EQ((std::vector<size_t>{1, 2, 1}), DoorbellSizes());
}

TEST_F(RmaIbProxyDoorbellTest, FullBatchRingsWithoutWaitingForTest) {
  SetWrBatchsize(8);
  CreateContext();
  void* last = nullptr;
  for (int i = 0; i < 8; i++) last = PutSignal(kPeer, 0x100 * i, kAggregate);
  EXPECT_EQ((std::vector<size_t>{8, 8}), DoorbellSizes());

  std::vector<uint64_t> putAddrs;
  for (const Doorbell& doorbell : ib_.doorbells) {
    for (const PostedWr& wr : doorbell.wrs) {
      if (wr.opcode == IBV_WR_RDMA_WRITE) putAddrs.push_back(wr.remoteAddr);
    }
  }
  std::vector<uint64_t> issued;
  for (int i = 0; i < 8; i++) issued.push_back(kDstBase + 0x100 * i);
  EXPECT_EQ(issued, putAddrs);

  Done(last);
  EXPECT_EQ((std::vector<size_t>{8, 8}), DoorbellSizes());
}

TEST_F(RmaIbProxyDoorbellTest, DefaultBatchHoldsThirtyTwoPutSignals) {
  CreateContext();
  void* last = nullptr;
  for (int i = 0; i < 40; i++) last = PutSignal(kPeer, 0x100 * i, kAggregate);
  EXPECT_EQ((std::vector<size_t>{64}), DoorbellSizes());
  Done(last);
  EXPECT_EQ((std::vector<size_t>{64, 16}), DoorbellSizes());
}

TEST_F(RmaIbProxyDoorbellTest, OversizedBatchsizeIsClampedToSixtyFour) {
  SetWrBatchsize(1000);
  CreateContext();
  void* last = nullptr;
  for (int i = 0; i < 40; i++) last = PutSignal(kPeer, 0x100 * i, kAggregate);
  EXPECT_EQ((std::vector<size_t>{64}), DoorbellSizes());
  Done(last);
  EXPECT_EQ((std::vector<size_t>{64, 16}), DoorbellSizes());
}

TEST_F(RmaIbProxyDoorbellTest, SwitchingPeersRingsThePendingBatch) {
  CreateContext();
  Put(kPeer, 0x100, kAggregate);
  Put(kPeer, 0x200, kAggregate);
  EXPECT_TRUE(ib_.doorbells.empty());

  void* self = Put(kSelf, 0x300, kAggregate);
  ASSERT_EQ(1u, ib_.doorbells.size());
  EXPECT_EQ(SendQp(kPeer), ib_.doorbells[0].qp);
  EXPECT_EQ(2u, ib_.doorbells[0].wrs.size());

  Done(self);
  ASSERT_EQ(2u, ib_.doorbells.size());
  EXPECT_EQ(SendQp(kSelf), ib_.doorbells[1].qp);
  EXPECT_EQ(1u, ib_.doorbells[1].wrs.size());
}

TEST_F(RmaIbProxyDoorbellTest, FlushRingsThePendingBatchFirst) {
  CreateContext();
  Put(kPeer, 0x100, kAggregate);
  Put(kPeer, 0x200, kAggregate);
  Flush(kPeer);

  ASSERT_EQ(2u, ib_.doorbells.size());
  EXPECT_EQ(SendQp(kPeer), ib_.doorbells[0].qp);
  EXPECT_EQ((std::vector<ibv_wr_opcode>{IBV_WR_RDMA_WRITE, IBV_WR_RDMA_WRITE}), Opcodes(ib_.doorbells[0]));
  EXPECT_EQ(&ib_.peers[kPeer].flushQp, ib_.doorbells[1].qp);
  EXPECT_EQ((std::vector<ibv_wr_opcode>{IBV_WR_RDMA_READ}), Opcodes(ib_.doorbells[1]));
}

TEST_F(RmaIbProxyDoorbellTest, DestroyContextRingsThePendingBatch) {
  CreateContext();
  Put(kPeer, 0x100, kAggregate);
  Put(kPeer, 0x200, kAggregate);
  EXPECT_TRUE(ib_.doorbells.empty());

  DestroyContext();
  EXPECT_EQ((std::vector<size_t>{2}), DoorbellSizes());
}

TEST_F(RmaIbProxyCompletionTest, OnlyTheLastWrOfEachDoorbellIsSignaled) {
  CreateContext();
  Put(kPeer, 0x100, kAggregate);
  PutSignal(kPeer, 0x200, kAggregate);
  Get(kPeer, 0x300, kAggregate);
  PutSignal(kPeer, 0x400, kNoFlags);
  Put(kPeer, 0x500, kNoFlags);

  ASSERT_FALSE(ib_.doorbells.empty());
  for (const Doorbell& doorbell : ib_.doorbells) {
    std::vector<bool> lastOnly(doorbell.wrs.size(), false);
    lastOnly.back() = true;
    EXPECT_EQ(lastOnly, Signaled(doorbell));
  }
}

TEST_F(RmaIbProxyCompletionTest, OneCqePerDoorbellCompletesEveryRequest) {
  CreateContext();
  void* put = Put(kPeer, 0x100, kAggregate);
  void* putSignal = PutSignal(kPeer, 0x200, kAggregate);
  void* get = Get(kPeer, 0x300, kAggregate);

  EXPECT_EQ(1, Done(put));
  EXPECT_EQ(1, Done(putSignal));
  EXPECT_EQ(1, Done(get));
  EXPECT_EQ(static_cast<int>(ib_.doorbells.size()), ib_.cqesRaised);
}

TEST_F(RmaIbProxyCompletionTest, RequestsStayPendingUntilTheCqeIsPolled) {
  CreateContext();
  ib_.deliverCqes = false;
  void* first = Put(kPeer, 0x100, kAggregate);
  void* second = Put(kPeer, 0x200, kAggregate);
  EXPECT_EQ(0, Done(first));
  EXPECT_EQ(0, Done(second));

  ib_.deliverCqes = true;
  EXPECT_EQ(1, Done(second));
  EXPECT_EQ(1, Done(first));
}

TEST_F(RmaIbProxyCompletionTest, FlushCompletesOnTheRecvComm) {
  CreateContext();
  void* flush = Flush(kPeer);
  ASSERT_NE(nullptr, flush);
  ASSERT_EQ(1u, ib_.doorbells.size());
  EXPECT_EQ(&ib_.peers[kPeer].flushQp, ib_.doorbells[0].qp);
  EXPECT_EQ(1, Done(flush));
}

TEST_F(RmaIbProxyCompletionTest, ErrorCqeFailsTheRequest) {
  CreateContext();
  ib_.cqeStatus = IBV_WC_REM_ACCESS_ERR;
  void* put = Put(kPeer, 0x100, kNoFlags);
  int done = -1;
  EXPECT_EQ(ncclRemoteError, ncclRmaIbProxyTest(&cComm_, put, &done));
  EXPECT_EQ(0, done);
}

}  // namespace
