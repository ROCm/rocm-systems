/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_ib/gin.cc (NET_IB_GIN_CC_PATH): the RMA IB proxy
// groups work requests and rings one doorbell per batch, signaling only the tail. The GDAKI arm
// is compiled out (__HIP_PLATFORM_AMD__), as in the shipping ROCm build.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"

#include NET_IB_GIN_CC_PATH

namespace {

template <typename T>
struct ZeroedPtr {
  ZeroedPtr()
      : p(static_cast<T*>(aligned_alloc(alignof(T), (sizeof(T) + alignof(T) - 1) / alignof(T) * alignof(T)))) {
    memset(p, 0, sizeof(T));
  }
  ~ZeroedPtr() { free(p); }
  ZeroedPtr(const ZeroedPtr&) = delete;
  ZeroedPtr& operator=(const ZeroedPtr&) = delete;
  T* operator->() const { return p; }
  T* get() const { return p; }
  T* p;
};

struct PostedWr {
  enum ibv_wr_opcode opcode;
  unsigned send_flags;
  bool hasNext;
  uint64_t remote;
  uint32_t rkey;
  uint32_t length;
  uint32_t lkey;
};

struct Post {
  struct ibv_qp* qp;
  std::vector<PostedWr> wrs;
};

struct BatchLog {
  std::vector<Post> posts;
  std::vector<struct ncclIbRequest*> completions;
  int postResult = 0;
};

BatchLog* g_log = nullptr;

int FakePostSend(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr**) {
  if (g_log->postResult != 0) return g_log->postResult;
  Post post;
  post.qp = qp;
  for (; wr != nullptr; wr = wr->next) {
    PostedWr snap{};
    snap.opcode = wr->opcode;
    snap.send_flags = wr->send_flags;
    snap.hasNext = wr->next != nullptr;
    snap.remote = wr->opcode == IBV_WR_ATOMIC_FETCH_AND_ADD ? wr->wr.atomic.remote_addr : wr->wr.rdma.remote_addr;
    snap.rkey = wr->opcode == IBV_WR_ATOMIC_FETCH_AND_ADD ? wr->wr.atomic.rkey : wr->wr.rdma.rkey;
    if (wr->sg_list != nullptr) {
      snap.length = wr->sg_list->length;
      snap.lkey = wr->sg_list->lkey;
    }
    post.wrs.push_back(snap);
  }
  g_log->posts.push_back(std::move(post));
  return 0;
}

int FakePollCq(struct ibv_cq*, int n, struct ibv_wc* wc) {
  int count = 0;
  while (count < n && !g_log->completions.empty()) {
    struct ncclIbRequest* req = g_log->completions.front();
    g_log->completions.erase(g_log->completions.begin());
    memset(&wc[count], 0, sizeof(wc[count]));
    wc[count].wr_id = static_cast<uint64_t>(req - req->base->reqs);
    wc[count].status = IBV_WC_SUCCESS;
    wc[count].opcode = IBV_WC_RDMA_WRITE;
    count++;
  }
  return count;
}

void ExpectSignaledTail(const Post& post) {
  if (post.wrs.empty()) {
    ADD_FAILURE() << "doorbell posted an empty chain";
    return;
  }
  for (size_t i = 0; i + 1 < post.wrs.size(); i++) {
    EXPECT_EQ(post.wrs[i].send_flags & IBV_SEND_SIGNALED, 0u) << "wr " << i;
    EXPECT_TRUE(post.wrs[i].hasNext) << "wr " << i;
  }
  EXPECT_NE(post.wrs.back().send_flags & IBV_SEND_SIGNALED, 0u);
  EXPECT_FALSE(post.wrs.back().hasNext);
}

class NetIbGinBatchMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    memset(&ibCtx_, 0, sizeof(ibCtx_));
    ibCtx_.ops.post_send = FakePostSend;
    ibCtx_.ops.poll_cq = FakePollCq;
    qp_.context = &ibCtx_;
    qpB_.context = &ibCtx_;
    flushQp_.context = &ibCtx_;
    cq_.context = &ibCtx_;

    send_->base.vProps.ndevs = 1;
    send_->base.qps[0].qp = &qp_;
    send_->base.qps[0].devIndex = 0;
    send_->devs[0].base.cq = &cq_;
    send_->devs[0].putSignalScratchpadMr = &scratchMr_;
    scratchMr_.lkey = 0x5c;

    sendB_->base.vProps.ndevs = 1;
    sendB_->base.qps[0].qp = &qpB_;
    sendB_->base.qps[0].devIndex = 0;
    sendB_->devs[0].base.cq = &cq_;

    recv_->base.vProps.ndevs = 1;
    recv_->flushEnabled = 1;
    recv_->devs[0].base.cq = &cq_;
    recv_->devs[0].gpuFlush.qp.qp = &flushQp_;
    recv_->devs[0].gpuFlush.qp.devIndex = 0;

    ctx_[0].rank = 0;
    ctx_[0].nranks = 3;
    ctx_[0].nContexts = 1;
    ctx_[0].wrBatchsize = NCCL_RMA_IB_WR_BATCHSIZE;
    ctx_[0].fullSendComm = sendComms_;
    ctx_[0].fullRecvComm = recvComms_;
    sendComms_[1] = send_.get();
    sendComms_[2] = sendB_.get();
    recvComms_[1] = recv_.get();

    src_ = Handle(0x1000, 0xa0, 0xb0);
    dst_ = Handle(0x2000, 0xa1, 0xb1);
    sig_ = Handle(0x3000, 0xa2, 0xb2);
    log_.posts.clear();
    log_.completions.clear();
    log_.postResult = 0;
    g_log = &log_;
  }

  void TearDown() override { g_log = nullptr; }

  struct ncclRmaIbProxyMrHandle Handle(uintptr_t base, uint32_t lkey, uint32_t rkey0) {
    mrs_.emplace_back(new ibv_mr{});
    mrs_.back()->lkey = lkey;
    mrHandles_.emplace_back(new ncclIbMrHandle{});
    mrHandles_.back()->mrs[0] = mrs_.back().get();
    vas_.emplace_back(new uintptr_t[3]{base, base + 0x10000, base + 0x20000});
    rkeys_.emplace_back(new uint32_t[3]{rkey0, rkey0 + 1, rkey0 + 2});
    return {mrHandles_.back().get(), vas_.back().get(), rkeys_.back().get()};
  }

  ncclResult_t Put(uint32_t rank, uint32_t flags, void** request) {
    return ncclRmaIbProxyIPut(ctx_, 0, 8, &src_, 64, 16, &dst_, rank, flags, request);
  }

  struct ibv_context ibCtx_ {};
  struct ibv_qp qp_ {};
  struct ibv_qp qpB_ {};
  struct ibv_qp flushQp_ {};
  struct ibv_cq cq_ {};
  struct ibv_mr scratchMr_ {};
  ZeroedPtr<ncclIbSendComm> send_;
  ZeroedPtr<ncclIbSendComm> sendB_;
  ZeroedPtr<ncclIbRecvComm> recv_;
  struct ncclRmaIbProxyCtx ctx_[1] {};
  void* sendComms_[3] = {};
  void* recvComms_[3] = {};
  struct ncclRmaIbProxyMrHandle src_ {};
  struct ncclRmaIbProxyMrHandle dst_ {};
  struct ncclRmaIbProxyMrHandle sig_ {};
  std::vector<std::unique_ptr<ibv_mr>> mrs_;
  std::vector<std::unique_ptr<ncclIbMrHandle>> mrHandles_;
  std::vector<std::unique_ptr<uintptr_t[]>> vas_;
  std::vector<std::unique_ptr<uint32_t[]>> rkeys_;
  BatchLog log_;
};

TEST_F(NetIbGinBatchMicrotest, AggregatePuts_StayUnpostedUntilTheBatchCannotHoldAPutSignal) {
  ctx_[0].wrBatchsize = 4;  // room for a following put+signal is 2, so the 3rd put posts
  void* reqs[3] = {};
  for (int i = 0; i < 2; i++) {
    ASSERT_EQ(Put(1, ncclRmaOptFlagsAggregateRequests, &reqs[i]), ncclSuccess);
  }
  EXPECT_TRUE(log_.posts.empty());

  ASSERT_EQ(Put(1, ncclRmaOptFlagsAggregateRequests, &reqs[2]), ncclSuccess);
  ASSERT_EQ(log_.posts.size(), 1u);
  EXPECT_EQ(log_.posts[0].qp, &qp_);
  ASSERT_EQ(log_.posts[0].wrs.size(), 3u);
  ExpectSignaledTail(log_.posts[0]);
  for (const PostedWr& wr : log_.posts[0].wrs) EXPECT_EQ(wr.opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(log_.posts[0].wrs[0].remote, 0x2000u + 0x10000 + 16);
  EXPECT_EQ(log_.posts[0].wrs[0].rkey, 0xb1u + 1);
  EXPECT_EQ(log_.posts[0].wrs[0].length, 64u);
  EXPECT_EQ(log_.posts[0].wrs[0].lkey, 0xa0u);
}

TEST_F(NetIbGinBatchMicrotest, DefaultPuts_RingTheDoorbellForEachRequest) {
  void* req = nullptr;
  for (int i = 0; i < 3; i++) {
    ASSERT_EQ(Put(1, ncclRmaOptFlagsDefault, &req), ncclSuccess);
  }
  ASSERT_EQ(log_.posts.size(), 3u);
  for (const Post& post : log_.posts) {
    ASSERT_EQ(post.wrs.size(), 1u);
    ExpectSignaledTail(post);
    EXPECT_EQ(post.qp, &qp_);
  }
}

TEST_F(NetIbGinBatchMicrotest, PutSignal_ClosesTheBatchOnceAnotherPutSignalWouldNotFit) {
  ctx_[0].wrBatchsize = 4;
  void* put = nullptr;
  ASSERT_EQ(Put(1, ncclRmaOptFlagsAggregateRequests, &put), ncclSuccess);
  ASSERT_EQ(Put(1, ncclRmaOptFlagsAggregateRequests, &put), ncclSuccess);
  EXPECT_TRUE(log_.posts.empty());

  // Two puts leave exactly enough room for this put+signal. Committing it leaves no room for
  // another put+signal, so the doorbell rings for the whole chain and only the atomic is signaled.
  void* signal = nullptr;
  ASSERT_EQ(ncclRmaIbProxyIPutSignal(ctx_, 0, 0, &src_, 32, 0, &dst_, 1, 8, &sig_, 5, NCCL_NET_SIGNAL_OP_ADD, false,
                                     ncclRmaOptFlagsAggregateRequests, &signal),
            ncclSuccess);
  ASSERT_EQ(log_.posts.size(), 1u);
  ASSERT_EQ(log_.posts[0].wrs.size(), 4u);
  EXPECT_EQ(log_.posts[0].wrs[0].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(log_.posts[0].wrs[1].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(log_.posts[0].wrs[2].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(log_.posts[0].wrs[3].opcode, IBV_WR_ATOMIC_FETCH_AND_ADD);
  EXPECT_EQ(log_.posts[0].wrs[3].remote, 0x3000u + 0x10000 + 8);
  ExpectSignaledTail(log_.posts[0]);
}

TEST_F(NetIbGinBatchMicrotest, PutSignal_StaysHeldUntilTheNextPutRingsTheDoorbell) {
  ctx_[0].wrBatchsize = 6;
  void* put = nullptr;
  ASSERT_EQ(Put(1, ncclRmaOptFlagsAggregateRequests, &put), ncclSuccess);
  ASSERT_EQ(Put(1, ncclRmaOptFlagsAggregateRequests, &put), ncclSuccess);
  void* signal = nullptr;
  ASSERT_EQ(ncclRmaIbProxyIPutSignal(ctx_, 0, 0, &src_, 32, 0, &dst_, 1, 8, &sig_, 5, NCCL_NET_SIGNAL_OP_ADD, false,
                                     ncclRmaOptFlagsAggregateRequests, &signal),
            ncclSuccess);
  EXPECT_TRUE(log_.posts.empty());  // 4 WRs, and a put+signal would still fit

  ASSERT_EQ(Put(1, ncclRmaOptFlagsDefault, &put), ncclSuccess);
  ASSERT_EQ(log_.posts.size(), 1u);
  ASSERT_EQ(log_.posts[0].wrs.size(), 5u);  // two puts, the signal's write+atomic, then the closing put
  EXPECT_EQ(log_.posts[0].wrs[2].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(log_.posts[0].wrs[3].opcode, IBV_WR_ATOMIC_FETCH_AND_ADD);
  EXPECT_EQ(log_.posts[0].wrs[3].remote, 0x3000u + 0x10000 + 8);
  EXPECT_EQ(log_.posts[0].wrs[3].send_flags & IBV_SEND_SIGNALED, 0u);
  EXPECT_EQ(log_.posts[0].wrs[4].opcode, IBV_WR_RDMA_WRITE);
  ExpectSignaledTail(log_.posts[0]);
}

TEST_F(NetIbGinBatchMicrotest, IGet_AggregatesAReadTheSameWay) {
  ctx_[0].wrBatchsize = 4;
  void* req = nullptr;
  ASSERT_EQ(ncclRmaIbProxyIGet(ctx_, 0, 4, &dst_, 32, 12, &src_, 1, ncclRmaOptFlagsAggregateRequests, &req),
            ncclSuccess);
  EXPECT_TRUE(log_.posts.empty());
  ASSERT_EQ(ncclRmaIbProxyIGet(ctx_, 0, 4, &dst_, 32, 12, &src_, 1, ncclRmaOptFlagsDefault, &req), ncclSuccess);
  ASSERT_EQ(log_.posts.size(), 1u);
  ASSERT_EQ(log_.posts[0].wrs.size(), 2u);
  EXPECT_EQ(log_.posts[0].wrs[0].opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(log_.posts[0].wrs[1].opcode, IBV_WR_RDMA_READ);
  ExpectSignaledTail(log_.posts[0]);
}

TEST_F(NetIbGinBatchMicrotest, QpChange_FlushesThePreviousPeerBeforeQueueingTheNext) {
  void* req = nullptr;
  ASSERT_EQ(Put(1, ncclRmaOptFlagsAggregateRequests, &req), ncclSuccess);
  EXPECT_TRUE(log_.posts.empty());
  ASSERT_EQ(Put(2, ncclRmaOptFlagsAggregateRequests, &req), ncclSuccess);
  ASSERT_EQ(log_.posts.size(), 1u);
  EXPECT_EQ(log_.posts[0].qp, &qp_);
  ASSERT_EQ(log_.posts[0].wrs.size(), 1u);
  ExpectSignaledTail(log_.posts[0]);
}

TEST_F(NetIbGinBatchMicrotest, Test_PostsTheHeldBatchAndOneTailCompletionRetiresEveryEarlierRequest) {
  ctx_[0].wrBatchsize = 4;
  void* reqs[4] = {};
  for (int i = 0; i < 3; i++) ASSERT_EQ(Put(1, ncclRmaOptFlagsAggregateRequests, &reqs[i]), ncclSuccess);
  ASSERT_EQ(log_.posts.size(), 1u);  // the 3rd put closed the batch
  ASSERT_EQ(Put(1, ncclRmaOptFlagsAggregateRequests, &reqs[3]), ncclSuccess);
  EXPECT_EQ(log_.posts.size(), 1u);  // the 4th is still held

  log_.completions.push_back(static_cast<ncclIbRequest*>(reqs[2]));
  int done = -1;
  ASSERT_EQ(ncclRmaIbProxyTest(nullptr, reqs[0], &done), ncclSuccess);
  EXPECT_EQ(done, 1);
  ASSERT_EQ(log_.posts.size(), 2u);
  EXPECT_EQ(log_.posts[1].qp, &qp_);
  ASSERT_EQ(log_.posts[1].wrs.size(), 1u);
  ExpectSignaledTail(log_.posts[1]);

  for (int i = 1; i <= 2; i++) {
    done = -1;
    ASSERT_EQ(ncclRmaIbProxyTest(nullptr, reqs[i], &done), ncclSuccess);
    EXPECT_EQ(done, 1) << "request " << i;
  }
  done = -1;
  ASSERT_EQ(ncclRmaIbProxyTest(nullptr, reqs[3], &done), ncclSuccess);
  EXPECT_EQ(done, 0);

  log_.completions.push_back(static_cast<ncclIbRequest*>(reqs[3]));
  ASSERT_EQ(ncclRmaIbProxyTest(nullptr, reqs[3], &done), ncclSuccess);
  EXPECT_EQ(done, 1);
}

TEST_F(NetIbGinBatchMicrotest, IFlush_RingsThePendingDoorbellBeforeTheFlush) {
  void* put = nullptr;
  ASSERT_EQ(Put(1, ncclRmaOptFlagsAggregateRequests, &put), ncclSuccess);
  EXPECT_TRUE(log_.posts.empty());

  void* flush = nullptr;
  ASSERT_EQ(ncclRmaIbProxyIFlush(ctx_, 0, &dst_, 1, &flush), ncclSuccess);
  ASSERT_NE(flush, nullptr);
  ASSERT_EQ(log_.posts.size(), 2u);
  EXPECT_EQ(log_.posts[0].qp, &qp_);
  ASSERT_EQ(log_.posts[0].wrs.size(), 1u);
  EXPECT_EQ(log_.posts[0].wrs[0].opcode, IBV_WR_RDMA_WRITE);
  ExpectSignaledTail(log_.posts[0]);
  EXPECT_EQ(log_.posts[1].qp, &flushQp_);
  ASSERT_EQ(log_.posts[1].wrs.size(), 1u);
  EXPECT_EQ(log_.posts[1].wrs[0].opcode, IBV_WR_RDMA_READ);
  EXPECT_NE(log_.posts[1].wrs[0].send_flags & IBV_SEND_SIGNALED, 0u);
  EXPECT_EQ(static_cast<ncclIbRequest*>(flush)->type, NCCL_NET_IB_REQ_FLUSH);
}

TEST_F(NetIbGinBatchMicrotest, DestroyContext_PostsAHeldBatchBeforeReleasingTheComms) {
  struct ncclRmaIbProxyCtx ctx {};
  ctx.rank = 0;
  ctx.nranks = 2;
  ctx.nContexts = 1;
  ctx.wrBatchsize = NCCL_RMA_IB_WR_BATCHSIZE;
  ctx.fullSendComm = static_cast<void**>(calloc(2, sizeof(void*)));
  ctx.fullRecvComm = static_cast<void**>(calloc(2, sizeof(void*)));
  ASSERT_NE(ctx.fullSendComm, nullptr);
  ASSERT_NE(ctx.fullRecvComm, nullptr);
  ctx.fullSendComm[1] = send_.get();
  ctx.fullRecvComm[1] = recv_.get();

  void* put = nullptr;
  ASSERT_EQ(ncclRmaIbProxyIPut(&ctx, 0, 8, &src_, 64, 16, &dst_, 1, ncclRmaOptFlagsAggregateRequests, &put),
            ncclSuccess);
  EXPECT_TRUE(log_.posts.empty());
  ASSERT_EQ(ncclRmaIbProxyDestroyContext(&ctx), ncclSuccess);
  ASSERT_EQ(log_.posts.size(), 1u);
  EXPECT_EQ(log_.posts[0].qp, &qp_);
  ASSERT_EQ(log_.posts[0].wrs.size(), 1u);
  ExpectSignaledTail(log_.posts[0]);
  EXPECT_EQ(ctx.fullSendComm, nullptr);
  EXPECT_EQ(ctx.fullRecvComm, nullptr);
}

TEST_F(NetIbGinBatchMicrotest, UnsupportedSignalOp_PostsNothing) {
  void* request = nullptr;
  EXPECT_EQ(ncclRmaIbProxyIPutSignal(ctx_, 0, 0, &src_, 8, 0, &dst_, 1, 0, &sig_, 1, 0x7, false,
                                     ncclRmaOptFlagsAggregateRequests, &request),
            ncclInvalidArgument);
  EXPECT_TRUE(log_.posts.empty());
}

}  // namespace

// Symbols gin.cc calls that live in other net_ib translation units. The data-path tests never
// reach them; they only have to link. NCCL_PARAM bodies inside gin.cc itself go through g_loadParam.

static ncclResult_t OkClose(void*) { return ncclSuccess; }
static ncclResult_t OkDereg(void*, void*) { return ncclSuccess; }

ncclNet_t ncclNetIb = [] {
  ncclNet_t net{};
  net.closeSend = OkClose;
  net.closeRecv = OkClose;
  net.closeListen = OkClose;
  net.deregMr = OkDereg;
  return net;
}();

int ncclNIbDevs = 0;
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
struct ncclIbMergedDev ncclIbMergedDevs[MAX_IB_VDEVS];

int64_t ncclParamIbTc() { return -1; }
size_t ncclOsGetPageSize() { return 4096; }

const char* ncclIbReqTypeStr[] = {"Unused", "Send", "Recv", "Flush", "IPut", "IGet", "Failed"};

ncclResult_t ncclIbGdrSupport() { return ncclSuccess; }
ncclResult_t ncclIbPeerMemSupport() { return ncclSuccess; }
ncclResult_t ncclIbDmaBufSupport(int) { return ncclSuccess; }
ncclResult_t ncclIbInitDevices(ncclDebugLogger_t, ncclProfilerCallback_t) { return ncclSuccess; }
ncclResult_t ncclIbFinalizeDevices() { return ncclSuccess; }
ncclResult_t ncclIbDevices(int* ndev) {
  *ndev = 0;
  return ncclSuccess;
}
ncclResult_t ncclIbListen(void*, int, void*, void**) { return ncclInternalError; }
ncclResult_t ncclIbCloseListen(void*) { return ncclSuccess; }
ncclResult_t ncclIbConnectImpl(void*, int, void*, void**, ncclNetDeviceHandle_t**, int, int) {
  return ncclInternalError;
}
ncclResult_t ncclIbAcceptImpl(void*, void**, ncclNetDeviceHandle_t**, int) { return ncclInternalError; }
ncclResult_t ncclIbGetProperties(int, ncclNetProperties_t*) { return ncclInternalError; }
ncclResult_t ncclIbGetGidIndex(struct ibv_context*, uint8_t, struct ibv_port_attr*, int*) {
  return ncclInternalError;
}
void ncclIbSetTrafficClass(void*, int) {}
ncclResult_t ncclIbRegMrDmaBufInternal(void*, void*, size_t, int, uint64_t, int, uint64_t, void**) {
  return ncclInternalError;
}
ncclResult_t ncclIbCreateFlushQp(struct ncclIbRecvComm*) { return ncclSuccess; }

ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req) {
  for (int i = 0; i < NET_IB_MAX_REQUESTS; i++) {
    struct ncclIbRequest* r = base->reqs + i;
    if (r->type != NCCL_NET_IB_REQ_UNUSED) continue;
    r->base = base;
    r->sock = nullptr;
    memset(r->devBases, 0, sizeof(r->devBases));
    memset(r->events, 0, sizeof(r->events));
    *req = r;
    return ncclSuccess;
  }
  *req = nullptr;
  return ncclInternalError;
}

ncclResult_t ncclIbFreeRequest(struct ncclIbRequest* r) {
  r->type = NCCL_NET_IB_REQ_UNUSED;
  return ncclSuccess;
}

void ncclIbAddEvent(struct ncclIbRequest* req, int devIndex) {
  req->events[devIndex]++;
  req->devBases[devIndex] = &((struct ncclIbSendComm*)req->base)->devs[devIndex].base;
}

ncclResult_t ncclSocketGetAddr(struct ncclSocket*, union ncclSocketAddress*) { return ncclSuccess; }

const char* ncclSocketToString(const union ncclSocketAddress*, char* buf, size_t bufLen, const int) {
  if (buf != nullptr && bufLen > 0) buf[0] = '\0';
  return buf;
}
