/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_ib_cast/connect.cc
// (NET_IB_CAST_CONNECT_CC_PATH, with the real qp_sharing.cc): QP setup, GID
// selection, and the listen/connect/accept handshake driven end to end through
// an in-memory socket pair and fake verbs.

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <fcntl.h>

#include <algorithm>
#include <cstdarg>
#include <cstring>
#include <deque>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "ScopedHook.h"
#include "fakes/env_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/ibvwrap_fakes.h"
#include "fakes/ionicdvwrap_fakes.h"
#include "fakes/libc_fakes.h"
#include "fakes/mlx5dvwrap_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/net_ib_cast_common_fakes.h"
#include "fakes/net_ib_cast_gdr_fakes.h"
#include "fakes/net_ib_cast_init_fakes.h"
#include "fakes/net_ib_cast_multiplane_fakes.h"
#include "fakes/net_ib_cast_p2p_resiliency_fakes.h"
#include "fakes/rocmwrap_fakes.h"
#include "fakes/socket_fakes.h"
#include "../common/LogCapture.hpp"

#include "fakes/param_redirect.h"

#include NET_IB_CAST_QP_SHARING_CC_PATH

// IbCastRoceGetVersionNum opens a sysfs path; libc_seam has no open().
static std::function<int(const char*)> g_testOpen;
static int NetIbCastConnectTestOpen(const char* path, int, ...) { return g_testOpen(path); }

#include "fakes/libc_seam.h"
#define open NetIbCastConnectTestOpen
#include NET_IB_CAST_CONNECT_CC_PATH
#undef open
#include "fakes/libc_seam_undef.h"

namespace {

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

union ibv_gid V4Gid(const char* ip) {
  union ibv_gid g;
  memset(&g, 0, sizeof(g));
  g.raw[10] = 0xff;
  g.raw[11] = 0xff;
  inet_pton(AF_INET, ip, g.raw + 12);
  return g;
}

union ibv_gid V6Gid(const char* ip) {
  union ibv_gid g;
  inet_pton(AF_INET6, ip, g.raw);
  return g;
}

union ncclSocketAddress V4Addr(const char* ip, uint16_t port) {
  union ncclSocketAddress a;
  memset(&a, 0, sizeof(a));
  a.sin.sin_family = AF_INET;
  a.sin.sin_port = htons(port);
  inet_pton(AF_INET, ip, &a.sin.sin_addr);
  return a;
}

bool SameGid(const union ibv_gid& a, const union ibv_gid& b) { return memcmp(a.raw, b.raw, sizeof(a.raw)) == 0; }

// A zeroed comm; the comm structs are alignas(64), more than calloc promises.
template <typename T>
T* NewComm() {
  auto* p = static_cast<T*>(aligned_alloc(alignof(T), sizeof(T)));
  memset(p, 0, sizeof(T));
  return p;
}

int FakePostRecv(struct ibv_qp*, struct ibv_recv_wr*, struct ibv_recv_wr**);

// Verbs objects handed out by the fakes, with a ledger of what is still live.
struct FakeVerbs {
  std::deque<ibv_pd> pds;
  std::deque<ibv_cq> cqs;
  std::deque<ibv_qp> qps;
  std::deque<ibv_mr> mrs;
  std::set<const void*> live;
  struct Modify {
    ibv_qp* qp;
    ibv_qp_attr attr;
    int mask;
  };
  std::vector<Modify> modifies;
  struct Create {
    ibv_qp* qp;
    ibv_pd* pd;
    ibv_qp_init_attr attr;
  };
  std::vector<Create> creates;
  struct Reg {
    ibv_mr* mr;
    ibv_pd* pd;
    void* addr;
    size_t length;
    int access;
  };
  std::vector<Reg> regs;
  std::vector<int> cqSizes;
  std::vector<const void*> destroyed;
  std::map<const ibv_context*, uint32_t> nextQpn;  // QP numbers are per HCA, as on hardware
  int postRecvs = 0;
  int postRecvFailAt = -1;

  ibv_qp* NewQp(ibv_pd* pd, ibv_qp_type type) {
    qps.emplace_back();
    ibv_qp* qp = &qps.back();
    memset(qp, 0, sizeof(*qp));
    qp->qp_type = type;
    qp->pd = pd;
    qp->context = pd ? pd->context : nullptr;
    auto it = nextQpn.emplace(qp->context, 0x100).first;
    qp->qp_num = it->second++;
    live.insert(qp);
    return qp;
  }
  void Release(const void* p) {
    destroyed.push_back(p);
    live.erase(p);
  }
  std::vector<Modify> ModifiesOf(const ibv_qp* qp) const {
    std::vector<Modify> out;
    for (const auto& m : modifies)
      if (m.qp == qp) out.push_back(m);
    return out;
  }
  const Modify* Rtr(const ibv_qp* qp) const {
    for (const auto& m : modifies)
      if (m.qp == qp && m.attr.qp_state == IBV_QPS_RTR) return &m;
    return nullptr;
  }
  const Modify* Rts(const ibv_qp* qp) const {
    for (const auto& m : modifies)
      if (m.qp == qp && m.attr.qp_state == IBV_QPS_RTS) return &m;
    return nullptr;
  }
  const Create* CreateOf(const ibv_qp* qp) const {
    for (const auto& c : creates)
      if (c.qp == qp) return &c;
    return nullptr;
  }
};

FakeVerbs* g_verbs = nullptr;

int FakePostRecv(struct ibv_qp*, struct ibv_recv_wr*, struct ibv_recv_wr**) {
  if (g_verbs->postRecvs == g_verbs->postRecvFailAt) return EIO;
  g_verbs->postRecvs++;
  return 0;
}

// An in-memory socket pair per ncclSocketConnect, accepted in connect order.
struct FakeSockets {
  struct Conn {
    std::string toServer, toClient;
    size_t serverRead = 0, clientRead = 0;
    union ncclSocketAddress serverAddr, clientAddr;
  };
  std::vector<std::unique_ptr<Conn>> conns;
  std::deque<Conn*> pending;
  std::map<const ncclSocket*, std::pair<Conn*, bool>> bound;  // sock -> (conn, isClient)
  union ncclSocketAddress listenAddr = V4Addr("10.0.0.1", 7000);
  union ncclSocketAddress nextClientAddr = V4Addr("10.0.0.2", 0);
  int listens = 0;
  int ready = 1;
  int chunk = 1 << 30;
  std::function<bool(int op, const ncclSocket*)> failProgress = [](int, const ncclSocket*) { return false; };
  std::vector<const ncclSocket*> closed;

  ncclResult_t Progress(int op, ncclSocket* sock, void* ptr, int size, int* offset) {
    if (failProgress(op, sock)) return ncclRemoteError;
    auto it = bound.find(sock);
    if (it == bound.end()) return ncclInternalError;
    Conn* c = it->second.first;
    bool client = it->second.second;
    int want = std::min(size - *offset, chunk);
    if (op == NCCL_SOCKET_SEND) {
      (client ? c->toServer : c->toClient).append(static_cast<char*>(ptr) + *offset, want);
      *offset += want;
    } else {
      const std::string& in = client ? c->toClient : c->toServer;
      size_t& rd = client ? c->clientRead : c->serverRead;
      int n = std::min<int>(want, in.size() - rd);
      memcpy(static_cast<char*>(ptr) + *offset, in.data() + rd, n);
      rd += n;
      *offset += n;
    }
    return ncclSuccess;
  }
};

class NetIbCastConnectMicrotest : public ::testing::Test {
 protected:
  static constexpr int kNDevs = 3;
  static constexpr int kGidIndex = 3;

  void SetUp() override {
    g_verbs = &verbs_;
    for (int i = 0; i < kNDevs; i++) {
      ctx_[i] = {};
      ctx_[i].ops.post_recv = FakePostRecv;
      SetDevice(i, IBV_LINK_LAYER_ETHERNET);
    }
    IbCastNDevs = kNDevs;
    SetMerged(0, {0});
    SetMerged(1, {1});
    SetMerged(2, {0, 1});
    SetMerged(3, {2});
    IbCastNMergedDevs = 4;
    IbCastIfAddr = sockets_.listenAddr;
    memset(nccl_channel_ud_map, 0, sizeof(nccl_channel_ud_map));
    memset(nccl_channel_last_ud, 0, sizeof(nccl_channel_last_ud));
    memset(nccl_udma_mask_warned, 0, sizeof(nccl_udma_mask_warned));
    for (auto& idx : nccl_p2p_udma_idx) idx.store(0);
    ResetQpSharing();
    InstallHooks();
  }

  void TearDown() override {
    for (void* c : sendComms_) IbCastCloseSend(c);
    for (void* c : recvComms_) IbCastCloseRecv(c);
    for (void* l : listens_) {
      // IbCastCloseListen leaves an unfinished accept's stage, and the comm in it, behind.
      ncclIbCommStage* stage = l ? static_cast<ncclIbListenComm*>(l)->stage : nullptr;
      if (stage) {
        if (stage->state != ncclIbCommStateStart) free(stage->comm);
        free(stage->buffer);
        free(stage);
      }
      IbCastCloseListen(l);
    }
    hooks_.clear();
    for (int i = 0; i < kNDevs; i++) {
      IbCastDevs[i].pdRefs = 0;
      IbCastDevs[i].pd = nullptr;
    }
    ResetQpSharing();
    ResetIbvwrapFakes();
    ResetIonicdvwrapFakes();
    ResetMlx5dvwrapFakes();
    ResetSocketFakes();
    ResetNetIbCastCommonFakes();
    ResetNetIbCastInitFakes();
    ResetNetIbCastGdrFakes();
    ResetNetIbCastMultiplaneFakes();
    ResetNetIbCastP2pResiliencyFakes();
    ResetHipFakes();
    ResetLibcFakes();
    ResetEnvFakes();
    ResetRocmWrapFakes();
    ResetNcclFakes();
    rcclTelemetryEnabled = 0;
    g_testOpen = nullptr;
    g_verbs = nullptr;
  }

  static void ResetQpSharing() {
    memset(g_IbCastSharedQpPool, 0, sizeof(g_IbCastSharedQpPool));
    g_IbCastSharedQpPoolCount = 0;
    g_IbCastSharedQpFreeTop = 0;
    memset(g_IbCastCommTable, 0, sizeof(g_IbCastCommTable));
    g_IbCastNextCommId = 1;
    g_IbCastCommIdFreeTop = 0;
    IbCastQpSharingGlobalEnable = false;
  }

  void SetDevice(int i, uint8_t linkLayer) {
    ncclIbDev& d = IbCastDevs[i];
    d.context = &ctx_[i];
    d.portNum = 1;
    d.pdRefs = 0;
    d.pd = nullptr;
    d.maxCqe = 0;
    d.ar = 0;
    d.oooRqSize = 0;
    d.ibProvider = IB_PROVIDER_NONE;
    memset(&d.capsProvider, 0, sizeof(d.capsProvider));
    snprintf(d.devName, sizeof(d.devName), "fake%d", i);
    memset(&d.portAttr, 0, sizeof(d.portAttr));
    d.portAttr.link_layer = linkLayer;
    d.portAttr.active_mtu = IBV_MTU_4096;
    d.portAttr.lid = 0x10 + i;
    d.portAttr.gid_tbl_len = 8;
    d.gidInfo.link_layer = linkLayer;
    d.gidInfo.localGidIndex = kGidIndex;
    d.gidInfo.localGid = V4Gid(i == 0 ? "10.0.0.1" : i == 1 ? "10.0.1.1" : "10.0.2.1");
  }

  // AINIC as IbCastInitDevices leaves it: each device holds its normalized uDMA engine count.
  void Ainic(uint8_t udmaCount = 2) {
    IbCastAinicRoce = true;
    for (int i = 0; i < kNDevs; i++) IbCastDevs[i].capsProvider.ionic.udmaCount = udmaCount;
  }

  void SetMerged(int m, std::vector<int> devs) {
    ncclIbMergedDev& md = IbCastMergedDevs[m];
    md.vProps.ndevs = devs.size();
    for (size_t i = 0; i < devs.size(); i++) md.vProps.devs[i] = devs[i];
    snprintf(md.devName, sizeof(md.devName), "merged%d", m);
  }

  void Param(const std::string& env, int64_t value) { params_[env] = value; }

  template <typename Sig, typename Fn>
  void Hook(std::function<Sig>& slot, Fn fn) {
    hooks_.push_back(std::make_shared<ScopedHook<Sig>>(slot, std::move(fn)));
  }

  void InstallHooks() {
    Hook(g_loadParam, [this](const char* env, int64_t deft) {
      auto it = params_.find(env);
      return it == params_.end() ? deft : it->second;
    });
    Hook(g_ibvAllocPd, [this](ibv_pd** ret, ibv_context* ctx) {
      if (allocPdResult_ != ncclSuccess) {
        *ret = nullptr;
        return allocPdResult_;
      }
      verbs_.pds.emplace_back();
      *ret = &verbs_.pds.back();
      (*ret)->context = ctx;
      verbs_.live.insert(*ret);
      return ncclSuccess;
    });
    Hook(g_ibvDeallocPd, [this](ibv_pd* pd) {
      verbs_.Release(pd);
      return ncclSuccess;
    });
    Hook(g_ibvCreateCq, [this](ibv_cq** ret, ibv_context* ctx, int cqe, void* cqCtx, ibv_comp_channel*, int) {
      verbs_.cqSizes.push_back(cqe);
      if (createCqResult_ != ncclSuccess) {
        *ret = nullptr;
        return createCqResult_;
      }
      verbs_.cqs.emplace_back();
      ibv_cq* cq = &verbs_.cqs.back();
      memset(cq, 0, sizeof(*cq));
      cq->context = ctx;
      cq->cq_context = cqCtx;
      cq->cqe = cqeCap_ > 0 ? std::min(cqe, cqeCap_) : cqe;
      verbs_.live.insert(cq);
      *ret = cq;
      return ncclSuccess;
    });
    Hook(g_ibvDestroyCq, [this](ibv_cq* cq) {
      verbs_.Release(cq);
      return destroyCqResult_;
    });
    Hook(g_ibvCreateQp, [this](ibv_qp** ret, ibv_pd* pd, ibv_qp_init_attr* attr) {
      if (createQpResult_ != ncclSuccess) {
        *ret = nullptr;
        return createQpResult_;
      }
      *ret = verbs_.NewQp(pd, attr->qp_type);
      (*ret)->qp_context = attr->qp_context;
      verbs_.creates.push_back({*ret, pd, *attr});
      return ncclSuccess;
    });
    Hook(g_mlx5dvCreateQp, [this](ibv_context*, ibv_qp_init_attr_ex* attr, mlx5dv_qp_init_attr* dv) -> ibv_qp* {
      mlx5Attrs_.push_back(*attr);
      mlx5DvAttrs_.push_back(*dv);
      if (!mlx5CreateOk_) return nullptr;
      return verbs_.NewQp(attr->pd, attr->qp_type);
    });
    Hook(g_ibvModifyQp, [this](ibv_qp* qp, ibv_qp_attr* attr, int mask) {
      verbs_.modifies.push_back({qp, *attr, mask});
      return modifyFailState_ >= 0 && attr->qp_state == modifyFailState_ ? ncclSystemError : ncclSuccess;
    });
    Hook(g_ibvDestroyQp, [this](ibv_qp* qp) {
      verbs_.Release(qp);
      return ncclSuccess;
    });
    Hook(g_ibvQueryEce, [this](ibv_qp*, ibv_ece* ece, int* supported) {
      ece->vendor_id = 0x15b3;
      ece->options = queryEceOptions_;
      ece->comp_mask = 0;
      *supported = eceSupported_;
      return ncclSuccess;
    });
    Hook(g_ibvSetEce, [this](ibv_qp* qp, ibv_ece* ece, int* supported) {
      setEce_.push_back({qp, *ece});
      *supported = setEceSupported_;
      return ncclSuccess;
    });
    Hook(g_ibvRegMr, [this](ibv_mr** ret, ibv_pd* pd, void* addr, size_t len, int access) {
      if (regMrFailAt_ == static_cast<int>(verbs_.regs.size())) {
        verbs_.regs.push_back({nullptr, pd, addr, len, access});
        *ret = nullptr;
        return ncclSystemError;
      }
      verbs_.mrs.emplace_back();
      ibv_mr* mr = &verbs_.mrs.back();
      memset(mr, 0, sizeof(*mr));
      mr->lkey = 0x1000 + verbs_.mrs.size();
      mr->rkey = 0x2000 + verbs_.mrs.size();
      verbs_.live.insert(mr);
      verbs_.regs.push_back({mr, pd, addr, len, access});
      *ret = mr;
      return ncclSuccess;
    });
    Hook(g_ibvDeregMr, [this](ibv_mr* mr) {
      verbs_.Release(mr);
      return ncclSuccess;
    });
    Hook(g_ibvQueryGid, [this](ibv_context* ctx, uint8_t, int index, union ibv_gid* gid) {
      queriedGids_.push_back(index);
      auto byCtx = gidByCtx_.find(ctx);
      if (byCtx != gidByCtx_.end()) {
        *gid = byCtx->second;
        return queryGidResult_;
      }
      auto it = gidTable_.find(index);
      if (it == gidTable_.end()) memset(gid, 0, sizeof(*gid));
      else *gid = it->second;
      return queryGidResult_;
    });
    Hook(g_ibvGetDeviceName, [](ibv_device*) { return "fakedev"; });
    Hook(g_ionicdvQpSetGda, [](ibv_qp*, bool, bool) { return ncclSuccess; });
    Hook(g_ionicdvPdSetUdmaMask, [this](ibv_pd* pd, uint8_t mask) {
      udmaMasks_.push_back(mask);
      return setUdmaResult_;
    });
    Hook(g_ncclSocketInit, [](ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t magic, ncclSocketType type,
                              volatile uint32_t*, int, int) {
      memset(sock, 0, sizeof(*sock));
      if (addr) sock->addr = *addr;
      sock->magic = magic;
      sock->type = type;
      return ncclSuccess;
    });
    // Each listener gets its own port, so a connect reaches only the listener its handle names.
    Hook(g_ncclSocketListen, [this](ncclSocket* sock) {
      sock->addr.sin.sin_port = htons(7000 + sockets_.listens++);
      return listenResult_;
    });
    Hook(g_ncclSocketGetAddr, [](ncclSocket* sock, union ncclSocketAddress* addr) {
      *addr = sock->addr;
      return ncclSuccess;
    });
    Hook(g_ncclSocketConnect, [this](ncclSocket* sock) {
      sockets_.conns.push_back(std::make_unique<FakeSockets::Conn>());
      FakeSockets::Conn* c = sockets_.conns.back().get();
      c->serverAddr = sock->addr;
      c->clientAddr = sockets_.nextClientAddr;
      c->clientAddr.sin.sin_port = htons(40000 + sockets_.conns.size());
      sockets_.bound[sock] = {c, true};
      sockets_.pending.push_back(c);
      return ncclSuccess;
    });
    Hook(g_ncclSocketReady, [this](ncclSocket*, int* running) {
      *running = sockets_.ready;
      return ncclSuccess;
    });
    Hook(g_ncclSocketAccept, [this](ncclSocket* sock, ncclSocket* listenSock, bool) {
      auto it = std::find_if(sockets_.pending.begin(), sockets_.pending.end(), [&](FakeSockets::Conn* c) {
        return memcmp(&c->serverAddr, &listenSock->addr, sizeof(c->serverAddr)) == 0;
      });
      if (it == sockets_.pending.end()) return ncclInternalError;
      FakeSockets::Conn* c = *it;
      sockets_.pending.erase(it);
      sockets_.bound[sock] = {c, false};
      sock->addr = c->clientAddr;
      return ncclSuccess;
    });
    Hook(g_ncclSocketProgress, [this](int op, ncclSocket* sock, void* ptr, int size, int* offset, bool*) {
      return sockets_.Progress(op, sock, ptr, size, offset);
    });
    Hook(g_ncclSocketClose, [this](ncclSocket* sock, bool) {
      sockets_.closed.push_back(sock);
      sockets_.bound.erase(sock);
      return ncclSuccess;
    });
    Hook(g_ibCastSendCommInit, [this](ncclIbSendComm* c) {
      InitBase(&c->base, true);
      return ncclSuccess;
    });
    Hook(g_ibCastRecvCommInit, [this](ncclIbRecvComm* c) {
      InitBase(&c->base, false);
      c->prepostReceiveWorkRequests = prepost_;
      c->ibRecvWorkRequest = {};
      return ncclSuccess;
    });
    Hook(g_ibCastInitOptRecvCompletion, [this](ncclIbNetCommBase* base, bool useCtsOffload) {
      optRecvCalls_.push_back(useCtsOffload);
      base->optRecvCompletion = useCtsOffload || optRecv_;
    });
  }

  // The part of IbCastBaseCommInit connect.cc relies on.
  void InitBase(ncclIbNetCommBase* base, bool isSend) {
    for (int i = 0; i < NCCL_IB_MAX_QPS; i++) {
      base->qps[i].devIndex = -1;
      base->qps[i].remDevIdx = -1;
      base->activeQps[i] = &base->qps[i];
    }
    base->isSend = isSend;
    base->nqps = -1;
    base->resiliency = resiliency_;
  }

  // Listens on merged device `dev` and returns the handle a peer connects with.
  ncclIbHandle Listen(int dev = 0, void** listenOut = nullptr) {
    ncclIbHandle handle;
    void* lc = nullptr;
    EXPECT_EQ(IbCastListen(nullptr, dev, &handle, &lc), ncclSuccess);
    listens_.push_back(lc);
    if (listenOut) *listenOut = lc;
    return handle;
  }

  // Alternates connect and accept until both sides report a comm.
  ncclResult_t Handshake(ncclIbHandle* handle, void* listenComm, ncclIbSendComm** send, ncclIbRecvComm** recv,
                         int dev = 0, void* ctx = nullptr, ncclNetDeviceHandle_t** sendDev = nullptr,
                         ncclNetDeviceHandle_t** recvDev = nullptr) {
    void* sc = nullptr;
    void* rc = nullptr;
    ncclResult_t res = ncclSuccess;
    for (int i = 0; i < 4096 && (!sc || !rc) && res == ncclSuccess; i++) {
      if (!sc) res = IbCastConnect(ctx, dev, handle, &sc, sendDev);
      if (!rc && res == ncclSuccess) res = IbCastAccept(listenComm, &rc, recvDev);
    }
    // A side that finished before the other failed still gets closed in TearDown.
    if (sc) sendComms_.push_back(sc);
    if (rc) recvComms_.push_back(rc);
    if (res != ncclSuccess) {
      Abandon(handle);
      return res;
    }
    *send = static_cast<ncclIbSendComm*>(sc);
    *recv = static_cast<ncclIbRecvComm*>(rc);
    return sc && rc ? ncclSuccess : ncclInProgress;
  }

  // Connect on a fresh listener; fails the test if the handshake does not finish.
  void Connected(ncclIbSendComm** send, ncclIbRecvComm** recv, int dev = 0) {
    void* lc = nullptr;
    ncclIbHandle h = Listen(dev, &lc);
    ASSERT_EQ(Handshake(&h, lc, send, recv, dev), ncclSuccess);
  }

  // A connect that fails without reaching its fail label keeps the comm in the handle's stage.
  static void Abandon(ncclIbHandle* handle) {
    if (handle->stage.state != ncclIbCommStateStart) {
      free(handle->stage.comm);
      free(handle->stage.buffer);
    }
    handle->stage = {};
  }

  void Forget(void* comm) {
    for (auto* v : {&sendComms_, &recvComms_, &listens_})
      for (auto it = v->begin(); it != v->end(); ++it)
        if (*it == comm) {
          v->erase(it);
          break;
        }
  }

  std::string InfoLog(const std::function<void()>& body) {
    RcclUnitTesting::ScopedDebugLogging info(NCCL_LOG_INFO, ~0ULL);
    return CaptureLog(body);
  }

  ibv_context ctx_[kNDevs] = {};
  FakeVerbs verbs_;
  FakeSockets sockets_;
  std::map<std::string, int64_t> params_;
  std::map<int, union ibv_gid> gidTable_;
  std::map<const ibv_context*, union ibv_gid> gidByCtx_;
  std::vector<int> queriedGids_;
  std::vector<std::shared_ptr<void>> hooks_;
  std::vector<void*> sendComms_, recvComms_, listens_;
  std::vector<std::pair<ibv_qp*, ibv_ece>> setEce_;
  std::vector<uint8_t> udmaMasks_;
  std::vector<ibv_qp_init_attr_ex> mlx5Attrs_;
  std::vector<mlx5dv_qp_init_attr> mlx5DvAttrs_;
  std::vector<bool> optRecvCalls_;
  ncclIbResiliency* resiliency_ = nullptr;
  ncclResult_t allocPdResult_ = ncclSuccess;
  ncclResult_t createCqResult_ = ncclSuccess;
  ncclResult_t destroyCqResult_ = ncclSuccess;
  ncclResult_t createQpResult_ = ncclSuccess;
  ncclResult_t queryGidResult_ = ncclSuccess;
  ncclResult_t setUdmaResult_ = ncclSuccess;
  ncclResult_t listenResult_ = ncclSuccess;
  int modifyFailState_ = -1;
  int regMrFailAt_ = -1;
  int cqeCap_ = 0;
  int eceSupported_ = 1;
  int setEceSupported_ = 1;
  uint32_t queryEceOptions_ = 0x7;
  bool mlx5CreateOk_ = true;
  bool prepost_ = false;
  bool optRecv_ = false;
};

TEST_F(NetIbCastConnectMicrotest, Handshake_WiresEverySenderQpToItsReceiverPeer) {
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  ASSERT_NE(s, nullptr);
  ASSERT_NE(r, nullptr);
  ASSERT_EQ(s->base.nqps, 2);  // IB_QPS_PER_CONNECTION default, one device each side
  ASSERT_EQ(r->base.nqps, 2);
  for (int q = 0; q < 2; q++) {
    SCOPED_TRACE(q);
    const auto* sRtr = verbs_.Rtr(s->base.qps[q].qp);
    const auto* rRtr = verbs_.Rtr(r->base.qps[q].qp);
    ASSERT_NE(sRtr, nullptr);
    ASSERT_NE(rRtr, nullptr);
    EXPECT_EQ(sRtr->attr.dest_qp_num, r->base.qps[q].qp->qp_num);
    EXPECT_EQ(rRtr->attr.dest_qp_num, s->base.qps[q].qp->qp_num);
    EXPECT_NE(verbs_.Rts(s->base.qps[q].qp), nullptr);
    EXPECT_NE(verbs_.Rts(r->base.qps[q].qp), nullptr);
    EXPECT_EQ(s->base.qps[q].remDevIdx, 0);
    EXPECT_EQ(r->base.qps[q].remDevIdx, 0);
    const auto sInit = verbs_.ModifiesOf(s->base.qps[q].qp).front();
    const auto rInit = verbs_.ModifiesOf(r->base.qps[q].qp).front();
    EXPECT_EQ(sInit.attr.qp_state, IBV_QPS_INIT);
    EXPECT_EQ(sInit.attr.qp_access_flags, (unsigned)IBV_ACCESS_REMOTE_WRITE);
    EXPECT_EQ(rInit.attr.qp_access_flags,
              (unsigned)(IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_REMOTE_READ));
    EXPECT_EQ(sInit.attr.port_num, 1);
    EXPECT_EQ(verbs_.Rts(s->base.qps[q].qp)->attr.timeout, 20);
    EXPECT_EQ(verbs_.Rts(r->base.qps[q].qp)->attr.retry_cnt, 7);
  }
  EXPECT_EQ(s->base.ready, 1);
  EXPECT_EQ(r->base.ready, 1);
  EXPECT_EQ(s->base.nRemDevs, 1);
  EXPECT_EQ(r->base.nRemDevs, 1);
}

TEST_F(NetIbCastConnectMicrotest, Handshake_ExchangesFifoAndCompletionRecordAddressesAndKeys) {
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  EXPECT_EQ(r->remCtsFifo.addr, reinterpret_cast<uint64_t>(s->ctsFifo));
  EXPECT_EQ(r->remCtsFifo.rkeys[0], s->devs[0].ctsFifoMr->rkey);
  EXPECT_EQ(s->remCmplsRecords.addr, reinterpret_cast<uint64_t>(r->cmplsRecords));
  EXPECT_EQ(s->remCmplsRecords.rkeys[0], r->devs[0].cmplsRecordsMr->rkey);
  EXPECT_EQ(s->devs[0].sge.lkey, s->devs[0].cmplsRecordsMr->lkey);
  EXPECT_EQ(r->devs[0].sge.lkey, r->devs[0].ctsFifoMr->lkey);
  EXPECT_TRUE(SameGid(s->base.remDevs[0].remoteGid, IbCastDevs[0].gidInfo.localGid));
}

TEST_F(NetIbCastConnectMicrotest, Close_ReleasesEveryVerbsObjectTheHandshakeTook) {
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  EXPECT_FALSE(verbs_.live.empty());
  Forget(s);
  Forget(r);
  EXPECT_EQ(IbCastCloseSend(s), ncclSuccess);
  EXPECT_EQ(IbCastCloseRecv(r), ncclSuccess);
  EXPECT_TRUE(verbs_.live.empty());
  EXPECT_EQ(IbCastDevs[0].pdRefs, 0);
}

class NetIbCastConnectGidMicrotest : public NetIbCastConnectMicrotest {
 protected:
  void SetUp() override {
    NetIbCastConnectMicrotest::SetUp();
    portAttr_.link_layer = IBV_LINK_LAYER_ETHERNET;
    portAttr_.gid_tbl_len = 8;
    // The sysfs RoCE-version file of GID index i answers on fd 100 + i.
    g_testOpen = [this](const char* path) {
      opened_.push_back(path);
      if (openFails_) return -1;
      return 100 + atoi(strrchr(path, '/') + 1);
    };
    Hook(g_read, [this](int fd, void* buf, size_t n) -> ssize_t {
      if (readErrno_) {
        errno = readErrno_;
        return -1;
      }
      const std::string& v = versions_[fd - 100];
      size_t len = std::min(n, v.size());
      memcpy(buf, v.data(), len);
      return len;
    });
  }

  // Index 0 a link-local IPv6 GID, then the given GIDs with their RoCE versions.
  void Table(const std::vector<std::pair<union ibv_gid, const char*>>& gids) {
    gidTable_[0] = V6Gid("fe80::1");
    versions_[0] = "RoCE v2";
    for (size_t i = 0; i < gids.size(); i++) {
      gidTable_[i + 1] = gids[i].first;
      versions_[i + 1] = gids[i].second;
    }
    portAttr_.gid_tbl_len = gids.size() + 1;
  }

  int Pick() {
    int idx = -1;
    EXPECT_EQ(IbCastGetGidIndex(&ctx_[0], 1, &portAttr_, &idx), ncclSuccess);
    return idx;
  }

  ibv_port_attr portAttr_ = {};
  std::map<int, std::string> versions_;
  std::vector<std::string> opened_;
  bool openFails_ = false;
  int readErrno_ = 0;
};

TEST_F(NetIbCastConnectGidMicrotest, Ib_PicksTheRoutableIndexWhenItCarriesAFlid) {
  portAttr_.link_layer = IBV_LINK_LAYER_INFINIBAND;
  gidTable_[1] = V6Gid("fe80:0:5::1");  // FLID 5 in bytes 4..5
  EXPECT_EQ(Pick(), 1);
  gidTable_[1] = V6Gid("fe80::1");
  EXPECT_EQ(Pick(), 0);
}

TEST_F(NetIbCastConnectGidMicrotest, Ib_RoutableIndexPastTheTable_PicksZeroWithoutQuerying) {
  portAttr_.link_layer = IBV_LINK_LAYER_INFINIBAND;
  Param("IB_ROUTABLE_FLID_GID_INDEX", portAttr_.gid_tbl_len);
  EXPECT_EQ(Pick(), 0);
  EXPECT_TRUE(queriedGids_.empty());
}

TEST_F(NetIbCastConnectGidMicrotest, Ib_QueryFailure_Propagates) {
  portAttr_.link_layer = IBV_LINK_LAYER_INFINIBAND;
  queryGidResult_ = ncclSystemError;
  int idx = -1;
  EXPECT_EQ(IbCastGetGidIndex(&ctx_[0], 1, &portAttr_, &idx), ncclSystemError);
}

TEST_F(NetIbCastConnectGidMicrotest, Roce_ExplicitIndexWinsWithoutScanning) {
  Param("IB_GID_INDEX", 5);
  EXPECT_EQ(Pick(), 5);
  EXPECT_TRUE(queriedGids_.empty());
}

TEST_F(NetIbCastConnectGidMicrotest, Roce_DefaultPrefersIpv4RoceV2) {
  Table({{V6Gid("fe80::2"), "RoCE v2"}, {V4Gid("10.0.0.1"), "RoCE v1"}, {V4Gid("10.0.0.1"), "RoCE v2"}});
  EXPECT_EQ(Pick(), 3);
}

TEST_F(NetIbCastConnectGidMicrotest, Roce_VersionParamOneKeepsTheV1Gid) {
  Table({{V6Gid("fe80::2"), "RoCE v2"}, {V4Gid("10.0.0.1"), "IB/RoCE v1"}, {V4Gid("10.0.0.1"), "RoCE v2"}});
  Param("IB_ROCE_VERSION_NUM", 1);
  EXPECT_EQ(Pick(), 2);
}

TEST_F(NetIbCastConnectGidMicrotest, Roce_UnrecognisedVersionNeverWins) {
  Table({{V4Gid("10.0.0.1"), "RoCE v2"}, {V4Gid("10.0.0.2"), "IB"}});
  EXPECT_EQ(Pick(), 1);
}

TEST_F(NetIbCastConnectGidMicrotest, Roce_Inet6FamilyLeavesLinkLocalForAGlobalGid) {
  SetMicroEnv("NCCL_IB_ADDR_FAMILY", "AF_INET6");
  Table({{V4Gid("10.0.0.1"), "RoCE v2"}, {V6Gid("2001:db8::1"), "RoCE v2"}});
  EXPECT_EQ(Pick(), 2);
}

TEST_F(NetIbCastConnectGidMicrotest, Roce_UnknownFamilyNameMeansInet) {
  SetMicroEnv("NCCL_IB_ADDR_FAMILY", "AF_UNIX");
  Table({{V4Gid("10.0.0.1"), "RoCE v2"}, {V6Gid("2001:db8::1"), "RoCE v2"}});
  EXPECT_EQ(Pick(), 1);
}

TEST_F(NetIbCastConnectGidMicrotest, Roce_AddrRangeSkipsGidsOutsideTheSubnet) {
  Table({{V4Gid("10.0.0.1"), "RoCE v2"}, {V4Gid("10.0.1.1"), "RoCE v2"}});
  EXPECT_EQ(Pick(), 1);
  SetMicroEnv("NCCL_IB_ADDR_RANGE", "10.0.1.0/24");
  EXPECT_EQ(Pick(), 2);
  SetMicroEnv("NCCL_IB_ADDR_RANGE", "10.0.1.1/32");  // the widest mask still accepted
  EXPECT_EQ(Pick(), 2);
}

TEST_F(NetIbCastConnectGidMicrotest, Roce_Inet6AddrRangeMatchesWholeAndPartialWords) {
  SetMicroEnv("NCCL_IB_ADDR_FAMILY", "AF_INET6");
  Table({{V6Gid("2001:db8:0:1::1"), "RoCE v2"}, {V6Gid("2001:db8:0:2::1"), "RoCE v2"}});
  SetMicroEnv("NCCL_IB_ADDR_RANGE", "2001:db8:0:2::/64");
  EXPECT_EQ(Pick(), 2);
  SetMicroEnv("NCCL_IB_ADDR_RANGE", "2001:db8:0:2::/48");  // both share the first 48 bits
  EXPECT_EQ(Pick(), 1);
  SetMicroEnv("NCCL_IB_ADDR_RANGE", "2001:db8:0:2::/63");  // bit 63 differs
  EXPECT_EQ(Pick(), 2);
  // A /64 range ignores the interface id, however its top bits are set.
  Table({{V6Gid("2001:db8:0:1::1"), "RoCE v2"}, {V6Gid("2001:db8:0:2:c000::1"), "RoCE v2"}});
  SetMicroEnv("NCCL_IB_ADDR_RANGE", "2001:db8:0:2::/64");
  EXPECT_EQ(Pick(), 2);
}

TEST_F(NetIbCastConnectGidMicrotest, Roce_MalformedAddrRangeIsIgnored) {
  Table({{V4Gid("10.0.0.1"), "RoCE v2"}, {V4Gid("10.0.1.1"), "RoCE v2"}});
  for (const char* range : {"10.0.1.0", "bogus/24", "10.0.1.0/33", ""}) {
    SCOPED_TRACE(range);
    SetMicroEnv("NCCL_IB_ADDR_RANGE", range);
    EXPECT_EQ(Pick(), 1);
  }
  SetMicroEnv("NCCL_IB_ADDR_FAMILY", "AF_INET6");
  Table({{V6Gid("2001:db8:0:1::1"), "RoCE v2"}, {V6Gid("2001:db8:0:2::1"), "RoCE v2"}});
  SetMicroEnv("NCCL_IB_ADDR_RANGE", "2001:db8:0:2::/129");
  EXPECT_EQ(Pick(), 1);
}

TEST_F(NetIbCastConnectGidMicrotest, Roce_ReadsTheVersionFileOfEachCandidate) {
  Table({{V4Gid("10.0.0.1"), "RoCE v1"}, {V4Gid("10.0.0.1"), "RoCE v2"}});
  Pick();
  ASSERT_FALSE(opened_.empty());
  EXPECT_EQ(opened_.back(), "/sys/class/infiniband/fakedev/ports/1/gid_attrs/types/2");
}

TEST_F(NetIbCastConnectGidMicrotest, Roce_VersionFileMissing_Fails) {
  Table({{V4Gid("10.0.0.1"), "RoCE v1"}, {V4Gid("10.0.0.1"), "RoCE v2"}});
  openFails_ = true;
  int idx = -1;
  const std::string log = CaptureLog([&] { EXPECT_EQ(IbCastGetGidIndex(&ctx_[0], 1, &portAttr_, &idx), ncclSystemError); });
  EXPECT_TRUE(LogHas(log, "open failed in IbCastRoceGetVersionNum")) << log;
}

TEST_F(NetIbCastConnectGidMicrotest, Roce_VersionReadError_FailsUnlessEinval) {
  Table({{V4Gid("10.0.0.1"), "RoCE v1"}, {V4Gid("10.0.0.1"), "RoCE v2"}});
  readErrno_ = EIO;
  int idx = -1;
  const std::string log = CaptureLog([&] { EXPECT_EQ(IbCastGetGidIndex(&ctx_[0], 1, &portAttr_, &idx), ncclSystemError); });
  EXPECT_TRUE(LogHas(log, "read failed in IbCastRoceGetVersionNum")) << log;
  readErrno_ = EINVAL;  // an unmapped GID index inside a container
  EXPECT_EQ(IbCastGetGidIndex(&ctx_[0], 1, &portAttr_, &idx), ncclSuccess);
  EXPECT_EQ(g_closedFds.size(), opened_.size());  // every opened file is closed, read error or not
}

TEST_F(NetIbCastConnectGidMicrotest, GidInfoQuery_FillsLinkLayerIndexAndGid) {
  Param("IB_GID_INDEX", 4);
  gidTable_[4] = V4Gid("10.9.9.9");
  ncclIbGidInfo info = {};
  EXPECT_EQ(IbCastGidInfoQuery(&ctx_[0], 1, &portAttr_, &info), ncclSuccess);
  EXPECT_EQ(info.link_layer, IBV_LINK_LAYER_ETHERNET);
  EXPECT_EQ(info.localGidIndex, 4);
  EXPECT_TRUE(SameGid(info.localGid, V4Gid("10.9.9.9")));
}

TEST_F(NetIbCastConnectGidMicrotest, GidInfoQuery_RejectsNullArgumentsAndANegativeIndex) {
  ncclIbGidInfo info = {};
  EXPECT_EQ(IbCastGidInfoQuery(nullptr, 1, &portAttr_, &info), ncclInternalError);
  EXPECT_EQ(IbCastGidInfoQuery(&ctx_[0], 1, nullptr, &info), ncclInternalError);
  EXPECT_EQ(IbCastGidInfoQuery(&ctx_[0], 1, &portAttr_, nullptr), ncclInternalError);
  portAttr_.link_layer = IBV_LINK_LAYER_INFINIBAND;
  Param("IB_ROUTABLE_FLID_GID_INDEX", -1);
  gidTable_[-1] = V6Gid("fe80:0:5::1");
  EXPECT_EQ(IbCastGidInfoQuery(&ctx_[0], 1, &portAttr_, &info), ncclInternalError);
}

TEST_F(NetIbCastConnectGidMicrotest, SameSubnet_Ipv4UsesThePrefixLengthIpv6TheTop64Bits) {
  auto same = [](union ibv_gid a, union ibv_gid b, int len) { return ncclIbCastTestGidSameSubnet(a.raw, b.raw, len); };
  EXPECT_EQ(same(V4Gid("10.0.0.1"), V4Gid("10.0.0.200"), 24), 1);
  EXPECT_EQ(same(V4Gid("10.0.0.1"), V4Gid("10.0.1.1"), 24), 0);
  EXPECT_EQ(same(V4Gid("10.0.0.1"), V4Gid("10.0.1.1"), 23), 1);
  EXPECT_EQ(same(V4Gid("10.0.0.1"), V4Gid("10.0.0.2"), 32), 0);
  EXPECT_EQ(same(V4Gid("10.0.0.1"), V6Gid("2001:db8::1"), 24), 0);
  EXPECT_EQ(same(V6Gid("2001:db8::1"), V6Gid("2001:db8::2"), 24), 1);
  EXPECT_EQ(same(V6Gid("2001:db8::1"), V6Gid("2001:db9::1"), 24), 0);
}

TEST_F(NetIbCastConnectGidMicrotest, SubnetMatchesAny_SkipsInvalidRemotesAndBoundsTheCount) {
  union ibv_gid local = V4Gid("10.0.0.1");
  union ibv_gid remotes[3] = {V6Gid("fe80::1"), {}, V4Gid("10.0.0.9")};
  EXPECT_EQ(ncclIbCastTestSubnetMatchesAny(local.raw, remotes[0].raw, 3, 24), 1);
  EXPECT_EQ(ncclIbCastTestSubnetMatchesAny(local.raw, remotes[0].raw, 2, 24), 0);
  union ibv_gid linkLocalV4 = V6Gid("fe80::");  // configured-but-link-local rejects too
  EXPECT_EQ(ncclIbCastTestSubnetMatchesAny(local.raw, linkLocalV4.raw, 1, 24), 0);
  EXPECT_EQ(ncclIbCastTestSubnetMatchesAny(local.raw, remotes[0].raw, -1, 24), 0);
  EXPECT_EQ(ncclIbCastTestSubnetMatchesAny(local.raw, remotes[0].raw, NCCL_IB_MAX_DEVS_PER_NIC + 1, 24), 0);
}

class NetIbCastConnectQpMicrotest : public NetIbCastConnectMicrotest {
 protected:
  ncclIbQp Qp(ibv_qp_type type = IBV_QPT_RC) {
    ncclIbQp qp = {};
    qp.qp = verbs_.NewQp(nullptr, type);
    return qp;
  }
  ncclIbQpCreateAttr Attr(bool ionic = false) {
    ncclIbQpCreateAttr a = {};
    a.type = IBV_QPT_RC;
    a.qpContext = &ctx_[0];
    a.cq = &cq_;
    a.pd = &pd_;
    a.maxRecvWorkRequest = 11;
    a.maxSendWorkRequest = 13;
    a.useIonic = ionic;
    a.ctsQpSlot = 4;
    IbCastQpCreateAttrInitSharing(&a);
    return a;
  }
  ibv_cq cq_ = {};
  ibv_pd pd_ = {&ctx_[0]};
};

TEST_F(NetIbCastConnectQpMicrotest, Init_MovesToInitWithPortPkeyAndAccess) {
  ncclIbQp qp = Qp();
  qp.initAttr = {IBV_QPS_INIT, 3, 2, IBV_ACCESS_REMOTE_WRITE};
  ASSERT_EQ(IbCastQpInit(&qp), ncclSuccess);
  const auto& m = verbs_.modifies.back();
  EXPECT_EQ(m.attr.qp_state, IBV_QPS_INIT);
  EXPECT_EQ(m.attr.pkey_index, 3);
  EXPECT_EQ(m.attr.port_num, 2);
  EXPECT_EQ(m.attr.qp_access_flags, (unsigned)IBV_ACCESS_REMOTE_WRITE);
  EXPECT_EQ(m.mask, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
}

TEST_F(NetIbCastConnectQpMicrotest, Rts_RcSetsRetryAndTimeoutUcOnlyTheState) {
  ncclIbQp rc = Qp();
  rc.rtsAttr = {14, 6};
  ASSERT_EQ(IbCastQpRts(&rc), ncclSuccess);
  const auto rcm = verbs_.modifies.back();
  EXPECT_EQ(rcm.attr.qp_state, IBV_QPS_RTS);
  EXPECT_EQ(rcm.attr.timeout, 14);
  EXPECT_EQ(rcm.attr.retry_cnt, 6);
  EXPECT_EQ(rcm.attr.rnr_retry, 7);
  EXPECT_EQ(rcm.attr.max_rd_atomic, 1);
  EXPECT_EQ(rcm.mask, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                          IBV_QP_MAX_QP_RD_ATOMIC);
  ncclIbQp uc = Qp(IBV_QPT_UC);
  ASSERT_EQ(IbCastQpRts(&uc), ncclSuccess);
  EXPECT_EQ(verbs_.modifies.back().mask, IBV_QP_STATE | IBV_QP_SQ_PSN);
}

TEST_F(NetIbCastConnectQpMicrotest, ResetAndError_MoveOnlyTheState) {
  ncclIbQp qp = Qp();
  ASSERT_EQ(IbCastQpReset(&qp), ncclSuccess);
  EXPECT_EQ(verbs_.modifies.back().attr.qp_state, IBV_QPS_RESET);
  EXPECT_EQ(verbs_.modifies.back().mask, IBV_QP_STATE);
  ASSERT_EQ(IbCastQpError(&qp), ncclSuccess);
  EXPECT_EQ(verbs_.modifies.back().attr.qp_state, IBV_QPS_ERR);
  EXPECT_EQ(verbs_.modifies.back().mask, IBV_QP_STATE);
  modifyFailState_ = IBV_QPS_ERR;
  EXPECT_EQ(IbCastQpError(&qp), ncclSystemError);
  modifyFailState_ = IBV_QPS_RESET;
  EXPECT_EQ(IbCastQpReset(&qp), ncclSystemError);
  modifyFailState_ = IBV_QPS_RTS;
  EXPECT_EQ(IbCastQpRts(&qp), ncclSystemError);
  modifyFailState_ = IBV_QPS_INIT;
  qp.initAttr.state = IBV_QPS_INIT;
  EXPECT_EQ(IbCastQpInit(&qp), ncclSystemError);
}

class NetIbCastConnectRtrMicrotest : public NetIbCastConnectQpMicrotest {
 protected:
  ncclIbQp RtrQp(uint8_t linkLayer, ibv_qp_type type = IBV_QPT_RC) {
    ncclIbQp qp = Qp(type);
    ncclIbQpRtrAttr& a = qp.rtrAttr;
    a.mtu = IBV_MTU_2048;
    a.linkLayer = linkLayer;
    a.tc = 104;
    a.sl = 3;
    a.remoteQpNum = 0x77;
    a.remoteLid = 0x21;
    a.remoteGid = V4Gid("10.0.0.9");
    a.localIbPort = 1;
    a.localGid = V4Gid("10.0.0.1");
    a.localGidIndex = kGidIndex;
    return qp;
  }
  const FakeVerbs::Modify& Last() { return verbs_.modifies.back(); }
};

TEST_F(NetIbCastConnectRtrMicrotest, Ethernet_AddressesTheRemoteGidThroughTheGrh) {
  ncclIbQp qp = RtrQp(IBV_LINK_LAYER_ETHERNET);
  ASSERT_EQ(IbCastQpRtr(&qp), ncclSuccess);
  const auto& m = Last();
  EXPECT_EQ(m.attr.qp_state, IBV_QPS_RTR);
  EXPECT_EQ(m.attr.path_mtu, IBV_MTU_2048);
  EXPECT_EQ(m.attr.dest_qp_num, 0x77u);
  EXPECT_EQ(m.attr.max_dest_rd_atomic, 1);
  EXPECT_EQ(m.attr.min_rnr_timer, 12);
  EXPECT_EQ(m.attr.ah_attr.is_global, 1);
  EXPECT_TRUE(SameGid(m.attr.ah_attr.grh.dgid, V4Gid("10.0.0.9")));
  EXPECT_EQ(m.attr.ah_attr.grh.sgid_index, kGidIndex);
  EXPECT_EQ(m.attr.ah_attr.grh.hop_limit, 255);
  EXPECT_EQ(m.attr.ah_attr.grh.traffic_class, 104);
  EXPECT_EQ(m.attr.ah_attr.sl, 3);
  EXPECT_EQ(m.attr.ah_attr.port_num, 1);
  EXPECT_EQ(m.mask, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

TEST_F(NetIbCastConnectRtrMicrotest, Uc_LeavesOutTheReadAtomicFields) {
  ncclIbQp qp = RtrQp(IBV_LINK_LAYER_ETHERNET, IBV_QPT_UC);
  ASSERT_EQ(IbCastQpRtr(&qp), ncclSuccess);
  EXPECT_EQ(Last().mask, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN);
  EXPECT_EQ(Last().attr.min_rnr_timer, 0);
}

TEST_F(NetIbCastConnectRtrMicrotest, Ib_SameSubnet_RoutesOnTheLidWithoutGrh) {
  ncclIbQp qp = RtrQp(IBV_LINK_LAYER_INFINIBAND);
  qp.rtrAttr.localGid = V6Gid("fe80::1");
  qp.rtrAttr.remoteGid = V6Gid("fe80:0:0:0:1::1");  // subnet prefixes differ only above the low 16 bits
  ASSERT_EQ(IbCastQpRtr(&qp), ncclSuccess);
  EXPECT_EQ(Last().attr.ah_attr.is_global, 0);
  EXPECT_EQ(Last().attr.ah_attr.dlid, 0x21);
}

TEST_F(NetIbCastConnectRtrMicrotest, Ib_GrhRequired_GoesGlobalOnTheLid) {
  ncclIbQp qp = RtrQp(IBV_LINK_LAYER_INFINIBAND);
  qp.rtrAttr.localGid = V6Gid("fe80::1");
  qp.rtrAttr.remoteGid = V6Gid("fe80::2");
  qp.rtrAttr.localPortFlags = IBV_QPF_GRH_REQUIRED;
  ASSERT_EQ(IbCastQpRtr(&qp), ncclSuccess);
  EXPECT_EQ(Last().attr.ah_attr.is_global, 1);
  EXPECT_EQ(Last().attr.ah_attr.dlid, 0x21);
  EXPECT_TRUE(SameGid(Last().attr.ah_attr.grh.dgid, V6Gid("fe80::2")));
  EXPECT_EQ(Last().attr.ah_attr.grh.hop_limit, 255);
  EXPECT_EQ(Last().attr.ah_attr.grh.sgid_index, kGidIndex);
}

TEST_F(NetIbCastConnectRtrMicrotest, Ib_OtherSubnet_RoutesToTheRemoteFlid) {
  ncclIbQp qp = RtrQp(IBV_LINK_LAYER_INFINIBAND);
  qp.rtrAttr.localGid = V6Gid("fe80:0:0:1::1");
  qp.rtrAttr.remoteGid = V6Gid("fe80:0:9:2::1");  // FLID 9, subnet 2
  ASSERT_EQ(IbCastQpRtr(&qp), ncclSuccess);
  EXPECT_EQ(Last().attr.ah_attr.is_global, 1);
  EXPECT_EQ(Last().attr.ah_attr.dlid, 9);
}

TEST_F(NetIbCastConnectRtrMicrotest, Ib_OtherSubnetWithoutFlid_WarnsAndFallsBackToTheLid) {
  ncclIbQp qp = RtrQp(IBV_LINK_LAYER_INFINIBAND);
  qp.rtrAttr.localGid = V6Gid("fe80:0:0:1::1");
  qp.rtrAttr.remoteGid = V6Gid("fe80:0:0:2::1");
  const std::string log = CaptureLog([&] { ASSERT_EQ(IbCastQpRtr(&qp), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, "remote FLID configured as zero")) << log;
  EXPECT_EQ(Last().attr.ah_attr.dlid, 0x21);
  EXPECT_EQ(Last().attr.ah_attr.is_global, 1);
}

TEST_F(NetIbCastConnectRtrMicrotest, ModifyFailure_Propagates) {
  ncclIbQp qp = RtrQp(IBV_LINK_LAYER_ETHERNET);
  modifyFailState_ = IBV_QPS_RTR;
  EXPECT_EQ(IbCastQpRtr(&qp), ncclSystemError);
}

class NetIbCastConnectMultiplaneMicrotest : public NetIbCastConnectRtrMicrotest {
 protected:
  void SetUp() override {
    NetIbCastConnectRtrMicrotest::SetUp();
    IbCastMultiplaneEnable = true;
    Hook(g_ibCastMultiplaneLoad, [] { return ncclSuccess; });
    Hook(g_ibCastMultiplaneGetPipGids, [this](const union ibv_gid* vip, union ibv_gid* pips, int* n) {
      const auto& list = SameGid(*vip, V4Gid("10.0.0.1")) ? localPips_ : remotePips_;
      *n = list.size();
      for (size_t i = 0; i < list.size(); i++) pips[i] = list[i];
      return ncclSuccess;
    });
    Hook(g_ionicdvQpSetPuecPlaneRoute, [this](ibv_qp*, uint8_t plane, ionic_dv_puec_route* route) {
      routes_.push_back({plane, *route});
      return routeFailAt_ == static_cast<int>(routes_.size()) - 1 ? ncclSystemError : ncclSuccess;
    });
  }
  std::vector<union ibv_gid> localPips_ = {V4Gid("192.168.0.1"), V4Gid("192.168.1.1")};
  std::vector<union ibv_gid> remotePips_ = {V4Gid("192.168.0.9"), V4Gid("192.168.1.9")};
  std::vector<std::pair<int, ionic_dv_puec_route>> routes_;
  int routeFailAt_ = -1;
};

TEST_F(NetIbCastConnectMultiplaneMicrotest, ProgramsOneRoutePerPlaneFromLocalToRemotePip) {
  ncclIbQp qp = RtrQp(IBV_LINK_LAYER_ETHERNET);
  ASSERT_EQ(IbCastQpRtr(&qp), ncclSuccess);
  ASSERT_EQ(routes_.size(), 2u);
  for (int i = 0; i < 2; i++) {
    SCOPED_TRACE(i);
    EXPECT_EQ(routes_[i].first, i);
    EXPECT_TRUE(SameGid(routes_[i].second.dgid, remotePips_[i]));
    EXPECT_TRUE(SameGid(routes_[i].second.sgid, localPips_[i]));
    EXPECT_EQ(routes_[i].second.hop_limit, 255);
    EXPECT_EQ(routes_[i].second.sl, 3);
    EXPECT_EQ(routes_[i].second.traffic_class, 104);
    EXPECT_EQ(routes_[i].second.flags, 0u);
  }
  EXPECT_EQ(Last().attr.qp_state, IBV_QPS_RTR);
}

TEST_F(NetIbCastConnectMultiplaneMicrotest, LoopbackQpAndDisabledMultiplane_ProgramNothing) {
  ncclIbQp qp = RtrQp(IBV_LINK_LAYER_ETHERNET);
  qp.rtrAttr.remoteGid = qp.rtrAttr.localGid;
  ASSERT_EQ(IbCastQpRtr(&qp), ncclSuccess);
  IbCastMultiplaneEnable = false;
  qp.rtrAttr.remoteGid = V4Gid("10.0.0.9");
  ASSERT_EQ(IbCastQpRtr(&qp), ncclSuccess);
  EXPECT_TRUE(routes_.empty());
}

TEST_F(NetIbCastConnectMultiplaneMicrotest, FewerLocalPips_WarnsAndProgramsOnlyThoseAndNoneWithoutLocal) {
  ncclIbQp qp = RtrQp(IBV_LINK_LAYER_ETHERNET);
  localPips_.resize(1);
  std::string log = CaptureLog([&] { ASSERT_EQ(IbCastQpRtr(&qp), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, "local VIP maps to 1 PIPs but remote maps to 2")) << log;
  EXPECT_EQ(routes_.size(), 1u);
  localPips_.clear();
  log = CaptureLog([&] { ASSERT_EQ(IbCastQpRtr(&qp), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, "local VIP has no PIP mapping")) << log;
  EXPECT_EQ(routes_.size(), 1u);
  remotePips_.clear();
  log = CaptureLog([&] { ASSERT_EQ(IbCastQpRtr(&qp), ncclSuccess); });
  EXPECT_FALSE(LogHas(log, "Multiplane")) << log;
}

TEST_F(NetIbCastConnectMultiplaneMicrotest, RouteFailure_WarnsAndStopsButStillMovesToRtr) {
  ncclIbQp qp = RtrQp(IBV_LINK_LAYER_ETHERNET);
  routeFailAt_ = 0;
  const std::string log = CaptureLog([&] { ASSERT_EQ(IbCastQpRtr(&qp), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, "PUEC route programming failed for plane 0")) << log;
  EXPECT_EQ(routes_.size(), 1u);
  EXPECT_EQ(Last().attr.qp_state, IBV_QPS_RTR);
}

TEST_F(NetIbCastConnectMultiplaneMicrotest, MapLoadOrLookupFailure_FailsTheTransition) {
  ncclIbQp qp = RtrQp(IBV_LINK_LAYER_ETHERNET);
  {
    ScopedHook load(g_ibCastMultiplaneLoad, [] { return ncclSystemError; });
    EXPECT_EQ(IbCastQpRtr(&qp), ncclSystemError);
  }
  ScopedHook pips(g_ibCastMultiplaneGetPipGids,
                  [](const union ibv_gid*, union ibv_gid*, int*) { return ncclInvalidArgument; });
  EXPECT_EQ(IbCastQpRtr(&qp), ncclInvalidArgument);
  EXPECT_TRUE(verbs_.modifies.empty());
}

TEST_F(NetIbCastConnectQpMicrotest, Create_GenericRcQpTakesItsQueuesFromTheAttributes) {
  ncclIbQpCreateAttr a = Attr();
  ncclIbQp qp = {};
  qp.telQpStats = reinterpret_cast<RcclQpStats*>(0x1);
  ASSERT_EQ(IbCastQpCreate(&qp, &a), ncclSuccess);
  ASSERT_NE(qp.qp, nullptr);
  EXPECT_EQ(qp.telQpStats, nullptr);
  const auto* c = verbs_.CreateOf(qp.qp);
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(c->pd, &pd_);
  EXPECT_EQ(c->attr.send_cq, &cq_);
  EXPECT_EQ(c->attr.recv_cq, &cq_);
  EXPECT_EQ(c->attr.qp_context, &ctx_[0]);
  EXPECT_EQ(c->attr.qp_type, IBV_QPT_RC);
  EXPECT_EQ(c->attr.cap.max_recv_wr, 11u);
  EXPECT_EQ(c->attr.cap.max_send_wr, 13u);
  EXPECT_EQ(c->attr.cap.max_send_sge, 1u);
  EXPECT_EQ(c->attr.cap.max_recv_sge, 1u);
  EXPECT_EQ(c->attr.cap.max_inline_data, 0u);
  IbCastUseInline = true;
  ASSERT_EQ(IbCastQpCreate(&qp, &a), ncclSuccess);
  EXPECT_EQ(verbs_.creates.back().attr.cap.max_inline_data, sizeof(ncclIbSendFifo) * NCCL_NET_IB_MAX_RECVS);
}

TEST_F(NetIbCastConnectQpMicrotest, Create_GenericFailure_Propagates) {
  ncclIbQpCreateAttr a = Attr();
  ncclIbQp qp = {};
  createQpResult_ = ncclSystemError;
  EXPECT_EQ(IbCastQpCreate(&qp, &a), ncclSystemError);
  a.useIonic = true;
  EXPECT_EQ(IbCastQpCreate(&qp, &a), ncclSystemError);
}

TEST_F(NetIbCastConnectQpMicrotest, Create_OooRq_UsesMlx5WithTheOutOfOrderFlag) {
  ncclIbQpCreateAttr a = Attr(/*ionic=*/true);
  a.oooRq = true;
  ncclIbQp qp = {};
  ASSERT_EQ(IbCastQpCreate(&qp, &a), ncclSuccess);
  ASSERT_EQ(mlx5Attrs_.size(), 1u);
  EXPECT_TRUE(verbs_.creates.empty());
  EXPECT_EQ(mlx5Attrs_[0].pd, &pd_);
  EXPECT_EQ(mlx5Attrs_[0].comp_mask, (uint32_t)IBV_QP_INIT_ATTR_PD);
  EXPECT_EQ(mlx5Attrs_[0].send_cq, &cq_);
  EXPECT_EQ(mlx5Attrs_[0].cap.max_recv_wr, 11u);
  EXPECT_EQ(mlx5Attrs_[0].cap.max_send_wr, 13u);
  EXPECT_EQ(mlx5Attrs_[0].cap.max_inline_data, 0u);
  EXPECT_TRUE(mlx5DvAttrs_[0].create_flags & MLX5DV_QP_CREATE_OOO_DP);
  EXPECT_TRUE(mlx5DvAttrs_[0].comp_mask & MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS);
  IbCastUseInline = true;
  a.oooRq = false;
  ASSERT_EQ(ncclIbCreateQpMlx5(&a, &qp), ncclSuccess);
  EXPECT_EQ(mlx5DvAttrs_.back().create_flags, 0u);
  EXPECT_EQ(mlx5Attrs_.back().cap.max_inline_data, sizeof(ncclIbSendFifo) * NCCL_NET_IB_MAX_RECVS);
}

TEST_F(NetIbCastConnectQpMicrotest, Create_Mlx5Failure_WarnsAndFails) {
  ncclIbQpCreateAttr a = Attr();
  a.oooRq = true;
  mlx5CreateOk_ = false;
  ncclIbQp qp = {};
  const std::string log = CaptureLog([&] { EXPECT_EQ(IbCastQpCreate(&qp, &a), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "mlx5dv_create_qp failed")) << log;
}

TEST_F(NetIbCastConnectQpMicrotest, Create_IonicSetsFeatureBitsAndCtsSlot) {
  ncclIbQpCreateAttr a = Attr(/*ionic=*/true);
  a.isDataQp = true;
  a.isCtsEnabled = true;
  a.isP2p = 1;
  ncclIbQp qp = {};
  ASSERT_EQ(IbCastQpCreate(&qp, &a), ncclSuccess);
  const auto& c = verbs_.creates.back().attr;
  EXPECT_EQ(c.sq_sig_all, (1 << 16) | (1 << 17) | (1 << 18) | (1 << 19) | (1 << 25));
  EXPECT_EQ(c.cap.max_inline_data, (uint32_t)MAX_INLINE_DATA_SIZE);
  EXPECT_EQ(c.cap.max_recv_wr, 11u);
  EXPECT_EQ(qp.ctsQpSlot, 4);

  a.isDataQp = false;
  a.isCtsEnabled = false;
  a.isP2p = 0;
  ASSERT_EQ(IbCastQpCreate(&qp, &a), ncclSuccess);
  EXPECT_EQ(verbs_.creates.back().attr.sq_sig_all, (1 << 16) | (1 << 18));
  EXPECT_EQ(verbs_.creates.back().attr.cap.max_inline_data, 0u);
  IbCastAinicCtsInlineData = true;
  ASSERT_EQ(IbCastQpCreate(&qp, &a), ncclSuccess);
  EXPECT_EQ(verbs_.creates.back().attr.cap.max_inline_data, sizeof(ncclIbSendFifoCtsInline));
}

TEST_F(NetIbCastConnectQpMicrotest, Create_IonicScalesQueueDepthsOnlyForSharedQps) {
  const int mult = rcclParamIbCastQpDepthMultiplier();
  ASSERT_GT(mult, 1);
  ncclIbQpCreateAttr a = Attr(/*ionic=*/true);
  a.cqDepthMultiplier = mult;
  ncclIbQp qp = {};
  ASSERT_EQ(IbCastQpCreate(&qp, &a), ncclSuccess);
  EXPECT_EQ(verbs_.creates.back().attr.cap.max_recv_wr, 11u);
  a.isQpSharingEnabled = true;
  ASSERT_EQ(IbCastQpCreate(&qp, &a), ncclSuccess);
  EXPECT_EQ(verbs_.creates.back().attr.cap.max_recv_wr, 11u * mult);
  EXPECT_EQ(verbs_.creates.back().attr.cap.max_send_wr, 13u * mult);
}

TEST_F(NetIbCastConnectQpMicrotest, Create_IonicUdQpTakesTheGenericPath) {
  ncclIbQpCreateAttr a = Attr(/*ionic=*/true);
  a.type = IBV_QPT_UD;
  ncclIbQp qp = {};
  ASSERT_EQ(IbCastQpCreate(&qp, &a), ncclSuccess);
  EXPECT_TRUE(udmaMasks_.empty());
  EXPECT_EQ(verbs_.creates.back().attr.sq_sig_all, 0);
}

TEST_F(NetIbCastConnectQpMicrotest, Create_IonicGdaOrUdmaFailureStillCreatesTheQp) {
  ncclIbQpCreateAttr a = Attr(/*ionic=*/true);
  setUdmaResult_ = ncclSystemError;
  ScopedHook gda(g_ionicdvQpSetGda, [](ibv_qp*, bool send, bool recv) {
    EXPECT_FALSE(send);
    EXPECT_TRUE(recv);
    return ncclSystemError;
  });
  ncclIbQp qp = {};
  ASSERT_EQ(IbCastQpCreate(&qp, &a), ncclSuccess);
  EXPECT_NE(qp.qp, nullptr);
  EXPECT_EQ(gda.calls, 1);
  EXPECT_TRUE(nccl_udma_mask_warned[0]);
  EXPECT_FALSE(nccl_udma_mask_warned[1]);
  ASSERT_EQ(IbCastQpCreate(&qp, &a), ncclSuccess);
  EXPECT_TRUE(nccl_udma_mask_warned[0]);
}

class NetIbCastConnectUdmaMicrotest : public NetIbCastConnectQpMicrotest {
 protected:
  // The uDMA mask the ionic path programs for a QP, or 0 when it leaves the PD alone.
  int Mask(int channelId, bool data, int qpIdx = 0, int isP2p = 0, int group = -1, int ibDevN = 0) {
    ncclIbQpCreateAttr a = Attr(/*ionic=*/true);
    a.channelId = channelId;
    a.isDataQp = data;
    a.qpIdx = qpIdx;
    a.isP2p = isP2p;
    a.ibDevN = ibDevN;
    if (group >= 0) {
      a.isQpSharingEnabled = true;
      a.qpSharingGroupIdx = group;
    }
    size_t before = udmaMasks_.size();
    ncclIbQp qp = {};
    EXPECT_EQ(IbCastQpCreate(&qp, &a), ncclSuccess);
    return udmaMasks_.size() == before ? 0 : udmaMasks_.back();
  }
  static constexpr int kLow = IONIC_UDMA_MASK_LOW;
  static constexpr int kHigh = IONIC_UDMA_MASK_HIGH;
};

TEST_F(NetIbCastConnectUdmaMicrotest, ChannelAffinity_AlternatesNewChannelsAndSticksPerChannel) {
  EXPECT_EQ(Mask(5, true), kLow);
  EXPECT_EQ(Mask(9, true), kHigh);
  EXPECT_EQ(Mask(5, true, 1), kLow);
  EXPECT_EQ(Mask(9, true, 1), kHigh);
  EXPECT_EQ(Mask(7, true), kLow);
  // CTS QPs and other devices keep their own rotation.
  EXPECT_EQ(Mask(9, false), kLow);
  EXPECT_EQ(Mask(9, true, 0, 0, -1, 1), kLow);
}

TEST_F(NetIbCastConnectUdmaMicrotest, DriverDefault_LeavesThePdMaskAlone) {
  Param("RCCL_IB_UDMA_POLICY", static_cast<int>(ncclIbCastUDMAPolicy::DriverDefault));
  EXPECT_EQ(Mask(5, true), 0);
}

TEST_F(NetIbCastConnectUdmaMicrotest, OutOfRangePolicy_MeansChannelAffinity) {
  for (int64_t policy : {-1, static_cast<int>(ncclIbCastUDMAPolicy::Last) + 1}) {
    SCOPED_TRACE(policy);
    Param("RCCL_IB_UDMA_POLICY", policy);
    memset(nccl_channel_ud_map, 0, sizeof(nccl_channel_ud_map));
    memset(nccl_channel_last_ud, 0, sizeof(nccl_channel_last_ud));
    EXPECT_EQ(Mask(5, true), kLow);
    EXPECT_EQ(Mask(9, true), kHigh);
    EXPECT_EQ(Mask(5, true), kLow);
  }
}

TEST_F(NetIbCastConnectUdmaMicrotest, SplitConnection_ChannelParityPlusQpIndex) {
  Param("RCCL_IB_UDMA_POLICY", static_cast<int>(ncclIbCastUDMAPolicy::SplitConnection));
  EXPECT_EQ(Mask(0b11, true, 0), kLow);   // even parity
  EXPECT_EQ(Mask(0b11, true, 1), kHigh);
  EXPECT_EQ(Mask(0b111, true, 0), kHigh);  // odd parity
  EXPECT_EQ(Mask(0b111, true, 3), kLow);
}

TEST_F(NetIbCastConnectUdmaMicrotest, P2pRoundRobin_AlternatesP2pQpsPerDevice) {
  Param("RCCL_IB_UDMA_POLICY", static_cast<int>(ncclIbCastUDMAPolicy::P2pRoundRobin));
  EXPECT_EQ(Mask(5, true, 0, 1), kLow);
  EXPECT_EQ(Mask(5, true, 0, 1), kHigh);
  EXPECT_EQ(Mask(5, true, 0, 1, -1, 1), kLow);
  EXPECT_EQ(Mask(5, true, 0, 1), kLow);
  // Collective QPs keep channel affinity.
  EXPECT_EQ(Mask(6, true), kLow);
  EXPECT_EQ(Mask(8, true), kHigh);
}

TEST_F(NetIbCastConnectUdmaMicrotest, SharedQps_FollowTheGroupOrTheQpIndexWithOneGroup) {
  EXPECT_EQ(Mask(5, true, 0, 0, 2), kLow);
  EXPECT_EQ(Mask(5, true, 1, 0, 3), kHigh);
  Param("RCCL_IB_COMM_NGROUPS", 1);
  EXPECT_EQ(Mask(5, true, 1, 0, 0), kHigh);
  EXPECT_EQ(Mask(5, true, 2, 0, 0), kLow);
}

TEST_F(NetIbCastConnectUdmaMicrotest, Ainic_EveryPolicyCyclesThroughTheDevicesEngines) {
  Ainic(4);
  IbCastDevs[1].capsProvider.ionic.udmaCount = 3;
  for (int ch : {1, 2, 3, 4, 5}) EXPECT_EQ(Mask(ch, true), 1 << ((ch - 1) % 4)) << ch;
  EXPECT_EQ(Mask(5, true, 0, 0, -1, 1), 1 << 0);
  EXPECT_EQ(Mask(6, true, 0, 0, -1, 1), 1 << 1);
  EXPECT_EQ(Mask(7, true, 0, 0, -1, 1), 1 << 2);
  EXPECT_EQ(Mask(8, true, 0, 0, -1, 1), 1 << 0);
  EXPECT_EQ(Mask(5, true, 0, 0, 3), 1 << 3);
  Param("RCCL_IB_COMM_NGROUPS", 1);
  EXPECT_EQ(Mask(5, true, 6, 0, 0), 1 << 2);
  Param("RCCL_IB_UDMA_POLICY", static_cast<int>(ncclIbCastUDMAPolicy::SplitConnection));
  EXPECT_EQ(Mask(0b111, true, 2), 1 << 3);  // odd parity + 2
  EXPECT_EQ(Mask(0b111, true, 3), 1 << 0);
  Param("RCCL_IB_UDMA_POLICY", static_cast<int>(ncclIbCastUDMAPolicy::P2pRoundRobin));
  for (int i = 0; i < 5; i++) EXPECT_EQ(Mask(5, true, 0, 1), 1 << (i % 4)) << i;
}

TEST_F(NetIbCastConnectUdmaMicrotest, NotAinic_KeepsTwoEnginesWhateverTheDeviceHolds) {
  IbCastDevs[0].capsProvider.ionic.udmaCount = 4;
  EXPECT_EQ(Mask(1, true), kLow);
  EXPECT_EQ(Mask(2, true), kHigh);
  EXPECT_EQ(Mask(3, true), kLow);
}

TEST_F(NetIbCastConnectUdmaMicrotest, Inspect_ReportsTheCountRoundRobinAndPerQpEngines) {
  Ainic(4);
  EXPECT_EQ(ncclIbCastTestGetDevUdmaCount(0), 4);
  EXPECT_EQ(ncclIbCastTestGetDevUdmaCount(-1), 0);
  IbCastNDevs = 1;
  EXPECT_EQ(ncclIbCastTestGetDevUdmaCount(1), 0);
  IbCastNDevs = kNDevs;

  uint8_t masks[5];
  ASSERT_EQ(ncclIbCastTestUdmaRoundRobin(3, 5, masks), ncclSuccess);
  EXPECT_EQ(std::vector<uint8_t>(masks, masks + 5), (std::vector<uint8_t>{1, 2, 4, 1, 2}));
  EXPECT_EQ(ncclIbCastTestUdmaRoundRobin(3, 5, nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastTestUdmaRoundRobin(0, 5, masks), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastTestUdmaRoundRobin(3, 0, masks), ncclInvalidArgument);

  auto* s = NewComm<ncclIbSendComm>();
  s->base.isSend = true;
  s->devs[0].base.ibDevN = 1;
  IbCastDevs[1].capsProvider.ionic.udmaCount = 3;
  ncclIbQp a = Qp(), b = Qp(), noQp = {};
  s->base.activeQps[0] = &a;
  s->base.activeQps[1] = &noQp;
  s->base.activeQps[2] = &b;
  s->base.nqps = 3;
  // b's driver cannot answer.
  ScopedHook idx(g_ionicdvQpGetUdmaIdx, [&](ibv_qp* qp) -> uint8_t { return qp == a.qp ? 2 : 0xFF; });
  ncclIbCastUdmaState st;
  ASSERT_EQ(ncclIbCastGetUdmaState(s, &st), ncclSuccess);
  EXPECT_EQ(st.nqps, 3);
  EXPECT_EQ(st.udmaCount, 3);
  EXPECT_EQ(st.udmaIdx[0], 2);
  EXPECT_EQ(std::vector<bool>(st.queryOk, st.queryOk + 3), (std::vector<bool>{true, false, false}));
  EXPECT_EQ(idx.calls, 2);
  EXPECT_EQ(ncclIbCastGetUdmaState(nullptr, &st), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGetUdmaState(s, nullptr), ncclInvalidArgument);
  free(s);
}

TEST_F(NetIbCastConnectQpMicrotest, BuildDataQpCreateAttr_SizesSendAndReceiveQueues) {
  auto* s = NewComm<ncclIbSendComm>();
  s->base.isSend = true;
  s->base.isP2p = 1;
  s->devs[1].base = {2, &pd_, &cq_};
  Ainic();
  ncclIbQpCreateAttr a;
  memset(&a, 0x5a, sizeof(a));
  IbCastBuildDataQpCreateAttr(&s->base, 1, &a);
  EXPECT_EQ(a.type, IBV_QPT_RC);
  EXPECT_EQ(a.qpContext, &s->base.stats);
  EXPECT_EQ(a.cq, &cq_);
  EXPECT_EQ(a.pd, &pd_);
  EXPECT_EQ(a.ibDevN, 2);
  EXPECT_TRUE(a.useIonic);
  EXPECT_EQ(a.isP2p, 1);
  EXPECT_EQ(a.maxRecvWorkRequest, 0u);
  EXPECT_EQ(a.maxSendWorkRequest, 2u * NET_IB_MAX_REQUESTS);
  EXPECT_EQ(a.qpIdx, 0);
  EXPECT_FALSE(a.isQpSharingEnabled);
  free(s);

  auto* r = NewComm<ncclIbRecvComm>();
  r->devs[0].base = {0, &pd_, &cq_};
  ScopedHook rq(g_ibCastResiliencyDataRqSizeGet, [&](ncclIbResiliency* res, uint dev, uint32_t* size) {
    EXPECT_EQ(res, r->base.resiliency);
    EXPECT_EQ(dev, 0u);
    *size = 321;
    return ncclSuccess;
  });
  IbCastBuildDataQpCreateAttr(&r->base, 0, &a);
  EXPECT_EQ(a.maxRecvWorkRequest, 321u);
  EXPECT_EQ(a.maxSendWorkRequest, (uint32_t)NET_IB_MAX_REQUESTS);
  free(r);
}

TEST_F(NetIbCastConnectQpMicrotest, InitCommDevBase_SharesOnePdPerDeviceAndSnapshotsTheGid) {
  ncclIbNetCommDevBase a = {}, b = {};
  ASSERT_EQ(IbCastInitCommDevBase(1, &a, &ctx_[1], 64), ncclSuccess);
  ASSERT_EQ(IbCastInitCommDevBase(1, &b, &ctx_[1], 32), ncclSuccess);
  EXPECT_EQ(a.ibDevN, 1);
  EXPECT_EQ(a.pd, b.pd);
  EXPECT_EQ(verbs_.pds.size(), 1u);
  EXPECT_EQ(IbCastDevs[1].pdRefs, 2);
  EXPECT_EQ(a.cq->cqe, 64);
  EXPECT_EQ(a.cq->cq_context, &ctx_[1]);
  EXPECT_TRUE(SameGid(a.gidInfo.localGid, IbCastDevs[1].gidInfo.localGid));
  EXPECT_EQ(a.gidInfo.localGidIndex, kGidIndex);

  ASSERT_EQ(IbCastDestroyBase(&a), ncclSuccess);
  EXPECT_EQ(IbCastDevs[1].pdRefs, 1);
  EXPECT_EQ(verbs_.live.count(b.pd), 1u);
  ASSERT_EQ(IbCastDestroyBase(&b), ncclSuccess);
  EXPECT_EQ(IbCastDevs[1].pdRefs, 0);
  EXPECT_TRUE(verbs_.live.empty());
}

TEST_F(NetIbCastConnectQpMicrotest, InitCommDevBase_ClampsTheCqToTheDeviceMaximum) {
  IbCastDevs[0].maxCqe = 100;
  ncclIbNetCommDevBase a = {};
  const std::string log = CaptureLog([&] { ASSERT_EQ(IbCastInitCommDevBase(0, &a, nullptr, 101), ncclSuccess); });
  EXPECT_EQ(verbs_.cqSizes.back(), 100);
  EXPECT_TRUE(LogHas(log, "requested CQ size 101 exceeds device fake0 max_cqe 100, clamping")) << log;
  ncclIbNetCommDevBase b = {};
  ASSERT_EQ(IbCastInitCommDevBase(0, &b, nullptr, 100), ncclSuccess);
  EXPECT_EQ(verbs_.cqSizes.back(), 100);
  IbCastDestroyBase(&a);
  IbCastDestroyBase(&b);
}

TEST_F(NetIbCastConnectQpMicrotest, InitCommDevBase_PdOrCqFailure_Propagates) {
  ncclIbNetCommDevBase a = {};
  createCqResult_ = ncclSystemError;
  EXPECT_EQ(IbCastInitCommDevBase(0, &a, nullptr, 8), ncclSystemError);
  IbCastDevs[0].pdRefs = 0;
  allocPdResult_ = ncclSystemError;
  EXPECT_EQ(IbCastInitCommDevBase(0, &a, nullptr, 8), ncclSystemError);
}

TEST_F(NetIbCastConnectQpMicrotest, DestroyBase_CqFailureKeepsThePdReference) {
  ncclIbNetCommDevBase a = {};
  ASSERT_EQ(IbCastInitCommDevBase(0, &a, nullptr, 8), ncclSuccess);
  destroyCqResult_ = ncclSystemError;
  EXPECT_EQ(IbCastDestroyBase(&a), ncclSystemError);
  EXPECT_EQ(IbCastDevs[0].pdRefs, 1);
  destroyCqResult_ = ncclSuccess;
  IbCastDevs[0].pdRefs = 0;
}

TEST_F(NetIbCastConnectMicrotest, MatchingScheme_FollowsOffloadSharingOooRqFailoverThenTheParam) {
  EXPECT_EQ(IbCastResolveRecvMatchingScheme(true), BY_ORDER);
  EXPECT_EQ(IbCastResolveRecvMatchingScheme(false), BY_INDEX);
  Ainic();
  EXPECT_EQ(IbCastResolveRecvMatchingScheme(false), BY_ORDER);
  Param("IB_RECEIVER_SIDE_MATCHING_SCHEME", BY_INDEX);
  EXPECT_EQ(IbCastResolveRecvMatchingScheme(false), BY_INDEX);
  Param("IB_RESILIENCY_PORT_FAILOVER", 1);
  EXPECT_EQ(IbCastResolveRecvMatchingScheme(false), BY_ID);
  Param("IB_RESILIENCY_PORT_FAILOVER", 2);
  EXPECT_EQ(IbCastResolveRecvMatchingScheme(false), BY_INDEX);
  Param("IB_OOO_RQ", 1);
  EXPECT_EQ(IbCastResolveRecvMatchingScheme(false), BY_ID);
  Param("IB_OOO_RQ", 0);
  IbCastQpSharingGlobalEnable = true;
  EXPECT_EQ(IbCastResolveRecvMatchingScheme(false), BY_ID);
  EXPECT_EQ(IbCastResolveRecvMatchingScheme(true), BY_ORDER);
}

TEST_F(NetIbCastConnectMicrotest, CtsOffload_IsOffForP2pOnlyWhenP2pDisablesIt) {
  EXPECT_FALSE(IbCastIsCtsOffloadEnabled(0));
  IbCastOffloadEnabled = true;
  EXPECT_TRUE(IbCastIsCtsOffloadEnabled(0));
  EXPECT_FALSE(IbCastIsCtsOffloadEnabled(1));
  Param("RCCL_IB_P2P_DISABLE_CTS", 0);
  EXPECT_TRUE(IbCastIsCtsOffloadEnabled(1));
  EXPECT_TRUE(IbCastByOrderRequested());
  IbCastOffloadEnabled = false;
  EXPECT_FALSE(IbCastByOrderRequested());
}

TEST_F(NetIbCastConnectMicrotest, ReportMatchingScheme_LogsCollectiveAndP2pSettings) {
  IbCastOffloadEnabled = true;
  const std::string log = InfoLog([] { IbCastReportMatchingScheme(); });
  EXPECT_TRUE(LogHas(log, "collectives communicators: CTS Offload: ON   RecvMatchingScheme: 2")) << log;
  EXPECT_TRUE(LogHas(log, "P2P communicators: CTS Offload: OFF   RecvMatchingScheme: 0")) << log;
}

TEST_F(NetIbCastConnectMicrotest, CalculateNqps_ScalesTheLargerSideByThePerConnectionCount) {
  EXPECT_EQ(IbCastCalculateNqps(0, 1, 3, "t"), 6);
  EXPECT_EQ(IbCastCalculateNqps(1, 2, 1, "t"), 4);
  Param("RCCL_IB_QPS_PER_P2P", 3);
  EXPECT_EQ(IbCastCalculateNqps(1, 2, 1, "t"), 6);
  EXPECT_EQ(IbCastCalculateNqps(0, 2, 1, "t"), 4);
  Param("IB_QPS_PER_CONNECTION", 1);
  EXPECT_EQ(IbCastCalculateNqps(0, 2, 1, "t"), 2);
}

TEST_F(NetIbCastConnectMicrotest, TrafficClass_ReadsAndWritesTheCommConfig) {
  ncclNetCommConfig_t cfg = {};
  cfg.trafficClass = 9;
  EXPECT_EQ(IbCastGetTrafficClass(&cfg), 9);
  IbCastSetTrafficClass(&cfg, 4);
  EXPECT_EQ(cfg.trafficClass, 4);
  EXPECT_EQ(IbCastGetTrafficClass(nullptr), NCCL_NET_TRAFFIC_CLASS_UNDEF);
  IbCastSetTrafficClass(nullptr, 4);
}

TEST_F(NetIbCastConnectMicrotest, CheckVProps_ReportsMismatchedRailsOnlyWhenAsked) {
  ncclNetVDeviceProps_t local = {}, remote = {};
  local.ndevs = 2;
  local.devs[0] = 0;
  local.devs[1] = 1;
  remote.ndevs = 1;
  remote.devs[0] = 1;
  std::string log = InfoLog([&] { EXPECT_EQ(IbCastCheckVProps(&local, &remote), ncclSuccess); });
  EXPECT_FALSE(LogHas(log, "mismatched physical devices")) << log;
  Param("IB_WARN_RAIL_LOCAL", 1);
  log = InfoLog([&] { EXPECT_EQ(IbCastCheckVProps(&local, &remote), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, "between local (0,1) and remote (1)")) << log;
  log = InfoLog([&] { EXPECT_EQ(IbCastCheckVProps(&remote, &local), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, "between local (1) and remote (0,1)")) << log;
  log = InfoLog([&] { EXPECT_EQ(IbCastCheckVProps(&local, &local), ncclSuccess); });
  EXPECT_FALSE(LogHas(log, "mismatched physical devices")) << log;
}

TEST_F(NetIbCastConnectMicrotest, P2pPolicy_TagsOnlyAValidHandle) {
  ncclIbHandle h = Listen();
  EXPECT_EQ(rcclCastNetP2pPolicy(nullptr, 1), ncclInvalidArgument);
  EXPECT_EQ(rcclCastNetP2pPolicy(&h, 1), ncclSuccess);
  EXPECT_EQ(h.isP2p, 1);
  h.magic++;
  EXPECT_EQ(rcclCastNetP2pPolicy(&h, 0), ncclInvalidArgument);
  EXPECT_EQ(h.isP2p, 1);
}

class NetIbCastConnectListenMicrotest : public NetIbCastConnectMicrotest {};

TEST_F(NetIbCastConnectListenMicrotest, FillsTheHandleFromTheListeningSocket) {
  ncclIbHandle h;
  memset(&h, 0x5a, sizeof(h));
  void* lc = nullptr;
  ASSERT_EQ(IbCastListen(nullptr, 1, &h, &lc), ncclSuccess);
  listens_.push_back(lc);
  auto* comm = static_cast<ncclIbListenComm*>(lc);
  EXPECT_EQ(comm->dev, 1);
  EXPECT_EQ(comm->stage, nullptr);
  EXPECT_EQ(comm->sock.type, ncclSocketTypeNetIb);
  EXPECT_EQ(h.magic, NCCL_SOCKET_MAGIC);
  EXPECT_EQ(memcmp(&h.connectAddr, &sockets_.listenAddr, sizeof(h.connectAddr)), 0);
  EXPECT_EQ(h.isP2p, 0);
  EXPECT_FALSE(h.isRMA);
  EXPECT_EQ(h.stage.state, ncclIbCommStateStart);
  EXPECT_FALSE(validGid(&h.listenGids[0]));
  EXPECT_FALSE(validGid(&h.listenGids[1]));
}

TEST_F(NetIbCastConnectListenMicrotest, SocketFailure_ClosesAndFails) {
  listenResult_ = ncclSystemError;
  ncclIbHandle h;
  void* lc = nullptr;
  EXPECT_EQ(IbCastListen(nullptr, 0, &h, &lc), ncclSystemError);
  EXPECT_EQ(lc, nullptr);
  EXPECT_EQ(sockets_.closed.size(), 1u);
  EXPECT_EQ(g_freedPointers.size(), 1u);
}

TEST_F(NetIbCastConnectListenMicrotest, SubnetAware_AdvertisesTheFirstTwoRocePfGids) {
  Param("IB_SUBNET_AWARE_ROUTING", 1);
  Param("IB_GID_INDEX", kGidIndex);
  gidByCtx_[&ctx_[0]] = V4Gid("10.0.0.1");
  gidByCtx_[&ctx_[1]] = V4Gid("10.0.1.1");
  gidByCtx_[&ctx_[2]] = V4Gid("10.0.2.1");
  SetDevice(0, IBV_LINK_LAYER_INFINIBAND);
  SetMerged(1, {0, 1, 2});
  ncclIbHandle h = Listen(1);
  EXPECT_TRUE(SameGid(h.listenGids[0], V4Gid("10.0.1.1")));
  EXPECT_TRUE(SameGid(h.listenGids[1], V4Gid("10.0.2.1")));

  SetDevice(0, IBV_LINK_LAYER_ETHERNET);
  std::string log = CaptureLog([&] { h = Listen(1); });
  EXPECT_TRUE(LogHas(log, "device 1 has 3 RoCE PFs but only the first 2 are advertised")) << log;
  EXPECT_TRUE(SameGid(h.listenGids[0], V4Gid("10.0.0.1")));
  EXPECT_TRUE(SameGid(h.listenGids[1], V4Gid("10.0.1.1")));
  EXPECT_EQ(queriedGids_, std::vector<int>(4, kGidIndex));

  h = Listen(IbCastNMergedDevs);
  EXPECT_FALSE(validGid(&h.listenGids[0]));
}

TEST_F(NetIbCastConnectListenMicrotest, SubnetAware_GidQueryFailure_FailsTheListen) {
  Param("IB_SUBNET_AWARE_ROUTING", 1);
  Param("IB_GID_INDEX", kGidIndex);
  queryGidResult_ = ncclSystemError;
  ncclIbHandle h;
  void* lc = nullptr;
  EXPECT_EQ(IbCastListen(nullptr, 0, &h, &lc), ncclSystemError);
  EXPECT_EQ(sockets_.closed.size(), 1u);
  Param("IB_GID_INDEX", -1);  // the scan's own query fails first
  EXPECT_EQ(IbCastListen(nullptr, 0, &h, &lc), ncclSystemError);
}

TEST_F(NetIbCastConnectListenMicrotest, CloseListen_ClosesTheSocketAndToleratesNull) {
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  Forget(lc);
  EXPECT_EQ(IbCastCloseListen(lc), ncclSuccess);
  EXPECT_EQ(sockets_.closed.back(), &static_cast<ncclIbListenComm*>(lc)->sock);
  EXPECT_EQ(IbCastCloseListen(nullptr), ncclSuccess);
}

class NetIbCastConnectHandshakeMicrotest : public NetIbCastConnectMicrotest {};

TEST_F(NetIbCastConnectHandshakeMicrotest, PartialSocketProgress_ResumesEveryStage) {
  sockets_.chunk = 97;
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  ASSERT_NE(s, nullptr);
  EXPECT_EQ(verbs_.Rtr(s->base.qps[1].qp)->attr.dest_qp_num, r->base.qps[1].qp->qp_num);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, SocketNotReady_RetriesFromTheConnectCheck) {
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  sockets_.ready = 0;
  void* sc = nullptr;
  ASSERT_EQ(IbCastConnect(nullptr, 0, &h, &sc, nullptr), ncclSuccess);
  EXPECT_EQ(sc, nullptr);
  EXPECT_EQ(h.stage.state, ncclIbCommStateConnect);
  void* rc = nullptr;
  ASSERT_EQ(IbCastAccept(lc, &rc, nullptr), ncclSuccess);
  EXPECT_EQ(rc, nullptr);
  EXPECT_EQ(static_cast<ncclIbListenComm*>(lc)->stage->state, ncclIbCommStateAccept);
  sockets_.ready = 1;
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  EXPECT_EQ(Handshake(&h, lc, &s, &r), ncclSuccess);
  EXPECT_EQ(sockets_.conns.size(), 1u);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, UnexpectedStageState_IsRejected) {
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  h.stage.state = ncclIbCommStateAccept;
  void* sc = nullptr;
  std::string log = CaptureLog([&] { EXPECT_EQ(IbCastConnect(nullptr, 0, &h, &sc, nullptr), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "trying to connect already connected sendComm")) << log;
  auto* l = static_cast<ncclIbListenComm*>(lc);
  ASSERT_EQ(ncclCalloc(&l->stage, 1), ncclSuccess);
  l->stage->state = ncclIbCommStateConnect;
  void* rc = nullptr;
  log = CaptureLog([&] { EXPECT_EQ(IbCastAccept(lc, &rc, nullptr), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "Listencomm in unknown state 1")) << log;
  free(l->stage);
  l->stage = nullptr;
}

TEST_F(NetIbCastConnectHandshakeMicrotest, NonexistentDevice_FailsEitherSide) {
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  void* sc = nullptr;
  std::string log =
      CaptureLog([&] { EXPECT_EQ(IbCastConnect(nullptr, IbCastNMergedDevs, &h, &sc, nullptr), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "non-existent virtual device 4")) << log;
  Abandon(&h);

  ncclIbHandle h2 = Listen(IbCastNMergedDevs, &lc);
  ASSERT_EQ(IbCastConnect(nullptr, 0, &h2, &sc, nullptr), ncclSuccess);
  void* rc = nullptr;
  log = CaptureLog([&] {
    ncclResult_t res = ncclSuccess;
    for (int i = 0; i < 4 && res == ncclSuccess; i++) res = IbCastAccept(lc, &rc, nullptr);
    EXPECT_EQ(res, ncclInternalError);
  });
  EXPECT_TRUE(LogHas(log, "non-existent virtual device 4")) << log;
  Abandon(&h2);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, FusedSenderToSingleNicReceiver_StripesQpsOverBothSenderDevices) {
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  const std::string log = InfoLog([&] { ASSERT_EQ(Handshake(&h, lc, &s, &r, /*dev=*/2), ncclSuccess); });
  ASSERT_EQ(s->base.nqps, 4);
  ASSERT_EQ(r->base.nqps, 4);
  EXPECT_EQ(s->base.nDataQps, 2);
  EXPECT_EQ(r->base.nDataQps, 2);
  for (int q = 0; q < 4; q++) {
    SCOPED_TRACE(q);
    EXPECT_EQ(s->base.qps[q].devIndex, q % 2);
    EXPECT_EQ(r->base.qps[q].devIndex, 0);
    EXPECT_EQ(s->base.qps[q].remDevIdx, 0);
    EXPECT_EQ(r->base.qps[q].remDevIdx, q % 2);
    EXPECT_EQ(verbs_.CreateOf(s->base.qps[q].qp)->pd, s->devs[q % 2].base.pd);
    EXPECT_EQ(verbs_.Rtr(r->base.qps[q].qp)->attr.dest_qp_num, s->base.qps[q].qp->qp_num);
  }
  EXPECT_EQ(r->base.nRemDevs, 2);
  EXPECT_EQ(s->remCmplsRecords.rkeys[0], r->devs[0].cmplsRecordsMr->rkey);
  EXPECT_EQ(r->remCtsFifo.rkeys[1], s->devs[1].ctsFifoMr->rkey);
  EXPECT_TRUE(LogHas(log, "has a different number of devices=1 as remote merged2 2")) << log;
}

TEST_F(NetIbCastConnectHandshakeMicrotest, QpCount_FollowsTheConnectionAndP2pParams) {
  Param("IB_QPS_PER_CONNECTION", 3);
  Param("RCCL_IB_QPS_PER_P2P", 1);
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  EXPECT_EQ(s->base.nqps, 3);
  EXPECT_EQ(r->base.nqps, 3);
  EXPECT_EQ(s->base.isP2p, 0);
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  ASSERT_EQ(rcclCastNetP2pPolicy(&h, 1), ncclSuccess);
  ASSERT_EQ(Handshake(&h, lc, &s, &r), ncclSuccess);
  EXPECT_EQ(s->base.nqps, 1);
  EXPECT_EQ(r->base.nqps, 1);
  EXPECT_EQ(s->base.isP2p, 1);
  EXPECT_EQ(r->base.isP2p, 1);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, ServiceLevelAndTrafficClass_DefaultToZero) {
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  EXPECT_EQ(verbs_.Rtr(s->base.qps[0].qp)->attr.ah_attr.sl, 0);
  EXPECT_EQ(verbs_.Rtr(s->base.qps[0].qp)->attr.ah_attr.grh.traffic_class, 0);
  EXPECT_EQ(verbs_.Rtr(r->base.qps[0].qp)->attr.ah_attr.grh.traffic_class, 0);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, ServiceLevelAndTrafficClass_ComeFromTheCommConfig) {
  ncclNetCommConfig_t cfg = {};
  cfg.trafficClass = 7;
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_EQ(Handshake(&h, lc, &s, &r, 0, &cfg), ncclSuccess);
  EXPECT_EQ(verbs_.Rtr(s->base.qps[0].qp)->attr.ah_attr.sl, 7);
  EXPECT_EQ(verbs_.Rtr(s->base.qps[0].qp)->attr.ah_attr.grh.traffic_class, 7);
  EXPECT_EQ(verbs_.Rtr(r->base.qps[0].qp)->attr.ah_attr.sl, 7);
  EXPECT_EQ(verbs_.Rtr(r->base.qps[0].qp)->attr.ah_attr.grh.traffic_class, 7);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, ServiceLevelAndTrafficClass_ParamsOverrideTheCommConfig) {
  Param("IB_SL", 5);
  Param("IB_TC", 106);
  Param("IB_FIFO_TC", 98);
  ncclNetCommConfig_t cfg = {};
  cfg.trafficClass = 7;
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_EQ(Handshake(&h, lc, &s, &r, 0, &cfg), ncclSuccess);
  EXPECT_EQ(verbs_.Rtr(s->base.qps[0].qp)->attr.ah_attr.sl, 5);
  EXPECT_EQ(verbs_.Rtr(r->base.qps[0].qp)->attr.ah_attr.sl, 5);
  EXPECT_EQ(verbs_.Rtr(s->base.qps[0].qp)->attr.ah_attr.grh.traffic_class, 106);
  EXPECT_EQ(verbs_.Rtr(r->base.qps[0].qp)->attr.ah_attr.grh.traffic_class, 98);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, InfinibandLink_UsesNoTrafficClass) {
  for (int i = 0; i < kNDevs; i++) SetDevice(i, IBV_LINK_LAYER_INFINIBAND);
  Param("IB_TC", 106);
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  const std::string log = InfoLog([&] { Connected(&s, &r); });
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_NE(s, nullptr);
  EXPECT_EQ(s->base.qps[0].rtrAttr.tc, 0xff);
  EXPECT_EQ(r->base.qps[0].rtrAttr.tc, 106);  // the receiver copies the sender's value regardless
  EXPECT_EQ(verbs_.Rtr(s->base.qps[0].qp)->attr.ah_attr.dlid, IbCastDevs[0].portAttr.lid);
  EXPECT_TRUE(LogHas(log, "LID 16 subnet-prefix")) << log;
}

TEST_F(NetIbCastConnectHandshakeMicrotest, Mtu_BothSidesSettleOnTheSmaller) {
  IbCastDevs[1].portAttr.active_mtu = IBV_MTU_1024;
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_EQ(Handshake(&h, lc, &s, &r, /*dev=*/1), ncclSuccess);
  EXPECT_EQ(verbs_.Rtr(r->base.qps[0].qp)->attr.path_mtu, IBV_MTU_1024);
  EXPECT_EQ(verbs_.Rtr(s->base.qps[0].qp)->attr.path_mtu, IBV_MTU_1024);
  EXPECT_EQ(IbCastDevs[0].portAttr.active_mtu, IBV_MTU_1024);
  // The larger side, here the sender, comes down too.
  IbCastDevs[1].portAttr.active_mtu = IBV_MTU_4096;
  h = Listen(0, &lc);
  ASSERT_EQ(Handshake(&h, lc, &s, &r, /*dev=*/1), ncclSuccess);
  EXPECT_EQ(verbs_.Rtr(s->base.qps[0].qp)->attr.path_mtu, IBV_MTU_1024);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, Ece_SenderAppliesTheReceiversReducedEce) {
  queryEceOptions_ = 0x3;
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  // Receiver: set the sender's ECE, then query the reduced one; sender: set that.
  ASSERT_EQ(setEce_.size(), 4u);
  EXPECT_EQ(setEce_[0].first, r->base.qps[0].qp);
  EXPECT_EQ(setEce_[2].first, s->base.qps[0].qp);
  EXPECT_EQ(setEce_[3].first, s->base.qps[1].qp);
  EXPECT_EQ(setEce_[3].second.options, 0x3u);
  EXPECT_EQ(s->base.qps[0].eceSupported, 1);
  EXPECT_EQ(s->base.qps[0].ece.options, 0x3u);
  EXPECT_EQ(r->base.qps[1].eceSupported, 1);
  EXPECT_EQ(r->base.qps[1].ece.vendor_id, 0x15b3u);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, Ece_DisabledOrUnsupported_IsClearedOnBothSides) {
  Param("IB_ECE_ENABLE", 0);
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  EXPECT_TRUE(setEce_.empty());
  EXPECT_EQ(s->base.qps[0].eceSupported, 0);
  EXPECT_EQ(r->base.qps[0].eceSupported, 0);
  Param("IB_ECE_ENABLE", 1);
  eceSupported_ = 0;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  EXPECT_TRUE(setEce_.empty());
  EXPECT_EQ(s->base.qps[0].eceSupported, 0);
  eceSupported_ = 1;
  setEceSupported_ = 0;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  EXPECT_EQ(setEce_.size(), 2u);  // the receiver's set reports no support, so it never queries back
  EXPECT_EQ(s->base.qps[0].eceSupported, 0);
  EXPECT_EQ(r->base.qps[0].eceSupported, 0);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, Rma_UsesOneQpAndAFlushQp) {
  Param("RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING", 0);
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  h.isRMA = true;
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_EQ(Handshake(&h, lc, &s, &r), ncclSuccess);
  EXPECT_EQ(s->base.nqps, 1);
  EXPECT_EQ(r->base.nqps, 1);
  EXPECT_FALSE(s->useCtsOffload);
  EXPECT_EQ(r->flushEnabled, 1);
  EXPECT_NE(r->devs[0].gpuFlush.qp.qp, nullptr);
  EXPECT_EQ(r->devs[0].gpuFlush.gpuFlushGpuMem, nullptr);
  EXPECT_EQ(r->devs[0].gpuFlush.dmabufFd, -1);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, CtsOffload_ForcesByOrderAndOptionalRecvCompletion) {
  IbCastOffloadEnabled = true;
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  EXPECT_TRUE(s->useCtsOffload);
  EXPECT_TRUE(r->useCtsOffload);
  EXPECT_EQ(s->base.recvMatchingScheme, BY_ORDER);
  EXPECT_EQ(r->base.recvMatchingScheme, BY_ORDER);
  EXPECT_TRUE(s->base.optRecvCompletion);
  EXPECT_TRUE(r->base.optRecvCompletion);
  EXPECT_EQ(optRecvCalls_, (std::vector<bool>{true, true}));
}

TEST_F(NetIbCastConnectHandshakeMicrotest, OptionalRecvCompletion_NeedsBothSides) {
  optRecv_ = true;
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  EXPECT_TRUE(s->base.optRecvCompletion);
  EXPECT_TRUE(r->base.optRecvCompletion);
  // Sender opts out: the receiver's later flag ANDs the sender's false in.
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  void* sc = nullptr;
  void* rc = nullptr;
  ASSERT_EQ(IbCastConnect(nullptr, 0, &h, &sc, nullptr), ncclSuccess);
  ASSERT_EQ(IbCastAccept(lc, &rc, nullptr), ncclSuccess);
  optRecv_ = false;
  ASSERT_EQ(IbCastConnect(nullptr, 0, &h, &sc, nullptr), ncclSuccess);
  optRecv_ = true;
  ASSERT_EQ(Handshake(&h, lc, &s, &r), ncclSuccess);
  EXPECT_FALSE(s->base.optRecvCompletion);
  EXPECT_FALSE(r->base.optRecvCompletion);
  // Receiver opts out: the sender ANDs the receiver's false in.
  h = Listen(0, &lc);
  sc = rc = nullptr;
  ASSERT_EQ(IbCastConnect(nullptr, 0, &h, &sc, nullptr), ncclSuccess);
  ASSERT_EQ(IbCastAccept(lc, &rc, nullptr), ncclSuccess);
  ASSERT_EQ(IbCastConnect(nullptr, 0, &h, &sc, nullptr), ncclSuccess);
  optRecv_ = false;
  ASSERT_EQ(Handshake(&h, lc, &s, &r), ncclSuccess);
  EXPECT_FALSE(s->base.optRecvCompletion);
  EXPECT_FALSE(r->base.optRecvCompletion);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, OooRq_CreatesOutOfOrderQpsWhenBothSidesSupportIt) {
  Param("IB_OOO_RQ", 1);
  for (int i = 0; i < kNDevs; i++) {
    IbCastDevs[i].ibProvider = IB_PROVIDER_MLX5;
    IbCastDevs[i].ar = 1;
    IbCastDevs[i].oooRqSize = NET_IB_MAX_REQUESTS;
  }
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  ASSERT_NE(s, nullptr);
  EXPECT_TRUE(s->base.localOooRq);
  EXPECT_TRUE(s->base.remOooRq);
  EXPECT_TRUE(r->base.remOooRq);
  EXPECT_EQ(mlx5Attrs_.size(), 4u);
  for (const auto& dv : mlx5DvAttrs_) EXPECT_TRUE(dv.create_flags & MLX5DV_QP_CREATE_OOO_DP);
  EXPECT_EQ(s->base.recvMatchingScheme, BY_ID);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, OooRq_FailsWithoutAdaptiveRoutingPeerSupportOrQueueDepth) {
  Param("IB_OOO_RQ", 1);
  for (int i = 0; i < kNDevs; i++) {
    IbCastDevs[i].ibProvider = IB_PROVIDER_MLX5;
    IbCastDevs[i].ar = 1;
    IbCastDevs[i].oooRqSize = NET_IB_MAX_REQUESTS;
  }
  struct Case {
    const char* what;
    std::function<void()> setup;
    const char* warn;
  };
  const std::vector<Case> cases = {
      {"sender AR", [] { IbCastDevs[1].ar = 0; }, "AR is not enabled"},
      {"receiver AR", [] { IbCastDevs[0].ar = 0; }, "AR is not enabled"},
      {"peer support", [] { IbCastDevs[0].oooRqSize = 0; }, "not supported on both sides"},
      {"queue depth", [] { IbCastDevs[0].oooRqSize = NET_IB_MAX_REQUESTS - 1; }, "less than the required"},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.what);
    for (int i = 0; i < kNDevs; i++) {
      IbCastDevs[i].ar = 1;
      IbCastDevs[i].oooRqSize = NET_IB_MAX_REQUESTS;
      IbCastDevs[i].pdRefs = 0;
    }
    c.setup();
    void* lc = nullptr;
    ncclIbHandle h = Listen(0, &lc);
    ncclIbSendComm* s = nullptr;
    ncclIbRecvComm* r = nullptr;
    ncclResult_t res = ncclSuccess;
    const std::string log = CaptureLog([&] { res = Handshake(&h, lc, &s, &r, /*dev=*/1); });
    EXPECT_EQ(res, ncclInternalError);
    EXPECT_TRUE(LogHas(log, c.warn)) << log;
  }
}

TEST_F(NetIbCastConnectHandshakeMicrotest, Prepost_PostsEveryReceiveSlotOnEveryQp) {
  prepost_ = true;
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  EXPECT_EQ(verbs_.postRecvs, 2 * NET_IB_MAX_REQUESTS);
  verbs_.postRecvFailAt = verbs_.postRecvs + 3;
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  EXPECT_EQ(Handshake(&h, lc, &s, &r), ncclSystemError);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, Telemetry_ClassifiesSenderDataAndReceiverCtsQps) {
  rcclTelemetryEnabled = 1;
  RcclChannelStats* ch = nullptr;
  int data0 = 0, cts0 = 0;
  if ((ch = rcclTelemetryChannel(0, 0)) != nullptr) {
    data0 = ch->num_data_qp;
    cts0 = ch->num_cts_qp;
  }
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  ch = rcclTelemetryChannel(0, 0);
  ASSERT_NE(ch, nullptr);
  EXPECT_EQ(ch->num_data_qp - data0, 3);  // two sender QPs and the receiver's non-CTS QP
  EXPECT_EQ(ch->num_cts_qp - cts0, 1);
  EXPECT_EQ(s->telChStats, ch);
  EXPECT_EQ(r->telChStats, ch);
  EXPECT_NE(s->base.qps[1].telQpStats, nullptr);
  EXPECT_NE(r->base.qps[1].telQpStats, nullptr);
  EXPECT_EQ(rcclTelemetryDevs[0].channels_unknown, 1);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, Telemetry_AinicCarriesTheChannelId) {
  rcclTelemetryEnabled = 1;
  Ainic();
  ncclNet_ctxt_t sendCtx = {12}, recvCtx = {12};
  void* lc = nullptr;
  ncclIbHandle h = Listen(3, &lc);
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_EQ(Handshake(&h, lc, &s, &r, 3, nullptr, reinterpret_cast<ncclNetDeviceHandle_t**>(&sendCtx),
                      reinterpret_cast<ncclNetDeviceHandle_t**>(&recvCtx)),
            ncclSuccess);
  EXPECT_EQ(s->telChId, 12);
  EXPECT_EQ(r->telChId, 12);
  EXPECT_EQ(s->base.qps[0].channelId, 12);
  EXPECT_EQ(s->telChStats, rcclTelemetryChannel(2, 12));
  EXPECT_EQ(rcclTelemetryDevs[2].channels_unknown, 0);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, SubnetAware_BothSidesMoveToTheNicOnThePeersSubnet) {
  Param("IB_SUBNET_AWARE_ROUTING", 1);
  Param("IB_GID_INDEX", kGidIndex);
  gidByCtx_[&ctx_[0]] = V4Gid("10.0.0.1");
  gidByCtx_[&ctx_[1]] = V4Gid("10.0.1.1");
  void* lc = nullptr;
  ncclIbHandle h = Listen(1, &lc);
  // Sender asks for dev 0, whose 10.0.0.x cannot reach the listener's 10.0.1.x.
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  std::string log = InfoLog([&] { ASSERT_EQ(Handshake(&h, lc, &s, &r, 0), ncclSuccess); });
  EXPECT_EQ(s->base.vProps.devs[0], 1);
  EXPECT_EQ(r->base.vProps.devs[0], 1);
  EXPECT_TRUE(LogHas(log, "overriding dev 0 with dev 1")) << log;

  // The receiver follows the sender's GIDs the same way; no advertised GIDs keeps the sender on dev 1.
  h = Listen(0, &lc);
  memset(h.listenGids, 0, sizeof(h.listenGids));
  ASSERT_EQ(Handshake(&h, lc, &s, &r, 1), ncclSuccess);
  EXPECT_EQ(s->base.vProps.devs[0], 1);
  EXPECT_EQ(r->base.vProps.devs[0], 1);
  EXPECT_EQ(static_cast<ncclIbListenComm*>(lc)->dev, 1);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, SubnetAware_FusedDefaultKeptOnlyWhenEveryPfReaches) {
  Param("IB_SUBNET_AWARE_ROUTING", 1);
  Param("IB_GID_INDEX", kGidIndex);
  gidByCtx_[&ctx_[0]] = V4Gid("10.0.0.1");
  gidByCtx_[&ctx_[1]] = V4Gid("10.0.0.2");
  gidByCtx_[&ctx_[2]] = V4Gid("10.0.2.1");
  union ibv_gid remote[2] = {V4Gid("10.0.0.9"), {}};
  int dev = -1;
  ASSERT_EQ(IbCastFindDevBySubnet(remote, 2, 2, &dev), ncclSuccess);
  EXPECT_EQ(dev, 2);
  gidByCtx_[&ctx_[1]] = V4Gid("10.0.2.2");
  ASSERT_EQ(IbCastFindDevBySubnet(remote, 2, 2, &dev), ncclSuccess);
  EXPECT_EQ(dev, 0);
  // No reachable device: the default stays.
  remote[0] = V4Gid("10.7.0.9");
  ASSERT_EQ(IbCastFindDevBySubnet(remote, 2, 2, &dev), ncclSuccess);
  EXPECT_EQ(dev, 2);
  // An out-of-range default is never probed but is still the fallback.
  ASSERT_EQ(IbCastFindDevBySubnet(remote, 2, -1, &dev), ncclSuccess);
  EXPECT_EQ(dev, -1);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, SubnetAware_SkipsInfinibandAndUnqueryableDevices) {
  Param("IB_GID_INDEX", kGidIndex);
  gidByCtx_[&ctx_[0]] = V4Gid("10.0.0.1");
  gidByCtx_[&ctx_[1]] = V4Gid("10.0.2.1");
  gidByCtx_[&ctx_[2]] = V4Gid("10.0.2.2");
  union ibv_gid remote[1] = {V4Gid("10.0.2.9")};
  SetDevice(1, IBV_LINK_LAYER_INFINIBAND);
  int dev = -1;
  ASSERT_EQ(IbCastFindDevBySubnet(remote, 1, 0, &dev), ncclSuccess);
  EXPECT_EQ(dev, 3);  // merged 1 is IB-only, merged 2 has a PF off-subnet
  queryGidResult_ = ncclSystemError;
  ASSERT_EQ(IbCastFindDevBySubnet(remote, 1, 0, &dev), ncclSuccess);
  EXPECT_EQ(dev, 0);
  remote[0] = {};
  queryGidResult_ = ncclSuccess;
  ASSERT_EQ(IbCastFindDevBySubnet(remote, 1, 0, &dev), ncclSuccess);
  EXPECT_EQ(dev, 0);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, SubnetAware_PrefixLengthOutOfRange_FailsTheConnect) {
  Param("IB_SUBNET_AWARE_ROUTING", 1);
  Param("IB_GID_INDEX", kGidIndex);
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  h.listenGids[0] = V4Gid("10.0.0.9");
  for (int64_t len : {0, 33}) {
    SCOPED_TRACE(len);
    Param("IB_SUBNET_PREFIX_LEN", len);
    void* sc = nullptr;
    const std::string log =
        CaptureLog([&] { EXPECT_EQ(IbCastConnect(nullptr, 0, &h, &sc, nullptr), ncclInvalidArgument); });
    const std::string want = "NCCL_IB_SUBNET_PREFIX_LEN=" + std::to_string(len) + " is out of range";
    EXPECT_TRUE(LogHas(log, want.c_str())) << log;
  }
}

TEST_F(NetIbCastConnectHandshakeMicrotest, CqSize_CoversThreeCompletionsPerRequestPerQpAndRespectsMaxCqe) {
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  const int want = 3 * NET_IB_MAX_REQUESTS * ncclParamIbCastQpsPerConn();
  EXPECT_EQ(verbs_.cqSizes, (std::vector<int>{want, want}));
  IbCastDevs[0].maxCqe = 1000;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  EXPECT_EQ(verbs_.cqSizes.back(), 1000);
  EXPECT_EQ(verbs_.cqSizes[verbs_.cqSizes.size() - 2], 1000);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, AdaptiveRouting_NeedsEverySenderDevice) {
  IbCastDevs[0].ar = 1;
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_EQ(Handshake(&h, lc, &s, &r, /*dev=*/2), ncclSuccess);
  EXPECT_EQ(s->ar, 0);
  IbCastDevs[1].ar = 1;
  h = Listen(0, &lc);
  ASSERT_EQ(Handshake(&h, lc, &s, &r, /*dev=*/2), ncclSuccess);
  EXPECT_EQ(s->ar, 1);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, LinkLayerMismatch_AmongTheSendersOwnDevices_Fails) {
  SetDevice(1, IBV_LINK_LAYER_INFINIBAND);
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&] { res = Handshake(&h, lc, &s, &r, /*dev=*/2); });
  EXPECT_EQ(res, ncclInternalError);
  EXPECT_TRUE(LogHas(log, "Attempted to connect incompatible devices: [1]fake1:1/IB and [0]fake0:1/RoCE")) << log;
}

TEST_F(NetIbCastConnectHandshakeMicrotest, LinkLayerMismatch_AmongTheReceiversOwnDevices_Fails) {
  SetDevice(1, IBV_LINK_LAYER_INFINIBAND);
  void* lc = nullptr;
  ncclIbHandle h = Listen(2, &lc);
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&] { res = Handshake(&h, lc, &s, &r, /*dev=*/0); });
  EXPECT_EQ(res, ncclInternalError);
  EXPECT_TRUE(LogHas(log, "Attempted to connect incompatible devices: [1]fake1:1/IB and [0]fake0:1/RoCE")) << log;
}

TEST_F(NetIbCastConnectHandshakeMicrotest, LinkLayerMismatch_WithTheRemote_FailsOnTheSideThatSeesIt) {
  SetDevice(1, IBV_LINK_LAYER_INFINIBAND);
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ncclResult_t res = ncclSuccess;
  std::string log = CaptureLog([&] { res = Handshake(&h, lc, &s, &r, /*dev=*/1); });
  EXPECT_EQ(res, ncclInternalError);
  EXPECT_TRUE(LogHas(log, "Remote IB device is incompatible with the local [0]fake0:1/RoCE")) << log;

  // The receiver advertises its GID's link layer, which the sender checks on its own.
  SetDevice(1, IBV_LINK_LAYER_ETHERNET);
  IbCastDevs[0].gidInfo.link_layer = IBV_LINK_LAYER_INFINIBAND;
  h = Listen(0, &lc);
  log = CaptureLog([&] { res = Handshake(&h, lc, &s, &r, /*dev=*/1); });
  EXPECT_EQ(res, ncclInternalError);
  EXPECT_TRUE(LogHas(log, "Remote IB device is incompatible with the local [1]fake1:1/RoCE")) << log;
}

TEST_F(NetIbCastConnectHandshakeMicrotest, ConnectFailure_FreesTheCommAndResetsTheStage) {
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  ScopedHook connect(g_ncclSocketConnect, [](ncclSocket*) { return ncclRemoteError; });
  void* sc = nullptr;
  EXPECT_EQ(IbCastConnect(nullptr, 0, &h, &sc, nullptr), ncclRemoteError);
  EXPECT_EQ(sc, nullptr);
  EXPECT_EQ(h.stage.state, ncclIbCommStateStart);
  EXPECT_EQ(std::count(g_freedPointers.begin(), g_freedPointers.end(), h.stage.comm), 1);
}

TEST_F(NetIbCastConnectHandshakeMicrotest, DeviceSetupFailure_FailsEitherSide) {
  struct Case {
    const char* what;
    std::function<void()> inject;
  };
  const std::vector<Case> cases = {
      {"sender PD", [this] { allocPdResult_ = ncclSystemError; }},
      {"sender QP", [this] { createQpResult_ = ncclSystemError; }},
      {"sender scratchpad MR", [this] { regMrFailAt_ = 0; }},
      {"sender CTS FIFO MR", [this] { regMrFailAt_ = 1; }},
      {"receiver CTS FIFO MR", [this] { regMrFailAt_ = 2; }},
      {"receiver records MR", [this] { regMrFailAt_ = 3; }},
      {"sender records MR", [this] { regMrFailAt_ = 4; }},
      {"RTR", [this] { modifyFailState_ = IBV_QPS_RTR; }},
      {"RTS", [this] { modifyFailState_ = IBV_QPS_RTS; }},
      {"INIT", [this] { modifyFailState_ = IBV_QPS_INIT; }},
      {"ECE query", [this] { Hook(g_ibvQueryEce, [](ibv_qp*, ibv_ece*, int*) { return ncclSystemError; }); }},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.what);
    allocPdResult_ = createQpResult_ = ncclSuccess;
    regMrFailAt_ = modifyFailState_ = -1;
    verbs_.regs.clear();
    IbCastDevs[0].pdRefs = IbCastDevs[1].pdRefs = 0;
    c.inject();
    void* lc = nullptr;
    ncclIbHandle h = Listen(0, &lc);
    ncclIbSendComm* s = nullptr;
    ncclIbRecvComm* r = nullptr;
    EXPECT_NE(Handshake(&h, lc, &s, &r, /*dev=*/1), ncclSuccess);
  }
}

TEST_F(NetIbCastConnectHandshakeMicrotest, SocketProgressFailure_FailsEveryStage) {
  // Fail the n-th progress call (counted over both sides) and check the error surfaces.
  for (int n = 0; n < 8; n++) {
    SCOPED_TRACE(n);
    int calls = 0;
    sockets_.failProgress = [&](int, const ncclSocket*) { return calls++ == n; };
    void* lc = nullptr;
    ncclIbHandle h = Listen(0, &lc);
    ncclIbSendComm* s = nullptr;
    ncclIbRecvComm* r = nullptr;
    EXPECT_EQ(Handshake(&h, lc, &s, &r), ncclRemoteError);
    IbCastDevs[0].pdRefs = 0;
  }
  sockets_.failProgress = [](int, const ncclSocket*) { return false; };
}

TEST_F(NetIbCastConnectHandshakeMicrotest, SocketAcceptFailure_FailsAndDropsTheStage) {
  void* lc = nullptr;
  Listen(0, &lc);
  ScopedHook accept(g_ncclSocketAccept, [](ncclSocket*, ncclSocket*, bool) { return ncclRemoteError; });
  void* rc = nullptr;
  EXPECT_EQ(IbCastAccept(lc, &rc, nullptr), ncclRemoteError);
  EXPECT_EQ(rc, nullptr);
  EXPECT_EQ(static_cast<ncclIbListenComm*>(lc)->stage, nullptr);
}

class NetIbCastConnectResiliencyMicrotest : public NetIbCastConnectMicrotest {
 protected:
  void SetUp() override {
    NetIbCastConnectMicrotest::SetUp();
    resiliency_ = reinterpret_cast<ncclIbResiliency*>(&token_);
    Hook(g_ibCastResiliencyDeviceNumSet, [this](ncclIbResiliency* res, int nLocal, int nRem) {
      EXPECT_EQ(res, resiliency_);
      deviceNums_.push_back({nLocal, nRem});
      return deviceNumResult_;
    });
    Hook(g_ibCastResiliencyDataCqSizeGet, [](ncclIbResiliency*, uint, int* cq) {
      *cq = 640;
      return ncclSuccess;
    });
    Hook(g_ibCastResiliencyDataRqSizeGet, [](ncclIbResiliency*, uint, uint32_t* rq) {
      *rq = 9;
      return ncclSuccess;
    });
    Hook(g_ibCastResiliencyDevInit, [this](ncclIbResiliency*, uint dev, ncclIbDev* ibDev) {
      devInits_.push_back(ibDev - IbCastDevs);
      return devInitResult_;
    });
    Hook(g_ibCastResiliencySenderCreateQps, [](ncclIbResiliency*, ncclIbResiliencyInfo* info) {
      info->probingQpsInfo[0].qpn = 0xabc;
      return ncclSuccess;
    });
    Hook(g_ibCastResiliencyReceiverQpsCreateToRts, [this](ncclIbResiliency*, ncclIbConnectionMetadata* rem,
                                                          ncclIbResiliencyInfo* info) {
      seenBySender_ = rem->resiliencyInfo.probingQpsInfo[0].qpn;
      info->portRecoveryQpsInfo[0].qpn = 0xdef;
      return ncclSuccess;
    });
    Hook(g_ibCastResiliencySenderQpsToRts, [this](ncclIbResiliency*, ncclIbConnectionMetadata* rem) {
      seenByReceiver_ = rem->resiliencyInfo.portRecoveryQpsInfo[0].qpn;
      return ncclSuccess;
    });
    Hook(g_ibCastResiliencyRemoteCompletionRecordsSet, [this](ncclIbResiliency*, uint32_t rkey, uint64_t addr,
                                                              uint dev) {
      records_.push_back({rkey, addr});
      return ncclSuccess;
    });
    Hook(g_ibCastResiliencyClose, [this](ncclIbResiliency*) {
      closes_++;
      return ncclSuccess;
    });
    Hook(g_ibCastResiliencyDevDestroy, [this](ncclIbResiliency*, uint) {
      devDestroys_++;
      return ncclSuccess;
    });
    Hook(g_ibCastResiliencyDestroy, [this](ncclIbResiliency** res) {
      *res = nullptr;
      destroys_++;
      return ncclSuccess;
    });
  }
  int token_ = 0;
  std::vector<std::pair<int, int>> deviceNums_;
  std::vector<long> devInits_;
  std::vector<std::pair<uint32_t, uint64_t>> records_;
  uint32_t seenBySender_ = 0, seenByReceiver_ = 0;
  int closes_ = 0, devDestroys_ = 0, destroys_ = 0;
  ncclResult_t deviceNumResult_ = ncclSuccess;
  ncclResult_t devInitResult_ = ncclSuccess;
};

TEST_F(NetIbCastConnectResiliencyMicrotest, Handshake_SizesQueuesAndExchangesResiliencyQps) {
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_EQ(Handshake(&h, lc, &s, &r, /*dev=*/2), ncclSuccess);
  EXPECT_EQ(deviceNums_, (std::vector<std::pair<int, int>>{{1, 2}, {2, 1}}));
  EXPECT_EQ(devInits_, (std::vector<long>{0, 1, 0}));
  EXPECT_EQ(verbs_.cqSizes, (std::vector<int>{640, 640, 640}));
  EXPECT_EQ(seenBySender_, 0xabcu);
  EXPECT_EQ(seenByReceiver_, 0xdefu);
  ASSERT_EQ(records_.size(), 1u);
  EXPECT_EQ(records_[0].first, r->devs[0].cmplsRecordsMr->rkey);
  EXPECT_EQ(records_[0].second, reinterpret_cast<uint64_t>(r->cmplsRecords));
  const auto* rq = verbs_.CreateOf(r->base.qps[0].qp);
  EXPECT_EQ(rq->attr.cap.max_recv_wr, 9u);
  EXPECT_EQ(rq->attr.cap.max_send_wr, (uint32_t)NET_IB_MAX_REQUESTS);
  EXPECT_EQ(verbs_.CreateOf(s->base.qps[0].qp)->attr.cap.max_send_wr, 2u * NET_IB_MAX_REQUESTS);

  Forget(s);
  Forget(r);
  EXPECT_EQ(IbCastCloseSend(s), ncclSuccess);
  EXPECT_EQ(IbCastCloseRecv(r), ncclSuccess);
  EXPECT_EQ(closes_, 2);
  EXPECT_EQ(devDestroys_, 3);
  EXPECT_EQ(destroys_, 2);
  EXPECT_TRUE(verbs_.live.empty());
}

TEST_F(NetIbCastConnectResiliencyMicrotest, Prepost_PostsTheResiliencyQueueDepth) {
  prepost_ = true;
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  EXPECT_EQ(verbs_.postRecvs, 2 * 9);
}

TEST_F(NetIbCastConnectResiliencyMicrotest, DeviceSetupFailure_FailsTheHandshake) {
  deviceNumResult_ = ncclInternalError;
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  EXPECT_EQ(Handshake(&h, lc, &s, &r), ncclInternalError);
  deviceNumResult_ = ncclSuccess;
  devInitResult_ = ncclSystemError;
  h = Listen(0, &lc);
  EXPECT_EQ(Handshake(&h, lc, &s, &r), ncclSystemError);
}

class NetIbCastConnectFlushMicrotest : public NetIbCastConnectMicrotest {
 protected:
  void SetUp() override {
    NetIbCastConnectMicrotest::SetUp();
    Param("GDR_FLUSH_DISABLE", 0);
    Hook(g_ibCastGdrSupport, [] { return ncclSuccess; });
    Hook(g_hipExtMallocWithFlags, [this](void** ptr, size_t size, unsigned flags) {
      mallocFlags_.push_back(flags);
      EXPECT_EQ(size, sizeof(int));
      *ptr = mallocResult_ == hipSuccess ? &gpuWord_ : nullptr;
      return mallocResult_;
    });
    Hook(g_hipThreadExchangeStreamCaptureMode, [this](hipStreamCaptureMode* mode) {
      if (captureFailAt_ == static_cast<int>(captureModes_.size())) return hipErrorInvalidValue;
      captureModes_.push_back(*mode);
      std::swap(*mode, threadMode_);
      return hipSuccess;
    });
    Hook(g_hipStreamCreateWithFlags, [this](hipStream_t* stream, unsigned flags) {
      streamFlags_.push_back(flags);
      *stream = streamResult_ == hipSuccess ? stream_ : nullptr;
      return streamResult_;
    });
    Hook(g_hipMemsetAsync, [this](void* dst, int value, size_t bytes, hipStream_t stream) {
      EXPECT_EQ(dst, &gpuWord_);
      EXPECT_EQ(value, 0);
      EXPECT_EQ(bytes, sizeof(int));
      EXPECT_EQ(stream, stream_);
      memsets_++;
      return memsetResult_;
    });
    Hook(g_hipStreamSynchronize, [this](hipStream_t stream) {
      syncs_++;
      return syncResult_;
    });
    Hook(g_hipStreamDestroy, [this](hipStream_t stream) {
      EXPECT_EQ(stream, stream_);
      destroys_++;
      return destroyResult_;
    });
    Hook(g_hipFree, [this](void* p) {
      freed_.push_back(p);
      return hipSuccess;
    });
    Hook(g_ibvRegDmabufMr, [this](ibv_mr** ret, ibv_pd* pd, uint64_t offset, size_t len, uint64_t iova, int fd,
                                  int access) {
      dmabufRegs_.push_back({offset, len, iova, fd, access});
      verbs_.mrs.emplace_back();
      *ret = &verbs_.mrs.back();
      verbs_.live.insert(*ret);
      return ncclSuccess;
    });
  }

  ncclIbRecvComm* Accepted() {
    ncclIbSendComm* s = nullptr;
    ncclIbRecvComm* r = nullptr;
    Connected(&s, &r);
    return r;
  }
  ncclResult_t TryAccept() {
    void* lc = nullptr;
    ncclIbHandle h = Listen(0, &lc);
    ncclIbSendComm* s = nullptr;
    ncclIbRecvComm* r = nullptr;
    return Handshake(&h, lc, &s, &r);
  }
  const FakeVerbs::Reg* RegOf(const void* addr) const {
    for (const auto& reg : verbs_.regs)
      if (reg.addr == addr) return &reg;
    return nullptr;
  }

  struct DmabufReg {
    uint64_t offset;
    size_t length;
    uint64_t iova;
    int fd;
    int access;
  };
  alignas(4096) int gpuWord_ = 7;
  hipStream_t stream_ = reinterpret_cast<hipStream_t>(0x5);
  hipStreamCaptureMode threadMode_ = hipStreamCaptureModeGlobal;
  std::vector<hipStreamCaptureMode> captureModes_;
  std::vector<unsigned> mallocFlags_, streamFlags_;
  std::vector<void*> freed_;
  std::vector<DmabufReg> dmabufRegs_;
  int memsets_ = 0, syncs_ = 0, destroys_ = 0;
  int captureFailAt_ = -1;
  hipError_t mallocResult_ = hipSuccess, streamResult_ = hipSuccess, memsetResult_ = hipSuccess,
             syncResult_ = hipSuccess, destroyResult_ = hipSuccess;
};

TEST_F(NetIbCastConnectFlushMicrotest, FlushQp_IsALoopbackQpOnTheReceiversDevice) {
  ncclIbRecvComm* r = Accepted();
  ASSERT_NE(r, nullptr);
  EXPECT_EQ(r->flushEnabled, 1);
  ibv_qp* fq = r->devs[0].gpuFlush.qp.qp;
  ASSERT_NE(fq, nullptr);
  const auto* create = verbs_.CreateOf(fq);
  EXPECT_EQ(create->attr.cap.max_recv_wr, 0u);
  EXPECT_EQ(create->attr.cap.max_send_wr, (uint32_t)NET_IB_MAX_REQUESTS);
  EXPECT_EQ(create->attr.send_cq, r->devs[0].base.cq);
  const auto mods = verbs_.ModifiesOf(fq);
  ASSERT_EQ(mods.size(), 3u);
  EXPECT_EQ(mods[0].attr.qp_access_flags,
            (unsigned)(IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE));
  EXPECT_EQ(mods[1].attr.dest_qp_num, fq->qp_num);
  EXPECT_TRUE(SameGid(mods[1].attr.ah_attr.grh.dgid, IbCastDevs[0].gidInfo.localGid));
  EXPECT_EQ(mods[1].attr.path_mtu, IBV_MTU_4096);
  EXPECT_EQ(mods[2].attr.qp_state, IBV_QPS_RTS);
  EXPECT_TRUE(r->devs[0].gpuFlush.qp.isDataQp);
}

TEST_F(NetIbCastConnectFlushMicrotest, GpuWord_IsAllocatedZeroedOnAPrivateStreamAndRegistered) {
  ncclIbRecvComm* r = Accepted();
  ASSERT_NE(r, nullptr);
#if defined(HIP_UNCACHED_MEMORY)
  EXPECT_EQ(mallocFlags_, std::vector<unsigned>{hipDeviceMallocUncached});
#else
  EXPECT_EQ(mallocFlags_, std::vector<unsigned>{hipDeviceMallocFinegrained});
#endif
  EXPECT_EQ(streamFlags_, std::vector<unsigned>{hipStreamNonBlocking});
  EXPECT_EQ(memsets_, 1);
  EXPECT_EQ(syncs_, 1);
  EXPECT_EQ(destroys_, 1);
  EXPECT_EQ(captureModes_, (std::vector<hipStreamCaptureMode>{hipStreamCaptureModeRelaxed,
                                                              hipStreamCaptureModeGlobal}));
  EXPECT_EQ(threadMode_, hipStreamCaptureModeGlobal);
  ncclIbGpuFlush& f = r->devs[0].gpuFlush;
  EXPECT_EQ(f.gpuFlushGpuMem, &gpuWord_);
  EXPECT_EQ(f.dmabufFd, -1);
  const auto* gpuReg = RegOf(&gpuWord_);
  ASSERT_NE(gpuReg, nullptr);
  EXPECT_EQ(gpuReg->mr, f.gpuMr);
  EXPECT_EQ(gpuReg->length, sizeof(int));
  EXPECT_EQ(gpuReg->access, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
  const auto* hostReg = RegOf(&r->gpuFlushHostMem);
  ASSERT_NE(hostReg, nullptr);
  EXPECT_EQ(hostReg->access, IBV_ACCESS_LOCAL_WRITE);
  EXPECT_EQ(f.sge.addr, reinterpret_cast<uint64_t>(&r->gpuFlushHostMem));
  EXPECT_EQ(f.sge.length, 1u);
  EXPECT_EQ(f.sge.lkey, f.hostMr->lkey);
}

TEST_F(NetIbCastConnectFlushMicrotest, Close_ReleasesTheGpuWordAndBothFlushMrs) {
  ncclIbRecvComm* r = Accepted();
  ASSERT_NE(r, nullptr);
  ibv_mr* gpuMr = r->devs[0].gpuFlush.gpuMr;
  ibv_mr* hostMr = r->devs[0].gpuFlush.hostMr;
  ibv_qp* fq = r->devs[0].gpuFlush.qp.qp;
  Forget(r);
  ASSERT_EQ(IbCastCloseRecv(r), ncclSuccess);
  EXPECT_EQ(freed_, std::vector<void*>{&gpuWord_});
  EXPECT_EQ(verbs_.live.count(gpuMr) + verbs_.live.count(hostMr) + verbs_.live.count(fq), 0u);
  EXPECT_TRUE(g_closedFds.empty());
}

TEST_F(NetIbCastConnectFlushMicrotest, DmaBuf_ExportsTheGpuWordAndClosesTheFdOnClose) {
  Hook(g_ibCastGdrSupport, [] { return ncclSystemError; });
  Hook(g_ibCastDmaBufSupport, [](int dev) {
    EXPECT_EQ(dev, 0);
    return ncclSuccess;
  });
  pfn_hsa_amd_portable_export_dmabuf = [](const void* ptr, size_t size, int* fd, uint64_t* offset) {
    *fd = 42;
    *offset = 0x10;
    return HSA_STATUS_SUCCESS;
  };
  ncclIbRecvComm* r = Accepted();
  ASSERT_NE(r, nullptr);
  ASSERT_EQ(dmabufRegs_.size(), 1u);
  EXPECT_EQ(dmabufRegs_[0].offset, 0x10u);
  EXPECT_EQ(dmabufRegs_[0].length, sizeof(int));
  EXPECT_EQ(dmabufRegs_[0].iova, reinterpret_cast<uint64_t>(&gpuWord_));
  EXPECT_EQ(dmabufRegs_[0].fd, 42);
  EXPECT_EQ(dmabufRegs_[0].access, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
  EXPECT_EQ(r->devs[0].gpuFlush.dmabufFd, 42);
  Forget(r);
  ASSERT_EQ(IbCastCloseRecv(r), ncclSuccess);
  EXPECT_EQ(g_closedFds, std::vector<int>{42});
}

TEST_F(NetIbCastConnectFlushMicrotest, DmaBufParamOff_FallsBackToGdrOrNoFlush) {
  Hook(g_ibCastDmaBufSupport, [](int) { return ncclSuccess; });
  Param("DMABUF_ENABLE", 0);
  ncclIbRecvComm* r = Accepted();
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(dmabufRegs_.empty());
  EXPECT_EQ(r->flushEnabled, 1);
  Hook(g_ibCastGdrSupport, [] { return ncclSystemError; });
  r = Accepted();
  ASSERT_NE(r, nullptr);
  EXPECT_EQ(r->flushEnabled, 0);
  EXPECT_EQ(r->devs[0].gpuFlush.qp.qp, nullptr);
}

// Pins AICOMRCCL-3081; a fix flips this pin.
TEST_F(NetIbCastConnectFlushMicrotest, Accept_KnownBug_DmaBufExportFailureReportsSuccess) {
  Hook(g_ibCastDmaBufSupport, [](int) { return ncclSuccess; });
  pfn_hsa_amd_portable_export_dmabuf = [](const void*, size_t, int* fd, uint64_t*) {
    *fd = -1;
    return HSA_STATUS_ERROR;
  };
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  void* sc = nullptr;
  void* rc = nullptr;
  ncclResult_t res = ncclInternalError;
  ASSERT_EQ(IbCastConnect(nullptr, 0, &h, &sc, nullptr), ncclSuccess);
  ASSERT_EQ(IbCastAccept(lc, &rc, nullptr), ncclSuccess);
  ASSERT_EQ(IbCastConnect(nullptr, 0, &h, &sc, nullptr), ncclSuccess);
  const std::string log = CaptureLog([&] { res = IbCastAccept(lc, &rc, nullptr); });
  EXPECT_TRUE(LogHas(log, "Failed to export DMA BUF")) << log;
  // The receiver comm is gone, yet the caller is told to keep polling.
  EXPECT_EQ(res, ncclSuccess);
  EXPECT_EQ(rc, nullptr);
  EXPECT_EQ(static_cast<ncclIbListenComm*>(lc)->stage, nullptr);
  Abandon(&h);
}

TEST_F(NetIbCastConnectFlushMicrotest, NoRelaxedOrderingOff_SkipsTheGpuWord) {
  Param("RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING", 0);
  ncclIbRecvComm* r = Accepted();
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(mallocFlags_.empty());
  EXPECT_EQ(r->devs[0].gpuFlush.gpuFlushGpuMem, nullptr);
  EXPECT_EQ(r->devs[0].gpuFlush.gpuMr, nullptr);
  EXPECT_NE(r->devs[0].gpuFlush.hostMr, nullptr);
  Forget(r);
  ASSERT_EQ(IbCastCloseRecv(r), ncclSuccess);
  EXPECT_TRUE(freed_.empty());
}

TEST_F(NetIbCastConnectFlushMicrotest, OffloadOrTheDisableParam_TurnsFlushOff) {
  Param("GDR_FLUSH_DISABLE", 1);
  ncclIbRecvComm* r = Accepted();
  ASSERT_NE(r, nullptr);
  EXPECT_EQ(r->flushEnabled, 0);
  Param("GDR_FLUSH_DISABLE", 0);
  IbCastOffloadEnabled = true;
  r = Accepted();
  ASSERT_NE(r, nullptr);
  EXPECT_EQ(r->flushEnabled, 0);
}

TEST_F(NetIbCastConnectFlushMicrotest, GpuWordFailures_FailTheAcceptAndRestoreTheCaptureMode) {
  struct Case {
    const char* what;
    std::function<void()> inject;
    int destroys;
  };
  const std::vector<Case> cases = {
      {"malloc", [this] { mallocResult_ = hipErrorOutOfMemory; }, 0},
      {"capture", [this] { captureFailAt_ = 0; }, 0},
      {"stream", [this] { streamResult_ = hipErrorInvalidValue; }, 0},
      {"memset", [this] { memsetResult_ = hipErrorInvalidValue; }, 1},
      {"sync", [this] { syncResult_ = hipErrorInvalidValue; }, 1},
      {"destroy", [this] { destroyResult_ = hipErrorInvalidValue; }, 1},
      {"restore", [this] { captureFailAt_ = 1; }, 1},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.what);
    mallocResult_ = streamResult_ = memsetResult_ = syncResult_ = destroyResult_ = hipSuccess;
    captureFailAt_ = -1;
    captureModes_.clear();
    destroys_ = 0;
    threadMode_ = hipStreamCaptureModeGlobal;
    IbCastDevs[0].pdRefs = 0;
    c.inject();
    EXPECT_EQ(TryAccept(), ncclUnhandledCudaError);
    EXPECT_EQ(destroys_, c.destroys);
    if (captureFailAt_ < 0) EXPECT_EQ(threadMode_, hipStreamCaptureModeGlobal);
  }
}

TEST_F(NetIbCastConnectFlushMicrotest, FlushMrFailure_FailsTheAccept) {
  // Registrations in order: sender scratchpad, sender CTS FIFO, receiver CTS FIFO,
  // receiver records, receiver GPU word, receiver host word.
  for (int n : {4, 5}) {
    SCOPED_TRACE(n);
    verbs_.regs.clear();
    regMrFailAt_ = n;
    IbCastDevs[0].pdRefs = 0;
    EXPECT_EQ(TryAccept(), ncclSystemError);
  }
  regMrFailAt_ = -1;
  verbs_.regs.clear();
  Hook(g_ibCastDmaBufSupport, [](int) { return ncclSuccess; });
  pfn_hsa_amd_portable_export_dmabuf = [](const void*, size_t, int* fd, uint64_t*) {
    *fd = 3;
    return HSA_STATUS_SUCCESS;
  };
  Hook(g_ibvRegDmabufMr, [](ibv_mr** ret, ibv_pd*, uint64_t, size_t, uint64_t, int, int) {
    *ret = nullptr;
    return ncclSystemError;
  });
  EXPECT_EQ(TryAccept(), ncclSystemError);
}

TEST_F(NetIbCastConnectFlushMicrotest, FlushQpCreateFailure_FailsTheAccept) {
  int creates = 0;
  Hook(g_ibvCreateQp, [&](ibv_qp** ret, ibv_pd* pd, ibv_qp_init_attr* attr) {
    if (++creates == 5) return ncclSystemError;  // after the two data QPs per side
    *ret = verbs_.NewQp(pd, attr->qp_type);
    verbs_.creates.push_back({*ret, pd, *attr});
    return ncclSuccess;
  });
  EXPECT_EQ(TryAccept(), ncclSystemError);
  EXPECT_EQ(creates, 5);
}

class NetIbCastConnectCloseMicrotest : public NetIbCastConnectMicrotest {};

TEST_F(NetIbCastConnectCloseMicrotest, NullComms_AreANoop) {
  EXPECT_EQ(IbCastCloseSend(nullptr), ncclSuccess);
  EXPECT_EQ(IbCastCloseRecv(nullptr), ncclSuccess);
}

TEST_F(NetIbCastConnectCloseMicrotest, PartlyBuiltComms_SkipMissingQpsAndMrs) {
  ncclIbNetCommDevBase base = {};
  ASSERT_EQ(IbCastInitCommDevBase(0, &base, nullptr, 8), ncclSuccess);
  auto* s = NewComm<ncclIbSendComm>();
  s->base.vProps.ndevs = 1;
  s->base.nqps = 2;
  s->devs[0].base = base;
  ASSERT_EQ(IbCastCloseSend(s), ncclSuccess);
  EXPECT_TRUE(verbs_.live.empty());

  ASSERT_EQ(IbCastInitCommDevBase(0, &base, nullptr, 8), ncclSuccess);
  auto* r = NewComm<ncclIbRecvComm>();
  r->base.vProps.ndevs = 1;
  r->base.nqps = 2;
  r->flushEnabled = 1;
  r->devs[0].base = base;
  ASSERT_EQ(IbCastCloseRecv(r), ncclSuccess);
  EXPECT_TRUE(verbs_.live.empty());
}

TEST_F(NetIbCastConnectCloseMicrotest, Telemetry_FoldsTheCommsCqPollsIntoItsFirstDevice) {
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  rcclTelemetryEnabled = 1;
  const uint64_t before = rcclTelemetryDevs[0].cq_poll_count;
  s->base.telCqPollCount = 5;
  r->base.telCqPollCount = 7;
  Forget(s);
  Forget(r);
  ASSERT_EQ(IbCastCloseSend(s), ncclSuccess);
  ASSERT_EQ(IbCastCloseRecv(r), ncclSuccess);
  EXPECT_EQ(rcclTelemetryDevs[0].cq_poll_count - before, 12u);
}

TEST_F(NetIbCastConnectCloseMicrotest, DestroyFailure_StopsTheClose) {
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connected(&s, &r));
  Forget(s);
  Forget(r);
  ScopedHook qp(g_ibvDestroyQp, [](ibv_qp*) { return ncclSystemError; });
  EXPECT_EQ(IbCastCloseSend(s), ncclSystemError);
  EXPECT_EQ(IbCastCloseRecv(r), ncclSystemError);
  // A failed close returns before it frees the comm.
  free(s);
  free(r);
}

class NetIbCastConnectSharingMicrotest : public NetIbCastConnectMicrotest {
 protected:
  void SetUp() override {
    NetIbCastConnectMicrotest::SetUp();
    IbCastQpSharingGlobalEnable = true;
  }
  // A connection on a fresh listener on merged device `listenDev`, sent from `dev`.
  void Connect(ncclIbSendComm** s, ncclIbRecvComm** r, int dev = 0, int listenDev = 0) {
    void* lc = nullptr;
    ncclIbHandle h = Listen(listenDev, &lc);
    ASSERT_EQ(Handshake(&h, lc, s, r, dev), ncclSuccess);
  }
  IbCastSharedQp* Slot(ibv_qp* qp, bool isSend) { return IbCastFindSharedQpByQpn(qp->qp_num, isSend); }
  int UsedSlots() {
    int n = 0;
    for (int i = 0; i < g_IbCastSharedQpPoolCount; i++) n += g_IbCastSharedQpPool[i].used;
    return n;
  }
  void CloseAll(std::initializer_list<void*> comms) {
    for (void* c : comms) {
      Forget(c);
      bool isSend = static_cast<ncclIbNetCommBase*>(c)->isSend;
      EXPECT_EQ(isSend ? IbCastCloseSend(c) : IbCastCloseRecv(c), ncclSuccess);
    }
  }
};

TEST_F(NetIbCastConnectSharingMicrotest, FirstConnection_MakesEachSideAPrimaryOwner) {
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connect(&s, &r));
  EXPECT_TRUE(IbCastCommIsPrimary(&s->base));
  EXPECT_TRUE(IbCastCommIsPrimary(&r->base));
  EXPECT_EQ(s->base.qpSharing.netIbCommId, 1);
  EXPECT_EQ(r->base.qpSharing.netIbCommId, 2);
  EXPECT_EQ(s->remCommId, 2);
  EXPECT_EQ(s->base.qpSharing.groupIdx, 0);
  EXPECT_EQ(r->base.qpSharing.groupIdx, 0);
  EXPECT_EQ(s->base.qpSharing.peerProcTag, IbCastLocalProcTag());
  EXPECT_EQ(s->base.recvMatchingScheme, BY_ID);
  EXPECT_EQ(r->base.recvMatchingScheme, BY_ID);
  EXPECT_FALSE(r->useCtsOffload);
  EXPECT_EQ(UsedSlots(), 4);
  for (int q = 0; q < 2; q++) {
    SCOPED_TRACE(q);
    IbCastSharedQp* ss = Slot(s->base.qps[q].qp, true);
    IbCastSharedQp* rs = Slot(r->base.qps[q].qp, false);
    ASSERT_NE(ss, nullptr);
    ASSERT_NE(rs, nullptr);
    EXPECT_EQ(ss->key.qpIdx, q);
    EXPECT_EQ(ss->refcount, 1);
    EXPECT_EQ(ss->cqRefcount, q == 0 ? 1 : 0);
    EXPECT_EQ(ss->primaryCq, s->devs[0].base.cq);
    EXPECT_EQ(rs->primaryCq, r->devs[0].base.cq);
    EXPECT_EQ(ss->key.peerAddr.sin.sin_port, 0);
    EXPECT_EQ(ss->key.peerAddr.sin.sin_addr.s_addr, sockets_.listenAddr.sin.sin_addr.s_addr);
    EXPECT_EQ(rs->key.peerAddr.sin.sin_addr.s_addr, sockets_.nextClientAddr.sin.sin_addr.s_addr);
  }
  const int cq = 3 * NET_IB_MAX_REQUESTS * ncclParamIbCastQpsPerConn() * rcclParamIbCastQpDepthMultiplier();
  EXPECT_EQ(verbs_.cqSizes, (std::vector<int>{cq, cq}));
}

TEST_F(NetIbCastConnectSharingMicrotest, Ainic_ScalesSharedQpDepthByTheMultiplierTheCqAchieved) {
  ASSERT_GT(rcclParamIbCastQpDepthMultiplier(), 3);
  Ainic();
  cqeCap_ = 3 * NET_IB_MAX_REQUESTS * ncclParamIbCastQpsPerConn() * 3;  // the device grants 3x, less than asked
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connect(&s, &r));
  EXPECT_EQ(verbs_.CreateOf(s->base.qps[0].qp)->attr.cap.max_send_wr, 2u * NET_IB_MAX_REQUESTS * 3);
  EXPECT_EQ(verbs_.CreateOf(r->base.qps[0].qp)->attr.cap.max_recv_wr, (uint32_t)NET_IB_MAX_REQUESTS * 3);
  cqeCap_ = 1;  // below one multiple: the depth never drops under 1x
  ASSERT_NO_FATAL_FAILURE(Connect(&s, &r));
  EXPECT_EQ(verbs_.CreateOf(s->base.qps[0].qp)->attr.cap.max_send_wr, 2u * NET_IB_MAX_REQUESTS);
}

TEST_F(NetIbCastConnectSharingMicrotest, OneGroup_SecondConnectionReusesTheQpsAndCqs) {
  Param("RCCL_IB_COMM_NGROUPS", 1);
  ncclIbSendComm *s1 = nullptr, *s2 = nullptr;
  ncclIbRecvComm *r1 = nullptr, *r2 = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connect(&s1, &r1));
  const size_t creates = verbs_.creates.size();
  const size_t modifies = verbs_.modifies.size();
  ASSERT_NO_FATAL_FAILURE(Connect(&s2, &r2));
  EXPECT_EQ(verbs_.creates.size(), creates);
  EXPECT_EQ(verbs_.modifies.size(), modifies);
  EXPECT_TRUE(IbCastCommIsSecondary(&s2->base));
  EXPECT_TRUE(IbCastCommIsSecondary(&r2->base));
  EXPECT_EQ(s2->base.qpSharing.groupNqps, 2);
  EXPECT_EQ(r2->base.qpSharing.groupNqps, 2);
  EXPECT_EQ(s2->remCommId, r2->base.qpSharing.netIbCommId);
  for (int q = 0; q < 2; q++) {
    SCOPED_TRACE(q);
    EXPECT_EQ(s2->base.qps[q].qp, s1->base.qps[q].qp);
    EXPECT_EQ(r2->base.qps[q].qp, r1->base.qps[q].qp);
    EXPECT_EQ(s2->base.activeQps[q], &s2->base.qps[q]);
    EXPECT_EQ(r2->base.qps[q].remDevIdx, 0);
    EXPECT_EQ(Slot(s1->base.qps[q].qp, true)->refcount, 2);
    EXPECT_EQ(Slot(r1->base.qps[q].qp, false)->refcount, 2);
  }
  // The secondaries' own CQs and PD references were given back at once.
  EXPECT_EQ(s2->devs[0].base.cq, s1->devs[0].base.cq);
  EXPECT_EQ(r2->devs[0].base.cq, r1->devs[0].base.cq);
  EXPECT_EQ(verbs_.cqs.size() - std::count_if(verbs_.cqs.begin(), verbs_.cqs.end(),
                                              [&](const ibv_cq& cq) { return verbs_.live.count(&cq); }),
            2u);
  EXPECT_EQ(IbCastDevs[0].pdRefs, 2);
  EXPECT_EQ(Slot(s1->base.qps[0].qp, true)->cqRefcount, 2);
  EXPECT_EQ(UsedSlots(), 4);
}

TEST_F(NetIbCastConnectSharingMicrotest, Close_SharedResourcesOutliveTheirFirstOwner) {
  Param("RCCL_IB_COMM_NGROUPS", 1);
  for (bool primaryFirst : {true, false}) {
    SCOPED_TRACE(primaryFirst);
    ncclIbSendComm *s1 = nullptr, *s2 = nullptr;
    ncclIbRecvComm *r1 = nullptr, *r2 = nullptr;
    ASSERT_NO_FATAL_FAILURE(Connect(&s1, &r1));
    ASSERT_NO_FATAL_FAILURE(Connect(&s2, &r2));
    ibv_qp* shared = s1->base.qps[0].qp;
    ibv_cq* cq = s1->devs[0].base.cq;
    if (primaryFirst) CloseAll({s1, r1});
    else CloseAll({s2, r2});
    EXPECT_EQ(verbs_.live.count(shared), 1u);
    EXPECT_EQ(verbs_.live.count(cq), 1u);
    if (primaryFirst) CloseAll({s2, r2});
    else CloseAll({s1, r1});
    EXPECT_TRUE(verbs_.live.empty());
    EXPECT_EQ(IbCastDevs[0].pdRefs, 0);
    EXPECT_EQ(UsedSlots(), 0);
    std::set<const void*> unique(verbs_.destroyed.begin(), verbs_.destroyed.end());
    EXPECT_EQ(unique.size(), verbs_.destroyed.size());
    EXPECT_EQ(std::count_if(std::begin(g_IbCastCommTable), std::end(g_IbCastCommTable),
                            [](const IbCastCommTableEntry& e) { return e.used; }),
              0);
    verbs_.destroyed.clear();
  }
}

TEST_F(NetIbCastConnectSharingMicrotest, DefaultGroups_RotateConnectionsAcrossFourGroups) {
  std::vector<ncclIbSendComm*> s(5);
  std::vector<ncclIbRecvComm*> r(5);
  for (int i = 0; i < 5; i++) ASSERT_NO_FATAL_FAILURE(Connect(&s[i], &r[i]));
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(s[i]->base.qpSharing.groupIdx, i);
    EXPECT_EQ(r[i]->base.qpSharing.groupIdx, i);
    EXPECT_TRUE(IbCastCommIsPrimary(&s[i]->base));
  }
  EXPECT_EQ(s[4]->base.qpSharing.groupIdx, 0);
  EXPECT_TRUE(IbCastCommIsSecondary(&s[4]->base));
  EXPECT_EQ(s[4]->base.qps[1].qp, s[0]->base.qps[1].qp);
}

TEST_F(NetIbCastConnectSharingMicrotest, SharedFlushQp_IsReusedAndReleasedWithItsLastUser) {
  Param("RCCL_IB_COMM_NGROUPS", 1);
  Param("GDR_FLUSH_DISABLE", 0);
  Param("RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING", 0);
  Hook(g_ibCastGdrSupport, [] { return ncclSuccess; });
  ncclIbSendComm *s1 = nullptr, *s2 = nullptr;
  ncclIbRecvComm *r1 = nullptr, *r2 = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connect(&s1, &r1));
  ibv_qp* flush = r1->devs[0].gpuFlush.qp.qp;
  IbCastSharedQp* slot = Slot(flush, false);
  ASSERT_NE(slot, nullptr);
  EXPECT_EQ(slot->key.qpIdx, IBCAST_FLUSH_QP_IDX);
  EXPECT_EQ(slot->devIndex, 0);
  const std::string log = InfoLog([&] { Connect(&s2, &r2); });
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(r2->devs[0].gpuFlush.qp.qp, flush);
  EXPECT_EQ(slot->refcount, 2);
  EXPECT_TRUE(LogHas(log, "SECONDARY recv sharing flush QP")) << log;
  CloseAll({s1, r1});
  EXPECT_EQ(verbs_.live.count(flush), 1u);
  CloseAll({s2, r2});
  EXPECT_EQ(verbs_.live.count(flush), 0u);
  EXPECT_FALSE(slot->used);
  EXPECT_TRUE(verbs_.live.empty());
}

TEST_F(NetIbCastConnectSharingMicrotest, SecondaryWithoutASharedFlushQp_WarnsAndRunsWithout) {
  Param("RCCL_IB_COMM_NGROUPS", 1);
  ncclIbSendComm *s1 = nullptr, *s2 = nullptr;
  ncclIbRecvComm *r1 = nullptr, *r2 = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connect(&s1, &r1));
  Param("GDR_FLUSH_DISABLE", 0);
  Param("RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING", 0);
  Hook(g_ibCastGdrSupport, [] { return ncclSuccess; });
  const std::string log = CaptureLog([&] { Connect(&s2, &r2); });
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_TRUE(LogHas(log, "SECONDARY recv could not find shared flush QP for dev=0 group=0")) << log;
  EXPECT_EQ(r2->devs[0].gpuFlush.qp.qp, nullptr);
  CloseAll({s1, r1, s2, r2});
  EXPECT_TRUE(verbs_.live.empty());
}

TEST_F(NetIbCastConnectSharingMicrotest, CommIdsExhausted_FallBackToPrivateQps) {
  g_IbCastNextCommId = IBCAST_MAX_COMMS;
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  const std::string log = CaptureLog([&] { Connect(&s, &r); });
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_TRUE(LogHas(log, "commId pool exhausted")) << log;
  EXPECT_FALSE(IbCastCommIsSharing(&s->base));
  EXPECT_FALSE(IbCastCommIsSharing(&r->base));
  EXPECT_EQ(UsedSlots(), 0);
  // Only the receiver runs out: the sender still shares, the receiver does not.
  g_IbCastNextCommId = IBCAST_MAX_COMMS - 1;
  ASSERT_NO_FATAL_FAILURE(Connect(&s, &r));
  EXPECT_TRUE(IbCastCommIsPrimary(&s->base));
  EXPECT_FALSE(IbCastCommIsSharing(&r->base));
  EXPECT_EQ(s->remCommId, 0);
  EXPECT_EQ(UsedSlots(), 2);
}

TEST_F(NetIbCastConnectSharingMicrotest, Rma_NeverShares) {
  Param("RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING", 0);
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  h.isRMA = true;
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  ASSERT_EQ(Handshake(&h, lc, &s, &r), ncclSuccess);
  EXPECT_FALSE(IbCastCommIsSharing(&s->base));
  EXPECT_FALSE(IbCastCommIsSharing(&r->base));
  EXPECT_EQ(UsedSlots(), 0);
}

TEST_F(NetIbCastConnectSharingMicrotest, PoolExhausted_FailsThePrimaryThatCannotRegister) {
  g_IbCastSharedQpPoolCount = IBCAST_MAX_SHARED_QPS - 1;
  void* lc = nullptr;
  ncclIbHandle h = Listen(0, &lc);
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  std::string log = CaptureLog([&] { EXPECT_EQ(Handshake(&h, lc, &s, &r), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "PRIMARY sender commId=1 group=0: shared-QP pool exhausted registering qpIdx=1/2")) << log;

  ResetQpSharing();
  IbCastQpSharingGlobalEnable = true;
  IbCastDevs[0].pdRefs = 0;
  g_IbCastSharedQpPoolCount = IBCAST_MAX_SHARED_QPS - 3;
  h = Listen(0, &lc);
  log = CaptureLog([&] { EXPECT_EQ(Handshake(&h, lc, &s, &r), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "PRIMARY receiver commId=2 group=0: shared-QP pool exhausted registering qpIdx=1/2"))
      << log;
}

TEST_F(NetIbCastConnectSharingMicrotest, FlushSlotExhausted_KeepsAPrivateFlushQp) {
  Param("GDR_FLUSH_DISABLE", 0);
  Param("RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING", 0);
  Hook(g_ibCastGdrSupport, [] { return ncclSuccess; });
  g_IbCastSharedQpPoolCount = IBCAST_MAX_SHARED_QPS - 4;
  ncclIbSendComm* s = nullptr;
  ncclIbRecvComm* r = nullptr;
  const std::string log = CaptureLog([&] { Connect(&s, &r); });
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_TRUE(LogHas(log, "secondaries will not be able to share it")) << log;
  ibv_qp* flush = r->devs[0].gpuFlush.qp.qp;
  ASSERT_NE(flush, nullptr);
  EXPECT_EQ(Slot(flush, false), nullptr);
  CloseAll({s, r});
  EXPECT_EQ(verbs_.live.count(flush), 0u);
}

TEST_F(NetIbCastConnectSharingMicrotest, ReceiverMissingASharedSlot_FallsBackToPrivateQps) {
  Param("RCCL_IB_COMM_NGROUPS", 1);
  ncclIbSendComm *s1 = nullptr, *s2 = nullptr;
  ncclIbRecvComm *r1 = nullptr, *r2 = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connect(&s1, &r1));
  // Re-key the receiver's second slot so the group still counts it but a lookup misses it.
  Slot(r1->base.qps[1].qp, false)->key.ibDevN = 2;
  const std::string log = CaptureLog([&] { Connect(&s2, &r2); });
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_TRUE(LogHas(log, "SECONDARY recv: could not find shared QP for qpIdx=1 group=0")) << log;
  EXPECT_FALSE(IbCastCommIsSharing(&r2->base));
  EXPECT_EQ(r2->base.qpSharing.groupIdx, -1);
  EXPECT_NE(r2->base.qps[0].qp, r1->base.qps[0].qp);
  EXPECT_FALSE(g_IbCastCommTable[4].used);
}

// Pins AICOMRCCL-3082; a fix flips this pin.
TEST_F(NetIbCastConnectSharingMicrotest, CloseSend_KnownBug_QpNumberCollisionAcrossNicsReleasesAnotherCommsQps) {
  // Two primaries on different NICs; each HCA numbers its QPs from the same base.
  ncclIbSendComm *a = nullptr, *b = nullptr;
  ncclIbRecvComm *ra = nullptr, *rb = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connect(&a, &ra, /*dev=*/0, /*listenDev=*/0));
  ASSERT_NO_FATAL_FAILURE(Connect(&b, &rb, /*dev=*/1, /*listenDev=*/1));
  ASSERT_TRUE(IbCastCommIsPrimary(&b->base));
  ASSERT_EQ(a->base.qps[0].qp->qp_num, b->base.qps[0].qp->qp_num);
  ibv_qp* aQp = a->base.qps[0].qp;
  ibv_qp* bQp = b->base.qps[0].qp;
  ibv_cq* aCq = a->devs[0].base.cq;
  CloseAll({b});
  // Closing b tore down a's QP and CQ while a is still open, and left b's own QP behind.
  EXPECT_EQ(verbs_.live.count(aQp), 0u);
  EXPECT_EQ(verbs_.live.count(aCq), 0u);
  EXPECT_EQ(verbs_.live.count(bQp), 1u);
}

// Pins AICOMRCCL-3083; a fix flips this pin.
TEST_F(NetIbCastConnectSharingMicrotest, CleanupGroupCqs_KnownBug_ReleasesTheGroupsCqsOnOtherLocalNics) {
  Param("RCCL_IB_COMM_NGROUPS", 1);
  // Two primaries of one group, on two local NICs, to the same remote NIC.
  ncclIbSendComm *a = nullptr, *b = nullptr;
  ncclIbRecvComm *ra = nullptr, *rb = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connect(&a, &ra, /*dev=*/0, /*listenDev=*/0));
  ASSERT_NO_FATAL_FAILURE(Connect(&b, &rb, /*dev=*/1, /*listenDev=*/0));
  ASSERT_TRUE(IbCastCommIsPrimary(&b->base));
  ASSERT_EQ(b->base.qpSharing.groupIdx, a->base.qpSharing.groupIdx);
  ibv_cq* bCq = b->devs[0].base.cq;
  ibv_pd* bPd = b->devs[0].base.pd;
  CloseAll({a});
  // b is still open, but a's last close destroyed b's CQ and dropped b's PD.
  EXPECT_EQ(verbs_.live.count(bCq), 0u);
  EXPECT_EQ(verbs_.live.count(bPd), 0u);
}

// Pins AICOMRCCL-3084; a fix flips this pin.
TEST_F(NetIbCastConnectSharingMicrotest, SenderSetup_KnownBug_FallbackLeaksTheSlotsItAlreadyReferenced) {
  Param("RCCL_IB_COMM_NGROUPS", 1);
  ncclIbSendComm *a = nullptr, *c = nullptr;
  ncclIbRecvComm *ra = nullptr, *rc = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connect(&a, &ra, /*dev=*/0));
  ibv_qp* shared = a->base.qps[0].qp;
  // A fused sender finds slot 0 of the group but not slot 1, and falls back.
  const std::string log = CaptureLog([&] { Connect(&c, &rc, /*dev=*/2); });
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_TRUE(LogHas(log, "QP sharing SECONDARY: could not find shared QP for qpIdx=1")) << log;
  ASSERT_FALSE(IbCastCommIsSharing(&c->base));
  EXPECT_EQ(Slot(shared, true)->refcount, 2);
  ibv_cq* cq = a->devs[0].base.cq;
  CloseAll({a, ra, c, rc});
  // Every comm is closed, yet the reference the fallback took keeps the QP alive past its CQ.
  EXPECT_EQ(verbs_.live.count(shared), 1u);
  EXPECT_EQ(verbs_.live.count(cq), 0u);
}

// Pins AICOMRCCL-3085; a fix flips this pin.
TEST_F(NetIbCastConnectQpMicrotest, InitCommDevBase_KnownBug_FailedPdAllocationLeavesAReferenceToNull) {
  ncclIbNetCommDevBase a = {}, b = {};
  allocPdResult_ = ncclSystemError;
  EXPECT_EQ(IbCastInitCommDevBase(0, &a, nullptr, 8), ncclSystemError);
  allocPdResult_ = ncclSuccess;
  EXPECT_EQ(IbCastDevs[0].pdRefs, 1);
  // The next comm on the device trusts the count and gets no PD at all.
  ASSERT_EQ(IbCastInitCommDevBase(0, &b, nullptr, 8), ncclSuccess);
  EXPECT_EQ(b.pd, nullptr);
  EXPECT_TRUE(verbs_.pds.empty());
  IbCastDevs[0].pdRefs = 0;
  verbs_.Release(b.cq);
}

TEST_F(NetIbCastConnectGidMicrotest, Roce_ExplicitInetFamilyMatchesTheDefault) {
  SetMicroEnv("NCCL_IB_ADDR_FAMILY", "AF_INET");
  Table({{V4Gid("10.0.0.1"), "RoCE v2"}, {V6Gid("2001:db8::1"), "RoCE v2"}});
  EXPECT_EQ(Pick(), 1);
}

TEST_F(NetIbCastConnectQpMicrotest, QpStripe_RejectsADeviceCountBelowOne) {
  int dev = -1, idx = -1;
  ASSERT_EQ(IbCastQpStripe(5, 2, &dev, &idx), ncclSuccess);
  EXPECT_EQ(dev, 1);
  EXPECT_EQ(idx, 2);
  const std::string log = CaptureLog([&] { EXPECT_EQ(IbCastQpStripe(5, 0, &dev, &idx), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "invalid device count 0 while striping QP 5")) << log;
}

TEST_F(NetIbCastConnectQpMicrotest, ReceiverOooRq_RequiresPeerSupport) {
  Param("IB_OOO_RQ", 1);
  IbCastDevs[0].ibProvider = IB_PROVIDER_MLX5;
  IbCastDevs[0].ar = 1;
  auto* r = NewComm<ncclIbRecvComm>();
  r->base.nqps = 1;
  r->base.vProps.ndevs = 1;
  r->base.localOooRq = true;
  ncclIbConnectionMetadata rem = {}, meta = {};
  const std::string log =
      CaptureLog([&] { EXPECT_EQ(IbCastReceiverQpsCreateToRts(r, &rem, &meta, 0, 1), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "not supported on both sides of the connection (device=fake0, localOooRq=1, remOooRq=0)"))
      << log;
  free(r);
}

// A secondary on a NIC its primary does not use holds the only reference to that NIC's PD.
TEST_F(NetIbCastConnectSharingMicrotest, SecondaryAloneOnItsNic_DeallocatesThePdItGivesBack) {
  Param("RCCL_IB_COMM_NGROUPS", 1);
  ibv_qp qp = {};
  qp.qp_num = 0x300;
  ibv_cq primaryCq = {};
  ncclIbNetCommDevBase own = {};
  ASSERT_EQ(IbCastInitCommDevBase(1, &own, nullptr, 8), ncclSuccess);
  ibv_pd* pd = own.pd;

  auto* s = NewComm<ncclIbSendComm>();
  InitBase(&s->base, true);
  IbCastCommInitSharingFields(&s->base);
  s->base.vProps.ndevs = 1;
  s->base.vProps.devs[0] = 1;
  s->base.nqps = 1;
  s->devs[0].base = own;
  ncclIbHandle h = {};
  h.connectAddr = V4Addr("10.0.0.1", 7000);
  ncclNetVDeviceProps_t remote = {};
  remote.ndevs = 1;
  IbCastSharedQpKey key = {};
  key.ibDevN = 1;
  key.peerAddr = V4Addr("10.0.0.1", 0);
  key.isSend = true;
  ASSERT_NE(IbCastRegisterSharedQp(&key, &qp, &primaryCq, 0, 0, 1), nullptr);
  ncclIbConnectionMetadata meta = {};
  IbCastQpSharingRole role = QP_SHARING_NONE;
  ASSERT_EQ(IbCastQpSharingSenderSetup(s, &h, &meta, &remote, 1, &role), ncclSuccess);
  EXPECT_EQ(role, QP_SHARING_SECONDARY);
  EXPECT_EQ(meta.qpInfo[0].qpn, 0x300u);
  EXPECT_EQ(s->devs[0].base.cq, &primaryCq);
  EXPECT_EQ(IbCastDevs[1].pdRefs, 0);
  EXPECT_EQ(verbs_.live.count(pd), 0u);
  free(s);

  ASSERT_EQ(IbCastInitCommDevBase(1, &own, nullptr, 8), ncclSuccess);
  auto* r = NewComm<ncclIbRecvComm>();
  InitBase(&r->base, false);
  IbCastCommInitSharingFields(&r->base);
  r->base.vProps.ndevs = 1;
  r->base.vProps.devs[0] = 1;
  r->base.nqps = 1;
  r->devs[0].base = own;
  r->base.sock.addr = V4Addr("10.0.0.2", 4000);
  key.isSend = false;
  key.peerAddr = V4Addr("10.0.0.2", 0);
  ASSERT_NE(IbCastRegisterSharedQp(&key, &qp, &primaryCq, 0, 0, 1), nullptr);
  ncclIbConnectionMetadata remMeta = {};
  remMeta.sharedGroupIdx = 0;
  ASSERT_EQ(IbCastQpSharingReceiverSetup(r, &remMeta, &meta, &role), ncclSuccess);
  EXPECT_EQ(role, QP_SHARING_SECONDARY);
  EXPECT_EQ(IbCastDevs[1].pdRefs, 0);
  EXPECT_TRUE(verbs_.live.empty());
  free(r);
}

// Pins AICOMRCCL-3087; a fix flips this pin.
TEST_F(NetIbCastConnectResiliencyMicrotest, SenderQpsToRts_KnownBug_SharingSecondaryTransitionsResiliencyQpsNeverCreated) {
  IbCastQpSharingGlobalEnable = true;
  Param("RCCL_IB_COMM_NGROUPS", 1);
  int creates = 0, toRts = 0;
  Hook(g_ibCastResiliencySenderCreateQps, [&](ncclIbResiliency*, ncclIbResiliencyInfo* info) {
    creates++;
    info->probingQpsInfo[0].qpn = 0xabc;
    return ncclSuccess;
  });
  Hook(g_ibCastResiliencySenderQpsToRts, [&](ncclIbResiliency*, ncclIbConnectionMetadata* rem) {
    toRts++;
    seenByReceiver_ = rem->resiliencyInfo.portRecoveryQpsInfo[0].qpn;
    return ncclSuccess;
  });
  ncclIbSendComm *s1 = nullptr, *s2 = nullptr;
  ncclIbRecvComm *r1 = nullptr, *r2 = nullptr;
  ASSERT_NO_FATAL_FAILURE(Connected(&s1, &r1));
  ASSERT_NO_FATAL_FAILURE(Connected(&s2, &r2));
  ASSERT_TRUE(IbCastCommIsSecondary(&s2->base));
  ASSERT_TRUE(IbCastCommIsSecondary(&r2->base));
  // The secondary sender never created resiliency QPs and its secondary peer sent none,
  // yet the sender still moves its (absent) probing QPs to RTS against QP number 0.
  EXPECT_EQ(creates, 1);
  EXPECT_EQ(toRts, 2);
  EXPECT_EQ(seenByReceiver_, 0u);
}

}  // namespace
