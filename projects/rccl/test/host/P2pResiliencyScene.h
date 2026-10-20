/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Shared scene for the net_ib_cast resiliency microtests: a fake verbs context
// whose ops table forwards to per-test hooks, heap-allocated send/recv comms, and
// the connect.cc / p2p.cc / common.cc seams wired to record what the unit asked for.

#ifndef RCCL_TEST_HOST_P2P_RESILIENCY_SCENE_H_
#define RCCL_TEST_HOST_P2P_RESILIENCY_SCENE_H_

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ScopedHook.h"
#include "fakes/ibvwrap_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/net_ib_cast_capability_fakes.h"
#include "fakes/net_ib_cast_common_fakes.h"
#include "fakes/net_ib_cast_connect_fakes.h"
#include "fakes/net_ib_cast_init_fakes.h"
#include "fakes/net_ib_cast_p2p_fakes.h"
#include "fakes/net_ib_cast_qp_sharing_fakes.h"
#include "p2p_cast.h"
#include "p2p_resiliency_cast.h"
#include "qp_sharing.h"

// The units read time through clockNano, which each test TU redirects here. Starts an hour
// in, like a steady clock on a host that has been up a while.
inline std::atomic<uint64_t> g_p2pResiliencyNowNs{3600ull * 1000000000ull};
inline uint64_t P2pResiliencyTestClockNano() { return g_p2pResiliencyNowNs.load(); }
inline void P2pResiliencyAdvanceMs(uint64_t ms) { g_p2pResiliencyNowNs += ms * 1000000ull; }

// Defined by p2p-resiliency-recovery-test.cc, which sees the recovery thread's statics.
// Stops the thread without racing its wait (AICOMRCCL-3103).
void P2pResiliencyStopRecoveryThread();
// Frees every recovery context queued for a thread that is not running; returns how many.
int P2pResiliencyDrainRecoveryInbox();

constexpr int kSceneIbDevBase = 3;  // physical IbCastDevs[] slot of local device 0

class P2pResiliencyScene {
 public:
  struct PostedSend {
    struct ibv_qp* qp;
    struct ibv_send_wr wr;
    std::vector<struct ibv_sge> sges;
  };
  struct PostedRecv {
    struct ibv_qp* qp;
    struct ibv_recv_wr wr;
    struct ibv_sge sge;
  };
  struct Modify {
    struct ibv_qp* qp;
    struct ibv_qp_attr attr;
    int mask;
  };
  struct RegMr {
    struct ibv_pd* pd;
    void* addr;
    size_t length;
    int access;
    struct ibv_mr* mr;
  };
  struct CreateCq {
    struct ibv_context* context;
    int cqe;
    void* cqContext;
    struct ibv_cq* cq;
  };
  struct QpCreate {
    struct ncclIbQp* qp;
    struct ncclIbQpCreateAttr attr;
  };
  struct Transition {
    struct ncclIbQp* qp;
    std::string step;
    struct ncclIbQpInitAttr initAttr;
    struct ncclIbQpRtrAttr rtrAttr;
    struct ncclIbQpRtsAttr rtsAttr;
  };
  struct MultiSend {
    struct ncclIbSendComm* comm;
    int slot, nqps, startQpIndex;
    bool wrrSched, useWriteOp;
  };
  struct PostFifo {
    struct ncclIbRecvComm* comm;
    struct ncclIbRequest* req;
    int slot, n;
  };
  struct AddEvent {
    struct ncclIbRequest* req;
    int devIndex;
  };

  // Per-test behaviour of the verbs ops the units call inline. Set before any thread starts.
  std::function<int(struct ibv_cq*, int, struct ibv_wc*)> pollCq;
  std::function<int(struct ibv_qp*, struct ibv_send_wr*)> postSend;
  std::function<int(struct ibv_qp*, struct ibv_recv_wr*)> postRecv;
  std::function<struct ibv_ah*(struct ibv_pd*, struct ibv_ah_attr*)> createAh;
  std::function<int(struct ibv_ah*)> destroyAh;

  std::map<std::string, int64_t> params;

  // What the units did. Guarded by mu because the recovery thread also posts.
  std::mutex mu;
  std::vector<PostedSend> sends;
  std::vector<PostedRecv> recvs;
  std::vector<Modify> modifies;
  std::vector<RegMr> regMrs;
  std::vector<CreateCq> createCqs;
  std::vector<struct ibv_cq*> destroyedCqs;
  std::vector<struct ibv_mr*> deregMrs;
  std::vector<struct ibv_qp*> destroyedQps;
  std::vector<struct ibv_ah*> destroyedAhs;
  std::vector<struct ibv_ah_attr> ahAttrs;
  std::vector<struct ibv_qp*> eceSets;
  std::vector<QpCreate> qpCreates;
  std::vector<Transition> transitions;
  std::vector<struct ncclIbQp*> postedRecvWrQps;
  std::vector<std::pair<struct ncclIbNetCommBase*, int>> buildAttrCalls;
  std::vector<MultiSend> multiSends;
  std::vector<PostFifo> postFifos;
  std::vector<AddEvent> addEvents;

  struct ibv_context ctx = {};
  struct ibv_pd pd = {};

  P2pResiliencyScene()
    : regMr_(g_ibvRegMr,
             [this](struct ibv_mr** ret, struct ibv_pd* p, void* addr, size_t len, int access) {
               std::lock_guard<std::mutex> lock(mu);
               mrs_.emplace_back();
               struct ibv_mr* mr = &mrs_.back();
               mr->addr = addr;
               mr->length = len;
               mr->lkey = 0x1000 + (uint32_t)mrs_.size();
               mr->rkey = 0x2000 + (uint32_t)mrs_.size();
               regMrs.push_back({p, addr, len, access, mr});
               *ret = mr;
               return ncclSuccess;
             }),
      deregMr_(g_ibvDeregMr,
               [this](struct ibv_mr* mr) {
                 std::lock_guard<std::mutex> lock(mu);
                 deregMrs.push_back(mr);
                 return ncclSuccess;
               }),
      createCq_(g_ibvCreateCq,
                [this](struct ibv_cq** ret, struct ibv_context* c, int cqe, void* cqContext,
                       struct ibv_comp_channel*, int) {
                  std::lock_guard<std::mutex> lock(mu);
                  cqs_.emplace_back();
                  struct ibv_cq* cq = &cqs_.back();
                  cq->context = c;
                  cq->cqe = cqe;
                  cq->cq_context = cqContext;
                  createCqs.push_back({c, cqe, cqContext, cq});
                  *ret = cq;
                  return ncclSuccess;
                }),
      destroyCq_(g_ibvDestroyCq,
                 [this](struct ibv_cq* cq) {
                   std::lock_guard<std::mutex> lock(mu);
                   destroyedCqs.push_back(cq);
                   return ncclSuccess;
                 }),
      modifyQp_(g_ibvModifyQp,
                [this](struct ibv_qp* qp, struct ibv_qp_attr* attr, int mask) {
                  std::lock_guard<std::mutex> lock(mu);
                  modifies.push_back({qp, *attr, mask});
                  return ncclSuccess;
                }),
      destroyQp_(g_ibvDestroyQp,
                 [this](struct ibv_qp* qp) {
                   std::lock_guard<std::mutex> lock(mu);
                   destroyedQps.push_back(qp);
                   return ncclSuccess;
                 }),
      setEce_(g_ibvSetEce,
              [this](struct ibv_qp* qp, struct ibv_ece*, int*) {
                std::lock_guard<std::mutex> lock(mu);
                eceSets.push_back(qp);
                return ncclSuccess;
              }),
      qpCreate_(g_ibCastQpCreate,
                [this](struct ncclIbQp* qp, struct ncclIbQpCreateAttr* attr) {
                  qp->qp = NewQp();
                  std::lock_guard<std::mutex> lock(mu);
                  qpCreates.push_back({qp, *attr});
                  return ncclSuccess;
                }),
      buildAttr_(g_ibCastBuildDataQpCreateAttr,
                 [this](struct ncclIbNetCommBase* base, int devIndex, struct ncclIbQpCreateAttr* out) {
                   std::memset(out, 0, sizeof(*out));
                   out->type = IBV_QPT_RC;
                   out->ibDevN = kSceneIbDevBase + devIndex;
                   out->maxSendWorkRequest = 77;
                   out->maxRecvWorkRequest = 88;
                   out->isP2p = 1;
                   std::lock_guard<std::mutex> lock(mu);
                   buildAttrCalls.push_back({base, devIndex});
                 }),
      qpInit_(g_ibCastQpInit, [this](struct ncclIbQp* qp) { return Record(qp, "INIT"); }),
      qpRtr_(g_ibCastQpRtr, [this](struct ncclIbQp* qp) { return Record(qp, "RTR"); }),
      qpRts_(g_ibCastQpRts, [this](struct ncclIbQp* qp) { return Record(qp, "RTS"); }),
      qpReset_(g_ibCastQpReset, [this](struct ncclIbQp* qp) { return Record(qp, "RESET"); }),
      qpError_(g_ibCastQpError, [this](struct ncclIbQp* qp) { return Record(qp, "ERROR"); }),
      postRecvWrs_(g_ibCastPostReceiveWorkRequestsOnQp,
                   [this](struct ncclIbRecvComm*, struct ncclIbQp* qp) {
                     std::lock_guard<std::mutex> lock(mu);
                     postedRecvWrQps.push_back(qp);
                     return ncclSuccess;
                   }),
      udSupported_(g_ibCastCapUdSupported, [](const struct ncclIbDev* dev) { return dev->udSupported == 1; }),
      multiSend_(g_ibCastMultiSend,
                 [this](struct ncclIbSendComm* comm, int slot, int nqps, int start, bool wrr, bool write) {
                   std::lock_guard<std::mutex> lock(mu);
                   multiSends.push_back({comm, slot, nqps, start, wrr, write});
                   return ncclSuccess;
                 }),
      postFifo_(g_ibCastPostFifo,
                [this](struct ncclIbRecvComm* comm, struct ncclIbRequest* req, int slot, int n) {
                  std::lock_guard<std::mutex> lock(mu);
                  postFifos.push_back({comm, req, slot, n});
                  return ncclSuccess;
                }),
      addEvent_(g_ibCastAddEvent,
                [this](struct ncclIbRequest* req, int devIndex) {
                  std::lock_guard<std::mutex> lock(mu);
                  addEvents.push_back({req, devIndex});
                }),
      loadParam_(g_loadParam, [this](const char* env, int64_t deftVal) {
        auto it = params.find(env);
        return it == params.end() ? deftVal : it->second;
      }) {
    current_ = this;
    ctx.ops.poll_cq = [](struct ibv_cq* cq, int n, struct ibv_wc* wc) { return current_->pollCq(cq, n, wc); };
    ctx.ops.post_send = [](struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad) {
      *bad = nullptr;
      return current_->RecordSend(qp, wr);
    };
    ctx.ops.post_recv = [](struct ibv_qp* qp, struct ibv_recv_wr* wr, struct ibv_recv_wr** bad) {
      *bad = nullptr;
      return current_->RecordRecv(qp, wr);
    };
    ctx.ops.create_ah = [](struct ibv_pd* p, struct ibv_ah_attr* attr) { return current_->createAh(p, attr); };
    ctx.ops.destroy_ah = [](struct ibv_ah* ah) {
      {
        std::lock_guard<std::mutex> lock(current_->mu);
        current_->destroyedAhs.push_back(ah);
      }
      return current_->destroyAh(ah);
    };
    pd.context = &ctx;
    pollCq = [this](struct ibv_cq* cq, int n, struct ibv_wc* wc) { return PopCqes(cq, n, wc); };
    postSend = [](struct ibv_qp*, struct ibv_send_wr*) { return 0; };
    postRecv = [](struct ibv_qp*, struct ibv_recv_wr*) { return 0; };
    createAh = [this](struct ibv_pd*, struct ibv_ah_attr* attr) {
      std::lock_guard<std::mutex> lock(mu);
      ahs_.emplace_back();
      ahAttrs.push_back(*attr);
      return &ahs_.back();
    };
    destroyAh = [](struct ibv_ah*) { return 0; };
    params["IB_RESILIENCY_PORT_FAILOVER"] = 1;
  }

  ~P2pResiliencyScene() {
    for (int d = 0; d < NCCL_IB_MAX_DEVS_PER_NIC; d++) {
      struct ncclIbDev* dev = &IbCastDevs[kSceneIbDevBase + d];
      dev->context = nullptr;
      dev->pd = nullptr;
      dev->portNum = 0;
      dev->udSupported = 0;
      dev->portAttr = {};
      dev->gidInfo = {};
      std::snprintf(dev->devName, sizeof(dev->devName), "%s", "");
    }
    ResetNetIbCastInitFakes();
    ResetNetIbCastQpSharingFakes();
    current_ = nullptr;
  }

  P2pResiliencyScene(const P2pResiliencyScene&) = delete;
  P2pResiliencyScene& operator=(const P2pResiliencyScene&) = delete;

  struct ibv_qp* NewQp() {
    std::lock_guard<std::mutex> lock(mu);
    qps_.emplace_back();
    struct ibv_qp* qp = &qps_.back();
    qp->context = &ctx;
    qp->qp_num = nextQpn_++;
    return qp;
  }

  struct ibv_cq* NewCq() {
    std::lock_guard<std::mutex> lock(mu);
    cqs_.emplace_back();
    cqs_.back().context = &ctx;
    return &cqs_.back();
  }

  struct ibv_mr* NewMr(uint32_t lkey) {
    std::lock_guard<std::mutex> lock(mu);
    mrs_.emplace_back();
    mrs_.back().lkey = lkey;
    return &mrs_.back();
  }

  struct ibv_ah* NewAh() {
    std::lock_guard<std::mutex> lock(mu);
    ahs_.emplace_back();
    ahs_.back().context = &ctx;
    return &ahs_.back();
  }

  void PushCqe(struct ibv_cq* cq, const struct ibv_wc& wc) {
    std::lock_guard<std::mutex> lock(mu);
    cqes_[cq].push_back(wc);
  }

  size_t PendingCqes(struct ibv_cq* cq) {
    std::lock_guard<std::mutex> lock(mu);
    return cqes_[cq].size();
  }

  // Physical device for local device d, with what the units read from it.
  struct ncclIbDev* Dev(int d) { return &IbCastDevs[kSceneIbDevBase + d]; }

  template <typename Comm>
  struct Freer {
    void operator()(Comm* c) const { std::free(c); }
  };
  template <typename Comm>
  using CommPtr = std::unique_ptr<Comm, Freer<Comm>>;

  // A comm over ndevs devices with nqps QPs striped across them (QP i on device i % ndevs).
  template <typename Comm>
  CommPtr<Comm> NewComm(bool isSend, int ndevs, int nqps) {
    size_t size = (sizeof(Comm) + 63) / 64 * 64;
    Comm* comm = static_cast<Comm*>(std::aligned_alloc(64, size));
    std::memset(comm, 0, size);
    struct ncclIbNetCommBase* base = &comm->base;
    base->isSend = isSend;
    base->vProps.ndevs = ndevs;
    base->nqps = nqps;
    for (int d = 0; d < ndevs; d++) {
      base->vProps.devs[d] = kSceneIbDevBase + d;
      struct ncclIbNetCommDevBase* devBase = &comm->devs[d].base;
      devBase->ibDevN = kSceneIbDevBase + d;
      devBase->pd = &pd;
      devBase->gidInfo.localGidIndex = 10 + d;
      devBase->gidInfo.localGid.global.interface_id = 0xA0 + d;
      struct ncclIbDev* dev = Dev(d);
      dev->context = &ctx;
      dev->pd = &pd;
      dev->portNum = 1 + d;
      dev->udSupported = 1;
      dev->portAttr.active_mtu = IBV_MTU_4096;
      dev->portAttr.flags = 0x40 + d;
      dev->gidInfo.localGidIndex = 20 + d;
      dev->gidInfo.localGid.global.interface_id = 0xB0 + d;
      std::snprintf(dev->devName, sizeof(dev->devName), "fake_ib%d", d);
    }
    for (int q = 0; q < nqps; q++) {
      base->qps[q].qp = NewQp();
      base->qps[q].devIndex = q % ndevs;
      base->activeQps[q] = &base->qps[q];
    }
    return CommPtr<Comm>(comm);
  }

 private:
  ncclResult_t Record(struct ncclIbQp* qp, const char* step) {
    std::lock_guard<std::mutex> lock(mu);
    transitions.push_back({qp, step, qp->initAttr, qp->rtrAttr, qp->rtsAttr});
    return ncclSuccess;
  }

  int RecordSend(struct ibv_qp* qp, struct ibv_send_wr* wr) {
    {
      std::lock_guard<std::mutex> lock(mu);
      for (struct ibv_send_wr* w = wr; w != nullptr; w = w->next) {
        PostedSend s{qp, *w, {}};
        for (int i = 0; i < w->num_sge; i++) s.sges.push_back(w->sg_list[i]);
        s.wr.next = nullptr;
        s.wr.sg_list = nullptr;
        sends.push_back(s);
      }
    }
    return postSend(qp, wr);
  }

  int RecordRecv(struct ibv_qp* qp, struct ibv_recv_wr* wr) {
    {
      std::lock_guard<std::mutex> lock(mu);
      recvs.push_back({qp, *wr, *wr->sg_list});
    }
    return postRecv(qp, wr);
  }

  int PopCqes(struct ibv_cq* cq, int n, struct ibv_wc* wc) {
    std::lock_guard<std::mutex> lock(mu);
    std::deque<struct ibv_wc>& q = cqes_[cq];
    int done = 0;
    while (done < n && !q.empty()) {
      wc[done++] = q.front();
      q.pop_front();
    }
    return done;
  }

  inline static P2pResiliencyScene* current_ = nullptr;

  std::deque<struct ibv_qp> qps_;
  std::deque<struct ibv_cq> cqs_;
  std::deque<struct ibv_mr> mrs_;
  std::deque<struct ibv_ah> ahs_;
  std::map<struct ibv_cq*, std::deque<struct ibv_wc>> cqes_;
  uint32_t nextQpn_ = 100;

  ScopedHook<ncclResult_t(struct ibv_mr**, struct ibv_pd*, void*, size_t, int)> regMr_;
  ScopedHook<ncclResult_t(struct ibv_mr*)> deregMr_;
  ScopedHook<ncclResult_t(struct ibv_cq**, struct ibv_context*, int, void*, struct ibv_comp_channel*, int)> createCq_;
  ScopedHook<ncclResult_t(struct ibv_cq*)> destroyCq_;
  ScopedHook<ncclResult_t(struct ibv_qp*, struct ibv_qp_attr*, int)> modifyQp_;
  ScopedHook<ncclResult_t(struct ibv_qp*)> destroyQp_;
  ScopedHook<ncclResult_t(struct ibv_qp*, struct ibv_ece*, int*)> setEce_;
  ScopedHook<ncclResult_t(struct ncclIbQp*, struct ncclIbQpCreateAttr*)> qpCreate_;
  ScopedHook<void(struct ncclIbNetCommBase*, int, struct ncclIbQpCreateAttr*)> buildAttr_;
  ScopedHook<ncclResult_t(struct ncclIbQp*)> qpInit_;
  ScopedHook<ncclResult_t(struct ncclIbQp*)> qpRtr_;
  ScopedHook<ncclResult_t(struct ncclIbQp*)> qpRts_;
  ScopedHook<ncclResult_t(struct ncclIbQp*)> qpReset_;
  ScopedHook<ncclResult_t(struct ncclIbQp*)> qpError_;
  ScopedHook<ncclResult_t(struct ncclIbRecvComm*, struct ncclIbQp*)> postRecvWrs_;
  ScopedHook<bool(const struct ncclIbDev*)> udSupported_;
  ScopedHook<ncclResult_t(struct ncclIbSendComm*, int, int, int, bool, bool)> multiSend_;
  ScopedHook<ncclResult_t(struct ncclIbRecvComm*, struct ncclIbRequest*, int, int)> postFifo_;
  ScopedHook<void(struct ncclIbRequest*, int)> addEvent_;
  ScopedHook<int64_t(const char*, int64_t)> loadParam_;
};

// A completion as the units see it.
inline struct ibv_wc P2pResiliencyWc(uint64_t wrId, enum ibv_wc_status status, enum ibv_wc_opcode opcode,
                                     uint32_t immData = 0, uint32_t byteLen = 0) {
  struct ibv_wc wc = {};
  wc.wr_id = wrId;
  wc.status = status;
  wc.opcode = opcode;
  wc.imm_data = immData;
  wc.byte_len = byteLen;
  return wc;
}

// For tests of the units' guards against a comm without the device's base.
inline struct ncclIbNetCommDevBase* P2pResiliencyNoDevBase(struct ncclIbNetCommBase*, int) { return nullptr; }

// The peer's connection metadata: probing QP q talks to remote device ndevs - 1 - q % ndevs, recovery
// QP q to remote device q % ndevs. A peer without port recovery leaves its recovery QPNs zero.
inline struct ncclIbConnectionMetadata P2pResiliencyRemoteInfo(int ndevs, bool withRecovery) {
  struct ncclIbConnectionMetadata meta;
  std::memset(&meta, 0, sizeof(meta));
  meta.ndevs = ndevs;
  meta.tc = 106;
  meta.sl = 3;
  for (int d = 0; d < ndevs; d++) {
    meta.devs[d].lid = 0x300 + d;
    meta.devs[d].ib_port = 5 + d;
    meta.devs[d].mtu = IBV_MTU_1024;
    meta.devs[d].link_layer = IBV_LINK_LAYER_ETHERNET;
    meta.devs[d].gid.global.interface_id = 0xC0 + d;
  }
  for (int q = 0; q < NCCL_IB_MAX_DEVS_PER_NIC; q++) {
    meta.resiliencyInfo.probingQpsInfo[q].qpn = 0x500 + q;
    meta.resiliencyInfo.probingQpsInfo[q].devIndex = ndevs - 1 - q % ndevs;
    if (withRecovery) {
      meta.resiliencyInfo.portRecoveryQpsInfo[q].qpn = 0x600 + q;
      meta.resiliencyInfo.portRecoveryQpsInfo[q].devIndex = q % ndevs;
    }
  }
  return meta;
}

#endif  // RCCL_TEST_HOST_P2P_RESILIENCY_SCENE_H_
