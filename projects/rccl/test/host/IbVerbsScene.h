/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_IB_VERBS_SCENE_H_
#define RCCL_TEST_HOST_IB_VERBS_SCENE_H_

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <vector>

#include "ibvwrap.h"

// One fake verbs device. wrap_ibv_post_send/post_recv/poll_cq are inline calls through ctx.ops, so pointing those
// at this scene lets a test see every work request the unit posts and hand it scripted completions.
struct IbVerbsScene {
  static constexpr int kQps = 8;
  static constexpr uint32_t kQpNumBase = 0x100;

  struct Wr {
    ibv_send_wr wr;
    bool hasSge;
    ibv_sge sge;  // sg_list[0] at post time; the unit reuses its sge buffers
  };
  struct Send {
    ibv_qp* qp;
    std::vector<Wr> chain;
  };
  struct Recv {
    ibv_qp* qp;
    uint64_t wrId;
  };
  // One posted send WR, flattened out of its chain.
  struct Posted {
    ibv_qp* qp;
    uint32_t qpNum;
    uint64_t wrId;
    ibv_wr_opcode opcode;
    int sendFlags;
    uint32_t immData;
    uint64_t remoteAddress;  // wr.atomic for atomic opcodes, wr.rdma otherwise
    uint32_t rkey;
    uint64_t compareAdd;     // atomic opcodes only
    ibv_sge sge;             // zero when the WR has none
  };

  ibv_device device{};
  ibv_context ctx{};
  ibv_pd pd{};
  ibv_cq cq{};
  ibv_qp qps[kQps]{};

  std::vector<Send> sends;
  std::vector<Posted> posted;        // the WRs of `sends`, in post order
  std::vector<Recv> recvs;
  std::deque<ibv_wc> wcs;
  std::vector<int> pollEntries;
  std::vector<ibv_qp*> postSendQps;  // every post_send call, failed ones included
  std::vector<ibv_qp*> postRecvQps;  // every post_recv call, failed ones included
  std::vector<ibv_cq*> polledCqs;    // every poll_cq call
  int postSendErr = 0;
  int postSendFailOnCall = 0;        // 1-based post_send call that fails with EIO; 0 for none
  int postRecvErr = 0;
  int pollErr = 0;

  // QP i is numbered qpNumBase + i.
  explicit IbVerbsScene(const char* name = "mlx5_0", uint32_t qpNumBase = kQpNumBase) {
    std::snprintf(device.name, sizeof(device.name), "%s", name);
    ctx.device = &device;
    ctx.abi_compat = this;
    ctx.ops.post_send = PostSend;
    ctx.ops.post_recv = PostRecv;
    ctx.ops.poll_cq = PollCq;
    pd.context = &ctx;
    cq.context = &ctx;
    for (int i = 0; i < kQps; i++) {
      qps[i].context = &ctx;
      qps[i].pd = &pd;
      qps[i].qp_num = qpNumBase + i;
    }
  }
  IbVerbsScene(const IbVerbsScene&) = delete;
  IbVerbsScene& operator=(const IbVerbsScene&) = delete;

  static IbVerbsScene* Of(ibv_context* c) { return static_cast<IbVerbsScene*>(c->abi_compat); }

  static int PostSend(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** badWr) {
    IbVerbsScene* s = Of(qp->context);
    s->postSendQps.push_back(qp);
    const int err = static_cast<int>(s->postSendQps.size()) == s->postSendFailOnCall ? EIO : s->postSendErr;
    if (err) {
      *badWr = wr;
      return err;
    }
    Send send{qp, {}};
    for (ibv_send_wr* w = wr; w; w = w->next) {
      Wr copy{*w, w->num_sge > 0 && w->sg_list != nullptr, {}};
      if (copy.hasSge) copy.sge = w->sg_list[0];
      send.chain.push_back(copy);
      s->posted.push_back(Flatten(qp, copy));
    }
    s->sends.push_back(send);
    return 0;
  }

  static int PostRecv(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr** badWr) {
    IbVerbsScene* s = Of(qp->context);
    s->postRecvQps.push_back(qp);
    if (s->postRecvErr) {
      *badWr = wr;
      return s->postRecvErr;
    }
    for (ibv_recv_wr* w = wr; w; w = w->next) s->recvs.push_back({qp, w->wr_id});
    return 0;
  }

  static int PollCq(ibv_cq* cq, int n, ibv_wc* out) {
    IbVerbsScene* s = Of(cq->context);
    s->polledCqs.push_back(cq);
    s->pollEntries.push_back(n);
    if (s->pollErr) return s->pollErr;
    int got = 0;
    while (got < n && !s->wcs.empty()) {
      out[got++] = s->wcs.front();
      s->wcs.pop_front();
    }
    return got;
  }

  ibv_wc& Complete(uint64_t wrId, ibv_wc_opcode opcode, uint32_t qpNum, ibv_wc_status status = IBV_WC_SUCCESS) {
    ibv_wc wc{};
    wc.wr_id = wrId;
    wc.opcode = opcode;
    wc.qp_num = qpNum;
    wc.status = status;
    wcs.push_back(wc);
    return wcs.back();
  }

 private:
  static Posted Flatten(ibv_qp* qp, const Wr& w) {
    Posted p{};
    p.qp = qp;
    p.qpNum = qp->qp_num;
    p.wrId = w.wr.wr_id;
    p.opcode = w.wr.opcode;
    p.sendFlags = static_cast<int>(w.wr.send_flags);
    p.immData = w.wr.imm_data;
    if (w.wr.opcode == IBV_WR_ATOMIC_FETCH_AND_ADD || w.wr.opcode == IBV_WR_ATOMIC_CMP_AND_SWP) {
      p.remoteAddress = w.wr.wr.atomic.remote_addr;
      p.rkey = w.wr.wr.atomic.rkey;
      p.compareAdd = w.wr.wr.atomic.compare_add;
    } else {
      p.remoteAddress = w.wr.wr.rdma.remote_addr;
      p.rkey = w.wr.wr.rdma.rkey;
    }
    if (w.hasSge) p.sge = w.sge;
    return p;
  }
};

#endif  // RCCL_TEST_HOST_IB_VERBS_SCENE_H_
