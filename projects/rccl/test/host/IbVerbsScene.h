/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_IB_VERBS_SCENE_H_
#define RCCL_TEST_HOST_IB_VERBS_SCENE_H_

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

  ibv_device device{};
  ibv_context ctx{};
  ibv_pd pd{};
  ibv_cq cq{};
  ibv_qp qps[kQps]{};

  std::vector<Send> sends;
  std::vector<Recv> recvs;
  std::deque<ibv_wc> wcs;
  std::vector<int> pollEntries;
  int postSendErr = 0;
  int postRecvErr = 0;
  int pollErr = 0;

  explicit IbVerbsScene(const char* name = "mlx5_0") {
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
      qps[i].qp_num = kQpNumBase + i;
    }
  }
  IbVerbsScene(const IbVerbsScene&) = delete;
  IbVerbsScene& operator=(const IbVerbsScene&) = delete;

  static IbVerbsScene* Of(ibv_context* c) { return static_cast<IbVerbsScene*>(c->abi_compat); }

  static int PostSend(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** badWr) {
    IbVerbsScene* s = Of(qp->context);
    if (s->postSendErr) {
      *badWr = wr;
      return s->postSendErr;
    }
    Send send{qp, {}};
    for (ibv_send_wr* w = wr; w; w = w->next) {
      Wr copy{*w, w->num_sge > 0 && w->sg_list != nullptr, {}};
      if (copy.hasSge) copy.sge = w->sg_list[0];
      send.chain.push_back(copy);
    }
    s->sends.push_back(send);
    return 0;
  }

  static int PostRecv(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr** badWr) {
    IbVerbsScene* s = Of(qp->context);
    if (s->postRecvErr) {
      *badWr = wr;
      return s->postRecvErr;
    }
    for (ibv_recv_wr* w = wr; w; w = w->next) s->recvs.push_back({qp, w->wr_id});
    return 0;
  }

  static int PollCq(ibv_cq* cq, int n, ibv_wc* out) {
    IbVerbsScene* s = Of(cq->context);
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
};

#endif  // RCCL_TEST_HOST_IB_VERBS_SCENE_H_
