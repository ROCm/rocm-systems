/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *****************************************************************************/

#include "net/ib_ctrl.hpp"

#include <cstdio>
#include <cstring>

namespace rocshmem {
namespace net {

IbCtrl::~IbCtrl() { destroy(); }

bool IbCtrl::open(Ibv *ibv, const char *dev_name, uint8_t port_hint) {
  ibv_ = ibv;
  dev_.port = port_hint;

  int num = 0;
  struct ibv_device **list = ibv_->get_device_list(&num);
  if (!list || num == 0) {
    fprintf(stderr, "[rocSHMEM] verbs: no RDMA devices found\n");
    return false;
  }

  struct ibv_device *chosen = nullptr;
  for (int i = 0; i < num; i++) {
    const char *name = ibv_->get_device_name(list[i]);
    if (dev_name && name && strstr(name, dev_name) == nullptr) {
      continue;  // name filter given and did not match
    }
    struct ibv_context *ctx = ibv_->open_device(list[i]);
    if (!ctx) {
      continue;
    }
    struct ibv_port_attr pa;
    memset(&pa, 0, sizeof(pa));
    if (ibv_->query_port(ctx, dev_.port, &pa) == 0 &&
        pa.state == IBV_PORT_ACTIVE) {
      chosen = list[i];
      dev_.ctx = ctx;
      dev_.port_attr = pa;
      dev_.mtu = pa.active_mtu;
      dev_.is_roce = (pa.link_layer == IBV_LINK_LAYER_ETHERNET);
      break;
    }
    ibv_->close_device(ctx);
  }

  if (!chosen) {
    fprintf(stderr, "[rocSHMEM] verbs: no device with an ACTIVE port%s%s\n",
            dev_name ? " matching " : "", dev_name ? dev_name : "");
    ibv_->free_device_list(list);
    return false;
  }
  fprintf(stderr, "[rocSHMEM] verbs: using %s port %u (%s)\n",
          ibv_->get_device_name(chosen), dev_.port,
          dev_.is_roce ? "RoCE" : "IB");
  ibv_->free_device_list(list);

  dev_.pd = ibv_->alloc_pd(dev_.ctx);
  if (!dev_.pd) {
    fprintf(stderr, "[rocSHMEM] verbs: alloc_pd failed\n");
    return false;
  }
  return select_gid();
}

bool IbCtrl::select_gid() {
  // Prefer a RoCEv2, non-zero, non-link-local GID on our port. On IB, GID 0 is
  // fine and gid_type detection is unnecessary.
  if (!dev_.is_roce) {
    dev_.gid_index = 0;
    ibv_->query_gid(dev_.ctx, dev_.port, 0, &dev_.gid);
    return true;
  }

  constexpr size_t kMax = 128;
  std::vector<struct ibv_gid_entry> entries(kMax);
  int n = ibv_->query_gid_table(dev_.ctx, entries.data(), kMax);
  if (n > 0) {
    int fallback = -1;
    for (int i = 0; i < n; i++) {
      const auto &e = entries[i];
      if (e.port_num != dev_.port) {
        continue;
      }
      // Skip the all-zero GID.
      bool zero = true;
      for (int b = 0; b < 16; b++) {
        if (e.gid.raw[b] != 0) {
          zero = false;
          break;
        }
      }
      if (zero) {
        continue;
      }
      if (fallback < 0) {
        fallback = static_cast<int>(e.gid_index);
      }
      if (e.gid_type == IBV_GID_TYPE_ROCE_V2) {
        dev_.gid_index = static_cast<int>(e.gid_index);
        dev_.gid = e.gid;
        return true;
      }
    }
    if (fallback >= 0) {
      dev_.gid_index = fallback;
      ibv_->query_gid(dev_.ctx, dev_.port, fallback, &dev_.gid);
      return true;
    }
  }

  // Last resort: index 0.
  dev_.gid_index = 0;
  ibv_->query_gid(dev_.ctx, dev_.port, 0, &dev_.gid);
  fprintf(stderr,
          "[rocSHMEM] verbs: GID-table query unavailable; using GID index 0\n");
  return true;
}

bool IbCtrl::create_queues(const LaneMap &lm, int cq_depth,
                           int qp_send_depth) {
  lm_ = lm;
  cqs_.assign(static_cast<size_t>(lm_.num_ctx), nullptr);
  qps_.assign(static_cast<size_t>(lm_.num_lanes()), nullptr);

  for (int ctx = 0; ctx < lm_.num_ctx; ctx++) {
    cqs_[ctx] = ibv_->create_cq(dev_.ctx, cq_depth, nullptr, nullptr, 0);
    if (!cqs_[ctx]) {
      fprintf(stderr, "[rocSHMEM] verbs: create_cq failed (ctx %d)\n", ctx);
      return false;
    }
  }

  for (int ctx = 0; ctx < lm_.num_ctx; ctx++) {
    for (int pe = 0; pe < lm_.num_pes; pe++) {
      struct ibv_qp_init_attr qa;
      memset(&qa, 0, sizeof(qa));
      qa.send_cq = cqs_[ctx];
      qa.recv_cq = cqs_[ctx];
      qa.qp_type = IBV_QPT_RC;
      qa.sq_sig_all = 0;  // conduit signals selectively
      qa.cap.max_send_wr = static_cast<uint32_t>(qp_send_depth);
      qa.cap.max_recv_wr = 1;
      qa.cap.max_send_sge = 1;
      qa.cap.max_recv_sge = 1;
      qa.cap.max_inline_data = 64;

      int lane = lm_.lane(ctx, pe);
      qps_[lane] = ibv_->create_qp(dev_.pd, &qa);
      if (!qps_[lane]) {
        fprintf(stderr, "[rocSHMEM] verbs: create_qp failed (lane %d)\n", lane);
        return false;
      }
      if (!to_init(qps_[lane])) {
        return false;
      }
    }
  }
  return true;
}

bool IbCtrl::to_init(struct ibv_qp *qp) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = dev_.port;
  attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                         IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  int mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
             IBV_QP_ACCESS_FLAGS;
  int rc = ibv_->modify_qp(qp, &attr, mask);
  if (rc != 0) {
    fprintf(stderr, "[rocSHMEM] verbs: modify_qp->INIT failed (%d)\n", rc);
    return false;
  }
  return true;
}

bool IbCtrl::to_rtr(struct ibv_qp *qp, const DestInfo &remote) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = dev_.mtu;
  attr.dest_qp_num = remote.qpn;
  attr.rq_psn = remote.psn;
  attr.max_dest_rd_atomic = 16;
  attr.min_rnr_timer = 12;

  attr.ah_attr.port_num = dev_.port;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.dlid = remote.lid;  // 0 on RoCE
  // RoCE (and best practice on IB with GRH) uses a global route.
  attr.ah_attr.is_global = 1;
  memcpy(attr.ah_attr.grh.dgid.raw, remote.gid, 16);
  attr.ah_attr.grh.sgid_index = static_cast<uint8_t>(dev_.gid_index);
  attr.ah_attr.grh.hop_limit = 255;
  attr.ah_attr.grh.traffic_class = 0;
  attr.ah_attr.grh.flow_label = 0;

  int mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
             IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  int rc = ibv_->modify_qp(qp, &attr, mask);
  if (rc != 0) {
    fprintf(stderr, "[rocSHMEM] verbs: modify_qp->RTR failed (%d)\n", rc);
    return false;
  }
  return true;
}

bool IbCtrl::to_rts(struct ibv_qp *qp) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = psn_;
  attr.max_rd_atomic = 16;
  int mask = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
             IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
  int rc = ibv_->modify_qp(qp, &attr, mask);
  if (rc != 0) {
    fprintf(stderr, "[rocSHMEM] verbs: modify_qp->RTS failed (%d)\n", rc);
    return false;
  }
  return true;
}

bool IbCtrl::connect_qp(int lane, const DestInfo &remote) {
  return to_rtr(qps_[lane], remote) && to_rts(qps_[lane]);
}

void IbCtrl::fill_local_dest(std::vector<DestInfo> &out) const {
  out.assign(static_cast<size_t>(lm_.num_lanes()), DestInfo{});
  for (int lane = 0; lane < lm_.num_lanes(); lane++) {
    DestInfo d;
    memset(&d, 0, sizeof(d));
    d.qpn = qps_[lane]->qp_num;
    d.psn = psn_;
    d.lid = dev_.port_attr.lid;
    d.port = dev_.port;
    d.gid_index = static_cast<uint8_t>(dev_.gid_index);
    memcpy(d.gid, dev_.gid.raw, 16);
    out[lane] = d;
  }
}

bool IbCtrl::connect(int my_pe, const std::vector<DestInfo> &global) {
  const int lanes = lm_.num_lanes();
  for (int ctx = 0; ctx < lm_.num_ctx; ctx++) {
    for (int pe = 0; pe < lm_.num_pes; pe++) {
      if (pe == my_pe) {
        continue;  // no self QP over the fabric
      }
      int my_lane = lm_.lane(ctx, pe);
      // Peer pe's QP that targets me lives at its lane (ctx, my_pe).
      int peer_lane = lm_.lane(ctx, my_pe);
      const DestInfo &remote =
          global[static_cast<size_t>(pe) * lanes + peer_lane];
      if (!connect_qp(my_lane, remote)) {
        return false;
      }
    }
  }
  return true;
}

void IbCtrl::destroy() {
  if (!ibv_) {
    return;
  }
  for (auto *qp : qps_) {
    if (qp) {
      ibv_->destroy_qp(qp);
    }
  }
  qps_.clear();
  for (auto *cq : cqs_) {
    if (cq) {
      ibv_->destroy_cq(cq);
    }
  }
  cqs_.clear();
  if (dev_.pd) {
    ibv_->dealloc_pd(dev_.pd);
    dev_.pd = nullptr;
  }
  if (dev_.ctx) {
    ibv_->close_device(dev_.ctx);
    dev_.ctx = nullptr;
  }
}

}  // namespace net
}  // namespace rocshmem
