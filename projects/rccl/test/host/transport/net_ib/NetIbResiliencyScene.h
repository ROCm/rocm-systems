/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only scene shared by the src/transport/net_ib/p2p_resiliency*.cc tests.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_TRANSPORT_NET_IB_NET_IB_RESILIENCY_SCENE_H_
#define RCCL_TEST_HOST_TRANSPORT_NET_IB_NET_IB_RESILIENCY_SCENE_H_

#include <array>
#include <initializer_list>

#include "IbVerbsScene.h"
#include "transport/net_ib/p2p_resiliency.h"

// A send comm and a recv comm with failover contexts on two fake verbs devices, every device healthy. Both comms
// sit on the same verbs QPs, so a test drives one direction at a time. Device d's verbs QPs are:
//   0, 1  data QPs d and d + kDevs (connect.cc stripes data QPs across devices)
//   2     the probing QP
//   3     the port-recovery QP
//   4..7  handed out by CreateQp
struct NetIbResiliencyScene {
  static constexpr int kDevs = 2;
  static constexpr int kDataQps = 2 * kDevs;
  static constexpr int kProbingQp = 2;
  static constexpr int kRecoveryQp = 3;
  static constexpr int kFirstCreatedQp = 4;

  std::array<IbVerbsScene, kDevs> verbs{{IbVerbsScene("mlx5_0", 0x100), IbVerbsScene("mlx5_1", 0x200)}};
  ncclIbSendComm send{};
  ncclIbRecvComm recv{};
  ncclIbResiliencySend sendRes{};
  ncclIbResiliency recvRes{};

  NetIbResiliencyScene() {
    send.base.isSend = true;
    send.base.resiliency = &sendRes.base;
    recv.base.resiliency = &recvRes;
    sendRes.base.baseComm = &send.base;
    recvRes.baseComm = &recv.base;
    for (ncclIbNetCommBase* base : {&send.base, &recv.base}) {
      base->nqps = kDataQps;
      base->nDataQps = kDevs;
      base->vProps.ndevs = kDevs;
      for (int i = 0; i < kDataQps; ++i) {
        base->qps[i].qp = &verbs[i % kDevs].qps[i / kDevs];
        base->qps[i].devIndex = i % kDevs;
        base->activeQps[i] = &base->qps[i];
      }
    }
    for (int d = 0; d < kDevs; ++d) {
      for (ncclIbNetCommDevBase* dev : {&send.devs[d].base, &recv.devs[d].base}) {
        dev->ibDevN = d;
        dev->pd = &verbs[d].pd;
      }
    }
    for (ncclIbResiliency* res : {&sendRes.base, &recvRes}) {
      res->ndevs = kDevs;
      res->nProbingQps = kDevs;
      res->nPortRecoveryQps = kDevs;
      for (int d = 0; d < kDevs; ++d) {
        res->devs[d].state.store(ncclIbResiliencyDevStateOk, std::memory_order_release);
        res->devs[d].probingCq = &verbs[d].cq;
        res->devs[d].portRecoveryCq = &verbs[d].cq;
        res->probingQps[d].qp = &verbs[d].qps[kProbingQp];
        res->probingQps[d].devIndex = d;
        res->portRecoveryQps[d].qp = &verbs[d].qps[kRecoveryQp];
        res->portRecoveryQps[d].devIndex = d;
      }
    }
  }
  NetIbResiliencyScene(const NetIbResiliencyScene&) = delete;
  NetIbResiliencyScene& operator=(const NetIbResiliencyScene&) = delete;

  // g_netIbQpCreate stand-in. Like ibv_create_qp it wants the CQ and the PD on one device, and the QP lands there.
  ncclResult_t CreateQp(ncclIbQp* qp, ncclIbQpCreateAttr* attr) {
    if (attr->pd == nullptr || attr->cq == nullptr || attr->pd->context == nullptr ||
        attr->cq->context != attr->pd->context) {
      return ncclSystemError;
    }
    IbVerbsScene* dev = IbVerbsScene::Of(attr->pd->context);
    int& created = created_[dev - verbs.data()];
    if (kFirstCreatedQp + created >= IbVerbsScene::kQps) return ncclSystemError;
    qp->qp = &dev->qps[kFirstCreatedQp + created++];
    return ncclSuccess;
  }

 private:
  std::array<int, kDevs> created_{};
};

#endif  // RCCL_TEST_HOST_TRANSPORT_NET_IB_NET_IB_RESILIENCY_SCENE_H_
