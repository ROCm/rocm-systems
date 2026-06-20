/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net_ib_ops_fault.h"

#ifdef ENABLE_FAULT_INJECTION

#include "core.h"  /* WARN / INFO */
#include <cerrno>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace {

// Per-QP fault configuration. A zero/disarmed value leaves the corresponding op
// forwarding to the saved real op.
struct QpFault {
  int  sendErrno      = 0;      // returned from post_send when != 0
  int  recvErrno      = 0;      // returned from post_recv when != 0
  int  pollWcStatus   = 0;      // ibv_wc_status applied when != 0 (0 == IBV_WC_SUCCESS)
  int  pollInjectCount = 0;     // remaining completions to corrupt; <0 == unlimited
  bool injectWhenIdle = false;  // synthesize an error WC when real poll returns 0
};

struct OpsEntry {
  int (*realPostSend)(struct ibv_qp*, struct ibv_send_wr*, struct ibv_send_wr**) = nullptr;
  int (*realPostRecv)(struct ibv_qp*, struct ibv_recv_wr*, struct ibv_recv_wr**) = nullptr;
  int (*realPollCq)(struct ibv_cq*, int, struct ibv_wc*) = nullptr;
  // Keyed by qp_num; NCCL_IB_OPS_FAULT_QP_ANY applies context-wide.
  std::unordered_map<uint32_t, QpFault> qpFaults;
};

std::mutex g_faultMtx;
std::unordered_map<struct ibv_context*, OpsEntry> g_ctxFaults;

// wr_id for synthesized idle completions; defined in the header and tied to the
// receiver's wr_id ranges by a static_assert in p2p.cc. Low byte 0 keeps the
// sender slot decode unchanged.
static constexpr uint64_t kSynthIdleWrId = NCCL_IB_OPS_FAULT_SYNTH_WR_ID;

// Look up the QpFault for (entry, qpNum), preferring an exact qp_num match and
// falling back to the context-wide ANY entry. Returns nullptr if neither armed.
// Caller must hold g_faultMtx.
QpFault* findFault(OpsEntry& entry, uint32_t qpNum) {
  auto it = entry.qpFaults.find(qpNum);
  if (it != entry.qpFaults.end()) return &it->second;
  auto any = entry.qpFaults.find(NCCL_IB_OPS_FAULT_QP_ANY);
  if (any != entry.qpFaults.end()) return &any->second;
  return nullptr;
}

// ── Shims ──────────────────────────────────────────────────────────────────

int shimPostSend(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr) {
  int (*real)(struct ibv_qp*, struct ibv_send_wr*, struct ibv_send_wr**) = nullptr;
  int errnoVal = 0;
  {
    std::lock_guard<std::mutex> lk(g_faultMtx);
    auto ctxIt = g_ctxFaults.find(qp->context);
    if (ctxIt != g_ctxFaults.end()) {
      real = ctxIt->second.realPostSend;
      QpFault* fault = findFault(ctxIt->second, qp->qp_num);
      if (fault) errnoVal = fault->sendErrno;
    }
  }
  if (errnoVal != 0) {
    if (bad_wr) *bad_wr = wr;
    return errnoVal;
  }
  // Lost real op: set *bad_wr (wrap_ibv_post_send logs it) and fail.
  if (!real) {
    if (bad_wr) *bad_wr = wr;
    return EFAULT;
  }
  return real(qp, wr, bad_wr);
}

int shimPostRecv(struct ibv_qp* qp, struct ibv_recv_wr* wr, struct ibv_recv_wr** bad_wr) {
  int (*real)(struct ibv_qp*, struct ibv_recv_wr*, struct ibv_recv_wr**) = nullptr;
  int errnoVal = 0;
  {
    std::lock_guard<std::mutex> lk(g_faultMtx);
    auto ctxIt = g_ctxFaults.find(qp->context);
    if (ctxIt != g_ctxFaults.end()) {
      real = ctxIt->second.realPostRecv;
      QpFault* fault = findFault(ctxIt->second, qp->qp_num);
      if (fault) errnoVal = fault->recvErrno;
    }
  }
  if (errnoVal != 0) {
    if (bad_wr) *bad_wr = wr;
    return errnoVal;
  }
  // Lost real op: set *bad_wr for ibv_post_* consistency and fail.
  if (!real) {
    if (bad_wr) *bad_wr = wr;
    return EFAULT;
  }
  return real(qp, wr, bad_wr);
}

int shimPollCq(struct ibv_cq* cq, int num_entries, struct ibv_wc* wc) {
  int (*real)(struct ibv_cq*, int, struct ibv_wc*) = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_faultMtx);
    auto ctxIt = g_ctxFaults.find(cq->context);
    if (ctxIt != g_ctxFaults.end()) real = ctxIt->second.realPollCq;
  }
  if (!real) return -1;

  int got = real(cq, num_entries, wc);
  if (got < 0) return got;

  std::lock_guard<std::mutex> lk(g_faultMtx);
  auto ctxIt = g_ctxFaults.find(cq->context);
  if (ctxIt == g_ctxFaults.end()) return got;
  OpsEntry& entry = ctxIt->second;

  // Rewrite the status of real completions whose qp_num is armed.
  for (int i = 0; i < got; i++) {
    QpFault* fault = findFault(entry, wc[i].qp_num);
    if (fault && fault->pollWcStatus != 0 && fault->pollInjectCount != 0) {
      wc[i].status = (enum ibv_wc_status)fault->pollWcStatus;
      if (fault->pollInjectCount > 0) fault->pollInjectCount--;
    }
  }

  // Synthesize an error completion when the queue was idle and an armed QP
  // requested idle injection. Only fabricate a single WC into slot 0.
  // NOTE: picks the first idle-armed QP on the context without checking it
  // belongs to this cq. Safe while only one QP is armed at a time; if several
  // QPs (on different CQs) are armed, filter by cq before fabricating.
  if (got == 0 && num_entries >= 1) {
    for (auto& kv : entry.qpFaults) {
      QpFault& fault = kv.second;
      if (fault.injectWhenIdle && fault.pollWcStatus != 0 && fault.pollInjectCount != 0) {
        memset(&wc[0], 0, sizeof(wc[0]));
        wc[0].status = (enum ibv_wc_status)fault.pollWcStatus;
        wc[0].qp_num = (kv.first == NCCL_IB_OPS_FAULT_QP_ANY) ? 0u : kv.first;
        wc[0].wr_id = kSynthIdleWrId;  // not 0; see kSynthIdleWrId
        if (fault.pollInjectCount > 0) fault.pollInjectCount--;
        got = 1;
        break;
      }
    }
  }
  return got;
}

}  // namespace

extern "C" {

ncclResult_t ncclIbOpsFaultInstall(struct ibv_context* ctx) {
  if (!ctx) return ncclInvalidArgument;
  std::lock_guard<std::mutex> lk(g_faultMtx);
  if (g_ctxFaults.find(ctx) != g_ctxFaults.end()) return ncclSuccess;  // already installed
  if (!ctx->ops.post_send || !ctx->ops.post_recv || !ctx->ops.poll_cq) {
    WARN("NET/IB ops-fault: context %p has NULL post_send/post_recv/poll_cq op; skipping install", ctx);
    return ncclSuccess;
  }
  OpsEntry entry;
  entry.realPostSend = ctx->ops.post_send;
  entry.realPostRecv = ctx->ops.post_recv;
  entry.realPollCq   = ctx->ops.poll_cq;
  g_ctxFaults.emplace(ctx, entry);
  ctx->ops.post_send = shimPostSend;
  ctx->ops.post_recv = shimPostRecv;
  ctx->ops.poll_cq   = shimPollCq;
  INFO(NCCL_NET, "NET/IB ops-fault: installed shims on context %p", ctx);
  return ncclSuccess;
}

ncclResult_t ncclIbOpsFaultRemove(struct ibv_context* ctx) {
  if (!ctx) return ncclInvalidArgument;
  std::lock_guard<std::mutex> lk(g_faultMtx);
  auto ctxIt = g_ctxFaults.find(ctx);
  if (ctxIt == g_ctxFaults.end()) return ncclSuccess;
  ctx->ops.post_send = ctxIt->second.realPostSend;
  ctx->ops.post_recv = ctxIt->second.realPostRecv;
  ctx->ops.poll_cq   = ctxIt->second.realPollCq;
  g_ctxFaults.erase(ctxIt);
  return ncclSuccess;
}

ncclResult_t ncclIbOpsFaultArmPostSend(struct ibv_context* ctx, uint32_t qpNum, int errnoVal) {
  if (!ctx) return ncclInvalidArgument;
  std::lock_guard<std::mutex> lk(g_faultMtx);
  auto ctxIt = g_ctxFaults.find(ctx);
  if (ctxIt == g_ctxFaults.end()) return ncclInvalidArgument;
  ctxIt->second.qpFaults[qpNum].sendErrno = errnoVal;
  return ncclSuccess;
}

ncclResult_t ncclIbOpsFaultArmPostRecv(struct ibv_context* ctx, uint32_t qpNum, int errnoVal) {
  if (!ctx) return ncclInvalidArgument;
  std::lock_guard<std::mutex> lk(g_faultMtx);
  auto ctxIt = g_ctxFaults.find(ctx);
  if (ctxIt == g_ctxFaults.end()) return ncclInvalidArgument;
  ctxIt->second.qpFaults[qpNum].recvErrno = errnoVal;
  return ncclSuccess;
}

ncclResult_t ncclIbOpsFaultArmPollCq(struct ibv_context* ctx, uint32_t qpNum,
                                     int wcStatus, int injectCount, bool injectWhenIdle) {
  if (!ctx) return ncclInvalidArgument;
  std::lock_guard<std::mutex> lk(g_faultMtx);
  auto ctxIt = g_ctxFaults.find(ctx);
  if (ctxIt == g_ctxFaults.end()) return ncclInvalidArgument;
  QpFault& fault = ctxIt->second.qpFaults[qpNum];
  fault.pollWcStatus = wcStatus;
  fault.pollInjectCount = injectCount;
  fault.injectWhenIdle = injectWhenIdle;
  return ncclSuccess;
}

ncclResult_t ncclIbOpsFaultClear(struct ibv_context* ctx) {
  if (!ctx) return ncclInvalidArgument;
  std::lock_guard<std::mutex> lk(g_faultMtx);
  auto ctxIt = g_ctxFaults.find(ctx);
  if (ctxIt == g_ctxFaults.end()) return ncclSuccess;
  ctxIt->second.qpFaults.clear();
  return ncclSuccess;
}

}  // extern "C"

#endif /* ENABLE_FAULT_INJECTION */
