/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ibv_ops_fakes.h"

#include <algorithm>
#include <cerrno>

std::vector<IbvPostedSend> g_ibvPostedSends;
std::vector<struct ibv_qp*> g_ibvPostSendQps;
std::vector<struct ibv_qp*> g_ibvPostRecvQps;
std::vector<struct ibv_cq*> g_ibvPolledCqs;
std::deque<struct ibv_wc> g_ibvCompletions;
int g_ibvPostSendResult = 0;
int g_ibvPostSendFailOnCall = -1;
int g_ibvPostRecvResult = 0;
int g_ibvPollResult = 0;

static IbvPostedSend RecordPostedSend(struct ibv_qp* qp, const struct ibv_send_wr& wr) {
  IbvPostedSend posted{};
  posted.qp = qp;
  posted.qpNum = qp->qp_num;
  posted.wrId = wr.wr_id;
  posted.opcode = wr.opcode;
  posted.sendFlags = static_cast<int>(wr.send_flags);
  posted.immData = wr.imm_data;
  if (wr.opcode == IBV_WR_ATOMIC_FETCH_AND_ADD || wr.opcode == IBV_WR_ATOMIC_CMP_AND_SWP) {
    posted.remoteAddress = wr.wr.atomic.remote_addr;
    posted.rkey = wr.wr.atomic.rkey;
    posted.compareAdd = wr.wr.atomic.compare_add;
  } else {
    posted.remoteAddress = wr.wr.rdma.remote_addr;
    posted.rkey = wr.wr.rdma.rkey;
  }
  if (wr.num_sge > 0 && wr.sg_list != nullptr) posted.sge = wr.sg_list[0];
  return posted;
}

int FakeIbvPostSend(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** badWr) {
  g_ibvPostSendQps.push_back(qp);
  const int result =
      static_cast<int>(g_ibvPostSendQps.size()) == g_ibvPostSendFailOnCall ? EIO : g_ibvPostSendResult;
  if (result != 0) {
    if (badWr) *badWr = wr;
    return result;
  }
  for (struct ibv_send_wr* current = wr; current != nullptr; current = current->next) {
    g_ibvPostedSends.push_back(RecordPostedSend(qp, *current));
  }
  return 0;
}

int FakeIbvPostRecv(struct ibv_qp* qp, struct ibv_recv_wr* wr, struct ibv_recv_wr** badWr) {
  g_ibvPostRecvQps.push_back(qp);
  if (g_ibvPostRecvResult != 0 && badWr) *badWr = wr;
  return g_ibvPostRecvResult;
}

int FakeIbvPollCq(struct ibv_cq* cq, int maxEntries, struct ibv_wc* completions) {
  g_ibvPolledCqs.push_back(cq);
  if (g_ibvPollResult < 0) return g_ibvPollResult;
  const int count = std::min<int>(maxEntries, static_cast<int>(g_ibvCompletions.size()));
  for (int i = 0; i < count; ++i) {
    completions[i] = g_ibvCompletions.front();
    g_ibvCompletions.pop_front();
  }
  return count;
}

FakeIbvVerbs::FakeIbvVerbs(uint32_t qpNum, enum ibv_qp_type type) {
  context.ops.post_send = FakeIbvPostSend;
  context.ops.post_recv = FakeIbvPostRecv;
  context.ops.poll_cq = FakeIbvPollCq;
  qp.context = &context;
  qp.qp_num = qpNum;
  qp.qp_type = type;
  cq.context = &context;
}

void ResetIbvOpsFakes() {
  g_ibvPostedSends.clear();
  g_ibvPostSendQps.clear();
  g_ibvPostRecvQps.clear();
  g_ibvPolledCqs.clear();
  g_ibvCompletions.clear();
  g_ibvPostSendResult = 0;
  g_ibvPostSendFailOnCall = -1;
  g_ibvPostRecvResult = 0;
  g_ibvPollResult = 0;
}
