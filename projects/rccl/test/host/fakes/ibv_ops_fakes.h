/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Recording libibverbs provider ops for the post/poll paths that the inline
// wrappers in ibvwrap.h reach through ibv_context::ops. FakeIbvVerbs wires a
// context, QP, and CQ to them; tests script results and completions through the
// globals below. Kept out of ibvwrap_fakes so binaries that compile the real
// src/misc/ibvwrap.cc can link it too.

#ifndef RCCL_TEST_HOST_IBV_OPS_FAKES_H_
#define RCCL_TEST_HOST_IBV_OPS_FAKES_H_

#include <cstdint>
#include <deque>
#include <vector>

#include "ibvwrap.h"

// One work request as the provider saw it at post time.
struct IbvPostedSend {
  struct ibv_qp* qp;
  uint32_t qpNum;
  uint64_t wrId;
  enum ibv_wr_opcode opcode;
  int sendFlags;
  uint32_t immData;
  uint64_t remoteAddress;  // wr.atomic for atomic opcodes, wr.rdma otherwise
  uint32_t rkey;
  uint64_t compareAdd;     // atomic opcodes only
  struct ibv_sge sge;      // first scatter/gather entry, zero when the WR has none
};

extern std::vector<IbvPostedSend> g_ibvPostedSends;  // successful posts, one entry per WR in each chain
extern std::vector<struct ibv_qp*> g_ibvPostSendQps;  // every post_send call, failed ones included
extern std::vector<struct ibv_qp*> g_ibvPostRecvQps;  // every post_recv call, failed ones included
extern std::vector<struct ibv_cq*> g_ibvPolledCqs;    // every poll_cq call
extern std::deque<struct ibv_wc> g_ibvCompletions;    // consumed front to back by poll_cq
extern int g_ibvPostSendResult;
extern int g_ibvPostSendFailOnCall;  // 1-based post_send call that fails with EIO; -1 never
extern int g_ibvPostRecvResult;
extern int g_ibvPollResult;  // a negative value is returned as the poll error

int FakeIbvPostSend(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** badWr);
int FakeIbvPostRecv(struct ibv_qp* qp, struct ibv_recv_wr* wr, struct ibv_recv_wr** badWr);
int FakeIbvPollCq(struct ibv_cq* cq, int maxEntries, struct ibv_wc* completions);

// The QP and CQ point at the context, so the object must stay where it was built.
struct FakeIbvVerbs {
  struct ibv_context context {};
  struct ibv_qp qp {};
  struct ibv_cq cq {};

  explicit FakeIbvVerbs(uint32_t qpNum = 100, enum ibv_qp_type type = IBV_QPT_RC);
  FakeIbvVerbs(const FakeIbvVerbs&) = delete;
  FakeIbvVerbs& operator=(const FakeIbvVerbs&) = delete;
};

void ResetIbvOpsFakes();

#endif  // RCCL_TEST_HOST_IBV_OPS_FAKES_H_
