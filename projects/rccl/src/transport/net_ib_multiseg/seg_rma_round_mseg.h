/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

#ifndef RCCL_TRANSPORT_MSEG_RMA_ROUND_MSEG_H_
#define RCCL_TRANSPORT_MSEG_RMA_ROUND_MSEG_H_

#include <stdlib.h>

#include "seg_rma_mseg.h"
#include "seg_wr_mseg.h"

// RCCL: multi-segment RMA ops wider than one round of RCCL_RMA_MAX_DATA_WRS data WRs post the next round when the
// previous one is done. Ops queue per peer in request order, so a later op never passes an earlier one.

struct rcclRmaRoundOp {
  struct rcclRmaRoundOp* next;
  void* req;     // plugin request
  void* comm;    // plugin comm the op posts on
  void* qp;      // plugin QP
  void* localH;  // plugin segment handles
  void* remoteH;
  int localRank;
  int remoteRank;
  enum ibv_wr_opcode opcode;
  uint64_t localOff;   // where the next round starts
  uint64_t remoteOff;
  uint64_t remaining;  // bytes not yet in a posted round
  bool hasSignal;  // the signal rides the last round
  struct ibv_send_wr signalWr;
  struct ibv_sge signalSge;
  bool roundInFlight;  // a non-final round is posted
};

struct rcclRmaRoundQueue {
  struct rcclRmaRoundOp* head;
  struct rcclRmaRoundOp* tail;
};

// Per-peer queues of one context, nranks long; NULL with multi-segment off. The plugin allocates the array and
// the queued ops with its calloc, and this header releases them with free().
struct rcclRmaRoundQueues {
  struct rcclRmaRoundQueue* send;
};

// The plugin's half of a round.
struct rcclRmaRoundHooks {
  // Up to maxWr WRs from the op's current offsets; *covered is the bytes they move.
  ncclResult_t (*build)(const struct rcclRmaRoundOp* op, struct ibv_send_wr* wr, struct ibv_sge* sge, int maxWr,
                        int* nWr, uint64_t* covered);
  // Asked only with roundInFlight set: true while that round has not completed.
  bool (*outstanding)(const struct rcclRmaRoundOp* op);
  // Stamp and post one round; *posted is the WRs the provider accepted.
  ncclResult_t (*post)(struct rcclRmaRoundOp* op, struct ibv_send_wr* wr, int nWr, int* posted);
  // The op leaves the queue: its last round posted (ok), or a round failed after posted of nWr WRs.
  void (*retire)(struct rcclRmaRoundOp* op, bool ok, int nWr, int posted);
  // True for a request already failed elsewhere; its op is dropped without posting.
  bool (*failed)(const struct rcclRmaRoundOp* op);
};

// ---- WRs ----

// Every WR carries the request's wrId; with signalLast, only the chain's tail is signaled.
static inline void rcclRmaStampWrs(struct ibv_send_wr* wr, int nWr, uint64_t wrId, int signalLast) {
  for (int i = 0; i < nWr; i++) {
    wr[i].wr_id = wrId;
    wr[i].send_flags = (signalLast && rcclRmaWrIsSignaled(i, nWr)) ? IBV_SEND_SIGNALED : 0;
  }
}

// A partial round touches at most maxWr segments on each side, so the key windows clamp to that.
static inline void rcclRmaRoundClampWindow(int maxWr, int* nRemote, int* nLocal) {
  if (*nRemote > maxWr) *nRemote = maxWr;
  if (*nLocal > maxWr) *nLocal = maxWr;
}

// ---- Op setup ----

static inline struct rcclRmaRoundOp rcclRmaRoundDataOp(void* comm, void* qp, enum ibv_wr_opcode opcode, void* localH,
                                                       int localRank, uint64_t localOff, void* remoteH, int remoteRank,
                                                       uint64_t remoteOff, uint64_t size) {
  struct rcclRmaRoundOp op = {};
  op.comm = comm;
  op.qp = qp;
  op.opcode = opcode;
  op.localH = localH;
  op.localRank = localRank;
  op.localOff = localOff;
  op.remoteH = remoteH;
  op.remoteRank = remoteRank;
  op.remoteOff = remoteOff;
  op.remaining = size;
  return op;
}

// An atomic add of `add` at remoteAddr; the fetched value lands in the scratch SGE.
static inline void rcclRmaRoundSetSignal(struct rcclRmaRoundOp* op, uint64_t remoteAddr, uint32_t rkey, uint64_t add,
                                         uint64_t scratchAddr, uint32_t scratchLen, uint32_t scratchLkey) {
  op->hasSignal = true;
  memset(&op->signalWr, 0, sizeof(op->signalWr));
  op->signalWr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  op->signalWr.wr.atomic.remote_addr = remoteAddr;
  op->signalWr.wr.atomic.compare_add = add;
  op->signalWr.wr.atomic.rkey = rkey;
  op->signalWr.num_sge = 1;
  op->signalSge.addr = scratchAddr;
  op->signalSge.length = scratchLen;
  op->signalSge.lkey = scratchLkey;
}

// ---- Queues ----

static inline struct rcclRmaRoundQueue* rcclRmaRoundQueueOf(const struct rcclRmaRoundQueues* qs, uint32_t rank) {
  return qs->send == NULL ? NULL : &qs->send[rank];
}

static inline bool rcclRmaRoundQueueBusy(const struct rcclRmaRoundQueue* q) { return q != NULL && q->head != NULL; }

// Appends op; true when it became the head and the caller must pump.
static inline bool rcclRmaRoundPush(struct rcclRmaRoundQueue* q, struct rcclRmaRoundOp* op) {
  op->next = NULL;
  bool head = q->head == NULL;
  if (head) q->head = op;
  else q->tail->next = op;
  q->tail = op;
  return head;
}

static inline void rcclRmaRoundPop(struct rcclRmaRoundQueue* q) {
  struct rcclRmaRoundOp* op = q->head;
  q->head = op->next;
  if (q->head == NULL) q->tail = NULL;
  free(op);
}

static inline void rcclRmaRoundQueuesFree(struct rcclRmaRoundQueues* qs, int nranks) {
  if (qs->send == NULL) return;
  for (int i = 0; i < nranks; i++) {
    while (qs->send[i].head != NULL) rcclRmaRoundPop(&qs->send[i]);
  }
  free(qs->send);
  qs->send = NULL;
}

// Copies op into a node the plugin allocated, ready to push.
static inline void rcclRmaRoundFillNode(struct rcclRmaRoundOp* node, const struct rcclRmaRoundOp* op) {
  *node = *op;
  node->next = NULL;
  node->roundInFlight = false;
}

// ---- Rounds ----

static inline void rcclRmaRoundAppendSignal(const struct rcclRmaRoundOp* op, struct ibv_send_wr* wr,
                                            struct ibv_sge* sge, int* n) {
  wr[*n] = op->signalWr;
  sge[*n] = op->signalSge;
  wr[*n].sg_list = &sge[*n];
  wr[*n].next = NULL;
  if (*n > 0) wr[*n - 1].next = &wr[*n];
  (*n)++;
}

// The op's next round into wr/sge (RCCL_RMA_MAX_SIGNAL_WRS long); advances its offsets.
static inline ncclResult_t rcclRmaRoundBuild(const struct rcclRmaRoundHooks* hooks, struct rcclRmaRoundOp* op,
                                             struct ibv_send_wr* wr, struct ibv_sge* sge, int* nWr, bool* last) {
  int n = 0;
  if (op->remaining > 0) {
    uint64_t covered = 0;
    ncclResult_t ret = hooks->build(op, wr, sge, RCCL_RMA_MAX_DATA_WRS, &n, &covered);
    if (ret != ncclSuccess) return ret;
    op->localOff += covered;
    op->remoteOff += covered;
    op->remaining -= covered;
  }
  *last = op->remaining == 0;
  if (*last && op->hasSignal) rcclRmaRoundAppendSignal(op, wr, sge, &n);
  *nWr = n;
  return ncclSuccess;
}

// The first round, built on a copy of op. *direct: it is the whole op (or empty) and nothing is queued ahead, so the
// plugin posts it as one chain without queueing.
static inline ncclResult_t rcclRmaRoundFirst(const struct rcclRmaRoundHooks* hooks, const struct rcclRmaRoundQueue* q,
                                             const struct rcclRmaRoundOp* op, struct ibv_send_wr* wr,
                                             struct ibv_sge* sge, int* nWr, bool* direct) {
  struct rcclRmaRoundOp first = *op;
  bool last = true;
  ncclResult_t ret = rcclRmaRoundBuild(hooks, &first, wr, sge, nWr, &last);
  if (ret != ncclSuccess) return ret;
  *direct = *nWr == 0 || (last && !rcclRmaRoundQueueBusy(q));
  return ncclSuccess;
}

// Posts the head's next round. True while a non-final round is in flight and the op stays queued.
static inline bool rcclRmaRoundPostNext(const struct rcclRmaRoundHooks* hooks, struct rcclRmaRoundOp* op) {
  struct ibv_send_wr wr[RCCL_RMA_MAX_SIGNAL_WRS];
  struct ibv_sge sge[RCCL_RMA_MAX_SIGNAL_WRS];
  int n = 0, posted = 0;
  bool last = true;
  ncclResult_t ret = rcclRmaRoundBuild(hooks, op, wr, sge, &n, &last);
  if (ret == ncclSuccess) ret = hooks->post(op, wr, n, &posted);
  if (ret != ncclSuccess || posted < n) {
    hooks->retire(op, false, n, posted);
    return false;
  }
  if (last) {
    hooks->retire(op, true, n, posted);
    return false;
  }
  op->roundInFlight = true;
  return true;
}

// Posts rounds from the head until one is in flight or the queue is empty. A failed round retires its op; the ops
// behind it still post.
static inline void rcclRmaRoundPump(const struct rcclRmaRoundHooks* hooks, struct rcclRmaRoundQueue* q) {
  if (q == NULL) return;
  while (q->head != NULL) {
    struct rcclRmaRoundOp* op = q->head;
    if (!hooks->failed(op)) {
      if (op->roundInFlight && hooks->outstanding(op)) return;
      if (rcclRmaRoundPostNext(hooks, op)) return;
    }
    rcclRmaRoundPop(q);
  }
}

// For plugins that complete a round by its tail CQE: true if that CQE ends req's in-flight round, which frees the
// head for its next round without completing req.
static inline bool rcclRmaRoundTailDone(struct rcclRmaRoundQueue* q, const void* req) {
  if (q == NULL || q->head == NULL || q->head->req != req || !q->head->roundInFlight) return false;
  q->head->roundInFlight = false;
  return true;
}

#endif
