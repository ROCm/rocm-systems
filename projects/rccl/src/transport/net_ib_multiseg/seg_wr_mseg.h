/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

#ifndef RCCL_TRANSPORT_MSEG_WR_MSEG_H_
#define RCCL_TRANSPORT_MSEG_WR_MSEG_H_

#include <string.h>

// Same selector as ibvwrap.h. Host tests compile against ibvcore.h.
#ifdef NCCL_BUILD_RDMA_CORE
#include <infiniband/verbs.h>
#else
#include "ibvcore.h"
#endif

#include "seg_mseg.h"

// RCCL: verbs fill for one slice of a local and a remote segment table. The caller links the chain and writes its
// own tail (signal, imm).
static inline void rcclIbSegFillWr(const struct rcclIbSegTable* local, uint32_t lkey,
                                   const struct rcclIbSegTable* remote, uint32_t rkey,
                                   const struct rcclIbSegSlice* slice, enum ibv_wr_opcode opcode,
                                   struct ibv_send_wr* wr, struct ibv_sge* sge) {
  memset(wr, 0, sizeof(*wr));
  memset(sge, 0, sizeof(*sge));
  wr->opcode = opcode;
  wr->sg_list = sge;
  wr->num_sge = 1;
  wr->wr.rdma.remote_addr = remote->base + slice->remoteOff;
  wr->wr.rdma.rkey = rkey;
  sge->addr = (uintptr_t)(local->base + slice->localOff);
  sge->length = (uint32_t)slice->len;
  sge->lkey = lkey;
}

// Returns the WR count, or -1 if the chain exceeds maxWr or runs past a table. With covered set, the chain stops at
// maxWr or at the end of a table instead, *covered is the bytes it moves, and -1 means it moved none.
struct rcclIbSegWrIn {
  const struct rcclIbSegTable* local;
  const uint32_t* lkeys;
  const struct rcclIbSegTable* remote;
  const uint32_t* rkeys;
  uint64_t localOff;
  uint64_t remoteOff;
  uint64_t len;
  enum ibv_wr_opcode opcode;
  int maxWr;
  uint64_t* covered;
};

static inline int rcclIbSegAppendWrs(const struct rcclIbSegWrIn* in, struct ibv_send_wr* wrs, struct ibv_sge* sges) {
  if (in == NULL || in->remote == NULL || in->rkeys == NULL) return -1;
  if (in->len == 0) return 0;
  if (wrs == NULL || sges == NULL || in->local == NULL || in->lkeys == NULL) return -1;
  struct rcclIbSegCursor cur;
  rcclIbSegCursorInit(&cur, in->local, in->remote, in->localOff, in->remoteOff, in->len);
  int n = 0;
  struct rcclIbSegSlice slice;
  while (rcclIbSegCursorNext(&cur, in->local, in->remote, &slice)) {
    if (n >= in->maxWr) return -1;
    rcclIbSegFillWr(in->local, in->lkeys[slice.localSeg], in->remote, in->rkeys[slice.remoteSeg], &slice, in->opcode,
                    &wrs[n], &sges[n]);
    if (n > 0) wrs[n - 1].next = &wrs[n];
    n++;
    if (in->covered != NULL && n == in->maxWr) break;
  }
  if (in->covered != NULL) {
    *in->covered = in->len - cur.remaining;
    return n > 0 ? n : -1;
  }
  if (cur.remaining != 0) return -1;
  return n;
}

#endif
