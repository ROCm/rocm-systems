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

#endif
