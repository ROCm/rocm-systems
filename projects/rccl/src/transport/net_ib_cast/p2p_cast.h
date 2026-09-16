/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_P2P_H_
#define NET_IB_P2P_H_

#include "common_cast.h"

#define NCCL_IB_FLUSH_REQ_WR_ID_OFFSET 0x1000
static_assert(NCCL_IB_FLUSH_REQ_WR_ID_OFFSET > NET_IB_MAX_REQUESTS,
              "wr_id offset for flush requests must be greater than NET_IB_MAX_REQUESTS");
static_assert(NCCL_IB_FLUSH_REQ_WR_ID_OFFSET <= UINT64_MAX - NET_IB_MAX_REQUESTS,
              "wr_id for flush requests must fit in 64 bits since ibv_send_wr::wr_id is 64 bits");

ncclResult_t IbCastPostFifo(struct ncclIbRecvComm* comm, struct ncclIbRequest* req, int slot, int n);
ncclResult_t IbCastMultiSend(struct ncclIbSendComm* comm, int slot, int nqps, int startQpIndex, bool wrrSched,
                             bool useWriteOp);

static inline ncclResult_t IbCastRecvCommGetQpForCts(struct ncclIbRecvComm* recvComm, uint32_t id, ncclIbQp** qp) {
  // AINIC requires CTS and data to be posted on the same QP.
  if (recvComm->useCtsOffload) {
    *qp = recvComm->base.activeQps[id % recvComm->base.nqps];
  } else {
    int devIndex = id % recvComm->base.vProps.ndevs;
    // CTS message is always posted the first QP on the device
    int qpIndex = 0;
    IbCastCommBaseGetQpByIndex(&recvComm->base, devIndex, qpIndex, qp);
  }
  assert(*qp != NULL);
  return ncclSuccess;
}

// Query whether a receiver QP is CTS-posting. Receiver QPs are dual-role: all
// take data receives and a subset posts CTS, so ask the selector rather than
// restate its rule. On selector failure answer false, since every QP takes
// data, so "not CTS" is the truthful fallback and true would corrupt roles.
static inline bool IbCastRecvCommIsCtsQp(struct ncclIbRecvComm* recvComm, int qpIndex) {
  for (uint32_t id = 0; id < (uint32_t)recvComm->base.nqps; id++) {
    ncclIbQp* ctsQp = NULL;
    if (IbCastRecvCommGetQpForCts(recvComm, id, &ctsQp) != ncclSuccess) return false;
    if (ctsQp == &recvComm->base.qps[qpIndex]) return true;
  }
  return false;
}

static inline ncclResult_t IbCastRequestRetrieveAsIndex(ncclIbRequest* reqs, uint32_t reqIndex, ncclIbRequest** req) {
  if (reqIndex >= NET_IB_MAX_REQUESTS) {
    WARN("NET/IB: %s: Invalid request index %u. Not in the range [%u, %u). Cannot retrieve request.", __func__,
         reqIndex, 0u, NET_IB_MAX_REQUESTS);
    return ncclInternalError;
  }
  *req = &reqs[reqIndex];
  return ncclSuccess;
}

// CTS FIFO element and stride helpers.
// The CTS FIFO buffers are declared as 2D array elements of
// ncclIbSendFifo (64B), but when IbCastAinicCtsInlineData
// is true, then the CTS FIFO buffer for each slot (which holds
// CTS messages upto MAX_RECV requests) is used as a contiguous
// buffer of the 32-byte of ncclIbSendFifoCtsInline type.
//
// Callers resolve the correct row pointer (slots) accessing the
// base address of CTS fifo for a given slot as
//   slot_base = comm->ctsFifo[slot];
// The accessors below take that pre-resolved row pointer (volatile void*)
// and only index by rxReqId within the row using the correct element type.

// --- Field accessors (layout-aware, row-based) ---
static inline uint32_t ctsFifoNreqs(volatile void* slotBase, uint32_t rxReqId) {
  if (IbCastAinicCtsInlineData) return (uint32_t)((volatile struct ncclIbSendFifoCtsInline*)slotBase)[rxReqId].nreqs;
  return ((volatile struct ncclIbSendFifo*)slotBase)[rxReqId].nreqs;
}
static inline uint64_t ctsFifoAddr(volatile void* slotBase, uint32_t rxReqId) {
  if (IbCastAinicCtsInlineData) return ((volatile struct ncclIbSendFifoCtsInline*)slotBase)[rxReqId].addr;
  return ((volatile struct ncclIbSendFifo*)slotBase)[rxReqId].addr;
}
static inline uint16_t ctsFifoRxReqIndex(volatile void* slotBase, uint32_t rxReqId) {
  if (IbCastAinicCtsInlineData)
    return (uint16_t)((volatile struct ncclIbSendFifoCtsInline*)slotBase)[rxReqId].rxReqIndex;
  return ((volatile struct ncclIbSendFifo*)slotBase)[rxReqId].rxReqIndex;
}
static inline uint32_t ctsFifoRkey(volatile void* slotBase, uint32_t rxReqId, int devIdx) {
  if (IbCastAinicCtsInlineData)
    // in SendFifoCtsInline only 1 key stored
    return ((volatile struct ncclIbSendFifoCtsInline*)slotBase)[rxReqId].rkeys[0];
  return ((volatile struct ncclIbSendFifo*)slotBase)[rxReqId].rkeys[devIdx];
}
static inline uint32_t ctsFifoIdx(volatile void* slotBase, uint32_t rxReqId) {
  if (IbCastAinicCtsInlineData) return ((volatile struct ncclIbSendFifoCtsInline*)slotBase)[rxReqId].idx;
  return ((volatile struct ncclIbSendFifo*)slotBase)[rxReqId].idx;
}
static inline size_t ctsFifoSize(volatile void* slotBase, uint32_t rxReqId) {
  if (IbCastAinicCtsInlineData) return (size_t)((volatile struct ncclIbSendFifoCtsInline*)slotBase)[rxReqId].size;
  return (size_t)((volatile struct ncclIbSendFifo*)slotBase)[rxReqId].size;
}
static inline uint32_t ctsFifoTag(volatile void* slotBase, uint32_t rxReqId) {
  if (IbCastAinicCtsInlineData) return (uint32_t)((volatile struct ncclIbSendFifoCtsInline*)slotBase)[rxReqId].tag;
  return ((volatile struct ncclIbSendFifo*)slotBase)[rxReqId].tag;
}

#endif // NET_IB_P2P_H_
