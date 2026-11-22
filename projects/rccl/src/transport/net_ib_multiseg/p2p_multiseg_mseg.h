/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_P2P_MULTISEG_MSEG_H_
#define NET_IB_P2P_MULTISEG_MSEG_H_

// RCCL: classic NET/IB work-request fill and CTS layout (p2p_multiseg.cc).
// p2p.cc calls these from IbMsegMultiSend, IbMsegIsend, IbMsegPostFifo, IbMsegIrecv and IbMsegIflush.

#include "common_mseg.h"

// RCCL: segmented chunk alignment; p2p.cc asserts it equals IB_WRITE_CHUNK_ALIGNMENT.
constexpr uint64_t rcclIbWriteChunkAlignment = 128;

// RCCL: for the segmented-send fixture tests. IbMsegArThreshold is defined in p2p.cc, the parameters in net_ib.
extern int64_t IbMsegArThreshold;
extern int64_t rcclParamIbSplitDataThreshold();
extern int64_t rcclParamIbGdrFlushGpuMemNoRelaxedOrdering();

// RCCL: true when a local or remote buffer spans several segments. The size record uses this.
bool rcclIbMultiSendNeedsSegmented(struct IbMsegSendComm* comm, int slot, int nreqs);
// RCCL: one QP's next post: one slice of each receive, advancing off[]. *last when it writes the tail.
ncclResult_t rcclIbFillSendWrs(struct IbMsegSendComm* comm, int slot, int devIndex, int remDevIdx, int qpIndex,
                               uint32_t* off, const uint32_t* end, bool first, uint64_t wrId, uint32_t immData,
                               bool needSizesWr, bool extraImmWr, bool flagFlush, struct ibv_send_wr** lastWr,
                               bool* last);
// RCCL: completion hook; true when it frees a QP of the slot still posting for its next slice.
bool rcclIbSendSliceDone(struct IbMsegSendComm* comm, int slot, int devIndex, uint32_t qpNum);
// RCCL: defined in p2p.cc. Posts next slices and then the queued slots, in fifo order.
ncclResult_t rcclIbSendProgress(struct IbMsegSendComm* comm);
ncclResult_t rcclIbSendLayoutsReady(struct IbMsegSendComm* comm, int slot, bool* ready);
ncclResult_t rcclIbSendLayoutReadDone(struct IbMsegSendComm* comm, uint64_t wrId);
ncclResult_t rcclIbSendLayoutCacheFree(struct IbMsegSendComm* comm);
// RCCL: CTS row for this slot. Multi-segment comms keep the layout prefix immediately before it.
static inline struct IbMsegSendFifo* rcclIbSendCtsRow(struct IbMsegSendComm* comm, int slot) {
  if (comm->ctsSlots != NULL) return comm->ctsSlots[slot].cts;
  return comm->ctsFifo[slot];
}
static inline const struct IbMsegSendFifo* rcclIbSendCtsRow(const struct IbMsegSendComm* comm, int slot) {
  if (comm->ctsSlots != NULL) return comm->ctsSlots[slot].cts;
  return comm->ctsFifo[slot];
}
static inline struct IbMsegSendFifo* rcclIbRecvCtsRow(struct IbMsegRecvComm* comm, int slot) {
  if (comm->ctsSlots != NULL) return comm->ctsSlots[slot].cts;
  return comm->remCtsFifo.elems[slot];
}
// RCCL: request r's layout, packed immediately before the CTS row. Valid for a long write of nreqs.
static inline struct IbMsegSegLayout* rcclIbCtsLayout(struct IbMsegSendFifo* ctsRow, int nreqs, int r) {
  return reinterpret_cast<struct IbMsegSegLayout*>(ctsRow) - nreqs + r;
}
static inline const volatile struct IbMsegSegLayout* rcclIbCtsLayout(const volatile struct IbMsegSendFifo* ctsRow,
                                                                    int nreqs, int r) {
  return reinterpret_cast<const volatile struct IbMsegSegLayout*>(ctsRow) - nreqs + r;
}

// RCCL: one CTS WR. A multi-segment slot extends the write backward over the layout prefix.
void rcclIbPrepareCtsWrite(struct IbMsegRecvComm* comm, struct IbMsegRequest* req, int slot, struct ibv_send_wr* wr);
ncclResult_t rcclIbIrecvCheckHandle(struct IbMsegRecvComm* comm, struct IbMsegMrHandle* mh, int i);
ncclResult_t rcclIbIrecvFillLayout(struct IbMsegRecvComm* comm, int slot, int i,
                                      struct IbMsegMrHandle* mhandleWrapper, bool* anyMultiSeg);
void rcclIbCtsFlagLayout(struct IbMsegSendFifo* localElem, int n, uint64_t idx);
// RCCL: true once every line of the row's nreqs layouts carries idx; a long write can land out of order.
bool rcclIbCtsLayoutsArrived(const volatile struct IbMsegSendFifo* ctsRow, int nreqs, uint64_t idx);

// RCCL: the MR covering [addr, addr+len) on devIndex: mrs[devIndex] for a single-segment handle, else the
// containing segment's MR, or NULL when the range crosses a segment boundary. IbMsegIflush reads with it.
ibv_mr* IbMsegMrForRange(const struct IbMsegMrHandle* h, uintptr_t addr, size_t len, int devIndex);

#endif // NET_IB_P2P_MULTISEG_MSEG_H_
