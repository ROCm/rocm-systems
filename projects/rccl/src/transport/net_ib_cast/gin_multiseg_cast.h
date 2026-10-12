/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Multi-segment RMA proxy paths, included once by gin.cc after IbCastRmaIbProxyCtx.
#ifndef NET_IB_CAST_GIN_MULTISEG_CAST_H_
#define NET_IB_CAST_GIN_MULTISEG_CAST_H_

#include "alloc.h"

#include <algorithm>
#include <sched.h>
#include <time.h>

extern int64_t ncclParamIbCastTimeout();
extern int64_t ncclParamIbCastRetryCnt();

struct IbCastRmaProxySegMrHandle {
  int nSegments;
  // segOff[0]==0, segOff[nSegments]==size; per-segment local MRs.
  // base_vas indexed [rank*nSegments + seg].
  // rkeys indexed [rank][seg][registration slot].
  // ibDevNs indexed [rank][registration slot] = physical ibDevN, or -1.
  // rankSegOff: per-rank [nSegments+1] after allgather (asymmetric windows).
  size_t segOff[NCCL_RMA_MAX_SEGMENTS + 1];
  struct ncclIbMrHandle* mrHandle[NCCL_RMA_MAX_SEGMENTS];
  uintptr_t* base_vas;
  uint32_t* rkeys;
  int* ibDevNs;
  size_t* rankSegOff;
};

// Fixed-size record every rank sends even on local failure; magic/version fail mismatched layouts collectively.
#define IBCAST_RMA_REGISTRATION_MAGIC 0x524d5347u
#define IBCAST_RMA_REGISTRATION_VERSION 1u
struct IbCastRmaProxyRegistration {
  uint32_t magic;
  uint32_t version;
  ncclResult_t status;
  int nSegments;
  size_t segOff[NCCL_RMA_MAX_SEGMENTS + 1];
};

static inline size_t IbCastRmaRkeyIndex(int nSegments, int rank, int seg, int devSlot) {
  return ((size_t)rank * nSegments + (size_t)seg) * NCCL_IB_MAX_DEVS_PER_NIC + (size_t)devSlot;
}

static inline uint32_t IbCastRmaRemoteRkey(const struct IbCastRmaProxySegMrHandle* h, int rank, int seg, int devSlot) {
  return h->rkeys[IbCastRmaRkeyIndex(h->nSegments, rank, seg, devSlot)];
}

static int IbCastRmaDevSlot(const struct IbCastRmaProxySegMrHandle* h, int rank, int ibDevN) {
  if (h == NULL) return -1;
  return ncclRmaDevSlotOf(h->ibDevNs, rank, ibDevN, NCCL_IB_MAX_DEVS_PER_NIC);
}

static inline const size_t* IbCastRmaRankOff(const struct IbCastRmaProxySegMrHandle* h, int rank) {
  return ncclRmaPeerSegOff(h->rankSegOff, h->segOff, rank);
}

static inline int IbCastRmaSegOf(const struct IbCastRmaProxySegMrHandle* h, int rank, uint64_t off) {
  return ncclRmaSegIndexOf(IbCastRmaRankOff(h, rank), h->nSegments, off);
}

static inline uint64_t IbCastRmaMrBytes(const struct IbCastRmaProxySegMrHandle* h, int rank) {
  return IbCastRmaRankOff(h, rank)[h->nSegments];
}

static inline bool IbCastRmaRangeOk(const struct IbCastRmaProxySegMrHandle* h, int rank, uint64_t off, size_t size) {
  return ncclRmaOffsetRangeOk(h->nSegments, IbCastRmaRankOff(h, rank), off, size);
}

// Build a chained RDMA WR list, splitting at segment boundaries so every WR
// stays within a single physical segment's MR.
//
// Paired mode (flushSge == NULL): move `size` bytes between a local and a
// remote symmetric buffer, splitting on both sides; each WR carries one local
// lkey and one remote rkey.
//
// Flush mode (flushSge != NULL): the local handle is ignored; emit one tiny
// RDMA_READ per remote segment touched by [remoteOff, remoteOff+size), each
// landing a single byte in the caller's flush scratch.
static ncclResult_t IbCastRmaBuildSegmentedWrs(struct ibv_send_wr* wr, struct ibv_sge* sge, int maxWr, int* nWr,
                                               enum ibv_wr_opcode opcode, struct ncclIbNetCommBase* commBase,
                                               const struct ncclIbQp* qp, struct IbCastRmaProxySegMrHandle* localH,
                                               int localRank, uint64_t localOff,
                                               struct IbCastRmaProxySegMrHandle* remoteH, int remoteRank,
                                               uint64_t remoteOff, size_t size, const struct ibv_sge* flushSge) {
  struct ncclIbNetCommDevBase* localDev = IbCastGetNetCommDevBase(commBase, qp->devIndex);
  int localIbDevN = localDev->ibDevN;
  int remoteIbDevN = commBase->remDevs[qp->remDevIdx].ibv_dev_index;
  int rkeyIbDevN = ncclRmaRkeyIbDevN(localIbDevN, remoteIbDevN, flushSge != NULL);
  int remoteDevSlot = IbCastRmaDevSlot(remoteH, remoteRank, rkeyIbDevN);
  int localDevSlot = flushSge == NULL ? IbCastRmaDevSlot(localH, localRank, localIbDevN) : -1;
  if (remoteDevSlot < 0 || (flushSge == NULL && localDevSlot < 0)) {
    WARN("NET/IB-CAST/RMA: no registration slot for local/remote IB devices %d/%d", localIbDevN, remoteIbDevN);
    return ncclInternalError;
  }
  int n = 0;
  uint64_t lOff = localOff, rOff = remoteOff;
  size_t rem = size;
  while (rem > 0) {
    if (n >= maxWr) {
      WARN("NET/IB-CAST/RMA: transfer of %zu bytes spans more than %d segment slices", size, maxWr);
      return ncclInternalError;
    }
    const size_t* rOffs = IbCastRmaRankOff(remoteH, remoteRank);
    int rs = IbCastRmaSegOf(remoteH, remoteRank, rOff);
    size_t remoteRemaining = rOffs[rs + 1] - rOff;

    int ls = 0;
    size_t localRemaining = SIZE_MAX;
    const size_t* lOffs = NULL;
    if (flushSge == NULL) {
      lOffs = IbCastRmaRankOff(localH, localRank);
      ls = IbCastRmaSegOf(localH, localRank, lOff);
      localRemaining = lOffs[ls + 1] - lOff;
    }
    // ibv_sge::length is 32-bit. Flush reads one byte but advances by the
    // complete portion of the physical segment; data WRs are additionally
    // split at UINT32_MAX by the shared classic/CAST helper.
    size_t chunk = flushSge == NULL ? ncclRmaSegmentSliceBytes(rem, localRemaining, remoteRemaining) :
                                      (rem < remoteRemaining ? rem : remoteRemaining);

    uintptr_t rAddr = remoteH->base_vas[(size_t)remoteRank * remoteH->nSegments + rs] + (rOff - rOffs[rs]);

    memset(&wr[n], 0, sizeof(wr[n]));
    memset(&sge[n], 0, sizeof(sge[n]));
    wr[n].opcode = opcode;
    wr[n].next = NULL;
    wr[n].wr.rdma.remote_addr = (uint64_t)rAddr;
    wr[n].wr.rdma.rkey = IbCastRmaRemoteRkey(remoteH, remoteRank, rs, remoteDevSlot);
    wr[n].sg_list = &sge[n];
    wr[n].num_sge = 1;
    if (flushSge != NULL) {
      sge[n] = *flushSge;
      sge[n].length = 1;
    } else {
      uintptr_t lAddr = localH->base_vas[(size_t)localRank * localH->nSegments + ls] + (lOff - lOffs[ls]);
      struct ibv_mr* lmr = localH->mrHandle[ls]->mrs[localDevSlot];
      if (lmr == NULL) {
        WARN("NET/IB-CAST/RMA: no local MR for segment %d device %d", ls, localIbDevN);
        return ncclInternalError;
      }
      sge[n].addr = (uintptr_t)lAddr;
      sge[n].length = (uint32_t)chunk;
      sge[n].lkey = lmr->lkey;
    }
    if (n > 0) wr[n - 1].next = &wr[n];

    lOff += chunk;
    rOff += chunk;
    rem -= chunk;
    n++;
  }
  *nWr = n;
  return ncclSuccess;
}

static ncclResult_t rcclIbCastRmaIbProxyConnectFinish(void** collComm, int nranks) {
  if (!rcclRmaMultiSegEnabled()) return ncclSuccess;
  // Registration must not fail locally before its first collective, so the
  // receive buffer is allocated here. It is page-aligned because allGather
  // registers whole pages for remote write.
  struct CastIbGinCollComm* cComm = (struct CastIbGinCollComm*)*collComm;
  ncclResult_t ret = ncclIbMalloc(&cComm->regConsensus, (size_t)nranks * sizeof(struct IbCastRmaProxyRegistration));
  if (ret != ncclSuccess) {
    IbCastGinIbCloseColl(*collComm);
    *collComm = NULL;
    return ret;
  }
  return ncclSuccess;
}

static ncclResult_t rcclIbCastRmaIbProxyRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type,
                                                       uint64_t offset, int fd, uint64_t mr_flags, void** mhandle) {
  struct CastIbGinCollComm* cComm = (struct CastIbGinCollComm*)collComm;
  struct IbCastRmaProxySegMrHandle* ginMrHandle = NULL;
  struct IbCastRmaProxyRegistration localRegistration = {};
  struct IbCastRmaProxyRegistration* registrations = (struct IbCastRmaProxyRegistration*)cComm->regConsensus;
  uintptr_t localVas[NCCL_RMA_MAX_SEGMENTS] = {};
  uint32_t localRkeys[NCCL_RMA_MAX_SEGMENTS * NCCL_IB_MAX_DEVS_PER_NIC] = {};
  int localIbDevNs[NCCL_IB_MAX_DEVS_PER_NIC];
  ncclResult_t ret = ncclSuccess;
  int nSeg = 1;
  int registered = 0;

  for (int d = 0; d < NCCL_IB_MAX_DEVS_PER_NIC; d++) localIbDevNs[d] = -1;

  *mhandle = NULL;
  ret = ncclCalloc(&ginMrHandle, 1);
  if (ret != ncclSuccess) goto reconcile;
  // calloc zeroes nSegments; fail paths below only dereg `registered` complete
  // handles, so a half-built ncclIbMrHandle is never passed to deregMr.

  // Count physical segments; ROCm/HIP describes only the first per export,
  // so multi-segment ranges register one MR per segment below.
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
  if (type == NCCL_PTR_CUDA && ncclCuMemEnable()) {
    CUdeviceptr base = 0;
    size_t baseSize = 0;
    NCCLCHECKGOTO(ncclCuMemGetAddressRange((CUdeviceptr)data, size, &base, &baseSize, &nSeg), ret, reconcile);
  }
#endif
  if (nSeg < 1) nSeg = 1;
  if (nSeg > NCCL_RMA_MAX_SEGMENTS) {
    WARN("NET/IB-CAST/RMA: buffer %p (size %zu) spans %d segments, exceeds NCCL_RMA_MAX_SEGMENTS=%d", data, size, nSeg,
         NCCL_RMA_MAX_SEGMENTS);
    ret = ncclInvalidUsage;
    goto reconcile;
  }
  ginMrHandle->nSegments = nSeg;
  ginMrHandle->segOff[0] = 0;

  if (nSeg == 1) {
    NCCLCHECKGOTO(IbCastRegMrDmaBufInternal(cComm->recvComm, data, size, type, offset, fd, mr_flags,
                                            (void**)&ginMrHandle->mrHandle[0]),
                  ret, reconcile);
    registered = 1;
    ginMrHandle->segOff[1] = size;
    localVas[0] = (uintptr_t)data;
  }
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
  else {
    uintptr_t segPtr = (uintptr_t)data;
    size_t remaining = size;
    size_t cum = 0;
    for (int s = 0; s < nSeg; s++) {
      CUdeviceptr segBase = 0;
      size_t segSize = 0;
      CUCHECKGOTO(cuMemGetAddressRange(&segBase, &segSize, (CUdeviceptr)segPtr), ret, reconcile);
      // Host VMM can report segBase==0; subtracting then wraps the length.
      size_t inSeg = (segBase == 0) ? segSize : segSize - (segPtr - (uintptr_t)segBase);
      size_t thisLen = remaining < inSeg ? remaining : inSeg;
      int segFd = -1;
      // The MR spans whole pages from segPtr's page at dmabuf offset 0 (reg.cc),
      // so the export must cover that same page range.
      static size_t hostPageSize = ncclOsGetPageSize();
      uintptr_t exportBase = segPtr & ~(uintptr_t)(hostPageSize - 1);
      size_t exportLen = segPtr + thisLen - exportBase;
      ALIGN_SIZE(exportLen, hostPageSize);
      CUCHECKGOTO(cuMemGetHandleForAddressRange((void*)&segFd, (CUdeviceptr)exportBase, exportLen,
                                                CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0),
                  ret, reconcile);
      ret = IbCastRegMrDmaBufInternal(cComm->recvComm, (void*)segPtr, thisLen, type, 0ULL, segFd, mr_flags,
                                      (void**)&ginMrHandle->mrHandle[s]);
      (void)close(segFd);
      if (ret != ncclSuccess) goto reconcile;
      registered = s + 1;
      localVas[s] = segPtr;
      cum += thisLen;
      ginMrHandle->segOff[s + 1] = cum;
      segPtr += thisLen;
      remaining -= thisLen;
    }
    if (remaining != 0) {
      WARN("NET/IB-CAST/RMA: enumerated %d/%d segments for buffer %p but %zu bytes remain unregistered", registered,
           nSeg, data, remaining);
      ret = ncclInvalidUsage;
      goto reconcile;
    }
    INFO(NCCL_NET | NCCL_REG, "NET/IB-CAST/RMA: registered multi-segment buffer %p size %zu as %d DMA-BUF MRs", data,
         size, nSeg);
  }
#endif

  {
    struct ncclIbNetCommBase* recvBase = (struct ncclIbNetCommBase*)cComm->recvComm;
    int ndevs = recvBase->vProps.ndevs;
    memset(localRkeys, 0, sizeof(localRkeys));
    if (ndevs < 1 || ndevs > NCCL_IB_MAX_DEVS_PER_NIC) {
      WARN("NET/IB-CAST/RMA: invalid ndevs %d for buffer %p", ndevs, data);
      ret = ncclInternalError;
      goto reconcile;
    }
    for (int s = 0; s < nSeg; s++) {
      for (int d = 0; d < ndevs; d++) {
        struct ibv_mr* mr = ginMrHandle->mrHandle[s]->mrs[d];
        if (mr == NULL) {
          WARN("NET/IB-CAST/RMA: missing MR for segment %d device %d", s, d);
          ret = ncclInternalError;
          goto reconcile;
        }
        localRkeys[(size_t)s * NCCL_IB_MAX_DEVS_PER_NIC + d] = mr->rkey;
      }
    }
    for (int d = 0; d < ndevs; d++) {
      localIbDevNs[d] = IbCastGetNetCommDevBase(recvBase, d)->ibDevN;
    }
  }

reconcile:
  localRegistration.magic = IBCAST_RMA_REGISTRATION_MAGIC;
  localRegistration.version = IBCAST_RMA_REGISTRATION_VERSION;
  localRegistration.status = ret;
  localRegistration.nSegments = nSeg;
  if (ncclRmaRegistrationHandleReady(ginMrHandle, nSeg))
    memcpy(localRegistration.segOff, ginMrHandle->segOff, sizeof(size_t) * (nSeg + 1));
  NCCLCHECKGOTO(cComm->allGather(cComm, &localRegistration, registrations, sizeof(localRegistration)), ret, fail);

  for (int r = 0; r < cComm->nranks; r++) {
    if (registrations[r].magic != IBCAST_RMA_REGISTRATION_MAGIC ||
        registrations[r].version != IBCAST_RMA_REGISTRATION_VERSION) {
      WARN("NET/IB-CAST/RMA: rank %d has unsupported registration transcript magic/version 0x%x/%u", r,
           registrations[r].magic, registrations[r].version);
      ret = ncclInvalidUsage;
      goto fail;
    }
    if (registrations[r].status != ncclSuccess) {
      WARN("NET/IB-CAST/RMA: symmetric registration rejected because rank %d failed with error %d", r,
           registrations[r].status);
      ret = registrations[r].status;
      goto fail;
    }
  }

  for (int r = 0; r < cComm->nranks; r++) {
    if (!ncclRmaSegmentCountsMatch(nSeg, registrations[r].nSegments)) {
      WARN("NET/IB-CAST/RMA: buffer %p segment count differs on rank %d (%d vs %d)", data, r,
           registrations[r].nSegments, nSeg);
      ret = ncclInvalidUsage;
      goto fail;
    }
    if (!ncclRmaSegOffTableValid(registrations[r].segOff, nSeg)) {
      WARN("NET/IB-CAST/RMA: buffer %p segment offsets from rank %d are not ordered from 0", data, r);
      ret = ncclInvalidUsage;
      goto fail;
    }
  }

  // Allocation failures are also reconciled before the VA/rkey gathers.
  ret = ncclCalloc(&ginMrHandle->rankSegOff, (size_t)cComm->nranks * (NCCL_RMA_MAX_SEGMENTS + 1));
  if (ret == ncclSuccess) {
    for (int r = 0; r < cComm->nranks; r++) {
      memcpy(ginMrHandle->rankSegOff + (size_t)r * (NCCL_RMA_MAX_SEGMENTS + 1), registrations[r].segOff,
             sizeof(size_t) * (nSeg + 1));
    }
  }
  if (ret == ncclSuccess) ret = ncclCalloc(&ginMrHandle->base_vas, (size_t)cComm->nranks * nSeg);
  if (ret == ncclSuccess)
    ret = ncclCalloc(&ginMrHandle->rkeys, (size_t)cComm->nranks * nSeg * NCCL_IB_MAX_DEVS_PER_NIC);
  if (ret == ncclSuccess)
    ret = ncclCalloc(&ginMrHandle->ibDevNs, (size_t)cComm->nranks * NCCL_IB_MAX_DEVS_PER_NIC);
  localRegistration.status = ret;
  NCCLCHECKGOTO(cComm->allGather(cComm, &localRegistration, registrations, sizeof(localRegistration)), ret, fail);
  for (int r = 0; r < cComm->nranks; r++) {
    if (registrations[r].status != ncclSuccess) {
      ret = registrations[r].status;
      goto fail;
    }
  }

  NCCLCHECKGOTO(cComm->allGather(cComm, localVas, ginMrHandle->base_vas, sizeof(uintptr_t) * nSeg), ret, fail);
  NCCLCHECKGOTO(cComm->allGather(cComm, localRkeys, ginMrHandle->rkeys,
                                 sizeof(uint32_t) * nSeg * NCCL_IB_MAX_DEVS_PER_NIC),
                ret, fail);
  NCCLCHECKGOTO(cComm->allGather(cComm, localIbDevNs, ginMrHandle->ibDevNs, sizeof(int) * NCCL_IB_MAX_DEVS_PER_NIC),
                ret, fail);

  *mhandle = ginMrHandle;
  return ncclSuccess;

fail:
  if (ginMrHandle) {
    for (int s = 0; s < registered; s++) {
      if (ginMrHandle->mrHandle[s]) (void)netIbCast.deregMr(cComm->recvComm, ginMrHandle->mrHandle[s]);
    }
    free(ginMrHandle->base_vas);
    free(ginMrHandle->rkeys);
    free(ginMrHandle->ibDevNs);
    free(ginMrHandle->rankSegOff);
    free(ginMrHandle);
  }
  return ret;
}

static ncclResult_t rcclIbCastRmaIbProxyDeregMrSym(void* collComm, void* mhandle) {
  struct CastIbGinCollComm* cComm = (struct CastIbGinCollComm*)collComm;
  struct IbCastRmaProxySegMrHandle* ginMrHandle = (struct IbCastRmaProxySegMrHandle*)mhandle;
  ncclResult_t ret = ncclSuccess;

  for (int s = 0; s < ginMrHandle->nSegments; s++) {
    if (ginMrHandle->mrHandle[s]) {
      ncclResult_t r = netIbCast.deregMr(cComm->recvComm, ginMrHandle->mrHandle[s]);
      if (ret == ncclSuccess && r != ncclSuccess) ret = r;
    }
  }
  free(ginMrHandle->base_vas);
  free(ginMrHandle->rkeys);
  free(ginMrHandle->ibDevNs);
  free(ginMrHandle->rankSegOff);
  free(ginMrHandle);
  return ret;
}

static ncclResult_t IbCastRmaIbProxyGetSendComm(struct IbCastRmaIbProxyCtx* ginProxyCtx, int rank,
                                                struct ncclIbSendComm** commPtr) {
  *commPtr = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
  if (*commPtr == NULL) {
    WARN("NET/IB-CAST/RMA: trying to send to non-connected peer %d", rank);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}
static ncclResult_t IbCastRmaIbProxyGetRecvComm(struct IbCastRmaIbProxyCtx* ginProxyCtx, int rank,
                                                struct ncclIbRecvComm** commPtr) {
  *commPtr = (struct ncclIbRecvComm*)ginProxyCtx->fullRecvComm[rank];
  if (*commPtr == NULL) {
    WARN("NET/IB-CAST/RMA: trying to receive from non-connected peer %d", rank);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

static ncclResult_t IbCastRmaRefundWrs(struct ncclIbRequest* req, int nWrs) {
  if (nWrs <= 0) return ncclSuccess;
  if (req->rmaNwrs < nWrs || req->base->rmaWrsOutstanding < nWrs) {
    WARN("NET/IB-CAST/RMA: work-request credit underflow");
    return ncclInternalError;
  }
  req->base->rmaWrsOutstanding -= nWrs;
  req->rmaNwrs -= nWrs;
  return ncclSuccess;
}

static ncclResult_t IbCastRmaAccountCompletion(struct ncclIbRequest* wcReq, int devIndex, enum ibv_wc_status status) {
  if (wcReq->events[devIndex] <= 0) {
    // After a QP error every flushed WQE completes with the owner's wr_id; only the first owns the event.
    if (status != IBV_WC_SUCCESS) return ncclSuccess;
    WARN("NET/IB-CAST/RMA: completion has no matching request event");
    return ncclInternalError;
  }
  wcReq->events[devIndex]--;
  if (wcReq->events[devIndex] == 0) return IbCastRmaRefundWrs(wcReq, wcReq->rmaNwrs);
  return ncclSuccess;
}

static ncclResult_t IbCastRmaPollCq(struct ncclIbNetCommBase* base, struct ncclIbNetCommDevBase* devBase, int devIndex,
                                   int* nCompleted) {
  struct ibv_wc wc[4];
  NCCLCHECK(wrap_ibv_poll_cq(devBase->cq, 4, wc, nCompleted));
  ncclResult_t batchRet = ncclSuccess;
  for (int i = 0; i < *nCompleted; i++) {
    ncclResult_t oneRet = ncclSuccess;
    if (wc[i].status != IBV_WC_SUCCESS) {
      char localGidString[INET6_ADDRSTRLEN] = "";
      char remoteGidString[INET6_ADDRSTRLEN] = "";
      const char *localGidStr = NULL, *remoteGidStr = NULL;
      if (devBase->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
        localGidStr = ibvGetGidStr(&devBase->gidInfo.localGid, localGidString, sizeof(localGidString));
        remoteGidStr = ibvGetGidStr(&base->remDevs[devIndex].remoteGid, remoteGidString, sizeof(remoteGidString));
      }
      char line[SOCKET_NAME_MAXLEN + 1];
      union ncclSocketAddress addr;
      const char* typeStr = "Unknown";
      struct ncclSocket* sock = &base->sock;
      if (wc[i].wr_id < NET_IB_MAX_REQUESTS) {
        struct ncclIbRequest* wcReq = base->reqs + wc[i].wr_id;
        if (wcReq->type >= 0 && wcReq->type <= NCCL_NET_IB_REQ_FAILED) typeStr = IbCastReqTypeStr[wcReq->type];
        if (wcReq->sock) sock = wcReq->sock;
      }
      ncclSocketGetAddr(sock, &addr);
      char* hcaName = devBase->pd->context->device->name;
      WARN("NET/IB-CAST/RMA: Got completion from peer %s with status=%s(%d) opcode=%s(%d) vendor_err=%u wr_id=%lu "
           "(%s)%s%s%s%s hca %s",
           ncclSocketToString(&addr, line), ibvWcStatusStr(wc[i].status), wc[i].status, ibvWcOpcodeStr(wc[i].opcode),
           wc[i].opcode, wc[i].vendor_err, (unsigned long)wc[i].wr_id, typeStr, localGidStr ? " localGid " : "",
           localGidString, remoteGidStr ? " remoteGids" : "", remoteGidString, hcaName);
      oneRet = ncclRemoteError;
    }
    if (wc[i].wr_id >= NET_IB_MAX_REQUESTS) {
      WARN("NET/IB-CAST/RMA: completion has invalid request id %lu", wc[i].wr_id);
      if (oneRet == ncclSuccess) oneRet = ncclInternalError;
    } else {
      struct ncclIbRequest* wcReq = base->reqs + wc[i].wr_id;
      ncclResult_t acc = IbCastRmaAccountCompletion(wcReq, devIndex, wc[i].status);
      if (oneRet == ncclSuccess) oneRet = acc;
      // The CQ may be drained by another request's Test(); the owner's next Test() must still fail.
      if (wc[i].status != IBV_WC_SUCCESS && wcReq->type != NCCL_NET_IB_REQ_UNUSED)
        wcReq->type = NCCL_NET_IB_REQ_FAILED;
    }
    if (batchRet == ncclSuccess) batchRet = oneRet;
  }
  return batchRet;
}

// Outlast the HCA's own retries (4.096us * 2^IB_TIMEOUT per attempt, IB_RETRY_CNT + 1
// attempts) so a recovering link does not abort the post; never below 2 s.
static uint64_t IbCastRmaCreditTimeoutNs() {
  const uint64_t kMinNs = 2ull * NSEC_PER_SEC;
  int64_t timeout = std::min<int64_t>(std::max<int64_t>(ncclParamIbCastTimeout(), 0), 31);
  int64_t retry = std::min<int64_t>(std::max<int64_t>(ncclParamIbCastRetryCnt(), 0), 7);
  uint64_t ns = (4096ull << timeout) * (uint64_t)(retry + 1);
  return std::max(ns, kMinNs);
}

static ncclResult_t IbCastRmaReserveWrs(struct ncclIbNetCommBase* base, struct ncclIbNetCommDevBase* devBase,
                                       int devIndex, struct ncclIbRequest* req, int nWrs) {
  int maxWrs = base->isSend ? NCCL_IB_RMA_MAX_SEND_WRS : NCCL_IB_RMA_MAX_FLUSH_WRS;
  if (nWrs < 0 || nWrs > maxWrs) return ncclInternalError;
  struct timespec start;
  const uint64_t kTimeoutNs = IbCastRmaCreditTimeoutNs();
  if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
    WARN("NET/IB-CAST/RMA: clock_gettime failed waiting for work-request credits");
    return ncclRemoteError;
  }
  while (!ncclRmaWrCreditsAvailable(base->rmaWrsOutstanding, nWrs, maxWrs)) {
    int nCompleted = 0;
    NCCLCHECK(IbCastRmaPollCq(base, devBase, devIndex, &nCompleted));
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
      WARN("NET/IB-CAST/RMA: clock_gettime failed waiting for work-request credits");
      return ncclRemoteError;
    }
    if ((TIMESPEC_TO_NSEC(&now) - TIMESPEC_TO_NSEC(&start)) >= kTimeoutNs) {
      WARN("NET/IB-CAST/RMA: timed out waiting for %d work-request credits (outstanding=%d max=%d)", nWrs,
           base->rmaWrsOutstanding, maxWrs);
      return ncclRemoteError;
    }
    if (nCompleted == 0) sched_yield();
  }
  base->rmaWrsOutstanding += nWrs;
  req->rmaNwrs = nWrs;
  return ncclSuccess;
}

static void IbCastRmaReleaseWrs(struct ncclIbRequest* req) {
  (void)IbCastRmaRefundWrs(req, req->rmaNwrs);
}

// wrap_ibv_post_send returns an error even after a prefix of the chain is
// accepted. Only the last WR is signaled, so NCCLCHECK-and-return would leak
// the request slot and leave unsignaled WRs without a CQE owner.
static ncclResult_t IbCastRmaPostWrs(struct ncclIbQp* qp, struct ncclIbRequest* req, struct ibv_send_wr* wr, int nWr,
                                    int* posted) {
  *posted = 0;
  if (nWr <= 0) return ncclSuccess;
  struct ibv_send_wr* bad_wr = nullptr;
  ncclResult_t ret = wrap_ibv_post_send(qp->qp, wr, &bad_wr);
  if (ret == ncclSuccess) {
    *posted = nWr;
    return ncclSuccess;
  }
  *posted = ncclRmaPostedWrCount(wr, nWr, bad_wr, offsetof(struct ibv_send_wr, next));
  if (*posted == 0) {
    IbCastRmaReleaseWrs(req);
    return ret;
  }
  (void)IbCastRmaRefundWrs(req, nWr - *posted);
  WARN("NET/IB-CAST/RMA: ibv_post_send failed after %d/%d WRs; leaving request for CQ drain", *posted, nWr);
  return ret;
}

// If nothing posted, free the slot and leave *request unset. If a prefix posted,
// keep the request so Test() can drain. A rejected signaled tail cannot produce
// a CQE; mark FAILED so Test() reports it instead of polling forever.
static ncclResult_t IbCastRmaCompletePostedRequest(struct ncclIbRequest* req, int devIndex, ncclResult_t postRet,
                                                  int posted, int nWr, void** request) {
  int keep = 0, failed = 0;
  ncclResult_t status = ncclRmaCompletePostedRequest(postRet, posted, nWr, &keep, &failed);
  if (!keep) {
    (void)IbCastFreeRequest(req);
    return status;
  }
  if (posted > 0) IbCastAddEvent(req, devIndex);
  *request = req;
  if (failed) req->type = NCCL_NET_IB_REQ_FAILED;
  return status;
}

static ncclResult_t IbCastRmaReservePostComplete(struct ncclIbNetCommBase* base, struct ncclIbNetCommDevBase* devBase,
                                                int devIndex, struct ncclIbQp* qp, struct ncclIbRequest* req,
                                                struct ibv_send_wr* wr, int nWr, void** request) {
  ncclResult_t ret = IbCastRmaReserveWrs(base, devBase, devIndex, req, nWr);
  if (ret != ncclSuccess) {
    (void)IbCastFreeRequest(req);
    return ret;
  }
  int posted = 0;
  ret = IbCastRmaPostWrs(qp, req, wr, nWr, &posted);
  return IbCastRmaCompletePostedRequest(req, devIndex, ret, posted, nWr, request);
}

static ncclResult_t rcclIbCastRmaIbProxyIPut(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                             uint64_t dstOff, void* dstMhandle, uint32_t rank, uint32_t optFlags,
                                             void** request) {
  (void)optFlags;
  struct IbCastRmaIbProxyCtx* ginProxyCtx = &((struct IbCastRmaIbProxyCtx*)ginCtx)[context];

  struct IbCastRmaProxySegMrHandle* srcMrHandle = (struct IbCastRmaProxySegMrHandle*)srcMhandle;
  struct IbCastRmaProxySegMrHandle* dstMrHandle = (struct IbCastRmaProxySegMrHandle*)dstMhandle;

  if (!IbCastRmaRangeOk(srcMrHandle, ginProxyCtx->rank, srcOff, size) ||
      !IbCastRmaRangeOk(dstMrHandle, rank, dstOff, size)) {
    WARN("NET/IB-CAST/RMA: iput out of range (srcOff=%lu dstOff=%lu size=%zu)", srcOff, dstOff, size);
    return ncclInvalidArgument;
  }

  struct ncclIbSendComm* comm;
  NCCLCHECK(IbCastRmaIbProxyGetSendComm(ginProxyCtx, rank, &comm));
  struct ncclIbQp* qp = &comm->base.qps[0];

  struct ibv_send_wr wr[NCCL_RMA_MAX_DATA_WRS];
  struct ibv_sge sge[NCCL_RMA_MAX_DATA_WRS];
  int nWr = 0;
  NCCLCHECK(IbCastRmaBuildSegmentedWrs(wr, sge, NCCL_RMA_MAX_DATA_WRS, &nWr, IBV_WR_RDMA_WRITE, &comm->base, qp,
                                       srcMrHandle, ginProxyCtx->rank, srcOff, dstMrHandle, rank, dstOff, size,
                                       /*flushSge=*/NULL));

  struct ncclIbRequest* req;
  NCCLCHECK(IbCastGetRequest(&comm->base, &req));
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  for (int i = 0; i < nWr; i++) {
    wr[i].wr_id = req - comm->base.reqs;
    wr[i].send_flags = ncclRmaWrIsSignaled(i, nWr) ? IBV_SEND_SIGNALED : 0;
  }
  return IbCastRmaReservePostComplete(&comm->base, &comm->devs[qp->devIndex].base, qp->devIndex, qp, req, &wr[0], nWr,
                                      request);
}

static ncclResult_t rcclIbCastRmaIbProxyIGet(void* ginCtx, int context, uint64_t remoteOffset, void* remoteMhandle,
                                             size_t size, uint64_t localOffset, void* localMhandle, uint32_t rank,
                                             uint32_t optFlags, void** request) {
  (void)optFlags;
  struct IbCastRmaIbProxyCtx* ginProxyCtx = &((struct IbCastRmaIbProxyCtx*)ginCtx)[context];

  struct IbCastRmaProxySegMrHandle* remoteMrHandle = (struct IbCastRmaProxySegMrHandle*)remoteMhandle;
  struct IbCastRmaProxySegMrHandle* localMrHandle = (struct IbCastRmaProxySegMrHandle*)localMhandle;

  if (!IbCastRmaRangeOk(remoteMrHandle, rank, remoteOffset, size) ||
      !IbCastRmaRangeOk(localMrHandle, ginProxyCtx->rank, localOffset, size)) {
    WARN("NET/IB-CAST/RMA: iget out of range (remoteOff=%lu localOff=%lu size=%zu)", remoteOffset, localOffset, size);
    return ncclInvalidArgument;
  }

  struct ncclIbSendComm* comm;
  NCCLCHECK(IbCastRmaIbProxyGetSendComm(ginProxyCtx, rank, &comm));
  struct ncclIbQp* qp = &comm->base.qps[0];

  struct ibv_send_wr wr[NCCL_RMA_MAX_DATA_WRS];
  struct ibv_sge sge[NCCL_RMA_MAX_DATA_WRS];
  int nWr = 0;
  NCCLCHECK(IbCastRmaBuildSegmentedWrs(wr, sge, NCCL_RMA_MAX_DATA_WRS, &nWr, IBV_WR_RDMA_READ, &comm->base, qp,
                                       localMrHandle, ginProxyCtx->rank, localOffset, remoteMrHandle, rank,
                                       remoteOffset, size, /*flushSge=*/NULL));

  struct ncclIbRequest* req;
  NCCLCHECK(IbCastGetRequest(&comm->base, &req));
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IGET;
  req->sock = &comm->base.sock;
  req->iget.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  for (int i = 0; i < nWr; i++) {
    wr[i].wr_id = req - comm->base.reqs;
    wr[i].send_flags = ncclRmaWrIsSignaled(i, nWr) ? IBV_SEND_SIGNALED : 0;
  }
  return IbCastRmaReservePostComplete(&comm->base, &comm->devs[qp->devIndex].base, qp->devIndex, qp, req, &wr[0], nWr,
                                      request);
}

static ncclResult_t rcclIbCastRmaIbProxyIPutSignal(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle,
                                                   size_t size, uint64_t dstOff, void* dstMhandle, uint32_t rank,
                                                   uint64_t signalOff, void* signalMhandle, uint64_t signalValue,
                                                   uint32_t signalOp, bool isStrongSignal, uint32_t optFlags,
                                                   void** request) {
  (void)isStrongSignal;
  (void)optFlags;
  if (signalOp != NCCL_NET_SIGNAL_OP_INC && signalOp != NCCL_NET_SIGNAL_OP_ADD) {
    WARN("IbCastRmaIbProxyIPutSignal: Unsupported signalOp %u", signalOp);
    return ncclInvalidArgument;
  }

  struct IbCastRmaIbProxyCtx* ginProxyCtx = &((struct IbCastRmaIbProxyCtx*)ginCtx)[context];

  struct IbCastRmaProxySegMrHandle* srcMrHandle = (struct IbCastRmaProxySegMrHandle*)srcMhandle;
  struct IbCastRmaProxySegMrHandle* dstMrHandle = (struct IbCastRmaProxySegMrHandle*)dstMhandle;
  struct IbCastRmaProxySegMrHandle* signalMrHandle = (struct IbCastRmaProxySegMrHandle*)signalMhandle;

  if ((size > 0 && (!srcMrHandle || !dstMrHandle || !IbCastRmaRangeOk(srcMrHandle, ginProxyCtx->rank, srcOff, size) ||
                    !IbCastRmaRangeOk(dstMrHandle, rank, dstOff, size))) ||
      !signalMrHandle || !IbCastRmaRangeOk(signalMrHandle, rank, signalOff, sizeof(uint64_t))) {
    WARN("NET/IB-CAST/RMA: iputSignal out of range (srcOff=%lu dstOff=%lu size=%zu signalOff=%lu)", srcOff, dstOff,
         size, signalOff);
    return ncclInvalidArgument;
  }
  int sig = IbCastRmaSegOf(signalMrHandle, rank, signalOff);
  if (!ncclRmaSignalOffsetValid(signalOff, IbCastRmaRankOff(signalMrHandle, rank)[sig + 1])) {
    WARN("NET/IB-CAST/RMA: iputSignal atomic must be 8-byte aligned and contained in one segment (signalOff=%lu)",
         signalOff);
    return ncclInvalidArgument;
  }

  struct ncclIbSendComm* comm;
  NCCLCHECK(IbCastRmaIbProxyGetSendComm(ginProxyCtx, rank, &comm));
  struct ncclIbQp* qp = &comm->base.qps[0];
  int devIndex = qp->devIndex;

  struct ibv_send_wr wr[NCCL_RMA_MAX_SIGNAL_WRS];
  struct ibv_sge sge[NCCL_RMA_MAX_SIGNAL_WRS];
  memset(&wr, 0, sizeof(wr));
  memset(&sge, 0, sizeof(sge));
  int nPut = 0;
  if (size > 0) {
    NCCLCHECK(IbCastRmaBuildSegmentedWrs(wr, sge, NCCL_RMA_MAX_DATA_WRS, &nPut, IBV_WR_RDMA_WRITE, &comm->base, qp,
                                         srcMrHandle, ginProxyCtx->rank, srcOff, dstMrHandle, rank, dstOff, size,
                                         /*flushSge=*/NULL));
    for (int i = 0; i < nPut; i++) wr[i].send_flags = 0;
  }

  int signalDevSlot = IbCastRmaDevSlot(signalMrHandle, rank, comm->base.remDevs[qp->remDevIdx].ibv_dev_index);
  if (signalDevSlot < 0) {
    WARN("NET/IB-CAST/RMA: no registration slot for remote IB device %d",
         comm->base.remDevs[qp->remDevIdx].ibv_dev_index);
    return ncclInternalError;
  }

  struct ncclIbRequest* req;
  NCCLCHECK(IbCastGetRequest(&comm->base, &req));
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }
  for (int i = 0; i < nPut; i++) wr[i].wr_id = req - comm->base.reqs;
  void* signalPtr = (void*)(signalMrHandle->base_vas[(size_t)rank * signalMrHandle->nSegments + sig] +
                            (signalOff - IbCastRmaRankOff(signalMrHandle, rank)[sig]));
  uint32_t signalRkey = IbCastRmaRemoteRkey(signalMrHandle, rank, sig, signalDevSlot);

  struct ibv_send_wr* sigWr = &wr[nPut];
  struct ibv_sge* sigSge = &sge[nPut];
  memset(sigWr, 0, sizeof(*sigWr));
  memset(sigSge, 0, sizeof(*sigSge));
  sigWr->opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  sigWr->send_flags = IBV_SEND_SIGNALED;
  sigWr->wr_id = req - comm->base.reqs;
  sigWr->next = NULL;
  sigWr->wr.atomic.remote_addr = (uint64_t)signalPtr;
  sigWr->wr.atomic.compare_add = signalOp == NCCL_NET_SIGNAL_OP_INC ? 1 : signalValue;
  sigWr->wr.atomic.rkey = signalRkey;
  sigWr->sg_list = sigSge;
  sigWr->num_sge = 1;

  sigSge->addr = (uintptr_t)&comm->putSignalScratchpad;
  sigSge->length = sizeof(comm->putSignalScratchpad);
  sigSge->lkey = comm->devs[devIndex].putSignalScratchpadMr->lkey;

  if (nPut > 0) wr[nPut - 1].next = sigWr;

  return IbCastRmaReservePostComplete(&comm->base, &comm->devs[devIndex].base, devIndex, qp, req,
                                      nPut > 0 ? &wr[0] : sigWr, nPut + 1, request);
}

static ncclResult_t rcclIbCastRmaIbProxyTest(void* collComm, void* request, int* done) {
  struct ncclIbRequest* req = (struct ncclIbRequest*)request;
  struct IbCastRmaIbProxyCtx* ginProxyCtx = (struct IbCastRmaIbProxyCtx*)req->ginProxyCtx;
  int rank = req->iput.rank;
  *done = 0;

  if (req->type == NCCL_NET_IB_REQ_FAILED) {
    *done = 1;
    IbCastRmaReleaseWrs(req);
    NCCLCHECK(IbCastFreeRequest(req));
    return ncclRemoteError;
  }

  ncclIbNetCommBase* commBase;
  ncclIbNetCommDevBase* devBase;
  int devIndex;
  if (req->type == NCCL_NET_IB_REQ_FLUSH) {
    struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)ginProxyCtx->fullRecvComm[rank];
    commBase = &comm->base;
    devIndex = comm->devs[0].gpuFlush.qp.devIndex;
    devBase = &comm->devs[devIndex].base;
  } else {
    struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
    commBase = &comm->base;
    devIndex = comm->base.qps[0].devIndex;
    devBase = &comm->devs[devIndex].base;
  }
  if (req->events[devIndex] != 0) {
    int nCompleted = 0;
    NCCLCHECK(IbCastRmaPollCq(commBase, devBase, devIndex, &nCompleted));
  }
  if (req->events[devIndex] == 0) {
    *done = 1;
    NCCLCHECK(IbCastFreeRequest(req));
  }
  return ncclSuccess;
}

static ncclResult_t rcclIbCastRmaIbProxyIFlush(void* ginCtx, int context, void* mhandle, uint32_t rank,
                                               void** request) {
  struct IbCastRmaIbProxyCtx* ginProxyCtx = &((struct IbCastRmaIbProxyCtx*)ginCtx)[context];
  struct IbCastRmaProxySegMrHandle* ginMrHandle = (struct IbCastRmaProxySegMrHandle*)mhandle;
  struct ncclIbRecvComm* comm;
  NCCLCHECK(IbCastRmaIbProxyGetRecvComm(ginProxyCtx, rank, &comm));
  struct ncclIbQp* qp = &comm->devs[0].gpuFlush.qp;

  struct ibv_send_wr wr[NCCL_RMA_MAX_FLUSH_WRS];
  struct ibv_sge sge[NCCL_RMA_MAX_FLUSH_WRS];
  int nWr = 0;
  NCCLCHECK(IbCastRmaBuildSegmentedWrs(wr, sge, NCCL_RMA_MAX_FLUSH_WRS, &nWr, IBV_WR_RDMA_READ, &comm->base, qp,
                                       /*localH=*/NULL, /*localRank=*/0, /*localOff=*/0,
                                       /*remoteH=*/ginMrHandle, /*remoteRank=*/ginProxyCtx->rank, /*remoteOff=*/0,
                                       /*size=*/IbCastRmaMrBytes(ginMrHandle, ginProxyCtx->rank),
                                       /*flushSge=*/&comm->devs[qp->devIndex].gpuFlush.sge));

  struct ncclIbRequest* req;
  NCCLCHECK(IbCastGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  req->ginProxyCtx = ginProxyCtx;
  for (int i = 0; i < nWr; i++) {
    wr[i].wr_id = req - comm->base.reqs;
    wr[i].send_flags = ncclRmaWrIsSignaled(i, nWr) ? IBV_SEND_SIGNALED : 0;
  }
  TRACE(NCCL_NET, "NET/IB: %s: Posting %d-segment flush request (req=%p, comm=%p)", __func__, nWr, req, req->base);
  TIME_START(4);
  ncclResult_t ret = IbCastRmaReservePostComplete(&comm->base, &comm->devs[qp->devIndex].base, qp->devIndex, qp, req,
                                                 &wr[0], nWr, request);
  TIME_STOP(4);
  return ret;
}

#endif
