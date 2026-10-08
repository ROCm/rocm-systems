/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// RCCL per-segment DMA-BUF MRs for the RMA proxy. Included once by gin.cc, after
// ncclRmaIbProxyGetRecvComm, so it can use that file's static helpers.

struct rcclRmaIbProxySegMrHandle {
  int nSegments;
  // segOff[0]==0, segOff[nSegments]==size; per-segment local MRs.
  // base_vas indexed [rank*nSegments + seg].
  // rkeys indexed (rank * nSegments + seg) * NCCL_IB_MAX_DEVS_PER_NIC + remDevIdx
  // rankSegOff: per-rank [nSegments+1] after allgather (asymmetric windows).
  size_t segOff[NCCL_RMA_MAX_SEGMENTS + 1];
  struct ncclIbMrHandle* mrHandle[NCCL_RMA_MAX_SEGMENTS];
  uintptr_t* base_vas;
  uint32_t* rkeys;
  size_t* rankSegOff;
};

struct ncclRmaIbProxyRegistration {
  ncclResult_t status;
  int nSegments;
  size_t segOff[NCCL_RMA_MAX_SEGMENTS + 1];
};

static inline size_t ncclRmaRkeyIndex(int nSegments, int rank, int seg, int remDevIdx) {
  return ((size_t)rank * nSegments + (size_t)seg) * NCCL_IB_MAX_DEVS_PER_NIC + (size_t)remDevIdx;
}

static inline uint32_t ncclRmaRemoteRkey(const struct rcclRmaIbProxySegMrHandle* h, int rank, int seg, int remDevIdx) {
  return h->rkeys[ncclRmaRkeyIndex(h->nSegments, rank, seg, remDevIdx)];
}

static inline const size_t* ncclRmaRankOff(const struct rcclRmaIbProxySegMrHandle* h, int rank) {
  return ncclRmaPeerSegOff(h->rankSegOff, h->segOff, rank);
}

static inline int ncclRmaSegOf(const struct rcclRmaIbProxySegMrHandle* h, int rank, uint64_t off) {
  return ncclRmaSegIndexOf(ncclRmaRankOff(h, rank), h->nSegments, off);
}

static inline uint64_t ncclRmaMrBytes(const struct rcclRmaIbProxySegMrHandle* h, int rank) {
  return ncclRmaRankOff(h, rank)[h->nSegments];
}

static inline bool ncclRmaRangeOk(const struct rcclRmaIbProxySegMrHandle* h, int rank, uint64_t off, size_t size) {
  const size_t* offs = ncclRmaRankOff(h, rank);
  uint64_t bytes = offs[h->nSegments];
  return off <= bytes && (uint64_t)size <= bytes - off;
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
// landing a single byte in the caller's flush scratch. Passing the whole
// window thus fences GPUDirect writes across every physical segment (the
// caller reads its own buffer via base_vas[self]/rkeys[self][qp->remDevIdx]
// on a loopback QP).
static ncclResult_t ncclRmaBuildSegmentedWrs(struct ibv_send_wr* wr, struct ibv_sge* sge, int maxWr, int* nWr,
                                             enum ibv_wr_opcode opcode, const struct ncclIbQp* qp,
                                             struct rcclRmaIbProxySegMrHandle* localH, int localRank, uint64_t localOff,
                                             struct rcclRmaIbProxySegMrHandle* remoteH, int remoteRank, uint64_t remoteOff,
                                             size_t size, const struct ibv_sge* flushSge) {
  int n = 0;
  uint64_t lOff = localOff, rOff = remoteOff;
  size_t rem = size;
  while (rem > 0) {
    if (ncclRmaDataWrBudgetFull(n, maxWr)) {
      WARN("NET/IB/RMA: transfer of %zu bytes spans more than %d segment slices", size, maxWr);
      return ncclInternalError;
    }
    const size_t* rOffs = ncclRmaRankOff(remoteH, remoteRank);
    int rs = ncclRmaSegOf(remoteH, remoteRank, rOff);
    size_t remoteRemaining = rOffs[rs + 1] - rOff;

    int ls = 0;
    size_t localRemaining = SIZE_MAX;
    const size_t* lOffs = NULL;
    if (flushSge == NULL) {
      lOffs = ncclRmaRankOff(localH, localRank);
      ls = ncclRmaSegOf(localH, localRank, lOff);
      localRemaining = lOffs[ls + 1] - lOff;
    }
    // Paired transfers are limited by ibv_sge::length. Flush WRs read one byte
    // but advance to the next physical segment in one step, regardless of the
    // segment's size.
    size_t chunk = flushSge == NULL ? ncclRmaSegmentSliceBytes(rem, localRemaining, remoteRemaining) :
                                      (rem < remoteRemaining ? rem : remoteRemaining);

    uintptr_t rAddr = remoteH->base_vas[(size_t)remoteRank * remoteH->nSegments + rs] + (rOff - rOffs[rs]);

    memset(&wr[n], 0, sizeof(wr[n]));
    memset(&sge[n], 0, sizeof(sge[n]));
    wr[n].opcode = opcode;
    wr[n].next = NULL;
    wr[n].wr.rdma.remote_addr = (uint64_t)rAddr;
    wr[n].wr.rdma.rkey = ncclRmaRemoteRkey(remoteH, remoteRank, rs, qp->remDevIdx);
    wr[n].sg_list = &sge[n];
    wr[n].num_sge = 1;
    if (flushSge != NULL) {
      // Touch one byte of this segment; the data lands in the fixed scratch.
      sge[n] = *flushSge;
      sge[n].length = 1;
    } else {
      uintptr_t lAddr = localH->base_vas[(size_t)localRank * localH->nSegments + ls] + (lOff - lOffs[ls]);
      struct ibv_mr* lmr = localH->mrHandle[ls]->mrs[qp->devIndex];
      if (lmr == NULL) {
        WARN("NET/IB/RMA: no local MR for segment %d device %d", ls, qp->devIndex);
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

// wrap_ibv_post_send returns ncclSystemError on any nonzero ibv_post_send, including
// after a prefix of the chain is accepted. Only the last WR is signaled, so
// NCCLCHECK-and-return would leak the request slot and leave unsignaled WRs
// without a CQE owner. Count how many WRs the HCA accepted.
static ncclResult_t ncclRmaPostWrs(struct ncclIbQp* qp, struct ibv_send_wr* wr, int nWr, int* posted) {
  *posted = 0;
  if (nWr <= 0) return ncclSuccess;
  struct ibv_send_wr* bad_wr = NULL;
  ncclResult_t ret = wrap_ibv_post_send(qp->qp, wr, &bad_wr);
  if (ret == ncclSuccess) {
    *posted = nWr;
    return ncclSuccess;
  }
  *posted = ncclRmaPostedWrCount(wr, nWr, bad_wr, offsetof(struct ibv_send_wr, next));
  if (*posted > 0) {
    WARN("NET/IB/RMA: ibv_post_send failed after %d/%d WRs; leaving request for CQ drain", *posted, nWr);
  }
  return ret;
}

static void ncclRmaStampSendWrs(struct ibv_send_wr* wr, int nWr, uint64_t wrId, int signalLast) {
  for (int i = 0; i < nWr; i++) {
    wr[i].wr_id = wrId;
    wr[i].send_flags = (signalLast && ncclRmaWrIsSignaled(i, nWr)) ? IBV_SEND_SIGNALED : 0;
  }
}

// Queue a segment-split chain behind earlier batched WRs. A chain longer than
// the batch posts directly after the batch so WRs reach the QP in request-id order.
static ncclResult_t ncclRmaIbProxyQueueWrs(struct ncclRmaIbProxyCtx* ctx, struct ncclIbQp* qp, struct ibv_send_wr* wr,
                                           struct ibv_sge* sge, int nWr, bool aggregate, int* posted) {
  *posted = 0;
  if (nWr == 0) return ncclSuccess;
  if (nWr <= ctx->wrBatchsize) {
    struct ibv_send_wr* slotWr;
    struct ibv_sge* slotSge;
    NCCLCHECK(ncclRmaIbProxyReserveWrs(ctx, qp->qp, nWr, &slotWr, &slotSge));
    for (int i = 0; i < nWr; i++) {
      slotWr[i] = wr[i];
      slotSge[i] = sge[i];
      slotWr[i].sg_list = &slotSge[i];
    }
    // Once queued, a failed doorbell may already have handed these WRs to the QP.
    *posted = nWr;
    return ncclRmaIbProxyCommitWrs(ctx, nWr, aggregate);
  }
  NCCLCHECK(ncclRmaIbProxyPostBatch(ctx));
  wr[nWr - 1].send_flags |= IBV_SEND_SIGNALED;
  wr[nWr - 1].next = NULL;
  return ncclRmaPostWrs(qp, wr, nWr, posted);
}

// If nothing posted, release the slot and its sequence id and leave *request unset.
// If a prefix posted, keep the request so Test() can drain. A rejected signaled
// tail cannot produce a CQE; mark FAILED so Test() reports it.
static ncclResult_t ncclRmaFinishPostedRequest(struct ncclIbRequest* req, uint64_t* postedSeq, ncclResult_t postRet,
                                               int posted, int nWr, void** request) {
  int keep = 0, failed = 0;
  ncclResult_t status = ncclRmaCompletePostedRequest(postRet, posted, nWr, &keep, &failed);
  if (!keep) {
    ncclRmaReleaseSeqId(req->id, postedSeq);
    (void)ncclIbFreeRequest(req);
    return status;
  }
  *request = req;
  if (failed) req->type = NCCL_NET_IB_REQ_FAILED;
  return status;
}

ncclResult_t rcclRmaIbProxyConnectFinish(void** collComm, int nranks) {
  if (!rcclRmaMultiSegEnabled()) return ncclSuccess;
  // Registration must not fail locally before its first collective, so the
  // receive buffer is allocated here. It is page-aligned because allGather
  // registers whole pages for remote write.
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)*collComm;
  ncclResult_t ret = ncclIbMalloc(&cComm->regConsensus, (size_t)nranks * sizeof(struct ncclRmaIbProxyRegistration));
  if (ret != ncclSuccess) {
    ncclGinIbCloseColl(*collComm);
    *collComm = NULL;
    return ret;
  }
  return ncclSuccess;
}

ncclResult_t rcclRmaIbProxyRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                          uint64_t mr_flags, void** mhandle) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;
  struct rcclRmaIbProxySegMrHandle* rmaMrHandle = NULL;
  struct ncclRmaIbProxyRegistration localRegistration = {};
  struct ncclRmaIbProxyRegistration* registrations = (struct ncclRmaIbProxyRegistration*)cComm->regConsensus;
  uintptr_t localVas[NCCL_RMA_MAX_SEGMENTS] = {};
  uint32_t localRkeys[NCCL_RMA_MAX_SEGMENTS * NCCL_IB_MAX_DEVS_PER_NIC] = {};
  ncclResult_t ret = ncclSuccess;
  int nSeg = 1;
  int registered = 0;

  *mhandle = NULL;
  ret = ncclCalloc(&rmaMrHandle, 1);
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
    WARN("NET/IB/RMA: buffer %p (size %zu) spans %d segments, exceeds NCCL_RMA_MAX_SEGMENTS=%d", data, size, nSeg,
         NCCL_RMA_MAX_SEGMENTS);
    ret = ncclInvalidUsage;
    goto reconcile;
  }
  rmaMrHandle->nSegments = nSeg;
  rmaMrHandle->segOff[0] = 0;

  if (nSeg == 1) {
    // Single segment: reuse the caller's fd (fd==-1 falls back to ibv_reg_mr).
    NCCLCHECKGOTO(ncclIbRegMrDmaBufInternal(cComm->recvComm, data, size, type, offset, fd, mr_flags,
                                            (void**)&rmaMrHandle->mrHandle[0]),
                  ret, reconcile);
    registered = 1;
    rmaMrHandle->segOff[1] = size;
    localVas[0] = (uintptr_t)data;
  } else {
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
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
      // Export this segment alone: one physical allocation, so its fd is complete.
      // The MR spans whole pages from segPtr's page at dmabuf offset 0 (reg.cc),
      // so the export must cover that same page range.
      static size_t hostPageSize = ncclOsGetPageSize();
      uintptr_t exportBase = segPtr & ~(uintptr_t)(hostPageSize - 1);
      size_t exportLen = segPtr + thisLen - exportBase;
      ALIGN_SIZE(exportLen, hostPageSize);
      CUCHECKGOTO(cuMemGetHandleForAddressRange((void*)&segFd, (CUdeviceptr)exportBase, exportLen,
                                                CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0),
                  ret, reconcile);
      ret = ncclIbRegMrDmaBufInternal(cComm->recvComm, (void*)segPtr, thisLen, type, 0ULL, segFd, mr_flags,
                                      (void**)&rmaMrHandle->mrHandle[s]);
      (void)close(segFd);
      if (ret != ncclSuccess) goto reconcile;
      registered = s + 1;
      localVas[s] = segPtr;
      cum += thisLen;
      rmaMrHandle->segOff[s + 1] = cum;
      segPtr += thisLen;
      remaining -= thisLen;
    }
    if (remaining != 0) {
      WARN("NET/IB/RMA: enumerated %d/%d segments for buffer %p but %zu bytes remain unregistered", registered, nSeg,
           data, remaining);
      ret = ncclInvalidUsage;
      goto reconcile;
    }
    INFO(NCCL_NET | NCCL_REG, "NET/IB/RMA: registered multi-segment buffer %p size %zu as %d DMA-BUF MRs", data, size,
         nSeg);
#else
    WARN("NET/IB/RMA: multi-segment (%d) registration requires HIP >= 7.12.60540", nSeg);
    ret = ncclInvalidUsage;
    goto reconcile;
#endif
  }

  {
    struct ncclIbNetCommBase* recvBase = (struct ncclIbNetCommBase*)cComm->recvComm;
    int ndevs = recvBase->vProps.ndevs;
    memset(localRkeys, 0, sizeof(localRkeys));
    if (ndevs < 1 || ndevs > NCCL_IB_MAX_DEVS_PER_NIC) {
      WARN("NET/IB/RMA: invalid ndevs %d for buffer %p", ndevs, data);
      ret = ncclInternalError;
      goto reconcile;
    }
    for (int s = 0; s < nSeg; s++) {
      for (int d = 0; d < ndevs; d++) {
        struct ibv_mr* mr = rmaMrHandle->mrHandle[s]->mrs[d];
        if (mr == NULL) {
          WARN("NET/IB/RMA: missing MR for segment %d device %d", s, d);
          ret = ncclInternalError;
          goto reconcile;
        }
        localRkeys[(size_t)s * NCCL_IB_MAX_DEVS_PER_NIC + d] = mr->rkey;
      }
    }
  }

reconcile:
  // No rank-local enumeration/export/registration/device-validation failure may
  // return before peers reach this first registration collective. Exchange the
  // complete status and layout in one fixed-size record.
  localRegistration.status = ret;
  localRegistration.nSegments = nSeg;
  if (ncclRmaRegistrationHandleReady(rmaMrHandle, nSeg)) {
    memcpy(localRegistration.segOff, rmaMrHandle->segOff, sizeof(size_t) * (nSeg + 1));
  }
  NCCLCHECKGOTO(cComm->allGather(cComm, &localRegistration, registrations, sizeof(struct ncclRmaIbProxyRegistration)),
                ret, fail);

  for (int r = 0; r < cComm->nranks; r++) {
    if (registrations[r].status != ncclSuccess) {
      WARN("NET/IB/RMA: symmetric registration rejected because rank %d failed with error %d", r,
           registrations[r].status);
      ret = registrations[r].status;
      goto fail;
    }
  }

  // VA/rkey allgathers use one nSeg. Terminal sizes may differ (asymmetric windows).
  for (int r = 0; r < cComm->nranks; r++) {
    if (!ncclRmaSegmentCountsMatch(nSeg, registrations[r].nSegments)) {
      WARN("NET/IB/RMA: buffer %p segment count differs on rank %d (%d vs %d)", data, r, registrations[r].nSegments,
           nSeg);
      ret = ncclInvalidUsage;
      goto fail;
    }
    if (!ncclRmaSegOffTableValid(registrations[r].segOff, nSeg)) {
      WARN("NET/IB/RMA: buffer %p segment offsets from rank %d are not ordered from 0", data, r);
      ret = ncclInvalidUsage;
      goto fail;
    }
  }

  // Reconcile allocation failures too, before entering the VA/rkey gathers.
  ret = ncclCalloc(&rmaMrHandle->rankSegOff, (size_t)cComm->nranks * (NCCL_RMA_MAX_SEGMENTS + 1));
  if (ret == ncclSuccess) {
    for (int r = 0; r < cComm->nranks; r++) {
      memcpy(rmaMrHandle->rankSegOff + (size_t)r * (NCCL_RMA_MAX_SEGMENTS + 1), registrations[r].segOff,
             sizeof(size_t) * (nSeg + 1));
    }
  }
  if (ret == ncclSuccess) ret = ncclCalloc(&rmaMrHandle->base_vas, (size_t)cComm->nranks * nSeg);
  if (ret == ncclSuccess)
    ret = ncclCalloc(&rmaMrHandle->rkeys, (size_t)cComm->nranks * nSeg * NCCL_IB_MAX_DEVS_PER_NIC);
  localRegistration.status = ret;
  NCCLCHECKGOTO(cComm->allGather(cComm, &localRegistration, registrations, sizeof(struct ncclRmaIbProxyRegistration)),
                ret, fail);
  for (int r = 0; r < cComm->nranks; r++) {
    if (registrations[r].status != ncclSuccess) {
      ret = registrations[r].status;
      goto fail;
    }
  }

  // Gather per-segment VAs and per-device rkeys. Equal nSeg keeps the VA/rkey
  // allgather size; rankSegOff supplies each peer's registration-relative map.
  NCCLCHECKGOTO(cComm->allGather(cComm, localVas, rmaMrHandle->base_vas, sizeof(uintptr_t) * nSeg), ret, fail);
  NCCLCHECKGOTO(cComm->allGather(cComm, localRkeys, rmaMrHandle->rkeys,
                                 sizeof(uint32_t) * nSeg * NCCL_IB_MAX_DEVS_PER_NIC),
                ret, fail);

  *mhandle = rmaMrHandle;
  return ncclSuccess;

fail:
  // `registered` counts complete ncclIbMrHandle objects. ncclIbRegMrDmaBufInternal
  // does not write *mhandle on failure, so a half-built wrapper is never dereg'd.
  if (rmaMrHandle) {
    for (int s = 0; s < registered; s++) {
      if (rmaMrHandle->mrHandle[s]) (void)ncclNetIb.deregMr(cComm->recvComm, rmaMrHandle->mrHandle[s]);
    }
    free(rmaMrHandle->base_vas);
    free(rmaMrHandle->rkeys);
    free(rmaMrHandle->rankSegOff);
    free(rmaMrHandle);
  }
  return ret;
}

ncclResult_t rcclRmaIbProxyDeregMrSym(void* collComm, void* mhandle) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;
  struct rcclRmaIbProxySegMrHandle* rmaMrHandle = (struct rcclRmaIbProxySegMrHandle*)mhandle;
  ncclResult_t ret = ncclSuccess;

  for (int s = 0; s < rmaMrHandle->nSegments; s++) {
    if (rmaMrHandle->mrHandle[s]) {
      ncclResult_t r = ncclNetIb.deregMr(cComm->recvComm, rmaMrHandle->mrHandle[s]);
      if (ret == ncclSuccess && r != ncclSuccess) ret = r;
    }
  }
  free(rmaMrHandle->base_vas);
  free(rmaMrHandle->rkeys);
  free(rmaMrHandle->rankSegOff);
  free(rmaMrHandle);
  return ret;
}

ncclResult_t rcclRmaIbProxyIPut(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                uint64_t dstOff, void* dstMhandle, uint32_t rank, uint32_t optFlags, void** request) {
  struct ncclRmaIbProxyCtx* rmaProxyCtx = &((struct ncclRmaIbProxyCtx*)rmaCtx)[context];

  struct rcclRmaIbProxySegMrHandle* srcMrHandle = (struct rcclRmaIbProxySegMrHandle*)srcMhandle;
  struct rcclRmaIbProxySegMrHandle* dstMrHandle = (struct rcclRmaIbProxySegMrHandle*)dstMhandle;

  // Reject out-of-range transfers before any WR is posted.
  if (!ncclRmaRangeOk(srcMrHandle, rmaProxyCtx->rank, srcOff, size) ||
      !ncclRmaRangeOk(dstMrHandle, rank, dstOff, size)) {
    WARN("NET/IB/RMA: iput out of range (srcOff=%lu dstOff=%lu size=%zu)", srcOff, dstOff, size);
    return ncclInvalidArgument;
  }

  struct ncclIbSendComm* comm;
  NCCLCHECK(ncclRmaIbProxyGetSendComm(rmaProxyCtx, rank, &comm));
  struct ncclIbQp* qp = &comm->base.qps[0];

  // Split the transfer at segment boundaries on both the local (src) and remote
  // (dst) buffers; each slice maps to one local lkey and one remote rkey.
  struct ibv_send_wr wr[NCCL_RMA_MAX_DATA_WRS];
  struct ibv_sge sge[NCCL_RMA_MAX_DATA_WRS];
  int nWr = 0;
  NCCLCHECK(ncclRmaBuildSegmentedWrs(wr, sge, NCCL_RMA_MAX_DATA_WRS, &nWr, IBV_WR_RDMA_WRITE, qp, srcMrHandle,
                                     rmaProxyCtx->rank, srcOff, dstMrHandle, rank, dstOff, size, /*flushSge=*/NULL));

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->rmaProxyCtx = rmaProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  req->id = ++comm->ginSeq.posted;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }
  ncclRmaStampSendWrs(wr, nWr, req - comm->base.reqs, /*signalLast=*/0);
  // No WR for size==0: complete with the previous request on this comm.
  if (nWr == 0) req->id = --comm->ginSeq.posted;

  int posted = 0;
  ncclResult_t postRet =
    ncclRmaIbProxyQueueWrs(rmaProxyCtx, qp, wr, sge, nWr, optFlags & ncclRmaOptFlagsAggregateRequests, &posted);
  ncclIbAddEvent(req, qp->devIndex);
  return ncclRmaFinishPostedRequest(req, &comm->ginSeq.posted, postRet, posted, nWr, request);
}

ncclResult_t rcclRmaIbProxyIGet(void* rmaCtx, int context, uint64_t remoteOffset, void* remoteMhandle, size_t size,
                                uint64_t localOffset, void* localMhandle, uint32_t rank, uint32_t optFlags,
                                void** request) {
  struct ncclRmaIbProxyCtx* rmaProxyCtx = &((struct ncclRmaIbProxyCtx*)rmaCtx)[context];

  struct rcclRmaIbProxySegMrHandle* remoteMrHandle = (struct rcclRmaIbProxySegMrHandle*)remoteMhandle;
  struct rcclRmaIbProxySegMrHandle* localMrHandle = (struct rcclRmaIbProxySegMrHandle*)localMhandle;

  // Reject out-of-range transfers before any WR is posted.
  if (!ncclRmaRangeOk(remoteMrHandle, rank, remoteOffset, size) ||
      !ncclRmaRangeOk(localMrHandle, rmaProxyCtx->rank, localOffset, size)) {
    WARN("NET/IB/RMA: iget out of range (remoteOff=%lu localOff=%lu size=%zu)", remoteOffset, localOffset, size);
    return ncclInvalidArgument;
  }

  struct ncclIbSendComm* comm;
  NCCLCHECK(ncclRmaIbProxyGetSendComm(rmaProxyCtx, rank, &comm));
  struct ncclIbQp* qp = &comm->base.qps[0];

  // RDMA READ: local buffer is the destination (lkey), remote buffer the source
  // (rkey). Split at segment boundaries on both sides.
  struct ibv_send_wr wr[NCCL_RMA_MAX_DATA_WRS];
  struct ibv_sge sge[NCCL_RMA_MAX_DATA_WRS];
  int nWr = 0;
  NCCLCHECK(ncclRmaBuildSegmentedWrs(wr, sge, NCCL_RMA_MAX_DATA_WRS, &nWr, IBV_WR_RDMA_READ, qp, localMrHandle,
                                     rmaProxyCtx->rank, localOffset, remoteMrHandle, rank, remoteOffset, size,
                                     /*flushSge=*/NULL));

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->rmaProxyCtx = rmaProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IGET;
  req->sock = &comm->base.sock;
  req->iget.rank = rank;
  req->id = ++comm->ginSeq.posted;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }
  ncclRmaStampSendWrs(wr, nWr, req - comm->base.reqs, /*signalLast=*/0);
  // No WR for size==0: complete with the previous request on this comm.
  if (nWr == 0) req->id = --comm->ginSeq.posted;

  int posted = 0;
  ncclResult_t postRet =
    ncclRmaIbProxyQueueWrs(rmaProxyCtx, qp, wr, sge, nWr, optFlags & ncclRmaOptFlagsAggregateRequests, &posted);
  return ncclRmaFinishPostedRequest(req, &comm->ginSeq.posted, postRet, posted, nWr, request);
}

ncclResult_t rcclRmaIbProxyIPutSignal(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                      uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff,
                                      void* signalMhandle, uint64_t signalValue, uint32_t signalOp, bool isStrongSignal,
                                      uint32_t optFlags, void** request) {
  (void)isStrongSignal;
  if (signalOp != NCCL_NET_SIGNAL_OP_INC && signalOp != NCCL_NET_SIGNAL_OP_ADD) {
    WARN("ncclRmaIbProxyIPutSignal: Unsupported signalOp %u", signalOp);
    return ncclInvalidArgument;
  }

  struct ncclRmaIbProxyCtx* rmaProxyCtx = &((struct ncclRmaIbProxyCtx*)rmaCtx)[context];

  struct rcclRmaIbProxySegMrHandle* srcMrHandle = (struct rcclRmaIbProxySegMrHandle*)srcMhandle;
  struct rcclRmaIbProxySegMrHandle* dstMrHandle = (struct rcclRmaIbProxySegMrHandle*)dstMhandle;
  struct rcclRmaIbProxySegMrHandle* signalMrHandle = (struct rcclRmaIbProxySegMrHandle*)signalMhandle;

  // Reject out-of-range payload/signal before any WR is posted (signal is an 8-byte atomic).
  if ((size > 0 && (!srcMrHandle || !dstMrHandle || !ncclRmaRangeOk(srcMrHandle, rmaProxyCtx->rank, srcOff, size) ||
                    !ncclRmaRangeOk(dstMrHandle, rank, dstOff, size))) ||
      !signalMrHandle || !ncclRmaRangeOk(signalMrHandle, rank, signalOff, sizeof(uint64_t))) {
    WARN("NET/IB/RMA: iputSignal out of range (srcOff=%lu dstOff=%lu size=%zu signalOff=%lu)", srcOff, dstOff, size,
         signalOff);
    return ncclInvalidArgument;
  }
  int sig = ncclRmaSegOf(signalMrHandle, rank, signalOff);
  if (!ncclRmaSignalOffsetValid(signalOff, ncclRmaRankOff(signalMrHandle, rank)[sig + 1])) {
    WARN("NET/IB/RMA: iputSignal atomic must be 8-byte aligned and contained in one segment (signalOff=%lu)",
         signalOff);
    return ncclInvalidArgument;
  }

  struct ncclIbSendComm* comm;
  NCCLCHECK(ncclRmaIbProxyGetSendComm(rmaProxyCtx, rank, &comm));
  struct ncclIbQp* qp = &comm->base.qps[0];
  int devIndex = qp->devIndex;

  // Up to NCCL_RMA_MAX_DATA_WRS slices for the segmented PUT plus one signal WR.
  struct ibv_send_wr wr[NCCL_RMA_MAX_SIGNAL_WRS];
  struct ibv_sge sge[NCCL_RMA_MAX_SIGNAL_WRS];
  int nPut = 0;

  // If size is 0, we only need to send the signal. srcMrHandle must be non-NULL
  if (size > 0 && dstMrHandle) {
    // PUT slices carry no CQE; only the trailing signal is signaled. Same-QP RC
    // ordering guarantees all writes land before the signal.
    NCCLCHECK(ncclRmaBuildSegmentedWrs(wr, sge, NCCL_RMA_MAX_DATA_WRS, &nPut, IBV_WR_RDMA_WRITE, qp, srcMrHandle,
                                       rmaProxyCtx->rank, srcOff, dstMrHandle, rank, dstOff, size, /*flushSge=*/NULL));
  }

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->rmaProxyCtx = rmaProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  req->id = ++comm->ginSeq.posted;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }
  ncclRmaStampSendWrs(wr, nPut, req - comm->base.reqs, /*signalLast=*/0);

  // SIGNAL (route to the segment that contains signalOff)
  void* signalPtr = (void*)(signalMrHandle->base_vas[(size_t)rank * signalMrHandle->nSegments + sig] +
                            (signalOff - ncclRmaRankOff(signalMrHandle, rank)[sig]));
  uint32_t signalRkey = ncclRmaRemoteRkey(signalMrHandle, rank, sig, qp->remDevIdx);

  struct ibv_send_wr* signalWr = &wr[nPut];
  struct ibv_sge* signalSge = &sge[nPut];
  memset(signalWr, 0, sizeof(*signalWr));
  memset(signalSge, 0, sizeof(*signalSge));
  signalWr->opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  signalWr->wr_id = req - comm->base.reqs;  // used for matching completions with request
  signalWr->next = NULL;
  signalWr->wr.atomic.remote_addr = (uint64_t)signalPtr;
  signalWr->wr.atomic.compare_add = signalOp == NCCL_NET_SIGNAL_OP_INC ? 1 : signalValue;
  signalWr->wr.atomic.rkey = signalRkey;
  signalWr->sg_list = signalSge;
  signalWr->num_sge = 1;

  signalSge->addr = (uintptr_t)&comm->putSignalScratchpad;
  signalSge->length = sizeof(comm->putSignalScratchpad);
  signalSge->lkey = comm->devs[devIndex].putSignalScratchpadMr->lkey;

  if (nPut > 0) wr[nPut - 1].next = signalWr;

  // Send the put and the signal in one go
  int posted = 0;
  ncclResult_t postRet =
    ncclRmaIbProxyQueueWrs(rmaProxyCtx, qp, wr, sge, nPut + 1, optFlags & ncclRmaOptFlagsAggregateRequests, &posted);
  return ncclRmaFinishPostedRequest(req, &comm->ginSeq.posted, postRet, posted, nPut + 1, request);
}

ncclResult_t rcclRmaIbProxyIFlush(void* rmaCtx, int context, void* mhandle, uint32_t rank, void** request) {
  struct ncclRmaIbProxyCtx* rmaProxyCtx = &((struct ncclRmaIbProxyCtx*)rmaCtx)[context];
  struct rcclRmaIbProxySegMrHandle* rmaMrHandle = (struct rcclRmaIbProxySegMrHandle*)mhandle;
  struct ncclIbRecvComm* comm;
  NCCLCHECK(ncclRmaIbProxyGetRecvComm(rmaProxyCtx, rank, &comm));
  if (comm->flushEnabled == 0) {
    *request = NULL;
    return ncclSuccess;
  }

  // drain batch before posting the flush
  NCCLCHECK(ncclRmaIbProxyPostBatch(rmaProxyCtx));
  NCCLCHECK(ncclIbCreateFlushQp(comm));
  struct ncclIbQp* qp = &comm->devs[0].gpuFlush.qp;

  // Fence GPUDirect writes across EVERY physical segment, not just segment 0.
  // The builder (flush mode) emits one loopback RDMA_READ per segment of the
  // local recv buffer -- read via base_vas[self] and rkeys[self][qp->remDevIdx]
  // -- each landing one byte in the flush scratch.
  struct ibv_send_wr wr[NCCL_RMA_MAX_FLUSH_WRS];
  struct ibv_sge sge[NCCL_RMA_MAX_FLUSH_WRS];
  int nWr = 0;
  NCCLCHECK(ncclRmaBuildSegmentedWrs(wr, sge, NCCL_RMA_MAX_FLUSH_WRS, &nWr, IBV_WR_RDMA_READ, qp,
                                     /*localH=*/NULL, /*localRank=*/0, /*localOff=*/0,
                                     /*remoteH=*/rmaMrHandle, /*remoteRank=*/rmaProxyCtx->rank, /*remoteOff=*/0,
                                     /*size=*/ncclRmaMrBytes(rmaMrHandle, rmaProxyCtx->rank),
                                     /*flushSge=*/&comm->devs[qp->devIndex].gpuFlush.sge));

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  req->rmaProxyCtx = rmaProxyCtx;
  req->id = ++comm->ginSeq.posted;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }
  ncclRmaStampSendWrs(wr, nWr, req - comm->base.reqs, /*signalLast=*/1);
  // No WR to post: complete with the previous request on this comm.
  if (nWr == 0) req->id = --comm->ginSeq.posted;

  TRACE(NCCL_NET, "NET/IB: %s: Posting %d-segment flush request (req=%p, comm=%p)", __func__, nWr, req, req->base);
  TIME_START(4);
  int posted = 0;
  ncclResult_t postRet = ncclRmaPostWrs(qp, &wr[0], nWr, &posted);
  TIME_STOP(4);
  return ncclRmaFinishPostedRequest(req, &comm->ginSeq.posted, postRet, posted, nWr, request);
}

// Called before the sequence check: later completions advance it past a lost tail.
ncclResult_t rcclRmaIbProxyTestFailed(struct ncclIbRequest* req, int* done) {
  *done = 1;
  NCCLCHECK(ncclIbFreeRequest(req));
  return ncclRemoteError;
}
