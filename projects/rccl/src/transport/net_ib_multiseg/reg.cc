/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common_mseg.h"
#include "rocmwrap.h" // RCCL: cuMem address ranges for one MR per segment.
#include "seg_reg_mseg.h" // RCCL: shared register helper and the per-segment entry.

ncclResult_t IbMsegRegMrDmaBufInternal2(IbMsegNetCommDevBase* base, void* data, size_t size, int type, uint64_t offset,
                                        int fd, uint64_t mrFlags, ibv_mr** mhandle) {
  static thread_local uintptr_t pageSize = 0;
  if (pageSize == 0) pageSize = sysconf(_SC_PAGESIZE);
  struct IbMsegMrCache* cache = &IbMsegDevs[base->ibDevN].mrCache;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize - 1) / pageSize;
  std::lock_guard<std::mutex> lock(IbMsegDevs[base->ibDevN].mutex);
  for (int slot = 0; /*true*/; slot++) {
    if (slot == cache->population || addr < cache->slots[slot].addr) { // didn't find in cache
      if (cache->population == cache->capacity) { // must grow cache
        cache->capacity = cache->capacity < 32 ? 32 : 2 * cache->capacity;
        NCCLCHECK(ncclRealloc(&cache->slots, cache->population, cache->capacity));
      }
      // Deregister / register
      struct ibv_mr* mr;
      unsigned int flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
      bool relaxedOrdering = IbMsegRelaxedOrderingEnabled && (mrFlags & NCCL_NET_MR_FLAG_FORCE_SO) == 0;
      if (relaxedOrdering) flags |= IBV_ACCESS_RELAXED_ORDERING;
      if (fd != -1) {
        /* DMA-BUF fd. dataDirect tries mlx5 first and falls back to ibv_reg_dmabuf_mr. */
        NCCLCHECK(wrap_reg_dmabuf_mr(&mr, base->pd, offset, pages * pageSize, addr, fd, flags,
                                     IbMsegDevs[base->ibDevN].capsProvider.mlx5.dataDirect));
      } else {
        if (relaxedOrdering) {
          // Use IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
          NCCLCHECK(wrap_ibv_reg_mr_iova2(&mr, base->pd, (void*)addr, pages * pageSize, addr, flags));
        } else {
          NCCLCHECK(wrap_ibv_reg_mr(&mr, base->pd, (void*)addr, pages * pageSize, flags));
        }
      }
      TRACE(NCCL_INIT | NCCL_NET, "regAddr=0x%lx size=%lld rkey=0x%x lkey=0x%x fd=%d", (unsigned long)addr,
            (long long)pages * pageSize, mr->rkey, mr->lkey, fd);
      if (slot != cache->population)
        memmove(cache->slots + slot + 1, cache->slots + slot, (cache->population - slot) * sizeof(struct IbMsegMr));
      cache->slots[slot].addr = addr;
      cache->slots[slot].pages = pages;
      cache->slots[slot].refs = 1;
      cache->slots[slot].mr = mr;
      cache->population += 1;
      *mhandle = mr;
      return ncclSuccess;
    } else if ((addr >= cache->slots[slot].addr) &&
               ((addr - cache->slots[slot].addr) / pageSize + pages) <= cache->slots[slot].pages) {
      cache->slots[slot].refs += 1;
      *mhandle = cache->slots[slot].mr;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

bool rcclIbMrAdmit(IbMsegNetCommBase* base, int nMr) {
  for (int i = 0; i < base->vProps.ndevs; i++) {
    struct IbMsegDev* dev = &IbMsegDevs[IbMsegGetNetCommDevBase(base, i)->ibDevN];
    std::lock_guard<std::mutex> lock(dev->mutex);
    if (dev->maxMr > 0 && (int64_t)dev->mrCache.population + nMr > dev->maxMr) return false;
  }
  return true;
}

// RCCL: comm-level declines for an nSeg-segment registration; the range walk runs them before exporting fds.
static ncclResult_t rcclIbMultiSegAdmit(struct IbMsegNetCommBase* base, int nSeg) {
  if (nSeg <= 1) return ncclSuccess;
  if (!rcclIbMrAdmit(base, nSeg)) {
    INFO(NCCL_NET | NCCL_REG, "NET/IB: %d more MRs would pass the device max_mr; declining", nSeg);
    return ncclInvalidUsage;
  }
  // RCCL: a resilient comm replays whole slots, so it stages multi-segment buffers on either side.
  if (base->resiliency != NULL) {
    INFO(NCCL_NET | NCCL_REG, "NET/IB: resiliency is on; declining %d-segment registration", nSeg);
    return ncclInvalidUsage;
  }
  // RCCL: the layout describes the receiver, so a recv comm on the legacy CTS (GIN) cannot take one.
  if (!base->isSend && !((struct IbMsegRecvComm*)base)->useMultiSeg) {
    INFO(NCCL_NET | NCCL_REG, "NET/IB: comm has no multi-segment CTS layout; declining %d-segment registration",
         nSeg);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

/* DMA-BUF support */
ncclResult_t IbMsegRegMrDmaBufInternal(void* comm, void* data, size_t size, int type, uint64_t offset, int fd,
                                       uint64_t mrFlags, void** mhandle) {
  ncclResult_t ret = ncclSuccess;
  if (size == 0) {
    WARN("NET/IB: Invalid memory registration size %zu, expected > 0", size);
    return ncclInternalError;
  }
  struct IbMsegNetCommBase* base = (struct IbMsegNetCommBase*)comm;
  struct IbMsegMrHandle* mhandleWrapper = NULL;
  NCCLCHECK(ncclCalloc(&mhandleWrapper, 1));
  mhandleWrapper->seg.nSegments = 1; // RCCL: a real fd stays one MR.
  for (int i = 0; i < base->vProps.ndevs; i++) {
    // Each IbMsegNetCommDevBase is at different offset in send and recv netComms
    struct IbMsegNetCommDevBase* devComm = IbMsegGetNetCommDevBase(base, i);
    NCCLCHECKGOTO(IbMsegRegMrDmaBufInternal2(devComm, data, size, type, offset, fd, mrFlags, mhandleWrapper->mrs + i),
                  ret, fail);
  }
  *mhandle = (void*)mhandleWrapper;
exit:
  return ret;
fail:
  free(mhandleWrapper);
  goto exit;
}

#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
// RCCL: fd < 0 walks one MR per segment. offset is the export flags. The count comes from the walk and stops one
// past the cap, which the caller declines.
static ncclResult_t rcclIbCountRangeSegments(void* data, size_t size, int* nSeg) {
  uintptr_t segPtr = (uintptr_t)data;
  size_t remaining = size;
  int n = 0;
  while (remaining > 0 && n <= RCCL_IB_REG_SEG_DEFAULT) {
    CUdeviceptr segBase = 0;
    size_t segSize = 0;
    CUCHECK(cuMemGetAddressRange(&segBase, &segSize, (CUdeviceptr)segPtr));
    size_t inSeg = IbMsegBytesRemainingInSegment(segPtr, (uintptr_t)segBase, segSize);
    if (inSeg == 0) return ncclInvalidUsage;
    size_t thisLen = remaining < inSeg ? remaining : inSeg;
    segPtr += thisLen;
    remaining -= thisLen;
    n++;
  }
  if (n < 1) return ncclInvalidUsage;
  *nSeg = n;
  return ncclSuccess;
}

static ncclResult_t rcclIbRegMrDmaBufFromRange(void* comm, void* data, size_t size, int type, uint64_t offset,
                                               void** mhandle) {
  const int handleFlags = (int)offset;
  ncclResult_t ret = ncclSuccess;
  char* scratch = NULL;
  void** segAddrs = NULL;
  size_t* segLens = NULL;
  uint64_t* segOffsets = NULL;
  int* segFds = NULL;
  int nSeg = 0;

  NCCLCHECKGOTO(rcclIbCountRangeSegments(data, size, &nSeg), ret, fail);
  // RCCL: decline before exporting one fd per segment.
  if (!rcclIbSegCountSupported(nSeg)) {
    INFO(NCCL_NET | NCCL_REG, "NET/IB: buffer %p has more than the supported %d segments; declining", data,
         RCCL_IB_REG_SEG_DEFAULT);
    ret = ncclInvalidUsage;
    goto fail;
  }
  ret = rcclIbMultiSegAdmit((struct IbMsegNetCommBase*)comm, nSeg);
  if (ret != ncclSuccess) goto fail;
  NCCLCHECKGOTO(ncclCalloc(&scratch, rcclIbSegWalkBytes(nSeg)), ret, fail);
  {
    struct rcclIbSegWalk walk;
    rcclIbSegWalkBind(scratch, nSeg, &walk);
    segAddrs = walk.addrs;
    segLens = walk.lens;
    segOffsets = walk.offs;
    segFds = walk.fds;
  }
  for (int s = 0; s < nSeg; s++) segFds[s] = -1;

  {
    uintptr_t segPtr = (uintptr_t)data;
    size_t remaining = size;
    for (int s = 0; s < nSeg; s++) {
      CUdeviceptr segBase = 0;
      size_t segSize = 0;
      CUCHECKGOTO(cuMemGetAddressRange(&segBase, &segSize, (CUdeviceptr)segPtr), ret, fail);
      size_t inSeg = IbMsegBytesRemainingInSegment(segPtr, (uintptr_t)segBase, segSize);
      size_t thisLen = remaining < inSeg ? remaining : inSeg;
      int fd = -1;
      CUCHECKGOTO(cuMemGetHandleForAddressRange((void*)&fd, (CUdeviceptr)segPtr, thisLen,
                                                CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, handleFlags),
                  ret, fail);
      segAddrs[s] = (void*)segPtr;
      segLens[s] = thisLen;
      segOffsets[s] = 0ULL;
      segFds[s] = fd;
      segPtr += thisLen;
      remaining -= thisLen;
    }
  }

  NCCLCHECKGOTO(IbMsegRegMrDmaBufMultiSeg(comm, nSeg, segAddrs, segLens, segOffsets, segFds, type, mhandle), ret, fail);

exit:
  if (segFds) {
    for (int s = 0; s < nSeg; s++) {
      if (segFds[s] != -1) (void)close(segFds[s]);
    }
  }
  free(scratch);
  return ret;
fail:
  goto exit;
}
#endif

ncclResult_t IbMsegRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
  // RCCL: fd < 0 registers one MR per physical segment. A real fd stays one MR.
  if (fd == -1) return rcclIbRegMrDmaBufFromRange(comm, data, size, type, offset, mhandle);
#endif
  return IbMsegRegMrDmaBufInternal(comm, data, size, type, offset, fd, 0ULL, mhandle);
}

ncclResult_t IbMsegRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  return IbMsegRegMrDmaBufInternal(comm, data, size, type, 0ULL, -1, 0, mhandle);
}

ncclResult_t IbMsegDeregMrInternal(IbMsegNetCommDevBase* base, ibv_mr* mhandle) {
  struct IbMsegMrCache* cache = &IbMsegDevs[base->ibDevN].mrCache;
  std::lock_guard<std::mutex> lock(IbMsegDevs[base->ibDevN].mutex);
  for (int i = 0; i < cache->population; i++) {
    if (mhandle == cache->slots[i].mr) {
      if (0 == --cache->slots[i].refs) {
        memmove(&cache->slots[i], &cache->slots[--cache->population], sizeof(struct IbMsegMr));
        if (cache->population == 0) {
          free(cache->slots);
          cache->slots = NULL;
          cache->capacity = 0;
        }
        NCCLCHECK(wrap_ibv_dereg_mr(mhandle));
      }
      return ncclSuccess;
    }
  }
  WARN("NET/IB: could not find mr %p inside cache of %d entries", mhandle, cache->population);
  return ncclInternalError;
}

// RCCL: one slot per (segment, device) for rcclIbSegRegister and rcclIbSegDeregister, laid out like segMrs.
struct rcclIbMultiSegRegCtx {
  struct IbMsegNetCommBase* base;
  void** segAddrs;
  size_t* segLens;
  uint64_t* segOffsets;
  int* segFds;
  int type;
};

static ncclResult_t rcclIbMultiSegRegOne(void* vctx, int idx, void** mr) {
  struct rcclIbMultiSegRegCtx* ctx = (struct rcclIbMultiSegRegCtx*)vctx;
  int s = idx / NCCL_IB_MAX_DEVS_PER_NIC;
  int i = idx % NCCL_IB_MAX_DEVS_PER_NIC;
  if (i >= ctx->base->vProps.ndevs) return ncclSuccess;
  struct IbMsegNetCommDevBase* devComm = IbMsegGetNetCommDevBase(ctx->base, i);
  ibv_mr* m = NULL;
  NCCLCHECK(IbMsegRegMrDmaBufInternal2(devComm, ctx->segAddrs[s], ctx->segLens[s], ctx->type, ctx->segOffsets[s],
                                       ctx->segFds[s], 0ULL, &m));
  *mr = m;
  return ncclSuccess;
}

static ncclResult_t rcclIbMultiSegDeregOne(void* vctx, int idx, void* mr) {
  struct rcclIbMultiSegRegCtx* ctx = (struct rcclIbMultiSegRegCtx*)vctx;
  int i = idx % NCCL_IB_MAX_DEVS_PER_NIC;
  struct IbMsegNetCommDevBase* devComm = IbMsegGetNetCommDevBase(ctx->base, i);
  return IbMsegDeregMrInternal(devComm, (ibv_mr*)mr);
}

// RCCL: free the counted segment arrays and the handle; the MRs are already released.
static void rcclIbMrHandleFreeSegments(struct IbMsegMrHandle* h) {
  rcclIbSegMrFreeArrays(&h->seg);
  free(h);
}

// RCCL: a receive overlapping more than RCCL_IB_CTS_INLINE_SEGS segments sends a reference to this layout.
static ncclResult_t rcclIbMrHandlePublishLayout(struct IbMsegNetCommBase* base, struct IbMsegMrHandle* h) {
  struct ibv_pd* pds[NCCL_IB_MAX_DEVS_PER_NIC];
  for (int d = 0; d < base->vProps.ndevs; d++) pds[d] = IbMsegGetNetCommDevBase(base, d)->pd;
  return rcclIbSegMrPublishLayout(&h->seg, base->vProps.ndevs, pds);
}

static ncclResult_t rcclIbDeregMrMultiSeg(void* comm, void* mhandle);

// RCCL: one MR per segment. A whole-range dma-buf MR fails with EINVAL because
// the export only describes the first physical segment.
ncclResult_t IbMsegRegMrDmaBufMultiSeg(void* comm, int nSeg, void** segAddrs, size_t* segLens, uint64_t* segOffsets,
                                       int* segFds, int type, void** mhandle) {
  ncclResult_t ret = ncclSuccess;
  struct IbMsegNetCommBase* base = (struct IbMsegNetCommBase*)comm;
  if (!rcclIbSegCountSupported(nSeg)) {
    WARN("NET/IB: multi-segment registration with %d segments is past the supported %d", nSeg,
         RCCL_IB_REG_SEG_DEFAULT);
    return ncclInvalidUsage;
  }
  ret = rcclIbMultiSegAdmit(base, nSeg);
  if (ret != ncclSuccess) return ret;
  struct IbMsegMrHandle* mhandleWrapper = NULL;
  NCCLCHECK(ncclCalloc(&mhandleWrapper, 1));
  NCCLCHECKGOTO(rcclIbSegMrInit(&mhandleWrapper->seg, nSeg, segAddrs, segLens), ret, fail);
  {
    // RCCL: slots past ndevs stay NULL and are skipped.
    struct rcclIbMultiSegRegCtx ctx = {base, segAddrs, segLens, segOffsets, segFds, type};
    struct rcclIbSegRegOps ops = {rcclIbMultiSegRegOne, rcclIbMultiSegDeregOne};
    NCCLCHECKGOTO(rcclIbSegRegister(&ops, &ctx, mhandleWrapper->seg.segMrs, nSeg * NCCL_IB_MAX_DEVS_PER_NIC), ret,
                  fail);
  }
  // Alias segment 0 into mrs[] so single-segment consumers keep working.
  for (int i = 0; i < base->vProps.ndevs; i++) mhandleWrapper->mrs[i] = rcclIbSegMrAt(&mhandleWrapper->seg, 0, i);
  if (!base->isSend && nSeg > RCCL_IB_CTS_INLINE_SEGS) {
    ret = rcclIbMrHandlePublishLayout(base, mhandleWrapper);
    if (ret != ncclSuccess) {
      (void)rcclIbDeregMrMultiSeg(comm, mhandleWrapper);
      return ret;
    }
  }
  INFO(NCCL_NET | NCCL_REG, "NET/IB: registered multi-segment buffer %p size %zu as %d DMA-BUF MRs", segAddrs[0],
       (size_t)mhandleWrapper->seg.segOff[nSeg], nSeg);
  *mhandle = (void*)mhandleWrapper;
  return ncclSuccess;
fail:
  // rcclIbSegRegister has already released a partial walk.
  rcclIbMrHandleFreeSegments(mhandleWrapper);
  return ret;
}

// mrs[] aliases segment 0, so it is freed via segMrs here (do not double-free).
// RCCL: deregister every MR, then free the handle, and return the first error.
static ncclResult_t rcclIbDeregMrMultiSeg(void* comm, void* mhandle) {
  struct IbMsegMrHandle* mhandleWrapper = (struct IbMsegMrHandle*)mhandle;
  struct rcclIbMultiSegRegCtx ctx = {(struct IbMsegNetCommBase*)comm, NULL, NULL, NULL, NULL, 0};
  struct rcclIbSegRegOps ops = {NULL, rcclIbMultiSegDeregOne};
  ncclResult_t layoutRet = rcclIbSegMrFreeLayout(&mhandleWrapper->seg);
  ncclResult_t ret = rcclIbSegDeregister(&ops, &ctx, mhandleWrapper->seg.segMrs,
                                         mhandleWrapper->seg.segMrs == NULL
                                           ? 0
                                           : mhandleWrapper->seg.nSegments * NCCL_IB_MAX_DEVS_PER_NIC);
  if (ret == ncclSuccess) ret = layoutRet;
  rcclIbMrHandleFreeSegments(mhandleWrapper);
  return ret;
}

ncclResult_t IbMsegDeregMr(void* comm, void* mhandle) {
  if (mhandle == NULL) return ncclSuccess;

  struct IbMsegMrHandle* mhandleWrapper = (struct IbMsegMrHandle*)mhandle;
  struct IbMsegNetCommBase* base = (struct IbMsegNetCommBase*)comm;
  // RCCL: a walked handle, one segment included, owns one MR per segment and the segment arrays.
  if (mhandleWrapper->seg.segMrs != NULL) return rcclIbDeregMrMultiSeg(comm, mhandle);
  for (int i = 0; i < base->vProps.ndevs; i++) {
    // Each IbMsegNetCommDevBase is at different offset in send and recv netComms
    struct IbMsegNetCommDevBase* devComm = IbMsegGetNetCommDevBase(base, i);
    NCCLCHECK(IbMsegDeregMrInternal(devComm, mhandleWrapper->mrs[i]));
  }
  free(mhandleWrapper);
  return ncclSuccess;
}
