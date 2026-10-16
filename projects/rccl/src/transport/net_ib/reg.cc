/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"
#include "rocmwrap.h"
#include "net_ib/multiseg.h"
#include "rccl_ib_multiseg.h"

ncclResult_t ncclIbRegMrDmaBufInternal2(ncclIbNetCommDevBase* base, void* data, size_t size, int type, uint64_t offset,
                                        int fd, uint64_t mrFlags, ibv_mr** mhandle) {
  static thread_local uintptr_t pageSize = 0;
  if (pageSize == 0) pageSize = sysconf(_SC_PAGESIZE);
  struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize - 1) / pageSize;
  std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);
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
      bool relaxedOrdering = ncclIbRelaxedOrderingEnabled && (mrFlags & NCCL_NET_MR_FLAG_FORCE_SO) == 0;
      if (relaxedOrdering) flags |= IBV_ACCESS_RELAXED_ORDERING;
      if (fd != -1) {
        /* DMA-BUF support */
        if (!ncclIbDevs[base->ibDevN].capsProvider.mlx5.dataDirect) {
          NCCLCHECK(wrap_ibv_reg_dmabuf_mr(&mr, base->pd, offset, pages * pageSize, addr, fd, flags));
        } else {
          NCCLCHECK(wrap_mlx5dv_reg_dmabuf_mr(&mr, base->pd, offset, pages * pageSize, addr, fd, flags,
                                              MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT));
        }
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
        memmove(cache->slots + slot + 1, cache->slots + slot, (cache->population - slot) * sizeof(struct ncclIbMr));
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

ncclResult_t ncclIbDeregMrInternal(ncclIbNetCommDevBase* base, ibv_mr* mhandle);

/* DMA-BUF support */
ncclResult_t ncclIbRegMrDmaBufInternal(void* comm, void* data, size_t size, int type, uint64_t offset, int fd,
                                       uint64_t mrFlags, void** mhandle) {
  ncclResult_t ret = ncclSuccess;
  if (size == 0) {
    WARN("NET/IB: Invalid memory registration size %zu, expected > 0", size);
    return ncclInternalError;
  }
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*)comm;
  struct ncclIbMrHandle* mhandleWrapper = NULL;
  int registered = 0;
  *mhandle = NULL;
  NCCLCHECK(ncclCalloc(&mhandleWrapper, 1));
  mhandleWrapper->nSegments = 1;
  for (int i = 0; i < base->vProps.ndevs; i++) {
    // Each ncclIbNetCommDevBase is at different offset in send and recv netComms
    struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
    NCCLCHECKGOTO(ncclIbRegMrDmaBufInternal2(devComm, data, size, type, offset, fd, mrFlags, mhandleWrapper->mrs + i),
                  ret, fail);
    registered++;
  }
  *mhandle = (void*)mhandleWrapper;
exit:
  return ret;
fail:
  for (int i = 0; i < registered; i++) {
    struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
    (void)ncclIbDeregMrInternal(devComm, mhandleWrapper->mrs[i]);
  }
  free(mhandleWrapper);
  goto exit;
}

#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
// RCCL: plugin regMrDmaBuf with fd < 0. offset is (cuMem export flags << 32) | segment count.
static ncclResult_t rcclIbRegMrDmaBufFromRange(void* comm, void* data, size_t size, int type, uint64_t packed,
                                               void** mhandle) {
  const int nExpect = (int)(packed & 0xffffffffu);
  const int handleFlags = (int)(packed >> 32);
  ncclResult_t ret = ncclSuccess;
  void** segAddrs = NULL;
  size_t* segLens = NULL;
  uint64_t* segOffsets = NULL;
  int* segFds = NULL;
  int nSeg = 0;
  size_t remaining = size;
  *mhandle = NULL;

  if (nExpect < 1 || nExpect > NCCL_IB_MAX_SEGMENTS) {
    WARN("NET/IB: multi-segment registration with %d segments exceeds NCCL_IB_MAX_SEGMENTS=%d", nExpect,
         NCCL_IB_MAX_SEGMENTS);
    return ncclInvalidUsage;
  }
  NCCLCHECKGOTO(ncclCalloc(&segAddrs, nExpect), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&segLens, nExpect), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&segOffsets, nExpect), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&segFds, nExpect), ret, fail);
  for (int s = 0; s < nExpect; s++) segFds[s] = -1;

  {
    uintptr_t segPtr = (uintptr_t)data;
    for (int s = 0; s < nExpect && remaining > 0; s++) {
      CUdeviceptr segBase = 0;
      size_t segSize = 0;
      CUCHECKGOTO(cuMemGetAddressRange(&segBase, &segSize, (CUdeviceptr)segPtr), ret, fail);
      size_t inSeg = ncclIbBytesRemainingInSegment(segPtr, (uintptr_t)segBase, segSize);
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
      nSeg++;
    }
  }

  if (remaining != 0 || nSeg != nExpect) {
    WARN("NET/IB: enumerated %d/%d segments for buffer %p but %zu/%zu bytes remain unregistered", nSeg, nExpect, data,
         remaining, size);
    ret = ncclInvalidUsage;
    goto fail;
  }
  if (!ncclIbSegmentsUniform(nSeg, segLens)) {
    INFO(NCCL_NET | NCCL_REG,
         "Buffer %p (size %zu) has non-uniform segments; declining NET user-buffer registration (staging fallback)",
         data, size);
    ret = ncclInvalidUsage;
    goto fail;
  }
  NCCLCHECKGOTO(ncclIbRegMrDmaBufMultiSeg(comm, nSeg, segAddrs, segLens, segOffsets, segFds, type, mhandle), ret, fail);

exit:
  if (segFds) {
    for (int s = 0; s < nExpect; s++) {
      if (segFds[s] != -1) (void)close(segFds[s]);
    }
  }
  free(segAddrs);
  free(segLens);
  free(segOffsets);
  free(segFds);
  return ret;
fail:
  goto exit;
}
#endif

ncclResult_t ncclIbRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
  // RCCL: fd < 0 registers one MR per physical segment. A real fd stays one MR.
  if (fd == -1) return rcclIbRegMrDmaBufFromRange(comm, data, size, type, offset, mhandle);
#endif
  return ncclIbRegMrDmaBufInternal(comm, data, size, type, offset, fd, 0ULL, mhandle);
}

ncclResult_t ncclIbRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  return ncclIbRegMrDmaBufInternal(comm, data, size, type, 0ULL, -1, 0, mhandle);
}

ncclResult_t ncclIbDeregMrInternal(ncclIbNetCommDevBase* base, ibv_mr* mhandle) {
  struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;
  std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);
  for (int i = 0; i < cache->population; i++) {
    if (mhandle == cache->slots[i].mr) {
      if (0 == --cache->slots[i].refs) {
        memmove(&cache->slots[i], &cache->slots[--cache->population], sizeof(struct ncclIbMr));
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

// RCCL: one MR per segment. A whole-range dma-buf MR fails with EINVAL because
// the export only describes the first physical segment.
ncclResult_t ncclIbRegMrDmaBufMultiSeg(void* comm, int nSeg, void** segAddrs, size_t* segLens, uint64_t* segOffsets,
                                       int* segFds, int type, void** mhandle) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*)comm;
  *mhandle = NULL;
  if (nSeg < 1 || nSeg > NCCL_IB_MAX_SEGMENTS) {
    WARN("NET/IB: multi-segment registration with %d segments exceeds NCCL_IB_MAX_SEGMENTS=%d", nSeg,
         NCCL_IB_MAX_SEGMENTS);
    return ncclInvalidUsage;
  }
  uint32_t peerCaps =
    base->isSend ? ((struct ncclIbSendComm*)comm)->peerCaps : ((struct ncclIbRecvComm*)comm)->peerCaps;
  // RCCL: the side table describes the receiver. A sender splits locally onto
  // one remote rkey, so only recv registration requires the peer cap.
  if (!base->isSend && ncclIbDeclineMultiSegRegistration(peerCaps, nSeg)) {
    INFO(NCCL_NET | NCCL_REG,
         "NET/IB: peer does not advertise multi-segment CTS side table; declining %d-segment registration", nSeg);
    return ncclInvalidUsage;
  }
  struct ncclIbMrHandle* mhandleWrapper = NULL;
  NCCLCHECK(ncclCalloc(&mhandleWrapper, 1));
  mhandleWrapper->nSegments = nSeg;
  for (int s = 0; s < nSeg; s++) {
    mhandleWrapper->segStart[s] = (uintptr_t)segAddrs[s];
    mhandleWrapper->segLen[s] = segLens[s];
    for (int i = 0; i < base->vProps.ndevs; i++) {
      struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
      NCCLCHECKGOTO(ncclIbRegMrDmaBufInternal2(devComm, segAddrs[s], segLens[s], type, segOffsets[s], segFds[s], 0ULL,
                                               &mhandleWrapper->segMrs[s][i]),
                    ret, fail);
    }
  }
  // Alias segment 0 into mrs[] so single-segment consumers keep working.
  for (int i = 0; i < base->vProps.ndevs; i++) mhandleWrapper->mrs[i] = mhandleWrapper->segMrs[0][i];
  INFO(NCCL_NET | NCCL_REG, "NET/IB: registered multi-segment buffer %p size %zu as %d DMA-BUF MRs", segAddrs[0],
       (size_t)(mhandleWrapper->segStart[nSeg - 1] + mhandleWrapper->segLen[nSeg - 1] - mhandleWrapper->segStart[0]),
       nSeg);
  *mhandle = (void*)mhandleWrapper;
  return ncclSuccess;
fail:
  for (int s = 0; s < nSeg; s++) {
    for (int i = 0; i < base->vProps.ndevs; i++) {
      if (mhandleWrapper->segMrs[s][i]) {
        struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
        (void)ncclIbDeregMrInternal(devComm, mhandleWrapper->segMrs[s][i]);
      }
    }
  }
  free(mhandleWrapper);
  return ret;
}

// mrs[] aliases segment 0, so it is freed via segMrs[0] here (do not double-free).
// RCCL: deregister every MR, then free the handle, and return the first error.
static ncclResult_t rcclIbDeregMrMultiSeg(void* comm, void* mhandle) {
  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*)mhandle;
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*)comm;
  ncclResult_t ret = ncclSuccess;
  for (int s = 0; s < mhandleWrapper->nSegments; s++) {
    for (int i = 0; i < base->vProps.ndevs; i++) {
      if (!mhandleWrapper->segMrs[s][i]) continue;
      struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
      ncclResult_t dr = ncclIbDeregMrInternal(devComm, mhandleWrapper->segMrs[s][i]);
      if (dr != ncclSuccess && ret == ncclSuccess) ret = dr;
    }
  }
  free(mhandleWrapper);
  return ret;
}

ncclResult_t ncclIbDeregMr(void* comm, void* mhandle) {
  if (mhandle == NULL) return ncclSuccess;

  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*)mhandle;
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*)comm;
  // RCCL: multi-segment handles own one MR per segment, not mrs[] alone.
  if (mhandleWrapper->nSegments > 1) return rcclIbDeregMrMultiSeg(comm, mhandle);
  for (int i = 0; i < base->vProps.ndevs; i++) {
    // Each ncclIbNetCommDevBase is at different offset in send and recv netComms
    struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
    NCCLCHECK(ncclIbDeregMrInternal(devComm, mhandleWrapper->mrs[i]));
  }
  free(mhandleWrapper);
  return ncclSuccess;
}
