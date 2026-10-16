/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common_cast.h"
#include "rocmwrap.h"
#include "net_ib/multiseg.h"
#include "rccl_ib_multiseg.h"

ncclResult_t IbCastDeregMrInternal(ncclIbNetCommDevBase* base, ibv_mr* mhandle);

// File-local helper. IB-CAST is a self-contained fork of net_ib; mark static so
// it does not collide with net_ib/reg.cc's identically-named definition at link time.
static ncclResult_t ncclIbRegMrDmaBufInternal2(ncclIbNetCommDevBase* base, void* data, size_t size, int type,
                                               uint64_t offset, int fd, uint64_t mrFlags, ibv_mr** mhandle) {
  static thread_local uintptr_t pageSize = 0;
  if (pageSize == 0) pageSize = sysconf(_SC_PAGESIZE);
  struct ncclIbMrCache* cache = &IbCastDevs[base->ibDevN].mrCache;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize - 1) / pageSize;
  std::lock_guard<std::mutex> lock(IbCastDevs[base->ibDevN].mutex);
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
      bool relaxedOrdering = IbCastRelaxedOrderingEnabled && (mrFlags & NCCL_NET_MR_FLAG_FORCE_SO) == 0;
      if (relaxedOrdering) flags |= IBV_ACCESS_RELAXED_ORDERING;
      if (fd != -1) {
        /* DMA-BUF support */
        if (!IbCastDevs[base->ibDevN].capsProvider.mlx5.dataDirect) {
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

/* DMA-BUF support */
ncclResult_t IbCastRegMrDmaBufInternal(void* comm, void* data, size_t size, int type, uint64_t offset, int fd,
                                       uint64_t mrFlags, void** mhandle) {
  ncclResult_t ret = ncclSuccess;
  // A caller that lost its connection reaches here with a null comm, and reading
  // the device list off it segfaults inside the plugin instead of telling the
  // caller what was wrong. size == 0 hit the same fate below via the page-count
  // math, so it is reported the same way rather than left as a release-build
  // assert.
  if (comm == NULL || mhandle == NULL || size == 0) {
    WARN("NET/IB-CAST: regMr/regMrDmaBuf called with comm=%p mhandle=%p size=%zu", comm, (void*)mhandle, size);
    return ncclInvalidArgument;
  }
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*)comm;
  struct ncclIbMrHandle* mhandleWrapper = NULL;
  int registered = 0;
  *mhandle = NULL;
  NCCLCHECK(ncclCalloc(&mhandleWrapper, 1));
  mhandleWrapper->nSegments = 1;
  for (int i = 0; i < base->vProps.ndevs; i++) {
    // Each ncclIbNetCommDevBase is at different offset in send and recv netComms
    struct ncclIbNetCommDevBase* devComm = IbCastGetNetCommDevBase(base, i);
    NCCLCHECKGOTO(ncclIbRegMrDmaBufInternal2(devComm, data, size, type, offset, fd, mrFlags, mhandleWrapper->mrs + i),
                  ret, fail);
    registered++;
  }
  *mhandle = (void*)mhandleWrapper;
exit:
  return ret;
fail:
  for (int i = 0; i < registered; i++) {
    struct ncclIbNetCommDevBase* devComm = IbCastGetNetCommDevBase(base, i);
    (void)IbCastDeregMrInternal(devComm, mhandleWrapper->mrs[i]);
  }
  free(mhandleWrapper);
  goto exit;
}

#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
// RCCL: plugin regMrDmaBuf with fd < 0. offset is (cuMem export flags << 32) | segment count.
static ncclResult_t IbCastRegMrDmaBufFromRange(void* comm, void* data, size_t size, int type, uint64_t packed,
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
    WARN("NET/IB: CAST multi-segment registration with %d segments exceeds NCCL_IB_MAX_SEGMENTS=%d", nExpect,
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
    WARN("NET/IB: CAST enumerated %d/%d segments for buffer %p but %zu/%zu bytes remain unregistered", nSeg, nExpect,
         data, remaining, size);
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
  NCCLCHECKGOTO(IbCastRegMrDmaBufMultiSeg(comm, nSeg, segAddrs, segLens, segOffsets, segFds, type, mhandle), ret, fail);

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

ncclResult_t IbCastRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
  // RCCL: fd < 0 registers one MR per physical segment. A real fd stays one MR.
  if (fd == -1) return IbCastRegMrDmaBufFromRange(comm, data, size, type, offset, mhandle);
#endif
  return IbCastRegMrDmaBufInternal(comm, data, size, type, offset, fd, 0ULL, mhandle);
}

ncclResult_t IbCastRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  return IbCastRegMrDmaBufInternal(comm, data, size, type, 0ULL, -1, 0, mhandle);
}

ncclResult_t IbCastDeregMrInternal(ncclIbNetCommDevBase* base, ibv_mr* mhandle) {
  struct ncclIbMrCache* cache = &IbCastDevs[base->ibDevN].mrCache;
  std::lock_guard<std::mutex> lock(IbCastDevs[base->ibDevN].mutex);
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

ncclResult_t IbCastDeregMr(void* comm, void* mhandle) {
  if (mhandle == NULL) return ncclSuccess;
  if (comm == NULL) {
    WARN("NET/IB-CAST: deregMr called with a null comm");
    return ncclInvalidArgument;
  }

  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*)mhandle;
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*)comm;
  if (mhandleWrapper->nSegments > 1) return rcclIbCastDeregMrMultiSeg(comm, mhandle);
  for (int i = 0; i < base->vProps.ndevs; i++) {
    // Each ncclIbNetCommDevBase is at different offset in send and recv netComms
    struct ncclIbNetCommDevBase* devComm = IbCastGetNetCommDevBase(base, i);
    NCCLCHECK(IbCastDeregMrInternal(devComm, mhandleWrapper->mrs[i]));
  }
  free(mhandleWrapper);
  return ncclSuccess;
}
