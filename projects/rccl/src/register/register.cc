/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "nccl.h"
#include "comm.h"
#include "net.h"
#include "register.h"
#include "register_inline.h"
#include "transport.h"
#include "group.h"
#include "api_trace.h"

using namespace rccl;

#if NCCL_CUMEM_VERSION_SUPPORTED(HIP_VERSION)
NCCL_PARAM(LocalRegister, "LOCAL_REGISTER", 1);
#else
NCCL_PARAM(LocalRegister, "LOCAL_REGISTER", 0);
#endif
RCCL_PARAM(RingAutoRegisterWarnBytes, "RING_AUTO_REGISTER_WARN_BYTES", 32LL << 30);
int64_t ncclParamLegacyCudaRegister();

ncclResult_t ncclRegLocalIsValid(struct ncclReg* reg, bool* isValid) {
  if (reg && isValid) {
    if (reg->localRefs) *isValid = true;
    else *isValid = false;
  }
  return ncclSuccess;
}

ncclResult_t ncclRegister(struct ncclComm* comm, void* data, size_t size, bool isGraph, void** handle) {
  NCCLCHECK(CommCheck(comm, "ncclCommRegister", "comm"));

  struct ncclRegCache* cache = &comm->regCache;
  uintptr_t pageSize = cache->pageSize;
  uintptr_t begAddr = (uintptr_t)data & -pageSize;
  uintptr_t endAddr = ((uintptr_t)data + size + pageSize - 1) & -pageSize;

  if (comm->checkMode != ncclCheckModeDefault) NCCLCHECK(CudaPtrCheck(data, comm, "buff", "ncclCommRegister"));

  bool hasSysmemSegment = false;
  if (ncclCuMemEnable()) {
    CUdeviceptr base;
    size_t baseSize;
    CUCHECK(cuMemGetAddressRange(&base, &baseSize, (CUdeviceptr)data));
    CUmemorytype memType;
    CUCHECK(cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)data));
    if (memType == CU_MEMORYTYPE_HOST) {
      hasSysmemSegment = true;
    } else {
      // Check for a Sysmem segment is only valid with cuMem based allocators, so a IS_LEGACY_CUDA_IPC check is required to ensure
      // that we're calling ncclCuMemGetAddressRange only when necessary.
#if HIP_VERSION >= 71260540
      int numSegments;
      int legacyIpcCap = 0;
      CUCHECK(cuPointerGetAttribute((void*)&legacyIpcCap, CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE,
                                    (CUdeviceptr)base));
      if (!legacyIpcCap) {
        NCCLCHECK(ncclCuMemGetAddressRange((CUdeviceptr)data, size, (CUdeviceptr*)&base, &baseSize, &numSegments,
                                           &hasSysmemSegment));
      }
#else
      // HIP 7.0.x rejects HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE with
      // hipErrorNotSupported (same HIP_VERSION gate as ipcRegisterBuffer in p2p.cc).
      // Do not call ncclCuMemGetAddressRange: that path retains a cuMem handle
      // and illegal-memory-accesses hipMalloc buffers on HIP 7.0. Skip
      // registration for managed/unified allocations (cannot walk segments).
      if (memType != CU_MEMORYTYPE_DEVICE) {
        hasSysmemSegment = true;
      }
#endif
    }
  }
  if (hasSysmemSegment) {
    INFO(NCCL_REG, "Skipping registration for buffer %p size %zi since it contains segments backed by CPU memory", data,
         size);
    return ncclSuccess;
  } else {
    INFO(NCCL_REG, "register comm %p buffer %p size %zi", comm, data, size);
  }

  for (int slot = 0; /*true*/; slot++) {
    if ((slot == cache->population) || (begAddr < cache->slots[slot]->begAddr)) {
      if (cache->population == cache->capacity) { // must grow cache
        cache->capacity = cache->capacity < 32 ? 32 : 2 * cache->capacity;
        NCCLCHECK(ncclRealloc(&cache->slots, cache->population, cache->capacity));
      }
      memmove(cache->slots + slot + 1, cache->slots + slot, (cache->population - slot) * sizeof(struct ncclReg*));
      NCCLCHECK(ncclCalloc(cache->slots + slot, 1));
      struct ncclReg* regSlot = cache->slots[slot];
      regSlot->begAddr = begAddr;
      regSlot->endAddr = endAddr;
      if (isGraph) regSlot->graphRefs = 1;
      else regSlot->localRefs = 1;
      cache->population += 1;
      *handle = regSlot;
      goto exit;
    } else if ((cache->slots[slot]->begAddr <= begAddr) && (cache->slots[slot]->endAddr >= endAddr)) {
      if (rcclRegHidden(cache, cache->slots[slot])) continue;
      if (cache->slots[slot]->rcclAuto.owned) {
        bool stale = false;
        NCCLCHECK(rcclRegCheckOwned(comm, data, cache->slots[slot], &stale));
        if (stale) {
          slot = -1;
          continue;
        }
      }
      if (isGraph) cache->slots[slot]->graphRefs++;
      else cache->slots[slot]->localRefs++;
      *handle = cache->slots[slot];
      goto exit;
    }
  }

exit:
  return ncclSuccess;
}

static void regNetCleanup(struct ncclComm* comm, struct ncclReg* reg) {
  if (reg->state & NET_REG_COMPLETE) {
    struct ncclRegNetHandles* netHandle = reg->netHandleHead;
    struct ncclRegNetHandles* netHandlePrev;
    while (netHandle) {
      if (ncclNetDeregBuffer(comm, netHandle->proxyConn, netHandle->handle) != ncclSuccess) {
        ATTN("rank %d deregister NET buffer handle %p proxy rank %d failed", comm->rank, netHandle->handle,
             netHandle->proxyConn->rank);
      }
      netHandlePrev = netHandle;
      netHandle = netHandle->next;
      free(netHandlePrev);
    }
  }
  reg->netHandleHead = NULL;
  reg->state &= ~NET_REG_COMPLETE;
}

static ncclResult_t regCleanup(struct ncclComm* comm, struct ncclReg* reg) {
  regNetCleanup(comm, reg);
  if (reg->state & NVLS_REG_COMPLETE) {
    NCCLCHECK(ncclNvlsUbDeregister(comm, reg));
  }
  if (reg->state & COLLNET_REG_COMPLETE) {
    if (ncclCollnetDeregBuffer(comm, reg->collnetProxyconn, reg->collnetHandle) != ncclSuccess) {
      ATTN("rank %d deregister COLLNET buffer handle %p proxy rank %d failed", comm->rank, reg->collnetHandle,
           reg->collnetProxyconn->rank);
    }
  }

  // RCCL: do not gate this on IPC_REG_COMPLETE. ipcInfos is allocated before the first peer
  // registration is attempted, so a record that never reaches the flag still owns the array.
  // Entries are non-NULL only for peers that completed.
  if (reg->ipcInfos) {
    for (int i = 0; i < reg->ipcInfosSize; ++i)
      if (reg->ipcInfos[i]) {
        if (ncclIpcDeregBuffer(comm, reg->ipcInfos[i]) != ncclSuccess) {
          ATTN("rank %d deregister IPC buffer %p peerRank %d failed", comm->rank, reg->ipcInfos[i]->baseAddr,
               reg->ipcInfos[i]->peerRank);
        }
        free(reg->ipcInfos[i]);
      }
    free(reg->ipcInfos);
  }
  if (reg->regIpcAddrs.hostPeerRmtAddrs) free(reg->regIpcAddrs.hostPeerRmtAddrs);
  if (reg->regIpcAddrs.devPeerRmtAddrs) NCCLCHECK(ncclCudaFree(reg->regIpcAddrs.devPeerRmtAddrs, comm->memManager));
  return ncclSuccess;
}

// Frees an RCCL-owned record without the proxy calls regCleanup makes. ncclRegCleanup runs after the proxies are
// destroyed, so a record rcclRegAutoRelease did not release (abort, or a failed destroy) cannot ask the peers to
// unmap it; their imports go with their own teardown.
static ncclResult_t rcclRegDropLocal(struct ncclComm* comm, struct ncclReg* reg) {
  ncclResult_t ret = ncclSuccess;
  for (struct ncclRegNetHandles* netHandle = reg->netHandleHead; netHandle;) {
    struct ncclRegNetHandles* next = netHandle->next;
    free(netHandle);
    netHandle = next;
  }
  if (reg->ipcInfos) {
    for (int i = 0; i < reg->ipcInfosSize; ++i) free(reg->ipcInfos[i]);
    free(reg->ipcInfos);
  }
  if (reg->regIpcAddrs.hostPeerRmtAddrs) free(reg->regIpcAddrs.hostPeerRmtAddrs);
  if (reg->regIpcAddrs.devPeerRmtAddrs)
    NCCLCHECKIGNORE(ncclCudaFree(reg->regIpcAddrs.devPeerRmtAddrs, comm->memManager), ret);
  free(reg);
  return ret;
}

// Frees every retired record, carrying on past a failed teardown and returning the first error. Without the proxies,
// drops them locally.
static ncclResult_t rcclRegFreeRetired(struct ncclComm* comm, bool proxiesRunning) {
  struct ncclRegCache* cache = &comm->regCache;
  ncclResult_t ret = ncclSuccess;
  while (cache->rcclRetired) {
    struct ncclReg* reg = cache->rcclRetired;
    cache->rcclRetired = reg->rcclAuto.retiredNext;
    if (proxiesRunning) {
      NCCLCHECKIGNORE(regCleanup(comm, reg), ret);
      free(reg);
    } else {
      NCCLCHECKIGNORE(rcclRegDropLocal(comm, reg), ret);
    }
  }
  cache->rcclRetiredBytes = 0;
  return ret;
}

ncclResult_t ncclRegCleanup(struct ncclComm* comm) {
  struct ncclRegCache* cache = &comm->regCache;
  for (int i = 0; i < cache->population; i++) {
    struct ncclReg* reg = cache->slots[i];
    INFO(NCCL_DESTROY, "Cleanup buffer %p pages %lx", (void*)reg->begAddr,
         (reg->endAddr - reg->begAddr) / cache->pageSize);
    if (reg->rcclAuto.owned && reg->localRefs == 1 && reg->graphRefs == 0) {
      NCCLCHECK(rcclRegDropLocal(comm, reg));
      continue;
    }
    NCCLCHECK(regCleanup(comm, reg));
    free(reg);
  }
  free(cache->slots);
  NCCLCHECK(rcclRegFreeRetired(comm, /*proxiesRunning=*/false));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommRegister, const ncclComm_t comm, void* buff, size_t size, void** handle);

ncclResult_t ncclCommRegister_impl(const ncclComm_t comm, void* buff, size_t size, void** handle) {
  ncclResult_t ret = ncclSuccess;

  if (!ncclParamLocalRegister()) *handle = NULL;
  else {
    INFO(NCCL_INIT, "RCCL: ncclCommRegister");
    // Explicit host registration must observe completed graph reclamation
    // before looking up a virtual address that VMM may have reused. Internal
    // graph registration intentionally relies on group.cc's throttled drain.
    NCCLCHECKGOTO(CommCheck(comm, "ncclCommRegister", "comm"), ret, end);
    NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/false), ret, end);
    NCCLCHECKGOTO(ncclRegister(comm, buff, size, false, handle), ret, end);
  }
end:
  // !recording at sink
  NCCLCHECK(Recorder::instance().record(rrCommRegister, comm, *handle, buff, size));
  return ret;
}

ncclResult_t ncclCommGraphRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
  if (ncclP2pUsesMemcpy()) {
    *handle = NULL;
    INFO(NCCL_REG, "Skipping graph registration for buffer %p size %zi (P2pUsesMemcpy=%d)", buff, size,
         ncclP2pUsesMemcpy());
  } else {
    NCCLCHECK(ncclRegister(comm, buff, size, true, handle));
  }
  return ncclSuccess;
}

static ncclResult_t commDeregister(struct ncclComm* comm, bool isGraph, struct ncclReg* reg) {
  NCCLCHECK(CommCheck(comm, "ncclCommDeregister", "comm"));

  struct ncclRegCache* cache = &comm->regCache;
  int slot;
  int saveDev;
  if (reg == NULL) goto exit;
  CUDACHECK(cudaGetDevice(&saveDev));
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  for (slot = 0; slot < cache->population && cache->slots[slot] != reg; slot++);
  if (slot == cache->population) {
    WARN("Deregister: Could not find handle");
    return ncclInvalidUsage;
  }
  if (isGraph) --reg->graphRefs;
  else --reg->localRefs;
  if (reg->localRefs || reg->graphRefs) return ncclSuccess;
  NCCLCHECK(regCleanup(comm, reg));
  free(reg);
  memmove(cache->slots + slot, cache->slots + slot + 1, (cache->population - slot - 1) * sizeof(struct ncclReg*));
  cache->population -= 1;
  CUDACHECK(cudaSetDevice(saveDev));
exit:
  return ncclSuccess;
}

ncclResult_t rcclRegAutoLocal(struct ncclComm* comm, void* buff, size_t size, struct ncclReg** reg) {
  *reg = NULL;
  if (buff == NULL || size == 0) return ncclSuccess;

  rcclRegAutoScope visible(&comm->regCache);
  // ncclRegFind drops a record whose allocation was freed and the address reused, so a hit is safe to use.
  NCCLCHECK(ncclRegFind(comm, buff, size, reg));
  if (*reg) return ncclSuccess;

  // The record covers the buffer's whole allocation: peers import the whole allocation anyway, later calls on
  // other parts of it reuse the record, and the buffer ID identifies exactly what the record maps. Anything that
  // cannot be identified, or spans more than one allocation, stays unregistered rather than failing the collective.
  CUmemorytype memType;
  unsigned long long bufferId = 0;
  CUdeviceptr base = 0;
  size_t baseSize = 0;
  if (cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)buff) != CUDA_SUCCESS ||
      memType != CU_MEMORYTYPE_DEVICE ||
      cuPointerGetAttribute(&bufferId, CU_POINTER_ATTRIBUTE_BUFFER_ID, (CUdeviceptr)buff) != CUDA_SUCCESS ||
      bufferId == 0 || cuMemGetAddressRange(&base, &baseSize, (CUdeviceptr)buff) != CUDA_SUCCESS ||
      (uintptr_t)buff + size > (uintptr_t)base + baseSize) {
    (void)cudaGetLastError();
    return ncclSuccess;
  }
#if HIP_VERSION >= 71260540
  // Peers import only cuMem allocations unless legacy IPC is on (see ipcRegisterBuffer), so a legacy allocation would
  // only hold a record that no peer ever maps.
  if (comm->directMode || !ncclParamLegacyCudaRegister()) {
    int legacyIpcCap = 0;
    if (cuPointerGetAttribute(&legacyIpcCap, CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE, (CUdeviceptr)buff) !=
          CUDA_SUCCESS ||
        legacyIpcCap) {
      (void)cudaGetLastError();
      return ncclSuccess;
    }
  }
#endif

  void* handle = NULL;
  if (ncclRegister(comm, (void*)base, baseSize, /*isGraph=*/false, &handle) != ncclSuccess || handle == NULL) {
    (void)cudaGetLastError();
    return ncclSuccess;
  }
  struct ncclReg* found = (struct ncclReg*)handle;
  // A record ncclRegister just created holds only our reference. The cache population cannot tell, because the same
  // call may also have retired a stale record.
  if (found->localRefs != 1 || found->graphRefs != 0) {
    // ncclRegister reused a record ncclRegFind did not return; keep its owner's reference count unchanged.
    --found->localRefs;
    return ncclSuccess;
  }
  found->rcclAuto.owned = true;
  found->rcclAuto.bufferId = bufferId;
  *reg = found;
  return ncclSuccess;
}

ncclResult_t rcclRegCheckOwned(struct ncclComm* comm, const void* buff, struct ncclReg* reg, bool* stale) {
  unsigned long long bufferId = 0;
  *stale = cuPointerGetAttribute(&bufferId, CU_POINTER_ATTRIBUTE_BUFFER_ID, (CUdeviceptr)buff) != CUDA_SUCCESS ||
           bufferId != reg->rcclAuto.bufferId;
  if (!*stale) return ncclSuccess;
  (void)cudaGetLastError();

  struct ncclRegCache* cache = &comm->regCache;
  reg->rcclAuto.owned = false;
  if (reg->localRefs > 1 || reg->graphRefs > 0) {
    // The user or a graph registered this range too; it is theirs to deregister.
    --reg->localRefs;
    return ncclSuccess;
  }
  INFO(NCCL_REG, "rank %d buffer %p is a new allocation; retiring the RCCL registration [%p, %p)", comm->rank, buff,
       (void*)reg->begAddr, (void*)reg->endAddr);
  int slot = 0;
  while (cache->slots[slot] != reg) slot++;
  memmove(cache->slots + slot, cache->slots + slot + 1, (cache->population - slot - 1) * sizeof(struct ncclReg*));
  cache->population -= 1;
  // The IB MR cache matches on address alone, so a live NET registration would hand this record's MR, still backed
  // by the freed pages, to the next registration at the address. Unlike unmapping peer imports, this is a local
  // proxy call and does not wait on peer kernels.
  regNetCleanup(comm, reg);
  reg->rcclAuto.retiredNext = cache->rcclRetired;
  cache->rcclRetired = reg;
  cache->rcclRetiredBytes += reg->endAddr - reg->begAddr;
  if (!cache->rcclRetiredWarned && cache->rcclRetiredBytes > (size_t)rcclParamRingAutoRegisterWarnBytes()) {
    cache->rcclRetiredWarned = true;
    WARN("rank %d: RCCL_RING_AUTO_REGISTER is holding %zu bytes of freed AllReduce/AllGather buffers until the "
         "communicator is destroyed, because the job keeps freeing and reallocating them; GPU memory use will keep "
         "growing. Set RCCL_RING_AUTO_REGISTER=0 to stop this; large unregistered AllReduce/AllGather may then run "
         "slower.",
         comm->rank, cache->rcclRetiredBytes);
  }
  return ncclSuccess;
}

ncclResult_t rcclRegAutoRelease(struct ncclComm* comm) {
  struct ncclRegCache* cache = &comm->regCache;
  ncclResult_t ret = ncclSuccess;
  int kept = 0;
  for (int slot = 0; slot < cache->population; slot++) {
    struct ncclReg* reg = cache->slots[slot];
    if (reg->rcclAuto.owned && reg->localRefs == 1 && reg->graphRefs == 0) {
      NCCLCHECKIGNORE(regCleanup(comm, reg), ret);
      free(reg);
      continue;
    }
    if (reg->rcclAuto.owned) {
      // The user or a graph registered this range too; leave the record to them.
      reg->rcclAuto.owned = false;
      --reg->localRefs;
    }
    cache->slots[kept++] = reg;
  }
  cache->population = kept;
  NCCLCHECKIGNORE(rcclRegFreeRetired(comm, /*proxiesRunning=*/true), ret);
  return ret;
}

NCCL_API(ncclResult_t, ncclCommDeregister, const ncclComm_t comm, void* handle);
ncclResult_t ncclCommDeregister_impl(const ncclComm_t comm, void* handle) {
  NCCLCHECK(Recorder::instance().record(rrCommDeregister, comm, handle));
  // Explicit host deregistration observes completed graph reclamation before
  // touching the cache. Internal graph deregistration relies on group.cc's
  // throttled drain and can itself run from a reclamation callback.
  NCCLCHECK(CommCheck(comm, "ncclCommDeregister", "comm"));
  NCCLCHECK(ncclCommPollCallbacks(comm, /*waitSome=*/false));
  NCCLCHECK(commDeregister(comm, false, (struct ncclReg*)handle));
  return ncclSuccess;
}

ncclResult_t ncclCommGraphDeregister(const ncclComm_t comm, struct ncclReg* handle) {
  NCCLCHECK(commDeregister(comm, true, handle));
  return ncclSuccess;
}
