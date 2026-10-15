/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "dev_runtime_internal.h"
#include "alloc.h"
#include "bootstrap.h"
#include "comm.h"
#include "rma.h"
#include "rocmwrap.h"
#include "transport.h"

#include <mutex>
#include <unistd.h>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
ncclResult_t rcclDevrProbeNonSymWindow(struct ncclComm* comm, void* userPtr, size_t userSize, int winFlags) {
  // Skip the probe when cuMem is off on any rank: ROCm 7.0.2.2 faults in hipMemRetainAllocationHandle.
  if (!comm->cuMemSupport) return ncclSuccess;

  // Classify the pointer before ncclCuMemGetAddressRange. hipMalloc is not
  // cuMem; retaining it WARNs on every registration and can fault on HIP 7.0.
  // A failed probe must still allgather: returning here hangs peers in it.
  ncclResult_t probeRet = ncclSuccess;
  CUmemorytype memType = CU_MEMORYTYPE_DEVICE;
  bool hasSysmemSegment = false;
  struct ncclDevrState* devr = &comm->devrState;
  // No rank-local allocation here: lsaSize is team-uniform, so this bound
  // fails on every rank alike and none is left waiting in the allgather.
  if (devr->lsaSize > NCCL_MAX_LOCAL_RANKS) {
    WARN("LSA team of %d exceeds %d local ranks", devr->lsaSize, NCCL_MAX_LOCAL_RANKS);
    return ncclInternalError;
  }
  int kinds[NCCL_MAX_LOCAL_RANKS] = {};
  auto noteProbeFailure = [&](hipError_t err) {
    if (err == hipSuccess || probeRet != ncclSuccess) return;
    WARN("HIP failure '%s' while classifying window %p", hipGetErrorString(err), userPtr);
    (void)hipGetLastError();
    probeRet = ncclUnhandledCudaError;
  };
  noteProbeFailure(cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)userPtr));
  if (probeRet == ncclSuccess && memType == CU_MEMORYTYPE_HOST) {
    hasSysmemSegment = true;
  } else if (probeRet == ncclSuccess) {
#if NCCL_VER_GE(HIP_VERSION, ROCM_VER_7_12_60540)
    int legacyIpcCap = 0;
    CUdeviceptr base = 0;
    size_t baseSize = 0;
    noteProbeFailure(cuMemGetAddressRange(&base, &baseSize, (CUdeviceptr)userPtr));
    if (probeRet == ncclSuccess) {
      // Host VMM can report base 0; query at userPtr then, as the 7.0.2.x walk does.
      noteProbeFailure(cuPointerGetAttribute((void*)&legacyIpcCap, CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE,
                                             base != 0 ? base : (CUdeviceptr)userPtr));
    }
    if (probeRet == ncclSuccess && !legacyIpcCap) {
      CUdeviceptr memAddr = 0;
      size_t memSize = 0;
      int numSegments = 0;
      probeRet = ncclCuMemGetAddressRange(reinterpret_cast<CUdeviceptr>(userPtr), userSize, &memAddr, &memSize,
                                          &numSegments, &hasSysmemSegment);
    }
#else
    // 7.0.2.x cannot retain the allocation. Walk every mapping: a device
    // first page must not hide a later host page.
    uintptr_t cursor = reinterpret_cast<uintptr_t>(userPtr);
    uintptr_t end = cursor + userSize;
    while (probeRet == ncclSuccess && !hasSysmemSegment && cursor < end) {
      CUdeviceptr base = 0;
      size_t baseSize = 0;
      noteProbeFailure(cuMemGetAddressRange(&base, &baseSize, (CUdeviceptr)cursor));
      if (probeRet != ncclSuccess || baseSize == 0) break;
      CUmemorytype segType = CU_MEMORYTYPE_DEVICE;
      uintptr_t attrAt = base != 0 ? (uintptr_t)base : cursor;
      noteProbeFailure(cuPointerGetAttribute(&segType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)attrAt));
      if (probeRet != ncclSuccess) break;
      if (segType != CU_MEMORYTYPE_DEVICE) hasSysmemSegment = true;
      uintptr_t next = attrAt + baseSize;
      if (next <= cursor) break;
      cursor = next;
    }
#endif
  }
  // Host VMM cannot cudaIpcGetMemHandle. Allgather the classify result so a
  // host rank cannot return while a device peer still enters the IPC exchange.
  if (devr->lsaSize > 1) {
    // 0 = device, 1 = host, 2 = probe failed.
    int localKind = (probeRet != ncclSuccess) ? 2 : (hasSysmemSegment ? 1 : 0);
    kinds[devr->lsaSelf] = localKind;
    NCCLCHECK(bootstrapIntraNodeAllGather(comm->bootstrap, devr->lsaRankList, devr->lsaSelf, devr->lsaSize, kinds,
                                          sizeof(int)));
    int anyHost = 0, anyFail = 0;
    for (int r = 0; r < devr->lsaSize; r++) {
      if (kinds[r] == 2) anyFail = 1;
      if (kinds[r] == 1) anyHost = 1;
    }
    if (anyFail) return (probeRet != ncclSuccess) ? probeRet : ncclInternalError;
    if (anyHost) {
      WARN("Host-backed VMM cannot be exported via IPC for an LSA team of %d", devr->lsaSize);
      return ncclInvalidArgument;
    }
    return ncclSuccess;
  }
  if (probeRet != ncclSuccess) return probeRet;
  if (hasSysmemSegment) {
    NCCLCHECK(ncclDevrCheckRegistrationSupport(userPtr, userSize, comm, hasSysmemSegment, winFlags));
  }
  return ncclSuccess;
}

ncclResult_t rcclDevrGinSegmentRegType(bool isHost, int maxGlobalNumSegments, int* ptrType, bool* needDmabuf) {
  // Host VMM is CPU memory, but NCCL_PTR_HOST would pin it with ibv_reg_mr and EFAULT; use the DMA-BUF MR path.
  *ptrType = NCCL_PTR_CUDA;
  *needDmabuf = maxGlobalNumSegments > 1 || isHost;
  if (isHost && !ncclParamDmaBufEnable()) {
    WARN("Host VMM GIN segment requires NCCL_DMABUF_ENABLE=1");
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}
#endif

bool rcclDevrRmaRegistersMultiSegment(struct ncclComm* comm) {
  // Only RMA backends that register one MR per physical segment can take a multi-segment window.
  return ncclParamMultiSegmentRegister() && comm->rmaState.rmaProxyState.ncclRma == &ncclRmaIbProxy;
}

namespace {
struct rcclHostSegmentFd {
  uintptr_t base;
  size_t size;
  int fd;
};
std::mutex rcclHostSegmentFdsMutex;
std::vector<rcclHostSegmentFd> rcclHostSegmentFds;
} // namespace

void rcclHostSegmentFdRegister(void* base, size_t size, int fd) {
  std::lock_guard<std::mutex> lock(rcclHostSegmentFdsMutex);
  rcclHostSegmentFds.push_back({(uintptr_t)base, size, fd});
}

void rcclHostSegmentFdDeregister(int fd) {
  std::lock_guard<std::mutex> lock(rcclHostSegmentFdsMutex);
  for (size_t i = 0; i < rcclHostSegmentFds.size(); i++) {
    if (rcclHostSegmentFds[i].fd == fd) {
      rcclHostSegmentFds.erase(rcclHostSegmentFds.begin() + i);
      return;
    }
  }
}

int rcclHostSegmentFdLookup(uintptr_t addr, size_t len, uint64_t* offset) {
  std::lock_guard<std::mutex> lock(rcclHostSegmentFdsMutex);
  for (const rcclHostSegmentFd& e : rcclHostSegmentFds) {
    if (addr >= e.base && addr + len <= e.base + e.size) {
      *offset = addr - e.base;
      return dup(e.fd);
    }
  }
  return -1;
}
