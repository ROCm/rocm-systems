/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "net.h"
#include "proxy.h"
#include "rccl_ib_multiseg.h"
#include "net_ib/multiseg.h"

// Register a multi-segment cuMem/VMM buffer for the classic NET/IB transport by exporting one dma-buf fd per
// physical segment and registering one MR per segment. ROCm/HIP dma-buf export only describes the first physical
// segment, so a single whole-range MR fails with EINVAL.
//
// Guards: only the built-in IB plugin is supported, and segments must be
// uniformly sized (interior segments equal the max stride; the first and last
// may be shorter when registration clips a physical allocation). Non-uniform
// layouts decline so the collective falls back to staging instead of the
// fatal QP path.
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
ncclResult_t rcclNetIbRegMrMultiSeg(struct ncclProxyState* proxyState, void* netComm, void* buffer, size_t totalSize,
                                    int numSegments, int handleFlags, void** handle) {
  ncclResult_t ret = ncclSuccess;
  void** segAddrs = NULL;
  size_t* segLens = NULL;
  uint64_t* segOffsets = NULL;
  int* segFds = NULL;
  int nSeg = 0;
  size_t remaining = totalSize;

  if (proxyState->ncclNet != &ncclNetIb) {
    INFO(NCCL_NET | NCCL_REG, "Multi-segment (%d) NET registration only supported on the IB transport", numSegments);
    return ncclInvalidUsage;
  }
  if (numSegments < 1 || numSegments > NCCL_IB_MAX_SEGMENTS) {
    WARN("NET/IB: multi-segment registration with %d segments exceeds NCCL_IB_MAX_SEGMENTS=%d", numSegments,
         NCCL_IB_MAX_SEGMENTS);
    return ncclInvalidUsage;
  }
  NCCLCHECKGOTO(ncclCalloc(&segAddrs, numSegments), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&segLens, numSegments), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&segOffsets, numSegments), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&segFds, numSegments), ret, fail);
  for (int s = 0; s < numSegments; s++) segFds[s] = -1;

  {
    uintptr_t segPtr = (uintptr_t)buffer;
    for (int s = 0; s < numSegments && remaining > 0; s++) {
      CUdeviceptr segBase = 0;
      size_t segSize = 0;
      CUCHECKGOTO(cuMemGetAddressRange(&segBase, &segSize, (CUdeviceptr)segPtr), ret, fail);
      size_t inSeg = ncclIbBytesRemainingInSegment(segPtr, (uintptr_t)segBase, segSize);
      size_t thisLen = remaining < inSeg ? remaining : inSeg;
      int fd = -1;
      // Export this segment alone: one physical allocation, so its fd is complete.
      CUCHECKGOTO(cuMemGetHandleForAddressRange((void*)&fd, (CUdeviceptr)segPtr, thisLen,
                                                CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                                handleFlags),
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

  // A count/range mismatch must never produce a partially registered handle:
  // the CTS path would otherwise advertise addresses beyond the final MR with
  // that MR's rkey, causing IBV_WC_REM_ACCESS_ERR on the remote write.
  if (remaining != 0 || nSeg != numSegments) {
    WARN("NET/IB: enumerated %d/%d segments for buffer %p but %zu/%zu bytes remain unregistered", nSeg, numSegments,
         buffer, remaining, totalSize);
    ret = ncclInvalidUsage;
    goto fail;
  }

  // Uniform-segment guard: interior segments equal the max stride; first/last
  // may be shorter, so every two-segment layout (including DeepEP [GPU][CPU])
  // registers. Layouts with a mismatched interior segment decline here and the
  // collective falls back to staging.
  if (!ncclIbSegmentsUniform(nSeg, segLens)) {
    INFO(NCCL_NET | NCCL_REG,
         "Buffer %p (size %zu) has non-uniform segments; declining NET user-buffer registration (staging fallback)",
         buffer, totalSize);
    ret = ncclInvalidUsage;
    goto fail;
  }

  NCCLCHECKGOTO(ncclIbRegMrDmaBufMultiSeg(netComm, nSeg, segAddrs, segLens, segOffsets, segFds, NCCL_PTR_CUDA,
                                          handle),
                ret, fail);

exit:
  if (segFds) {
    for (int s = 0; s < numSegments; s++) {
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
