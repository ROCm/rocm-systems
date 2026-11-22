/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

#ifndef RCCL_TRANSPORT_MSEG_EXPORT_MSEG_H_
#define RCCL_TRANSPORT_MSEG_EXPORT_MSEG_H_

#include <stddef.h>
#include <stdint.h>
#include <unistd.h>

#include "alloc.h" // RCCL: cuMem wrappers for the build's platform.
#include "bitops.h"
#include "nccl.h"
#include "os.h"
#include "rccl_common.h"

// RCCL: backend-neutral walk of a VA range one physical segment at a time. Each step exports that segment alone,
// because a HIP DMA-BUF export covers only the first physical allocation of a range.

// Walk position plus the local layout it fills: segOff[s + 1] is the end of segment s, vas[s] its start.
struct rcclIbSegWalkState {
  uintptr_t ptr;
  size_t remaining;
  size_t cum;
  size_t* segOff;
  uintptr_t* vas;
};

// One segment's export. fd is the owner's published fd (borrowed) or one exported here (exported == true).
struct rcclIbSegExport {
  uintptr_t ptr;
  size_t len;
  int fd;
  uint64_t fdOffset;
  bool exported;
};

static inline void rcclIbSegExportClose(struct rcclIbSegExport* e) {
  if (e->exported && e->fd >= 0) (void)close(e->fd);
  e->fd = -1;
  e->exported = false;
}

#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
// Exports the segment at w->ptr over the whole pages the MR spans; the caller registers it, closes it, then advances.
static inline ncclResult_t rcclIbSegExportNext(const struct rcclIbSegWalkState* w, struct rcclIbSegExport* e) {
  CUdeviceptr segBase = 0;
  size_t segSize = 0;
  e->fd = -1;
  e->exported = false;
  CUCHECK(cuMemGetAddressRange(&segBase, &segSize, (CUdeviceptr)w->ptr));
  // Host VMM can report segBase==0; subtracting then wraps the length.
  size_t inSeg = (segBase == 0) ? segSize : segSize - (w->ptr - (uintptr_t)segBase);
  e->ptr = w->ptr;
  e->len = w->remaining < inSeg ? w->remaining : inSeg;
  static size_t hostPageSize = ncclOsGetPageSize();
  uintptr_t exportBase = w->ptr & ~(uintptr_t)(hostPageSize - 1);
  size_t exportLen = w->ptr + e->len - exportBase;
  ALIGN_SIZE(exportLen, hostPageSize);
  // Imported host segments cannot be exported here; use the owner's fd published by the test or window.
  e->fdOffset = 0;
  int fd = rcclHostSegmentFdLookup(exportBase, exportLen, &e->fdOffset);
  if (fd < 0) {
    CUCHECK(cuMemGetHandleForAddressRange((void*)&fd, (CUdeviceptr)exportBase, exportLen,
                                          CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
    e->exported = true;
  }
  e->fd = fd;
  return ncclSuccess;
}

// An owner-published fd is always a host segment; otherwise the allocation's location says.
static inline ncclResult_t rcclIbSegExportIsHost(const struct rcclIbSegExport* e, bool* host) {
  *host = true;
  if (e->fd >= 0 && !e->exported) return ncclSuccess;
  return ncclCuMemIsSysmemSegment((CUdeviceptr)e->ptr, host);
}
#endif

// Records segment seg after it registered and moves the walk past it.
static inline void rcclIbSegWalkAdvance(struct rcclIbSegWalkState* w, int seg, size_t len) {
  w->vas[seg] = w->ptr;
  w->cum += len;
  w->segOff[seg + 1] = w->cum;
  w->ptr += len;
  w->remaining -= len;
}

#endif // RCCL_TRANSPORT_MSEG_EXPORT_MSEG_H_
