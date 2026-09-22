/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gdr_peermem.h"
#include "core.h"
#include "ibvwrap.h"

int ncclIbProbeGdrSupport(struct ibv_context* context, int relaxedOrderingEnabled) {
  void* gpuBuf = nullptr;
  if (hipMalloc(&gpuBuf, 4096) != hipSuccess) return 0;

  struct ibv_pd* pd;
  if (wrap_ibv_alloc_pd(&pd, context) != ncclSuccess) {
    (void)hipFree(gpuBuf);
    return 0;
  }

  unsigned int flags =
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  struct ibv_mr* mr;
  if (relaxedOrderingEnabled) {
    mr = wrap_direct_ibv_reg_mr_iova2(pd, gpuBuf, 4096, (uint64_t)gpuBuf, flags | IBV_ACCESS_RELAXED_ORDERING);
  } else {
    mr = wrap_direct_ibv_reg_mr(pd, gpuBuf, 4096, flags);
  }
  int result = (mr != nullptr);
  if (mr) (void)wrap_ibv_dereg_mr(mr);
  (void)wrap_ibv_dealloc_pd(pd);
  (void)hipFree(gpuBuf);
  return result;
}
