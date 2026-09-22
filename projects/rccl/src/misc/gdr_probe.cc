/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gdr_peermem.h"
#include "core.h"
#include "ibvwrap.h"

int ncclIbProbeGdrSupport(struct ibv_context* context, int relaxedOrderingEnabled) {
  int result = 0;
  void* gpuBuf = nullptr;
  struct ibv_pd* pd = nullptr;
  struct ibv_mr* mr = nullptr;
  unsigned int flags =
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

  // Single exit path below (cleanup:) so gpuBuf/pd always get released, even
  // if a future early return gets added here.
  if (hipMalloc(&gpuBuf, 4096) != hipSuccess) goto cleanup;
  if (wrap_ibv_alloc_pd(&pd, context) != ncclSuccess) goto cleanup;

  if (relaxedOrderingEnabled) {
    mr = wrap_direct_ibv_reg_mr_iova2(pd, gpuBuf, 4096, (uint64_t)gpuBuf, flags | IBV_ACCESS_RELAXED_ORDERING);
  } else {
    mr = wrap_direct_ibv_reg_mr(pd, gpuBuf, 4096, flags);
  }
  result = (mr != nullptr);
  if (mr && wrap_ibv_dereg_mr(mr) != ncclSuccess) {
    WARN("NET/IB: peermem probe: ibv_dereg_mr failed");
  }

cleanup:
  if (pd && wrap_ibv_dealloc_pd(pd) != ncclSuccess) {
    WARN("NET/IB: peermem probe: ibv_dealloc_pd failed");
  }
  if (gpuBuf) (void)hipFree(gpuBuf);
  return result;
}
