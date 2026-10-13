/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_FAULT_INJECTION

#include "common.h"
#include "net_ib_flush_inspect.h"

static_assert(NCCL_IB_MAX_DEVS_PER_NIC <= NCCL_IB_FLUSH_INSPECT_MAX_DEVS,
              "ncclIbFlushState must hold every device of a recv comm");

extern "C" ncclResult_t ncclIbFlushGetState(void* recvComm, struct ncclIbFlushState* out) {
  struct ncclIbRecvComm* rComm = (struct ncclIbRecvComm*)recvComm;
  if (rComm == NULL || out == NULL) return ncclInvalidArgument;
  memset(out, 0, sizeof(*out));
  out->flushEnabled = rComm->flushEnabled;
  out->ndevs = std::min(std::max(rComm->base.vProps.ndevs, 0), NCCL_IB_MAX_DEVS_PER_NIC);
  for (int i = 0; i < out->ndevs; i++) {
    const struct ncclIbGpuFlush* gpuFlush = &rComm->devs[i].gpuFlush;
    out->scratchpadRegistered[i] = gpuFlush->gpuFlushGpuMem != nullptr && gpuFlush->gpuMr != nullptr;
    out->scratchpadDmaBuf[i] = out->scratchpadRegistered[i] && gpuFlush->dmabuf_fd >= 0;
  }
  return ncclSuccess;
}
#endif /* ENABLE_FAULT_INJECTION */
