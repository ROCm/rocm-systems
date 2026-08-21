/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Mixed-architecture guard. gfx942 implements e4m3 as fnuz and gfx950 the OCP
 * form, and HIP exposes one per architecture, so a communicator spanning both
 * would exchange bytes each side decodes differently -- silently wrong numerics
 * rather than an error. Fatal at construction, for every dtype.
 * RCCL_EP_FAKE_ARCH exercises the rejection without a heterogeneous node.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_EP_ARCH_H_
#define RCCL_EP_ARCH_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <hip/hip_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include "device/dispatch.h"
#include "include/ep_layout.h"

namespace rccl_ep {

// Publish this rank's architecture string into every peer's arch region.
__global__ void k_ep_publish_arch(EpConfig cfg, WindowView* __restrict__ peers, const char* __restrict__ mine) {
  const int dst = blockIdx.x;
  if (dst >= cfg.num_ranks) return;
  char* slot = peers[dst].arch() + (size_t)cfg.rank * kArchLen;
  for (int i = threadIdx.x; i < kArchLen; i += blockDim.x) slot[i] = mine[i];
}

// Returns 0 if every rank reports the same architecture, -1 otherwise.
// On mismatch `err` receives a message naming both architectures.
//
// Must be called with the window registered and the device communicator
// created; `d_views` is the peer table and `self` this rank's own view.
inline int check_uniform_arch(const EpConfig& cfg, int device, WindowView* d_views, const WindowView& self,
                              ncclDevComm devComm, char* err, size_t err_len) {
  char mine[kArchLen];
  memset(mine, 0, sizeof(mine));
  hipDeviceProp_t pr;
  if (hipGetDeviceProperties(&pr, device) != hipSuccess) {
    snprintf(err, err_len, "hipGetDeviceProperties(%d) failed", device);
    return -1;
  }
  snprintf(mine, sizeof(mine), "%s", pr.gcnArchName);

  const char* fake = getenv("RCCL_EP_FAKE_ARCH");
  const char* fake_rank = getenv("RCCL_EP_FAKE_ARCH_RANK");
  if (fake && *fake && (!fake_rank || atoi(fake_rank) == cfg.rank)) snprintf(mine, sizeof(mine), "%s", fake);

  char* d_mine = nullptr;
  if (hipMalloc(&d_mine, kArchLen) != hipSuccess) {
    snprintf(err, err_len, "hipMalloc for the arch check failed");
    return -1;
  }
  hipMemcpy(d_mine, mine, kArchLen, hipMemcpyHostToDevice);

  hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(8), dim3(kWarpSize), 0, 0, devComm);
  hipLaunchKernelGGL(k_ep_publish_arch, dim3(cfg.num_ranks), dim3(kArchLen), 0, 0, cfg, d_views, d_mine);
  hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(8), dim3(kWarpSize), 0, 0, devComm);
  hipError_t he = hipDeviceSynchronize();
  hipFree(d_mine);
  if (he != hipSuccess) {
    snprintf(err, err_len, "arch exchange: %s", hipGetErrorString(he));
    return -1;
  }

  // WindowView's accessors are device-side, so the offset is applied here
  // rather than calling self.arch().
  char* all = (char*)malloc((size_t)cfg.num_ranks * kArchLen);
  hipMemcpy(all, self.base + self.l.off_arch, (size_t)cfg.num_ranks * kArchLen, hipMemcpyDeviceToHost);

  int rc = 0;
  for (int r = 0; r < cfg.num_ranks; ++r) {
    all[(size_t)r * kArchLen + kArchLen - 1] = '\0';
    if (strcmp(all, all + (size_t)r * kArchLen) != 0) {
      snprintf(err, err_len,
               "mixed-architecture communicator: rank 0 is %s but rank %d is %s. "
               "FP8 e4m3 is fnuz on gfx942 and OCP on gfx950, so the ranks would "
               "not agree on the wire format",
               all, r, all + (size_t)r * kArchLen);
      rc = -1;
      break;
    }
  }
  free(all);
  return rc;
}

}  // namespace rccl_ep

#endif  // RCCL_EP_ARCH_H_
