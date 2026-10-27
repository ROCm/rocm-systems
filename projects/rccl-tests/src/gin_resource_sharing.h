/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Runtime selection of the ncclGin resource-sharing mode for the GIN device
// testers (alltoall / all_gather / broadcast). The host --gin_resource_sharing
// value is mirrored into a device symbol once per device; the kernels read it to
// construct ncclGin. Default is THREAD (exclusive QP, drops the SQ lock on the
// rocshmem-GDA bnxt/mlx5 post path).
#ifndef GIN_RESOURCE_SHARING_H_
#define GIN_RESOURCE_SHARING_H_

#include "common.h"  // ginResourceSharingMode, ncclGinResourceSharingMode, CUDACHECK
#include "cuda_runtime.h"  // cudaGetDevice

// Device mirror of the host flag, as the resolved enum value. static so each GIN
// tester TU gets its own copy (device symbols are not shared across -fgpu-rdc TUs).
static __device__ uint8_t d_ginResourceSharingMode = (uint8_t)NCCL_GIN_RESOURCE_SHARING_THREAD;

// Mode the GIN tester kernel bodies pass to the ncclGin constructor.
static __device__ __forceinline__ ncclGinResourceSharingMode ginResourceSharingModeDev() {
  return (ncclGinResourceSharingMode)d_ginResourceSharingMode;
}

// Map the CLI value (0=thread, 1=CTA, 2=device) to the enum (GPU=0, CTA=1,
// THREAD=2 -- deliberately a different numbering, so the mapping is explicit).
static inline ncclGinResourceSharingMode ginResourceSharingModeFromCli(int cli) {
  switch (cli) {
    case 1:  return NCCL_GIN_RESOURCE_SHARING_CTA;
    case 2:  return NCCL_GIN_RESOURCE_SHARING_GPU;
    default: return NCCL_GIN_RESOURCE_SHARING_THREAD;
  }
}

// Enqueue the selected mode into the current device's d_ginResourceSharingMode.
// Call after cudaSetDevice() and before a GIN launch. Notes:
//  - Once per device: the value is a fixed CLI flag, so later calls are no-ops.
//    Keep it off the per-iteration path (a blocking copy there inflates latency).
//  - ASYNC on `stream`: a sync copy is illegal under -G >= 1 graph capture; the
//    same-stream enqueue stays capturable and orders before the kernel.
//  - Persistent host source (static, not stack): the copy may run at graph-launch,
//    after this returns.
//  - Thread races (-t N) only re-write an identical byte, so no locking needed.
#define GIN_RESOURCE_SHARING_MAX_DEVICES 64
static inline testResult_t uploadGinResourceSharingMode(cudaStream_t stream) {
  static int8_t uploaded[GIN_RESOURCE_SHARING_MAX_DEVICES] = {0};  // -1 once uploaded, per device
  static uint8_t hostMode[GIN_RESOURCE_SHARING_MAX_DEVICES] = {0};  // persistent async copy source
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= GIN_RESOURCE_SHARING_MAX_DEVICES) dev = 0;
  if (uploaded[dev]) return testSuccess;
  hostMode[dev] = (uint8_t)ginResourceSharingModeFromCli(ginResourceSharingMode);
  CUDACHECK(hipMemcpyToSymbolAsync(d_ginResourceSharingMode, &hostMode[dev], sizeof(hostMode[dev]), 0,
                                   hipMemcpyHostToDevice, stream));
  uploaded[dev] = -1;
  return testSuccess;
}

#endif  // GIN_RESOURCE_SHARING_H_
