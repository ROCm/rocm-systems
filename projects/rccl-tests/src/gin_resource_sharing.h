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

// Ensure the current device's d_ginResourceSharingMode holds the selected mode.
// The value is fixed for the whole run (a CLI flag), so this does the synchronous
// hipMemcpyToSymbol at most ONCE per device and is a no-op on every later call --
// it must stay out of the per-iteration launch path (a blocking symbol copy there
// serializes the device and inflates small-message latency). Call after
// cudaSetDevice() and before a GIN launch. Not thread-safe across the -t N threads
// of one process, but they share the device and upload the same value, so a race
// only re-writes an identical byte.
#define GIN_RESOURCE_SHARING_MAX_DEVICES 64
static inline testResult_t uploadGinResourceSharingMode() {
  static int8_t uploaded[GIN_RESOURCE_SHARING_MAX_DEVICES] = {0};  // -1 once uploaded, per device
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= GIN_RESOURCE_SHARING_MAX_DEVICES) dev = 0;
  if (uploaded[dev]) return testSuccess;
  uint8_t v = (uint8_t)ginResourceSharingModeFromCli(ginResourceSharingMode);
  CUDACHECK(hipMemcpyToSymbol(d_ginResourceSharingMode, &v, sizeof(v), 0, hipMemcpyHostToDevice));
  uploaded[dev] = -1;
  return testSuccess;
}

#endif  // GIN_RESOURCE_SHARING_H_
