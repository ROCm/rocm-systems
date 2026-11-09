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

// Write the selected mode into the current device's d_ginResourceSharingMode.
// Call once per device at the top of the tester's RunTest, before the benchmark
// loop (synchronous, so outside graph capture).
static inline testResult_t uploadGinResourceSharingMode() {
  uint8_t v = (uint8_t)ginResourceSharingModeFromCli(ginResourceSharingMode);
  CUDACHECK(hipMemcpyToSymbol(d_ginResourceSharingMode, &v, sizeof(v), 0, hipMemcpyHostToDevice));
  return testSuccess;
}

#endif  // GIN_RESOURCE_SHARING_H_
