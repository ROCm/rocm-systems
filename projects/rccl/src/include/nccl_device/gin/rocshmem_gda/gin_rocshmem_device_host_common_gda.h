/*************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ROCSHMEM_DEVICE_HOST_COMMON_GDA_H_
#define _NCCL_DEVICE_GIN_ROCSHMEM_DEVICE_HOST_COMMON_GDA_H_

#include <stdint.h>

#define NCCL_GIN_ROCSHMEM_VERSION 100

namespace rocshmem { class QueuePair; }

struct ncclGinRocshmemGdaGPUContext {
  rocshmem::QueuePair** qps;        // Array of nRanks QP pointers (GPU-accessible)
  uint64_t* signals;                // GPU-allocated signal array
  uint64_t* counters;               // GPU-allocated counter array
  uint32_t* signal_rkeys;           // Per-peer rkeys for signal memory
  uintptr_t* signal_raddrs;         // Per-peer remote base addresses for signals
  uint32_t nSignals;
  uint32_t nCounters;
  int nRanks;
  int rank;
};

struct ncclGinRocshmemGdaMemHandle {
  uintptr_t local_va;    // This rank's VA of the registered buffer (for src laddr)
  uintptr_t *remote_vas; // Per-peer VAs (GPU array, nRanks entries, allgathered)
  uint32_t lkey;          // Local key for this MR
  uint32_t *rkeys;        // Per-peer rkeys (GPU array, nRanks entries, allgathered)
};

#endif // _NCCL_DEVICE_GIN_ROCSHMEM_DEVICE_HOST_COMMON_GDA_H_
