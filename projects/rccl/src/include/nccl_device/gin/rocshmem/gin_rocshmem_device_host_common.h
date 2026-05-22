/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ROCSHMEM_DEVICE_HOST_COMMON_H_
#define _NCCL_DEVICE_GIN_ROCSHMEM_DEVICE_HOST_COMMON_H_

#include <stdint.h>

#define NCCL_GIN_ROCSHMEM_VERSION 100

struct ncclGinRocshmemGPUContext {
  uint64_t* signals;      // Symmetric signal array (rocshmem_malloc'd)
  uint64_t* counters;     // Counter array (hipMalloc'd, local only)
  uint32_t nSignals;
  uint32_t nCounters;
  int nRanks;
  int rank;
};

struct ncclGinRocshmemMemHandle {
  uintptr_t baseAddr;     // VA of the registered buffer
};

#endif // _NCCL_DEVICE_GIN_ROCSHMEM_DEVICE_HOST_COMMON_H_
