/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#ifndef RCCL_CPU_MEM_H_
#define RCCL_CPU_MEM_H_

#include "checks.h"

#include <cstddef>

// Returns a host pointer suitable for load/store. Device-only pointers are
// copied into thread-local staging (synced on demand).
ncclResult_t rcclCpuMapDevicePtr(void* devicePtr, size_t bytes, void** hostPtr, bool* needsUnmap);
void rcclCpuUnmapDevicePtr(void* devicePtr, void* hostPtr, bool needsUnmap);

#endif
