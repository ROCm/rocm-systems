/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Real implementations of the src/os/linux.cc allocation shims. Not stubs: the
// host-only microtests want working aligned allocation, and posix_memalign is
// exactly what production uses.
//
// The other src/os/*.cc entry points (ncclOsCpuCount, ncclOsSetAffinity,
// ncclOsTopoGetStrFromSys) are still split between collective_stubs.cc and
// nccl_stubs.cc. That predates this file and is left alone here rather than
// widening the change; consolidate them when one of those targets next moves.

#include <cstdlib>

#include "os.h"

void* ncclOsAlignedAlloc(size_t alignment, size_t size) {
  void* p = nullptr;
  if (posix_memalign(&p, alignment, size) != 0) return nullptr;
  return p;
}

void ncclOsAlignedFree(void* ptr) { free(ptr); }
