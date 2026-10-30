/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <cerrno>
#include <cstdlib>
#include <string>

// C++ linkage, matching rocmwrap.h / librccl.so.
extern int ncclCuMemRuntimeSupported();

namespace RCCLTestGuards {

// Returns a skip reason when NCCL_CUMEM_ENABLE requests symmetric memory but the
// current kernel/runtime cannot provide cuMem (dma-buf / P2PDMA). Empty => OK.
inline std::string symmetricMemEnvAndRuntimeSkipReason() {
  const char* cumem = std::getenv("NCCL_CUMEM_ENABLE");
  if (!cumem || cumem[0] == '\0')
    return "Symmetric memory required (set NCCL_CUMEM_ENABLE to a non-zero value)";
  errno = 0;
  if (std::strtoll(cumem, nullptr, 0) == 0 && errno == 0)
    return "Symmetric memory required (NCCL_CUMEM_ENABLE must be non-zero)";
  if (!ncclCuMemRuntimeSupported())
    return "Symmetric memory not supported on this platform (cuMem requires kernel dma-buf/P2PDMA support)";
  return "";
}

}  // namespace RCCLTestGuards
