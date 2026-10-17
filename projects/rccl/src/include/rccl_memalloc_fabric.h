/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef RCCL_MEMALLOC_FABRIC_H_
#define RCCL_MEMALLOC_FABRIC_H_

#include <cstdint>

// RCCL_MEMALLOC_FABRIC (ncclMemAlloc, allocator.cc): whether a buffer that would get the stock
// POSIX_FD handle type is allocated with a FABRIC handle instead.
//   mode 1: force (if the device supports FABRIC handles), 0: off (stock), < 0 (default): auto.
// Auto upgrades only where the FABRIC-only buffer is the one that works: gfx1250, whose UALoE
// MNNVL cliques span trays (a POSIX fd cannot reach a peer on another host), the device supports
// FABRIC handles, and NCCL_MNNVL_ENABLE != 0. Everywhere else ncclMemAlloc is unchanged: HIP takes
// exactly one requested handle type, so a FABRIC buffer has no POSIX fd export and a non-MNNVL
// comm could not register it.
inline bool rcclMemAllocWantFabric(int64_t mode, int64_t mnnvlEnable, bool fabricSupported, bool isGfx1250) {
  if (!fabricSupported || mode == 0) return false;
  if (mode > 0) return true;
  return mnnvlEnable != 0 && isGfx1250;
}

#endif  // RCCL_MEMALLOC_FABRIC_H_
