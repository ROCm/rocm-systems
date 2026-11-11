/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef LL128_LAYOUT_H_
#define LL128_LAYOUT_H_

#include <hip/hip_runtime.h>

// Dense 16-byte-chunk index for register pair `g` of lane `wid`. The LL128 wire
// format interleaves a flag word into every line, so the payload slots are not
// contiguous in register order; this compensates for those holes to keep the
// packed user buffer gap-free and coalesced.
//
// Templated on the line geometry rather than reading NCCL_LL128_LINEELEMS /
// WARP_SIZE directly so it stays usable from host code: the unit tests compute
// the expected layout on the CPU, and the TDM store leg's correctness is
// precisely the claim that these indices densely pack [0, DataBytes).
template <int LineElems, int WarpSz>
__host__ __device__ __forceinline__ constexpr int ll128SlotIx(int g, int wid) {
  constexpr int LineSkip = 2 * WarpSz / LineElems;
  return g * WarpSz - LineSkip * (g / 2) + wid - (g % 2) * (wid / (LineElems / 2));
}

#endif  // LL128_LAYOUT_H_
