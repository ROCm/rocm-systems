/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Direct LSA-flat-from-fabric AllToAll device body for GIN-SDMA Test#5.
 * Requires nccl_device.h (or equivalent) before this include.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LSA_A2A_DEVICE_H_
#define _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LSA_A2A_DEVICE_H_

#include <cstddef>
#include <cstdint>

#include "gin_fabric_lsa_policy.h"

namespace gin {
namespace fabric {

__device__ __forceinline__ void ginFabricLsaA2ACopyCta(char* dst, const char* src, size_t bytes) {
  size_t nVec = bytes / sizeof(uint4);
  for (size_t i = threadIdx.x; i < nVec; i += blockDim.x) {
    ((uint4*)dst)[i] = ((const uint4*)src)[i];
  }
  for (size_t i = nVec * sizeof(uint4) + threadIdx.x; i < bytes; i += blockDim.x) {
    dst[i] = src[i];
  }
}

// 2D grid: blockIdx.x == peer, blockIdx.y == chunk. Extra CTAs from a padded
// host grid (gin.put / LL) return before the LSA barrier. Barrier indices are
// flattened as (chunk * nRanks + peer) so they stay in the LSA pool even when
// grid.x is larger than nRanks.
__device__ __forceinline__ void ginFabricLsaAlltoAllBody(ncclWindow_t sendWin, size_t sendOff, ncclWindow_t recvWin,
                                                         size_t recvOff, size_t bytesPerPeer,
                                                         struct ncclDevComm devComm) {
  int chunks = 0;
  int threads = 0;
  size_t chunkBytes = 0;
  ginFabricLsaA2ALaunchConfig(devComm.nRanks, bytesPerPeer, &chunks, &threads, &chunkBytes);
  (void)threads;
  if ((int)blockIdx.x >= devComm.nRanks || (int)blockIdx.y >= chunks) return;

  const unsigned int ctaIndex = (unsigned int)blockIdx.y * (unsigned int)devComm.nRanks + (unsigned int)blockIdx.x;
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), devComm, ncclTeamTagLsa(), ctaIndex};
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);

  const int peer = (int)blockIdx.x;
  const size_t off = chunkBytes * (size_t)blockIdx.y;
  if (off < bytesPerPeer) {
    size_t bytes = bytesPerPeer - off;
    if (bytes > chunkBytes) bytes = chunkBytes;
    char* dst = (char*)ncclGetLsaPointer(recvWin, recvOff + (size_t)devComm.rank * bytesPerPeer + off, peer);
    const char* src = (const char*)ncclGetLocalPointer(sendWin, sendOff + (size_t)peer * bytesPerPeer + off);
    ginFabricLsaA2ACopyCta(dst, src, bytes);
  }

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

}  // namespace fabric
}  // namespace gin

#endif  // _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LSA_A2A_DEVICE_H_
