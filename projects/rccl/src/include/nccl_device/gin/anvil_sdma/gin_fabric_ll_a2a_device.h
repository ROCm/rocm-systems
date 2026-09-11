/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL-protocol all-to-all device body for the GIN-SDMA fabric DDA path.
 * Installed under include/nccl_device/gin/anvil_sdma/.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LL_A2A_DEVICE_H_
#define _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LL_A2A_DEVICE_H_

#include "gin_fabric_ll_device_prims.h"

namespace dda {
namespace common {

constexpr size_t kDdaLLA2ASlotStridePkts = kDdaLLMaxBytes / sizeof(LLPacket16);

// 2D grid: grid.x == nRanks selects the peer; grid.y == blocksPerPeer splits
// that peer's packets. The self column copies locally; other columns scatter
// then poll.
template <typename T, int NRANKS_CT>
__device__ __forceinline__ void ddaAllToAllFabricLLBody(
    T* const* __restrict__ peerScratch, T* __restrict__ recvbuff, const T* __restrict__ sendbuff,
    size_t perChunkBytes, int selfRank, int nRanksRt, uint32_t* __restrict__ epochDev, int epochLen) {
  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const int peer = blockIdx.x;
  if (peer >= nRanks) return;
  const int chunk = blockIdx.y;
  const int nChunks = gridDim.y;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const size_t nPk = perChunkBytes >> 3;
  const size_t slot = kDdaLLA2ASlotStridePkts;

  const int flatBlockId = blockIdx.x * gridDim.y + blockIdx.y;
  const int total = gridDim.x * gridDim.y;
  const uint32_t flag = ddaGetLLEpochInc(epochDev, flatBlockId, 1);
  const size_t bankOffsetPkts = (size_t)(flag & 1u) * (size_t)nRanks * slot;

  const size_t pkPerChunk = (nPk + (size_t)nChunks - 1) / (size_t)nChunks;
  const size_t pkBegin = (size_t)chunk * pkPerChunk;
  size_t pkEnd = pkBegin + pkPerChunk;
  if (pkEnd > nPk) pkEnd = nPk;

  if (peer == selfRank) {
    const uint4* s4 =
      reinterpret_cast<const uint4*>(reinterpret_cast<const char*>(sendbuff) + (size_t)selfRank * perChunkBytes);
    uint4* d4 = reinterpret_cast<uint4*>(reinterpret_cast<char*>(recvbuff) + (size_t)selfRank * perChunkBytes);
    const size_t nVec = perChunkBytes >> 4;
    const size_t vecPerChunk = (nVec + (size_t)nChunks - 1) / (size_t)nChunks;
    const size_t vBegin = (size_t)chunk * vecPerChunk;
    size_t vEnd = vBegin + vecPerChunk;
    if (vEnd > nVec) vEnd = nVec;
    for (size_t i = vBegin + tid; i < vEnd; i += nthreads) {
      const uint4* p = &s4[i];
      uint4 v;
      v.x = __builtin_nontemporal_load(&p->x);
      v.y = __builtin_nontemporal_load(&p->y);
      v.z = __builtin_nontemporal_load(&p->z);
      v.w = __builtin_nontemporal_load(&p->w);
      uint4* q = &d4[i];
      __builtin_nontemporal_store(v.x, &q->x);
      __builtin_nontemporal_store(v.y, &q->y);
      __builtin_nontemporal_store(v.z, &q->z);
      __builtin_nontemporal_store(v.w, &q->w);
    }
  } else {
    const uint32_t* sw =
      reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(sendbuff) + (size_t)peer * perChunkBytes);
    LLPacket16* dst = reinterpret_cast<LLPacket16*>(peerScratch[peer]) + (size_t)selfRank * slot + bankOffsetPkts;
    for (size_t pk = pkBegin + tid; pk < pkEnd; pk += nthreads) {
      ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(&dst[pk]), sw[2 * pk], flag, sw[2 * pk + 1], flag);
    }

    volatile LLPacket16* src =
      reinterpret_cast<LLPacket16*>(peerScratch[selfRank]) + bankOffsetPkts + (size_t)peer * slot;
    uint32_t* out = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(recvbuff) + (size_t)peer * perChunkBytes);
    for (size_t pk = pkBegin + tid; pk < pkEnd; pk += nthreads) {
      uint32_t d0, f0, d1, f1;
      do {
        ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(const_cast<LLPacket16*>(&src[pk])), d0, f0, d1, f1);
      } while (f0 != flag || f1 != flag);
      out[2 * pk] = d0;
      out[2 * pk + 1] = d1;
    }
  }

  ddaSetLLEpoch(epochDev, epochLen, flatBlockId, total, flag);
}

} // namespace common
} // namespace dda

#endif
