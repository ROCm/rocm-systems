/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL-protocol all-to-all device kernel for the DDA fabric path. Each 16-byte
 * line holds 8 bytes of payload and two 4-byte flags; the flags carry cross-rank
 * sync, so no GPU barrier is needed. Staging uses the DDA scratch
 * (comm->ddaScratch, reachable via comm->ddaPeerPtrsDev).
 *
 * All-to-all is the "personalized" analogue of all-gather: instead of sending
 * one chunk to every peer, each rank sends a distinct chunk (sendbuff[peer]) to
 * each peer, and recvbuff[src] receives the chunk sent by rank src. The scatter
 * source and self-copy source are therefore per-peer offsets into sendbuff;
 * the gather/poll is identical to all-gather.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "nccl_device/gin/anvil_sdma/gin_fabric_ll_a2a_device.h"

namespace dda::common {

// Host-initiated RCCL entry point. Device-API kernels call the same body
// directly, avoiding device-side dynamic parallelism.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllToAllFabricLL(T* const* __restrict__ peerScratch,
                                    T* __restrict__ recvbuff,
                                    const T* __restrict__ sendbuff,
                                    size_t perChunkBytes,
                                    int selfRank, int nRanksRt,
                                    uint32_t* __restrict__ epochDev,
                                    int epochLen) {
  ddaAllToAllFabricLLBody<T, NRANKS_CT>(peerScratch, recvbuff, sendbuff, perChunkBytes,
                                        selfRank, nRanksRt, epochDev, epochLen);
}

} // namespace dda::common
