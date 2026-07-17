/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

 #pragma once

 #include <algorithm>
 #include <cstring>
 
 #include <hip/hip_runtime.h>
 
 #include "comm.h"
 #include "ce_coll.h"
 #include "nccl.h"
 
 namespace RcclUnitTesting
 {
 
 // Runtime driver-version gate mirroring ncclCeImplemented().
 inline bool isCeRuntimeDriverSupported()
 {
     int driverVer = 0;
     if(hipDriverGetVersion(&driverVer) != hipSuccess)
         return false;
     return (driverVer >= 71200000) ||
            (driverVer >= 70051831 && driverVer < 70060000);
 }
 
 // Mirrors chooseChunkBytes() in ce_coll.cc.
 inline size_t ceAllReduceChooseChunkBytes(size_t shardBytes, size_t maxChunkBytes)
 {
     const size_t kMinChunkBytes = 4 * 1024 * 1024ULL;
     const size_t kMaxChunkBytes = 256 * 1024 * 1024ULL;
     if(shardBytes <= kMinChunkBytes)
         return shardBytes;
     size_t targetChunkBytes = shardBytes / 4;
     targetChunkBytes        = std::min(targetChunkBytes, kMaxChunkBytes);
     targetChunkBytes        = std::max(targetChunkBytes, kMinChunkBytes);
     targetChunkBytes        = std::min(targetChunkBytes, maxChunkBytes);
     return targetChunkBytes;
 }
 
 inline size_t ceAllReduceMaxChunkBytes(int nRanks)
 {
     return (NCCL_CE_AR_MAX_MSG_BYTES / 2) / static_cast<size_t>(nRanks);
 }
 
 // Mirrors ncclCeAllReduce() chunking math (per-rank shard layout).
 inline void ceAllReduceComputeChunking(size_t count,
                                        size_t eltSize,
                                        int nRanks,
                                        size_t& baseChunkElems,
                                        size_t& tailChunkElems,
                                        size_t& chunksPerShard,
                                        size_t& slotChunkElems)
 {
     const size_t shardElems    = count / static_cast<size_t>(nRanks);
     const size_t shardBytes    = shardElems * eltSize;
     const size_t maxChunkBytes = ceAllReduceMaxChunkBytes(nRanks);
     const size_t chunkBytes    = ceAllReduceChooseChunkBytes(shardBytes, maxChunkBytes);
 
     baseChunkElems  = chunkBytes / eltSize;
     slotChunkElems  = maxChunkBytes / eltSize;
     chunksPerShard  = shardElems / baseChunkElems;
     tailChunkElems  = shardElems % baseChunkElems;
     if(tailChunkElems != 0)
         ++chunksPerShard;
 }
 
 // Minimal ncclComm stand-in for CE AllReduce eligibility unit tests.
 struct CeAllReduceMockComm
 {
     ncclComm comm{};
 
     CeAllReduceMockComm() { reset(); }
 
     void reset()
     {
         std::memset(&comm, 0, sizeof(comm));
         comm.nNodes           = 1;
         comm.nRanks           = 4;
         comm.rank             = 0;
         comm.symmetricSupport = true;
         comm.config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
     }
 
     ncclComm* get() { return &comm; }
 };
 
 } // namespace RcclUnitTesting
 
