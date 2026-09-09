/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only GIN Anvil-SDMA fabric AllToAll small-message lane. Not a device
// header and not part of struct ncclDevComm (keeps the public sizeof stable).

#ifndef NCCL_GIN_FABRIC_A2A_HOST_H_
#define NCCL_GIN_FABRIC_A2A_HOST_H_

#include "nccl.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ncclDevComm;

// 0-threshold means the lane is disabled (same encoding as gin_host setup).
struct ncclGinFabricA2ALane {
  int enabled;
  void** peerScratch;
  uint32_t* llEpoch;
  int llEpochLen;
  size_t scratchBytes;
  size_t llThreshold;
};

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
void ncclGinFabricA2ALanePublish(void* ginHandle, ncclGinFabricA2ALane const& lane);
void ncclGinFabricA2ALaneErase(void* ginHandle);
#endif

#ifdef __cplusplus
extern "C" {
#endif

ncclResult_t ncclGinQueryFabricA2ALane(struct ncclDevComm const* devComm, struct ncclGinFabricA2ALane* out);

#ifdef __cplusplus
}
#endif

#endif
