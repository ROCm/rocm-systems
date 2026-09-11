/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_GIN_FABRIC_A2A_LANE_H_
#define NCCL_GIN_FABRIC_A2A_LANE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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

#endif
