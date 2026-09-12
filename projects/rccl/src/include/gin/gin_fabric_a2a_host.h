/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only GIN Anvil-SDMA fabric AllToAll small-message lane. Not a device
// header and not part of struct ncclDevComm (keeps the public sizeof stable).

#ifndef NCCL_GIN_FABRIC_A2A_HOST_H_
#define NCCL_GIN_FABRIC_A2A_HOST_H_

#include <stddef.h>
#include <stdint.h>

#include "nccl.h"
#include "nccl_device/gin/anvil_sdma/gin_fabric_a2a_lane.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ncclDevComm;
struct ncclComm;

ncclResult_t ncclGinQueryFabricA2ALane(struct ncclDevComm const* devComm, struct ncclGinFabricA2ALane* out);

#ifdef __cplusplus
}

void ncclGinFabricA2ALanePublish(void* ginHandle, ncclGinFabricA2ALane const& lane);
void ncclGinFabricA2ALaneErase(void* ginHandle);
void ncclGinFabricA2ALaneClearAll();
// AND-reduce localEnabled across comm->bootstrap (comm->nRanks / comm->rank).
ncclResult_t ncclGinFabricA2ALaneAgreeEnabled(struct ncclComm* comm, int localEnabled, int* allEnabled);
#endif

#endif
