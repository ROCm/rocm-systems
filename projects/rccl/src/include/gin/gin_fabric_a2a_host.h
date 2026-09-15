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
// ncclSuccess + *allEnabled=1: every rank voted enabled.
// ncclSuccess + *allEnabled=0: allgather succeeded, at least one rank voted 0 (soft disable).
// Any other return: bootstrap/allgather failed; *allEnabled is 0. Callers must not treat
// that as "not eligible". A caller-side gate must not skip this call: it is a
// collective, so a rank that returns early hangs the peers blocked inside it.
ncclResult_t ncclGinFabricA2ALaneAgreeEnabled(struct ncclComm* comm, int localEnabled, int* allEnabled);
// Build a device peer-scratch table pointing at the GIN LL carve-out in ddaScratch.
ncclResult_t ginFabricA2ALaneBuildPeerScratchDev(struct ncclComm* comm, void*** outPeerDev, size_t* outRegionBytes);
#endif

#endif
