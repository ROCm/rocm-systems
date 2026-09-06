/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "nccl.h"
#include "alloc.h"

struct ncclComm;

// Returns true when the DDA fabric/VMM path should be used for this comm,
// false to use the legacy IPC path
bool ncclDdaUseFabricPath(struct ncclComm* comm);

// Returns true when the Anvil SDMA GIN plugin should use fabric DDA peer
// memory for this comm (MI455 single-clique path).
bool ginAnvilUseFabricMem(struct ncclComm* comm);

ncclResult_t ncclDdaFabricCommInit(struct ncclComm* comm);
ncclResult_t ncclDdaFabricCommFini(struct ncclComm* comm);
