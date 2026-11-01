/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Link floor for the GIN Anvil-SDMA fabric AllToAll lane, for the binaries that
// compile the real src/gin/gin_host.cc (rccl-UnitTestsMicroGinHost and
// rccl-UnitTestsMicroGinHostFinalize). No test there has DDA fabric memory, so
// the gate reports false and ncclGinDevCommSetup never publishes a lane; erase
// on DevCommFree and Finalize is then a no-op on an empty table.

#include "algorithms/dda/fabric/fabric_init.h"
#include "gin/gin_fabric_a2a_host.h"

// src/algorithms/dda/fabric/fabric_init.cu
bool ginAnvilUseFabricMem(struct ncclComm*) { return false; }

// src/gin/gin_fabric_a2a_lane.cc
void ncclGinFabricA2ALanePublish(void*, ncclGinFabricA2ALane const&) {}
void ncclGinFabricA2ALaneErase(void*) {}
