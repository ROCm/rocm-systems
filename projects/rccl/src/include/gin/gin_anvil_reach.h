/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _GIN_ANVIL_REACH_H_
#define _GIN_ANVIL_REACH_H_

struct ncclComm;

// Readers of comm state, defined in src/gin/gin_anvil_reach.cc.

// True when comm could register Anvil SDMA windows through exported fabric VMM
// handles: MI450 with MNNVL, one clique covering comm, and cuMem. Depends on
// topology only, so every rank of comm gets the same answer.
bool ginAnvilFabricMemCapable(struct ncclComm* comm);

// ginAnvilFabricMemCapable plus this rank's RCCL_GIN_ANVIL_FABRIC_MEM=1 opt-in.
// The opt-in can differ between ranks, so the plugin agrees on it at connect.
bool ginAnvilUseFabricMem(struct ncclComm* comm);

#endif  // _GIN_ANVIL_REACH_H_
