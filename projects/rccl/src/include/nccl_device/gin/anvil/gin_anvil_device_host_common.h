/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_DEVICE_HOST_COMMON_H_
#define _NCCL_DEVICE_GIN_ANVIL_DEVICE_HOST_COMMON_H_

#include <stdint.h>

#define NCCL_GIN_ANVIL_VERSION 100

// Device-resident context handle for Anvil-SDMA based GIN.
// - queues[peer] points to an SDMA queue handle for issuing ops to that peer.
// - signalsBase[peer] is peer P's native GPU VA for SDMA signal atomics (bootstrap exchange).
// - signals is this rank's indexed signal array (local readSignal / waitSignal).
struct ncclGinAnvilGPUContext {
  void** queues;              // device array [nRanks] of rocshmem::anvil::SdmaQueueDeviceHandle*
  uint64_t** signalsBase;     // device array [nRanks] of base pointers to peer signals
  uint64_t* signals;          // device pointer to this rank's indexed signal array
  uint64_t* counters;         // device pointer to local counters array
  uint32_t nSignals;
  uint32_t nCounters;
  int nRanks;
  int rank;
  int myNode;                 // rankToNode[rank]; peers on other nodes have null queues
  // Symmetric VA layout (LSA flat space)
  uintptr_t lsaRank0Base;     // base pointer for rank0 slice for a registered window
  uint64_t lsaStrideBytes;    // bytes between consecutive ranks in flat space (devr->bigSize)
};

// Device window handle for Anvil: stores the rank0 base in LSA flat space.
struct ncclGinAnvilMemHandle {
  uintptr_t lsaRank0Base;
  uint64_t lsaStrideBytes;
};

#endif // _NCCL_DEVICE_GIN_ANVIL_DEVICE_HOST_COMMON_H_

