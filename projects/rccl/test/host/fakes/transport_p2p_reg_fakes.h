/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams for the src/transport/p2p.cc registration entry points, for a unit compiled in a binary that also
// compiles the real p2p.cc. transport_p2p_reg_seam.h renames the unit's calls to the micro_* trampolines below.
// Defaults fail loudly, except ncclP2pUsesMemcpy (false, as with NCCL_P2P_USE_CUDA_MEMCPY unset).

#ifndef RCCL_TEST_HOST_TRANSPORT_P2P_REG_FAKES_H_
#define RCCL_TEST_HOST_TRANSPORT_P2P_REG_FAKES_H_

#include <cstddef>
#include <cstdint>
#include <functional>

#include "nccl.h"
#include "p2p.h"

extern std::function<ncclResult_t(ncclComm*, const void*, size_t, int* /*peerRanks*/, int /*nPeers*/, ncclIpcRegType,
                                  int* /*regBufFlag*/, uintptr_t* /*offsetOut*/, uintptr_t** /*peerRmtAddrsOut*/)>
  g_ncclIpcLocalRegisterBuffer;
extern std::function<ncclResult_t(ncclComm*, const void*, size_t, int*, int, ncclIpcRegType, int*, uintptr_t*,
                                  uintptr_t**, void* /*cleanupQueuePtr*/, int* /*nCleanupQueueElts*/)>
  g_ncclIpcGraphRegisterBuffer;
extern std::function<ncclResult_t(struct ncclComm*, struct ncclIpcRegInfo*)> g_ncclIpcDeregBuffer;
extern std::function<bool()> g_ncclP2pUsesMemcpy;

ncclResult_t micro_ncclIpcLocalRegisterBuffer(ncclComm*, const void*, size_t, int*, int, ncclIpcRegType, int*,
                                              uintptr_t*, uintptr_t**);
ncclResult_t micro_ncclIpcGraphRegisterBuffer(ncclComm*, const void*, size_t, int*, int, ncclIpcRegType, int*,
                                              uintptr_t*, uintptr_t**, void*, int*);
ncclResult_t micro_ncclIpcDeregBuffer(struct ncclComm*, struct ncclIpcRegInfo*);
bool micro_ncclP2pUsesMemcpy();

void ResetTransportP2pRegFakes();

#endif  // RCCL_TEST_HOST_TRANSPORT_P2P_REG_FAKES_H_
