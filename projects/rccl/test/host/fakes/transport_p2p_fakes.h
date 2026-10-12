/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// src/transport/p2p.cc shareable buffers; never link with the real p2p.cc.

#ifndef RCCL_TEST_HOST_FAKES_TRANSPORT_P2P_FAKES_H_
#define RCCL_TEST_HOST_FAKES_TRANSPORT_P2P_FAKES_H_

#include <cstddef>
#include <functional>

#include "nccl.h"
#include "p2p.h"

// Default: host calloc as the device buffer, its address in a zeroed *ipcDesc; the caller owns and frees it.
extern std::function<ncclResult_t(size_t /*size*/, int /*directMap*/, ncclIpcDesc* /*ipcDesc*/, void** /*ptr*/,
                                  int /*peerRank*/, struct ncclMemManager* /*manager*/, ncclMemType_t /*memtype*/)>
    g_ncclP2pAllocateShareableBuffer;

// Default: one host address space, so the import maps ownerPtr, and only if *ipcDesc starts with that address.
extern std::function<ncclResult_t(struct ncclComm* /*comm*/, int /*peer*/, size_t /*size*/,
                                  ncclIpcDesc* /*ipcDesc*/, void** /*devMemPtr*/, void* /*ownerPtr*/,
                                  ncclMemType_t /*memType*/)>
    g_ncclP2pImportShareableBuffer;

void ResetTransportP2pFakes();

#endif  // RCCL_TEST_HOST_FAKES_TRANSPORT_P2P_FAKES_H_
