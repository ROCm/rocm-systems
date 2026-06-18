/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <cstring>

#include "comm.h"
#include "ipc_init_detail.h"

namespace RcclUnitTesting
{

// Minimal ncclComm stand-in for DDA IPC eligibility unit tests.
struct DdaIpcMockComm
{
    ncclComm comm{};
    char     bootstrapPlaceholder{0};

    DdaIpcMockComm() { reset(); }

    void reset()
    {
        std::memset(&comm, 0, sizeof(comm));
        comm.bootstrap           = &bootstrapPlaceholder;
        comm.nNodes              = 1;
        comm.nRanks              = nccl_dda_ipc_detail::kDdaNranks;
        comm.ddaIpcScratchBytes  = DDA_IPC_BUFFER_SIZE;
        setIpcResourcesPresent(true);
    }

    void setIpcResourcesPresent(bool present)
    {
        if (present)
        {
            comm.ddaIpcMemHandler =
                reinterpret_cast<ncclIpcMemHandler*>(0x1);
            comm.ddaIpcScratch       = reinterpret_cast<void*>(0x2);
            comm.ddaIpcPeerPtrsDev   = reinterpret_cast<void*>(0x3);
            comm.ddaIpcBarrierState  =
                reinterpret_cast<nccl_dda_ipc_detail::DdaIpcBarrierState*>(0x4);
        }
        else
        {
            comm.ddaIpcMemHandler   = nullptr;
            comm.ddaIpcScratch      = nullptr;
            comm.ddaIpcPeerPtrsDev  = nullptr;
            comm.ddaIpcBarrierState = nullptr;
        }
    }

    ncclComm* get() { return &comm; }
};

} // namespace RcclUnitTesting
