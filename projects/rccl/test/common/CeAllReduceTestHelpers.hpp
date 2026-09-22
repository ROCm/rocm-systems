/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <algorithm>
#include <cstring>

#include <hip/hip_runtime.h>

#include "comm.h"
#include "ce_coll.h"
#include "nccl.h"

namespace RcclUnitTesting
{

// Runtime driver-version gate mirroring ncclCeImplemented().
inline bool isCeRuntimeDriverSupported()
{
    int driverVer = 0;
    if(hipDriverGetVersion(&driverVer) != hipSuccess)
        return false;
    return (driverVer >= 71200000) ||
           (driverVer >= 70051831 && driverVer < 70060000);
}

// Default ceARTmpBuf capacity. Matches NCCL_CE_AR_TMPBUF_DEFAULT_BYTES.
// Per-rank chunk capacity is that size / nRanks, same as ncclCeInit.
constexpr size_t kCeArMaxMsgBytesDefault = NCCL_CE_AR_TMPBUF_DEFAULT_BYTES;

inline size_t ceAllReduceMaxChunkBytes(int nRanks,
                                       size_t ceArMaxBytes = kCeArMaxMsgBytesDefault)
{
    return ceArMaxBytes / static_cast<size_t>(nRanks);
}

// Minimal ncclComm stand-in for CE AllReduce eligibility unit tests.
struct CeAllReduceMockComm
{
    ncclComm comm{};

    CeAllReduceMockComm() { reset(); }

    void reset()
    {
        std::memset(&comm, 0, sizeof(comm));
        comm.nNodes           = 1;
        comm.nRanks           = 4;
        comm.rank             = 0;
        comm.symmetricSupport = true;
        comm.config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
        // Keep mock LSA state initialized so eligibility does not inspect absent topology.
        comm.devrState.bigSize = 1;
        comm.devrState.lsaSize = comm.nRanks;
        comm.devrState.lsaSelf = comm.rank;
        comm.ceColl.ceArMaxBytes  = kCeArMaxMsgBytesDefault;
    }

    ncclComm* get() { return &comm; }
};

} // namespace RcclUnitTesting

