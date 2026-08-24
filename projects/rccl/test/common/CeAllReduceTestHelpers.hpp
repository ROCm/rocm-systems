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

// Staging cap used when no RCCL_CE_AR_MAX_MSG_BYTES override is set. Matches
// rcclArchThresholds::ceArMax in graph/tuning.cc (256 MiB). Per-rank chunk
// capacity is that cap / nRanks, same as ncclCeInit.
constexpr size_t kCeArMaxMsgBytesDefault = 256ull * 1024 * 1024;

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
        comm.nNodes               = 1;
        comm.nRanks               = 4;
        comm.rank                 = 0;
        comm.symmetricSupport     = true;
        comm.config.CTAPolicy     = NCCL_CTA_POLICY_ZERO;
        comm.ceColl.ceArMaxBytes  = kCeArMaxMsgBytesDefault;
    }

    ncclComm* get() { return &comm; }
};

} // namespace RcclUnitTesting

