/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

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

// The chunk-size helpers live in ce_coll.h (ncclCeAllReduceMaxChunkBytes,
// ncclCeAllReduceSlotChunkBytes, ncclCeAllReduceChooseChunkBytes). CE ReduceScatter
// reuses that staging layout, so these tests exercise the same code
// ncclCeReduceScatter() uses instead of a copy that can drift.

// 2-shot size cap rcclUseCeReduceScatter() applies when the comm has no arch
// table (rcclCeAr2ShotMax fallback), which is the case for the mock comm below.
constexpr size_t kCeRsMaxMsgBytesDefault = NCCL_CE_AR_TMPBUF_DEFAULT_BYTES;

// Minimal ncclComm stand-in for CE ReduceScatter eligibility unit tests.
struct CeReduceScatterMockComm
{
    ncclComm comm{};

    CeReduceScatterMockComm() { reset(); }

    void reset()
    {
        std::memset(&comm, 0, sizeof(comm));
        comm.nNodes           = 1;
        comm.nRanks           = 4;
        comm.rank             = 0;
        comm.symmetricSupport = true;
        comm.config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
    }

    ncclComm* get() { return &comm; }
};

} // namespace RcclUnitTesting
