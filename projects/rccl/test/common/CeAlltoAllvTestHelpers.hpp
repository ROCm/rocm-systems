/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_COMMON_CE_ALLTOALLV_TEST_HELPERS_HPP
#define RCCL_TEST_COMMON_CE_ALLTOALLV_TEST_HELPERS_HPP

#include <cstring>

#include <hip/hip_runtime.h>

#include "alltoallv_meta.h"
#include "comm.h"
#include "nccl.h"

namespace RcclUnitTesting
{

// Runtime driver-version gate mirroring ncclCeImplemented().
inline bool isCeRuntimeDriverSupported()
{
    int driverVer = 0;
    if (hipDriverGetVersion(&driverVer) != hipSuccess)
    {
        return false;
    }
    return (driverVer >= 71200000) ||
           (driverVer >= 70051831 && driverVer < 70060000);
}

// Minimal ncclComm stand-in for CE AlltoAllv eligibility unit tests.
struct CeAlltoAllvMockComm
{
    ncclComm comm{};

    CeAlltoAllvMockComm() { reset(); }

    void reset()
    {
        std::memset(&comm, 0, sizeof(comm));
        comm.nNodes            = 1;
        comm.nRanks            = 4;
        comm.rank              = 0;
        comm.symmetricSupport  = true;
        comm.config.CTAPolicy  = NCCL_CTA_POLICY_ZERO;
    }

    ncclComm* get() { return &comm; }
};

} // namespace RcclUnitTesting

#endif  // RCCL_TEST_COMMON_CE_ALLTOALLV_TEST_HELPERS_HPP
