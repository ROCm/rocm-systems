/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <cstring>

#include "comm.h"

namespace RcclUnitTesting
{

// Minimal ncclComm for GIN AllReduce eligibility early-outs that run *before*
// isSymmetricKernelRequested(). Do not set symmetricSupport=true with gfx950,
// ncclSum, and a supported datatype together: that path calls ncclSymkInitOnce
// on this stand-in and is not safe.
struct GinAllReduceMockComm
{
    ncclComm comm{};
    char     archName_[16]{};

    GinAllReduceMockComm() { reset(); }

    void reset()
    {
        std::memset(&comm, 0, sizeof(comm));
        std::memset(archName_, 0, sizeof(archName_));
        std::memcpy(archName_, "gfx950", 6);
        comm.archName         = archName_;
        comm.nNodes           = 1;
        comm.nRanks           = 8;
        comm.rank             = 0;
        comm.symmetricSupport = false;
    }

    void setArch(const char* arch)
    {
        std::memset(archName_, 0, sizeof(archName_));
        std::strncpy(archName_, arch, sizeof(archName_) - 1);
    }

    ncclComm* get() { return &comm; }
};

} // namespace RcclUnitTesting
