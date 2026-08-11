/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_DDA_FABRIC_TEST_HELPERS_HPP
#define RCCL_TEST_DDA_FABRIC_TEST_HELPERS_HPP

#include <cstring>

#include "comm.h"
#include "dda_init_detail.h"
#include "fabric_gpu_barrier.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting
{

// Minimal ncclComm stand-in for DDA fabric (VMM) eligibility unit tests.
struct DdaFabricMockComm
{
    ncclComm comm{};
    char     bootstrapPlaceholder{0};

    DdaFabricMockComm() { reset(); }

    void reset()
    {
        std::memset(&comm, 0, sizeof(comm));
        comm.bootstrap          = &bootstrapPlaceholder;
        comm.nNodes             = 1;
        comm.nRanks             = 8; // any value in [2, kDdaMaxNranks]
        comm.ddaScratchBytes    = DDA_FABRIC_BUFFER_SIZE;
        comm.ddaFabricMaxBlocks = DDA_FABRIC_MAXBLOCKS;
        setFabricResourcesPresent(true);
    }

    void setFabricResourcesPresent(bool present)
    {
        if (present)
        {
            comm.ddaFabricMemHandler =
                reinterpret_cast<ncclFabricMemHandler*>(0x1);
            comm.ddaScratch     = reinterpret_cast<void*>(0x2);
            comm.ddaPeerPtrsDev = reinterpret_cast<void*>(0x3);
            comm.ddaFabricBarrierState =
                reinterpret_cast<nccl_dda_detail::DdaFabricBarrierState*>(0x4);
        }
        else
        {
            comm.ddaFabricMemHandler   = nullptr;
            comm.ddaScratch            = nullptr;
            comm.ddaPeerPtrsDev        = nullptr;
            comm.ddaFabricBarrierState = nullptr;
        }
    }

    ncclComm* get() { return &comm; }
};

// Shared fixture for DDA fabric host tests: a mock comm plus dummy user-buffer
// pointers (every fabric eligibility predicate ignores the buffer pointers).
class DdaFabricFixture : public ::testing::Test
{
protected:
    DdaFabricMockComm mockComm_;
    void*             sendbuff_{reinterpret_cast<void*>(0x1000)};
    void*             recvbuff_{reinterpret_cast<void*>(0x2000)};
};

} // namespace RcclUnitTesting

#endif // RCCL_TEST_DDA_FABRIC_TEST_HELPERS_HPP
