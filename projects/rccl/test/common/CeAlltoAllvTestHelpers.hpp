/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <cstring>

#include <hip/hip_runtime.h>

#include "comm.h"
#include "nccl.h"

namespace RcclUnitTesting
{

// Metadata layout mirrors ce_coll.cc / collectives.cc gathered AlltoAllv sizes:
// per-rank block of [sendSizes, sendDispls, recvSizes, recvDispls] x nRanks (bytes).
inline size_t ceAlltoAllvMetaBlockOffset(int rank, int nRanks)
{
    return static_cast<size_t>(rank) * 4u * static_cast<size_t>(nRanks);
}

inline size_t* ceAlltoAllvSendSizes(size_t* gathered, int myRank, int nRanks)
{
    return gathered + ceAlltoAllvMetaBlockOffset(myRank, nRanks);
}

inline size_t* ceAlltoAllvSendDispls(size_t* gathered, int myRank, int nRanks)
{
    return ceAlltoAllvSendSizes(gathered, myRank, nRanks) + nRanks;
}

inline size_t* ceAlltoAllvRecvSizes(size_t* gathered, int myRank, int nRanks)
{
    return ceAlltoAllvSendDispls(gathered, myRank, nRanks) + nRanks;
}

inline size_t* ceAlltoAllvRecvDispls(size_t* gathered, int myRank, int nRanks)
{
    return ceAlltoAllvRecvSizes(gathered, myRank, nRanks) + nRanks;
}

// Pack local size metadata the same way ncclAlltoAllv_impl() does before exchange.
inline void packLocalAlltoAllvSizes(size_t* sizes,
                                    int nRanks,
                                    const size_t* sendcounts,
                                    const size_t* sdispls,
                                    const size_t* recvcounts,
                                    const size_t* rdispls)
{
    for (int i = 0; i < nRanks; ++i)
    {
        sizes[i]              = sendcounts[i];
        sizes[nRanks + i]     = sdispls[i];
        sizes[2 * nRanks + i] = recvcounts[i];
        sizes[3 * nRanks + i] = rdispls[i];
    }
}

// Traffic bytes for a rank: sum of outgoing send sizes in that rank's metadata block.
inline size_t ceAlltoAllvTrafficBytes(const size_t* gathered, int rank, int nRanks)
{
    size_t bytes = 0;
    const size_t* sendSizes = ceAlltoAllvSendSizes(const_cast<size_t*>(gathered), rank, nRanks);
    for (int r = 0; r < nRanks; ++r)
        bytes += sendSizes[r];
    return bytes;
}

// Runtime driver-version gate mirroring ncclCeImplemented().
inline bool isCeRuntimeDriverSupported()
{
    int driverVer = 0;
    if (hipDriverGetVersion(&driverVer) != hipSuccess)
        return false;
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
    }

    ncclComm* get() { return &comm; }
};

} // namespace RcclUnitTesting
