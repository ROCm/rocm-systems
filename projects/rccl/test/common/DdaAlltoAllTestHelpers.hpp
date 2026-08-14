/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <cstring>

#include "archinfo.h"
#include "collectives.h"
#include "comm.h"
#include "enqueue.h"
#include "group.h"
#include "dda_init_detail.h"
#include "rccl_common.h"

namespace RcclUnitTesting
{

// Use the production rcclDdaEnabled() from rccl_common.h directly.
// Threshold constants kDdaAlltoAllGfx{942,950,1250}ThresholdBytes are also in rccl_common.h.

inline size_t testAlltoAllTotalBytes(size_t count, int nRanks, ncclDataType_t datatype) {
  return static_cast<size_t>(nRanks) * count * static_cast<size_t>(ncclTypeSize(datatype));
}

inline bool testRcclDdaAlltoAllThresholdEnabled(
    const ncclComm* comm,
    size_t count,
    ncclDataType_t datatype) {
  return rcclDdaEnabled(
      comm,
      testAlltoAllTotalBytes(count, comm->nRanks, datatype),
      kDdaAlltoAllGfx942ThresholdBytes,
      kDdaAlltoAllGfx950ThresholdBytes,
      kDdaAlltoAllGfx1250ThresholdBytes);
}

inline size_t testAlltoAllDdaIpcStagingBytes(size_t count, int nRanks, size_t typeSize) {
  return count * static_cast<size_t>(nRanks) * typeSize;
}

struct DdaAlltoAllMockComm
{
    ncclComm comm{};
    char archNameBuf[64]{};

    DdaAlltoAllMockComm() { reset("gfx950:sramecc+:xnack-"); }

    void reset(const char* archName)
    {
        std::memset(&comm, 0, sizeof(comm));
        std::strncpy(archNameBuf, archName, sizeof(archNameBuf) - 1);
        archNameBuf[sizeof(archNameBuf) - 1] = '\0';
        comm.archName = archNameBuf;
        comm.nNodes = 1;
        comm.nRanks = nccl_dda_detail::kDdaNranks;
        comm.symmetricSupport = 0;
    }

    ncclComm* get() { return &comm; }
};

// Largest float32 per-rank count whose 8-rank AlltoAll totals exactly 4 MiB.
constexpr size_t kAlltoAllFloat32CountAt4MbThreshold =
    kDdaAlltoAllGfx942ThresholdBytes /
    (static_cast<size_t>(nccl_dda_detail::kDdaNranks) * sizeof(float));

} // namespace RcclUnitTesting
