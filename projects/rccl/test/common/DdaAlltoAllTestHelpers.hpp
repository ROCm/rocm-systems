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

// Mirrors collectives.cc rcclDdaEnabled() for unit tests only.
constexpr size_t kDdaAlltoAllGfx942ThresholdBytes = 4194304;
constexpr size_t kDdaAlltoAllGfx950ThresholdBytes = 4194304;

inline bool testRcclDdaEnabled(
    const ncclComm* comm,
    size_t totalBytes,
    size_t gfx942Default,
    size_t gfx950Default = 0) {
  if (!rcclParamDdaEnable() || ncclParamLaunchOrderImplicit() || ncclGroupDepth != 0 ||
      comm->nRanks < 8 || comm->symmetricSupport) {
    return false;
  }
  size_t threshold;
  if (IsArchMatch(comm->archName, "gfx942")) {
    threshold = gfx942Default;
  } else if (IsArchMatch(comm->archName, "gfx950")) {
    threshold = gfx950Default ? gfx950Default : static_cast<size_t>(rcclParamDdaThreshold());
  } else {
    return false;
  }
  return threshold > 0 && totalBytes <= threshold;
}

inline size_t testAlltoAllTotalBytes(size_t count, int nRanks, ncclDataType_t datatype) {
  return static_cast<size_t>(nRanks) * count * static_cast<size_t>(ncclTypeSize(datatype));
}

inline bool testRcclDdaAlltoAllThresholdEnabled(
    const ncclComm* comm,
    size_t count,
    ncclDataType_t datatype) {
  return testRcclDdaEnabled(
      comm,
      testAlltoAllTotalBytes(count, comm->nRanks, datatype),
      kDdaAlltoAllGfx942ThresholdBytes,
      kDdaAlltoAllGfx950ThresholdBytes);
}

inline size_t testAlltoAllDdaIpcStagingBytes(size_t count, int nRanks, size_t typeSize) {
  return count * static_cast<size_t>(nRanks) * typeSize;
}

struct DdaAlltoAllMockComm
{
    ncclComm comm{};

    DdaAlltoAllMockComm() { reset("gfx950:sramecc+:xnack-"); }

    void reset(const char* archName)
    {
        std::memset(&comm, 0, sizeof(comm));
        std::strncpy(comm.archName, archName, sizeof(comm.archName) - 1);
        comm.archName[sizeof(comm.archName) - 1] = '\0';
        comm.nNodes = 1;
        comm.nRanks = nccl_dda_detail::kDdaNranks;
        comm.symmetricSupport = 0;
    }

    ncclComm* get() { return &comm; }
};

// Largest float32 per-rank count whose 8-rank AlltoAll totals exactly 4 MiB.
constexpr size_t kAlltoAllFloat32CountAt4MbThreshold =
    kDdaAlltoAllGfx950ThresholdBytes /
    (static_cast<size_t>(nccl_dda_detail::kDdaNranks) * sizeof(float));

} // namespace RcclUnitTesting
