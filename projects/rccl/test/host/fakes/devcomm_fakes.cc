/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm_fakes.h"

#include <cstddef>
#include <cstring>

#include "comm.h"
#include "dev_runtime.h"

ncclTeam_t DefaultNcclTeamLsa(ncclComm_t comm) {
  ncclTeam_t t{};
  t.nRanks = comm ? comm->nRanks : 0;
  t.rank = comm ? comm->rank : 0;
  t.stride = 1;
  return t;
}

void DefaultNcclDevCommCopyLsaData(void* dst, void const* src) {
  std::memcpy(dst, src,
              offsetof(struct ncclDevComm, railGinBarrier) - offsetof(struct ncclDevComm, rank));
}

std::function<ncclTeam_t(ncclComm_t)> g_ncclTeamLsa = DefaultNcclTeamLsa;
std::function<void(void*, void const*)> g_ncclDevCommCopyLsaData = DefaultNcclDevCommCopyLsaData;

// The externals the #included devcomm .cc files link against.
extern "C" ncclTeam_t ncclTeamLsa(ncclComm_t comm) { return g_ncclTeamLsa(comm); }
void ncclDevCommCopyLsaData(void* dst, void const* src) { g_ncclDevCommCopyLsaData(dst, src); }

void ResetDevcommFakes() {
  g_ncclTeamLsa = DefaultNcclTeamLsa;
  g_ncclDevCommCopyLsaData = DefaultNcclDevCommCopyLsaData;
}
