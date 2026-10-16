/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_DEVCOMM_FAKES_H_
#define RCCL_TEST_HOST_DEVCOMM_FAKES_H_

#include <functional>

#include "nccl.h"

struct ncclTeam;
typedef struct ncclTeam ncclTeam_t;

// Seams for the two externals the devcomm compat shims reach. Install per-test
// behaviour with ScopedHook; both have working defaults.
extern std::function<ncclTeam_t(ncclComm_t)> g_ncclTeamLsa;
extern std::function<void(void* dst, void const* src)> g_ncclDevCommCopyLsaData;

// Default: reports the LSA team as spanning the whole communicator, so
// deviceApiSupport survives the filter unless a test says otherwise. That
// makes the filters' team-size comparison unconditionally true -- a test
// targeting the mismatch arm must install its own hook. See devcomm_fakes.cc.
ncclTeam_t DefaultNcclTeamLsa(ncclComm_t comm);
// Default: a mirror of the production memcpy of the LSA prefix, so copy shims
// move real bytes. Update in lockstep with src/dev_runtime.cc; see the note on
// the definition in devcomm_fakes.cc.
void DefaultNcclDevCommCopyLsaData(void* dst, void const* src);

void ResetDevcommFakes();

#endif  // RCCL_TEST_HOST_DEVCOMM_FAKES_H_
