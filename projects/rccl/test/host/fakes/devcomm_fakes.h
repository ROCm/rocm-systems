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

// Seam for the external the devcomm compat shims reach. Install per-test
// behaviour with ScopedHook; the default reports the LSA team as spanning
// the whole communicator, so deviceApiSupport survives the filter unless a
// test says otherwise. That makes the filters' team-size comparison
// unconditionally true -- a test targeting the mismatch arm must install
// its own hook. See devcomm_fakes.cc.
extern std::function<ncclTeam_t(ncclComm_t)> g_ncclTeamLsa;

ncclTeam_t DefaultNcclTeamLsa(ncclComm_t comm);

void ResetDevcommFakes();

#endif  // RCCL_TEST_HOST_DEVCOMM_FAKES_H_
