/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams nccl_device_core_fakes.cc OWNS (src/nccl_device/core.cc -- the host-side
// ncclTeam* accessors), declared once so a signature change is a compile error
// rather than a link mismatch.
//
// These are seams rather than fixed returns because ncclTeamRankIsMember /
// ncclTeamRankToTeam are real inline code that divides by the team's stride, so a
// zero-initialised ncclTeam_t SIGFPEs the moment a test reaches a strided arm. The
// defaults describe the contiguous, stride-1 team the comm already says it has; a
// test that needs a strided or offset team installs its own.

#ifndef RCCL_TEST_HOST_NCCL_DEVICE_CORE_FAKES_H_
#define RCCL_TEST_HOST_NCCL_DEVICE_CORE_FAKES_H_

#include <functional>

#include "nccl.h"
#include "nccl_device/core.h"  // ncclTeam_t, ncclCftTeamMode_t

// Team shape seen by (among others) ncclDevrWorldToLsaRank's symmetric arm and
// ncclGinConnectOnce. Defaults to the comm's own contiguous stride-1 team.
extern std::function<ncclTeam_t(ncclComm_t)> g_ncclTeamWorld;

// The rail team. Unlike the world team this has NO safe shape to guess -- it is
// comm->nRanks/lsaSize wide with stride lsaSize, and neither is reachable from a
// hand-built comm -- so the default is the zero team, i.e. "no rail". A test that
// takes a rail path must install its own.
extern std::function<ncclTeam_t(ncclComm_t)> g_ncclTeamRail;

// Maps a team-relative rank to a world rank. The default is production's own
// arithmetic: the mapping is the contract every caller is written against, so a
// fake that returned anything else would test the fake.
extern std::function<int(ncclComm_t, ncclTeam_t, int)> g_ncclTeamRankToWorld;

// The CFT teams, read by symMemoryObtain and ncclDevrCommCreateInternal. Same
// shape as the real accessors, reading the sizes ncclDevrInitOnce already
// computed, minus their ncclDevrInitOnce call: these are reached from inside that
// very function's callees, so calling it here would recurse.
extern std::function<ncclTeam_t(ncclComm_t, ncclCftTeamMode_t)> g_ncclTeamCft;
extern std::function<ncclTeam_t(ncclComm_t)> g_ncclTeamCftMultimem;

void ResetNcclDeviceCoreFakes();

#endif  // RCCL_TEST_HOST_NCCL_DEVICE_CORE_FAKES_H_
