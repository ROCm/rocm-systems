/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for the host-side ncclTeam* accessors src/nccl_device/core.cc owns.
//
// ASSERT_HOOK_MATCHES_SIG rather than ASSERT_HOOK_MATCHES_PROD: each symbol is also
// a device entry point taking ncclDevComm const&, so `&ncclTeamWorld` is ambiguous.
//
// ncclTeamLsa is a core.cc symbol too, but is faked in fakes/devcomm_fakes.cc; moving
// it is tracked separately.

#include "comm.h"
#include "nccl_device/core.h"

#include "fakes/nccl_device_core_fakes.h"
#include "fakes/signature-drift.h"

static ncclTeam_t DefaultTeamWorld(ncclComm_t comm) {
  if (comm == nullptr) return ncclTeam_t{0, 0, 1};
  return ncclTeam_t{comm->nRanks, comm->rank, 1};
}
std::function<ncclTeam_t(ncclComm_t)> g_ncclTeamWorld = DefaultTeamWorld;
ncclTeam_t ncclTeamWorld(ncclComm_t comm) { return g_ncclTeamWorld(comm); }
ASSERT_HOOK_MATCHES_SIG(g_ncclTeamWorld, ncclTeam_t(ncclComm_t));

static ncclTeam_t DefaultTeamRail(ncclComm_t) { return ncclTeam_t{}; }
std::function<ncclTeam_t(ncclComm_t)> g_ncclTeamRail = DefaultTeamRail;
ncclTeam_t ncclTeamRail(ncclComm_t comm) { return g_ncclTeamRail(comm); }
ASSERT_HOOK_MATCHES_SIG(g_ncclTeamRail, ncclTeam_t(ncclComm_t));

static int DefaultTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}
std::function<int(ncclComm_t, ncclTeam_t, int)> g_ncclTeamRankToWorld = DefaultTeamRankToWorld;
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return g_ncclTeamRankToWorld(comm, team, rank);
}
ASSERT_HOOK_MATCHES_SIG(g_ncclTeamRankToWorld, int(ncclComm_t, ncclTeam_t, int));

static ncclTeam_t DefaultTeamCft(ncclComm_t comm, ncclCftTeamMode_t) {
  if (comm == nullptr) return ncclTeam_t{0, 0, 1};
  return ncclTeam_t{comm->devrState.cftSize, comm->devrState.cftSelf, 1};
}
std::function<ncclTeam_t(ncclComm_t, ncclCftTeamMode_t)> g_ncclTeamCft = DefaultTeamCft;
ncclTeam_t ncclTeamCft(ncclComm_t comm, ncclCftTeamMode_t mode) { return g_ncclTeamCft(comm, mode); }
ASSERT_HOOK_MATCHES_SIG(g_ncclTeamCft, ncclTeam_t(ncclComm_t, ncclCftTeamMode_t));

static ncclTeam_t DefaultTeamCftMultimem(ncclComm_t comm) {
  if (comm == nullptr) return ncclTeam_t{0, 0, 1};
  return ncclTeam_t{comm->devrState.cftMcSize, comm->devrState.cftMcSelf, 1};
}
std::function<ncclTeam_t(ncclComm_t)> g_ncclTeamCftMultimem = DefaultTeamCftMultimem;
ncclTeam_t ncclTeamCftMultimem(ncclComm_t comm) { return g_ncclTeamCftMultimem(comm); }
ASSERT_HOOK_MATCHES_SIG(g_ncclTeamCftMultimem, ncclTeam_t(ncclComm_t));

void ResetNcclDeviceCoreFakes() {
  g_ncclTeamWorld = DefaultTeamWorld;
  g_ncclTeamRail = DefaultTeamRail;
  g_ncclTeamRankToWorld = DefaultTeamRankToWorld;
  g_ncclTeamCft = DefaultTeamCft;
  g_ncclTeamCftMultimem = DefaultTeamCftMultimem;
}
