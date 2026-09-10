/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm_fakes.h"

#include <type_traits>

#include "comm.h"
#include "dev_runtime.h"

// Signature-drift watchdog: assert each hook still matches the production symbol it shadows
// (templates + macro live in fakes/signature-drift.h).
#include "signature-drift.h"

// ncclTeamLsa cannot go through the macro: core.h declares two overloads (a device one taking
// ncclDevComm const&, and the host one taking ncclComm_t), so `&ncclTeamLsa` is ambiguous. Naming
// the host overload's type explicitly is the same guarantee by hand.
static_assert(std::is_same_v<::rccl_test_host::FnSigOf_t<decltype(g_ncclTeamLsa)>,
                             ncclTeam_t(ncclComm_t)>,
              "signature drift: g_ncclTeamLsa no longer matches the host ncclTeamLsa(ncclComm_t) "
              "declared in nccl_device/core.h -- update the std::function hook signature to match");

#undef ASSERT_HOOK_MATCHES_PROD

// Default: reports the LSA team as exactly spanning the communicator.
//
// TRAP: every filter under test compares ncclTeamLsa(comm).nRanks against comm->nRanks, so under
// this default that comparison is unconditionally true. A test that means to exercise the
// mismatch arm MUST install its own ScopedHook -- forgetting one does not fail, it silently
// re-tests the equal arm. Do not "simplify" a test by dropping its g_ncclTeamLsa hook.
ncclTeam_t DefaultNcclTeamLsa(ncclComm_t comm) {
  ncclTeam_t t{};
  t.nRanks = comm ? comm->nRanks : 0;
  t.rank = comm ? comm->rank : 0;
  t.stride = 1;
  return t;
}

std::function<ncclTeam_t(ncclComm_t)> g_ncclTeamLsa = DefaultNcclTeamLsa;

// The external the #included devcomm .cc files link against.
extern "C" ncclTeam_t ncclTeamLsa(ncclComm_t comm) { return g_ncclTeamLsa(comm); }

void ResetDevcommFakes() { g_ncclTeamLsa = DefaultNcclTeamLsa; }
