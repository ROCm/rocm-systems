/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm_fakes.h"

#include <cstddef>
#include <cstring>
#include <type_traits>

#include "comm.h"
#include "dev_runtime.h"

// Signature-drift watchdog: assert each hook still matches the production symbol it shadows
// (templates + macro live in fakes/signature-drift.h).
#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_ncclDevCommCopyLsaData, ncclDevCommCopyLsaData);
// ncclTeamLsa cannot go through the macro above: core.h declares two overloads (a device one taking
// ncclDevComm const&, and the host one taking ncclComm_t), so `&ncclTeamLsa` is ambiguous.
// ASSERT_HOOK_MATCHES_SIG names the host overload's type instead -- the same guarantee.
ASSERT_HOOK_MATCHES_SIG(g_ncclTeamLsa, ncclTeam_t(ncclComm_t));

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

void DefaultNcclDevCommCopyLsaData(void* dst, void const* src) {
  std::memcpy(dst, src,
              offsetof(struct ncclDevComm, railGinBarrier) - offsetof(struct ncclDevComm, rank));
}

std::function<ncclTeam_t(ncclComm_t)> g_ncclTeamLsa = DefaultNcclTeamLsa;
std::function<void(void*, void const*)> g_ncclDevCommCopyLsaData = DefaultNcclDevCommCopyLsaData;

// The externals the #included devcomm .cc files link against.
extern "C" ncclTeam_t ncclTeamLsa(ncclComm_t comm) { return g_ncclTeamLsa(comm); }
// No wrapper definition here: dev_runtime.cc supplies the real
// ncclDevCommCopyLsaData in this binary. The hook above survives because
// devcomm-test.cc macro-shims its own call sites onto it.

void ResetDevcommFakes() {
  g_ncclTeamLsa = DefaultNcclTeamLsa;
  g_ncclDevCommCopyLsaData = DefaultNcclDevCommCopyLsaData;
}
