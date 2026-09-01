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

// MIRROR of src/dev_runtime.cc's ncclDevCommCopyLsaData body, not a call into it: dev_runtime.cc is
// not in RCCL_MICRO_TEST_SOURCES and cannot be host-compiled standalone, so the seam is the honest
// microtest boundary. Keep this span expression in lockstep with the production one -- a hand-edit
// there is invisible to every test whose name says "RealCopy" / "MovesLsaPrefix". What IS pinned
// against the real struct is the 128-byte span itself, static_asserted in devcomm-test.cc.
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
