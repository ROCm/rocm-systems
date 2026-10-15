/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See cft_dev_runtime_fakes.h.

#include "cft_dev_runtime_fakes.h"

#include "dev_runtime_internal.h"
#include "fail_loud.h"
#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_computeLsaSize, computeLsaSize);
ASSERT_HOOK_MATCHES_PROD(g_devrWinRegEnabled, ncclDevrWinRegEnabled);
ASSERT_HOOK_MATCHES_PROD(g_findCommAndHostWindowFromDeviceWindow, findCommAndHostWindowFromDeviceWindow);
ASSERT_HOOK_MATCHES_PROD(g_symTeamObtain, symTeamObtain);
#undef ASSERT_HOOK_MATCHES_PROD

static int DefaultComputeLsaSize(struct ncclComm*) { return 1; }
std::function<int(struct ncclComm*)> g_computeLsaSize = DefaultComputeLsaSize;
int computeLsaSize(struct ncclComm* comm) { return g_computeLsaSize(comm); }

static bool DefaultDevrWinRegEnabled(int, ncclDevrRegisterCapability) { return true; }
std::function<bool(int, ncclDevrRegisterCapability)> g_devrWinRegEnabled = DefaultDevrWinRegEnabled;
bool ncclDevrWinRegEnabled(int winFlags, ncclDevrRegisterCapability capability) {
  return g_devrWinRegEnabled(winFlags, capability);
}

ncclComm_t g_foundComm = nullptr;
struct ncclDevrWindow* g_foundWinHost = nullptr;
static ncclResult_t FailLoudFindCommAndHostWindow(ncclWindow_t, ncclComm_t*, struct ncclDevrWindow**) {
  FailLoudUnfaked("cft_dev_runtime_fakes", "findCommAndHostWindowFromDeviceWindow");
}
std::function<ncclResult_t(ncclWindow_t, ncclComm_t*, struct ncclDevrWindow**)>
    g_findCommAndHostWindowFromDeviceWindow = FailLoudFindCommAndHostWindow;
ncclResult_t findCommAndHostWindowFromDeviceWindow(ncclWindow_t devWindow, ncclComm_t* foundComm,
                                                   struct ncclDevrWindow** hostWindow) {
  return g_findCommAndHostWindowFromDeviceWindow(devWindow, foundComm, hostWindow);
}
static ncclResult_t ResolveFoundCommAndWindowImpl(ncclWindow_t, ncclComm_t* foundComm,
                                                  struct ncclDevrWindow** hostWindow) {
  *foundComm = g_foundComm;
  *hostWindow = g_foundWinHost;
  return ncclSuccess;
}
void UseFoundCommAndWindow() { g_findCommAndHostWindowFromDeviceWindow = ResolveFoundCommAndWindowImpl; }

static ncclResult_t FailLoudSymTeamObtain(struct ncclComm*, struct ncclTeam, bool, bool, bool, bool,
                                          struct ncclDevrTeam**, bool*) {
  FailLoudUnfaked("cft_dev_runtime_fakes", "symTeamObtain");
}
std::function<ncclResult_t(struct ncclComm*, struct ncclTeam, bool, bool, bool, bool, struct ncclDevrTeam**, bool*)>
    g_symTeamObtain = FailLoudSymTeamObtain;
ncclResult_t symTeamObtain(struct ncclComm* comm, struct ncclTeam team, bool multimem, bool counted, bool wantsLeUc,
                           bool wantsLeMc, struct ncclDevrTeam** outTeam, bool* needBarrier) {
  return g_symTeamObtain(comm, team, multimem, counted, wantsLeUc, wantsLeMc, outTeam, needBarrier);
}

void ResetCftDevRuntimeFakes() {
  g_computeLsaSize = DefaultComputeLsaSize;
  g_devrWinRegEnabled = DefaultDevrWinRegEnabled;
  g_findCommAndHostWindowFromDeviceWindow = FailLoudFindCommAndHostWindow;
  g_foundComm = nullptr;
  g_foundWinHost = nullptr;
  g_symTeamObtain = FailLoudSymTeamObtain;
}
