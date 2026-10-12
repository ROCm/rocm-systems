/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams for src/cft_dev_runtime.cc's callees that live in src/dev_runtime.cc
// (computeLsaSize, symTeamObtain, findCommAndHostWindowFromDeviceWindow) and
// src/dev_runtime.cc's ncclDevrWinRegEnabled. Targets linking this file
// compile the real cft_dev_runtime.cc but not the much larger dev_runtime.cc,
// so these four symbols need a seam of their own rather than the full file.

#ifndef RCCL_TEST_HOST_CFT_DEV_RUNTIME_FAKES_H_
#define RCCL_TEST_HOST_CFT_DEV_RUNTIME_FAKES_H_

#include <functional>

#include "nccl.h"
// ncclDevrRegisterCapability is a plain (non-fixed-underlying-type) enum, which C++
// does not allow forward-declaring -- pull in the real declarations instead.
#include "dev_runtime.h"

struct ncclComm;
struct ncclTeam;

// computeLsaSize: generous default (1) -- matches a comm with no LSA peers.
extern std::function<int(struct ncclComm*)> g_computeLsaSize;

// ncclDevrWinRegEnabled: default true -- matches a window registered for every capability.
extern std::function<bool(int, ncclDevrRegisterCapability)> g_devrWinRegEnabled;

// findCommAndHostWindowFromDeviceWindow: fail-loud default; a test must set
// g_foundComm/g_foundWinHost and install ResolveFoundCommAndWindow (or its own
// hook) before calling a function that depends on it.
extern std::function<ncclResult_t(ncclWindow_t, ncclComm_t*, struct ncclDevrWindow**)>
    g_findCommAndHostWindowFromDeviceWindow;
extern ncclComm_t g_foundComm;
extern struct ncclDevrWindow* g_foundWinHost;
// Installs the default hook that returns g_foundComm/g_foundWinHost directly.
void UseFoundCommAndWindow();

// symTeamObtain: fail-loud default; a test must install its own hook (e.g. one
// that heap-allocates a struct ncclDevrTeam with a populated worldRankList)
// before calling ncclGetMultimemDeviceLeInfo.
extern std::function<ncclResult_t(struct ncclComm*, struct ncclTeam, bool, bool, bool, bool, struct ncclDevrTeam**,
                                   bool*)>
    g_symTeamObtain;

void ResetCftDevRuntimeFakes();

#endif // RCCL_TEST_HOST_CFT_DEV_RUNTIME_FAKES_H_
