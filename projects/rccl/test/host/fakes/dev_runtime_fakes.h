/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams for src/dev_runtime.cc, for targets that do not compile the real file: window-shape
// queries default to "not registered" so NVLS/CollNet registration arms stay off unless a test
// asks, and ncclDevrInitOnce defaults to success.

#ifndef RCCL_TEST_HOST_DEV_RUNTIME_FAKES_H_
#define RCCL_TEST_HOST_DEV_RUNTIME_FAKES_H_

#include <cstddef>
#include <functional>

#include "nccl.h"
// ncclDevrRegisterCapability is a plain enum, which C++ cannot forward-declare.
#include "dev_runtime.h"

struct ncclComm;
struct ncclDevrWindow;

extern std::function<ncclResult_t(struct ncclComm*, void const*, struct ncclDevrWindow**)> g_devrFindWindow;
extern std::function<bool(struct ncclDevrWindow*)> g_devrWindowIsMultiSegment;
extern std::function<bool(struct ncclDevrWindow*)> g_devrWindowHasSysmemSegment;
extern bool g_devrWindowIsMultiSegmentValue;
extern bool g_devrWindowHasSysmemSegmentValue;

// src/dev_runtime.cc's ncclDevrInitOnce: generous default (ncclSuccess), matching a comm that
// never sets symmetricSupport with a null peerInfo (dev_runtime.cc's own real error condition).
extern std::function<ncclResult_t(struct ncclComm*)> g_devrInitOnce;

// ncclDevrGetLsaRankPtr: fail-loud by default, since a peer address only exists for a window layout the test defines.
extern std::function<ncclResult_t(struct ncclComm*, struct ncclDevrWindow*, size_t /*offset*/, int /*lsaRank*/,
                                  void** /*outPtr*/)>
    g_devrGetLsaRankPtr;
// Fail-loud default: a window registration is collective, so a test must opt in to reaching one.
extern std::function<ncclResult_t(struct ncclComm*, void*, size_t, int, ncclWindow_t*)> g_devrWindowRegisterInGroup;
// Fail-loud default; same spelling as dev_runtime_micro_fakes.h, which no target links alongside this file.
extern std::function<ncclResult_t(ncclComm_t, ncclWindow_t)> g_devrNcclCommWindowDeregister;

// The dev_runtime.cc functions src/cft_dev_runtime.cc calls. All fail loud by default.
struct ncclTeam;
struct ncclDevrTeam;
extern std::function<int(struct ncclComm*)> g_computeLsaSize;
extern std::function<bool(int, ncclDevrRegisterCapability)> g_devrWinRegEnabled;
extern std::function<ncclResult_t(struct ncclComm*, struct ncclTeam, bool, bool, bool, bool, struct ncclDevrTeam**,
                                  bool*)>
    g_symTeamObtain;
extern std::function<ncclResult_t(ncclWindow_t, ncclComm_t*, struct ncclDevrWindow**)>
    g_findCommAndHostWindowFromDeviceWindow;
// UseFoundCommAndWindow() installs a g_findCommAndHostWindowFromDeviceWindow that returns these.
extern ncclComm_t g_foundComm;
extern struct ncclDevrWindow* g_foundWinHost;
void UseFoundCommAndWindow();

void ResetDevRuntimeFakes();

#endif  // RCCL_TEST_HOST_DEV_RUNTIME_FAKES_H_
