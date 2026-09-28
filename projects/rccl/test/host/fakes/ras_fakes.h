/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Shared seams for RAS helpers used by multiple directly included source units.

#ifndef RCCL_TEST_HOST_RAS_FAKES_H_
#define RCCL_TEST_HOST_RAS_FAKES_H_

#include <poll.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "ras/diagnostics.h"
#include "ras/ras_internal.h"

extern struct pollfd* g_rasFakePfds;
extern std::vector<struct pollfd> g_rasFakePfdsStorage;
extern int g_getNewPollEntryCalls;
extern int g_msgAllocCalls;
extern int g_msgFreeCalls;
extern ncclResult_t g_msgAllocResult;
extern int g_diagContextInitCalls;
extern ncclResult_t g_diagContextInitResult;
extern const struct ncclComm* g_diagnosticsInitComm;

extern std::function<ncclResult_t(int*)> g_rasGetNewPollEntry;
extern std::function<ncclResult_t(struct rasMsg**, size_t)> g_rasMsgAlloc;
extern std::function<void(struct rasMsg*)> g_rasMsgFree;
extern std::function<int64_t(int64_t)> g_rasTimeoutFactorNs;
extern std::function<ncclResult_t(struct rasDiagnosticsContext*, const struct ncclComm*)>
  g_rasDiagnosticsContextInit;

ncclResult_t RasTestGetNewPollEntry(int* index);
ncclResult_t RasTestMsgAlloc(struct rasMsg** msg, size_t msgLen);
void RasTestMsgFree(struct rasMsg* msg);

void ResetRasFakes();

#endif  // RCCL_TEST_HOST_RAS_FAKES_H_
