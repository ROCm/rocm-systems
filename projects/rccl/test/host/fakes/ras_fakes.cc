/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "fakes/ras_fakes.h"

#include <cstdlib>
#include <cstring>

#include "fakes/signature-drift.h"
#include "ras/ras_param.h"

struct pollfd* g_rasFakePfds = nullptr;
std::vector<struct pollfd> g_rasFakePfdsStorage;
int g_getNewPollEntryCalls = 0;
int g_msgAllocCalls = 0;
int g_msgFreeCalls = 0;
ncclResult_t g_msgAllocResult = ncclSuccess;
int g_diagContextInitCalls = 0;
ncclResult_t g_diagContextInitResult = ncclSuccess;
const struct ncclComm* g_diagnosticsInitComm = nullptr;

namespace {

ncclResult_t DefaultRasGetNewPollEntry(int* index) {
  ++g_getNewPollEntryCalls;
  g_rasFakePfdsStorage.push_back(pollfd{});
  g_rasFakePfds = g_rasFakePfdsStorage.data();
  *index = static_cast<int>(g_rasFakePfdsStorage.size()) - 1;
  return ncclSuccess;
}

ncclResult_t DefaultRasMsgAlloc(struct rasMsg** msg, size_t msgLen) {
  ++g_msgAllocCalls;
  if (g_msgAllocResult != ncclSuccess) return g_msgAllocResult;
  const size_t totalSize = offsetof(struct rasMsgMeta, msg) + msgLen;
  auto* meta = static_cast<struct rasMsgMeta*>(calloc(1, totalSize));
  if (meta == nullptr) return ncclSystemError;
  *msg = &meta->msg;
  return ncclSuccess;
}

void DefaultRasMsgFree(struct rasMsg* msg) {
  ++g_msgFreeCalls;
  if (msg == nullptr) return;
  auto* meta = reinterpret_cast<struct rasMsgMeta*>(reinterpret_cast<char*>(msg) - offsetof(struct rasMsgMeta, msg));
  free(meta);
}

int64_t DefaultRasTimeoutFactorNs(int64_t baseSeconds) { return baseSeconds * CLOCK_UNITS_PER_SEC; }

ncclResult_t DefaultRasDiagnosticsContextInit(struct rasDiagnosticsContext* ctx, const struct ncclComm* comm) {
  ++g_diagContextInitCalls;
  g_diagnosticsInitComm = comm;
  if (ctx) std::memset(ctx, 0, sizeof(*ctx));
  return g_diagContextInitResult;
}

}  // namespace

std::function<ncclResult_t(int*)> g_rasGetNewPollEntry = DefaultRasGetNewPollEntry;
std::function<ncclResult_t(struct rasMsg**, size_t)> g_rasMsgAlloc = DefaultRasMsgAlloc;
std::function<void(struct rasMsg*)> g_rasMsgFree = DefaultRasMsgFree;
std::function<int64_t(int64_t)> g_rasTimeoutFactorNs = DefaultRasTimeoutFactorNs;
std::function<ncclResult_t(struct rasDiagnosticsContext*, const struct ncclComm*)> g_rasDiagnosticsContextInit =
  DefaultRasDiagnosticsContextInit;

ASSERT_HOOK_MATCHES_PROD(g_rasGetNewPollEntry, rasGetNewPollEntry);
ASSERT_HOOK_MATCHES_PROD(g_rasMsgAlloc, rasMsgAlloc);
ASSERT_HOOK_MATCHES_PROD(g_rasMsgFree, rasMsgFree);
ASSERT_HOOK_MATCHES_PROD(g_rasTimeoutFactorNs, rasTimeoutFactorNs);
ASSERT_HOOK_MATCHES_PROD(g_rasDiagnosticsContextInit, rasDiagnosticsContextInit);

#undef ASSERT_HOOK_MATCHES_PROD

ncclResult_t RasTestGetNewPollEntry(int* index) { return g_rasGetNewPollEntry(index); }
ncclResult_t RasTestMsgAlloc(struct rasMsg** msg, size_t msgLen) { return g_rasMsgAlloc(msg, msgLen); }
void RasTestMsgFree(struct rasMsg* msg) { g_rasMsgFree(msg); }

int64_t rasTimeoutFactorNs(int64_t baseSeconds) { return g_rasTimeoutFactorNs(baseSeconds); }

ncclResult_t rasDiagnosticsContextInit(struct rasDiagnosticsContext* ctx, const struct ncclComm* comm) {
  return g_rasDiagnosticsContextInit(ctx, comm);
}

void ResetRasFakes() {
  g_rasGetNewPollEntry = DefaultRasGetNewPollEntry;
  g_rasMsgAlloc = DefaultRasMsgAlloc;
  g_rasMsgFree = DefaultRasMsgFree;
  g_rasTimeoutFactorNs = DefaultRasTimeoutFactorNs;
  g_rasDiagnosticsContextInit = DefaultRasDiagnosticsContextInit;
  g_rasFakePfdsStorage.clear();
  g_rasFakePfds = nullptr;
  g_getNewPollEntryCalls = 0;
  g_msgAllocCalls = 0;
  g_msgFreeCalls = 0;
  g_msgAllocResult = ncclSuccess;
  g_diagContextInitCalls = 0;
  g_diagContextInitResult = ncclSuccess;
  g_diagnosticsInitComm = nullptr;
}
