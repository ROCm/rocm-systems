/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fail-loud stub floor for the transport subsystem (transport/proxy/NVLS/
// CollNet/PXN), shared by host-only microtests. These satisfy a unit-under-
// test's link-time symbol closure; the shallower tests never call them
// (abort-on-call, except benign teardown returning ncclSuccess). A test that
// needs to drive one of these replaces that individual entry with a real fake.

#include "transport_stubs.h"

#include <cstdlib>
#include <functional>

#include "nccl.h"

struct ncclComm;
struct ncclTopoGraph;

// src/transport/net.cc:343. Was a fail-loud stub in nccl_stubs.cc, which forced
// the enqueue target to omit it via a macro and supply its own; a seam here
// serves both. `false` means "no AINIC", which is what a host-only binary with
// no device actually has, so no test is silently steered by the default.
bool g_rcclUseAinic = false;
bool rcclUseAinic() { return g_rcclUseAinic; }

// src/proxy.cc: comm teardown stops the proxy through this, so it needs a working default, not a fail-loud one.
static ncclResult_t DefaultNcclProxyStop(struct ncclComm*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclComm*)> g_ncclProxyStop = DefaultNcclProxyStop;

void ResetTransportStubs() {
  g_rcclUseAinic = false;
  g_ncclProxyStop = DefaultNcclProxyStop;
}

ncclResult_t ncclCollNetChainBufferSetup(ncclComm_t comm) { ::abort(); }
ncclResult_t ncclCollNetDirectBufferSetup(ncclComm_t comm) { ::abort(); }
ncclResult_t ncclCollNetSetup(ncclComm_t comm, ncclComm_t parent, struct ncclTopoGraph* graphs[]) { ::abort(); }
// Controllable (was fail-loud). A std::function, not a result code: initTransportsRank:1506 branches on the WRITTEN
// *level, so a result-only seam could not drive it. Single call site (:1501) -> a success default is safe;
// see MICROTEST_README.md "Adding more controllable seams" for when to default a seam to failure instead.
extern std::function<ncclResult_t(int*)> g_ncclGetUserP2pLevel;
ncclResult_t ncclGetUserP2pLevel(int* level) { return g_ncclGetUserP2pLevel(level); }
ncclResult_t ncclNvlsBufferSetup(struct ncclComm* comm) { ::abort(); }
// Controllable (was fail-loud). :1618 uses bare NCCLCHECK, not NCCLCHECKGOTO, so a failure here returns
// WITHOUT running exit: -- the counter on ncclOsCpuCount is what makes that bypass observable.
extern ncclResult_t g_ncclNvlsInitResult;
extern int g_ncclNvlsInitCalls;
ncclResult_t ncclNvlsInit(struct ncclComm* comm) {
  g_ncclNvlsInitCalls++;
  return g_ncclNvlsInitResult;
}
ncclResult_t ncclNvlsSetup(struct ncclComm* comm, struct ncclComm* parent) { ::abort(); }
ncclResult_t ncclNvlsTreeConnect(struct ncclComm* comm) { ::abort(); }
// Controllable (was fail-loud). init.cc:2185, gated on comm->nvlsSupport surviving the :2182 fold.
extern ncclResult_t g_ncclNvlsTuningResult;
extern int g_ncclNvlsTuningCalls;
ncclResult_t ncclNvlsTuning(struct ncclComm* comm) {
  g_ncclNvlsTuningCalls++;
  return g_ncclNvlsTuningResult;
}
ncclResult_t ncclProxyCreate(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclProxyDestroy(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclProxyShmUnlink(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclProxyStop(struct ncclComm* comm) { return g_ncclProxyStop(comm); }
int ncclPxnDisable(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTransportPatConnect(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTransportRingConnect(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm) { ::abort(); }
// Controllable (was fail-loud). init.cc:2215, gated on comm->topo->treeDefined.
extern ncclResult_t g_ncclTreeBasePostsetResult;
extern int g_ncclTreeBasePostsetCalls;
extern struct ncclTopoGraph* g_ncclTreeBasePostsetGraph;
ncclResult_t ncclTreeBasePostset(struct ncclComm* comm, struct ncclTopoGraph* treeGraph) {
  g_ncclTreeBasePostsetCalls++;
  g_ncclTreeBasePostsetGraph = treeGraph;
  return g_ncclTreeBasePostsetResult;
}
ncclResult_t ncclTransportCheckP2pType(struct ncclComm*, bool*, bool*, bool*) { ::abort(); }
ncclResult_t ncclTransportP2pConnect(struct ncclComm*, int, int, int*, int, int*, int) { ::abort(); }
ncclResult_t ncclTransportP2pSetup(struct ncclComm*, struct ncclTopoGraph*, int, bool*) { ::abort(); }
